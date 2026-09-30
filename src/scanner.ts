// Loop da câmera: retícula viva (scanic), gates de estabilidade/foco, captura
// com correção de perspectiva e o edge-trigger do modo sling.
//
// Portado do scanner do CartinhasDaJu. Os limiares vieram de telemetria real
// de scans no celular; a razão de cada um está no comentário ao lado.
import * as Q from "./quad";
import { frameDiff, glare, presence, sharpness } from "./pixels";
import { recognize, type Recognition } from "./recognize";
import { detectYolo, preloadYolo, YOLO_MIN_SCORE } from "./yolo";
import { applyCamera, openSavedCamera, tuneCamera, type CameraInfo } from "./camera";

type Scanic = typeof import("scanic");

/** Callbacks from the scanner to the UI. */
export interface ScannerHooks {
  status(msg: string): void;
  /**
   * A capture started or its crop is ready. Handheld: called first with a quick
   * preview cut from the video, then with `photo`, the image sent to the server.
   */
  captured(photo: HTMLCanvasElement): void;
  /** Recognition finished. `result` is null on error or no match. */
  result(result: Recognition | null, photo: HTMLCanvasElement): void;
  /** Readiness for the next capture, 0..1 (drives the progress bar). */
  progress(frac: number): void;
}

const TICK_MS = 150;
const STABLE_HAND = 10;        // ~1.5s firme antes de disparar: dá tempo de enquadrar
const STABLE_SLING = 5;        // ~0.75s: rig fixo converge rápido, mas não dispara com carta meio-posta
const FOCUS_RATIO = 0.85;      // dispara com foco a 85% do pico visto (espera o frame nítido)
const DIFF_THRESHOLD = 40;     // tolera ruído de autofoco/exposição do celular
const SLING_DIFF = 35;         // diff acima disso vs a carta capturada = carta nova
const PRESENCE_MIN = 16;       // desvio-padrão abaixo disso = quadro vazio
const GLARE_HINT = 0.12;       // >12% de pixels estourados: pede para inclinar
// Com o servidor, um frame ruim custa ~0.5s e é recusado barato; se os gates
// nunca abrem (mão tremendo), dispara mesmo assim depois da espera.
const FAILSAFE_HAND_MS = 3000;
const FAILSAFE_SLING_MS = 2500;
// Se o diff nunca passa do limiar nem o quadro esvazia, o sling travava.
// 4s desarmado com carta presente força o re-arme.
const SLING_REARM_TIMEOUT_MS = 4000;

const QUAD_INTERVAL_MS = 200;
const QUAD_PROXY_DIM = 480;
// Lado maior da foto nítida (takePhoto) usada para os recortes.
const SHOT_MAX_DIM = 2400;
const QUAD_STABLE_SLING = 2;
const QUAD_STABLE_HAND = 4;
const QUAD_STILL_PX = 6;       // movimento médio por canto abaixo disso = parado
const QUAD_ADOPT_PX = 40;      // salto maior só é adotado com 2 detecções confirmando
const QUAD_MISS_REARM = 2;
const QUAD_MOVED_REARM_PX = 40;

export class Scanner {
  private stream: MediaStream | null = null;
  private interval: number | null = null;
  private raf: number | null = null;
  private capturing = false;
  private paused = false;

  private coords: { px: number; py: number; pw: number; ph: number } | null = null;
  private coordsDirty = true;
  private proxyCtx: CanvasRenderingContext2D | null = null;
  private stable = 0;
  private sharpPeak = 0;
  private lastData: Uint8ClampedArray | null = null;
  private presentSince = 0;

  private sling = false;
  private armed = true;
  private disarmedAt = 0;
  private lastAddData: Uint8ClampedArray | null = null;
  private movedSince = false;
  private pin = "";
  /** Same card re-read without leaving the frame must not be added twice. */
  private lastAddedId: string | null = null;
  private cardGoneSinceAdd = true;

  private scanic: Scanic | null = null;
  private quadLast: Q.Quad | null = null;
  private quadRaw: Q.Quad | null = null;
  private quadPending: Q.Quad | null = null;
  private quadStable = 0;
  private quadMiss = 0;
  private quadBusy = false;
  private quadLastRun = 0;
  /** Dimensions of the tracking space: the full frame scaled to QUAD_PROXY_DIM. */
  private quadProxy: { width: number; height: number } | null = null;
  private quadCropCanvas: HTMLCanvasElement | null = null;
  private imageCapture: { takePhoto(): Promise<Blob> } | null = null;
  private quadFiredAt: Q.Quad | null = null;
  private reticle: Q.Quad | null = null;

  private readonly ro: ResizeObserver;

  constructor(
    private readonly video: HTMLVideoElement,
    private readonly frame: HTMLElement,
    private readonly overlay: SVGSVGElement,
    private readonly hooks: ScannerHooks,
  ) {
    this.ro = new ResizeObserver(() => { this.coordsDirty = true; });
    import("scanic").then((m) => { this.scanic = m; document.body.classList.add("scanic-on"); }).catch(() => {});
    preloadYolo();
  }

  get slingMode(): boolean {
    return this.sling;
  }

  setSling(on: boolean): void {
    this.sling = on;
    this.armed = true;
    this.pin = "";
  }

  /** Open the camera and start the loop. Returns an error message, or null on success. */
  async start(): Promise<string | null> {
    if (!window.isSecureContext) return "A câmera exige HTTPS.";
    if (!navigator.mediaDevices?.getUserMedia) return "Este navegador não tem acesso à câmera.";
    // 4:3 em alta: frame mais alto que 16:9, então o object-fit:cover corta menos as laterais.
    const attempts: MediaStreamConstraints[] = [
      { video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1440 } } },
      { video: { facingMode: "environment" } },
      { video: true },
    ];
    let lastErr: DOMException | null = null;
    this.stream = await openSavedCamera();
    for (const c of this.stream ? [] : attempts) {
      try {
        this.stream = await navigator.mediaDevices.getUserMedia(c);
        break;
      } catch (err) {
        lastErr = err as DOMException;
        if (lastErr.name === "NotAllowedError" || lastErr.name === "SecurityError") break;
      }
    }
    if (!this.stream) {
      const map: Record<string, string> = {
        NotAllowedError: "Permissão de câmera negada. Libere a câmera nas configurações do site.",
        NotFoundError: "Nenhuma câmera encontrada.",
        NotReadableError: "A câmera está em uso por outro app.",
      };
      return map[lastErr?.name || ""] || `Câmera indisponível (${lastErr?.name || "erro"}).`;
    }
    ({ stream: this.stream, info: this.camInfo } = await tuneCamera(this.stream));
    this.video.srcObject = this.stream;
    // iOS: o play() pode resolver antes do primeiro frame. Espera o metadata DESTE stream.
    await new Promise<void>((resolve) => {
      if (this.video.readyState >= 1 && this.video.videoWidth) return resolve();
      const done = () => resolve();
      this.video.addEventListener("loadedmetadata", done, { once: true });
      setTimeout(done, 2500);
    });
    await this.video.play().catch(() => {});
    if (!this.video.videoWidth) return "Câmera sem imagem. Feche outros apps que usam a câmera.";
    this.ro.observe(this.frame);
    this.coordsDirty = true;
    setTimeout(() => { this.coordsDirty = true; }, 400);
    this.interval = window.setInterval(() => this.tick(), TICK_MS);
    this.startReticle();
    return null;
  }

  stop(): void {
    if (this.interval) clearInterval(this.interval);
    this.interval = null;
    if (this.raf) cancelAnimationFrame(this.raf);
    this.raf = null;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.imageCapture = null;   // preso à faixa antiga; recriado no próximo disparo
    this.video.srcObject = null;
    this.ro.disconnect();
  }

  /** Rebuild the camera. Android often keeps a "live" track that paints black after background. */
  async restart(): Promise<string | null> {
    this.stop();
    return this.start();
  }

  /** Resume after a handheld result was closed. */
  resume(): void {
    this.video.play().catch(() => {});   // descongela a imagem do disparo
    this.paused = false;
    this.capturing = false;
    this.stable = 0;
    this.sharpPeak = 0;
    this.lastData = null;
    this.presentSince = 0;
  }

  /** Capture now (shutter button). */
  captureNow(): void {
    if (this.capturing || !this.coords) return;
    this.capture(this.currentFrameQuad() || undefined);
  }

  hasTorch(): boolean {
    const track = this.stream?.getVideoTracks()[0];
    const caps = track?.getCapabilities?.() as (MediaTrackCapabilities & { torch?: boolean }) | undefined;
    return !!caps?.torch;
  }

  async setTorch(on: boolean): Promise<boolean> {
    const track = this.stream?.getVideoTracks()[0];
    if (!track) return false;
    return applyCamera(track, { torch: on });
  }

  /**
   * Sling only: decide what to do with a recognized card and re-arm the loop.
   *
   * @returns "added" when the caller must add the card, "duplicate" when the same card is still in frame.
   */
  slingAccept(rec: Recognition): "added" | "duplicate" {
    const id = rec.card.api_id;
    // Sem janela de tempo: a carta parada no quadro furava um guard de 30s.
    // Sair do quadro é o único jeito de somar a mesma carta de novo.
    if (id === this.lastAddedId && !this.cardGoneSinceAdd) {
      this.pin = `${rec.card.name} já está na lista — tire e ponha a próxima`;
      this.resume();
      return "duplicate";
    }
    this.lastAddedId = id;
    this.cardGoneSinceAdd = false;
    this.pin = `✓ ${rec.card.name} — tire e ponha a próxima`;
    this.flashFrame("is-added", 700);
    this.resume();
    return "added";
  }

  /** Sling only: the capture was not a strong match. Wait for the frame to empty and try again. */
  slingReject(): void {
    this.pin = "Não reconheci — tire a carta e ponha de novo";
    this.flashFrame("is-rejected", 600);
    this.resume();
  }

  private flashFrame(cls: string, ms: number): void {
    this.frame.classList.add(cls);
    setTimeout(() => this.frame.classList.remove(cls), ms);
  }

  private updateCoords(): void {
    const v = this.video;
    const vw = v.videoWidth, vh = v.videoHeight, ew = v.clientWidth, eh = v.clientHeight;
    // Math.max = object-fit:cover. Tem que casar com o CSS do <video>, senão o
    // corte mapeia para outra região do sensor.
    const s = Math.max(ew / vw, eh / vh);
    const cropX = (vw - ew / s) / 2, cropY = (vh - eh / s) / 2;
    const fr = this.frame.getBoundingClientRect(), vr = v.getBoundingClientRect();
    this.coords = {
      px: cropX + (fr.left - vr.left) / s,
      py: cropY + (fr.top - vr.top) / s,
      pw: fr.width / s,
      ph: fr.height / s,
    };
    this.coordsDirty = false;
  }

  private setStatus(msg: string): void {
    this.hooks.status(msg);
  }

  private tick(): void {
    const v = this.video;
    if (!v.videoWidth || this.paused) return;
    if (this.capturing) {
      this.observeWhileCapturing();
      return;
    }
    const track = this.stream?.getVideoTracks()[0];
    if (track && track.readyState === "ended") { void this.restart(); return; }
    if (!this.coords || this.coordsDirty) this.updateCoords();
    if (this.scanic) void this.quadTick();

    const need = this.sling ? STABLE_SLING : STABLE_HAND;
    const { px, py, pw, ph } = this.coords!;
    if (!this.proxyCtx) {
      const c = document.createElement("canvas");
      c.width = 160;
      c.height = Math.round(160 * (88 / 63));
      this.proxyCtx = c.getContext("2d", { willReadFrequently: true });
    }
    const ctx = this.proxyCtx!, cw = ctx.canvas.width, chh = ctx.canvas.height;
    ctx.drawImage(v, px, py, pw, ph, 0, 0, cw, chh);
    const data = ctx.getImageData(0, 0, cw, chh).data;
    const sharp = sharpness(data, cw, chh);

    const present = presence(data) >= PRESENCE_MIN;
    this.frame.classList.toggle("card-present", present);
    if (present && !this.presentSince) this.presentSince = Date.now();
    if (!present) {
      // Quadro vazio: a carta anterior saiu, então o sling rearma (edge-trigger).
      this.presentSince = 0;
      this.stable = 0;
      this.sharpPeak = 0;
      this.lastData = data;
      this.armed = true;
      this.movedSince = false;
      this.cardGoneSinceAdd = true;
      this.pin = "";
      this.hooks.progress(0);
      this.setStatus(this.sling ? "Sling pronto — ponha a próxima carta" : "Aponte a câmera para uma carta");
      return;
    }

    if (!this.lastData) {
      this.sharpPeak = sharp;
      this.lastData = data;
      return;
    }
    const diff = frameDiff(data, this.lastData);
    const moved = diff > DIFF_THRESHOLD * 3;
    this.stable = diff < DIFF_THRESHOLD ? Math.min(this.stable + 1, need) : moved ? 0 : Math.max(this.stable - 1, 0);
    // O pico decai devagar: um pico transitório na hora de pôr a carta travava
    // o "Focando…" para sempre, porque o platô real nunca chegava a 85% dele.
    if (moved) this.sharpPeak = 0;
    else if (sharp > this.sharpPeak) this.sharpPeak = sharp;
    else this.sharpPeak *= 0.97;
    const focused = this.sharpPeak > 0 && sharp >= this.sharpPeak * FOCUS_RATIO;
    const focusFrac = this.sharpPeak > 0 ? Math.min(1, sharp / (this.sharpPeak * FOCUS_RATIO)) : 0;
    this.hooks.progress(Math.min(this.stable / need, focusFrac));

    if (this.sling && !this.armed) {
      // Rig fixo: troca carta-sobre-carta sem quadro vazio. Rearma quando o
      // conteúdo difere bastante da carta que acabou de entrar na lista.
      if (diff > SLING_DIFF) this.movedSince = true;
      if (this.movedSince && this.lastAddData && frameDiff(data, this.lastAddData) > SLING_DIFF) this.armed = true;
      if (this.disarmedAt && Date.now() - this.disarmedAt > SLING_REARM_TIMEOUT_MS) this.armed = true;
    }
    const waiting = this.sling && !this.armed;
    this.frame.classList.toggle("sling-waiting", waiting);
    const glary = this.stable >= Math.ceil(need * 0.4) && glare(data) > GLARE_HINT;
    if (waiting) this.setStatus(this.pin || "Tire a carta e ponha a próxima");
    else if (glary) this.setStatus("Reflexo — incline um pouco a carta");
    else if (this.stable >= need && !focused) this.setStatus("Focando…");
    else if (this.stable >= Math.ceil(need * 0.4)) this.setStatus("Estabilizando…");
    else this.setStatus("Enquadre a carta");

    this.lastData = data;
    if (waiting) return;
    if (this.stable >= need && focused) {
      this.capture(this.currentFrameQuad() || undefined);
      return;
    }
    const failsafe = this.sling ? FAILSAFE_SLING_MS : FAILSAFE_HAND_MS;
    if (this.presentSince && Date.now() - this.presentSince > failsafe) {
      this.capture(this.currentFrameQuad() || undefined);
    }
  }

  // Durante o reconhecimento (até alguns segundos), a troca de carta passava
  // despercebida e o sling ficava desarmado com carta nova no quadro.
  private observeWhileCapturing(): void {
    if (!this.sling || this.armed || !this.proxyCtx || !this.coords) return;
    const { px, py, pw, ph } = this.coords;
    const ctx = this.proxyCtx, cw = ctx.canvas.width, chh = ctx.canvas.height;
    ctx.drawImage(this.video, px, py, pw, ph, 0, 0, cw, chh);
    const d = ctx.getImageData(0, 0, cw, chh).data;
    if (presence(d) < PRESENCE_MIN) {
      this.armed = true;
      this.movedSince = false;
      this.cardGoneSinceAdd = true;
      this.pin = "";
    } else if (this.lastAddData && frameDiff(d, this.lastAddData) > SLING_DIFF) {
      this.movedSince = true;
    }
  }

  private currentFrameQuad(): Q.Quad | null {
    if (!this.quadRaw || this.quadMiss > 0 || !this.quadProxy || !this.video.videoWidth) return null;
    return Q.scale(this.quadRaw, this.video.videoWidth / this.quadProxy.width);
  }

  // Detecção em duas camadas (teste no celular em 2026-09-29/30: 9 de 10 scans
  // adicionados, zero "não é essa"):
  // - overlay: scanic clássico no quadro inteiro, ~270ms no celular. É leve o
  //   bastante para seguir a mão e medir se a carta parou.
  // - foto capturada: YOLO de cantos (src/yolo.ts), uma vez só, ~600ms. Acha a
  //   borda externa mesmo com dedo no canto ou quando o clássico não travou.
  // O YOLO contínuo levava ~400-1000ms por quadro e o contorno ficava para trás.
  // O scanic ML (DocCornerNet) punha os cantos para dentro e cortava o rodapé.

  // Cantos crus da carta em px de `src` (vw×vh) pelo scanic clássico, ou null.
  private async detectClassic(src: CanvasImageSource, vw: number, vh: number, canvas: HTMLCanvasElement): Promise<Q.Quad | null> {
    const s2 = QUAD_PROXY_DIM / Math.max(vw, vh);
    const w = Math.round(vw * s2), h = Math.round(vh * s2);
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    canvas.getContext("2d", { willReadFrequently: true })!.drawImage(src, 0, 0, vw, vh, 0, 0, w, h);
    const r = await this.scanic!.scanDocument(canvas, { mode: "detect", maxProcessingDimension: QUAD_PROXY_DIM });
    if (!(r?.success && r.corners)) return null;
    const out = {} as Q.Quad;
    for (const k of ["topLeft", "topRight", "bottomRight", "bottomLeft"] as const)
      out[k] = { x: r.corners[k].x / s2, y: r.corners[k].y / s2 };
    return out;
  }

  // Cantos crus da carta na foto capturada: YOLO; enquanto o modelo baixa, o
  // clássico. null = sem carta.
  private async detectPhoto(photo: HTMLCanvasElement): Promise<Q.Quad | null> {
    const t0 = performance.now();
    const canvas = document.createElement("canvas");
    const s2 = QUAD_PROXY_DIM / Math.max(photo.width, photo.height);
    canvas.width = Math.round(photo.width * s2);
    canvas.height = Math.round(photo.height * s2);
    canvas.getContext("2d")!.drawImage(photo, 0, 0, canvas.width, canvas.height);
    const y = await detectYolo(canvas);
    let out: Q.Quad | null = null;
    if (y) {
      if (y.score >= YOLO_MIN_SCORE) {
        out = {} as Q.Quad;
        for (const k of ["topLeft", "topRight", "bottomRight", "bottomLeft"] as const)
          out[k] = { x: y[k].x / s2, y: y[k].y / s2 };
      }
    } else {
      out = await this.detectClassic(photo, photo.width, photo.height, canvas);
    }
    this.capDetMs = performance.now() - t0;
    return out;
  }

  /** What the browser reports about the open camera (lens, focus, resolution). */
  camInfo: CameraInfo | null = null;

  /** Moving average of the overlay detector time per call, in ms. */
  detMs = 0;
  /** Detector time on the last captured photo, in ms. */
  capDetMs = 0;

  private async quadTick(): Promise<void> {
    const scanic = this.scanic;
    if (!scanic || this.quadBusy || this.capturing) return;
    const now = Date.now();
    if (this.quadLast && now - this.quadLastRun < QUAD_INTERVAL_MS) return;
    this.quadLastRun = now;
    this.quadBusy = true;
    try {
      const vw = this.video.videoWidth, vh = this.video.videoHeight;
      const s = QUAD_PROXY_DIM / Math.max(vw, vh);
      const pw = Math.round(vw * s), ph = Math.round(vh * s);
      if (!this.quadProxy || this.quadProxy.width !== pw || this.quadProxy.height !== ph) this.quadProxy = { width: pw, height: ph };
      if (!this.quadCropCanvas) this.quadCropCanvas = document.createElement("canvas");
      // Cantos em px do quadro → espaço de rastreio (quadro inteiro reduzido): filtro,
      // suavização, disparo e retícula continuam iguais.
      const t0 = performance.now();
      const fc = await this.detectClassic(this.video, vw, vh, this.quadCropCanvas);
      const ms = performance.now() - t0;
      this.detMs = this.detMs ? this.detMs * 0.8 + ms * 0.2 : ms;
      if (this.capturing) return;
      const ordered = fc ? Q.inset(Q.order(Q.scale(fc, s))) : null;
      if (!ordered || !Q.isSane(ordered, pw, ph)) {
        this.quadMiss++;
        this.quadStable = 0;
        this.quadPending = null;
        // Um miss isolado não apaga o contorno (reflexo/blur num frame só);
        // apagar a cada falha deixava a retícula piscando.
        if (this.quadMiss >= 2) {
          this.quadLast = null;
          this.quadRaw = null;
          this.cardGoneSinceAdd = true;
        }
        if (this.sling && !this.armed && this.quadMiss >= QUAD_MISS_REARM) {
          this.armed = true;
          this.movedSince = false;
        }
        return;
      }
      this.quadMiss = 0;
      this.quadRaw = ordered;
      if (this.quadLast && Q.avgDelta(ordered, this.quadLast) < QUAD_ADOPT_PX) {
        // Mesma carta: suaviza (EMA) para o contorno deslizar em vez de tremer.
        this.quadStable = Q.avgDelta(ordered, this.quadLast) < QUAD_STILL_PX ? this.quadStable + 1 : 1;
        this.quadLast = Q.lerp(this.quadLast, ordered, 0.45);
        this.quadPending = null;
      } else if (this.quadPending && Q.avgDelta(ordered, this.quadPending) < QUAD_ADOPT_PX) {
        this.quadLast = ordered;
        this.quadStable = 1;
        this.quadPending = null;
      } else {
        // Salto de um frame só (reflexo, objeto errado): espera confirmação.
        this.quadPending = ordered;
        return;
      }
      if (this.sling && !this.armed && this.quadFiredAt && Q.avgDelta(this.quadLast, this.quadFiredAt) > QUAD_MOVED_REARM_PX) {
        this.armed = true;
      }
      const need = this.sling ? QUAD_STABLE_SLING : QUAD_STABLE_HAND;
      if (this.quadStable >= need && (!this.sling || this.armed) && this.coords && !this.paused) {
        this.quadFiredAt = this.quadLast;
        this.quadStable = 0;
        // Corta pelos cantos crus desta detecção: a EMA fica meio passo atrás da borda real.
        this.capture(Q.scale(ordered, vw / pw));
      }
    } catch {
      /* o detector nunca derruba o loop */
    } finally {
      this.quadBusy = false;
    }
  }

  private startReticle(): void {
    const loop = () => {
      if (!this.interval) { this.raf = null; return; }
      if (!this.capturing) this.drawReticle();
      this.raf = requestAnimationFrame(loop);
    };
    this.raf = requestAnimationFrame(loop);
  }

  // `final`: contorno do recorte (em px do vídeo). Desenha sem suavizar, para a
  // tela mostrar exatamente o que vai para o servidor.
  private drawReticle(final?: Q.Quad): void {
    const v = this.video;
    if (!this.scanic || !v.videoWidth || !this.coords) return;
    let target: Q.Quad;
    let tracking = false;
    if (final) {
      target = final;
      tracking = true;
    } else if (this.quadLast && this.quadMiss === 0 && this.quadProxy) {
      target = Q.scale(this.quadLast, v.videoWidth / this.quadProxy.width);
      tracking = true;
    } else {
      const { px, py, pw, ph } = this.coords;
      target = Q.fromRect(px, py, pw, ph);
    }
    // 0.35 por quadro: com a detecção a cada ~250ms no celular, 0.18 deixava o
    // contorno sempre atrás da carta.
    this.reticle = this.reticle && !final ? Q.lerp(this.reticle, target, 0.35) : target;
    const vw = v.videoWidth, vh = v.videoHeight, ew = v.clientWidth, eh = v.clientHeight;
    const s = Math.max(ew / vw, eh / vh);
    const cropX = (vw - ew / s) / 2, cropY = (vh - eh / s) / 2;
    const pts = Q.points(this.reticle).map((p) => ({ x: (p.x - cropX) * s, y: (p.y - cropY) * s }));
    const path = Q.roundedPath(pts);
    const [mask, poly] = [this.overlay.children[0], this.overlay.children[1]];
    poly.setAttribute("d", path);
    mask.setAttribute("d", `M0 0H${ew}V${eh}H0Z ${path}`);
    this.overlay.classList.toggle("tracking", tracking);
    this.overlay.style.opacity = "1";
  }

  /**
   * Perspective-corrected photo of the card inside `quad`.
   * Returns the photo, or `drop` (the quad to fall back to) when no warp is plausible.
   * `used` is the quad of the photo, in photo pixels.
   * `detected`: corners already found on this photo (skips a second detector run).
   */
  private async warp(full: HTMLCanvasElement, quad: Q.Quad, detected?: Q.Quad | null): Promise<{ photo: HTMLCanvasElement | null; drop: Q.Quad | null; used: Q.Quad }> {
    const scanic = this.scanic;
    if (!scanic) return { photo: null, drop: null, used: quad };
    // Re-detecta NA PRÓPRIA FOTO: os cantos são do mesmo instante dela (a mão
    // mexe durante o refoco do takePhoto). O contorno rastreado — em escala da
    // foto — fica de reserva se o warp novo falhar. (Uma guarda que preferia o
    // rastreado piorou: dedo cobrindo canto, caixa de ataque.)
    let use = quad;
    let backup: Q.Quad | null = null;
    try {
      const fc = detected !== undefined ? detected : await this.detectPhoto(full);
      if (fc) {
        const fresh = Q.inset(Q.order(fc));
        if (Q.isSane(fresh, full.width, full.height)) { backup = quad; use = fresh; }
      }
    } catch { /* fica com o quad anterior */ }
    let photo = await this.warpOne(full, use);
    if (!photo && backup) { photo = await this.warpOne(full, backup); if (photo) use = backup; }
    return { photo, drop: photo ? null : use, used: use };
  }

  private async warpOne(full: HTMLCanvasElement, quad: Q.Quad): Promise<HTMLCanvasElement | null> {
    const scanic = this.scanic!;
    const r = await scanic.extractDocument(full, quad, { output: "canvas" });
    let cap = r?.success && r.output instanceof HTMLCanvasElement && r.output.width > 40 ? r.output : null;
    if (!cap) return null;
    // Carta é retrato. O embedding é sensível a rotação, então gira o warp deitado.
    if (cap.width > cap.height) {
      const rot = document.createElement("canvas");
      rot.width = cap.height;
      rot.height = cap.width;
      const ctx = rot.getContext("2d")!;
      ctx.translate(rot.width, 0);
      ctx.rotate(Math.PI / 2);
      ctx.drawImage(cap, 0, 0);
      cap = rot;
    }
    // Em full-art/holo o detector agarra uma aresta interna e o warp sai com
    // um pedaço da carta. Fora da faixa, a carta inteira do corte fixo vale mais.
    const asp = cap.width / cap.height;
    if (asp < Q.WARP_ASPECT_MIN || asp > Q.WARP_ASPECT_MAX) return null;
    // O scanic dimensiona o warp pelo MAIOR lado de cada par, então a
    // perspectiva residual sai como carta esticada (0.69-0.80 medido vs 0.716
    // real). Reamostra para a proporção exata da carta.
    const norm = document.createElement("canvas");
    norm.width = cap.width;
    norm.height = Math.round(cap.width * 88 / 63);
    norm.getContext("2d")!.drawImage(cap, 0, 0, norm.width, norm.height);
    return norm;
  }

  // Prévia instantânea do disparo: a caixa do contorno rastreado (ou a retícula)
  // recortada do quadro do vídeo, reduzida. Só para a tela; não vai ao servidor.
  private quickPreview(quad?: Q.Quad): HTMLCanvasElement | null {
    const v = this.video;
    if (!v.videoWidth || !this.coords) return null;
    let x: number, y: number, w: number, h: number;
    if (quad) {
      ({ x, y, w, h } = Q.bbox(quad, v.videoWidth, v.videoHeight, 0.04));
    } else {
      ({ px: x, py: y, pw: w, ph: h } = this.coords);
    }
    if (w < 20 || h < 20) return null;
    const s = Math.min(1, 480 / Math.max(w, h));
    const c = document.createElement("canvas");
    c.width = Math.round(w * s);
    c.height = Math.round(h * s);
    c.getContext("2d")!.drawImage(v, x, y, w, h, 0, 0, c.width, c.height);
    return c;
  }

  /**
   * Sharp photo of the capture moment, in the video's field of view.
   * Uses ImageCapture.takePhoto() (refocus, full resolution) cropped to the
   * stream framing and capped at SHOT_MAX_DIM. Falls back to the video frame
   * when ImageCapture is missing (Firefox, iOS) or fails within 2.5s.
   * `k` = photo px per video px.
   */
  private async sharpFrame(): Promise<{ canvas: HTMLCanvasElement; k: number; still: boolean }> {
    // O quadro do vídeo fica mole de perto (o AF contínuo não trava macro) e a
    // carta deixava de ser reconhecida: as capturas saíam lavadas e borradas.
    const v = this.video, vw = v.videoWidth, vh = v.videoHeight;
    try {
      const track = this.stream?.getVideoTracks()[0];
      const IC = (window as unknown as { ImageCapture?: new (t: MediaStreamTrack) => { takePhoto(): Promise<Blob> } }).ImageCapture;
      if (track && IC) {
        if (!this.imageCapture) this.imageCapture = new IC(track);
        const blob = await Promise.race([
          this.imageCapture.takePhoto(),
          new Promise<never>((_, rej) => setTimeout(() => rej(new Error("still timeout")), 2500)),
        ]);
        const bmp = await createImageBitmap(blob);
        // O vídeo é um center-crop da foto: uma escala só + offsets centrando o
        // campo excedente. Escalar cada eixo separado recortava a região errada.
        const s = Math.min(bmp.width / vw, bmp.height / vh);
        const offX = (bmp.width - vw * s) / 2, offY = (bmp.height - vh * s) / 2;
        const k = Math.min(s, SHOT_MAX_DIM / Math.max(vw, vh));
        const c = document.createElement("canvas");
        c.width = Math.round(vw * k); c.height = Math.round(vh * k);
        c.getContext("2d")!.drawImage(bmp, offX, offY, vw * s, vh * s, 0, 0, c.width, c.height);
        bmp.close();
        return { canvas: c, k, still: true };
      }
    } catch { /* cai no quadro do vídeo */ }
    const c = document.createElement("canvas");
    c.width = vw; c.height = vh;
    c.getContext("2d")!.drawImage(v, 0, 0);
    return { canvas: c, k: 1, still: false };
  }

  private async capture(quad?: Q.Quad): Promise<void> {
    if (this.capturing || !this.coords) return;
    this.capturing = true;
    this.presentSince = 0;
    this.armed = false;
    this.disarmedAt = Date.now();
    if (this.sling) {
      this.lastAddData = this.lastData;
      this.movedSince = false;
    } else {
      // Resposta imediata: a foto nítida + o YOLO + o servidor levam ~2s, e sem
      // nada na tela o disparo parecia não ter acontecido ("lag quando
      // identifica"). Congela a imagem, vibra e já mostra a carta recortada do
      // vídeo; o recorte final troca a prévia quando fica pronto. No sling o
      // vídeo segue contínuo (esteira).
      navigator.vibrate?.(15);
      this.frame.classList.add("is-locked");
      setTimeout(() => this.frame.classList.remove("is-locked"), 400);
      this.video.pause();
      const preview = this.quickPreview(quad);
      if (preview) this.hooks.captured(preview);
    }
    // Todos os recortes saem da foto nítida (ver sharpFrame); `quad` e o
    // retângulo da retícula estão em px do vídeo, então escalam por `k`.
    const shot = await this.sharpFrame();
    const src = shot.canvas, k = shot.k;
    let photo: HTMLCanvasElement | null = null;
    let preCropped = false;
    // Sem contorno rastreado (mão tapando a borda, fundo claro): o YOLO ainda
    // acha a carta na foto, então é ele que dá o contorno do recorte.
    // `detected` evita rodar o detector de novo na mesma foto dentro do warp().
    let detected: Q.Quad | null | undefined;
    if (!quad && this.scanic) {
      try {
        const fc = detected = await this.detectPhoto(src);
        const q = fc ? Q.inset(Q.order(fc)) : null;
        if (q && Q.isSane(q, src.width, src.height)) quad = Q.scale(q, 1 / k);
      } catch { /* segue sem contorno */ }
    }
    if (quad) {
      let drop: Q.Quad | null = null;
      let used: Q.Quad | null = null;
      try { ({ photo, drop, used } = await this.warp(src, Q.scale(quad, k), detected)); } catch { photo = null; }
      if (photo && used) this.drawReticle(Q.scale(used, 1 / k));
      preCropped = !!photo;
      // Warp descartado: a carta está DENTRO do contorno que o usuário vê, não
      // necessariamente no corte fixo (que cortava as laterais de carta grande).
      // Recorta a caixa do contorno e deixa a detecção do servidor achar a carta.
      if (!photo && drop) {
        const b = Q.bbox(drop, src.width, src.height, 0.06);
        if (b.w >= 40 && b.h >= 40) {
          photo = document.createElement("canvas");
          photo.width = Math.round(b.w);
          photo.height = Math.round(b.h);
          photo.getContext("2d")!.drawImage(src, b.x, b.y, b.w, b.h, 0, 0, photo.width, photo.height);
        }
      }
    }
    if (!photo) {
      const { px, py, pw, ph } = this.coords;
      photo = document.createElement("canvas");
      photo.width = Math.max(1, Math.round(pw * k));
      photo.height = Math.max(1, Math.round(ph * k));
      photo.getContext("2d")!.drawImage(src, px * k, py * k, pw * k, ph * k, 0, 0, photo.width, photo.height);
    }
    if (this.sling) {
      navigator.vibrate?.(15);
      this.frame.classList.add("is-locked");
      setTimeout(() => this.frame.classList.remove("is-locked"), 400);
    } else {
      this.paused = true;
    }
    this.hooks.captured(photo);
    this.setStatus("Identificando carta…");
    let res: Recognition | null = null;
    try { res = await recognize(photo, preCropped); } catch { res = null; }
    this.hooks.result(res, photo);
  }
}
