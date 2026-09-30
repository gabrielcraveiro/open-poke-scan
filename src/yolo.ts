// Detector de cantos YOLOv8-pose (duclvQ/tcg-card-detector, AGPL-3.0): prevê
// direto os 4 cantos da carta, em vez de procurar o maior retângulo. Treinado
// com fotos de cartas Pokémon. Bancada com os 13 quadros reais da coleta: 13/13
// na borda externa, contra 12/13 do scanic ML (que põe os cantos para dentro).
// Quadro vazio e jeans dão score 0.0; carta, ~0.97.
//
// Roda uma vez por captura, na foto (ver Scanner.detectPhoto). Baixa em segundo
// plano ao abrir o site: ~12.5 MB de modelo (fp32; o int8 desviava os cantos
// até 3% do quadro) + ~2.5 MB de WASM do onnxruntime-web.
import type * as Ort from "onnxruntime-web";

// Fixado no commit para o modelo não mudar por baixo do app.
const MODEL_URL =
  "https://huggingface.co/duclvQ/tcg-card-detector/resolve/419633c3986296bf2e3df93ce94d28b1a3fb965d/card_pose_480.onnx";
// A mesma versão do package.json; o WASM vem do CDN para não entrar no bundle.
const WASM_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/";
const SIZE = 480;
/** Minimum card score. Empty frame and jeans give 0.0; real cards give ~0.97. */
export const YOLO_MIN_SCORE = 0.5;

type Corner = { x: number; y: number };
/** The four card corners in the pixels of the input canvas, in model order. */
export interface YoloCorners {
  topLeft: Corner;
  topRight: Corner;
  bottomRight: Corner;
  bottomLeft: Corner;
  score: number;
}

let loading: Promise<{ ort: typeof Ort; session: Ort.InferenceSession }> | null = null;
let ready: { ort: typeof Ort; session: Ort.InferenceSession } | null = null;

/** Start the model download in the background. Safe to call many times. */
export function preloadYolo(): void {
  load().then((m) => { ready = m; }).catch(() => {});
}

function load() {
  if (!loading) {
    loading = (async () => {
      const ort = await import("onnxruntime-web/wasm");
      ort.env.wasm.wasmPaths = WASM_BASE;
      // Sem cross-origin isolation não há SharedArrayBuffer: multithread falharia.
      ort.env.wasm.numThreads = 1;
      const session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] });
      return { ort, session };
    })();
    loading.catch(() => { loading = null; });
  }
  return loading;
}

const letterbox = typeof document !== "undefined" ? document.createElement("canvas") : null;

/**
 * Find the card corners in `src`.
 *
 * @param src Canvas with the frame (any size; it is letterboxed to 480×480).
 * @returns The corners in `src` pixels and the card score, or null while the
 *   model downloads (see preloadYolo). Check `score` against YOLO_MIN_SCORE:
 *   the model always returns its best guess.
 */
export async function detectYolo(src: HTMLCanvasElement): Promise<YoloCorners | null> {
  // Não espera o download (~15 MB): o rastreio segue sem contorno até ficar pronto.
  if (!ready) { preloadYolo(); return null; }
  const { ort, session } = ready;
  const w = src.width, h = src.height;
  const r = SIZE / Math.max(w, h);
  const nw = Math.round(w * r), nh = Math.round(h * r);
  const px = Math.floor((SIZE - nw) / 2), py = Math.floor((SIZE - nh) / 2);
  const c = letterbox!;
  c.width = SIZE; c.height = SIZE;
  const ctx = c.getContext("2d", { willReadFrequently: true })!;
  // Cinza 114 = o preenchimento do letterbox do treino do Ultralytics.
  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, SIZE, SIZE);
  ctx.drawImage(src, 0, 0, w, h, px, py, nw, nh);
  const rgba = ctx.getImageData(0, 0, SIZE, SIZE).data;
  const n = SIZE * SIZE;
  const chw = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    chw[i] = rgba[i * 4] / 255;
    chw[n + i] = rgba[i * 4 + 1] / 255;
    chw[2 * n + i] = rgba[i * 4 + 2] / 255;
  }
  const out = await session.run({ images: new ort.Tensor("float32", chw, [1, 3, SIZE, SIZE]) });
  // Saída [1, 17, A]: caixa (4) + score (1) + 4 cantos × (x, y, visibilidade).
  const o = out[session.outputNames[0]];
  const d = o.data as Float32Array;
  const anchors = o.dims[2];
  let best = 0;
  for (let a = 1; a < anchors; a++) if (d[4 * anchors + a] > d[4 * anchors + best]) best = a;
  const kp = (k: number): Corner => ({
    x: (d[(5 + k * 3) * anchors + best] - px) / r,
    y: (d[(6 + k * 3) * anchors + best] - py) / r,
  });
  return { topLeft: kp(0), topRight: kp(1), bottomRight: kp(2), bottomLeft: kp(3), score: d[4 * anchors + best] };
}
