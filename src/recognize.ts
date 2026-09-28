// Cliente do servidor de reconhecimento (FastAPI, POST /recognize).

export const RECOGNIZE_URL = (import.meta.env.VITE_RECOGNIZE_URL || "https://cartinhas-recognize.fly.dev")
  .replace(/\/+$/, "");

/** One catalog card, as returned by the recognition server. */
export interface Card {
  api_id: string;
  name: string;
  number: string;
  set_id: string;
  set_name: string;
  printed_total: number | null;
  image_url: string;
  /** Cosine similarity to the photo, 0..1. */
  cos: number;
}

/** Result of one recognition call. */
export interface Recognition {
  card: Card;
  /** True when the OCR number+total matched or the cosine is high. */
  confident: boolean;
  /** True when the OCR number+total picked the card. */
  numberMatch: boolean;
  /** Top candidates, best first. `card` is always in the list. */
  candidates: Card[];
  ms: number;
}

// Orçamento do POST: a máquina do Fly suspende quando ociosa. Acordada do
// zero leva ~20s; quente responde em <1s. Timeout curto com máquina fria
// reprovava todo scan, então o teto depende do estado.
const POST_WARM_MS = 12000;
const POST_COLD_MS = 40000;
const UPLOAD_MAX_DIM = 1280;

let warm = false;

export function isWarm(): boolean {
  return warm;
}

/** Wake the server early. Call it when the page opens. Never throws. */
export function wakeServer(): void {
  fetch(RECOGNIZE_URL + "/health").then((r) => { if (r.ok) warm = true; }).catch(() => {});
}

function downscale(canvas: HTMLCanvasElement): HTMLCanvasElement {
  const long = Math.max(canvas.width, canvas.height);
  if (long <= UPLOAD_MAX_DIM) return canvas;
  const s = UPLOAD_MAX_DIM / long;
  const out = document.createElement("canvas");
  out.width = Math.round(canvas.width * s);
  out.height = Math.round(canvas.height * s);
  out.getContext("2d")!.drawImage(canvas, 0, 0, out.width, out.height);
  return out;
}

// Sem o teto, o callback do toBlob às vezes nunca vem em WebView Android
// com pouca memória, e o scanner congelava esperando.
function toJpeg(canvas: HTMLCanvasElement, quality = 0.8): Promise<Blob | null> {
  return new Promise((resolve) => {
    const t = setTimeout(() => resolve(null), 5000);
    try {
      canvas.toBlob((b) => { clearTimeout(t); resolve(b); }, "image/jpeg", quality);
    } catch {
      clearTimeout(t);
      resolve(null);
    }
  });
}

async function post(blob: Blob, preCropped: boolean, budgetMs: number) {
  const fd = new FormData();
  fd.append("file", blob, "scan.jpg");
  if (preCropped) fd.append("pre", "1");
  const ctl = new AbortController();
  let timedOut = false;
  const t0 = Date.now();
  const t = setTimeout(() => { timedOut = true; ctl.abort(); }, budgetMs);
  try {
    const r = await fetch(RECOGNIZE_URL + "/recognize", { method: "POST", body: fd, signal: ctl.signal });
    const data = r.ok ? await r.json() : null;
    return { data, ms: Date.now() - t0, timedOut };
  } catch {
    return { data: null, ms: Date.now() - t0, timedOut };
  } finally {
    clearTimeout(t);
  }
}

/**
 * Send a card photo to the server and return the identified card.
 *
 * @param capture Photo of the card. With `preCropped`, the card fills the whole image.
 * @param preCropped Set to true when the client already cropped the card (skips server detection).
 * @returns The recognition, or null on network error, timeout or no match.
 */
export async function recognize(capture: HTMLCanvasElement, preCropped: boolean): Promise<Recognition | null> {
  const blob = await toJpeg(downscale(capture));
  if (!blob) return null;
  const budget = warm ? POST_WARM_MS : POST_COLD_MS;
  let r = await post(blob, preCropped, budget);
  // Reenvia só em falha rápida. Reenviar depois de timeout dobra a carga
  // quando o servidor já está afogado (1 vCPU) e colapsa a fila.
  if (!r.data && !r.timedOut && r.ms < 3000) {
    await new Promise((res) => setTimeout(res, 350));
    r = await post(blob, preCropped, budget);
  }
  const d = r.data;
  warm = !!(d && d.card);
  if (!d || !d.card) return null;
  const candidates: Card[] = Array.isArray(d.candidates) ? d.candidates : [];
  if (!candidates.some((c) => c.api_id === d.card.api_id)) candidates.unshift(d.card);
  return {
    card: d.card,
    confident: !!d.confident,
    numberMatch: !!(d.ocr && d.ocr.match),
    candidates,
    ms: r.ms,
  };
}

// O catálogo mistura URLs do TCGdex (com ou sem /low.webp no fim) e URLs
// completas de .jpg da Liga (usar como estão: sufixo em .jpg dá 404).
export function cardImage(card: Pick<Card, "image_url">, quality: "low" | "high" = "low"): string {
  const u = card.image_url || "";
  if (!u) return "";
  if (u.includes("assets.tcgdex.net")) return `${u.replace(/\/(low|high)\.(webp|png|jpg)$/i, "")}/${quality}.webp`;
  return u;
}
