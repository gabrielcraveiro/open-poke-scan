// Geometria do contorno da carta (quad) — portado do scanner do CartinhasDaJu.
// Os limiares vieram de telemetria real; mexer neles muda acerto, não só visual.
import type { CornerPoints, Point } from "scanic";

export type Quad = CornerPoints;

const KEYS = ["topLeft", "topRight", "bottomRight", "bottomLeft"] as const;

// Proporção aceita para o warp (largura/altura, em retrato). Acertos medidos
// ficam em 0.658–0.796; warps tortos (pedaço da carta) vão de 0.535 a 1.231.
export const WARP_ASPECT_MIN = 0.62;
export const WARP_ASPECT_MAX = 0.85;

export function avgDelta(a: Quad, b: Quad): number {
  let sum = 0;
  for (const k of KEYS) sum += Math.hypot(a[k].x - b[k].x, a[k].y - b[k].y);
  return sum / 4;
}

// Re-rotula os cantos por geometria: com a carta na diagonal os rótulos do
// detector trocam entre frames, e um warp com rótulos trocados sai cisalhado.
export function order(c: Quad): Quad {
  const pts = KEYS.map((k) => c[k]);
  let tl = pts[0], br = pts[0], tr = pts[0], bl = pts[0];
  for (const p of pts) {
    if (p.x + p.y < tl.x + tl.y) tl = p;
    if (p.x + p.y > br.x + br.y) br = p;
    if (p.x - p.y > tr.x - tr.y) tr = p;
    if (p.x - p.y < bl.x - bl.y) bl = p;
  }
  return { topLeft: tl, topRight: tr, bottomRight: br, bottomLeft: bl };
}

// Rejeita detecções-lixo: lados opostos díspares, área irrisória ou tela
// inteira, e proporção impossível para uma carta (0.716 em retrato).
// Quase-quadrado nunca é carta: é o detector agarrando uma aresta interna.
export function isSane(c: Quad, w: number, h: number): boolean {
  const d = (a: Point, b: Point) => Math.hypot(a.x - b.x, a.y - b.y);
  const top = d(c.topLeft, c.topRight), bottom = d(c.bottomLeft, c.bottomRight);
  const left = d(c.topLeft, c.bottomLeft), right = d(c.topRight, c.bottomRight);
  if (!top || !bottom || !left || !right) return false;
  const rH = top / bottom, rV = left / right;
  if (rH < 0.6 || rH > 1.67 || rV < 0.6 || rV > 1.67) return false;
  const avgW = (top + bottom) / 2, avgH = (left + right) / 2;
  const aspect = avgW / avgH;
  const portrait = aspect >= 0.55 && aspect <= 0.95;
  const landscape = aspect >= 1.05 && aspect <= 1.82;
  if (!portrait && !landscape) return false;
  const area = avgW * avgH;
  if (area < w * h * 0.015 || area > w * h * 0.95) return false;
  for (const k of KEYS) {
    const p = c[k];
    if (p.x < -w * 0.05 || p.x > w * 1.05 || p.y < -h * 0.05 || p.y > h * 1.05) return false;
  }
  return true;
}

// O contorno do detector marca a borda externa (dilatação do Canny) e fica um
// pouco fora da carta. Encolher ~2% alinha o visual e tira a lasca de fundo.
export function inset(c: Quad, f = 0.02): Quad {
  let cx = 0, cy = 0;
  for (const k of KEYS) { cx += c[k].x; cy += c[k].y; }
  cx /= 4; cy /= 4;
  const out = {} as Quad;
  for (const k of KEYS) out[k] = { x: c[k].x + (cx - c[k].x) * f, y: c[k].y + (cy - c[k].y) * f };
  return out;
}

export function lerp(a: Quad, b: Quad, t: number): Quad {
  const out = {} as Quad;
  for (const k of KEYS) out[k] = { x: a[k].x + (b[k].x - a[k].x) * t, y: a[k].y + (b[k].y - a[k].y) * t };
  return out;
}

export function scale(c: Quad, s: number): Quad {
  const out = {} as Quad;
  for (const k of KEYS) out[k] = { x: c[k].x * s, y: c[k].y * s };
  return out;
}

export function fromRect(x: number, y: number, w: number, h: number): Quad {
  return {
    topLeft: { x, y },
    topRight: { x: x + w, y },
    bottomRight: { x: x + w, y: y + h },
    bottomLeft: { x, y: y + h },
  };
}

// Path SVG com cantos arredondados, como a própria carta.
export function roundedPath(pts: Point[]): string {
  const n = pts.length;
  const d: string[] = [];
  for (let i = 0; i < n; i++) {
    const p = pts[i], prev = pts[(i + n - 1) % n], next = pts[(i + 1) % n];
    const lp = Math.hypot(p.x - prev.x, p.y - prev.y) || 1;
    const ln = Math.hypot(next.x - p.x, next.y - p.y) || 1;
    const r = Math.min(18, lp * 0.3, ln * 0.3);
    const inX = p.x - ((p.x - prev.x) / lp) * r, inY = p.y - ((p.y - prev.y) / lp) * r;
    const outX = p.x + ((next.x - p.x) / ln) * r, outY = p.y + ((next.y - p.y) / ln) * r;
    d.push(`${i ? "L" : "M"}${inX.toFixed(1)},${inY.toFixed(1)}`,
      `Q${p.x.toFixed(1)},${p.y.toFixed(1)} ${outX.toFixed(1)},${outY.toFixed(1)}`);
  }
  return d.join(" ") + " Z";
}

export function points(c: Quad): Point[] {
  return KEYS.map((k) => c[k]);
}
