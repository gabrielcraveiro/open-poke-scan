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
// Ordem cíclica pelo ângulo em torno do centro (nunca repete canto, nunca
// espelha — os extremos de x±y antigos perdiam um canto perto de 45°); topo e
// base = o par de lados MAIS HORIZONTAL, e topo = o de cima. Até 45° dá os
// mesmos rótulos da versão original. A regra anterior ("lado curto em cima")
// errava com a carta inclinada para trás na mão: a perspectiva encurta a
// altura, um lado vertical virava "topo" e a retícula não travava. Acima de
// 45° o contorno sai em paisagem e isSane() rejeita (vai pelo corte fixo).
export function order(c: Quad): Quad {
  const pts = KEYS.map((k) => c[k]);
  const cx = (pts[0].x + pts[1].x + pts[2].x + pts[3].x) / 4;
  const cy = (pts[0].y + pts[1].y + pts[2].y + pts[3].y) / 4;
  // y cresce para baixo: ângulo crescente = sentido horário na tela (TL→TR→BR→BL).
  const cyc = pts.slice().sort((a, b) => Math.atan2(a.y - cy, a.x - cx) - Math.atan2(b.y - cy, b.x - cx));
  const hor = (a: Point, b: Point) => Math.abs(b.x - a.x) - Math.abs(b.y - a.y);
  const s = hor(cyc[0], cyc[1]) + hor(cyc[2], cyc[3]) >= hor(cyc[1], cyc[2]) + hor(cyc[3], cyc[0]) ? 0 : 1;
  const midY = (i: number) => (cyc[i % 4].y + cyc[(i + 1) % 4].y) / 2;
  const start = midY(s) <= midY(s + 2) ? s : s + 2;
  const at = (i: number) => cyc[(start + i) % 4];
  return { topLeft: at(0), topRight: at(1), bottomRight: at(2), bottomLeft: at(3) };
}

/** Axis-aligned box around the quad, grown by `pad` on each side and clamped to w×h. */
export function bbox(c: Quad, w: number, h: number, pad: number): { x: number; y: number; w: number; h: number } {
  const xs = KEYS.map((k) => c[k].x), ys = KEYS.map((k) => c[k].y);
  let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const px = (x1 - x0) * pad, py = (y1 - y0) * pad;
  x0 = Math.max(0, x0 - px); y0 = Math.max(0, y0 - py);
  x1 = Math.min(w, x1 + px); y1 = Math.min(h, y1 + py);
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
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
  // Só retrato: contorno em paisagem (topo/base = lados horizontais, ver
  // order()) é região interna da carta — caixa de ataque, faixa do nome — cujo
  // warp mandava meia carta de lado, ou carta inclinada mais de 45°.
  if (aspect < 0.55 || aspect > 0.95) return false;
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
