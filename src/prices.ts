// Preço de referência. Em real: índice de preços do Brasil. Em EUR/USD:
// Cardmarket e TCGplayer via API pública do TCGdex. As duas chamadas passam
// pelos rewrites do vercel.json (mesma origem), porque a edge do TCGdex às
// vezes responde sem o header de CORS e o navegador bloqueia.
import type { Card } from "./recognize";

/** Reference prices for one card. Any field can be missing. */
export interface Price {
  /** Brazilian market price, BRL. */
  brl?: { low?: number; mid?: number };
  /** Cardmarket trend price, EUR. */
  eur?: number;
  /** TCGplayer market price, USD (first variant with a price). */
  usd?: number;
}

const cache = new Map<string, Promise<Price | null>>();

function firstNumber(...vals: unknown[]): number | undefined {
  for (const v of vals) if (typeof v === "number" && v > 0) return v;
  return undefined;
}

async function loadBrl(card: Card): Promise<Price["brl"]> {
  try {
    const q = new URLSearchParams({ set_id: card.set_id, number: card.number });
    const r = await fetch(`/api/price-brl?${q}`);
    if (!r.ok) return undefined;
    const d = await r.json();
    const low = firstNumber(d.low), mid = firstNumber(d.mid);
    return low || mid ? { low, mid } : undefined;
  } catch {
    return undefined;
  }
}

async function loadIntl(apiId: string): Promise<Pick<Price, "eur" | "usd">> {
  try {
    const r = await fetch(`/api/tcgdex/en/cards/${encodeURIComponent(apiId)}`);
    if (!r.ok) return {};
    const pricing = (await r.json())?.pricing;
    if (!pricing) return {};
    const cm = pricing.cardmarket || {};
    const out: Pick<Price, "eur" | "usd"> = { eur: firstNumber(cm.trend, cm.avg, cm["trend-holo"], cm["avg-holo"]) };
    for (const [key, variant] of Object.entries(pricing.tcgplayer || {})) {
      if (key === "unit" || key === "updated" || !variant || typeof variant !== "object") continue;
      const v = variant as Record<string, unknown>;
      const usd = firstNumber(v.marketPrice, v.midPrice);
      if (usd) { out.usd = usd; break; }
    }
    return out;
  } catch {
    return {};
  }
}

async function load(card: Card): Promise<Price | null> {
  const [brl, intl] = await Promise.all([loadBrl(card), loadIntl(card.api_id)]);
  const price: Price = { brl, ...intl };
  return price.brl || price.eur || price.usd ? price : null;
}

/**
 * Get reference prices for a card.
 *
 * @param card The recognized card.
 * @returns The prices, or null when no source has a price. Never throws.
 */
export function getPrice(card: Card): Promise<Price | null> {
  let p = cache.get(card.api_id);
  if (!p) {
    p = load(card);
    cache.set(card.api_id, p);
  }
  return p;
}

const fmt = (v: number, currency: string) => v.toLocaleString("pt-BR", { style: "currency", currency });

/** Main price line: BRL when available, otherwise EUR/USD. */
export function formatPrice(p: Price | null): string {
  if (!p) return "sem preço de referência";
  if (p.brl?.mid) return fmt(p.brl.mid, "BRL");
  if (p.brl?.low) return fmt(p.brl.low, "BRL");
  return formatIntl(p) || "sem preço de referência";
}

/** Secondary line: BRL minimum plus international prices. */
export function formatPriceDetail(p: Price | null): string {
  if (!p) return "";
  const parts: string[] = [];
  if (p.brl?.mid && p.brl.low) parts.push(`no Brasil a partir de ${fmt(p.brl.low, "BRL")}`);
  if (p.brl) {
    const intl = formatIntl(p);
    if (intl) parts.push(intl);
  }
  return parts.join(" · ");
}

function formatIntl(p: Price): string {
  const parts: string[] = [];
  if (p.eur) parts.push(fmt(p.eur, "EUR"));
  if (p.usd) parts.push(fmt(p.usd, "USD"));
  return parts.join(" · ");
}
