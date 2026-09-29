// Preço de referência em real, do índice de preços do mercado brasileiro.
// A chamada passa pelo rewrite do vercel.json (mesma origem, sem CORS).
import type { Card } from "./recognize";

/** Reference price for one card, BRL. Any field can be missing. */
export interface Price {
  /** Lowest listed price. */
  low?: number;
  /** Average price. */
  mid?: number;
}

const cache = new Map<string, Promise<Price | null>>();

function positive(v: unknown): number | undefined {
  return typeof v === "number" && v > 0 ? v : undefined;
}

async function load(card: Card): Promise<Price | null> {
  try {
    const q = new URLSearchParams({ set_id: card.set_id, number: card.number });
    const r = await fetch(`/api/price-brl?${q}`);
    if (!r.ok) return null;
    const d = await r.json();
    const price: Price = { low: positive(d.low), mid: positive(d.mid) };
    return price.low || price.mid ? price : null;
  } catch {
    return null;
  }
}

/**
 * Get the reference price of a card, in BRL.
 *
 * @param card The recognized card.
 * @returns The price, or null when there is no price or the call fails. Never throws.
 */
export function getPrice(card: Card): Promise<Price | null> {
  let p = cache.get(card.api_id);
  if (!p) {
    p = load(card);
    cache.set(card.api_id, p);
  }
  return p;
}

const brl = (v: number) => v.toLocaleString("pt-BR", { style: "currency", currency: "BRL" });

/** Main price line: the average price, or the lowest when there is no average. */
export function formatPrice(p: Price | null): string {
  const v = p?.mid ?? p?.low;
  return v ? brl(v) : "sem preço de referência";
}

/** Secondary line: the lowest price, when it differs from the main line. */
export function formatPriceDetail(p: Price | null): string {
  return p?.mid && p.low ? `a partir de ${brl(p.low)}` : "";
}
