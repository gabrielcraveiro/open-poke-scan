// Preço de referência vindo da API pública do TCGdex (Cardmarket em EUR e
// TCGplayer em USD). Cartas fora do TCGdex (sets semeados só da Liga) ficam sem preço.

/** Reference prices for one card. Any field can be missing. */
export interface Price {
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

async function load(apiId: string): Promise<Price | null> {
  try {
    const r = await fetch(`https://api.tcgdex.net/v2/en/cards/${encodeURIComponent(apiId)}`);
    if (!r.ok) return null;
    const pricing = (await r.json())?.pricing;
    if (!pricing) return null;
    const cm = pricing.cardmarket || {};
    const price: Price = { eur: firstNumber(cm.trend, cm.avg, cm["trend-holo"], cm["avg-holo"]) };
    const tp = pricing.tcgplayer || {};
    for (const [key, variant] of Object.entries(tp)) {
      if (key === "unit" || key === "updated" || !variant || typeof variant !== "object") continue;
      const v = variant as Record<string, unknown>;
      const usd = firstNumber(v.marketPrice, v.midPrice);
      if (usd) { price.usd = usd; break; }
    }
    return price.eur || price.usd ? price : null;
  } catch {
    return null;
  }
}

/**
 * Get reference prices for a card from TCGdex.
 *
 * @param apiId TCGdex card id, for example `sv02-199`.
 * @returns The prices, or null when TCGdex has no price or the call fails. Never throws.
 */
export function getPrice(apiId: string): Promise<Price | null> {
  let p = cache.get(apiId);
  if (!p) {
    p = load(apiId);
    cache.set(apiId, p);
  }
  return p;
}

export function formatPrice(p: Price | null): string {
  if (!p) return "sem preço de referência";
  const parts: string[] = [];
  if (p.eur) parts.push(p.eur.toLocaleString("pt-BR", { style: "currency", currency: "EUR" }));
  if (p.usd) parts.push(p.usd.toLocaleString("pt-BR", { style: "currency", currency: "USD" }));
  return parts.join(" · ");
}
