// Links de busca nas lojas brasileiras. Só montam a URL de busca: nenhum dado
// é lido desses sites. O código da carta segue o formato que a Liga indexa
// (verificado página a página no CartinhasDaJu).
import type { Card } from "./recognize";

// Subconjuntos "RC" que a Liga indexa com contagem própria.
const RC_SUBSET_TOTALS: Record<string, number> = { g1: 32, bw11: 25 };
// Sets em que só o número é ambíguo: a Liga precisa do &ed= para achar a página.
const LIGA_EDITION_CODES: Record<string, string> = {
  g1: "CP3",
  svp: "SVP",
  mep: "MEP",
  me03: "POR",
  sv8a: "SV8a",
  sv4a: "SV4a",
  swsh10: "ASR",
  "swsh12.5": "CRZ",
};
const JP_RE = /[぀-ヿ㐀-鿿]/;

function pad3(n: string | number, width: number): string {
  return String(n).padStart(width, "0");
}

function ligaCardCode(number: string, printedTotal: number | null, setId: string): string {
  if (!number) return "";
  const sid = setId.toLowerCase();
  if (/^RC/i.test(number)) {
    const rcTotal = RC_SUBSET_TOTALS[sid];
    return rcTotal ? `${number}/RC${rcTotal}` : number;
  }
  if (!printedTotal) return "";
  // Promo com letras (SWSH266, XY-P): a Liga guarda sem o total.
  if (!/^\d+$/.test(number)) return number;
  const threeDigits = /^z?sv/.test(sid) || /^me/.test(sid) || ["swsh10", "swsh12.5"].includes(sid);
  if (threeDigits) {
    const d = Math.max(number.length, 3);
    return `${pad3(number, d)}/${pad3(printedTotal, d)}`;
  }
  if (/^swsh/.test(sid)) return `${parseInt(number, 10)}/${printedTotal}`;
  return `${number}/${pad3(printedTotal, number.length)}`;
}

/** Build the Liga Pokémon search URL for a card. */
export function ligaUrl(card: Card): string {
  const sid = (card.set_id || "").toLowerCase();
  const ed = LIGA_EDITION_CODES[sid];
  const edParam = ed ? `&ed=${ed}` : "";
  const base = "https://www.ligapokemon.com.br/?view=cards/card&card=";
  if (card.number && /^RC/i.test(card.number)) {
    const rcTotal = RC_SUBSET_TOTALS[sid];
    const rcNum = parseInt(card.number.replace(/^RC/i, ""), 10);
    if (ed && rcTotal && !isNaN(rcNum)) {
      const numStr = pad3(rcNum, 3);
      const query = `${card.name} (${numStr}/${pad3(rcTotal, 3)})`;
      return `${base}${encodeURIComponent(query)}&ed=${ed}&num=${numStr}`;
    }
  }
  const code = ligaCardCode(card.number, card.printed_total, sid);
  // A Liga não indexa nomes em japonês: busca só pelo código.
  if (JP_RE.test(card.name || "")) return `${base}${encodeURIComponent(code)}${edParam}`;
  const query = code ? `${card.name} (${code})` : card.name;
  return `${base}${encodeURIComponent(query)}${edParam}`;
}

/** Build the MYP Cards search URL for a card. */
export function mypUrl(card: Card): string {
  const n = card.number ? String(card.number).trim() : "";
  const t = card.printed_total ? String(card.printed_total).trim() : "";
  const numberPart = n && t ? `(${n}/${t})` : n;
  const q = `${(card.name || "").trim()} ${numberPart}`.replace(/\s+/g, " ").trim();
  const p = new URLSearchParams();
  p.set("ProdutoSearch[marca]", "pokemon");
  p.set("ProdutoSearch[query]", ` ${q} `);
  return `https://mypcards.com/pokemon?${p.toString()}`;
}
