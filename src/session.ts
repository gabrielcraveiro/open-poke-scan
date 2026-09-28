// Lista de cartas escaneadas. Fica só no aparelho (localStorage): não há conta nem servidor.
import type { Card } from "./recognize";

/** One scanned card in the list. */
export interface Entry {
  uid: number;
  card: Card;
  at: number;
}

const KEY = "openpokescan.session.v1";
let entries: Entry[] = load();
let seq = entries.reduce((m, e) => Math.max(m, e.uid), 0);
const listeners = new Set<() => void>();

function load(): Entry[] {
  try {
    const raw = localStorage.getItem(KEY);
    const v = raw ? JSON.parse(raw) : [];
    return Array.isArray(v) ? v : [];
  } catch {
    return [];
  }
}

function save(): void {
  try { localStorage.setItem(KEY, JSON.stringify(entries)); } catch { /* cota cheia: a lista segue em memória */ }
  listeners.forEach((fn) => fn());
}

export function onChange(fn: () => void): void {
  listeners.add(fn);
}

export function all(): readonly Entry[] {
  return entries;
}

export function add(card: Card): Entry {
  const e = { uid: ++seq, card, at: Date.now() };
  entries = [e, ...entries];
  save();
  return e;
}

export function remove(uid: number): void {
  entries = entries.filter((e) => e.uid !== uid);
  save();
}

export function clear(): void {
  entries = [];
  save();
}

function code(c: Card): string {
  return c.printed_total ? `${c.number}/${c.printed_total}` : c.number;
}

/** Plain-text list, one card per line. Good for pasting in a chat or post. */
export function toText(): string {
  return entries
    .slice()
    .reverse()
    .map((e) => `${e.card.name} — ${e.card.set_name} ${code(e.card)}`)
    .join("\n");
}

/** CSV with one row per scanned card. */
export function toCsv(): string {
  const esc = (v: unknown) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const head = ["name", "set", "set_id", "number", "printed_total", "tcgdex_id"];
  const rows = entries.slice().reverse().map((e) =>
    [e.card.name, e.card.set_name, e.card.set_id, e.card.number, e.card.printed_total, e.card.api_id].map(esc).join(","));
  return [head.join(","), ...rows].join("\n");
}
