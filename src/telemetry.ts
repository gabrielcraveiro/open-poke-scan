// Telemetria anônima de uso. Desligada por padrão: só envia quando o build
// define VITE_TELEMETRY_URL. Um fork sem essa variável não manda nada.
//
// O que vai: um ID aleatório por aparelho (localStorage), o nome do evento e
// dados do scan (cartas candidatas, confiança, tempo). Não vai foto, IP nem
// nada que identifique a pessoa.

const URL_ = import.meta.env.VITE_TELEMETRY_URL || "";
const DEVICE_KEY = "openpokescan.device";

/** Event names accepted by the telemetry endpoint. */
export type EventName = "open" | "scan" | "add" | "wrong" | "skip" | "alt_pick" | "sling_reject" | "link" | "export";

function deviceId(): string {
  try {
    let id = localStorage.getItem(DEVICE_KEY);
    if (!id) {
      id = crypto.randomUUID();
      localStorage.setItem(DEVICE_KEY, id);
    }
    return id;
  } catch {
    return "";
  }
}

/**
 * Send one usage event. Fire-and-forget: never throws and never blocks the scan.
 *
 * @param event Event name.
 * @param data Small JSON object (the server drops bodies over 4KB).
 */
export function track(event: EventName, data: Record<string, unknown> = {}): void {
  if (!URL_) return;
  try {
    const body = JSON.stringify({ device_id: deviceId(), event, data });
    // Texto puro no sendBeacon evita o preflight de CORS e sobrevive ao fechar a aba.
    if (!navigator.sendBeacon?.(URL_, body)) {
      fetch(URL_, { method: "POST", body, keepalive: true, headers: { "Content-Type": "text/plain" } }).catch(() => {});
    }
  } catch {
    /* telemetria nunca derruba o app */
  }
}

/** Referrer host only (for example `reddit.com`), or null for direct visits. */
export function referrerHost(): string | null {
  try {
    const params = new URLSearchParams(location.search);
    const utm = params.get("utm_source") || params.get("ref");
    if (utm) return utm.slice(0, 40);
    return document.referrer ? new URL(document.referrer).hostname.replace(/^www\./, "") : null;
  } catch {
    return null;
  }
}
