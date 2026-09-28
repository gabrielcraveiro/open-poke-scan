import "./style.css";
import { Scanner } from "./scanner";
import { cardImage, isWarm, wakeServer, type Card, type Recognition } from "./recognize";
import { ligaUrl, mypUrl } from "./links";
import { formatPrice, formatPriceDetail, getPrice } from "./prices";
import * as session from "./session";
import { referrerHost, track } from "./telemetry";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

const video = $<HTMLVideoElement>("video");
const frame = $("frame");
const quadSvg = document.getElementById("quad") as unknown as SVGSVGElement;
const statusEl = $("status");
const progressBar = $("progress-bar");
const sheet = $("sheet");
const loading = $("loading");
const verify = $("verify");
const drawer = $("drawer");
const serverNote = $("server-note");

// Sling só adiciona sozinho com sinal forte. Sem isso, um frame ruim durante
// a troca de carta entrava na lista como a carta errada.
const SLING_MIN_COS = 0.62;
const SLING_KEY = "openpokescan.sling";

let current: Recognition | null = null;
let started = false;

function code(c: Card): string {
  return c.printed_total ? `#${c.number}/${c.printed_total}` : `#${c.number}`;
}

let toastTimer = 0;
function toast(msg: string): void {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => { t.hidden = true; }, 2200);
}

const scanner = new Scanner(video, frame, quadSvg, {
  status(msg) {
    if (statusEl.textContent !== msg) statusEl.textContent = msg;
  },
  progress(frac) {
    progressBar.style.transform = `scaleX(${frac.toFixed(3)})`;
  },
  captured(photo) {
    serverNote.hidden = isWarm();
    if (scanner.slingMode) return;
    ($<HTMLImageElement>("loading-img")).src = photo.toDataURL("image/jpeg", 0.6);
    loading.hidden = false;
  },
  result(res) {
    track("scan", {
      mode: scanner.slingMode ? "sling" : "hand",
      ok: !!res,
      confident: !!res?.confident,
      number_match: !!res?.numberMatch,
      cos: res?.card.cos ?? null,
      api_id: res?.card.api_id ?? null,
      top: res?.candidates.map((c) => c.api_id) ?? [],
      ms: res?.ms ?? null,
      warm: isWarm(),
    });
    serverNote.hidden = true;
    loading.hidden = true;
    if (scanner.slingMode) return onSlingResult(res);
    if (!res) {
      toast("Não consegui identificar. Tente de novo com a carta inteira no quadro.");
      scanner.resume();
      return;
    }
    showSheet(res);
  },
});

function onSlingResult(res: Recognition | null): void {
  const strong = !!res && res.confident && (res.numberMatch || res.card.cos >= SLING_MIN_COS);
  if (!res || !strong) {
    navigator.vibrate?.(25);
    track("sling_reject", { api_id: res?.card.api_id ?? null, cos: res?.card.cos ?? null });
    scanner.slingReject();
    return;
  }
  if (scanner.slingAccept(res) === "duplicate") return;
  session.add(res.card);
  track("add", { api_id: res.card.api_id, via: "sling" });
  navigator.vibrate?.([12, 40, 12]);
  showVerify(res.card);
}

let verifyTimer = 0;
function showVerify(card: Card): void {
  verify.replaceChildren();
  const img = document.createElement("img");
  img.src = cardImage(card);
  img.alt = "";
  const info = document.createElement("div");
  const name = document.createElement("strong");
  name.textContent = card.name;
  const meta = document.createElement("span");
  meta.textContent = `${card.set_name} ${code(card)}`;
  const price = document.createElement("span");
  price.className = "price";
  price.textContent = "…";
  info.append(name, meta, price);
  verify.append(img, info);
  verify.hidden = false;
  getPrice(card).then((p) => { price.textContent = formatPrice(p); });
  clearTimeout(verifyTimer);
  verifyTimer = window.setTimeout(() => { verify.hidden = true; }, 2600);
}

function fillSheet(card: Card, rec: Recognition): void {
  ($<HTMLImageElement>("sheet-img")).src = cardImage(card, "high");
  $("sheet-title").textContent = card.name;
  $("sheet-set").textContent = `${card.set_name} · ${code(card)}`;
  const priceEl = $("sheet-price");
  const detailEl = $("sheet-price-detail");
  priceEl.textContent = "Buscando preço…";
  detailEl.textContent = "";
  getPrice(card).then((p) => {
    if (!current || current.card.api_id !== card.api_id) return;
    priceEl.textContent = formatPrice(p);
    detailEl.textContent = formatPriceDetail(p);
  });
  const conf = card === rec.card && rec.numberMatch ? "Número conferido pelo OCR"
    : `Semelhança ${(card.cos * 100).toFixed(0)}%${rec.confident ? "" : " — confira se é essa"}`;
  $("sheet-confidence").textContent = conf;
  ($<HTMLAnchorElement>("link-liga")).href = ligaUrl(card);
  ($<HTMLAnchorElement>("link-myp")).href = mypUrl(card);
}

function showSheet(rec: Recognition): void {
  current = rec;
  fillSheet(rec.card, rec);
  renderAlternatives(rec);
  sheet.hidden = false;
}

function hideSheet(): void {
  sheet.hidden = true;
  current = null;
  scanner.resume();
}

// As outras candidatas ficam sempre à mostra: "Não é essa" volta direto para
// a câmera, e quem reconhece a carta certa aqui escolhe com um toque.
function renderAlternatives(rec: Recognition): void {
  const alts = $("sheet-alts");
  alts.replaceChildren();
  const others = rec.candidates.filter((c) => c.api_id !== rec.card.api_id).slice(0, 4);
  alts.hidden = !others.length;
  if (!others.length) return;
  const label = document.createElement("p");
  label.className = "muted small";
  label.textContent = "Ou é alguma destas?";
  alts.append(label);
  for (const c of others) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "alt";
    const img = document.createElement("img");
    img.src = cardImage(c);
    img.alt = "";
    const cap = document.createElement("span");
    cap.textContent = `${c.name} ${code(c)}`;
    b.append(img, cap);
    b.onclick = () => {
      track("alt_pick", { from: rec.card.api_id, to: c.api_id });
      const picked: Recognition = { ...rec, card: c, numberMatch: false, confident: true };
      current = picked;
      fillSheet(c, picked);
      renderAlternatives({ ...picked, candidates: [c, ...rec.candidates.filter((x) => x.api_id !== c.api_id)] });
    };
    alts.append(b);
  }
}

function renderList(): void {
  const items = session.all();
  $("list-count").textContent = String(items.length);
  $("list-empty").hidden = items.length > 0;
  const ul = $("list");
  ul.replaceChildren(...items.map((e) => {
    const li = document.createElement("li");
    const img = document.createElement("img");
    img.src = cardImage(e.card);
    img.alt = "";
    img.loading = "lazy";
    const info = document.createElement("div");
    const name = document.createElement("strong");
    name.textContent = e.card.name;
    const meta = document.createElement("span");
    meta.className = "muted small";
    meta.textContent = `${e.card.set_name} ${code(e.card)}`;
    const price = document.createElement("span");
    price.className = "price small";
    getPrice(e.card).then((p) => { price.textContent = p ? formatPrice(p) : ""; });
    const links = document.createElement("span");
    links.className = "small";
    const liga = document.createElement("a");
    liga.href = ligaUrl(e.card); liga.target = "_blank"; liga.rel = "noopener"; liga.textContent = "Liga ↗";
    const myp = document.createElement("a");
    myp.href = mypUrl(e.card); myp.target = "_blank"; myp.rel = "noopener"; myp.textContent = "MYP ↗";
    links.append(liga, " · ", myp);
    info.append(name, meta, price, links);
    const rm = document.createElement("button");
    rm.type = "button";
    rm.className = "icon-btn";
    rm.setAttribute("aria-label", `Remover ${e.card.name}`);
    rm.textContent = "✕";
    rm.onclick = () => session.remove(e.uid);
    li.append(img, info, rm);
    return li;
  }));
}

function download(name: string, text: string, type: string): void {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function setSlingUI(on: boolean): void {
  const b = $("sling-btn");
  b.setAttribute("aria-pressed", String(on));
  b.classList.toggle("on", on);
  document.body.classList.toggle("sling-on", on);
}

async function startCamera(): Promise<void> {
  $("start").hidden = true;
  statusEl.textContent = "Abrindo a câmera…";
  const err = await scanner.start();
  if (err) {
    statusEl.textContent = err;
    $("start").hidden = false;
    return;
  }
  started = true;
  $("torch-btn").hidden = !scanner.hasTorch();
}

// ── Eventos ─────────────────────────────────────────────────────────────
$("start-btn").onclick = () => void startCamera();
$("shutter").onclick = () => { if (started) scanner.captureNow(); };
$("again-btn").onclick = hideSheet;
$("wrong-btn").onclick = () => {
  if (current) track("wrong", { api_id: current.card.api_id, top: current.candidates.map((c) => c.api_id) });
  hideSheet();
  toast("Ok — aponte de novo para a carta");
};
$("link-liga").addEventListener("click", () => track("link", { store: "liga", api_id: current?.card.api_id }));
$("link-myp").addEventListener("click", () => track("link", { store: "myp", api_id: current?.card.api_id }));
$("add-btn").onclick = () => {
  if (!current) return;
  session.add(current.card);
  track("add", { api_id: current.card.api_id, via: "sheet" });
  toast(`✓ ${current.card.name} adicionada`);
  hideSheet();
};

const slingOn = localStorage.getItem(SLING_KEY) === "1";
scanner.setSling(slingOn);
setSlingUI(slingOn);
$("sling-btn").onclick = () => {
  const on = !scanner.slingMode;
  scanner.setSling(on);
  setSlingUI(on);
  localStorage.setItem(SLING_KEY, on ? "1" : "0");
  if (on && !sheet.hidden) hideSheet();
  toast(on ? "Sling ligado: adiciona sozinho e vai para a próxima" : "Sling desligado: você confirma cada carta");
};

let torchOn = false;
$("torch-btn").onclick = async () => {
  if (await scanner.setTorch(!torchOn)) {
    torchOn = !torchOn;
    $("torch-btn").setAttribute("aria-pressed", String(torchOn));
  }
};

$("list-btn").onclick = () => { drawer.hidden = false; };
$("drawer-close").onclick = () => { drawer.hidden = true; };
$("copy-btn").onclick = async () => {
  try {
    await navigator.clipboard.writeText(session.toText());
    track("export", { kind: "copy", n: session.all().length });
    toast("Lista copiada");
  } catch {
    toast("Não deu para copiar neste navegador");
  }
};
$("csv-btn").onclick = () => {
  track("export", { kind: "csv", n: session.all().length });
  download("cartas.csv", session.toCsv(), "text/csv;charset=utf-8");
};
$("clear-btn").onclick = () => {
  if (!session.all().length) return;
  const btn = $("clear-btn");
  // Dois toques em vez de confirm(): diálogo nativo trava a câmera em alguns WebViews.
  if (btn.dataset.armed === "1") {
    session.clear();
    btn.dataset.armed = "";
    btn.textContent = "Limpar";
  } else {
    btn.dataset.armed = "1";
    btn.textContent = "Toque de novo para limpar";
    setTimeout(() => { btn.dataset.armed = ""; btn.textContent = "Limpar"; }, 3000);
  }
};

const feedback = import.meta.env.VITE_FEEDBACK_URL;
if (feedback) {
  const a = $<HTMLAnchorElement>("feedback-link");
  a.href = feedback;
  a.hidden = false;
}

// Volta do background: o Android costuma manter um track "vivo" que pinta
// preto. Recriar a câmera é o único jeito confiável.
document.addEventListener("visibilitychange", () => {
  if (!document.hidden && started) void scanner.restart();
});

session.onChange(renderList);
renderList();
// Acorda o servidor já na abertura da página: o cold start (~20s) passa
// enquanto a pessoa lê a tela e libera a câmera.
wakeServer();
track("open", {
  ref: referrerHost(),
  mobile: /Mobi|Android|iPhone/i.test(navigator.userAgent),
  lang: navigator.language,
  sling: scanner.slingMode,
});
// A máquina suspende depois de alguns minutos ociosa. Pingar enquanto a
// câmera está aberta evita pagar o cold start no meio de um lote.
setInterval(() => { if (started && !document.hidden) wakeServer(); }, 150_000);
