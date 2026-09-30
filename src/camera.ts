// Escolha da lente e do foco. Em celular com várias câmeras traseiras, o
// facingMode "environment" às vezes abre a grande-angular, que costuma ter
// foco fixo: a carta de perto nunca fica nítida, nem no preview. Os scanners
// web de código de barras resolvem assim: ficam com a traseira que anuncia
// foco automático contínuo.

const CAM_KEY = "openpokescan.camera";
const HI_RES = { width: { ideal: 1920 }, height: { ideal: 1440 } };

/** What the browser reports about the open camera. Sent with the scan telemetry. */
export interface CameraInfo {
  label: string;
  /** Current focus mode, when the browser reports it. */
  focus: string | null;
  /** Focus modes the lens supports, or null when the browser does not say. */
  focusModes: string[] | null;
  /** Focus distance range in meters, when reported. */
  focusMin: number | null;
  focusMax: number | null;
  width: number;
  height: number;
  /** True when we left the lens the browser picked for one with autofocus. */
  switched: boolean;
  backCameras: number;
}

type Caps = MediaTrackCapabilities & { focusMode?: string[]; focusDistance?: { min?: number; max?: number } };
type Settings = MediaTrackSettings & { focusMode?: string };

function caps(track: MediaStreamTrack): Caps {
  try { return (track.getCapabilities?.() || {}) as Caps; } catch { return {}; }
}

function hasAutofocus(track: MediaStreamTrack): boolean | null {
  const modes = caps(track).focusMode;
  return Array.isArray(modes) ? modes.includes("continuous") : null;
}

// "camera2 0, facing back" → 0. A câmera principal costuma ter o menor número.
function lensIndex(label: string): number {
  const m = /camera2?\s*(\d+)/i.exec(label);
  return m ? Number(m[1]) : 99;
}

async function open(deviceId: string): Promise<MediaStream | null> {
  try {
    return await navigator.mediaDevices.getUserMedia({ video: { deviceId: { exact: deviceId }, ...HI_RES } });
  } catch {
    return null;
  }
}

// Tudo o que já foi pedido à faixa (foco, lanterna). Vai inteiro em cada
// applyConstraints, que substitui o conjunto anterior.
let advanced: Record<string, unknown> = {};

/**
 * Change camera settings (focus, torch) without losing the ones set before or
 * the resolution.
 *
 * @param track The video track.
 * @param patch Settings to change, for example `{ torch: true }`.
 * @returns True when the browser accepted the change.
 */
export async function applyCamera(track: MediaStreamTrack, patch: Record<string, unknown>): Promise<boolean> {
  const next = { ...advanced, ...patch };
  try {
    await track.applyConstraints({ ...HI_RES, advanced: [next as MediaTrackConstraintSet] });
    advanced = next;
    return true;
  } catch {
    return false;
  }
}

/** Open the lens saved by a previous tuneCamera(), or null when there is none or it fails. */
export async function openSavedCamera(): Promise<MediaStream | null> {
  let id: string | null = null;
  try { id = localStorage.getItem(CAM_KEY); } catch { /* sem storage */ }
  return id ? open(id) : null;
}

/**
 * Make sure `stream` comes from a back lens with autofocus, and turn on
 * continuous focus at the center.
 *
 * Switches lens only when the browser reports that the current lens has no
 * continuous autofocus. When the browser does not report focus modes
 * (Safari, Firefox), the stream stays as it is.
 *
 * @param stream The stream from getUserMedia. It can be stopped and replaced.
 * @returns The stream to use and the camera info for telemetry.
 */
export async function tuneCamera(stream: MediaStream): Promise<{ stream: MediaStream; info: CameraInfo }> {
  let track = stream.getVideoTracks()[0];
  let switched = false;
  let backCameras = 0;
  try {
    const devices = (await navigator.mediaDevices.enumerateDevices())
      .filter((d) => d.kind === "videoinput" && /back|rear|traseir|environment/i.test(d.label));
    backCameras = devices.length;
    if (hasAutofocus(track) === false && devices.length > 1) {
      const current = track.getSettings().deviceId;
      const others = devices.filter((d) => d.deviceId !== current).sort((a, b) => lensIndex(a.label) - lensIndex(b.label));
      // Muitos Android não abrem duas câmeras ao mesmo tempo: fecha antes de testar.
      stream.getTracks().forEach((t) => t.stop());
      let found: MediaStream | null = null;
      for (const d of others) {
        const s = await open(d.deviceId);
        if (!s) continue;
        if (hasAutofocus(s.getVideoTracks()[0])) { found = s; break; }
        s.getTracks().forEach((t) => t.stop());
      }
      stream = found || (current && (await open(current))) || (await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", ...HI_RES } }));
      switched = !!found;
      track = stream.getVideoTracks()[0];
    }
  } catch { /* fica com o stream que tiver */ }

  const c = caps(track);
  // applyConstraints SUBSTITUI todas as restrições do stream. Sem repetir a
  // resolução, o Firefox voltava ao padrão de 640×480 e o recorte da carta
  // saía com ~300px (scan_log 1472-1476). Por isso a resolução vai junto, e
  // cada pedido só sai quando o navegador anuncia suporte.
  const focus: Record<string, unknown> = {};
  if (c.focusMode?.includes("continuous")) focus.focusMode = "continuous";
  // Foco no centro, onde fica a retícula.
  if ((navigator.mediaDevices.getSupportedConstraints() as Record<string, boolean>).pointsOfInterest) {
    focus.pointsOfInterest = [{ x: 0.5, y: 0.5 }];
  }
  advanced = {};
  if (Object.keys(focus).length) await applyCamera(track, focus);

  const st = track.getSettings() as Settings;
  if (hasAutofocus(track) && st.deviceId) {
    try { localStorage.setItem(CAM_KEY, st.deviceId); } catch { /* sem storage */ }
  }
  return {
    stream,
    info: {
      label: (track.label || "").slice(0, 40),
      focus: st.focusMode ?? null,
      focusModes: Array.isArray(c.focusMode) ? c.focusMode : null,
      focusMin: c.focusDistance?.min ?? null,
      focusMax: c.focusDistance?.max ?? null,
      width: st.width ?? 0,
      height: st.height ?? 0,
      switched,
      backCameras,
    },
  };
}
