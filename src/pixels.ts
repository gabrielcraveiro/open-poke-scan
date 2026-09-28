// Métricas baratas sobre o proxy da retícula (160px), rodadas a cada tick.
// Todas usam passo (stride) para caber no orçamento de 150ms no celular.

export function frameDiff(a: Uint8ClampedArray, b: Uint8ClampedArray): number {
  let sum = 0, n = 0;
  for (let i = 0; i < a.length; i += 16) {
    sum += Math.abs(a[i] - b[i]) + Math.abs(a[i + 1] - b[i + 1]) + Math.abs(a[i + 2] - b[i + 2]);
    n++;
  }
  return sum / (n * 3);
}

// Brenner: gradiente de luminância entre vizinhos horizontais. Maior = mais nítido.
export function sharpness(data: Uint8ClampedArray, w: number, h: number): number {
  let sum = 0, n = 0;
  const stride = 2;
  for (let y = 0; y < h; y += 2) {
    const row = y * w * 4;
    for (let x = 0; x < w - stride; x += stride) {
      const i = row + x * 4, j = row + (x + stride) * 4;
      const l1 = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
      const l2 = data[j] * 0.299 + data[j + 1] * 0.587 + data[j + 2] * 0.114;
      const d = l1 - l2;
      sum += d * d; n++;
    }
  }
  return n ? sum / n : 0;
}

// Desvio-padrão da luminância: carta tem estrutura (alto), mesa vazia é plana (baixo).
export function presence(data: Uint8ClampedArray): number {
  let sum = 0, sum2 = 0, n = 0;
  for (let i = 0; i < data.length; i += 16) {
    const l = (data[i] + data[i + 1] + data[i + 2]) / 3;
    sum += l; sum2 += l * l; n++;
  }
  const mean = sum / n;
  return Math.sqrt(Math.max(0, sum2 / n - mean * mean));
}

// Fração de pixels estourados (reflexo do foil holo).
export function glare(data: Uint8ClampedArray): number {
  let blown = 0, n = 0;
  for (let i = 0; i < data.length; i += 16) {
    if (data[i] > 240 && data[i + 1] > 240 && data[i + 2] > 240) blown++;
    n++;
  }
  return n ? blown / n : 0;
}
