/** True when canvas has no visible content (all white / near-white). */
export function isCanvasBlank(canvas: HTMLCanvasElement): boolean {
  const ctx = canvas.getContext("2d");
  if (!ctx) return true;

  const { data, width, height } = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const step = Math.max(1, Math.floor(Math.sqrt((width * height) / 4096)));
  let colored = 0;

  for (let y = 0; y < height; y += step) {
    for (let x = 0; x < width; x += step) {
      const i = (y * width + x) * 4;
      if (data[i] < 250 || data[i + 1] < 250 || data[i + 2] < 250) {
        colored += 1;
        if (colored >= 8) return false;
      }
    }
  }
  return true;
}

function parseColor(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  const full = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
  return [
    parseInt(full.slice(0, 2), 16),
    parseInt(full.slice(2, 4), 16),
    parseInt(full.slice(4, 6), 16),
  ];
}

function colorMatch(
  data: Uint8ClampedArray,
  i: number,
  target: [number, number, number],
  tolerance: number
): boolean {
  return (
    Math.abs(data[i] - target[0]) <= tolerance &&
    Math.abs(data[i + 1] - target[1]) <= tolerance &&
    Math.abs(data[i + 2] - target[2]) <= tolerance
  );
}

/** Paint-style flood fill at (x, y). */
export function floodFill(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  fillHex: string,
  width: number,
  height: number,
  tolerance = 36
): void {
  const startX = Math.floor(x);
  const startY = Math.floor(y);
  if (startX < 0 || startY < 0 || startX >= width || startY >= height) return;

  const imageData = ctx.getImageData(0, 0, width, height);
  const data = imageData.data;
  const startIdx = (startY * width + startX) * 4;
  const target: [number, number, number] = [
    data[startIdx],
    data[startIdx + 1],
    data[startIdx + 2],
  ];
  const fill = parseColor(fillHex);

  if (
    Math.abs(fill[0] - target[0]) <= tolerance &&
    Math.abs(fill[1] - target[1]) <= tolerance &&
    Math.abs(fill[2] - target[2]) <= tolerance
  ) {
    return;
  }

  const stack: [number, number][] = [[startX, startY]];
  const visited = new Uint8Array(width * height);

  while (stack.length > 0) {
    const [px, py] = stack.pop()!;
    const pi = py * width + px;
    if (visited[pi]) continue;

    const i = pi * 4;
    if (!colorMatch(data, i, target, tolerance)) continue;

    visited[pi] = 1;
    data[i] = fill[0];
    data[i + 1] = fill[1];
    data[i + 2] = fill[2];
    data[i + 3] = 255;

    if (px > 0) stack.push([px - 1, py]);
    if (px < width - 1) stack.push([px + 1, py]);
    if (py > 0) stack.push([px, py - 1]);
    if (py < height - 1) stack.push([px, py + 1]);
  }

  ctx.putImageData(imageData, 0, 0);
}
