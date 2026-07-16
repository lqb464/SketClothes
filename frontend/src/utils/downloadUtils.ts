/** Trigger a browser download from a data URL or blob URL. */
export function downloadUrl(url: string, filename: string): void {
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
}

/** Side-by-side composite of sketch + photo as a PNG data URL. */
export async function composeSketchAndPhoto(
  sketchDataUrl: string,
  photoDataUrl: string,
  size = 512
): Promise<string> {
  const [sketch, photo] = await Promise.all([
    loadImage(sketchDataUrl),
    loadImage(photoDataUrl),
  ]);

  const canvas = document.createElement("canvas");
  canvas.width = size * 2;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas unsupported");

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(sketch, 0, 0, size, size);
  ctx.drawImage(photo, size, 0, size, size);

  return canvas.toDataURL("image/png");
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Failed to load image"));
    img.src = src;
  });
}
