import { useEffect, useRef, useState } from "react";
import { composeSketchAndPhoto, downloadUrl } from "../utils/downloadUtils";

interface PreviewPanelProps {
  imageSrc: string | null;
  sketchDataUrl: string | null;
  loading: boolean;
  statusMessage: string;
  streaming: boolean;
  step?: number;
}

export default function PreviewPanel({
  imageSrc,
  sketchDataUrl,
  loading,
  statusMessage,
  streaming,
  step,
}: PreviewPanelProps) {
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!menuOpen) return;
    const onDoc = (e: MouseEvent) => {
      if (!menuRef.current?.contains(e.target as Node)) setMenuOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [menuOpen]);

  const downloadSketch = () => {
    if (!sketchDataUrl) return;
    downloadUrl(sketchDataUrl, "sketclothes-sketch.png");
    setMenuOpen(false);
  };

  const downloadPhoto = () => {
    if (!imageSrc) return;
    downloadUrl(imageSrc, "sketclothes-photo.jpg");
    setMenuOpen(false);
  };

  const downloadBoth = async () => {
    if (!sketchDataUrl || !imageSrc) return;
    try {
      const composed = await composeSketchAndPhoto(sketchDataUrl, imageSrc);
      downloadUrl(composed, "sketclothes-sketch-photo.png");
    } catch {
      // ignore
    }
    setMenuOpen(false);
  };

  const canDownload = Boolean(sketchDataUrl || imageSrc);

  return (
    <div className="preview-panel">
      <div className="panel-controls result-toolbar">
        <div className="controls-row">
          <p className="status result-status">{statusMessage}</p>
          <div className="download-menu" ref={menuRef}>
            <button
              type="button"
              className="btn-download"
              disabled={!canDownload}
              onClick={() => setMenuOpen((o) => !o)}
              aria-expanded={menuOpen}
            >
              Tải ảnh ▾
            </button>
            {menuOpen && (
              <div className="download-menu-panel download-menu-panel-down" role="menu">
                <button
                  type="button"
                  role="menuitem"
                  disabled={!sketchDataUrl}
                  onClick={downloadSketch}
                >
                  Sketch
                </button>
                <button
                  type="button"
                  role="menuitem"
                  disabled={!imageSrc}
                  onClick={downloadPhoto}
                >
                  Photo
                </button>
                <button
                  type="button"
                  role="menuitem"
                  disabled={!sketchDataUrl || !imageSrc}
                  onClick={() => void downloadBoth()}
                >
                  Sketch + Photo
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="panel-visual">
        <div className="preview-frame">
          {imageSrc ? (
            <img src={imageSrc} alt="Generated fashion" />
          ) : (
            <div className="placeholder">
              <span>Ảnh trang phục sẽ hiện ở đây</span>
            </div>
          )}
          {loading && (
            <div className="loading-overlay">
              <div className="spinner" />
              <p>{statusMessage || "Đang sinh ảnh..."}</p>
              {streaming && step !== undefined && <p>Bước {step}</p>}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
