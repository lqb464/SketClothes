import { FashionCategory } from "../types";
import FashionControls from "./FashionControls";

interface PreviewPanelProps {
  imageSrc: string | null;
  loading: boolean;
  hasSketch: boolean;
  statusMessage: string;
  streaming: boolean;
  step?: number;
  category: FashionCategory;
  style: string;
  onCategoryChange: (category: FashionCategory) => void;
  onStyleChange: (style: string) => void;
  onGenerate: () => void;
}

export default function PreviewPanel({
  imageSrc,
  loading,
  hasSketch,
  statusMessage,
  streaming,
  step,
  category,
  style,
  onCategoryChange,
  onStyleChange,
  onGenerate,
}: PreviewPanelProps) {
  const handleDownload = () => {
    if (!imageSrc) return;
    const link = document.createElement("a");
    link.href = imageSrc;
    link.download = "sketch-to-fashion.jpg";
    link.click();
  };

  return (
    <div className="preview-panel">
      <FashionControls
        category={category}
        style={style}
        onCategoryChange={onCategoryChange}
        onStyleChange={onStyleChange}
        onGenerate={onGenerate}
        loading={loading}
        hasSketch={hasSketch}
      />
      <div className="panel-visual">
        <div className="preview-frame">
          {imageSrc ? (
            <img src={imageSrc} alt="Generated fashion" />
          ) : (
            <div className="placeholder">
              <span>Ảnh sinh ra sẽ hiện ở đây</span>
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
      <div className="panel-footer">
        <p className="status">{statusMessage}</p>
        <button type="button" onClick={handleDownload} disabled={!imageSrc}>
          Tải ảnh
        </button>
      </div>
    </div>
  );
}
