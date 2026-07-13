import { CATEGORY_OPTIONS, FashionCategory } from "../types";

interface FashionControlsProps {
  category: FashionCategory;
  style: string;
  onCategoryChange: (category: FashionCategory) => void;
  onStyleChange: (style: string) => void;
  onGenerate: () => void;
  loading: boolean;
  hasSketch: boolean;
}

export default function FashionControls({
  category,
  style,
  onCategoryChange,
  onStyleChange,
  onGenerate,
  loading,
  hasSketch,
}: FashionControlsProps) {
  return (
    <div className="panel-controls">
      <div className="controls-row">
        <div className="controls-row-primary category-tabs">
          {CATEGORY_OPTIONS.map((opt) => (
            <button
              key={opt.id}
              type="button"
              className={category === opt.id ? "active" : ""}
              onClick={() => onCategoryChange(opt.id)}
            >
              {opt.label}
            </button>
          ))}
        </div>
        <div className="controls-row-actions">
          <button
            type="button"
            className="btn-generate"
            onClick={onGenerate}
            disabled={loading || !hasSketch}
          >
            {loading ? "Đang sinh..." : "Tạo ảnh"}
          </button>
        </div>
      </div>
      <label className="style-input">
        Phong cách / chất liệu (tuỳ chọn)
        <input
          type="text"
          placeholder="vd: denim xanh, lụa đỏ, casual..."
          value={style}
          onChange={(e) => onStyleChange(e.target.value)}
        />
      </label>
    </div>
  );
}
