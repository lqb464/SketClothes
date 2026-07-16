interface FashionControlsProps {
  style: string;
  adherence: number;
  liveMode: boolean;
  onStyleChange: (style: string) => void;
  onAdherenceChange: (value: number) => void;
  onLiveModeChange: (value: boolean) => void;
  onGenerate: () => void;
  loading: boolean;
  hasSketch: boolean;
}

export default function FashionControls({
  style,
  adherence,
  liveMode,
  onStyleChange,
  onAdherenceChange,
  onLiveModeChange,
  onGenerate,
  loading,
  hasSketch,
}: FashionControlsProps) {
  const adherenceLabel =
    adherence <= 0.45 ? "Lỏng" : adherence >= 0.75 ? "Chặt" : "Vừa";

  return (
    <div className="global-toolbar">
      <label className="style-input toolbar-prompt">
        <span className="toolbar-label">Mô tả</span>
        <input
          type="text"
          placeholder="vd: áo denim xanh, váy lụa đỏ, casual linen…"
          value={style}
          onChange={(e) => onStyleChange(e.target.value)}
        />
      </label>

      <label className="adherence-input toolbar-adherence">
        <span className="toolbar-label">
          Bám sketch <em>{adherenceLabel}</em>
        </span>
        <input
          type="range"
          min={0.3}
          max={1}
          step={0.05}
          value={adherence}
          onChange={(e) => onAdherenceChange(Number(e.target.value))}
        />
      </label>

      <label className="live-toggle toolbar-live">
        <input
          type="checkbox"
          checked={liveMode}
          onChange={(e) => onLiveModeChange(e.target.checked)}
        />
        <span className="live-toggle-box" aria-hidden />
        <span className="live-toggle-text">
          <strong>Live</strong>
          <small>Tự sinh khi vẽ</small>
        </span>
      </label>

      <button
        type="button"
        className="btn-generate"
        onClick={onGenerate}
        disabled={loading || !hasSketch}
        title={liveMode ? "Ép sinh lại ngay" : "Sinh ảnh từ sketch hiện tại"}
      >
        {loading ? "Đang sinh..." : liveMode ? "Tạo lại" : "Tạo ảnh"}
      </button>
    </div>
  );
}
