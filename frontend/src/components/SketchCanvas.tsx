import { useCallback, useEffect, useRef, useState } from "react";
import { isCanvasBlank } from "../utils/canvasUtils";

export type Tool = "pen" | "eraser";

interface SketchCanvasProps {
  onSketchChange: (dataUrl: string | null) => void;
  width?: number;
  height?: number;
}

const MAX_UNDO = 20;
const INK = "#111111";

export default function SketchCanvas({
  onSketchChange,
  width = 512,
  height = 512,
}: SketchCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tool, setTool] = useState<Tool>("pen");
  const [penSize, setPenSize] = useState(3);
  const [isDrawing, setIsDrawing] = useState(false);
  const undoStack = useRef<ImageData[]>([]);

  const getContext = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    return canvas.getContext("2d");
  }, []);

  const initCanvas = useCallback(() => {
    const ctx = getContext();
    if (!ctx) return;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    // Snapshots are states *before* each stroke; undo pops & restores that state.
    undoStack.current = [];
  }, [getContext, width, height]);

  useEffect(() => {
    initCanvas();
  }, [initCanvas]);

  const pushUndo = useCallback(() => {
    const ctx = getContext();
    if (!ctx) return;
    const snapshot = ctx.getImageData(0, 0, width, height);
    undoStack.current = [...undoStack.current.slice(-(MAX_UNDO - 1)), snapshot];
  }, [getContext, width, height]);

  const exportSketch = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (isCanvasBlank(canvas)) {
      onSketchChange(null);
      return;
    }
    onSketchChange(canvas.toDataURL("image/png"));
  }, [onSketchChange]);

  const handleClear = () => {
    initCanvas();
    onSketchChange(null);
  };

  const getPoint = (event: React.MouseEvent | React.TouchEvent) => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    // Use content box (exclude CSS border) so ink lands under the crosshair.
    const scaleX = canvas.width / canvas.clientWidth;
    const scaleY = canvas.height / canvas.clientHeight;

    const clientX =
      "touches" in event
        ? (event.touches[0] ?? event.changedTouches[0]).clientX
        : event.clientX;
    const clientY =
      "touches" in event
        ? (event.touches[0] ?? event.changedTouches[0]).clientY
        : event.clientY;

    return {
      x: (clientX - rect.left - canvas.clientLeft) * scaleX,
      y: (clientY - rect.top - canvas.clientTop) * scaleY,
    };
  };

  const handlePointerDown = (event: React.MouseEvent | React.TouchEvent) => {
    event.preventDefault();
    const ctx = getContext();
    if (!ctx) return;
    const { x, y } = getPoint(event);

    pushUndo();
    setIsDrawing(true);
    ctx.beginPath();
    ctx.moveTo(x, y);
  };

  const handlePointerMove = (event: React.MouseEvent | React.TouchEvent) => {
    if (!isDrawing) return;
    event.preventDefault();
    const ctx = getContext();
    if (!ctx) return;
    const { x, y } = getPoint(event);
    ctx.strokeStyle = tool === "eraser" ? "#ffffff" : INK;
    ctx.lineWidth = tool === "pen" ? penSize : penSize * 3;
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
  };

  const handlePointerUp = () => {
    if (!isDrawing) return;
    setIsDrawing(false);
    exportSketch();
  };

  const handleUndo = () => {
    const ctx = getContext();
    if (!ctx || undoStack.current.length === 0) return;
    const prev = undoStack.current.pop();
    if (!prev) return;
    ctx.putImageData(prev, 0, 0);
    exportSketch();
  };

  const cursorClass = tool === "eraser" ? "canvas-eraser" : "canvas-pen";

  return (
    <div className="sketch-canvas">
      <div className="panel-controls sketch-toolbar">
        <div className="controls-row">
          <div className="controls-row-primary toolbar-tools">
            <button
              type="button"
              className={tool === "pen" ? "active" : ""}
              onClick={() => setTool("pen")}
            >
              Bút mực
            </button>
            <button
              type="button"
              className={tool === "eraser" ? "active" : ""}
              onClick={() => setTool("eraser")}
            >
              Tẩy
            </button>
            <label className="pen-size">
              Nét
              <input
                type="range"
                min={2}
                max={12}
                value={penSize}
                onChange={(e) => setPenSize(Number(e.target.value))}
              />
              <span>{penSize}px</span>
            </label>
          </div>
          <div className="controls-row-actions">
            <button
              type="button"
              className="btn-undo"
              onClick={handleUndo}
              title="Hoàn tác nét vừa vẽ"
            >
              <span className="btn-icon" aria-hidden>
                ↺
              </span>
              Hoàn tác
            </button>
            <button
              type="button"
              className="btn-clear"
              onClick={handleClear}
              title="Xóa toàn bộ canvas"
            >
              <span className="btn-icon" aria-hidden>
                ⌫
              </span>
              Xóa
            </button>
          </div>
        </div>
      </div>

      <div className="panel-visual">
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          className={`canvas ${cursorClass}`}
          onMouseDown={handlePointerDown}
          onMouseMove={handlePointerMove}
          onMouseUp={handlePointerUp}
          onMouseLeave={handlePointerUp}
          onTouchStart={handlePointerDown}
          onTouchMove={handlePointerMove}
          onTouchEnd={handlePointerUp}
        />
      </div>
    </div>
  );
}
