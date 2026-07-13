import { useState } from "react";
import PreviewPanel from "./components/PreviewPanel";
import SketchCanvas from "./components/SketchCanvas";
import { useSketchStream } from "./hooks/useSketchStream";
import { FashionCategory } from "./types";

export default function App() {
  const [category, setCategory] = useState<FashionCategory>("shirt");
  const [style, setStyle] = useState("");

  const { health, imageSrc, loading, hasSketch, statusMessage, step, updateSketch, generate } =
    useSketchStream({ category, style });

  const isStreaming = health?.streaming ?? false;
  const deviceLabel =
    health?.device === "cuda" ? "GPU" : health ? "CPU" : "Offline";

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>Sketch-to-Fashion</h1>
          <p>Vẽ sketch tay → nhấn Tạo ảnh để sinh trang phục</p>
        </div>
        <div className="badges">
          <span className={`badge device-${health?.device ?? "unknown"}`}>
            {deviceLabel}
            {health?.resolution ? ` · ${health.resolution}px` : ""}
          </span>
          {isStreaming && <span className="badge mode">Streaming</span>}
          {health && !health.streaming && (
            <span className="badge warn">Chậm (~30-90s)</span>
          )}
        </div>
      </header>

      <main className="workspace">
        <section className="panel sketch-panel">
          <h2>Sketch của bạn</h2>
          <SketchCanvas onSketchChange={updateSketch} />
        </section>
        <section className="panel preview-panel-wrap">
          <h2>Ảnh sinh ra</h2>
          <PreviewPanel
            imageSrc={imageSrc}
            loading={loading}
            hasSketch={hasSketch}
            statusMessage={statusMessage}
            streaming={isStreaming}
            step={step}
            category={category}
            style={style}
            onCategoryChange={setCategory}
            onStyleChange={setStyle}
            onGenerate={generate}
          />
        </section>
      </main>

      <footer className="footer">
        <p>
          Model: Stable Diffusion 1.5 + ControlNet Scribble
          {health?.resolution ? ` · ${health.resolution}px` : ""}
        </p>
      </footer>
    </div>
  );
}
