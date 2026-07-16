import { useCallback, useState } from "react";
import FashionControls from "./components/FashionControls";
import PreviewPanel from "./components/PreviewPanel";
import SketchCanvas from "./components/SketchCanvas";
import { useSketchStream } from "./hooks/useSketchStream";

export default function App() {
  const [style, setStyle] = useState("");
  const [adherence, setAdherence] = useState(0.6);
  const [liveMode, setLiveMode] = useState(false);
  const [sketchDataUrl, setSketchDataUrl] = useState<string | null>(null);

  const { health, imageSrc, loading, hasSketch, statusMessage, step, updateSketch, generate } =
    useSketchStream({ style, adherence, liveMode });

  const handleSketchChange = useCallback(
    (dataUrl: string | null) => {
      setSketchDataUrl(dataUrl);
      updateSketch(dataUrl);
    },
    [updateSketch]
  );

  const deviceStreaming = health?.streaming ?? false;
  const deviceLabel =
    health?.device === "cuda" ? "GPU" : health ? "CPU" : "Offline";

  return (
    <div className="app">
      <header className="header">
        <div>
          <p className="brand">SketClothes</p>
          <h1>Sketch → Fashion</h1>
        </div>
        <div className="badges">
          <span className={`badge device-${health?.device ?? "unknown"}`}>
            {deviceLabel}
            {health?.resolution ? ` · ${health.resolution}px` : ""}
          </span>
          {liveMode && <span className="badge mode">Live draw</span>}
          {deviceStreaming && <span className="badge mode">Frame stream</span>}
          {health && !deviceStreaming && (
            <span className="badge warn">Chậm (~30–90s)</span>
          )}
        </div>
      </header>

      <FashionControls
        style={style}
        adherence={adherence}
        liveMode={liveMode}
        onStyleChange={setStyle}
        onAdherenceChange={setAdherence}
        onLiveModeChange={setLiveMode}
        onGenerate={generate}
        loading={loading}
        hasSketch={hasSketch}
      />

      <main className="workspace">
        <section className="panel sketch-panel">
          <h2>Sketch</h2>
          <SketchCanvas onSketchChange={handleSketchChange} />
        </section>
        <section className="panel preview-panel-wrap">
          <h2>Kết quả</h2>
          <PreviewPanel
            imageSrc={imageSrc}
            sketchDataUrl={sketchDataUrl}
            loading={loading}
            statusMessage={statusMessage}
            streaming={deviceStreaming}
            step={step}
          />
        </section>
      </main>
    </div>
  );
}
