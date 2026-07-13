import { useCallback, useEffect, useRef, useState } from "react";
import { FashionCategory, HealthInfo, StreamMessage } from "../types";

const WS_URL =
  import.meta.env.VITE_WS_URL ||
  `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}/ws/generate`;

interface UseSketchStreamOptions {
  category: FashionCategory;
  style: string;
}

export function useSketchStream({ category, style }: UseSketchStreamOptions) {
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [hasSketch, setHasSketch] = useState(false);
  const [statusMessage, setStatusMessage] = useState("Vẽ sketch rồi nhấn Tạo ảnh");
  const [step, setStep] = useState<number | undefined>();

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const intentionalCloseRef = useRef(false);

  const requestIdRef = useRef<string>("");
  const latestSketchRef = useRef<string | null>(null);
  const queuedSketchRef = useRef<string | null>(null);
  const isGeneratingRef = useRef(false);

  const categoryRef = useRef(category);
  const styleRef = useRef(style);
  categoryRef.current = category;
  styleRef.current = style;

  const fetchHealth = useCallback(async () => {
    try {
      const res = await fetch("/api/health");
      setHealth(await res.json());
    } catch {
      setHealth(null);
    }
  }, []);

  const flushQueued = useCallback(() => {
    const next = queuedSketchRef.current;
    queuedSketchRef.current = null;
    if (!next) return;

    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      queuedSketchRef.current = next;
      return;
    }

    requestIdRef.current = crypto.randomUUID();
    isGeneratingRef.current = true;
    setLoading(true);
    setStatusMessage("Đang sinh ảnh...");

    ws.send(
      JSON.stringify({
        type: "generate",
        sketch: next,
        category: categoryRef.current,
        style: styleRef.current,
        request_id: requestIdRef.current,
      })
    );
  }, []);

  const sendGenerate = useCallback((sketch: string) => {
    if (isGeneratingRef.current) {
      queuedSketchRef.current = sketch;
      setStatusMessage("Đang sinh ảnh — sẽ cập nhật sau khi xong");
      return;
    }

    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      queuedSketchRef.current = sketch;
      setStatusMessage("Đang kết nối server...");
      return;
    }

    requestIdRef.current = crypto.randomUUID();
    isGeneratingRef.current = true;
    setLoading(true);
    setStatusMessage("Đang sinh ảnh...");

    ws.send(
      JSON.stringify({
        type: "generate",
        sketch,
        category: categoryRef.current,
        style: styleRef.current,
        request_id: requestIdRef.current,
      })
    );
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    if (wsRef.current?.readyState === WebSocket.CONNECTING) return;

    intentionalCloseRef.current = false;
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      if (queuedSketchRef.current) {
        flushQueued();
      } else if (!isGeneratingRef.current) {
        setStatusMessage(
          latestSketchRef.current
            ? "Sẵn sàng — nhấn Tạo ảnh"
            : "Vẽ sketch rồi nhấn Tạo ảnh"
        );
      }
    };

    ws.onmessage = (event) => {
      const data: StreamMessage = JSON.parse(event.data);

      if (data.type === "progress") {
        setLoading(true);
        setStatusMessage(data.message ?? "Đang sinh ảnh...");
        return;
      }

      if (data.type === "frame" && data.image) {
        setLoading(true);
        setImageSrc(`data:image/jpeg;base64,${data.image}`);
        setStep(data.step);
        setStatusMessage("Đang cập nhật...");
        return;
      }

      if (data.type === "done" && data.image) {
        setImageSrc(`data:image/jpeg;base64,${data.image}`);
        setLoading(false);
        setStep(undefined);
        isGeneratingRef.current = false;
        setStatusMessage("Sinh ảnh xong!");
        if (queuedSketchRef.current) {
          flushQueued();
        }
        return;
      }

      if (data.type === "error") {
        setLoading(false);
        isGeneratingRef.current = false;
        setStatusMessage(data.message ?? "Lỗi không xác định");
        queuedSketchRef.current = null;
        return;
      }

      if (data.type === "cancelled") {
        setLoading(false);
        isGeneratingRef.current = false;
        setStatusMessage(data.message ?? "Đã huỷ");
        return;
      }
    };

    ws.onclose = () => {
      wsRef.current = null;
      if (!intentionalCloseRef.current) {
        reconnectTimerRef.current = window.setTimeout(connect, 3000);
      }
    };

    ws.onerror = () => {
      setStatusMessage("Lỗi kết nối WebSocket");
    };
  }, [flushQueued]);

  const updateSketch = useCallback((sketch: string | null) => {
    latestSketchRef.current = sketch;
    setHasSketch(sketch !== null);

    if (sketch === null) {
      queuedSketchRef.current = null;
      setImageSrc(null);
      setLoading(false);
      setStep(undefined);
      isGeneratingRef.current = false;
      setStatusMessage("Vẽ sketch rồi nhấn Tạo ảnh");
      return;
    }

    if (!isGeneratingRef.current) {
      setStatusMessage("Sẵn sàng — nhấn Tạo ảnh");
    }
  }, []);

  const generate = useCallback(() => {
    const sketch = latestSketchRef.current;
    if (!sketch) {
      setStatusMessage("Hãy vẽ sketch trước");
      return;
    }

    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      queuedSketchRef.current = sketch;
      connect();
      setStatusMessage("Đang kết nối server...");
      return;
    }

    sendGenerate(sketch);
  }, [connect, sendGenerate]);

  useEffect(() => {
    fetchHealth();
    connect();

    return () => {
      intentionalCloseRef.current = true;
      if (reconnectTimerRef.current) window.clearTimeout(reconnectTimerRef.current);
      wsRef.current?.close();
    };
  }, [connect, fetchHealth]);

  return {
    health,
    imageSrc,
    loading,
    hasSketch,
    statusMessage,
    step,
    updateSketch,
    generate,
    refreshHealth: fetchHealth,
  };
}
