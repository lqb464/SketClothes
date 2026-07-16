import { useCallback, useEffect, useRef, useState } from "react";
import { HealthInfo, StreamMessage } from "../types";

const WS_URL =
  import.meta.env.VITE_WS_URL ||
  `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}/ws/generate`;

const LIVE_DEBOUNCE_MS = 700;

interface UseSketchStreamOptions {
  style: string;
  adherence: number;
  liveMode: boolean;
}

export function useSketchStream({ style, adherence, liveMode }: UseSketchStreamOptions) {
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [hasSketch, setHasSketch] = useState(false);
  const [statusMessage, setStatusMessage] = useState("Vẽ sketch rồi nhấn Tạo ảnh");
  const [step, setStep] = useState<number | undefined>();

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const liveTimerRef = useRef<number | null>(null);
  const intentionalCloseRef = useRef(false);

  const requestIdRef = useRef<string>("");
  const latestSketchRef = useRef<string | null>(null);
  const queuedSketchRef = useRef<string | null>(null);
  const isGeneratingRef = useRef(false);

  const styleRef = useRef(style);
  const adherenceRef = useRef(adherence);
  const liveModeRef = useRef(liveMode);
  styleRef.current = style;
  adherenceRef.current = adherence;
  liveModeRef.current = liveMode;

  const idleStatus = useCallback((has: boolean) => {
    if (liveModeRef.current) {
      return has ? "Live — vẽ tiếp để cập nhật ảnh" : "Live — vẽ sketch để tự sinh ảnh";
    }
    return has ? "Sẵn sàng — nhấn Tạo ảnh" : "Vẽ sketch rồi nhấn Tạo ảnh";
  }, []);

  const fetchHealth = useCallback(async () => {
    try {
      const res = await fetch("/api/health");
      setHealth(await res.json());
    } catch {
      setHealth(null);
    }
  }, []);

  const payload = useCallback(
    (sketch: string, requestId: string) =>
      JSON.stringify({
        type: "generate",
        sketch,
        style: styleRef.current,
        conditioning_scale: adherenceRef.current,
        request_id: requestId,
      }),
    []
  );

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
    ws.send(payload(next, requestIdRef.current));
  }, [payload]);

  const sendGenerate = useCallback(
    (sketch: string) => {
      if (isGeneratingRef.current) {
        queuedSketchRef.current = sketch;
        setStatusMessage(
          liveModeRef.current
            ? "Đang sinh — sẽ dùng bản sketch mới nhất"
            : "Đang sinh ảnh — sẽ cập nhật sau khi xong"
        );
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
      ws.send(payload(sketch, requestIdRef.current));
    },
    [payload]
  );

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
        setStatusMessage(idleStatus(Boolean(latestSketchRef.current)));
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
        setStatusMessage(liveModeRef.current ? "Đã cập nhật — vẽ tiếp nếu muốn" : "Sinh ảnh xong!");
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
        if (queuedSketchRef.current) {
          flushQueued();
        }
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
  }, [flushQueued, idleStatus]);

  const clearLiveTimer = useCallback(() => {
    if (liveTimerRef.current !== null) {
      window.clearTimeout(liveTimerRef.current);
      liveTimerRef.current = null;
    }
  }, []);

  const updateSketch = useCallback(
    (sketch: string | null) => {
      latestSketchRef.current = sketch;
      setHasSketch(sketch !== null);
      clearLiveTimer();

      if (sketch === null) {
        queuedSketchRef.current = null;
        setImageSrc(null);
        setLoading(false);
        setStep(undefined);
        isGeneratingRef.current = false;
        setStatusMessage(idleStatus(false));
        return;
      }

      if (liveModeRef.current) {
        setStatusMessage("Live — đang chờ nét vẽ ổn định...");
        liveTimerRef.current = window.setTimeout(() => {
          liveTimerRef.current = null;
          if (!latestSketchRef.current || !liveModeRef.current) return;
          sendGenerate(latestSketchRef.current);
        }, LIVE_DEBOUNCE_MS);
        return;
      }

      if (!isGeneratingRef.current) {
        setStatusMessage(idleStatus(true));
      }
    },
    [clearLiveTimer, idleStatus, sendGenerate]
  );

  const generate = useCallback(() => {
    clearLiveTimer();
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
  }, [clearLiveTimer, connect, sendGenerate]);

  useEffect(() => {
    if (!isGeneratingRef.current) {
      setStatusMessage(idleStatus(Boolean(latestSketchRef.current)));
    }
    if (!liveMode) {
      clearLiveTimer();
      queuedSketchRef.current = null;
    }
  }, [liveMode, idleStatus, clearLiveTimer]);

  useEffect(() => {
    fetchHealth();
    connect();

    return () => {
      intentionalCloseRef.current = true;
      clearLiveTimer();
      if (reconnectTimerRef.current) window.clearTimeout(reconnectTimerRef.current);
      wsRef.current?.close();
    };
  }, [connect, fetchHealth, clearLiveTimer]);

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
