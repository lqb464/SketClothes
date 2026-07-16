export interface HealthInfo {
  status: string;
  device: string;
  mode: string;
  streaming: boolean;
  models_loaded: boolean;
  resolution: number;
}

export interface StreamMessage {
  type: "frame" | "done" | "progress" | "error" | "cancelled" | "pong";
  image?: string;
  step?: number;
  message?: string;
}
