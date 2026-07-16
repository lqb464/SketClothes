import asyncio
import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import DEVICE, PORT, RESOLUTION
from inference.base import GenerationEvent, InferenceEngine
from inference.factory import create_engine
from inference.sketch_utils import decode_sketch_b64, is_blank_sketch
from prompts import build_prompt

engine: InferenceEngine | None = None
active_request_id: str | None = None
load_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = create_engine()
    asyncio.create_task(_warmup_engine())
    yield


async def _warmup_engine() -> None:
    global engine
    if engine is None:
        return
    async with load_lock:
        try:
            await engine.load()
            print(f"[INFO] Inference engine ready ({engine.mode}, streaming={engine.supports_streaming})")
        except Exception as exc:
            print(f"[ERROR] Failed to load models: {exc}")


app = FastAPI(title="Sketch-to-Fashion API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def event_to_dict(event: GenerationEvent) -> dict:
    payload = {"type": event.type}
    if event.image_b64 is not None:
        payload["image"] = event.image_b64
    if event.step is not None:
        payload["step"] = event.step
    if event.message is not None:
        payload["message"] = event.message
    return payload


async def safe_send_json(websocket: WebSocket, payload: dict) -> bool:
    try:
        await websocket.send_json(payload)
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "mode": engine.mode if engine else "unknown",
        "streaming": engine.supports_streaming if engine else False,
        "models_loaded": engine.is_loaded if engine else False,
        "resolution": RESOLUTION,
    }


from pydantic import BaseModel

class GenerateRequest(BaseModel):
    sketch: str  # base64 encoded image
    category: str = "shirt"
    style: str = ""
    conditioning_scale: float | None = None


@app.get("/api/models/status")
async def models_status():
    return {
        "loaded": engine.is_loaded if engine else False,
        "mode": engine.mode if engine else "unknown",
    }


@app.post("/api/generate")
async def http_generate(body: GenerateRequest):
    global engine
    if engine is None:
        raise HTTPException(status_code=500, detail="Engine not initialized")

    try:
        sketch = decode_sketch_b64(body.sketch)
        if is_blank_sketch(sketch):
            raise HTTPException(status_code=400, detail="Sketch trống — hãy vẽ trước khi sinh ảnh")

        prompt = build_prompt(body.style)

        async with load_lock:
            if not engine.is_loaded:
                await engine.load()

        request_id = str(uuid.uuid4())
        final_image_b64 = None

        # Iterate over generation events to find the final image
        async for event in engine.generate(
            sketch,
            prompt,
            request_id,
            conditioning_scale=body.conditioning_scale,
        ):
            if event.type == "image":
                final_image_b64 = event.image_b64

        if not final_image_b64:
            raise HTTPException(status_code=500, detail="Sinh ảnh không thành công — không có kết quả trả về")

        return {
            "image": final_image_b64,
            "category": body.category,
            "style": body.style,
            "mode": engine.mode
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))



@app.websocket("/ws/generate")
async def ws_generate(websocket: WebSocket):
    global active_request_id, engine

    await websocket.accept()

    if engine is None:
        await websocket.send_json({"type": "error", "message": "Engine not initialized"})
        await websocket.close()
        return

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)

            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if data.get("type") != "generate":
                await websocket.send_json(
                    {"type": "error", "message": f"Unknown message type: {data.get('type')}"}
                )
                continue

            sketch_b64 = data.get("sketch")
            if not sketch_b64:
                await websocket.send_json({"type": "error", "message": "Missing sketch data"})
                continue

            style = data.get("style", "")
            request_id = data.get("request_id") or str(uuid.uuid4())
            raw_scale = data.get("conditioning_scale")
            conditioning_scale = float(raw_scale) if raw_scale is not None else None

            if active_request_id and active_request_id != request_id:
                await engine.cancel(active_request_id)

            active_request_id = request_id

            try:
                sketch = decode_sketch_b64(sketch_b64)
                if is_blank_sketch(sketch):
                    await safe_send_json(
                        websocket,
                        {"type": "cancelled", "message": "Sketch trống — hãy vẽ trước khi sinh ảnh"},
                    )
                    continue

                prompt = build_prompt(style)

                async with load_lock:
                    if not engine.is_loaded:
                        if not await safe_send_json(
                            websocket,
                            {
                                "type": "progress",
                                "message": "Loading models (first run may download ~5GB)...",
                            },
                        ):
                            break
                        await engine.load()

                async for event in engine.generate(
                    sketch,
                    prompt,
                    request_id,
                    conditioning_scale=conditioning_scale,
                ):
                    if active_request_id != request_id:
                        break
                    if not await safe_send_json(websocket, event_to_dict(event)):
                        break

            except Exception as exc:
                print(f"[ERROR] Generation failed: {exc}")
                await safe_send_json(websocket, {"type": "error", "message": str(exc)})
            finally:
                if active_request_id == request_id:
                    active_request_id = None

    except WebSocketDisconnect:
        if active_request_id:
            await engine.cancel(active_request_id)
            active_request_id = None


# Mount standalone built Vite frontend (`frontend/dist`) if available
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse

FRONTEND_DIST_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if FRONTEND_DIST_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST_DIR / "assets")), name="assets")

    @app.get("/")
    async def serve_root():
        return FileResponse(FRONTEND_DIST_DIR / "index.html")

    @app.get("/{full_path:path}")
    async def catch_all_spa(full_path: str):
        file_path = FRONTEND_DIST_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIST_DIR / "index.html")
else:
    @app.get("/")
    async def serve_root_fallback():
        return {"status": "ok", "message": "SketClothes API running. Build frontend/dist for UI."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
