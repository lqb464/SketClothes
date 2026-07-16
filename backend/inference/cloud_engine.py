"""Cloud-based sketch-to-fashion generation engine."""
import base64
import io
import os
import logging
from collections.abc import AsyncIterator
from PIL import Image, ImageOps
import httpx

from .base import InferenceEngine, GenerationEvent
import config

logger = logging.getLogger(__name__)


class CloudEngine(InferenceEngine):
    """
    Implements sketch-to-fashion generation by calling a Cloud API.

    Supports Stability AI's ControlNet/Sketch API if STABILITY_API_KEY is configured.
    Otherwise, falls back to a Gemini-driven text-to-image (Imagen 3) pipeline
    or a mock generative overlay.
    """

    @property
    def mode(self) -> str:
        return "cloud"

    @property
    def supports_streaming(self) -> bool:
        return False

    @property
    def is_loaded(self) -> bool:
        return True

    async def load(self) -> None:
        pass

    def _pil_to_bytes(self, image: Image.Image) -> bytes:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return buffered.getvalue()

    def _bytes_to_b64(self, img_bytes: bytes) -> str:
        return base64.b64encode(img_bytes).decode("utf-8")

    async def generate(
        self,
        sketch: Image.Image,
        prompt: str,
        request_id: str,
        conditioning_scale: float | None = None,
    ) -> AsyncIterator[GenerationEvent]:
        yield GenerationEvent(type="status", message="Gửi dữ liệu lên Cloud API...")

        stability_key = os.getenv("STABILITY_API_KEY", "")
        gemini_key = os.getenv("GEMINI_API_KEY", "")

        # Scenario 1: Stability AI ControlNet/Sketch API (True sketch-to-image)
        if stability_key:
            try:
                yield GenerationEvent(type="status", message="Đang sinh ảnh bằng Stability ControlNet...")
                img_bytes = self._pil_to_bytes(sketch)
                
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "https://api.stability.ai/v2beta/generation/image-to-image/control/sketch",
                        headers={
                            "authorization": f"Bearer {stability_key}",
                            "accept": "image/*"
                        },
                        files={
                            "image": ("sketch.png", img_bytes, "image/png")
                        },
                        data={
                            "prompt": f"{prompt}, high quality fashion photo, studio lighting",
                            "output_format": "webp",
                            "control_strength": 0.7
                        },
                        timeout=45.0
                    )
                    
                    if response.status_code == 200:
                        b64_res = base64.b64encode(response.content).decode("utf-8")
                        yield GenerationEvent(type="image", image_b64=b64_res)
                        yield GenerationEvent(type="status", message="Hoàn thành!")
                        return
                    else:
                        logger.error("Stability API error: %s", response.text)
                        yield GenerationEvent(type="status", message=f"Stability API Error: {response.status_code}")
            except Exception as e:
                logger.error("Stability generation failed: %s", e)
                yield GenerationEvent(type="status", message="Lỗi Stability API, chuyển sang phương án dự phòng...")

        # Scenario 2: Gemini Imagen 3 Text-to-Image (Multimodal prompt fallback)
        if gemini_key:
            try:
                yield GenerationEvent(type="status", message="Đang phân tích sketch bằng Gemini...")
                # Convert sketch to base64 for Gemini vision
                sketch_b64 = self._bytes_to_b64(self._pil_to_bytes(sketch))
                
                async with httpx.AsyncClient() as client:
                    # Step 1: Use Gemini to describe the sketch colors and layout
                    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')}:generateContent?key={gemini_key}"
                    vision_payload = {
                        "contents": [{
                            "parts": [
                                {"text": f"Describe the fashion sketch in detail. List the item type (dress, shirt, pants), colors, and outline. User prompt: {prompt}"},
                                {
                                    "inlineData": {
                                        "mimeType": "image/png",
                                        "data": sketch_b64
                                    }
                                }
                            ]
                        }]
                    }
                    
                    res = await client.post(gemini_url, json=vision_payload, timeout=20.0)
                    res.raise_for_status()
                    description = res.json()["candidates"][0]["content"]["parts"][0]["text"]
                    
                    yield GenerationEvent(type="status", message="Đang sinh ảnh thời trang bằng Imagen 3...")
                    
                    # Step 2: Use Imagen 3 to generate the image
                    imagen_url = f"https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-002:generateImages?key={gemini_key}"
                    imagen_payload = {
                        "prompt": f"Professional fashion catalog photo of a garment based on this description: {description}. Style: {prompt}. Solid background, studio lighting, photorealistic.",
                        "numberOfImages": 1,
                        "outputMimeType": "image/jpeg",
                        "aspectRatio": "1:1"
                    }
                    
                    res_img = await client.post(imagen_url, json=imagen_payload, timeout=30.0)
                    res_img.raise_for_status()
                    img_data_b64 = res_img.json()["generatedImages"][0]["image"]["imageBytesBase64"]
                    
                    yield GenerationEvent(type="image", image_b64=img_data_b64)
                    yield GenerationEvent(type="status", message="Hoàn thành!")
                    return
            except Exception as e:
                logger.error("Gemini Imagen generation failed: %s", e)

        # Scenario 3: Local Mock Overlay (Allows app to run fully without keys)
        yield GenerationEvent(type="status", message="Mô phỏng sinh ảnh thời trang (Mock mode)...")
        
        # Colorized overlay: paste sketch on a white background with a colorful filter
        bg = Image.new("RGB", sketch.size, (245, 245, 247))
        # Invert sketch edges to make them colored lines on white
        inverted = ImageOps.invert(sketch.convert("RGB"))
        bg.paste(inverted, (0, 0), sketch.split()[-1] if sketch.mode == "RGBA" else None)
        
        b64_mock = self._bytes_to_b64(self._pil_to_bytes(bg))
        yield GenerationEvent(type="image", image_b64=b64_mock)
        yield GenerationEvent(type="status", message="Mô phỏng hoàn tất! Hãy cấu hình API keys để sử dụng thật.")
