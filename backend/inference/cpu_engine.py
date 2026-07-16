import asyncio
import os
from collections.abc import AsyncIterator

import torch
from PIL import Image

import config
from .base import GenerationEvent, InferenceEngine
from .pipeline_loader import load_controlnet_pipeline
from .sketch_utils import image_to_jpeg_b64, preprocess_sketch


class CPUEngine(InferenceEngine):
    def __init__(self) -> None:
        self._pipe = None
        self._loaded = False
        self._cancel_flags: dict[str, bool] = {}

    @property
    def mode(self) -> str:
        return "cpu"

    @property
    def supports_streaming(self) -> bool:
        return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    async def load(self) -> None:
        if self._loaded:
            return

        torch.set_num_threads(os.cpu_count() or 4)

        def _load_pipeline():
            print("[INFO] Loading ControlNet + SD pipeline (first run downloads weights)...")
            pipe = load_controlnet_pipeline("cpu", torch.float32)
            print("[INFO] Pipeline loaded successfully.")
            return pipe

        self._pipe = await asyncio.to_thread(_load_pipeline)
        self._loaded = True

    async def cancel(self, request_id: str) -> None:
        self._cancel_flags[request_id] = True

    async def generate(
        self,
        sketch: Image.Image,
        prompt: str,
        request_id: str,
        conditioning_scale: float | None = None,
    ) -> AsyncIterator[GenerationEvent]:
        if not self._loaded or self._pipe is None:
            await self.load()

        self._cancel_flags[request_id] = False
        control_image = preprocess_sketch(sketch, size=config.RESOLUTION)
        scale = (
            conditioning_scale
            if conditioning_scale is not None
            else config.CONTROLNET_CONDITIONING_SCALE
        )

        yield GenerationEvent(type="progress", message="Generating on CPU (may take 30-90s)...")
        print(f"[INFO] Generating image (CPU, {config.RESOLUTION}px)...")

        def _run():
            if self._cancel_flags.get(request_id):
                return None
            result = self._pipe(
                prompt=prompt,
                negative_prompt=config.NEGATIVE_PROMPT,
                image=control_image,
                num_inference_steps=config.NUM_INFERENCE_STEPS,
                guidance_scale=config.GUIDANCE_SCALE,
                controlnet_conditioning_scale=scale,
            )
            return result.images[0]

        image = await asyncio.to_thread(_run)
        self._cancel_flags.pop(request_id, None)

        if image is None:
            yield GenerationEvent(type="cancelled", message="Request cancelled")
            return

        print("[INFO] Generation complete.")
        yield GenerationEvent(
            type="done",
            image_b64=image_to_jpeg_b64(image),
        )
