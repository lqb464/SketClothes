import asyncio
from collections.abc import AsyncIterator

import torch
from PIL import Image

import config
from .base import GenerationEvent, InferenceEngine
from .pipeline_loader import load_controlnet_pipeline
from .sketch_utils import image_to_jpeg_b64, preprocess_sketch


class DiffusersGPUEngine(InferenceEngine):
    """GPU inference via diffusers with per-step frame streaming."""

    def __init__(self) -> None:
        self._pipe = None
        self._loaded = False
        self._cancel_flags: dict[str, bool] = {}
        self._dtype = torch.float16

    @property
    def mode(self) -> str:
        return "gpu"

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    async def load(self) -> None:
        if self._loaded:
            return

        def _load_pipeline():
            return load_controlnet_pipeline("cuda", self._dtype)

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
        frames: asyncio.Queue[GenerationEvent | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _callback(pipe, step_index, timestep, callback_kwargs):
            if self._cancel_flags.get(request_id):
                return callback_kwargs
            latent = callback_kwargs.get("latents")
            if latent is not None:
                with torch.no_grad():
                    decoded = pipe.vae.decode(
                        latent / pipe.vae.config.scaling_factor,
                        return_dict=False,
                    )[0]
                    image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
                    event = GenerationEvent(
                        type="frame",
                        image_b64=image_to_jpeg_b64(image),
                        step=step_index + 1,
                    )
                    loop.call_soon_threadsafe(frames.put_nowait, event)
            return callback_kwargs

        def _run():
            try:
                if self._cancel_flags.get(request_id):
                    return None
                result = self._pipe(
                    prompt=prompt,
                    negative_prompt=config.NEGATIVE_PROMPT,
                    image=control_image,
                    num_inference_steps=config.NUM_INFERENCE_STEPS,
                    guidance_scale=config.GUIDANCE_SCALE,
                    controlnet_conditioning_scale=scale,
                    callback_on_step_end=_callback,
                )
                return result.images[0]
            finally:
                loop.call_soon_threadsafe(frames.put_nowait, None)

        task = asyncio.create_task(asyncio.to_thread(_run))

        while True:
            event = await frames.get()
            if event is None:
                break
            yield event

        final_image = await task
        self._cancel_flags.pop(request_id, None)

        if final_image is None:
            yield GenerationEvent(type="cancelled", message="Request cancelled")
            return

        yield GenerationEvent(
            type="done",
            image_b64=image_to_jpeg_b64(final_image),
        )


def create_gpu_engine() -> InferenceEngine:
    return DiffusersGPUEngine()
