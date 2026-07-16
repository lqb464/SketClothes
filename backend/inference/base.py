from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

from PIL import Image


@dataclass
class GenerationEvent:
    type: str
    image_b64: str | None = None
    step: int | None = None
    message: str | None = None


class InferenceEngine(ABC):
    @property
    @abstractmethod
    def mode(self) -> str:
        """'gpu' or 'cpu'."""

    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        pass

    @abstractmethod
    async def load(self) -> None:
        pass

    @abstractmethod
    async def generate(
        self,
        sketch: Image.Image,
        prompt: str,
        request_id: str,
        conditioning_scale: float | None = None,
    ) -> AsyncIterator[GenerationEvent]:
        pass

    async def cancel(self, request_id: str) -> None:
        pass
