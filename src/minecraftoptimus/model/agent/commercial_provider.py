import base64
import logging
import os
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import numpy as np
from PIL import Image


logger = logging.getLogger(__name__)


def _get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float env %s=%s, using default=%s", name, raw, default)
        return default


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid int env %s=%s, using default=%s", name, raw, default)
        return default


@dataclass
class CommercialLLMProviderConfig:
    backend: str
    model: str
    api_key: Optional[str]
    api_base: Optional[str]
    temperature: float = 0.2
    max_tokens: int = 512
    timeout_s: float = 60.0

    @classmethod
    def from_env(cls) -> Optional["CommercialLLMProviderConfig"]:
        model = (os.getenv("OPTIMUS_COMMERCIAL_LLM_MODEL") or "").strip()
        if not model:
            return None
        backend = (os.getenv("OPTIMUS_COMMERCIAL_LLM_BACKEND") or "litellm").strip().lower()
        api_key = (os.getenv("OPTIMUS_COMMERCIAL_LLM_API_KEY") or "").strip() or None
        api_base = (os.getenv("OPTIMUS_COMMERCIAL_LLM_API_BASE") or "").strip() or None
        temperature = _get_env_float("OPTIMUS_COMMERCIAL_LLM_TEMPERATURE", 0.2)
        max_tokens = _get_env_int("OPTIMUS_COMMERCIAL_LLM_MAX_TOKENS", 512)
        timeout_s = _get_env_float("OPTIMUS_COMMERCIAL_LLM_TIMEOUT_S", 60.0)
        return cls(
            backend=backend,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )


class CommercialLLMProvider:
    def __init__(self, config: CommercialLLMProviderConfig):
        self.config = config
        if self.config.backend != "litellm":
            raise ValueError(f"Unsupported backend: {self.config.backend}. Expected: litellm")
        try:
            from litellm import completion
        except ImportError as exc:
            raise RuntimeError(
                "litellm is required for commercial provider mode. Install it with `uv pip install litellm`."
            ) from exc
        self._completion = completion

    def describe(self) -> str:
        return f"{self.config.backend}:{self.config.model}"

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image: np.ndarray | str | None = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        message_content: list[dict] = [{"type": "text", "text": user_prompt}]
        image_url = self._normalize_image_to_url(image)
        if image_url is not None:
            message_content.append({"type": "image_url", "image_url": {"url": image_url}})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message_content},
        ]
        request = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "timeout": self.config.timeout_s,
        }
        if self.config.api_key:
            request["api_key"] = self.config.api_key
        if self.config.api_base:
            request["api_base"] = self.config.api_base

        try:
            response = self._completion(**request)
            return self._extract_text(response)
        except Exception as exc:
            # Some models/endpoints reject multimodal content even with OpenAI-compatible schemas.
            if image_url is not None:
                logger.warning("Multimodal request failed, retrying text-only: %s", exc)
                request["messages"] = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                response = self._completion(**request)
                return self._extract_text(response)
            raise

    def _extract_text(self, response) -> str:
        content = response.choices[0].message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for chunk in content:
                if isinstance(chunk, dict) and chunk.get("type") == "text":
                    parts.append(chunk.get("text", ""))
                elif isinstance(chunk, str):
                    parts.append(chunk)
            return "\n".join([part for part in parts if part]).strip()
        return str(content)

    def _normalize_image_to_url(self, image: np.ndarray | str | None) -> Optional[str]:
        if image is None:
            return None
        if isinstance(image, str):
            if image.startswith("http://") or image.startswith("https://") or image.startswith("data:"):
                return image
            with Image.open(image) as img:
                rgb = img.convert("RGB")
                return self._encode_pil_to_data_url(rgb)
        if isinstance(image, np.ndarray):
            arr = image.astype(np.uint8, copy=False)
            pil_image = Image.fromarray(arr).convert("RGB")
            return self._encode_pil_to_data_url(pil_image)
        raise TypeError(f"Unsupported image type: {type(image)}")

    @staticmethod
    def _encode_pil_to_data_url(image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"


def build_provider_from_env() -> Optional[CommercialLLMProvider]:
    config = CommercialLLMProviderConfig.from_env()
    if config is None:
        return None
    provider = CommercialLLMProvider(config)
    logger.info("Commercial LLM provider enabled: %s", provider.describe())
    return provider
