"""
LangChain-compatible text embeddings using Qwen/Qwen3-VL-Embedding-2B.

The model is a decoder-architecture vision-language embedding model.
For text-only inputs it produces high-quality dense embeddings via
last-non-padding-token pooling followed by L2 normalisation.

The singleton getter get_qwen_embeddings() loads the model once per
process and caches it, so VectorStore instances share the same weights.
"""
from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F
from langchain_core.embeddings import Embeddings
from transformers import AutoModel, AutoTokenizer

from src.logger.custom_logger import logger
from src.settings.config import settings


class QwenVLEmbeddings(Embeddings):

    def __init__(
        self,
        model_name: str,
        device: str,
        max_length: int = 512,
        batch_size: int = 8,
    ) -> None:
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.max_length = max_length
        self.batch_size = batch_size

        logger.info(f"Loading embedding model '{model_name}' on {self.device}…")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        # Use float16 on GPU for speed; float32 on CPU/MPS for numerical stability
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=dtype,
        ).to(self.device)
        self.model.eval()
        logger.info(f"Embedding model ready: {model_name}")

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _last_token_pool(
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(
            last_hidden_state.size(0), device=last_hidden_state.device
        )
        return last_hidden_state[batch_idx, seq_lengths]

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**enc)
        vecs = self._last_token_pool(out.last_hidden_state, enc["attention_mask"])
        vecs = F.normalize(vecs, p=2, dim=-1)
        return vecs.cpu().float().tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            results.extend(self._encode_batch(batch))
        return results

    def embed_query(self, text: str) -> list[float]:
        return self._encode_batch([text])[0]


@lru_cache(maxsize=1)
def get_qwen_embeddings() -> QwenVLEmbeddings:
    return QwenVLEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        device=settings.EMBEDDING_DEVICE,
    )
