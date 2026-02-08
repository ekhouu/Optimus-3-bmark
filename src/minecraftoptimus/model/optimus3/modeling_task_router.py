import os
from pathlib import Path

import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from lightning.pytorch.loggers import WandbLogger  # noqa
from sentence_transformers import SentenceTransformer


class TaskRouterModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super().__init__()

        # Prefer explicit override, then local checkpoint, then HF hub id.
        sentence_bert_path = os.getenv("OPTIMUS_SENTENCE_BERT_PATH")
        if not sentence_bert_path:
            local_ckpt = Path(__file__).resolve().parents[4] / "checkpoint" / "sentence-bert-base"
            sentence_bert_path = str(local_ckpt) if local_ckpt.exists() else "efederici/sentence-bert-base"
        self.bert = SentenceTransformer(sentence_bert_path)
        self.head = nn.Sequential(nn.Linear(768, 768 * 4), nn.ReLU(), nn.Linear(768 * 4, 5))

    def forward(self, x):
        embed = self.bert.encode(x, convert_to_tensor=True, device=self.bert.device, show_progress_bar=False)
        return self.head(embed)

    def router(self, query: str) -> int:
        logits = self.forward(query)
        return logits.argmax(dim=-1).item()
