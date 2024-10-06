import os
import torch
from transformers import AutoModel, AutoTokenizer, pipeline
from .base import Featurizer

FOLDSEEK_MISSING_IDX = 20



class ProtBertFeaturizer(Featurizer):
    def __init__(self, protbert_path):
        super().__init__("ProtBert", 1024)

        self._max_len = 1024

        self._protbert_tokenizer = AutoTokenizer.from_pretrained(
            protbert_path,
            do_lower_case=False,
        )
        self._protbert_model = AutoModel.from_pretrained(
            protbert_path,
        )
        self._protbert_feat = pipeline(
            "feature-extraction",
            model=self._protbert_model,
            tokenizer=self._protbert_tokenizer,
        )

        self._register_cuda("model", self._protbert_model)
        self._register_cuda("featurizer", self._protbert_feat, self._feat_to_device)

    def _feat_to_device(self, pipe, device):
        from transformers import pipeline

        if device.type == "cpu":
            d = -1
        else:
            d = device.index

        pipe = pipeline(
            "feature-extraction",
            model=self._protbert_model,
            tokenizer=self._protbert_tokenizer,
            device=d,
        )
        self._protbert_feat = pipe
        return pipe

    def _space_sequence(self, x):
        return " ".join(list(x))

    def _transform(self, seq: str):
        if len(seq) > self._max_len - 2:
            seq = seq[: self._max_len - 2]

        embedding = torch.tensor(
            self._cuda_registry["featurizer"][0](self._space_sequence(seq))
        )
        seq_len = len(seq)
        start_Idx = 1
        end_Idx = seq_len + 1
        feats = embedding.squeeze()[start_Idx:end_Idx]

        return feats.mean(0)
