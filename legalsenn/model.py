# model.py

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class LegalSENN(nn.Module):
    """
    Transformer encoder (e.g. Legal-BERT/BERT) + SENN-inspired head
    for interpretable legal text classification (Model C in the ablation).

        h(x) = W_c z(x) + b_c
        a(x) = W_2 ReLU(W_1 z(x) + b_1) + b_2
        c(x) = h(x) * a(x)
        logits = W c(x) + b
    """

    arch = "senn"

    def __init__(
        self,
        encoder_name: str,
        num_labels: int,
        num_concepts: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 1) Pretrained transformer encoder
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)

        # 2) Concept encoder: z -> h(x)
        self.concept_layer = nn.Linear(hidden_size, num_concepts)

        # 3) Relevance predictor: z -> α(x)
        self.relevance_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_concepts),
        )

        # 4) Classifier: c(x) -> logits
        self.classifier = nn.Linear(num_concepts, num_labels)

    def encode(self, input_ids, attention_mask, inputs_embeds=None):
        enc_out = self.encoder(
            input_ids=None if inputs_embeds is not None else input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        # CLS representation: (B, H)
        return self.dropout(enc_out.last_hidden_state[:, 0, :])

    def forward(self, input_ids, attention_mask, inputs_embeds=None):
        """
        input_ids: (B, T)
        attention_mask: (B, T)
        Returns dict with:
            logits: (B, num_labels)
            concepts: (B, num_concepts)
            relevances: (B, num_concepts)
            contributions: (B, num_concepts)
        """
        cls = self.encode(input_ids, attention_mask, inputs_embeds)

        # Concepts h_i(x)
        concepts = self.concept_layer(cls)

        # Relevances α_i(x)
        relevances = self.relevance_mlp(cls)

        # Element-wise concept contributions c_i(x)
        contributions = concepts * relevances

        logits = self.classifier(contributions)

        return {
            "logits": logits,
            "concepts": concepts,
            "relevances": relevances,
            "contributions": contributions,
        }

    @torch.no_grad()
    def predict(self, input_ids, attention_mask):
        out = self.forward(input_ids, attention_mask)
        probs = F.softmax(out["logits"], dim=-1)
        preds = probs.argmax(dim=-1)
        out["probs"] = probs
        out["preds"] = preds
        return out


class ConceptLayerClassifier(LegalSENN):
    """
    Legal-BERT + concept layer without relevance network (Model B, Eqs. 9-10):

        h(x) = W_c z(x) + b_c
        logits = V h(x) + d

    `contributions` is h(x) so that effect_i(x, y) = V[y, i] * h_i(x) is computed
    exactly as for the full model.
    """

    arch = "concept"

    def __init__(self, encoder_name: str, num_labels: int, num_concepts: int = 32, dropout: float = 0.1):
        nn.Module.__init__(self)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.concept_layer = nn.Linear(hidden_size, num_concepts)
        self.classifier = nn.Linear(num_concepts, num_labels)

    def forward(self, input_ids, attention_mask, inputs_embeds=None):
        cls = self.encode(input_ids, attention_mask, inputs_embeds)
        concepts = self.concept_layer(cls)
        logits = self.classifier(concepts)
        return {"logits": logits, "concepts": concepts, "contributions": concepts}


class LegalBertClassifier(LegalSENN):
    """Standard Legal-BERT document classifier: dropout + linear layer on [CLS] (Model A)."""

    arch = "bert"

    def __init__(self, encoder_name: str, num_labels: int, dropout: float = 0.1, **_):
        nn.Module.__init__(self)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, inputs_embeds=None):
        cls = self.encode(input_ids, attention_mask, inputs_embeds)
        return {"logits": self.classifier(cls)}


class HierarchicalLegalBert(nn.Module):
    """
    Hierarchical Legal-BERT in the style of the LexGLUE SCOTUS baseline
    (Chalkidis et al., 2022): each document is split into up to `max_segments`
    segments of `segment_length` tokens, each segment is encoded independently
    by Legal-BERT ([CLS] vector), a 2-layer transformer encoder with learned
    segment-position embeddings contextualises the segment vectors, and the
    document representation is the max-pool over non-empty segments.
    """

    arch = "hier"

    def __init__(self, encoder_name: str, num_labels: int, max_segments: int = 64, dropout: float = 0.1, **_):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.max_segments = max_segments
        self.seg_pos = nn.Embedding(max_segments + 1, hidden, padding_idx=0)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=self.encoder.config.num_attention_heads,
            dim_feedforward=self.encoder.config.intermediate_size,
            activation="gelu", dropout=dropout, batch_first=True,
        )
        self.seg_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        # input_ids: (B, S, T)
        B, S, T = input_ids.shape
        seg_mask = attention_mask.sum(-1) > 0  # (B, S)
        flat_ids = input_ids[seg_mask]
        flat_am = attention_mask[seg_mask]
        cls = self.encoder(input_ids=flat_ids, attention_mask=flat_am).last_hidden_state[:, 0, :]
        seg = cls.new_zeros(B, S, cls.shape[-1])
        seg[seg_mask] = cls
        pos = torch.arange(1, S + 1, device=input_ids.device).unsqueeze(0).expand(B, S) * seg_mask
        seg = seg + self.seg_pos(pos)
        seg = self.seg_encoder(seg, src_key_padding_mask=~seg_mask)
        seg = seg.masked_fill(~seg_mask.unsqueeze(-1), float("-inf"))
        doc = seg.max(dim=1).values
        return {"logits": self.classifier(self.dropout(doc))}


ARCHS = {
    "senn": LegalSENN,
    "concept": ConceptLayerClassifier,
    "bert": LegalBertClassifier,
    "hier": HierarchicalLegalBert,
}


def build_model(arch: str, encoder_name: str, num_labels: int, num_concepts: int = 32,
                dropout: float = 0.1, **kwargs) -> nn.Module:
    cls = ARCHS[arch]
    if arch in ("senn", "concept"):
        return cls(encoder_name=encoder_name, num_labels=num_labels, num_concepts=num_concepts, dropout=dropout)
    return cls(encoder_name=encoder_name, num_labels=num_labels, dropout=dropout, **kwargs)


def load_from_run(run_dir: str, device="cpu", ckpt_name: str = "best_model.pt") -> nn.Module:
    """
    Rebuild the exact trained architecture from the run's config.json and load its
    checkpoint. The concept dimension is read from the training config, never from a
    concept-name display file.
    """
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    extra = {"max_segments": cfg["max_segments"]} if cfg.get("arch") == "hier" else {}
    model = build_model(
        cfg.get("arch", "senn"), cfg["encoder_name"], cfg["num_labels"],
        num_concepts=cfg.get("num_concepts", 16), **extra,
    )
    state = torch.load(os.path.join(run_dir, ckpt_name), map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
