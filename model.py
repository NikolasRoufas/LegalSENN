# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class LegalSENN(nn.Module):
    """
    Transformer encoder (e.g. Legal-BERT/BERT) + SENN head
    for interpretable legal text classification.
    """

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

    def forward(self, input_ids, attention_mask):
        """
        input_ids: (B, T)
        attention_mask: (B, T)
        Returns dict with:
            logits: (B, num_labels)
            concepts: (B, num_concepts)
            relevances: (B, num_concepts)
            contributions: (B, num_concepts)
        """
        enc_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # CLS representation: (B, H)
        cls = enc_out.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)

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
