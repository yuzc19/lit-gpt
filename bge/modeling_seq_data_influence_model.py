import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn, Tensor
from transformers import AutoModel
from transformers.file_utils import ModelOutput

logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None


class BiEncoderModel(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        normlized: bool = True,
        sentence_pooling_method: str = "cls",
        temperature: float = 0.01,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

        hidden_size = self.model.config.hidden_size
        self.qmat = nn.Linear(hidden_size, hidden_size)
        self.kmat = nn.Linear(hidden_size, hidden_size)
        self.vmat = nn.Linear(hidden_size, hidden_size)

        classifier_dropout = (
            self.model.config.classifier_dropout
            if self.model.config.classifier_dropout is not None
            else self.model.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.model.config.hidden_size, 1)

        with torch.no_grad():
            self.qmat.weight.copy_(torch.eye(hidden_size))
            self.qmat.bias.zero_()
            self.kmat.weight.copy_(torch.eye(hidden_size))
            self.kmat.bias.zero_()
            self.vmat.weight.copy_(torch.eye(hidden_size))
            self.vmat.bias.zero_()

        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.mse = nn.MSELoss()

        self.normlized = normlized
        self.sentence_pooling_method = sentence_pooling_method
        self.temperature = temperature
        self.config = self.model.config

    def encode(self, features):
        if features is None:
            return None
        psg_out = self.model(**features, return_dict=True)
        p_reps = psg_out.last_hidden_state[:, 0]
        if self.normlized:
            p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
        return p_reps.contiguous()

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def attention_pooling(self, q_reps, p_reps):
        q_reps = self.qmat(q_reps)
        k_reps = self.kmat(p_reps)
        v_reps = self.vmat(p_reps)
        group_size = p_reps.size(0) // q_reps.size(0)
        # p_reps = p_reps.view(q_reps.size(0), group_size, -1)
        k_reps = k_reps.view(q_reps.size(0), group_size, -1)
        v_reps = v_reps.view(q_reps.size(0), group_size, -1)
        q_reps_expanded = q_reps.unsqueeze(1)  # Shape: (B, 1, H)
        attention_scores = torch.bmm(
            q_reps_expanded, k_reps.transpose(1, 2)
        )  # Shape: (B, 1, G)
        attention_scores = attention_scores.squeeze(1)  # Shape: (B, G)
        attention_weights = torch.softmax(attention_scores, dim=1)  # Shape: (B, G)
        attention_weights = attention_weights.unsqueeze(1)  # Shape: (B, 1, G)
        weighted_p_reps = torch.bmm(attention_weights, v_reps)  # Shape: (B, G, H)
        # attention_weighted_p_reps = weighted_p_reps.sum(dim=1)  # Shape: (B, H)
        return weighted_p_reps.squeeze(1)

    def forward(
        self,
        query: Dict[str, Tensor] = None,
        passage: Dict[str, Tensor] = None,
        label: Tensor = None,
        use_independent: bool = True,
    ):
        # query size (B, seq_len)
        # passage size (B*G, seq_len)
        q_reps = self.encode(query)
        if not use_independent:
            p_reps = self.encode(passage)
            batch_size = q_reps.size(0)
            group_size = p_reps.size(0) // q_reps.size(0)

            # p_reps = p_reps.view(q_reps.size(0), group_size, -1).mean(dim=1)
            # p_reps = self.attention_pooling(q_reps, p_reps)
            scores = self.compute_similarity(q_reps, p_reps) / self.temperature  # B B*G
            reshaped_scores = scores.view(batch_size, -1, group_size)
            dependent_scores = reshaped_scores.max(dim=-1).values.diagonal()

        pooled_output = self.dropout(q_reps)
        independent_scores = self.classifier(pooled_output).squeeze()

        if use_independent:
            scores = independent_scores
        else:
            scores = independent_scores + dependent_scores
        loss = self.compute_loss(scores, label)
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            # p_reps=p_reps,
        )

    def compute_loss(self, scores, target):
        return self.mse(scores, target)
        # return self.cross_entropy(scores, target)
