"""
models_dl.py — Deep-learning model architectures for the ViHSD project.

Public API:
    BiLSTMClassifier   — 2-layer bidirectional LSTM with pretrained
                          embeddings + dropout + linear head.
    TextCNN            — Kim-style sentence CNN with parallel conv filters
                          over [3, 4, 5]-gram windows + global max-pool.
    count_parameters   — pretty-print trainable / total parameter counts.

Both classifiers expect the BiLSTM-mode collate output:
    input_ids  : LongTensor (batch, seq_len)
    lengths    : LongTensor (batch,) — real lengths on the CPU side, since
                 ``pack_padded_sequence`` requires CPU lengths.

The classification head outputs raw logits over ``num_classes`` (no softmax);
loss is expected to be ``nn.CrossEntropyLoss(weight=class_weights)`` so the
weighted handling lives at the loss, not the model.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMClassifier(nn.Module):
    """2-layer Bi-LSTM classifier with last-hidden pooling.

    The pooling strategy is the standard "final hidden state of the last
    layer in each direction" — concatenated to produce a 2*hidden_dim
    sentence vector. Attention is not used here (kept for a follow-up
    notebook) because the base BiLSTM is the comparison point against the
    Week-3 LR champion.

    Parameters
    ----------
    vocab_size, embedding_dim
        Shape of the embedding lookup. Must match the pretrained matrix
        when one is provided.
    hidden_dim
        Hidden size per direction (so output dim is 2*hidden_dim).
    num_layers
        Stacked LSTM depth. ``dropout`` applies only between layers
        (PyTorch convention — no dropout when num_layers == 1).
    dropout
        Inter-layer LSTM dropout. A separate fixed ``Dropout(0.5)`` runs
        after pooling, before the linear head (this is the regulariser
        that usually matters most).
    pretrained_embeddings
        Optional ``(vocab_size, embedding_dim)`` tensor — typically the
        PhoW2V matrix built in ``04a``.
    freeze_embeddings
        If True, ``embedding.weight.requires_grad = False`` and the
        embeddings stay fixed during training.
    padding_idx
        Index whose embedding is zeroed and excluded from gradient updates.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
        padding_idx: int = 0,
        head_dropout: float = 0.5,
    ) -> None:
        super().__init__()

        if pretrained_embeddings is not None:
            if tuple(pretrained_embeddings.shape) != (vocab_size, embedding_dim):
                raise ValueError(
                    f"pretrained_embeddings shape {tuple(pretrained_embeddings.shape)} "
                    f"≠ expected ({vocab_size}, {embedding_dim})"
                )
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings.clone().float(),
                freeze=freeze_embeddings,
                padding_idx=padding_idx,
            )
        else:
            self.embedding = nn.Embedding(
                vocab_size, embedding_dim, padding_idx=padding_idx,
            )

        # PyTorch only applies inter-layer dropout when num_layers > 1.
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(head_dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.padding_idx = padding_idx

    def set_embedding_frozen(self, frozen: bool) -> None:
        """Toggle ``self.embedding.weight.requires_grad`` for warmup schedules."""
        self.embedding.weight.requires_grad = not frozen

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # pack_padded_sequence silently corrupts state on length=0; surface it
        # as a real error so a buggy Dataset/collate is caught immediately.
        if (lengths < 1).any():
            n_bad = int((lengths < 1).sum())
            raise ValueError(
                f"BiLSTMClassifier got {n_bad} sequences with length<1; "
                f"filter empty samples in the Dataset before batching."
            )

        emb = self.embedding(input_ids)               # (B, L, D)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        _, (h_n, _) = self.lstm(packed)
        # h_n: (num_layers * 2, B, hidden). Last layer fwd=-2, bwd=-1.
        h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (B, 2*hidden_dim)
        return self.fc(self.dropout(h_last))


class TextCNN(nn.Module):
    """Kim-style sentence CNN (Kim, 2014).

    Parallel Conv1d filters over n-gram windows, ReLU, global max-pool over
    time, concat across filter sizes, dropout, linear head. ``lengths`` is
    accepted (for API parity with the BiLSTM) but unused — TextCNN doesn't
    care about padding tokens once we global-max-pool.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        num_filters: int = 100,
        kernel_sizes: Tuple[int, ...] = (3, 4, 5),
        num_classes: int = 3,
        dropout: float = 0.5,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()

        if pretrained_embeddings is not None:
            if tuple(pretrained_embeddings.shape) != (vocab_size, embedding_dim):
                raise ValueError(
                    f"pretrained_embeddings shape {tuple(pretrained_embeddings.shape)} "
                    f"≠ expected ({vocab_size}, {embedding_dim})"
                )
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings.clone().float(),
                freeze=freeze_embeddings,
                padding_idx=padding_idx,
            )
        else:
            self.embedding = nn.Embedding(
                vocab_size, embedding_dim, padding_idx=padding_idx,
            )

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters,
                      kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

        self.kernel_sizes = kernel_sizes
        self.padding_idx = padding_idx

    def forward(self, input_ids: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        # input_ids: (B, L) → emb: (B, L, D) → Conv1d wants (B, D, L)
        x = self.embedding(input_ids).transpose(1, 2)

        # Each conv → (B, num_filters, L'), then ReLU + adaptive max → (B, num_filters)
        pooled = [
            F.adaptive_max_pool1d(F.relu(conv(x)), 1).squeeze(-1)
            for conv in self.convs
        ]
        h = torch.cat(pooled, dim=-1)
        return self.fc(self.dropout(h))


def count_parameters(model: nn.Module, verbose: bool = True) -> Dict[str, int]:
    """Return + optionally print trainable/total/frozen parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    info = {"trainable": trainable, "total": total, "frozen": total - trainable}
    if verbose:
        print(f"Model: {model.__class__.__name__}")
        print(f"  trainable params : {trainable:>10,d}")
        print(f"  total params     : {total:>10,d}")
        print(f"  frozen params    : {info['frozen']:>10,d}")
    return info


if __name__ == "__main__":
    # Smoke test (CPU): build small models and run one forward pass each.
    torch.manual_seed(42)
    V, D, B, L = 100, 16, 4, 8

    bilstm = BiLSTMClassifier(vocab_size=V, embedding_dim=D, hidden_dim=8, num_layers=2)
    ids = torch.randint(1, V, (B, L))
    lengths = torch.tensor([L, L - 1, L - 2, L - 3])
    out = bilstm(ids, lengths)
    print("BiLSTM out:", tuple(out.shape))
    count_parameters(bilstm)

    cnn = TextCNN(vocab_size=V, embedding_dim=D, num_filters=4, kernel_sizes=(2, 3))
    out_cnn = cnn(ids)
    print("TextCNN out:", tuple(out_cnn.shape))
    count_parameters(cnn)
