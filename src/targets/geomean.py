"""Geometric-mean confidence target.

Source: notebooks/4digit/epochs_100/seed_0/geomean.ipynb, cell 17.
"""
from src.data import PAD_TOKEN

import torch.nn.functional as F

def Q_LOSS(q_halt, f_out, y_true, pad_token=PAD_TOKEN):
    logp = F.log_softmax(f_out, dim=-1)                                        # (B, L, V)
    logp_correct = logp.gather(-1, y_true.unsqueeze(-1)).squeeze(-1)           # (B, L): log probability of the correct token
    non_pad = (y_true != pad_token).float()                                    # (B, L)
    mean_logp = (logp_correct * non_pad).sum(dim=-1, keepdim=True) / (non_pad.sum(dim=-1, keepdim=True) + 1e-8)
    # Detach the confidence target; Q-loss must not backpropagate through it.
    target = mean_logp.exp().detach()                                          # (B, 1), also exp(-mean CE)
    return F.binary_cross_entropy_with_logits(q_halt, target, reduction='none')
