import torch
from torch import Tensor


def generate_mask(max_seq_len: int) -> Tensor:
    return torch.tril(torch.ones(max_seq_len, max_seq_len))


def extract_mask(mask: Tensor, T: int) -> Tensor:
    return mask[:T, :T]
