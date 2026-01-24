from enum import StrEnum

import torch
from torch import Tensor, nn


def _norm_squared(x: Tensor) -> Tensor:
    return x.pow(2).sum(-1)


def generate_mask(max_seq_len: int) -> Tensor:
    return torch.tril(torch.ones(max_seq_len, max_seq_len))


def extract_mask(mask: Tensor, T: int) -> Tensor:
    return mask[:T, :T]


def naive_norm_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None, eps=1e-8) -> Tensor:
    attn_map = _norm_squared(q.unsqueeze(-2) + k.unsqueeze(-3))  # [..., T, T]
    attn_map = attn_map + eps
    if mask is not None:
        attn_map = attn_map * mask
    z = 1 / attn_map.sum(-1, keepdim=True)
    return (z * attn_map) @ v


class NormCacheNames(StrEnum):
    T = "T"
    k_sum = "k_sum"
    v_sum = "v_sum"
    k_norm_sum = "k_norm_sum"
    k_norm_v = "k_norm_v"
    kv = "kv"


class NonCausalNormAttention(nn.Module):

    def __init__(self, kv_cache=False):
        super().__init__()
        self.kv_cache = kv_cache
        for name in NormCacheNames:
            self.register_buffer(name, torch.tensor(0), persistent=False)

    def reset_cache(self):
        for name in NormCacheNames:
            setattr(self, name, torch.tensor(0))

    def _add_cache(self, value, name):
        value += getattr(self, name)
        setattr(self, name, value)
        return value

    def forward(self, q, k, v, eps=1e-8):
        """
         :param q: query of shape [..., T, d]
         :param k: key of shape [..., T, d]
         :param v: value of shape [..., T, m]
         :param eps: small value to avoid division by zero
         :return: output of shape [..., T, m]
         """
        T = q.shape[-2]
        q_norm = _norm_squared(q) + eps  # [...T]
        k_norm = _norm_squared(k)  # [...T]

        k_sum = k.sum(-2)  # [...d]
        v_sum = v.sum(-2)  # [...m]
        k_norm_sum = k_norm.sum(-1, keepdim=True)  # [...1]
        k_norm_v = torch.einsum('...T, ...Tm -> ...m', k_norm, v)  # [...m]
        kv = k.transpose(-2, -1) @ v  # [...,d,m]

        if self.kv_cache:
            T = self._add_cache(T, NormCacheNames.T)
            k_sum = self._add_cache(k_sum, NormCacheNames.k_sum)
            v_sum = self._add_cache(v_sum, NormCacheNames.v_sum)
            k_norm_sum = self._add_cache(k_norm_sum, NormCacheNames.k_norm_sum)
            k_norm_v = self._add_cache(k_norm_v, NormCacheNames.k_norm_v)
            kv = self._add_cache(kv, NormCacheNames.kv)

        # denominator
        z = T * q_norm  # [...T]
        z = z + 2 * torch.einsum('...Td, ...d -> ...T', q, k_sum)
        z = z + k_norm_sum + eps
        z = 1 / z  # [...T]

        # numerator
        o = torch.einsum('...T, ...m -> ...Tm', q_norm, v_sum)
        o = o + k_norm_v.unsqueeze(-2)
        o = o + 2 * (q @ kv)  # [...,T,m]
        return o * z.unsqueeze(-1)


def causal_norm_attention(q: Tensor, k: Tensor, v: Tensor, eps=1e-8):
    """
    :param q: query of shape [..., T, d]
    :param k: key of shape [..., T, d]
    :param v: value of shape [..., T, m]
    :param eps: small value to avoid division by zero
    :return: output of shape [..., T, m]
    """
    T = q.shape[-2]
    q_norm = _norm_squared(q) + eps  # [...T]
    k_norm = _norm_squared(k)  # [...T]

    # causal mask is NOT needed during inference, hence caching does not need to be implemented
    k_sum = k.cumsum(-2)  # [...T, d]
    v_sum = v.cumsum(-2)  # [...T, m]
    k_norm_sum = k_norm.cumsum(-1)  # [...T]
    k_norm_v = torch.einsum('...T, ...Tm -> ...Tm', k_norm, v).cumsum(-2)  # [..., T, m]
    kv = torch.einsum('...Td, ...Tm -> ...Tdm', k, v).cumsum(-3)  # [...T,d,m]

    # denominator
    z = torch.arange(start=1, end=T + 1, device=q.device) * q_norm  # [...T]
    z = z + 2 * torch.einsum('...Td, ...Td -> ...T', q, k_sum)
    z = z + k_norm_sum
    z = 1 / z  # [...T]

    # numerator
    o = torch.einsum('...T, ...Tm -> ...Tm', q_norm, v_sum)
    o = o + k_norm_v
    o = o + 2 * torch.einsum('...Td, ...Tdm -> ...Tm', q, kv)  # [...,T,m]
    return o * z.unsqueeze(-1)
