from enum import StrEnum

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _norm_squared(x: Tensor) -> Tensor:
    return x.pow(2).sum(-1)


def generate_mask(max_seq_len: int) -> Tensor:
    return torch.tril(torch.ones(max_seq_len, max_seq_len))


def extract_mask(mask: Tensor, T: int) -> Tensor:
    return mask[:T, :T]


def _rms_norm(x):
    return F.rms_norm(x, (x.size(-1),))
    # return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True))


def naive_qk_split_value_norm_attention(q: Tensor, k: Tensor, v_p: Tensor, v_n: Tensor, mask: Tensor = None) -> Tensor:
    q = _rms_norm(q)
    k = _rms_norm(k)

    attn_map_p = _norm_squared(q.unsqueeze(-2) + k.unsqueeze(-3))  # [..., T, T]
    attn_map_n = _norm_squared(q.unsqueeze(-2) + (-k).unsqueeze(-3))  # [..., T, T]
    if mask is not None:
        attn_map_p = attn_map_p * mask
        attn_map_n = attn_map_n * mask
    z = attn_map_p.sum(-1, keepdim=True) + attn_map_n.sum(-1, keepdim=True)
    z = 1 / z
    return z * (attn_map_p @ v_p + attn_map_n @ v_n)


class QKSVCacheNames(StrEnum):
    T = "T"
    vp_sum = "vp_sum"
    k_vn = "k_vn"


class NonCausalQKSplitValueNormAttention(nn.Module):

    def __init__(self, kv_cache=False):
        super().__init__()
        self.kv_cache = kv_cache
        for name in QKSVCacheNames:
            self.register_buffer(name, torch.tensor(0), persistent=False)

    def reset_cache(self):
        for name in QKSVCacheNames:
            setattr(self, name, torch.tensor(0))

    def _add_cache(self, value, name):
        value += getattr(self, name)
        setattr(self, name, value)
        return value

    def forward(self, q, k, vp, vn):
        """
        :param q: query of shape [..., T, d]
        :param k: key of shape [..., T, d]
        :param vp: value of shape [..., T, m]
        :param vn: value of shape [..., T, m]
        :return: output of shape [..., T, m]
        """
        q = _rms_norm(q)
        k = _rms_norm(k)

        vp, vn = vp + vn, vp - vn

        T, d = q.shape[-2:]

        vp_sum = vp.sum(-2)  # [...m]
        k_vn = k.transpose(-2, -1) @ vn  # [...,d,m]

        if self.kv_cache:
            T = self._add_cache(T, QKSVCacheNames.T)
            vp_sum = self._add_cache(vp_sum, QKSVCacheNames.vp_sum)
            k_vn = self._add_cache(k_vn, QKSVCacheNames.k_vn)

        # denominator
        z = 1 / (2 * T)

        # numerator
        o = vp_sum.unsqueeze(-2)  # [...,1,m]
        o = o + (1 / d) * (q @ k_vn)  # [...,T,m]
        return o * z


def causal_qk_split_value_norm_attention(q: Tensor, k: Tensor, vp: Tensor, vn: Tensor):
    """
    :param q: query of shape [..., T, d]
    :param k: key of shape [..., T, d]
    :param vp: value of shape [..., T, m]
    :param vn: value of shape [..., T, m]
    :return: output of shape [..., T, m]
    """
    q = _rms_norm(q)
    k = _rms_norm(k)
    vp, vn = vp + vn, vp - vn

    T, d = q.shape[-2:]

    vp_sum = vp.cumsum(-2)  # [...T, m]
    k_vn = torch.einsum('...Td, ...Tm -> ...Tdm', k, vn).cumsum(-3)  # [...T,d,m]

    # denominator
    z = 2 * torch.arange(start=1, end=T + 1, device=q.device)  # [...T]
    z = 1 / z

    # numerator
    o = vp_sum
    o = o + (1 / d) * torch.einsum('...Td, ...Tdm -> ...Tm', q, k_vn)  # [...,T,m]
    return o * z.unsqueeze(-1)
