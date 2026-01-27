from enum import StrEnum

import torch
from torch import Tensor, nn


def _norm_squared(x: Tensor) -> Tensor:
    return x.pow(2).sum(-1)


def generate_mask(max_seq_len: int) -> Tensor:
    return torch.tril(torch.ones(max_seq_len, max_seq_len))


def extract_mask(mask: Tensor, T: int) -> Tensor:
    return mask[:T, :T]


def naive_ar_split_value_norm_attention(q: Tensor, k: Tensor, v_p: Tensor, v_n: Tensor, mask: Tensor = None) -> tuple[
    Tensor, Tensor]:
    attn_map_p = _norm_squared(q.unsqueeze(-2) + k.unsqueeze(-3))  # [..., T, T]
    attn_map_n = _norm_squared(q.unsqueeze(-2) + (-k).unsqueeze(-3))  # [..., T, T]
    if mask is not None:
        attn_map_p = attn_map_p * mask
        attn_map_n = attn_map_n * mask
    z = attn_map_p.sum(-1, keepdim=True) + attn_map_n.sum(-1, keepdim=True)
    return (attn_map_p @ v_p + attn_map_n @ v_n), z # multiply z outside


class ARSVCacheNames(StrEnum):
    T = "T"
    vp_sum = "vp_sum"
    vn_sum = "vn_sum"
    k_norm_sum = "k_norm_sum"
    k_norm_vp = "k_norm_vp"
    k_vn = "k_vn"
    k_vn_prev = "k_vn_prev"


class ARNonCausalSplitValueNormAttention(nn.Module):

    def __init__(self, kv_cache=False):
        super().__init__()
        self.kv_cache = kv_cache
        for name in ARSVCacheNames:
            self.register_buffer(name, torch.tensor(0), persistent=False)

    def reset_cache(self):
        for name in ARSVCacheNames:
            setattr(self, name, torch.tensor(0))

    def _add_cache(self, value, name):
        value += getattr(self, name)
        setattr(self, name, value)
        return value

    def forward(self,
                q,
                k,
                vp,
                vn,
                q_prev = None,
                k_prev = None):
        """
        :param k_prev:
        :param q_prev:
        :param q: query of shape [..., T, d]
        :param k: key of shape [..., T, d]
        :param vp: value of shape [..., T, m]
        :param vn: value of shape [..., T, m]
        :return: output of shape [..., T, m]
        """
        vp, vn = vp + vn, vp - vn

        T = q.shape[-2]

        q_norm = _norm_squared(q)   # [...T]
        k_norm = _norm_squared(k)   # [...T]

        if q_prev is not None:
            q_norm = q_norm + _norm_squared(q_prev)
            k_norm = k_norm + _norm_squared(k_prev)

        vp_sum = vp.sum(-2)  # [...m]
        k_norm_sum = k_norm.sum(-1, keepdim=True)  # [...1]
        k_norm_vp = torch.einsum('...T, ...Tm -> ...m', k_norm, vp)  # [...m]
        k_vn = k.transpose(-2, -1) @ vn  # [...,d,m]

        if q_prev is not None:
            k_vn_prev = k_prev.transpose(-2, -1) @ vn  # [...,d,m]

        if self.kv_cache:
            T = self._add_cache(T, ARSVCacheNames.T)
            vp_sum = self._add_cache(vp_sum, ARSVCacheNames.vp_sum)
            k_norm_sum = self._add_cache(k_norm_sum, ARSVCacheNames.k_norm_sum)
            k_norm_vp = self._add_cache(k_norm_vp, ARSVCacheNames.k_norm_vp)
            k_vn = self._add_cache(k_vn, ARSVCacheNames.k_vn)

            if q_prev is not None:
                k_vn_prev = self._add_cache(k_vn_prev, ARSVCacheNames.k_vn_prev)


        # denominator
        z = T * q_norm  # [...T]
        z = z + k_norm_sum
        z = 1 / (2 * z)  # [...T]

        # numerator
        o = torch.einsum('...T, ...m -> ...Tm', q_norm, vp_sum)
        o = o + k_norm_vp.unsqueeze(-2)
        o = o + 2 * (q @ k_vn)  # [...,T,m]

        if q_prev is not None:
            o = o + 2 * (q_prev @ k_vn_prev)  # [...,T,m]
        return o * z.unsqueeze(-1)


def ar_causal_sv_norm_attention(
        q: Tensor,
        k: Tensor,
        vp: Tensor,
        vn: Tensor,
        q_prev: Tensor = None,
        k_prev: Tensor = None):
    """
    :param k_prev:
    :param q_prev:
    :param q: query of shape [..., T, d]
    :param k: key of shape [..., T, d]
    :param vp: value of shape [..., T, m]
    :param vn: value of shape [..., T, m]
    :return: output of shape [..., T, m]
    """
    vp, vn = vp + vn, vp - vn

    T = q.shape[-2]
    q_norm = _norm_squared(q)  # [...T]
    k_norm = _norm_squared(k)  # [...T]
    if q_prev is not None:
        q_norm = q_norm + _norm_squared(q_prev)
        k_norm = k_norm + _norm_squared(k_prev)

    vp_sum = vp.cumsum(-2)  # [...T, m]
    k_norm_sum = k_norm.cumsum(-1)  # [...T]
    k_norm_vp = torch.einsum('...T, ...Tm -> ...Tm', k_norm, vp).cumsum(-2)  # [..., T, m]
    k_vn = torch.einsum('...Td, ...Tm -> ...Tdm', k, vn).cumsum(-3)  # [...T,d,m]
    if q_prev is not None:
        k_vn_prev = torch.einsum('...Td, ...Tm -> ...Tdm', k_prev, vn).cumsum(-3)  # [...T,d,m]

    # denominator
    z = torch.arange(start=1, end=T + 1, device=q.device) * q_norm  # [...T]
    z = z + k_norm_sum
    z = 1 / (2 * z)  # [...T]

    # numerator
    o = torch.einsum('...T, ...Tm -> ...Tm', q_norm, vp_sum)
    o = o + k_norm_vp
    o = o + 2 * torch.einsum('...Td, ...Tdm -> ...Tm', q, k_vn)  # [...,T,m]
    if q_prev is not None:
        o = o + 2 * torch.einsum('...Td, ...Tdm -> ...Tm', q_prev, k_vn_prev)  # [...,T,m]
    return o * z.unsqueeze(-1)
