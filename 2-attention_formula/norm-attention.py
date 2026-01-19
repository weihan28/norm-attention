from sympy.printing.pytorch import torch
from torch import Tensor


def naive_norm_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None) -> Tensor:
    attn_map = ((q.unsqueeze(-2) + k.unsqueeze(-3)) ** 2).sum(-1) # [..., T, T]
    attn_map = attn_map / attn_map.sum(-1, keepdim=True)

    if mask is not None:
        attn_map = attn_map.masked_fill(mask == 0, 0)
    return attn_map @ v

def non_causal_decoupled_norm_attention(q: Tensor, k: Tensor, v: Tensor):
    T = q.shape[-2]
    # sums and squared norms (should be cached during inference)
    q_ns = q.pow(2).sum(-1) # [...T]
    k_ns = k.pow(2).sum(-1) # [...T]
    k_sum = k.sum(-2) # [...d]
    v_sum = v.sum(-2) # [...m]

    # denominator
    z = T*q_ns # [...T]
    z = z + 2*torch.einsum('...Td, ...d -> ...T', q, k_sum) # +[...T]
    z = z + k_ns.sum(-1, keepdim=True) # +[...1]
    z = 1/z # [...T]

    # numerator
    o = torch.einsum('...T, ...m -> ...Tm', q_ns, v_sum)
    o = o + torch.einsum('...T, ...Tm -> ...m', k_ns, v).unsqueeze(-2)
    o = o + 2*(q @ (k.transpose(-2, -1) @ v)) # [...,T,m]
    return o * z.unsqueeze(-1)

if __name__ == '__main__':
    B, T, H = 2, 10, 3
    D, M = 64, 4

    atol = 1e-6
    tests = 10000
    for i in range(tests):
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, M)
        o = non_causal_decoupled_norm_attention(q, k, v)
        o2 = naive_norm_attention(q, k, v)
        assert torch.allclose(o, o2, atol=atol)