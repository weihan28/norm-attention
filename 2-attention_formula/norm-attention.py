from sympy.printing.pytorch import torch
from torch import Tensor

def _norm_squared(x: Tensor) -> Tensor:
    return x.pow(2).sum(-1)

def generate_mask(max_seq_len: int) -> Tensor:
    return torch.tril(torch.ones(max_seq_len, max_seq_len))

def extract_mask(mask: Tensor, T: int) -> Tensor:
    return mask[:T, :T]

def naive_norm_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None) -> Tensor:
    attn_map = _norm_squared(q.unsqueeze(-2) + k.unsqueeze(-3)) # [..., T, T]
    if mask is not None:
        attn_map = attn_map * mask
    z = 1/attn_map.sum(-1, keepdim=True)
    return (z * attn_map) @ v

def non_causal_decoupled_norm_attention(q: Tensor, k: Tensor, v: Tensor):
    """
    :param q: query of shape [..., T, d]
    :param k: key of shape [..., T, d]
    :param v: value of shape [..., T, m]
    :return: output of shape [..., T, m]
    """
    T = q.shape[-2]
    q_ns = _norm_squared(q) # [...T]
    k_ns = _norm_squared(k)  # [...T]

    # cache during inference (all of these do not contain T)
    k_sum = k.sum(-2) # [...d]
    v_sum = v.sum(-2) # [...m]
    k_ns_sum = k_ns.sum(-1, keepdim=True) # [...1]
    k_ns_v = torch.einsum('...T, ...Tm -> ...m', k_ns, v) # [...m]
    kv = k.transpose(-2, -1) @ v # [...,d,m]

    # denominator
    z = T*q_ns # [...T]
    z = z + 2*torch.einsum('...Td, ...d -> ...T', q, k_sum)
    z = z + k_ns_sum
    z = 1/z # [...T]

    # numerator
    o = torch.einsum('...T, ...m -> ...Tm', q_ns, v_sum)
    o = o + k_ns_v.unsqueeze(-2)
    o = o + 2*(q @ kv) # [...,T,m]
    return o * z.unsqueeze(-1)

def causal_decoupled_norm_attention(q: Tensor, k: Tensor, v: Tensor):
    """
    :param q: query of shape [..., T, d]
    :param k: key of shape [..., T, d]
    :param v: value of shape [..., T, m]
    :return: output of shape [..., T, m]
    """
    T = q.shape[-2]
    q_ns = _norm_squared(q)  # [...T]
    k_ns = _norm_squared(k)  # [...T]

    # causal mask is NOT needed during inference, hence caching does not need to be implemented
    k_sum = k.cumsum(-2)  # [...T, d]
    v_sum = v.cumsum(-2)  # [...T, m]
    k_ns_sum = k_ns.cumsum(-1)  # [...T]
    k_ns_v = torch.einsum('...T, ...Tm -> ...Tm', k_ns, v).cumsum(-2) # [..., T, m]
    kv = torch.einsum('...Td, ...Tm -> ...Tdm', k, v).cumsum(-3)  # [...T,d,m]

    # denominator
    z = torch.arange(start=1, end=T+1, device=q.device) * q_ns  # [...T]
    z = z + 2 * torch.einsum('...Td, ...Td -> ...T', q, k_sum)
    z = z + k_ns_sum
    z = 1 / z  # [...T]

    # numerator
    o = torch.einsum('...T, ...Tm -> ...Tm', q_ns, v_sum)
    o = o + k_ns_v
    o = o + 2*torch.einsum('...Td, ...Tdm -> ...Tm', q, kv)  # [...,T,m]
    return o * z.unsqueeze(-1)

if __name__ == '__main__':
    import time
    B, T, H = 2, 10, 3
    D, M = 64, 4

    atol = 1e-6
    tests = 10000

    print("Running Test for non causal norm attention")
    start = time.perf_counter()
    for i in range(tests):
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, M)
        o = non_causal_decoupled_norm_attention(q, k, v)
        o2 = naive_norm_attention(q, k, v)
        assert torch.allclose(o, o2, atol=atol)
    end = time.perf_counter()
    print(f"{end - start:.6f} s")

    print("Running Test for causal norm attention")
    start = time.perf_counter()
    mask = generate_mask(2 * T)
    for i in range(tests):
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, M)
        o = causal_decoupled_norm_attention(q, k, v)
        o2 = naive_norm_attention(q, k, v, mask=extract_mask(mask, T))
        assert torch.allclose(o, o2, atol=atol)
    end = time.perf_counter()
    print(f"{end - start:.6f} s")