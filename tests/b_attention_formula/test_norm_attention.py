import unittest

from b_attention_formula.norm_attention import *
from utils.decorators import Timed


class TestNormAttention(unittest.TestCase):

    def setUp(self):
        self.atol = 1e-6
        self.tests = 10000

        self.B = 2
        self.T = 10
        self.H = 3
        self.D = 64
        self.M = 4

    def _generate_data(self):
        q = torch.randn(self.B, self.H, self.T, self.D)
        k = torch.randn(self.B, self.H, self.T, self.D)
        v = torch.randn(self.B, self.H, self.T, self.M)
        return q, k, v

    @Timed(tag="NonCausalAttention")
    def test_non_causal_norm_attention(self):
        non_causal_attention = NonCausalNormAttention(kv_cache=False)
        for i in range(self.tests):
            q, k, v = self._generate_data()
            o = non_causal_attention(q, k, v)
            o2 = naive_norm_attention(q, k, v)
            assert torch.allclose(o, o2, atol=self.atol)

    @Timed(tag="CausalNormAttention")
    def test_causal_norm_attention(self):
        mask = generate_mask(2 * self.T)
        for i in range(self.tests):
            q, k, v = self._generate_data()
            o = causal_norm_attention(q, k, v)
            o2 = naive_norm_attention(q, k, v, mask=extract_mask(mask, self.T))
            assert torch.allclose(o, o2, atol=self.atol)

    @Timed(tag="NonCausalNormAttention (KV Cache)")
    def test_non_causal_kv_cache(self):
        non_causal_attention = NonCausalNormAttention(kv_cache=True)
        for i in range(self.tests):
            q, k, v = self._generate_data()
            half, last = self.T // 2, -1

            # prepare cache
            o2 = naive_norm_attention(q[:, :, :last], k[:, :, :last], v[:, :, :last])
            _ = non_causal_attention(q[:, :, :half], k[:, :, :half], v[:, :, :half])

            # batched input
            o = non_causal_attention(q[:, :, half:last], k[:, :, half:last], v[:, :, half:last])
            assert torch.allclose(o, o2[:, :, half:], atol=self.atol)

            # single query input
            o2 = naive_norm_attention(q, k, v)
            o = non_causal_attention(q[:, :, last:], k[:, :, last:], v[:, :, last:])
            assert torch.allclose(o, o2[:, :, last:], atol=self.atol)

            # reset cache
            non_causal_attention.reset_cache()


if __name__ == '__main__':
    unittest.main()
