import unittest

from src.b_attention_formula.sv_norm_attention import *
from src.utils.decorators import Timed


class TestSplitValueNormAttention(unittest.TestCase):

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
        vp = torch.randn(self.B, self.H, self.T, self.M)
        vn = torch.randn(self.B, self.H, self.T, self.M)
        return q, k, vp, vn

    @Timed(tag="NonCausalSVNormAttention")
    def test_non_causal_sv_norm_attention(self):
        non_causal_attention = NonCausalSplitValueNormAttention(kv_cache=False)
        for i in range(self.tests):
            q, k, vp, vn = self._generate_data()
            o = non_causal_attention(q, k, vp, vn)
            o2 = naive_split_value_norm_attention(q, k, vp, vn)
            assert torch.allclose(o, o2, atol=self.atol)

    @Timed(tag="NonCausalSVNormAttention (KV Cache)")
    def test_non_causal_kv_cache(self):
        non_causal_attention = NonCausalSplitValueNormAttention(kv_cache=True)
        for i in range(self.tests):
            q, k, vp, vn = self._generate_data()
            half, last = self.T // 2, -1

            # prepare cache
            o2 = naive_split_value_norm_attention(q[:, :, :last], k[:, :, :last], vp[:, :, :last], vn[:, :, :last])
            _ = non_causal_attention(q[:, :, :half], k[:, :, :half], vp[:, :, :half], vn[:, :, :half])

            # batched input
            o = non_causal_attention(q[:, :, half:last], k[:, :, half:last], vp[:, :, half:last], vn[:, :, half:last])
            assert torch.allclose(o, o2[:, :, half:], atol=self.atol)

            # single query input
            o2 = naive_split_value_norm_attention(q, k, vp, vn)
            o = non_causal_attention(q[:, :, last:], k[:, :, last:], vp[:, :, last:], vn[:, :, last:])
            assert torch.allclose(o, o2[:, :, last:], atol=self.atol)

            # reset cache
            non_causal_attention.reset_cache()


if __name__ == '__main__':
    unittest.main()
