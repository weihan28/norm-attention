import unittest

from src.b_attention_formula.qksv_norm_attention import *
from src.utils.decorators import Timed


class TestQKNormSplitValueNormAttention(unittest.TestCase):

    def setUp(self):
        self.tests = 100

        self.B = 2
        self.T = 100
        self.H = 3
        self.D = 64
        self.M = 4

    def _generate_data(self):
        q = torch.randn(self.B, self.H, self.T, self.D)
        k = torch.randn(self.B, self.H, self.T, self.D)
        vp = torch.randn(self.B, self.H, self.T, self.M)
        vn = torch.randn(self.B, self.H, self.T, self.M)
        self.atol = torch.finfo(q.dtype).eps
        return q, k, vp, vn

    @Timed(tag="NonCausalQKSVNormAttention")
    def test_non_causal(self):
        non_causal_attention = NonCausalQKSplitValueNormAttention(kv_cache=False)
        for _ in range(self.tests):
            q, k, vp, vn = self._generate_data()
            o = non_causal_attention(q, k, vp, vn)
            o2 = naive_qk_split_value_norm_attention(q, k, vp, vn)
            assert torch.allclose(o, o2, atol=self.atol)

    @Timed(tag="CausalQKSVNormAttention")
    def test_causal(self):
        mask = generate_mask(2 * self.T)
        for _ in range(self.tests):
            q, k, vp, vn = self._generate_data()
            o = causal_qk_split_value_norm_attention(q, k, vp, vn)
            o2 = naive_qk_split_value_norm_attention(q, k, vp, vn, mask=extract_mask(mask, self.T))
            assert torch.allclose(o, o2, atol=self.atol)

    @Timed(tag="NonCausalQKSVNormAttention (KV Cache)")
    def test_non_causal_kv_cache(self):
        non_causal_attention = NonCausalQKSplitValueNormAttention(kv_cache=True)
        for _ in range(self.tests):
            q, k, vp, vn = self._generate_data()
            half, last = self.T // 2, -1

            # prepare cache
            o2 = naive_qk_split_value_norm_attention(q[:, :, :last], k[:, :, :last], vp[:, :, :last], vn[:, :, :last])
            _ = non_causal_attention(q[:, :, :half], k[:, :, :half], vp[:, :, :half], vn[:, :, :half])

            # batched input
            o = non_causal_attention(q[:, :, half:last], k[:, :, half:last], vp[:, :, half:last], vn[:, :, half:last])
            assert torch.allclose(o, o2[:, :, half:], atol=self.atol)

            # single query input
            o2 = naive_qk_split_value_norm_attention(q, k, vp, vn)
            o = non_causal_attention(q[:, :, last:], k[:, :, last:], vp[:, :, last:], vn[:, :, last:])
            assert torch.allclose(o, o2[:, :, last:], atol=self.atol)

            # reset cache
            non_causal_attention.reset_cache()


if __name__ == '__main__':
    unittest.main()
