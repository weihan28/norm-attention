import unittest

import torch

from src.b_attention_formula.ar_norm_attention import naive_ar_split_value_norm_attention, \
    ARNonCausalSplitValueNormAttention, extract_mask, generate_mask, ar_causal_sv_norm_attention
from src.utils.decorators import Timed


class TestARSplitValueNormAttention(unittest.TestCase):

    def setUp(self):
        self.atol = 1e-6
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
        return q, k, vp, vn

    @Timed(tag="NonCausalARSVNormAttention")
    def test_non_causal(self):
        non_causal_attention = ARNonCausalSplitValueNormAttention(kv_cache=False)
        for i in range(self.tests):
            q, k, vp, vn = self._generate_data()
            q_2, k_2, vp_2, vn_2 = self._generate_data()

            # run 2 layers of ar attention
            o, z1 = naive_ar_split_value_norm_attention(q, k, vp_2, vn_2)
            o_t, z2 = naive_ar_split_value_norm_attention(q_2, k_2, vp_2, vn_2)
            o = o + o_t
            z = (1/(z1 + z2))
            o = o * z

            o2 = non_causal_attention(q_2, k_2, vp_2, vn_2, q_prev=q, k_prev=k)
            assert torch.allclose(o, o2, atol=self.atol)

    @Timed(tag="CausalARSVNormAttention")
    def test_causal(self):
        mask = generate_mask(2 * self.T)
        mask  = extract_mask(mask, self.T)
        for i in range(self.tests):
            q, k, vp, vn = self._generate_data()
            q_2, k_2, vp_2, vn_2 = self._generate_data()

            o, z1 = naive_ar_split_value_norm_attention(q, k, vp_2, vn_2, mask=mask)
            o_t, z2 = naive_ar_split_value_norm_attention(q_2, k_2, vp_2, vn_2, mask=mask)
            o = o + o_t
            z = (1 / (z1 + z2))
            o = o * z

            o2 = ar_causal_sv_norm_attention(q_2, k_2, vp_2, vn_2, q_prev=q, k_prev=k)
            assert torch.allclose(o, o2, atol=self.atol)


if __name__ == '__main__':
    unittest.main()
