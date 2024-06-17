import unittest
import torch

import matplotlib.pyplot as plt

from spline import bezier, cubic, optimize


class OptimizeTestCase(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_bezier_opt(self):
        # Random points, bezier spline will not be c0
        x = torch.rand([5, 6, 4, 3], device=self.device) * 10
        t = torch.tensor([5.], device=self.device)

        spline = bezier.BezierSpline(t, x)
        print(bezier.c0_c1_violation(spline))

        x, eq_mult, ineq_mult, data = optimize.optimize_spline(t, x, optimize.null_fn_, spline_type="bezier",
                                                               equality_fn=bezier.c0_c1_violation)
        spline = bezier.BezierSpline(t, x)
        print(bezier.c0_c1_violation(spline))

        self.assertAlmostEqual(torch.norm(
            bezier.c0_c1_violation(spline)).detach().cpu().item(), 0, 4)
