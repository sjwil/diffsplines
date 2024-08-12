import unittest
import torch

import matplotlib.pyplot as plt

import time

from spline import bezier, cubic, optimize


class OptimizeTestCase(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_bezier_opt(self):
        # Random points, bezier spline will not be c1
        x = torch.rand([5, 6, 3, 3], device=self.device) * 10
        t = torch.tensor(5., device=self.device)

        control_points = bezier.adapt_c0_bezier(x)
        spline = bezier.BezierSpline(t, control_points)
        print("begin")
        print(bezier.c0_c1_violation(spline))


        t0 = time.time()
        x, eq_mult, ineq_mult, data = optimize.optimize_spline(t, x, optimize.null_fn_, spline_type="bezier",
                                                               equality_fn=bezier.c0_c1_violation)
        
        # print(data["dual_traj"])
        print(eq_mult)
        print(ineq_mult)
        
        elapsed = time.time() - t0
        print(elapsed, "time elapsed")
        control_points = bezier.adapt_c0_bezier(x)
        spline = bezier.BezierSpline(t, control_points)
        print("end")
        print(bezier.c0_c1_violation(spline))

        self.assertAlmostEqual(torch.norm(
            bezier.c0_c1_violation(spline)).detach().cpu().item(), 0, 4)
