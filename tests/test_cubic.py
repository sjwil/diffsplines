import unittest
from spline import cubic
import torch


class CubicTestCase(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_uniform_coeffs(self):
        # 5 splines, 5 points (4 polynomials), 3 dim
        x = torch.rand([5, 5, 3], device=self.device) * 10
        t = torch.tensor(4, device=self.device)
        times = torch.linspace(0, 4, 5, device=self.device)
        coeffs = cubic.solve_cubic_coeffs(t, x)
        spline = cubic.CubicSpline(coeffs)
        # Spline is interpolating
        self.assertAlmostEqual(torch.linalg.norm(
            spline.position(times) - x).detach().cpu().item(), 0, 4)

        t, a, b, c, d = coeffs
        xtplus1 = a + b + c + d
        # Spline is continuous
        self.assertAlmostEqual(torch.linalg.norm(
            xtplus1[..., :-1, :] - a[..., 1:, :]).detach().cpu().item(), 0, 4)

        xtdotplus1 = b + 2 * c + 3 * d
        # Spline is continuously differentiable
        self.assertAlmostEqual(torch.linalg.norm(
            xtdotplus1[..., :-1, :] - b[..., 1:, :]).detach().cpu().item(), 0, 4)

        ciplus1 = c + 3 * d
        # Spline is twice continuously differentiable (for natural cubic splines)
        self.assertAlmostEqual(torch.linalg.norm(
            ciplus1[..., :-1, :] - c[..., 1:, :]).detach().cpu().item(), 0, 4)

    def test_nonuniform_coeffs(self):
        # 5 splines, 5 points (4 polynomials), 3 dim
        x = torch.rand([5, 5, 3], device=self.device) * 10
        t = torch.tensor([0., 0.5, 2, 3.5, 4.], device=self.device)
        coeffs = cubic.solve_cubic_coeffs(t, x)
        spline = cubic.CubicSpline(coeffs)
        delta = (t[1:] - t[:-1]).unsqueeze(-1)

        t, a, b, c, d = coeffs
        # Spline is interpolating
        self.assertAlmostEqual(torch.linalg.norm(
            spline.position(t) - x).detach().cpu().item(), 0, 4)

        xtplus1 = a + b * delta + c * delta ** 2 + d * delta ** 3
        # # Spline is continuous
        self.assertAlmostEqual(torch.linalg.norm(
            xtplus1[..., :-1, :] - a[..., 1:, :]).detach().cpu().item(), 0, 4)

        xtdotplus1 = b + 2 * c * delta + 3 * d * delta ** 2
        # Spline is continuously differentiable
        self.assertAlmostEqual(torch.linalg.norm(
            xtdotplus1[..., :-1, :] - b[..., 1:, :]).detach().cpu().item(), 0, 4)

    def test_clamped_coeffs(self):
        # 5 splines, 5 points (4 polynomials), 3 dim
        x = torch.rand([5, 5, 3], device=self.device) * 10
        t = torch.tensor([0., 1, 2, 3.5, 4.], device=self.device)
        with self.assertRaises(ValueError) as v:
            coeffs = cubic.solve_cubic_coeffs(
                t, x, end_condition=cubic.EndCondition.CLAMPED)
        coeffs = cubic.solve_cubic_coeffs(
            t, x, end_condition=cubic.EndCondition.CLAMPED, v_begin=2., v_end=-4.)
        spline = cubic.CubicSpline(coeffs)
        delta = (t[1:] - t[:-1]).unsqueeze(-1)

        t, a, b, c, d = coeffs
        # Spline is interpolating
        self.assertAlmostEqual(torch.linalg.norm(
            spline.position(t) - x).detach().cpu().item(), 0, 4)

        xtplus1 = a + b * delta + c * delta ** 2 + d * delta ** 3
        # # Spline is continuous
        self.assertAlmostEqual(torch.linalg.norm(
            xtplus1[..., :-1, :] - a[..., 1:, :]).detach().cpu().item(), 0, 4)

        xtdotplus1 = b + 2 * c * delta + 3 * d * delta ** 2
        # Spline is continuously differentiable
        self.assertAlmostEqual(torch.linalg.norm(
            xtdotplus1[..., :-1, :] - b[..., 1:, :]).detach().cpu().item(), 0, 4)

    def test_closed_coeffs(self):
        # 5 splines, 5 points (4 polynomials), 3 dim
        x = torch.rand([5, 5, 3], device=self.device) * 10
        # t = torch.tensor([0., 1, 2, 3.5, 4.], device=self.device)
        t = torch.tensor(5., device=self.device)
        coeffs = cubic.solve_cubic_coeffs(
            t, x, end_condition=cubic.EndCondition.CLOSED)
        spline = cubic.CubicSpline(coeffs)
        # Spline is interpolating
        t = torch.tensor([0., 1., 2., 3., 4.], device=self.device)
        self.assertAlmostEqual(torch.linalg.norm(
            spline.position(t) - x).detach().cpu().item(), 0, 4)

        t, a, b, c, d = coeffs
        delta = (t[1:] - t[:-1]).unsqueeze(-1)
        xtplus1 = a + b * delta + c * delta ** 2 + d * delta ** 3
        # # Spline is continuous
        self.assertAlmostEqual(torch.linalg.norm(
            xtplus1[..., :-1, :] - a[..., 1:, :]).detach().cpu().item(), 0, 4)

        xtdotplus1 = b + 2 * c * delta + 3 * d * delta ** 2
        # Spline is continuously differentiable
        self.assertAlmostEqual(torch.linalg.norm(
            xtdotplus1[..., :-1, :] - b[..., 1:, :]).detach().cpu().item(), 0, 4)

    def test_looping_coeffs(self):
        # 5 splines, 5 points (4 polynomials), 3 dim
        x = torch.rand([5, 5, 3], device=self.device) * 10
        t = torch.tensor([0., 1, 2, 3.5, 4., 5.], device=self.device)

        coeffs = cubic.solve_cubic_coeffs(
            t, x, end_condition=cubic.EndCondition.LOOPING, loop_index=0)
        closed_coeffs = cubic.solve_cubic_coeffs(
            t, x, end_condition=cubic.EndCondition.CLOSED)

        t, a, b, c, d = coeffs
        t, c_a, c_b, c_c, c_d = closed_coeffs

        def norm_diff(c1, c2): return torch.linalg.norm(
            c1 - c2).detach().cpu().item()
        # Looping and Closed splines should coincide when loop_index = 0
        self.assertAlmostEqual(norm_diff(a, c_a), 0)
        self.assertAlmostEqual(norm_diff(b, c_b), 0)
        self.assertAlmostEqual(norm_diff(c, c_c), 0)
        self.assertAlmostEqual(norm_diff(d, c_d), 0)

        loop_index = 2
        coeffs = cubic.solve_cubic_coeffs(
            t, x, end_condition=cubic.EndCondition.LOOPING, loop_index=loop_index)

        t, a, b, c, d = coeffs
        delta = (t[1:] - t[:-1]).unsqueeze(-1)
        xtplus1 = a + b * delta + c * delta ** 2 + d * delta ** 3
        # # Spline is continuous
        self.assertAlmostEqual(torch.linalg.norm(
            xtplus1[..., :-1, :] - a[..., 1:, :]).detach().cpu().item(), 0, 4)

        # Spline loops on the correct index
        self.assertAlmostEqual(norm_diff(xtplus1[..., -1, :], a[..., loop_index, :]), 0, 4)

        xtdotplus1 = b + 2 * c * delta + 3 * d * delta ** 2
        # Spline is continuously differentiable
        self.assertAlmostEqual(torch.linalg.norm(
            xtdotplus1[..., :-1, :] - b[..., 1:, :]).detach().cpu().item(), 0, 4)
        
        # Spline loop point is continuously differentiable
        self.assertAlmostEqual(norm_diff(xtdotplus1[..., -1, :], b[..., loop_index, :]), 0, 4)

