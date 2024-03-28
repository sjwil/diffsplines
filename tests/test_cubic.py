import unittest
from spline import cubic
import torch


class CubicTestCase(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def test_uniform_coeffs(self):
        # 5 splines, 5 length, 3 dim
        x = torch.rand([5, 5, 3], device=self.device) * 10
        t = torch.tensor(5, device=self.device)
        times = torch.linspace(0, 4, 5, device=self.device)
        coeffs = cubic.solve_cubic_coeffs(t, x)
        spline = cubic.CubicSpline(coeffs)
        # Spline is interpolating
        self.assertAlmostEqual(torch.linalg.norm(spline.position(times) - x).detach().cpu().item(), 0, 4)

        t, a, b, c, d = coeffs
        xtplus1 = a + b + c + d
        # Spline is continuous
        self.assertAlmostEqual(torch.linalg.norm(xtplus1[..., :-1, :] - a[..., 1:, :]).detach().cpu().item(), 0, 4)

        xtdotplus1 = b + 2 * c + 3 * d
        # Spline is continuously differentiable
        self.assertAlmostEqual(torch.linalg.norm(xtdotplus1[..., :-1, :] - b[..., 1:, :]).detach().cpu().item(), 0, 4)

        ciplus1 = c + 3 * d
        # Spline is twice continuously differentiable (for natural cubic splines)
        self.assertAlmostEqual(torch.linalg.norm(ciplus1[..., :-1, :] - c[..., 1:, :]).detach().cpu().item(), 0, 4)
        
