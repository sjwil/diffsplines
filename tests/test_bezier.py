import unittest
import torch
from spline import bezier


class BezierTestCase(unittest.TestCase):
    def setUp(self):
        torch.set_printoptions(precision=10)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_uniform_spline(self):
        # 5 splines, 4 segments, cubic, 2 dimensional
        x = torch.rand([5, 5, 2, 2], device=self.device) * 10
        t = torch.tensor(4, device=self.device)
        times = torch.linspace(0, 4, 5, device=self.device)

        spline = bezier.BezierSpline(t, x)
        # Spline is interpolating
        self.assertAlmostEqual(torch.linalg.norm(spline.position(
            times) - x[..., 0, :]).detach().cpu().item(), 0, 4)

        # Pos, vel, and acc have correct shapes
        pos_shape = spline.position(times).shape
        vel_shape = spline.velocity(times).shape
        acc_shape = spline.acceleration(times).shape
        self.assertEqual(pos_shape[0], vel_shape[0])
        self.assertEqual(pos_shape[1], vel_shape[1])
        self.assertEqual(pos_shape[2], vel_shape[2])
        self.assertEqual(pos_shape[0], acc_shape[0])
        self.assertEqual(pos_shape[1], acc_shape[1])
        self.assertEqual(pos_shape[2], acc_shape[2])

        # Spline is continuous
        self.assertAlmostEqual(torch.linalg.norm(
            spline.position(times[1:] - 0.000001) - x[:, 1:, 0, :]).detach().cpu().item(), 0, 2)

        # Spline is continuously differentiable
        self.assertAlmostEqual(torch.linalg.norm(spline.velocity(
            times[1:]) - spline.velocity(times[1:] - 0.000001)).detach().cpu().item(), 0, 2)

    def test_nonuniform_spline(self):
        # 5 splines, 5 segments, cubic, 3 dimensional
        x = torch.rand([5, 3, 2, 3], device=self.device) * 10
        # t = torch.tensor([0., 1.2, 1.4, 3., 4.], device=self.device)
        t = torch.tensor([0., 1., 3.], device=self.device)

        spline = bezier.BezierSpline(t, x)
        # Spline is interpolating
        self.assertAlmostEqual(torch.linalg.norm(spline.position(
            t) - x[..., 0, :]).detach().cpu().item(), 0, 4)

        # Spline is continuous
        self.assertAlmostEqual(torch.linalg.norm(
            spline.position(t[1:] - 0.000001) - x[:, 1:, 0, :]).detach().cpu().item(), 0, 2)

        # Spline is continuously differentiable
        self.assertAlmostEqual(torch.linalg.norm(spline.velocity(
            t[1:]) - spline.velocity(t[1:] - 0.000001)).detach().cpu().item(), 0, 2)

    def test_nonuniform_noncubic_spline(self):
        # 5 splines, 5 segments, 7 points per curve, 3 dimensional
        x = torch.rand([5, 3, 5, 3], device=self.device) * 10
        # t = torch.tensor([0., 1.2, 1.4, 3., 4.], device=self.device)
        t = torch.tensor([0., 1., 3.], device=self.device)

        spline = bezier.BezierSpline(t, x)
        # Spline is interpolating
        self.assertAlmostEqual(torch.linalg.norm(spline.position(
            t) - x[..., 0, :]).detach().cpu().item(), 0, 4)

        # Spline is continuous
        self.assertAlmostEqual(torch.linalg.norm(
            spline.position(t[1:] - 0.000001) - x[:, 1:, 0, :]).detach().cpu().item(), 0, 2)

        # Spline is continuously differentiable
        self.assertAlmostEqual(torch.linalg.norm(spline.velocity(
            t[1:]) - spline.velocity(t[1:] - 0.000001)).detach().cpu().item(), 0, 2)
