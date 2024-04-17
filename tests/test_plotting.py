import unittest
from spline import cubic
from spline import plotting
import torch
import matplotlib.pyplot as plt

class PlottingTestCase(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'



    def test_2d_trajectory(self):
        x = torch.rand([5, 5, 2], device=self.device) * 10
        t = torch.tensor(4, device=self.device)
        coeffs = cubic.solve_cubic_coeffs(t, x)
        
        plotting.plot_2d_trajectory(coeffs)
        
        # Just make sure all code ran
        self.assertEqual(0, 0)
    
    def test_figure(self):
        x = torch.rand([5, 5, 2], device=self.device) * 10
        t = torch.tensor(4, device=self.device)
        coeffs = cubic.solve_cubic_coeffs(t, x)
        eval_t = torch.linspace(0, 4, 200, device=self.device)

        fig, (ax1, ax2) = plt.subplots(2, 1)
        axes = [ax1, ax2]
        plotting.generate_figure(coeffs, eval_t, [50, 150], axes)

        # Just make sure all code ran 
        self.assertEqual(0, 0)
