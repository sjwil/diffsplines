import unittest
from spline import natural
import torch


class NaturalTestCase(unittest.TestCase):
    def setUp(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def test_pulse(self):
        self.assertEqual(0, 0)