# Design inspired by https://github.com/patrick-kidger/torchcubicspline
import torch
from enum import Enum


# Cubic Hermite splines with natural, clamped, and closed end conditions
class EndCondition(Enum):
    NATURAL = 1
    CLAMPED = 2
    CLOSED = 3


def solve_cubic_coeffs(t, x, end_condition=EndCondition.NATURAL):
    # t: T_max or (..., length)
    # x: (..., length, channels)
    uniform = len(t.size()) == 0
    if uniform:
        # Note this rhs is specific to natural
        first = (3 * (x[..., 1, :] - x[..., 0, :])).unsqueeze(-2)
        last = (3 * (x[..., -1, :] - x[..., -2, :])).unsqueeze(-2)
        rhs = 3 * (x[..., 2:, :] - x[..., :-2, :])
        rhs = torch.cat([first, rhs, last], dim=-2)
        # Setup tridiagonal
        tridiagonal = torch.zeros(
            (*x.shape[:-2], x.shape[-2], x.shape[-2]), device=x.device)
        diag = torch.arange(0, x.shape[-2] - 1, 1, dtype=int)
        tridiagonal[..., diag, diag] = 4
        tridiagonal[..., diag + 1, diag] = 1
        tridiagonal[..., diag, diag + 1] = 1
        # Fix first and last row
        if end_condition == EndCondition.NATURAL:
            tridiagonal[..., 0, 0] = 2
            tridiagonal[..., -1, -2] = 2
        # TODO:: efficient tridiagonal solver
        xdot = torch.linalg.solve(tridiagonal, rhs)
        a = x[..., :-1, :]
        b = xdot[..., :-1, :]
        c = 3 * (x[..., 1:, :] - x[..., :-1, :]) - xdot[..., 1:, :] - 2 * xdot[..., :-1, :]
        d = 2 * (x[..., :-1, :] - x[..., 1:, :]) + xdot[..., 1:, :] + xdot[..., :-1, :]

        # TODO: Currently assumes unit time between each endpoint
        t = torch.linspace(0, x.shape[0] - 1, x.shape[0], device=x.device)
        return t, a, b, c, d
    else:
        raise NotImplementedError("Not implemented!")


class CubicSpline:
    def __init__(self, coeffs):
        t, a, b, c, d = coeffs
        self.t = t
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def _times_to_indices(self, times):
        index = torch.bucketize(times, self.t, right=True) - 1
        index = index.clamp(0, self.a.size(-2) - 1)
        fractional_part = times - self.t[index]
        return fractional_part, index

    def position(self, times):
        fractional_part, index = self._times_to_indices(times)
        fractional_part = fractional_part.unsqueeze(-1)

        return self.a[..., index, :] + self.b[..., index, :] * fractional_part + \
            self.c[..., index, :] * fractional_part ** 2 + \
            self.d[..., index, :] * fractional_part ** 3

    def velocity(self, times):
        fractional_part, index = self._times_to_indices(times)
        return self.b[..., index, :] + \
            2 * self.c[..., index, :] * fractional_part + \
            3 * self.d[..., index, :] * fractional_part ** 2

    def acceleration(self, times):
        fractional_part, index = self._times_to_indices(times)
        return 2 * self.c[..., index, :] + 6 * self.d[..., index, :] * fractional_part
