# Some design elements inspired by https://github.com/patrick-kidger/torchcubicspline
import torch
from enum import Enum


# Cubic Hermite splines with natural, clamped, and closed end conditions
class EndCondition(Enum):
    NATURAL = 1
    CLAMPED = 2
    CLOSED = 3


def solve_cubic_coeffs(t, x, end_condition=EndCondition.NATURAL, **kwargs):
    # t: T_max or (..., length)
    # x: (..., length, channels)
    if len(t.size()) == 0:
        t = torch.linspace(0, t, x.shape[-2], device=x.device)
    # Check v_begin and v_end for CLAMPED end conditions
    if end_condition == EndCondition.CLAMPED:
        if "v_begin" not in kwargs.keys() or "v_end" not in kwargs.keys():
            raise ValueError("missing v_begin or v_end for clamped end condition")
    # Check t_closed for CLOSED end conditions
    if end_condition == EndCondition.CLOSED:
        if "t_closed" not in kwargs.keys():
            raise ValueError("missing t_closed for closed end condition")

    delta = (t[1:] - t[:-1]).unsqueeze(-1)
    delta_sq = delta ** 2
    delta_reciprocal = 1 / delta
    delta_reciprocal_sq = 1 / delta_sq

    # TODO: implement clamped and closed end conditions
    if end_condition == EndCondition.NATURAL:
        first = (3 * (x[..., 1, :] - x[..., 0, :])).unsqueeze(-2)
        last = (3 * (x[..., -1, :] - x[..., -2, :])).unsqueeze(-2)
    elif end_condition == EndCondition.CLAMPED:
        first = torch.ones_like(x[..., 0, :]).unsqueeze(-2) * kwargs["v_begin"]
        last = torch.ones_like(x[..., 0, :]).unsqueeze(-2) * kwargs["v_end"]
    elif end_condition == EndCondition.CLOSED:
        first = (3 * (x[..., 1, :] - x[..., -1, :])).unsqueeze(-2)
        last = (3 * (x[..., 0, :] - x[..., -2, :])).unsqueeze(-2)

    # i - 1, i, i + 1
    rhs = 3 * (x[..., 1:-1, :] - x[..., :-2, :]) * delta_reciprocal_sq[:-1] + 3 * (x[..., 2:, :] - x[..., 1:-1, :]) * delta_reciprocal_sq[1:]
    rhs = torch.cat([first, rhs, last], dim=-2)
    # Setup tridiagonal
    tridiagonal = torch.zeros(
        (*x.shape[:-2], x.shape[-2], x.shape[-2]), device=x.device)
    diag = torch.arange(0, x.shape[-2], 1, dtype=int)
    tridiagonal[..., diag[1:-1], diag[1:-1]] = 2 * delta_reciprocal[:-1, 0] + 2 * delta_reciprocal[1:, 0]
    tridiagonal[..., diag[1:-1], diag[2:]] = delta_reciprocal[:-1, 0]
    tridiagonal[..., diag[1:-1], diag[:-2]] = delta_reciprocal[1:, 0]

    # Fix first and last row
    if end_condition == EndCondition.NATURAL:
        tridiagonal[..., 0, 0] = 2 * delta[0]
        tridiagonal[..., 0, 1] = delta[0]
        tridiagonal[..., -1, -1] = 2 * delta[-1]
        tridiagonal[..., -1, -2] = delta[-1]
    elif end_condition == EndCondition.CLAMPED:
        tridiagonal[..., 0, 0] = 1
        tridiagonal[..., -1, -1] = 1
    elif end_condition == EndCondition.CLOSED:
        # Not sure about this here
        t_closed = torch.tensor(kwargs["t_closed"], device=x.device)
        t_closed_reciprocal = 1 / t_closed
        tridiagonal[..., 0, 0] = 2 * delta_reciprocal[0, 0] + 2 * t_closed_reciprocal
        tridiagonal[..., 0, 1] = delta_reciprocal[0, 0]
        tridiagonal[..., 0, -1] = t_closed_reciprocal
        tridiagonal[..., -1, -1] =  2 * t_closed_reciprocal + 2 * delta_reciprocal[-1, 0]
        tridiagonal[..., -1, -2] = delta_reciprocal[-1, 0]
        tridiagonal[..., -1, 0] = t_closed_reciprocal
        # tridiagonal[..., 0, 0] = 2 * delta[0] + 2 * t_closed
        # tridiagonal[..., 0, 1] = delta[0]
        # tridiagonal[..., 0, -1] = t_closed
        # tridiagonal[..., -1, -1] =  2 * t_closed + 2 * delta[-1]
        # tridiagonal[..., -1, -2] = delta[-1]
        # tridiagonal[..., -1, 0] = t_closed

        # Fixing dimensions to add the additional spline
        # Doesn't matter what we cat to it since we will throw away the last element below. Just need to 
        # Have a spline that starts on the previous last element 
        # x = torch.cat([x, x[..., 0, :].unsqueeze(-2)], dim=-2)
        # xdot = torch.cat([xdot, xdot[..., 0, :].unsqueeze(-2)], dim=-2)
        delta_reciprocal = torch.cat([delta_reciprocal.squeeze(), t_closed_reciprocal.unsqueeze(-1)]).unsqueeze(-1)
        delta_reciprocal_sq = torch.cat([delta_reciprocal_sq.squeeze(), t_closed_reciprocal.unsqueeze(-1) ** 2]).unsqueeze(-1)

    # TODO:: efficient tridiagonal solver
    xdot = torch.linalg.solve(tridiagonal, rhs)
    if end_condition == EndCondition.CLOSED:
        # Fixing dimensions to add the additional spline
        x = torch.cat([x, x[..., 0, :].unsqueeze(-2)], dim=-2)
        xdot = torch.cat([xdot, xdot[..., 0, :].unsqueeze(-2)], dim=-2)

    a = x[..., :-1, :]
    b = xdot[..., :-1, :]
    c = 3 * (x[..., 1:, :] - x[..., :-1, :]) * delta_reciprocal_sq - xdot[..., 1:, :] * delta_reciprocal - 2 * xdot[..., :-1, :] * delta_reciprocal
    d = 2 * (x[..., :-1, :] - x[..., 1:, :]) * delta_reciprocal * delta_reciprocal_sq + xdot[..., 1:, :] * delta_reciprocal_sq + xdot[..., :-1, :] * delta_reciprocal_sq

    return t, a, b, c, d


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
        fractional_part = fractional_part.unsqueeze(-1)
        return self.b[..., index, :] + \
            2 * self.c[..., index, :] * fractional_part + \
            3 * self.d[..., index, :] * fractional_part ** 2

    def acceleration(self, times):
        fractional_part, index = self._times_to_indices(times)
        fractional_part = fractional_part.unsqueeze(-1)
        return 2 * self.c[..., index, :] + 6 * self.d[..., index, :] * fractional_part
