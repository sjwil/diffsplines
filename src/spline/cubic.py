# Some design elements inspired by https://github.com/patrick-kidger/torchcubicspline
import torch
from enum import Enum


# Cubic Hermite splines with natural, clamped, and closed end conditions
class EndCondition(Enum):
    NATURAL = 1
    CLAMPED = 2
    CLOSED = 3
    LOOPING = 4


def _closed_or_looping(end_condition):
    return end_condition == EndCondition.CLOSED or end_condition == EndCondition.LOOPING


def solve_cubic_coeffs(t, x, end_condition=EndCondition.NATURAL, **kwargs):
    # t: T_max or (..., length)
    # x: (..., length, channels)
    if len(t.size()) == 0:
        t = torch.linspace(0, t, x.shape[-2] if not _closed_or_looping(
            end_condition) else x.shape[-2] + 1, device=x.device)
    # Check v_begin and v_end for CLAMPED end conditions
    if end_condition == EndCondition.CLAMPED:
        if "v_begin" not in kwargs.keys() or "v_end" not in kwargs.keys():
            raise ValueError(
                "missing v_begin or v_end for clamped end condition")
    # Check t_closed for CLOSED end conditions
    if _closed_or_looping(end_condition):
        if t.shape[0] != x.shape[-2] + 1:
            raise ValueError(
                "incorrect number of time points for closed or looping cubic spline")
        if end_condition == EndCondition.LOOPING:
            if "loop_index" not in kwargs.keys():
                raise ValueError(
                    "missing loop_index for looping end condition")

    delta = (t[1:] - t[:-1]).unsqueeze(-1)
    delta_sq = delta ** 2
    delta_reciprocal = 1 / delta
    delta_reciprocal_sq = 1 / delta_sq
    delta_plus1_index = x.shape[-2] if not _closed_or_looping(
        end_condition) else -1
    delta_plus0_index = -1 if not _closed_or_looping(end_condition) else -2

    if end_condition == EndCondition.NATURAL:
        first = (3 * (x[..., 1, :] - x[..., 0, :])).unsqueeze(-2)
        last = (3 * (x[..., -1, :] - x[..., -2, :])).unsqueeze(-2)
    elif end_condition == EndCondition.CLAMPED:
        first = torch.ones_like(x[..., 0, :]).unsqueeze(-2) * kwargs["v_begin"]
        last = torch.ones_like(x[..., 0, :]).unsqueeze(-2) * kwargs["v_end"]
    elif _closed_or_looping(end_condition):
        loop_index = kwargs.get("loop_index", 0)
        if loop_index == 0:
            first = (3 * (x[..., 1, :] - x[..., -1, :])).unsqueeze(-2)
            last = (3 * (x[..., 0, :] - x[..., -2, :])).unsqueeze(-2)
        else:
            # Natural condition for start of curve
            first = (3 * (x[..., 1, :] - x[..., 0, :])).unsqueeze(-2)
            # Equivalent to first condition for closed spline, C^2 continuity between last and loop
            last = (3 * (x[..., loop_index + 1, :] -
                    x[..., -1, :])).unsqueeze(-2)

    rhs = 3 * (x[..., 1:-1, :] - x[..., :-2, :]) * delta_reciprocal_sq[:delta_plus0_index] + \
        3 * (x[..., 2:, :] - x[..., 1:-1, :]) * \
        delta_reciprocal_sq[1:delta_plus1_index]
    rhs = torch.cat([first, rhs, last], dim=-2)
    # Setup tridiagonal matrix
    tridiagonal = torch.zeros(
        (*x.shape[:-2], x.shape[-2], x.shape[-2]), device=x.device)
    diag = torch.arange(0, x.shape[-2], 1, dtype=int)
    tridiagonal[..., diag[1:-1], diag[1:-1]] = 2 * delta_reciprocal[:delta_plus0_index,
                                                                    0] + 2 * delta_reciprocal[1:delta_plus1_index, 0]
    tridiagonal[..., diag[1:-1], diag[2:]
                ] = delta_reciprocal[:delta_plus0_index, 0]
    tridiagonal[..., diag[1:-1], diag[:-2]
                ] = delta_reciprocal[1:delta_plus1_index, 0]

    # Fix first and last row
    if end_condition == EndCondition.NATURAL:
        tridiagonal[..., 0, 0] = 2 * delta[0]
        tridiagonal[..., 0, 1] = delta[0]
        tridiagonal[..., -1, -1] = 2 * delta[-1]
        tridiagonal[..., -1, -2] = delta[-1]
    elif end_condition == EndCondition.CLAMPED:
        tridiagonal[..., 0, 0] = 1
        tridiagonal[..., -1, -1] = 1
    elif _closed_or_looping(end_condition):
        if loop_index == 0:
            tridiagonal[..., 0, 0] = 2 * delta_reciprocal[0, 0] + \
                2 * delta_reciprocal[-1, 0]
            tridiagonal[..., 0, 1] = delta_reciprocal[0, 0]
            tridiagonal[..., 0, -1] = delta_reciprocal[-1, 0]
            tridiagonal[..., -1, -1] = 2 * \
                delta_reciprocal[-1, 0] + 2 * delta_reciprocal[-2, 0]
            tridiagonal[..., -1, -2] = delta_reciprocal[-2, 0]
            tridiagonal[..., -1, 0] = delta_reciprocal[-1, 0]
        else:
            # Natural beginning
            tridiagonal[..., 0, 0] = 2 * delta[0]
            tridiagonal[..., 0, 1] = delta[0]
            # Looping end
            tridiagonal[..., -1, loop_index] = 2 * \
                delta_reciprocal[loop_index, 0] + 2 * delta_reciprocal[-1, 0]
            tridiagonal[..., -1, loop_index +
                        1] = delta_reciprocal[loop_index, 0]
            tridiagonal[..., -1, -1] = delta_reciprocal[-1, 0]

    # TODO:: efficient tridiagonal solver
    xdot = torch.linalg.solve(tridiagonal, rhs)
    if _closed_or_looping(end_condition):
        # Fixing dimensions to add the additional spline
        x = torch.cat([x, x[..., loop_index, :].unsqueeze(-2)], dim=-2)
        xdot = torch.cat(
            [xdot, xdot[..., loop_index, :].unsqueeze(-2)], dim=-2)

    a = x[..., :-1, :]
    b = xdot[..., :-1, :]
    c = 3 * (x[..., 1:, :] - x[..., :-1, :]) * delta_reciprocal_sq - xdot[...,
                                                                          1:, :] * delta_reciprocal - 2 * xdot[..., :-1, :] * delta_reciprocal
    d = 2 * (x[..., :-1, :] - x[..., 1:, :]) * delta_reciprocal * delta_reciprocal_sq + \
        xdot[..., 1:, :] * delta_reciprocal_sq + \
        xdot[..., :-1, :] * delta_reciprocal_sq

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
