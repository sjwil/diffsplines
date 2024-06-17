import torch
from spline.utils import torch_binomial


class BezierSpline:
    def __init__(self, t, control_points, loop_index=None):
        # t: T_max or tensor (..., length)
        # control_points: tensor (..., length + 1, order + 1, channels).
        # Currently assumes order >= 3. This class constructs C^0 bezier splines
        # so control_points does not need to provide the last point per curve.
        if t.dim() == 0:
            self.t = torch.linspace(
                0, t, control_points.shape[-3] + 1, device=control_points.device)
        else:
            self.t = t
        self.t_diffs = self.t[1:] - self.t[:-1]
        self.control_points = control_points
        # Fix c0 continuity
        # self.control_points = torch.cat([control_points[..., :-1, :, :],
        #                                  control_points[..., 1:, 0, :].unsqueeze(-2)], dim=-2)

        self.d_control_points = (self.control_points.shape[-2] - 1) * (
            self.control_points[..., 1:, :] - self.control_points[..., :-1, :])
        self.d2_control_points = (self.control_points.shape[-2] - 2) * (
            self.d_control_points[..., 1:, :] - self.d_control_points[..., :-1, :])

        self.k = torch.tensor(
            self.control_points.shape[-2] - 1, device=control_points.device)
        self.i_ = torch.tensor(
            range(self.control_points.shape[-2]), device=control_points.device)

        self.loop_index = loop_index
        # Precompute binomial coefficients
        self.pos_binom = torch_binomial(self.k, self.i_)
        self.vel_binom = torch_binomial(self.k - 1, self.i_[:-1])
        self.acc_binom = torch_binomial(self.k - 2, self.i_[:-2])

    def _times_to_indices(self, times):
        index = torch.bucketize(times, self.t, right=True) - 1

        if self.loop_index is not None:
            index = index.clamp_min(0)
            # For each time, where in the lasso segment would it be
            loop_index = torch.bucketize(
                (times - self.t[-1]) % self.loop_t[-1], self.loop_t, right=True) - 1
            # Which indices are actually in the lasso segment
            lasso_indices = index > self.control_points.size(-3) - 1
            # Reset lasso indices to correct index
            index[lasso_indices] = loop_index[lasso_indices] + self.loop_index

        else:
            index = index.clamp(0, self.control_points.size(-3) - 1)

        fractional_part = times - self.t[index]

        if self.loop_index is not None:
            # How far are we in the current loop
            fractional_part[lasso_indices] = (
                (times[lasso_indices] - self.t[-1]) % self.loop_t[-1]) - self.loop_t[loop_index[lasso_indices]]
        return fractional_part, index

    def position(self, times):
        fractional_part, index = self._times_to_indices(times)
        fractional_part = (fractional_part / self.t_diffs[index]).unsqueeze(-1)

        # times x order + 1
        res = self.pos_binom * \
            torch.pow(1 - fractional_part, self.k - self.i_) * \
            torch.pow(fractional_part, self.i_)

        res = res.unsqueeze(-1)
        result = torch.sum(self.control_points[..., index, :, :] * res, dim=-2)
        return result

    def velocity(self, times):
        fractional_part, index = self._times_to_indices(times)
        fractional_part = (fractional_part / self.t_diffs[index]).unsqueeze(-1)

        res = self.vel_binom * \
            torch.pow(1 - fractional_part, self.k - self.i_[1:]) * \
            torch.pow(fractional_part, self.i_[:-1])
        res = res.unsqueeze(-1)
        result = torch.sum(
            self.d_control_points[..., index, :, :] * res, dim=-2)
        return result

    def acceleration(self, times):
        fractional_part, index = self._times_to_indices(times)
        fractional_part = (fractional_part / self.t_diffs[index]).unsqueeze(-1)

        res = self.acc_binom * \
            torch.pow(1 - fractional_part, self.k - self.i_[2:]) * \
            torch.pow(fractional_part, self.i_[:-2])

        res = res.unsqueeze(-1)
        result = torch.sum(
            self.d2_control_points[..., index, :, :] * res, dim=-2)
        return result


def adapt_c1_bezier(control_points):
    # control_points: tensor (..., length + 1, order - 1, channels)
    # Appends an additional control point to each curve to ensure c1 continuity
    # through knots.
    diffs = (2 * control_points[..., 1:, -2,
                                :] - control_points[..., 1:, -1, :])
    # create (..., length, order, channels) shape for full representation.
    return torch.cat([control_points[..., :-1, :, :], diffs.unsqueeze(-2),
                      control_points[..., 1:, 0, :].unsqueeze(-2)], dim=-2)
