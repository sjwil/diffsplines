import torch
from spline.utils import torch_binomial


class BezierSpline:
    def __init__(self, t, control_points, loop_index=None):
        # t: T_max or tensor (..., length)
        # control_points: tensor (..., length + 1, order + 1, channels).
        # Currently assumes order >= 3. This class constructs C^0 bezier splines
        # so control_points does not need to provide the last point per curve.
        # length: Number of segments
        # order: Number of control points per segment, also determines the number of derivatives we can take
        # channels: Number of splines
        if t.dim() == 0:
            self.t = torch.linspace(
                0, t, control_points.shape[-3] + 1, device=control_points.device)
        else:
            self.t = t
        self.t_diffs = self.t[1:] - self.t[:-1]
        self.t_diffs_recip = 1 / self.t_diffs
        self.t_diffs_recip_sq = self.t_diffs_recip ** 2
        self.t_diffs_recip_cube = self.t_diffs_recip ** 3
        self.t_diffs_recip_quad = self.t_diffs_recip ** 4

        self.control_points = control_points
        self.device = control_points.device

        # Control points of the derivatives of the spline are related to the control points of the spline
        self.d_control_points = (self.control_points.shape[-2] - 1) * (self.t_diffs_recip.unsqueeze(-1).unsqueeze(-1)) * (
            self.control_points[..., 1:, :] - self.control_points[..., :-1, :])
        self.d2_control_points = (self.control_points.shape[-2] - 2) * (self.t_diffs_recip_sq.unsqueeze(-1).unsqueeze(-1)) * (
            self.d_control_points[..., 1:, :] - self.d_control_points[..., :-1, :])
        self.d3_control_points = (self.control_points.shape[-2] - 3) * (self.t_diffs_recip_cube.unsqueeze(-1).unsqueeze(-1)) * (
            self.d2_control_points[..., 1:, :] - self.d2_control_points[..., :-1, :])
        self.d4_control_points = (self.control_points.shape[-2] - 4) * (self.t_diffs_recip_quad.unsqueeze(-1).unsqueeze(-1)) * (
            self.d3_control_points[..., 1:, :] - self.d3_control_points[..., :-1, :])
        self.order_control_points = [self.control_points, self.d_control_points,
                                     self.d2_control_points, self.d3_control_points, self.d4_control_points]

        self.k = torch.tensor(
            self.control_points.shape[-2] - 1, device=control_points.device)
        self.i_ = torch.tensor(
            range(self.control_points.shape[-2]), device=control_points.device)

        self.loop_index = loop_index
        if self.loop_index is not None:
            self.loop_t = (self.t - self.t[loop_index])[loop_index:]

        # Precompute binomial coefficients
        self.pos_binom = torch_binomial(self.k, self.i_)
        self.vel_binom = torch_binomial(self.k - 1, self.i_[:-1])
        self.acc_binom = torch_binomial(self.k - 2, self.i_[:-2])
        self.jerk_binom = torch_binomial(self.k - 3, self.i_[:-3])
        self.snap_binom = torch_binomial(self.k - 4, self.i_[:-4])
        self.order_binom_coeffs = [self.pos_binom, self.vel_binom,
                                   self.acc_binom, self.jerk_binom, self.snap_binom]

        self.times_cache = None
        self.fractional_cache = None
        self.index_cache = None

    def _times_to_indices(self, times):
        # Check cache
        if self.times_cache is not None and torch.equal(self.times_cache, times):
            return self.fractional_cache, self.index_cache

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

        # Set cache
        self.times_cache = times
        self.fractional_cache = fractional_part
        self.index_cache = index
        return fractional_part, index

    def _times_to_scaled_indices(self, times):
        fractional_part, index = self._times_to_indices(times)
        fractional_part = (fractional_part / self.t_diffs[index]).unsqueeze(-1)
        return fractional_part, index

    def _compute_b_polynomial(self, fractional_part, index, order):
        # Evaluate bernstein polynomial of the correct order
        terms = self.order_binom_coeffs[order] * \
            torch.pow(1 - fractional_part, self.k - self.i_[order:]) * \
            torch.pow(fractional_part, self.i_[
                      :-order] if order > 0 else self.i_)
        terms = terms.unsqueeze(-1)
        result = torch.sum(self.order_control_points[order][..., index, :, :] * terms, dim=-2)
        return result

    def compute_spline(self, times, order):
        # Evaluate the order-th derivative of the spline at the given times
        fractional_part, index = self._times_to_scaled_indices(times)
        return self._compute_b_polynomial(fractional_part, index, order)

    def position(self, times):
        return self.compute_spline(times, 0)

    def velocity(self, times):
        return self.compute_spline(times, 1)
    
    def acceleration(self, times):
        return self.compute_spline(times, 2)

    def jerk(self, times):
        return self.compute_spline(times, 3)

    def snap(self, times):
        return self.compute_spline(times, 4)

def adapt_c0_bezier(control_points, loop_index=None):
    # control_points: tensor (..., length + 1, order, channels)
    # Appends an additional control point to each curve to ensure c0 continuity
    # through knots.
    if loop_index is None:
        return torch.cat([control_points[..., :-1, :, :],
                          control_points[..., 1:, 0, :].unsqueeze(-2)], dim=-2)
    else:
        return torch.cat([control_points,
                          torch.cat([control_points[..., 1:, 0, :], control_points[...,
                                                                                   loop_index, 0, :].unsqueeze(-2)], dim=-2).unsqueeze(-2),
                          ], dim=-2)


def adapt_c1_bezier(control_points, delta_t=None, loop_index=None):
    # control_points: tensor (..., length + 1, order - 1, channels)
    # Appends two additional control points to each curve to ensure c1 continuity
    # through knots.
    if delta_t is None:
        delta_t = torch.ones(
            control_points.shape[-3], device=control_points.device).unsqueeze(-1)

    else:
        delta_t = torch.cat([delta_t, delta_t[-1].unsqueeze(0)], dim=0)

    if loop_index is None:
        diffs = control_points[..., 1:, -2, :] + delta_t[:-1] / delta_t[1:] * (control_points[..., 1:, -2,
                                                                                              :] - control_points[..., 1:, -1, :])
        # create (..., length, order, channels) shape for full representation.
        return torch.cat([control_points[..., :-1, :, :], diffs.unsqueeze(-2),
                          control_points[..., 1:, 0, :].unsqueeze(-2)], dim=-2)

    else:
        diffs = torch.cat([control_points[..., 1:, -2, :] + delta_t[:-1] / delta_t[1:] * (control_points[..., 1:, -2,
                                                                                                         :] - control_points[..., 1:, -1, :]),
                           (control_points[..., loop_index, -2, :] + delta_t[-1] / delta_t[loop_index] *
                            (control_points[..., loop_index, -2, :] -
                               control_points[..., loop_index, -1, :])).unsqueeze(-2)], dim=-2)

        # diffs = torch.cat([(2 * control_points[..., 1:, -2, :] - control_points[..., 1:, -1, :]),
        #                    (2 * control_points[..., loop_index, -2, :] - control_points[..., loop_index, -1, :]).unsqueeze(-2)], dim=-2)
        # create (..., length + 1, order, channels) shape for full representation.
        next_points = torch.cat(
            [control_points[..., 1:, 0, :], control_points[..., loop_index, 0, :].unsqueeze(-2)], dim=-2)

        return torch.cat([control_points, diffs.unsqueeze(-2), next_points.unsqueeze(-2)], dim=-2)


def c0_violation(spline):
    return torch.norm(spline.control_points[..., 1:, 0, :] - spline.control_points[..., :-1, -1, :])


def c1_violation(spline, verbose=False):

    violation = torch.norm(
        spline.d_control_points[..., 1:, 0, :] - spline.d_control_points[..., :-1, -1, :])
    # violation = torch.norm((spline.control_points[..., 1:, 1, :] - spline.control_points[..., 1:, 0, :]) -
    #                        (spline.control_points[..., :-1, -1, :] - spline.control_points[..., :-1, -2, :]))
    if spline.loop_index is not None:
        # Should we work this calculation into the previous norm?
        loop_violation = torch.norm((spline.control_points[..., spline.loop_index, 1, :] - spline.control_points[..., spline.loop_index, 0, :]) -
                                    (spline.control_points[..., -1, -1, :] -
                                     spline.control_points[..., -1, -2, :]))

        if verbose:
            print("Violation: ", violation)
            print((spline.control_points[..., 1:, 1, :] - spline.control_points[..., 1:, 0, :]) -
                  (spline.control_points[..., :-1, -1, :] - spline.control_points[..., :-1, -2, :]))

            print("Loop violation: ", loop_violation)

        return violation + torch.norm((spline.control_points[..., spline.loop_index, 1, :] - spline.control_points[..., spline.loop_index, 0, :]) -
                                      (spline.control_points[..., -1, -1, :] -
                                       spline.control_points[..., -1, -2, :]))

    return violation


def c0_c1_violation(spline):
    return torch.stack([c0_violation(spline), c1_violation(spline)])
