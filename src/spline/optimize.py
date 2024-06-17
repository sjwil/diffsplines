import torch
import numpy as np
from spline import bezier, cubic


def null_fn_(x):
    return 0


def dual(spline, cost_fn, c, equality_fn=null_fn_, eq_multipliers=0, ineq_fn=null_fn_, ineq_multipliers=0):
    # This dual function is actually smooth even though there is a
    # torch.maximum

    return cost_fn(spline) + eq_multipliers @ equality_fn(spline) + 0.5 * c * \
        (torch.sum(torch.clamp(ineq_multipliers + c * ineq_fn(spline), 0, None)
         ** 2 - ineq_multipliers ** 2) + torch.norm(equality_fn(spline)) ** 2)


def merit(x, eq_mult, ineq_mult, dx, deq_mult, dineq_mult, gamma, c):
    # zero = torch.tensor(0)

    return 0.5 * (torch.norm(dx) ** 2 + torch.norm(deq_mult) ** 2 +
                  torch.norm(dineq_mult) ** 2)
    # TODO: amend to include cost lagrangian value
    # + \
    #     γ * (f(x) + λ @ h(x) + torch.sum(torch.maximum(zero, μ + c * g(x))))


def optimize_spline(t, x, cost_fn, spline_type="cubic", equality_fn=null_fn_, inequality_fn=null_fn_, **kwargs):
    # t: T_max or tensor (..., length)
    # x: if cubic, tensor (..., length, channels). If bezier, tensor(..., length + 1, order + 1, channels).
    # cost_fn: function that takes the spline as input and returns a scalar cost.
    # spline_type: "cubic" or "bezier"
    # equality_fn: Equality constraints satisfied when g(x) = 0 that takes the spline and returns a 1d tensor
    # of violations.
    # inequality_fn: Inequality constraints satisfied when h(x) <= 0 that takes the spline and returns a
    # 1d tensor of violations.
    #
    # Uses an augmented lagrangian approach to minimize cost_fn

    # constraint violation penalty
    c = kwargs.get("c", 1.)
    # Additional weight on merit function on lagrangian value
    gamma = kwargs.get("gamma", 0.5)
    # Maximum constraint violation penalty, used as stopping criteria
    max_c = kwargs.get("max_c", 1000)
    # Maximum step size
    max_alpha = kwargs.get("max_alpha", 1.)
    # Minimum step size, increase constraint penalty if unable to improve merit with a step larger than alpha
    min_alpha = kwargs.get("min_alpha", 1e-3)
    # Maximum number of iterations
    max_iters = kwargs.get("max_iters", int(1e4))
    # Index the spline loops back on, default no loop.
    loop_index = kwargs.get("loop_index", None)

    data = {}
    x_traj = []

    # Set spline type
    if spline_type == "cubic":
        def fit_spline(t, x):
            coeffs = cubic.solve_cubic_coeffs(t, x, **kwargs)
            return cubic.CubicSpline(coeffs, loop_index)
    elif spline_type == "bezier":
        def fit_spline(t, x):
            return bezier.BezierSpline(t, x, loop_index)

    # Initialize multipliers
    if equality_fn is null_fn_:
        eq_multipliers = torch.zeros(1, requires_grad=True)
    else:
        # get shape of multipliers
        violation = equality_fn(fit_spline(t, x))
        if violation.dim() != 1:
            raise ValueError(
                "equality_fn should return a 1d tensor, instead tensor has shape ", violation.shape)
        eq_multipliers = torch.zeros(violation.shape, requires_grad=True)

    if inequality_fn is null_fn_:
        ineq_multipliers = torch.zeros(1, requires_grad=True)
    else:
        violation = inequality_fn(fit_spline(t, x))
        if violation.dim() != 1:
            raise ValueError(
                "inequality_fn should return a 1d tensor, instead tensor hasa shape ", violation.shape)
        ineq_multipliers = torch.zeros(violation.shape, requires_grad=True)

    x.requires_grad_(True)

    for i in range(max_iters):
        x_traj += [x.detach().cpu().numpy().copy()]

        # Evaluate dual and gradients
        spline = fit_spline(t, x)
        dual_old = dual(spline, cost_fn, c, equality_fn,
                        eq_multipliers, inequality_fn, ineq_multipliers)
        print(dual_old)
        dual_old.backward()
        dx = x.grad
        if equality_fn is null_fn_:
            deq_mult = torch.zeros(1)
        else:
            deq_mult = eq_multipliers.grad
        if inequality_fn is null_fn_:
            dineq_mult = torch.zeros(1)
        else:
            dineq_mult = ineq_multipliers.grad

        # Backtracking line search to find the next x & multipliers
        m = merit(spline, eq_multipliers, ineq_multipliers,
                  dx, deq_mult, dineq_mult, gamma, c)
        alpha = max_alpha
        x_test = x - alpha * dx
        eq_mult_test = eq_multipliers + alpha * deq_mult
        ineq_mult_test = ineq_multipliers + alpha * dineq_mult
        spline_test = fit_spline(t, x_test)

        dx_test, deq_test, dineq_test = torch.autograd.grad(
            dual(spline_test, cost_fn, c, equality_fn,
                 eq_mult_test, inequality_fn, ineq_mult_test),
            [x_test, eq_mult_test, ineq_mult_test]
        )
        grad_norm = torch.norm(
            dx) ** 2 + torch.norm(deq_mult) ** 2 + torch.norm(dineq_mult) ** 2
        # grad_norm = torch.norm(torch.cat([dx, deq_mult, dineq_mult])) ** 2
        while m - merit(spline_test, eq_mult_test, ineq_mult_test, dx_test, deq_test, dineq_test, gamma, c) < alpha * grad_norm * 0.125 \
                and alpha > min_alpha:
            alpha *= 0.5

            x_test = x - alpha * dx
            eq_mult_test = eq_multipliers + alpha * deq_mult
            ineq_mult_test = ineq_multipliers + alpha * dineq_mult
            spline_test = fit_spline(t, x_test)

            dx_test, deq_test, dineq_test = torch.autograd.grad(
                dual(spline_test, cost_fn, c, equality_fn,
                     eq_mult_test, inequality_fn, ineq_mult_test),
                [x_test, eq_mult_test, ineq_mult_test]
            )
        print(merit(spline_test, eq_mult_test, ineq_mult_test, dx_test, deq_test, dineq_test, gamma, c))
        if alpha <= min_alpha:
            # BLS failed, step constraint violation penalty
            c *= 1.1
            if c > max_c:
                break
        else:
            # Step x and multipliers
            with torch.no_grad():
                x -= alpha * dx
                eq_multipliers += alpha * deq_mult
                ineq_multipliers += alpha * dineq_mult

                # clamp
                ineq_multipliers[ineq_multipliers < 0] = 0

    data["x_traj"] = np.stack(x_traj)
    return x, eq_multipliers, ineq_multipliers, data
