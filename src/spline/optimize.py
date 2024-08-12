import torch
from torch.func import hessian
import numpy as np
from spline import bezier, cubic


def null_fn_(x):
    return torch.zeros(1, device=x.device)


def dual(spline, cost_fn, c, equality_fn=null_fn_, eq_multipliers=0, ineq_fn=null_fn_, ineq_multipliers=0):
    # This dual function is actually smooth even though there is a
    # torch.maximum
    return cost_fn(spline) + eq_multipliers @ equality_fn(spline) + 0.5 * c * \
        (torch.sum(torch.clamp(ineq_multipliers + c * ineq_fn(spline), 0, None)
         ** 2 - ineq_multipliers ** 2) + torch.sum(equality_fn(spline) ** 2))


def merit(spline, cost_fn, c, equality_fn, eq_mult, ineq_fn, ineq_mult, dx, deq_mult, dineq_mult, gamma):
    return 0.5 * (torch.norm(dx) ** 2 + torch.norm(deq_mult) ** 2 +
                  torch.norm(dineq_mult) ** 2) + gamma * (cost_fn(spline) + eq_mult @ equality_fn(spline) +
                                                          torch.sum(torch.clamp(ineq_mult + c * ineq_fn(spline), 0, None)))


def grad_l2_norm(grad):
    return torch.sum(grad ** 2)


def optimize_fo_spline(t, x, cost_fn, spline_type="cubic", equality_fn=null_fn_, inequality_fn=null_fn_, **kwargs):
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
    gamma = kwargs.get("gamma", 2.)
    # Maximum constraint violation penalty, used as stopping criteria
    max_c = kwargs.get("max_c", 1e2)
    # Maximum step size
    max_alpha = kwargs.get("max_alpha", 1.)
    # Minimum step size, increase constraint penalty if unable to improve merit with a step larger than alpha
    min_alpha = kwargs.get("min_alpha", 1e-4)
    # Maximum number of iterations
    max_iters = kwargs.get("max_iters", int(1e4))
    # Index the spline loops back on, default no loop.
    loop_index = kwargs.get("loop_index", None)
    # Whether the bezier spline will be adapted to enforce c0 continuity
    enforce_c0 = kwargs.get("enforce_c0", True)
    # Maximum gradient abs
    max_grad = kwargs.get("max_grad", 10.)

    data = {}
    x_traj = []
    cost_traj = []
    eq_viol_traj = []
    ineq_viol_traj = []
    dual_traj = []

    # Set spline type
    if spline_type == "cubic":
        def fit_spline(t, x):
            coeffs = cubic.solve_cubic_coeffs(t, x, **kwargs)
            return cubic.CubicSpline(coeffs, loop_index)
    elif spline_type == "bezier":
        if enforce_c0:
            def fit_spline(t, x):
                control_points = bezier.adapt_c0_bezier(x, loop_index)
                return bezier.BezierSpline(t, control_points, loop_index)
        else:
            def fit_spline(t, x):
                return bezier.BezierSpline(t, x, loop_index)

    # Initialize multipliers
    if equality_fn is null_fn_:
        eq_multipliers = torch.zeros(1, device=x.device, requires_grad=True)
    else:
        # get shape of multipliers
        violation = equality_fn(fit_spline(t, x))
        if violation.dim() != 1:
            raise ValueError(
                "equality_fn should return a 1d tensor, instead tensor has shape ", violation.shape)
        eq_multipliers = torch.zeros(
            violation.shape, device=x.device, requires_grad=True)

    if inequality_fn is null_fn_:
        ineq_multipliers = torch.zeros(1, device=x.device, requires_grad=True)
    else:
        violation = inequality_fn(fit_spline(t, x))
        if violation.dim() != 1:
            raise ValueError(
                "inequality_fn should return a 1d tensor, instead tensor hasa shape ", violation.shape)
        ineq_multipliers = torch.zeros(
            violation.shape, device=x.device, requires_grad=True)

    x.requires_grad_(True)

    for i in range(max_iters):

        # Evaluate dual and gradients
        spline = fit_spline(t, x)
        dx, deq_mult, dineq_mult = torch.autograd.grad(dual(spline, cost_fn, c, equality_fn,
                                                            eq_multipliers, inequality_fn, ineq_multipliers), [x, eq_multipliers, ineq_multipliers])
        torch.clip_(dx, -max_grad, max_grad)
        torch.clip_(deq_mult, -max_grad, max_grad)
        torch.clip_(dineq_mult, -max_grad, max_grad)

        # Backtracking line search to find the next x & multipliers
        m = merit(spline, cost_fn, c, equality_fn, eq_multipliers, inequality_fn, ineq_multipliers,
                  dx, deq_mult, dineq_mult, gamma)
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
        torch.clip_(dx_test, -max_grad, max_grad)
        torch.clip_(deq_test, -max_grad, max_grad)
        torch.clip_(dineq_test, -max_grad, max_grad)

        grad_norm = torch.norm(
            dx) ** 2 + torch.norm(deq_mult) ** 2 + torch.norm(dineq_mult) ** 2
        while m - merit(spline_test, cost_fn, c, equality_fn, eq_mult_test, inequality_fn, ineq_mult_test, dx_test, deq_test, dineq_test, gamma) < alpha * grad_norm * 0.125 \
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

            torch.clip_(dx_test, -max_grad, max_grad)
            torch.clip_(deq_test, -max_grad, max_grad)
            torch.clip_(dineq_test, -max_grad, max_grad)

        if alpha <= min_alpha:
            # BLS failed, step constraint violation penalty
            c *= 1.5
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

        x_traj += [x.detach().cpu().numpy().copy()]
        cost_traj += [cost_fn(spline).item()]
        eq_viol_traj += [eq_multipliers.detach().cpu().numpy().copy()]
        ineq_viol_traj += [ineq_mult_test.detach().cpu().numpy().copy()]
        dual_traj += [dual(spline, cost_fn, c, equality_fn,
                           eq_multipliers, inequality_fn, ineq_multipliers).item()]

    data["x_traj"] = np.stack(x_traj)
    data["cost_traj"] = np.stack(cost_traj)
    data["eq_viol_traj"] = np.stack(eq_viol_traj)
    data["ineq_viol_traj"] = np.stack(ineq_viol_traj)
    data["dual_traj"] = np.stack(dual_traj)
    return x, eq_multipliers, ineq_multipliers, data


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
    # constraint scaling
    c_scaling = kwargs.get("c_scaling", 3.)
    # Maximum constraint violation penalty, used as stopping criteria
    max_c = kwargs.get("max_c", 1e2)
    # Maximum step size
    max_alpha = kwargs.get("max_alpha", 1.)
    # Minimum step size, increase constraint penalty if unable to improve merit with a step larger than alpha
    min_alpha = kwargs.get("min_alpha", 1e-2)
    # Maximum number of iterations
    max_iters = kwargs.get("max_iters", int(1e4))
    # Index the spline loops back on, default no loop.
    loop_index = kwargs.get("loop_index", None)
    # Whether the bezier spline will be adapted to enforce c0 continuity
    enforce_c0 = kwargs.get("enforce_c0", True)
    # Maximum gradient abs
    max_grad = kwargs.get("max_grad", 100.)
    # regularizaton added to diagonal of the hessian
    regularization = kwargs.get("regularization", 0.01)

    data = {}
    x_traj = []
    cost_traj = []
    eq_viol_traj = []
    ineq_viol_traj = []
    dual_traj = []
    alpha_traj = []

    # Set spline type
    if spline_type == "cubic":
        def fit_spline(t, x):
            coeffs = cubic.solve_cubic_coeffs(t, x, **kwargs)
            return cubic.CubicSpline(coeffs, loop_index)
    elif spline_type == "bezier":
        if enforce_c0:
            def fit_spline(t, x):
                control_points = bezier.adapt_c0_bezier(x, loop_index)
                return bezier.BezierSpline(t, control_points, loop_index)
        else:
            def fit_spline(t, x):
                return bezier.BezierSpline(t, x, loop_index)

    # Initialize multipliers
    if equality_fn is null_fn_:
        eq_multipliers = torch.zeros(1, device=x.device)
    else:
        # get shape of multipliers
        violation = equality_fn(fit_spline(t, x))
        if violation.dim() != 1:
            raise ValueError(
                "equality_fn should return a 1d tensor, instead tensor has shape ", violation.shape)
        eq_multipliers = torch.zeros(
            violation.shape, device=x.device)

    if inequality_fn is null_fn_:
        ineq_multipliers = torch.zeros(1, device=x.device)
    else:
        violation = inequality_fn(fit_spline(t, x))
        if violation.dim() != 1:
            raise ValueError(
                "inequality_fn should return a 1d tensor, instead tensor hasa shape ", violation.shape)
        ineq_multipliers = torch.zeros(
            violation.shape, device=x.device)

    def eval_dual(t, x, cost_fn, c, equality_fn, eq_mult, inequality_fn, ineq_mult):
        spline = fit_spline(t, x)
        return dual(spline, cost_fn, c, equality_fn, eq_mult, inequality_fn, ineq_mult)

    x.requires_grad_(True)
    print(x.shape)

    # Hessian w.r.t. x
    h = torch.func.hessian(eval_dual, argnums=1)
    I = torch.eye(x.flatten().shape[0], device=x.device) * regularization
    # TEMP
    alpha = max_alpha

    for i in range(max_iters):
        # Evaluate dual and gradients
        spline = fit_spline(t, x)
        dx, = torch.autograd.grad(dual(spline, cost_fn, c, equality_fn,
                                       eq_multipliers, inequality_fn, ineq_multipliers), x)
        dx_shape = dx.flatten().shape
        torch.clip_(dx, -max_grad, max_grad)

        hessian = h(t, x, cost_fn, c, equality_fn, eq_multipliers,
                    inequality_fn, ineq_multipliers)
        d = torch.linalg.solve(hessian.view(dx_shape[0], dx_shape[0]) + I, dx.flatten()).view(dx.shape)
        torch.clip_(d, -max_grad, max_grad)
        
        # Backtracking line search to find the next x & multipliers
        m = grad_l2_norm(dx)
        # TEMP
        alpha = max_alpha
        x_test = x - alpha * d
        spline_test = fit_spline(t, x_test)

        dx_test, = torch.autograd.grad(
            dual(spline_test, cost_fn, c, equality_fn,
                 eq_multipliers, inequality_fn, ineq_multipliers),
            x_test
        )
        torch.clip_(dx_test, -max_grad, max_grad)

        while grad_l2_norm(dx_test) >= (1 - (alpha * 0.25)) * m and alpha > min_alpha:
            # while m - merit(spline_test, cost_fn, c, equality_fn, eq_mult_test, inequality_fn, ineq_mult_test, dx_test, deq_test, dineq_test, gamma) < alpha * grad_norm * 0.125 \
            # and alpha > min_alpha:
            alpha *= 0.5

            x_test = x - alpha * d
            spline_test = fit_spline(t, x_test)

            dx_test, = torch.autograd.grad(
                dual(spline_test, cost_fn, c, equality_fn,
                     eq_multipliers, inequality_fn, ineq_multipliers),
                x_test
            )
            torch.clip_(dx_test, -max_grad, max_grad)

        with torch.no_grad():
            # Step multipliers, dual ascent
            if alpha <= min_alpha:
                eq_multipliers += c * equality_fn(spline)
                ineq_multipliers += c * inequality_fn(spline)
                c *= c_scaling
                # Clamp
                ineq_multipliers[ineq_multipliers < 0] = 0
                # TEMP
                # alpha = max_alpha
            # Step x
            else:
                x -= alpha * d
                # alpha *= 2
        x_traj += [x.detach().cpu().numpy().copy()]
        cost_traj += [cost_fn(spline).item()]
        eq_viol_traj += [equality_fn(spline).detach().cpu().numpy().copy()]
        ineq_viol_traj += [inequality_fn(spline).detach().cpu().numpy().copy()]
        dual_traj += [dual(spline, cost_fn, c, equality_fn,
                           eq_multipliers, inequality_fn, ineq_multipliers).item()]
        alpha_traj += [alpha]
        if c >= max_c:
            break

    data["x_traj"] = np.stack(x_traj)
    data["cost_traj"] = np.stack(cost_traj)
    data["eq_viol_traj"] = np.stack(eq_viol_traj)
    data["ineq_viol_traj"] = np.stack(ineq_viol_traj)
    data["dual_traj"] = np.stack(dual_traj)
    data["alpha_traj"] = np.stack(alpha_traj)
    return x, eq_multipliers, ineq_multipliers, data
