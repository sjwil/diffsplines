import torch
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

from . import cubic

def plot_2d_trajectory(coeffs, points=200):
    t, a, b, c, d = coeffs
    eval_t = torch.linspace(t[0], t[-1], points, device=t.device)
    spline = cubic.CubicSpline(coeffs)
    trajectory = spline.position(eval_t).detach().cpu().numpy()

    plt.plot(trajectory[..., :, 0], trajectory[..., :, 1])

def generate_figure(coeffs, t, plot_idx, axes, goals=None, circleRadius=None):
    # Coeffs: spline coeffs (t, a, b, c, d)
    # t: times to evaluate spline position
    # plot_idx: time indices to plot the spline position
    # axes: list or single matplotlib axis, same shape as plot_idx
    # goals: optional goal point to display for each spline
    # circleRadius: optional radius of circle around the current spline position

    spline = cubic.CubicSpline(coeffs)
    # shape (..., m, d) for length m and dim d 
    trajectory = spline.position(t).detach().cpu().numpy()
    trajectory.reshape(-1, *trajectory.shape[-2:])

    if goals is not None:
        goals_np = goals.detach().cpu().numpy()
    n_splines = trajectory.shape[0]
    cmap = matplotlib.colormaps["tab10"]
    colors = [cmap(index) for index in torch.linspace(0, 1, n_splines)]
    
    # TODO: Expand projection to choose axes to plot
    projection = "2d" if (trajectory.shape[-1] == 2) else "3d"

    if type(axes) != list:
        axes = [axes]
        
    for i, axis in enumerate(axes):
        # draw appropriate index and line to that index
        if projection == "2d":
            for j in range(n_splines):
                axis.plot(trajectory[j, :plot_idx[i], 0],
                          trajectory[j, :plot_idx[i], 1], c=colors[j])
            axis.scatter(trajectory[:, plot_idx[i], 0],
                         trajectory[:, plot_idx[i], 1], c=colors)
        else:
            for j in range(n_splines):
                axis.plot(trajectory[j, :plot_idx[i], 0], trajectory[j,
                          :plot_idx[i], 1], trajectory[j, :plot_idx[i], 2], c=colors[j])
            axis.scatter(trajectory[:, plot_idx[i], 0], trajectory[:,
                         plot_idx[i], 1], trajectory[:, plot_idx[i], 2], c=colors)

        if circleRadius is not None:
            for j in range(n_splines):
                # No circles for 3d plots
                if projection == "2d":
                    axis.add_patch(Circle((trajectory[j, plot_idx[i], 0], trajectory[j,
                                   plot_idx[i], 1]), circleRadius, edgecolor=colors[j], animated=False, fill=False))

        # TODO: May need to reshape goal plotting
        if goals is not None:
            if projection == "2d":
                axis.scatter(goals_np[:, 0],
                             goals_np[:, 1], marker="x", c=colors)
            else:
                axis.scatter(goals_np[:, 0], goals_np[:, 1],
                             goals_np[:, 2], marker="x", c=colors)
