import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from collision import get_collision_marker_points
from config import (
    PARAMS,
    CENTER_WALL_X,
    PARKING_SPOT_WIDTH,
    RIGHT_WALL_X,
    ROW1_X_END,
    ROW1_X_START,
    ROW2_X_END,
    ROW2_X_START,
    ROW3_X_END,
    ROW3_X_START,
    ROW4_X_END,
    ROW4_X_START,
)
from heuristics import get_three_closest_wall_points
from smoothing import smooth_trajectory_cg


def draw_vehicle_rectangle(ax, state, color='green', linestyle='-', linewidth=2.0, show_heading_arrow=True):
    corners = state.get_corners()
    xs = [corners[0][0], corners[1][0], corners[2][0], corners[3][0], corners[0][0]]
    ys = [corners[0][1], corners[1][1], corners[2][1], corners[3][1], corners[0][1]]
    ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=linewidth, zorder=6)

    if show_heading_arrow:
        arrow_len = PARAMS['vehicle_L'] * 0.55
        ax.arrow(
            state.x,
            state.y,
            arrow_len * math.cos(state.theta),
            arrow_len * math.sin(state.theta),
            color=color,
            width=0.03,
            head_width=0.45,
            head_length=0.65,
            length_includes_head=True,
            zorder=7,
        )


def draw_vehicle_footprints(ax, nodes, color='deepskyblue'):
    for node in nodes:
        corners = node.get_corners()
        poly = Polygon(corners, closed=True, fill=False, edgecolor=color, linewidth=0.6, alpha=0.22, zorder=3)
        ax.add_patch(poly)


def plot_results(start, goal, final_node, all_explored, obs_map, title_suffix='', save_path=None, show_plot=True):
    fig, ax = plt.subplots(figsize=(10, 12))
    fig.subplots_adjust(right=0.76)

    ox = [p[0] for p in obs_map.obstacles]
    oy = [p[1] for p in obs_map.obstacles]
    ax.scatter(ox, oy, c='black', marker='s', s=10)

    wall_style = {'color': 'black', 'linewidth': 3}
    env_style = {'color': 'gray', 'linewidth': 1.5, 'linestyle': '-'}
    open_style = {'color': 'lightgray', 'linewidth': 1, 'linestyle': '--'}

    ax.plot([0, ROW3_X_END], [45, 45], **wall_style)
    ax.plot([ROW4_X_START, RIGHT_WALL_X], [45, 45], **wall_style)
    ax.plot([0, RIGHT_WALL_X], [0, 0], **wall_style)
    ax.plot([0, 0], [0, 45], **wall_style)
    ax.plot([RIGHT_WALL_X, RIGHT_WALL_X], [0, 45], **wall_style)
    ax.plot([CENTER_WALL_X, CENTER_WALL_X], [4.5, 36.5], **wall_style)

    start_y, end_y = 4.5, 36.5
    y_lines = np.arange(start_y, end_y + PARKING_SPOT_WIDTH, PARKING_SPOT_WIDTH)
    row1_x = [ROW1_X_START, ROW1_X_END]
    row2_x = [ROW2_X_START, ROW2_X_END]
    row3_x = [ROW3_X_START, ROW3_X_END]
    row4_x = [ROW4_X_START, ROW4_X_END]
    for y in y_lines:
        for rx in [row1_x, row2_x, row3_x, row4_x]:
            ax.plot(rx, [y, y], **env_style)

    for x in [ROW1_X_END, ROW2_X_START, ROW3_X_END, ROW4_X_START]:
        ax.plot([x, x], [start_y, end_y], **open_style)

    ax.text((ROW3_X_END + ROW4_X_START) / 2, 45.5, 'Entrance', ha='center', va='bottom', fontsize=12, color='blue', fontweight='bold')

    path_nodes = []
    curr = final_node
    while curr is not None:
        path_nodes.append(curr)
        curr = curr.parent
    path_nodes = path_nodes[::-1]

    smoothed_nodes = smooth_trajectory_cg(path_nodes, obs_map)

    draw_vehicle_footprints(ax, smoothed_nodes)

    collision_markers = []
    collision_state_count = 0
    for node in smoothed_nodes:
        markers = get_collision_marker_points(node, obs_map)
        if markers:
            collision_state_count += 1
            collision_markers.extend(markers)

    unique_collision_markers = []
    for mx, my in collision_markers:
        if not any(math.hypot(mx - ux, my - uy) < 0.1 for ux, uy in unique_collision_markers):
            unique_collision_markers.append((mx, my))

    if unique_collision_markers:
        marker_x = [pt[0] for pt in unique_collision_markers]
        marker_y = [pt[1] for pt in unique_collision_markers]
        ax.scatter(marker_x, marker_y, c='red', marker='x', s=120, linewidths=2.2, zorder=11, label='Collision')

    for i in range(1, len(smoothed_nodes)):
        prev, curr = smoothed_nodes[i - 1], smoothed_nodes[i]
        linestyle = '-' if curr.gear == 1 else '--'
        color = 'red' if curr.gear == 1 else 'orange'
        ax.plot([prev.x, curr.x], [prev.y, curr.y], c=color, linewidth=2, linestyle=linestyle)

    draw_vehicle_rectangle(ax, start, color='blue', linestyle='--', linewidth=1.6)
    draw_vehicle_rectangle(ax, goal, color='magenta', linestyle='--', linewidth=1.6)
    final_pose = smoothed_nodes[-1]
    draw_vehicle_rectangle(ax, final_pose, color='green', linestyle='-', linewidth=2.2)

    closest_three = get_three_closest_wall_points(final_pose, obs_map)
    summary_lines = []
    marker_offsets = {
        1: (0.55, 0.45),
        2: (0.55, -0.55),
        3: (-0.65, 0.45),
    }
    for idx, (dist, car_pt, wall_pt, part_name) in enumerate(closest_three, start=1):
        ax.plot([car_pt[0], wall_pt[0]], [car_pt[1], wall_pt[1]], c='green', linestyle=':', linewidth=1.5, zorder=7)
        ax.scatter(car_pt[0], car_pt[1], c='limegreen', s=30, zorder=8)
        ax.scatter(wall_pt[0], wall_pt[1], c='black', marker='x', s=35, zorder=8)

        dx, dy = marker_offsets.get(idx, (0.4, 0.4))
        label_x, label_y = car_pt[0] + dx, car_pt[1] + dy
        ax.scatter(label_x, label_y, s=120, c='white', edgecolors='green', linewidths=1.2, zorder=9)
        ax.text(
            label_x,
            label_y,
            str(idx),
            va='center',
            ha='center',
            fontsize=9,
            color='green',
            fontweight='bold',
            zorder=10,
        )

        summary_lines.append(
            f'{idx}. d={dist:.3f}  P=({car_pt[0]:.2f}, {car_pt[1]:.2f})  W=({wall_pt[0]:.2f}, {wall_pt[1]:.2f})'
        )

    summary_text = 'Closest 1/2/3\n' + '\n'.join(summary_lines)
    if collision_state_count > 0:
        summary_text += f'\nCollision states: {collision_state_count}'
    else:
        summary_text += '\nCollision states: 0'
    ax.text(
        1.01,
        0.02,
        summary_text,
        transform=ax.transAxes,
        va='bottom',
        ha='left',
        fontsize=8,
        color='green',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='green', alpha=0.95),
        clip_on=False,
        zorder=10,
    )

    ax.scatter(start.x, start.y, c='blue', s=100, label='Start')
    ax.scatter(goal.x, goal.y, c='magenta', s=100, label='Goal')
    title = 'Hybrid A-star with Reeds-Shepp, Dual Heuristic & Smoothing'
    if title_suffix:
        title = f'{title}\n{title_suffix}'
    ax.set_title(title)
    ax.legend(loc='lower left')
    ax.axis('equal')
    if save_path is not None:
        fig.savefig(save_path, dpi=220, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
