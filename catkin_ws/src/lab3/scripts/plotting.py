#!/usr/bin/env python3
"""
ME571 Lab 3 - Tasks 6 & 7: Plot and compare planning results
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PLANNERS = ["RRTConnectkConfigDefault",
            "PRMkConfigDefault",
            "KPIECEkConfigDefault"]
PLANNER_LABELS = ["RRTConnect", "PRM", "KPIECE"]
COLORS = ["#2196F3", "#FF5722", "#4CAF50"]

df = pd.read_csv("/tmp/planning_results.csv")

# ── Plot 1: Planning time by planner and space ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, space, title in zip(axes,
                             ["task_space", "joint_space"],
                             ["Task-Space Planning", "Joint-Space Planning"]):
    sub = df[df["space"] == space]
    x = np.arange(len(PLANNERS))
    bars = ax.bar(x, [sub[sub["planner"]==p]["planning_time"].mean()
                      for p in PLANNERS],
                  color=COLORS, width=0.5)
    ax.bar_label(bars, fmt="%.3fs", padding=3)
    ax.set_xticks(x)
    ax.set_xticklabels(PLANNER_LABELS)
    ax.set_ylabel("Mean Planning Time (s)")
    ax.set_title(title)
    ax.set_ylim(0, df["planning_time"].max() * 1.3)

plt.suptitle("Planning Time Comparison", fontweight="bold")
plt.tight_layout()
plt.savefig("/tmp/planning_time.png", dpi=150)
plt.show()

# ── Plot 2: Planning time per pose per planner ─────────────────────────────
poses = df["pose"].unique()
fig, axes = plt.subplots(1, len(poses), figsize=(14, 5), sharey=True)
for ax, pose in zip(axes, poses):
    sub = df[df["pose"] == pose]
    for i, (planner, label, color) in enumerate(
            zip(PLANNERS, PLANNER_LABELS, COLORS)):
        row = sub[sub["planner"] == planner]
        task = row[row["space"] == "task_space"]["planning_time"].values
        jnt  = row[row["space"] == "joint_space"]["planning_time"].values
        ax.bar(i - 0.2, task[0] if len(task) else 0,
               width=0.35, color=color, alpha=0.9, label=f"{label} (task)")
        ax.bar(i + 0.2, jnt[0]  if len(jnt)  else 0,
               width=0.35, color=color, alpha=0.45, label=f"{label} (joint)")
    ax.set_title(pose)
    ax.set_xticks(range(len(PLANNERS)))
    ax.set_xticklabels(PLANNER_LABELS, rotation=20)
    ax.set_ylabel("Planning Time (s)")

plt.suptitle("Planning Time per Pose", fontweight="bold")
plt.tight_layout()
plt.savefig("/tmp/planning_time_per_pose.png", dpi=150)
plt.show()

# ── Plot 3: Task-space vs joint-space path length proxy (from final pos) ───
fig, ax = plt.subplots(figsize=(8, 5))
for space, label, marker in [("task_space", "Task-Space", "o"),
                               ("joint_space", "Joint-Space", "s")]:
    sub = df[df["space"] == space]
    means = [sub[sub["planner"]==p]["planning_time"].mean() for p in PLANNERS]
    ax.plot(PLANNER_LABELS, means, marker=marker, linewidth=2, label=label)

ax.set_ylabel("Mean Planning Time (s)")
ax.set_title("Task-Space vs Joint-Space: Planning Time by Planner")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/tmp/space_comparison.png", dpi=150)
plt.show()

print("All plots saved to /tmp/")