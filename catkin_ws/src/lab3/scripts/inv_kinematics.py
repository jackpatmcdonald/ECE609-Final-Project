#!/usr/bin/env python3

import numpy as np

# DH Parameters [a, alpha, d, theta_offset] (From HW5 Solutions)
DH = np.array([
    [0.000, np.pi/2, 0.077, 0.0],
    [0.130, 0.0, 0.0, np.pi/2],
    [0.135, 0.000, 0.0, -np.pi/2],
    [0.126, 0.000, 0.0, 0.0],])

# DH transform matrix
def dh_transform(a, d, alpha, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,      sa,     ca,    d],
        [0,       0,      0,    1],
    ])

#compute end effector pose from joint angles
def forward_kinematics(q):
    T_ee = np.eye(4)
    # go through each matrix, multiply to get end effector matrix
    for i, (a, alpha, d, offset) in enumerate(DH):
        T_ee = T_ee @ dh_transform(a, alpha, d, q[i] + offset)
    return T_ee

# Jacobian matrix [linear vel; omega]
def jacobian(q, delta=1e-5):
    T0 = forward_kinematics(q)
    p0 = T0[:3, 3]
    J = np.zeros((6, len(q)))
    for i in range(len(q)):
        dq = np.zeros(len(q))
        dq[i] = delta
        T1 = forward_kinematics(q + dq)
        p1 = T1[:3, 3]
        # Linear part
        J[:3, i] = (p1 - p0) / delta
        # Angular part
        dR = (T1[:3, :3] - T0[:3, :3]) / delta @ T0[:3, :3].T
        J[3, i] = dR[2, 1]
        J[4, i] = dR[0, 2]
        J[5, i] = dR[1, 0]
    return J

# pose error - where it is vs where we want it
def pose_error(T_current, T_desired):
    # position error
    dp = T_desired[:3, 3] - T_current[:3, 3]
    # orientation error
    dR = T_desired[:3, :3] @ T_current[:3, :3].T
    # skew-symetric part
    orientation_err = np.array([dR[2,1] - dR[1,2], dR[0,2] - dR[2,0], dR[1,0] - dR[0,1]]) / 2.0
    return np.concatenate([dp, orientation_err])

# IK via newton raphson
def inverse_kinematics(T_desired, q_init=None, alpha=0.5, max_iter=1000, tol=1e-4, damping=1e-3):
    q = q_init if q_init is not None else np.zeros(4)

    for i in range(max_iter):
        # compute where arm is
        T_curr = forward_kinematics(q)
        # compute error
        err = pose_error(T_curr, T_desired)
        # check if error is small enough
        if np.linalg.norm(err) < tol:
            return q, True, i
        # Jacobian
        J = jacobian(q)
        # Damped pseudoinverse: J^T (J J^T + λI)^{-1}
        JJT = J @ J.T
        J_pinv = J.T @ np.linalg.inv(JJT + damping * np.eye(6))
        dq = alpha * J_pinv @ err
        # joint update
        q = q + dq
        # Clamp to joint limits
        q = np.clip(q, -np.pi, np.pi)
    return q, False, max_iter


# Make poses based on specied input
def make_pose(x, y, z, rx=0, ry=0, rz=0):
    cx, cy, cz = np.cos(rx), np.cos(ry), np.cos(rz)
    sx, sy, sz = np.sin(rx), np.sin(ry), np.sin(rz)
    R = np.array([
        [cy*cz, cz*sx*sy - cx*sz, cx*cz*sy + sx*sz],
        [cy*sz, cx*cz + sx*sy*sz, cx*sy*sz - cz*sx],
        [  -sy,            cy*sx,            cx*cy],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = [x, y, z]
    return T
# define target poses
poses = {make_pose(0.25, 0.00, 0.15, ry=np.pi/6),
         make_pose(0.15, 0.15, 0.20, ry=np.pi/4),
         make_pose(0.20, -0.10, 0.08, ry=np.pi/3),}

# run poses
for name, T_des in poses.items():
    q_sol, success, iters = inverse_kinematics(T_des, q_init=np.zeros(4))

    # Verification (forward kinematics)
    T_check = forward_kinematics(q_sol)
    pos_error = np.linalg.norm(T_check[:3, 3] - T_des[:3, 3])

    print(f"Target: {name}")
    print(f"  Desired position : {T_des[:3,3]}")
    print(f"  Solved joints (rad): {np.round(q_sol, 4)}")
    print(f"  Solved joints (deg): {np.round(np.degrees(q_sol), 2)}")
    print(f"  FK-verified position: {np.round(T_check[:3,3], 5)}")
    print(f"  Position error : {pos_error:.6f} m")
    print(f"  Converged: {success} in {iters} iterations")
    print()