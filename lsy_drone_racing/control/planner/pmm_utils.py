import numpy as np
from scipy.optimize import brentq

def compute_1d_minimum_time(p0, v0, pf, vf, u_min, u_max):
    """
    Computes the minimum time 1D trajectory using bang-bang control.
    Returns (t1, t2, u1, u2) or None if no valid trajectory.
    """
    best_T = float('inf')
    best_sol = None
    
    for u1, u2 in [(u_max, u_min), (u_min, u_max)]:
        if u1 == u2:
            continue
        C = (vf**2 - v0**2 - 2 * u2 * (pf - p0)) / (u2 - u1)
        discriminant = v0**2 - u1 * C
        if discriminant >= 0:
            sqrt_disc = np.sqrt(discriminant)
            for sign in [-1, 1]:
                t1 = (-v0 + sign * sqrt_disc) / u1
                if t1 >= -1e-6:
                    t1 = max(0.0, t1)
                    t2 = (vf - v0 - u1 * t1) / u2
                    if t2 >= -1e-6:
                        t2 = max(0.0, t2)
                        T = t1 + t2
                        if T < best_T:
                            best_T = T
                            best_sol = (t1, t2, u1, u2)
    return best_sol

def evaluate_1d_trajectory(p0, v0, t1, t2, u1, u2, t):
    """
    Evaluates the 1D trajectory at time t.
    """
    t = np.clip(t, 0, t1 + t2)
    if t <= t1:
        p = p0 + v0 * t + 0.5 * u1 * t**2
        v = v0 + u1 * t
    else:
        dt = t - t1
        p1 = p0 + v0 * t1 + 0.5 * u1 * t1**2
        v1 = v0 + u1 * t1
        p = p1 + v1 * dt + 0.5 * u2 * dt**2
        v = v1 + u2 * dt
    return p, v

def solve_alpha_for_target_time(p0, v0, pf, vf, u_min, u_max, T_target):
    """
    Finds the scaling factor alpha in [0, 1] such that the minimum time is T_target.
    Returns (t1, t2, alpha*u1, alpha*u2) or None.
    """
    def time_diff(alpha):
        sol = compute_1d_minimum_time(p0, v0, pf, vf, alpha * u_min, alpha * u_max)
        if sol is None:
            return float('inf')
        return (sol[0] + sol[1]) - T_target

    try:
        # T increases as alpha decreases
        # check alpha = 1.0
        sol_1 = compute_1d_minimum_time(p0, v0, pf, vf, u_min, u_max)
        if sol_1 is None: return None
        T_1 = sol_1[0] + sol_1[1]
        if abs(T_1 - T_target) < 1e-4:
            return sol_1
        
        # Binary search range for alpha
        alpha_min = 1e-3
        alpha_max = 1.0
        
        alpha_root = brentq(time_diff, alpha_min, alpha_max, xtol=1e-4)
        sol = compute_1d_minimum_time(p0, v0, pf, vf, alpha_root * u_min, alpha_root * u_max)
        return sol
    except Exception:
        # Fallback to simple binary search if brentq fails
        alpha_low = 1e-3
        alpha_high = 1.0
        best_sol = None
        for _ in range(20):
            alpha = (alpha_low + alpha_high) / 2
            sol = compute_1d_minimum_time(p0, v0, pf, vf, alpha * u_min, alpha * u_max)
            if sol is None:
                alpha_low = alpha
                continue
            T = sol[0] + sol[1]
            if T > T_target:
                alpha_low = alpha
            else:
                alpha_high = alpha
                best_sol = sol
        return best_sol

def generate_3d_trajectory(p0, v0, pf, vf, u_min, u_max):
    """
    Generates time-synchronized 3D trajectory components.
    Returns max_T and a list of 3 tuples (t1, t2, u1, u2) for x, y, z.
    """
    sols = []
    max_T = 0.0
    for i in range(3):
        sol = compute_1d_minimum_time(p0[i], v0[i], pf[i], vf[i], u_min, u_max)
        if sol is None:
            return None, None
        T = sol[0] + sol[1]
        max_T = max(max_T, T)
        sols.append(sol)
        
    final_sols = []
    for i in range(3):
        T = sols[i][0] + sols[i][1]
        if abs(T - max_T) > 1e-4:
            sol = solve_alpha_for_target_time(p0[i], v0[i], pf[i], vf[i], u_min, u_max, max_T)
            if sol is None:
                return None, None
            final_sols.append(sol)
        else:
            final_sols.append(sols[i])
            
    return max_T, final_sols

def dijkstra_search(p0, v0, p_waypoints, u_min, u_max, s_bins, v_min, v_max, theta_min, theta_max, psi_min, psi_max):
    """
    Velocity search graph using DP on layered DAG.
    p0, v0: current state
    p_waypoints: array of waypoints of shape (Hg, 3)
    Returns max_T, best_sols_per_layer, best_velocities
    """
    Hg = len(p_waypoints)
    
    v_norms = np.linspace(v_min, v_max, s_bins)
    yaws = np.linspace(theta_min, theta_max, s_bins)
    pitches = np.linspace(psi_min, psi_max, s_bins)
    
    sampled_velocities = []
    for v in v_norms:
        for yaw in yaws:
            for pitch in pitches:
                vx = v * np.cos(pitch) * np.cos(yaw)
                vy = v * np.cos(pitch) * np.sin(yaw)
                vz = v * np.sin(pitch)
                sampled_velocities.append(np.array([vx, vy, vz]))
    sampled_velocities = np.array(sampled_velocities)
    num_samples = len(sampled_velocities)
    
    # dp[layer][j] = min total time to reach waypoint layer with velocity j
    dp = np.full((Hg, num_samples), float('inf'))
    parent = np.zeros((Hg, num_samples), dtype=int)
    sols = [[None]*num_samples for _ in range(Hg)]
    
    # layer 0: from p0, v0 to p_waypoints[0], sampled_velocities[j]
    for j in range(num_samples):
        T, sol = generate_3d_trajectory(p0, v0, p_waypoints[0], sampled_velocities[j], u_min, u_max)
        if T is not None:
            dp[0, j] = T
            sols[0][j] = sol
            
    # layer i: from p_waypoints[i-1], sampled_velocities[k] to p_waypoints[i], sampled_velocities[j]
    for i in range(1, Hg):
        for k in range(num_samples):
            if dp[i-1, k] == float('inf'):
                continue
            vk = sampled_velocities[k]
            for j in range(num_samples):
                vj = sampled_velocities[j]
                T, sol = generate_3d_trajectory(p_waypoints[i-1], vk, p_waypoints[i], vj, u_min, u_max)
                if T is not None:
                    if dp[i-1, k] + T < dp[i, j]:
                        dp[i, j] = dp[i-1, k] + T
                        parent[i, j] = k
                        sols[i][j] = sol
                        
    # Find best in last layer
    best_j = np.argmin(dp[-1])
    if dp[-1, best_j] == float('inf'):
        return None, None, None
        
    best_T = dp[-1, best_j]
    
    # Backtrack
    curr_j = best_j
    best_sols = []
    best_vs = []
    for i in range(Hg - 1, -1, -1):
        best_sols.append(sols[i][curr_j])
        best_vs.append(sampled_velocities[curr_j])
        curr_j = parent[i, curr_j]
        
    best_sols.reverse()
    best_vs.reverse()
    return best_T, best_sols, best_vs

def pmm_cone_refocusing(p0, v0, p_waypoints, config):
    """
    Implements Algorithm 1: PMM generation via cone refocusing.
    """
    v_min, v_max = config.v_min, config.v_max
    theta_min, theta_max = -np.pi, np.pi
    psi_min, psi_max = -np.pi/2 + 0.1, np.pi/2 - 0.1
    
    prev_T = float('inf')
    best_sols = None
    best_vs = None
    
    for k in range(config.K):
        T, sols, vs = dijkstra_search(
            p0, v0, p_waypoints, config.u_min, config.u_max, config.s, 
            v_min, v_max, theta_min, theta_max, psi_min, psi_max
        )
        
        if T is None:
            break
            
        best_sols = sols
        best_vs = vs
        
        if prev_T < float('inf') and prev_T / T < config.epsilon:
            break
            
        prev_T = T
        
        # Refocus cone around the first velocity in the path
        v_opt = vs[0]
        v_norm_opt = np.linalg.norm(v_opt)
        yaw_opt = np.arctan2(v_opt[1], v_opt[0])
        pitch_opt = np.arcsin(np.clip(v_opt[2] / (v_norm_opt + 1e-6), -1.0, 1.0))
        
        # Shrink window
        dv = (v_max - v_min) / 2.0
        dtheta = (theta_max - theta_min) / 2.0
        dpsi = (psi_max - psi_min) / 2.0
        
        v_min = max(0.1, v_norm_opt - dv)
        v_max = v_norm_opt + dv
        theta_min = yaw_opt - dtheta
        theta_max = yaw_opt + dtheta
        psi_min = max(-np.pi/2 + 0.1, pitch_opt - dpsi)
        psi_max = min(np.pi/2 - 0.1, pitch_opt + dpsi)
        
    return best_sols, best_vs
