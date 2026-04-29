import numpy as np

# ==========================================
# SFC Attitude Controller Tuning Parameters
# ==========================================

# Position-controller gains (Newtons / metre)
KP = np.array([1.044, 1.044, 1.25])
KI = np.array([0.05, 0.05, 0.05])
KD = np.array([0.598, 0.598, 0.4])
KI_RANGE = np.array([2.0, 2.0, 2.0])         # symmetric integrator clamp

# Saturation / smoothing
TILT_LIMIT = 1.176                            # rad (~67°)
TILT_RATE_LIMIT = 0.804                       # rad per 50 Hz tick


# ==========================================
# SFC Planner Tuning Parameters
# ==========================================

# TOPP (variable-speed schedule) tunables
V_MAX_GLOBAL = 2.0          # m/s. Speed ceiling on straights.
TILT_LIMIT_PLANNER = 0.35   # rad. Assumed max tilt for planning (should be < controller TILT_LIMIT).
A_LONG_MAX_FACTOR = 0.7     # a_long_max = factor * a_lat_max. Vertical thrust eats some accel budget.
V_FLOOR = 0.3               # m/s. Floor on scheduled speed.
N_TOPP_SAMPLES = 200        # Number of points to sample u in [0, 1] when building the schedule.
