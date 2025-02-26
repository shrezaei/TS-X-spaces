import numpy as np
import matplotlib.pyplot as plt
from utils.utils import create_path_if_not_exists


def uncertainty_plot_sort_based_integral_based(ts, ps, percentage_covered=1.0, save_path=None):
    ts = ts.reshape(-1)
    ps = ps.reshape(-1)
    ts_size = ts.shape[0]
    ts = np.abs(ts)
    full_ps = np.concatenate((ps, ps[1:-1]))
    ps = np.abs(full_ps)

    ts_normalization_factor = np.sqrt(np.sum(ts ** 2))
    ts = ts / ts_normalization_factor
    ps_normalization_factor = np.sqrt(np.sum(ps ** 2))
    ps = ps / ps_normalization_factor


    ts_uniques = np.unique(ts)
    ps_uniques = np.unique(ps)
    ts_steps = np.min([int(ts.shape[-1] * percentage_covered), ts_uniques.shape[0]])
    ps_steps = np.min([int(ps.shape[-1] * percentage_covered), ps_uniques.shape[0]])
    ts_epsilon_valuse = ts_uniques[:ts_steps]
    ps_epsilon_valuse = ps_uniques[:ps_steps]

    bound_gaps = np.ones((ts_steps, ps_steps, 3), dtype=float)
    violation_status = np.zeros((ts_steps, ps_steps), dtype=float)
    b_ratios = np.zeros((ts_steps, ps_steps), dtype=float)
    violation_counter = 0
    invalid_thresholds = 0
    t_concentration = 0
    f_concentration = 0

    if len(ts_uniques) == 1 or len(ps_uniques) == 1:
        return bound_gaps, -1, t_concentration, f_concentration

    for i in range(0, ts_steps):
        for j in range(0, ps_steps):
            ts_e = ts_epsilon_valuse[i]
            ps_e = ps_epsilon_valuse[j]
            Nt = np.sum(np.abs(ts) > ts_e)
            Nf = np.sum(ps > ps_e)

            ts_e_norm = np.sqrt(np.sum(ts[ts <= ts_e] ** 2))
            ps_e_norm = np.sqrt(np.sum(ps[ps <= ps_e] ** 2))
            b_ratio = (Nt*Nf) / (ts_size * ((1 - (ts_e_norm + ps_e_norm)) ** 2))

            t_concentration += Nt / ts_size
            f_concentration += Nf / ts_size

            if (ts_e_norm + ps_e_norm) >= 1:
                bound_gaps[i, j, 1] = 0
                invalid_thresholds += 1
                violation_status[i, j] = -1
            elif b_ratio >= 1:
                bound_gaps[i, j, 0] = b_ratio
                violation_status[i, j] = 0
            else:
                bound_gaps[i, j, 0] = 0
                bound_gaps[i, j, 2] = 0
                violation_counter += 1
                violation_status[i, j] = 1
                b_ratios[i, j] = 1 - b_ratio


    # compute the integral of violation
    violation_integral = 0
    non_violation_integral = 0
    for i in range(1, ts_steps):
        for j in range(1, ps_steps):
            if violation_status[i, j] == 1:
                if violation_status[i-1, j-1] == 1:
                    # violation_integral += (ts_epsilon_valuse[i] - ts_epsilon_valuse[i-1]) * (ps_epsilon_valuse[j] - ps_epsilon_valuse[j-1]) * b_ratios[i, j]
                    violation_integral += (ts_epsilon_valuse[i] - ts_epsilon_valuse[i - 1]) * (ps_epsilon_valuse[j] - ps_epsilon_valuse[j - 1])
                    continue
            non_violation_integral += (ts_epsilon_valuse[i] - ts_epsilon_valuse[i-1]) * (ps_epsilon_valuse[j] - ps_epsilon_valuse[j-1])
            
    violation_percentage = violation_integral / (violation_integral + non_violation_integral) * 100

    return bound_gaps, violation_percentage
