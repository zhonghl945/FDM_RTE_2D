import numpy as np
import h5py
import random
import numba
from scipy.special import roots_legendre
from scipy.interpolate import interp1d
import os
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

# Create the directory if it doesn't exist
os.makedirs('./data', exist_ok=True)


def generate_time_steps(total_time=1., initial_step=5e-5, max_step=1e-3, front_ratio=0.1):
    front_time = total_time * front_ratio
    steps_front = []
    current_time = 0
    step = initial_step
    growth_factor = 1.005

    num_steps_in_front = 0
    max_loop_iterations = 100000

    while current_time < front_time and num_steps_in_front < max_loop_iterations:
        step = min(step * growth_factor, max_step)
        step_to_add = min(step, front_time - current_time)

        if current_time > 0 and step_to_add < initial_step * 0.1:
            step_to_add = min(initial_step, front_time - current_time)

        step_to_add = max(step_to_add, 1e-9)

        if front_time - current_time < step_to_add:
            step_to_add = front_time - current_time

        if step_to_add <= 1e-9:
            break

        steps_front.append(step_to_add)
        current_time += step_to_add
        num_steps_in_front += 1

    if num_steps_in_front >= max_loop_iterations:
        print("警告: 在前段时间步生成中达到最大安全计数")

    idx_end_front = num_steps_in_front  # t_moment 中代表前段时间结束点的索引

    rear_time = total_time - current_time
    steps_rear = []
    if rear_time > 1e-9:
        num_steps_rear_ideal = np.ceil(rear_time / max_step)
        if num_steps_rear_ideal > 0:
            actual_step_rear = rear_time / num_steps_rear_ideal
            steps_rear = [actual_step_rear] * int(num_steps_rear_ideal)
            # 如果需要，对总和进行微小校正
            if steps_rear and not np.isclose(np.sum(steps_rear), rear_time):
                steps_rear[-1] += rear_time - np.sum(steps_rear)
        elif rear_time > 1e-9:  # rear_time 很小但为正
            steps_rear = [rear_time]

    all_steps = steps_front + steps_rear
    t_moment = np.cumsum([0] + all_steps)

    return all_steps, t_moment, idx_end_front


def make_s(num_S, Nt_minus_1, t_points_for_s):
    # random.seed(99)
    # np.random.seed(99)  # 为 numpy 的随机操作 (如 shuffle) 设置种子

    S_t = np.zeros((num_S, Nt_minus_1))
    increase_factor = 100
    increase = np.linspace(0.5, 1.01, increase_factor)

    if increase_factor == 0:
        interval = float(num_S)  # 避免除以零
    else:
        interval = np.ceil(num_S / increase_factor)
    if interval == 0 and num_S > 0: interval = 1.0  # 如果 num_S > 0，确保 interval 至少为 1

    j = 1
    for i in range(num_S):
        type = random.randint(1, 8)
        # Lambda 函数定义 (f 需要一个参数: t_val)
        if type == 1:
            f = lambda t_val: np.abs(np.sin(4 * np.pi * np.random.rand() * t_val))
        elif type == 2:
            a = 2 * np.random.rand() - 1
            f = lambda t_val: np.exp(a * (t_val - (0. if a < 0 else 1.)))
        elif type == 3:
            a = 2 * np.random.rand() - 1
            f = lambda t_val: np.abs(a * (t_val - (0. if a > 0 else 1.)))
        elif type == 4:
            f = lambda t_val: (t_val - np.random.rand()) ** 2
        elif type == 5:
            f = lambda t_val: 0.3333 * np.abs((t_val - np.random.rand()) ** 3) + 0.6667 * np.random.rand()
        elif type == 6:
            f = lambda t_val: 0.6666 * np.arctan(20 * np.random.rand() * t_val)
        elif type == 7:
            mu, sigma_rand = np.random.rand(), 0.2 * np.random.rand()
            f = lambda t_val: np.exp(-((t_val - mu) ** 2) / (2 * sigma_rand ** 2))
        elif type == 8:
            a = 2 * np.random.rand(2) - 1
            f = lambda t_val: abs(a[0]) / (a[1] * t_val + 2)

        if Nt_minus_1 > 0:
            S_t[i, :] = np.array([f(ti) for ti in t_points_for_s])

        if j <= len(increase) and interval > 0 and np.floor((i + 1) / interval) >= j:
            start_idx = int((j - 1) * interval)
            end_idx = min(int(j * interval), num_S)
            S_t[start_idx:end_idx, :] *= increase[j - 1]
            j += 1

    if num_S > 0 and Nt_minus_1 > 0:  # 仅当有内容时才进行 shuffle
        np.random.shuffle(S_t)

    print('BC max: ', S_t.max())
    print('BC min: ', S_t.min())
    print('BC mean: ', S_t.mean())
    print('BC std: ', S_t.std())
    return S_t


def create_resampled_bc(S_t_original, t_points_for_s, idx_end_front_in_t_points,
                        num_total_samples=64, front_sample_proportion=0.1):
    """
    根据给定的采样策略，对原始S_t进行重采样和分片线性插值。

    参数:
    S_t_original (np.array): 原始的边界条件数据 (num_S x Nt_minus_1)。
    t_points_for_s (np.array): 对应S_t_original的时间点 (t_moment[1:])。
    idx_end_front_in_t_points (int): 在 t_points_for_s 中，属于“前段”的点有多少个。
                                     这对应于 generate_time_steps 返回的 idx_end_front。
    num_total_samples (int): 总共要选取的采样点数量。
    front_sample_proportion (float): 从“前段”采样点所占总采样点数量的比例。

    返回:
    np.array: 重采样和插值后的新S_t数据。
    """
    num_S, Nt_minus_1 = S_t_original.shape
    S_t_new = np.zeros_like(S_t_original)
    num_samples_front = np.ceil(num_total_samples * front_sample_proportion).astype(int)
    num_samples_rear = num_total_samples - num_samples_front

    # 选择采样点的索引
    # 前段的可用点索引范围: [0, idx_end_front_in_t_points - 1]
    # 后段的可用点索引范围: [idx_end_front_in_t_points, Nt_minus_1 - 1]
    selected_indices_front = np.round(np.linspace(0, idx_end_front_in_t_points - 1, num_samples_front)).astype(int)
    selected_indices_front = np.unique(selected_indices_front)  # 确保唯一性
    selected_indices_rear = np.round(np.linspace(idx_end_front_in_t_points, Nt_minus_1 - 1, num_samples_rear)).astype(
        int)
    selected_indices_rear = np.unique(selected_indices_rear)  # 确保唯一性

    # 合并并排序采样索引
    combined_sampled_indices = np.sort(np.concatenate((selected_indices_front, selected_indices_rear)))
    t_sampled = t_points_for_s[combined_sampled_indices]

    # 对S_t_original的每一行进行插值
    for i in range(num_S):
        s_original_row = S_t_original[i, :]
        s_values_at_sampled_points = s_original_row[combined_sampled_indices]

        # 创建分片线性插值函数
        # fill_value用于处理t_points_for_s中的点可能超出t_sampled边界的情况（理论上不应发生很多）
        # 使用采样点的边界值进行填充，避免外插引入的较大误差
        interp_function = interp1d(t_sampled, s_values_at_sampled_points, kind='linear',
                                   bounds_error=False,
                                   fill_value=(s_values_at_sampled_points[0], s_values_at_sampled_points[-1]))

        S_t_new[i, :] = interp_function(t_points_for_s)

    print(f"\n通过 {len(combined_sampled_indices)} 个采样点进行插值后的新BC统计:")
    print('BC_new max: ', S_t_new.max())
    print('BC_new min: ', S_t_new.min())
    print('BC_new mean: ', S_t_new.mean())
    print('BC_new std: ', S_t_new.std())

    return S_t_new, combined_sampled_indices


def generate_sn_directions(N_theta, N_phi_per_theta):
    mu, w_mu = roots_legendre(N_theta)
    theta = np.arccos(mu)
    M = 0
    delta = (2 * np.pi) / (4 * N_phi_per_theta)
    phi = np.linspace(delta, 2 * np.pi + delta, N_phi_per_theta, endpoint=False)
    theta_list, phi_list, weight_list = [], [], []
    for i in range(N_theta):
        for j in range(N_phi_per_theta):
            theta_list.append(theta[i])
            phi_list.append(phi[j])
            # Ensure correct weight calculation
            weight_list.append(w_mu[i] * (2 * np.pi / N_phi_per_theta))
            M += 1
    return np.array(theta_list), np.array(phi_list), np.array(weight_list), M


@numba.njit(fastmath=True, parallel=True, cache=True)
def calculate_I_new_numba_optimized(I_old_cell, I_old_node, I_BC_left, I_BC_right, I_BC_top, I_BC_bottom, T_k4,
                                    sigma_T, mu_x, mu_y, dx, dy, dt, c, a, M):
    num_problems_batch, _, Nx, Ny = I_old_node.shape
    I_new_cell = np.zeros_like(I_old_cell)
    I_new_node = np.zeros_like(I_old_node)

    for p in numba.prange(num_problems_batch):
        for m in numba.prange(M):
            mux = mu_x[m]
            muy = mu_y[m]

            bc_x_left = I_BC_left[p]
            bc_x_right = I_BC_right[p]
            bc_y_left = I_BC_bottom[p]
            bc_y_right = I_BC_top[p]

            if mux > 0 and muy > 0:
                I_new_node[p, m, 0, :] = bc_x_left
                I_new_node[p, m, :, 0] = bc_y_left
                for i in range(Nx - 1):
                    for j in range(Ny - 1):
                        # 菱形格式
                        source = a * c * sigma_T[p, i, j] * T_k4[p, i, j] / (4 * np.pi) + \
                                 I_old_cell[p, m, i, j] / (c * dt)
                        term_x = (abs(mux) / dx) * (I_new_node[p, m, i, j] + I_new_node[p, m, i, j + 1])
                        term_y = (abs(muy) / dy) * (I_new_node[p, m, i, j] + I_new_node[p, m, i + 1, j])
                        denom = 1.0 / (c * dt) + (2.0 * abs(mux) / dx) + (2.0 * abs(muy) / dy) + sigma_T[p, i, j]

                        I_new_cell[p, m, i, j] = max((source + term_x + term_y) / denom, 0)
                        I_new_node[p, m, i + 1, j + 1] = 4 * I_new_cell[p, m, i, j] - I_new_node[p, m, i, j] \
                                                         - I_new_node[p, m, i, j + 1] - I_new_node[p, m, i + 1, j]

                        if I_new_node[p, m, i + 1, j + 1] < 0:
                            I_new_node[p, m, i + 1, j + 1] = 0.

            elif mux < 0 and muy < 0:
                I_new_node[p, m, -1, :] = bc_x_right
                I_new_node[p, m, :, -1] = bc_y_right
                for i in range(Nx - 2, -1, -1):
                    for j in range(Ny - 2, -1, -1):
                        # 菱形格式
                        source = a * c * sigma_T[p, i, j] * T_k4[p, i, j] / (4 * np.pi) + \
                                 I_old_cell[p, m, i, j] / (c * dt)
                        term_x = (abs(mux) / dx) * (I_new_node[p, m, i + 1, j] + I_new_node[p, m, i + 1, j + 1])
                        term_y = (abs(muy) / dy) * (I_new_node[p, m, i, j + 1] + I_new_node[p, m, i + 1, j + 1])
                        denom = 1.0 / (c * dt) + (2.0 * abs(mux) / dx) + (2.0 * abs(muy) / dy) + sigma_T[p, i, j]

                        I_new_cell[p, m, i, j] = max((source + term_x + term_y) / denom, 0)
                        I_new_node[p, m, i, j] = 4 * I_new_cell[p, m, i, j] - I_new_node[p, m, i + 1, j + 1] \
                                                 - I_new_node[p, m, i, j + 1] - I_new_node[p, m, i + 1, j]

                        if I_new_node[p, m, i, j] < 0:
                            I_new_node[p, m, i, j] = 0.

            elif mux > 0 and muy < 0:
                I_new_node[p, m, 0, :] = bc_x_left
                I_new_node[p, m, :, -1] = bc_y_right
                for i in range(Nx - 1):
                    for j in range(Ny - 2, -1, -1):
                        # 菱形格式
                        source = a * c * sigma_T[p, i, j] * T_k4[p, i, j] / (4 * np.pi) + \
                                 I_old_cell[p, m, i, j] / (c * dt)
                        term_x = (abs(mux) / dx) * (I_new_node[p, m, i, j] + I_new_node[p, m, i, j + 1])
                        term_y = (abs(muy) / dy) * (I_new_node[p, m, i, j + 1] + I_new_node[p, m, i + 1, j + 1])
                        denom = 1.0 / (c * dt) + (2.0 * abs(mux) / dx) + (2.0 * abs(muy) / dy) + sigma_T[p, i, j]

                        I_new_cell[p, m, i, j] = max((source + term_x + term_y) / denom, 0)
                        I_new_node[p, m, i + 1, j] = 4 * I_new_cell[p, m, i, j] - I_new_node[p, m, i, j] \
                                                     - I_new_node[p, m, i, j + 1] - I_new_node[p, m, i + 1, j + 1]

                        if I_new_node[p, m, i + 1, j] < 0:
                            I_new_node[p, m, i + 1, j] = 0.

            elif mux < 0 and muy > 0:
                I_new_node[p, m, -1, :] = bc_x_right
                I_new_node[p, m, :, 0] = bc_y_left
                for i in range(Nx - 2, -1, -1):
                    for j in range(Ny - 1):
                        # 菱形格式
                        source = a * c * sigma_T[p, i, j] * T_k4[p, i, j] / (4 * np.pi) + \
                                 I_old_cell[p, m, i, j] / (c * dt)
                        term_x = (abs(mux) / dx) * (I_new_node[p, m, i + 1, j] + I_new_node[p, m, i + 1, j + 1])
                        term_y = (abs(muy) / dy) * (I_new_node[p, m, i, j] + I_new_node[p, m, i + 1, j])
                        denom = 1.0 / (c * dt) + (2.0 * abs(mux) / dx) + (2.0 * abs(muy) / dy) + sigma_T[p, i, j]

                        I_new_cell[p, m, i, j] = max((source + term_x + term_y) / denom, 0)
                        I_new_node[p, m, i, j + 1] = 4 * I_new_cell[p, m, i, j] - I_new_node[p, m, i, j] \
                                                     - I_new_node[p, m, i + 1, j] - I_new_node[p, m, i + 1, j + 1]

                        if I_new_node[p, m, i, j + 1] < 0:
                            I_new_node[p, m, i, j + 1] = 0.

    return I_new_cell, I_new_node


def run_sn_batch(BC_batch, current_batch_size, T_initial,
                 dt_inho, Nt, indices,
                 mu_x, mu_y, weights, M,
                 Nx, Ny, dx, dy,
                 c, sigma_map, a, C_v,
                 max_outer_iters, max_source_iters,
                 I_tolerance, T_tolerance,
                 batch_num):
    # Initialize
    T_cell = np.full((current_batch_size, Nx - 1, Ny - 1), T_initial, dtype=np.float64)
    I_initial_val = a * c * T_initial ** 4 / (4 * np.pi)
    I_cell = np.full((current_batch_size, M, Nx - 1, Ny - 1), I_initial_val, dtype=np.float64)
    I_node = np.full((current_batch_size, M, Nx, Ny), I_initial_val, dtype=np.float64)
    pho_batch = np.zeros((current_batch_size, len(indices), Nx - 1, Ny - 1), dtype=np.float64)
    ss = 0

    # Time Stepping Loop
    for n in tqdm(range(Nt - 1), desc=f"Batch {batch_num} Time Steps", leave=False):
        dt = dt_inho[n]

        # Extract BC for the current time step
        I_BC_left_np = 0. * BC_batch[:, n].astype(np.float64)
        I_BC_right_np = 0. * BC_batch[:, n].astype(np.float64)
        I_BC_top_np = BC_batch[:, n].astype(np.float64)
        I_BC_bottom_np = 0.2 * BC_batch[:, n].astype(np.float64)

        T_old = 1.0001 * T_cell
        S_old = np.sum(I_cell * weights[None, :, None, None], axis=1)

        # Outer Iteration Loop
        outer_converged = False
        for outer_iter in range(max_outer_iters):
            T_old_3 = T_old ** 3
            T_old_4 = T_old_3 * T_old
            sigma_T = sigma_map[None, :, :] / T_old_3  # Shape: (current_batch_size, Nx - 1, Ny - 1)

            # Source Iteration Loop
            source_iter_converged = False
            for src_iter in range(max_source_iters):
                # calculate T_for_I_calc
                T_for_I_calc = (C_v / dt * T_cell + sigma_T * S_old + 3 * sigma_T * a * c * T_old_4) / \
                               (C_v / dt + 4 * sigma_T * a * c * T_old_3)
                T_for_I_calc_4 = T_old_4 + 4 * T_old_3 * (T_for_I_calc - T_old)

                # calculate I_new
                I_new_cell, I_new_node = calculate_I_new_numba_optimized(
                    I_cell, I_node,
                    I_BC_left_np, I_BC_right_np, I_BC_top_np, I_BC_bottom_np,
                    T_for_I_calc_4, sigma_T,
                    mu_x, mu_y, dx, dy, dt, c, a, M)

                S_new = np.sum(I_new_cell * weights[None, :, None, None], axis=1)
                delta_S_rel = np.max(np.abs(S_new - S_old)) / np.max(np.abs(S_new))
                S_old = S_new

                if delta_S_rel < I_tolerance and src_iter > 0:
                    # print(f"    Source Iter {src_iter + 1}: Converged (Rel ΔS={delta_S_rel:.2e})")
                    source_iter_converged = True
                    break

            if not source_iter_converged:
                print(
                    f" Batch {batch_num}, Time {n + 1}: Source iter max reached ({max_source_iters} iters, "
                    f" (Rel ΔS={delta_S_rel:.2e})")
                raise ValueError("Source Iteration is not converged")
            # End Source Iteration

            # 源迭代收敛后，更新温度 T_old -> T_new
            T_new = (dt * (sigma_T * S_old - a * c * sigma_T * T_for_I_calc_4) + C_v * T_cell) / C_v
            delta_T_rel = np.max(np.abs(T_new - T_old)) / np.max(np.abs(T_new))
            T_old = T_new
            # print(f"  Outer Iter {outer_iter + 1}: Rel ΔT={delta_T_rel:.2e}")

            if delta_T_rel < T_tolerance and outer_iter > 0:
                # print(f"  Outer Iter {outer_iter + 1}: Converged (Rel ΔT={delta_T_rel:.2e})")
                outer_converged = True
                break

        if not outer_converged:
            print(f"  Batch {batch_num}, Time {n + 1}: Outer iter max reached ({max_outer_iters} iters, "
                  f"  (Rel ΔT={delta_T_rel:.2e})")
            raise ValueError("Outer Iteration is not converged")
        # End Outer Iteration

        T_cell = T_old
        I_cell = I_new_cell
        I_node = I_new_node
        if n in indices:
            pho_batch[:, ss, ...] = S_old / (4 * np.pi)
            ss += 1
    # End Time Stepping Loop

    return pho_batch


def process_and_save_batches(BC_array_full, BC_sampled, total_num_problems, batch_size, T_initial,
                             dt_inho, Nt, indices,
                             mu_x, mu_y, weights, M,
                             Nx, Ny, dx, dy,
                             c, sigma_map, a, C_v,
                             max_outer_iters, max_source_iters,
                             I_tolerance, T_tolerance,
                             output_filename, group_name):
    print(f"Initializing HDF5 output file: {output_filename}")
    sampling_num_t = len(indices)
    with h5py.File(output_filename, 'w') as f1:
        group = f1.create_group(group_name)
        rho_dset = group.create_dataset('rho',
                                        shape=(total_num_problems, sampling_num_t, Nx - 1, Ny - 1),
                                        dtype=np.float64,
                                        chunks=(1, sampling_num_t, Nx - 1, Ny - 1))
        bc_dset = group.create_dataset('BC',
                                       shape=(total_num_problems, sampling_num_t),
                                       dtype=np.float64,
                                       chunks=(1, sampling_num_t))

        num_batches = int(np.ceil(total_num_problems / batch_size))
        for i_batch in range(num_batches):
            batch_start_time = time.time()

            start_idx = i_batch * batch_size
            end_idx = min((i_batch + 1) * batch_size, total_num_problems)
            current_batch_size = end_idx - start_idx
            if current_batch_size == 0:
                continue
            BC_batch = BC_array_full[start_idx:end_idx, :]
            BC_batch_sampled = BC_sampled[start_idx:end_idx, :]

            print(f"\n--- Starting Batch {i_batch + 1}/{num_batches} (Size: {current_batch_size}) ---")
            pho_batch_result = run_sn_batch(
                BC_batch, current_batch_size, T_initial,
                dt_inho, Nt, indices,
                mu_x, mu_y, weights, M,
                Nx, Ny, dx, dy,
                c, sigma_map, a, C_v,
                max_outer_iters, max_source_iters,
                I_tolerance, T_tolerance,
                i_batch + 1
            )

            rho_dset[start_idx:end_idx, :, :, :] = pho_batch_result
            bc_dset[start_idx:end_idx, :] = BC_batch_sampled

            batch_end_time = time.time()
            print(
                f"Batch {i_batch + 1}/{num_batches} finished. Time: {batch_end_time - batch_start_time:.2f}s. Wrote rows {start_idx}-{end_idx - 1}")

    print(f"\nResults saved to {output_filename} in group '{group_name}'.")


def Sn(BC=None, filename='2D_Sn_Python', Num=1000, T_initial=0.001, BatchSize=100):
    start_main = time.time()

    # Physics and Geometry constants
    c = 29.98
    a = 0.01372
    C_v = 0.3
    Lx, Ly = 0.2, 0.4
    Nx, Ny = 33, 65
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    S_n_order = 8

    # Tolerances and Iteration Limits
    max_outer_iters = 30000
    max_source_iters = 30000
    I_tolerance = 1e-5
    T_tolerance = 1e-4

    # time sampling num
    num_samples_overall = 64  # 期望的总采样时间点数
    front_sample_proportion = 0.1  # 对前段插值点的比例

    # Define sigma values
    sigma_background = 30.0
    sigma_region1 = 300.0
    # sigma_region2 = 100.0
    # Add more as needed...
    sigma_map = np.full((Nx - 1, Ny - 1), sigma_background, dtype=np.float64)
    x_vector = np.linspace(dx / 2, Lx - dx / 2, Nx - 1)
    y_vector = np.linspace(dy / 2, Ly - dy / 2, Ny - 1)
    X_coords, Y_coords = np.meshgrid(x_vector, y_vector, indexing='ij')
    # region1
    x_start1, x_end1, y_start1, y_end1 = 0.05, 0.15, 0.1, 0.3
    mask_1 = (X_coords >= x_start1) & (X_coords <= x_end1) & (Y_coords >= y_start1) & (Y_coords <= y_end1)
    sigma_map[mask_1] = sigma_region1
    # region2
    # x_start2, x_end2, y_start2, y_end2 = 0.3, 0.45, 0.3, 0.45
    # mask_2 = (X_coords >= x_start2) & (X_coords <= x_end2) & (Y_coords >= y_start2) & (Y_coords <= y_end2)
    # sigma_map[mask_2] = sigma_region2

    # Visualize the sigma_map
    plt.figure(figsize=(8, 6))
    plt.imshow(sigma_map.T, origin='lower', extent=[0, Lx, 0, Ly], cmap='viridis', aspect='auto')
    cbar = plt.colorbar()
    # cbar.set_label(r'$\sigma$', fontsize=20)
    cbar.ax.tick_params(labelsize=12)
    plt.xlabel(r'$x$', fontsize=20)
    plt.ylabel(r'$y$', fontsize=20)
    plt.tick_params(axis='both', labelsize=12)
    plt.savefig('Medium Distribution')
    plt.close()

    # generate Sn directions
    theta, phi, weights, M = generate_sn_directions(S_n_order, S_n_order)
    mu_x = np.sin(theta) * np.cos(phi)
    mu_y = np.sin(theta) * np.sin(phi)
    print(f"Number of Sn directions (M): {M}")

    # generate time steps
    dt_inho, t_moment, idx_end_front = generate_time_steps()
    Nt = len(t_moment)
    print(f"总模拟时间: {t_moment[-1]:.4f}s")
    print(f"生成的总时间点数 (Nt, 包括 t=0): {Nt}")
    print(f"标记前段时间结束的索引 (idx_end_front): {idx_end_front}")
    print(f"前段时间结束时的时间 (t_moment[idx_end_front]): {t_moment[idx_end_front]:.4f}s")

    # Prepare BCs
    if (BC is None) and (filename == '2D_Sn_Python'):
        # Training mode: Generate BCs
        total_num_problems = Num
        # 使用原始 make_s 生成初始 BC 数组, make_s 需要不包括 t=0 的时间点，因此我们传递 t_moment[1:]
        # BC_array_original 的列数将是 Nt-1
        BC_array_original = make_s(Num, Nt - 1, t_moment[1:])
        print(f"原始 BC_array 形状: {BC_array_original.shape}")
        BC_array_final, sampled_indices = create_resampled_bc(BC_array_original, t_moment[1:], idx_end_front,
                                                              num_total_samples=num_samples_overall,
                                                              front_sample_proportion=front_sample_proportion)
        BC_sampled = BC_array_final[:, sampled_indices]
        print(f"最终修改后的 BC_array 形状: {BC_array_final.shape}")

        output_filename = f'./data/{filename}.h5'
        group_name = 'Train'

    elif (BC is not None) and (filename == 'Test_Data'):
        # Testing mode: Use provided BCs
        total_num_problems = BC.shape[0]
        BC_array_original = BC
        print(f"原始 BC_array 形状: {BC_array_original.shape}")
        BC_array_final, sampled_indices = create_resampled_bc(BC_array_original, t_moment[1:], idx_end_front,
                                                              num_total_samples=num_samples_overall,
                                                              front_sample_proportion=front_sample_proportion)
        BC_sampled = BC_array_final[:, sampled_indices]
        print(f"最终修改后的 BC_array 形状: {BC_array_final.shape}")

        output_filename = f'./data/{filename}.h5'
        group_name = 'Test'

    elif (BC is not None) and (filename == 'Test_Results'):
        # Testing mode: Use provided BCs
        total_num_problems = BC.shape[0]
        BC_sampled = BC
        print(f"BC_array 形状: {BC_sampled.shape}")

        if BC.shape[1] != num_samples_overall:
            raise ValueError(f"输入BC序列的长度 ({BC.shape[1]}) 与期望的采样点数 ({num_samples_overall}) 不符")

        num_samples_front = np.ceil(num_samples_overall * front_sample_proportion).astype(int)
        num_samples_rear = num_samples_overall - num_samples_front

        # 从 t_moment[1:] 的选择索引
        indices_front = np.round(np.linspace(0, idx_end_front - 1, num_samples_front)).astype(int)
        sampled_indices_front_list = list(np.unique(indices_front))
        indices_rear = np.round(np.linspace(idx_end_front, Nt - 2, num_samples_rear)).astype(int)
        sampled_indices_rear_list = list(np.unique(indices_rear))

        # 合并并排序索引
        sampled_indices = np.sort(np.unique(np.array(sampled_indices_front_list + sampled_indices_rear_list, dtype=int)))

        if len(sampled_indices) != num_samples_overall:
            print(
                f"错误: 重建的采样点索引数量 ({len(sampled_indices)})与期望的BC序列长度 ({num_samples_overall}) 不符。\n"
            )

        # 获取这些索引对应的时间点
        t_values_for_given_bc = t_moment[1:][sampled_indices]

        # 使用分片线性插值
        interp_func = interp1d(
            t_values_for_given_bc,
            BC_sampled,  # 长度为64的BC值
            kind='linear',
            bounds_error=False,
            fill_value=(BC_sampled[:, 0], BC_sampled[:, -1])  # 使用首尾值填充边界外区域
        )

        # 在所有原始时间点上计算插值结果
        BC_array_final = interp_func(t_moment[1:])
        print(f"最终修改后的 BC_array 形状: {BC_array_final.shape}")

        output_filename = f'./data/{filename}.h5'
        group_name = 'Test'

    else:
        raise ValueError("BC mode and filename is illegal")

    # Run Batch Processing
    process_and_save_batches(
        BC_array_final, BC_sampled, total_num_problems, BatchSize, T_initial,
        dt_inho, Nt, sampled_indices,
        mu_x, mu_y, weights, M,
        Nx, Ny, dx, dy,
        c, sigma_map, a, C_v,
        max_outer_iters, max_source_iters,
        I_tolerance, T_tolerance,
        output_filename, group_name
    )

    end_main = time.time()
    print(f"\nTotal simulation and saving completed in {end_main - start_main:.2f} seconds.")


if __name__ == '__main__':
    print("--- Running Training Data Generation ---")
    random.seed(99)
    np.random.seed(99)
    Sn(BC=None, filename='2D_Sn_Python', Num=1, T_initial=1e-6, BatchSize=1)

    print("\n" + "=" * 30 + "\n")

    print("--- Running Testing with Provided BC ---")
    _, t_moment, _ = generate_time_steps()
    t = t_moment[1:]
    S1 = 0.5 * np.ones((len(t),))
    S2 = 2 ** t - 1
    S3 = -np.log(0.5 * t + 0.4)
    # S1 = t ** 0.3
    # S2 = 0.2 * np.ones((len(t),))
    # S2[-500:] = 0.8
    # S3 = 0.5 * np.arccos(np.clip(t, 0.0, 1.0))
    Test_St = np.stack((S1, S2, S3), axis=0)
    Sn(BC=Test_St, filename='Test_Data', Num=Test_St.shape[0], T_initial=1e-6, BatchSize=Test_St.shape[0])
