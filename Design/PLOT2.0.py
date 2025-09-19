import numpy as np
import h5py
import os
import numba
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from tqdm import tqdm


# =============================================================================
# 1. 高性能求解器 (保持不变)
# =============================================================================
@numba.jit(nopython=True, fastmath=True)
def _time_loop_numba(T, S, mask_x_in_rect, mask_y_in_rect, results_3d, sample_indices, nt, nx, ny, dt, dx, dy, Cv,
                     T_floor):
    sample_idx_counter = 0
    for n in range(nt):
        Tn = T.copy()
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                t_avg_right = (Tn[j, i + 1] + Tn[j, i]) / 2.0
                kappa_right = (t_avg_right ** 3) / 50.0 if mask_x_in_rect[j, i] else (t_avg_right ** 3) / 300.0
                flux_right = kappa_right * (Tn[j, i + 1] - Tn[j, i]) / dx
                t_avg_left = (Tn[j, i] + Tn[j, i - 1]) / 2.0
                kappa_left = (t_avg_left ** 3) / 50.0 if mask_x_in_rect[j, i - 1] else (t_avg_left ** 3) / 300.0
                flux_left = kappa_left * (Tn[j, i] - Tn[j, i - 1]) / dx
                t_avg_top = (Tn[j + 1, i] + Tn[j, i]) / 2.0
                kappa_top = (t_avg_top ** 3) / 50.0 if mask_y_in_rect[j, i] else (t_avg_top ** 3) / 300.0
                flux_top = kappa_top * (Tn[j + 1, i] - Tn[j, i]) / dy
                t_avg_bottom = (Tn[j, i] + Tn[j - 1, i]) / 2.0
                kappa_bottom = (t_avg_bottom ** 3) / 50.0 if mask_y_in_rect[j - 1, i] else (t_avg_bottom ** 3) / 300.0
                flux_bottom = kappa_bottom * (Tn[j, i] - Tn[j - 1, i]) / dy
                divergence = (flux_right - flux_left) / dx + (flux_top - flux_bottom) / dy
                T[j, i] = Tn[j, i] + (dt / Cv) * (divergence + S[j, i])
        for j in range(ny):
            for i in range(nx):
                if T[j, i] < T_floor: T[j, i] = T_floor
        if n == nt - 1:
            results_3d[0, :, :] = T[1:-1, 1:-1]
    return T, results_3d


def solve_final_temperature(nx, ny, nt, S_interior):
    Lx, Ly, t_final = 1.0, 1.0, 0.5
    Cv, T0, T_floor = 1.0, 0.1, 1e-9
    dx, dy, dt = Lx / (nx - 1), Ly / (ny - 1), t_final / nt
    T = np.full((ny, nx), T0, dtype=np.float64)
    S = np.zeros((ny, nx), dtype=np.float64)
    if S_interior is not None: S[1:-1, 1:-1] = S_interior
    rect_x_range, rect_y_range = (0.25, 0.75), (0.25, 0.75)
    x_coords_full, y_coords_full = np.linspace(0, Lx, nx), np.linspace(0, Ly, ny)
    x_faces_v = x_coords_full[:-1] + dx / 2
    X_faces_v, Y_faces_v = np.meshgrid(x_faces_v, y_coords_full)
    mask_x_in_rect = (X_faces_v >= rect_x_range[0]) & (X_faces_v < rect_x_range[1]) & (Y_faces_v >= rect_y_range[0]) & (
            Y_faces_v < rect_y_range[1])
    y_faces_h = y_coords_full[:-1] + dy / 2
    X_faces_h, Y_faces_h = np.meshgrid(x_coords_full, y_faces_h)
    mask_y_in_rect = (X_faces_h >= rect_x_range[0]) & (X_faces_h < rect_x_range[1]) & (Y_faces_h >= rect_y_range[0]) & (
            Y_faces_h < rect_y_range[1])
    sample_indices = np.array([nt - 1])
    results_3d = np.zeros((1, ny - 2, nx - 2), dtype=np.float64)
    _, results_3d = _time_loop_numba(T, S, mask_x_in_rect, mask_y_in_rect, results_3d, sample_indices, nt, nx, ny, dt,
                                     dx, dy, Cv, T_floor)
    return results_3d[0]


# =============================================================================
# 2. 新增：计算次级目标的函数
# =============================================================================
def calculate_secondary_objectives(S_interior, dx, dy):
    """
    计算给定源项的三个次级目标。
    """
    # 目标1: 总输入能量 (∫S^2 dA)
    total_energy = np.sum(S_interior ** 2) * dx * dy

    # 目标2: 控制平滑度 (∫|∇S|^2 dA), 使用梯度近似
    grad_y, grad_x = np.gradient(S_interior, dy, dx)
    smoothness = np.sum(grad_x ** 2 + grad_y ** 2) * dx * dy

    # 目标3: 峰值功率 (max(S))
    peak_power = np.max(S_interior)

    return total_energy, smoothness, peak_power


# =============================================================================
# 3. 可视化函数
# =============================================================================
def create_exploration_plot(ref_data, sample1_data, sample2_data, sample3_data, constraint_params, grid_params):
    # ... (此函数保持不变) ...
    S_ref, T_ref = ref_data
    S1, T1, info1 = sample1_data
    S2, T2, info2 = sample2_data
    S3, T3, info3 = sample3_data

    NX, NY = grid_params['NX'], grid_params['NY']
    x = np.linspace(0, 1.0, NX)
    y = np.linspace(0, 1.0, NY)
    X_full, Y_full = np.meshgrid(x, y)
    X_interior, Y_interior = X_full[1:-1, 1:-1], Y_full[1:-1, 1:-1]
    plot_extent = [x[1], x[-2], y[1], y[-2]]

    fig = plt.figure(figsize=(15, 22))
    gs = gridspec.GridSpec(5, 3, figure=fig)

    # --- Row 1: 参考解 ---
    ax_s_ref = fig.add_subplot(gs[0, 0], projection='3d')
    ax_s_ref.set_title("$F_{ref}$")
    ax_s_ref.plot_surface(X_interior, Y_interior, S_ref, cmap='viridis', rstride=2, cstride=2)

    ax_t_ref = fig.add_subplot(gs[0, 1:])
    ax_t_ref.set_title("$T_{ref}$")
    im_t_ref = ax_t_ref.imshow(T_ref, extent=plot_extent, origin='lower', cmap='inferno')
    fig.colorbar(im_t_ref, ax=ax_t_ref, fraction=0.046, pad=0.04)

    # --- Row 2: 三个被选出的生成源项 ---
    s_plots = [(S1, " "),
               (S2, "$F_{gen}$"),
               (S3, " ")]
    for i, (s_data, title) in enumerate(s_plots):
        ax = fig.add_subplot(gs[1, i], projection='3d')
        ax.set_title(title)
        ax.plot_surface(X_interior, Y_interior, s_data, cmap='viridis', rstride=2, cstride=2)

    # --- Row 3: 源项差异 ---
    s_diff_plots = [(S1 - S_ref, " "),
                    (S2 - S_ref, "$F_{gen} - F_{ref}$"),
                    (S3 - S_ref, " ")]
    for i, (s_diff, title) in enumerate(s_diff_plots):
        ax = fig.add_subplot(gs[2, i], projection='3d')
        ax.set_title(title)
        ax.plot_surface(X_interior, Y_interior, s_diff, cmap='coolwarm', rstride=2, cstride=2)

    # --- Row 4: 对应的求解温度场 ---
    t_plots = [(T1, " "),
               (T2, "$T_{sol}$ (t=0.5)"),
               (T3, " ")]
    vmin_t = min(T1.min(), T2.min(), T3.min(), T_ref.min())
    vmax_t = max(T1.max(), T2.max(), T3.max(), T_ref.max())
    for i, (t_data, title) in enumerate(t_plots):
        ax = fig.add_subplot(gs[3, i])
        ax.set_title(title)
        im = ax.imshow(t_data, extent=plot_extent, origin='lower', cmap='inferno', vmin=vmin_t, vmax=vmax_t)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- Row 5: 相对误差图 ---
    err_plots = [(T1, info1, " "),
                 (T2, info2, " Relative pointwise absolute error"),
                 (T3, info3, " ")]
    epsilon = 1e-9
    for i, (t_data, info, title) in enumerate(err_plots):
        ax = fig.add_subplot(gs[4, i])
        ax.set_title(title)
        rel_err_map = np.abs(t_data - T_ref) / (np.abs(T_ref) + epsilon)
        im = ax.imshow(rel_err_map, extent=plot_extent, origin='lower', cmap='magma')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        x_range, y_range = constraint_params['x_range'], constraint_params['y_range']
        rect_width = x_range[1] - x_range[0]
        rect_height = y_range[1] - y_range[0]
        rect = patches.Rectangle((x_range[0], y_range[0]), rect_width, rect_height,
                                 linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)

        error_text = f"Constrained area:\n{info['error']:.2e}"
        ax.text(0.95, 0.05, error_text, ha='right', va='bottom', transform=ax.transAxes,
                fontsize=12, bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.8))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.08, top=0.90, hspace=0.3, wspace=0.3)


# =============================================================================
# 4. 多目标性能分析图的可视化函数
# =============================================================================
def create_multiobjective_plot(all_errors, all_secondary_metrics, highlighted_indices):
    errors_np = np.array([e[0] for e in all_errors])
    energies_np = np.array([m[0] for m in all_secondary_metrics])
    smoothness_np = np.array([m[1] for m in all_secondary_metrics])
    peaks_np = np.array([m[2] for m in all_secondary_metrics])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- 图1: 控制误差直方图 ---
    axes[0, 0].hist(errors_np, bins=15, color='skyblue', edgecolor='black')
    axes[0, 0].set_title("Primary objective: control error")
    axes[0, 0].set_xlabel("Relative $L_2$ error")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)

    # --- 散点图通用设置 ---
    scatter_plots_info = [
        (axes[0, 1], energies_np, "Subsidiary objective: energy"),
        (axes[1, 0], smoothness_np, "Subsidiary objective: unsmoothness"),
        (axes[1, 1], peaks_np, "Subsidiary objective: peak")
    ]

    # 提取高亮样本的数据
    highlight_errors = errors_np[highlighted_indices]
    highlight_energies = energies_np[highlighted_indices]
    highlight_smoothness = smoothness_np[highlighted_indices]
    highlight_peaks = peaks_np[highlighted_indices]
    highlight_data = [highlight_energies, highlight_smoothness, highlight_peaks]

    for i, (ax, y_data, title) in enumerate(scatter_plots_info):
        # 绘制所有样本的散点图
        sc = ax.scatter(errors_np, y_data, alpha=0.6, label='All samples')

        # 高亮显示选定的样本
        ax.scatter(highlight_errors, highlight_data[i],
                   color='red', s=150, marker='*', edgecolor='black', zorder=5, label='Selected samples')

        ax.set_title(title)
        ax.set_xlabel("Relative $L_2$ error")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.4, wspace=0.3)


# =============================================================================
# 5. 主程序入口
# =============================================================================
if __name__ == '__main__':
    # --- 参数定义 ---
    NX, NY, NT = 66, 66, 40000
    GEN_FILE_PATH = './data/Test_Gen.h5'
    DATA_FILE_PATH = './data/Test_Data.h5'
    OUTPUT_DIR = 'exploration_results'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # --- 1. 加载所有必要数据 ---
    print("正在从HDF5文件加载数据...")
    with h5py.File(GEN_FILE_PATH, 'r') as f:
        S_gen_all = f['Test']['S'][:]
    with h5py.File(DATA_FILE_PATH, 'r') as f:
        S_ref_interior = f['Test']['S'][0]
        T_ref_final = f['Test']['T'][0][-1]

    num_samples = S_gen_all.shape[0]
    print(f"加载了 {num_samples} 个生成的源项和1个参考解。")

    # --- 2. 计算所有样本的主目标（控制误差）和次级目标 ---
    print("\n正在计算所有样本的控制误差和次级目标 (这可能需要一些时间)...")
    errors = []
    secondary_metrics = []
    solved_temperatures = {}

    constraint = {'x_range': (0.4, 0.6), 'y_range': (0.4, 0.6)}
    dx = 1.0 / (NX - 1)
    dy = 1.0 / (NY - 1)
    x_start_idx = int(round((constraint['x_range'][0] / dx) - 1))
    x_end_idx = int(round((constraint['x_range'][1] / dx) - 1))
    y_start_idx = int(round((constraint['y_range'][0] / dy) - 1))
    y_end_idx = int(round((constraint['y_range'][1] / dy) - 1))
    constraint_slice = (slice(y_start_idx, y_end_idx + 1), slice(x_start_idx, x_end_idx + 1))

    for i in tqdm(range(num_samples), desc="评估样本"):
        s_gen = S_gen_all[i]
        t_sol = solve_final_temperature(NX, NY, NT, s_gen)

        # 计算主目标：控制误差
        t_sol_zone = t_sol[constraint_slice]
        t_ref_zone = T_ref_final[constraint_slice]
        error_l2 = np.sqrt(np.mean((t_sol_zone - t_ref_zone) ** 2))
        norm_ref = np.sqrt(np.mean(t_ref_zone ** 2))
        relative_error = error_l2 / norm_ref if norm_ref > 1e-9 else error_l2
        errors.append((relative_error, i))

        # 计算次级目标
        energy, smoothness, peak = calculate_secondary_objectives(s_gen, dx, dy)
        secondary_metrics.append((energy, smoothness, peak, i))

        solved_temperatures[i] = t_sol

    # 根据误差排序
    errors.sort(key=lambda x: x[0])
    # 确保次级目标的顺序与误差排序后的顺序一致
    original_indices_sorted = [idx for err, idx in errors]
    secondary_metrics_sorted = sorted(secondary_metrics, key=lambda x: original_indices_sorted.index(x[3]))

    print("所有目标的计算和排序完成。")

    # --- 3. 手动选择要详细可视化的样本 ---
    ranks_to_plot = [7, 8, 9]
    print(f"\n已选择排名为 {ranks_to_plot} 的样本进行详细可视化。")

    # --- 4. 准备详细绘图所需的数据 ---
    selected_data_for_plot = []
    highlight_indices_for_scatter = []
    for rank in ranks_to_plot:
        error, original_idx = errors[rank - 1]
        highlight_indices_for_scatter.append(original_indices_sorted.index(original_idx))

        info = {'rank': rank, 'idx': original_idx, 'error': error}
        data_tuple = (S_gen_all[original_idx], solved_temperatures[original_idx], info)
        selected_data_for_plot.append(data_tuple)

    ref_data_tuple = (S_ref_interior, T_ref_final)
    grid_params_dict = {'NX': NX, 'NY': NY}

    # --- 5. 生成并保存详细的探索图 ---
    print("\n正在生成详细的探索对比图表...")
    create_exploration_plot(
        ref_data=ref_data_tuple,
        sample1_data=selected_data_for_plot[0],
        sample2_data=selected_data_for_plot[1],
        sample3_data=selected_data_for_plot[2],
        constraint_params=constraint,
        grid_params=grid_params_dict
    )
    output_filename_detail = os.path.join(OUTPUT_DIR, "design_exploration_detail_plot.png")
    plt.savefig(output_filename_detail, dpi=300, bbox_inches='tight')
    print(f"详细图表已保存至: '{output_filename_detail}'")
    plt.close('all')

    # --- 6. 生成并保存多目标性能分析图 ---
    print("\n正在生成多目标性能分析图表...")
    create_multiobjective_plot(
        all_errors=errors,
        all_secondary_metrics=secondary_metrics_sorted,
        highlighted_indices=highlight_indices_for_scatter
    )
    output_filename_multi = os.path.join(OUTPUT_DIR, "multiobjective_performance_plot.png")
    plt.savefig(output_filename_multi, dpi=300, bbox_inches='tight')
    print(f"多目标分析图表已保存至: '{output_filename_multi}'")
    plt.close('all')

    print("\n所有任务已完成。")
