import os
import numpy as np
import h5py
import numba
from tqdm import tqdm
from perlin_noise import PerlinNoise
import multiprocessing as mp
import time
import random


def scan_hdf5_for_anomalies(file_path, group_name, chunk_size):
    """
    高效地扫描HDF5文件中的数据集，查找NaN和Inf值。
    """
    if not os.path.exists(file_path):
        print(f"🛑 错误：找不到文件 '{file_path}'。请检查路径是否正确。")
        return

    print(f"--- 开始检查文件: {file_path} ---")
    print(f"--- 目标组: '{group_name}' ---")

    try:
        with h5py.File(file_path, 'r') as file:
            if group_name not in file:
                print(f"🛑 错误：在文件中找不到组 '{group_name}'。可用组: {list(file.keys())}")
                return

            t_dataset = file[group_name]['T']
            s_dataset = file[group_name]['S']
            num_samples = len(t_dataset)

            print(f"总共 {num_samples} 个样本，开始扫描...")

            # 检查 'T' 数据集
            print("\n--- 正在扫描 'T' 数据集 ---")
            for i in tqdm(range(0, num_samples, chunk_size), desc="扫描 T"):
                t_chunk = t_dataset[i: i + chunk_size]

                # 检查 NaN
                if np.isnan(t_chunk).any():
                    for j in range(len(t_chunk)):
                        if np.isnan(t_chunk[j]).any():
                            sample_idx = i + j
                            nan_count = np.isnan(t_chunk[j]).sum()
                            print(f"\n🛑 在 'T' 数据集的第 {sample_idx} 号样本中发现 {nan_count} 个 NaN 值！")
                            print("--- 检查终止 ---")
                            return

                # 检查 Inf
                if np.isinf(t_chunk).any():
                    for j in range(len(t_chunk)):
                        if np.isinf(t_chunk[j]).any():
                            sample_idx = i + j
                            inf_count = np.isinf(t_chunk[j]).sum()
                            print(f"\n🛑 在 'T' 数据集的第 {sample_idx} 号样本中发现 {inf_count} 个 Inf 值！")
                            print("--- 检查终止 ---")
                            return

            print("✅ 'T' 数据集检查完毕，未发现异常。")

            # 检查 'S' 数据集
            print("\n--- 正在扫描 'S' 数据集 ---")
            for i in tqdm(range(0, num_samples, chunk_size), desc="扫描 S"):
                s_chunk = s_dataset[i: i + chunk_size]

                # 检查 NaN
                if np.isnan(s_chunk).any():
                    for j in range(len(s_chunk)):
                        if np.isnan(s_chunk[j]).any():
                            sample_idx = i + j
                            nan_count = np.isnan(s_chunk[j]).sum()
                            print(f"\n🛑 在 'S' 数据集的第 {sample_idx} 号样本中发现 {nan_count} 个 NaN 值！")
                            print("--- 检查终止 ---")
                            return

                # 检查 Inf
                if np.isinf(s_chunk).any():
                    for j in range(len(s_chunk)):
                        if np.isinf(s_chunk[j]).any():
                            sample_idx = i + j
                            inf_count = np.isinf(s_chunk[j]).sum()
                            print(f"\n🛑 在 'S' 数据集的第 {sample_idx} 号样本中发现 {inf_count} 个 Inf 值！")
                            print("--- 检查终止 ---")
                            return

            print("✅ 'S' 数据集检查完毕，未发现异常。")

    except Exception as e:
        print(f"处理文件时发生错误: {e}")
        return

    print("\n🎉 恭喜！完整性检查通过，数据集中没有发现任何 NaN 或 Inf 值。")


# =============================================================================
# 1. 高性能求解器
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
        if sample_idx_counter < len(sample_indices) and n == sample_indices[sample_idx_counter]:
            for j in range(ny - 2):
                for i in range(nx - 2):
                    results_3d[sample_idx_counter, j, i] = T[j + 1, i + 1]
            sample_idx_counter += 1
    return T, results_3d


def solve_nonlinear_heat_explicit(nx, ny, nt, S_interior, n_time_samples):
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
    sample_indices = np.linspace(0, nt - 1, num=n_time_samples + 1, dtype=int)[1:]
    results_3d = np.zeros((n_time_samples, ny - 2, nx - 2), dtype=np.float64)
    _, results_3d = _time_loop_numba(T, S, mask_x_in_rect, mask_y_in_rect, results_3d, sample_indices, nt, nx, ny, dt,
                                     dx, dy, Cv, T_floor)
    return results_3d, sample_indices * dt


# =============================================================================
# 训练源项生成函数库
# =============================================================================
def generate_gaussian_base(X, Y, amp_range=(5, 15), center_margin=0.4):
    amp = np.random.uniform(*amp_range)
    cx = np.random.uniform(center_margin, 1 - center_margin)
    cy = np.random.uniform(center_margin, 1 - center_margin)
    return amp * np.exp(-(((X - cx) ** 2 + (Y - cy) ** 2) / 0.2))


def generate_sine_perturbation(X, Y, amp_range=(0, 5), freq_range=(7, 16)):
    amp = np.random.uniform(*amp_range)
    freq_x = np.random.uniform(*freq_range)
    freq_y = np.random.uniform(*freq_range)
    return amp * np.sin(freq_x * np.pi * X) * np.sin(freq_y * np.pi * Y)


def generate_checkerboard_perturbation(X, Y, amp_range=(1, 5), freq_range=(10, 15)):
    amp = np.random.uniform(*amp_range)
    freq = np.random.randint(*freq_range)
    return amp * (2 * ((np.floor(X * freq) + np.floor(Y * freq)) % 2) - 1)


def generate_perlin_perturbation(X, Y, amp_range=(2, 8), octaves_range=(4, 10)):
    amp = np.random.uniform(*amp_range)
    octaves = np.random.randint(*octaves_range)
    seed = np.random.randint(1, 1000)
    noise_gen = PerlinNoise(octaves=octaves, seed=seed)
    nx, ny = X.shape[1], X.shape[0]
    p_noise = np.array([[noise_gen([i / nx, j / ny]) for j in range(ny)] for i in range(nx)]).T
    return amp * p_noise


def generate_diagonal_wave_perturbation(X, Y, amp_range=(2, 5), freq_range=(7, 15)):
    amp = np.random.uniform(*amp_range)
    freq = np.random.uniform(*freq_range)
    direction = np.random.choice([-1, 1])
    return amp * np.sin(freq * np.pi * (X + direction * Y))


def generate_concentric_wave_perturbation(X, Y, amp_range=(1, 3), freq_range=(20, 30), center_margin=0.4):
    amp = np.random.uniform(*amp_range)
    freq = np.random.uniform(*freq_range)
    cx = np.random.uniform(center_margin, 1 - center_margin)
    cy = np.random.uniform(center_margin, 1 - center_margin)
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    return amp * np.sin(freq * R)


def generate_sparse_impulse_perturbation(X, Y, amp_range=(5, 10), sparsity_range=(0.005, 0.05)):
    amp = np.random.uniform(*amp_range)
    sparsity = np.random.uniform(*sparsity_range)
    ny, nx = X.shape
    delta_S = np.zeros_like(X)
    num_points = int(sparsity * nx * ny)
    rows = np.random.randint(0, ny, size=num_points)
    cols = np.random.randint(0, nx, size=num_points)
    values = amp * (2 * np.random.randint(0, 2, size=num_points) - 1)
    delta_S[rows, cols] = values
    return delta_S


# 全局变量，用于worker进程初始化 ---
# 将这些变量设为全局，以便worker进程可以访问它们
# 注意：这种方式适用于简单的参数传递，更复杂的情况可能需要不同的方法
g_nx, g_ny, g_nt, g_num_time_snapshots = 0, 0, 0, 0
g_X, g_Y = None, None
g_base_generators, g_perturbation_generators = [], []


def initialize_worker(nx, ny, nt, num_time_snapshots):
    """初始化worker进程的全局变量"""
    global g_nx, g_ny, g_nt, g_num_time_snapshots, g_X, g_Y
    global g_base_generators, g_perturbation_generators

    g_nx, g_ny, g_nt, g_num_time_snapshots = nx, ny, nt, num_time_snapshots

    Lx, Ly = 1.0, 1.0
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    g_X, g_Y = np.meshgrid(x, y)

    g_base_generators = [generate_gaussian_base]
    g_perturbation_generators = [
        generate_sine_perturbation,
        generate_checkerboard_perturbation,
        generate_perlin_perturbation,
        generate_diagonal_wave_perturbation,
        generate_concentric_wave_perturbation,
        generate_sparse_impulse_perturbation
    ]
    # 首次调用以编译numba函数
    solve_nonlinear_heat_explicit(2, 2, 1, np.zeros((0, 0)), 1)


def worker_generate_sample(sample_index):
    """
    【新增】这是单个工作进程要执行的任务：生成一个(S, T)样本。
    它会返回样本索引和计算结果。
    """
    # 为了保证每个进程的随机性不同，需要重新设置随机种子
    np.random.seed(os.getpid() + sample_index)

    # 从全局变量获取参数
    base_gen = np.random.choice(g_base_generators)
    S_base = base_gen(g_X, g_Y)

    if np.random.rand() > 0.3:
        pert_gen = np.random.choice(g_perturbation_generators)
        delta_S = pert_gen(g_X, g_Y)
    else:
        delta_S = np.zeros_like(g_X)

    S_full = S_base + delta_S
    S_interior = S_full[1:-1, 1:-1]

    T_interior_samples, _ = solve_nonlinear_heat_explicit(
        g_nx, g_ny, g_nt, S_interior, n_time_samples=g_num_time_snapshots
    )

    return sample_index, S_interior.astype('f4'), T_interior_samples.astype('f4')


# =============================================================================
# 并行化数据生成逻辑
# =============================================================================
def generate_dataset_parallel(num_samples, file_path, nx, ny, nt, num_time_snapshots, num_workers=4):
    """
    使用多进程并行生成数据集。
    """
    dir_name = os.path.dirname(file_path)
    if dir_name: os.makedirs(dir_name, exist_ok=True)

    print(f"将在 {num_workers} 个CPU核心上并行生成数据...")

    with h5py.File(file_path, 'w') as f:
        print(f"创建HDF5文件: {file_path}")
        group = f.create_group('Train')
        s_dataset = group.create_dataset('S', (num_samples, ny - 2, nx - 2), dtype='f4')
        t_dataset = group.create_dataset('T', (num_samples, num_time_snapshots, ny - 2, nx - 2), dtype='f4')

        # 使用进程池
        # initializer函数用于向每个worker进程传递只读的全局参数
        with mp.Pool(processes=num_workers, initializer=initialize_worker,
                     initargs=(nx, ny, nt, num_time_snapshots)) as pool:
            # 使用 imap_unordered 来分发任务，并用tqdm显示进度
            # imap_unordered 效率更高，因为它一有结果就返回，不保证顺序
            results_iterator = pool.imap_unordered(worker_generate_sample, range(num_samples))

            for index, s_result, t_result in tqdm(results_iterator, total=num_samples, desc="正在生成数据"):
                # 主进程负责将结果写入HDF5文件
                s_dataset[index, :, :] = s_result
                t_dataset[index, :, :, :] = t_result

    print(f"\n成功并行生成 {num_samples} 组数据并保存到 {file_path}")


# =============================================================================
# 测试数据生成函数
# =============================================================================
def generate_test_dataset(num_test, file_path, nx, ny, nt, num_time_snapshots):
    """
    生成并保存测试数据集。
    该函数会构造一个固定的源项，求解后，将结果复制num_test次并保存。
    """
    dir_name = os.path.dirname(file_path)
    if dir_name: os.makedirs(dir_name, exist_ok=True)

    # --- 1. 构造唯一的源项 S ---
    Lx, Ly = 1.0, 1.0
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    # 泛化内测试集
    # S_full_single = 10 * np.exp(-(((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.2))

    # 泛化外测试集
    # 定义十字的参数
    center_x, center_y = 0.5, 0.5  # 中心点
    arm_thickness = 0.2             # 十字臂的厚度
    arm_length = 0.8                # 十字臂的长度
    amplitude = 10.0                # 热源强度 (在训练数据范围内，但形状是新的)
    # 创建水平臂的掩码
    horizontal_mask = (np.abs(X - center_x) < arm_length / 2) & (np.abs(Y - center_y) < arm_thickness / 2)
    # 创建垂直臂的掩码
    vertical_mask = (np.abs(X - center_x) < arm_thickness / 2) & (np.abs(Y - center_y) < arm_length / 2)
    # 合并两个臂来创建十字形
    cross_mask = horizontal_mask | vertical_mask
    # 根据掩码生成源项 S
    S_full_single = np.where(cross_mask, amplitude, 0.0)

    S_interior_single = S_full_single[1:-1, 1:-1]

    # --- 2. 调用求解器获得对应的温度场 T ---
    print("--> 正在调用求解器为测试集计算温度场 (可能需要一些时间)...")
    T_samples_single, _ = solve_nonlinear_heat_explicit(
        nx=nx, ny=ny, nt=nt,
        S_interior=S_interior_single,
        n_time_samples=num_time_snapshots
    )

    # --- 3. 将S和T复制多次 ---
    print(f"--> 正在将S和T复制 {num_test} 份...")
    S_replicated = np.repeat(S_interior_single[np.newaxis, :, :], num_test, axis=0)
    T_replicated = np.repeat(T_samples_single[np.newaxis, :, :, :], num_test, axis=0)

    # --- 4. 保存到HDF5文件 ---
    with h5py.File(file_path, 'w') as f:
        test_group = f.create_group('Test')
        test_group.create_dataset('S', data=S_replicated.astype('f4'), dtype='f4')
        test_group.create_dataset('T', data=T_replicated.astype('f4'), dtype='f4')

    print(f"\n成功生成 {num_test} 组测试数据并保存到 {file_path}")


# =============================================================================
# 4. 程序入口
# =============================================================================
if __name__ == "__main__":
    # --- 一个稳健的锁获取函数 ---
    def acquire_lock(lock_file_path, timeout=60):
        """
        尝试在指定时间内获取一个文件锁。
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                lock_fd = os.open(lock_file_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return lock_fd
            except (FileExistsError, BlockingIOError):
                wait_time = random.uniform(0.5, 2.0)
                print(f"锁被占用或系统繁忙，等待 {wait_time:.2f} 秒后重试...")
                time.sleep(wait_time)
        return None


    # --- 通用参数设置 ---
    NX, NY, NT = 66, 66, 40000
    NUM_TIME_SNAPSHOTS = 16
    np.random.seed(99)

    # 强制限制底层库的线程数，避免并行冲突
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

    # # --- 生成训练集 ---
    # print("\n" + "=" * 50)
    # print("--- 开始执行：生成训练集 ---")
    # print("=" * 50)
    #
    # NUM_WORKERS = 60
    # NUM_TRAIN_SAMPLES = 50000
    # TRAIN_FILE = './data/2D_Diff_Python.h5'
    # LOCK_FILE_TRAIN = './data/train_generation.lock'
    #
    # os.makedirs(os.path.dirname(TRAIN_FILE), exist_ok=True)
    #
    # lock_fd_train = acquire_lock(LOCK_FILE_TRAIN)
    #
    # if lock_fd_train is not None:
    #     try:
    #         print("成功获取训练集文件锁，开始生成...")
    #         generate_dataset_parallel(
    #             num_samples=NUM_TRAIN_SAMPLES,
    #             file_path=TRAIN_FILE,
    #             nx=NX,
    #             ny=NY,
    #             nt=NT,
    #             num_time_snapshots=NUM_TIME_SNAPSHOTS,
    #             num_workers=NUM_WORKERS
    #         )
    #         print("\n--- 训练集生成完毕 ---")
    #         print("正在扫描训练集以确保数据完整性...")
    #         scan_hdf5_for_anomalies(TRAIN_FILE, 'Train', 50)
    #     finally:
    #         print("释放训练集文件锁...")
    #         os.close(lock_fd_train)
    #         os.remove(LOCK_FILE_TRAIN)
    # else:
    #     print("在超时时间内未能获取训练集锁，说明另一个进程可能正在长时间运行。跳过训练集生成。")

    # --- 生成测试集 ---
    print("\n" + "=" * 50)
    print("--- 开始执行：生成测试集 ---")
    print("=" * 50)

    NUM_TEST_SAMPLES = 50
    TEST_FILE = './data/Test_Data.h5'
    LOCK_FILE_TEST = './data/test_generation.lock'

    os.makedirs(os.path.dirname(TEST_FILE), exist_ok=True)

    lock_fd_test = acquire_lock(LOCK_FILE_TEST)

    if lock_fd_test is not None:
        try:
            print("成功获取测试集文件锁，开始生成...")
            generate_test_dataset(
                num_test=NUM_TEST_SAMPLES,
                file_path=TEST_FILE,
                nx=NX,
                ny=NY,
                nt=NT,
                num_time_snapshots=NUM_TIME_SNAPSHOTS
            )
            print("\n--- 测试集生成完毕 ---")
            print("正在扫描测试集以确保数据完整性...")
            scan_hdf5_for_anomalies(TEST_FILE, 'Test', 50)
        finally:
            print("释放测试集文件锁...")
            os.close(lock_fd_test)
            os.remove(LOCK_FILE_TEST)
    else:
        print("在超时时间内未能获取测试集锁，说明另一个进程可能正在长时间运行。跳过测试集生成。")

    print("\n所有任务已执行完毕。")
