import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import warnings
import os

from Data_generation import Sn, generate_time_steps
from Diffusion import Unet3D, GaussianDiffusion3D
from Main_Train import get_dataset
from Trainer import Trainer

# 忽略警告
warnings.filterwarnings("ignore")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
os.makedirs('./fig', exist_ok=True)


def BC_plot(true_data, pre_data, time_indices, batch_indices=None):
    # 设置默认显示前3个样本
    if batch_indices is None:
        batch_indices = range(min(3, true_data.shape[0]))

    # 3个样本为一组可视化
    for b in range(len(batch_indices) // 3):
        plt.figure(figsize=(8, 6))
        plt.plot(time_indices, pre_data[0, :], label='$S_1(t)$_ $gen$', linestyle='--',
                 color='red', linewidth=2.5)
        plt.plot(time_indices, pre_data[1, :], label='$S_2(t)$_ $gen$', linestyle='--',
                 color='black', linewidth=2.5)
        plt.plot(time_indices, pre_data[2, :], label='$S_3(t)$_ $gen$', linestyle='--',
                 color='orange', linewidth=2.5)
        plt.plot(time_indices, true_data[0, :], label='$S_1(t)$_ $ref$', color='red', linewidth=2.5)
        plt.plot(time_indices, true_data[1, :], label='$S_2(t)$_ $ref$', color='black', linewidth=2.5)
        plt.plot(time_indices, true_data[2, :], label='$S_3(t)$_ $ref$', color='orange', linewidth=2.5)

        plt.legend()
        plt.xlabel(r'$t$', fontsize=20)
        plt.ylabel(r'$S(t)$', fontsize=20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig(f"./fig/Prediction_{3 * b:03d}to{3 * b + 2:03d}.png", format='png', dpi=300)


def rho_plot(rho_true, rho_sol, time_indices, freeze_indices_rho,
             Lx=0.2, Ly=0.4, num=5, batch_index=0):
    rho_sample = rho_true[batch_index]
    rho_sol_sample = rho_sol[batch_index]

    Nt, Nx, Ny = rho_sample.shape
    L1_rel_error = np.abs(rho_sample - rho_sol_sample) / np.abs(rho_sample)

    time_indices_to_plot_idx = np.linspace(0, Nt - 1, num, dtype=int)[1:]
    physical_times = time_indices[time_indices_to_plot_idx]

    vmin_rho = min(rho_sample.min(), rho_sol_sample.min())
    vmax_rho = max(rho_sample.max(), rho_sol_sample.max())

    # --- 标记约束区域 (索引坐标) ---
    x_slice = freeze_indices_rho[1]
    y_slice = freeze_indices_rho[2]
    x_start_idx = x_slice.start if x_slice.start is not None else 0
    x_end_idx = x_slice.stop if x_slice.stop is not None else Nx
    y_start_idx = y_slice.start if y_slice.start is not None else 0
    y_end_idx = y_slice.stop if y_slice.stop is not None else Ny

    # --- 核心修正：将索引坐标转换为物理坐标 ---
    dx = Lx / Nx
    dy = Ly / Ny
    rect_x_phys = x_start_idx * dx
    rect_y_phys = y_start_idx * dy
    rect_width_phys = (x_end_idx - x_start_idx) * dx
    rect_height_phys = (y_end_idx - y_start_idx + 1) * dy

    for i, t_idx in enumerate(time_indices_to_plot_idx):
        t_val = physical_times[i]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

        subplot_titles = [rf'True $\rho$ (t={t_val:.3f})', rf'Solved $\rho$ (t={t_val:.3f})',
                          rf'Relative pointwise absolute error (t={t_val:.3f})']
        data_to_plot = [rho_sample[t_idx].T, rho_sol_sample[t_idx].T, L1_rel_error[t_idx].T]
        cmaps = ['viridis', 'viridis', 'magma']

        current_error_data = L1_rel_error[t_idx]
        local_vmax_error = np.max(current_error_data[np.isfinite(current_error_data)])
        vmins = [vmin_rho, vmin_rho, 0]
        vmaxs = [vmax_rho, vmax_rho, local_vmax_error]

        images = []
        for ax_idx, ax in enumerate(axes):
            # 使用 extent 将图像映射到物理坐标系
            im = ax.imshow(data_to_plot[ax_idx], cmap=cmaps[ax_idx], origin='lower', vmin=vmins[ax_idx],
                           vmax=vmaxs[ax_idx], extent=[0, Lx, 0, Ly], interpolation='bilinear')
            images.append(im)
            ax.set_title(subplot_titles[ax_idx])
            ax.set_xlabel(r'$x$', fontsize=15)
            ax.set_ylabel(r'$y$', fontsize=15)
            ax.tick_params(axis='x', labelsize=12)
            ax.tick_params(axis='y', labelsize=12)

            # 使用转换后的物理坐标来创建矩形
            rect = patches.Rectangle((rect_x_phys, rect_y_phys), rect_width_phys, rect_height_phys,
                                     linewidth=2.5, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

        fig.colorbar(images[0], ax=axes[0])
        fig.colorbar(images[1], ax=axes[1])
        fig.colorbar(images[2], ax=axes[2])

        save_path = f'./fig/rho_batch{batch_index}_time_idx_{t_idx}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)


def cosine_beta_J_schedule(t, s=0.008):
    timesteps = 1000
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)[t]


def load_3dconv_model(rho, BC, config):
    N, Nt_rho, H, W = rho.shape
    Nt_bc = BC.shape[1]
    assert Nt_rho == Nt_bc
    Nt = Nt_rho
    print(f"rho shape (N, Nt, H, W): {rho.shape}")
    print(f"BC shape (N, Nt): {BC.shape}")

    # 1. Configure Unet3D
    unet = Unet3D(
        dim=config['dim'],
        dim_mults=config['dim_mults'],
        time_emb_dim_input=config['time_emb_dim'],
        resnet_block_groups=config['resnet_groups'],
        padding_mode=config['padding_mode']
    ).to(device)

    # 2. Configure GaussianDiffusion3D
    diffusion_model = GaussianDiffusion3D(
        model=unet,
        rho_shape=(Nt, H, W),
        timesteps=config['timesteps'],
        beta_schedule=config['beta_schedule'],
        freeze_indices_rho=config['freeze_indices_rho'],
    ).to(device)

    trainer = Trainer(
        diffusion_model=diffusion_model,
        rho=rho,
        BC=BC,
        device=device,
        use_fp16=config['use_fp16']
    )

    trainer.load(config['load_milestone'])

    return diffusion_model


def diffuse_3dconv(rho, BC, rescaler, config):
    ddpm = load_3dconv_model(rho=rho, BC=BC, config=config)
    ddpm.eval()
    ddpm.to(device)

    rho_pre, BC_pre = ddpm.sample(batch_size=rho.shape[0], x_f_known_rho_orig=rho, rescaler=rescaler)
    rho_pre = (rho_pre + 1) * rescaler / 2
    BC_pre = (BC_pre + 1) * rescaler / 2

    BC_pre = BC_pre.cpu().numpy()
    Sn(BC=BC_pre, filename='Test_Results', Num=BC_pre.shape[0], T_initial=1e-6, BatchSize=BC_pre.shape[0])
    rho_sol, _, _ = get_dataset(mode='Results')

    rho = rho.cpu().numpy()
    BC = BC.cpu().numpy()
    rho_pre = rho_pre.cpu().numpy()
    rho_sol = rho_sol.cpu().numpy()

    "Test set"
    rho_constrained = rho[(slice(None),) + config['freeze_indices_rho']]
    rho_pre_constrained = rho_pre[(slice(None),) + config['freeze_indices_rho']]
    rho_sol_constrained = rho_sol[(slice(None),) + config['freeze_indices_rho']]

    # L2 norm for generated
    BC_L2_norm = np.sum(BC_pre ** 2, axis=-1) / BC_pre.shape[-1]
    print(f'Testing set, BC: Generated L2 norm: {BC_L2_norm}')
    print(f'Testing set, BC: Batch mean: {BC_L2_norm.mean()}\n')

    # L2 (relative) error between generated and true
    ddpm_error = np.sqrt(np.mean((rho_constrained - rho_pre_constrained) ** 2, axis=(-1, -2, -3)))
    ddpm_relative_error = ddpm_error / np.sqrt(np.mean(rho_constrained ** 2, axis=(-1, -2, -3)))
    print('ddpm error (between generated and true)')
    print(f'Testing set, rho: L2 error: {ddpm_error}')
    print(f'Testing set, rho: Batch mean: {ddpm_error.mean()}')
    print(f'Testing set, rho: L2 relative error: {ddpm_relative_error}')
    print(f'Testing set, rho: Batch mean: {ddpm_relative_error.mean()}\n')

    # L2 (relative) error between solved and true
    J_error = np.sqrt(np.mean((rho_constrained - rho_sol_constrained) ** 2, axis=(-1, -2, -3)))
    J_relative_error = J_error / np.sqrt(np.mean(rho_constrained ** 2, axis=(-1, -2, -3)))
    print('J error (between solved and true)')
    print(f'Testing set, rho: L2 error: {J_error}')
    print(f'Testing set, rho: Batch mean: {J_error.mean()}')
    print(f'Testing set, rho: L2 relative error: {J_relative_error}')
    print(f'Testing set, rho: Batch mean: {J_relative_error.mean()}')

    # PLOT
    _, t_moment, idx_end_front = generate_time_steps()
    num_samples_front = np.ceil(config['num_samples_overall'] * config['front_sample_proportion']).astype(int)
    num_samples_rear = config['num_samples_overall'] - num_samples_front
    # 从 t_moment[1:] 的选择索引
    indices_front = np.round(np.linspace(0, idx_end_front - 1, num_samples_front)).astype(int)
    sampled_indices_front_list = list(np.unique(indices_front))
    indices_rear = np.round(np.linspace(idx_end_front, len(t_moment) - 2, num_samples_rear)).astype(int)
    sampled_indices_rear_list = list(np.unique(indices_rear))
    # 合并并排序索引
    sampled_indices = np.sort(np.unique(np.array(sampled_indices_front_list + sampled_indices_rear_list, dtype=int)))
    if len(sampled_indices) != config['num_samples_overall']:
        print(
            f"错误: 重建的采样点索引数量 ({len(sampled_indices)})与期望的BC序列长度 ({config['num_samples_overall']}) 不符。\n"
        )
    # 获取这些索引对应的时间点
    t_values_for_sampled = t_moment[1:][sampled_indices]

    BC_plot(true_data=BC,
            pre_data=BC_pre,
            time_indices=t_values_for_sampled,
            batch_indices=None)

    for b_idx in range(min(rho.shape[0], 3)):
        rho_plot(rho_true=rho,
                 rho_sol=rho_sol,
                 time_indices=t_values_for_sampled,
                 freeze_indices_rho=config['freeze_indices_rho'],
                 num=6,
                 batch_index=b_idx)


def get_test_data(test_batch):
    _, _, training_rescaler = get_dataset(mode='Train')

    rho, BC, _ = get_dataset(mode='Test')
    batch_size = test_batch
    batch_indices = torch.randperm(rho.size(0))[:batch_size]
    rho_test_batch = rho[batch_indices, ...].to(device)
    BC_test_batch = BC[batch_indices, ...].to(device)
    training_rescaler = training_rescaler.to(device)

    return rho_test_batch, BC_test_batch, training_rescaler


if __name__ == '__main__':
    # Set seeds
    torch.manual_seed(88)
    torch.cuda.manual_seed_all(88)
    np.random.seed(99)

    config = {
        # time sampling
        'front_sample_proportion': 0.1,
        'num_samples_overall': 64,

        # Model Config (U-Net)
        'dim': 64,
        'dim_mults': (1, 2, 4),
        'time_emb_dim': 256,
        'resnet_groups': 8,
        'padding_mode': 'reflect',  # 'zeros', 'reflect', 'replicate', 'circular'

        # Model Config (Diffusion)
        'timesteps': 1000,
        'beta_schedule': 'cosine',  # 'linear' or 'cosine'
        'freeze_indices_rho': (slice(None), slice(10, 21), slice(0, 63)),  # (Nt_rho, H, W)

        # Training Config
        'use_fp16': 'no',
        'load_milestone': 10
    }

    # load batch test data and training rescaler
    rho_test_batch, BC_test_batch, training_rescaler = get_test_data(test_batch=3)

    # Testing
    diffuse_3dconv(
        rho=rho_test_batch,
        BC=BC_test_batch,
        rescaler=training_rescaler,
        config=config
    )
