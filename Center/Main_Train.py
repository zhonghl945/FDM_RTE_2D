import torch
import numpy as np
import h5py
import warnings

from Diffusion import GaussianDiffusion3D, Unet3D
from Trainer import Trainer

# 忽略所有警告
warnings.filterwarnings("ignore")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")


def get_dataset(mode):
    if mode == 'Train':
        with h5py.File('./data/2D_Sn_Python.h5', 'r') as file:
            group_train = file['Train']
            rho = group_train['rho'][:]
            BC = group_train['BC'][:]

        rho = torch.tensor(rho, dtype=torch.float32)
        BC = torch.tensor(BC, dtype=torch.float32)

        print(f"归一化前 训练rho: 范围[{rho.min()}, {rho.max()}], 均值[{rho.mean()}], 方差[{rho.std()}]")
        print(f"归一化前 训练BC: 范围[{BC.min()}, {BC.max()}], 均值[{BC.mean()}], 方差[{BC.std()}]")

        rescaler = torch.max(rho.abs().max(), BC.abs().max())
        rho = 2 * (rho / rescaler) - 1
        BC = 2 * (BC / rescaler) - 1

    elif mode == 'Test':
        with h5py.File('./data/Test_Data.h5', 'r') as file:
            group_train = file['Test']
            rho = group_train['rho'][:]
            BC = group_train['BC'][:]

        rho = torch.tensor(rho, dtype=torch.float32)
        BC = torch.tensor(BC, dtype=torch.float32)
        rescaler = torch.max(rho.abs().max(), BC.abs().max())

    elif mode == 'Results':
        with h5py.File('./data/Test_Results.h5', 'r') as file:
            group_train = file['Test']
            rho = group_train['rho'][:]
            BC = group_train['BC'][:]

        rho = torch.tensor(rho, dtype=torch.float32)
        BC = torch.tensor(BC, dtype=torch.float32)
        rescaler = torch.max(rho.abs().max(), BC.abs().max())

    else:
        raise ValueError('Bad data mode')

    return rho, BC, rescaler


def get_diffusion_setup(rho_shape_tuple, config, rescaler):
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
        rho_shape=rho_shape_tuple,  # (Nt, H, W)
        timesteps=config['timesteps'],
        beta_schedule=config['beta_schedule'],
        lambda_bc_x0=config['lambda_bc_x0'],
        freeze_indices_rho=config['freeze_indices_rho'],
        rescaler=rescaler,
    ).to(device)

    return diffusion_model


def run_3d_diffusion_training(config):
    # 1. Load and Normalize Data
    rho, BC, rescaler = get_dataset(mode='Train')

    # 2. Get Data Shape Info
    assert rho.shape[1] == BC.shape[1]
    N, Nt, H, W = rho.shape
    rho_shape_no_batch = (Nt, H, W)
    print(f"rho shape (N, Nt, H, W): {rho.shape}")
    print(f"BC shape (N, Nt): {BC.shape}")

    # 3. Get Configured Diffusion Model
    diffusion_model = get_diffusion_setup(rho_shape_tuple=rho_shape_no_batch, config=config, rescaler=rescaler)

    # 4. Instantiate Trainer
    trainer = Trainer(
        diffusion_model=diffusion_model,
        rho=rho,
        BC=BC,
        device=device,
        train_batch_size=config['batch_size'],
        train_num_steps=config['train_steps'],
        save_and_sample_every=config['save_every'],
        record_step=config['record_every'],
        train_lr=config['learning_rate'],
        num_warmup_steps=config['num_warmup_steps'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        use_fp16=config['use_fp16'],
    )

    # 5. (Optional) Load Checkpoint
    if config['load_milestone'] is not None:
        try:
            milestone_num = int(config['load_milestone'])
            print(f"\nLoading model milestone: {milestone_num}")
            trainer.load(milestone_num)  # Assuming Trainer.load takes integer step/milestone
        except FileNotFoundError:
            print(f"Milestone {config['load_milestone']} not found. Starting from scratch.")
        except ValueError:
            print(f"Invalid milestone format: {config['load_milestone']}. Should be an integer. Starting from scratch.")
        except Exception as e:
            print(f"Error loading milestone {config['load_milestone']}: {e}. Starting from scratch.")

    # 6. Start Training
    print("\n--- Starting Training ---")
    trainer.train()
    print("--- Training Finished ---")


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    np.random.seed(99)

    config = {

        # Model Config (U-Net)
        'dim': 64,
        'dim_mults': (1, 2, 4),
        'time_emb_dim': 256,
        'resnet_groups': 8,
        'padding_mode': 'reflect',  # 'zeros', 'reflect', 'replicate', 'circular'

        # Model Config (Diffusion)
        'timesteps': 1000,
        'beta_schedule': 'cosine',  # 'linear' or 'cosine'
        'lambda_bc_x0': 1.0,  # Weight for bc prediction loss
        'freeze_indices_rho': (slice(None), slice(10, 21), slice(0, 63)),  # (Nt_rho, H, W)

        # Training Config
        'batch_size': 8,
        'train_steps': 100000,
        'learning_rate': 1e-4,
        'num_warmup_steps': 5000,
        'gradient_accumulation_steps': 1,
        'save_every': 10000,
        'record_every': 100,
        'use_fp16': 'fp16' if torch.cuda.is_available() else 'no',
        'load_milestone': 10
    }

    # Run Training
    run_3d_diffusion_training(config)
