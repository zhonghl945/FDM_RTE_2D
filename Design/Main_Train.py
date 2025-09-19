import torch
import numpy as np
import h5py
import warnings
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from Diffusion import GaussianDiffusion3D, Unet3D
from Trainer import Trainer

# 忽略所有警告
warnings.filterwarnings("ignore")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")


class H5LazyDataset(Dataset):
    """
    一个用于从HDF5文件进行懒加载的Dataset类。
    它在初始化时不加载任何数据到内存，只在__getitem__被调用时才读取单个样本。
    """

    def __init__(self, file_path, mode, rescaler):
        self.file_path = file_path
        self.mode = mode
        self.rescaler = rescaler

        # 为了获取数据集的长度，我们需要打开文件一次
        # 这个操作非常快，不会加载数据
        with h5py.File(self.file_path, 'r') as file:
            self._data_len = len(file[self.mode]['T'])

    def __len__(self):
        return self._data_len

    def __getitem__(self, idx):
        # 每次只为单个索引打开文件和读取数据
        # 这是懒加载的核心
        with h5py.File(self.file_path, 'r') as file:
            group = file[self.mode]

            # 读取单个样本
            t_sample = group['T'][idx]
            s_sample = group['S'][idx]

        # 将Numpy数组转换为PyTorch张量
        t_sample = torch.tensor(t_sample, dtype=torch.float32)
        s_sample = torch.tensor(s_sample, dtype=torch.float32)

        # 应用归一化
        t_sample = 2 * (t_sample / self.rescaler) - 1
        s_sample = 2 * (s_sample / self.rescaler) - 1

        return t_sample, s_sample


def get_diffusion_setup(T_shape_tuple, config, rescaler):
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
        T_shape=T_shape_tuple,  # (Nt, H, W)
        timesteps=config['timesteps'],
        beta_schedule=config['beta_schedule'],
        lambda_S_x0=config['lambda_S_x0'],
        freeze_indices_T=config['freeze_indices_T'],
        rescaler=rescaler,
    ).to(device)

    return diffusion_model


def run_3d_diffusion_training(config):
    # 1. 定义数据文件路径和模式
    h5_file_path = './data/2D_Diff_Python.h5'
    data_mode = 'Train'

    # 2. 通过分块遍历，高效计算全局元数据（rescaler 和 shape）
    global_max_abs_val = torch.tensor(0.0, dtype=torch.float32)
    chunk_size = 50  # 每次处理50个样本，可以根据需要调整

    with h5py.File(h5_file_path, 'r') as file:
        group_train = file[data_mode]
        t_dataset_h5 = group_train['T']  # 这是一个h5py对象，不是Tensor
        s_dataset_h5 = group_train['S']
        N, Nt, H, W = t_dataset_h5.shape
        T_shape_no_batch = (Nt, H, W)

        # 直接在HDF5对象上分块读取，不会消耗大量内存
        for i in tqdm(range(0, N, chunk_size), desc="Scanning T for max value"):
            t_chunk = torch.tensor(t_dataset_h5[i:i + chunk_size], dtype=torch.float32)
            s_chunk = torch.tensor(s_dataset_h5[i:i + chunk_size], dtype=torch.float32)

            current_max = torch.max(t_chunk.abs().max(), s_chunk.abs().max())
            if current_max > global_max_abs_val:
                global_max_abs_val = current_max

    rescaler = global_max_abs_val
    print(f"Global rescaler value calculated: {rescaler.item()}")
    print(f"T_Chunk shape (N, Nt, H, W): {t_chunk.shape}")
    print(f"S_Chunk shape (N, H, W): {s_chunk.shape}")

    # 3. 创建懒加载的数据集实例
    lazy_dataset = H5LazyDataset(file_path=h5_file_path,
                                 mode=data_mode,
                                 rescaler=rescaler)
    print("Lazy HDF5 dataset created successfully.")

    # 4. 获取配置好的 Diffusion 模型
    diffusion_model = get_diffusion_setup(T_shape_tuple=T_shape_no_batch,
                                          config=config,
                                          rescaler=rescaler)

    # 5. 实例化 Trainer
    trainer = Trainer(
        diffusion_model=diffusion_model,
        dataset=lazy_dataset,
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
        'lambda_S_x0': 1.0,  # Weight for S prediction loss
        'freeze_indices_T': (slice(-1, None), slice(25, 39), slice(25, 39)),  # (Nt_T, H, W)

        # Training Config
        'batch_size': 8,
        'train_steps': 200000,
        'learning_rate': 1e-4,
        'num_warmup_steps': 5000,
        'gradient_accumulation_steps': 1,
        'save_every': 10000,
        'record_every': 100,
        'use_fp16': 'fp16' if torch.cuda.is_available() else 'no',
        'load_milestone': None
    }

    # Run Training
    run_3d_diffusion_training(config)
