import torch
import numpy as np
import h5py
import warnings
import os
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from Diffusion import GaussianDiffusion3D, Unet3D

# 忽略警告
warnings.filterwarnings("ignore")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")


class H5LazyDataset(Dataset):
    """
    用于从HDF5文件进行懒加载的Dataset类。
    它直接读取原始数据，不进行归一化。
    """

    def __init__(self, file_path, mode):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件未找到: {file_path}")
        self.file_path = file_path
        self.mode = mode

        with h5py.File(self.file_path, 'r') as file:
            if self.mode not in file:
                raise KeyError(
                    f"组 '{self.mode}' 在HDF5文件 '{self.file_path}' 中未找到。可用组: {list(file.keys())}")
            self._data_len = len(file[self.mode]['T'])

    def __len__(self):
        return self._data_len

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as file:
            group = file[self.mode]
            t_sample = group['T'][idx]
            s_sample = group['S'][idx]

        t_sample = torch.tensor(t_sample, dtype=torch.float32)
        s_sample = torch.tensor(s_sample, dtype=torch.float32)
        return t_sample, s_sample


def get_diffusion_setup(T_shape_tuple, config, rescaler):
    """
    配置并返回扩散模型。
    """
    unet = Unet3D(
        dim=config['dim'],
        dim_mults=config['dim_mults'],
        time_emb_dim_input=config['time_emb_dim'],
        resnet_block_groups=config['resnet_groups'],
        padding_mode=config['padding_mode']
    ).to(device)

    diffusion_model = GaussianDiffusion3D(
        model=unet,
        T_shape=T_shape_tuple,
        timesteps=config['timesteps'],
        beta_schedule=config['beta_schedule'],
        lambda_S_x0=config['lambda_S_x0'],
        freeze_indices_T=config['freeze_indices_T'],
        rescaler=rescaler,
    ).to(device)

    return diffusion_model


def get_data_and_rescaler(train_h5_path, test_h5_path, test_batch_size):
    """
    从训练数据计算rescaler，并加载一批原始（RAW）测试数据。
    """
    global_max_abs_val = torch.tensor(0.0, dtype=torch.float32)
    chunk_size = 100

    with h5py.File(train_h5_path, 'r') as file:
        group_train = file['Train']
        t_dataset_train = group_train['T']
        s_dataset_train = group_train['S']
        N_train = len(t_dataset_train)

        for i in tqdm(range(0, N_train, chunk_size), desc="扫描训练数据"):
            t_chunk = torch.from_numpy(t_dataset_train[i:i + chunk_size]).abs()
            s_chunk = torch.from_numpy(s_dataset_train[i:i + chunk_size]).abs()
            current_max = torch.max(t_chunk.max(), s_chunk.max())
            if current_max > global_max_abs_val:
                global_max_abs_val = current_max

    rescaler = global_max_abs_val
    print(f"从训练集计算得到的全局 rescaler 值为: {rescaler.item()}\n")

    test_dataset = H5LazyDataset(file_path=test_h5_path, mode='Test')
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    T_test_batch, S_test_batch = next(iter(test_dataloader))

    print(f"已加载一批原始测试数据 T，形状为: {T_test_batch.shape}")
    print(f"已加载一批原始测试数据 S，形状为: {S_test_batch.shape}")

    return T_test_batch.to(device), S_test_batch.to(device), rescaler.to(device)


def run_3d_diffusion_test(config, train_h5_path, test_h5_path, test_batch_size, results_folder):
    """
    运行测试流程的主函数。
    """
    # 1. 获取原始测试数据和rescaler
    T_test_batch, S_test_batch, rescaler = get_data_and_rescaler(train_h5_path, test_h5_path, test_batch_size)

    # 2. 配置模型，并将rescaler传入
    T_shape_no_batch = T_test_batch.shape[1:]
    diffusion_model = get_diffusion_setup(T_shape_tuple=T_shape_no_batch, config=config, rescaler=rescaler)

    # 3. 直接加载模型状态，不再依赖Trainer
    milestone = config['load_milestone']
    model_path = os.path.join(results_folder, f'model-{milestone}.pt')

    try:
        print(f"\n正在加载模型检查点: {model_path}")
        data = torch.load(model_path, map_location=device)
        # 直接加载模型的状态字典
        diffusion_model.load_state_dict(data['model'])
        print("模型权重加载成功。")
    except FileNotFoundError:
        print(f"🛑 错误: 模型文件 '{model_path}' 未找到。测试中止。")
        return
    except Exception as e:
        print(f"🛑 加载模型时出错: {e}。测试中止。")
        return

    # 4. 运行采样
    diffusion_model.eval()  # 切换到评估模式

    print("\n--- 开始采样过程 ---")
    # 运行模型生成预测结果
    T_pred_norm, S_pred_norm = diffusion_model.sample(
        batch_size=test_batch_size,
        x_f_known_T_orig=T_test_batch,  # 输入的是原始尺度的T
        rescaler=rescaler
    )
    print("--- 采样完成 ---\n")

    # 5. 手动反归一化，得到物理尺度的预测结果
    T_pred = (T_pred_norm.cpu() + 1) * rescaler.cpu() / 2
    S_pred = (S_pred_norm.cpu() + 1) * rescaler.cpu() / 2

    print(f"预测T的形状: {T_pred.shape}")
    print(f"预测S的形状: {S_pred.shape}")

    # 6. 保存预测结果
    output_filename = './data/Test_Gen.h5'
    if not os.path.exists('./data'):
        os.makedirs('./data')

    with h5py.File(output_filename, 'w') as f:
        group = f.create_group('Test')
        group.create_dataset('S', data=S_pred.numpy(), dtype='f4')
        group.create_dataset('T', data=T_pred.numpy(), dtype='f4')
    print(f"预测结果已成功保存到: {output_filename}")


if __name__ == "__main__":
    torch.manual_seed(13)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(13)
    np.random.seed(13)

    config = {
        # 模型参数，必须与训练时完全一致
        'dim': 64,
        'dim_mults': (1, 2, 4),
        'time_emb_dim': 256,
        'resnet_groups': 8,
        'padding_mode': 'reflect',

        # 扩散过程参数，必须与训练时完全一致
        'timesteps': 1000,
        'beta_schedule': 'cosine',
        'lambda_S_x0': 1.0,
        'freeze_indices_T': (slice(-1, None), slice(25, 39), slice(25, 39)),

        # 测试时用到的参数
        'use_fp16': 'no',  # 测试时通常不需要混合精度
        'load_milestone': 20  # 要加载的模型检查点编号
    }

    # 文件路径
    train_data_path = './data/2D_Diff_Python.h5'
    test_data_path = './data/Test_Data.h5'
    results_folder_path = './results'  # 训练结果（模型.pt文件）所在的文件夹

    # 设置要从测试集中取多少样本进行测试
    num_test_samples = 50

    run_3d_diffusion_test(config, train_data_path, test_data_path, num_test_samples, results_folder_path)