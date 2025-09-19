import torch
import datetime
import matplotlib.pyplot as plt
import math

import torch.optim.lr_scheduler as lr_scheduler
from torch.optim import Adam

from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from ema_pytorch import EMA
from tqdm.auto import tqdm
from tensorboardX import SummaryWriter

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def exists(x):
    return x is not None


def cycle(dl):
    while True:
        for data in dl:
            yield data


def make_dir(filename):
    import os
    import errno
    if not os.path.exists(os.path.dirname(filename)):
        print("directory {0} does not exist, created.".format(os.path.dirname(filename)))
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                print(exc)
            raise


class Trainer(object):
    def __init__(
            self,
            diffusion_model,
            dataset,
            train_batch_size=16,
            train_lr=1e-4,
            train_num_steps=100000,
            gradient_accumulation_steps=1,
            ema_update_every=10,
            ema_decay=0.995,
            adam_betas=(0.9, 0.999),
            num_warmup_steps=5000,
            save_and_sample_every=10000,
            results_folder='./results',
            max_grad_norm=1.,
            record_step=1,
            use_fp16='fp16'
    ):
        super().__init__()

        self.device = device
        self.accelerator = Accelerator(
            split_batches=True,
            mixed_precision=use_fp16,
            gradient_accumulation_steps=gradient_accumulation_steps
        )
        self.model = diffusion_model
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.train_num_steps = train_num_steps

        self.save_and_sample_every = save_and_sample_every
        self.batch_size = train_batch_size
        self.max_grad_norm = max_grad_norm
        self.record_step = record_step

        dl = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=2)
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        self.train_lr = train_lr
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)
        self.num_warmup_steps = num_warmup_steps

        self.scheduler = lr_scheduler.LambdaLR(self.opt, self.lr_lambda)
        # self.scheduler = CosineAnnealingLR(self.opt, T_max=self.train_num_steps, eta_min=0)

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)
        else:
            self.ema = None

        self.results_folder = Path(results_folder)
        if self.accelerator.is_main_process:
            make_dir(str(results_folder) + '/')
        self.results_folder.mkdir(exist_ok=True)

        self.step = 0

        self.model, self.opt, self.scheduler = self.accelerator.prepare(self.model, self.opt, self.scheduler)

    def lr_lambda(self, current_step: int):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        else:
            progress = float(current_step - self.num_warmup_steps) / float(
                max(1, self.train_num_steps - self.num_warmup_steps))
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            final_lr_ratio = 0. / self.train_lr  # Assuming eta_min is 0
            factor = final_lr_ratio + (1.0 - final_lr_ratio) * cosine_decay
            return factor

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return
        model_state_dict = self.accelerator.get_state_dict(self.model)

        data = {
            'step': self.step,
            'model': model_state_dict,
            'opt': self.opt.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'ema': self.ema.state_dict() if self.ema is not None else None,
            'scaler': self.accelerator.scaler.state_dict() if self.accelerator.scaler is not None else None,
            'loss': getattr(self, 'total_loss', None),
        }
        save_path = str(self.results_folder / f'model-{milestone}.pt')
        self.accelerator.save(data, save_path)
        print(f"Milestone {milestone} saved to {save_path}")

    def load(self, milestone):
        device = self.accelerator.device
        if isinstance(milestone, int):
            load_path = str(self.results_folder / f'model-{milestone}.pt')
        else:
            load_path = str(self.results_folder / milestone)

        print(f"Loading checkpoint from {load_path}")
        try:
            data = torch.load(load_path, map_location=device)

            print('Loaded loss: ', data.get('loss', 'N/A'))  # Use .get for safety
            model_to_load = self.accelerator.unwrap_model(self.model)
            model_to_load.load_state_dict(data['model'])
            print("Model state loaded.")

            self.step = data['step']
            self.step_initial = self.step  # 用于 loss 记录的起始点
            print(f"Resuming from optimizer step: {self.step}")

            self.opt.load_state_dict(data['opt'])
            print("Optimizer state loaded.")

            # 加载 scheduler 状态 (如果已保存)
            if 'scheduler' in data:
                self.scheduler.load_state_dict(data['scheduler'])
                print("Scheduler state loaded.")

            # 加载 EMA 状态 (仅主进程)
            if self.accelerator.is_main_process and self.ema is not None and 'ema' in data and data['ema'] is not None:
                self.ema.load_state_dict(data["ema"])
                print("EMA state loaded.")

            # 加载混合精度 scaler 状态
            if self.accelerator.scaler is not None and 'scaler' in data and data['scaler'] is not None:
                self.accelerator.scaler.load_state_dict(data['scaler'])
                print("Scaler state loaded.")

            if 'version' in data:
                print(f"Checkpoint version: {data['version']}")

        except FileNotFoundError:
            print(f"Checkpoint file not found at {load_path}. Starting from scratch.")
            self.step = 0  # 确保从 0 开始
            self.step_initial = 0
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
            self.step = 0  # 确保从 0 开始
            self.step_initial = 0

    def train(self):
        print(f"Using device: {self.accelerator.device}")
        logdir = Path(self.results_folder) / 'tensorboard_logs' / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        writer = None
        if self.accelerator.is_main_process:
            make_dir(str(logdir) + '/')
            writer = SummaryWriter(logdir=str(logdir))
            print(f"TensorBoard logs will be saved to: {logdir}")

        accelerator = self.accelerator
        loss_record = []
        if not hasattr(self, 'step_initial'):
            self.step_initial = 0

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                with accelerator.accumulate(self.model):
                    batch_data = next(self.dl)
                    loss_components = self.model(batch_data)
                    total_loss = loss_components['total_loss']
                    accelerator.backward(total_loss)

                    # 日志记录更新
                    if accelerator.is_main_process:
                        # 使用 .get() 安全地获取字典中的值，如果键不存在则返回0.0
                        total_loss_val = total_loss.item()
                        l_bc_x0_val = loss_components.get('S_x0_loss', torch.tensor(0.0)).item()

                        # 更新进度条，显示关键信息
                        pbar.set_description(
                            f'loss: {total_loss_val:.3e} | '  # 使用科学计数法显示，因为值可能很小
                            f'L_BC_x0: {l_bc_x0_val:.3e} | '
                        )

                    if accelerator.sync_gradients:
                        if self.max_grad_norm is not None:
                            accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                        self.opt.step()
                        self.scheduler.step()
                        self.opt.zero_grad()

                        if accelerator.is_main_process:
                            if (self.step + 1 - self.step_initial) % self.record_step == 0:
                                loss_record.append(total_loss.item())

                            # --- TensorBoard 日志更新 ---
                            if writer is not None:
                                # 记录所有重要的标量
                                writer.add_scalar('loss/total', total_loss.item(), self.step)
                                writer.add_scalar('learning_rate', self.opt.param_groups[0]['lr'], self.step)
                            # ---------------------------

                            if self.ema is not None:
                                self.ema.update()

                            if (self.step + 1) % self.save_and_sample_every == 0:
                                self.total_loss = total_loss.item()
                                milestone = (self.step + 1) // self.save_and_sample_every
                                self.save(milestone)

                        self.step += 1
                        pbar.update(1)

            # --- 训练结束 ---
            print("\nTraining finished. Generating loss plot...")
            if accelerator.is_main_process and writer is not None:
                record_len = len(loss_record)
                if record_len > 0:
                    # 计算绘图对应的步数 (optimizer steps)
                    interval_indices = range(
                        self.step_initial + self.record_step,  # 起始步数
                        self.step_initial + record_len * self.record_step + 1,  # 结束步数
                        self.record_step  # 间隔
                    )
                    # 确保长度匹配
                    interval_indices = list(interval_indices)[:record_len]

                    plt.figure(figsize=(12, 6))
                    plt.yscale('log')
                    plt.plot(interval_indices, loss_record, label='Total Loss (Avg per Opt Step)')

                    # 绘制特定点
                    points_to_plot = [0.6, 0.7, 0.8, 0.9, 1.0]
                    for p in points_to_plot:
                        idx = int(record_len * p) - 1
                        if 0 <= idx < record_len:
                            step_num = interval_indices[idx]
                            plt.scatter(step_num, loss_record[idx], color='red', zorder=5)

                    plt.title('Average Total Loss Over Optimizer Steps (Log Scale)')
                    plt.xlabel(f'Optimizer Step (recorded every {self.record_step} steps)')
                    plt.ylabel('Average Total Loss')
                    plt.legend()
                    plt.grid(True, which="both", ls="--", linewidth=0.5)
                    plot_save_path = str(self.results_folder / 'Total_loss_over_steps.png')
                    plt.savefig(plot_save_path)
                    print(f"Loss plot saved to {plot_save_path}")
                    plt.close()
                else:
                    print("No loss data recorded for plotting.")

                writer.close()

            accelerator.print('Training complete')
