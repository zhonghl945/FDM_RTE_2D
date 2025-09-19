import torch
import math
import torch.nn.functional as F
from torch import nn
from functools import partial

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape_obj):  # x_shape_obj 应该是 torch.Size 对象, 例如 x_t.shape
    batch_size = t.shape[0]
    # target_device 的确定可以简化，通常从参与运算的张量获取
    # 假设 a, t, x_shape_obj 对应的张量都在同一device或能正确转换
    target_device = t.device  # 或者 a.device，或者从 x_shape_obj 关联的张量获取

    if not isinstance(a, torch.Tensor):
        a = torch.from_numpy(a)
    a = a.to(target_device)
    t = t.to(target_device)  # 确保 t 和 a 在相同设备

    out = a.gather(-1, t)  # out 的形状是 [batch_size]

    # 使用 x_shape_obj (torch.Size) 的长度 (即rank) 来确定 reshape 后的维度
    # len(x_shape_obj) 是 x_t 张量的维度数量
    # 例如，如果 x_t 是 [B, C, Nt, H, W]，len(x_shape_obj) 是 5
    # 那么 reshape 后的形状是 [B, 1, 1, 1, 1]
    return out.float().reshape(batch_size, *((1,) * (len(x_shape_obj) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, t):
        module_device = next(self.parameters()).device if list(self.parameters()) else \
            (t.device if isinstance(t, torch.Tensor) else torch.device('cpu'))  # Robust device detection
        t = t.to(module_device)
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=module_device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)  # Use broadcasting: t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# --- BCEncoder and BCDecoder are REMOVED ---

class Block3D(nn.Module):
    def __init__(self, dim, dim_out, groups=8, padding_mode='zeros'):  # Added padding_mode
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, 3, padding=1, padding_mode=padding_mode)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            scale = scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) if exists(scale) else 1.0
            shift = shift.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) if exists(shift) else 0.0
            x = x * (scale + 1) + shift  # Original used scale, common to use (scale+1)
        x = self.act(x)
        return x


class ResnetBlock3D(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8, padding_mode='zeros'):  # Added padding_mode
        super().__init__()
        self.mlp_time = None
        if exists(time_emb_dim):
            self.mlp_time = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, dim_out * 2)
            )

        self.block1 = Block3D(dim, dim_out, groups=groups, padding_mode=padding_mode)
        self.block2 = Block3D(dim_out, dim_out, groups=groups, padding_mode=padding_mode)
        # For 1x1 conv, padding_mode has no effect if padding=0
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp_time) and exists(time_emb):
            time_scale_shift = self.mlp_time(time_emb)
            scale_shift = time_scale_shift.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


def Upsample3D(dim, dim_out=None, padding_mode='zeros'):  # Added padding_mode
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
        nn.Conv3d(dim, default(dim_out, dim), 3, padding=1, padding_mode=padding_mode)
    )


def Downsample3D(dim, dim_out=None, padding_mode='zeros'):  # Added padding_mode
    # Using kernel 3, stride 2, padding 1 is common for halving spatial dimensions
    return nn.Conv3d(dim, default(dim_out, dim), kernel_size=3, stride=2, padding=1, padding_mode=padding_mode)


class Unet3D(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            dim_mults=(1, 2, 4, 8),
            input_bc_channels=1,
            time_emb_dim_input=None,
            time_emb_dim_internal=None,
            resnet_block_groups=8,
            padding_mode='zeros'  # Added padding_mode
    ):
        super().__init__()
        # Input channels = 1 (for rho) + input_bc_channels (for spatially expanded BC)
        self.in_channels = 1 + input_bc_channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv3d(self.in_channels, init_dim, 7, padding=3, padding_mode=padding_mode)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)

        if exists(time_emb_dim_input):
            time_emb_dim_internal = default(time_emb_dim_internal, dim * 4)
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(time_emb_dim_input),
                nn.Linear(time_emb_dim_input, time_emb_dim_internal),
                nn.GELU(),
                nn.Linear(time_emb_dim_internal, time_emb_dim_internal)
            )
            self.pos_emb_input_layer = self.time_mlp[0]
        else:
            time_emb_dim_internal = None
            self.time_mlp = None
            self.pos_emb_input_layer = None

        block_klass = partial(ResnetBlock3D, groups=resnet_block_groups, time_emb_dim=time_emb_dim_internal,
                              padding_mode=padding_mode)
        attn_klass = lambda _dim: nn.Identity()  # Attention not used

        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in),
                block_klass(dim_in, dim_in),
                attn_klass(dim_in),
                Downsample3D(dim_in, dim_out, padding_mode=padding_mode) if not is_last else \
                    nn.Conv3d(dim_in, dim_out, 3, padding=1, padding_mode=padding_mode)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim)
        self.mid_attn = attn_klass(mid_dim)
        self.mid_block2 = block_klass(mid_dim, mid_dim)

        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out),
                block_klass(dim_out + dim_in, dim_out),  # Adjusted based on typical U-Net skip connection sums
                attn_klass(dim_out),
                Upsample3D(dim_out, dim_in, padding_mode=padding_mode) if not is_last else \
                    nn.Conv3d(dim_out, dim_in, 3, padding=1, padding_mode=padding_mode)
            ]))

        self.final_res_block = block_klass(init_dim * 2, init_dim)

        # Head 1: Rho noise prediction (epsilon)
        self.final_conv_rho = nn.Conv3d(init_dim, 1, 1)

        # Head 2: Original BC (x0) prediction (shape [B, Nt_rho])
        # Takes features [B, init_dim, Nt_rho, H, W]
        # 1. Average pool H, W dimensions
        # 2. Use a 1D conv (acting as a linear layer per time step) to project channels
        self.spatial_avg_pool_bc = nn.AdaptiveAvgPool3d((None, 1, 1))  # Keep T dim, pool H, W
        self.final_projection_bc = nn.Conv1d(init_dim, 1,
                                             kernel_size=1)  # Project init_dim channels to 1 channel for BC

    def forward(self, x, time):
        # x: [B, 1+input_bc_channels, Nt, H, W]
        # time: [B]
        module_device = next(self.parameters()).device
        time = time.to(module_device)

        t_emb = None
        if exists(self.time_mlp):
            if exists(self.pos_emb_input_layer):
                time_pos_emb = self.pos_emb_input_layer(time)
                t_emb = self.time_mlp[1:](time_pos_emb)
            else:
                t_emb = self.time_mlp(time)

        x = self.init_conv(x)
        r = x.clone()
        h_skips = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, time_emb=t_emb)
            h_skips.append(x)
            x = block2(x, time_emb=t_emb)
            x = attn(x)
            h_skips.append(x)
            x = downsample(x)

        x = self.mid_block1(x, time_emb=t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, time_emb=t_emb)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h_skips.pop()), dim=1)
            x = block1(x, time_emb=t_emb)
            x = torch.cat((x, h_skips.pop()), dim=1)
            x = block2(x, time_emb=t_emb)
            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x_final_feat = self.final_res_block(x, time_emb=t_emb)  # [B, init_dim, Nt, H, W]

        # Rho head
        rho_eps_pred = self.final_conv_rho(x_final_feat)  # [B, 1, Nt, H, W]

        # BC head
        # x_final_feat is [B, init_dim, Nt, H, W]
        pooled_bc_features = self.spatial_avg_pool_bc(x_final_feat)  # [B, init_dim, Nt, 1, 1]
        pooled_bc_features_squeezed = pooled_bc_features.squeeze(-1).squeeze(-1)  # [B, init_dim, Nt]
        # Conv1d expects [B, Channels, Length], so current shape is correct
        bc_x0_pred_chan = self.final_projection_bc(pooled_bc_features_squeezed)  # [B, 1, Nt]
        bc_x0_pred = bc_x0_pred_chan.squeeze(1)  # [B, Nt]

        return rho_eps_pred, bc_x0_pred


class GaussianDiffusion3D(nn.Module):
    def __init__(
            self,
            model: Unet3D,
            *,
            rho_shape: tuple,  # Shape: (Nt, H, W), e.g., (64, Nx, Nx)
            timesteps=1000,
            beta_schedule='cosine',
            lambda_bc_x0=1.0,  # Loss weight for the direct BC (x0) prediction
            freeze_indices_rho: tuple = (slice(None), slice(None), slice(None)),
            rescaler=1.0,
    ):
        super().__init__()
        self.model = model
        self.device = next(model.parameters()).device

        self.rho_shape = rho_shape
        self.nt = rho_shape[0]  # Temporal dimension for rho and aligned BC
        self.H = rho_shape[1]
        self.W = rho_shape[2]

        # Assuming Unet3D input_bc_channels=1 for this strategy
        self.num_input_bc_channels = 1  # Fixed for this strategy
        self.diffusion_input_channels = 1 + self.num_input_bc_channels  # 1 for rho, 1 for expanded BC

        self.data_shape_diffusion = (self.diffusion_input_channels, self.nt, self.H, self.W)

        self.num_timesteps = int(timesteps)
        self.lambda_bc_x0 = lambda_bc_x0

        self.freeze_indices_train = (slice(None), 0, *freeze_indices_rho)  # For rho channel (index 0)
        self.freeze_indices_sample = (slice(None), 0, *freeze_indices_rho)

        self.rescaler = rescaler

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32).to(self.device))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        noise = noise.to(x_start.device)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return (
                       sqrt_alphas_cumprod_t * x_start +
                       sqrt_one_minus_alphas_cumprod_t * noise
               ), noise

    def predict_x0_from_eps(self, x_t, t, noise):
        # x_t 是例如 [B, 1, Nt, H, W] (rho部分)
        # t 是 [B]
        # noise 是例如 [B, 1, Nt, H, W] (对应的噪声)
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -  # 注意这里是 x_t.shape
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise  # 注意这里是 x_t.shape
        )

    def p_losses(self, rho_start, bc_start, t):
        # rho_start: [B, Nt, H, W]
        # bc_start: [B, Nt] (MUST have same Nt as rho_start)
        # t: [B]
        B = rho_start.shape[0]

        # 1. Prepare rho_start and bc_start for concatenation
        rho_start_expanded_ch = rho_start.unsqueeze(1)  # [B, 1, Nt, H, W]

        # Spatially expand bc_start
        # bc_start: [B, Nt] -> [B, 1, Nt] -> [B, 1, Nt, 1] -> [B, 1, Nt, 1, 1] -> expand to [B, 1, Nt, H, W]
        bc_start_expanded_spatial = bc_start.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).expand(
            B, self.num_input_bc_channels, self.nt, self.H, self.W
        )

        x_start_combined = torch.cat((rho_start_expanded_ch, bc_start_expanded_spatial), dim=1)
        # x_start_combined shape: [B, 1+1=2, Nt, H, W]

        # 2. Apply diffusion to combined state
        x_t, noise_target = self.q_sample(x_start=x_start_combined, t=t)

        # 3. Apply Freezing Constraint to rho part of x_t and noise_target
        x_t_frozen = x_t.clone()
        x_t_frozen[self.freeze_indices_train] = rho_start_expanded_ch[self.freeze_indices_train]

        noise_target_frozen = noise_target.clone()
        noise_target_frozen[self.freeze_indices_train] = 0.  # More precise for rho channel only

        # 4. Predict using the model
        rho_eps_pred, bc_x0_pred = self.model(x_t_frozen, t)
        # rho_eps_pred: [B, 1, Nt, H, W]
        # bc_x0_pred: [B, Nt]

        # 5. Calculate Losses
        loss_dict = {}

        # 扩散损失
        loss_rho_eps = F.mse_loss(rho_eps_pred, noise_target_frozen[:, 0:1, ...])
        loss_bc_direct_x0 = F.mse_loss(bc_x0_pred, bc_start)
        L_diffusion = loss_rho_eps + self.lambda_bc_x0 * loss_bc_direct_x0

        total_loss = L_diffusion
        # 日志记录
        loss_dict['total_loss'] = total_loss
        loss_dict['L_BC_x0'] = loss_bc_direct_x0.detach()

        return loss_dict

    @torch.no_grad()
    def p_sample_standard(self, x_k, k, x_f_known_normalized_rho):
        # x_k: [B, 2, Nt, H, W]
        # x_f_known_normalized_rho: [B, Nt, H, W] (normalized known rho for freezing)
        B = x_k.shape[0]
        batched_k = torch.full((B,), k, device=self.device, dtype=torch.long)

        x_k_frozen = x_k.clone()
        x_f_known_norm_rho_chan = x_f_known_normalized_rho.unsqueeze(1)  # [B, 1, Nt, H, W]
        x_k_frozen[self.freeze_indices_sample] = x_f_known_norm_rho_chan[self.freeze_indices_sample]

        rho_eps_pred, bc_x0_pred = self.model(x_k_frozen, batched_k)
        # rho_eps_pred: [B, 1, Nt, H, W]
        # bc_x0_pred: [B, Nt]

        # Estimate clean rho from its noise prediction
        # x_k_frozen[:, 0:1, ...] is the rho part of the noisy input
        rho_0_pred = self.predict_x0_from_eps(x_k_frozen[:, 0:1, ...], batched_k, rho_eps_pred)

        # Spatially expand predicted clean bc_x0_pred
        bc_x0_pred_expanded_spatial = bc_x0_pred.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).expand(
            B, self.num_input_bc_channels, self.nt, self.H, self.W
        )

        x_0_pred_combined = torch.cat((rho_0_pred, bc_x0_pred_expanded_spatial), dim=1)
        x_0_pred_combined[self.freeze_indices_sample] = x_f_known_norm_rho_chan[self.freeze_indices_sample]
        x_0_pred_combined.clamp_(-1., 1.)

        posterior_mean_coef1_t = extract(self.posterior_mean_coef1, batched_k, x_k.shape)
        posterior_mean_coef2_t = extract(self.posterior_mean_coef2, batched_k, x_k.shape)
        model_mean = (posterior_mean_coef1_t * x_0_pred_combined +
                      posterior_mean_coef2_t * x_k_frozen)

        model_log_variance = extract(self.posterior_log_variance_clipped, batched_k, x_k.shape)
        posterior_variance = model_log_variance.exp()

        noise = torch.randn_like(x_k) if k > 0 else 0.
        nonzero_mask = (1 - (batched_k == 0).float()).reshape(B, *((1,) * (len(x_k.shape) - 1)))
        x_prev = model_mean + nonzero_mask * torch.sqrt(posterior_variance) * noise
        return x_prev

    @torch.no_grad()
    def p_sample_loop(self, batch_size, x_f_known_rho_orig, rescaler):
        # x_f_known_rho_orig: [B, Nt, H, W] (original scale)
        img = torch.randn((batch_size, *self.data_shape_diffusion), device=self.device)

        x_f_known_rho_orig = x_f_known_rho_orig.to(self.device)
        rescaler = rescaler.to(self.device)
        x_f_known_normalized_rho = 2 * (x_f_known_rho_orig / rescaler) - 1  # Assuming standard normalization

        # tqdm can be added in the calling script (e.g., Trainer or inference script)
        for k_val in reversed(range(0, self.num_timesteps)):
            img = self.p_sample_standard(img, k_val, x_f_known_normalized_rho)
        return img  # This is the predicted x0_combined (normalized) [B, 2, Nt, H, W]

    @torch.no_grad()
    def sample(self, batch_size: int, x_f_known_rho_orig, rescaler):
        # x_f_known_rho_orig: [B, Nt, H, W] (original scale, for freezing)

        x0_approx_combined_norm = self.p_sample_loop(batch_size, x_f_known_rho_orig, rescaler)
        # x0_approx_combined_norm is [B, 2, Nt, H, W]

        # Better: Use a final model call with t=0 for clean predictions, especially for BC.
        k_zero = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        # The input to the model should be the "denoised" state x0_approx_combined_norm,
        # as if it were x_t at t=0 (where noise is zero).
        final_rho_eps_pred, final_bc_x0_pred_norm = self.model(x0_approx_combined_norm, k_zero)

        # For rho, use the prediction from x0_approx_combined_norm directly or re-derive
        # If using x0_approx_combined_norm directly for rho:
        rho_0_final_norm = x0_approx_combined_norm[:, 0:1, ...].squeeze(1).clamp(-1., 1.)

        # For BC, use the direct x0 prediction from the model at t=0
        bc_0_final_norm = final_bc_x0_pred_norm.clamp(-1., 1.)  # [B, Nt]

        return rho_0_final_norm, bc_0_final_norm  # Return normalized predictions

    def forward(self, data_tuple):  # For training
        rho_start, bc_start = data_tuple  # rho_start: [B,Nt,H,W], bc_start: [B,Nt]

        rho_start = rho_start.to(self.device)
        bc_start = bc_start.to(self.device)

        # Validate that bc_start has the same temporal dimension as rho_start
        if rho_start.shape[1] != bc_start.shape[1]:
            raise ValueError(f"Temporal dimension of rho_start ({rho_start.shape[1]}) "
                             f"must match temporal dimension of bc_start ({bc_start.shape[1]}) for this strategy.")

        b = rho_start.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=self.device).long()

        return self.p_losses(rho_start, bc_start, t)
