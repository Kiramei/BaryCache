import hashlib
import os
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Any, Union

import torch
from diffusers import AutoencoderKL
from torchvision.utils import save_image
from transformers import T5Tokenizer, T5EncoderModel

from diffusion import IDDPM
from diffusion.data import ASPECT_RATIO_1024_TEST
from diffusion.model import PixArtMS_XL_2
from diffusion.model.nets.PixArt import get_2d_sincos_pos_embed
from diffusion.model.nets.PixArt_blocks import t2i_modulate
from diffusion.model.utils import auto_grad_checkpoint, prepare_prompt_ar
from diffusion.utils.misc import set_random_seed

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def prompt_hash(prompt: str) -> str:
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()


ScheduleFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def make_weight_fn(
        decay_alpha: float = 0.75,
        denom_eps: float = 1e-6,
        clip_abs: float = 10.0,
        normalize: bool = True,
):
    def schedule(
            t: torch.Tensor,
            x: torch.Tensor,
            eps: float = 1e-8,
    ) -> torch.Tensor:
        w_anchor = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
        w_anchor[1::2] *= -1
        w_anchor[0] = w_anchor[0] * 0.5
        w_anchor[-1] = w_anchor[-1] * 0.5
        diff = t - x
        diff = torch.where(diff.abs() < eps, diff.sign() * eps, diff)
        return w_anchor / diff

    def weight_fn(t, x):
        u = schedule(t, x)
        if clip_abs is not None:
            u = u.clamp(min=-clip_abs, max=clip_abs)
        u = torch.sign(u) * torch.pow(u.abs() + 1e-12, decay_alpha)
        if not normalize:
            return u
        s = u.sum()
        s = torch.where(s.abs() < denom_eps, s.sign() * denom_eps, s)
        return u / s

    return weight_fn


def apply_bary_weights(alpha: torch.Tensor, feats: Sequence[torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(list(feats), dim=0)
    view_shape = (-1,) + (1,) * (stacked.ndim - 1)
    return (alpha.view(*view_shape) * stacked).sum(dim=0)


class BaryCache:
    mode_schedule = "eq-space"
    fresh_start = 3
    fresh_end = 0
    fresh_interval = 4
    rel_l1_thresh = 0.16
    bary_probe_eps = 1e-8
    mode_cache = "stepwise"
    cache_ratio = 0.0

    max_history = 2
    blend_factor = None

    _GLOBAL_ALPHA_CACHE: Dict[Tuple[Any, Union[int, float]], torch.Tensor] = {}
    _CACHE_PATH = "bary_alpha_cache.pt"
    _CACHE_LOADED = False

    suggested_compress_ratio_map = {
        2: 1.0,
        3: 0.95,
        4: 0.7
    }

    def __init__(self, num_steps: int, weight_fn: Optional[Callable] = None):
        self.num_steps = num_steps
        self.current_step = num_steps - 1

        if not BaryCache._CACHE_LOADED:
            if os.path.exists(BaryCache._CACHE_PATH):
                try:
                    loaded_data = torch.load(BaryCache._CACHE_PATH, map_location="cuda:0")
                    if isinstance(loaded_data, dict):
                        BaryCache._GLOBAL_ALPHA_CACHE = loaded_data
                except Exception as e:
                    print(f"[BaryCache] Failed to load cache: {e}. Starting empty.")
            BaryCache._CACHE_LOADED = True
        self.full_forward: bool = True
        self.interval_counter: int = 0
        self.rel_accum: float = 0.0
        self.prev_probe: Optional[torch.Tensor] = None
        self.full_record: List[int] = []
        self.history: Dict[str, List[Dict]] = {}
        self.last_value: Dict[str, torch.Tensor] = {}
        if weight_fn is None:
            self.weight_fn = make_weight_fn(
                normalize=True,
                decay_alpha=self.suggested_compress_ratio_map[self.max_history]
            )
        else:
            self.weight_fn = weight_fn

    def save_anchor(self, key: str, value: torch.Tensor) -> None:
        self.last_value[key] = value.detach()
        hist = self.history.setdefault(key, [])
        hist.append({"step": int(self.current_step), "value": value.detach()})
        if len(hist) > self.max_history:
            hist.pop(0)

    def predict(self, key: str, ref: Optional[torch.Tensor] = None) -> torch.Tensor:
        hist = self.history.get(key, [])
        if len(hist) == 0:
            raise KeyError(f"[BaryCache] No history for key={key}. Full-forward at least once is required.")
        if len(hist) == 1:
            out = hist[-1]["value"]
            return out if ref is None else out.to(device=ref.device, dtype=ref.dtype)

        n = min(self.max_history, len(hist))
        recent = hist[-n:]
        feats = [e["value"] for e in recent]
        device = feats[0].device
        dtype = feats[0].dtype

        for e in recent:
            if int(self.current_step) == int(e["step"]):
                out = e["value"]
                return out if ref is None else out.to(device=ref.device, dtype=ref.dtype)

        diff_key = tuple((*(self.current_step - int(e["step"]) for e in recent),
                          self.suggested_compress_ratio_map[self.max_history]))

        alpha = BaryCache._GLOBAL_ALPHA_CACHE.get(diff_key)
        if alpha is not None:
            if alpha.device != device:
                alpha = alpha.to(device=device, dtype=dtype)
                BaryCache._GLOBAL_ALPHA_CACHE[diff_key] = alpha
            elif alpha.dtype != dtype:
                alpha = alpha.to(dtype=dtype)

        else:
            steps = torch.tensor([e["step"] for e in recent], device=device, dtype=torch.float16)
            t = torch.tensor(float(self.current_step), device=device, dtype=torch.float16)
            alpha = self.weight_fn(t, steps)
            BaryCache._GLOBAL_ALPHA_CACHE[diff_key] = alpha.detach()

        out = apply_bary_weights(alpha, feats)
        if self.blend_factor is not None and self.blend_factor > 0:
            last_anchor = hist[-1]["value"]
            lam = float(self.blend_factor)
            out = lam * last_anchor + (1 - lam) * out

        return out if ref is None else out.to(device=ref.device, dtype=ref.dtype)

    def solve_type(self, probe: Optional[torch.Tensor] = None) -> bool:
        step = int(self.current_step)
        total = int(self.num_steps)

        is_near_start = step >= total - int(self.fresh_start)
        is_near_end = step < int(self.fresh_end)

        if self.mode_schedule == "eq-space":
            interval_hit = (self.fresh_interval is not None) and (not is_near_start) and (
                    step <= self.full_record[-1] - self.fresh_interval)
            self.full_forward = bool(is_near_start or is_near_end or interval_hit)
            return self.full_forward

        if self.mode_schedule == "adaptive":
            if is_near_start or is_near_end:
                need_full = True
            else:
                if (self.rel_l1_thresh is None) or (probe is None) or (self.prev_probe is None):
                    need_full = True
                else:
                    rel_change = (probe - self.prev_probe).abs().mean() / (
                            self.prev_probe.abs().mean() + self.bary_probe_eps)
                    delta_score = float(rel_change.detach().float().cpu().item())
                    self.rel_accum += delta_score
                    need_full = self.rel_accum >= float(self.rel_l1_thresh)

            self.full_forward = bool(need_full)

            if self.full_forward:
                self.interval_counter = 0
                self.rel_accum = 0.0
            else:
                self.interval_counter += 1
            if probe is not None:
                self.prev_probe = probe.detach()

            return self.full_forward

        raise ValueError(f"[BaryCache] Unknown mode_schedule={self.mode_schedule}")

    def step(self) -> None:
        if self.full_forward:
            self.full_record.append(int(self.current_step))

        self.current_step -= 1

        if self.current_step <= 0:
            self.demolish()

    def demolish(self) -> None:
        if BaryCache._CACHE_PATH:
            try:
                cpu_cache = {k: v.cpu() for k, v in BaryCache._GLOBAL_ALPHA_CACHE.items()}
                torch.save(cpu_cache, BaryCache._CACHE_PATH)
            except Exception as e:
                print(f"[BaryCache] Warning: Could not save cache: {e}")

        self.history.clear()
        self.last_value.clear()
        self.prev_probe = None
        self.rel_accum = 0.0
        self.interval_counter = 0
        self.full_record.clear()
        self.current_step = self.num_steps - 1
        self.full_forward = True


def bary_block_forward(
        self,
        x,
        y,
        t,
        mask=None,
        HW=None,
        **kwargs
):
    B, N, C = x.shape

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
    ).chunk(6, dim=1)

    layer = kwargs.get("layer", None)
    need_internal = (layer is not None) and (layer >= getattr(self.cache, "skip_upto", 0))

    if (not self.cache.full_forward) and (self.cache.mode_cache == "blockwise"):
        self_attn_out = self.cache.predict(f"self_attn.{layer}", ref=x)
    else:
        self_attn_out = self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa), HW=HW)
        if self.cache.full_forward and need_internal and (self.cache.mode_cache == "blockwise"):
            self.cache.save_anchor(f"self_attn.{layer}", self_attn_out)

    x = x + self.drop_path(gate_msa * self_attn_out)

    if (not self.cache.full_forward) and (self.cache.mode_cache == "blockwise"):
        cross_attn_out = self.cache.predict(f"cross_attn.{layer}", ref=x)
    else:
        cross_attn_out = self.cross_attn(x, y, mask)
        if self.cache.full_forward and need_internal and (self.cache.mode_cache == "blockwise"):
            self.cache.save_anchor(f"cross_attn.{layer}", cross_attn_out)

    x = x + cross_attn_out

    if (not self.cache.full_forward) and (self.cache.mode_cache == "blockwise"):
        mlp_out = self.cache.predict(f"mlp.{layer}", ref=x)
    else:
        mlp_out = self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp))
        if self.cache.full_forward and need_internal and (self.cache.mode_cache == "blockwise"):
            self.cache.save_anchor(f"mlp.{layer}", mlp_out)

    x = x + self.drop_path(gate_mlp * mlp_out)

    return x


def bary_forward(
        self,
        x,
        timestep,
        y,
        mask=None,
        data_info=None,
        **kwargs
):
    bs = x.shape[0]
    x = x.to(self.dtype)
    timestep = timestep.to(self.dtype)
    y = y.to(self.dtype)

    self.h, self.w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
    pos_embed = torch.from_numpy(
        get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            (self.h, self.w),
            pe_interpolation=self.pe_interpolation,
            base_size=self.base_size
        )
    ).unsqueeze(0).to(x.device).to(self.dtype)

    x = self.x_embedder(x) + pos_embed
    t = self.t_embedder(timestep)

    if self.micro_conditioning:
        c_size, ar = data_info["img_hw"].to(self.dtype), data_info["aspect_ratio"].to(self.dtype)
        csize = self.csize_embedder(c_size, bs)
        ar = self.ar_embedder(ar, bs)
        t = t + torch.cat([csize, ar], dim=1)

    t0 = self.t_block(t)
    y = self.y_embedder(y, self.training)

    if mask is not None:
        if mask.shape[0] != y.shape[0]:
            mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
        mask = mask.squeeze(1).squeeze(1)
        y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
        y_lens = mask.sum(dim=1).tolist()
    else:
        y_lens = [y.shape[2]] * y.shape[0]
        y = y.squeeze(1).view(1, -1, x.shape[-1])

    probe = None
    if getattr(self.cache, "mode_schedule", "eq-space") == "adaptive":
        B = x.shape[0]
        shift_msa, scale_msa, *_ = (
                self.blocks[0].scale_shift_table[None] + t0.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        probe = t2i_modulate(self.blocks[0].norm1(x), shift_msa, scale_msa)

    solved_type = self.cache.solve_type(probe=probe)

    if self.cache.mode_cache == "stepwise":
        if solved_type:
            for ind, block in enumerate(self.blocks):
                x = auto_grad_checkpoint(
                    block,
                    x,
                    y,
                    t0,
                    y_lens,
                    (self.h, self.w),
                    **(kwargs | {"layer": ind})
                )
            self.cache.save_anchor("stepwise", x)
        else:
            x = self.cache.predict("stepwise", ref=x)

    elif self.cache.mode_cache == "blockwise":
        n_blocks = len(self.blocks)
        skip_upto = int(self.cache.cache_ratio * n_blocks)
        self.cache.skip_upto = skip_upto

        for ind, block in enumerate(self.blocks):
            if (not self.cache.full_forward) and (ind < skip_upto):
                x = self.cache.predict(f"block_out.{ind}", ref=x)
                continue

            x = auto_grad_checkpoint(
                block,
                x,
                y,
                t0,
                y_lens,
                (self.h, self.w),
                **(kwargs | {"layer": ind})
            )

            if self.cache.full_forward:
                if ind < self.cache.skip_upto:
                    self.cache.save_anchor(f"block_out.{ind}", x)

    else:
        for block in self.blocks:
            x = auto_grad_checkpoint(block, x, y, t0, y_lens, (self.h, self.w), **kwargs)

    x = self.final_layer(x, t)
    x = self.unpatchify(x)

    self.cache.step()
    return x


def pipeline():
    seed = 2025
    sample_steps = 50
    latent_size = 1024 // 8
    pe_interpolation = 2
    micro_condition = False
    max_sequence_length = 300
    config_scale = 4.5
    weight_dtype = torch.float16
    t5_cpu = False
    device = torch.device("cuda:0")
    SAVE_PATH = "sample.png"
    TRANSFORMER_PATH = "YOUR_PATH/PixArt-Sigma-XL-2-1024-MS.pth"
    VAE_PATH = "YOUR_PATH/pixart_sigma_sdxlvae_T5_diffusers/vae"
    T5_PIPELINE_PATH = "YOUR_PATH/pixart_sigma_sdxlvae_T5_diffusers"
    embed_cache_dir = Path("path-to-embed-cache")
    embed_cache_dir.mkdir(parents=True, exist_ok=True)
    prompt = "Your Prompt"

    diffusion = IDDPM(str(sample_steps))

    model = PixArtMS_XL_2(
        input_size=latent_size,
        pe_interpolation=pe_interpolation,
        micro_condition=micro_condition,
        model_max_length=max_sequence_length,
    ).to(device)
    state_dict = torch.load(TRANSFORMER_PATH, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict['state_dict'], strict=False)
    model.to(weight_dtype).eval()

    def init_params():
        shared_cache = BaryCache(sample_steps)
        model.__class__.forward = bary_forward
        model.blocks[0].__class__.forward = bary_block_forward
        model.cache = shared_cache
        for blk in model.blocks:
            blk.cache = shared_cache

    tokenizer = None
    text_encoder = None

    def ensure_t5_loaded():
        nonlocal tokenizer, text_encoder
        if tokenizer is None or text_encoder is None:
            tokenizer = T5Tokenizer.from_pretrained(T5_PIPELINE_PATH, subfolder="tokenizer")
            text_encoder_device = torch.device("cpu") if t5_cpu else device
            text_encoder = T5EncoderModel.from_pretrained(T5_PIPELINE_PATH, subfolder="text_encoder").to(
                text_encoder_device
            )
            text_encoder.eval()

    null_emb_path = embed_cache_dir / "null_embedding.pt"

    if null_emb_path.exists():
        cached = torch.load(null_emb_path, map_location="cpu")
        null_caption_embs = cached["caption_embs"].to(device)
    else:
        ensure_t5_loaded()
        te_dev = next(text_encoder.parameters()).device
        null_caption_token = tokenizer(
            "",
            max_length=max_sequence_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(te_dev)
        null_caption_embs = text_encoder(
            null_caption_token.input_ids, attention_mask=null_caption_token.attention_mask
        )[0].to(device)
        torch.save({"caption_embs": null_caption_embs.cpu()}, null_emb_path)

    vae = AutoencoderKL.from_pretrained(VAE_PATH).to(device).to(weight_dtype)

    base_ratios = ASPECT_RATIO_1024_TEST
    prompt_clean, _, hw, ar, _ = prepare_prompt_ar(prompt, base_ratios, device=device, show=False)
    prompt_clean = prompt_clean.strip()

    ph = prompt_hash(prompt)
    embed_cache_path = embed_cache_dir / f"{ph}_embedding.pt"

    if embed_cache_path.exists():
        cached = torch.load(embed_cache_path, map_location="cpu")
        caption_embs = cached["caption_embs"].to(device)
        emb_masks = cached["emb_masks"].to(device)
    else:
        ensure_t5_loaded()
        te_dev = next(text_encoder.parameters()).device
        caption_token = tokenizer(
            [prompt_clean],
            max_length=max_sequence_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(te_dev)
        ce = text_encoder(caption_token.input_ids, attention_mask=caption_token.attention_mask)[0]
        caption_embs = ce.to(device)
        emb_masks = caption_token.attention_mask.to(device)

        torch.save({"caption_embs": caption_embs.cpu(), "emb_masks": emb_masks.cpu()}, embed_cache_path)

    caption_embs = caption_embs[:, None]
    null_y = null_caption_embs.repeat(1, 1, 1)[:, None]

    set_random_seed(seed)

    latent_size_h, latent_size_w = int(hw[0, 0] // 8), int(hw[0, 1] // 8)
    y = torch.cat([caption_embs] + [null_y])
    z = torch.randn(1, 4, latent_size_h, latent_size_w, device=device).repeat(2, 1, 1, 1)

    model_kwargs = dict(
        y=y,
        cfg_scale=config_scale,
        data_info={'img_hw': hw, 'aspect_ratio': ar},
        mask=emb_masks,
        sample_folder_dir="samples",
        output_folder_dir="test",
    )

    init_params()
    path_to_save = SAVE_PATH

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record()
    with torch.inference_mode():
        latents = diffusion.ddim_sample_loop(
            noise=z,
            shape=z.shape,
            model=model.forward_with_cfg,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device=device
        ).chunk(2, dim=0)[0].to(weight_dtype)
        sample = vae.decode(latents / vae.config.scaling_factor).sample[0]

    end.record()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    os.umask(0o000)

    elapsed_time = start.elapsed_time(end) * 1e-3
    save_image(sample, path_to_save, nrow=1, normalize=True, value_range=(-1, 1))
    print(f"Saved sample to {path_to_save}, elapsed time: {elapsed_time}")


if __name__ == '__main__':
    pipeline()
