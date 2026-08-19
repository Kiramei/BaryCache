import os
from typing import Optional, List, Dict, Any, Union, Callable, Tuple, Sequence

import torch
from diffusers import AutoencoderKL
from torchvision.utils import save_image

from diffusion import create_diffusion
from download import find_model
from models import DiT_models, modulate


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
    fresh_interval = 2
    rel_l1_thresh = 0.13
    bary_probe_eps = 1e-8
    mode_cache = "stepwise"
    cache_ratio = 0.0
    max_history = 2
    blend_factor = None
    _GLOBAL_ALPHA_CACHE: Dict[Tuple[Any, Union[int, float]], torch.Tensor] = {}
    _CACHE_PATH = "bary_alpha_cache.pt"
    _CACHE_LOADED = False

    suggested_compress_ratio_map = {
        2: 1,
        3: 0.95,
        4: 0.75
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
        print(self.full_record)
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

def _macro_upto(depth: int, cache_ratio: float) -> int:
    r = float(cache_ratio)
    r = 0.0 if r < 0.0 else (1.0 if r > 1.0 else r)
    return int(round(r * depth))

def bary_block_forward(self, x, c, *, layer: Optional[int] = None):
    cache: Optional[BaryCache] = getattr(self, "cache", None)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
        self.adaLN_modulation(c).chunk(6, dim=1)
    skip_upto = int(getattr(cache, "skip_upto", 0)) if cache is not None else 0
    need_internal = (cache is not None) and (cache.mode_cache == "blockwise") and (layer is not None) and (layer >= skip_upto)
    if need_internal and (not cache.full_forward):
        attn_out = cache.predict(f"attn.{layer}", ref=x)
    else:
        attn_out = self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        if need_internal and cache.full_forward:
            cache.save_anchor(f"attn.{layer}", attn_out)
    x = x + gate_msa.unsqueeze(1) * attn_out
    if need_internal and (not cache.full_forward):
        mlp_out = cache.predict(f"mlp.{layer}", ref=x)
    else:
        mlp_out = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        if need_internal and cache.full_forward:
            cache.save_anchor(f"mlp.{layer}", mlp_out)

    x = x + gate_mlp.unsqueeze(1) * mlp_out
    return x


def bary_forward(self, x, t, y):
    cache: Optional[BaryCache] = getattr(self, "cache", None)
    if cache is None:
        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t + y
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x

    x = self.x_embedder(x) + self.pos_embed
    t_emb = self.t_embedder(t)
    y_emb = self.y_embedder(y, self.training)
    c = t_emb + y_emb
    probe = None
    if getattr(cache, "mode_schedule", "eq-space") == "adaptive":
        shift_msa, scale_msa, *_ = self.blocks[0].adaLN_modulation(c).chunk(6, dim=1)
        probe = modulate(self.blocks[0].norm1(x), shift_msa, scale_msa)

    solved_type = cache.solve_type(probe=probe)

    if cache.mode_cache == "stepwise":
        if solved_type:
            for ind, block in enumerate(self.blocks):
                x = block(x, c, layer=ind)
            cache.save_anchor("stepwise", x)
        else:
            x = cache.predict("stepwise", ref=x)

    elif cache.mode_cache == "blockwise":
        n_blocks = len(self.blocks)
        skip_upto = int(cache.cache_ratio * n_blocks)
        cache.skip_upto = skip_upto
        for ind, block in enumerate(self.blocks):
            if (not cache.full_forward) and (ind < skip_upto):
                x = cache.predict(f"block_out.{ind}", ref=x)
                continue
            x = block(x, c, layer=ind)
            if cache.full_forward and (ind < skip_upto):
                cache.save_anchor(f"block_out.{ind}", x)

    else:
        for block in self.blocks:
            x = block(x, c)

    x = self.final_layer(x, c)
    x = self.unpatchify(x)

    cache.step()
    return x


def pipeline():
    seed = 0
    sample_steps = 50
    image_size = 256
    latent_size = image_size // 8
    num_classes = 1000
    cfg_scale = 4.0

    model_type = 'DiT-XL/2'
    dit_ckpt_path = "YOUR_PATH/DiT-XL-2-256x256.pt"
    vae_path = "YOUR_PATH/sd-vae-ft-ema"

    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DiT_models[model_type](
        input_size=latent_size,
        num_classes=num_classes,
    ).to(device)
    state_dict = find_model(dit_ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()

    def init_params():
        shared_cache = BaryCache(sample_steps)
        shared_cache.mode_cache = "stepwise"
        shared_cache.mode_schedule = "eq-space"
        shared_cache.fresh_start = 3
        shared_cache.fresh_interval = 1
        shared_cache.cache_ratio = 0.0
        shared_cache.rel_l1_thresh = 0.45
        model.__class__.forward = bary_forward
        model.blocks[0].__class__.forward = bary_block_forward
        model.cache = shared_cache
        for blk in model.blocks:
            blk.cache = shared_cache

    init_params()

    diffusion = create_diffusion(str(sample_steps))
    vae = AutoencoderKL.from_pretrained(vae_path).to(device)
    class_labels = [985]
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=cfg_scale)
    samples = diffusion.ddim_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)
    samples = vae.decode(samples / 0.18215).sample
    save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == '__main__':
    pipeline()
