import contextlib
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Any, Union

from loguru import logger

from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler
from hyvideo.utils.file_utils import save_videos_grid

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import sys

ScheduleFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def make_weight_fn(
        decay_alpha: float = 0.75,
        denom_eps: float = 1e-6,
        clip_abs: float = 10.0,
        normalize: bool = True,
):
    def schedule(t: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        w_anchor = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
        w_anchor[1::2] *= -1
        if x.shape[0] >= 1:
            w_anchor[0] = w_anchor[0] * 0.5
            w_anchor[-1] = w_anchor[-1] * 0.5

        diff = t - x
        tiny = torch.full_like(diff, eps)
        diff = torch.where(diff.abs() < eps, torch.where(diff >= 0, tiny, -tiny), diff)
        return w_anchor / diff

    def weight_fn(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        u = schedule(t, x)

        if clip_abs is not None:
            u = u.clamp(min=-clip_abs, max=clip_abs)
        u = torch.sign(u) * torch.pow(u.abs() + 1e-12, decay_alpha)

        if not normalize:
            return u

        s = u.sum()
        tiny = torch.tensor(denom_eps, device=s.device, dtype=s.dtype)
        s = torch.where(s.abs() < denom_eps, torch.where(s >= 0, tiny, -tiny), s)
        return u / s

    return weight_fn


def apply_bary_weights(alpha: torch.Tensor, feats: Sequence[torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(list(feats), dim=0)
    view_shape = (-1,) + (1,) * (stacked.ndim - 1)
    return (alpha.view(*view_shape) * stacked).sum(dim=0)


class BaryCache:
    mode_schedule: str = "eq-space"
    fresh_start: int = 3
    fresh_end: int = 0
    fresh_interval: int = 6

    rel_l1_thresh: float = 0.13
    bary_probe_eps: float = 1e-8

    mode_cache: str = "stepwise"
    cache_ratio: float = 0.0

    max_history: int = 2
    blend_factor: Optional[float] = None

    _GLOBAL_ALPHA_CACHE: Dict[Tuple[Any, ...], torch.Tensor] = {}
    _CACHE_PATH: str = "bary_alpha_cache.pt"
    _CACHE_LOADED: bool = False

    suggested_compress_ratio_map = {
        2: 1.00,
        3: 0.95,
        4: 0.70,
    }

    def __init__(self, num_steps: int, weight_fn: Optional[Callable] = None):
        self.num_steps = int(num_steps)
        self.current_step = int(num_steps) - 1

        if not BaryCache._CACHE_LOADED and BaryCache._CACHE_PATH:
            if os.path.exists(BaryCache._CACHE_PATH):
                try:
                    loaded_data = torch.load(BaryCache._CACHE_PATH, map_location="cpu")
                    if isinstance(loaded_data, dict):
                        BaryCache._GLOBAL_ALPHA_CACHE = loaded_data
                except Exception as e:
                    print(f"[BaryCache] Failed to load alpha cache: {e}. Starting empty.")
            BaryCache._CACHE_LOADED = True

        self.full_forward: bool = True
        self.rel_accum: float = 0.0
        self.prev_probe: Optional[torch.Tensor] = None
        self.full_record: List[int] = []

        self.history: Dict[str, List[Dict[str, Any]]] = {}
        self.last_value: Dict[str, torch.Tensor] = {}

        if weight_fn is None:
            mh = int(self.max_history)
            decay = self.suggested_compress_ratio_map.get(mh, 0.75)
            self.weight_fn = make_weight_fn(normalize=True, decay_alpha=decay)
        else:
            self.weight_fn = weight_fn
        self.skip_upto_double: int = 0
        self.skip_upto_single: int = 0

    def has_history(self, key: str) -> bool:
        return key in self.history and len(self.history[key]) > 0

    def save_anchor(self, key: str, value: torch.Tensor) -> None:
        v = value.detach()
        self.last_value[key] = v

        hist = self.history.setdefault(key, [])
        hist.append({"step": int(self.current_step), "value": v})

        if len(hist) > int(self.max_history):
            hist.pop(0)

    def predict(self, key: str, ref: Optional[torch.Tensor] = None) -> torch.Tensor:
        hist = self.history.get(key, [])
        if len(hist) == 0:
            raise KeyError(f"[BaryCache] No history for key={key}. Full-forward at least once is required.")
        if len(hist) == 1:
            out = hist[-1]["value"]
            return out if ref is None else out.to(device=ref.device, dtype=ref.dtype)

        n = min(int(self.max_history), len(hist))
        recent = hist[-n:]

        feats = [e["value"] for e in recent]
        device = feats[0].device
        dtype = feats[0].dtype

        for e in recent:
            if int(self.current_step) == int(e["step"]):
                out = e["value"]
                return out if ref is None else out.to(device=ref.device, dtype=ref.dtype)

        mh = int(self.max_history)
        decay = self.suggested_compress_ratio_map.get(mh, 0.75)
        diff_key = tuple([*(int(self.current_step) - int(e["step"]) for e in recent), mh, float(decay)])

        alpha = BaryCache._GLOBAL_ALPHA_CACHE.get(diff_key)
        if alpha is not None:
            if alpha.device != device:
                alpha = alpha.to(device=device)
                BaryCache._GLOBAL_ALPHA_CACHE[diff_key] = alpha
        else:
            steps = torch.tensor([float(e["step"]) for e in recent], device=device, dtype=torch.float32)
            t = torch.tensor(float(self.current_step), device=device, dtype=torch.float32)
            alpha = self.weight_fn(t, steps).detach()
            BaryCache._GLOBAL_ALPHA_CACHE[diff_key] = alpha

        alpha = alpha.to(device=device, dtype=dtype)

        out = apply_bary_weights(alpha, feats)

        if self.blend_factor is not None and float(self.blend_factor) > 0:
            lam = float(self.blend_factor)
            out = lam * recent[-1]["value"] + (1.0 - lam) * out

        return out if ref is None else out.to(device=ref.device, dtype=ref.dtype)

    def solve_type(self, probe: Optional[torch.Tensor] = None) -> bool:
        step = int(self.current_step)
        total = int(self.num_steps)

        is_near_start = step >= total - int(self.fresh_start)
        is_near_end = step < int(self.fresh_end)

        if self.mode_schedule == "eq-space":
            last_full = self.full_record[-1] if len(self.full_record) > 0 else (total - 1)
            interval = int(self.fresh_interval) if self.fresh_interval is not None else None
            interval_hit = (interval is not None) and (not is_near_start) and (not is_near_end) and (
                    step <= last_full - interval)
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
                            self.prev_probe.abs().mean() + float(self.bary_probe_eps))
                    self.rel_accum += float(rel_change.detach().float().cpu().item())
                    need_full = self.rel_accum >= float(self.rel_l1_thresh)

            self.full_forward = bool(need_full)

            if self.full_forward:
                self.rel_accum = 0.0

            if probe is not None:
                self.prev_probe = probe.detach()

            return self.full_forward

        raise ValueError(f"[BaryCache] Unknown mode_schedule={self.mode_schedule}")

    def step(self) -> None:
        if self.full_forward:
            self.full_record.append(int(self.current_step))

        self.current_step -= 1

        if self.current_step < 0:
            self.finalize()

    def finalize(self) -> None:
        if BaryCache._CACHE_PATH:
            try:
                cpu_cache = {k: v.cpu() for k, v in BaryCache._GLOBAL_ALPHA_CACHE.items()}
                torch.save(cpu_cache, BaryCache._CACHE_PATH)
            except Exception as e:
                print(f"[BaryCache] Warning: Could not save alpha cache: {e}")

        self.history.clear()
        self.last_value.clear()
        self.prev_probe = None
        self.rel_accum = 0.0
        self.full_record.clear()


from einops import rearrange
from hyvideo.modules.modulate_layers import apply_gate, modulate
from hyvideo.modules.posemb_layers import apply_rotary_emb
from hyvideo.modules.attenion import get_cu_seqlens, parallel_attention, attention


def bary_double_forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: tuple = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cache: Optional[BaryCache] = getattr(self, "cache", None)
    layer: int = int(getattr(self, "_bary_layer", -1))

    (
        img_mod1_shift, img_mod1_scale, img_mod1_gate,
        img_mod2_shift, img_mod2_scale, img_mod2_gate,
    ) = self.img_mod(vec).chunk(6, dim=-1)
    (
        txt_mod1_shift, txt_mod1_scale, txt_mod1_gate,
        txt_mod2_shift, txt_mod2_scale, txt_mod2_gate,
    ) = self.txt_mod(vec).chunk(6, dim=-1)

    skip_this_layer = (
            cache is not None
            and cache.mode_cache == "blockwise"
            and (not cache.full_forward)
            and 0 <= layer < int(getattr(cache, "skip_upto_double", 0))
    )
    save_this_layer = (
            cache is not None
            and cache.mode_cache == "blockwise"
            and cache.full_forward
            and 0 <= layer < int(getattr(cache, "skip_upto_double", 0))
    )

    if skip_this_layer:
        try:
            k_img_attn = f"double_stream.{layer}.img_attn"
            k_img_mlp = f"double_stream.{layer}.img_mlp"
            k_txt_attn = f"double_stream.{layer}.txt_attn"
            k_txt_mlp = f"double_stream.{layer}.txt_mlp"

            img_attn_branch = cache.predict(k_img_attn, ref=img)
            img = img + apply_gate(img_attn_branch, gate=img_mod1_gate)

            img_mlp_branch = cache.predict(k_img_mlp, ref=img)
            img = img + apply_gate(img_mlp_branch, gate=img_mod2_gate)

            txt_attn_branch = cache.predict(k_txt_attn, ref=txt)
            txt = txt + apply_gate(txt_attn_branch, gate=txt_mod1_gate)

            txt_mlp_branch = cache.predict(k_txt_mlp, ref=txt)
            txt = txt + apply_gate(txt_mlp_branch, gate=txt_mod2_gate)

            return img, txt

        except KeyError:
            pass

    img_modulated = self.img_norm1(img)
    img_modulated = modulate(img_modulated, shift=img_mod1_shift, scale=img_mod1_scale)
    img_qkv = self.img_attn_qkv(img_modulated)
    img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
    img_q = self.img_attn_q_norm(img_q).to(img_v)
    img_k = self.img_attn_k_norm(img_k).to(img_v)

    if freqs_cis is not None:
        img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
        img_q, img_k = img_qq, img_kk

    txt_modulated = self.txt_norm1(txt)
    txt_modulated = modulate(txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale)
    txt_qkv = self.txt_attn_qkv(txt_modulated)
    txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
    txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
    txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

    q = torch.cat((img_q, txt_q), dim=1)
    k = torch.cat((img_k, txt_k), dim=1)
    v = torch.cat((img_v, txt_v), dim=1)

    if not self.hybrid_seq_parallel_attn:
        attn = attention(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            batch_size=img_k.shape[0],
        )
    else:
        attn = parallel_attention(
            self.hybrid_seq_parallel_attn,
            q, k, v,
            img_q_len=img_q.shape[1],
            img_kv_len=img_k.shape[1],
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
        )

    img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1]:]

    img_attn_branch = self.img_attn_proj(img_attn)
    if save_this_layer:
        cache.save_anchor(f"double_stream.{layer}.img_attn", img_attn_branch)
    img = img + apply_gate(img_attn_branch, gate=img_mod1_gate)

    img_mlp_branch = self.img_mlp(modulate(self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale))
    if save_this_layer:
        cache.save_anchor(f"double_stream.{layer}.img_mlp", img_mlp_branch)
    img = img + apply_gate(img_mlp_branch, gate=img_mod2_gate)

    txt_attn_branch = self.txt_attn_proj(txt_attn)
    if save_this_layer:
        cache.save_anchor(f"double_stream.{layer}.txt_attn", txt_attn_branch)
    txt = txt + apply_gate(txt_attn_branch, gate=txt_mod1_gate)

    txt_mlp_branch = self.txt_mlp(modulate(self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale))
    if save_this_layer:
        cache.save_anchor(f"double_stream.{layer}.txt_mlp", txt_mlp_branch)
    txt = txt + apply_gate(txt_mlp_branch, gate=txt_mod2_gate)

    return img, txt


def bary_single_forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
) -> torch.Tensor:
    cache: Optional[BaryCache] = getattr(self, "cache", None)
    layer: int = int(getattr(self, "_bary_layer", -1))

    mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)

    skip_this_layer = (
            cache is not None
            and cache.mode_cache == "blockwise"
            and (not cache.full_forward)
            and 0 <= layer < int(getattr(cache, "skip_upto_single", 0))
    )
    save_this_layer = (
            cache is not None
            and cache.mode_cache == "blockwise"
            and cache.full_forward
            and 0 <= layer < int(getattr(cache, "skip_upto_single", 0))
    )

    if skip_this_layer:
        try:
            k_total = f"single_stream.{layer}.total"
            out_branch = cache.predict(k_total, ref=x)
            return x + apply_gate(out_branch, gate=mod_gate)
        except KeyError:
            pass

    x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)
    qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
    q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)

    q = self.q_norm(q).to(v)
    k = self.k_norm(k).to(v)

    if freqs_cis is not None:
        img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
        img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
        img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
        q = torch.cat((img_qq, txt_q), dim=1)
        k = torch.cat((img_kk, txt_k), dim=1)

    if not self.hybrid_seq_parallel_attn:
        attn = attention(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            batch_size=x.shape[0],
        )
    else:
        attn = parallel_attention(
            self.hybrid_seq_parallel_attn,
            q, k, v,
            img_q_len=(q[:, :-txt_len].shape[1] if txt_len > 0 else q.shape[1]),
            img_kv_len=(k[:, :-txt_len].shape[1] if txt_len > 0 else k.shape[1]),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
        )

    out_branch = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
    if save_this_layer:
        cache.save_anchor(f"single_stream.{layer}.total", out_branch)

    return x + apply_gate(out_branch, gate=mod_gate)


def bary_forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_states: torch.Tensor = None,
        text_mask: torch.Tensor = None,
        text_states_2: Optional[torch.Tensor] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        guidance: torch.Tensor = None,
        return_dict: bool = True,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    cache: Optional[BaryCache] = getattr(self, "cache", None)

    if cache is None:
        out = {}
        img = x
        txt = text_states
        _, _, ot, oh, ow = x.shape
        tt, th, tw = (ot // self.patch_size[0], oh // self.patch_size[1], ow // self.patch_size[2])

        vec = self.time_in(t)
        vec = vec + self.vector_in(text_states_2)
        if self.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(guidance)

        img = self.img_in(img)
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]
        cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
        cu_seqlens_kv = cu_seqlens_q
        max_seqlen_q = img_seq_len + txt_seq_len
        max_seqlen_kv = max_seqlen_q
        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

        for i, block in enumerate(self.double_blocks):
            setattr(block, "_bary_layer", i)
            img, txt = block(img, txt, vec, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, freqs_cis)

        x_cat = torch.cat((img, txt), 1)
        if len(self.single_blocks) > 0:
            for i, block in enumerate(self.single_blocks):
                setattr(block, "_bary_layer", i)
                x_cat = block(x_cat, vec, txt_seq_len, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv,
                              (freqs_cos, freqs_sin))

        img = x_cat[:, :img_seq_len, ...]
        img = self.final_layer(img, vec)
        img = self.unpatchify(img, tt, th, tw)
        if return_dict:
            out["x"] = img
            return out
        return img

    if cache.mode_schedule == "eq-space":
        cache.solve_type(probe=None)
    else:
        vec_probe = self.time_in(t)
        vec_probe = vec_probe + self.vector_in(text_states_2)
        if self.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec_probe = vec_probe + self.guidance_in(guidance)

        img_probe = self.img_in(x)
        blk0 = self.double_blocks[0]
        img_mod1_shift, img_mod1_scale, *_ = blk0.img_mod(vec_probe).chunk(6, dim=-1)
        probe = modulate(blk0.img_norm1(img_probe), shift=img_mod1_shift, scale=img_mod1_scale)
        cache.solve_type(probe=probe)

    if cache.mode_cache == "stepwise" and (not cache.full_forward) and (not cache.has_history("stepwise")):
        cache.full_forward = True

    if cache.mode_cache == "stepwise" and (not cache.full_forward):
        pred = cache.predict("stepwise", ref=x)
        cache.step()
        if return_dict:
            return {"x": pred}
        return pred
    out = {}
    img = x
    txt = text_states
    _, _, ot, oh, ow = x.shape
    tt, th, tw = (ot // self.patch_size[0], oh // self.patch_size[1], ow // self.patch_size[2])

    vec = self.time_in(t)
    vec = vec + self.vector_in(text_states_2)

    if self.guidance_embed:
        if guidance is None:
            raise ValueError("Didn't get guidance strength for guidance distilled model.")
        vec = vec + self.guidance_in(guidance)

    img = self.img_in(img)
    if self.text_projection == "linear":
        txt = self.txt_in(txt)
    elif self.text_projection == "single_refiner":
        txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
    else:
        raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")

    txt_seq_len = txt.shape[1]
    img_seq_len = img.shape[1]

    cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
    cu_seqlens_kv = cu_seqlens_q
    max_seqlen_q = img_seq_len + txt_seq_len
    max_seqlen_kv = max_seqlen_q

    freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
    if cache.mode_cache == "blockwise":
        cache.skip_upto_double = int(cache.cache_ratio * len(self.double_blocks))
        cache.skip_upto_single = int(cache.cache_ratio * len(self.single_blocks))
    else:
        cache.skip_upto_double = 0
        cache.skip_upto_single = 0

    for i, block in enumerate(self.double_blocks):
        setattr(block, "_bary_layer", i)
        img, txt = block(img, txt, vec, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, freqs_cis)

    x_cat = torch.cat((img, txt), 1)
    if len(self.single_blocks) > 0:
        for i, block in enumerate(self.single_blocks):
            setattr(block, "_bary_layer", i)
            x_cat = block(x_cat, vec, txt_seq_len, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv,
                          (freqs_cos, freqs_sin))

    img = x_cat[:, :img_seq_len, ...]

    img = self.final_layer(img, vec)
    img = self.unpatchify(img, tt, th, tw)

    if cache.mode_cache == "stepwise" and cache.full_forward:
        cache.save_anchor("stepwise", img)

    cache.step()

    if return_dict:
        out["x"] = img
        return out
    return img


@contextlib.contextmanager
def suppress_output():
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
            logger.disable("")
            yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        logger.enable("")


def main():
    args = parse_args()
    print(args)
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    save_path = args.save_path if args.save_path_suffix == "" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    args = hunyuan_video_sampler.args
    hunyuan_video_sampler.pipeline.transformer.__class__.cnt = 0
    hunyuan_video_sampler.pipeline.transformer.__class__.num_steps = args.infer_steps
    transformer = hunyuan_video_sampler.pipeline.transformer
    transformer.__class__.forward = bary_forward
    transformer.double_blocks[0].__class__.forward = bary_double_forward
    transformer.single_blocks[0].__class__.forward = bary_single_forward

    cache_mode = getattr(args, "cache_mode", "stepwise")
    cache_schedule_mode = getattr(args, "cache_schedule_mode", "eq-space")
    cache_interval = int(getattr(args, "cache_interval", 7))
    cache_ratio = float(getattr(args, "cache_ratio", 0.0))
    rel_l1_thresh = float(getattr(args, "rel_l1_thresh", 0.2))
    max_history = int(getattr(args, "max_history", 1))
    blend_factor = getattr(args, "blend_factor", 0.8)
    blend_factor = None if blend_factor is None else float(blend_factor)

    fresh_start = int(getattr(args, "fresh_start", 1))
    fresh_end = int(getattr(args, "fresh_end", 0))

    def reset_bary_cache():
        shared_cache = BaryCache(num_steps=int(args.infer_steps))
        shared_cache.mode_schedule = str(cache_schedule_mode)
        shared_cache.fresh_start = int(fresh_start)
        shared_cache.fresh_end = int(fresh_end)
        shared_cache.fresh_interval = int(cache_interval)
        shared_cache.rel_l1_thresh = float(rel_l1_thresh)
        shared_cache.mode_cache = str(cache_mode)
        shared_cache.cache_ratio = float(cache_ratio)
        shared_cache.max_history = int(max_history)
        shared_cache.blend_factor = blend_factor
        transformer.cache = shared_cache
        for blk in transformer.double_blocks:
            blk.cache = shared_cache
        for blk in transformer.single_blocks:
            blk.cache = shared_cache

        return shared_cache

    prompt_text = "Your prompt."
    reset_bary_cache()
    with suppress_output():
        outputs = hunyuan_video_sampler.predict(
            prompt=prompt_text,
            height=args.video_size[0],
            width=args.video_size[1],
            video_length=args.video_length,
            seed=args.seed,
            negative_prompt=args.neg_prompt,
            infer_steps=args.infer_steps,
            guidance_scale=args.cfg_scale,
            num_videos_per_prompt=args.num_videos,
            flow_shift=args.flow_shift,
            batch_size=args.batch_size,
            embedded_guidance_scale=args.embedded_cfg_scale
        )
    samples = outputs['samples']
    if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
        for i, sample in enumerate(samples):
            sample = samples[i].unsqueeze(0)
            time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
            cur_save_path = f"{save_path}/{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/', '')}.mp4"
            save_videos_grid(sample, cur_save_path, fps=24)
            logger.info(f'Sample save to: {cur_save_path}')


if __name__ == "__main__":
    main()
