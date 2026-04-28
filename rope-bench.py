# Copyright (c) 2025, Tri Dao.
# As of 2025-04-23, we require triton >= 3.0

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Union

import torch

import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_gpt_parcae as pg

FUSED_BLOCK_H = 2
FUSED_BLOCK_M = 0


@triton.jit
def rotary_kernel(
    OUT,  # Pointers to matrices
    X,
    COS,
    SIN,
    CU_SEQLENS,
    SEQLEN_OFFSETS,  # this could be int or a pointer
    # Matrix dimensions
    seqlen,
    nheads,
    seqlen_ro,
    # strides
    stride_out_batch,
    stride_out_seqlen,
    stride_out_nheads,
    stride_out_headdim,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_nheads,
    stride_x_headdim,
    # Meta-parameters
    # We want ROTARY_DIM to be constexpr, otherwise the triton compiler doesn't know that
    # the mask is constant every 8 elements, and it will generate LDG.16 instead of LDG.128
    ROTARY_DIM: tl.constexpr,
    IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    BLOCK_K: tl.constexpr = triton.next_power_of_2(ROTARY_DIM)
    ROTARY_DIM_HALF = ROTARY_DIM // 2
    pid_head = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_batch = tl.program_id(axis=2)

    if not IS_VARLEN:
        X = X + pid_batch * stride_x_batch
        OUT = OUT + pid_batch * stride_out_batch
    else:
        start_idx = tl.load(CU_SEQLENS + pid_batch)
        seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
        X = X + start_idx * stride_x_seqlen
        OUT = OUT + start_idx * stride_out_seqlen

    if pid_m * BLOCK_M >= seqlen:
        return

    rh = pid_head * BLOCK_H + tl.arange(0, BLOCK_H)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if not IS_SEQLEN_OFFSETS_TENSOR:
        rm_cs = rm + SEQLEN_OFFSETS
    else:
        rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)

    rk_half = tl.arange(0, BLOCK_K // 2)
    COS = COS + (rm_cs[:, None] * ROTARY_DIM_HALF + rk_half[None, :])
    SIN = SIN + (rm_cs[:, None] * ROTARY_DIM_HALF + rk_half[None, :])
    mask_cs = (rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < ROTARY_DIM_HALF)
    cos = tl.load(COS, mask=mask_cs, other=1.0).to(tl.float32)
    sin = tl.load(SIN, mask=mask_cs, other=0.0).to(tl.float32)
    if CONJUGATE:
        sin = -sin

    if not INTERLEAVED:
        # Load the 1st and 2nd halves of X, do calculation, then store to 1st and 2nd halves of OUT
        X = X + (rh[:, None, None] * stride_x_nheads + rm[None, :, None] * stride_x_seqlen + rk_half[None, None, :] * stride_x_headdim)
        OUT = OUT + (rh[:, None, None] * stride_out_nheads + rm[None, :, None] * stride_out_seqlen + rk_half[None, None, :] * stride_out_headdim)
        mask = (rh[:, None, None] < nheads) & (rm[None, :, None] < seqlen) & (rk_half[None, None, :] < ROTARY_DIM_HALF)
        x0 = tl.load(X, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(X + ROTARY_DIM_HALF * stride_x_headdim, mask=mask, other=0.0,).to(tl.float32)
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        tl.store(OUT, o0, mask=mask)
        tl.store(OUT + ROTARY_DIM_HALF * stride_out_headdim, o1, mask=mask)
    else:
        rk = tl.arange(0, BLOCK_K)
        X = X + (rh[:, None, None] * stride_x_nheads + rm[None, :, None] * stride_x_seqlen + rk[None, None, :] * stride_x_headdim)
        OUT = OUT + (rh[:, None, None] * stride_out_nheads + rm[None, :, None] * stride_out_seqlen + rk[None, None, :] * stride_out_headdim)
        mask = (rh[:, None, None] < nheads) & (rm[None, :, None] < seqlen) & (rk[None, None, :] < ROTARY_DIM)
        x = tl.load(X, mask=mask, other=0.0).to(tl.float32)
        x0, x1 = tl.split(tl.reshape(x, [BLOCK_H, BLOCK_M, BLOCK_K // 2, 2]))
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        o = tl.reshape(tl.join(o0, o1), [BLOCK_H, BLOCK_M, BLOCK_K])
        tl.store(OUT, o, mask=mask)


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    interleaved=False,
    inplace=False,
    conjugate=False,
) -> torch.Tensor:
    """
    Arguments:
        x: (batch, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim).
        cos: (seqlen_ro, rotary_dim / 2)
        sin: (seqlen_ro, rotary_dim / 2)
        seqlen_offsets: integer or integer tensor of size (batch,)
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Returns:
        y: (batch, seqlen, nheads, headdim)
    """
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        assert max_seqlen is not None, "If cu_seqlens is passed in, then max_seqlen must be passed"
        total_seqlen, nheads, headdim = x.shape
        batch_p_1 = cu_seqlens.shape[0]
        batch = batch_p_1 - 1
        seqlen = max_seqlen
    seqlen_ro, rotary_dim = cos.shape
    assert sin.shape == cos.shape
    rotary_dim *= 2
    assert rotary_dim <= headdim, "rotary_dim must be <= headdim"
    assert headdim <= 256, "Only support headdim <= 256"
    assert seqlen_ro >= seqlen, "seqlen_ro must be >= seqlen"

    cos, sin = cos.contiguous(), sin.contiguous()
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (batch,)
        assert seqlen_offsets.dtype in [torch.int32, torch.int64]
        seqlen_offsets = seqlen_offsets.contiguous()
    else:
        assert seqlen_offsets + seqlen <= seqlen_ro

    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and not inplace:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])

    grid = lambda META: (triton.cdiv(nheads, META["BLOCK_H"]), triton.cdiv(seqlen, META["BLOCK_M"]), batch)  # noqa
    BLOCK_M = 8 if rotary_dim <= 128 else 4

    # Need this, otherwise Triton tries to launch from cuda:0 and we get
    # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
    with torch.cuda.device(x.device.index):
        torch.library.wrap_triton(rotary_kernel)[grid](
            output,  # data ptrs
            x,
            cos,
            sin,
            cu_seqlens,
            seqlen_offsets,
            seqlen,  # shapes
            nheads,
            seqlen_ro,
            output.stride(0) if not is_varlen else 0,  # batch_strides if not varlen else 0
            output.stride(-3),  # seqlen_stride or total_seqlen_stride
            output.stride(-2),  # nheads_stride
            output.stride(-1),  # headdim_stride
            x.stride(0) if not is_varlen else 0,  # batch_strides if not varlen else 0
            x.stride(-3),  # seqlen stride or total_seqlen_stride
            x.stride(-2),  # nheads stride
            x.stride(-1),  # headdim stride
            rotary_dim,
            isinstance(seqlen_offsets, torch.Tensor),
            is_varlen,
            interleaved,
            conjugate,
            BLOCK_M=BLOCK_M,
            BLOCK_H=2,
        )
    return output


def _apply_rotary_emb_qkv(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    interleaved: bool = False,
    inplace: bool = False,
    conjugate: bool = False,
    num_heads_q: Optional[int] = None,
) -> torch.Tensor:
    apply_rotary_fn = lambda x: apply_rotary(
        x,
        cos,
        sin,
        interleaved=interleaved,
        inplace=inplace,
        conjugate=conjugate,
    )
    if qkv.is_contiguous():
        if qkv.dim() == 5:
            batch, seqlen, three, nheads, headdim = qkv.shape
            assert three == 3
            qk = qkv[:, :, :2].reshape(batch, seqlen, -1, headdim)
            qk = apply_rotary_fn(qk)
            if not inplace:
                qkv = torch.cat([qk.reshape(batch, seqlen, 2, nheads, headdim), qkv[:, :, 2:]], dim=2)
        else:
            assert qkv.dim() == 4
            assert num_heads_q is not None
            num_heads_k = (qkv.shape[2] - num_heads_q) // 2
            assert qkv.shape[2] == num_heads_q + 2 * num_heads_k
            qk = qkv[:, :, : num_heads_q + num_heads_k]
            qk = apply_rotary_fn(qk)
            if not inplace:
                qkv = torch.cat([qk, qkv[:, :, num_heads_q + num_heads_k :]], dim=2)
    else:
        if qkv.dim() == 5:
            q = qkv[:, :, 0]
            k = qkv[:, :, 1]
            q = apply_rotary_fn(q)
            k = apply_rotary_fn(k)
            if not inplace:
                qkv = torch.stack([q, k, qkv[:, :, 2]], dim=2)
        else:
            assert qkv.dim() == 4
            assert num_heads_q is not None
            num_heads_k = (qkv.shape[2] - num_heads_q) // 2
            assert qkv.shape[2] == num_heads_q + 2 * num_heads_k
            q = qkv[:, :, :num_heads_q]
            k = qkv[:, :, num_heads_q : num_heads_q + num_heads_k]
            q = apply_rotary_fn(q)
            k = apply_rotary_fn(k)
            if not inplace:
                qkv = torch.cat([q, k, qkv[:, :, num_heads_q + num_heads_k :]], dim=2)
    return qkv


class ApplyRotaryEmbQKV(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        interleaved: bool,
        num_heads_q: int,
    ):
        ctx.save_for_backward(cos, sin)
        ctx.interleaved = interleaved
        ctx.num_heads_q = num_heads_q
        return _apply_rotary_emb_qkv(
            qkv,
            cos,
            sin,
            interleaved=interleaved,
            inplace=True,
            num_heads_q=num_heads_q,
        )

    @staticmethod
    def backward(ctx, dqkv: torch.Tensor):
        cos, sin = ctx.saved_tensors
        dqkv = _apply_rotary_emb_qkv(
            dqkv.contiguous(),
            cos,
            sin,
            interleaved=ctx.interleaved,
            inplace=True,
            conjugate=True,
            num_heads_q=ctx.num_heads_q,
        )
        return dqkv, None, None, None, None


@triton.jit
def fused_qk_rotary_kernel(
    OUT_Q,
    Q,
    OUT_K,
    K,
    COS,
    SIN,
    seqlen,
    n_qheads,
    n_kheads,
    seqlen_ro,
    stride_oq_batch,
    stride_oq_seqlen,
    stride_oq_nheads,
    stride_oq_headdim,
    stride_q_batch,
    stride_q_seqlen,
    stride_q_nheads,
    stride_q_headdim,
    stride_ok_batch,
    stride_ok_seqlen,
    stride_ok_nheads,
    stride_ok_headdim,
    stride_k_batch,
    stride_k_seqlen,
    stride_k_nheads,
    stride_k_headdim,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    CONJUGATE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    BLOCK_K: tl.constexpr = triton.next_power_of_2(ROTARY_DIM)
    ROTARY_DIM_HALF: tl.constexpr = ROTARY_DIM // 2
    pid_head = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_batch = tl.program_id(axis=2)

    rh = pid_head * BLOCK_H + tl.arange(0, BLOCK_H)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if pid_m * BLOCK_M >= seqlen:
        return

    rk_half = tl.arange(0, BLOCK_K // 2)
    cos_offsets = rm[:, None] * ROTARY_DIM_HALF + rk_half[None, :]
    cos_mask = (rm[:, None] < seqlen_ro) & (rk_half[None, :] < ROTARY_DIM_HALF)
    cos = tl.load(COS + cos_offsets, mask=cos_mask, other=1.0).to(tl.float32)
    sin = tl.load(SIN + cos_offsets, mask=cos_mask, other=0.0).to(tl.float32)
    if CONJUGATE:
        sin = -sin

    rk = tl.arange(0, BLOCK_K)
    q_offsets = (
        pid_batch * stride_q_batch
        + rh[:, None, None] * stride_q_nheads
        + rm[None, :, None] * stride_q_seqlen
        + rk[None, None, :] * stride_q_headdim
    )
    oq_offsets = (
        pid_batch * stride_oq_batch
        + rh[:, None, None] * stride_oq_nheads
        + rm[None, :, None] * stride_oq_seqlen
        + rk[None, None, :] * stride_oq_headdim
    )
    q_mask = (rh[:, None, None] < n_qheads) & (rm[None, :, None] < seqlen) & (rk[None, None, :] < ROTARY_DIM)
    q = tl.load(Q + q_offsets, mask=q_mask, other=0.0).to(tl.float32)
    q0, q1 = tl.split(tl.reshape(q, [BLOCK_H, BLOCK_M, BLOCK_K // 2, 2]))
    oq0 = q0 * cos - q1 * sin
    oq1 = q0 * sin + q1 * cos
    oq = tl.reshape(tl.join(oq0, oq1), [BLOCK_H, BLOCK_M, BLOCK_K])
    tl.store(OUT_Q + oq_offsets, oq, mask=q_mask)

    k_offsets = (
        pid_batch * stride_k_batch
        + rh[:, None, None] * stride_k_nheads
        + rm[None, :, None] * stride_k_seqlen
        + rk[None, None, :] * stride_k_headdim
    )
    ok_offsets = (
        pid_batch * stride_ok_batch
        + rh[:, None, None] * stride_ok_nheads
        + rm[None, :, None] * stride_ok_seqlen
        + rk[None, None, :] * stride_ok_headdim
    )
    k_mask = (rh[:, None, None] < n_kheads) & (rm[None, :, None] < seqlen) & (rk[None, None, :] < ROTARY_DIM)
    k = tl.load(K + k_offsets, mask=k_mask, other=0.0).to(tl.float32)
    k0, k1 = tl.split(tl.reshape(k, [BLOCK_H, BLOCK_M, BLOCK_K // 2, 2]))
    ok0 = k0 * cos - k1 * sin
    ok1 = k0 * sin + k1 * cos
    ok = tl.reshape(tl.join(ok0, ok1), [BLOCK_H, BLOCK_M, BLOCK_K])
    tl.store(OUT_K + ok_offsets, ok, mask=k_mask)

    if ROTARY_DIM < HEAD_DIM:
        BLOCK_TAIL: tl.constexpr = triton.next_power_of_2(HEAD_DIM - ROTARY_DIM)
        rt = ROTARY_DIM + tl.arange(0, BLOCK_TAIL)
        tail_q_offsets = (
            pid_batch * stride_q_batch
            + rh[:, None, None] * stride_q_nheads
            + rm[None, :, None] * stride_q_seqlen
            + rt[None, None, :] * stride_q_headdim
        )
        tail_oq_offsets = (
            pid_batch * stride_oq_batch
            + rh[:, None, None] * stride_oq_nheads
            + rm[None, :, None] * stride_oq_seqlen
            + rt[None, None, :] * stride_oq_headdim
        )
        tail_q_mask = (rh[:, None, None] < n_qheads) & (rm[None, :, None] < seqlen) & (rt[None, None, :] < HEAD_DIM)
        q_tail = tl.load(Q + tail_q_offsets, mask=tail_q_mask, other=0.0)
        tl.store(OUT_Q + tail_oq_offsets, q_tail, mask=tail_q_mask)

        tail_k_offsets = (
            pid_batch * stride_k_batch
            + rh[:, None, None] * stride_k_nheads
            + rm[None, :, None] * stride_k_seqlen
            + rt[None, None, :] * stride_k_headdim
        )
        tail_ok_offsets = (
            pid_batch * stride_ok_batch
            + rh[:, None, None] * stride_ok_nheads
            + rm[None, :, None] * stride_ok_seqlen
            + rt[None, None, :] * stride_ok_headdim
        )
        tail_k_mask = (rh[:, None, None] < n_kheads) & (rm[None, :, None] < seqlen) & (rt[None, None, :] < HEAD_DIM)
        k_tail = tl.load(K + tail_k_offsets, mask=tail_k_mask, other=0.0)
        tl.store(OUT_K + tail_ok_offsets, k_tail, mask=tail_k_mask)


def apply_fused_qk_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    conjugate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seqlen, n_qheads, head_dim = q.shape
    n_kheads = k.shape[2]
    seqlen_ro, rotary_dim_half = cos.shape
    rotary_dim = rotary_dim_half * 2
    if k.shape[:2] != q.shape[:2] or k.shape[-1] != head_dim:
        raise ValueError(f"q/k shape mismatch: q={tuple(q.shape)} k={tuple(k.shape)}")
    if rotary_dim > head_dim:
        raise ValueError(f"rotary_dim={rotary_dim} must be <= head_dim={head_dim}")
    out_q = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    out_k = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    grid = lambda META: (
        triton.cdiv(max(n_qheads, n_kheads), META["BLOCK_H"]),
        triton.cdiv(seqlen, META["BLOCK_M"]),
        batch,
    )
    block_m = FUSED_BLOCK_M or (8 if rotary_dim <= 128 else 4)
    with torch.cuda.device(q.device.index):
        torch.library.wrap_triton(fused_qk_rotary_kernel)[grid](
            out_q,
            q,
            out_k,
            k,
            cos.contiguous(),
            sin.contiguous(),
            seqlen,
            n_qheads,
            n_kheads,
            seqlen_ro,
            out_q.stride(0),
            out_q.stride(1),
            out_q.stride(2),
            out_q.stride(3),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            out_k.stride(0),
            out_k.stride(1),
            out_k.stride(2),
            out_k.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            head_dim,
            rotary_dim,
            conjugate,
            BLOCK_H=FUSED_BLOCK_H,
            BLOCK_M=block_m,
        )
    return out_q, out_k


class ApplyRotary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool):
        ctx.save_for_backward(cos, sin)
        ctx.interleaved = interleaved
        return apply_rotary(x, cos, sin, interleaved=interleaved)

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        cos, sin = ctx.saved_tensors
        dx = apply_rotary(dout.contiguous(), cos, sin, interleaved=ctx.interleaved, conjugate=True)
        return dx, None, None, None


def tridao_interleaved_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    rope_dims: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = freqs_cos[0, :, 0, : rope_dims // 2].contiguous()
    sin = freqs_sin[0, :, 0, : rope_dims // 2].contiguous()
    return (
        ApplyRotary.apply(q, cos, sin, True),
        ApplyRotary.apply(k, cos, sin, True),
    )


class InplaceContiguousRotary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        ctx.save_for_backward(cos, sin)
        q_out = q.contiguous()
        k_out = k.contiguous()
        apply_rotary(q_out, cos, sin, interleaved=True, inplace=True)
        apply_rotary(k_out, cos, sin, interleaved=True, inplace=True)
        return q_out, k_out

    @staticmethod
    def backward(ctx, dq: torch.Tensor, dk: torch.Tensor):
        cos, sin = ctx.saved_tensors
        dq = dq.contiguous()
        dk = dk.contiguous()
        apply_rotary(dq, cos, sin, interleaved=True, inplace=True, conjugate=True)
        apply_rotary(dk, cos, sin, interleaved=True, inplace=True, conjugate=True)
        return dq, dk, None, None


def tridao_inplace_contiguous_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    rope_dims: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = freqs_cos[0, :, 0, : rope_dims // 2].contiguous()
    sin = freqs_sin[0, :, 0, : rope_dims // 2].contiguous()
    return InplaceContiguousRotary.apply(q, k, cos, sin)


class FusedQKRotary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        ctx.save_for_backward(cos, sin)
        return apply_fused_qk_rotary(q, k, cos, sin)

    @staticmethod
    def backward(ctx, dq: torch.Tensor, dk: torch.Tensor):
        cos, sin = ctx.saved_tensors
        return (*apply_fused_qk_rotary(dq.contiguous(), dk.contiguous(), cos, sin, conjugate=True), None, None)


def fused_qk_interleaved_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    rope_dims: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = freqs_cos[0, :, 0, : rope_dims // 2].contiguous()
    sin = freqs_sin[0, :, 0, : rope_dims // 2].contiguous()
    return FusedQKRotary.apply(q, k, cos, sin)


def pytorch_packed_rope(
    qkv: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    rope_dims: int,
    q_heads: int,
    kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = qkv[:, :, :q_heads]
    k = qkv[:, :, q_heads : q_heads + kv_heads]
    return pg.apply_rotary_emb_complex_like(q, k, freqs_cos, freqs_sin, rope_dims)


def tridao_packed_qkv_rope(
    qkv: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    rope_dims: int,
    q_heads: int,
    kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del kv_heads
    cos = freqs_cos[0, :, 0, : rope_dims // 2].contiguous()
    sin = freqs_sin[0, :, 0, : rope_dims // 2].contiguous()
    qkv = ApplyRotaryEmbQKV.apply(qkv, cos, sin, True, q_heads)
    return qkv[:, :, :q_heads], qkv[:, :, q_heads : q_heads + (qkv.shape[2] - q_heads) // 2]


def _bench(
    name: str,
    fn,
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    rope_dims: int,
    iters: int,
    warmup: int,
    backward: bool,
) -> tuple[float, float]:
    for _ in range(warmup):
        qi = q.detach().clone().requires_grad_(True) if backward else q
        ki = k.detach().clone().requires_grad_(True) if backward else k
        qo, ko = fn(qi, ki, freqs_cos, freqs_sin, rope_dims)
        if backward:
            (qo.float().square().mean() + ko.float().square().mean()).backward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        qi = q.detach().clone().requires_grad_(True) if backward else q
        ki = k.detach().clone().requires_grad_(True) if backward else k
        qo, ko = fn(qi, ki, freqs_cos, freqs_sin, rope_dims)
        if backward:
            (qo.float().square().mean() + ko.float().square().mean()).backward()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    peak_mib = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"{name}: {ms:.4f} ms/iter peak_alloc:{peak_mib:.1f} MiB")
    return ms, peak_mib


def _bench_packed(
    name: str,
    fn,
    qkv: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    rope_dims: int,
    q_heads: int,
    kv_heads: int,
    iters: int,
    warmup: int,
    backward: bool,
    mutates: bool = False,
) -> tuple[float, float]:
    forward_qkv = qkv.detach().clone() if mutates and not backward else qkv
    for _ in range(warmup):
        qkvi = qkv.detach().clone().requires_grad_(True) if backward else forward_qkv
        qo, ko = fn(qkvi, freqs_cos, freqs_sin, rope_dims, q_heads, kv_heads)
        if backward:
            (qo.float().square().mean() + ko.float().square().mean()).backward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        qkvi = qkv.detach().clone().requires_grad_(True) if backward else forward_qkv
        qo, ko = fn(qkvi, freqs_cos, freqs_sin, rope_dims, q_heads, kv_heads)
        if backward:
            (qo.float().square().mean() + ko.float().square().mean()).backward()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    peak_mib = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"{name}: {ms:.4f} ms/iter peak_alloc:{peak_mib:.1f} MiB")
    return ms, peak_mib


def _check_correctness(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    rope_dims: int,
) -> None:
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    q_tri = q.detach().clone().requires_grad_(True)
    k_tri = k.detach().clone().requires_grad_(True)
    q_fused = q.detach().clone().requires_grad_(True)
    k_fused = k.detach().clone().requires_grad_(True)
    q_inplace = q.detach().clone().requires_grad_(True)
    k_inplace = k.detach().clone().requires_grad_(True)

    ref_q, ref_k = pg.apply_rotary_emb_complex_like(q_ref, k_ref, freqs_cos, freqs_sin, rope_dims)
    tri_q, tri_k = tridao_interleaved_rope(q_tri, k_tri, freqs_cos, freqs_sin, rope_dims)
    fused_q, fused_k = fused_qk_interleaved_rope(q_fused, k_fused, freqs_cos, freqs_sin, rope_dims)
    inplace_q, inplace_k = tridao_inplace_contiguous_rope(q_inplace, k_inplace, freqs_cos, freqs_sin, rope_dims)
    ref_loss = ref_q.float().square().mean() + ref_k.float().square().mean()
    tri_loss = tri_q.float().square().mean() + tri_k.float().square().mean()
    fused_loss = fused_q.float().square().mean() + fused_k.float().square().mean()
    inplace_loss = inplace_q.float().square().mean() + inplace_k.float().square().mean()
    ref_loss.backward()
    tri_loss.backward()
    fused_loss.backward()
    inplace_loss.backward()

    print(
        "tridao_correctness "
        f"forward_maxdiff_q:{(ref_q - tri_q).abs().max().item():.6g} "
        f"forward_maxdiff_k:{(ref_k - tri_k).abs().max().item():.6g} "
        f"grad_maxdiff_q:{(q_ref.grad - q_tri.grad).abs().max().item():.6g} "
        f"grad_maxdiff_k:{(k_ref.grad - k_tri.grad).abs().max().item():.6g}"
    )
    print(
        "fused_qk_correctness "
        f"forward_maxdiff_q:{(ref_q - fused_q).abs().max().item():.6g} "
        f"forward_maxdiff_k:{(ref_k - fused_k).abs().max().item():.6g} "
        f"grad_maxdiff_q:{(q_ref.grad - q_fused.grad).abs().max().item():.6g} "
        f"grad_maxdiff_k:{(k_ref.grad - k_fused.grad).abs().max().item():.6g}"
    )
    print(
        "inplace_contig_correctness "
        f"forward_maxdiff_q:{(ref_q - inplace_q).abs().max().item():.6g} "
        f"forward_maxdiff_k:{(ref_k - inplace_k).abs().max().item():.6g} "
        f"grad_maxdiff_q:{(q_ref.grad - q_inplace.grad).abs().max().item():.6g} "
        f"grad_maxdiff_k:{(k_ref.grad - k_inplace.grad).abs().max().item():.6g}"
    )

    if rope_dims < q.shape[-1]:
        q_tail_diff = (tri_q[..., rope_dims:] - q[..., rope_dims:]).abs().max().item()
        k_tail_diff = (tri_k[..., rope_dims:] - k[..., rope_dims:]).abs().max().item()
        fused_q_tail_diff = (fused_q[..., rope_dims:] - q[..., rope_dims:]).abs().max().item()
        fused_k_tail_diff = (fused_k[..., rope_dims:] - k[..., rope_dims:]).abs().max().item()
        inplace_q_tail_diff = (inplace_q[..., rope_dims:] - q[..., rope_dims:]).abs().max().item()
        inplace_k_tail_diff = (inplace_k[..., rope_dims:] - k[..., rope_dims:]).abs().max().item()
        print(
            f"tridao_partial_tail_maxdiff_q:{q_tail_diff:.6g} "
            f"tridao_partial_tail_maxdiff_k:{k_tail_diff:.6g}"
        )
        print(
            f"fused_partial_tail_maxdiff_q:{fused_q_tail_diff:.6g} "
            f"fused_partial_tail_maxdiff_k:{fused_k_tail_diff:.6g}"
        )
        print(
            f"inplace_partial_tail_maxdiff_q:{inplace_q_tail_diff:.6g} "
            f"inplace_partial_tail_maxdiff_k:{inplace_k_tail_diff:.6g}"
        )


def _check_packed_correctness(
    qkv: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    rope_dims: int,
    q_heads: int,
    kv_heads: int,
) -> None:
    qkv_ref = qkv.detach().clone().requires_grad_(True)
    qkv_tri = qkv.detach().clone().requires_grad_(True)

    ref_q, ref_k = pytorch_packed_rope(qkv_ref, freqs_cos, freqs_sin, rope_dims, q_heads, kv_heads)
    tri_q, tri_k = tridao_packed_qkv_rope(qkv_tri, freqs_cos, freqs_sin, rope_dims, q_heads, kv_heads)
    ref_loss = ref_q.float().square().mean() + ref_k.float().square().mean()
    tri_loss = tri_q.float().square().mean() + tri_k.float().square().mean()
    ref_loss.backward()
    tri_loss.backward()

    print(
        "packed_qkv_correctness "
        f"forward_maxdiff_q:{(ref_q - tri_q).abs().max().item():.6g} "
        f"forward_maxdiff_k:{(ref_k - tri_k).abs().max().item():.6g} "
        f"grad_maxdiff_qkv:{(qkv_ref.grad - qkv_tri.grad).abs().max().item():.6g}"
    )
    if rope_dims < qkv.shape[-1]:
        q_ref_tail = qkv[:, :, :q_heads, rope_dims:]
        k_ref_tail = qkv[:, :, q_heads : q_heads + kv_heads, rope_dims:]
        q_tail_diff = (tri_q[..., rope_dims:] - q_ref_tail).abs().max().item()
        k_tail_diff = (tri_k[..., rope_dims:] - k_ref_tail).abs().max().item()
        print(f"packed_qkv_partial_tail_maxdiff_q:{q_tail_diff:.6g} k:{k_tail_diff:.6g}")
    v_diff = (qkv_tri[:, :, q_heads + kv_heads :] - qkv[:, :, q_heads + kv_heads :]).abs().max().item()
    print(f"packed_qkv_v_untouched_maxdiff:{v_diff:.6g}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--q-heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=96)
    parser.add_argument("--rope-dims", type=int, default=32)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--backward-iters", type=int, default=100)
    parser.add_argument("--backward-warmup", type=int, default=10)
    parser.add_argument("--skip-backward", action="store_true")
    parser.add_argument("--contiguous-inputs", action="store_true")
    parser.add_argument("--fused-block-h", type=int, default=2)
    parser.add_argument("--fused-block-m", type=int, default=0)
    args = parser.parse_args()
    global FUSED_BLOCK_H, FUSED_BLOCK_M
    FUSED_BLOCK_H = args.fused_block_h
    FUSED_BLOCK_M = args.fused_block_m

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.rope_dims <= 0 or args.rope_dims > args.head_dim or args.rope_dims % 2 != 0:
        raise SystemExit("--rope-dims must be positive, even, and <= --head-dim")
    if args.head_dim > 256:
        raise SystemExit("Tri Dao rotary kernel supports head_dim <= 256")

    dtype = torch.bfloat16
    if args.contiguous_inputs:
        q = torch.randn(
            args.batch_size,
            args.seq_len,
            args.q_heads,
            args.head_dim,
            device="cuda",
            dtype=dtype,
        )
        k = torch.randn(
            args.batch_size,
            args.seq_len,
            args.kv_heads,
            args.head_dim,
            device="cuda",
            dtype=dtype,
        )
        qkv = torch.randn(
            args.batch_size,
            args.seq_len,
            args.q_heads + 2 * args.kv_heads,
            args.head_dim,
            device="cuda",
            dtype=dtype,
        )
    else:
        q_dim = args.q_heads * args.head_dim
        kv_dim = args.kv_heads * args.head_dim
        packed = torch.randn(
            args.batch_size,
            args.seq_len,
            q_dim + 2 * kv_dim,
            device="cuda",
            dtype=dtype,
        )
        qkv = packed.view(args.batch_size, args.seq_len, args.q_heads + 2 * args.kv_heads, args.head_dim)
        q = qkv[:, :, : args.q_heads]
        k = qkv[:, :, args.q_heads : args.q_heads + args.kv_heads]
    freqs_cos, freqs_sin = pg.precompute_freqs_cos_sin(args.rope_dims, args.seq_len)
    freqs_cos = freqs_cos.cuda()
    freqs_sin = freqs_sin.cuda()

    print(
        f"shape B={args.batch_size} T={args.seq_len} Hq={args.q_heads} "
        f"Hkv={args.kv_heads} D={args.head_dim} rope_dims={args.rope_dims} dtype={dtype}"
    )
    print(
        f"q_shape:{tuple(q.shape)} k_shape:{tuple(k.shape)} "
        f"q_stride:{q.stride()} k_stride:{k.stride()} "
        f"qkv_shape:{tuple(qkv.shape)} qkv_stride:{qkv.stride()} "
        f"freqs_shape:{tuple(freqs_cos.shape)} tridao_cos_shape:{(args.seq_len, args.rope_dims // 2)}"
    )
    _check_correctness(q, k, freqs_cos, freqs_sin, args.rope_dims)
    _check_packed_correctness(qkv, freqs_cos, freqs_sin, args.rope_dims, args.q_heads, args.kv_heads)
    _bench(
        "pytorch_adjacent_pair_rope_forward",
        pg.apply_rotary_emb_complex_like,
        q,
        k,
        freqs_cos,
        freqs_sin,
        args.rope_dims,
        args.iters,
        args.warmup,
        backward=False,
    )
    _bench_packed(
        "pytorch_packed_split_rope_forward",
        pytorch_packed_rope,
        qkv,
        freqs_cos,
        freqs_sin,
        args.rope_dims,
        args.q_heads,
        args.kv_heads,
        args.iters,
        args.warmup,
        backward=False,
    )
    _bench_packed(
        "tridao_packed_qkv_inplace_rope_forward",
        tridao_packed_qkv_rope,
        qkv,
        freqs_cos,
        freqs_sin,
        args.rope_dims,
        args.q_heads,
        args.kv_heads,
        args.iters,
        args.warmup,
        backward=False,
        mutates=True,
    )
    _bench(
        "tridao_interleaved_rope_forward",
        tridao_interleaved_rope,
        q,
        k,
        freqs_cos,
        freqs_sin,
        args.rope_dims,
        args.iters,
        args.warmup,
        backward=False,
    )
    _bench(
        "fused_qk_interleaved_rope_forward",
        fused_qk_interleaved_rope,
        q,
        k,
        freqs_cos,
        freqs_sin,
        args.rope_dims,
        args.iters,
        args.warmup,
        backward=False,
    )
    _bench(
        "tridao_inplace_contiguous_rope_forward",
        tridao_inplace_contiguous_rope,
        q,
        k,
        freqs_cos,
        freqs_sin,
        args.rope_dims,
        args.iters,
        args.warmup,
        backward=False,
    )
    if not args.skip_backward:
        _bench(
            "pytorch_adjacent_pair_rope_forward_backward",
            pg.apply_rotary_emb_complex_like,
            q,
            k,
            freqs_cos,
            freqs_sin,
            args.rope_dims,
            args.backward_iters,
            args.backward_warmup,
            backward=True,
        )
        _bench_packed(
            "pytorch_packed_split_rope_forward_backward",
            pytorch_packed_rope,
            qkv,
            freqs_cos,
            freqs_sin,
            args.rope_dims,
            args.q_heads,
            args.kv_heads,
            args.backward_iters,
            args.backward_warmup,
            backward=True,
        )
        _bench_packed(
            "tridao_packed_qkv_inplace_rope_forward_backward",
            tridao_packed_qkv_rope,
            qkv,
            freqs_cos,
            freqs_sin,
            args.rope_dims,
            args.q_heads,
            args.kv_heads,
            args.backward_iters,
            args.backward_warmup,
            backward=True,
            mutates=True,
        )
        _bench(
            "fused_qk_interleaved_rope_forward_backward",
            fused_qk_interleaved_rope,
            q,
            k,
            freqs_cos,
            freqs_sin,
            args.rope_dims,
            args.backward_iters,
            args.backward_warmup,
            backward=True,
        )
        _bench(
            "tridao_inplace_contiguous_rope_forward_backward",
            tridao_inplace_contiguous_rope,
            q,
            k,
            freqs_cos,
            freqs_sin,
            args.rope_dims,
            args.backward_iters,
            args.backward_warmup,
            backward=True,
        )
        _bench(
            "tridao_interleaved_rope_forward_backward",
            tridao_interleaved_rope,
            q,
            k,
            freqs_cos,
            freqs_sin,
            args.rope_dims,
            args.backward_iters,
            args.backward_warmup,
            backward=True,
        )


if __name__ == "__main__":
    main()
