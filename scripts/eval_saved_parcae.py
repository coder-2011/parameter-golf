from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_gpt_parcae as pg


def main() -> None:
    run_id = os.environ.get("RUN_ID", f"eval_saved_{int(time.time())}")
    default_model = (
        "final_model.mixed_int.ptz"
        if os.environ.get("MIXED_QUANT_BITS", "0") == "1"
        else f"final_model.int{os.environ.get('QUANT_BITS', '8')}.ptz"
    )
    model_path = Path(os.environ.get("MODEL_PATH", default_model))
    log_path = Path("logs") / f"{run_id}.txt"
    log_path.parent.mkdir(exist_ok=True)

    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            print(msg, file=f)

    args = pg.Hyperparameters()
    pg.signed_quant_max(args.quant_bits)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    (
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    ), tokenizer_metadata = pg.load_tokenizer_luts(
        args.tokenizer_path,
        args.tokenizer_meta_path,
        args.vocab_size,
        device,
        validate_meta=args.tokenizer_meta_validate,
    )
    token_bytes_lut = None
    if args.ppm_enabled or args.lzp_enabled:
        token_bytes_lut = pg.load_token_bytes_lut(args.tokenizer_path, tokenizer_metadata, args.vocab_size)

    val_tokens = pg.load_validation_tokens(args.val_files, args.train_seq_len)
    log(
        f"eval_saved:run_id={run_id} model={model_path} stride={args.eval_stride} "
        f"val_tokens={val_tokens.numel() - 1} sliding={int(args.sliding_window_enabled)} "
        f"lzp={int(args.lzp_enabled)} ppm={int(args.ppm_enabled)} ngram_order={args.ngram_eval_order} "
        f"ngram_chunk_tokens={args.ngram_chunk_tokens}"
    )

    model = pg.GPT(args).to(device).bfloat16()
    pg.restore_low_dim_params_to_fp32(model)
    with model_path.open("rb") as f:
        quant_state = pg.load_quant_artifact(f.read())
    model.load_state_dict(pg.dequantize_state_dict_int(quant_state), strict=True)
    model.eval()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if args.sliding_window_enabled:
        val_loss, val_bpb, context_result, ngram_result = pg.eval_val_sliding(
            args,
            model,
            0,
            1,
            device,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            token_bytes_lut,
            log_fn=log,
        )
        torch.cuda.synchronize()
        log(
            f"eval_saved_sliding_exact val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f} "
            f"stride:{args.eval_stride} eval_time:{time.perf_counter() - t0:.2f}s"
        )
        if context_result is not None:
            log(
                "eval_saved_context_exact "
                + " ".join(
                    f"{k}:{v:.8f}" if isinstance(v, float) else f"{k}:{v}"
                    for k, v in context_result.items()
                )
            )
        if ngram_result is not None:
            log(
                "eval_saved_ngram_exact "
                + " ".join(
                    f"{k}:{v:.8f}" if isinstance(v, float) else f"{k}:{v}"
                    for k, v in ngram_result.items()
                )
            )
    else:
        val_loss, val_bpb = pg.eval_val(
            args,
            model,
            0,
            1,
            device,
            1,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
        )
        torch.cuda.synchronize()
        log(
            f"eval_saved_exact val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f} "
            f"eval_time:{time.perf_counter() - t0:.2f}s"
        )


if __name__ == "__main__":
    main()
