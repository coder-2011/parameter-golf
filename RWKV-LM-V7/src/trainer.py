import datetime
import json
import math
import time

import pytorch_lightning as pl
import torch
from lightning_utilities.core.rank_zero import rank_zero_info, rank_zero_only

from .quant import save_quantized_state_dict


def my_save(args, trainer, dd, ff):
    if "deepspeed_stage_3" in args.strategy:
        trainer.save_checkpoint(ff, weights_only=True)
    else:
        torch.save(dd, ff)


def maybe_save_quantized_final(args, state_dict, final_path):
    bits = int(getattr(args, "quant_bits", 0))
    if bits <= 0:
        return
    quant_path = final_path.removesuffix(".pth") + f".int{bits}.ptz"
    raw_bytes, file_bytes = save_quantized_state_dict(state_dict, quant_path, bits=bits)
    rank_zero_info(
        f"Serialized quantized final checkpoint: {quant_path} "
        f"(raw_torch:{raw_bytes} compressed:{file_bytes})"
    )


METRICS_CSV_HEADER = [
    "run_timestamp",
    "event",
    "step",
    "epoch",
    "epoch_step",
    "tokens",
    "gtokens",
    "train_loss",
    "avg_loss",
    "ppl",
    "lr",
    "weight_decay",
    "kt_s",
    "step_time_ms",
    "elapsed_s",
    "samples_per_s",
    "tokens_per_step",
    "real_bsz",
    "ctx_len",
    "micro_bsz",
    "global_rank",
    "world_size",
    "loss_rank_mean",
    "loss_rank_std",
    "loss_rank_min",
    "loss_rank_max",
    "gpu_mem_allocated_mb",
    "gpu_mem_reserved_mb",
    "wall_time",
]


def _write_metrics_csv_header(f):
    f.write(",".join(METRICS_CSV_HEADER) + "\n")


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _safe_exp(value):
    try:
        return math.exp(min(float(value), 20.0))
    except (OverflowError, ValueError):
        return float("nan")


def _rank_loss_stats(trainer):
    loss_all = getattr(trainer, "my_loss_all", None)
    if loss_all is None:
        return {}
    try:
        values = loss_all.detach().float().reshape(-1).cpu()
        if values.numel() == 0:
            return {}
        return {
            "loss_rank_mean": float(values.mean().item()),
            "loss_rank_std": float(values.std(unbiased=False).item()),
            "loss_rank_min": float(values.min().item()),
            "loss_rank_max": float(values.max().item()),
        }
    except BaseException:
        return {}


def _gpu_memory_metrics():
    if not torch.cuda.is_available():
        return {"gpu_mem_allocated_mb": 0.0, "gpu_mem_reserved_mb": 0.0}
    return {
        "gpu_mem_allocated_mb": float(torch.cuda.max_memory_allocated() / 1024 / 1024),
        "gpu_mem_reserved_mb": float(torch.cuda.max_memory_reserved() / 1024 / 1024),
    }


def _append_metrics(trainer, metrics):
    if hasattr(trainer, "my_metrics_log"):
        trainer.my_metrics_log.write(
            ",".join(_csv_value(metrics.get(k)) for k in METRICS_CSV_HEADER) + "\n"
        )
        trainer.my_metrics_log.flush()
    if hasattr(trainer, "my_metrics_jsonl"):
        trainer.my_metrics_jsonl.write(json.dumps(metrics, sort_keys=True) + "\n")
        trainer.my_metrics_jsonl.flush()


def _should_emit_step_metrics(args, step):
    if int(getattr(args, "extreme_logging", 0)) > 0:
        return True
    interval = max(0, int(getattr(args, "metrics_log_interval", 50)))
    return interval > 0 and (step <= 1 or step % interval == 0)


def _wandb_log(trainer, metrics, step):
    wandb = getattr(trainer, "my_wandb", None)
    if wandb is None:
        return
    payload = {
        "train/loss": metrics.get("train_loss"),
        "train/avg_loss": metrics.get("avg_loss"),
        "train/ppl": metrics.get("ppl"),
        "optim/lr": metrics.get("lr"),
        "optim/weight_decay": metrics.get("weight_decay"),
        "throughput/kt_s": metrics.get("kt_s"),
        "throughput/step_time_ms": metrics.get("step_time_ms"),
        "throughput/samples_per_s": metrics.get("samples_per_s"),
        "tokens/total": metrics.get("tokens"),
        "tokens/gtokens": metrics.get("gtokens"),
        "system/gpu_mem_allocated_mb": metrics.get("gpu_mem_allocated_mb"),
        "system/gpu_mem_reserved_mb": metrics.get("gpu_mem_reserved_mb"),
        "trainer/epoch": metrics.get("epoch"),
        "trainer/epoch_step": metrics.get("epoch_step"),
    }
    for key in ("loss_rank_mean", "loss_rank_std", "loss_rank_min", "loss_rank_max"):
        if metrics.get(key) is not None:
            payload[f"dist/{key}"] = metrics[key]
    wandb.log({k: v for k, v in payload.items() if v is not None}, step=int(step))


def scheduled_lr(args, step: int) -> float:
    warmup_steps = int(args.warmup_steps)
    lr = float(args.lr_init)

    if args.my_exit_tokens != 0:
        real_tokens = step * args.ctx_len * args.real_bsz
        warmup_tokens = warmup_steps * args.ctx_len * args.real_bsz
        progress = (real_tokens - warmup_tokens) / (
            abs(args.my_exit_tokens) - warmup_tokens
        )
        progress = max(0.0, min(1.0, progress))
        lr_final_factor = args.lr_final / args.lr_init
        lr_mult = (0.5 + lr_final_factor / 2) + (
            0.5 - lr_final_factor / 2
        ) * math.cos(math.pi * progress)
        if args.my_exit_tokens > 0:
            lr = args.lr_init * lr_mult
        else:
            lr = (lr + args.lr_init * lr_mult) / 2
    else:
        cooldown_steps = max(0, int(getattr(args, "cooldown_steps", 0)))
        total_steps = max(0, int(getattr(args, "epoch_steps", 0)))
        if cooldown_steps > 0 and total_steps > 0:
            cooldown_start = max(total_steps - cooldown_steps, 0)
            if step >= cooldown_start:
                progress = (step - cooldown_start) / max(cooldown_steps, 1)
                progress = max(0.0, min(1.0, progress))
                lr_final_factor = args.lr_final / args.lr_init
                lr_mult = (0.5 + lr_final_factor / 2) + (
                    0.5 - lr_final_factor / 2
                ) * math.cos(math.pi * progress)
                lr = args.lr_init * lr_mult

    if step < warmup_steps:
        lr = lr * (0.01 + 0.99 * step / warmup_steps)
    return lr


class train_callback(pl.Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_start_time = None
        self.last_batch_time_ns = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        args = self.args
        if self.train_start_time is None:
            self.train_start_time = time.time()

        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps
        token_per_step = args.ctx_len * args.real_bsz

        lr = scheduled_lr(args, trainer.global_step)
        wd_now = args.weight_decay

        if args.my_exit_tokens != 0:  # cosine decay
            real_tokens = real_step * args.ctx_len * args.real_bsz
            warmup_tokens = args.warmup_steps * args.ctx_len * args.real_bsz
            progress = (real_tokens - warmup_tokens) / (
                abs(args.my_exit_tokens) - warmup_tokens
            )
            progress = max(0, min(1, progress))
            if progress >= 1:
                if (trainer.is_global_zero) or ("deepspeed_stage_3" in args.strategy):
                    to_save_dict = pl_module.state_dict()
                    final_path = f"{args.proj_dir}/rwkv-final.pth"
                    my_save(
                        args,
                        trainer,
                        to_save_dict,
                        final_path,
                    )
                    if trainer.is_global_zero:
                        maybe_save_quantized_final(args, to_save_dict, final_path)
                    rank_zero_info(
                        f"\n✅ End of training. Model saved to: {final_path}\n"
                    )
                    if trainer.is_global_zero:
                        _append_metrics(
                            trainer,
                            {
                                "run_timestamp": args.my_timestamp,
                                "event": "saved_final_tokens",
                                "step": int(real_step),
                                "epoch": int(args.epoch_begin + trainer.current_epoch),
                                "tokens": int(real_step * token_per_step),
                                "gtokens": float(real_step * token_per_step / 1e9),
                                "lr": float(lr),
                                "weight_decay": float(wd_now),
                                "tokens_per_step": int(token_per_step),
                                "real_bsz": int(args.real_bsz),
                                "ctx_len": int(args.ctx_len),
                                "micro_bsz": int(args.micro_bsz),
                                "global_rank": int(trainer.global_rank),
                                "world_size": int(trainer.world_size),
                                "wall_time": str(datetime.datetime.now()),
                            },
                        )
                    import sys

                    sys.exit(0)

        lr_scale = lr / args.lr_init if args.lr_init != 0 else 1.0
        muon_momentum = None
        if getattr(args, "optimizer", "adamw") == "muon":
            warmup_steps = max(0, int(getattr(args, "muon_momentum_warmup_steps", 0)))
            frac = min(trainer.global_step / warmup_steps, 1.0) if warmup_steps > 0 else 1.0
            muon_momentum = (
                (1.0 - frac) * args.muon_momentum_warmup_start
                + frac * args.muon_momentum
            )

        for optimizer in trainer.optimizers:
            for param_group in optimizer.param_groups:
                if param_group.get("schedule_weight_decay", True) and param_group.get("weight_decay", 0) > 0:
                    param_group["weight_decay"] = wd_now
                if "base_lr" in param_group:
                    param_group["lr"] = param_group["base_lr"] * lr_scale
                else:
                    param_group["lr"] = lr * param_group.get("my_lr_scale", 1.0)
                if muon_momentum is not None and param_group.get("is_muon", False):
                    param_group["momentum"] = muon_momentum

        trainer.my_lr = lr
        trainer.my_wd = wd_now

        if trainer.global_step == 0:
            if trainer.is_global_zero:  # logging
                trainer.my_loss_sum = 0
                trainer.my_loss_count = 0
                trainer.my_log = open(args.proj_dir + "/train_log.txt", "a")
                trainer.my_loss_log = open(args.proj_dir + "/loss_log.csv", "a")
                trainer.my_metrics_log = open(args.proj_dir + "/metrics_log.csv", "a")
                trainer.my_metrics_jsonl = open(args.proj_dir + "/metrics_log.jsonl", "a")
                if trainer.my_loss_log.tell() == 0:
                    trainer.my_loss_log.write(
                        "run_timestamp,step,gtokens,loss,avg_loss,lr,weight_decay,kt_s,wall_time\n"
                    )
                if trainer.my_metrics_log.tell() == 0:
                    _write_metrics_csv_header(trainer.my_metrics_log)
                trainer.my_log.write(
                    f"NEW RUN {args.my_timestamp}\n{vars(self.args)}\n"
                )
                rank_zero_info(
                    (
                        f"logging: train_log={args.proj_dir}/train_log.txt "
                        f"loss_csv={args.proj_dir}/loss_log.csv "
                        f"metrics_csv={args.proj_dir}/metrics_log.csv "
                        f"metrics_jsonl={args.proj_dir}/metrics_log.jsonl "
                        f"metrics_interval={getattr(args, 'metrics_log_interval', 50)} "
                        f"extreme_logging={int(getattr(args, 'extreme_logging', 0))} "
                        f"wandb={'enabled' if len(args.wandb) > 0 else 'disabled'}"
                    )
                )
                try:
                    rank_zero_info(f"\n{trainer.strategy.config}\n")
                    trainer.my_log.write(f"{trainer.strategy.config}\n")
                except BaseException:
                    pass
                trainer.my_log.flush()
                _append_metrics(
                    trainer,
                    {
                        "run_timestamp": args.my_timestamp,
                        "event": "run_start",
                        "step": int(real_step),
                        "epoch": int(args.epoch_begin + trainer.current_epoch),
                        "epoch_step": 0,
                        "tokens": 0,
                        "gtokens": 0.0,
                        "lr": lr,
                        "weight_decay": wd_now,
                        "tokens_per_step": args.ctx_len * args.real_bsz,
                        "real_bsz": args.real_bsz,
                        "ctx_len": args.ctx_len,
                        "micro_bsz": args.micro_bsz,
                        "global_rank": trainer.global_rank,
                        "world_size": trainer.world_size,
                        "wall_time": str(datetime.datetime.now()),
                    },
                )
                if len(args.wandb) > 0:
                    rank_zero_info("Login to wandb...")
                    import wandb

                    wandb.init(
                        project=args.wandb,
                        name=args.wandb_run_name or (args.run_name + " " + args.my_timestamp),
                        config=vars(args),
                        mode=args.wandb_mode or None,
                        save_code=False,
                    )
                    trainer.my_wandb = wandb

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        args = self.args
        token_per_step = args.ctx_len * args.real_bsz
        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps

        if trainer.is_global_zero:  # logging
            t_now = time.time_ns()
            kt_s = 0.0
            t_cost = 0.0
            try:
                previous_time_ns = (
                    self.last_batch_time_ns
                    if self.last_batch_time_ns is not None
                    else trainer.my_time_ns
                )
                t_cost = (t_now - previous_time_ns) / 1e9
                kt_s = token_per_step / max(t_cost, 1e-9) / 1000
                self.log("REAL it/s", 1.0 / max(t_cost, 1e-9), prog_bar=True, on_step=True)
                self.log("Kt/s", kt_s, prog_bar=True, on_step=True)
            except BaseException:
                pass
            self.last_batch_time_ns = t_now
            trainer.my_time_ns = t_now
            if isinstance(outputs, dict):
                current_loss = outputs['loss'].item()
            else:
                current_loss = outputs.item()
            trainer.my_loss = current_loss
            trainer.my_loss_sum += current_loss
            trainer.my_loss_count += 1
            trainer.my_epoch_loss = trainer.my_loss_sum / trainer.my_loss_count
            self.log("lr", trainer.my_lr, prog_bar=True, on_step=True)
            self.log("loss", trainer.my_epoch_loss, prog_bar=True, on_step=True)

            step = int(real_step)
            log_interval = max(0, int(getattr(args, "loss_log_interval", 50)))
            emit_step_metrics = _should_emit_step_metrics(args, step)
            extreme_logging = int(getattr(args, "extreme_logging", 0)) > 0
            tokens = real_step * token_per_step
            gtokens = tokens / 1e9
            elapsed_s = (
                max(time.time() - self.train_start_time, 0.0)
                if self.train_start_time is not None
                else 0.0
            )
            samples_per_s = args.real_bsz / max(t_cost, 1e-9) if t_cost > 0 else 0.0
            metrics = {
                "run_timestamp": args.my_timestamp,
                "event": "train_step",
                "step": step,
                "epoch": int(args.epoch_begin + trainer.current_epoch),
                "epoch_step": int(trainer.global_step % max(args.epoch_steps, 1)),
                "tokens": int(tokens),
                "gtokens": float(gtokens),
                "train_loss": float(trainer.my_loss),
                "avg_loss": float(trainer.my_epoch_loss),
                "ppl": _safe_exp(trainer.my_epoch_loss),
                "lr": float(trainer.my_lr),
                "weight_decay": float(trainer.my_wd),
                "kt_s": float(kt_s),
                "step_time_ms": float(t_cost * 1000.0),
                "elapsed_s": float(elapsed_s),
                "samples_per_s": float(samples_per_s),
                "tokens_per_step": int(token_per_step),
                "real_bsz": int(args.real_bsz),
                "ctx_len": int(args.ctx_len),
                "micro_bsz": int(args.micro_bsz),
                "global_rank": int(trainer.global_rank),
                "world_size": int(trainer.world_size),
                "wall_time": str(datetime.datetime.now()),
            }
            if extreme_logging:
                metrics.update(_rank_loss_stats(trainer))
                metrics.update(_gpu_memory_metrics())
            if hasattr(trainer, "my_loss_log"):
                trainer.my_loss_log.write(
                    (
                        f"{args.my_timestamp},"
                        f"{step},"
                        f"{gtokens:.9f},"
                        f"{trainer.my_loss:.8f},"
                        f"{trainer.my_epoch_loss:.8f},"
                        f"{trainer.my_lr:.10f},"
                        f"{trainer.my_wd:.10f},"
                        f"{kt_s:.4f},"
                        f"{datetime.datetime.now()}\n"
                    )
                )
                trainer.my_loss_log.flush()
            if emit_step_metrics:
                _append_metrics(trainer, metrics)

            if log_interval > 0 and (step <= 1 or step % log_interval == 0):
                mem_text = ""
                if extreme_logging:
                    mem_text = (
                        f" mem:{metrics.get('gpu_mem_allocated_mb', 0.0):.0f}/"
                        f"{metrics.get('gpu_mem_reserved_mb', 0.0):.0f}MiB"
                    )
                rank_zero_info(
                    (
                        f"step:{step} train_loss:{trainer.my_loss:.6f} "
                        f"avg_loss:{trainer.my_epoch_loss:.6f} "
                        f"ppl:{metrics['ppl']:.4f} "
                        f"lr:{trainer.my_lr:.8f} "
                        f"kt_s:{kt_s:.1f} "
                        f"step_time:{metrics['step_time_ms']:.1f}ms "
                        f"elapsed:{elapsed_s:.1f}s "
                        f"gtokens:{gtokens:.6f}"
                        f"{mem_text}"
                    )
                )

            if len(args.wandb) > 0 and emit_step_metrics:
                _wandb_log(trainer, metrics, step=real_step)

        if (trainer.is_global_zero) or (
            "deepspeed_stage_3" in args.strategy
        ):  # save pth
            if args.magic_prime > 0:
                if int(real_step) == int(args.magic_prime // args.real_bsz) - 1:
                    to_save_dict = pl_module.state_dict()
                    final_path = f"{args.proj_dir}/rwkv-final.pth"
                    my_save(
                        args,
                        trainer,
                        to_save_dict,
                        final_path,
                    )
                    if trainer.is_global_zero:
                        maybe_save_quantized_final(args, to_save_dict, final_path)
                    rank_zero_info(
                        f"\n✅ End of training. Model saved to: {final_path}\n"
                    )
                    if trainer.is_global_zero:
                        _append_metrics(
                            trainer,
                            {
                                "run_timestamp": args.my_timestamp,
                                "event": "saved_final_magic_prime",
                                "step": int(real_step),
                                "epoch": int(args.epoch_begin + trainer.current_epoch),
                                "tokens": int(real_step * token_per_step),
                                "gtokens": float(real_step * token_per_step / 1e9),
                                "lr": float(trainer.my_lr),
                                "weight_decay": float(trainer.my_wd),
                                "tokens_per_step": int(token_per_step),
                                "real_bsz": int(args.real_bsz),
                                "ctx_len": int(args.ctx_len),
                                "micro_bsz": int(args.micro_bsz),
                                "global_rank": int(trainer.global_rank),
                                "world_size": int(trainer.world_size),
                                "wall_time": str(datetime.datetime.now()),
                            },
                        )

            if args.my_exit_seconds > 0 and self.train_start_time is not None:
                if time.time() - self.train_start_time >= args.my_exit_seconds:
                    to_save_dict = pl_module.state_dict()
                    final_path = f"{args.proj_dir}/rwkv-final.pth"
                    my_save(
                        args,
                        trainer,
                        to_save_dict,
                        final_path,
                    )
                    if trainer.is_global_zero:
                        maybe_save_quantized_final(args, to_save_dict, final_path)
                    rank_zero_info(
                        f"\n✅ Timed training stop. Model saved to: {final_path}\n"
                    )
                    if trainer.is_global_zero:
                        _append_metrics(
                            trainer,
                            {
                                "run_timestamp": args.my_timestamp,
                                "event": "saved_final_time",
                                "step": int(real_step),
                                "epoch": int(args.epoch_begin + trainer.current_epoch),
                                "tokens": int(real_step * token_per_step),
                                "gtokens": float(real_step * token_per_step / 1e9),
                                "elapsed_s": float(time.time() - self.train_start_time),
                                "lr": float(trainer.my_lr),
                                "weight_decay": float(trainer.my_wd),
                                "tokens_per_step": int(token_per_step),
                                "real_bsz": int(args.real_bsz),
                                "ctx_len": int(args.ctx_len),
                                "micro_bsz": int(args.micro_bsz),
                                "global_rank": int(trainer.global_rank),
                                "world_size": int(trainer.world_size),
                                "wall_time": str(datetime.datetime.now()),
                            },
                        )
                    import sys

                    sys.exit(0)

    def on_train_epoch_start(self, trainer, pl_module):
        args = self.args
        dataset = trainer.train_dataloader.dataset
        if hasattr(dataset, "global_rank"):
            dataset.global_rank = trainer.global_rank
            dataset.real_epoch = int(args.epoch_begin + trainer.current_epoch)
            dataset.world_size = trainer.world_size

    def on_train_epoch_end(self, trainer, pl_module):
        args = self.args
        to_save_dict = {}
        if (trainer.is_global_zero) or (
            "deepspeed_stage_3" in args.strategy
        ):  # save pth
            if (
                args.epoch_save > 0 and trainer.current_epoch % args.epoch_save == 0
            ) or (trainer.current_epoch == args.epoch_count - 1):
                if args.data_type == "wds_img":
                    raw_dict = pl_module.state_dict()
                    for k in raw_dict:
                        if k.startswith("encoder.") or k.startswith("decoder."):
                            to_save_dict[k] = raw_dict[k]
                else:
                    to_save_dict = pl_module.state_dict()
                try:
                    checkpoint_path = f"{args.proj_dir}/rwkv-{args.epoch_begin + trainer.current_epoch}.pth"
                    my_save(
                        args,
                        trainer,
                        to_save_dict,
                        checkpoint_path,
                    )
                    if trainer.is_global_zero:
                        _append_metrics(
                            trainer,
                            {
                                "run_timestamp": args.my_timestamp,
                                "event": "saved_epoch",
                                "step": int(trainer.global_step + args.epoch_begin * args.epoch_steps),
                                "epoch": int(args.epoch_begin + trainer.current_epoch),
                                "train_loss": float(getattr(trainer, "my_loss", float("nan"))),
                                "avg_loss": float(getattr(trainer, "my_epoch_loss", float("nan"))),
                                "ppl": _safe_exp(getattr(trainer, "my_epoch_loss", float("nan"))),
                                "lr": float(getattr(trainer, "my_lr", 0.0)),
                                "weight_decay": float(getattr(trainer, "my_wd", 0.0)),
                                "tokens_per_step": int(args.ctx_len * args.real_bsz),
                                "real_bsz": int(args.real_bsz),
                                "ctx_len": int(args.ctx_len),
                                "micro_bsz": int(args.micro_bsz),
                                "global_rank": int(trainer.global_rank),
                                "world_size": int(trainer.world_size),
                                "wall_time": str(datetime.datetime.now()),
                            },
                        )
                except Exception as e:
                    rank_zero_info("Error\n\n", e, "\n\n")

        if trainer.is_global_zero:  # logging
            trainer.my_log.write(
                (
                    f"{args.epoch_begin + trainer.current_epoch} "
                    f"{trainer.my_epoch_loss:.6f} "
                    f"{math.exp(trainer.my_epoch_loss):.4f} "
                    f"{trainer.my_lr:.8f} "
                    f"{datetime.datetime.now()} "
                    f"{trainer.current_epoch}\n"
                )
            )
            trainer.my_log.flush()

            trainer.my_loss_sum = 0
            trainer.my_loss_count = 0

    def on_train_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        args = self.args
        _append_metrics(
            trainer,
            {
                "run_timestamp": args.my_timestamp,
                "event": "train_end",
                "step": int(trainer.global_step + args.epoch_begin * args.epoch_steps),
                "epoch": int(args.epoch_begin + trainer.current_epoch),
                "elapsed_s": (
                    float(time.time() - self.train_start_time)
                    if self.train_start_time is not None
                    else 0.0
                ),
                "lr": float(getattr(trainer, "my_lr", 0.0)),
                "weight_decay": float(getattr(trainer, "my_wd", 0.0)),
                "tokens_per_step": int(args.ctx_len * args.real_bsz),
                "real_bsz": int(args.real_bsz),
                "ctx_len": int(args.ctx_len),
                "micro_bsz": int(args.micro_bsz),
                "global_rank": int(trainer.global_rank),
                "world_size": int(trainer.world_size),
                "wall_time": str(datetime.datetime.now()),
            },
        )
        for attr in ("my_metrics_jsonl", "my_metrics_log", "my_loss_log", "my_log"):
            f = getattr(trainer, attr, None)
            if f is not None:
                f.flush()
                f.close()
                setattr(trainer, attr, None)
        wandb = getattr(trainer, "my_wandb", None)
        if wandb is not None:
            wandb.finish()


@rank_zero_only
def generate_init_weight(model, init_weight_name):
    mm = model.generate_init_weight()

    if model.args.train_stage == 1:
        if len(model.args.load_model) > 0:
            rank_zero_info(f"Combine weights from {model.args.load_model}...")
            load_dict = torch.load(model.args.load_model, map_location="cpu")
            for k in load_dict:
                try:
                    assert k in mm
                except BaseException:
                    rank_zero_info("missing", k)
                    import sys
                    sys.exit(0)
                src = load_dict[k]
                try:
                    mm[k] = src.reshape(mm[k].shape)
                except BaseException:
                    tmp = mm[k].squeeze().clone()
                    rank_zero_info(k, src.shape, "-->", mm[k].shape)
                    ss = src.shape[0]
                    dd = tmp.shape[0]
                    for i in range(dd):
                        pos = i / dd * ss
                        if pos >= ss - 1:
                            tmp[i] = src[ss - 1]
                        else:
                            p0 = int(math.floor(pos))
                            ii = pos - p0
                            tmp[i] = src[p0] * (1 - ii) + src[p0 + 1] * (ii)
                    mm[k] = tmp.reshape(mm[k].shape)
                    sss = src.squeeze().float().cpu().numpy()
                    rank_zero_info(sss[:10], "...", sss[-10:])
                    mmm = mm[k].squeeze().float().cpu().numpy()
                    rank_zero_info(mmm[:10], "...", mmm[-10:])

    rank_zero_info(f"Save to {init_weight_name}...")
    torch.save(mm, init_weight_name)

    if model.args.train_stage == 1:
        rank_zero_info("Done. Now go for stage 2.")
        import sys
        sys.exit(0)
