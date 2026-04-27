#################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
#################################################################

import glob
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .binidx import MMapIndexedDataset

try:
    from lightning_utilities.core.rank_zero import rank_zero_info
except ModuleNotFoundError:
    def rank_zero_info(*args, **kwargs):
        print(*args, **kwargs)


def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args

        self.vocab_size = args.vocab_size
        rank_zero_info(
            f"Current vocab size = {self.vocab_size} (make sure it's correct)")

        if args.data_type == "binidx":
            self.data = MMapIndexedDataset(args.data_file)
            self.data_size = len(
                self.data._bin_buffer) // self.data._index._dtype_size
            rank_zero_info(f"Data has {self.data_size} tokens.")
        elif args.data_type == "fineweb_u16":
            self.data = FineWebU16Dataset(args.data_file)
            self.data_size = self.data.data_size
            rank_zero_info(
                f"FineWeb data has {self.data_size} tokens across {len(self.data.files)} shard(s).")
        else:
            raise ValueError(f"Unsupported data_type: {args.data_type}")

        self.samples_per_epoch = args.epoch_steps * args.real_bsz
        if args.data_type == "binidx":
            assert self.samples_per_epoch == 40320 * args.devices
        rank_zero_info(f"########## train stage {args.train_stage} ##########")
        # add default rank parameter
        self.global_rank = 0
        self.real_epoch = 0
        self.world_size = 1

        if args.data_type == "binidx":
            dataset_slot = self.data_size // args.ctx_len
            assert is_prime(args.magic_prime)
            assert args.magic_prime % 3 == 2
            assert args.magic_prime / dataset_slot > 0.9 and args.magic_prime / dataset_slot <= 1

    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        args = self.args
        rank = self.global_rank
        epoch = self.real_epoch
        world_size = self.world_size
        # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size}")

        ctx_len = args.ctx_len
        req_len = ctx_len + 1

        ii = 1 + epoch * self.samples_per_epoch + (idx * world_size) + rank

        if args.data_type == "fineweb_u16":
            i = (ii * ctx_len) % (self.data_size - req_len)
            dix = self.data.get(offset=i, length=req_len).astype(int)
        else:
            magic_prime = args.magic_prime
            factor = (math.sqrt(5) - 1) / 2
            factor = int(magic_prime * factor)
            i = ((factor * ii * ii * ii) % magic_prime) * ctx_len
            dix = self.data.get(idx=0, offset=i, length=req_len).astype(int)
            # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size} ii {ii} pos {round(i / self.data_size, 3)}")


        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)

        return x, y


class FineWebU16Dataset:
    def __init__(self, data_file):
        path = Path(data_file)
        if path.is_dir():
            files = sorted(path.glob("fineweb_train_*.bin"))
        else:
            files = [Path(p) for p in sorted(glob.glob(data_file))]
            if not files and path.exists():
                files = [path]
        if not files:
            raise FileNotFoundError(
                f"No FineWeb train shards found for {data_file}. "
                "Pass a directory containing fineweb_train_*.bin or a shard glob.")

        self.files = files
        self.shards = []
        self.sizes = []
        for file in self.files:
            tokens = self._open_shard(file)
            self.shards.append(tokens)
            self.sizes.append(int(tokens.shape[0]))
        self.cumulative_sizes = np.cumsum(self.sizes)
        self.data_size = int(self.cumulative_sizes[-1])

    @staticmethod
    def _open_shard(file):
        header_bytes = 256 * np.dtype("<i4").itemsize
        token_bytes = np.dtype("<u2").itemsize
        header = np.fromfile(file, dtype="<i4", count=256)
        if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
            raise ValueError(f"Unexpected FineWeb shard header for {file}")
        num_tokens = int(header[2])
        expected_size = header_bytes + num_tokens * token_bytes
        if file.stat().st_size != expected_size:
            raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
        return np.memmap(file, dtype="<u2", mode="r", offset=header_bytes, shape=(num_tokens,))

    def get(self, offset, length):
        if length > self.data_size:
            raise ValueError(f"Requested {length} tokens from dataset of size {self.data_size}")
        offset %= self.data_size
        remaining = length
        chunks = []
        while remaining > 0:
            shard_idx = int(np.searchsorted(self.cumulative_sizes, offset, side="right"))
            shard_start = 0 if shard_idx == 0 else int(self.cumulative_sizes[shard_idx - 1])
            local_offset = offset - shard_start
            available = self.sizes[shard_idx] - local_offset
            take = min(remaining, available)
            chunks.append(np.asarray(self.shards[shard_idx][local_offset: local_offset + take]))
            remaining -= take
            offset = (offset + take) % self.data_size
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
