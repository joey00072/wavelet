from __future__ import annotations

from functools import partial

from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from wavelet.configs.sft import DataConfig
from wavelet.data.collation import collate_batch


def setup_dataloader(
    dataset: IterableDataset,
    config: DataConfig,
    pad_token_id: int,
) -> StatefulDataLoader:
    if config.pack_function in ("pad", "cat"):
        include_attention_mask = config.pack_function != "cat"
        return StatefulDataLoader(
            dataset,
            batch_size=config.micro_batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=config.num_workers > 0,
            snapshot_every_n_steps=1,
            collate_fn=partial(
                collate_batch,
                pad_token_id=pad_token_id,
                include_attention_mask=include_attention_mask,
            ),
        )
    raise NotImplementedError(
        f"Pack function '{config.pack_function}' not implemented yet"
    )
