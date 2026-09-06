import torch

from wavelet.configs.sft import SchedulerConfig
from wavelet.trainer.optim import setup_scheduler


def test_constant_scheduler_honors_warmup_then_holds_lr() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=1e-6)
    scheduler = setup_scheduler(
        optimizer,
        SchedulerConfig(type="constant", warmup_steps=10, min_lr_factor=0.1),
        total_steps=100,
        lr=1e-6,
    )

    lrs = []
    for _ in range(15):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    assert lrs[0] > 1e-7
    assert lrs[8] < 1e-6
    assert lrs[9] == 1e-6
    assert lrs[-1] == 1e-6
