import math

import torch
from torch.optim.lr_scheduler import LambdaLR


class LinearScheduleWithWarmupConfig(LambdaLR):
    """
    Inherit LambdaLR so that it can be defined in config.
    https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#LambdaLR
    https://huggingface.co/transformers/_modules/transformers/optimization.html#get_linear_schedule_with_warmup
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        epochs = int,
        last_epoch: int = -1,
        num_training_steps: int = 6036000
    ) -> None:

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )
        super().__init__(optimizer, lr_lambda, last_epoch)
