from typing import Optional, Sequence

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmseg.registry import HOOKS


@HOOKS.register_module()
class IterInjectHook(Hook):

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: Optional[Sequence[dict]] = None) -> None:
        model = runner.model.module if is_model_wrapper(runner.model) else runner.model
        model.iter = runner.iter