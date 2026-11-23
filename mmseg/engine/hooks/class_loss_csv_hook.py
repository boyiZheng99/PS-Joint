import csv
from pathlib import Path

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmseg.registry import HOOKS


@HOOKS.register_module()
class ClassLossCSVHook(Hook):
    """Save per-class classification losses from each training iteration."""

    priority = 'LOW'

    def before_train(self, runner):
        self.path = Path(runner.work_dir) / 'classification_losses.csv'
        self.file = None
        self.writer = None
        if getattr(runner, 'rank', 0) == 0:
            self.file = self.path.open('w', newline='')

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if getattr(runner, 'rank', 0) != 0 or not outputs:
            return
        log_vars = outputs.get('log_vars', {})
        row = {'iter': int(runner.iter) + 1}
        for key, value in log_vars.items():
            if key.startswith('loss_cls_') or key.startswith('loss_dino_cls_'):
                if hasattr(value, 'item'):
                    value = value.item()
                row[key] = float(value)
        if len(row) == 1:
            return
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(row))
            self.writer.writeheader()
        else:
            for key in row:
                if key not in self.writer.fieldnames:
                    continue
            row = {key: row.get(key, '') for key in self.writer.fieldnames}
        self.writer.writerow(row)
        self.file.flush()

    def after_train(self, runner):
        if self.file is not None:
            self.file.close()
