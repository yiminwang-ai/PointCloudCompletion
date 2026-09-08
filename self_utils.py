import numpy as np


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        初始化早停机制的参数。

        :param patience: 整数，允许的没有改善的 epoch 数。
        :param min_delta: 最小改善量，小于此值将被视为没有改进。
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = np.inf  # 假设我们在监控损失
        self.epochs_without_improvement = 0
        self.should_stop = False

    def check(self, current_metric):
        """
        检查当前的性能指标是否有改进。

        :param current_metric: 当前 epoch 的性能指标（如测试损失）。
        """
        if self.best_metric - current_metric > self.min_delta:
            self.best_metric = current_metric
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.patience:
            self.should_stop = True
            print(f"Early stopping triggered (best metric: {self.best_metric:.4f})")


class Printf:
    def __init__(self, is_open=True):
        self.is_open = is_open

    def say(self, *x):
        if self.is_open:
            print(x)
