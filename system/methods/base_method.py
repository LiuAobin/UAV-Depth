from pytorch_lightning import LightningModule
from system.core import get_optim_scheduler, timm_schedulers


class BaseMethod(LightningModule):
    def __init__(self, config):
        super().__init__()
        # 保存超参数
        self.save_hyperparameters()
        self.models = self._build_model()  # 子类具体实验模型构建逻辑


    def _build_model(self):
        """
        构建模型，具体逻辑由子类实现
        Returns:
        """
        raise NotImplementedError

    def configure_optimizers(self):
        """
        配置优化器和学习率调度器
        Returns:
        """
        optimizer, scheduler, by_epoch = get_optim_scheduler(  # 根据超参数设置优化器和学习率调度器
            self.hparams.config,
            self.hparams.config.epochs,
            self.models,
            self.hparams.config.epoch_size
        )
        return {
            "optimizer": optimizer,
            'lr_scheduler':{
                "scheduler": scheduler,
                "interval": "epoch" if by_epoch else "step"
            }
        }

    def lr_scheduler_step(self, scheduler, metric):
        """
        学习率调度器调整步骤
        Args:
            scheduler (): 调度器
            metric ():
        Returns:
        """
        # 如果是timm提供的调度器，则按epoch调整
        if any(isinstance(scheduler,sch) for sch in timm_schedulers):
            scheduler.step(epoch=self.current_epoch)
        else:  # 根据指标或者默认规则调整
            if metric is None:
                scheduler.step()
            else:
                scheduler.step(metric)

    def forward(self, batch):
        """
        前向传播逻辑，需由子类实现。————可以只获取深度，这样的话，测试步骤实现就比较方便
        """
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        """
        训练阶段逻辑，需由子类实现。
        """
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        """
        验证阶段的单步逻辑。
        """
        pass

    def test_step(self, batch, batch_idx):
        """
        测试阶段的单步逻辑。
        """
        pass

    def on_test_epoch_end(self):
        """
        测试阶段结束时的逻辑。
        """
        pass

