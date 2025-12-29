import torch
import lightning as L
import segmentation_models_pytorch as smp
from torchmetrics.segmentation import MeanIoU, GeneralizedDiceScore

from losses import DiceLoss, FusionLoss, FocalLoss
from learningRate import CyclicLR, WarmUpLR, CosineLR  

import os
import numpy as np
from ..config import TEMP_DIR
from monsoonToolBox.filetools import pJoin
from monsoonToolBox.arraytools.draw2d import Drawer2D
import matplotlib.pyplot as plt


class UNetPPModule(L.LightningModule):
    """
    Lightning version of TrainerTMJ:
    - smp.UnetPlusPlus(resnet50, in_channels=1, classes=4)
    - FusionLoss([DiceLoss, FocalLoss])
    - custom LR schedule (Cosine + Cyclic + WarmUp) like TrainerTMJ.getLr()
    - optional prob image logging like TrainerTMJ.probImg()

    Expected batch (from TMJDataset.py):
      batch = {"image": (B,1,H,W), "mask": (B,H,W), "filename": ...}
    """

    def __init__(
        self,
        num_classes: int = 4,
        encoder_name: str = "resnet50",
        in_channels: int = 1,
        learning_rate: float = 1e-4,
        total_epochs: int = 1000,
        weight_decay: float = 0.01,
        prob_every: int = 5,
        save_dir: str | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.base_lr = learning_rate
        self.total_epochs = total_epochs

        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )

        loss_fn0 = DiceLoss(weights=[0.1, 5, 1, 1], use_softmax=True)
        loss_fn1 = FocalLoss(gamma=3)
        self.criterion = FusionLoss([loss_fn0, loss_fn1], weights=[0.5, 0.5], device="cpu")  # device set in setup()

        self.iou_metrics = {
            "train": MeanIoU(num_classes=num_classes, per_class=False),
            "val": MeanIoU(num_classes=num_classes, per_class=False),
            "test": MeanIoU(num_classes=num_classes, per_class=False),
        }
        for stage, metric in self.iou_metrics.items():
            self.add_module(f"{stage}_mean_iou", metric)

        self.dice_metrics = {
            "train": GeneralizedDiceScore(num_classes=num_classes),
            "val": GeneralizedDiceScore(num_classes=num_classes),
            "test": GeneralizedDiceScore(num_classes=num_classes),
        }
        for stage, metric in self.dice_metrics.items():
            self.add_module(f"{stage}_dice_score", metric)

        self.prob_every = prob_every
        self.save_dir = save_dir or pJoin(TEMP_DIR, "ProbImgs")

        base_instance = CosineLR()
        cycle_instance = CyclicLR(cycle_at=[50, 200], lr_snippet=base_instance, decay=[0.5, 0.1])
        self.lr_instance = WarmUpLR(lr_instance=cycle_instance, warmup_frac=0.05)

        self.prob_img = None
        self.prob_msk = None

    def setup(self, stage: str | None = None):
        if hasattr(self.criterion, "device"):
            self.criterion.device = self.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _common_step(self, batch: dict, stage: str) -> torch.Tensor:
        images = batch["image"]          # (B,1,H,W)
        masks = batch["mask"]            # (B,H,W)

        logits = self(images)            # (B,C,H,W)
        loss = self.criterion(logits, masks)

        preds = torch.argmax(logits, dim=1)  # (B,H,W)

        self.iou_metrics[stage].update(preds, masks)
        self.dice_metrics[stage].update(preds, masks)

        self.log(
            f"{stage}/loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            batch_size=len(images),
        )
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._common_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._common_step(batch, "val")

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._common_step(batch, "test")

    def predict_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images = batch["image"]
        logits = self(images)
        return torch.argmax(logits, dim=1)

    # ---------- epoch end metrics ----------
    def _common_epoch_end(self, stage: str):
        iou = self.iou_metrics[stage].compute()
        dice = self.dice_metrics[stage].compute()
        self.log(f"{stage}/iou", iou, prog_bar=True)
        self.log(f"{stage}/dice", dice, prog_bar=False)

        self.iou_metrics[stage].reset()
        self.dice_metrics[stage].reset()

    def on_train_epoch_end(self):
        self._common_epoch_end("train")

    def on_validation_epoch_end(self):
        self._common_epoch_end("val")

    def on_test_epoch_end(self):
        self._common_epoch_end("test")

    # ---------- optional: prob image logging (like TrainerTMJ) ----------
    def serveProbImgs(self, img: torch.Tensor, msk: torch.Tensor):
        """
        img: (1,H,W)
        msk: (H,W)
        """
        self.prob_img = img
        self.prob_msk = msk

    def on_train_epoch_start(self):
        if (self.current_epoch % self.prob_every) == 0:
            self._probImg()

    def _probImg(self, flag: str = ""):
        if self.prob_img is None or self.prob_msk is None:
            return

        self.model.eval()
        with torch.no_grad():
            img = self.prob_img.unsqueeze(0).to(self.device)  # (1,1,H,W)
            out = self.model(img)
            pred_msk = out.argmax(dim=1)[0]  # (H,W)

        im = self.prob_img[0].detach().cpu().numpy()
        pred_msk = pred_msk.detach().cpu().numpy()
        gt_msk = self.prob_msk.detach().cpu().numpy()

        color_dict = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 100, 255)}
        compare_im = Drawer2D.visualCompareSegmentations(
            im,
            [gt_msk, pred_msk],
            color_dict=color_dict,
            alpha=0.5,
            tags=["Ground truth", "Model prediction"],
        )

        os.makedirs(self.save_dir, exist_ok=True)
        fname = pJoin(self.save_dir, f"epoch-{self.current_epoch}{flag}.png")
        plt.imsave(fname, compare_im)

        self.model.train()

    # ---------- optimizer + custom LR schedule ----------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.base_lr,
            weight_decay=self.hparams.weight_decay,
        )
        
        def lr_lambda(epoch: int):
            lr_abs = float(self.lr_instance(epoch, self.total_epochs, self.base_lr))
            return lr_abs / float(self.base_lr)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
