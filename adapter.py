from datamint.mlflow.flavors.model import DatamintModel, PredictionResult
from datamint.entities.annotations import VolumeSegmentation
from datamint.entities import Resource
import torch
import cv2
import numpy as np
from typing_extensions import override
import lightning as L
from medimgkit.readers import read_array_normalized

PROJECT_NAME = "TMJ Test"
NUM_CLASSES = 4
CLASS_NAMES = {
    0: "Background",
    1: "Disc",
    2: "Condyle",
    3: "Eminence"
}


class UNetPPSegmentationAdapter(DatamintModel):
    """Datamint adapter for UNet++ segmentation model deployment."""

    def __init__(self):
        super().__init__(
            mlflow_torch_models_uri={
                'unetpp': f'models:/{PROJECT_NAME}/latest'
            },
            settings={'need_gpu': False}
        )
        self.class_names = CLASS_NAMES

    def _preprocess(self, slice_2d):
        """ Preprocess a 2D slice before feeding it to the model:"""
        img = slice_2d.astype(np.float32)
        img = np.rot90(img, k=3)
        img = np.flip(img, axis=1)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        vmin, vmax = np.percentile(img, [1, 99])
        img = np.clip(img, vmin, vmax)
        img = (img - vmin) / (vmax - vmin + 1e-8)
        return torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()

    def _undo_transforms(self, pred_2d, original_shape):
        """ Undo the preprocessing transforms on the predicted 2D slice."""
        pred = np.flip(pred_2d, axis=1)
        pred = np.rot90(pred, k=1)
        pred = cv2.resize(pred.astype(np.uint8),
                          (original_shape[1], original_shape[0]),
                          interpolation=cv2.INTER_NEAREST)
        return pred

    def _create_3d_annotations(self, full_pred_volume):
        """ Convert a full 3D predicted volume into Datamint ImageSegmentation annotations."""
        annotations = []
        for class_idx, class_name in self.class_names.items():
            if class_idx == 0:
                continue
            mask_3d = (full_pred_volume == class_idx).astype(np.uint8) * 255
            if mask_3d.any():
                seg = VolumeSegmentation.from_semantic_segmentation(
                    segmentation=mask_3d,
                    class_map=class_name,
                )
                annotations.append(seg)
        return annotations

    @override
    def predict_image(self, model_input: list[Resource], **kwargs):
        pytorch_model = self.get_mlflow_torch_models()['unetpp']
        pytorch_model.eval()

        fabric = L.Fabric(accelerator=self.inference_device)
        pytorch_model = fabric.setup_module(pytorch_model)

        all_predictions = []
        with torch.inference_mode():
            for res in model_input:

                data = res.fetch_file_data(auto_convert=False, use_cache=True)
                vol = read_array_normalized(data, mime_type=res.mimetype)
                # (Z, C, H, W) -> (Z, H, W)
                vol = vol[:, 0, :, :]
                # (Z, H, W) -> (W, H, Z)
                vol = np.transpose(vol, (2, 1, 0))

                print(f"Dimensions of volume: {vol.ndim}")
                print(f"Shape: {vol.shape}")
                # print(f"slices sum: {vol[6].sum()} {vol[:,7].sum()} {vol[:,:,8].sum()}") # Debugging line to check the content of the volume

                if vol.ndim == 3:
                    h, w, z_slices = vol.shape
                    ''' New volume '''
                    full_pred_volume = np.zeros((h, w, z_slices), dtype=np.uint8)

                    for z in range(z_slices):
                        slice_2d = vol[:, :, z]
                        input_tensor = self._preprocess(slice_2d).to(fabric.device)
                        logits = pytorch_model(input_tensor)
                        pred_2d = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

                        full_pred_volume[:, :, z] = self._undo_transforms(pred_2d, (h, w))
                    annotations = self._create_3d_annotations(full_pred_volume)
                    all_predictions.append(annotations)
                else:
                    print(f"Unexpected volume dimensions: {vol.ndim}. Expected a 3D volume.")

        return all_predictions

    def predict_default(self, model_input: list[Resource], **kwargs):
        return self.predict_image(model_input, **kwargs)