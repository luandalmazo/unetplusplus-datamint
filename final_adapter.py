import math
from operator import itemgetter

import cv2
import lightning as L
import numpy as np
import torch
from datamint.entities import Resource
from datamint.entities.annotations import BoxAnnotation, LineAnnotation, VolumeSegmentation
from datamint.mlflow.flavors.model import DatamintModel
from medimgkit.readers import read_array_normalized
from typing_extensions import override
import logging

_LOGGER = logging.getLogger(__name__)

PROJECT_NAME = "TMJ Test"
CLASS_NAMES = {
    0: "Background",
    1: "Disc",
    2: "Condyle",
    3: "Eminence",
}
DISC_CLASS_ID = 1
CONDYLE_CLASS_ID = 2
EMINENCE_CLASS_ID = 3
POINT_BOX_HALF_SIZE = 3
NORMAL_DISTANCE_THRESHOLD_MM = 10.0

Point2D = tuple[int, int]


class UNetPP_TMJ_Adapter(DatamintModel):
    """Datamint adapter with TMJ segmentations and slice-level geometry annotations."""

    def __init__(self) -> None:
        super().__init__(
            mlflow_torch_models_uri={"unetpp": f"models:/{PROJECT_NAME}/latest"},
            settings={"need_gpu": False},
        )
        self.class_names = CLASS_NAMES

    @staticmethod
    def _preprocess(slice_2d: np.ndarray) -> torch.Tensor:
        img = slice_2d.astype(np.float32)
        img = np.rot90(img, k=3)
        img = np.flip(img, axis=1)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        vmin, vmax = np.percentile(img, [1, 99])
        img = np.clip(img, vmin, vmax)
        img = (img - vmin) / (vmax - vmin + 1e-8)
        return torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()

    @staticmethod
    def _undo_transforms(pred_2d: np.ndarray, original_shape: tuple[int, int]) -> np.ndarray:
        pred_2d = np.flip(pred_2d, axis=1)
        pred_2d = np.rot90(pred_2d, k=1)
        pred_2d = cv2.resize(
            pred_2d.astype(np.uint8),
            (original_shape[1], original_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        return pred_2d

    @staticmethod
    def _to_display_space(mask_slice: np.ndarray) -> np.ndarray:
        return np.flip(np.rot90(mask_slice, k=3), axis=1)

    @staticmethod
    def _display_to_volume_point(point: Point2D) -> Point2D:
        # The demo computes geometry on the displayed slice, which is the transposed volume slice.
        return point[1], point[0]

    @staticmethod
    def _extract_spacing(metadata: object) -> tuple[float, float]:
        header = getattr(metadata, "header", None)
        if header is not None and hasattr(header, "get_zooms"):
            zooms = header.get_zooms()
            if len(zooms) >= 2:
                return float(zooms[0]), float(zooms[1])
        return 1.0, 1.0

    @staticmethod
    def _mask_to_points(mask: np.ndarray) -> list[Point2D]:
        return sorted({(int(point[1]), int(point[0])) for point in np.argwhere(mask > 0)})

    @staticmethod
    def _get_mean_point(points: list[Point2D]) -> Point2D | None:
        if not points:
            return None

        x_mean = round(sum(point[0] for point in points) / len(points))
        y_mean = round(sum(point[1] for point in points) / len(points))
        return x_mean, y_mean

    @staticmethod
    def _get_extrema_point(points: list[Point2D], extrema_wanted: str) -> Point2D | None:
        if not points:
            return None
        if extrema_wanted == "max":
            return min(points, key=itemgetter(1))
        if extrema_wanted == "min":
            return max(points, key=itemgetter(1))
        raise ValueError("extrema_wanted must be 'max' or 'min'")

    @staticmethod
    def _get_extrema_point_x(points: list[Point2D], extrema_wanted: str) -> Point2D | None:
        if not points:
            return None
        if extrema_wanted == "min":
            return min(points, key=itemgetter(0))
        if extrema_wanted == "max":
            return max(points, key=itemgetter(0))
        raise ValueError("extrema_wanted must be 'min' or 'max'")

    @staticmethod
    def _bresenham_line(x0: int, y0: int, x1: int, y1: int) -> list[Point2D]:
        points: list[Point2D] = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        step_x = -1 if x0 > x1 else 1
        step_y = -1 if y0 > y1 else 1

        if dx > dy:
            err = dx / 2.0
            while x != x1:
                points.append((x, y))
                err -= dy
                if err < 0:
                    y += step_y
                    err += dx
                x += step_x
        else:
            err = dy / 2.0
            while y != y1:
                points.append((x, y))
                err -= dx
                if err < 0:
                    x += step_x
                    err += dy
                y += step_y

        points.append((x, y))
        return points

    @staticmethod
    def _get_distances_between_two_points(
        point1: Point2D,
        point2: Point2D,
    ) -> tuple[int, int, float]:
        x_distance = abs(point1[0] - point2[0])
        y_distance = abs(point1[1] - point2[1])
        euclidean_distance = math.hypot(x_distance, y_distance)
        return x_distance, y_distance, euclidean_distance

    @staticmethod
    def _get_shortest_distance(
        points1: list[Point2D],
        points2: list[Point2D],
    ) -> tuple[Point2D, Point2D] | None:
        if not points1 or not points2:
            return None

        point_array_1 = np.asarray(points1, dtype=np.float32)
        point_array_2 = np.asarray(points2, dtype=np.float32)
        deltas = point_array_1[:, None, :] - point_array_2[None, :, :]
        distances = np.sum(deltas * deltas, axis=2)
        index_1, index_2 = np.unravel_index(np.argmin(distances), distances.shape)
        point1 = tuple(point_array_1[index_1].astype(int))
        point2 = tuple(point_array_2[index_2].astype(int))
        return point1, point2

    @staticmethod
    def _find_midpoint(points: list[Point2D]) -> Point2D | None:
        if not points:
            return None

        x_mid = round((points[0][0] + points[-1][0]) / 2)
        y_mid = round((points[0][1] + points[-1][1]) / 2)
        return x_mid, y_mid

    @classmethod
    def _make_disc_line(
        cls,
        disc_points: list[Point2D],
    ) -> tuple[list[Point2D] | None, float, Point2D | None, Point2D | None]:
        unique_points = sorted(set(disc_points))
        if not unique_points:
            return None, 0.0, None, None

        if len({point[0] for point in unique_points}) < 2:
            top_point = cls._get_extrema_point(unique_points, "max")
            bottom_point = cls._get_extrema_point(unique_points, "min")
            if top_point is None or bottom_point is None:
                return None, 0.0, None, None
            line_points = cls._bresenham_line(top_point[0], top_point[1], bottom_point[0], bottom_point[1])
            line_length = cls._get_distances_between_two_points(top_point, bottom_point)[2]
            return line_points, line_length, top_point, bottom_point

        xs = np.asarray([point[0] for point in unique_points], dtype=np.float32)
        ys = np.asarray([point[1] for point in unique_points], dtype=np.float32)
        slope, intercept = np.polyfit(xs, ys, 1)

        best_fit_points = []
        for x_value in xs:
            y_value = round(slope * x_value + intercept)
            best_fit_points.append((int(round(x_value)), int(y_value)))

        top_right_best = cls._get_extrema_point(best_fit_points, "max")
        bottom_left_best = cls._get_extrema_point(best_fit_points, "min")
        if top_right_best is None or bottom_left_best is None:
            return None, 0.0, None, None

        disc_length = cls._get_distances_between_two_points(top_right_best, bottom_left_best)[2]
        return best_fit_points, disc_length, top_right_best, bottom_left_best

    @classmethod
    def _make_emcon_line(
        cls,
        eminence_points: list[Point2D],
        condyle_points: list[Point2D],
        left_low_disc_point: Point2D,
    ) -> tuple[list[Point2D] | None, tuple[int, int, float] | None, Point2D | None, Point2D | None]:
        unique_eminence = sorted(set(eminence_points))
        unique_condyle = sorted(set(condyle_points))

        updated_eminence = [point for point in unique_eminence if point[0] >= left_low_disc_point[0]]
        if not updated_eminence or not unique_condyle:
            return None, None, None, None

        highest_eminence_point = cls._get_extrema_point(updated_eminence, "max")
        if highest_eminence_point is None:
            return None, None, None, None

        cleaned_eminence = [
            point for point in updated_eminence if point[0] <= highest_eminence_point[0]
        ]
        if not cleaned_eminence:
            return None, None, None, None

        closest_pair = cls._get_shortest_distance(unique_condyle, cleaned_eminence)
        if closest_pair is None:
            return None, None, None, None

        emcon_point_1, emcon_point_2 = closest_pair
        emcon_line_points = cls._bresenham_line(
            emcon_point_1[0],
            emcon_point_1[1],
            emcon_point_2[0],
            emcon_point_2[1],
        )
        line_lengths = cls._get_distances_between_two_points(emcon_point_1, emcon_point_2)
        return emcon_line_points, line_lengths, emcon_point_1, emcon_point_2

    @classmethod
    def _calculate_ratios(
        cls,
        disc_mask_display: np.ndarray,
        emcon_line_points: list[Point2D],
    ) -> tuple[float, float]:
        disc_points = cls._mask_to_points(disc_mask_display)
        if not disc_points or not emcon_line_points:
            return 0.0, 0.0

        endpoint_1 = cls._get_extrema_point(emcon_line_points, "max")
        endpoint_2 = cls._get_extrema_point(emcon_line_points, "min")
        if endpoint_1 is None or endpoint_2 is None:
            return 0.0, 0.0

        below_line = 0
        above_line = 0
        for x_value, y_value in disc_points:
            determinant = (
                (x_value - endpoint_1[0]) * (endpoint_2[1] - endpoint_1[1])
                - (y_value - endpoint_1[1]) * (endpoint_2[0] - endpoint_1[0])
            )
            if determinant > 0:
                above_line += 1
            elif determinant < 0:
                below_line += 1

        total = below_line + above_line
        if total == 0:
            return 0.0, 0.0

        anterior_percent = (below_line / total) * 100.0
        posterior_percent = (above_line / total) * 100.0
        return anterior_percent, posterior_percent

    @staticmethod
    def _box_geometry(point1: Point2D, point2: Point2D, frame_index: int) -> dict[str, object]:
        return {
            "type": "square",
            "viewPlaneNormal": None,
            "viewUp": None,
            "coordinate_system": "pixel",
            "points": (
                (int(point1[0]), int(point1[1]), int(frame_index)),
                (int(point2[0]), int(point2[1]), int(frame_index)),
            ),
        }

    @staticmethod
    def _line_geometry(point1: Point2D, point2: Point2D, frame_index: int) -> dict[str, object]:
        return {
            "type": "line",
            "viewPlaneNormal": None,
            "viewUp": None,
            "coordinate_system": "pixel",
            "points": (
                (int(point1[0]), int(point1[1]), int(frame_index)),
                (int(point2[0]), int(point2[1]), int(frame_index)),
            ),
        }

    @classmethod
    def _create_line_annotation(
        cls,
        point1: Point2D,
        point2: Point2D,
        *,
        identifier: str,
        frame_index: int,
        numeric_value: float | None = None,
        units: str | None = None,
        text_value: str | None = None,
        user_info: dict[str, object] | None = None,
        metadata: object = None,
    ) -> LineAnnotation:
        return LineAnnotation.from_points(
            point1,
            point2,
            identifier=identifier,
            frame_index=frame_index,
            metadata=metadata,
            numeric_value=numeric_value,
            units=units,
            text_value=text_value,
            user_info=user_info,
        )

    @classmethod
    def _create_point_box_annotation(
        cls,
        point: Point2D,
        *,
        identifier: str,
        frame_index: int,
        width: int,
        height: int,
        text_value: str | None = None,
        user_info: dict[str, object] | None = None,
    ) -> BoxAnnotation:
        x_value, y_value = point
        top_left = (
            max(x_value - POINT_BOX_HALF_SIZE, 0),
            max(y_value - POINT_BOX_HALF_SIZE, 0),
        )
        bottom_right = (
            min(x_value + POINT_BOX_HALF_SIZE, width - 1),
            min(y_value + POINT_BOX_HALF_SIZE, height - 1),
        )
        return BoxAnnotation(
            identifier=identifier,
            frame_index=frame_index,
            geometry=cls._box_geometry(top_left, bottom_right, frame_index),
            text_value=text_value,
            user_info=user_info,
        )

    def _create_3d_segmentations(self, full_pred_volume: np.ndarray) -> list[VolumeSegmentation]:
        annotations: list[VolumeSegmentation] = []
        for class_idx, class_name in self.class_names.items():
            if class_idx == 0:
                continue
            mask_3d = (full_pred_volume == class_idx).astype(np.uint8) * 255
            if mask_3d.any():
                annotations.append(
                    VolumeSegmentation.from_semantic_segmentation(
                        segmentation=mask_3d,
                        class_map=class_name,
                    )
                )
        return annotations

    def _create_slice_annotations(
        self,
        full_pred_volume: np.ndarray,
        spacing: tuple[float, float],
        metadata: object = None,
    ) -> list[LineAnnotation | BoxAnnotation]:
        annotations: list[LineAnnotation | BoxAnnotation] = []
        height, width, z_slices = full_pred_volume.shape
        slice_spacing = float(spacing[0])

        for frame_index in range(z_slices):
            mask_slice = full_pred_volume[:, :, frame_index]
            mask_display = self._to_display_space(mask_slice)

            disc_mask = mask_display == DISC_CLASS_ID
            condyle_mask = mask_display == CONDYLE_CLASS_ID
            eminence_mask = mask_display == EMINENCE_CLASS_ID

            disc_points = self._mask_to_points(disc_mask)
            if not disc_points:
                continue

            disc_centroid_display = self._get_mean_point(disc_points)
            disc_line_points, disc_length_px, disc_line_start, disc_line_end = self._make_disc_line(disc_points)

            if disc_centroid_display is not None:
                disc_centroid = disc_centroid_display
                annotations.append(
                    self._create_point_box_annotation(
                        disc_centroid,
                        identifier="Q (Disc Center)",
                        frame_index=frame_index,
                        width=width,
                        height=height,
                        user_info={"slice_index": frame_index, "role": "disc_center"},
                    )
                )
            else:
                disc_centroid = None

            if disc_line_start is not None and disc_line_end is not None:
                annotations.append(
                    self._create_line_annotation(
                        disc_line_start,
                        disc_line_end,
                        identifier="Disc Line",
                        frame_index=frame_index,
                        numeric_value=float(disc_length_px),
                        units="px",
                        user_info={
                            "slice_index": frame_index,
                            "line_role": "disc_best_fit",
                            "point_count": len(disc_line_points or []),
                        },
                        metadata=metadata,
                    )
                )

            if (
                disc_centroid_display is None
                or disc_line_end is None
                or not condyle_mask.any()
                or not eminence_mask.any()
            ):
                continue

            condyle_points = self._mask_to_points(condyle_mask)
            eminence_points = self._mask_to_points(eminence_mask)
            emcon_line_points, emcon_lengths, emcon_point_1, emcon_point_2 = self._make_emcon_line(
                eminence_points,
                condyle_points,
                disc_line_end,
            )
            if (
                emcon_line_points is None
                or emcon_lengths is None
                or emcon_point_1 is None
                or emcon_point_2 is None
            ):
                continue

            emcon_midpoint_display = self._find_midpoint(emcon_line_points)
            if emcon_midpoint_display is None or disc_centroid is None:
                continue

            # emcon_midpoint = self._display_to_volume_point(emcon_midpoint_display)
            # emcon_start = self._display_to_volume_point(emcon_point_1)
            # emcon_end = self._display_to_volume_point(emcon_point_2)
            emcon_midpoint = emcon_midpoint_display
            emcon_start = emcon_point_1
            emcon_end = emcon_point_2

            distance_px = self._get_distances_between_two_points(
                disc_centroid_display,
                emcon_midpoint_display,
            )[2]
            distance_mm = distance_px * slice_spacing
            anterior_percent, posterior_percent = self._calculate_ratios(disc_mask, emcon_line_points)
            status = (
                "Normal"
                if distance_mm < NORMAL_DISTANCE_THRESHOLD_MM
                else "Anterior Dislocated (ADD)"
            )

            annotations.append(
                self._create_line_annotation(
                    emcon_start,
                    emcon_end,
                    identifier="EMCON Line",
                    frame_index=frame_index,
                    numeric_value=float(emcon_lengths[2]),
                    units="px",
                    user_info={
                        "slice_index": frame_index,
                        "line_role": "emcon",
                        "distance_x_px": int(emcon_lengths[0]),
                        "distance_y_px": int(emcon_lengths[1]),
                    },
                    metadata=metadata,
                )
            )
            annotations.append(
                self._create_line_annotation(
                    disc_centroid,
                    emcon_midpoint,
                    identifier="Q-P Distance",
                    frame_index=frame_index,
                    numeric_value=float(distance_mm),
                    units="mm",
                    text_value=status,
                    user_info={
                        "slice_index": frame_index,
                        "line_role": "dislocation_distance",
                        "distance_px": float(distance_px),
                        "distance_mm": float(distance_mm),
                        "anterior_percent": float(anterior_percent),
                        "posterior_percent": float(posterior_percent),
                        "status": status,
                    },
                    metadata=metadata,
                )
            )
            annotations.append(
                self._create_point_box_annotation(
                    emcon_midpoint,
                    identifier="P (EMCON Midpoint)",
                    frame_index=frame_index,
                    width=width,
                    height=height,
                    text_value=status,
                    user_info={
                        "slice_index": frame_index,
                        "role": "emcon_midpoint",
                        "distance_mm": float(distance_mm),
                    },
                )
            )
            annotations.append(
                self._create_point_box_annotation(
                    emcon_start,
                    identifier="Closest Condyle Point",
                    frame_index=frame_index,
                    width=width,
                    height=height,
                    user_info={"slice_index": frame_index, "role": "closest_condyle"},
                )
            )
            annotations.append(
                self._create_point_box_annotation(
                    emcon_end,
                    identifier="Closest Eminence Point",
                    frame_index=frame_index,
                    width=width,
                    height=height,
                    user_info={"slice_index": frame_index, "role": "closest_eminence"},
                )
            )

        return annotations

    @override
    def predict_volume(self, model_input: list[Resource], **kwargs) -> list[list[VolumeSegmentation | LineAnnotation | BoxAnnotation]]:
        """Process 3D medical imaging volumes and predict TMJ anatomical structures with annotations.

        Args:
            model_input: List of Resource objects containing medical image data (NIFTI, DICOM, etc.)
            **kwargs: Additional keyword arguments passed to parent class

        Returns:
            list[list[VolumeSegmentation | LineAnnotation | BoxAnnotation]]: 
                For each input resource, returns a list of annotations containing:
                
                1. VolumeSegmentation (3D):
                   - One per segmented class (Disc, Condyle, Eminence)
                   - Binary masks marking anatomical structures in 3D space
                   - Each pixel value is 0 (background) or 255 (structure)
                
                2. LineAnnotation (slice-level):
                   Identifies:
                   - "Disc Line": Best-fit line through disc segmentation
                     * numeric_value: Length in pixels
                     * units: "px"
                     * user_info["line_role"]: "disc_best_fit"
                     * user_info["point_count"]: Number of points forming the line
                   
                   - "EMCON Line": Line connecting closest condyle and eminence points
                     * numeric_value: Euclidean distance in pixels
                     * units: "px"
                     * user_info["line_role"]: "emcon"
                     * user_info["distance_x_px"]: Horizontal distance
                     * user_info["distance_y_px"]: Vertical distance
                   
                   - "Q-P Distance": Distance from disc center to EMCON midpoint
                     * numeric_value: Distance in millimeters
                     * units: "mm"
                     * text_value: "Normal" or "Anterior Dislocated (ADD)"
                     * user_info["distance_px"]: Distance in pixels
                     * user_info["distance_mm"]: Distance in millimeters
                     * user_info["anterior_percent"]: Percentage of disc anterior to EMCON line
                     * user_info["posterior_percent"]: Percentage of disc posterior to EMCON line
                     * user_info["status"]: "Normal" or "Anterior Dislocated (ADD)"
                
                3. BoxAnnotation (slice-level):
                   Identifies anatomical point locations:
                   - "Q (Disc Center)": Centroid of disc segmentation
                     * user_info["role"]: "disc_center"
                   
                   - "P (EMCON Midpoint)": Midpoint of EMCON line
                     * user_info["role"]: "emcon_midpoint"
                     * user_info["distance_mm"]: Distance to disc center
                   
                   - "Closest Condyle Point": Closest point in condyle to eminence
                     * user_info["role"]: "closest_condyle"
                   
                   - "Closest Eminence Point": Closest point in eminence to condyle
                     * user_info["role"]: "closest_eminence"

        Notes:
            - Returns empty list if input volume has invalid dimensions (not 3D)
            - All slice-level annotations include frame_index referencing the z-slice
            - Spatial coordinates are in pixel space with metadata spacing available
            - Status determination uses NORMAL_DISTANCE_THRESHOLD_MM = 10.0 mm
        """
        pytorch_model = self.get_mlflow_torch_models()["unetpp"]
        pytorch_model.eval()

        fabric = L.Fabric(accelerator=self.inference_device)
        pytorch_model = fabric.setup_module(pytorch_model)

        all_predictions = []
        with torch.inference_mode():
            for resource in model_input:
                data = resource.fetch_file_data(auto_convert=False, use_cache=True)
                volume, metadata = read_array_normalized(data, mime_type=resource.mimetype,
                                                         return_metainfo=True)
                volume = volume[:, 0, :, :]
                volume = np.transpose(volume, (2, 1, 0))

                if volume.ndim != 3:
                    _LOGGER.warning(f"Unexpected volume dimensions: {volume.ndim}. Expected a 3D volume.")
                    all_predictions.append([])
                    continue

                # metadata = resource.fetch_file_data(auto_convert=True, use_cache=True)
                spacing = self._extract_spacing(metadata)
                height, width, z_slices = volume.shape
                full_pred_volume = np.zeros((height, width, z_slices), dtype=np.uint8)

                for frame_index in range(z_slices):
                    slice_2d = volume[:, :, frame_index]
                    input_tensor = self._preprocess(slice_2d).to(fabric.device)
                    logits = pytorch_model(input_tensor)  # shape (1, num_classes+1, height, width)
                    pred_2d = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
                    full_pred_volume[:, :, frame_index] = self._undo_transforms(pred_2d, (height, width))

                annotations: list[VolumeSegmentation | LineAnnotation | BoxAnnotation] = []
                annotations.extend(self._create_3d_segmentations(full_pred_volume))
                annotations.extend(self._create_slice_annotations(full_pred_volume, spacing, metadata))
                all_predictions.append(annotations)

        return all_predictions

    def predict_default(self, model_input: list[Resource], **kwargs) -> list[list[VolumeSegmentation | LineAnnotation | BoxAnnotation]]:
        """Default prediction method (delegates to predict_volume).
        
        See predict_volume() for detailed documentation of outputs and annotation types.
        """
        return self.predict_volume(model_input, **kwargs)
