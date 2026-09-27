"""Per-line-exact forward projection and AD reconstruction.

The continuous-rotation generator stores volumes as ``[z, y, x]`` and images
as ``[Y, X]``.  FastTomo stores volumes as ``[Ny, Nz, Nx]``, where ``Ny`` is
the independent tilt-axis dimension, and returns sinograms as
``[angle, Ny, detector]``.  Consequently the mappings used here are::

    generator volume [z, y, x] -> FastTomo volume [x, z, y]
    generator image  [Y, X]    -> FastTomo image  [X, Y]

The angle sign is unchanged.  For fast-par scans one detector line is gathered
from each expanded projection; for fast-perp scans one tilt-axis slice is
gathered.  Reconstruction calls backward once per pseudo-angle chunk so a
36,000-angle dataset never materialises a 36,000-projection autograd graph.

At even spatial sizes FastTomo's ray-marched forward operator is not identical
to the generator's rotate-then-sum operator.  Production-sized 180/200 data is
expected to have a roughly 2--5% relative model mismatch and a corresponding
reconstruction fidelity floor around 1e-3.  This accepted inverse-crime
avoidance is an integration-scheme difference, not a centering bug.

Pose refinement exactly mirrors the generator's operation order. FastTomo's
verified scalar projector first predicts each full nominal-angle projection.
That projection is transposed to generator image order ``[Y, X]``, rotated in
2-D about the image centre by post-projection ``z3=phi``, translated by
``(dy, dx)``, and only then sampled at the measured row/column. Both image
transforms use bilinear ``grid_sample`` with ``align_corners=False`` and zero
padding, matching the generator. The only remaining discrepancy is the known
base scalar-projector integration floor described above; it does not grow with
``abs(theta) * abs(phi)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import fasttomo as ft


Arm = Literal["fast-par", "fast-perp"]
LossName = Literal["mse", "l1"]
POSE_SHIFT_BOUND_PX = 15.0
POSE_AXIS_BOUND_DEG = 2.5
POSE_CONVENTION = (
    "Forward-model pose in generator coordinates: shifts are (dy, dx) pixels "
    "toward increasing image indices and axis_deg is the in-plane tilt-axis "
    "angle phi in degrees. Perfect recovery is learned ~= dataset GT (same "
    "sign). The scalar FastTomo projector predicts the nominal-theta image, "
    "which is transposed to generator [Y,X] order; phi is then applied exactly "
    "as the generator's post-projection in-plane z3 rotation about image centre, "
    "followed by translation with output(y,x)=input(y-dy,x-dx). Rotation and "
    "translation are separate bilinear grid_sample operations with zero padding "
    "and align_corners=False. Phi and both shift signs therefore match dataset "
    "GT directly. No ZYZ Euler approximation is used; residual mismatch is only "
    "the nominal scalar-projector integration floor."
)


@dataclass(frozen=True)
class ContinuousDataset:
    """Validated continuous-rotation inputs loaded from one dataset directory."""

    path: Path
    tilt_series: np.ndarray
    angles: np.ndarray
    angles_per_line: np.ndarray
    valid_lines: np.ndarray
    arm: Arm
    line_angle_path: Path
    valid_lines_path: Path | None
    generator_manifest: dict[str, Any]
    input_paths: dict[str, Path]

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        return tuple(int(size) for size in self.tilt_series.shape)


def detect_continuous_arm(dataset: str | Path) -> tuple[Arm, Path]:
    """Return the scan arm and its sole per-line angle file."""
    dataset_path = Path(dataset).expanduser().resolve()
    row_path = dataset_path / "angles_per_row.npy"
    col_path = dataset_path / "angles_per_col.npy"
    if row_path.is_file() == col_path.is_file():
        raise ValueError(
            "dataset must contain exactly one of angles_per_row.npy or "
            "angles_per_col.npy"
        )
    return ("fast-par", row_path) if row_path.is_file() else ("fast-perp", col_path)


def load_continuous_dataset(dataset: str | Path) -> ContinuousDataset:
    """Load and validate a continuous-rotation generator dataset."""
    dataset_path = Path(dataset).expanduser().resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {dataset_path}")

    arm, line_angle_path = detect_continuous_arm(dataset_path)
    input_paths = {
        "tilt_series.npy": dataset_path / "tilt_series.npy",
        "angles.npy": dataset_path / "angles.npy",
        line_angle_path.name: line_angle_path,
        "gen_manifest.yaml": dataset_path / "gen_manifest.yaml",
    }
    valid_lines_path = dataset_path / "valid_lines.npy"
    if valid_lines_path.is_file():
        input_paths[valid_lines_path.name] = valid_lines_path
    missing = [str(path) for path in input_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing required dataset inputs: {', '.join(missing)}"
        )

    with input_paths["gen_manifest.yaml"].open("r", encoding="utf-8") as stream:
        loaded_manifest = yaml.safe_load(stream)
    if not isinstance(loaded_manifest, dict):
        raise ValueError("gen_manifest.yaml must contain a mapping")
    manifest_arm = loaded_manifest.get("scan_orientation")
    if manifest_arm is not None and manifest_arm != arm:
        raise ValueError(
            f"gen_manifest scan_orientation={manifest_arm!r} disagrees with {arm!r}"
        )

    tilt_series = np.load(input_paths["tilt_series.npy"], allow_pickle=False)
    angles = np.load(input_paths["angles.npy"], allow_pickle=False)
    angles_per_line = np.load(line_angle_path, allow_pickle=False)
    if tilt_series.ndim != 3:
        raise ValueError(f"tilt_series.npy must be 3-D, got {tilt_series.shape}")
    n_frames, image_y, image_x = (int(size) for size in tilt_series.shape)
    if angles.shape != (n_frames,):
        raise ValueError(
            f"angles.npy must have shape {(n_frames,)}, got {angles.shape}"
        )
    n_lines = image_y if arm == "fast-par" else image_x
    if angles_per_line.shape != (n_frames, n_lines):
        raise ValueError(
            f"{line_angle_path.name} must have shape {(n_frames, n_lines)}, "
            f"got {angles_per_line.shape}"
        )
    if valid_lines_path.is_file():
        valid_lines = np.load(valid_lines_path, allow_pickle=False)
        if valid_lines.dtype != np.bool_:
            raise ValueError(
                f"valid_lines.npy must have boolean dtype, got {valid_lines.dtype}"
            )
        if valid_lines.shape != (n_frames, n_lines):
            raise ValueError(
                f"valid_lines.npy must have shape {(n_frames, n_lines)}, "
                f"got {valid_lines.shape}"
            )
        if not valid_lines.any():
            raise ValueError("valid_lines.npy must retain at least one line")
    else:
        valid_lines = np.ones((n_frames, n_lines), dtype=np.bool_)
    for name, array in (
        ("tilt_series.npy", tilt_series),
        ("angles.npy", angles),
        (line_angle_path.name, angles_per_line),
    ):
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains a non-finite value")

    return ContinuousDataset(
        path=dataset_path,
        tilt_series=tilt_series,
        angles=angles,
        angles_per_line=angles_per_line,
        valid_lines=valid_lines,
        arm=arm,
        line_angle_path=line_angle_path,
        valid_lines_path=valid_lines_path if valid_lines_path.is_file() else None,
        generator_manifest=loaded_manifest,
        input_paths=input_paths,
    )


def generator_to_fasttomo_volume(volume_zyx: torch.Tensor) -> torch.Tensor:
    """Permute a generator-frame ``[z, y, x]`` volume to ``[x, z, y]``."""
    if volume_zyx.ndim != 3:
        raise ValueError(f"volume must be three-dimensional, got {volume_zyx.shape}")
    return volume_zyx.permute(2, 0, 1).contiguous()


def generator_to_fasttomo_sinogram(tilt_series_kyx: torch.Tensor) -> torch.Tensor:
    """Transpose generator frames ``[k, Y, X]`` to FastTomo ``[k, X, Y]``."""
    if tilt_series_kyx.ndim != 3:
        raise ValueError(
            "tilt series must be three-dimensional, "
            f"got {tilt_series_kyx.shape}"
        )
    return tilt_series_kyx.permute(0, 2, 1).contiguous()


def fasttomo_to_generator_volume(volume_xzy: torch.Tensor) -> torch.Tensor:
    """Permute a FastTomo ``[x, z, y]`` volume back to generator ``[z, y, x]``."""
    if volume_xzy.ndim != 3:
        raise ValueError(f"volume must be three-dimensional, got {volume_xzy.shape}")
    return volume_xzy.permute(1, 2, 0).contiguous()


@dataclass(frozen=True)
class PerLineGeometry:
    """Deduplicated expanded-angle geometry for one continuous tilt series."""

    arm: Arm
    frame_shape: tuple[int, int, int]
    angles_deg: torch.Tensor
    line_to_angle: torch.Tensor

    @classmethod
    def from_angles(
        cls,
        angles_per_line: np.ndarray | torch.Tensor,
        *,
        arm: Arm,
        frame_shape: tuple[int, int, int],
    ) -> "PerLineGeometry":
        """Build geometry from ``[n_frames, n_lines]`` exact angles.

        ``frame_shape`` is always in generator image order
        ``(n_frames, image_Y, image_X)``.  Deduplication is exact in float32,
        matching the angle precision consumed by FastTomo.
        """
        if arm not in ("fast-par", "fast-perp"):
            raise ValueError(f"arm must be 'fast-par' or 'fast-perp', got {arm!r}")
        if len(frame_shape) != 3 or any(int(size) < 1 for size in frame_shape):
            raise ValueError(f"invalid frame shape {frame_shape}")

        per_line = torch.as_tensor(angles_per_line, dtype=torch.float32, device="cpu")
        if per_line.ndim != 2:
            raise ValueError(
                "angles_per_line must have shape [n_frames, n_lines], "
                f"got {tuple(per_line.shape)}"
            )
        n_frames, image_y, image_x = (int(size) for size in frame_shape)
        expected = image_y if arm == "fast-par" else image_x
        if tuple(per_line.shape) != (n_frames, expected):
            raise ValueError(
                f"{arm} angles must have shape {(n_frames, expected)}, "
                f"got {tuple(per_line.shape)}"
            )
        if not torch.isfinite(per_line).all():
            raise ValueError("angles_per_line contains a non-finite value")

        unique, inverse = torch.unique(
            per_line.reshape(-1), sorted=True, return_inverse=True
        )
        return cls(
            arm=arm,
            frame_shape=(n_frames, image_y, image_x),
            angles_deg=unique.contiguous(),
            line_to_angle=inverse.contiguous(),
        )

    @property
    def n_pseudo_angles(self) -> int:
        return int(self.angles_deg.numel())

    @property
    def n_lines(self) -> int:
        """Total measured lines across all frames."""
        return int(self.line_to_angle.numel())

    @property
    def n_scan_lines(self) -> int:
        """Measured rows/columns in one frame."""
        _, image_y, image_x = self.frame_shape
        return image_y if self.arm == "fast-par" else image_x

    @property
    def fasttomo_volume_shape(self) -> tuple[int, int, int]:
        """Default cubic-depth FastTomo shape ``[image_X, image_Y, image_Y]``."""
        _, image_y, image_x = self.frame_shape
        return image_x, image_y, image_y

    def _chunk_slots(self, start: int, stop: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        inverse = self.line_to_angle
        mask = (inverse >= start) & (inverse < stop)
        slots = torch.nonzero(mask, as_tuple=False).flatten()
        local_angles = inverse[slots] - start
        return slots.to(device=device), local_angles.to(device=device)

    def _gather_chunk(
        self,
        expanded: torch.Tensor,
        slots: torch.Tensor,
        local_angles: torch.Tensor,
    ) -> torch.Tensor:
        """Gather valid measured lines from ``[chunk, image_X, image_Y]``."""
        _, image_y, image_x = self.frame_shape
        line_index = torch.remainder(slots, self.n_scan_lines)
        if self.arm == "fast-par":
            # One detector coordinate, shared across every tilt-axis slice.
            return expanded[local_angles, :, line_index]
        # One independent tilt-axis slice, retaining every detector coordinate.
        return expanded[local_angles, line_index, :]

    def measurement_lines(self, measurements: torch.Tensor) -> torch.Tensor:
        """View generator-frame measurements in the gather operator's line order."""
        if tuple(measurements.shape) != self.frame_shape:
            raise ValueError(
                f"measurements must have shape {self.frame_shape}, "
                f"got {tuple(measurements.shape)}"
            )
        if self.arm == "fast-par":
            return measurements.reshape(self.n_lines, self._line_width)
        return measurements.permute(0, 2, 1).reshape(
            self.n_lines, self._line_width
        )

    def predict(
        self,
        volume: torch.Tensor,
        *,
        chunk_size: int,
        num_threads: int | None = None,
        shifts_per_line: torch.Tensor | None = None,
        axis_deg_per_line: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return exact predicted frames in generator image order ``[k, Y, X]``.

        This convenience method is intended for inference and correctness
        checks.  :func:`reconstruct` uses immediate per-chunk backward calls
        instead, which bounds training memory more tightly.
        """
        _validate_volume(volume, self)
        _validate_chunk(chunk_size)
        if shifts_per_line is not None or axis_deg_per_line is not None:
            shifts, axis = _validate_pose_lines(
                shifts_per_line, axis_deg_per_line, self, volume.device
            )
            return self._predict_with_pose(
                volume,
                shifts,
                axis,
                chunk_size=chunk_size,
                num_threads=num_threads,
            )

        device = volume.device
        # Zero initialization prevents future incomplete slot coverage from leaking garbage.
        gathered = volume.new_zeros((self.n_lines, self._line_width))
        for start in range(0, self.n_pseudo_angles, chunk_size):
            stop = min(start + chunk_size, self.n_pseudo_angles)
            slots, local = self._chunk_slots(start, stop, device)
            expanded = ft.forward_project(
                volume,
                self.angles_deg[start:stop].to(device=device),
                N_det=self.frame_shape[1],
                num_threads=num_threads,
            )
            gathered = gathered.index_copy(0, slots, self._gather_chunk(expanded, slots, local))

        n_frames, image_y, image_x = self.frame_shape
        if self.arm == "fast-par":
            return gathered.reshape(n_frames, image_y, image_x)
        return gathered.reshape(n_frames, image_x, image_y).permute(0, 2, 1).contiguous()

    def _predict_with_pose(
        self,
        volume: torch.Tensor,
        shifts_per_line: torch.Tensor,
        axis_deg_per_line: torch.Tensor,
        *,
        chunk_size: int,
        num_threads: int | None,
    ) -> torch.Tensor:
        """Project every measured line with its generator-frame pose.

        Pose prevents angle deduplication because equal nominal angles may have
        different shifts or axis deviations.  The flattened line order is the
        same global time order used by the acquisition generator.
        """
        device = volume.device
        gathered = volume.new_zeros((self.n_lines, self._line_width))
        shifts_flat = shifts_per_line.reshape(self.n_lines, 2)
        axis_flat = axis_deg_per_line.reshape(self.n_lines)
        nominal_flat = self.angles_deg[self.line_to_angle].to(device=device)
        for start in range(0, self.n_lines, chunk_size):
            stop = min(start + chunk_size, self.n_lines)
            slots = torch.arange(start, stop, device=device)
            local = torch.arange(stop - start, device=device)
            expanded = _forward_project_pose_chunk(
                volume,
                nominal_flat[start:stop],
                shifts_flat[start:stop],
                axis_flat[start:stop],
                detector_size=self.frame_shape[1],
                num_threads=num_threads,
            )
            gathered = gathered.index_copy(
                0, slots, self._gather_chunk(expanded, slots, local)
            )

        n_frames, image_y, image_x = self.frame_shape
        if self.arm == "fast-par":
            return gathered.reshape(n_frames, image_y, image_x)
        return gathered.reshape(n_frames, image_x, image_y).permute(0, 2, 1).contiguous()

    @property
    def _line_width(self) -> int:
        _, image_y, image_x = self.frame_shape
        return image_x if self.arm == "fast-par" else image_y


@dataclass(frozen=True)
class ADConfig:
    """Configuration for :func:`reconstruct`."""

    iters: int = 1000
    lr: float = 0.05
    tv: float = 0.0
    sparsity: float = 0.0
    chunk: int = 16
    loss: LossName = "mse"
    seed: int = 0
    num_threads: int | None = None
    refine_pose: bool = False
    pose_lr_shifts: float = 1e-2
    pose_lr_angles: float = 1e-2
    pose_start: int = 0
    anchor_reference: bool = False

    def validate(self) -> None:
        if self.iters < 1:
            raise ValueError("iters must be at least one")
        if not np.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be positive and finite")
        if not np.isfinite(self.tv) or self.tv < 0:
            raise ValueError("tv must be nonnegative and finite")
        if not np.isfinite(self.sparsity) or self.sparsity < 0:
            raise ValueError("sparsity must be nonnegative and finite")
        _validate_chunk(self.chunk)
        if self.loss not in ("mse", "l1"):
            raise ValueError(f"loss must be 'mse' or 'l1', got {self.loss!r}")
        if self.num_threads is not None and self.num_threads < 1:
            raise ValueError("num_threads must be at least one when supplied")
        if not np.isfinite(self.pose_lr_shifts) or self.pose_lr_shifts <= 0:
            raise ValueError("pose_lr_shifts must be positive and finite")
        if not np.isfinite(self.pose_lr_angles) or self.pose_lr_angles <= 0:
            raise ValueError("pose_lr_angles must be positive and finite")
        if (
            isinstance(self.pose_start, bool)
            or not isinstance(self.pose_start, int)
            or self.pose_start < 0
        ):
            raise ValueError("pose_start must be a nonnegative integer")
        if self.anchor_reference and not self.refine_pose:
            raise ValueError("anchor_reference requires refine_pose=True")


@dataclass(frozen=True)
class LossRecord:
    iteration: int
    total: float
    fidelity: float
    tv: float
    sparsity: float


@dataclass(frozen=True)
class ADResult:
    volume: torch.Tensor
    loss_log: list[LossRecord]
    poses: LearnedPoses | None = None


@dataclass(frozen=True)
class LearnedPoses:
    """Learned or fixed poses in the generator/dataset coordinate convention."""

    shifts_frames: torch.Tensor
    axis_deg_frames: torch.Tensor
    shifts_per_line: torch.Tensor
    axis_deg_per_line: torch.Tensor
    convention: str = POSE_CONVENTION
    anchor_reference: bool = False
    anchor_frame_index: int | None = None


def interpolate_pose_nodes(nodes: torch.Tensor, n_lines: int) -> torch.Tensor:
    """Linearly interpolate frame-center nodes onto global scan-line times.

    Frame ``f`` is centred at global line
    ``f * n_lines + (n_lines - 1) / 2``. Values before the first centre and
    after the last centre use clamped extrapolation.
    """
    if nodes.ndim < 1 or nodes.shape[0] < 1:
        raise ValueError("nodes must have shape [n_frames, ...] with n_frames >= 1")
    if isinstance(n_lines, bool) or not isinstance(n_lines, int) or n_lines < 1:
        raise ValueError("n_lines must be a positive integer")

    n_frames = int(nodes.shape[0])
    flat_time = torch.arange(
        n_frames * n_lines, device=nodes.device, dtype=nodes.dtype
    )
    frame_coordinate = (flat_time - (n_lines - 1) / 2.0) / float(n_lines)
    left_unclamped = torch.floor(frame_coordinate).to(dtype=torch.long)
    fraction = frame_coordinate - left_unclamped.to(dtype=nodes.dtype)
    left = left_unclamped.clamp(0, n_frames - 1)
    right = (left_unclamped + 1).clamp(0, n_frames - 1)
    trailing_dims = (1,) * (nodes.ndim - 1)
    fraction = fraction.reshape(-1, *trailing_dims)
    interpolated = nodes[left] * (1.0 - fraction) + nodes[right] * fraction
    return interpolated.reshape(n_frames, n_lines, *nodes.shape[1:])


def tv_charbonnier(volume: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean isotropic 3-D Charbonnier total variation."""
    if min(volume.shape) < 2:
        return volume.sum() * 0.0
    d0 = volume[1:, :, :] - volume[:-1, :, :]
    d1 = volume[:, 1:, :] - volume[:, :-1, :]
    d2 = volume[:, :, 1:] - volume[:, :, :-1]
    return (
        d0[:, :-1, :-1].square()
        + d1[:-1, :, :-1].square()
        + d2[:-1, :-1, :].square()
        + eps
    ).sqrt().mean()


def reconstruct(
    tilt_series: np.ndarray | torch.Tensor,
    geometry: PerLineGeometry,
    *,
    config: ADConfig,
    device: str | torch.device,
    fixed_shifts_per_line: np.ndarray | torch.Tensor | None = None,
    fixed_axis_deg_per_line: np.ndarray | torch.Tensor | None = None,
    initial_shifts_frames: np.ndarray | torch.Tensor | None = None,
    initial_axis_deg_frames: np.ndarray | torch.Tensor | None = None,
    valid_lines: np.ndarray | torch.Tensor | None = None,
) -> ADResult:
    """Reconstruct a nonnegative volume with per-line-exact data fidelity.

    Fidelity is divided by the total number of measured pixels before each
    chunk's backward call, so accumulated gradients are exactly those of a
    mean MSE/L1 loss over valid measured pixels. Invalid lines contribute no
    loss or gradients. TV and sparsity are evaluated once per iteration.
    """
    config.validate()
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    target = torch.as_tensor(tilt_series, dtype=torch.float32, device=resolved_device)
    if tuple(target.shape) != geometry.frame_shape:
        raise ValueError(
            f"tilt_series shape must be {geometry.frame_shape}, got {tuple(target.shape)}"
        )
    if not torch.isfinite(target).all():
        raise ValueError("tilt_series contains a non-finite value")
    valid_line_mask = _validate_valid_lines(valid_lines, geometry, resolved_device)

    fixed_requested = (
        fixed_shifts_per_line is not None or fixed_axis_deg_per_line is not None
    )
    initial_requested = (
        initial_shifts_frames is not None or initial_axis_deg_frames is not None
    )
    if fixed_requested and (
        fixed_shifts_per_line is None or fixed_axis_deg_per_line is None
    ):
        raise ValueError("both fixed per-line pose arrays must be supplied together")
    if initial_requested and (
        initial_shifts_frames is None or initial_axis_deg_frames is None
    ):
        raise ValueError("both initial frame pose arrays must be supplied together")
    if fixed_requested and config.refine_pose:
        raise ValueError("fixed per-line poses cannot be combined with pose refinement")
    if fixed_requested and initial_requested:
        raise ValueError("fixed and initialized learnable poses are mutually exclusive")
    if initial_requested and not config.refine_pose:
        raise ValueError("initial frame poses require refine_pose=True")

    fixed_shifts = None
    fixed_axis = None
    if fixed_requested:
        fixed_shifts, fixed_axis = _validate_pose_lines(
            fixed_shifts_per_line,
            fixed_axis_deg_per_line,
            geometry,
            resolved_device,
        )

    torch.manual_seed(config.seed)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    volume = torch.zeros(
        geometry.fasttomo_volume_shape,
        dtype=torch.float32,
        device=resolved_device,
        requires_grad=True,
    )
    optimizer_groups: list[dict[str, Any]] = [
        {"params": [volume], "lr": config.lr}
    ]
    raw_shifts = None
    raw_axis = None
    if config.refine_pose:
        n_frames = geometry.frame_shape[0]
        if config.anchor_reference and n_frames < 2:
            raise ValueError(
                "anchor_reference requires at least one ramp frame plus the "
                "final reference frame"
            )
        n_optimized_frames = n_frames - int(config.anchor_reference)
        if initial_requested:
            shifts_init, axis_init = _validate_pose_nodes(
                initial_shifts_frames,
                initial_axis_deg_frames,
                n_frames=n_frames,
                device=resolved_device,
            )
            # The appended reference is a separate calibration exposure, not
            # part of scan time. Its supplied initialization is intentionally
            # ignored and its physical pose is constructed as exact zero.
            shifts_init = shifts_init[:n_optimized_frames]
            axis_init = axis_init[:n_optimized_frames]
            raw_shifts_init, raw_axis_init = _raw_from_bounded_pose(
                shifts_init, axis_init
            )
        else:
            raw_shifts_init = torch.zeros(
                (n_optimized_frames, 2), device=resolved_device
            )
            raw_axis_init = torch.zeros(n_optimized_frames, device=resolved_device)
        raw_shifts = raw_shifts_init.detach().requires_grad_(True)
        raw_axis = raw_axis_init.detach().requires_grad_(True)
        optimizer_groups.extend(
            (
                {"params": [raw_shifts], "lr": config.pose_lr_shifts},
                {"params": [raw_axis], "lr": config.pose_lr_angles},
            )
        )
    optimizer = torch.optim.Adam(optimizer_groups)
    loss_log: list[LossRecord] = []

    for iteration in range(1, config.iters + 1):
        optimizer.zero_grad(set_to_none=True)
        if fixed_requested:
            assert fixed_shifts is not None and fixed_axis is not None
            fidelity_value = _backward_fidelity_fixed_pose_chunks(
                volume,
                target,
                geometry,
                config,
                fixed_shifts,
                fixed_axis,
                valid_line_mask,
            )
        elif config.refine_pose:
            assert raw_shifts is not None and raw_axis is not None
            # pose_start is a zero-based iteration index: 0 activates pose in
            # the first iteration, while N performs N volume-only iterations.
            pose_active = iteration - 1 >= config.pose_start
            fidelity_value = _backward_fidelity_pose_chunks(
                volume,
                target,
                geometry,
                config,
                raw_shifts,
                raw_axis,
                valid_line_mask,
                pose_active=pose_active,
            )
        else:
            fidelity_value = _backward_fidelity_chunks(
                volume, target, geometry, config, valid_line_mask
            )

        tv_term = (
            config.tv * tv_charbonnier(volume)
            if config.tv > 0.0
            else volume.new_zeros(())
        )
        sparsity_term = (
            config.sparsity * volume.abs().mean()
            if config.sparsity > 0.0
            else volume.new_zeros(())
        )
        regularization = tv_term + sparsity_term
        if regularization.requires_grad:
            regularization.backward()

        tv_value = float(tv_term.detach())
        sparsity_value = float(sparsity_term.detach())
        total_value = fidelity_value + tv_value + sparsity_value
        loss_log.append(
            LossRecord(iteration, total_value, fidelity_value, tv_value, sparsity_value)
        )

        optimizer.step()
        with torch.no_grad():
            volume.clamp_(min=0.0)

    learned_poses = None
    if fixed_requested:
        assert fixed_shifts is not None and fixed_axis is not None
        learned_poses = LearnedPoses(
            shifts_frames=fixed_shifts.mean(dim=1).detach(),
            axis_deg_frames=fixed_axis.mean(dim=1).detach(),
            shifts_per_line=fixed_shifts.detach(),
            axis_deg_per_line=fixed_axis.detach(),
            convention=(
                f"{POSE_CONVENTION} Pose source is fixed dataset GT oracle; "
                "these values were not learned and bypass the tanh bounds."
            ),
        )
    elif config.refine_pose:
        assert raw_shifts is not None and raw_axis is not None
        shifts_frames, axis_frames, shifts_per_line, axis_per_line = (
            _expand_learned_pose(
                raw_shifts,
                raw_axis,
                n_scan_lines=geometry.n_scan_lines,
                anchor_reference=config.anchor_reference,
            )
        )
        convention = POSE_CONVENTION
        anchor_frame_index = None
        if config.anchor_reference:
            anchor_frame_index = geometry.frame_shape[0] - 1
            convention = (
                f"{convention} The final frame (index {anchor_frame_index}) is an "
                "appended static theta=0 reference anchored exactly at zero pose. "
                "It has no optimizer node and is excluded from the continuous "
                "scan interpolation timeline; ramp interpolation clamps at the "
                "final ramp-frame node before a separate all-zero reference block "
                "is appended."
            )
        learned_poses = LearnedPoses(
            shifts_frames=shifts_frames.detach(),
            axis_deg_frames=axis_frames.detach(),
            shifts_per_line=shifts_per_line.detach(),
            axis_deg_per_line=axis_per_line.detach(),
            convention=convention,
            anchor_reference=config.anchor_reference,
            anchor_frame_index=anchor_frame_index,
        )
    return ADResult(
        volume=volume.detach(), loss_log=loss_log, poses=learned_poses
    )


def _backward_fidelity_chunks(
    volume: torch.Tensor,
    target: torch.Tensor,
    geometry: PerLineGeometry,
    config: ADConfig,
    valid_lines: torch.Tensor | None = None,
) -> float:
    device = volume.device
    target_lines = geometry.measurement_lines(target)
    valid_flat = _validate_valid_lines(valid_lines, geometry, device).reshape(-1)
    normalizer = int(valid_flat.count_nonzero()) * geometry._line_width
    fidelity_value = 0.0
    for start in range(0, geometry.n_pseudo_angles, config.chunk):
        stop = min(start + config.chunk, geometry.n_pseudo_angles)
        slots, local = geometry._chunk_slots(start, stop, device)
        expanded = ft.forward_project(
            volume,
            geometry.angles_deg[start:stop].to(device=device),
            N_det=geometry.frame_shape[1],
            num_threads=config.num_threads,
        )
        prediction = geometry._gather_chunk(expanded, slots, local)
        chunk_valid = valid_flat[slots]
        if not chunk_valid.any():
            continue
        residual = prediction[chunk_valid] - target_lines[slots[chunk_valid]]
        if config.loss == "mse":
            chunk_loss = residual.square().sum() / normalizer
        else:
            chunk_loss = residual.abs().sum() / normalizer
        fidelity_value += float(chunk_loss.detach())
        chunk_loss.backward()
    return fidelity_value


def _backward_fidelity_pose_chunks(
    volume: torch.Tensor,
    target: torch.Tensor,
    geometry: PerLineGeometry,
    config: ADConfig,
    raw_shifts: torch.Tensor,
    raw_axis: torch.Tensor,
    valid_lines: torch.Tensor | None = None,
    *,
    pose_active: bool,
) -> float:
    device = volume.device
    target_lines = geometry.measurement_lines(target)
    valid_flat = _validate_valid_lines(valid_lines, geometry, device).reshape(-1)
    normalizer = int(valid_flat.count_nonzero()) * geometry._line_width
    fidelity_value = 0.0
    nominal_flat = geometry.angles_deg[geometry.line_to_angle].to(device=device)
    for start in range(0, geometry.n_lines, config.chunk):
        stop = min(start + config.chunk, geometry.n_lines)
        slots = torch.arange(start, stop, device=device)
        valid_slots = slots[valid_flat[slots]]
        if valid_slots.numel() == 0:
            continue
        local = torch.arange(valid_slots.numel(), device=device)
        # Rebuild the tiny bound/interpolation graph for every chunk. This
        # permits immediate backward without retain_graph, preserving the
        # harness's bounded-memory behavior for projector graphs.
        _, _, shifts_per_line, axis_per_line = _expand_learned_pose(
            raw_shifts,
            raw_axis,
            n_scan_lines=geometry.n_scan_lines,
            anchor_reference=config.anchor_reference,
        )
        shifts_flat = shifts_per_line.reshape(geometry.n_lines, 2)
        axis_flat = axis_per_line.reshape(geometry.n_lines)
        if not pose_active:
            shifts_flat = shifts_flat.detach()
            axis_flat = axis_flat.detach()
        expanded = _forward_project_pose_chunk(
            volume,
            nominal_flat[valid_slots],
            shifts_flat[valid_slots],
            axis_flat[valid_slots],
            detector_size=geometry.frame_shape[1],
            num_threads=config.num_threads,
        )
        prediction = geometry._gather_chunk(expanded, valid_slots, local)
        residual = prediction - target_lines[valid_slots]
        if config.loss == "mse":
            chunk_loss = residual.square().sum() / normalizer
        else:
            chunk_loss = residual.abs().sum() / normalizer
        fidelity_value += float(chunk_loss.detach())
        chunk_loss.backward()
    return fidelity_value


def _backward_fidelity_fixed_pose_chunks(
    volume: torch.Tensor,
    target: torch.Tensor,
    geometry: PerLineGeometry,
    config: ADConfig,
    shifts_per_line: torch.Tensor,
    axis_deg_per_line: torch.Tensor,
    valid_lines: torch.Tensor | None = None,
) -> float:
    """Backpropagate volume-only fidelity with exact fixed per-line poses.

    Values are already in the generator convention and deliberately bypass
    the learnable-pose tanh bounds. No pose tensor requires gradients.
    """
    device = volume.device
    target_lines = geometry.measurement_lines(target)
    valid_flat = _validate_valid_lines(valid_lines, geometry, device).reshape(-1)
    normalizer = int(valid_flat.count_nonzero()) * geometry._line_width
    fidelity_value = 0.0
    nominal_flat = geometry.angles_deg[geometry.line_to_angle].to(device=device)
    shifts_flat = shifts_per_line.reshape(geometry.n_lines, 2)
    axis_flat = axis_deg_per_line.reshape(geometry.n_lines)
    for start in range(0, geometry.n_lines, config.chunk):
        stop = min(start + config.chunk, geometry.n_lines)
        slots = torch.arange(start, stop, device=device)
        valid_slots = slots[valid_flat[slots]]
        if valid_slots.numel() == 0:
            continue
        local = torch.arange(valid_slots.numel(), device=device)
        expanded = _forward_project_pose_chunk(
            volume,
            nominal_flat[valid_slots],
            shifts_flat[valid_slots],
            axis_flat[valid_slots],
            detector_size=geometry.frame_shape[1],
            num_threads=config.num_threads,
        )
        prediction = geometry._gather_chunk(expanded, valid_slots, local)
        residual = prediction - target_lines[valid_slots]
        if config.loss == "mse":
            chunk_loss = residual.square().sum() / normalizer
        else:
            chunk_loss = residual.abs().sum() / normalizer
        fidelity_value += float(chunk_loss.detach())
        chunk_loss.backward()
    return fidelity_value


def _forward_project_pose_chunk(
    volume: torch.Tensor,
    nominal_deg: torch.Tensor,
    shifts_dy_dx: torch.Tensor,
    axis_deg: torch.Tensor,
    *,
    detector_size: int,
    num_threads: int | None,
) -> torch.Tensor:
    """Project nominal tilt, then apply the exact generator 2-D pose.

    The scalar projection is returned as FastTomo ``[X,Y]`` and transposed to
    generator image ``[Y,X]``. Generator post-projection ``z3=phi`` rotation is
    applied first, followed by ``(dy,dx)`` translation, before transposing back
    for the existing row/column gather. This operation order, interpolation,
    padding, centre, and sign convention match the generator exactly.
    """
    expanded = ft.forward_project(
        volume,
        nominal_deg,
        N_det=detector_size,
        num_threads=num_threads,
    )
    generator_frames = expanded.permute(0, 2, 1)
    posed_frames = _apply_generator_pose_2d(
        generator_frames, shifts_dy_dx, axis_deg
    )
    return posed_frames.permute(0, 2, 1).contiguous()


def _apply_generator_pose_2d(
    frames_yx: torch.Tensor,
    shifts_dy_dx: torch.Tensor,
    axis_deg: torch.Tensor,
) -> torch.Tensor:
    """Apply exact generator post-projection rotation then translation.

    ``frames_yx`` has shape ``[n_lines,Y,X]``. Positive ``phi`` uses the same
    affine matrix as generator ``z3``. Translation samples
    ``output(y,x)=input(y-dy,x-dx)``, so positive shifts move content toward
    increasing row/column indices. Both stages remain differentiable with
    respect to the image and pose tensors.
    """
    if frames_yx.ndim != 3:
        raise ValueError(f"frames_yx must be [n,Y,X], got {tuple(frames_yx.shape)}")
    n_frames, height, width = frames_yx.shape
    if tuple(shifts_dy_dx.shape) != (n_frames, 2):
        raise ValueError(
            f"shifts_dy_dx must have shape {(n_frames, 2)}, "
            f"got {tuple(shifts_dy_dx.shape)}"
        )
    if tuple(axis_deg.shape) != (n_frames,):
        raise ValueError(
            f"axis_deg must have shape {(n_frames,)}, got {tuple(axis_deg.shape)}"
        )

    radians = torch.deg2rad(axis_deg)
    cosines, sines = radians.cos(), radians.sin()
    zeros = torch.zeros_like(radians)
    ones = torch.ones_like(radians)
    rotations = torch.stack(
        (cosines, -sines, zeros, sines, cosines, zeros), dim=-1
    ).reshape(-1, 2, 3)
    image_batch = frames_yx[:, None]
    rotated = F.grid_sample(
        image_batch,
        F.affine_grid(rotations, image_batch.shape, align_corners=False),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )

    translations = torch.stack(
        (
            ones,
            zeros,
            -2.0 * shifts_dy_dx[:, 1] / width,
            zeros,
            ones,
            -2.0 * shifts_dy_dx[:, 0] / height,
        ),
        dim=-1,
    ).reshape(-1, 2, 3)
    translated = F.grid_sample(
        rotated,
        F.affine_grid(translations, rotated.shape, align_corners=False),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return translated[:, 0]


def _bounded_pose(
    raw_shifts: torch.Tensor, raw_axis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply smooth ±15 px and ±2.5° bounds to unconstrained parameters."""
    shifts = POSE_SHIFT_BOUND_PX * torch.tanh(raw_shifts / POSE_SHIFT_BOUND_PX)
    axis = POSE_AXIS_BOUND_DEG * torch.tanh(raw_axis / POSE_AXIS_BOUND_DEG)
    return shifts, axis


def _expand_learned_pose(
    raw_shifts: torch.Tensor,
    raw_axis: torch.Tensor,
    *,
    n_scan_lines: int,
    anchor_reference: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bound/interpolate ramp nodes and optionally append a fixed reference.

    With ``anchor_reference=True``, the inputs contain only ramp-frame nodes.
    They are interpolated on the ramp timeline with the normal endpoint clamp;
    the temporally last static reference is excluded from that timeline and
    receives separate exactly-zero frame and per-line poses. Consequently no
    reference parameter exists in the optimizer and the final ramp frame is
    not pulled toward the reference pose.
    """
    shifts_frames, axis_frames = _bounded_pose(raw_shifts, raw_axis)
    shifts_per_line = interpolate_pose_nodes(shifts_frames, n_scan_lines)
    axis_per_line = interpolate_pose_nodes(axis_frames, n_scan_lines)
    if not anchor_reference:
        return shifts_frames, axis_frames, shifts_per_line, axis_per_line

    zero_shift_frame = shifts_frames.new_zeros((1, 2))
    zero_axis_frame = axis_frames.new_zeros(1)
    zero_shift_lines = shifts_frames.new_zeros((1, n_scan_lines, 2))
    zero_axis_lines = axis_frames.new_zeros((1, n_scan_lines))
    return (
        torch.cat((shifts_frames, zero_shift_frame), dim=0),
        torch.cat((axis_frames, zero_axis_frame), dim=0),
        torch.cat((shifts_per_line, zero_shift_lines), dim=0),
        torch.cat((axis_per_line, zero_axis_lines), dim=0),
    )


def _raw_from_bounded_pose(
    shifts: torch.Tensor, axis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert smooth pose bounds for a requested physical initialization.

    Initial values outside a learnable bound are clipped just inside it;
    fixed oracle poses do not call this helper and remain unbounded.
    """
    eps = 1e-6
    shifts_unit = (shifts / POSE_SHIFT_BOUND_PX).clamp(-1.0 + eps, 1.0 - eps)
    axis_unit = (axis / POSE_AXIS_BOUND_DEG).clamp(-1.0 + eps, 1.0 - eps)
    return (
        POSE_SHIFT_BOUND_PX * torch.atanh(shifts_unit),
        POSE_AXIS_BOUND_DEG * torch.atanh(axis_unit),
    )


def _validate_pose_nodes(
    shifts_frames: np.ndarray | torch.Tensor | None,
    axis_deg_frames: np.ndarray | torch.Tensor | None,
    *,
    n_frames: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if shifts_frames is None or axis_deg_frames is None:
        raise ValueError("both initial frame pose arrays must be supplied together")
    shifts = torch.as_tensor(shifts_frames, dtype=torch.float32, device=device)
    axis = torch.as_tensor(axis_deg_frames, dtype=torch.float32, device=device)
    if tuple(shifts.shape) != (n_frames, 2):
        raise ValueError(
            f"initial shifts must have shape {(n_frames, 2)}, got {tuple(shifts.shape)}"
        )
    if tuple(axis.shape) != (n_frames,):
        raise ValueError(
            f"initial axis must have shape {(n_frames,)}, got {tuple(axis.shape)}"
        )
    if not torch.isfinite(shifts).all() or not torch.isfinite(axis).all():
        raise ValueError("initial pose arrays contain a non-finite value")
    return shifts, axis


def _validate_pose_lines(
    shifts_per_line: np.ndarray | torch.Tensor | None,
    axis_deg_per_line: np.ndarray | torch.Tensor | None,
    geometry: PerLineGeometry,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if shifts_per_line is None or axis_deg_per_line is None:
        raise ValueError(
            "shifts_per_line and axis_deg_per_line must be supplied together"
        )
    n_frames = geometry.frame_shape[0]
    shifts = torch.as_tensor(
        shifts_per_line, dtype=torch.float32, device=device
    )
    axis = torch.as_tensor(
        axis_deg_per_line, dtype=torch.float32, device=device
    )
    expected = (n_frames, geometry.n_scan_lines)
    if tuple(shifts.shape) != (*expected, 2):
        raise ValueError(
            f"shifts_per_line must have shape {(*expected, 2)}, "
            f"got {tuple(shifts.shape)}"
        )
    if tuple(axis.shape) != expected:
        raise ValueError(
            f"axis_deg_per_line must have shape {expected}, got {tuple(axis.shape)}"
        )
    if not torch.isfinite(shifts).all() or not torch.isfinite(axis).all():
        raise ValueError("pose arrays contain a non-finite value")
    return shifts, axis


def _validate_valid_lines(
    valid_lines: np.ndarray | torch.Tensor | None,
    geometry: PerLineGeometry,
    device: torch.device,
) -> torch.Tensor:
    """Return a validated boolean mask in frame/scan-line order."""
    expected = (geometry.frame_shape[0], geometry.n_scan_lines)
    if valid_lines is None:
        return torch.ones(expected, dtype=torch.bool, device=device)
    mask = torch.as_tensor(valid_lines, device=device)
    if mask.dtype != torch.bool:
        raise ValueError(f"valid_lines must have boolean dtype, got {mask.dtype}")
    if tuple(mask.shape) != expected:
        raise ValueError(
            f"valid_lines must have shape {expected}, got {tuple(mask.shape)}"
        )
    if not mask.any():
        raise ValueError("valid_lines must retain at least one line")
    return mask.contiguous()


def _validate_chunk(chunk_size: int) -> None:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size/chunk must be a positive integer")


def _validate_volume(volume: torch.Tensor, geometry: PerLineGeometry) -> None:
    if volume.ndim != 3:
        raise ValueError(f"volume must have shape [image_X, z, image_Y], got {volume.shape}")
    _, image_y, image_x = geometry.frame_shape
    if int(volume.shape[0]) != image_x or int(volume.shape[2]) != image_y:
        raise ValueError(
            "volume must have FastTomo shape [image_X, z, image_Y]; "
            f"expected first/last dimensions {(image_x, image_y)}, got {tuple(volume.shape)}"
        )
