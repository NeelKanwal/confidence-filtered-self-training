"""Author: Neel Kanwal <neel.kanwal0@gmail.com>."""

from __future__ import annotations

from .image_geometry import to_sitk, from_sitk, ants_to_sitk, sitk_to_ants
from .bias_correction import shared_n4_in_memory
from .registration import _ants_metric_names, register_ants
from .resampling import resample_stack
from .normalization import global_normalize_4d

import dataclasses
import hashlib
import json
import os
import multiprocessing
import re
import shutil
import socket
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


os.environ.setdefault("ITK_DEBUG", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")

import nibabel as nib
import numpy as np
from nibabel.processing import resample_to_output, resample_from_to
import SimpleITK as sitk
from scipy.ndimage import (
    binary_closing as ndi_binary_closing,
    binary_dilation,
    binary_fill_holes,
    distance_transform_edt,
    gaussian_filter,
    gaussian_filter1d,
    label,
)
from skimage.filters import frangi
from skimage.morphology import ball, binary_closing, binary_opening, skeletonize

warnings.filterwarnings("ignore", category=FutureWarning)


DATASET_ROOT = Path(
    "work/selected"
)
OUTPUT_ROOT = Path(
    "work/preprocessed"
)


WORK_ROOT = OUTPUT_ROOT.parent / f".{OUTPUT_ROOT.name}_work"


STATE_ROOT = OUTPUT_ROOT.parent / f".{OUTPUT_ROOT.name}_state"


VESSEL_QC_ROOT = OUTPUT_ROOT.parent / f"{OUTPUT_ROOT.name}_QC"


PATIENT_WORKERS = 1
PHASE_WORKERS = 4
REGISTRATION_WORKERS = 3
ITK_THREADS = 0
STALE_LOCK_HOURS = 48.0


OVERWRITE_EXISTING = False
KEEP_WORK_ON_FAILURE = False
KEEP_WORK_ON_SUCCESS = False
STRICT_OUTPUT_MIRROR = True


EXPECTED_PHASES = (0, 1, 2, 3)
REGISTRATION_BASELINE_PHASE = 3
STRICT_EXPECTED_PHASES = True


DO_BIAS_CORRECTION = True
DO_BREAST_MASK = True
DO_REGISTRATION = True
DO_RESAMPLE = True
DO_NORMALIZE = True
DO_VESSEL_MASK = False
APPLY_VESSEL_SUPPRESSION = False


SAVE_INTERMEDIATE = False
SAVE_4D_QC = False
DO_PERCENT_ENHANCEMENT = False
DO_SUBTRACTIONS = False
DO_DISPLACEMENT = False


ANTS_TRANSFORM = "SyN"
ANTS_METRIC = "MI"
ANTS_RANDOM_SEED = 0
ANTS_HISTOGRAM_MATCHING = False
N4_ITERATIONS = (50, 40, 30, 20)
N4_SHRINK_FACTOR = 2
ALLOW_MISSING_ORIENTATION = False
ASSUME_MM_IF_UNITS_UNKNOWN = False
USE_BREAST_MASK_FOR_REGISTRATION = True
FAST_MODE = True


NORMALIZE_TYPE = "zscore"
NEW_SPACING = (1.0, 1.0, 1.0)
RESAMPLE_INTERPOLATOR = "bspline"


BREAST_ROI_SETTINGS = {'breast_mask_foreground_fraction': 0.05,
    'breast_mask_gaussian_sigma_mm': 0.8,
    'breast_mask_central_strip_mm': 30.0,
    'breast_mask_occupancy_window_mm': 10.0,
    'breast_mask_chest_wall_margin_mm': 40.0,
    'breast_mask_central_posterior_margin_mm': 10.0,
    'breast_mask_central_half_width_mm': 45.0,
    'breast_mask_max_lateral_curve_mm': 35.0,
    'breast_mask_lateral_extension_mm': 15.0,
    'breast_mask_min_projection_mm': 10.0,
    'breast_mask_si_margin_mm': 8.0,
    'breast_mask_min_component_ml': 5.0,
    'breast_mask_close_mm': 2.0,
    'breast_mask_dilate_mm': 1.0}


VESSEL_POST_PHASE = 1


VESSEL_BREAST_FOREGROUND_FRACTION = 0.05
VESSEL_BREAST_CENTRAL_STRIP_MM = 30.0
VESSEL_BREAST_OCCUPANCY_WINDOW_MM = 10.0
VESSEL_BREAST_CHEST_MARGIN_MM = 12.0
VESSEL_BOUNDARY_MARGIN_MM = 5.0


VESSEL_FILTER_SIGMAS_MM = (0.8, 1.2, 1.8, 2.5)
VESSEL_FILTER_ALPHA = 0.5
VESSEL_FILTER_BETA = 0.5
VESSEL_RESPONSE_NORM_LOW_PERCENTILE = 30.0
VESSEL_RESPONSE_NORM_HIGH_PERCENTILE = 99.7


VESSEL_ENHANCEMENT_LOW_PERCENTILE = 20.0
VESSEL_ENHANCEMENT_HIGH_PERCENTILE = 99.5


VESSEL_CONSENSUS_HIGH_PERCENTILE = 98.0
VESSEL_CONSENSUS_LOW_PERCENTILE = 92.0
VESSEL_TEMPORAL_FRACTION_START = 0.25
VESSEL_TEMPORAL_FRACTION_WIDTH = 0.50
VESSEL_TEMPORAL_MIN_WEIGHT = 0.35


VESSEL_MIN_COMPONENT_MM3 = 8.0
VESSEL_LARGE_COMPONENT_MM3 = 60.0
VESSEL_LARGE_COMPONENT_MIN_EXTENT_MM = 10.0


VESSEL_LUMEN_DILATION_MM = 1.0
VESSEL_LUMEN_MIN_SCALED_ENHANCEMENT = 0.15


VESSEL_HESSIAN_LINE_BETA = 0.35
VESSEL_HESSIAN_STRENGTH_PERCENTILE = 90.0


VESSEL_MIN_PCA_ELONGATION = 1.25
VESSEL_MIN_SKELETON_LENGTH_MM = 8.0
VESSEL_MIN_LENGTH_DIAMETER_RATIO = 2.0
VESSEL_LARGE_TREE_SKELETON_MM = 15.0


VESSEL_REMOVE_ALL_PHASES = True
VESSEL_INPAINT_SIGMA_MM = 2.5
VESSEL_REMOVAL_STRENGTH = 1.0


SAVE_VESSEL_QC = True
SAVE_VESSEL_QC_NIFTI = True
SAVE_VESSEL_QC_GRID = True
VESSEL_QC_GRID_MAX_SLICES = 6

LOCK_DIR_NAME = "locks"
SCRATCH_DIR_NAME = "scratch"
PIPELINE_VERSION = "2026-10-29-training-deployment-2.5.12-parity"


@dataclass
class PatientResult:
    patient_id: str
    status: str
    message: str
    out_dir: str = ""


@dataclass
class PipelineResult:
    out_dir: str
    primary_outputs: List[str]
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclass
class Config:
    out_dir: str = "preprocess"
    save_intermediate: bool = False


    use_dce_only: bool = False
    dce_keys: Tuple[str, ...] = (
        "dce", "dyn", "ax_dyn", "pre", "1st", "2nd", "3rd", "4th",
        "post", "vibrant", "multiphase", "ph1ax", "tra", "dynamic",
        "vibrant_multiphase", "ph1", "ph2", "ph3", "ph4",
    )
    t1_exclude: Tuple[str, ...] = ("t1", "t1w", "t1-weighted")


    max_workers: int = 4
    registration_workers: int = 3
    itk_threads: int = 1
    serial_itk_threads: int = 1
    n4_iterations: Tuple[int, ...] = (50, 40, 30, 20)
    n4_shrink_factor: int = 2
    allow_missing_orientation: bool = False
    assume_mm_if_units_unknown: bool = False
    breast_roi: Dict[str, float] = dataclasses.field(default_factory=lambda: dict(BREAST_ROI_SETTINGS))
    patient_workers: int = 1
    stale_lock_hours: float = 48.0


    do_bias_correction: bool = True
    do_breast_mask: bool = True
    do_registration: bool = True
    do_resample: bool = True
    do_normalize: bool = True
    do_make_4d: bool = False
    do_percent_enhancement: bool = False
    do_subtractions: bool = False
    do_vessel_mask: bool = False
    apply_vessel_suppression: bool = False
    do_displacement: bool = False


    ants_transform: str = "SyN"
    ants_metric: str = "MI"
    ants_random_seed: int = 0
    ants_histogram_matching: bool = False
    registration_baseline_phase: int = 3
    use_breast_mask: bool = True
    fast_mode: bool = True


    normalize_type: str = "zscore"
    new_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    resample_interpolator: str = "bspline"


    vessel_post_phase: int = 1
    vessel_breast_foreground_fraction: float = 0.05
    vessel_breast_central_strip_mm: float = 30.0
    vessel_breast_occupancy_window_mm: float = 10.0
    vessel_breast_chest_margin_mm: float = 12.0
    vessel_boundary_margin_mm: float = 5.0
    vessel_filter_sigmas_mm: Tuple[float, ...] = (0.8, 1.2, 1.8, 2.5)
    vessel_filter_alpha: float = 0.5
    vessel_filter_beta: float = 0.5
    vessel_response_norm_low_percentile: float = 30.0
    vessel_response_norm_high_percentile: float = 99.7
    vessel_enhancement_low_percentile: float = 20.0
    vessel_enhancement_high_percentile: float = 99.5
    vessel_consensus_high_percentile: float = 98.0
    vessel_consensus_low_percentile: float = 92.0
    vessel_temporal_fraction_start: float = 0.25
    vessel_temporal_fraction_width: float = 0.50
    vessel_temporal_min_weight: float = 0.35
    vessel_min_component_mm3: float = 8.0
    vessel_large_component_mm3: float = 60.0
    vessel_large_component_min_extent_mm: float = 10.0
    vessel_lumen_dilation_mm: float = 1.0
    vessel_lumen_min_scaled_enhancement: float = 0.15
    vessel_hessian_line_beta: float = 0.35
    vessel_hessian_strength_percentile: float = 90.0
    vessel_min_pca_elongation: float = 1.25
    vessel_min_skeleton_length_mm: float = 8.0
    vessel_min_length_diameter_ratio: float = 2.0
    vessel_large_tree_skeleton_mm: float = 15.0
    vessel_remove_all_phases: bool = True
    vessel_inpaint_sigma_mm: float = 2.5
    vessel_removal_strength: float = 1.0
    save_vessel_qc: bool = False
    save_vessel_qc_nifti: bool = False
    save_vessel_qc_grid: bool = False
    vessel_qc_grid_max_slices: int = 6
    vessel_qc_root: str = ""

    def __post_init__(self) -> None:
        if self.registration_baseline_phase not in (0, 1, 2, 3):
            raise ValueError("registration_baseline_phase must be 0, 1, 2, or 3")
        if self.registration_workers < 1 or self.itk_threads < 1 or self.serial_itk_threads < 1:
            raise ValueError("Worker and thread counts must be positive")
        if self.n4_shrink_factor < 1 or not self.n4_iterations or any(int(v) < 1 for v in self.n4_iterations):
            raise ValueError("N4 shrink factor and iteration counts must be positive")
        if len(self.new_spacing) != 3 or any(not np.isfinite(v) or v <= 0 for v in self.new_spacing):
            raise ValueError("new_spacing must contain three positive finite values")
        if self.normalize_type.lower() not in {"zscore", "minmax"}:
            raise ValueError("normalize_type must be 'zscore' or 'minmax'.")
        if self.resample_interpolator.lower() not in {"linear", "bspline"}:
            raise ValueError("resample_interpolator must be 'linear' or 'bspline'.")
        if self.max_workers < 1 or self.patient_workers < 1:
            raise ValueError("max_workers and patient_workers must be >= 1.")
        if not (0.0 <= self.vessel_removal_strength <= 1.0):
            raise ValueError("vessel_removal_strength must be in [0, 1].")
        if self.vessel_inpaint_sigma_mm <= 0:
            raise ValueError("vessel_inpaint_sigma_mm must be > 0.")
        if self.vessel_min_skeleton_length_mm < 0:
            raise ValueError("vessel_min_skeleton_length_mm must be >= 0.")
        if self.vessel_response_norm_low_percentile >= self.vessel_response_norm_high_percentile:
            raise ValueError("vessel response normalization percentiles are invalid.")
        if self.vessel_enhancement_low_percentile >= self.vessel_enhancement_high_percentile:
            raise ValueError("vessel enhancement percentiles are invalid.")
        if self.vessel_consensus_low_percentile >= self.vessel_consensus_high_percentile:
            raise ValueError("vessel consensus percentiles are invalid.")
        if not (0.0 <= self.vessel_temporal_min_weight <= 1.0):
            raise ValueError("vessel_temporal_min_weight must be in [0, 1].")
        if self.vessel_qc_grid_max_slices < 1:
            raise ValueError("vessel_qc_grid_max_slices must be >= 1.")
        if self.apply_vessel_suppression:
            self.do_vessel_mask = True
        if self.do_vessel_mask:
            self.do_breast_mask = True


@dataclass
class ProcessingState:
    current_paths: List[Path]
    current_images: Dict[int, sitk.Image]
    reference_nii: Optional[nib.Nifti1Image] = None
    breast_mask: Optional[sitk.Image] = None
    final_breast_mask: Optional[sitk.Image] = None
    vessel_mask: Optional[np.ndarray] = None

    def get_image(self, idx: int) -> sitk.Image:
        if idx not in self.current_images:
            self.current_images[idx] = load_sitk(self.current_paths[idx])
        return self.current_images[idx]

    def clear_cache(self) -> None:
        self.current_images.clear()

    def update_paths(self, new_paths: Sequence[Path], clear_cache: bool = True) -> None:
        self.current_paths = list(new_paths)
        if clear_cache:
            self.clear_cache()


def log(msg: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [MRI-PRE] {msg}", flush=True)


@contextmanager
def timer(step_name: str):
    start = time.time()
    log(f"[START] {step_name}")
    try:
        yield
    finally:
        elapsed = time.time() - start
        log(f"[FINISHED] {step_name} in {elapsed:.2f}s ({elapsed / 60:.2f} min)")


def ensure_float32(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32, copy=False) if arr.dtype != np.float32 else arr


def save_nifti_like(data: np.ndarray, ref_nii: nib.Nifti1Image, out_path: Path) -> None:
    try:
        data = ensure_float32(np.asarray(data))
        header = ref_nii.header.copy()
        img = nib.Nifti1Image(data, affine=ref_nii.affine, header=header)
        img.set_data_dtype(np.float32)
        nib.save(img, str(out_path))
        log(f"      [IO] wrote {out_path.name} shape={data.shape}")
    except Exception as exc:
        raise RuntimeError(f"Failed to save {out_path}: {exc}") from exc


def save_mask_nifti(mask: np.ndarray, ref_nii: nib.Nifti1Image, out_path: Path) -> None:
    mask = np.asarray(mask, dtype=np.uint8)
    header = ref_nii.header.copy()
    img = nib.Nifti1Image(mask, affine=ref_nii.affine, header=header)
    img.set_data_dtype(np.uint8)
    nib.save(img, str(out_path))


def load_nifti(path: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
    try:
        nii = nib.load(str(path))
        arr = nii.get_fdata(dtype=np.float32)
        return ensure_float32(arr), nii
    except Exception as exc:
        raise RuntimeError(f"Failed to load {path}: {exc}") from exc


def load_sitk(path: Path) -> sitk.Image:
    data, nii = load_nifti(Path(path))
    return to_sitk(data, nii.affine)


def write_sitk(img: sitk.Image, out_path: Union[str, Path]) -> None:
    if not isinstance(img, sitk.Image):
        raise TypeError(f"Expected SimpleITK.Image, got {type(img)}")
    try:
        sitk.WriteImage(img, str(out_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to save {out_path}: {exc}") from exc


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def set_threads(threads: int = 1) -> None:
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(int(threads))
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(int(threads))


def validate_volume_shapes(vols: Sequence[np.ndarray], step_name: str) -> None:
    if not vols:
        return
    shapes = [v.shape[:3] for v in vols]
    if len(set(shapes)) > 1:
        raise ValueError(f"Incompatible shapes in {step_name}: {shapes}")


def looks_like_t1(name: str, exclude: Tuple[str, ...]) -> bool:
    low = name.lower()
    return any(k.lower() in low for k in exclude)


def looks_like_dce(name: str, dce_keys: Tuple[str, ...]) -> bool:
    low = name.lower()
    return any(k.lower() in low for k in dce_keys)


def order_key(name: str) -> int:
    low = name.lower()
    if "pre" in low:
        return 0
    for i, tag in enumerate(("1st", "2nd", "3rd", "4th", "post1", "post2", "post3", "post4"), start=1):
        if tag in low:
            return i
    match = re.search(r"_(\d+)\.nii(?:\.gz)?$", low)
    if match:
        return int(match.group(1))
    numbers = re.findall(r"\d+", low)
    return int(numbers[-1]) if numbers else 999


def phase_index_from_name(name: str) -> Optional[int]:

    match = re.search(r"_(\d{4})\.nii(?:\.gz)?$", name.lower())
    return int(match.group(1)) if match else None


def baseline_index_from_original_files(files: Sequence[Path], baseline_phase: int) -> int:
    matches = [i for i, path in enumerate(files) if phase_index_from_name(path.name) == baseline_phase]
    if len(matches) != 1:
        suffix = f"_{baseline_phase:04d}.nii(.gz)"
        raise RuntimeError(
            f"Expected exactly one registration baseline matching {suffix}; "
            f"found {len(matches)} among {[p.name for p in files]}."
        )
    return matches[0]


def validate_expected_phases(files: Sequence[Path], expected_phases: Sequence[int]) -> None:
    actual = [phase_index_from_name(path.name) for path in files]
    if any(v is None for v in actual):
        bad = [path.name for path, phase in zip(files, actual) if phase is None]
        raise RuntimeError(f"Could not parse DCE phase suffix from: {bad}")
    actual_int = [int(v) for v in actual if v is not None]
    expected = list(expected_phases)
    if actual_int != expected:
        raise RuntimeError(
            f"Expected phases {expected}, but found {actual_int} in {[p.name for p in files]}."
        )


def get_dce_files_from_list(files: Sequence[Path], cfg: Config) -> List[Path]:
    dce_files: List[Path] = []
    for path in files:
        if looks_like_t1(path.name, cfg.t1_exclude):
            continue
        if not cfg.use_dce_only or looks_like_dce(path.name, cfg.dce_keys):
            dce_files.append(path)

    result = sorted(set(dce_files), key=lambda p: order_key(p.name))
    log(f"Isolated {len(result)} DCE volumes.")
    for idx, path in enumerate(result):
        log(f"    phase[{idx}] -> {path.name}")
    return result


BREAST_MASK_ALGORITHM_VERSION = "sternum-curved-chestwall-v8"

@dataclass
class BreastROIConfig:

    breast_mask_foreground_fraction: float = 0.05
    breast_mask_gaussian_sigma_mm: float = 0.8
    breast_mask_central_strip_mm: float = 30.0
    breast_mask_occupancy_window_mm: float = 10.0
    breast_mask_chest_wall_margin_mm: float = 40.0
    breast_mask_central_posterior_margin_mm: float = 10.0
    breast_mask_central_half_width_mm: float = 45.0
    breast_mask_max_lateral_curve_mm: float = 35.0
    breast_mask_lateral_extension_mm: float = 15.0
    breast_mask_min_projection_mm: float = 10.0
    breast_mask_si_margin_mm: float = 8.0
    breast_mask_min_component_ml: float = 5.0
    breast_mask_close_mm: float = 2.0
    breast_mask_dilate_mm: float = 1.0
    breast_mask_qc_slices: int = 1


def _mask_ellipsoid(radius_mm: float, spacing_xyz: Sequence[float]) -> np.ndarray:
    if radius_mm <= 0:
        return np.ones((1, 1, 1), dtype=bool)
    spacing = np.asarray(spacing_xyz[:3], dtype=np.float64)
    radii = np.maximum(1, np.ceil(float(radius_mm) / spacing).astype(int))
    rx, ry, rz = [int(v) for v in radii]
    xx, yy, zz = np.ogrid[-rx:rx + 1, -ry:ry + 1, -rz:rz + 1]
    d2 = (xx * spacing[0]) ** 2 + (yy * spacing[1]) ** 2 + (zz * spacing[2]) ** 2
    return d2 <= float(radius_mm) ** 2 + 1e-8


def _mask_keep_components(mask: np.ndarray, spacing_xyz: Sequence[float], min_component_ml: float) -> np.ndarray:
    labels, n_labels = label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if n_labels == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    voxel_mm3 = float(np.prod(np.asarray(spacing_xyz[:3], dtype=np.float64)))
    min_voxels = max(1, int(round(float(min_component_ml) * 1000.0 / max(voxel_mm3, 1e-8))))
    ids = np.flatnonzero(sizes >= min_voxels)
    if ids.size == 0:
        ids = np.asarray([int(np.argmax(sizes))])
    ids = ids[np.argsort(sizes[ids])[::-1]][:2]
    return np.isin(labels, ids)


def _mask_fill_holes_xyz(mask: np.ndarray) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    for z in range(out.shape[2]):
        if out[:, :, z].any():
            out[:, :, z] = binary_fill_holes(out[:, :, z])
    return binary_fill_holes(out)


def _postprocess_breast_mask_xyz(
    mask: np.ndarray, spacing_xyz: Sequence[float], cfg: BreastROIConfig,
    allowed_mask: Optional[np.ndarray] = None, dilate: bool = True,
) -> np.ndarray:

    out = _mask_keep_components(mask, spacing_xyz, cfg.breast_mask_min_component_ml)
    out = _mask_fill_holes_xyz(out)
    if cfg.breast_mask_close_mm > 0:
        structure = _mask_ellipsoid(cfg.breast_mask_close_mm, spacing_xyz)
        pad = tuple((int(s // 2), int(s // 2)) for s in structure.shape)
        padded = ndi_binary_closing(np.pad(out, pad), structure=structure)
        out = padded[tuple(slice(p, -p if p else None) for p, _ in pad)]
    out = _mask_fill_holes_xyz(out)
    if dilate and cfg.breast_mask_dilate_mm > 0:
        out = binary_dilation(out, structure=_mask_ellipsoid(cfg.breast_mask_dilate_mm, spacing_xyz))
    if allowed_mask is not None:
        out &= np.asarray(allowed_mask, dtype=bool)
    out = _mask_keep_components(out, spacing_xyz, cfg.breast_mask_min_component_ml)
    return out.astype(np.uint8)


def _create_roi(reference: nib.Nifti1Image, cfg: BreastROIConfig) -> Tuple[np.ndarray, nib.Nifti1Image, Dict[str, float]]:


    original = reference
    source_name = "reference MRI"
    if len(original.shape) != 3:
        raise ValueError(f"Expected a 3D MRI, got {original.shape}: {source_name}")
    canonical = nib.as_closest_canonical(original)
    image = canonical.get_fdata(dtype=np.float32)
    spacing = np.asarray(canonical.header.get_zooms()[:3], dtype=np.float64)
    if np.any(~np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError(f"Invalid voxel spacing: {spacing}")
    if not 0 < cfg.breast_mask_foreground_fraction < 1:
        raise ValueError("breast_mask_foreground_fraction must be between zero and one")
    finite_values = image[np.isfinite(image)]
    if finite_values.size < 100:
        raise RuntimeError(f"Too few finite voxels in {source_name}")
    lo, hi = np.percentile(finite_values, [0.5, 99.5])
    if not np.isfinite(lo + hi) or hi <= lo:
        raise RuntimeError(f"Invalid intensity range in {source_name}: {lo}, {hi}")

    scaled = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    nx, ny, nz = image.shape
    body = np.zeros(image.shape, dtype=bool)
    fronts = np.full((nx, nz), np.nan, dtype=np.float32)
    sternum = np.full(nz, np.nan, dtype=np.float32)
    centers = np.full(nz, np.nan, dtype=np.float32)
    half_widths = np.full(nz, np.nan, dtype=np.float32)
    ap_radii = np.full(nz, np.nan, dtype=np.float32)
    projection = np.zeros(nz, dtype=np.float32)
    central_half = max(1, int(round(0.5 * cfg.breast_mask_central_strip_mm / spacing[0])))
    window = max(1, int(round(cfg.breast_mask_occupancy_window_mm / spacing[1])))
    xx = np.arange(nx)
    for z in range(nz):
        sl = scaled[:, :, z]
        finite = np.isfinite(sl)
        positive = sl[finite & (sl > 0)]
        if positive.size < 100:
            continue
        high = float(np.percentile(positive, 99.0))
        if high <= 0:
            continue
        smooth = gaussian_filter(np.where(finite, sl, 0.0), cfg.breast_mask_gaussian_sigma_mm / spacing[:2])
        outline = _largest_2d_component(finite & (smooth > cfg.breast_mask_foreground_fraction * high))
        if int(outline.sum()) < 100:
            continue
        outline = binary_fill_holes(outline)
        coords = np.argwhere(outline)
        xmin, ymin = coords.min(axis=0)
        xmax, ymax = coords.max(axis=0)
        center = int(round((xmin + xmax) / 2.0))
        x0, x1 = max(0, center - central_half), min(nx, center + central_half + 1)
        occupancy = gaussian_filter1d(outline[x0:x1, :].mean(axis=0).astype(float), max(0.5, 1.5 / spacing[1]), mode="nearest")
        chest = None
        for y in range(int(ymax), int(ymin) + window - 1, -1):
            if occupancy[y] >= 0.5 and occupancy[y - window:y + 1].mean() >= 0.6:
                chest = float(y)
                break
        if chest is None:
            continue

        row = max(0, int(round(chest - 40.0 / spacing[1])))
        spans, _ = label(outline[:, row])
        sid = int(spans[center])
        row_x = np.flatnonzero(spans == sid) if sid else np.flatnonzero(outline[:, row])
        half_width = max(40.0, 0.5 * (float(row_x[-1] - row_x[0]) + 1.0) * spacing[0]) if row_x.size else max(40.0, 0.5 * (xmax - xmin) * spacing[0])
        front = ny - 1 - np.argmax(outline[:, ::-1], axis=1)
        front = np.where(outline.any(axis=1), front, np.nan)
        lateral_mm = np.abs(xx - center) * spacing[0]
        breast_columns = np.isfinite(front) & (lateral_mm >= 20.0) & (lateral_mm <= 0.85 * half_width)
        if breast_columns.any():
            projection[z] = max(0.0, float(np.percentile(front[breast_columns], 90.0) - chest) * spacing[1])
        body[:, :, z] = outline
        fronts[:, z] = front
        sternum[z], centers[z], half_widths[z] = chest, center, half_width
        ap_radii[z] = np.clip(0.5 * (chest - ymin) * spacing[1], 40.0, 100.0)
    valid = np.flatnonzero(np.isfinite(sternum))
    if valid.size < 2:
        raise RuntimeError(f"Cannot locate a stable anterior chest wall in {source_name}")

    support = np.isfinite(sternum) & (projection >= cfg.breast_mask_min_projection_mm)
    runs, nruns = label(support)
    if nruns == 0:
        raise RuntimeError(f"No breast projection found in {source_name}; inspect this case manually")
    sizes = np.bincount(runs); sizes[0] = 0
    best = int(np.argmax(sizes))
    active = np.flatnonzero(runs == best)
    z_margin = max(0, int(np.ceil(cfg.breast_mask_si_margin_mm / spacing[2])))
    z0, z1 = max(0, int(active[0]) - z_margin), min(nz, int(active[-1]) + z_margin + 1)
    zcoords = np.arange(nz)
    sigma_z = max(0.5, 4.0 / spacing[2])
    sternum = gaussian_filter1d(np.interp(zcoords, valid, sternum[valid]), sigma_z, mode="nearest")
    centers = gaussian_filter1d(np.interp(zcoords, valid, centers[valid]), sigma_z, mode="nearest")
    half_widths = gaussian_filter1d(np.interp(zcoords, valid, half_widths[valid]), sigma_z, mode="nearest")
    ap_radii = gaussian_filter1d(np.interp(zcoords, valid, ap_radii[valid]), sigma_z, mode="nearest")
    allowed = np.zeros_like(body)
    anterior_y = np.arange(ny)[None, :]
    x_mm = (xx[:, None] - centers[None, :]) * spacing[0]
    blend = np.clip(np.abs(x_mm) / max(cfg.breast_mask_central_half_width_mm, spacing[0]), 0.0, 1.0)
    blend = blend * blend * (3.0 - 2.0 * blend)
    margins = cfg.breast_mask_central_posterior_margin_mm + blend * (cfg.breast_mask_chest_wall_margin_mm - cfg.breast_mask_central_posterior_margin_mm)
    lateral_ratio = np.clip(np.abs(x_mm) / np.maximum(half_widths[None, :], spacing[0]), 0.0, 0.98)
    curve = np.minimum(cfg.breast_mask_max_lateral_curve_mm, ap_radii[None, :] * (1.0 - np.sqrt(1.0 - lateral_ratio ** 2))) * blend
    cuts = sternum[None, :] - (margins + curve) / spacing[1]
    for z in range(z0, z1):


        torso_span = (anterior_y >= sternum[z]) | (np.abs(x_mm[:, z, None]) <= half_widths[z] + cfg.breast_mask_lateral_extension_mm)
        allowed[:, :, z] = body[:, :, z] & (anterior_y >= cuts[:, z, None]) & torso_span
    breast = _postprocess_breast_mask_xyz(allowed, spacing, cfg, allowed_mask=allowed)
    to_native = nib.orientations.ornt_transform(nib.orientations.io_orientation(canonical.affine), nib.orientations.io_orientation(original.affine))
    native_mask = nib.orientations.apply_orientation(breast, to_native).copy().astype(np.uint8)
    if native_mask.shape != original.shape or not native_mask.any():
        raise RuntimeError(f"Invalid or empty breast mask for {source_name}")
    stats = {
        "si_start_index_in_canonical_roi_grid": int(z0),
        "si_stop_index_exclusive_in_canonical_roi_grid": int(z1),
        "mask_voxels": float(native_mask.sum()),
        "mask_volume_ml": float(native_mask.sum() * abs(np.linalg.det(original.affine[:3,:3])) / 1000.0),
        "remaining_internal_hole_voxels": float((_mask_fill_holes_xyz(breast) & (breast == 0)).sum()),
        "valid_mask_slices": float(np.count_nonzero(breast.sum(axis=(0, 1)))),
        "chest_wall_margin_mm": float(cfg.breast_mask_chest_wall_margin_mm),
        "central_posterior_margin_mm": float(cfg.breast_mask_central_posterior_margin_mm),
        "maximum_lateral_reach_mm": float(cfg.breast_mask_chest_wall_margin_mm + cfg.breast_mask_max_lateral_curve_mm),
        "lateral_extension_mm": float(cfg.breast_mask_lateral_extension_mm),
    }
    return native_mask, original, stats


def _largest_2d_component(mask: np.ndarray) -> np.ndarray:
    labels, n_labels = label(mask)
    if n_labels == 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(np.argmax(counts))


def build_breast_mask(data, affine, opts):

    reference = nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine)
    canonical = nib.as_closest_canonical(reference)
    diagonal = np.diag(np.diag(canonical.affine[:3, :3]))
    oblique = not np.allclose(canonical.affine[:3, :3], diagonal, atol=1e-5)
    working = resample_to_output(reference, nib.affines.voxel_sizes(canonical.affine), order=1) if oblique else reference
    cfg = BreastROIConfig(**{field.name: opts[field.name] for field in fields(BreastROIConfig) if field.name != 'breast_mask_qc_slices'})
    mask, _, stats = _create_roi(working, cfg)
    if oblique:
        image = nib.Nifti1Image(mask, working.affine)
        mask = np.asarray(resample_from_to(image, reference, order=0).dataobj, dtype=np.uint8)
    stats.update(algorithm=BREAST_MASK_ALGORITHM_VERSION,
                 source_signal='original unnormalized reference MRI',
                 oblique_roi_reformat=oblique, anatomical_segmentation_validated=False,
                 native_voxel_count=int(mask.sum()),
                 native_volume_mm3=float(mask.sum()*abs(np.linalg.det(affine[:3,:3]))))
    return mask.astype(bool), stats


def preview_slice_index(mask, spacing):

    areas=mask.sum(axis=(0,1));columns=mask.any(axis=1)
    occupied=np.flatnonzero(columns.any(axis=1))
    if not len(occupied):
        raise ValueError('Cannot preview an empty ROI')
    center=int(round((occupied[0]+occupied[-1])/2))
    half=max(1,int(round(15./spacing[0])))
    fronts=mask.shape[1]-1-np.argmax(mask[:,::-1,:],axis=1)
    scores=np.zeros(mask.shape[2],float)
    for z in np.flatnonzero(areas):
        ids=np.arange(max(0,center-half),min(mask.shape[0],center+half+1))
        ids=ids[columns[ids,z]]
        if len(ids):
            chest=float(np.median(fronts[ids,z]))
            y0=min(mask.shape[1],max(0,int(np.ceil(chest+5./spacing[1]))))
            scores[z]=float(mask[:,y0:,z].sum())
    return int(np.argmax(scores if scores.any() else areas))


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    labels, n_labels = label(mask)
    if n_labels == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= int(min_size)
    keep[0] = False
    return keep[labels]


def resample_mask_to_reference(mask: sitk.Image, reference: sitk.Image) -> sitk.Image:
    return sitk.Resample(
        mask,
        reference,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )


def load_volume(path, opts):
    image = nib.load(str(path))
    if len(image.shape) != 3 or min(image.shape) < 2:
        raise ValueError(f"Expected a 3D volume: {path}, shape={image.shape}")
    if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-8:
        raise ValueError(f"Invalid NIfTI affine: {path}")
    orientation_known = bool(image.header["sform_code"] or image.header["qform_code"])
    if not orientation_known and not opts["allow_missing_orientation"]:
        raise ValueError(f"{path}: no coded qform/sform; location cannot be trusted")
    unit = image.header.get_xyzt_units()[0]
    if unit != "mm":
        if unit != "unknown" or not opts["assume_mm_if_units_unknown"]:
            raise ValueError(f"{path}: spatial unit is {unit}; provide millimetre NIfTI")
    data = np.asarray(image.dataobj, dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f"Nonfinite input intensities: {path}")
    if float(data.std(dtype=np.float64)) < 1e-8:
        raise ValueError(f"Constant or empty input volume: {path}")
    return image, data, orientation_known


def bias_correction_n4_ants(in_paths: Sequence[Path], out_dir: Path, cfg: Config) -> List[Path]:
    loaded = [load_volume(p, dataclasses.asdict(cfg)) for p in in_paths]
    reference = loaded[0][0]
    if any(image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-4)
           for image, _, _ in loaded):
        raise ValueError("Shared N4 requires matching original input grids")
    corrected = shared_n4_in_memory([v[1] for v in loaded], reference,
        {"n4_iterations": cfg.n4_iterations, "n4_shrink_factor": cfg.n4_shrink_factor,
         "phase_workers": cfg.max_workers}, to_sitk)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = []
    for i, (array, (image, _, _)) in enumerate(zip(corrected, loaded)):
        target = out_dir / f"step01_bias_corrected_{i}.nii"
        save_nifti_like(array, image, target)
        result.append(target)
    return result


def save_jacobian_determinant(warp_path: str, domain_image_path: Path, out_path: Path) -> None:
    import ants
    try:
        domain = ants.image_read(str(domain_image_path))
        jac = ants.create_jacobian_determinant_image(domain, warp_path, do_log=False)
        ants.image_write(jac, str(out_path))
    except Exception as exc:
        log(f"      [WARN] Jacobian computation failed: {exc}")


def resample_to_spacing_sitk(
    img: sitk.Image,
    new_spacing: Tuple[float, float, float],
    interpolator_name: str,
) -> sitk.Image:
    spacing = np.asarray(img.GetSpacing(), dtype=float)
    size = np.asarray(img.GetSize(), dtype=int)
    target_spacing = np.asarray(new_spacing, dtype=float)
    target_size = np.maximum(2, np.round(size * spacing / target_spacing).astype(int))

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([int(v) for v in target_size])
    resampler.SetOutputSpacing(tuple(target_spacing.tolist()))
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetTransform(sitk.Transform())
    if interpolator_name.lower() == "bspline":
        resampler.SetInterpolator(sitk.sitkBSpline)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
    return sitk.Cast(resampler.Execute(img), sitk.sitkFloat32)


def resample_volume(idx: int, in_path: str, out_dir: str,
                    new_spacing: Tuple[float, float, float], interpolator_name: str) -> str:
    array, nii = load_nifti(Path(in_path))
    result, affine = resample_stack([array], nii.affine,
        {"resample": True, "target_spacing_mm": new_spacing,
         "interpolation_order": 3 if interpolator_name.lower() == "bspline" else 1,
         "phase_workers": 1})
    out_path = Path(out_dir) / f"step04_resampled_{idx}.nii"
    save_nifti_like(result[0], nib.Nifti1Image(np.zeros(result.shape[1:], np.uint8), affine), out_path)
    return str(out_path)


def stack_4d(paths: Sequence[Path]) -> Tuple[np.ndarray, nib.Nifti1Image]:
    vols: List[np.ndarray] = []
    reference: Optional[nib.Nifti1Image] = None
    for path in paths:
        arr, nii = load_nifti(path)
        if reference is None:
            reference = nii
        vols.append(arr)
    validate_volume_shapes(vols, "4D stacking")
    for path in paths:
        if not np.allclose(nib.load(str(path)).affine, reference.affine, atol=1e-4):
            raise ValueError("4D stacking requires matching physical grids")
    if reference is None:
        raise RuntimeError("No volumes available for 4D stacking.")
    return np.stack(vols, axis=-1).astype(np.float32, copy=False), reference


def percent_enhancement_4d(vol4d: np.ndarray) -> np.ndarray:
    pre = vol4d[..., 0].astype(np.float32, copy=False)
    nonzero = np.abs(pre[np.isfinite(pre) & (np.abs(pre) > 0)])
    robust_floor = float(np.percentile(nonzero, 5)) * 0.1 if nonzero.size else 1e-6
    denom = np.maximum(np.abs(pre), max(robust_floor, 1e-6))
    out = np.zeros_like(vol4d, dtype=np.float32)
    for t in range(1, vol4d.shape[-1]):
        out[..., t] = 100.0 * (vol4d[..., t] - pre) / denom
    return out


def subtraction_volume(idx: int, baseline_path: str, in_path: str, out_dir: str) -> str:
    base, base_nii = load_nifti(Path(baseline_path))
    arr, _ = load_nifti(Path(in_path))
    out_path = Path(out_dir) / f"sub_{idx}.nii"
    save_nifti_like(arr - base, base_nii, out_path)
    return str(out_path)


def build_vessel_breast_roi_from_precontrast(
    pre: np.ndarray,
    spacing_xyz: Sequence[float],
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:


    pre = np.asarray(pre, dtype=np.float32)
    if pre.ndim != 3:
        raise ValueError(f"Expected 3D precontrast image, got {pre.shape}")
    spacing = np.asarray(spacing_xyz[:3], dtype=np.float64)
    if np.any(spacing <= 0):
        raise ValueError(f"Invalid spacing: {spacing}")

    full = np.zeros_like(pre, dtype=bool)
    valid_slices = 0

    central_half_width = max(1, int(round(float(cfg.vessel_breast_central_strip_mm) / spacing[0])))
    occupancy_window = max(1, int(round(float(cfg.vessel_breast_occupancy_window_mm) / spacing[1])))
    chest_margin = max(0, int(round(float(cfg.vessel_breast_chest_margin_mm) / spacing[1])))

    for z in range(pre.shape[2]):
        image = pre[:, :, z]
        finite = np.isfinite(image)
        positive = image[finite & (image > 0)]
        if positive.size < 100:
            continue

        p99 = float(np.percentile(positive, 99.0))
        threshold = max(1e-6, float(cfg.vessel_breast_foreground_fraction) * p99)
        body = finite & (image > threshold)
        body = _largest_2d_component(body)
        if not body.any():
            continue
        body = binary_fill_holes(body)

        coords = np.argwhere(body)
        if coords.shape[0] < 100:
            continue
        xmin, ymin = coords.min(axis=0)
        xmax, ymax = coords.max(axis=0)
        center_x = int(round((xmin + xmax) / 2.0))
        x0 = max(0, center_x - central_half_width)
        x1 = min(body.shape[0], center_x + central_half_width + 1)
        if x1 <= x0:
            continue

        occupancy = body[x0:x1, :].mean(axis=0).astype(np.float32)
        occupancy = gaussian_filter1d(occupancy, sigma=2.0, mode="nearest")

        chest_y: Optional[int] = None
        for y in range(int(ymax), int(ymin + occupancy_window), -1):
            lo = max(int(ymin), y - occupancy_window)
            if occupancy[y] > 0.50 and float(np.mean(occupancy[lo : y + 1])) > 0.45:
                chest_y = y
                break

        if chest_y is None:
            chest_y = int(round(ymin + 0.60 * (ymax - ymin)))

        cut = max(int(ymin), int(chest_y) - chest_margin)
        breast = body.copy()
        breast[:, :cut] = False
        breast = ndi_binary_closing(breast, structure=np.ones((3, 3), dtype=bool), iterations=2)
        breast = binary_fill_holes(breast)


        breast = binary_dilation(breast, structure=np.ones((3, 3), dtype=bool), iterations=1)
        full[:, :, z] = breast
        valid_slices += 1

    if not full.any():
        raise RuntimeError("Could not construct a breast ROI from the precontrast phase.")


    full = ndi_binary_closing(full, structure=np.ones((3, 3, 3), dtype=bool), iterations=1)
    full = binary_fill_holes(full)


    inner = distance_transform_edt(full, sampling=tuple(spacing)) >= float(cfg.vessel_boundary_margin_mm)


    full_voxels = int(full.sum())
    inner_voxels = int(inner.sum())
    if inner_voxels < max(500, int(0.15 * full_voxels)):
        relaxed_mm = max(2.0, 0.60 * float(cfg.vessel_boundary_margin_mm))
        inner = distance_transform_edt(full, sampling=tuple(spacing)) >= relaxed_mm
        inner_voxels = int(inner.sum())

    voxel_volume = float(np.prod(spacing))
    stats = {
        "breast_roi_voxels": float(full_voxels),
        "breast_roi_volume_ml": float(full_voxels * voxel_volume / 1000.0),
        "inner_roi_voxels": float(inner_voxels),
        "inner_roi_volume_ml": float(inner_voxels * voxel_volume / 1000.0),
        "valid_mask_slices": float(valid_slices),
        "boundary_margin_mm": float(cfg.vessel_boundary_margin_mm),
    }
    return full.astype(np.uint8), inner.astype(np.uint8), stats


def _robust_unit_scale(
    image: np.ndarray,
    domain: np.ndarray,
    low_percentile: float,
    high_percentile: float,
) -> Tuple[np.ndarray, float, float]:
    image = np.asarray(image, dtype=np.float32)
    domain = np.asarray(domain, dtype=bool)
    values = image[domain & np.isfinite(image) & (image > 0)]
    if values.size < 100:
        raise ValueError("Too few positive voxels for robust vessel-source scaling.")
    lo, hi = np.percentile(values, [low_percentile, high_percentile])
    lo = float(lo)
    hi = float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError(f"Invalid robust scaling range: {lo}, {hi}")
    scaled = np.clip((image - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)
    scaled[~domain] = 0.0
    return scaled, lo, hi


def _normalize_filter_response(
    response: np.ndarray,
    domain: np.ndarray,
    low_percentile: float,
    high_percentile: float,
) -> np.ndarray:
    response = np.nan_to_num(np.asarray(response, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    domain = np.asarray(domain, dtype=bool)
    values = response[domain & (response > 0)]
    if values.size < 20:
        return np.zeros_like(response, dtype=np.float32)
    lo, hi = np.percentile(values, [low_percentile, high_percentile])
    lo = float(lo)
    hi = float(hi)
    if hi <= lo:
        return np.zeros_like(response, dtype=np.float32)
    out = np.clip((response - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)
    out[~domain] = 0.0
    return out


def _filter_sigmas_pixels(sigmas_mm: Sequence[float], spacing_xy: Sequence[float]) -> Tuple[float, ...]:


    mean_xy = max(float(np.mean(np.asarray(spacing_xy[:2], dtype=np.float64))), 1e-6)
    sigmas = tuple(max(0.5, float(mm) / mean_xy) for mm in sigmas_mm if float(mm) > 0)
    if not sigmas:
        raise ValueError("At least one positive vessel filter sigma is required.")
    return sigmas


def _retain_vessel_components(
    mask: np.ndarray,
    spacing_xyz: Sequence[float],
    cfg: Config,
) -> np.ndarray:


    labels, n_labels = label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if n_labels == 0:
        return np.zeros_like(mask, dtype=bool)

    spacing = np.asarray(spacing_xyz[:3], dtype=np.float64)
    voxel_volume = float(np.prod(spacing))
    skeleton_voxel_mm = float(np.min(spacing))
    kept = np.zeros_like(mask, dtype=bool)

    for component_id in range(1, n_labels + 1):
        component = labels == component_id
        coords = np.argwhere(component)
        if coords.size == 0:
            continue

        volume_mm3 = float(coords.shape[0] * voxel_volume)
        if volume_mm3 < float(cfg.vessel_min_component_mm3):
            continue

        ext_vox = coords.max(axis=0) - coords.min(axis=0) + 1
        ext_mm = ext_vox.astype(np.float64) * spacing
        max_extent = float(np.max(ext_mm))

        physical = coords.astype(np.float64) * spacing[None, :]
        if physical.shape[0] >= 4:
            eigvals = np.sort(np.linalg.eigvalsh(np.cov(physical, rowvar=False)))[::-1]
            pca_elongation = float(np.sqrt(max(eigvals[0], 1e-8) / max(eigvals[1], 1e-8)))
        else:
            pca_elongation = 1.0

        skeleton = skeletonize(component)
        skeleton_length_mm = float(np.count_nonzero(skeleton) * skeleton_voxel_mm)
        equivalent_diameter_mm = float((6.0 * volume_mm3 / np.pi) ** (1.0 / 3.0))
        length_diameter_ratio = skeleton_length_mm / max(equivalent_diameter_mm, 1e-6)

        elongated = (
            pca_elongation >= float(cfg.vessel_min_pca_elongation)
            and skeleton_length_mm >= float(cfg.vessel_min_skeleton_length_mm)
            and length_diameter_ratio >= float(cfg.vessel_min_length_diameter_ratio)
        )
        branching_tree = (
            volume_mm3 >= float(cfg.vessel_large_component_mm3)
            and max_extent >= float(cfg.vessel_large_component_min_extent_mm)
            and skeleton_length_mm >= float(cfg.vessel_large_tree_skeleton_mm)
            and length_diameter_ratio >= float(cfg.vessel_min_length_diameter_ratio)
        )
        if elongated or branching_tree:
            kept[component] = True

    return kept

def _inplane_dilation_structure(radius_mm: float, spacing_xyz: Sequence[float]) -> np.ndarray:
    if radius_mm <= 0:
        return np.ones((1, 1, 1), dtype=bool)
    sx, sy = float(spacing_xyz[0]), float(spacing_xyz[1])
    hx = max(1, int(np.ceil(radius_mm / sx)))
    hy = max(1, int(np.ceil(radius_mm / sy)))
    xx, yy = np.ogrid[-hx : hx + 1, -hy : hy + 1]
    disk2d = (xx * sx) ** 2 + (yy * sy) ** 2 <= radius_mm**2 + 1e-8
    return disk2d[:, :, None]


def _hessian_line_score_2d(
    image2d: np.ndarray,
    domain2d: np.ndarray,
    sigmas_px: Sequence[float],
    cfg: Config,
) -> np.ndarray:


    image2d = np.asarray(image2d, dtype=np.float32)
    domain2d = np.asarray(domain2d, dtype=bool)
    best = np.zeros_like(image2d, dtype=np.float32)
    beta = max(float(cfg.vessel_hessian_line_beta), 1e-3)

    for sigma in sigmas_px:
        sigma = float(sigma)
        hxx = gaussian_filter(image2d, sigma=sigma, order=(2, 0), mode="reflect") * (sigma ** 2)
        hyy = gaussian_filter(image2d, sigma=sigma, order=(0, 2), mode="reflect") * (sigma ** 2)
        hxy = gaussian_filter(image2d, sigma=sigma, order=(1, 1), mode="reflect") * (sigma ** 2)

        trace_half = 0.5 * (hxx + hyy)
        disc = np.sqrt((0.5 * (hxx - hyy)) ** 2 + hxy * hxy)
        e1 = trace_half - disc
        e2 = trace_half + disc

        swap = np.abs(e1) > np.abs(e2)
        lambda_small = np.where(swap, e2, e1)
        lambda_large = np.where(swap, e1, e2)

        bright_ridge = lambda_large < 0
        ratio = np.abs(lambda_small) / (np.abs(lambda_large) + 1e-8)
        line_shape = np.exp(-(ratio * ratio) / (2.0 * beta * beta))
        strength = np.maximum(-lambda_large, 0.0)

        strength_values = strength[domain2d & bright_ridge & (strength > 0)]
        if strength_values.size:
            scale_c = max(
                float(np.percentile(strength_values, cfg.vessel_hessian_strength_percentile)),
                1e-8,
            )
        else:
            scale_c = 1.0
        strength_term = 1.0 - np.exp(-(strength * strength) / (2.0 * scale_c * scale_c))
        response = line_shape * strength_term * bright_ridge * domain2d
        best = np.maximum(best, response.astype(np.float32, copy=False))

    best[~domain2d] = 0.0
    return best


def detect_vessels_frangi_hessian(
    vol4d: np.ndarray,
    cfg: Config,
    spacing_xyz: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:


    vol4d = np.asarray(vol4d, dtype=np.float32)
    if vol4d.ndim != 4 or vol4d.shape[-1] < 2:
        raise ValueError("Vessel detection requires a 4D DCE series with pre + post phases.")

    post_idx = min(max(int(cfg.vessel_post_phase), 1), vol4d.shape[-1] - 1)
    pre = vol4d[..., 0]
    post_deltas = np.maximum(vol4d[..., 1:] - pre[..., None], 0.0).astype(np.float32)
    early = post_deltas[..., post_idx - 1]
    max_post = np.max(post_deltas, axis=-1)

    breast_roi, inner_roi, roi_stats = build_vessel_breast_roi_from_precontrast(
        pre, spacing_xyz, cfg
    )
    breast = breast_roi.astype(bool)
    inner = inner_roi.astype(bool)

    early_masked = early * inner
    scaled, enh_lo, enh_hi = _robust_unit_scale(
        early_masked,
        inner,
        cfg.vessel_enhancement_low_percentile,
        cfg.vessel_enhancement_high_percentile,
    )

    sigmas_px = _filter_sigmas_pixels(cfg.vessel_filter_sigmas_mm, spacing_xyz[:2])
    frangi_raw = np.zeros_like(scaled, dtype=np.float32)
    hessian_raw = np.zeros_like(scaled, dtype=np.float32)


    for z in range(scaled.shape[2]):
        slice_domain = inner[:, :, z]
        if int(slice_domain.sum()) < 200:
            continue
        image2d = scaled[:, :, z]
        fr = frangi(
            image2d,
            sigmas=sigmas_px,
            alpha=float(cfg.vessel_filter_alpha),
            beta=float(cfg.vessel_filter_beta),
            gamma=None,
            black_ridges=False,
            mode="reflect",
        )
        hs = _hessian_line_score_2d(image2d, slice_domain, sigmas_px, cfg)
        frangi_raw[:, :, z] = np.nan_to_num(fr, nan=0.0, posinf=0.0, neginf=0.0) * slice_domain
        hessian_raw[:, :, z] = np.nan_to_num(hs, nan=0.0, posinf=0.0, neginf=0.0) * slice_domain

    frangi_norm = _normalize_filter_response(
        frangi_raw,
        inner,
        cfg.vessel_response_norm_low_percentile,
        cfg.vessel_response_norm_high_percentile,
    )
    hessian_norm = _normalize_filter_response(
        hessian_raw,
        inner,
        cfg.vessel_response_norm_low_percentile,
        cfg.vessel_response_norm_high_percentile,
    )


    consensus = np.sqrt(frangi_norm * hessian_norm) * np.sqrt(np.clip(scaled, 0.0, 1.0))
    early_fraction = np.divide(
        early,
        max_post + 1e-8,
        out=np.zeros_like(early, dtype=np.float32),
        where=max_post > 0,
    )
    temporal_weight = np.clip(
        (early_fraction - float(cfg.vessel_temporal_fraction_start))
        / max(float(cfg.vessel_temporal_fraction_width), 1e-6),
        float(cfg.vessel_temporal_min_weight),
        1.0,
    )
    consensus *= temporal_weight
    consensus *= inner

    response_values = consensus[inner & (consensus > 0)]
    if response_values.size < 20:
        vessel_mask = np.zeros_like(inner, dtype=np.uint8)
        stats = dict(roi_stats)
        stats.update({
            "post_phase": float(post_idx),
            "detected_voxels": 0.0,
            "warning": 1.0,
        })
        return early_masked, frangi_norm, hessian_norm, consensus, breast_roi, inner_roi, vessel_mask, stats

    high_threshold = float(np.percentile(response_values, cfg.vessel_consensus_high_percentile))
    low_threshold = float(np.percentile(response_values, cfg.vessel_consensus_low_percentile))
    strong = inner & (consensus >= high_threshold)
    weak = inner & (consensus >= low_threshold)

    weak_labels, n_weak = label(weak, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if n_weak > 0 and strong.any():
        strong_ids = np.unique(weak_labels[strong])
        strong_ids = strong_ids[strong_ids > 0]
        vessel = np.isin(weak_labels, strong_ids) & inner
    else:
        vessel = np.zeros_like(inner, dtype=bool)

    vessel = _retain_vessel_components(vessel, spacing_xyz, cfg)


    if vessel.any() and cfg.vessel_lumen_dilation_mm > 0:
        vessel = binary_dilation(
            vessel,
            structure=_inplane_dilation_structure(cfg.vessel_lumen_dilation_mm, spacing_xyz),
            iterations=1,
        )
        vessel &= inner
        vessel &= scaled > float(cfg.vessel_lumen_min_scaled_enhancement)
        vessel = _retain_vessel_components(vessel, spacing_xyz, cfg)

    voxel_volume = float(np.prod(np.asarray(spacing_xyz[:3], dtype=np.float64)))
    detected_voxels = int(vessel.sum())
    stats: Dict[str, float] = dict(roi_stats)
    stats.update({
        "post_phase": float(post_idx),
        "enhancement_scale_low": float(enh_lo),
        "enhancement_scale_high": float(enh_hi),
        "consensus_high_threshold": high_threshold,
        "consensus_low_threshold": low_threshold,
        "strong_seed_voxels": float(strong.sum()),
        "weak_candidate_voxels": float(weak.sum()),
        "detected_voxels": float(detected_voxels),
        "detected_volume_ml": float(detected_voxels * voxel_volume / 1000.0),
        "detected_fraction_of_inner_roi": float(detected_voxels / max(int(inner.sum()), 1)),
        "filter_sigma_count": float(len(sigmas_px)),
    })
    return (
        early_masked.astype(np.float32),
        frangi_norm.astype(np.float32),
        hessian_norm.astype(np.float32),
        consensus.astype(np.float32),
        breast_roi.astype(np.uint8),
        inner_roi.astype(np.uint8),
        vessel.astype(np.uint8),
        stats,
    )


def remove_vessels_all_phases(
    vol4d: np.ndarray,
    vessel_mask: np.ndarray,
    breast_roi: np.ndarray,
    spacing_xyz: Sequence[float],
    cfg: Config,
) -> Tuple[np.ndarray, Dict[str, float], np.ndarray]:


    out = np.asarray(vol4d, dtype=np.float32).copy()
    binary = np.asarray(vessel_mask, dtype=bool)
    breast = np.asarray(breast_roi, dtype=bool)
    strength = float(np.clip(cfg.vessel_removal_strength, 0.0, 1.0))
    spacing = np.asarray(spacing_xyz[:3], dtype=np.float64)
    sigma_vox = np.maximum(float(cfg.vessel_inpaint_sigma_mm) / spacing, 0.5)
    abs_delta_4d = np.zeros_like(out, dtype=np.float32)

    if not binary.any() or strength <= 0:
        return out, {
            "removed_voxels": float(binary.sum()),
            "mean_abs_change_all_phases": 0.0,
            "max_abs_change_all_phases": 0.0,
            "removal_strength": strength,
            "inpaint_sigma_mm": float(cfg.vessel_inpaint_sigma_mm),
        }, abs_delta_4d

    valid_source = breast & (~binary)
    if int(valid_source.sum()) < 100:
        raise RuntimeError("Too little non-vessel breast tissue is available for vessel inpainting.")

    phase_means: List[float] = []
    phase_maxima: List[float] = []
    for t in range(out.shape[-1]):
        image = out[..., t]
        numerator = gaussian_filter(
            np.where(valid_source, image, 0.0).astype(np.float32),
            sigma=sigma_vox,
            mode="nearest",
        )
        denominator = gaussian_filter(
            valid_source.astype(np.float32),
            sigma=sigma_vox,
            mode="nearest",
        )
        local_fill = np.divide(
            numerator,
            denominator + 1e-6,
            out=image.copy(),
            where=denominator > 1e-5,
        )

        replacement = (1.0 - strength) * image + strength * local_fill
        before = image[binary].copy()
        out[..., t][binary] = replacement[binary]
        changed = np.abs(before - out[..., t][binary])
        abs_delta_4d[..., t][binary] = changed
        phase_means.append(float(changed.mean()) if changed.size else 0.0)
        phase_maxima.append(float(changed.max()) if changed.size else 0.0)

    changed_all = abs_delta_4d[binary, :]
    stats: Dict[str, float] = {
        "removed_voxels": float(binary.sum()),
        "mean_abs_change_all_phases": float(changed_all.mean()) if changed_all.size else 0.0,
        "max_abs_change_all_phases": float(changed_all.max()) if changed_all.size else 0.0,
        "removal_strength": strength,
        "inpaint_sigma_mm": float(cfg.vessel_inpaint_sigma_mm),
    }
    for t, (mean_change, max_change) in enumerate(zip(phase_means, phase_maxima)):
        stats[f"phase_{t}_mean_abs_change"] = mean_change
        stats[f"phase_{t}_max_abs_change"] = max_change
    return out.astype(np.float32, copy=False), stats, abs_delta_4d


def _select_qc_slices(vessel_mask: np.ndarray, inner_roi: np.ndarray, max_slices: int) -> List[int]:
    counts = np.asarray(vessel_mask, dtype=bool).sum(axis=(0, 1))
    selected: List[int] = []
    for z in np.argsort(counts)[::-1]:
        z = int(z)
        if counts[z] <= 0:
            break
        if all(abs(z - prev) >= 3 for prev in selected):
            selected.append(z)
        if len(selected) >= max_slices:
            break

    if len(selected) < max_slices:
        area = np.asarray(inner_roi, dtype=bool).sum(axis=(0, 1))
        valid = np.flatnonzero(area > 200)
        if valid.size:
            for z in np.linspace(valid.min(), valid.max(), max_slices).round().astype(int):
                if z not in selected and all(abs(int(z) - prev) >= 2 for prev in selected):
                    selected.append(int(z))
                if len(selected) >= max_slices:
                    break
    return sorted(selected[:max_slices])


def save_vessel_qc_grid(
    output_path: Path,
    raw_4d: np.ndarray,
    vessel_removed_raw_4d: np.ndarray,
    post_idx: int,
    breast_roi: np.ndarray,
    inner_roi: np.ndarray,
    early_subtraction: np.ndarray,
    frangi_response: np.ndarray,
    hessian_response: np.ndarray,
    consensus: np.ndarray,
    vessel_mask: np.ndarray,
    abs_delta_4d: np.ndarray,
    max_slices: int,
) -> None:

    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        log(f"      [WARN] Could not import matplotlib for vessel QC grid: {exc}")
        return

    slices = _select_qc_slices(vessel_mask, inner_roi, max_slices)
    if not slices:
        log("      [WARN] No suitable slices found for vessel QC grid.")
        return

    pre = raw_4d[..., 0]
    post = raw_4d[..., post_idx]
    post_removed = vessel_removed_raw_4d[..., post_idx]
    nrows = len(slices)
    ncols = 10
    fig, axes = plt.subplots(nrows, ncols, figsize=(30, 3.0 * nrows), squeeze=False)

    for row, z in enumerate(slices):
        roi = breast_roi[:, :, z].astype(bool)
        inner = inner_roi[:, :, z].astype(bool)
        vessels = vessel_mask[:, :, z].astype(bool)
        pre_slice = pre[:, :, z]
        post_slice = post[:, :, z]
        removed_post = post_removed[:, :, z]

        display_values = np.concatenate([pre_slice[roi], post_slice[roi], removed_post[roi]]) if roi.any() else post_slice.ravel()
        vmin, vmax = np.percentile(display_values, [1.0, 99.5]) if display_values.size else (0.0, 1.0)
        if vmax <= vmin:
            vmax = vmin + 1.0

        early = early_subtraction[:, :, z]
        early_vals = early[inner]
        early_max = max(float(np.percentile(early_vals, 99.5)) if early_vals.size else 1.0, 1e-6)

        cons = consensus[:, :, z]
        cons_vals = cons[inner & (cons > 0)]
        cons_max = max(float(np.percentile(cons_vals, 99.5)) if cons_vals.size else 1.0, 1e-6)

        delta = abs_delta_4d[:, :, z, post_idx]
        delta_vals = delta[vessels]
        delta_max = max(float(np.percentile(delta_vals, 99.0)) if delta_vals.size else 1.0, 1e-6)

        panels = [
            (pre_slice, "gray", vmin, vmax, f"pre 0000  z={z}"),
            (post_slice, "gray", vmin, vmax, "early post 0001"),
            (post_slice, "gray", vmin, vmax, "breast / inner ROI"),
            (early, "gray", 0.0, early_max, "0001 - 0000"),
            (frangi_response[:, :, z], "magma", 0.0, 1.0, "Frangi"),
            (hessian_response[:, :, z], "magma", 0.0, 1.0, "Hessian line score"),
            (cons, "magma", 0.0, cons_max, "consensus"),
            (post_slice, "gray", vmin, vmax, "vessel mask"),
            (removed_post, "gray", vmin, vmax, "0001 vessel removed"),
            (delta, "hot", 0.0, delta_max, "|change| 0001"),
        ]

        for col, (panel, cmap, pmin, pmax, title) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(panel.T, cmap=cmap, origin="lower", vmin=pmin, vmax=pmax)
            if col == 2:
                if roi.any():
                    ax.contour(roi.T, [0.5], colors="cyan", linewidths=0.55)
                if inner.any():
                    ax.contour(inner.T, [0.5], colors="yellow", linewidths=0.45)
            if col in (7, 8) and vessels.any():
                ax.contour(vessels.T, [0.5], colors="lime", linewidths=0.70)
            ax.set_title(title, fontsize=8)
            ax.axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_all_phase_removal_qc_grid(
    output_path: Path,
    raw_4d: np.ndarray,
    vessel_removed_raw_4d: np.ndarray,
    breast_roi: np.ndarray,
    inner_roi: np.ndarray,
    vessel_mask: np.ndarray,
    max_slices: int,
) -> None:

    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        log(f"      [WARN] Could not import matplotlib for all-phase QC grid: {exc}")
        return

    slices = _select_qc_slices(vessel_mask, inner_roi, max_slices)
    if not slices:
        return

    phases = min(raw_4d.shape[-1], 4)
    fig, axes = plt.subplots(
        len(slices), phases * 2,
        figsize=(5.0 * phases, 2.8 * len(slices)),
        squeeze=False,
    )

    for row, z in enumerate(slices):
        roi = breast_roi[:, :, z].astype(bool)
        vessels = vessel_mask[:, :, z].astype(bool)
        for t in range(phases):
            before = raw_4d[:, :, z, t]
            after = vessel_removed_raw_4d[:, :, z, t]
            vals = np.concatenate([before[roi], after[roi]]) if roi.any() else before.ravel()
            vmin, vmax = np.percentile(vals, [1.0, 99.5]) if vals.size else (0.0, 1.0)
            if vmax <= vmin:
                vmax = vmin + 1.0

            ax0 = axes[row, 2 * t]
            ax1 = axes[row, 2 * t + 1]
            ax0.imshow(before.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
            if vessels.any():
                ax0.contour(vessels.T, [0.5], colors="lime", linewidths=0.60)
            ax0.set_title(f"z={z}  {t:04d} before", fontsize=8)
            ax1.imshow(after.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
            ax1.set_title(f"z={z}  {t:04d} vessel removed", fontsize=8)
            ax0.axis("off")
            ax1.axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _register_array_worker(payload):
    idx, fixed, moving, mask, cfg_dict, prefix = payload
    cfg = Config(**cfg_dict)
    set_threads(cfg.itk_threads)
    started = time.perf_counter()
    warped, tx = register_ants(fixed, moving, cfg, mask, prefix)
    return idx, from_sitk(warped), {
        "phase": idx, "reference_phase": cfg.registration_baseline_phase,
        "backend": "ants", "type": cfg.ants_transform, "metric": cfg.ants_metric,
        "transforms": tx["fwdtransforms"], "inverse_transforms": tx["invtransforms"],
        "elapsed_seconds": time.perf_counter() - started,
    }


def preprocess_spatial(cfg: Config, original_files: Sequence[Path], work_dir: Path):

    stages = {}
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
        loaded = list(pool.map(lambda p: load_volume(p, dataclasses.asdict(cfg)), original_files))
    images, arrays = [v[0] for v in loaded], [v[1] for v in loaded]
    fixed_idx = baseline_index_from_original_files(original_files, cfg.registration_baseline_phase) if cfg.do_registration else 0
    reference = images[fixed_idx]
    same_grid = all(image.shape == reference.shape and np.allclose(image.affine, reference.affine, atol=1e-4) for image in images)
    if not same_grid and not cfg.do_registration:
        raise ValueError("DCE grids differ; enable registration or supply physically aligned phases")
    stages["input_loading"] = time.perf_counter() - started
    started = time.perf_counter()
    if cfg.do_bias_correction:
        if not same_grid:
            raise ValueError("Shared N4 requires matching original input grids")
        log("Shared N4 correction: field from raw phase 0000")
        arrays = shared_n4_in_memory(arrays, images[0],
            {"n4_shrink_factor": cfg.n4_shrink_factor, "n4_iterations": cfg.n4_iterations,
             "phase_workers": cfg.max_workers}, to_sitk)
    stages["n4_correction"] = time.perf_counter() - started
    started = time.perf_counter()
    breast_native, breast_details = None, None
    if cfg.do_breast_mask:
        breast_native, breast_details = build_breast_mask(
            np.asarray(reference.dataobj, dtype=np.float32), reference.affine, cfg.breast_roi)
        if cfg.save_intermediate:
            save_mask_nifti(breast_native, reference, work_dir / "step01_breast_mask_baseline.nii.gz")
    stages["breast_roi"] = time.perf_counter() - started
    started = time.perf_counter()
    records = []
    if cfg.do_registration:
        fixed = to_sitk(arrays[fixed_idx], reference.affine)
        mask = to_sitk(breast_native.astype(np.uint8), reference.affine) if breast_native is not None and cfg.use_breast_mask else None
        jobs = [(i, fixed, to_sitk(a, images[i].affine), mask, dataclasses.asdict(cfg),
                 str(work_dir / f"phase_{i}_")) for i, a in enumerate(arrays) if i != fixed_idx]
        log(f"Registration: {cfg.ants_transform}/{cfg.ants_metric}; phase {cfg.registration_baseline_phase:04d} fixed; {cfg.registration_workers} workers x {cfg.itk_threads} threads")
        with ProcessPoolExecutor(max_workers=cfg.registration_workers,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            for i, data, record in pool.map(_register_array_worker, jobs):
                arrays[i] = data
                records.append(record)
                log(f"Registration phase {i:04d}: {record['elapsed_seconds']:.2f}s")
        if cfg.do_displacement:
            fixed_path = work_dir / "step02_baseline_fixed.nii.gz"
            save_nifti_like(arrays[fixed_idx], reference, fixed_path)
            for record in records:
                for tx in record["transforms"]:
                    if str(tx).endswith(("Warp.nii.gz", "Warp.nii")):
                        save_jacobian_determinant(tx, fixed_path, work_dir / f"step02_jacobian_{record['phase']}.nii.gz")
    stages["registration"] = time.perf_counter() - started
    started = time.perf_counter()

    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(cfg.itk_threads)
    stack, model_affine = resample_stack(arrays, reference.affine,
        {"resample": cfg.do_resample, "target_spacing_mm": cfg.new_spacing,
         "interpolation_order": 3 if cfg.resample_interpolator.lower() == "bspline" else 1,
         "phase_workers": cfg.max_workers})
    stages["resampling"] = time.perf_counter() - started
    target = nib.Nifti1Image(np.zeros(stack.shape[1:], np.uint8), model_affine, reference.header.copy())
    breast_model = None
    if breast_native is not None:

        breast_model = np.asarray(resample_from_to(nib.Nifti1Image(breast_native.astype(np.uint8), reference.affine),
                                                  target, order=0).dataobj) > 0
        if cfg.save_intermediate:
            save_mask_nifti(breast_model, target, work_dir / "step04_breast_mask_final_geometry.nii.gz")
    metadata = {"reference_phase": cfg.registration_baseline_phase if cfg.do_registration else 0,
        "reference_file": original_files[fixed_idx].name, "reference_affine": reference.affine.tolist(),
        "model_affine": model_affine.tolist(), "model_shape": list(stack.shape[1:]),
        "breast_roi": breast_details, "stage_timings_seconds": stages,
        "registration": records, "itk_threads": cfg.itk_threads, "serial_itk_threads": cfg.serial_itk_threads}
    return np.moveaxis(stack, 0, -1), target, breast_model, metadata


def preprocess_main(cfg: Config, patient_files: Sequence[Path]) -> PipelineResult:


    started = time.time()
    work_dir = Path(cfg.out_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    set_threads(cfg.serial_itk_threads)
    dce_files = get_dce_files_from_list(patient_files, cfg)
    if len(dce_files) != 4 or [phase_index_from_name(p.name) for p in dce_files] != [0, 1, 2, 3]:
        raise ValueError("Training/deployment parity requires exactly phases _0000 through _0003")
    original_files = list(dce_files)
    original_names = [path.name for path in original_files]
    patient_id = re.sub(r"_\d{4}\.nii(?:\.gz)?$", "", original_names[0], flags=re.IGNORECASE)
    raw_4d, ref_nii, breast_model, metadata = preprocess_spatial(cfg, original_files, work_dir)
    state = ProcessingState(current_paths=list(dce_files), current_images={}, reference_nii=ref_nii)
    if breast_model is not None:
        state.final_breast_mask = sitk.Cast(to_sitk(breast_model, ref_nii.affine), sitk.sitkUInt8)


    with timer("Step 5: vessel removal + sequence-global normalization"):
        normalization_started = time.perf_counter()
        state.reference_nii = ref_nii
        model_raw_4d = raw_4d

        if cfg.do_make_4d:
            save_nifti_like(raw_4d, ref_nii, work_dir / "4d_beforeVesselRemoval.nii.gz")

        if cfg.do_vessel_mask:
            abs_delta_4d = np.zeros_like(raw_4d, dtype=np.float32)
            spacing_xyz = tuple(float(v) for v in ref_nii.header.get_zooms()[:3])
            (
                vessel_source,
                frangi_response,
                hessian_response,
                vessel_consensus,
                vessel_breast_roi,
                vessel_inner_roi,
                vessel_mask,
                vessel_stats,
            ) = detect_vessels_frangi_hessian(raw_4d, cfg, spacing_xyz)
            state.vessel_mask = vessel_mask

            log(
                "      [VESSEL] "
                f"breast={vessel_stats.get('breast_roi_volume_ml', 0.0):.1f} mL, "
                f"inner={vessel_stats.get('inner_roi_volume_ml', 0.0):.1f} mL, "
                f"seeds={int(vessel_stats.get('strong_seed_voxels', 0))}, "
                f"final={int(vessel_stats.get('detected_voxels', 0))} voxels, "
                f"vessel_volume={vessel_stats.get('detected_volume_ml', 0.0):.2f} mL"
            )
            if vessel_stats.get("detected_voxels", 0.0) < 50:
                log("      [WARN] Very little vasculature was detected; inspect the QC grids/maps.")

            removal_stats: Dict[str, float] = {}
            if cfg.apply_vessel_suppression:
                if not cfg.vessel_remove_all_phases:
                    raise RuntimeError(
                        "This pipeline version is configured for all-phase vessel removal; "
                        "set VESSEL_REMOVE_ALL_PHASES=True."
                    )
                model_raw_4d, removal_stats, abs_delta_4d = remove_vessels_all_phases(
                    raw_4d,
                    vessel_mask,
                    vessel_breast_roi,
                    spacing_xyz,
                    cfg,
                )
                log(
                    "      [VESSEL] all-phase inpainting "
                    f"mean_abs_change={removal_stats.get('mean_abs_change_all_phases', 0.0):.4f}, "
                    f"max_abs_change={removal_stats.get('max_abs_change_all_phases', 0.0):.4f}, "
                    f"sigma={removal_stats.get('inpaint_sigma_mm', 0.0):.2f} mm"
                )

            combined_stats = dict(vessel_stats)
            combined_stats.update({f"removal_{k}": v for k, v in removal_stats.items()})

            if cfg.save_intermediate:
                save_mask_nifti(vessel_breast_roi, ref_nii, work_dir / "vessel_breast_roi.nii.gz")
                save_mask_nifti(vessel_inner_roi, ref_nii, work_dir / "vessel_inner_roi.nii.gz")
                save_nifti_like(vessel_source, ref_nii, work_dir / "vessel_early_subtraction.nii.gz")
                save_nifti_like(frangi_response, ref_nii, work_dir / "vessel_frangi.nii.gz")
                save_nifti_like(hessian_response, ref_nii, work_dir / "vessel_hessian_line_score.nii.gz")
                save_nifti_like(vessel_consensus, ref_nii, work_dir / "vessel_consensus.nii.gz")
                save_mask_nifti(vessel_mask, ref_nii, work_dir / "vessel_mask.nii.gz")
                save_nifti_like(np.max(abs_delta_4d, axis=-1), ref_nii, work_dir / "vessel_removal_max_abs_delta.nii.gz")
                write_json_atomic(work_dir / "vessel_detection_stats.json", combined_stats)

            if cfg.save_vessel_qc and cfg.vessel_qc_root:
                qc_dir = Path(cfg.vessel_qc_root) / patient_id
                qc_dir.mkdir(parents=True, exist_ok=True)
                write_json_atomic(qc_dir / f"{patient_id}_vessel_stats.json", combined_stats)

                if cfg.save_vessel_qc_nifti:
                    save_mask_nifti(vessel_breast_roi, ref_nii, qc_dir / f"{patient_id}_breast_roi.nii.gz")
                    save_mask_nifti(vessel_inner_roi, ref_nii, qc_dir / f"{patient_id}_inner_breast_roi.nii.gz")
                    save_nifti_like(vessel_source, ref_nii, qc_dir / f"{patient_id}_early_subtraction.nii.gz")
                    save_nifti_like(frangi_response, ref_nii, qc_dir / f"{patient_id}_frangi.nii.gz")
                    save_nifti_like(hessian_response, ref_nii, qc_dir / f"{patient_id}_hessian_line_score.nii.gz")
                    save_nifti_like(vessel_consensus, ref_nii, qc_dir / f"{patient_id}_consensus.nii.gz")
                    save_mask_nifti(vessel_mask, ref_nii, qc_dir / f"{patient_id}_vessel_mask.nii.gz")
                    save_nifti_like(np.max(abs_delta_4d, axis=-1), ref_nii, qc_dir / f"{patient_id}_max_abs_change_all_phases.nii.gz")

                if cfg.save_vessel_qc_grid:
                    post_idx = min(max(int(cfg.vessel_post_phase), 1), raw_4d.shape[-1] - 1)
                    save_vessel_qc_grid(
                        qc_dir / f"{patient_id}_vessel_detection_grid.png",
                        raw_4d,
                        model_raw_4d,
                        post_idx,
                        vessel_breast_roi,
                        vessel_inner_roi,
                        vessel_source,
                        frangi_response,
                        hessian_response,
                        vessel_consensus,
                        vessel_mask,
                        abs_delta_4d,
                        cfg.vessel_qc_grid_max_slices,
                    )
                    save_all_phase_removal_qc_grid(
                        qc_dir / f"{patient_id}_vessel_removal_all_phases_grid.png",
                        raw_4d,
                        model_raw_4d,
                        vessel_breast_roi,
                        vessel_inner_roi,
                        vessel_mask,
                        cfg.vessel_qc_grid_max_slices,
                    )

        if cfg.do_make_4d and cfg.apply_vessel_suppression:
            save_nifti_like(model_raw_4d, ref_nii, work_dir / "4d_afterVesselRemoval_beforeNorm.nii.gz")


        if cfg.do_percent_enhancement:
            pe = percent_enhancement_4d(model_raw_4d)
            save_nifti_like(pe, ref_nii, work_dir / "percent_enhancement_4d.nii.gz")


        final_4d = model_raw_4d
        if cfg.do_normalize:
            final_4d, norm_stats = global_normalize_4d(model_raw_4d, cfg)
            metadata["normalization"] = norm_stats
            if cfg.save_intermediate:
                write_json_atomic(work_dir / "normalization_stats.json", norm_stats)
            if cfg.do_make_4d:
                save_nifti_like(final_4d, ref_nii, work_dir / "4d_final_afterNorm.nii.gz")

    metadata["stage_timings_seconds"]["vessel_normalization"] = time.perf_counter() - normalization_started

    if cfg.do_subtractions:
        with timer("Step 7: post-minus-pre subtraction QC"):
            subtraction_dir = work_dir / "subtractions"
            subtraction_dir.mkdir(parents=True, exist_ok=True)
            final_phase_tmp = work_dir / "_final_for_subtractions"
            final_phase_tmp.mkdir(parents=True, exist_ok=True)
            temp_paths: List[Path] = []
            for t, name in enumerate(original_names):
                path = final_phase_tmp / name
                save_nifti_like(final_4d[..., t], ref_nii, path)
                temp_paths.append(path)
            workers = min(cfg.max_workers, max(len(temp_paths) - 1, 1))
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = [
                    pool.submit(
                        subtraction_volume,
                        i,
                        str(temp_paths[0]),
                        str(path),
                        str(subtraction_dir),
                    )
                    for i, path in enumerate(temp_paths[1:], start=1)
                ]
                for future in as_completed(futures):
                    future.result()


    final_stage_dir = work_dir / "final_staging"
    final_stage_dir.mkdir(parents=True, exist_ok=True)
    final_paths: List[str] = []
    if final_4d.shape[-1] != len(original_names):
        raise RuntimeError(
            f"Final 4D phase count {final_4d.shape[-1]} does not match input count {len(original_names)}."
        )
    for t, original_name in enumerate(original_names):
        final_path = final_stage_dir / original_name
        save_nifti_like(final_4d[..., t], ref_nii, final_path)
        final_paths.append(str(final_path))

    elapsed = time.time() - started
    log(f"Patient pipeline completed in {elapsed:.2f}s ({elapsed / 60:.2f} min)")
    metadata["total_preprocessing_seconds"] = elapsed
    return PipelineResult(str(work_dir), final_paths, metadata)


def output_config_fingerprint(cfg: Config) -> str:

    payload = dataclasses.asdict(cfg)
    for key in (
        "out_dir", "max_workers", "patient_workers", "stale_lock_hours",
        "save_intermediate", "do_make_4d", "do_percent_enhancement",
        "do_subtractions", "do_displacement", "save_vessel_qc",
        "save_vessel_qc_nifti", "save_vessel_qc_grid", "vessel_qc_grid_max_slices",
        "vessel_qc_root",
    ):
        payload.pop(key, None)
    payload["pipeline_version"] = PIPELINE_VERSION
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def patient_source_signature(patient_files: Sequence[Path]) -> List[Dict[str, Any]]:
    signature: List[Dict[str, Any]] = []
    for path in patient_files:
        stat = path.stat()
        signature.append({
            "name": path.name,
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        })
    return signature


def patient_outputs_complete(
    output_root: Path,
    patient_files: Sequence[Path],
    state_root: Path,
    patient_id: str,
    config_fingerprint: str,
    source_signature: Sequence[Dict[str, Any]],
) -> bool:

    expected = [output_root / path.name for path in patient_files]
    if not expected or not all(path.is_file() and path.stat().st_size > 0 for path in expected):
        return False

    state_path = state_root / f"{patient_id}.json"
    if not state_path.is_file():
        return False
    try:
        state = json.loads(state_path.read_text())
    except Exception:
        return False
    return (
        state.get("pipeline_version") == PIPELINE_VERSION
        and state.get("config_fingerprint") == config_fingerprint
        and state.get("source_signature") == list(source_signature)
        and state.get("output_names") == [path.name for path in patient_files]
    )


@contextmanager
def patient_lock(work_root: Path, patient_id: str, stale_lock_hours: float):
    lock_root = work_root / LOCK_DIR_NAME
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_dir = lock_root / f"{patient_id}.lock"

    def is_stale() -> bool:
        if stale_lock_hours <= 0:
            return False
        try:
            age_hours = (time.time() - lock_dir.stat().st_mtime) / 3600.0
            return age_hours > stale_lock_hours
        except FileNotFoundError:
            return False

    acquired = False
    while True:
        try:
            lock_dir.mkdir()
            acquired = True
            break
        except FileExistsError:
            if not is_stale():
                break
            stale_target = lock_root / f"{patient_id}.stale.{os.getpid()}.{time.time_ns()}"
            try:
                os.rename(lock_dir, stale_target)
                shutil.rmtree(stale_target, ignore_errors=True)
            except OSError:
                break

    if acquired:
        write_json_atomic(
            lock_dir / "owner.json",
            {
                "patient_id": patient_id,
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "started_unix": time.time(),
            },
        )

    try:
        yield acquired
    finally:
        if acquired:
            shutil.rmtree(lock_dir, ignore_errors=True)


def commit_final_outputs(staged_paths: Sequence[str], output_root: Path, expected_names: Sequence[str]) -> None:

    staged = [Path(path) for path in staged_paths]
    if [path.name for path in staged] != list(expected_names):
        raise RuntimeError(
            f"Staged output names {[p.name for p in staged]} do not match expected {list(expected_names)}."
        )
    for path in staged:
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"Final staged output is missing or empty: {path}")

    output_root.mkdir(parents=True, exist_ok=True)
    for staged_path in staged:
        os.replace(staged_path, output_root / staged_path.name)


def preserve_training_transforms(metadata: Dict[str, Any], state_root: Path, patient_id: str) -> None:

    records = metadata.get("registration", [])
    if not records:
        return
    target = state_root / "transforms" / patient_id
    target.mkdir(parents=True, exist_ok=True)
    copied = {}
    for record in records:
        for key in ("transforms", "inverse_transforms"):
            mapped = []
            for source_name in record.get(key, []):
                source = Path(source_name)
                if source_name not in copied:
                    destination = target / source.name
                    temporary = target / f".{source.name}.{os.getpid()}.tmp"
                    try:
                        shutil.copy2(source, temporary)
                        os.replace(temporary, destination)
                    finally:
                        temporary.unlink(missing_ok=True)
                    copied[source_name] = str(destination)
                mapped.append(copied[source_name])
            record[key] = mapped
    write_json_atomic(target / "transform_manifest.json", {
        "reference_phase": metadata["reference_phase"], "reference_file": metadata["reference_file"],
        "reference_affine": metadata["reference_affine"], "model_affine": metadata["model_affine"],
        "model_shape": metadata["model_shape"], "registration": records,
        "label_interpolator": "nearestNeighbor",
        "label_note": "Apply a phase transform only if the annotation source phase is known; these transforms do not include model-grid resampling.",
    })


def process_patient_case(
    patient_id: str,
    patient_files: Sequence[str],
    cfg_dict: Dict[str, Any],
    output_root_str: str,
    work_root_str: str,
    state_root_str: str,
    overwrite_existing: bool,
    keep_work_on_failure: bool,
    keep_work_on_success: bool,
    strict_expected_phases: bool,
    expected_phases: Sequence[int],
) -> PatientResult:
    output_root = Path(output_root_str)
    work_root = Path(work_root_str)
    state_root = Path(state_root_str)
    files = sorted((Path(p) for p in patient_files), key=lambda p: order_key(p.name))
    cfg_for_state = Config(**cfg_dict)
    config_fingerprint = output_config_fingerprint(cfg_for_state)
    source_signature = patient_source_signature(files)

    if strict_expected_phases:
        try:
            validate_expected_phases(files, expected_phases)
        except Exception as exc:
            return PatientResult(patient_id, "failed", str(exc), str(output_root))

    if not overwrite_existing and patient_outputs_complete(
        output_root, files, state_root, patient_id, config_fingerprint, source_signature
    ):
        return PatientResult(patient_id, "skipped_complete", "outputs match current preprocessing configuration", str(output_root))

    with patient_lock(work_root, patient_id, float(cfg_dict.get("stale_lock_hours", 48.0))) as acquired:
        if not acquired:
            return PatientResult(patient_id, "skipped_locked", "another worker is processing this patient", str(output_root))

        if not overwrite_existing and patient_outputs_complete(
            output_root, files, state_root, patient_id, config_fingerprint, source_signature
        ):
            return PatientResult(patient_id, "skipped_complete", "completed while waiting for lock", str(output_root))

        scratch_root = work_root / SCRATCH_DIR_NAME
        scratch_root.mkdir(parents=True, exist_ok=True)
        patient_work = scratch_root / f"{patient_id}.{os.getpid()}.{time.time_ns()}"
        patient_work.mkdir(parents=True, exist_ok=False)

        cfg = Config(**cfg_dict)
        cfg.out_dir = str(patient_work)

        try:
            result = preprocess_main(cfg, files)
            expected_names = [path.name for path in files]
            commit_final_outputs(result.primary_outputs, output_root, expected_names)
            preserve_training_transforms(result.metadata, state_root, patient_id)
            write_json_atomic(
                state_root / f"{patient_id}.json",
                {
                    "patient_id": patient_id,
                    "pipeline_version": PIPELINE_VERSION,
                    "config_fingerprint": config_fingerprint,
                    "source_signature": source_signature,
                    "output_names": expected_names,
                    "completed_unix": time.time(),
                    "preprocessing": result.metadata,
                },
            )
            if not keep_work_on_success:
                shutil.rmtree(patient_work, ignore_errors=True)
            return PatientResult(patient_id, "processed", "ok", str(output_root))
        except Exception as exc:
            if not keep_work_on_failure:
                shutil.rmtree(patient_work, ignore_errors=True)
            return PatientResult(
                patient_id,
                "failed",
                f"{exc}\n{traceback.format_exc()}",
                str(output_root),
            )


def group_patient_files(dataset_root: Path) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    for path in sorted(list(dataset_root.glob("*.nii")) + list(dataset_root.glob("*.nii.gz"))):
        match = re.fullmatch(r"(.+)_(\d{4})\.nii(?:\.gz)?", path.name, re.IGNORECASE)
        if match and int(match.group(2)) in EXPECTED_PHASES:
            groups.setdefault(match.group(1), []).append(path)
    return {patient: sorted(files, key=lambda p: order_key(p.name)) for patient, files in groups.items()}


def validate_flat_output_mirror(dataset_root: Path, output_root: Path) -> Tuple[bool, List[str], List[str]]:

    input_names = {p.name for files in group_patient_files(dataset_root).values() for p in files}
    output_entries = {p.name for p in output_root.iterdir()} if output_root.exists() else set()
    missing = sorted(input_names - output_entries)
    extra = sorted(output_entries - input_names)
    return not missing and not extra, missing, extra


def cleanup_empty_work_root(work_root: Path) -> None:

    for child in (work_root / SCRATCH_DIR_NAME, work_root / LOCK_DIR_NAME):
        try:
            child.rmdir()
        except OSError:
            pass
    try:
        work_root.rmdir()
    except OSError:
        pass


def run_dataset(
    dataset_root: Path,
    output_root: Path,
    work_root: Path,
    state_root: Path,
    cfg: Config,
) -> List[PatientResult]:
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    state_root.mkdir(parents=True, exist_ok=True)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"DATASET_ROOT does not exist: {dataset_root}")
    patient_groups = group_patient_files(dataset_root)
    if not patient_groups:
        raise ValueError(f"No patient DCE phases _0000 through _0003 found in {dataset_root}")
    patient_ids = sorted(patient_groups)
    log(f"Found {len(patient_ids)} patients / {sum(len(v) for v in patient_groups.values())} DCE files.")

    cpu_count = os.cpu_count() or 1
    potential_processes = cfg.patient_workers * max(cfg.max_workers, 1)
    if potential_processes > cpu_count:
        log(
            f"[WARN] PATIENT_WORKERS({cfg.patient_workers}) * PHASE_WORKERS({cfg.max_workers}) "
            f"= {potential_processes}, above detected CPU count {cpu_count}."
        )

    cfg_dict = dataclasses.asdict(cfg)
    results: List[PatientResult] = []
    common_args = (
        cfg_dict, str(output_root), str(work_root), str(state_root), OVERWRITE_EXISTING,
        KEEP_WORK_ON_FAILURE, KEEP_WORK_ON_SUCCESS, STRICT_EXPECTED_PHASES, EXPECTED_PHASES,
    )

    if cfg.patient_workers == 1:
        for index, patient_id in enumerate(patient_ids, start=1):
            log(f"[{index}/{len(patient_ids)}] patient {patient_id}")
            result = process_patient_case(
                patient_id,
                [str(p) for p in patient_groups[patient_id]],
                *common_args,
            )
            results.append(result)
            log(f"    {patient_id}: {result.status} - {result.message.splitlines()[0]}")
        return results

    with ProcessPoolExecutor(max_workers=cfg.patient_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {
            pool.submit(
                process_patient_case,
                patient_id,
                [str(p) for p in patient_groups[patient_id]],
                *common_args,
            ): patient_id
            for patient_id in patient_ids
        }
        for future in as_completed(futures):
            patient_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = PatientResult(
                    patient_id, "failed", f"patient worker crashed: {exc}", str(output_root)
                )
            results.append(result)
            log(f"patient {result.patient_id}: {result.status} - {result.message.splitlines()[0]}")

    return sorted(results, key=lambda r: r.patient_id)


def cpu_settings():
    limits = [os.cpu_count() or 1]
    try:
        limits.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    if os.environ.get("SLURM_CPUS_PER_TASK"):
        limits.append(int(os.environ["SLURM_CPUS_PER_TASK"]))
    budget = max(1, min(limits))
    patients = min(PATIENT_WORKERS, budget)
    per_patient = max(1, budget // patients)
    phases = min(PHASE_WORKERS, per_patient)
    registration = min(REGISTRATION_WORKERS, 3, per_patient)
    if ITK_THREADS < 0:
        raise ValueError("ITK_THREADS must be zero (automatic) or positive")
    threads = ITK_THREADS or min(8, max(1, per_patient // max(phases, registration)))
    serial = ITK_THREADS or min(8, per_patient)
    if threads * max(phases, registration) > per_patient:
        raise ValueError("Patient/phase/thread settings exceed the CPU allocation")
    return patients, phases, registration, threads, serial


def make_config() -> Config:

    patients, phases, registration, threads, serial = cpu_settings()
    if APPLY_VESSEL_SUPPRESSION:
        warnings.warn("Training vessel inpainting differs from deployment; leave APPLY_VESSEL_SUPPRESSION=False for parity.", RuntimeWarning)
    return Config(
        max_workers=phases,
        patient_workers=patients,
        registration_workers=registration,
        itk_threads=threads,
        serial_itk_threads=serial,
        n4_iterations=tuple(N4_ITERATIONS),
        n4_shrink_factor=N4_SHRINK_FACTOR,
        allow_missing_orientation=ALLOW_MISSING_ORIENTATION,
        assume_mm_if_units_unknown=ASSUME_MM_IF_UNITS_UNKNOWN,
        breast_roi=dict(BREAST_ROI_SETTINGS),
        stale_lock_hours=STALE_LOCK_HOURS,
        save_intermediate=SAVE_INTERMEDIATE,
        do_bias_correction=DO_BIAS_CORRECTION,
        do_breast_mask=DO_BREAST_MASK,
        do_registration=DO_REGISTRATION,
        do_resample=DO_RESAMPLE,
        do_normalize=DO_NORMALIZE,
        do_make_4d=SAVE_4D_QC,
        do_percent_enhancement=DO_PERCENT_ENHANCEMENT,
        do_subtractions=DO_SUBTRACTIONS,
        do_vessel_mask=DO_VESSEL_MASK or APPLY_VESSEL_SUPPRESSION,
        apply_vessel_suppression=APPLY_VESSEL_SUPPRESSION,
        do_displacement=DO_DISPLACEMENT,
        ants_transform=ANTS_TRANSFORM,
        ants_metric=ANTS_METRIC,
        ants_random_seed=ANTS_RANDOM_SEED,
        ants_histogram_matching=ANTS_HISTOGRAM_MATCHING,
        registration_baseline_phase=REGISTRATION_BASELINE_PHASE,
        use_breast_mask=USE_BREAST_MASK_FOR_REGISTRATION,
        fast_mode=FAST_MODE,
        normalize_type=NORMALIZE_TYPE,
        new_spacing=NEW_SPACING,
        resample_interpolator=RESAMPLE_INTERPOLATOR,
        vessel_post_phase=VESSEL_POST_PHASE,
        vessel_breast_foreground_fraction=VESSEL_BREAST_FOREGROUND_FRACTION,
        vessel_breast_central_strip_mm=VESSEL_BREAST_CENTRAL_STRIP_MM,
        vessel_breast_occupancy_window_mm=VESSEL_BREAST_OCCUPANCY_WINDOW_MM,
        vessel_breast_chest_margin_mm=VESSEL_BREAST_CHEST_MARGIN_MM,
        vessel_boundary_margin_mm=VESSEL_BOUNDARY_MARGIN_MM,
        vessel_filter_sigmas_mm=VESSEL_FILTER_SIGMAS_MM,
        vessel_filter_alpha=VESSEL_FILTER_ALPHA,
        vessel_filter_beta=VESSEL_FILTER_BETA,
        vessel_response_norm_low_percentile=VESSEL_RESPONSE_NORM_LOW_PERCENTILE,
        vessel_response_norm_high_percentile=VESSEL_RESPONSE_NORM_HIGH_PERCENTILE,
        vessel_enhancement_low_percentile=VESSEL_ENHANCEMENT_LOW_PERCENTILE,
        vessel_enhancement_high_percentile=VESSEL_ENHANCEMENT_HIGH_PERCENTILE,
        vessel_consensus_high_percentile=VESSEL_CONSENSUS_HIGH_PERCENTILE,
        vessel_consensus_low_percentile=VESSEL_CONSENSUS_LOW_PERCENTILE,
        vessel_temporal_fraction_start=VESSEL_TEMPORAL_FRACTION_START,
        vessel_temporal_fraction_width=VESSEL_TEMPORAL_FRACTION_WIDTH,
        vessel_temporal_min_weight=VESSEL_TEMPORAL_MIN_WEIGHT,
        vessel_min_component_mm3=VESSEL_MIN_COMPONENT_MM3,
        vessel_large_component_mm3=VESSEL_LARGE_COMPONENT_MM3,
        vessel_large_component_min_extent_mm=VESSEL_LARGE_COMPONENT_MIN_EXTENT_MM,
        vessel_lumen_dilation_mm=VESSEL_LUMEN_DILATION_MM,
        vessel_lumen_min_scaled_enhancement=VESSEL_LUMEN_MIN_SCALED_ENHANCEMENT,
        vessel_hessian_line_beta=VESSEL_HESSIAN_LINE_BETA,
        vessel_hessian_strength_percentile=VESSEL_HESSIAN_STRENGTH_PERCENTILE,
        vessel_min_pca_elongation=VESSEL_MIN_PCA_ELONGATION,
        vessel_min_skeleton_length_mm=VESSEL_MIN_SKELETON_LENGTH_MM,
        vessel_min_length_diameter_ratio=VESSEL_MIN_LENGTH_DIAMETER_RATIO,
        vessel_large_tree_skeleton_mm=VESSEL_LARGE_TREE_SKELETON_MM,
        vessel_remove_all_phases=VESSEL_REMOVE_ALL_PHASES,
        vessel_inpaint_sigma_mm=VESSEL_INPAINT_SIGMA_MM,
        vessel_removal_strength=VESSEL_REMOVAL_STRENGTH,
        save_vessel_qc=SAVE_VESSEL_QC,
        save_vessel_qc_nifti=SAVE_VESSEL_QC_NIFTI,
        save_vessel_qc_grid=SAVE_VESSEL_QC_GRID,
        vessel_qc_grid_max_slices=VESSEL_QC_GRID_MAX_SLICES,
        vessel_qc_root=str(VESSEL_QC_ROOT),
    )


def main() -> int:
    dataset_root = DATASET_ROOT.resolve()
    output_root = OUTPUT_ROOT.resolve()
    work_root = WORK_ROOT.resolve()
    state_root = STATE_ROOT.resolve()
    if dataset_root == output_root:
        raise RuntimeError("DATASET_ROOT and OUTPUT_ROOT must be different directories.")
    if work_root == output_root or work_root == dataset_root:
        raise RuntimeError("WORK_ROOT must be separate from both DATASET_ROOT and OUTPUT_ROOT.")
    if state_root in {dataset_root, output_root, work_root}:
        raise RuntimeError("STATE_ROOT must be separate from DATASET_ROOT, OUTPUT_ROOT and WORK_ROOT.")

    cfg = make_config()
    results = run_dataset(dataset_root, output_root, work_root, state_root, cfg)
    failed = [r for r in results if r.status == "failed"]
    cleanup_empty_work_root(work_root)

    log(
        f"Finished: processed={sum(r.status == 'processed' for r in results)}, "
        f"complete={sum(r.status == 'skipped_complete' for r in results)}, "
        f"locked={sum(r.status == 'skipped_locked' for r in results)}, "
        f"failed={len(failed)}"
    )

    if STRICT_OUTPUT_MIRROR:
        ok, missing, extra = validate_flat_output_mirror(dataset_root, output_root)
        if not ok:
            if missing:
                log(f"[ERROR] Output mirror is missing {len(missing)} file(s): {missing[:10]}")
            if extra:
                log(f"[ERROR] Output mirror contains {len(extra)} unexpected root entry/entries: {extra[:10]}")
            return 1
        log("Output mirror verified: root-level output NIfTI filenames exactly match the input set.")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
