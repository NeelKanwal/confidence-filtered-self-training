"""Author: Neel Kanwal <neel.kanwal0@gmail.com>."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import (
    binary_closing as ndi_binary_closing, binary_dilation, binary_fill_holes,
    gaussian_filter, gaussian_filter1d, label,
)
from nibabel.processing import resample_to_output, resample_from_to


INPUT_FOLDER = Path("work/preprocessed")
OUTPUT_FOLDER = INPUT_FOLDER.parent / f"{INPUT_FOLDER.name}_BreastMasks"


RAW_REFERENCE_FOLDER = Path("work/selected")
REGISTRATION_BASELINE_PHASE = 3
RECURSIVE = True
BREAST_MASK_PNG_ALPHA = 0.30
BREAST_MASK_PNG_DPI = 180
ALLOW_MISSING_ORIENTATION = False
ASSUME_MM_IF_UNITS_UNKNOWN = False


BREAST_MASK_ALGORITHM_VERSION = "sternum-curved-chestwall-v8"

@dataclass
class Config:

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
    mask: np.ndarray, spacing_xyz: Sequence[float], cfg: Config,
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


def _create_roi(reference: nib.Nifti1Image, cfg: Config) -> Tuple[np.ndarray, nib.Nifti1Image, Dict[str, float]]:


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
    cfg = Config(**{field.name: opts[field.name] for field in fields(Config) if field.name != 'breast_mask_qc_slices'})
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


def create_strict_breast_mask_nifti(path: Path, cfg: Config) -> Tuple[np.ndarray, nib.Nifti1Image, Dict[str, object]]:

    original, data, _ = load_volume(path, {"allow_missing_orientation": ALLOW_MISSING_ORIENTATION,
        "assume_mm_if_units_unknown": ASSUME_MM_IF_UNITS_UNKNOWN})
    opts = {field.name: getattr(cfg, field.name) for field in fields(Config)}
    mask, stats = build_breast_mask(data, original.affine, opts)
    return mask.astype(np.uint8), original, stats


def save_red_mask_overlay_png(
    image: np.ndarray, mask: np.ndarray, path: Path, title: str, max_slices: int,
    reference: Optional[nib.Nifti1Image] = None,
) -> None:

    if max_slices < 1:
        return
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    image = np.asarray(image, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if image.ndim != 3 or image.shape != mask.shape:
        raise ValueError(f"MRI/mask shapes must match and be 3D: {image.shape}, {mask.shape}")
    spacing = np.ones(3)
    if reference is not None:
        orientation = nib.orientations.ornt_transform(nib.orientations.io_orientation(reference.affine), nib.orientations.axcodes2ornt(("R", "A", "S")))
        image = nib.orientations.apply_orientation(image, orientation)
        mask = nib.orientations.apply_orientation(mask, orientation)
        spacing = nib.affines.voxel_sizes(nib.as_closest_canonical(reference).affine)
    z = preview_slice_index(mask, spacing)
    sl, roi = image[:, :, z].T, mask[:, :, z].T
    vals = sl[np.isfinite(sl) & (sl > 0)]
    if not vals.size:
        vals = sl[np.isfinite(sl) & roi]
    vmin, vmax = np.percentile(vals, [1.0, 99.0]) if vals.size else (0.0, 1.0)
    if vmax <= vmin:
        vmax = vmin + 1.0
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor="#080e18")
    extent = (0.0, image.shape[0] * spacing[0], 0.0, image.shape[1] * spacing[1])
    ax.imshow(sl, cmap="gray", origin="lower", vmin=vmin, vmax=vmax, extent=extent, interpolation="nearest")
    rgba = np.zeros((*roi.shape, 4), dtype=np.float32)
    rgba[..., 0] = 1.0
    rgba[..., 3] = roi * BREAST_MASK_PNG_ALPHA
    ax.imshow(rgba, origin="lower", extent=extent, interpolation="nearest")
    if roi.any() and not roi.all():
        ax.contour(roi, levels=[0.5], colors=["#ff4545"], linewidths=0.8, origin="lower", extent=extent)
    ax.set_xlim(*sorted(ax.get_xlim(), reverse=True))
    ax.set_aspect("equal")
    ax.set_title(f"{title} | axial slice {z}", color="white", fontsize=10, pad=9)
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.91)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=BREAST_MASK_PNG_DPI, facecolor=fig.get_facecolor(), metadata={"Software": "6.create_breast_mask.py | " + BREAST_MASK_ALGORITHM_VERSION})
    finally:
        plt.close(fig)


def patient_and_phase(path: Path) -> Tuple[str, Optional[int]]:

    name = re.sub(r"\.nii(?:\.gz)?$", "", path.name, flags=re.IGNORECASE)
    match = re.fullmatch(r"(.+)_(\d{4})", name)
    return (match.group(1), int(match.group(2))) if match else (name, None)


def select_patient_mris(input_folder: Path, output_folder: Path, recursive: bool = True, preferred_phase: int = REGISTRATION_BASELINE_PHASE) -> Dict[str, Path]:

    groups: Dict[str, List[Tuple[Optional[int], Path]]] = {}
    seen = set()
    finder = input_folder.rglob if recursive else input_folder.glob
    paths = list(finder("*.nii.gz")) + list(finder("*.nii"))
    derived_suffix = re.compile(r"(?:^|_)(?:mask|masks|breastmask|breast_mask|seg|segmentation|label|labels)(?:_|$)", re.IGNORECASE)
    derived_dirs = {"masks", "breast_masks", "processed_masks", "labels", "labelsTr", "labelsTs", "segmentations"}
    for path in sorted(paths):
        if not path.is_file():
            continue
        if path.resolve().is_relative_to(output_folder.resolve()):
            continue
        relative = path.relative_to(input_folder)
        if any(part.lower() in {d.lower() for d in derived_dirs} for part in relative.parts[:-1]):
            continue
        stem = re.sub(r"\.nii(?:\.gz)?$", "", path.name, flags=re.IGNORECASE)
        if derived_suffix.search(stem):
            continue
        patient, phase = patient_and_phase(path)
        identity = (patient, phase, path.resolve())
        if identity in seen:
            continue
        seen.add(identity)
        groups.setdefault(patient, []).append((phase, path))
    selected = {}
    for patient, candidates in sorted(groups.items()):

        _, selected[patient] = min(candidates, key=lambda item: (
            0 if item[0] == preferred_phase else 1 if item[0] is not None else 2,
            item[0] if item[0] is not None else 0,
            str(item[1].relative_to(input_folder)),
        ))
    return selected


def _map_raw_roi_to_target(mask: np.ndarray, raw: nib.Nifti1Image, target: nib.Nifti1Image) -> np.ndarray:

    if len(target.shape) != 3:
        raise ValueError(f"Expected a 3D target MRI, got {target.shape}")
    raw_spacing = nib.affines.voxel_sizes(raw.affine)
    target_spacing = nib.affines.voxel_sizes(target.affine)
    if np.any(~np.isfinite(target_spacing)) or np.any(target_spacing <= 0):
        raise ValueError("Target MRI has invalid physical spacing")
    raw_direction = raw.affine[:3, :3] / raw_spacing
    target_direction = target.affine[:3, :3] / target_spacing
    if not np.allclose(raw_direction, target_direction, rtol=0, atol=1e-4) or not np.allclose(raw.affine[:3, 3], target.affine[:3, 3], rtol=0, atol=0.1):
        raise ValueError("Raw/target origin or axis directions differ. Use the same fixed-reference/model grid from 5.preprocessing_mri.py; ROI resampling cannot replace registration.")
    raw_extent = np.asarray(raw.shape) * raw_spacing
    target_extent = np.asarray(target.shape) * target_spacing
    if np.any(np.abs(raw_extent - target_extent) > np.maximum(raw_spacing, target_spacing) + 0.1):
        raise ValueError("Raw/target fields of view differ beyond spacing-rounding tolerance")
    if mask.shape == target.shape and np.allclose(raw.affine, target.affine, rtol=0, atol=1e-6):
        return mask.astype(np.uint8)
    result = resample_from_to(nib.Nifti1Image(mask.astype(np.uint8), raw.affine), target, order=0)
    result = np.asarray(result.dataobj, dtype=np.uint8)
    if not result.any():
        raise ValueError("Raw ROI does not overlap target MRI")
    return result


def create_patient_outputs(patient: str, source: Path, output_folder: Path, cfg: Config, raw_source: Optional[Path] = None) -> Dict[str, object]:

    raw_source = source if raw_source is None else raw_source
    mask, raw_reference, stats = create_strict_breast_mask_nifti(raw_source, cfg)
    reference = raw_reference
    if raw_source != source:
        reference, _, _ = load_volume(source, {"allow_missing_orientation": ALLOW_MISSING_ORIENTATION,
            "assume_mm_if_units_unknown": ASSUME_MM_IF_UNITS_UNKNOWN})
    mask = _map_raw_roi_to_target(mask, raw_reference, reference)
    stats.update(roi_source=str(raw_source), target_image=str(source),
        target_mask_voxels=int(mask.sum()),
        target_mask_volume_mm3=float(mask.sum() * abs(np.linalg.det(reference.affine[:3, :3]))))
    mask_path = output_folder / f"{patient}_mask.nii.gz"
    png_path = output_folder / f"{patient}_mask.png"
    mask_tmp = output_folder / f".{patient}_mask.tmp.nii.gz"
    png_tmp = output_folder / f".{patient}_mask.tmp.png"
    output_folder.mkdir(parents=True, exist_ok=True)
    try:
        header = reference.header.copy()
        header["descrip"] = ("breast ROI v8; raw=" + raw_source.name).encode("ascii", errors="replace")[:79]
        saved = nib.Nifti1Image(mask.astype(np.uint8), reference.affine, header)
        saved.set_data_dtype(np.uint8)
        saved.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
        saved.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
        nib.save(saved, str(mask_tmp))
        save_red_mask_overlay_png(
            reference.get_fdata(dtype=np.float32), mask, png_tmp,
            f"{patient} | target {source.name}", 1, reference=reference,
        )
        check = nib.load(str(mask_tmp))
        if check.shape != reference.shape or not np.allclose(check.affine, reference.affine, rtol=0, atol=1e-5):
            raise RuntimeError(f"Saved mask geometry differs from {source.name}")
        mask_tmp.replace(mask_path)
        png_tmp.replace(png_path)
    finally:
        mask_tmp.unlink(missing_ok=True)
        png_tmp.unlink(missing_ok=True)
    return stats


def main() -> int:
    input_folder = INPUT_FOLDER.expanduser().resolve()
    output_folder = OUTPUT_FOLDER.expanduser().resolve()
    if not input_folder.is_dir():
        print(f"[ERROR] Input folder does not exist: {input_folder}", file=sys.stderr)
        return 2
    if output_folder == input_folder:
        print("[ERROR] Choose an OUTPUT_FOLDER different from INPUT_FOLDER", file=sys.stderr)
        return 2
    if output_folder.exists() and not output_folder.is_dir():
        print(f"[ERROR] Output path is not a folder: {output_folder}", file=sys.stderr)
        return 2
    selected = select_patient_mris(input_folder, output_folder, RECURSIVE, REGISTRATION_BASELINE_PHASE)
    if not selected:
        print("[ERROR] No eligible MRI .nii/.nii.gz files found (existing masks/labels are excluded)", file=sys.stderr)
        return 2
    raw_selected = None
    if RAW_REFERENCE_FOLDER is not None:
        raw_folder = Path(RAW_REFERENCE_FOLDER).expanduser().resolve()
        if not raw_folder.is_dir():
            print(f"[ERROR] Raw reference folder does not exist: {raw_folder}", file=sys.stderr)
            return 2
        raw_selected = select_patient_mris(raw_folder, output_folder, RECURSIVE, REGISTRATION_BASELINE_PHASE)
        missing = [patient for patient in selected if patient not in raw_selected or patient_and_phase(raw_selected[patient])[1] != REGISTRATION_BASELINE_PHASE]
        if missing:
            print(f"[ERROR] Missing raw _{REGISTRATION_BASELINE_PHASE:04d} reference for: {', '.join(missing)}", file=sys.stderr)
            return 2
    cfg = Config()
    print(f"Selected {len(selected)} unique patient(s). Output: {output_folder}", flush=True)
    failures = 0
    for patient, path in selected.items():
        try:
            raw_source = raw_selected[patient] if raw_selected is not None else path
            print(f"[SELECT] {patient}: raw ROI {raw_source} | target {path}", flush=True)
            stats = create_patient_outputs(patient, path, output_folder, cfg, raw_source=raw_source)
            print(f"[OK] {patient}_mask.nii.gz + {patient}_mask.png | {stats['target_mask_volume_mm3'] / 1000.0:.1f} mL", flush=True)
        except Exception as exc:
            failures += 1
            print(f"[FAILED] {patient}: {exc}", file=sys.stderr, flush=True)
    print(f"Finished: {len(selected) - failures} succeeded, {failures} failed.", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
