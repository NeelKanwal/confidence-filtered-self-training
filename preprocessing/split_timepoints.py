"""Author: Neel Kanwal <neel.kanwal0@gmail.com>."""

import re
import json
import shutil
import datetime
from pathlib import Path

import numpy as np
import nibabel as nib


BASE_DIR      = "work/converted"
KEEP_ORIGINAL = True
DRY_RUN       = False
RECURSIVE     = True


PATIENT_DIR_PATTERN = re.compile(r'^[13]\d{3}$')


DCE_PATTERN = re.compile(
    r'^(?:'
    r'1\d{3}_T1_tra_vibe(?!.*(?:SUB|MIP)).*(?:p4|DRBh).*W.*?'
    r'|'
    r'3\d{3}_twist_dyn(?!.*(?:SUB|MIP)).*wex.*?'
    r')(?:\.nii(?:\.gz)?)?$',
    re.IGNORECASE,
)


SPLIT_OUTPUT_PATTERN = re.compile(r'_t\d{2}$')

ARCHIVE_DIR_NAME = "_originals"


def nii_base(path):

    name = path.name
    for ext in (".nii.gz", ".nii"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return path.stem


def nii_ext(path):
    return ".nii.gz" if path.name.lower().endswith(".nii.gz") else ".nii"


def sidecar_json(path):
    return path.parent / f"{nii_base(path)}.json"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent="\t")


def is_target_file(nii_path):

    if SPLIT_OUTPUT_PATTERN.search(nii_base(nii_path)):
        return False
    return bool(DCE_PATTERN.match(nii_path.name))


def n_volumes(img):

    if img.ndim == 4:
        return img.shape[3]
    if img.ndim == 5 and img.shape[3] == 1:
        return img.shape[4]
    return 1


def find_candidates(patient_dir):
    pattern_iter = patient_dir.rglob("*") if RECURSIVE else patient_dir.glob("*")
    files = []
    for f in sorted(pattern_iter):
        if not f.is_file():
            continue
        if ARCHIVE_DIR_NAME in f.relative_to(patient_dir).parts:
            continue
        if not f.name.lower().endswith((".nii", ".nii.gz")):
            continue
        if is_target_file(f):
            files.append(f)
    return files


def bump_acquisition_time(base_str, offset_sec):

    if not base_str:
        return None
    try:
        h, m, s = base_str.split(":")
        base = datetime.datetime(2000, 1, 1, int(h), int(m)) + datetime.timedelta(seconds=float(s))
        return (base + datetime.timedelta(seconds=offset_sec)).strftime("%H:%M:%S.%f")
    except (ValueError, TypeError):
        return None


def frame_times(json_data, n_vol):


    raw = json_data.get("FrameTimesStart")
    if raw:
        raw = [float(t) for t in raw]
        if max(raw) > 3600:
            raw = [t / 1000.0 for t in raw]
        if len(raw) != n_vol:
            raise ValueError(f"FrameTimesStart has {len(raw)} entries for {n_vol} volumes")
        return raw, "FrameTimesStart"

    tr = json_data.get("RepetitionTime")
    if tr is not None and float(tr) >= 1.0:
        tr = float(tr)
        print(f"    INFO: No FrameTimesStart. Estimating frame times from RepetitionTime ({tr} s).")
        return [i * tr for i in range(n_vol)], "RepetitionTime estimate"

    print("    WARNING: No FrameTimesStart and RepetitionTime is not a per-frame interval. "
          "Splitting without time stamps.")
    return None, "none"


def split_collated_file(nii_path, json_data):
    img      = nib.load(str(nii_path))
    n_vol    = n_volumes(img)
    base     = nii_base(nii_path)
    ext      = nii_ext(nii_path)
    out_dir  = nii_path.parent
    base_acq = json_data.get("AcquisitionTime")
    orig_comment = json_data.get("ImageComments")

    times, source = frame_times(json_data, n_vol)
    n_out = min(n_vol, len(times)) if times else n_vol

    if img.ndim == 5:
        img = nib.Nifti1Image(np.asanyarray(img.dataobj)[:, :, :, 0, :], img.affine, img.header)

    for t in range(n_out):
        tag = f"_t{t:02d}"
        out_nii  = out_dir / f"{base}{tag}{ext}"
        out_json = out_dir / f"{base}{tag}.json"

        if not DRY_RUN:

            nib.save(img.slicer[:, :, :, t], str(out_nii))

            out_data = dict(json_data)
            out_data.pop("FrameTimesStart", None)
            out_data.pop("BidsGuess", None)
            out_data["TimePointIndex"] = t
            out_data["NumberOfTimePoints"] = n_vol
            out_data["FrameTimeSource"] = source
            if orig_comment is not None:
                out_data["OriginalImageComments"] = orig_comment

            if times:
                ttc = round(times[t], 3)
                out_data["FrameTimeStart"] = ttc
                out_data["ImageComments"] = f"TTC {ttc} sec"
                acq = bump_acquisition_time(base_acq, ttc)
                if acq:
                    out_data["AcquisitionTime"] = acq

            save_json(out_data, str(out_json))

        stamp = f"t={round(times[t], 3)}s" if times else "t=unknown"
        print(f"    t{t:02d}  {stamp:<14} ->  {out_nii.name}")

    return n_out


def archive_original(nii_path):
    orig_dir = nii_path.parent / ARCHIVE_DIR_NAME
    orig_dir.mkdir(exist_ok=True)
    for src in (nii_path, sidecar_json(nii_path)):
        if src.exists():
            shutil.move(str(src), str(orig_dir / src.name))
            print(f"    Archived -> {ARCHIVE_DIR_NAME}/{src.name}")


def remove_original(nii_path):
    for src in (nii_path, sidecar_json(nii_path)):
        if src.exists():
            src.unlink()


def process_patient(patient_dir, summary):
    print(f"\n{'─' * 60}\nPatient {patient_dir.name}")
    candidates = find_candidates(patient_dir)

    if not candidates:
        print("  No DCE series found.")
        summary["no_dce"].append(patient_dir.name)
        return

    for nii_path in candidates:
        rel = nii_path.relative_to(patient_dir)
        json_path = sidecar_json(nii_path)
        if not json_path.exists():
            print(f"  SKIP (no JSON sidecar): {rel}")
            summary["no_json"].append(f"{patient_dir.name}/{rel}")
            continue

        try:
            img = nib.load(str(nii_path))
        except Exception as e:
            print(f"  ERROR reading {rel}: {e}")
            summary["errors"].append(f"{patient_dir.name}/{rel}: {e}")
            continue

        n_vol = n_volumes(img)
        if n_vol <= 1:

            print(f"  Already a single time point (3D): {rel}")
            summary["presplit"].append(f"{patient_dir.name}/{rel}")
            continue

        json_data = load_json(json_path)
        print(f"  Splitting {rel}  ({n_vol} time points)")
        try:
            n_saved = split_collated_file(nii_path, json_data)
        except Exception as e:
            print(f"  ERROR splitting {rel}: {e}")
            summary["errors"].append(f"{patient_dir.name}/{rel}: {e}")
            continue

        if not DRY_RUN:
            if KEEP_ORIGINAL:
                archive_original(nii_path)
            else:
                remove_original(nii_path)

        print(f"  ✓ {'Would save' if DRY_RUN else 'Saved'} {n_saved} time points")
        summary["split"].append(f"{patient_dir.name}/{rel} ({n_saved})")


def print_summary(summary):
    print(f"\n{'═' * 60}\nSUMMARY{' (DRY RUN, nothing written)' if DRY_RUN else ''}")
    print(f"  Split 4D series            : {len(summary['split'])}")
    print(f"  Already single time point  : {len(summary['presplit'])}")
    print(f"  Patients without DCE match : {len(summary['no_dce'])}")
    print(f"  Missing JSON sidecar       : {len(summary['no_json'])}")
    print(f"  Errors                     : {len(summary['errors'])}")
    for key, title in (("no_dce", "Patients without a matching DCE file"),
                       ("no_json", "Files without JSON"),
                       ("errors", "Errors")):
        if summary[key]:
            print(f"\n  {title}:")
            for item in summary[key]:
                print(f"    - {item}")


def main():
    base = Path(BASE_DIR)
    patient_dirs = sorted(d for d in base.iterdir()
                          if d.is_dir() and PATIENT_DIR_PATTERN.match(d.name))
    print(f"Found {len(patient_dirs)} patient folders in {base}")

    summary = {k: [] for k in ("split", "presplit", "no_dce", "no_json", "errors")}
    for pd in patient_dirs:
        process_patient(pd, summary)

    print_summary(summary)
    print(f"\n{'─' * 60}\nDone.")


if __name__ == "__main__":
    main()
