"""Author: Neel Kanwal <neel.kanwal0@gmail.com>."""

import os
import json
import re
import shutil
import numpy as np
from typing import Dict, List, Optional


PREFIX_CONFIG = {
    "3": {
        "mode": "fixed",
        "indices": [0, 2, 5, 7]
    },
    "1": {
        "mode": "adaptive",
        "map": {
            11: [0, 3, 7, 9],
            8:  [0, 2, 5, 7],
            7:  [0, 2, 4, 6],
            "default": [0, 2, 4, 6]
        }
    }
}


PROCESS_PREFIXES = ["3", "1"]


TIME_POINT_PATTERN = re.compile(r'_t(\d{2})\.nii.gz$')


def extract_ttc_from_filename(filename: str) -> Optional[float]:
    patterns = [
        r'TTC[=\s]*(\d+(?:\.\d+)?)\s*s',
        r'TTC[=\s]*(\d+(?:\.\d+)?)',
        r'_TTC[=\s]*(\d+(?:\.\d+)?)'
    ]
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None

def extract_ttc_from_json(json_path: str) -> Optional[float]:
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        series_desc = data.get('SeriesDescription', '')
        patterns = [
            r'TTC[=\s]*(\d+(?:\.\d+)?)\s*s',
            r'TTC[=\s]*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*sec',
            r'(\d+(?:\.\d+)?)\s*s'
        ]
        for pattern in patterns:
            match = re.search(pattern, series_desc, re.IGNORECASE)
            if match:
                return float(match.group(1))
    except:
        pass
    return None


def find_files_hospital3(patient_dir: str) -> List[Dict]:

    dce_files = []
    for file_name in os.listdir(patient_dir):
        if not file_name.endswith('.nii.gz'):
            continue
        if '_SUB' in file_name:
            continue
        if 'twist_dyn' in file_name.lower():
            json_file = file_name.replace('.nii.gz', '.json')
            json_path = os.path.join(patient_dir, json_file)
            ttc = extract_ttc_from_filename(file_name)
            if ttc is None and os.path.exists(json_path):
                ttc = extract_ttc_from_json(json_path)
            if ttc is not None:
                dce_files.append({
                    'nii_file': file_name,
                    'json_file': json_file if os.path.exists(json_path) else None,
                    'key': ttc,
                    'display': f"TTC={ttc}s"
                })
    dce_files.sort(key=lambda x: x['key'])
    return dce_files

def find_files_hospital1(patient_dir: str) -> List[Dict]:

    dce_files = []
    for file_name in os.listdir(patient_dir):
        if not file_name.endswith('.nii.gz'):
            continue
        match = TIME_POINT_PATTERN.search(file_name)
        if match:
            index = int(match.group(1))
            json_file = file_name.replace('.nii.gz', '.json')
            json_path = os.path.join(patient_dir, json_file)
            dce_files.append({
                'nii_file': file_name,
                'json_file': json_file if os.path.exists(json_path) else None,
                'key': index,
                'display': f"t{index:02d}"
            })
    dce_files.sort(key=lambda x: x['key'])
    return dce_files


def select_indices(total: int, config: Dict, verbose: bool = True) -> List[int]:


    mode = config.get("mode", "fixed")
    if mode == "fixed":
        idx_list = config.get("indices", [0, 1, 2, 3])

        indices = [i for i in idx_list if i < total]
        if len(indices) < 4:

            remaining = [i for i in range(total) if i not in indices]
            needed = 4 - len(indices)
            if remaining:
                step = max(1, len(remaining) // needed)
                fill = remaining[0::step][:needed]
                indices.extend(fill)
            indices = sorted(indices)[:4]
        return indices

    elif mode == "adaptive":
        idx_map = config.get("map", {})
        idx_list = idx_map.get(total)
        if idx_list is None:
            idx_list = idx_map.get("default", [0, 1, 2, 3])

        indices = [i for i in idx_list if i < total]
        if len(indices) < 4:
            remaining = [i for i in range(total) if i not in indices]
            needed = 4 - len(indices)
            if remaining:
                step = max(1, len(remaining) // needed)
                fill = remaining[0::step][:needed]
                indices.extend(fill)
            indices = sorted(indices)[:4]
        return indices

    else:
        raise ValueError(f"Unknown mode: {mode}")


def process_patient(patient_dir: str, output_base_dir: str, verbose: bool = True) -> bool:
    patient_id = os.path.basename(patient_dir)
    prefix = patient_id[0]


    config = PREFIX_CONFIG.get(prefix)
    if config is None:
        if verbose:
            print(f"  ✗ Patient {patient_id}: No configuration for prefix '{prefix}'.")
        return False


    if prefix == "3":
        dce_files = find_files_hospital3(patient_dir)
    elif prefix == "1":
        dce_files = find_files_hospital1(patient_dir)
    else:
        if verbose:
            print(f"  ✗ Patient {patient_id}: Unknown prefix '{prefix}'. Only '1' and '3' are supported.")
        return False

    total = len(dce_files)
    if total < 4:
        if verbose:
            print(f"  ✗ Patient {patient_id}: Only {total} files found (need 4).")
        return False


    selected_indices = select_indices(total, config, verbose)
    if len(selected_indices) < 4:
        if verbose:
            print(f"  ✗ Patient {patient_id}: Could not select 4 indices.")
        return False


    selected_files = [dce_files[i] for i in selected_indices]

    if verbose:
        print(f"  Patient {patient_id} ({prefix}***) – {total} files found.")
        print(f"    Selected indices: {selected_indices}")
        print(f"    Corresponding files: {[f['display'] for f in selected_files]}")


    for out_idx, file_info in enumerate(selected_files):
        src_nii = os.path.join(patient_dir, file_info['nii_file'])
        dest_nii = os.path.join(output_base_dir, f"{patient_id}_{out_idx:04d}.nii.gz")
        shutil.copy2(src_nii, dest_nii)
        if verbose:
            print(f"      -> {os.path.basename(dest_nii)}")

    return True

def batch_process(input_dir: str, output_dir: str,
                  max_patients: int = None, verbose: bool = True):
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)


    patients = []
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path):
            prefix = item[0] if item else ""
            if prefix in PROCESS_PREFIXES:
                patients.append(item_path)

    patients.sort()
    if max_patients:
        patients = patients[:max_patients]

    print(f"\n{'='*80}")
    print(f"UNIFIED DCE PROCESSING FOR nnUNet PREDICT")
    print(f"{'='*80}")
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Processing prefixes: {PROCESS_PREFIXES}")
    print(f"Total patients: {len(patients)}")
    print(f"{'='*80}\n")

    successful = []
    failed = []

    for i, patient_dir in enumerate(patients, 1):
        patient_id = os.path.basename(patient_dir)
        try:
            success = process_patient(patient_dir, output_dir, verbose=verbose)
            if success:
                successful.append(patient_id)
                if not verbose:
                    print(f"[{i:3d}/{len(patients)}] {patient_id} ✓")
            else:
                failed.append(patient_id)
        except Exception as e:
            failed.append(patient_id)
            print(f"  ✗ ERROR on {patient_id}: {str(e)}")

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Successfully processed: {len(successful)} patients")
    print(f"Failed: {len(failed)} patients")
    print(f"Output ready for nnUNetv2_predict")
    print(f"{'='*80}\n")

    return successful, failed
