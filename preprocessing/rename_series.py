"""Author: Neel Kanwal <neel.kanwal0@gmail.com>."""

import fnmatch
import logging
import filecmp
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time


from .conversion import _visible_entries

def _top_level_nifti_files(folder):

    return [p for p in Path(folder).iterdir()
            if p.is_file() and not p.is_symlink() and fnmatch.fnmatchcase(p.name, "*.nii*")]


def _folder_patient_id(folder):
    files = _top_level_nifti_files(folder)
    if not files:
        return ""
    name = files[0].name
    marker = name.rfind(".nii")
    stem = name[:marker] if marker >= 0 else name
    first_field = stem.split("_", 1)[0]
    return first_field if re.fullmatch(r"[0-9]+", first_field) else ""


def run_patient_folder_rename(converted_dir, log_dir, *, confirm_rename=True,
                              logger=None):


    logger = logger or logging.getLogger(__name__)
    converted_dir, log_dir = Path(converted_dir), Path(log_dir)
    if not converted_dir.is_dir():
        raise FileNotFoundError(f"Converted directory does not exist: {converted_dir}")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    operation_log = log_dir / f"rename_{timestamp}.log"
    rollback_path = log_dir / f"rollback_{timestamp}.sh"
    folders = [p for p in _visible_entries(converted_dir) if p.is_dir()]
    primary, duplicates, mappings = {}, {}, {}
    summary = {"folders": len(folders), "valid": 0, "renamed": 0, "merged": 0,
               "orphan_renamed": 0, "orphan_merged": 0, "failed": 0,
               "cancelled": False, "log_file": str(operation_log),
               "rollback_file": str(rollback_path)}
    with operation_log.open("w", encoding="utf-8") as operation,\
            rollback_path.open("w", encoding="utf-8") as rollback:
        rollback.write("#!/bin/bash\n# Reverts folder names only; merged contents cannot be recovered.\n")
        rollback_path.chmod(0o700)
        for folder in folders:
            patient = _folder_patient_id(folder)
            if not patient:
                logger.info("Skip folder rename: %s (no numeric ID in first NIfTI filename)", folder.name)
                operation.write(f"SKIP {folder.name}: no numeric patient ID\n")
                continue
            mappings[folder.name] = patient
            summary["valid"] += 1
            if patient not in primary:
                primary[patient] = folder.name
            else:
                duplicates.setdefault(patient, []).append(folder.name)
            logger.info("Patient folder mapping: %s -> %s", folder.name, patient)
            operation.write(f"FOUND {folder.name} -> {patient}\n")
        rename_ops = [(name, patient) for name, patient in mappings.items()
                      if name == primary[patient] and name != patient]
        logger.info("Rename plan: %s renames, %s duplicate patient groups",
                    len(rename_ops), len(duplicates))
        if confirm_rename:
            try:
                answer = input(f"Proceed with renaming {len(rename_ops)} folders and merging duplicates? (y/n): ")
            except EOFError:
                answer = ""
            if not answer or answer[0].lower() != "y":
                summary["cancelled"] = True
                logger.warning("Folder renaming cancelled")
                operation.write("Cancelled by user\n")
                return summary

        def add_rollback(old_path, new_path):
            rollback.write(f"mv -- {shlex.quote(str(new_path))} {shlex.quote(str(old_path))} 2>/dev/null\n")
            rollback.flush()

        def merge_folder(old_path, new_path):


            if old_path.resolve() == new_path.resolve():
                logger.info("Keep numeric patient folder already at destination: %s", old_path)
                return 0
            archive = converted_dir / ".merged_sources" / old_path.name
            if archive.exists():
                raise FileExistsError(f"A previous source archive exists: {archive}")
            files = list(old_path.iterdir())
            for source in files:
                target = new_path / source.name
                if source.is_symlink() or not source.is_file():
                    raise ValueError(f"Review this entry before merging: {source}")
                if target.exists() and (target.is_symlink() or not target.is_file()
                                        or not filecmp.cmp(source, target, shallow=False)):
                    raise FileExistsError(f"Conflicting converted files: {source} and {target}")
            count = 0
            for source in files:
                target = new_path / source.name
                if not target.exists():
                    shutil.copy2(source, target)
                    count += 1
            archive.parent.mkdir(exist_ok=True)
            old_path.rename(archive)
            operation.write(f"MERGED {old_path} -> {new_path}; source retained at {archive}\n")
            return count

        for folder_name, patient in rename_ops:
            old_path, new_path = converted_dir / folder_name, converted_dir / patient
            if new_path.is_dir():
                merge_folder(old_path, new_path)
                summary["merged"] += 1
            else:
                try:
                    old_path.rename(new_path)
                except OSError as error:
                    summary["failed"] += 1
                    logger.error("Folder rename failed %s -> %s: %s", old_path, new_path, error)
                    operation.write(f"FAILED RENAME {old_path}: {error}\n")
                else:
                    summary["renamed"] += 1
                    add_rollback(old_path, new_path)
                    operation.write(f"RENAMED {old_path} -> {new_path}\n")
                    logger.info("Renamed patient folder %s -> %s", folder_name, patient)

        for patient, duplicate_names in duplicates.items():
            target = converted_dir / patient
            if not target.is_dir() and (converted_dir / primary[patient]).is_dir():
                target = converted_dir / primary[patient]
            for folder_name in duplicate_names:
                old_path = converted_dir / folder_name
                if old_path.is_dir() and old_path.resolve() != target.resolve():
                    merge_folder(old_path, target)
                    summary["merged"] += 1

        for folder in [p for p in _visible_entries(converted_dir) if p.is_dir()]:
            if re.fullmatch(r"[0-9]+", folder.name):
                continue
            patient = _folder_patient_id(folder)
            if not patient:
                continue
            target = converted_dir / patient
            if target.is_dir():
                merge_folder(folder, target)
                summary["orphan_merged"] += 1
            else:
                try:
                    folder.rename(target)
                except OSError as error:
                    summary["failed"] += 1
                    logger.error("Orphan folder rename failed %s -> %s: %s", folder, target, error)
                else:
                    add_rollback(folder, target)
                    summary["orphan_renamed"] += 1
        summary["final_folders"] = len([p for p in converted_dir.iterdir() if p.is_dir() and not p.is_symlink()])
        operation.write(f"SUMMARY {summary}\n")
    logger.info("Patient-folder rename complete: %s renamed, %s merged, %s failed; log: %s",
                summary["renamed"] + summary["orphan_renamed"],
                summary["merged"] + summary["orphan_merged"],
                summary["failed"], operation_log)
    return summary
