"""Author: Neel Kanwal <neel.kanwal0@gmail.com>."""

import fnmatch
import logging
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time


def _visible_entries(folder):
    return sorted((p for p in Path(folder).iterdir() if not p.name.startswith(".")),
                  key=lambda p: p.name)


def _conversion_series_at_depth(patient_path, depth=3):

    patient_path = Path(patient_path)
    if not patient_path.is_dir() or patient_path.is_symlink():
        return []
    current = [patient_path]
    for _ in range(int(depth)):
        following = []
        for parent in current:
            try:
                following.extend(p for p in parent.iterdir()
                                 if p.is_dir() and not p.is_symlink())
            except OSError:
                continue
        current = following
    return sorted(current, key=str)


def _dicom_series_description(first_file, series_folder, dcmdump_executable):
    if dcmdump_executable:
        result = subprocess.run([str(dcmdump_executable), str(first_file)],
                                text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, check=False)
        for tag in ("Series Description", "Protocol Name"):
            for line in result.stdout.splitlines():
                if tag in line:
                    match = re.search(r".*\[(.*)\].*", line)
                    if match and match.group(1):
                        return match.group(1)
    return "Series_" + Path(series_folder).name


def run_dicom_conversion(source_dir, converted_dir, log_dir, *,
                         dcm2niix_executable="dcm2niix",
                         dcm2niix_options=("-z", "n", "-v", "2", "-f", "%i_%d"),
                         min_files_threshold=8, series_depth=3, logger=None):


    logger = logger or logging.getLogger(__name__)
    source_dir, converted_dir, log_dir = map(Path, (source_dir, converted_dir, log_dir))
    executable = shutil.which(str(dcm2niix_executable))
    if not executable:
        raise FileNotFoundError(f"dcm2niix not found: {dcm2niix_executable}")
    file_executable = shutil.which("file")
    if not file_executable:
        raise FileNotFoundError("The file utility is required for the original DICOM-folder check")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"DICOM source directory does not exist: {source_dir}")
    converted_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    patients = _visible_entries(source_dir)
    if not patients:
        raise RuntimeError(f"No patients found in DICOM source directory: {source_dir}")
    options = (shlex.split(dcm2niix_options) if isinstance(dcm2niix_options, str)
               else [str(option) for option in dcm2niix_options])
    conversion_log = log_dir / f"conversion_{time.strftime('%Y%m%d_%H%M%S')}.log"
    summary = {"patients": len(patients), "series": 0, "converted": 0,
               "failed": 0, "skipped": 0, "log_file": str(conversion_log)}
    dcmdump_executable = shutil.which("dcmdump")
    with conversion_log.open("w", encoding="utf-8") as detail:
        detail.write(f"DICOM conversion started: {time.ctime()}\nSource: {source_dir}\n"
                     f"Destination: {converted_dir}\nOptions: {shlex.join(options)}\n")
        detail.flush()
        for patient_number, patient in enumerate(patients, 1):
            destination = converted_dir / patient.name
            destination.mkdir(parents=True, exist_ok=True)
            series_folders = _conversion_series_at_depth(patient, series_depth)
            logger.info("DICOM patient %s/%s: %s; %s potential series at depth %s",
                        patient_number, len(patients), patient.name,
                        len(series_folders), series_depth)
            if not series_folders:
                logger.warning("No series directories at depth %s below %s", series_depth, patient)
            for number, folder in enumerate(series_folders, 1):
                summary["series"] += 1
                try:
                    entries = _visible_entries(folder)
                except OSError as error:
                    entries = []
                    logger.warning("Cannot list series %s: %s", folder, error)
                is_dicom = False
                if len(entries) >= min_files_threshold:
                    probe = subprocess.run([file_executable, "-b", str(entries[0])],
                                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                           text=True, check=False)
                    is_dicom = "dicom" in probe.stdout.lower()
                if not is_dicom:
                    summary["skipped"] += 1
                    detail.write(f"Series: {folder}; status: No DICOM\n")
                    logger.info("Skip series %s/%s: %s (original DICOM check failed)",
                                number, len(series_folders), folder.name)
                    continue
                description = _dicom_series_description(entries[0], folder, dcmdump_executable)
                clean_description = re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9_-]", "_", description)).strip("_")
                command = [executable, *options, "-o", str(destination), str(folder)]
                detail.write(f"\nSeries: {clean_description}\nPath: {folder}\n"
                             f"Description: {description}\nCommand: {shlex.join(command)}\n")
                detail.flush()
                started = time.perf_counter()
                logger.info("Converting series %s/%s: %s", number, len(series_folders), clean_description)
                result = subprocess.run(command, stdout=detail, stderr=subprocess.STDOUT, check=False)
                elapsed = time.perf_counter() - started
                if result.returncode == 0:
                    summary["converted"] += 1
                    logger.info("Converted %s in %.1f s", clean_description, elapsed)
                else:
                    summary["failed"] += 1
                    logger.error("Conversion failed for %s (exit %s; %.1f s); details: %s",
                                 folder, result.returncode, elapsed, conversion_log)
                detail.write(f"Exit: {result.returncode}; elapsed_seconds: {elapsed:.3f}\n")
                detail.flush()
        detail.write(f"\nSummary: {summary}\nCompleted: {time.ctime()}\n")
    logger.info("Conversion complete: %s converted, %s failed, %s skipped; log: %s",
                summary["converted"], summary["failed"], summary["skipped"], conversion_log)
    return summary
