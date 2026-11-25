"""Author: Neel Kanwal <neel.kanwal0@gmail.com>."""

from pathlib import Path


DICOM_INPUT_DIR = Path('work/dicom')
CONVERTED_DIR = Path('work/converted')
SELECTED_DCE_DIR = Path('work/selected')
PREPROCESSED_DIR = Path('work/preprocessed')
BREAST_MASK_DIR = PREPROCESSED_DIR.parent / f'{PREPROCESSED_DIR.name}_BreastMasks'
LOG_DIR = PREPROCESSED_DIR.parent / f'{PREPROCESSED_DIR.name}_Logs'

RUN_STAGES = (1, 2, 3, 4, 5, 6)
CONFIRM_FOLDER_RENAME = True
DCM2NIIX_EXECUTABLE = 'dcm2niix'
DCM2NIIX_OPTIONS = ('-z', 'n', '-v', '2', '-f', '%i_%d')
MIN_DICOM_FILES = 8
DICOM_SERIES_DEPTH = 3

KEEP_ORIGINAL_4D = True
SPLIT_DRY_RUN = False
SPLIT_RECURSIVE = True
PROCESS_PREFIXES = ['3', '1']
MAX_PATIENTS = None
SELECTION_VERBOSE = True


ENABLE_NIFTI_HANDOFF = True

USE_SPLIT_FRAME_TIMES = True


PREFIX_CONFIG = {'3': {'mode': 'fixed', 'indices': [0, 2, 5, 7]},
 '1': {'mode': 'adaptive',
       'map': {11: [0, 3, 7, 9], 8: [0, 2, 5, 7], 7: [0, 2, 4, 6], 'default': [0, 2, 4, 6]}}}


PREPROCESSING_SETTINGS = {'PATIENT_WORKERS': 1,
 'PHASE_WORKERS': 4,
 'REGISTRATION_WORKERS': 3,
 'ITK_THREADS': 0,
 'STALE_LOCK_HOURS': 48.0,
 'OVERWRITE_EXISTING': False,
 'KEEP_WORK_ON_FAILURE': False,
 'KEEP_WORK_ON_SUCCESS': False,
 'STRICT_OUTPUT_MIRROR': True,
 'EXPECTED_PHASES': (0, 1, 2, 3),
 'REGISTRATION_BASELINE_PHASE': 3,
 'STRICT_EXPECTED_PHASES': True,
 'DO_BIAS_CORRECTION': True,
 'DO_BREAST_MASK': True,
 'DO_REGISTRATION': True,
 'DO_RESAMPLE': True,
 'DO_NORMALIZE': True,
 'DO_VESSEL_MASK': False,
 'APPLY_VESSEL_SUPPRESSION': False,
 'SAVE_INTERMEDIATE': False,
 'SAVE_4D_QC': False,
 'DO_PERCENT_ENHANCEMENT': False,
 'DO_SUBTRACTIONS': False,
 'DO_DISPLACEMENT': False,
 'ANTS_TRANSFORM': 'SyN',
 'ANTS_METRIC': 'MI',
 'ANTS_RANDOM_SEED': 0,
 'ANTS_HISTOGRAM_MATCHING': False,
 'N4_ITERATIONS': (50, 40, 30, 20),
 'N4_SHRINK_FACTOR': 2,
 'ALLOW_MISSING_ORIENTATION': False,
 'ASSUME_MM_IF_UNITS_UNKNOWN': False,
 'USE_BREAST_MASK_FOR_REGISTRATION': True,
 'FAST_MODE': True,
 'NORMALIZE_TYPE': 'zscore',
 'NEW_SPACING': (1.0, 1.0, 1.0),
 'RESAMPLE_INTERPOLATOR': 'bspline',
 'VESSEL_POST_PHASE': 1,
 'VESSEL_BREAST_FOREGROUND_FRACTION': 0.05,
 'VESSEL_BREAST_CENTRAL_STRIP_MM': 30.0,
 'VESSEL_BREAST_OCCUPANCY_WINDOW_MM': 10.0,
 'VESSEL_BREAST_CHEST_MARGIN_MM': 12.0,
 'VESSEL_BOUNDARY_MARGIN_MM': 5.0,
 'VESSEL_FILTER_SIGMAS_MM': (0.8, 1.2, 1.8, 2.5),
 'VESSEL_FILTER_ALPHA': 0.5,
 'VESSEL_FILTER_BETA': 0.5,
 'VESSEL_RESPONSE_NORM_LOW_PERCENTILE': 30.0,
 'VESSEL_RESPONSE_NORM_HIGH_PERCENTILE': 99.7,
 'VESSEL_ENHANCEMENT_LOW_PERCENTILE': 20.0,
 'VESSEL_ENHANCEMENT_HIGH_PERCENTILE': 99.5,
 'VESSEL_CONSENSUS_HIGH_PERCENTILE': 98.0,
 'VESSEL_CONSENSUS_LOW_PERCENTILE': 92.0,
 'VESSEL_TEMPORAL_FRACTION_START': 0.25,
 'VESSEL_TEMPORAL_FRACTION_WIDTH': 0.5,
 'VESSEL_TEMPORAL_MIN_WEIGHT': 0.35,
 'VESSEL_MIN_COMPONENT_MM3': 8.0,
 'VESSEL_LARGE_COMPONENT_MM3': 60.0,
 'VESSEL_LARGE_COMPONENT_MIN_EXTENT_MM': 10.0,
 'VESSEL_LUMEN_DILATION_MM': 1.0,
 'VESSEL_LUMEN_MIN_SCALED_ENHANCEMENT': 0.15,
 'VESSEL_HESSIAN_LINE_BETA': 0.35,
 'VESSEL_HESSIAN_STRENGTH_PERCENTILE': 90.0,
 'VESSEL_MIN_PCA_ELONGATION': 1.25,
 'VESSEL_MIN_SKELETON_LENGTH_MM': 8.0,
 'VESSEL_MIN_LENGTH_DIAMETER_RATIO': 2.0,
 'VESSEL_LARGE_TREE_SKELETON_MM': 15.0,
 'VESSEL_REMOVE_ALL_PHASES': True,
 'VESSEL_INPAINT_SIGMA_MM': 2.5,
 'VESSEL_REMOVAL_STRENGTH': 1.0,
 'SAVE_VESSEL_QC': True,
 'SAVE_VESSEL_QC_NIFTI': True,
 'SAVE_VESSEL_QC_GRID': True,
 'VESSEL_QC_GRID_MAX_SLICES': 6,
 'LOCK_DIR_NAME': 'locks',
 'SCRATCH_DIR_NAME': 'scratch',
 'PIPELINE_VERSION': '2026-10-29-training-deployment-2.5.12-parity'}

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


MASK_RECURSIVE = True
BREAST_MASK_PNG_ALPHA = 0.30
BREAST_MASK_PNG_DPI = 180


import contextlib
import copy
import dataclasses
import gzip
import hashlib
import io
import json
import logging
import math
import multiprocessing
import os
import re
import shutil
import sys
import time
import types

LOG = logging.getLogger('dicom_training')


from preprocessing import conversion, rename_series
from preprocessing import split_timepoints as STAGE3, select_frames as STAGE4
from preprocessing import mri_preprocessing as STAGE5, breast_roi as STAGE6
_ORIGINAL_TTC_FROM_JSON = STAGE4.extract_ttc_from_json


def selection_ttc_from_json(json_path):
    if USE_SPLIT_FRAME_TIMES:
        try:
            with open(json_path) as stream:
                metadata = json.load(stream)
            index = metadata.get('TimePointIndex')
            value = metadata.get('FrameTimeStart')
            if type(index) is int and index >= 0 and type(value) in (int, float) and math.isfinite(value) and value >= 0:
                return float(value)
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    return _ORIGINAL_TTC_FROM_JSON(json_path)


def configure_stages(settings=None):

    if settings:
        allowed = {'DICOM_INPUT_DIR', 'CONVERTED_DIR', 'SELECTED_DCE_DIR', 'PREPROCESSED_DIR',
                   'BREAST_MASK_DIR', 'LOG_DIR', 'RUN_STAGES', 'CONFIRM_FOLDER_RENAME',
                   'DCM2NIIX_EXECUTABLE', 'DCM2NIIX_OPTIONS', 'MIN_DICOM_FILES', 'DICOM_SERIES_DEPTH',
                   'KEEP_ORIGINAL_4D', 'SPLIT_DRY_RUN', 'SPLIT_RECURSIVE', 'PROCESS_PREFIXES',
                   'MAX_PATIENTS', 'SELECTION_VERBOSE', 'ENABLE_NIFTI_HANDOFF', 'USE_SPLIT_FRAME_TIMES',
                   'PREFIX_CONFIG', 'PREPROCESSING_SETTINGS', 'BREAST_ROI_SETTINGS', 'MASK_RECURSIVE',
                   'BREAST_MASK_PNG_ALPHA', 'BREAST_MASK_PNG_DPI'}
        unknown = set(settings)-allowed
        if unknown:
            raise ValueError(f'Unknown combined settings: {sorted(unknown)}')
        globals().update(copy.deepcopy(settings))
    STAGE3.BASE_DIR = str(Path(CONVERTED_DIR).expanduser())
    STAGE3.KEEP_ORIGINAL = KEEP_ORIGINAL_4D
    STAGE3.DRY_RUN = SPLIT_DRY_RUN
    STAGE3.RECURSIVE = SPLIT_RECURSIVE
    STAGE4.PREFIX_CONFIG = copy.deepcopy(PREFIX_CONFIG)
    STAGE4.PROCESS_PREFIXES = list(PROCESS_PREFIXES)
    STAGE4.extract_ttc_from_json = selection_ttc_from_json
    for key, value in PREPROCESSING_SETTINGS.items():
        if not key.isupper() or not hasattr(STAGE5, key):
            raise ValueError(f'Unknown preprocessing setting: {key}')
        setattr(STAGE5, key, copy.deepcopy(value))
    STAGE5.DATASET_ROOT = Path(SELECTED_DCE_DIR).expanduser()
    STAGE5.OUTPUT_ROOT = Path(PREPROCESSED_DIR).expanduser()
    STAGE5.WORK_ROOT = STAGE5.OUTPUT_ROOT.parent/f'.{STAGE5.OUTPUT_ROOT.name}_work'
    STAGE5.STATE_ROOT = STAGE5.OUTPUT_ROOT.parent/f'.{STAGE5.OUTPUT_ROOT.name}_state'
    STAGE5.VESSEL_QC_ROOT = STAGE5.OUTPUT_ROOT.parent/f'{STAGE5.OUTPUT_ROOT.name}_QC'
    STAGE5.BREAST_ROI_SETTINGS = copy.deepcopy(BREAST_ROI_SETTINGS)
    STAGE6.INPUT_FOLDER = STAGE5.OUTPUT_ROOT
    STAGE6.OUTPUT_FOLDER = Path(BREAST_MASK_DIR).expanduser()
    STAGE6.RAW_REFERENCE_FOLDER = STAGE5.DATASET_ROOT
    STAGE6.REGISTRATION_BASELINE_PHASE = STAGE5.REGISTRATION_BASELINE_PHASE if STAGE5.DO_REGISTRATION else 0
    STAGE6.RECURSIVE = MASK_RECURSIVE
    STAGE6.BREAST_MASK_PNG_ALPHA = BREAST_MASK_PNG_ALPHA
    STAGE6.BREAST_MASK_PNG_DPI = BREAST_MASK_PNG_DPI
    STAGE6.ALLOW_MISSING_ORIENTATION = STAGE5.ALLOW_MISSING_ORIENTATION
    STAGE6.ASSUME_MM_IF_UNITS_UNKNOWN = STAGE5.ASSUME_MM_IF_UNITS_UNKNOWN
    roi_fields = dataclasses.fields(STAGE6.Config)
    unknown = set(BREAST_ROI_SETTINGS)-{field.name for field in roi_fields}
    if unknown:
        raise ValueError(f'Unknown breast ROI settings: {sorted(unknown)}')


    STAGE6.Config.__init__.__defaults__ = tuple(BREAST_ROI_SETTINGS.get(field.name, field.default) for field in roi_fields)


def prepare_selection_inputs(input_dir=None):

    root = Path(CONVERTED_DIR if input_dir is None else input_dir)
    if not ENABLE_NIFTI_HANDOFF:
        return {'compressed': 0, 'existing': 0}
    compressed = existing = 0
    for patient in sorted(root.iterdir()):
        if not patient.is_dir() or not patient.name or patient.name[0] not in PROCESS_PREFIXES:
            continue
        for source in sorted(patient.iterdir()):
            if not source.is_file() or not source.name.endswith('.nii'):
                continue
            prefix = patient.name[0]
            if prefix == '1':
                eligible = bool(re.search(r'_t\d{2}\.nii$', source.name))
            elif prefix == '3':
                metadata = source.with_suffix('.json')
                eligible = ('twist_dyn' in source.name.lower() and '_SUB' not in source.name
                            and (STAGE4.extract_ttc_from_filename(source.name) is not None
                                 or (metadata.is_file() and selection_ttc_from_json(metadata) is not None)))
            else:
                eligible = False
            if not eligible:
                continue
            destination = source.with_name(source.name+'.gz')
            if destination.exists():

                with source.open('rb') as plain, gzip.open(destination,'rb') as zipped:
                    while True:
                        left, right = plain.read(1024*1024), zipped.read(1024*1024)
                        if left != right:
                            raise RuntimeError(f'Conflicting uncompressed/compressed inputs: {source} and {destination}')
                        if not left:
                            break
                existing += 1
                continue
            temporary = destination.with_name('.'+destination.name+'.tmp')
            try:
                with source.open('rb') as stream, temporary.open('wb') as target:
                    with gzip.GzipFile(filename='',mode='wb',fileobj=target,mtime=0) as zipped:
                        shutil.copyfileobj(stream,zipped)
                temporary.replace(destination)
                shutil.copystat(source,destination)
            finally:
                temporary.unlink(missing_ok=True)
            compressed += 1
            LOG.info('Lossless selection handoff: %s -> %s (original retained)',source.name,destination.name)
    return {'compressed': compressed, 'existing': existing}


class _StageOutput(io.TextIOBase):
    def __init__(self, stage, level):
        self.stage, self.level, self.pending = stage, level, ''

    def write(self, text):
        self.pending += text
        while '\n' in self.pending:
            line,self.pending = self.pending.split('\n',1)
            if line.strip():
                LOG.log(self.level,'[Stage %s] %s',self.stage,line.rstrip())
        return len(text)

    def flush(self):
        if self.pending.strip():
            LOG.log(self.level,'[Stage %s] %s',self.stage,self.pending.rstrip())
        self.pending = ''


def _run_logged(stage, label, function):
    started = time.perf_counter()
    LOG.info('START %s/6 | %s',stage,label)
    output,error = _StageOutput(stage,logging.INFO),_StageOutput(stage,logging.ERROR)
    try:
        with contextlib.redirect_stdout(output),contextlib.redirect_stderr(error):
            result = function()
    finally:
        output.flush();error.flush()
        LOG.info('END %s/6 | %s | %.2f seconds',stage,label,time.perf_counter()-started)
    return result


def main():
    stages = tuple(RUN_STAGES)
    if not stages or stages != tuple(sorted(set(stages))) or any(type(n) is not int or n not in range(1,7) for n in stages):
        raise ValueError('RUN_STAGES must be a nonempty increasing subset of (1,2,3,4,5,6)')
    configure_stages()
    paths = [Path(p).expanduser().resolve() for p in (DICOM_INPUT_DIR,CONVERTED_DIR,SELECTED_DCE_DIR,PREPROCESSED_DIR,BREAST_MASK_DIR)]
    if len(set(paths)) != len(paths):
        raise ValueError('DICOM, converted, selected, preprocessed and mask directories must be different')
    directory = Path(LOG_DIR).expanduser().resolve();directory.mkdir(parents=True,exist_ok=True)
    log_path = directory/f'pipeline_{time.strftime("%Y%m%d_%H%M%S")}.log'
    LOG.setLevel(logging.INFO);LOG.propagate=False
    for handler in list(LOG.handlers):
        handler.close();LOG.removeHandler(handler)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    for handler in (logging.StreamHandler(sys.stdout),logging.FileHandler(log_path,encoding='utf-8')):
        handler.setFormatter(formatter);LOG.addHandler(handler)
    LOG.info('DICOM-to-training pipeline | stages %s | log %s',stages,log_path)
    for label,path in zip(('DICOM input','Converted / renamed','Selected four DCE','Preprocessed MRI','Breast ROI masks'),paths):
        LOG.info('%s: %s',label,path)
    started = time.perf_counter();completed=[];failures=0
    try:
        for stage in stages:
            if stage == 1:
                result = _run_logged(1,'DICOM conversion',lambda: conversion.run_dicom_conversion(
                    DICOM_INPUT_DIR,CONVERTED_DIR,directory,dcm2niix_executable=DCM2NIIX_EXECUTABLE,
                    dcm2niix_options=DCM2NIIX_OPTIONS,min_files_threshold=MIN_DICOM_FILES,
                    series_depth=DICOM_SERIES_DEPTH,logger=LOG))
                failures += result['failed']
                if not result['converted']:
                    raise RuntimeError('No DICOM series converted; check the conversion log, series depth and minimum file count')
            elif stage == 2:
                result = _run_logged(2,'Patient folder renaming',lambda: rename_series.run_patient_folder_rename(
                    CONVERTED_DIR,directory,confirm_rename=CONFIRM_FOLDER_RENAME,logger=LOG))
                if result['cancelled']:
                    LOG.info('Pipeline stopped after the original rename confirmation was declined')
                    return 0
                failures += result['failed']
            elif stage == 3:
                _run_logged(3,'Split DCE time points',STAGE3.main)
            elif stage == 4:
                _run_logged(4,'Lossless input handoff',prepare_selection_inputs)
                successful,failed = _run_logged(4,'Select four DCE frames',lambda: STAGE4.batch_process(
                    str(CONVERTED_DIR),str(SELECTED_DCE_DIR),MAX_PATIENTS,SELECTION_VERBOSE))
                failures += len(failed)
                if not successful:
                    raise RuntimeError('No patient has four selected DCE frames; see the stage-4 log')
            elif stage == 5:
                status = _run_logged(5,'MRI preprocessing',STAGE5.main)
                if status:
                    raise RuntimeError(f'Preprocessing reported failures (exit {status}); see preceding patient messages')
            else:
                status = _run_logged(6,'Breast ROI masks and previews',STAGE6.main)
                if status:
                    raise RuntimeError(f'Breast ROI export reported failures (exit {status})')
            completed.append(stage)
            if stage == 3 and SPLIT_DRY_RUN:
                LOG.info('Stage 3 was a dry run; subsequent stages were not started')
                break
    except Exception:
        LOG.exception('Pipeline failed after completed stages %s',completed)
        return 1
    finally:
        LOG.info('TOTAL %.2f seconds | completed stages %s | reported conversion/selection failures %s',
                 time.perf_counter()-started,completed,failures)
    LOG.info('Finished. Final MRI: %s | Breast ROI masks: %s',PREPROCESSED_DIR,BREAST_MASK_DIR)
    return 1 if failures else 0


configure_stages()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    raise SystemExit(main())
