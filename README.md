# Breast MRI lesion segmentation with confidence-filtered self-training

**arXiv preprint:** link pending. <!-- Replace “link pending” with the arXiv URL when available. -->

## Abstract

Deep learning (DL) models for lesion segmentation in breast dynamic contrast-enhanced magnetic resonance imaging (DCE-MRI) are typically developed on curated multi-centre data and can lose accuracy when applied at hospitals whose scanners and protocols differ from those represented in the training data. Motion and differences in image quality complicate transfer, while obtaining expert annotations for local retraining is time-consuming. We develop a preprocessing pipeline and propose an unsupervised domain adaptation method based on confidence-filtered self-training. First, we use a multi-step preprocessing pipeline to correct intensity inhomogeneity, co-register DCE-MRI sequences, standardize voxel spacing, and normalize image intensity. Next, we use an nnU-Net pretrained on the public MAMA-MIA dataset for binary lesion segmentation. We apply it to unlabelled DCE-MRI examinations from two hospitals and threshold the predicted lesion probabilities to generate binary pseudo-labels. Finally, we use these binary pseudo-labels to fine-tune three models, two trained separately on data from the respective hospitals and a third trained on pooled data from both. We combine their predictions by mean-logit fusion. We retain the source architecture and normalization and evaluate segmentation against expert lesion masks. In high-motion examinations, mean Dice was 43.2% higher after combined registration, resampling, and normalization. The self-trained ensemble achieved a mean Dice of 0.4382 compared with 0.3637 for the source baseline and improved overlap in 18 of 20 examinations. The proposed method enables local fine-tuning without new expert training annotations, with effects that vary by hospital.

[![Breast MRI lesion segmentation workflow](docs/Figure_1.png)](docs/Figure_1.pdf)

*Figure 1. Breast MRI lesion segmentation at inference.*

## Where to start

| What you have | Where to go after installation |
|---|---|
| DICOM series or four selected NIfTI phases | [Prepare the four channels](#1-prepare-four-dce-channels) |
| Labelled examinations for training a source model | [Pretrain the source model](#2-pretrain-the-source-model) |
| A trained source model and unlabelled examinations | [Self-train the hospital models](#3-self-train-m1-m2-and-m3) |
| Prepared images and compatible trained weights | [Run single-model inference](#single-model-example) or [run the ensemble](#three-model-ensemble) |

## Installation

1. Clone the repository and enter its root directory.
   ~~~bash
   git clone https://github.com/NeelKanwal/confidence-filtered-self-training.git
   cd confidence-filtered-self-training
   ~~~
2. On Linux, create a Python 3.12 environment and install the pinned dependencies. If you plan to use a GPU, use a PyTorch build compatible with your driver.
   ~~~bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   python -c "import ants, nibabel, SimpleITK, torch; print(torch.__version__, torch.cuda.is_available())"
   ~~~
3. For DICOM input, install the system utilities `file` and `dcm2niix`. The optional `dcmdump` utility from DCMTK reads series descriptions. Work with de-identified copies because conversion uses DICOM fields in filenames.

Run the scripts from the repository root. The default working directory is `work/`. To use another drive, edit `WORK` in [study.py](study.py), for example `WORK = Path("/data/breast_mri")`. The modelling scripts use this setting. [preprocess.py](preprocess.py) has its own input and output paths.

## 1. Prepare four DCE channels

~~~bash
python preprocess.py
~~~

Set DICOM_INPUT_DIR, CONVERTED_DIR, SELECTED_DCE_DIR, PREPROCESSED_DIR, and BREAST_MASK_DIR in preprocess.py. RUN_STAGES selects the stages to execute in order. For example, use (5, 6) when starting with four selected NIfTI volumes per case. This avoids the original acquisition-specific DICOM routing. Input names must end in _0000 through _0003 with a .nii or .nii.gz extension. The modelling scripts use .nii.gz. Channel 0 is pre-contrast and channels 1–3 are three selected post-contrast acquisitions in chronological order.

| Stage | Source in preprocessing/ | Operation |
|---|---|---|
| 1 | conversion.py | Convert each eligible DICOM series with dcm2niix. Configure the expected directory depth and minimum file count for your export. |
| 2 | rename_series.py | Group the converted series under a common examination folder. Run against the converted working copy. |
| 3 | split_timepoints.py | Split supported collated 4D DCE sequences into 3D time points and update the timing sidecars. The default retains the original 4D input. |
| 4 | select_frames.py | Select one pre-contrast and three post-contrast phases and write four numbered NIfTI channels. |
| 5 | mri_preprocessing.py | Apply shared N4 correction, estimate the breast ROI, register the first three phases to the fourth, resample to 1 mm isotropic spacing and normalize the stack. |
| 6 | breast_roi.py | Write an examination-level binary breast ROI on the processed image grid. The original fourth volume provides the anatomical reference. |

Stages 3 and 4 retain the study's two acquisition naming conventions. Check PATIENT_DIR_PATTERN, DCE_PATTERN, PROCESS_PREFIXES and PREFIX_CONFIG before trying another scanner export. One rule selects zero-based indices [0, 2, 5, 7] when all four are available. The other selects frames according to the available count. Inspect the selected frame times for your acquisition. The digits used in filenames are routing conventions and should not be used to infer hospital membership.

For independent use, the shared N4 algorithm is in bias_correction.py, ANTs registration in registration.py, spacing conversion in resampling.py, and stack-wide intensity normalization in normalization.py. The standalone ROI exporter also runs as `python -m preprocessing.breast_roi` after setting its input, output and raw reference paths. For other individual stages, set RUN_STAGES to a one-element tuple, for example (4,), and use preprocess.py so the stage receives the same settings as a full run.

## 2. Pretrain the source model

Prepare a local nnU-Net dataset of labelled breast examinations. The following paths show the naming convention with a fictional case. Keep all examinations from the same patient in the same partition.

~~~text
work/nnUNet_raw/Dataset501_BreastDCE/
  imagesTr/example_0000.nii.gz
  imagesTr/example_0001.nii.gz
  imagesTr/example_0002.nii.gz
  imagesTr/example_0003.nii.gz
  labelsTr/example.nii.gz
~~~

The label is a 3D integer image on the channel grid, with 0 for background and 1 for lesion. Preprocess expert masks onto that grid with nearest-neighbour interpolation if necessary. Obtain any source data under the dataset provider's own access conditions.

~~~bash
python pretrain.py
~~~

The entry point writes dataset.json, checks dataset integrity, plans the full-resolution 3D configuration and runs the configured fold. By default the nnU-Net trainer uses supervised Dice and cross-entropy. Dataset planning derives patch size and batch size from the local data and available training configuration. Archive the resulting plans.json, dataset.json, splits_final.json, environment and training logs alongside your own experiment. Use study.py to point subsequent stages to the trained source model and fold. The inference code expects a two-class softmax model with SimpleITKIO.

## 3. Self-train M1, M2 and M3

1. For each model, place prepared four-channel examinations in work/adaptation/M1/train/ and work/adaptation/M1/validation/.
2. Supply their binary breast masks in work/adaptation/M1/roi/ as example_mask.nii.gz. Use the same layout for M2 and M3.
3. Assign M1 the first site's training and validation subsets, M2 the second site's, and M3 the pooled subsets. Preserve patient-level separation among training, validation and evaluation, including when pooling sites.

~~~bash
python self_train.py
~~~

Set `MODEL_NAME` in `self_train.py` before each run. Each run starts from the same configured source weights. It generates fixed binary targets from lesion probabilities of at least 0.70 within the breast ROI after two iterations of dilation. No expert evaluation mask is used in this step.

The source nnU-Net preprocessor transforms channels and pseudo-labels together. The adaptation loop samples one training patch per examination per epoch and uses fixed central patches for validation. The source plan determines the patch shape. This implementation uses AdamW, a five-epoch warm-up to 1e-5, cosine decay and early stopping. Record the seed and training membership for each run.

The selected weights and plans are saved locally in work/models/M1, M2 or M3. Each output directory must be new.

## 4. Predict with one model or the ensemble

### Single-model example

1. Supply your own trained two-class nnU-Net model using SimpleITKIO. The example below uses M1 and one fictional examination. Place the four **prepared** images and the model files under the working directory as shown. Channel 0000 is pre-contrast, followed by three post-contrast phases in time order.

   ~~~text
   work/
   ├── evaluation/images/
   │   ├── example_0000.nii.gz
   │   ├── example_0001.nii.gz
   │   ├── example_0002.nii.gz
   │   └── example_0003.nii.gz
   └── models/M1/
       ├── dataset.json
       ├── plans.json
       └── fold_0/checkpoint_best.pth
   ~~~

2. Edit the settings at the top of [inference.py](inference.py). Point `MODEL` to the directory containing the two JSON files. The script uses `FOLD` and `WEIGHTS` to locate the weights inside it.

   ~~~python
   INPUT = WORK / "evaluation" / "images"
   MODEL = WORK / "models" / "M1"
   FOLD = 0
   WEIGHTS = "checkpoint_best.pth"
   OUTPUT = WORK / "probabilities" / "M1"
   ~~~

3. Run `python inference.py`. Open `work/probabilities/M1/example.nii.gz` to view the lesion probability at each voxel on the prepared image grid. For source-baseline inference, use the original defaults in `inference.py` and set the source model location in `study.py`.

### Three-model ensemble

1. Use the same four prepared images. Place all three model directories under `work/models/` as `M1/`, `M2/` and `M3/`, each with the model layout shown above.
2. Add the binary breast ROI at `work/evaluation/roi/example_mask.nii.gz` on the prepared image grid. Place the original fourth-phase volume at `work/evaluation/native/example.nii.gz`. The native and prepared volumes must share a physical coordinate system.
3. Run `python ensemble.py`. If your folders differ, edit `INPUT`, `ROI`, `NATIVE`, `MODEL_DIRS` and `OUTPUT` in [ensemble.py](ensemble.py).
4. Open the outputs in `work/ensemble/`. The file `example_prob.nii.gz` contains fused lesion probabilities on the prepared grid. Overlay `example_mask.nii.gz` on the native fourth-phase volume to inspect the binary segmentation.

The ensemble averages lesion log-odds across models with equal weights and thresholds the fused probability at 0.50. It retains six-connected components of at least 8 mm³ with at least 1% overlap with the dilated breast ROI. After mapping the mask to the native grid with nearest-neighbour interpolation, it reapplies the volume cutoff.

## Results reported in the manuscript

The comparison included 20 examinations, ten from each hospital. These are manuscript results. The public code does not include the original hospital images, annotations, study splits or trained weights. Running the reference training workflow on other data produces a new experiment.

| Model | Overall mean Dice | Hospital 1 | Hospital 2 |
|---|---:|---:|---:|
| Source baseline | 0.3637 | 0.3835 | 0.3440 |
| M1 | 0.4190 | 0.4626 | 0.3754 |
| M2 | 0.3667 | 0.3922 | 0.3412 |
| M3 | 0.4128 | 0.4496 | 0.3760 |
| Equal-weight ensemble | 0.4382 | 0.4627 | 0.4137 |

The archived models behind the headline ensemble score used Dice and cross-entropy. The fixed-target focal Tversky workflow in self_train.py represents a separate training configuration. The training provenance available for this release does not establish that running self_train.py reproduces the headline model weights.

## Citation

> Neel Kanwal, Matthan Caan and Oliver Marcel Geier. *Self-Training with Confidence-Filtered Pseudo-Labels for Domain-Adapted Breast MRI Lesion Segmentation*. Unpublished manuscript.

Publication details and DOI can be added here when available.

Contact: Neel Kanwal (neel.kanwal0@gmail.com).
