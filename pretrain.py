"""Author: Neel Kanwal <neel.kanwal0@gmail.com>."""

import json
import shutil
import subprocess

from study import RAW, SOURCE_DATASET, SOURCE_FOLD, configure_environment

DATASET_ID = int(SOURCE_DATASET[7:10])
CONFIGURATION = "3d_fullres"


def main():
    configure_environment()
    folder = RAW / SOURCE_DATASET
    cases = sorted((folder / "imagesTr").glob("*_0000.nii.gz"))
    if not cases:
        raise ValueError(f"Prepare four-channel imagesTr and binary labelsTr in {folder}")
    metadata = dict(channel_names={str(i): f"DCE phase {i}" for i in range(4)},
                    labels={"background": 0, "lesion": 1}, numTraining=len(cases),
                    file_ending=".nii.gz", overwrite_image_reader_writer="SimpleITKIO")
    metadata_file = folder / "dataset.json"
    if metadata_file.exists() and json.loads(metadata_file.read_text()) != metadata:
        raise ValueError("Existing dataset.json differs from the expected four-channel dataset")
    metadata_file.write_text(json.dumps(metadata, indent=2))
    for name, arguments in [
        ("nnUNetv2_plan_and_preprocess", ["-d", str(DATASET_ID), "-c", CONFIGURATION, "--verify_dataset_integrity"]),
        ("nnUNetv2_train", [str(DATASET_ID), CONFIGURATION, str(SOURCE_FOLD)]),
    ]:
        executable = shutil.which(name)
        if not executable:
            raise RuntimeError(f"Missing {name}. Activate the repository environment.")
        subprocess.run([executable, *arguments], check=True)


if __name__ == "__main__":
    main()
