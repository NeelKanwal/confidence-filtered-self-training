"""Author: Neel Kanwal <neel.kanwal0@gmail.com>."""

from __future__ import annotations

import numpy as np
import nibabel as nib
from .image_geometry import from_sitk, to_sitk


def resample_stack(arrays, affine, opts):

    if not opts["resample"]:
        return np.stack(arrays),affine.copy()
    import SimpleITK as sitk
    spacing=nib.affines.voxel_sizes(affine)
    target=np.asarray(opts["target_spacing_mm"],float)
    size=np.maximum(2,np.round(np.asarray(arrays[0].shape)*spacing/target).astype(int))
    def resample_phase(array):
        image=to_sitk(array,affine)
        out=sitk.Resample(image,[int(v) for v in size],sitk.Transform(),
            sitk.sitkBSpline if opts["interpolation_order"]==3 else sitk.sitkLinear,
            image.GetOrigin(),tuple(target),image.GetDirection(),0.,sitk.sitkFloat32)
        return from_sitk(out)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(opts["phase_workers"],len(arrays))) as pool:
        result=list(pool.map(resample_phase,arrays))
    model_affine=affine.copy()
    model_affine[:3,:3]=affine[:3,:3]/spacing*target
    return np.stack(result).astype(np.float32),model_affine
