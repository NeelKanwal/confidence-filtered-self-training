"""Author: Neel Kanwal <neel.kanwal0@gmail.com>."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk


def to_sitk(data, affine):
    import SimpleITK as sitk
    lps = np.diag([-1., -1., 1.]) @ affine[:3, :3]
    spacing = np.linalg.norm(lps, axis=0)
    direction = lps / spacing
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-4):
        raise ValueError("N4/registration requires an orthogonal NIfTI grid (no shear)")
    image = sitk.GetImageFromArray(np.asarray(data, dtype=np.float32).transpose(2, 1, 0))
    image.SetSpacing(tuple(spacing))
    image.SetOrigin(tuple(np.diag([-1., -1., 1.]) @ affine[:3, 3]))
    image.SetDirection(tuple(direction.ravel()))
    return image


def from_sitk(image):
    import SimpleITK as sitk
    return sitk.GetArrayFromImage(image).transpose(2, 1, 0).astype(np.float32)


def ants_to_sitk(img_ants) -> sitk.Image:
    arr = np.transpose(img_ants.numpy(), (2, 1, 0))
    out = sitk.GetImageFromArray(arr)
    out.SetSpacing(tuple(img_ants.spacing))
    out.SetOrigin(tuple(img_ants.origin))
    out.SetDirection(tuple(np.ravel(img_ants.direction)))
    return sitk.Cast(out, sitk.sitkFloat32)


def sitk_to_ants(img: sitk.Image):
    import ants
    arr = np.transpose(sitk.GetArrayFromImage(img), (2, 1, 0))
    return ants.from_numpy(
        arr,
        spacing=img.GetSpacing(),
        origin=img.GetOrigin(),
        direction=np.asarray(img.GetDirection()).reshape(3, 3),
    )
