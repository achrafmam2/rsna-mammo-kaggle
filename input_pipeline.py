"""Input pipeline for the breast model."""

import cv2
import dcm_imaging
import functools
import pathlib
import pydicom
import numpy as np

from typing import Tuple


def loadbreastimg(
    p: pathlib.Path,
    output_shape: Tuple[int, int] = (384, 768),
) -> np.ndarray:

  def nodcm(fn):
    return lambda img, ignore_dcm: fn(img)

  def touint8(img: np.ndarray) -> np.ndarray:
    return (img * 255).astype(np.uint8)

  def fitimg(img: np.ndarray) -> np.ndarray:
    img = img[10:-10, 10:-10]  # Hack.

    mask = (img > 1).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    return img[y:y + h, x:x + w]

  tfns = [
      pydicom.pixel_data_handlers.apply_voi_lut,
      nodcm(dcm_imaging.normalize),
      dcm_imaging.tomonochrome2,
      nodcm(touint8),
      nodcm(fitimg),
      nodcm(functools.partial(dcm_imaging.resize, shape=output_shape)),
  ]

  return dcm_imaging.dcmreadimg(p, tfns)
