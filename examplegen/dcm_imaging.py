import numpy as np
import cv2
import pathlib
import pydicom
import functools
import inspect

from typing import Tuple, Callable, Sequence, Union

DcmImgTransform = Callable[[np.ndarray, pydicom.Dataset], np.ndarray]


def dcmreadimg(
    path: pathlib.Path,
    tfns: Sequence[DcmImgTransform] = (),
) -> np.ndarray:
  """Reads the image stored in the dicom specified in the input path, and applies the specified transforms on it.
  
  Args:
    - path: path to a dicom.
    - tfns: sequence of transforms that will be applied to image stored in the dicom.
    
  Returns: a 2D image.
  """
  dcm = pydicom.dcmread(path)
  voxels = dcm.pixel_array

  for tfn in tfns:
    voxels = tfn(voxels, dcm)

  return voxels


def rescale(img: np.ndarray, dcm: pydicom.Dataset) -> np.ndarray:
  if ('RescaleSlope' in dcm) and ('RescaleIntercept' in dcm):
    img = dcm.RescaleSlope * img + dcm.RescaleIntercept
  return img


def normalize(img: np.ndarray) -> np.ndarray:
  return (img - img.min()) / (img.max() - img.min())


def tomonochrome2(img: np.ndarray, dcm: pydicom.Dataset):
  if dcm.PhotometricInterpretation == "MONOCHROME1":
    img = invertpixels(img)
  return img


def invertpixels(img: np.ndarray) -> np.ndarray:
  pxmin = img.min()
  img = img - pxmin
  img = img.max() - img
  return img + pxmin


def resize(img: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
  cur_res = np.prod(img.shape)
  target_res = np.prod(shape)

  # TODO: Read about interpolation
  # https://stackoverflow.com/questions/23853632/which-kind-of-interpolation-best-for-resizing-image
  interpolation = cv2.INTER_CUBIC if target_res > cur_res else cv2.INTER_AREA

  return cv2.resize(img, shape, interpolation=interpolation)
