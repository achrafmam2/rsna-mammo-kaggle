from absl import logging
import numpy as np
import pandas as pd
import more_itertools as mit
import epath
import dataclasses
import functools
import multiprocessing as mp
import pydicom
import cv2

from typing import Tuple, Iterable

from rsna_mammo_dataset import dcm_imaging


@dataclasses.dataclass(frozen=True)
class Options:
  """
  Attributes:
    - metadata_path: path to the csv file that contains metadata about the patients.
    - split_path: path to the csv file that contains the patient in the split.
    - images_dir: path to the root directory that contains the patient images.
    - output_shape: the shape to resize the images to.
    - allowed_views: a sequence of views to generate examples for (e.g., 'CC', 'MLO').
  """
  metadata_path: epath.Path
  split_path: epath.Path
  images_dir: epath.Path
  shape: Tuple[int, int]
  allowed_views: Tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Scan:
  scan_id: str
  patient_id: int
  image: np.ndarray
  laterality: str
  view: str
  cancer: bool


def load_scans(opts: Options) -> Iterable[Scan]:
  """Load scans."""
  metadata_df = pd.read_csv(opts.metadata_path)
  metadata_df = metadata_df[metadata_df.view.isin(opts.allowed_views)]

  split_df = pd.read_csv(opts.split_path)
  df = metadata_df.join(split_df.set_index('patient_id'), on='patient_id', how='inner')
  
  inputs = df.to_dict(orient='records')
  with mp.Pool() as p:
    load_fn = functools.partial(
        _load_scan,
        images_dir=opts.images_dir,
        shape=opts.shape,
    )
    for scan in p.imap(load_fn, inputs):
      yield scan


def _load_scan(inpt, images_dir: epath.Path, shape: Tuple[int, int]) -> Scan:
  scan_id = f'{inpt["patient_id"]}/{inpt["image_id"]}'
  relpath = images_dir / f'{scan_id}.dcm'
  return Scan(
      scan_id=scan_id,
      patient_id=inpt['patient_id'],
      image=_load_image(relpath, shape),
      laterality=inpt['laterality'],
      view=inpt['view'],
      cancer=bool(inpt['cancer']))


def _load_image(
    p: epath.Path,
    output_shape: Tuple[int, int],
) -> np.ndarray:
  """Reads and process the breast image stored in the dicom located by the input path `p`.

  The image is cropped to highlight the breast.

  Note:
    This function asssumes that only one breast exists per image.

  Returns: 
    A 2D numpy array.
  """
  def nodcm(fn):
    return lambda img, ignore_dcm: fn(img)

  def touint8(img: np.ndarray) -> np.ndarray:
    return (img * 255).astype(np.uint8)

  def fitimg(img: np.ndarray) -> np.ndarray:
    img = img[10:-10, 10:-10] # Hack.

    mask = (img > 1).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    c = max(contours, key = cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    return img[y:y + h, x:x + w]
  
  return dcm_imaging.dcmreadimg(
    p,
    tfns = [
        pydicom.pixel_data_handlers.apply_voi_lut,
        nodcm(dcm_imaging.normalize), 
        dcm_imaging.tomonochrome2, 
        nodcm(touint8),
        nodcm(fitimg),
        nodcm(functools.partial(
            dcm_imaging.resize,
            shape=output_shape)),
    ])
