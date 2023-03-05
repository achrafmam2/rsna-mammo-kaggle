from absl import logging
import concurrent.futures
import numpy as np
import pandas as pd
import more_itertools as mit
import epath
import dataclasses
import functools
import pydicom
import cv2

from typing import Any, Tuple, Iterable, Optional

from rsna_mammo_dataset import dcm_imaging


@dataclasses.dataclass(frozen=True)
class Options:
  """
  Attributes:
    - metadata_path: Path to the csv file that contains metadata about the patients.
    - split_path: An optional path to a csv file that contains which patients should be included.
    - images_dir: Path to the root directory that contains the patient images.
    - output_shape: The shape to resize the images to.
    - allowed_views: A sequence of views to generate examples for (e.g., 'CC', 'MLO').
    - labels_exist: Indicates whether labels exist or not.
  """
  metadata_path: epath.Path
  split_path: Optional[epath.Path]
  images_dir: epath.Path
  shape: Tuple[int, int]
  allowed_views: Tuple[str, ...]
  labels_exist: bool


@dataclasses.dataclass(frozen=True)
class Scan:
  scan_id: str
  patient_id: int
  image: np.ndarray
  laterality: str
  view: str
  cancer: Optional[bool]


def load_scans(opts: Options) -> Iterable[Scan]:
  """Load scans."""
  df = pd.read_csv(opts.metadata_path)
  df = df[df.view.isin(opts.allowed_views)]

  if opts.split_path:
    split_df = pd.read_csv(opts.split_path)
    df = df.join(split_df.set_index('patient_id'), on='patient_id', how='inner')

  inputs = df.to_dict(orient='records')
  with concurrent.futures.ThreadPoolExecutor() as executor:
    load_fn = functools.partial(
        _load_scan,
        images_dir=opts.images_dir,
        shape=opts.shape,
        labels_exist=opts.labels_exist,
    )
    for scan in executor.map(load_fn, inputs):
      yield scan


def _load_scan(
    inpt: dict[str, Any],
    images_dir: epath.Path,
    shape: Tuple[int, int],
    labels_exist: bool,
) -> Scan:
  scan_id = f'{inpt["patient_id"]}/{inpt["image_id"]}'
  relpath = images_dir / f'{scan_id}.dcm'

  cancer = None
  if labels_exist:
    cancer = bool(inpt['cancer'])

  return Scan(scan_id=scan_id,
              patient_id=inpt['patient_id'],
              image=_load_image(relpath, shape),
              laterality=inpt['laterality'],
              view=inpt['view'],
              cancer=cancer)


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
    img = img[10:-10, 10:-10]  # Hack.

    mask = (img > 1).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    return img[y:y + h, x:x + w]

  transforms = [
      pydicom.pixel_data_handlers.apply_voi_lut,
      nodcm(dcm_imaging.normalize),
      dcm_imaging.tomonochrome2,
      nodcm(touint8),
      nodcm(fitimg),
      nodcm(functools.partial(dcm_imaging.resize, shape=output_shape)),
  ]

  return dcm_imaging.dcmreadimg(p, tfns=transforms)
