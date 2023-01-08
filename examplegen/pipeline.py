from absl import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import more_itertools as mit
import pathlib
import dataclasses
import functools
import joblib
import multiprocessing
import os
import pydicom
import dcm_imaging
import math
import cv2

from typing import Sequence


@dataclasses.dataclass(frozen=True)
class Options():
  """
  Attributes:
    - patients_csv_path: path to the csv file that contains metadata about the patients.
    - root_dir: path to the root directory that contains the patient images.
    - output_path: the path to the root directory where the tfrecords should be written.
    - output_shape: the shape to resize the images with.
    - num_examples_per_record: the maximum number of examples in a single output tfrecords.
  """
  patients_csv_path: pathlib.Path
  root_dir: pathlib.Path
  output_path: pathlib.Path
  output_shape: tuple[int, int]
  num_examples_per_record: int


def make_dataset(opts: Options) -> None:
  """Generates examples.
  
  The built examples are bundled within TFRecords and sharded according
  user options.
  
  The TFRecords can be used to build a tf.data.Data object.
  """
  df = pd.read_csv(opts.patients_csv_path)

  logging.info('Standardizing DF.')
  df = _standardize_dataframe(df)
  assert min(df.groupby(['patient_id']).image_id.count()) == max(
      df.groupby(['patient_id']).image_id.count()) == 4

  logging.info('Making patient cases.')
  cases = _make_cases(df)

  logging.info('Make cancer tfrecords.')
  cancer_cases = [c for c in cases if c.cancer]
  _make_tfrecords(
      cancer_cases,
      images_dir=opts.root_dir,
      output_dir=opts.output_path / 'cancer',
      shape=opts.output_shape,
      num_examples_per_record=opts.num_examples_per_record,
  )

  logging.info('Make non cancer tfrecords.')
  noncancer_cases = [c for c in cases if not c.cancer]
  _make_tfrecords(
      noncancer_cases,
      images_dir=opts.root_dir,
      output_dir=opts.output_path / 'noncancer',
      shape=opts.output_shape,
      num_examples_per_record=opts.num_examples_per_record,
  )


def _standardize_dataframe(df):
  """Standardizes the input dataframe.
  
  Only the four standard images are kept.
  
  If the patient has more than one standard image, for example two
  left CC images, then one at random will be chosen.
  
  Returns:
    Pandas dataframe.
  """
  # TODO: Make the the sampling deterministic.
  allowlist = (df.view == 'CC') | (df.view == 'MLO')
  df = df[allowlist]

  grpby_cols = ['patient_id', 'view', 'laterality']
  return df.groupby(grpby_cols).apply(lambda x: x.sample(1)).reset_index(
      drop=True)


@dataclasses.dataclass(frozen=True)
class _Scan():
  laterality: str
  view: str
  relpath: pathlib.Path
  cancer: bool


@dataclasses.dataclass(frozen=True)
class _Case():
  patient_id: int
  scans: tuple[_Scan, ...]

  @property
  def cancer(self) -> bool:
    return any(s.cancer for s in self.scans)


def _make_cases(df) -> Sequence[_Case]:
  jobs = []
  for (patient_id, g) in df.groupby('patient_id'):
    jobs.append(joblib.delayed(_make_case)(patient_id, g))

  return joblib.Parallel(
      n_jobs=multiprocessing.cpu_count(),
      verbose=0,
      backend='multiprocessing',
      prefer='threads',
  )(jobs)


def _make_case(patient_id: int, df) -> _Case:
  scans = []
  for lt in ('L', 'R'):
    for v in ('CC', 'MLO'):
      cond = (df.view == v) & (df.laterality == lt)
      img_id = df.loc[cond].image_id.values.squeeze()
      relpath = pathlib.Path(f'{patient_id}/{img_id}.dcm')
      cancer = bool(df.loc[cond].cancer.values.squeeze())

      scan = _Scan(laterality=lt, view=v, cancer=cancer, relpath=relpath)

      scans.append(scan)

  return _Case(scans=tuple(scans), patient_id=patient_id)


def _make_tfexample(
    case: _Case,
    root_dir: pathlib.Path,
    shape: tuple[int, int],
) -> tf.train.Example:

  def loadscan(relpath: pathlib.Path) -> bytes:
    scan = _loadbreastimg(root_dir / relpath, shape)
    scan = np.expand_dims(scan, 2)
    return tf.io.encode_png(scan, compression=9).numpy()

  images = {
      f'image/{s.laterality}_{s.view}': _bytes_feature(loadscan(s.relpath))
      for s in case.scans
  }

  labels = {
      f'cancer/{s.laterality}': _int64_feature(s.cancer) for s in case.scans
  }

  context = {
      'cancer/case': _int64_feature(case.cancer),
      'patient_id': _int64_feature(case.patient_id),
  }

  return tf.train.Example(features=tf.train.Features(feature={
      **images,
      **labels,
      **context
  }))


def _make_tfrecords(
    cases: Sequence[_Case],
    images_dir: pathlib.Path,
    output_dir: pathlib.Path,
    shape: tuple[int, int],
    num_examples_per_record: int,
) -> None:
  output_dir.mkdir(parents=True, exist_ok=False)

  n = num_examples_per_record
  num_chunks = math.ceil(len(cases) / n)

  for i, chunk in enumerate(mit.chunked(cases, n)):
    logging.info('Making TFRecords %d/%d.', i + 1, num_chunks)

    examples = _load_examples(chunk, images_dir=images_dir, output_shape=shape)

    output_path = output_dir / f'{i}.tfrecords'
    _write_examples(examples, output_path)


def _load_examples(
    cases: Sequence[_Case],
    images_dir: pathlib.Path,
    output_shape: tuple[int, int],
) -> Sequence[tf.train.Example]:
  make = functools.partial(_make_tfexample,
                           root_dir=images_dir,
                           shape=output_shape)

  jobs = [joblib.delayed(make)(c) for c in cases]

  return joblib.Parallel(
      n_jobs=multiprocessing.cpu_count(),
      verbose=0,
      backend='multiprocessing',
      prefer='threads',
  )(jobs)


def _write_examples(
    examples: Sequence[tf.train.Example],
    output_path: pathlib.Path,
) -> None:
  options = tf.io.TFRecordOptions(compression_type='GZIP')
  with tf.io.TFRecordWriter(os.fspath(output_path), options=options) as fw:
    for e in examples:
      fw.write(e.SerializeToString())


def _loadbreastimg(
    p: pathlib.Path,
    output_shape: tuple[int, int],
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

  tfns = [
      pydicom.pixel_data_handlers.apply_voi_lut,
      nodcm(dcm_imaging.normalize),
      dcm_imaging.tomonochrome2,
      nodcm(touint8),
      nodcm(fitimg),
      nodcm(functools.partial(dcm_imaging.resize, shape=output_shape)),
  ]
  return dcm_imaging.dcmreadimg(p, tfns)


def _bytes_feature(x: bytes) -> tf.train.Feature:
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[x]))


def _int64_feature(x: int) -> tf.train.Feature:
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[x]))
