"""Splits a set of patients into train and validation."""

import os

from absl import app
from absl import flags
from absl import logging
from etils import epath

import pandas as pd

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'metadata_path', None,
    'Path to the metadata csv file that contains all patients.')
flags.mark_flag_as_required('metadata_path')

flags.DEFINE_float(
    'validation_frac', None,
    'Fraction of the patients to be allocated to the validation split.')
flags.register_validator('validation_frac',
                         lambda value: 0 < value < 1,
                         message='--eval_frac must be in the range (0, 1).')

flags.DEFINE_integer('seed', 0, 'Seed for the random number generator.')

flags.DEFINE_string('output_dir', None, 'Directory where to write the splits.')
flags.mark_flag_as_required('output_dir')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.info('Reading metadata file at %s', FLAGS.metadata_path)

  patients = pd.read_csv(FLAGS.metadata_path)

  logging.info('Finished reading file %s', FLAGS.metadata_path)
  logging.info('Splitting patients based on the "cancer" attribute')

  patients = patients.groupby('patient_id').agg(
      cancer=pd.NamedAgg(column='cancer', aggfunc='max'))

  validation_split = patients.groupby('cancer').sample(
      frac=FLAGS.validation_frac, random_state=FLAGS.seed)
  train_split = patients.drop(validation_split.index)

  for name, split in zip(['train', 'validation'],
                         [train_split, validation_split]):
    output_path = epath.Path(FLAGS.output_dir) / f'{name}.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(output_path, columns=[], index=True)
    logging.info('Wrote "%s" split to %s', name, os.fspath(output_path))


if __name__ == '__main__':
  app.run(main)
