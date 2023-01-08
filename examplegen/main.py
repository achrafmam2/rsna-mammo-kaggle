"""Main file for generating examples."""

from absl import app
from absl import flags
from absl import logging

import pathlib
import os

import pipeline

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'patients_csv', None,
    'Path to the csv file that contains metadata about the patients.')

flags.DEFINE_string(
    'images_dir', None,
    'Path to the root directory that contains the patient images.')

flags.DEFINE_string('output_path', None,
                    'Path where the tfrecords should be writtend.')

flags.DEFINE_string(
    'shape', None,
    'Shape to resize the images with (e.g., 384x768 width then height).')

flags.DEFINE_integer(
    'num_examples_per_record', None,
    'The maximum number of examples in a single output tfrecords.')


def _parse_shape(s: str) -> tuple[int, int]:
  [w, h] = s.split('x')
  return (int(w), int(h))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Tensorflow is very verbose.
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

  csv_path = pathlib.Path(FLAGS.patients_csv)
  images_dir = pathlib.Path(FLAGS.images_dir)
  output_path = pathlib.Path(FLAGS.output_path)
  shape = _parse_shape(FLAGS.shape)
  n = FLAGS.num_examples_per_record

  opts = pipeline.Options(
      patients_csv_path=csv_path,
      root_dir=images_dir,
      output_path=output_path,
      output_shape=shape,
      num_examples_per_record=n,
  )

  pipeline.make_dataset(opts)


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'patients_csv', 'images_dir', 'output_path', 'shape',
      'num_examples_per_record'
  ])
  app.run(main)
