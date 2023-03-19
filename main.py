"""Main file for training the breast model."""

import input_pipeline

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  ds = input_pipeline.load(split='train',
                           batch_size=32,
                           prefetch_buffer_size=1,
                           deterministic=False)


if __name__ == '__main__':
  app.run(main)
