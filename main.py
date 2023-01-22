"""Main file for training the breast model."""

from absl import app
from absl import flags
from absl import logging

import tensorflow_datasets as tfds

from rsna_mammo_dataset import rsna_mammo_dataset

FLAGS = flags.FLAGS

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tfds.load(
    'rsna_mammo_dataset',
    split='train',
    download=True,
    download_and_prepare_kwargs={
      'download_config': tfds.download.DownloadConfig(
        manual_dir='~/Downloads/results',
      ),
    })


if __name__ == '__main__':
  app.run(main)
