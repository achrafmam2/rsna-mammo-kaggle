"""RSNA Mammo Dataset Builder."""

from typing import Tuple, Iterable

import tensorflow_datasets as tfds
import numpy as np

from rsna_mammo_dataset import rsna_mammo_pipeline as rmp

Example = tfds.core.split_builder.Example


class RsnaMammoDataset(tfds.core.GeneratorBasedBuilder):
  """TFDS builder for the RSNA Mammo dataset."""

  VERSION = tfds.core.Version('0.3.2')
  RELEASE_NOTES = {
      '0.1.0': 'Initial release.',
      '0.2.0': 'Add an eval split.',
      '0.3.0': 'Add a test split.',
      '0.3.1': 'Fix incorrect labels.',
      '0.3.2': 'Rename eval split to validation.',
  }

  MANUAL_DOWNLOAD_INSTRUCTIONS = 'Download the dataset from Kaggle.'

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            'patient_id':
                tfds.features.Scalar(np.int64),
            'image':
                tfds.features.Image(shape=(768, 384, 1)),
            'laterality':
                tfds.features.ClassLabel(names=['L', 'R']),
            'view':
                tfds.features.ClassLabel(names=['CC', 'MLO']),
            'cancer':
                tfds.features.ClassLabel(
                    names=['no', 'yes'],
                    doc='Whether the breast has cancer',
                ),
        }),
        supervised_keys=('image', 'cancer'),
        homepage=
        'https://www.kaggle.com/competitions/rsna-breast-cancer-detection',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""

    root = dl_manager.manual_dir
    return {
        'train':
            self._generate_examples(metadata_path=root / 'train.csv',
                                    split_path=root / 'splits/train.csv',
                                    images_dir=root / 'train_images'),
        'validation':
            self._generate_examples(
                metadata_path=root / 'train.csv',  # This correct.
                split_path=root / 'splits/validation.csv',
                images_dir=root / 'train_images'),
        'test':
            self._generate_examples(metadata_path=root / 'test.csv',
                                    images_dir=root / 'test_images',
                                    labels_exist=False),
    }

  def _generate_examples(
      self,
      *,
      metadata_path,
      images_dir,
      labels_exist: bool = True,
      split_path=None,
  ) -> Iterable[Tuple[str, Example]]:
    """Yields examples."""

    def to_example(s: rmp.Scan):
      ex = {
          'image': s.image[
              ..., np.newaxis],  # Channel dimension is required by tfds.
          'patient_id': s.patient_id,
          'laterality': s.laterality,
          'view': s.view,
      }
      if labels_exist:
        ex['cancer'] = int(s.cancer)
      else:
        ex['cancer'] = -1  # For the test split.

      return s.scan_id, ex

    opts = rmp.Options(metadata_path=metadata_path,
                       split_path=split_path,
                       images_dir=images_dir,
                       shape=(768, 384),
                       allowed_views=('CC', 'MLO'),
                       labels_exist=labels_exist)
    return map(to_example, rmp.load_scans(opts))
