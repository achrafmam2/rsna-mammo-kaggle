"""tfds dataset."""

import tensorflow_datasets as tfds
import rsna_mammo_pipeline as rmp
import numpy as np
import epath

from typing import Tuple, Iterable

Example = tfds.core.split_builder.Example
SplitGenerator = tfds.core.split_builder.SplitBuilder


class RsnaMammoBuilder(tfds.core.GeneratorBasedBuilder):
  """TFDS builder for the RSNA Mammo dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            'patient_id': tfds.features.Scalar(np.int64),
            'image': tfds.features.Image(shape=(768, 384, 1)),
            'laterality': tfds.features.ClassLabel(names=['L', 'R']),            
            'view': tfds.features.ClassLabel(names=['CC', 'MLO']),
            'cancer': tfds.features.ClassLabel(
                names=['no', 'yes'],
                doc='Whether the breast has cancer',
            ),
        }),
        supervised_keys=('image', 'cancer'),
        homepage='https://www.kaggle.com/competitions/rsna-breast-cancer-detection',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    
    root = dl_manager.manual_dir
    return {
        'train': self._generate_examples(
            csv_path=root / 'train.csv', images_dir=root / 'train_images'),
    }

  def _generate_examples(self, csv_path: epath.Path, images_dir: epath.Path) -> Iterable[Tuple[str, Example]]:
    """Yields examples."""
    
    def to_example(s: rmp.Scan) -> Example:
      return s.scan_id, {
        'image': s.image[..., np.newaxis], # Channel dimension is required by tfds.
        'patient_id': s.patient_id,
        'laterality': s.laterality,
        'view': s.view,
        'cancer': int(s.cancer),
      }

    opts = rmp.Options(
      csv_path=csv_path,
      images_dir=images_dir,
      shape=(768, 384),
      allowed_views=('CC', 'MLO'))

    return map(to_example, rmp.load_scans(opts))
