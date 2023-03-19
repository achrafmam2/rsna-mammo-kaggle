"""Input pipeline for RSNA mamo."""

import functools
from typing import Any

import jax
import tensorflow as tf
import tensorflow_datasets as tfds

from rsna_mammo_dataset import rsna_mammo_dataset

TensorMapping = dict[str, tf.Tensor]


def load(
    *,
    split: str,
    batch_size: int,
    prefetch_buffer_size: int,
    deterministic: bool,
) -> tf.data.Dataset:
  """Loads a dataset split.

  Args:
    - split: Which split to read. Valid values are: 'train', 'validation', and
        'test'.
    - batch_size: Global batch size.
    - prefetch_buffer_size: The maximum number of elements that will be buffered
        when prefetching.
    - deterministic: Whether the outputs need to be produced in deterministic order.
  """
  ds = tfds.load('rsna_mammo_dataset', split=split)

  decode_fn = functools.partial(
      decode_example,
      as_supervised=split in ['train', 'validation'],
  )
  ds = ds.map(decode_fn, num_parallel_calls=tf.data.AUTOTUNE)

  # TODO(achraf): Add image augmentation.

  ds = ds.batch(batch_size, drop_remainder=True)
  ds = ds.prefetch(prefetch_buffer_size)

  opts = tf.data.Options()
  opts.deterministic = deterministic
  ds = ds.with_options(opts)

  return ds


def decode_example(
    example: TensorMapping,
    as_supervised: bool,
) -> TensorMapping:
  image = tf.cast(example['image'], dtype=tf.float32)
  image = image / 255.0

  if not as_supervised:
    return {'image': image}

  return {'image': image, 'label': example['cancer']}
