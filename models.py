"""Flax implementation of the Breast Cancer Detection Model"""

import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
import functools

from typing import Any, Sequence, Tuple

ModuleDef = Any


class ResNetBlock(nn.Module):
  features: int
  strides: Tuple[int, int]
  conv: ModuleDef
  norm: ModuleDef
  act: ModuleDef

  @nn.compact
  def __call__(self, x):
    residual = x

    y = self.conv(self.features, kernel_size=(3, 3), strides=self.strides)(x)
    y = self.norm()(y)
    y = self.act(y)

    y = self.conv(self.features, kernel_size=(3, 3), strides=(1, 1))(y)
    y = self.norm()(y)

    if y.shape != residual.shape:
      residual = self.conv(self.features,
                           kernel_size=(1, 1),
                           strides=self.strides)(residual)
      residual = self.norm()(residual)

    return self.act(y + residual)


class ResNet(nn.Module):
  num_classes: int
  blocks: Sequence[int]
  features: int = 64
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x, train=True):
    conv = functools.partial(nn.Conv, use_bias=False, dtype=self.dtype)
    norm = functools.partial(nn.BatchNorm,
                             use_running_average=not train,
                             momentum=0.9,
                             epsilon=1e-5,
                             dtype=self.dtype)
    act = nn.relu

    x = conv(self.features,
             kernel_size=(7, 7),
             strides=(2, 2),
             name='conv_init')(x)
    x = norm(name='bn_init')(x)
    x = act(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')

    for i, block_size in enumerate(self.blocks):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = ResNetBlock(self.features * (2**i),
                        strides=strides,
                        conv=conv,
                        norm=norm,
                        act=act)(x)

    x = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
    x = jnp.asarray(x, self.dtype)
    return x


class CaseModel(nn.Module):
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x, train=True):
    lcc, lmlo = x['image/L_CC'], x['image/L_MLO']
    rcc, rmlo = x['image/R_CC'], x['image/R_MLO']

    breast_model = ResNet(num_classes=1024,
                          blocks=[2, 2, 2, 2],
                          dtype=self.dtype)

    x = []
    for img in [lcc, lmlo, rcc, rmlo]:
      x.append(breast_model(img, train=train))

    l = jnp.concatenate(jnp.array(x[:2]), axis=-1, dtype=self.dtype)
    l = nn.Dense(512, dtype=self.dtype)(l)

    r = jnp.concatenate(jnp.array(x[2:]), axis=-1, dtype=self.dtype)
    r = nn.Dense(512, dtype=self.dtype)(r)

    x = jnp.concatenate([l, r], axis=-1)
    x = nn.Dense(256, dtype=self.dtype)(x)
    x = nn.Dense(2, dtype=self.dtype)(x)

    return {
        'output/L': x[..., 0],
        'output/R': x[..., 1],
    }
