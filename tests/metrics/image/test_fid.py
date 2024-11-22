# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Optional

import numpy as np

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from torcheval.metrics.image.fid import FrechetInceptionDistance
from torcheval.utils.test_utils.metric_class_tester import (
    BATCH_SIZE,
    IMG_CHANNELS,
    MetricClassTester,
    NUM_TOTAL_UPDATES,
)
from torchvision import models


class ResnetFeatureExtractor(nn.Module):
    def __init__(
        self,
        weights: Optional[str] = "DEFAULT",
    ) -> None:
        """
        This class wraps the InceptionV3 model to compute FID.

        Args:
            weights Optional[str]: Defines the pre-trained weights to use.
        """
        super().__init__()
        # pyre-ignore
        self.model = models.resnet.resnet18(weights=weights)
        # Do not want fc layer
        self.model.fc = nn.Identity()
        self.model.eval()

    def forward(self, x: Tensor) -> Tensor:
        # Interpolating the input image tensors to be of size 224 x 224
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = self.model(x)

        return x


class TestFrechetInceptionDistance(MetricClassTester):
    def setUp(self) -> None:
        super(TestFrechetInceptionDistance, self).setUp()
        torch.manual_seed(0)

    def _get_random_data_FrechetInceptionDistance(
        self,
        num_updates: int,
        batch_size: int,
        num_channels: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        imgs = torch.rand(
            size=(num_updates, batch_size, num_channels, height, width),
        )

        return imgs

    def test_fid_random_data_default_model(self) -> None:
        imgs = self._get_random_data_FrechetInceptionDistance(
            NUM_TOTAL_UPDATES,
            BATCH_SIZE,
            IMG_CHANNELS,
            299,
            299,
        )
        self._test_fid(
            imgs=imgs, feature_dim=2048, expected_result=torch.tensor(4.58449)
        )

    def test_fid_random_data_custom_model(self) -> None:
        imgs = self._get_random_data_FrechetInceptionDistance(
            NUM_TOTAL_UPDATES,
            BATCH_SIZE,
            IMG_CHANNELS,
            224,
            224,
        )

        feature_extractor = ResnetFeatureExtractor()

        self._test_fid(
            imgs=imgs,
            feature_dim=512,
            model=feature_extractor,
            expected_result=torch.tensor(0.990241),
        )

    def _test_fid(
        self,
        imgs: torch.Tensor,
        feature_dim: int,
        expected_result: torch.Tensor,
        model: Optional[torch.nn.Module] = None,
    ) -> None:
        # create an alternating list of boolean values to
        # simulate a sequence of alternating real and generated images
        real_or_gen = [idx % 2 == 0 for idx in range(NUM_TOTAL_UPDATES)]

        state_names = {
            "real_sum",
            "real_cov_sum",
            "num_real_images",
            "fake_sum",
            "fake_cov_sum",
            "num_fake_images",
        }

        self.run_class_implementation_tests(
            metric=FrechetInceptionDistance(model=model, feature_dim=feature_dim),
            state_names=state_names,
            update_kwargs={
                "images": imgs,
                "is_real": real_or_gen,
            },
            compute_result=expected_result,
            min_updates_before_compute=2,
            test_merge_with_one_update=False,
            atol=1e-2,
            rtol=1e-2,
            test_devices=["cpu"],
        )

    def test_fid_invalid_input(self) -> None:
        metric = FrechetInceptionDistance()
        with self.assertRaisesRegex(
            ValueError,
            "Expected 3 channels as input. Got 4.",
        ):
            metric.update(torch.rand(4, 4, 256, 256), is_real=False)

        with self.assertRaisesRegex(
            ValueError, "Expected 'real' to be of type bool but got <class 'float'>."
        ):
            # pyre-ignore
            metric.update(torch.rand(4, 3, 256, 256), is_real=1.0)

        with self.assertRaisesRegex(
            ValueError,
            "Expected 4D tensor as input. But input has 3 dimenstions",
        ):
            metric.update(torch.rand(3, 256, 256), is_real=True)

        with self.assertRaisesRegex(
            ValueError,
            "Expected tensor as input, but got .*",
        ):
            metric.update(np.random.rand(4, 3, 256, 256), is_real=True)

        with self.assertRaisesRegex(
            ValueError,
            "When default inception-v3 model is used, images expected to be `torch.float32`, but got torch.uint8.",
        ):
            metric.update(torch.rand(4, 3, 256, 256).byte(), is_real=False)

        with self.assertRaisesRegex(
            ValueError,
            r"When default inception-v3 model is used, images are expected to be in the \[0, 1\] interval",
        ):
            metric.update(torch.rand(4, 3, 256, 256) * 2, is_real=False)

    def test_fid_invalid_params(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "feature_dim has to be a positive integer",
        ):
            FrechetInceptionDistance(feature_dim=-1)

        with self.assertRaisesRegex(
            RuntimeError,
            "When the default Inception v3 model is used, feature_dim needs to be set to 2048",
        ):
            FrechetInceptionDistance(feature_dim=256)

    def test_fid_with_similar_inputs(self) -> None:
        real_images = torch.ones(2, 3, 224, 224)
        fake_images = torch.ones(2, 3, 224, 224)

        metric = FrechetInceptionDistance()

        metric.update(real_images, is_real=True)
        metric.update(fake_images, is_real=False)
        fid_score = metric.compute().item()
        metric.reset()

        assert fid_score < 10, "FID must be low for similar inputs."

    def test_fid_with_dissimilar_inputs(self) -> None:
        real_images = torch.zeros(2, 3, 224, 224)
        # The differnet fake images are alternating 1s and 0s which should result in a higher FID
        fake_images = torch.zeros(2 * 3 * 224 * 224)
        fake_images[::2] = 1
        fake_images = fake_images.reshape(2, 3, 224, 224)

        metric = FrechetInceptionDistance()

        metric.update(real_images, is_real=True)
        metric.update(fake_images, is_real=False)
        fid_score = metric.compute().item()
        metric.reset()

        assert fid_score > 100, "FID must be high for dissimilar inputs."
