# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import warnings
from importlib.util import find_spec
from typing import Any, Iterable, Optional, TypeVar, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torcheval.metrics.functional.frechet import gaussian_frechet_distance
from torcheval.metrics.metric import Metric

if find_spec("torchvision") is not None:
    from torchvision import models

    _TORCHVISION_AVAILABLE = True
else:
    _TORCHVISION_AVAILABLE = False

TFrechetInceptionDistance = TypeVar("TFrechetInceptionDistance")

# pyre-ignore-all-errors[16]: Undefined attribute of metric states.


def _validate_torchvision_available() -> None:
    if not _TORCHVISION_AVAILABLE:
        raise RuntimeError(
            "You must have torchvision installed to use FID, please install torcheval[image]"
        )


class FIDInceptionV3(nn.Module):
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
        self.model = models.inception_v3(weights=weights)
        # Do not want fc layer
        self.model.fc = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # Interpolating the input image tensors to be of size 299 x 299
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = self.model(x)

        return x


class FrechetInceptionDistance(Metric[torch.Tensor]):
    def __init__(
        self: TFrechetInceptionDistance,
        model: Optional[nn.Module] = None,
        feature_dim: int = 2048,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Computes the Frechet Inception Distance (FID) between two distributions of images (real and generated).

        The original paper: https://arxiv.org/pdf/1706.08500.pdf

        Args:
            model (nn.Module): Module used to compute feature activations.
                If None, a default InceptionV3 model will be used.
            feature_dim (int): The number of features in the model's output,
                the default number is 2048 for default InceptionV3.
            device (torch.device): The device where the computations will be performed.
                If None, the default device will be used.
        """
        _validate_torchvision_available()

        super().__init__(device=device)

        self._FID_parameter_check(model=model, feature_dim=feature_dim)

        if model is None:
            model = FIDInceptionV3()

        # Set the model and put it in evaluation mode
        self.model = model.to(device)
        self.model.eval()
        self.model.requires_grad_(False)

        # Initialize state variables used to compute FID
        self._add_state(
            "real_sum", torch.zeros(feature_dim, device=device, dtype=torch.float64)
        )
        self._add_state(
            "real_cov_sum",
            torch.zeros((feature_dim, feature_dim), device=device, dtype=torch.float64),
        )
        self._add_state(
            "fake_sum", torch.zeros(feature_dim, device=device, dtype=torch.float64)
        )
        self._add_state(
            "fake_cov_sum",
            torch.zeros((feature_dim, feature_dim), device=device, dtype=torch.float64),
        )
        self._add_state("num_real_images", torch.tensor(0, device=device).int())
        self._add_state("num_fake_images", torch.tensor(0, device=device).int())

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(
        self: TFrechetInceptionDistance, images: Tensor, is_real: bool
    ) -> TFrechetInceptionDistance:
        """
        Update the states with a batch of real and fake images.

        Args:
            images (Tensor): A batch of images.
            is_real (Boolean): Denotes if images are real or not.
        """

        self._FID_update_input_check(images=images, is_real=is_real)

        images = images.to(self.device)

        # Compute activations for images using the given model
        activations = self.model(images)

        batch_size = images.shape[0]

        # Update the state variables used to compute FID
        if is_real:
            self.num_real_images += batch_size
            self.real_sum += torch.sum(activations, dim=0)
            self.real_cov_sum += torch.matmul(activations.T, activations)
        else:
            self.num_fake_images += batch_size
            self.fake_sum += torch.sum(activations, dim=0)
            self.fake_cov_sum += torch.matmul(activations.T, activations)

        return self

    @torch.inference_mode()
    def merge_state(
        self: TFrechetInceptionDistance, metrics: Iterable[TFrechetInceptionDistance]
    ) -> TFrechetInceptionDistance:
        """
        Merge the state of another FID instance into this instance.

        Args:
            metrics (Iterable[FID]): The other FID instance(s) whose state will be merged into this instance.
        """
        for metric in metrics:
            self.real_sum += metric.real_sum.to(self.device)
            self.real_cov_sum += metric.real_cov_sum.to(self.device)
            self.fake_sum += metric.fake_sum.to(self.device)
            self.fake_cov_sum += metric.fake_cov_sum.to(self.device)
            self.num_real_images += metric.num_real_images.to(self.device)
            self.num_fake_images += metric.num_fake_images.to(self.device)

        return self

    @torch.inference_mode()
    def compute(self: TFrechetInceptionDistance) -> Tensor:
        """
        Compute the FID.

        Returns:
            tensor: The FID.
        """

        # If the user has not already updated with at lease one
        # image from each distribution, then we raise an Error.
        if (self.num_real_images < 2) or (self.num_fake_images < 2):
            warnings.warn(
                "Computing FID requires at least 2 real images and 2 fake images,"
                f"but currently running with {self.num_real_images} real images and {self.num_fake_images} fake images."
                "Returning 0.0",
                RuntimeWarning,
                stacklevel=2,
            )

            return torch.tensor(0.0)

        # Compute the mean activations for each distribution
        real_mean = (self.real_sum / self.num_real_images).unsqueeze(0)
        fake_mean = (self.fake_sum / self.num_fake_images).unsqueeze(0)

        # Compute the covariance matrices for each distribution
        real_cov_num = self.real_cov_sum - self.num_real_images * torch.matmul(
            real_mean.T, real_mean
        )
        real_cov = real_cov_num / (self.num_real_images - 1)
        fake_cov_num = self.fake_cov_sum - self.num_fake_images * torch.matmul(
            fake_mean.T, fake_mean
        )
        fake_cov = fake_cov_num / (self.num_fake_images - 1)

        # Compute the Frechet Distance between the distributions
        fid = gaussian_frechet_distance(
            real_mean.squeeze(), real_cov, fake_mean.squeeze(), fake_cov
        )
        fid = fid.to(torch.float32)
        return fid

    def _FID_parameter_check(
        self: TFrechetInceptionDistance,
        model: Optional[nn.Module],
        feature_dim: int,
    ) -> None:
        # Whatever the model, the feature_dim needs to be set
        if feature_dim is None or feature_dim <= 0:
            raise RuntimeError("feature_dim has to be a positive integer")

        if model is None and feature_dim != 2048:
            raise RuntimeError(
                "When the default Inception v3 model is used, feature_dim needs to be set to 2048"
            )

    def _FID_update_input_check(
        self: TFrechetInceptionDistance, images: torch.Tensor, is_real: bool
    ) -> None:
        if not torch.is_tensor(images):
            raise ValueError(f"Expected tensor as input, but got {type(images)}.")

        if images.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor as input. But input has {images.dim()} dimenstions."
            )

        if images.size()[1] != 3:
            raise ValueError(f"Expected 3 channels as input. Got {images.size()[1]}.")

        if type(is_real) != bool:
            raise ValueError(
                f"Expected 'real' to be of type bool but got {type(is_real)}.",
            )

        if isinstance(self.model, FIDInceptionV3):
            if images.dtype != torch.float32:
                raise ValueError(
                    f"When default inception-v3 model is used, images expected to be `torch.float32`, but got {images.dtype}."
                )

            if images.min() < 0 or images.max() > 1:
                raise ValueError(
                    "When default inception-v3 model is used, images are expected to be in the [0, 1] interval"
                )

    def to(
        self: TFrechetInceptionDistance,
        device: Union[str, torch.device],
        *args: Any,
        **kwargs: Any,
    ) -> TFrechetInceptionDistance:
        super().to(device=device)
        self.model.to(self.device)
        return self
