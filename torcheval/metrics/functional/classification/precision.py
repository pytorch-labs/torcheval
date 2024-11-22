# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[6]: expect int got Optional[int] for num_classes
# pyre-ignore-all-errors[58]: - is not supported for operand types Optional[int] and int

import logging
from typing import Optional, Tuple

import torch


@torch.inference_mode()
def binary_precision(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Compute precision score for binary classification class, which is calculated as the ratio between the number of
    true positives (TP) and the total number of predicted positives (TP + FP).
    Its class version is ``torcheval.metrics.BinaryPrecision``.
    See also :func:`multiclass_precision <torcheval.metrics.functional.multiclass_precision>`

    Args:
        input (Tensor): Tensor of label predictions
            It could be the predicted labels, with shape of (n_sample, ).
            ``torch.where(input < threshold, 0, 1)`` will be applied to the input.
        target (Tensor): Tensor of ground truth labels with shape of (n_sample,).
        threshold (float, default 0.5): Threshold for converting input into predicted labels for each sample.
    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional import binary_precision
        >>> input = torch.tensor([0, 0, 1, 1])
        >>> target = torch.tensor([1, 0, 1, 1])
        >>> binary_precision(input, target)
        tensor(1.)  # 2 / 2

        >>> metric = BinaryPrecision(threshold=0.7)
        >>> input = torch.tensor([0, 0.8, 0.6, 0.7])
        >>> target = torch.tensor([1, 0, 1, 1])
        >>> binary_precision(input, target)
        tensor(0.5)  # 1 / 2
    """

    num_tp, num_fp, num_label = _binary_precision_update(input, target, threshold)
    return _precision_compute(num_tp, num_fp, num_label, "micro")


@torch.inference_mode()
def multiclass_precision(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: Optional[int] = None,
    average: Optional[str] = "micro",
) -> torch.Tensor:
    """
    Compute precision score, which is the ratio of the true positives (TP) and the
    total number of points classified as positives (TP + FP).
    Its class version is ``torcheval.metrics.MultiClassPrecision``.
    See also :func:`binary_precision <torcheval.metrics.functional.binary_precision>`

    Args:
        input (Tensor): Tensor of label predictions
            It could be the predicted labels, with shape of (n_sample, ).
            It could also be probabilities or logits with shape of (n_sample, n_class).
            ``torch.argmax`` will be used to convert input into predicted labels.
        target (Tensor): Tensor of ground truth labels with shape of (n_sample, ).
        num_classes:
            Number of classes.
        average:
            - ``'micro'`` [default]:
                Calculate the metrics globally, by using the total true positives and false
                positives across all classes.
            - ``'macro'``:
                Calculate metrics for each class separately, and return their unweighted
                mean. Classes with 0 true instances and predicted instances are ignored.
            - ``'weighted'``:
                Calculate metrics for each class separately, and return their average weighted
                by the number of instances for each class in the ``target`` tensor. Classes with
                0 true instances and predicted instances are ignored.
            - ``None``:
                Calculate the metric for each class separately, and return
                the metric for every class.
                NaN is returned if a class has no sample in ``target``.

    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional import multiclass_precision
        >>> input = torch.tensor([0, 2, 1, 3])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> multiclass_precision(input, target)
        tensor(0.5)
        >>> multiclass_precision(input, target, average=None, num_classes=4)
        tensor([1., 0., 0., 1.])
        >>> multiclass_precision(input, target, average="macro", num_classes=4)
        tensor(0.5)
        >>> input = torch.tensor([[0.9, 0.1, 0, 0], [0.1, 0.2, 0.4, 0,3], [0, 1.0, 0, 0], [0, 0, 0.2, 0.8]])
        >>> multiclass_precision(input, target)
        tensor(0.5)
    """

    _precision_param_check(num_classes, average)
    num_tp, num_fp, num_class = _precision_update(input, target, num_classes, average)
    return _precision_compute(num_tp, num_fp, num_class, average)


def _precision_update(
    input: torch.Tensor,
    target: torch.Tensor,
    num_classes: Optional[int],
    average: Optional[str],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _precision_update_input_check(input, target, num_classes)

    if input.ndim == 2:
        input = torch.argmax(input, dim=1)

    if average == "micro":
        num_tp = (input == target).sum()
        num_fp = (input != target).sum()
        return num_tp, num_fp, torch.tensor(0.0)

    num_label = target.new_zeros(num_classes).scatter_(0, target, 1, reduce="add")
    num_tp = target.new_zeros(num_classes).scatter_(
        0, target[input == target], 1, reduce="add"
    )
    num_fp = target.new_zeros(num_classes).scatter_(
        0, input[input != target], 1, reduce="add"
    )

    return num_tp, num_fp, num_label


def _precision_compute(
    num_tp: torch.Tensor,
    num_fp: torch.Tensor,
    num_label: torch.Tensor,
    average: Optional[str],
) -> torch.Tensor:
    if average in ("macro", "weighted"):
        # Ignore the class that has no samples in both `input` and `target`
        mask = (num_label != 0) | ((num_tp + num_fp) != 0)
        num_tp, num_fp = num_tp[mask], num_fp[mask]

    precision = num_tp / (num_tp + num_fp)

    if average in (None, "None") and torch.isnan(precision).sum():
        # Warning in case there are classes with zero representation in both the
        # predictions and the ground truth.
        bad_class = torch.nonzero(torch.isnan(precision))
        logging.warning(
            f"{bad_class} classes have zero instances in either the "
            "predictions or the ground truth labels. Precision is still logged "
            "as zero."
        )

    # If precision is NaN, convert it to 0.
    precision = torch.nan_to_num(precision)

    if average == "micro":
        return precision
    elif average == "macro":
        return precision.mean()
    elif average == "weighted":
        # pyre-fixme[61]: `mask` is undefined, or not always defined.
        return torch.inner(precision, (num_label[mask] / num_label.sum()))
    else:  # average is None
        return precision


def _precision_param_check(
    num_classes: Optional[int],
    average: Optional[str],
) -> None:
    average_options = ("micro", "macro", "weighted", "None", None)
    if average not in average_options:
        raise ValueError(
            f"`average` was not in the allowed value of {average_options}, got {average}."
        )
    if average != "micro" and (num_classes is None or num_classes <= 0):
        raise ValueError(
            f"num_classes should be a positive number when average={average}."
            f" Got num_classes={num_classes}."
        )


def _precision_update_input_check(
    input: torch.Tensor,
    target: torch.Tensor,
    num_classes: Optional[int],
) -> None:
    if input.size(0) != target.size(0):
        raise ValueError(
            "The `input` and `target` should have the same first dimension, "
            f"got shapes {input.shape} and {target.shape}."
        )

    if target.ndim != 1:
        raise ValueError(
            f"target should be a one-dimensional tensor, got shape {target.shape}."
        )

    if not input.ndim == 1 and not (
        input.ndim == 2 and (num_classes is None or input.shape[1] == num_classes)
    ):
        raise ValueError(
            "input should have shape of (num_sample,) or (num_sample, num_classes), "
            f"got {input.shape}."
        )


def _binary_precision_update(
    input: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _binary_precision_update_input_check(input, target)

    input = torch.where(input < threshold, 0, 1)
    num_tp = (input * target).sum(dim=-1)
    num_fp = input.sum(dim=-1) - num_tp

    num_label = torch.tensor(0.0)

    return num_tp, num_fp, num_label


def _binary_precision_update_input_check(
    input: torch.Tensor,
    target: torch.Tensor,
) -> None:
    if input.shape != target.shape:
        raise ValueError(
            "The `input` and `target` should have the same dimensions, "
            f"got shapes {input.shape} and {target.shape}."
        )
    if target.ndim != 1:
        raise ValueError(
            f"target should be a one-dimensional tensor, got shape {target.shape}."
        )
