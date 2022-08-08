# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional, Tuple

import torch


@torch.inference_mode()
def multiclass_recall(
    input: torch.Tensor,
    target: torch.Tensor,
    num_classes: Optional[int] = None,
    average: Optional[str] = "micro",
) -> torch.Tensor:
    """
    Compute recall score, which is calculated as the ratio between the number of
    true positives (TP) and the total number of actual positives (TP + FN).
    Its class version is ``torcheval.metrics.MultiClassRecall``.

    Args:
        input: Tensor of label predictions
            It could be the predicted labels, with shape of (n_sample, ).
            It could also be probabilities or logits with shape of (n_sample, n_class).
            ``torch.argmax`` will be used to convert input into predicted labels.
        target: Tensor of ground truth labels with shape of (n_sample, ).
        num_classes:
            Number of classes.
        average:
            - ``'micro'``[default]:
                Calculate the metrics globally, by using the total true positives and false
                negatives across all classes.
            - ``'macro'``:
                Calculate metrics for each class separately, and return their unweighted
                mean. Classes with 0 true and predicted instances are ignored.
            - ``'weighted'``:
                Calculate metrics for each class separately, and return their average weighted
                by the number of instances for each class in the ``target`` tensor. Classes with
                0 true and predicted instances are ignored.
            - ``None``:
                Calculate the metric for each class separately, and return
                the metric for every class.

    Example:
        >>> import torch
        >>> from torcheval.metrics.functional.classification import multiclass_recall
        >>> input = torch.tensor([0, 2, 1, 3])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> multiclass_recall(input, target)
        tensor(0.5000)
        >>> multiclass_recall(input, target, average=None, num_classes=4)
        tensor([1., 0., 0., 1.])
        >>> multiclass_recall(input, target, average="macro", num_classes=4)
        tensor(0.5000)
        >>> input = torch.tensor([[0.9, 0.1, 0, 0], [0.1, 0.2, 0.4, 0.3], [0, 1.0, 0, 0], [0, 0, 0.2, 0.8]])
        >>> multiclass_recall(input, target)
        tensor(0.5000)
    """
    _recall_param_check(num_classes, average)
    num_tp, num_labels, num_predictions = _recall_update(
        input, target, num_classes, average
    )
    return _recall_compute(num_tp, num_labels, num_predictions, average)


def _recall_update(
    input: torch.Tensor,
    target: torch.Tensor,
    num_classes: Optional[int],
    average: Optional[str],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _recall_update_input_check(input, target, num_classes)

    if input.ndim == 2:
        input = torch.argmax(input, dim=1)

    if average == "micro":
        num_tp = (input == target).sum()
        num_labels = target.new_tensor(target.numel())
        num_predictions = num_labels
        return num_tp, num_labels, num_predictions

    assert isinstance(
        num_classes, int
    ), f"`num_classes` must be an integer, but received {num_classes}."
    input = input.to(dtype=torch.float32)
    target = target.to(dtype=torch.float32)
    num_tp = torch.histc(
        input[input == target], bins=num_classes, min=0, max=num_classes - 1
    )
    num_labels = torch.histc(target, bins=num_classes, min=0, max=num_classes - 1)
    num_predictions = torch.histc(input, bins=num_classes, min=0, max=num_classes - 1)
    return num_tp, num_labels, num_predictions


def _recall_compute(
    num_tp: torch.Tensor,
    num_labels: torch.Tensor,
    num_predictions: torch.Tensor,
    average: Optional[str],
) -> torch.Tensor:
    if average in ("macro", "weighted"):
        # Ignore classes which have no samples in `target` and `input`
        mask = (num_labels != 0) | (num_predictions != 0)
        num_tp = num_tp[mask]

    recall = num_tp / num_labels

    isnan_class = torch.isnan(recall)
    if isnan_class.any():
        nan_classes = isnan_class.nonzero(as_tuple=True)[0]
        logging.warning(
            f"One or more NaNs identified, as no ground-truth instances of "
            f"{nan_classes.tolist()} have been seen. These have been converted to zero."
        )
        recall = torch.nan_to_num(recall)

    if average == "micro":
        return recall
    elif average == "macro":
        return recall.mean()
    elif average == "weighted":
        weights = num_labels[mask] / num_labels.sum()
        return (recall * weights).sum()
    else:  # average is None
        return recall


def _recall_param_check(num_classes: Optional[int], average: Optional[str]) -> None:
    average_options = ("micro", "macro", "weighted", None)
    if average not in average_options:
        raise ValueError(
            f"`average` was not in the allowed values of {average_options}, "
            f"got {average}."
        )
    if average != "micro" and (num_classes is None or num_classes <= 0):
        raise ValueError(
            f"`num_classes` should be a positive number when average={average}, "
            f"got num_classes={num_classes}."
        )


def _recall_update_input_check(
    input: torch.Tensor, target: torch.Tensor, num_classes: Optional[int]
) -> None:
    if input.size(0) != target.size(0):
        raise ValueError(
            f"The `input` and `target` should have the same first dimension, "
            f"got shapes {input.shape} and {target.shape}."
        )
    if target.ndim != 1:
        raise ValueError(
            f"`target` should be a one-dimensional tensor, got shape {target.shape}."
        )
    if input.ndim != 1 and not (
        input.ndim == 2 and (num_classes is None or input.shape[1] == num_classes)
    ):
        raise ValueError(
            f"`input` should have shape (num_samples,) or (num_samples, num_classes), "
            f"got {input.shape}."
        )
