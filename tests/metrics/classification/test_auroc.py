# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from sklearn.metrics import roc_auc_score

from torcheval.metrics import AUROC
from torcheval.utils.test_utils.metric_class_tester import (
    BATCH_SIZE,
    MetricClassTester,
    NUM_TOTAL_UPDATES,
)


class TestAUROC(MetricClassTester):
    def _test_auroc_class_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        input_tensors = input.reshape(-1, 1)
        target_tensors = target.reshape(-1, 1)
        compute_result = torch.tensor(
            roc_auc_score(target_tensors, input_tensors), dtype=torch.float32
        )

        self.run_class_implementation_tests(
            metric=AUROC(),
            state_names={"inputs", "targets"},
            update_kwargs={
                "input": input,
                "target": target,
            },
            compute_result=compute_result,
        )

    def test_auroc_class_base(self) -> None:
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        target = torch.randint(high=2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        self._test_auroc_class_with_input(input, target)

        input = torch.randint(high=2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        target = torch.randint(high=2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        self._test_auroc_class_with_input(input, target)

    def test_auroc_class_update_input_shape_different(self) -> None:
        num_classes = 2
        update_input = [
            torch.rand(5),
            torch.rand(8),
            torch.rand(2),
            torch.rand(5),
        ]

        update_target = [
            torch.randint(high=num_classes, size=(5,)),
            torch.randint(high=num_classes, size=(8,)),
            torch.randint(high=num_classes, size=(2,)),
            torch.randint(high=num_classes, size=(5,)),
        ]
        compute_result = torch.tensor(
            roc_auc_score(
                torch.cat(update_target, dim=0),
                torch.cat(update_input, dim=0),
            ),
            dtype=torch.float32,
        )

        self.run_class_implementation_tests(
            metric=AUROC(),
            state_names={"inputs", "targets"},
            update_kwargs={
                "input": update_input,
                "target": update_target,
            },
            compute_result=compute_result,
            num_total_updates=4,
            num_processes=2,
        )

    def test_auroc_class_invalid_input(self) -> None:
        metric = AUROC()
        with self.assertRaisesRegex(
            ValueError,
            "input should be a one-dimensional tensor, "
            r"got shape torch.Size\(\[3, 2\]\).",
        ):
            metric.update(torch.rand(3, 2), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError,
            "target should be a one-dimensional tensor, "
            r"got shape torch.Size\(\[3, 2\]\).",
        ):
            metric.update(torch.rand(3), torch.rand(3, 2))

        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same shape, "
            r"got shapes torch.Size\(\[4\]\) and torch.Size\(\[3\]\).",
        ):
            metric.update(torch.rand(4), torch.rand(3))
