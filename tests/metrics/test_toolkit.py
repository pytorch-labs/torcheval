# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
import uuid
from typing import Callable, Type, Union

import torch
import torch.distributed.launcher as pet
from torcheval.metrics import Metric
from torcheval.metrics.toolkit import (
    clone_metric,
    clone_metrics,
    get_synced_metric,
    get_synced_state_dict,
    reset_metrics,
    sync_and_compute,
    to_device,
)
from torcheval.utils.env import init_from_env
from torcheval.utils.test_utils.dummy_metric import (
    DummySumListStateMetric,
    DummySumMetric,
)
from typing_extensions import Literal


class MetricToolkitTest(unittest.TestCase):
    def test_metric_sync(self) -> None:
        num_processes = 4
        input_tensor = torch.rand(num_processes * 2)
        # recipient_rank = 0
        self._launch_metric_sync_test(num_processes, input_tensor, DummySumMetric)
        self._launch_metric_sync_test(
            num_processes, input_tensor, DummySumListStateMetric
        )
        # recipient_rank = 1
        self._launch_metric_sync_test(num_processes, input_tensor, DummySumMetric, 1)
        self._launch_metric_sync_test(
            num_processes, input_tensor, DummySumListStateMetric, 1
        )
        # recipient_rank = "all"
        self._launch_metric_sync_test(
            num_processes, input_tensor, DummySumMetric, "all"
        )
        self._launch_metric_sync_test(
            num_processes, input_tensor, DummySumListStateMetric, "all"
        )

    def test_metric_sync_world_size_1(self) -> None:
        metric = DummySumMetric()
        synced_metric = get_synced_metric(metric)
        self.assertIsNotNone(synced_metric)
        self.assertDictEqual(metric.state_dict(), synced_metric.state_dict())

        self.assertDictEqual(get_synced_state_dict(metric), metric.state_dict())
        self.assertDictEqual(
            get_synced_state_dict(metric, recipient_rank="all"), metric.state_dict()
        )

        torch.testing.assert_close(sync_and_compute(metric), metric.compute())
        torch.testing.assert_close(
            sync_and_compute(metric, recipient_rank="all"), metric.compute()
        )

    def test_metric_sync_state_invalid_recipient_rank(self) -> None:
        metric = DummySumMetric()
        with self.assertRaisesRegex(
            ValueError,
            "``recipient_rank`` should be an integer or 'all', got ALL instead.",
        ):
            # pyre-ignore[6]: It's intended to test the behaviour when recipient_rank's value is not correct.
            get_synced_metric(metric, recipient_rank="ALL")

    def test_clone_metric(self) -> None:
        metric = DummySumMetric()
        self.assertDictEqual(clone_metric(metric).state_dict(), metric.state_dict())
        metric.update(torch.tensor(2.0))
        self.assertDictEqual(clone_metric(metric).state_dict(), metric.state_dict())
        if torch.cuda.is_available():
            metric.to("cuda")
            cloned_metric = clone_metric(metric)
            self.assertEqual(cloned_metric.device, torch.device("cuda"))
            # pyre-ignore[16]: Undefined attribute [16]: `Metric` has no attribute `sum`.
            self.assertEqual(cloned_metric.sum.device, "cuda")

    def test_clone_metrics(self) -> None:
        metrics = [DummySumMetric(), DummySumMetric()]
        cloned = clone_metrics(metrics)
        for original, clone in zip(metrics, cloned):
            self.assertDictEqual(original.state_dict(), clone.state_dict())

        if torch.cuda.is_available():
            metrics = to_device(metrics, torch.device("cuda"))
            cloned = clone_metrics(metrics)
            for original, clone in zip(metrics, cloned):
                self.assertEqual(original.device, clone.device)
                self.assertEqual(clone.device, torch.device("cuda"))
                self.assertEqual(clone.sum.device, torch.device("cuda"))

    @staticmethod
    def _test_per_process_metric_sync(
        input_tensor: torch.Tensor,
        metric_class: Callable[[], Metric],
        recipient_rank: Union[int, Literal["all"]],
    ) -> None:
        device = init_from_env()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        tc = unittest.TestCase()

        metric = metric_class().to(device)
        input_tensor = input_tensor.to(device)
        num_total_updates = len(input_tensor)
        for i in range(rank, num_total_updates, world_size):
            metric.update(input_tensor[i])

        state_dict_before_sync = metric.state_dict()
        synced_metric = get_synced_metric(metric, recipient_rank=recipient_rank)
        compute_result = sync_and_compute(metric, recipient_rank=recipient_rank)
        synced_state_dict = get_synced_state_dict(metric, recipient_rank=recipient_rank)

        # input metric state unchanged
        tc.assertDictEqual(metric.state_dict(), state_dict_before_sync)

        if rank == recipient_rank or recipient_rank == "all":
            metric_with_all_updates = metric_class().to(device)
            for i in range(num_total_updates):
                metric_with_all_updates.update(input_tensor[i])
            tc.assertIsNotNone(synced_metric)
            torch.testing.assert_close(
                synced_metric.compute(),
                metric_with_all_updates.compute(),
                check_device=False,
            )
            torch.testing.assert_close(
                compute_result,
                metric_with_all_updates.compute(),
                check_device=False,
            )
            tc.assertGreater(len(synced_state_dict), 0)
        else:
            tc.assertIsNone(synced_metric)
            tc.assertIsNone(compute_result)
            tc.assertDictEqual(synced_state_dict, {})

    def _launch_metric_sync_test(
        self,
        num_processes: int,
        input_tensor: torch.Tensor,
        metric_class: Type[Metric],
        recipient_rank: Union[int, Literal["all"]] = 0,
    ) -> None:
        lc = pet.LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=num_processes,
            run_id=str(uuid.uuid4()),
            rdzv_backend="c10d",
            rdzv_endpoint="localhost:0",
            max_restarts=0,
            monitor_interval=1,
        )
        pet.elastic_launch(lc, entrypoint=self._test_per_process_metric_sync)(
            input_tensor,
            metric_class,
            recipient_rank,
        )

    def test_reset_metrics(self) -> None:
        metric1 = DummySumMetric()
        metric2 = DummySumMetric()
        metric1.update(torch.tensor(2.0))
        metric2.update(torch.tensor(3.0))
        metric1, metric2 = reset_metrics((metric1, metric2))
        torch.testing.assert_close(metric1.compute(), torch.tensor(0.0))
        torch.testing.assert_close(metric2.compute(), torch.tensor(0.0))

    def test_to_device(self) -> None:
        metric1, metric2 = DummySumMetric(), DummySumMetric()
        self.assertEqual(metric1.device.type, "cpu")
        self.assertEqual(metric2.device.type, "cpu")
        if torch.cuda.is_available():
            cuda = torch.device("cuda")
            metric1, metric2 = to_device((metric1, metric2), cuda)
            self.assertTrue(metric1.device, torch.device("cuda"))
            self.assertTrue(metric2.device, torch.device("cuda"))
            metric1, metric2 = to_device((metric1, metric2), torch.device("cpu"))
            self.assertTrue(metric1.device, torch.device("cpu"))
            self.assertTrue(metric2.device, torch.device("cpu"))
