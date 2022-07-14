# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torcheval.metrics.functional.aggregation.mean import mean
from torcheval.metrics.functional.aggregation.sum import sum

__all__ = ["mean", "sum"]
