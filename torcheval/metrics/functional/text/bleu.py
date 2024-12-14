# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections import Counter as counter
from typing import Counter, Optional, Sequence, Tuple, Union

import torch


def bleu_score(
    input: Union[str, Sequence[str]],
    target: Sequence[Union[str, Sequence[str]]],
    n_gram: int = 4,
    weights: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute BLEU score given translations and references for each translation.
    Its class version is ``torcheval.metrics.texBLEUScore``.

    Args:
        input: Translations to score.
        target: List of references for each translation. Requires len(input) = len(target)
        n_gram: Maximum n-gram to use when computing BLEU score. Can be 1, 2, 3, or 4.
        weights: Optional weight distribution of n-grams. Requires len(weights) = n_gram. If unspecified,
            will use uniform weights.

        Examples:
            >>> import torch
            >>> from torcheval.metrics.functional.text import bleu_score
            >>> candidates = ["the squirrel is eating the nut"]
            >>> references = [["a squirrel is eating a nut", "the squirrel is eating a tasty nut"]]
            >>> bleu_score(candidates, references, n_gram=4)
            tensor(0.53728497)
            >>> candidates = ["the squirrel is eating the nut", "the cat is on the mat"]
            >>> references = [["a squirrel is eating a nut", "the squirrel is eating a tasty nut"], ["there is a cat on the mat", "a cat is on the mat"]]
            >>> bleu_score(candidates, references, n_gram=4)
            tensor(0.65341892)
    """
    (
        input_len,
        target_len,
        matches_by_order,
        possible_matches_by_order,
    ) = _bleu_score_update(
        input,
        target,
        n_gram,
        device,
    )

    return _bleu_score_compute(
        input_len,
        target_len,
        matches_by_order,
        possible_matches_by_order,
        n_gram,
        weights,
    )


def _bleu_score_update(
    input: Union[str, Sequence[str]],
    target: Sequence[Union[str, Sequence[str]]],
    n_gram: int,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ = [input] if isinstance(input, str) else input
    target_ = [[tgt] if isinstance(tgt, str) else tgt for tgt in target]

    if len(input_) != len(target_):
        raise ValueError(
            f"Input and target corpus should have same sizes, but input corpus size = {len(input_)}, target corpus size = {len(target_)} "
        )

    input_len = torch.tensor(0, device=device)
    target_len = torch.tensor(0, device=device)
    matches_by_order = torch.zeros(n_gram, device=device)
    possible_matches_by_order = torch.zeros(n_gram, device=device)

    for candidate, references in zip(input_, target_):
        candidate_tokenized = candidate.split()
        references_tokenized = [ref.split() for ref in references]

        len_candidate = len(candidate_tokenized)
        len_reference = min(
            [len(ref) for ref in references_tokenized],
            key=lambda ref_len: (abs(ref_len - len_candidate), ref_len),
        )
        input_len += len_candidate
        target_len += len_reference

        candidate_ngram_counter = _get_ngrams(candidate_tokenized, n_gram)
        reference_ngram_counter = counter()
        for ref in references_tokenized:
            reference_ngram_counter |= _get_ngrams(ref, n_gram)
        overlap = candidate_ngram_counter & reference_ngram_counter

        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]

        for i in range(n_gram):
            if len_candidate - i > 0:
                possible_matches_by_order[i] += len_candidate - i

    if torch.min(possible_matches_by_order) == 0:
        raise ValueError(
            f"the input is too short to find all n-gram matches with n_gram={n_gram}"
        )

    return input_len, target_len, matches_by_order, possible_matches_by_order


def _bleu_score_compute(
    input_len: torch.Tensor,
    target_len: torch.Tensor,
    matches_by_order: torch.Tensor,
    possible_matches_by_order: torch.Tensor,
    n_gram: int,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if weights is not None and n_gram != weights.size(dim=0):
        raise ValueError(
            f"the length of weights should equal n_gram, got len(weights)={weights.size(dim=0)}, n_gram={n_gram}"
        )

    if weights is None:
        weights = torch.tensor([1 / n_gram] * n_gram, device=input_len.device)

    precisions = matches_by_order / possible_matches_by_order
    geometric_mean = torch.exp(torch.sum(weights * torch.log(precisions)))

    brevity_penalty = _calc_brevity_penalty(input_len, target_len)

    return brevity_penalty * geometric_mean


def _calc_brevity_penalty(
    input_len: torch.Tensor, target_len: torch.Tensor
) -> torch.Tensor:
    if input_len > target_len:
        return torch.tensor(1.0, device=input_len.device)
    else:
        return torch.exp(1 - target_len / input_len)


def _get_ngrams(sentence: Sequence[str], n_gram: int) -> Counter[str]:
    """
    Args:
        sentence: text from which we get n-grams
        n_gram: length of n-gram
    """
    if n_gram not in [1, 2, 3, 4]:
        raise ValueError(f"n_gram should be 1, 2, 3, or 4, got {n_gram}.")
    ngram_counts = counter()
    for n_val in range(1, n_gram + 1):
        for i in range(0, len(sentence) - n_val + 1):
            ngram = tuple(sentence[i : i + n_val])
            ngram_counts[ngram] += 1
    return ngram_counts
