"""Simple bigram language model expressed using log probabilities."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List


class BigramLanguageModel:
    """Bigram language model with fixed probabilities."""

    def __init__(self) -> None:
        # Vocabulary used by the model
        self.vocabulary: List[str] = [
            "the",
            "cat",
            "sat",
            "on",
            "mat",
        ]

        # Prior log probabilities for the first word in a sequence
        prior_probs: Dict[str, float] = {
            "the": 0.4,
            "cat": 0.2,
            "sat": 0.1,
            "on": 0.1,
            "mat": 0.2,
        }
        self.prior: Dict[str, float] = {
            w: math.log(p) for w, p in prior_probs.items()
        }

        # Conditional log probabilities P(current | previous)
        conditional_probs: Dict[str, Dict[str, float]] = {
            "the": {
                "cat": 0.4,
                "mat": 0.3,
                "sat": 0.1,
                "on": 0.1,
                "the": 0.1,
            },
            "cat": {
                "sat": 0.3,
                "on": 0.3,
                "the": 0.2,
                "mat": 0.1,
                "cat": 0.1,
            },
            "sat": {
                "on": 0.4,
                "the": 0.2,
                "mat": 0.2,
                "cat": 0.1,
                "sat": 0.1,
            },
            "on": {
                "the": 0.5,
                "mat": 0.2,
                "cat": 0.1,
                "sat": 0.1,
                "on": 0.1,
            },
            "mat": {
                "the": 0.3,
                "cat": 0.2,
                "on": 0.2,
                "sat": 0.2,
                "mat": 0.1,
            },
        }
        self.conditional: Dict[str, Dict[str, float]] = {
            prev: {w: math.log(p) for w, p in dist.items()}
            for prev, dist in conditional_probs.items()
        }

    def log_probability(self, word: str, prev_word: str | None = None) -> float:
        """Return log P(word | prev_word)."""
        if prev_word is None:
            return self.prior.get(word, float("-inf"))
        return self.conditional.get(prev_word, {}).get(word, float("-inf"))

    def probability(self, word: str, prev_word: str | None = None) -> float:
        """Return P(word | prev_word)."""
        logp = self.log_probability(word, prev_word)
        if logp == float("-inf"):
            return 0.0
        return math.exp(logp)

    def sequence_probability(self, words: Iterable[str]) -> float:
        """Return the probability of a word sequence."""
        prob = 1.0
        prev_word = None
        for word in words:
            prob *= self.probability(word, prev_word)
            prev_word = word
        return prob

    def sequence_log_probability(self, words: Iterable[str]) -> float:
        """Return log probability of a word sequence."""
        logp = 0.0
        prev_word = None
        for word in words:
            logp += self.log_probability(word, prev_word)
            prev_word = word
        return logp
