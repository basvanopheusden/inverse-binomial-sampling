import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bigram_model import BigramLanguageModel


def test_logprobabilities_sum_to_one_and_vocab():
    model = BigramLanguageModel()
    # Prior should sum to 1 when exponentiated
    from math import exp, isclose

    total = sum(exp(lp) for lp in model.prior.values())
    assert isclose(total, 1.0, rel_tol=1e-12)

    # Each conditional distribution should sum to 1 when exponentiated
    for prev_word, dist in model.conditional.items():
        assert prev_word in model.vocabulary
        for word in dist.keys():
            assert word in model.vocabulary
        total = sum(exp(lp) for lp in dist.values())
        assert isclose(total, 1.0, rel_tol=1e-12)


def test_sequence_probability():
    model = BigramLanguageModel()
    seq = ["the", "cat", "sat"]
    # Compute probability manually
    import math
    expected = (
        math.exp(model.prior[seq[0]])
        * math.exp(model.conditional[seq[0]][seq[1]])
        * math.exp(model.conditional[seq[1]][seq[2]])
    )
    assert abs(model.sequence_probability(seq) - expected) < 1e-12
    expected_logp = (
        model.prior[seq[0]]
        + model.conditional[seq[0]][seq[1]]
        + model.conditional[seq[1]][seq[2]]
    )
    assert abs(model.sequence_log_probability(seq) - expected_logp) < 1e-12
