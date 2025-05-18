# Inverse Binomial Sampling

This repository contains a minimal implementation of the *inverse binomial sampling* (IBS) method described in the paper ["Unbiased and Efficient Log-Likelihood Estimation with Inverse Binomial Sampling"](https://arxiv.org/abs/2001.03985v3).

IBS provides unbiased estimates of the log-likelihood for models that can generate stochastic responses but do not have a tractable likelihood function. For each observation, the simulator is repeatedly sampled until its output matches the recorded response. If `K` samples are required, the contribution of that observation to the log-likelihood is given by `psi(1) - psi(K)` where `psi` is the digamma function. Summing over all observations yields an unbiased estimate of the total log-likelihood.

## Matplotlib setup

To ensure all figures share the same look across experiments, this repository
provides a small helper in `setup_matplotlib.py`.  Import and call
`setup_matplotlib()` at the start of your scripts to apply a consistent style for
all plots.

## Fixed sampling helpers

The file `fixed_sampling.py` provides several simple log-probability estimators
for binary observations based on a fixed number of samples. Methods such as the
naive empirical mean, a ``+1`` adjustment, Laplace smoothing, and Jeffreys
smoothing are available via the ``FIXED_SAMPLING_METHODS`` dictionary.  In
addition, ``fixed_analytical_mean`` and ``fixed_analytical_variance`` can be
used to compute the expected value and variance of these estimators given the
true success probability and number of samples ``M``.


## Animated demonstration

A basic animation script, `animate_inverse_sampling.py`, is provided to
visualise how inverse sampling stops as soon as a matching model response
is observed. The script compares this behaviour to a fixed-sample approach
and displays the total number of samples used by each method. Run it with

```bash
python animate_inverse_sampling.py
```

which will open a window showing the animation.
