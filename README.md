# Traceable uncertainty in federated learning — simulation code

Code reproducing **Table I** of:

> M. Vivekanandan and S. Ramasamy, "Federated Learning Has No Concept of
> Traceable Uncertainty," *submitted*.

## What this is, and what it is not

This is a **synthetic scalar construction**, not a federated training
benchmark. It exists to make one confound concrete and to record one failed
repair. It establishes no new result; the prior work that establishes the
underlying phenomenon is cited in Section II-D of the paper.

## Setup

A scalar estimand with two regimes:

| | population weight | true value | reported `u_k` |
|---|---|---|---|
| Regime A | 0.9 | 0.0 | 0.20 |
| Regime B | 0.1 | 3.0 | 0.50 |

giving `theta* = 0.300`. Twenty participants: 18 honest in regime A, one
honest in regime B (the *rare* peer), and one adversary reporting `y = 0.50`
while claiming `u = 0.05`. 2000 independent trials.

Each participant reports the triple `(y_k, u_k, c_k)` — estimate, claimed
standard uncertainty, declared operating condition. Rules 1–4 ignore `c_k`;
rule 5 uses it.

## The five rules

1. **Mean (FedAvg)** — keeps everything, including the adversary.
2. **Trimmed mean** — 10% symmetric trim.
3. **Krum** — single-point selection, `n_byz = 2`.
4. **Inverse-variance + chi-squared gate** — the naive metrological transfer.
   Iteratively drops the peer whose residual, in units of *its own* claimed
   uncertainty, is most inconsistent with honesty. **This fails**, and the
   failure is the point: within-regime uncertainty does not describe
   between-regime distance, so the honest rare peer is excluded ~99.6% of the
   time despite perfect integrity.
5. **Stratified by declared `c_k`** — integrity tested *within* a declared
   stratum, heterogeneity tested *across* strata. Within-stratum
   inconsistency indicates dishonesty; across-stratum inconsistency is not an
   error but the informative output.

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python simulation_table1.py
```

Runs in a few seconds. Output is deterministic under seed `20260805` and
should match `expected_output.txt` byte for byte on the pinned versions.

## Known limitation

Rule 5 does not stop an adversary that declares a **novel** stratum rather
than joining A: a stratum of one cannot be tested against itself, and
detection falls to 0.0%. This is discussed in Section V of the paper and is
regarded there as the central unsolved difficulty, not a detail. Closing it
requires attestation of declared conditions, which is outside the statistical
layer.

## Citing

Please cite the paper. If you need to cite the code specifically, use the
archived release DOI (see the badge above once the Zenodo release is made).

## License

MIT — see `LICENSE`.
