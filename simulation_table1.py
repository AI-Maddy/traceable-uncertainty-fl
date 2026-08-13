"""
Reproduces Table I of

    M. Vivekanandan and S. Ramasamy,
    "Federated Learning Has No Concept of Traceable Uncertainty."

Five aggregation rules are evaluated on a synthetic scalar estimand with two
regimes and one adversary. The point of the table is the contrast between
rows 2-4 and row 5, not the absolute numbers.

-----------------------------------------------------------------------------
Why row 4 exists
-----------------------------------------------------------------------------
Row 4 ("Inverse-variance + chi-squared gate") is the naive metrological
transfer, and it FAILS. It excludes the honest rare-regime peer 99.6% of the
time, because that peer's residual measured in units of its own WITHIN-REGIME
uncertainty is large. Within-regime uncertainty does not describe
between-regime distance. The residual test alone therefore confounds integrity
with competence -- the same defect we attribute to Krum.

Row 5 is the repair: the observation triple (y_k, u_k, c_k) carries declared
operating conditions. Integrity is tested WITHIN a declared stratum;
heterogeneity is tested ACROSS strata. Failing the within-stratum test is
dishonesty. Failing the across-strata test is information.

Run:  python simulation_table1.py
"""
import numpy as np
from scipy import stats

SEED = 20260805
N_TRIALS = 2000
THETA_STAR = 0.300


# ---------- baselines -------------------------------------------------------
def krum(v, n_byz):
    K = len(v); m = max(K - n_byz - 2, 1)
    D = (v[:, None] - v[None, :]) ** 2
    return int(np.argmin([np.sort(D[i])[1:m + 1].sum() for i in range(K)]))


def trimmed_keep(v, frac):
    K = len(v); t = int(np.floor(frac * K)); o = np.argsort(v)
    return np.sort(o[t:K - t] if t else o)


def ivw(y, u, mask):
    w = np.where(mask, 1.0 / u ** 2, 0.0)
    return (w * y).sum() / w.sum(), 1.0 / np.sqrt(w.sum())


# ---------- row 4: naive transfer (fails) -----------------------------------
def flat_gate(y, u, alpha=1e-3, het_alpha=0.01):
    """Inverse-variance weighting under a chi-squared consistency gate, with
    NO use of declared conditions. Iteratively drops the peer whose residual,
    in units of its own claimed uncertainty, is most inconsistent with
    honesty. This is the obvious thing to try, and it does not work."""
    mask = np.ones(len(y), dtype=bool)
    crit = stats.norm.ppf(1 - alpha / 2)
    for _ in range(len(y)):
        th, _ = ivw(y, u, mask)
        r = np.where(mask, (y - th) / u, 0.0)
        j = int(np.argmax(np.abs(r)))
        if np.abs(r[j]) > crit and mask.sum() > 2:
            mask[j] = False
        else:
            break
    theta, _ = ivw(y, u, mask)
    idx = np.where(mask)[0]
    chi2 = (((y[idx] - theta) / u[idx]) ** 2).sum()
    dof = max(len(idx) - 1, 1)
    heterogeneous = stats.chi2.sf(chi2, dof) < het_alpha
    return theta, heterogeneous, mask


# ---------- row 5: stratified integrity gate --------------------------------
def stratified_gate(y, u, c, alpha=1e-3, het_alpha=0.01, w_regime=None):
    """Within each declared stratum: iteratively drop peers whose residual,
    in units of their own claimed uncertainty, is inconsistent with honesty.
    Across strata: chi-squared test for a common value."""
    mask = np.ones(len(y), dtype=bool)
    strata = np.unique(c)
    crit = stats.norm.ppf(1 - alpha / 2)

    for s in strata:
        idx = np.where(c == s)[0]
        if len(idx) < 3:                     # cannot test a stratum of 1-2
            continue
        local = np.ones(len(idx), dtype=bool)
        for _ in range(len(idx)):
            th, _ = ivw(y[idx], u[idx], local)
            r = np.where(local, (y[idx] - th) / u[idx], 0.0)
            j = int(np.argmax(np.abs(r)))
            if np.abs(r[j]) > crit and local.sum() > 2:
                local[j] = False
            else:
                break
        mask[idx] = local

    # per-stratum consensus from surviving peers
    th_s, u_s, keys = [], [], []
    for s in strata:
        idx = np.where((c == s) & mask)[0]
        if len(idx) == 0:
            continue
        t, uu = ivw(y[idx], u[idx], np.ones(len(idx), dtype=bool))
        th_s.append(t); u_s.append(uu); keys.append(s)
    th_s = np.array(th_s); u_s = np.array(u_s)

    # across-strata homogeneity test
    if len(th_s) > 1:
        pooled, _ = ivw(th_s, u_s, np.ones(len(th_s), dtype=bool))
        chi2 = (((th_s - pooled) / u_s) ** 2).sum()
        dof = len(th_s) - 1
        birge = np.sqrt(chi2 / dof)
        heterogeneous = stats.chi2.sf(chi2, dof) < het_alpha
    else:
        birge, heterogeneous = 1.0, False

    # population estimate: reweight strata by their population share
    if w_regime is not None:
        wts = np.array([w_regime[k] for k in keys]); wts = wts / wts.sum()
        theta = float((wts * th_s).sum())
    else:
        theta, _ = ivw(th_s, u_s, np.ones(len(th_s), dtype=bool))
    return theta, birge, heterogeneous, mask


# ---------- trial -----------------------------------------------------------
def one_trial(rng, n_A=18, theta_B=3.0, p_B=0.10, sigma_A=0.20,
              sigma_B=0.50, adv_bias=0.50, adv_u=0.05):
    theta_star = (1 - p_B) * 0.0 + p_B * theta_B
    y, u, c, kind = [], [], [], []
    for _ in range(n_A):
        y.append(rng.normal(0, sigma_A)); u.append(sigma_A)
        c.append("A"); kind.append("A")
    y.append(rng.normal(theta_B, sigma_B)); u.append(sigma_B)
    c.append("B"); kind.append("rare")
    y.append(adv_bias); u.append(adv_u)
    c.append("A"); kind.append("adv")
    y, u, c, kind = map(np.array, (y, u, c, kind))
    i_r = int(np.where(kind == "rare")[0][0])
    i_a = int(np.where(kind == "adv")[0][0])

    res = {"theta_star": theta_star}
    res["mean"] = (y.mean(), True, True, False)
    k = trimmed_keep(y, 0.10)
    res["trimmed"] = (y[k].mean(), i_r in k, i_a in k, False)
    s = krum(y, 2)
    res["krum"] = (y[s], s == i_r, s == i_a, False)
    th_f, het_f, m_f = flat_gate(y, u)
    res["flat"] = (th_f, bool(m_f[i_r]), bool(m_f[i_a]), het_f)
    th, birge, het, m = stratified_gate(y, u, c, w_regime={"A": 1 - p_B, "B": p_B})
    res["strat"] = (th, bool(m[i_r]), bool(m[i_a]), het)
    res["birge"] = birge
    return res


def main(n=N_TRIALS, seed=SEED):
    rng = np.random.default_rng(seed)
    ms = ["mean", "trimmed", "krum", "flat", "strat"]
    A = {m: {"r": [], "a": [], "d": [], "e": []} for m in ms}
    B = []
    for _ in range(n):
        t = one_trial(rng); B.append(t["birge"])
        for m in ms:
            th, rk, ak, det = t[m]
            A[m]["r"].append(rk); A[m]["a"].append(ak)
            A[m]["d"].append(det); A[m]["e"].append(abs(th - t["theta_star"]))
    nm = {"mean": "Mean (FedAvg)",
          "trimmed": "Trimmed mean",
          "krum": "Krum",
          "flat": "Inverse-variance + chi^2 gate",
          "strat": "Stratified by declared c_k"}
    print(f"Table I  --  {n} trials, seed {seed}, theta* = {THETA_STAR:.3f}\n")
    print(f"{'Rule':<32}{'rare kept':>11}{'adv kept':>10}"
          f"{'heterog. signalled':>20}{'|err|':>9}")
    print("-" * 82)
    for m in ms:
        print(f"{nm[m]:<32}{100*np.mean(A[m]['r']):>10.1f}%"
              f"{100*np.mean(A[m]['a']):>9.1f}%"
              f"{100*np.mean(A[m]['d']):>19.1f}%{np.mean(A[m]['e']):>9.3f}")
    print(f"\nacross-strata Birge ratio: median {np.median(B):.1f} "
          f"[{np.percentile(B,5):.1f}, {np.percentile(B,95):.1f}]")


if __name__ == "__main__":
    main()
