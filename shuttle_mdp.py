"""Part 2: Shuttle dispatching MDP -- enumeration (finite horizon), value iteration, policy iteration.

Cost-minimisation formulation (value = expected discounted COST; lower is better).
State  s in {0..200}: customers waiting at the beginning of a decision epoch.
Action a in {0 (hold), 1 (dispatch)}.
Dynamics: if a=1, min(s,K) customers leave and a cost cf is paid; the customers still waiting
          (r = s - a*min(s,K)) cost ch each; then A~Unif{1..5} arrive; s' = min(r + A, 200).
Immediate cost  c(s,a) = cf*a + ch*(s - a*min(s,K)).
"""
import itertools, json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K, CF, CH, SMAX, GAMMA = 15, 100.0, 2.0, 200, 0.95
ARR = np.arange(1, 6); PARR = np.full(5, 0.2)


def build_mdp(K=K, cf=CF, ch=CH, smax=SMAX, arr=ARR, parr=PARR):
    """Return cost c[a,s] and transition P[a,s,s']."""
    n = smax + 1
    c = np.zeros((2, n)); P = np.zeros((2, n, n))
    for s in range(n):
        for a in (0, 1):
            r = s - a * min(s, K)
            c[a, s] = cf * a + ch * r
            for A, p in zip(arr, parr):
                P[a, s, min(r + A, smax)] += p
    return c, P


def backward_induction(c, P, T, gamma=GAMMA):
    """Exact finite-horizon DP == expected cost of the best of all 2^T action sequences (enumeration
    by dynamic programming / memoised recursion). V_T = 0. Returns V_0 and V_t for all t, policy_0."""
    n = c.shape[1]; V = np.zeros(n); Vs = [V]; pol = None
    for t in range(T - 1, -1, -1):
        Q = c + gamma * P @ V                      # (2,n)
        pol = (Q[1] < Q[0] - 1e-12).astype(int)     # ties -> hold
        V = Q.min(0); Vs.append(V)
    return V, pol


def brute_force(s, T, c, P, gamma=GAMMA):
    """Literal enumeration: recurse over every action and every arrival (no memoisation). Tiny T only."""
    if T == 0: return 0.0
    best = np.inf
    for a in (0, 1):
        nxt = np.flatnonzero(P[a, s]); v = c[a, s]
        v += gamma * sum(P[a, s, j] * brute_force(j, T - 1, c, P, gamma) for j in nxt)
        best = min(best, v)
    return best


def value_iteration(c, P, gamma=GAMMA, tol=1e-10, maxit=100000):
    V = np.zeros(c.shape[1]); hist = []
    for it in range(maxit):
        Q = c + gamma * P @ V; Vn = Q.min(0)
        d = np.abs(Vn - V).max(); hist.append(d); V = Vn
        if d < tol: break
    Q = c + gamma * P @ V
    return V, (Q[1] < Q[0] - 1e-9).astype(int), it + 1, hist


def eval_policy(pol, c, P, gamma=GAMMA):
    n = c.shape[1]; idx = np.arange(n)
    Pp = P[pol, idx]; cp = c[pol, idx]
    return np.linalg.solve(np.eye(n) - gamma * Pp, cp)


def policy_iteration(c, P, gamma=GAMMA, init=None):
    n = c.shape[1]; pol = np.zeros(n, int) if init is None else init.copy(); it = 0; hist = []
    while True:
        it += 1; V = eval_policy(pol, c, P, gamma)
        Q = c + gamma * P @ V
        new = pol.copy()
        better = Q[1 - pol, np.arange(n)] < Q[pol, np.arange(n)] - 1e-9   # switch only on strict improvement
        new[better] = 1 - pol[better]
        hist.append(int(better.sum()))
        if not better.any(): return V, pol, it, hist
        pol = new


def simulate(pol, s0, gamma=GAMMA, K=K, cf=CF, ch=CH, smax=SMAX, H=400, n=20000, seed=0):
    """Vectorised Monte-Carlo estimate of the discounted cost of `pol` from s0 (gamma^400 ~ 1e-9 truncation)."""
    rng = np.random.default_rng(seed); pol = np.asarray(pol)
    s = np.full(n, s0); tot = np.zeros(n); g = 1.0
    for t in range(H):
        a = pol[s]; r = s - a * np.minimum(s, K)
        tot += g * (cf * a + ch * r); g *= gamma
        s = np.minimum(r + rng.integers(1, 6, n), smax)
    return tot.mean(), tot.std(ddof=1) / np.sqrt(n)


def thr(pol): return int(np.argmax(pol == 1)) if pol.any() else None


if __name__ == "__main__":
    log = {}
    c, P = build_mdp(); n = SMAX + 1
    # ---------------- solve ----------------
    V500, pol500 = backward_induction(c, P, 500)
    Vvi, polvi, itvi, hvi = value_iteration(c, P)
    Vpi, polpi, itpi, hpi = policy_iteration(c, P)
    log.update(vi_iters=itvi, pi_iters=itpi, pi_switches_per_iter=hpi, threshold_vi=thr(polvi), threshold_pi=thr(polpi),
               threshold_T500=thr(pol500), V0_at_0=[V500[0], Vvi[0], Vpi[0]])
    # ---------------- plots ----------------
    s = np.arange(n)
    for name, V, ttl in [("part2a_enumeration_T500", V500, "(a) Enumeration, T=500: optimal $V_0(s)$"),
                         ("part2b_value_iteration", Vvi, "(b) Value iteration ($T=\\infty$): optimal $V(s)$")]:
        plt.figure(figsize=(6, 4)); plt.plot(s, V, lw=2); plt.xlabel("Number of customers waiting"); plt.ylabel("Optimal expected discounted cost")
        plt.title(ttl); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(f"figures/{name}.png", dpi=150); plt.close()
    plt.figure(figsize=(6, 3.5)); plt.step(s, polpi, where="post", lw=2); plt.yticks([0, 1], ["hold (0)", "dispatch (1)"])
    plt.xlabel("Number of customers waiting"); plt.title("(c) Policy iteration: optimal policy"); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig("figures/part2c_policy_iteration.png", dpi=150); plt.close()
    plt.figure(figsize=(6, 3.5)); plt.semilogy(hvi); plt.xlabel("VI iteration"); plt.ylabel("max |V_{k+1}-V_k|"); plt.title("Value-iteration convergence")
    plt.axhline(0.95 ** 0 * 1e-10, ls="--", c="r"); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig("figures/part2_vi_convergence.png", dpi=150); plt.close()
    # ---------------- tests ----------------
    T = {}
    # T1 transition matrices are stochastic, costs as expected at boundaries
    T["rows_sum_to_one"] = float(np.abs(P.sum(2) - 1).max())
    T["cost_s0_hold,dispatch"] = [c[0, 0], c[1, 0]]
    T["cost_s10_dispatch(partial batch)"] = c[1, 10]          # = cf, nobody left
    T["cost_s20_dispatch"] = c[1, 20]                         # = cf + ch*5
    T["boundary_s200_hold_goes_to_200"] = float(P[0, 200, 200])
    # T2 literal enumeration vs DP for tiny horizons
    errs = []
    for Th in (1, 2, 3, 4):
        Vd, _ = backward_induction(c, P, Th)
        for s0 in (0, 3, 14, 15, 16, 40, 199, 200):
            errs.append(abs(brute_force(s0, Th, c, P) - Vd[s0]))
    T["bruteforce_vs_DP_max_abs_err(T<=4)"] = max(errs)
    # T3 finite horizon -> infinite horizon (gamma^500 ~ 6.9e-12)
    T["max|V_T500 - V_VI|"] = float(np.abs(V500 - Vvi).max())
    T["max|V_PI - V_VI|"] = float(np.abs(Vpi - Vvi).max())
    T["policy_VI==policy_PI"] = bool((polvi == polpi).all()); T["policy_T500==policy_VI"] = bool((pol500 == polvi).all())
    # T4 Bellman optimality residual
    T["bellman_residual"] = float(np.abs((c + GAMMA * P @ Vvi).min(0) - Vvi).max())
    # T5 structure: monotone V and threshold policy
    T["V_nondecreasing"] = bool((np.diff(Vvi) >= -1e-9).all()); T["policy_is_threshold"] = bool(np.all(np.diff(polvi) >= 0))
    T["policy_at_s0"] = int(polvi[0])
    # T6 sensitivity
    sens = {}
    for tag, kw in {"cf=0": dict(cf=0.0), "ch=0": dict(ch=0.0), "cf=50": dict(cf=50.0), "cf=100 (base)": {}, "cf=200": dict(cf=200.0),
                    "cf=400": dict(cf=400.0), "ch=1": dict(ch=1.0), "ch=4": dict(ch=4.0), "K=5": dict(K=5), "K=30": dict(K=30)}.items():
        cc, PP = build_mdp(**kw); V_, p_, _, _ = value_iteration(cc, PP); sens[tag] = dict(threshold=thr(p_), V0=float(V_[0]))
    T["sensitivity"] = sens
    # T7 gamma -> 0 : myopic policy
    cc, PP = build_mdp(); V_, p_, _, _ = value_iteration(cc, PP, gamma=0.0)
    T["gamma=0 threshold (None = never dispatch, since cf=100 > ch*K=30)"] = thr(p_)
    # T8 Monte Carlo vs analytic, and optimal beats baselines (common random numbers via seed)
    mc = {}
    for s0 in (0, 20, 100):
        m, se = simulate(polvi, s0, seed=100 + s0); mc[s0] = dict(MC=m, SE=se, analytic=float(Vvi[s0]), z=(m - Vvi[s0]) / se)
    T["monte_carlo"] = mc
    base = {}
    for name, p_ in {"optimal": polvi, "never": np.zeros(n, int), "always": np.ones(n, int),
                     "thr=K": (s >= K).astype(int), "thr=opt-5": (s >= thr(polvi) - 5).astype(int), "thr=opt+10": (s >= thr(polvi) + 10).astype(int)}.items():
        base[name] = dict(V0_analytic=float(eval_policy(p_, c, P)[0]), V0_MC=simulate(p_, 0, seed=7)[0])
    T["baseline_policies"] = base
    # T9 first 10 steps with deterministic arrivals (A=3) -- readable trace
    cd, Pd = build_mdp(arr=np.array([3]), parr=np.array([1.0])); Vd, pold, _, _ = value_iteration(cd, Pd)
    st, trace = 0, []
    for t in range(10):
        a = int(pold[st]); r = st - a * min(st, K); trace.append(dict(t=t, s=st, a=a, cost=cd[a, st], V=round(float(Vd[st]), 3))); st = min(r + 3, SMAX)
    T["trace_det_arrivals3"] = trace; T["threshold_det_arrivals3"] = thr(pold)
    # T10 policy iteration from different initialisations converges to same policy
    inits = {"never": np.zeros(n, int), "always": np.ones(n, int), "random": np.random.default_rng(1).integers(0, 2, n)}
    T["PI_inits_same_policy"] = {k: bool((policy_iteration(c, P, init=v)[1] == polpi).all()) for k, v in inits.items()}
    T["PI_iters_by_init"] = {k: policy_iteration(c, P, init=v)[2] for k, v in inits.items()}
    # sensitivity plot
    cfs = [0, 25, 50, 100, 150, 200, 300, 400]; ths = []
    for cf_ in cfs:
        cc, PP = build_mdp(cf=float(cf_)); ths.append(thr(value_iteration(cc, PP)[1]))
    plt.figure(figsize=(6, 3.5)); plt.plot(cfs, ths, "o-"); plt.xlabel("dispatch cost $c_f$"); plt.ylabel("dispatch threshold (customers)")
    plt.title("Sensitivity: threshold vs $c_f$"); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig("figures/part2_sensitivity_cf.png", dpi=150); plt.close()
    T["threshold_vs_cf"] = dict(zip(cfs, ths))
    plt.figure(figsize=(6, 3.5)); plt.plot(s[:60], Vvi[:60], label="VI"); plt.plot(s[:60], V500[:60], "--", label="Enumeration T=500")
    for nm, p_ in [("never", np.zeros(n, int)), ("thr=K", (s >= K).astype(int))]: plt.plot(s[:60], eval_policy(p_, c, P)[:60], label=nm)
    plt.legend(); plt.xlabel("customers waiting"); plt.ylabel("expected discounted cost"); plt.title("Optimal vs naive policies"); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig("figures/part2_policy_comparison.png", dpi=150); plt.close()
    log["tests"] = T
    json.dump(log, open("results/part2_results.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(log, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
