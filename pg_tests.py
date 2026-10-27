"""Validation tests for the policy-gradient code. Usage: python pg_tests.py [fast|negated]
fast    : returns, log-prob, gradient-unbiasedness/variance (bandit with known exact gradient), preprocessing, determinism, Pong speed
negated : sanity experiment -- train CartPole with the reward sign flipped; the agent should learn to FAIL fast."""
import sys, json, time, numpy as np, torch, torch.nn.functional as F
import pg

def naive_returns(r, g, reset=False):
    out = []
    for t in range(len(r)):
        tot = 0.0
        for k in range(t, len(r)):
            if reset and k > t and r[k - 1] != 0: break      # segment ends right after a scored point
            tot += g ** (k - t) * r[k]
        out.append(tot)
    return np.array(out, dtype=np.float32)

def fast():
    R = {}
    rng = np.random.default_rng(0); torch.manual_seed(0)
    # 1. discounted returns vs O(T^2) reference, plus hand-checked value
    errs = [np.abs(pg.discounted_returns(r, 0.95) - naive_returns(r, 0.95)).max() for r in (rng.normal(size=50) for _ in range(20))]
    R["returns_vs_naive_max_err"] = float(max(errs))
    R["returns_hand_check [1,1,1],g=.5 -> [1.75,1.5,1]"] = pg.discounted_returns([1, 1, 1], .5).tolist()
    pong_r = np.zeros(60); pong_r[20] = -1; pong_r[45] = 1
    R["pong_returns_reset_vs_naive_max_err"] = float(np.abs(pg.discounted_returns(pong_r, .99, True) - naive_returns(pong_r, .99, True)).max())
    R["pong_no_leak_across_points (G at t=21..45 uses only +1 point)"] = bool(pg.discounted_returns(pong_r, .99, True)[21] > 0 and pg.discounted_returns(pong_r, .99, True)[19] < 0)
    # 2. log-prob correctness
    ag = pg.Agent("cartpole", "none", 1e-3, 1e-3, 0); x = torch.randn(7, 4); a = torch.randint(0, 2, (7,))
    R["cartpole_logp_vs_log_softmax"] = float((ag.logp(x, a) - torch.log(F.softmax(ag.pi(x), -1)[torch.arange(7), a])).abs().max())
    agp = pg.Agent("pong", "none", 1e-3, 1e-3, 0); xp = torch.randn(5, 6400); ap = torch.randint(0, 2, (5,)); pr = torch.sigmoid(agp.pi(xp).squeeze(-1))
    R["pong_logp_vs_sigmoid"] = float((agp.logp(xp, ap) - torch.log(torch.where(ap == 1, pr, 1 - pr))).abs().max())
    big = torch.zeros(1, 6400); agp.pi.f[2].bias.data.fill_(80.0); R["pong_logp_finite_at_saturation"] = bool(torch.isfinite(agp.logp(big, torch.tensor([0]))).all())
    # 3. REINFORCE gradient: unbiased and lower variance with baseline. 3-armed bandit, softmax(theta); exact grad = sum_a pi_a r_a (e_a - pi)
    theta = torch.tensor([0.2, -0.3, 0.1]); rew = torch.tensor([1.0, 3.0, 2.0]); pi_ = F.softmax(theta, 0)
    exact = (pi_[:, None] * rew[:, None] * (torch.eye(3) - pi_[None])).sum(0)
    N = 200000; acts = torch.multinomial(pi_, N, replacement=True); rr = rew[acts]
    score = torch.eye(3)[acts] - pi_[None]                      # grad log pi(a)
    for name, b in [("no baseline", 0.0), ("constant baseline = mean reward", float((pi_ * rew).sum())), ("badly-scaled baseline b=10", 10.0)]:
        g = score * (rr - b)[:, None]; m = g.mean(0); se = g.std(0) / np.sqrt(N)
        R[f"grad_estimator[{name}]"] = dict(mean=m.tolist(), exact=exact.tolist(), max_abs_z=float(((m - exact).abs() / se).max()),
                                          cosine=float(F.cosine_similarity(m, exact, dim=0)), per_sample_variance=float(g.var(0).sum()))
    # 4. preprocessing / env interface
    env = ag.make_env(); envp = agp.make_env(); o, _ = envp.reset(seed=0); f = pg.preprocess(o)
    R["preprocess_shape,min,max,frac_nonzero"] = [list(f.shape), float(f.min()), float(f.max()), float((f != 0).mean())]
    S, A, Rw = agp.run_episode(envp, seed=0, record=True)
    S = np.asarray(S); R["pong_first_state_is_zero_diff"] = bool((S[0] == 0).all()); R["pong_diff_values"] = sorted(set(np.unique(S).tolist()))
    R["pong_episode_len,reward_random_init"] = [len(A), float(sum(Rw))]; R["pong_action_meanings"] = envp.unwrapped.get_action_meanings()[2:4]
    R["pong_min_max_episode_reward_bounds_ok"] = bool(-21 <= sum(Rw) <= 21)
    # 5. determinism with the same seed
    def short(seed):
        ag_ = pg.Agent("cartpole", "value", 1e-3, 1e-3, seed); e = ag_.make_env(); out = []
        for ep in range(30):
            b = ag_.run_episode(e, seed=seed if ep == 0 else None); out.append(sum(b[2])); ag_.update([b])
        return out
    R["same_seed_identical_rewards"] = short(5) == short(5); R["different_seed_differs"] = short(5) != short(6)
    # 6. random-policy reference + Pong throughput
    rnd = [sum(pg.Agent("cartpole", "none", 1e-3, 1e-3, 0).run_episode(env, record=False)[2]) for _ in range(100)]
    R["cartpole_untrained_policy_mean,std"] = [float(np.mean(rnd)), float(np.std(rnd))]
    t0 = time.time(); n = 0
    for _ in range(2): n += len(agp.run_episode(envp, record=False)[2])
    R["pong_agent_steps_per_second_1cpu"] = n / (time.time() - t0)
    json.dump(R, open("results/pg_tests_fast.json", "w"), indent=1, default=str); print(json.dumps(R, indent=1, default=str))

def negated():
    orig = pg.discounted_returns; pg.discounted_returns = lambda r, g, reset=False: -orig(r, g, reset)
    out = {}
    for seed in (0, 1):
        ag = pg.Agent("cartpole", "none", 1e-3, 1e-3, seed); env = ag.make_env(); rs = []
        for ep in range(600):
            b = ag.run_episode(env, seed=seed if ep == 0 else None); rs.append(sum(b[2])); ag.update([b])
        out[f"seed{seed}"] = dict(first100=float(np.mean(rs[:100])), last100=float(np.mean(rs[-100:])))
    json.dump(out, open("results/pg_tests_negated.json", "w"), indent=1); print(out)

if __name__ == "__main__": {"fast": fast, "negated": negated}[sys.argv[1]]()
