import numpy as np, json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for k, (t, ttl) in enumerate([("nobase", "Part 1: no baseline"), ("value", "Part 2: learned value baseline")]):
    for s in range(5):
        r = np.load(f"results/cp_{t}_s{s}_train_rewards.npy"); ax[k].plot(np.arange(99, len(r)), np.convolve(r, np.ones(100) / 100, "valid"), label=f"seed {s}")
    ax[k].set_title(f"CartPole-v1, {ttl}"); ax[k].set_xlabel("Training episode"); ax[k].grid(alpha=.3); ax[k].legend()
ax[0].set_ylabel("Moving average (100 episodes)"); plt.tight_layout(); plt.savefig("figures/cartpole_seeds_comparison.png", dpi=150)
rows = []
for t in ("nobase", "value"):
    R = [json.load(open(f"results/cp_{t}_s{s}.json")) for s in range(5)]
    rows.append(dict(method=t, eval_means=[r["eval_mean"] for r in R], eval_stds=[r["eval_std"] for r in R],
                     grad_norm_mean=float(np.mean([r["train_grad_norm_mean"] for r in R])), grad_norm_std_within_run=float(np.mean([r["train_grad_norm_std"] for r in R])),
                     last100=[r["train_last100_mean"] for r in R]))
json.dump(rows, open("results/cartpole_summary.json", "w"), indent=1); print(rows)
