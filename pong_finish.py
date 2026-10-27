"""Evaluate a saved Pong policy (500 episodes) and make plots/json. Used because the first run was interrupted before evaluation."""
import json, numpy as np, torch, pg
tag = "pong_nobase"
ag = pg.Agent("pong", "none", 1e-3, 1e-3, 0); ag.pi.load_state_dict(torch.load(f"results/{tag}_policy.pt")); env = ag.make_env()
rewards = np.load(f"results/{tag}_train_rewards.npy"); ev = pg.evaluate(ag, env, 500); np.save(f"results/{tag}_eval.npy", ev)
pg.plots(tag, rewards, ev, "Pong-v5 (no baseline)")
json.dump(dict(n_train_episodes=len(rewards), train_last100_mean=float(rewards[-100:].mean()), eval_mean=float(ev.mean()), eval_std=float(ev.std())), open(f"results/{tag}.json", "w"), indent=1)
print("done", ev.mean(), ev.std())
