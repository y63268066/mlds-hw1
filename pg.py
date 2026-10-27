"""Policy gradient (REINFORCE) with optional learned state-value baseline, in PyTorch.
Part 1 : --baseline none     Part 2 : --baseline value (non-constant, state-dependent baseline b(s)=V_phi(s))
Environments: CartPole-v1 (gamma=0.95) and ALE/Pong-v5 (gamma=0.99, actions restricted to {2:RIGHT, 3:LEFT}).
Usage examples:
  python pg.py --env cartpole --baseline none  --episodes 1500 --seed 0 --tag cp_nobase_s0
  python pg.py --env pong     --baseline value --episodes 3000 --batch 4 --lr 1e-3 --tag pong_value
"""
import argparse, json, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import gymnasium as gym, ale_py
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

gym.register_envs(ale_py)
torch.set_num_threads(1)


def preprocess(image):
    """prepro 210x160x3 uint8 frame into 6400 (80x80) 2D float array (function given in the assignment)"""
    image = image[35:195]; image = image[::2, ::2, 0]
    image = image.copy()
    image[image == 144] = 0; image[image == 109] = 0; image[image != 0] = 1
    return np.reshape(image.astype(np.float32).ravel(), [80, 80])


def discounted_returns(rewards, gamma, reset_on_nonzero=False):
    """Reward-to-go G_t = r_t + gamma*G_{t+1}. For Pong (reset_on_nonzero) the running sum is reset
    at every non-zero reward, i.e. each point (+1/-1) is its own game segment (standard Pong convention)."""
    out = np.zeros(len(rewards), dtype=np.float32); run = 0.0
    for t in reversed(range(len(rewards))):
        if reset_on_nonzero and rewards[t] != 0: run = 0.0
        run = rewards[t] + gamma * run; out[t] = run
    return out


class Net(nn.Module):
    """MLP. Policy: out=2 logits (CartPole) or out=1 logit for P(RIGHT) (Pong). Baseline: out=1 value."""
    def __init__(self, inp, hid, out):
        super().__init__(); self.f = nn.Sequential(nn.Linear(inp, hid), nn.ReLU(), nn.Linear(hid, out))
    def forward(self, x): return self.f(x)


class Agent:
    def __init__(self, env, baseline, lr, blr, seed, hidden=None):
        self.env_name = env; self.pong = env == "pong"
        self.gamma = 0.99 if self.pong else 0.95
        inp = 6400 if self.pong else 4; hid = hidden or (200 if self.pong else 128)
        torch.manual_seed(seed); np.random.seed(seed)
        self.pi = Net(inp, hid, 1 if self.pong else 2)
        self.v = Net(inp, hid, 1) if baseline == "value" else None
        self.opt = torch.optim.Adam(self.pi.parameters(), lr=lr)
        self.vopt = torch.optim.Adam(self.v.parameters(), lr=blr) if self.v is not None else None
        self.rng = np.random.default_rng(seed)

    def make_env(self):
        return gym.make("ALE/Pong-v5") if self.pong else gym.make("CartPole-v1")

    def logp(self, x, a):
        """log pi(a|x), numerically stable (logsigmoid / log_softmax)."""
        out = self.pi(x)
        if self.pong:
            z = out.squeeze(-1)                   # a=1 -> RIGHT(2), a=0 -> LEFT(3)
            return torch.where(a == 1, F.logsigmoid(z), F.logsigmoid(-z))
        return F.log_softmax(out, -1).gather(1, a[:, None]).squeeze(1)

    @torch.no_grad()
    def act(self, x, greedy=False):
        out = self.pi(torch.as_tensor(x, dtype=torch.float32)[None])
        if self.pong:
            p = torch.sigmoid(out)[0, 0].item()
            a = int(p > 0.5) if greedy else int(self.rng.random() < p)
            return a, (2 if a == 1 else 3)
        pr = F.softmax(out, -1)[0].numpy().astype(np.float64); pr /= pr.sum()
        a = int(pr.argmax()) if greedy else int(self.rng.choice(2, p=pr)); return a, a

    def run_episode(self, env, seed=None, greedy=False, record=True):
        obs, _ = env.reset(seed=seed); prev = None; S, A, R = [], [], []; done = False
        while not done:
            if self.pong:
                cur = preprocess(obs); x = cur - prev if prev is not None else np.zeros_like(cur); prev = cur; x = x.ravel()
            else: x = obs
            a, ea = self.act(x, greedy)
            obs, r, term, trunc, _ = env.step(ea); done = term or trunc
            if record: S.append(x.astype(np.int8) if self.pong else x.astype(np.float32)); A.append(a)
            R.append(r)
        return S, A, R

    def update(self, batch):
        """batch: list of (S,A,R). Loss = -(1/B) sum_episodes sum_t log pi(a_t|s_t) * (G_t - b(s_t))."""
        S = torch.as_tensor(np.concatenate([np.asarray(b[0]) for b in batch]), dtype=torch.float32)
        A = torch.as_tensor(np.concatenate([b[1] for b in batch]), dtype=torch.long)
        G = torch.as_tensor(np.concatenate([discounted_returns(b[2], self.gamma, self.pong) for b in batch]))
        info = {}
        if self.v is not None:
            with torch.no_grad(): base = self.v(S).squeeze(-1)
            adv = G - base
        else: adv = G
        loss = -(self.logp(S, A) * adv).sum() / len(batch)
        self.opt.zero_grad(); loss.backward()
        info["grad_norm"] = float(torch.sqrt(sum((p.grad ** 2).sum() for p in self.pi.parameters())))
        self.opt.step()
        if self.v is not None:
            vl = F.mse_loss(self.v(S).squeeze(-1), G); self.vopt.zero_grad(); vl.backward(); self.vopt.step(); info["v_loss"] = float(vl)
        return info


def train(a):
    ag = Agent(a.env, a.baseline, a.lr, a.blr, a.seed); env = ag.make_env()
    rewards, infos, batch, t0 = [], [], [], time.time()
    for ep in range(a.episodes):
        S, A, R = ag.run_episode(env, seed=a.seed if ep == 0 else None)
        rewards.append(float(sum(R))); batch.append((S, A, R))
        if len(batch) == a.batch: infos.append(ag.update(batch)); batch = []
        if (ep + 1) % a.log_every == 0:
            print(f"[{a.tag}] ep {ep+1}/{a.episodes} avg100={np.mean(rewards[-100:]):.2f} elapsed={time.time()-t0:.0f}s", flush=True)
            np.save(f"results/{a.tag}_train_rewards.npy", np.array(rewards)); torch.save(ag.pi.state_dict(), f"results/{a.tag}_policy.pt")
        if a.max_seconds and time.time() - t0 > a.max_seconds:
            print(f"[{a.tag}] stopping at ep {ep+1}: time budget reached", flush=True); break
    np.save(f"results/{a.tag}_train_rewards.npy", np.array(rewards)); torch.save(ag.pi.state_dict(), f"results/{a.tag}_policy.pt")
    return ag, env, np.array(rewards), infos


def evaluate(ag, env, n, greedy=False, seed=12345):
    return np.array([sum(ag.run_episode(env, seed=seed + i if i == 0 else None, greedy=greedy, record=False)[2]) for i in range(n)])


def plots(tag, rewards, ev, title):
    ma = np.convolve(rewards, np.ones(100) / 100, mode="valid") if len(rewards) >= 100 else np.array([])
    plt.figure(figsize=(7, 4)); plt.plot(rewards, alpha=.35, label="episode reward")
    if len(ma): plt.plot(np.arange(99, len(rewards)), ma, "r", lw=2, label="moving average (100 episodes)")
    plt.xlabel("Training episode"); plt.ylabel("Episode reward"); plt.title(title + " - training"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(f"figures/{tag}_training.png", dpi=150); plt.close()
    plt.figure(figsize=(6, 4)); plt.hist(ev, bins=25, edgecolor="k"); plt.axvline(ev.mean(), c="r", ls="--", label=f"mean={ev.mean():.2f}, std={ev.std():.2f}")
    plt.xlabel("Episode reward"); plt.ylabel("Count"); plt.title(title + " - 500 evaluation episodes"); plt.legend(); plt.tight_layout()
    plt.savefig(f"figures/{tag}_hist.png", dpi=150); plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["cartpole", "pong"], required=True); p.add_argument("--baseline", choices=["none", "value"], default="none")
    p.add_argument("--episodes", type=int, default=1500); p.add_argument("--batch", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.01); p.add_argument("--blr", type=float, default=None); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", required=True); p.add_argument("--eval_episodes", type=int, default=500); p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--max_seconds", type=float, default=0); p.add_argument("--skip_eval", action="store_true")
    a = p.parse_args(); a.blr = a.blr or a.lr
    os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)
    ag, env, rewards, infos = train(a)
    name = {"cartpole": "CartPole-v1", "pong": "Pong-v5"}[a.env] + (" + baseline" if a.baseline == "value" else " (no baseline)")
    res = dict(args=vars(a), n_train_episodes=len(rewards), train_last100_mean=float(rewards[-100:].mean()),
               train_grad_norm_mean=float(np.mean([i["grad_norm"] for i in infos])), train_grad_norm_std=float(np.std([i["grad_norm"] for i in infos])))
    if not a.skip_eval:
        ev = evaluate(ag, env, a.eval_episodes); np.save(f"results/{a.tag}_eval.npy", ev); plots(a.tag, rewards, ev, name)
        res.update(eval_mean=float(ev.mean()), eval_std=float(ev.std()), eval_min=float(ev.min()), eval_max=float(ev.max()))
    else: plots(a.tag, rewards, np.array([0.0]), name)
    json.dump(res, open(f"results/{a.tag}.json", "w"), indent=1); print(json.dumps(res, indent=1))
