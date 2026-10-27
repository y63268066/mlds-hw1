import json, os, numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from PIL import Image as PI

st = getSampleStyleSheet()
B = ParagraphStyle("B", parent=st["BodyText"], fontName="Times-Roman", fontSize=10.5, leading=13.5)
H1 = ParagraphStyle("H1", parent=st["Heading1"], fontName="Times-Bold", fontSize=15)
H2 = ParagraphStyle("H2", parent=st["Heading2"], fontName="Times-Bold", fontSize=12)
CAP = ParagraphStyle("CAP", parent=B, fontSize=9, leading=11, alignment=1)
CD = ParagraphStyle("CD", parent=B, fontName="Courier", fontSize=8.5, leading=10.5)
S = []
P = lambda t: S.append(Paragraph(t, B)); sp = lambda h=6: S.append(Spacer(1, h))
def IMG(path, w=4.2, cap=None):
    iw, ih = PI.open(path).size; it = [Image(path, w * inch, w * inch * ih / iw)]
    if cap: it.append(Paragraph(cap, CAP))
    S.append(KeepTogether(it + [Spacer(1, 6)]))
def IMG2(a, b, cap, w=3.3):
    cells = []
    for p in (a, b):
        iw, ih = PI.open(p).size; cells.append(Image(p, w * inch, w * inch * ih / iw))
    S.append(KeepTogether([Table([cells], colWidths=[3.5 * inch] * 2), Paragraph(cap, CAP), Spacer(1, 6)]))
def TBL(rows, cw=None, fs=9):
    ps = ParagraphStyle("t", parent=B, fontSize=fs, leading=fs + 2)
    t = Table([[Paragraph(str(c), ps) for c in r] for r in rows], colWidths=cw, repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .4, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")), ("VALIGN", (0, 0), (-1, -1), "TOP")])); S.append(t); sp(8)
f = lambda x, n=1: f"{x:.{n}f}"
J = lambda p: json.load(open(p))
R2 = J("results/part2_results.json"); T2 = R2["tests"]; FT = J("results/pg_tests_fast.json"); NG = J("results/pg_tests_negated.json")
LR = J("results/cp_lr0.01_probe.json"); LRV = J("results/cp_value_lr0.01_s0.json")
NS = 5
CP = {t: [J(f"results/cp_{t}_s{s}.json") for s in range(NS)] for t in ("nobase", "value")}
PN, PV = J("results/pong_nobase.json"), J("results/pong_value.json")
ev = lambda t: np.array([r["eval_mean"] for r in CP[t]]); gn = lambda t: np.mean([r["train_grad_norm_mean"] for r in CP[t]])
gs = lambda t: np.mean([r["train_grad_norm_std"] for r in CP[t]])
GV = lambda k: FT[k]["per_sample_variance"]
bn, bm, bb = "grad_estimator[no baseline]", "grad_estimator[constant baseline = mean reward]", "grad_estimator[badly-scaled baseline b=10]"
mc, bp, sens = T2["monte_carlo"], T2["baseline_policies"], T2["sensitivity"]

S.append(Paragraph("Assignment 1: Value/Policy Iteration and Policy Gradient", st["Title"]))
P("Name: ______________ &nbsp;&nbsp; GitHub username: y63268066 &nbsp;&nbsp; Repository: &lt;REPO-NAME&gt; &nbsp;&nbsp; Link: https://github.com/y63268066/&lt;REPO-NAME&gt;")
S.append(Paragraph("How to run", H2))
S.append(Paragraph("pip install numpy matplotlib torch gymnasium ale-py reportlab<br/>"
  "python shuttle_mdp.py &nbsp;&nbsp;# Part 2 solvers, plots and all validation tests<br/>"
  "python pg.py --env cartpole --baseline none --episodes 1500 --lr 0.001 --seed 0 --tag cp_nobase_s0<br/>"
  "python pg.py --env cartpole --baseline value --episodes 1500 --lr 0.001 --seed 0 --tag cp_value_s0<br/>"
  "python pg.py --env pong --baseline none --batch 4 --lr 0.001 --episodes 100000 --max_seconds 1500 --tag pong_nobase<br/>"
  "python pg.py --env pong --baseline value --batch 4 --lr 0.001 --episodes 100000 --max_seconds 1500 --tag pong_value<br/>"
  "python pg_tests.py fast; python pg_tests.py negated &nbsp;&nbsp;# policy gradient tests<br/>"
  "python make_summary_figs.py; python make_report.py", CD))
P("Each pg.py run trains the agent, saves the reward curve with a 100-episode moving average, plays 500 episodes with the final policy and saves the histogram, mean and standard deviation. Results are in results/ and figures in figures/. README.md lists every file.")

# ---------------- Part 2: MDP
S.append(Paragraph("1. Value and Policy Iteration", H1))
S.append(Paragraph("1.1 Model", H2))
P("The state is the number of customers waiting, s in {0,...,200}. The action is hold (0) or dispatch (1). Dispatching moves min(s, 15) customers and costs 100. Each customer still waiting after the decision costs 2. Then A ~ Unif{1,...,5} customers arrive and the next state is min(s_after + A, 200). So the cost is c(s,a) = 100a + 2(s - a*min(s,15)). "
  "Assumptions: I minimise expected discounted cost, so the value function is a cost (lower is better); gamma = 0.95 everywhere, including the T = 500 problem; the terminal value at T = 500 is 0; arrivals above 200 are dropped at no cost; ties are broken toward holding.")
S.append(Paragraph("1.2 Methods and results", H2))
P(f"(a) Enumerating all 2^500 action sequences is not possible, so I use backward induction, which gives the value of the best sequence exactly; a literal enumeration is used only to check it for small T. (b) Value iteration stops when the largest change is below 1e-10 ({R2['vi_iters']} iterations). (c) Policy iteration solves the linear system for each policy and only switches an action when it strictly improves; it stopped after {R2['pi_iters']} iterations. "
  f"All three give V(0) = {f(R2['V0_at_0'][1],2)}. The optimal policy holds when fewer than {R2['threshold_pi']} customers are waiting and dispatches otherwise.")
IMG("figures/part2a_enumeration_T500.png", 4.0, "Figure 1. (a) Enumeration, T = 500: optimal value at time 0 versus customers waiting.")
IMG("figures/part2b_value_iteration.png", 4.0, "Figure 2. (b) Value iteration: optimal value versus customers waiting.")
IMG("figures/part2c_policy_iteration.png", 4.0, "Figure 3. (c) Policy iteration: optimal policy versus customers waiting.")
S.append(Paragraph("1.3 Edge cases", H2))
P(f"Empty station (s = 0): dispatching costs 100 and moves nobody, so the policy holds; the solver gives action {T2['policy_at_s0']}. Fewer than 15 waiting: only s customers leave, so the cost is exactly 100 (c(10,1) = 100). More than 15 waiting: the rest stay and pay holding cost (c(20,1) = 110). Upper limit s = 200: arrivals are cut off, and holding at 200 stays at 200 with probability {f(T2['boundary_s200_hold_goes_to_200'],0)}. Every row of both transition matrices sums to one (largest error {T2['rows_sum_to_one']}). Ties between actions could make policy iteration loop forever, so an action changes only on strict improvement. 0.95^500 is 7e-12, so the T = 500 and infinite-horizon answers should agree.")
S.append(Paragraph("1.4 Tests", H2))
P("Everything below is produced by running shuttle_mdp.py (results/part2_results.json).")
TBL([["Test", "Expected", "Result"],
 ["Enumerate every action and every arrival (no shortcuts) for T = 1 to 4 and 8 start states; compare to backward induction", "same value", f"largest difference {T2['bruteforce_vs_DP_max_abs_err(T<=4)']:.1e}"],
 ["T = 500 enumeration vs value iteration", "difference about 1e-9 or less", f"{T2['max|V_T500 - V_VI|']:.1e}"],
 ["Policy iteration vs value iteration", "same V and policy", f"V differs by {T2['max|V_PI - V_VI|']:.1e}; same policy: {T2['policy_VI==policy_PI']}"],
 ["Bellman residual of V", "near 0", f"{T2['bellman_residual']:.1e}"],
 ["Policy iteration from never / always / random initial policy", "same answer", f"{T2['PI_inits_same_policy']}, iterations {T2['PI_iters_by_init']}"],
 ["Simulate 20,000 runs of 400 steps from s = 0, 20, 100 and compare with V", "within about 2 standard errors", f"{f(mc['0']['MC'],2)} vs {f(mc['0']['analytic'],2)}; {f(mc['20']['MC'],2)} vs {f(mc['20']['analytic'],2)}; {f(mc['100']['MC'],2)} vs {f(mc['100']['analytic'],2)} (z = {f(mc['0']['z'],2)}, {f(mc['20']['z'],2)}, {f(mc['100']['z'],2)})"],
 ["Cost from s = 0 of other policies (exact evaluation)", "optimal is smallest", f"optimal {f(bp['optimal']['V0_analytic'])}; dispatch at 15: {f(bp['thr=K']['V0_analytic'])}; at 8: {f(bp['thr=opt-5']['V0_analytic'])}; at 23: {f(bp['thr=opt+10']['V0_analytic'])}; always: {f(bp['always']['V0_analytic'])}; never: {f(bp['never']['V0_analytic'])}"],
 ["Raise the dispatch cost", "threshold goes up; cost 0 gives threshold 1", f"threshold for cost 0/25/50/100/150/200/300/400: {list(T2['threshold_vs_cf'].values())}"],
 ["Holding cost 0, 1, 2, 4", "never dispatch at 0; then earlier dispatch", f"threshold {sens['ch=0']['threshold']}, {sens['ch=1']['threshold']}, {sens['cf=100 (base)']['threshold']}, {sens['ch=4']['threshold']}"],
 ["Capacity 5, 15, 30", "threshold follows capacity", f"{sens['K=5']['threshold']}, {R2['threshold_pi']}, {sens['K=30']['threshold']}"],
 ["gamma = 0 (only the current cost counts)", "never dispatch, since 100 > 2 x 15", f"threshold {T2['gamma=0 threshold (None = never dispatch, since cf=100 > ch*K=30)']} (None = never)"],
 ["V non-decreasing in s, policy is a threshold", "true", f"{T2['V_nondecreasing']}, {T2['policy_is_threshold']}"]],
 cw=[2.9 * inch, 1.5 * inch, 2.7 * inch])
P("First 10 steps with deterministic arrivals (A = 3 each step). With these arrivals the state goes 0, 3, 6, 9, 12, 15; the policy waits until 15 and then dispatches, which empties the station, and the cycle restarts at 3. The values obey V(15) = 100 + 0.95 V(3): " + f"{T2['trace_det_arrivals3'][5]['V']} = 100 + 0.95 x {T2['trace_det_arrivals3'][6]['V']}.")
TBL([["t", "state", "action", "cost", "V(state)"]] + [[x["t"], x["s"], x["a"], x["cost"], x["V"]] for x in T2["trace_det_arrivals3"]], cw=[.6 * inch] * 5)
IMG("figures/part2_policy_comparison.png", 3.7, "Figure 4. Optimal value compared with two simple policies.")

# ---------------- PG part 1
S.append(PageBreak()); S.append(Paragraph("2. Policy Gradient, Part 1 (no baseline)", H1))
S.append(Paragraph("2.1 Implementation", H2))
P("I implemented REINFORCE in PyTorch (pg.py). The loss is minus the sum over steps of log pi(a_t|s_t) times G_t, averaged over the episodes in a batch, where G_t is the discounted reward from step t onward (same expected gradient as the full-episode discounted return, with less noise). Optimiser: Adam. "
  "CartPole-v1: gamma = 0.95, a network with one hidden layer of 128 ReLU units and 2 outputs, one episode per update. "
  "ALE/Pong-v5: gamma = 0.99, input is the difference between two consecutive preprocessed frames (6400 values, so the network can see motion), one hidden layer of 200 units and one sigmoid output giving the probability of RIGHT (action 2); LEFT (action 3) has the remaining probability, so only these two actions are used. The running discounted sum is reset after each point, and 4 games are used per update.")
P(f"Learning rate. With the suggested 0.01 and no baseline, CartPole did not learn in my run: the average of the last 100 training episodes was {f(LR['train_last100_mean'])} after 1,000 episodes and the evaluation mean was {f(LR['eval_mean'])}, compared with {f(FT['cartpole_untrained_policy_mean,std'][0])} for an untrained policy. I used 0.001 for all other runs. I only compared these two values and did not do a wider search.")
S.append(Paragraph("2.2 CartPole-v1", H2))
IMG("figures/cp_nobase_s0_training.png", 4.2, "Figure 5. CartPole without baseline (seed 0): episode reward and 100-episode moving average.")
IMG("figures/cp_nobase_s0_hist.png", 3.8, f"Figure 6. Rewards of 500 evaluation episodes (seed 0): mean {f(CP['nobase'][0]['eval_mean'])}, standard deviation {f(CP['nobase'][0]['eval_std'])}.")
TBL([["Seed", "Last-100 training mean", "Evaluation mean", "Evaluation std"]] + [[s, f(CP['nobase'][s]['train_last100_mean']), f(CP['nobase'][s]['eval_mean']), f(CP['nobase'][s]['eval_std'])] for s in range(NS)] + [["Average", "", f(ev('nobase').mean()), ""]], cw=[1 * inch, 1.8 * inch, 1.6 * inch, 1.4 * inch])
P("The maximum score is 500. Runs differ a lot between seeds, and for several seeds the evaluation mean is below the best training moving average. Learning is not steady: the moving average goes up and down.")
S.append(Paragraph("2.3 Pong-v5", H2))
P(f"Pong was trained for 12 minutes on one CPU core, which was {PN['n_train_episodes']} episodes (about {PN['n_train_episodes']//4} updates). Average of the last 100 training episodes: {f(PN['train_last100_mean'])}. Evaluation over 500 episodes: mean {f(PN['eval_mean'])}, standard deviation {f(PN['eval_std'])}. Pong scores run from -21 to +21. This is far less training than Pong normally needs, so the agent is still close to the worst possible score.")
IMG("figures/pong_nobase_training.png", 4.2, "Figure 7. Pong without baseline: episode reward and 100-episode moving average.")
IMG("figures/pong_nobase_hist.png", 3.8, f"Figure 8. Rewards of 500 evaluation episodes: mean {f(PN['eval_mean'],2)}, standard deviation {f(PN['eval_std'],2)}.")
S.append(Paragraph("2.4 Edge cases", H2))
P("CartPole episodes end when the pole falls or at 500 steps. The method uses full returns without bootstrapping, so both cases are handled the same way. The loss is divided by the number of episodes, not steps, so long episodes are not down-weighted. "
  "Pong: the first frame has no previous frame, so its difference image is zero; the previous frame is cleared at the start of each episode; the given preprocess function changes its input in place, so I copy before modifying; returns are reset after each point so later points do not affect earlier decisions; log-probabilities use logsigmoid and log_softmax so a saturated output stays finite; action index 1 is sent to the environment as action 2 (RIGHT) and index 0 as action 3 (LEFT), checked with the environment's action names.")
S.append(Paragraph("2.5 Tests", H2))
TBL([["Test", "Expected", "Result"],
 ["Discounted returns against a slow direct sum on 20 random reward sequences; hand example [1,1,1] with gamma 0.5", "0 error; [1.75, 1.5, 1]", f"error {FT['returns_vs_naive_max_err']}; {FT['returns_hand_check [1,1,1],g=.5 -> [1.75,1.5,1]']}"],
 ["Pong return reset: points far apart do not affect each other", "0 error", f"error {FT['pong_returns_reset_vs_naive_max_err']}; {FT['pong_no_leak_across_points (G at t=21..45 uses only +1 point)']}"],
 ["Log-probabilities against a direct softmax/sigmoid; finite with a saturated output", "match", f"difference {FT['cartpole_logp_vs_log_softmax']:.0e} and {FT['pong_logp_vs_sigmoid']:.0e}; finite: {FT['pong_logp_finite_at_saturation']}"],
 ["Three-armed bandit where the true policy gradient can be computed. Average the same estimator used in the loss over 200,000 samples and compare to the true gradient", "close to the true gradient", f"cosine similarity {FT[bn]['cosine']:.5f}, largest z-score {FT[bn]['max_abs_z']:.2f}"],
 ["Same seed run twice; different seed", "same results; different results", f"{FT['same_seed_identical_rewards']}; {FT['different_seed_differs']}"],
 ["Flip the sign of the reward and train CartPole for 600 episodes (the agent should now learn to fail quickly)", "episodes get shorter", f"average length of first 100 vs last 100 episodes: seed 0 {NG['seed0']['first100']} to {NG['seed0']['last100']}, seed 1 {NG['seed1']['first100']} to {NG['seed1']['last100']}"],
 ["Pong preprocessing", "80x80, input differences in {-1,0,1}, first difference zero, actions RIGHT and LEFT", f"shape {FT['preprocess_shape,min,max,frac_nonzero'][0]}, values {FT['pong_diff_values']}, first zero {FT['pong_first_state_is_zero_diff']}, actions {FT['pong_action_meanings']}"]],
 cw=[3.0 * inch, 1.6 * inch, 2.5 * inch])
P("These tests show that the returns, probabilities and gradient are computed correctly and that training follows the sign of the reward. They do not show that the hyperparameters are well tuned.")

# ---------------- PG part 2
S.append(PageBreak()); S.append(Paragraph("3. Policy Gradient, Part 2 (baseline)", H1))
S.append(Paragraph("3.1 Baseline", H2))
P("I use a learned, state-dependent baseline b(s): a second network (same layout as the policy but with one output) trained by mean-squared error to predict the discounted return G_t, using the same Adam learning rate. The policy loss uses G_t - b(s_t) with b treated as a constant. Subtracting a function of the state alone does not change the expected gradient, and it removes the common offset in the returns. In CartPole every return is positive, so without a baseline every sampled action is pushed up. Everything else is the same as Part 1.")
S.append(Paragraph("3.2 CartPole-v1", H2))
IMG("figures/cp_value_s0_training.png", 4.2, "Figure 9. CartPole with baseline (seed 0): episode reward and 100-episode moving average.")
IMG("figures/cp_value_s0_hist.png", 3.8, f"Figure 10. Rewards of 500 evaluation episodes (seed 0): mean {f(CP['value'][0]['eval_mean'])}, standard deviation {f(CP['value'][0]['eval_std'])}.")
IMG("figures/cartpole_seeds_comparison.png", 6.3, "Figure 11. Moving average over 100 episodes for all seeds, without baseline (left) and with baseline (right).")
TBL([["Seed", "No baseline: mean / std", "Baseline: mean / std"]] + [[s, f"{f(CP['nobase'][s]['eval_mean'])} / {f(CP['nobase'][s]['eval_std'])}", f"{f(CP['value'][s]['eval_mean'])} / {f(CP['value'][s]['eval_std'])}"] for s in range(NS)] +
 [["Average", f(ev('nobase').mean()), f(ev('value').mean())], ["Gradient norm per update, mean (std within a run)", f"{f(gn('nobase'),0)} ({f(gs('nobase'),0)})", f"{f(gn('value'),0)} ({f(gs('value'),0)})"]], cw=[2.8 * inch, 2.1 * inch, 2.1 * inch])
better = int((np.array([r['eval_mean'] for r in CP['value']]) > np.array([r['eval_mean'] for r in CP['nobase']])).sum())
P(f"The baseline lowers the gradient norm by a factor of {f(gn('nobase')/gn('value'))} and its variation within a run by {f(gs('nobase')/gs('value'))}. The average evaluation score is {f(ev('value').mean(),0)} with the baseline against {f(ev('nobase').mean(),0)} without it, and the baseline run is higher for {better} of {NS} seeds. With this few seeds and this much spread I would not call the score difference conclusive; the gradient-norm change is clearer.")
P(f"With a baseline and the suggested learning rate 0.01 (seed 0), the last-100 training average was {f(LRV['train_last100_mean'])} and the evaluation mean was {f(LRV['eval_mean'])} (standard deviation {f(LRV['eval_std'])}); without a baseline at 0.01 it was {f(LR['eval_mean'])}.")
S.append(Paragraph("3.3 Pong-v5", H2))
P(f"Same setup as Part 1 but an 8-minute budget ({PV['n_train_episodes']} episodes, about the same count as Part 1). Average of the last 100 training episodes: {f(PV['train_last100_mean'])}. Evaluation over 500 episodes: mean {f(PV['eval_mean'])}, standard deviation {f(PV['eval_std'])}. Without baseline the mean was {f(PN['eval_mean'])}. This is one run per method and both are still far from winning, so I do not draw a conclusion about the baseline on Pong.")
IMG("figures/pong_value_training.png", 4.2, "Figure 12. Pong with baseline: episode reward and 100-episode moving average.")
IMG("figures/pong_value_hist.png", 3.8, f"Figure 13. Rewards of 500 evaluation episodes: mean {f(PV['eval_mean'],2)}, standard deviation {f(PV['eval_std'],2)}.")
S.append(Paragraph("3.4 Edge cases and tests", H2))
P(f"Edge cases: the baseline is detached when the advantage is formed, so the policy loss cannot change the value network. The baseline is fitted to the same returns as the policy uses, including the reset after each point in Pong. An inaccurate baseline early in training should only add noise, not bias. "
  f"Tests: in the bandit test above I compared three versions of the estimator with the true gradient. All three match it (cosine similarity at least {min(FT[k]['cosine'] for k in (bn, bm, bb)):.4f}), so the estimator stays unbiased with a baseline; the per-sample variance is {f(GV(bn),2)} without baseline, {f(GV(bm),2)} with the mean-reward baseline and {f(GV(bb),1)} with a badly chosen constant of 10. This also shows that a bad baseline can make things worse. I also compared gradient norms and seeds as in 3.2.")

S.append(Paragraph("4. With three more months", H1))
P("MDP: try other arrival distributions (Poisson, rush-hour patterns) and check that the threshold form of the policy still holds; solve the same problem with Q-learning and see whether it finds the same threshold; make the cap larger than 200 and check that the values we care about do not change; check how wrong the policy is if the costs are estimated incorrectly. "
  "Policy gradient: run at least 20 seeds per method with confidence intervals; search over learning rate, batch size and network size; train Pong for tens of thousands of episodes on a GPU with and without baseline and compare sample efficiency; try other baselines and entropy regularisation; test the gradient on environments with known optimal policies and against finite differences on the full network; compare ALE settings (sticky actions on or off); study why evaluation scores are often below the best moving average, for example by checkpointing and using a decaying learning rate.")
P("Limits of this work: the CartPole learning rate is 0.001 rather than 0.01, and Pong was trained for a short time on a CPU, so its scores are still close to -21.")
SimpleDocTemplate("report.pdf", pagesize=letter, leftMargin=.75 * inch, rightMargin=.75 * inch, topMargin=.7 * inch, bottomMargin=.7 * inch, title="Assignment 1").build(S)
print("built")
