# MLDS Assignment 1 - RL (Value/Policy Iteration, Policy Gradient)
GitHub user: y63268066   

## Setup
pip install numpy matplotlib torch gymnasium ale-py reportlab   (ale-py bundles the Atari ROMs)

## Part 2 - Shuttle MDP (seconds)
python shuttle_mdp.py          # solves (a) enumeration T=500, (b) value iteration, (c) policy iteration, runs ALL validation tests
                               # -> figures/part2*.png, results/part2_results.json

## Policy gradient (pg.py)
# Part 1 (no baseline)           # Part 2 (learned state-value baseline)
python pg.py --env cartpole --baseline none  --episodes 1500 --lr 0.001 --seed 0 --tag cp_nobase_s0
python pg.py --env cartpole --baseline value --episodes 1500 --lr 0.001 --seed 0 --tag cp_value_s0
python pg.py --env pong --baseline none  --episodes 100000 --batch 4 --lr 0.001 --max_seconds 720 --tag pong_nobase
python pg.py --env pong --baseline value --episodes 100000 --batch 4 --lr 0.001 --max_seconds 720 --tag pong_value
#  each call: trains, saves reward curve + 100-episode moving average, rolls out 500 episodes, saves histogram, mean and std.
#  For a serious Pong run use e.g. --max_seconds 0 --episodes 10000 (hours on CPU; much faster on a GPU).

python pg.py --env cartpole --baseline none --episodes 1000 --lr 0.01 --seed 0 --tag cp_lr0.01_probe --eval_episodes 100   # the lr=0.01 trial quoted in the report
python pong_finish.py   # evaluates a saved Pong policy if a run was interrupted before its evaluation

## Validation tests
python pg_tests.py fast        # returns, log-prob, unbiased-gradient/variance (bandit), preprocessing, determinism, throughput
python pg_tests.py negated     # reward sign-flip sanity experiment
python make_summary_figs.py    # multi-seed CartPole comparison
python make_report.py          # builds report.pdf from the saved results
