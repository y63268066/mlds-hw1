for s in 0 1 2; do
 python3 pg.py --env cartpole --baseline none  --episodes 1500 --lr 0.001 --seed $s --tag cp_nobase_s$s 2>&1 | grep -E "avg100" | tail -1
 python3 pg.py --env cartpole --baseline value --episodes 1500 --lr 0.001 --seed $s --tag cp_value_s$s  2>&1 | grep -E "avg100" | tail -1
done
echo DONE
