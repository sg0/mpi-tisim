#!/usr/bin/env bash
# Send live control tokens to a running mpisim.
#   ./simctl.sh p:0.9          raise inference priority (running training tasks checkpoint)
#   ./simctl.sh s:1            suspend rank 1 (checkpoint to checkpoints/rank_1.ckpt)
#   ./simctl.sh r:1            resume rank 1 (drains its checkpoints first)
#   ./simctl.sh e:1            exit, saving checkpoints      (e:0 = exit without saving)
# Several tokens may be given at once. Set SIM_CTL to use a non-default control file.
CTL="${SIM_CTL:-sim_control.txt}"
[ $# -ge 1 ] || { sed -n '2,8p' "$0"; exit 1; }
echo "$*" >> "$CTL"
