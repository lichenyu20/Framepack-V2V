python v2v_cli.py \
  --history_video ./prefix.mp4 \
  --prompt "A simple pendulum oscillation on a uniform white background. The motion starts with the pendulum bob displaced to the left of the equilibrium position and released, then it swings rightward and continues periodic motion." \
  --total_second_length 5 \
  --steps 25 \
  --gs 10 \
  --seed 31337 \
  --use_teacache \
  --prepend_history