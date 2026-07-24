"""Train entry point: registers K2 tasks, then runs mjlab's trainer.

  python -m k2_rl.train Mjlab-InPlace-K2 --env.scene.num-envs 4096
"""

import k2_rl  # noqa: F401  (registers K2 tasks)
from mjlab.scripts.train import main

if __name__ == "__main__":
  main()
