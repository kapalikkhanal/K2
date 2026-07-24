"""Play/visualize entry point: registers K2 tasks, then runs mjlab's player.

  python -m k2_rl.play Mjlab-InPlace-K2 --checkpoint-file <path.pt>
"""

import k2_rl  # noqa: F401  (registers K2 tasks)
from mjlab.scripts.play import main

if __name__ == "__main__":
  main()
