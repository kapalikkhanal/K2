"""K2 reinforcement-learning tasks for mjlab.

Importing this package registers the K2 in-place tasks with mjlab's task
registry (so ``list_tasks()`` / train / play can find them).
"""

from k2_rl import tasks  # noqa: F401  (registers tasks on import)
