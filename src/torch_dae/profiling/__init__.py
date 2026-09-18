"""Profiling v1 / Technical Card subsystem (independent of Model Card lifecycle).

Submodules are intentionally leaf-level: importing this package never requires PyTorch,
CodeCarbon, or `psutil`. Modules that need those optional dependencies import them lazily inside
functions so this package is safe to import from both the root control-plane environment and a
model-runtime worker environment (Section 25).
"""

from __future__ import annotations
