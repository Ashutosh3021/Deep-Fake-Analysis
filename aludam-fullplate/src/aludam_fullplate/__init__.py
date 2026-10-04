"""aludam-fullplate - the neural tier of aludam.

Same import, same syntax, same :class:`aludam.result.DetectionResult` as
``aludam``. Installing this package additionally pulls aludam's ``neural``
extra (torch, transformers, huggingface_hub, timm, peft,
opencv-python-headless, librosa, soundfile, scipy, onnxruntime) - the stack
the four detectors need to actually run::

    pip install aludam-fullplate          # == pip install "aludam[neural]"

    from aludam import load_detector      # or from aludam_fullplate import ...
    det = load_detector("img_det")
    result = det.predict("photo.png")

Everything here is a re-export, deliberately: ``aludam`` owns the one
dependency list, and this module guarantees the two names are the *same*
objects, not look-alikes (ALUDAM_PLAN.md sec 6.2).

Weights are never bundled. Run ``aludam fetch`` once after installing.
Missing dependencies and missing weights raise; they never degrade into a
heuristic-looking score.
"""

from __future__ import annotations

import aludam as _aludam

from aludam import *  # noqa: F401,F403 - the surface, re-exported verbatim
from aludam import __version__ as __version__  # noqa: F401

TIER = "neural"

__all__ = [*_aludam.__all__, "TIER"]
