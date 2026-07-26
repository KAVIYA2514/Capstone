# app package
import sys
import types
import importlib.machinery

try:
    import torch  # noqa: F401
except (OSError, ImportError):
    if "torch" not in sys.modules or sys.modules["torch"] is None:
        t = types.ModuleType("torch")
        t.__file__ = "dummy"
        t.__version__ = "2.0.0"
        t.__spec__ = importlib.machinery.ModuleSpec("torch", None)
        t.Tensor = type("Tensor", (), {})
        c = types.ModuleType("torch.cuda")
        c.is_available = lambda: False
        t.cuda = c
        sys.modules["torch"] = t
        sys.modules["torch.cuda"] = c

