"""Compatibility shim so the talc YOLO-seg checkpoint loads under stock ultralytics.

The `talc_seg_v2/weights/best.pt` was trained with a forked ultralytics that had
custom loss classes (`BCEDiceLoss`, `MultiChannelDiceLoss`, …) living in
`ultralytics.utils.loss`. Those class references are pickled into the checkpoint,
so a plain `YOLO(best.pt)` under stock ultralytics dies with
`AttributeError: Can't get attribute 'BCEDiceLoss'`.

We only need the model for INFERENCE, where the training loss is never used, so we
vend a harmless dummy `nn.Module` subclass for any missing symbol the unpickler
asks `ultralytics.utils.loss` for. Call `install_loss_stubs()` once, before the
first `YOLO(<weights>)` call.
"""
import torch.nn as nn
import ultralytics.utils.loss as _loss

_cache: dict[str, type] = {}


def _make(name: str) -> type:
    if name not in _cache:
        _cache[name] = type(name, (nn.Module,),
                            {"forward": lambda self, *a, **k: None})
    return _cache[name]


def install_loss_stubs() -> None:
    """Patch ultralytics.utils.loss to resolve any missing loss class to a stub."""
    if getattr(_loss, "_talc_stubs_installed", False):
        return

    def _module_getattr(name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return _make(name)

    _loss.__getattr__ = _module_getattr  # PEP 562 module-level __getattr__
    _loss._talc_stubs_installed = True
