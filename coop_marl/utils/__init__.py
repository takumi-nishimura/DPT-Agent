try:
    from rebar import arrdict as _arrdict_mod, dotdict as _dotdict_mod

    Arrdict = _arrdict_mod.arrdict
    Dotdict = _dotdict_mod.dotdict
    CatArrdict = _arrdict_mod.cat
except Exception:  # pragma: no cover - optional upstream dependency
    import logging

    class _FallbackDict(dict):
        """Lightweight dot-accessible dict used when rebar is unavailable."""

        def __getattr__(self, item):
            try:
                return self[item]
            except KeyError as exc:
                raise AttributeError(item) from exc

        def __setattr__(self, key, value):
            self[key] = value

    class Arrdict(_FallbackDict):
        pass

    class Dotdict(_FallbackDict):
        pass

    def CatArrdict(*args, **kwargs):
        raise NotImplementedError("cat is unavailable without rebar")

    logging.getLogger(__name__).warning(
        "rebar not found; using fallback arrdict/dotdict shim"
    )

arrdict = Arrdict
dotdict = Dotdict
cat = CatArrdict

from .logger import *
from .metrics import *
from .nn import *
from .parser import *
from .rl import *
from .utils import *
