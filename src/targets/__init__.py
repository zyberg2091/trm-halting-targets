"""Expose the notebook Q_LOSS selected by config.TARGET."""
from src.config import TARGET

if TARGET == "softmean":
    from src.targets.softmean import Q_LOSS
elif TARGET == "geomean":
    from src.targets.geomean import Q_LOSS
elif TARGET == "binary_em":
    from src.targets.binary_em import Q_LOSS
else:
    raise ValueError(f"unknown TRM_TARGET {TARGET!r}; expected softmean, geomean or binary_em")

__all__ = ["Q_LOSS"]
