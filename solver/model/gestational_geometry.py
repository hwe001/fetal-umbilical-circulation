"""Gestational-age geometry inputs for the reduced UA/UV model.

All dimensions are *population reference inputs*, not patient-specific
segmentations.  Functions reject unsupported extrapolation by default.
"""

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.interpolate import PchipInterpolator


# Reference specimen trunk lengths. Keep these separate from the GA-dependent
# diameter curves and override them explicitly in population sensitivity runs.
UA_TRUNK_LENGTH_CM = 72.2
UV_TRUNK_LENGTH_CM = 61.78


@dataclass(frozen=True)
class ReferenceCurve:
    name: str
    ga_weeks: tuple
    values: tuple
    units: str
    measurement_site: str
    citation: str

    def evaluate(self, ga, allow_extrapolation=False):
        ga_array = np.asarray(ga, dtype=float)
        lo, hi = self.ga_weeks[0], self.ga_weeks[-1]
        if np.any((ga_array < lo) | (ga_array > hi)):
            message = f"{self.name} is anchored only over {lo:g}-{hi:g} weeks"
            if not allow_extrapolation:
                raise ValueError(message)
            warnings.warn(message + "; extrapolating", RuntimeWarning, stacklevel=2)
        curve = PchipInterpolator(self.ga_weeks, self.values, extrapolate=allow_extrapolation)
        value = curve(ga_array)
        return float(value) if value.ndim == 0 else value


# Existing project UA inputs. These should be replaced by a documented fit to
# the deidentified diameter extractions before final manuscript analysis.
UA_DIAMETER_MM = ReferenceCurve(
    name="Mean diameter of either umbilical artery",
    ga_weeks=(12.0, 20.0, 30.0),
    values=(0.74, 1.545, 3.6),
    units="mm",
    measurement_site="umbilical cord; project source table",
    citation="Project targets_and_original_fit.json; provenance requires confirmation",
)

# Prospective longitudinal normal cohort, intra-amniotic/free-cord UV, inner
# diameter. These sparse means define a transparent provisional PCHIP curve;
# final analysis should use published percentile coefficients where licensed.
UV_DIAMETER_MM = ReferenceCurve(
    name="Umbilical vein inner diameter",
    ga_weeks=(18.0, 26.0, 34.0),
    values=(2.8, 5.8, 7.6),
    units="mm",
    measurement_site="intra-amniotic portion of the umbilical vein",
    citation="Spurway et al., Australas J Ultrasound Med (2016), PMID 34760451",
)


def ua_radius_cm(ga, allow_extrapolation=False):
    return UA_DIAMETER_MM.evaluate(ga, allow_extrapolation) / 20.0


def uv_radius_cm(ga, allow_extrapolation=False):
    return UV_DIAMETER_MM.evaluate(ga, allow_extrapolation) / 20.0


def circular_area_cm2(radius_cm):
    radius_cm = np.asarray(radius_cm, dtype=float)
    if np.any(radius_cm <= 0):
        raise ValueError("radius must be positive")
    return np.pi * radius_cm**2


def flow_from_tav_mL_min(diameter_mm, tav_cm_s, profile_factor=1.0):
    """Convert UV diameter and time-averaged *spatial mean* velocity to flow.

    ``profile_factor`` must remain 1 when TAV is already spatially averaged.
    Use a study-specific factor only when the reported velocity convention
    requires it (for example, a maximum/centreline velocity).
    """
    if diameter_mm <= 0 or profile_factor <= 0:
        raise ValueError("diameter and profile_factor must be positive")
    area_cm2 = np.pi * (diameter_mm / 20.0) ** 2
    return float(area_cm2 * tav_cm_s * profile_factor * 60.0)
