"""
Literature/cohort-anchored diameter-vs-gestational-age growth model for the
umbilical vein (UV) and umbilical artery (UA). Same anchors as the
companion manuscript's Section II-D.

UV (mm): 14.5w -> 2.0, 20w -> 5.1, 40w -> 9.1
UA (mm, mean of the two arteries): 12w -> 0.74, 20w -> 1.545, 30w -> 3.6
  (UA's valid interpolation range is 12-30 weeks; outside is extrapolation)
"""
import numpy as np
from scipy.interpolate import PchipInterpolator

UV_GA = np.array([14.5, 20.0, 40.0])
UV_DIAM_MM = np.array([2.0, 5.1, 9.1])
_uv_curve = PchipInterpolator(UV_GA, UV_DIAM_MM, extrapolate=True)

UA_GA = np.array([12.0, 20.0, 30.0])
UA_DIAM_MM = np.array([0.74, 1.545, 3.6])
_ua_curve = PchipInterpolator(UA_GA, UA_DIAM_MM, extrapolate=True)


def uv_diameter_mm(ga_weeks):
    return float(_uv_curve(ga_weeks))


def ua_diameter_mm(ga_weeks):
    return float(_ua_curve(ga_weeks))
