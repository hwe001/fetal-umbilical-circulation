# UA recalibration and reduced UV-return simulation

## Configuration

- Two identical 1D UAs, each 72.2 cm long.
- One simplified central UV, 61.78 cm long.
- UA diameter: project GA curve, directly anchored over 12-30 weeks.
- UV diameter: literature PCHIP curve, directly anchored at 18, 26 and 34 weeks.
- UA indices: minimum/pre-systolic velocity used as EDV; no artificial floor.
- Coarse fit mesh: dx approximately 0.73 cm.
- Production verification mesh: dx approximately 0.25 cm.
- Every reported production run records periodic convergence and outlet fallback.

## Table I: UA PI/RI recalibration

| GA | R_total | dp_inlet | PI target | PI production | RI target | RI production | Two-UA flow (mL/min) | Periodic |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 19 | 17238.0 | 40978.0 | 1.295 | 1.266 | 0.740 | 0.789 | 5.9 | yes |
| 23 | 7863.8 | 16004.2 | 1.190 | 1.173 | 0.710 | 0.732 | 18.1 | yes |
| 25 | 4683.9 | 11085.1 | 1.140 | 1.132 | 0.700 | 0.699 | 34.0 | yes |
| 29 | 1199.4 | 6223.9 | 0.960 | 0.974 | 0.630 | 0.620 | 110.2 | yes |

GA 11, 33 and 37 are excluded from the primary Table-I calibration because
the present UA diameter curve is not directly anchored at those ages. They
must not be represented as equally validated extrapolations.

## Table II: simultaneous PI/RI/UV-flow fit

| GA | PI target | PI production | RI target | RI production | UV flow target | UV flow production | UV mean velocity | Periodic |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 19 | 1.295 | 1.252 | 0.740 | 0.786 | 31.2 | 2.5 | 0.50 | yes |
| 23 | 1.190 | 1.119 | 0.710 | 0.709 | 71.7 | 30.3 | 2.75 | yes |
| 25 | 1.140 | 1.132 | 0.700 | 0.700 | 92.7 | 38.5 | 2.69 | yes |
| 29 | 0.960 | 0.975 | 0.630 | 0.621 | 141.8 | 65.3 | 3.19 | yes |
| 33 | 0.905 | 0.924 | 0.600 | 0.594 | 202.3 | 199.8 | 7.66 | no (15 cycles) |

Flow and velocity units are mL/min and cm/s. The GA-33 UA radius is an
extrapolation, although its UV diameter is within the literature-supported
range.

## Interpretation

The corrected 72.2 cm UA model can reproduce the PI/RI trend reasonably at
23-29 weeks, with all production checks periodic and no outlet fallbacks.
The fit is weaker at 19 weeks.

The current three-parameter model does **not** jointly reproduce UA PI/RI
and the literature UV-flow targets at 19-29 weeks. The shortfall is about
54-92%. Only GA 33 approaches its flow target, and that production run did
not yet satisfy the periodicity tolerance. A wider resistance search was
attempted at GA 19, but was computationally expensive and did not supersede
this completed table.

This is evidence of structural incompatibility or parameter
non-identifiability in the present pressure-driven UA + RCR formulation,
not validation of the original Table-II claim. The UV calculation is a
one-way, mass-conserving reduced 1D return model: summed UA flow drives the
UV, and the UV model calculates velocity and viscous/inertial pressure drop.
UV pressure has not yet been fed back into the placental outlet boundary.

## Required next model change

Replace the terminal RCR-only return assumption with an explicitly coupled
closed loop:

`two UAs -> placental arterial resistance/compliance -> exchange resistance -> placental venous compliance -> 1D UV -> fetal venous pressure`.

That adds a physical venous pressure state instead of forcing p0 to serve as
an empirical flow-control parameter. Only after this bidirectional coupling
should the model be recalibrated simultaneously to UA indices and UV flow.
