# Characteristic-direction repair

## Root cause

For the implemented tube law, `c(A) = c_ref*(A0/A)**0.25`, so

`integral(c/A dA) = -4c`.

The Riemann invariant transported by `lambda+ = u+c` is therefore `u-4c`,
and the invariant transported by `lambda- = u-c` is `u+4c`.  The production
solver used the opposite assignment at both boundaries.  This made a positive
pressure inlet wave generate negative forward velocity and made the low-
resistance outlet follow an incorrect negative-flow branch.

## Minimal production correction

The isolated `solve_arterial_flow_UA.py` contains the minimal sign repair:

- inlet incoming invariant: `W1 = 4*c_ref - 8*c_target`;
- inlet outgoing invariant: `W2 = u_foot + 4*c_foot`;
- reconstruction: `c = (W2-W1)/8`;
- outlet shifted outgoing invariant: `J_plus = u - 4*(c-c_ref)`;
- outlet reconstruction: `c_b = c_ref + (u_b-J_plus)/4`.

It also adds explicit CFL control and fixed-physical-offset diagnostics.

## Independent reference implementation

`solve_arterial_flow_UA_fv.py` is a conservative cell-centred reference
solver using Rusanov fluxes and SSP-RK2.  The RCR pressure is advanced at both
Runge-Kutta stages, and the outlet state contributes through a numerical flux
rather than overwriting the final interior cell.

## Verification results

Late parameters, corrected legacy solver, original inlet profile:

| Mesh | location | minimum u | maximum u | mean u | raw PI |
|---:|---:|---:|---:|---:|---:|
| 100 | L-0.5 cm | -6.107 | 48.017 | 13.670 | 3.959 |
| 200 | L-0.5 cm | -6.189 | 47.891 | 13.637 | 3.9656 |
| 400 | L-0.5 cm | -6.211 | 47.852 | 13.631 | 3.9664 |

All three runs reached the periodicity criterion with zero outlet fallback.
The M=200 to M=400 change is small, unlike the pre-fix terminal results.

Independent finite-volume checks at `L-0.5 cm`:

- M=50, CFL=0.4: min -5.109, max 46.864, mean 13.466 cm/s.
- M=50, CFL=0.2: min -5.101, max 46.862, mean 13.466 cm/s.
- M=100, CFL=0.4: min -5.368, max 47.343, mean 13.549 cm/s.

The reference method is temporally stable under timestep halving and shows
the same qualitative, bounded Late waveform as the corrected legacy method.

## Consequences

All calibration, sensitivity, mesh-convergence, figures, and manuscript
tables generated with the previous invariant orientation are stale and must
be recomputed.  The fitted parameters cannot simply be retained, because
they compensated for a boundary condition with reversed characteristic
directions.

The Early corrected run remains numerically stable but still has a reverse-
flow interval.  That feature should be reassessed after recalibration and
waveform-level fitting; it is no longer evidence about the Late boundary bug.

