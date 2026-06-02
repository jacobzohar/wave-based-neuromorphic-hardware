# `GEOMETRY.md` — simulator parameters

This file documents every parameter used by the SWRC simulator: symbol, value,
units, physical meaning, and where each lives in the code. It is the canonical
reference for the substrate physics — both `.mx3` files
([`build_relax_cache.mx3`](build_relax_cache.mx3) and
[`test_sample.mx3`](test_sample.mx3)) and the embedded templates in
[`run_sweep.py`](run_sweep.py) use these exact values.

## Geometry

| Symbol | Value | Units | Description |
|--------|------:|:------|-------------|
| Disk diameter | 1000 | nm | Circular magnetic disk, single z-layer (perpendicular thin film). |
| Cell size (x, y) | 2 × 2 | nm | In-plane discretisation. Below the smallest spin-wave wavelength of interest (~10 nm at 800 MHz in these material parameters). |
| Cell thickness (z) | 1.5 | nm | Single-layer thickness, characteristic of thin Pt/CoFeB/MgO stacks. |
| Grid | 512 × 512 × 1 | cells | Total simulation cell count. Disk occupies ~π/4 ≈ 78 % of the box. |
| Transducer (actuator) radius | 50 | nm | Circular VCMA actuator region. |
| Transducer radial position | 300 | nm | Distance of actuator centres from disk centre. |
| Number of actuator regions | `(num_inputs + 1) · 2 = 6` | — | Evenly spaced around the disk circumference; **2 are driven inputs** (the others are unused passive regions, included so the grid symmetry matches the experimental device with up to 6 active electrodes). |

## Magnetic material (CoFeB-like, perpendicular anisotropy)

| Symbol | Value | Units | Description |
|--------|------:|:------|-------------|
| `Msat` | 1 × 10⁶ | A/m | Saturation magnetisation. Representative of thin CoFeB. |
| `Aex` | 1.5 × 10⁻¹¹ | J/m | Exchange stiffness. Standard CoFeB value. |
| `alpha` | 0.012 | — | Gilbert damping. Low-damping thin-film value (set below 0.02 to allow spin-wave propagation across the disk). |
| `Ku1` (static) | 6.3 × 10⁵ | J/m³ | First-order uniaxial perpendicular anisotropy constant. Chosen to sit **just above the spin-reorientation transition** (SRT) so the equilibrium is canted near the SRT and the disk is at maximum susceptibility to dynamic Ku1 modulation (= maximum spin-wave drive efficiency per applied voltage). |
| `anisU` | (0, 0, 1) | — | Easy-axis direction (out-of-plane). |
| `DisableZhangLiTorque` | true | — | Spin-transfer torques disabled — drive is via VCMA only, not current injection. |

## Bias and drive

| Symbol | Value | Units | Description |
|--------|------:|:------|-------------|
| `B_ext` (bias) | (45, 0, 0) | mT | Static in-plane (+x) bias field. Tilts the equilibrium magnetisation in-plane, setting the static spin-wave dispersion. |
| Drive carrier frequency `f` | 800 | MHz | RF carrier of the Gaussian-windowed input pulse. Sits in the dispersion band of the bias-tilted disk. |
| Drive pulsewidth `σ` | 1.25 | ns | Gaussian envelope 1/e half-width. Sets the spectral bandwidth of one input event (≈ 1/(2πσ) ≈ 130 MHz). |
| Drive amplitude (peak dynamic Ku1) | 8 × 10⁵ | J/m³ | **The peak amplitude of the dynamic `Ku1` modulation that one input pulse imposes on its actuator region** (see § VCMA below). Larger than the static `Ku1 = 6.3 × 10⁵` J/m³ to drive the disk firmly across the SRT during the pulse. |
| Pulse-centre offset | 0.2 | ns | Extra delay added on top of `coord · T_step` so the leading edge of each pulse sits inside the simulation window. |

## VCMA — voltage-controlled magnetic anisotropy

In a physical Pt/CoFeB/MgO heterostructure, an applied voltage across the
MgO barrier modulates the interfacial perpendicular anisotropy via the VCMA
effect:

> ΔKu1(t) = ξ_VCMA · E_MgO(t) / t_MgO

where ξ_VCMA is the device VCMA coefficient (typically in the
**50 – 200 fJ / V·m** range for sputtered Pt/CoFeB/MgO at room temperature),
E_MgO is the electric field across the barrier, and t_MgO its thickness.

**In this simulator we specify `ΔKu1(t)` directly** as a Gaussian-windowed
800 MHz tone with peak amplitude `8 × 10⁵ J/m³` on the chosen actuator region:

```
Ku1.setRegion(i, Amplitude * exp(-((t - tc - offset)/σ)²) * sin(2π·f·(t - tc - offset)))
```

with `Amplitude = 8e5`, `σ = 1.25 ns`, `f = 800 MHz`, `tc = coord · T_step`.
We do **not** model the voltage-to-anisotropy conversion explicitly — the
`Amplitude` parameter is the *effective* peak ΔKu1 that a particular
experimental voltage would produce, given the device's VCMA coefficient. To
match a specific physical device, set
`Amplitude_simulator = ξ_VCMA · V_peak / t_MgO`.

This abstraction lets the simulator be device-agnostic: any Pt/CoFeB/MgO-class
heterostructure with the documented Msat / Aex / α / Ku1 can be matched by
solving for the V_peak that gives the same effective ΔKu1.

## Encoding (coordinate → pulse-centre time)

| Symbol | Default | Units | Description |
|--------|------:|:------|-------------|
| `T_step` | 0.05 | ns | Per-coordinate-step delay. Pulse-centre time for input `i` at coordinate `c` is `tc_i = c · T_step + 0.2 ns`. |
| `Tmax` | 4 | ns | Total simulation time per sample. Long enough to include the input pulse + spin-wave ringdown. |
| `FixDt` | 2 × 10⁻¹³ | s | Fixed integration step. ≈ 250 steps per RF carrier period. |
| `100·FixDt` | 20 | ps | Output sample period — one m_z snapshot every 20 ps → `Tmax / 20 ps + 1` frames per sample (201 frames at `Tmax = 4 ns`). |
| Coordinate range | 0 .. `GRID − 1` | — | Each input takes an integer coordinate; default `GRID = 32` → 32 × 32 = 1024 unique input pairs per sweep. |

## Coarse-graining for downstream analysis

The `m_z` field is stored per cell at the simulation resolution (512²), then
downsampled before being handed to the classifier:

| Symbol | Default | Description |
|--------|------:|-------------|
| Crop window | `[64 : 448]` of 512 | Centred 384×384 crop around the disk (transducers ~ centred, edge cells dropped). |
| Block-mean factor | 6 | 384 / 64 = 6 → output frames are 64×64 (≈ 12 nm per output cell). |

The crop + block-mean step is performed in-process by `run_sweep.py` so that
no full-resolution OVF files are retained on disk (they're 1 MB / frame each,
which would be ~200 GB for the production sweep — at 64×64 the per-sample
output is ~5 MB).

## Provenance of each value

Every value above is in `build_relax_cache.mx3`, `test_sample.mx3`, and
`run_sweep.py` (the script's `_GEOM` template and module-level constants).
Changing a value in one place without updating the others will produce
inconsistent samples; the recommended workflow is to change a parameter in
`run_sweep.py` (single source of truth for sweeps) and copy it into the
`.mx3` files if you need a matching standalone reference.
