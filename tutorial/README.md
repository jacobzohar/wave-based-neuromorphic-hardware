# tutorial/ — minimal walkthrough of the spin-wave-disk reservoir simulator

`SWRC_MuMax3.ipynb` is the original, single-file walkthrough that documents the
SWRC mumax³ substrate end-to-end in ~200 lines:

- `ReservoirInput(...)` — emit a mumax³ script for the 1 µm disk substrate,
  with configurable bias field, drive frequency / pulsewidth / amplitude and a
  list of input pulse times.
- `Run / ReadFiles / ReadTable` — invoke `mumax3`, parse the OVF outputs.
- `Compress` — crop + block-mean the per-frame `m_z` field into a 50×50 CSV.
- `DataRun(...)` — outer loop over a list of input patterns (× 5 repeats).

Read this notebook first if you want to understand the substrate physics and
pipeline without parsing the production code. **For the production
data-generation pipeline — multi-GPU sweeps, resumable runs, externalised
`.mx3` sources, and the cube-stacking step that produces the `(N, T, H, W)`
arrays the analysis loads — see [`../simulator/`](../simulator/)**, which is
the direct descendant of this notebook with the same geometry and material.

Requirements: a `mumax3` binary on PATH (https://mumax.github.io/), plus
`numpy` and `pandas` (see top-level `requirements.txt`).
