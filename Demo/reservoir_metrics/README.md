# `reservoir_metrics/` — characterising the spin-wave-disk reservoir

Two demonstrations that quantify intrinsic properties of the
spin-wave-disk reservoir as a computational substrate, rather than its
performance on a specific application task. Both consume `m_z` cubes
produced by the simulator in [`../../simulator/`](../../simulator/) and
back the supplementary-text discussion of the reservoir's capacity and
expressivity.

| Folder | What it does | Manuscript element |
|--------|--------------|--------------------|
| [`xor_chequerboard/`](xor_chequerboard/) | SWR + MLP + CNN compared on an XOR-chequerboard task ladder (balanced accuracy vs FLOPs); shows the substrate performs a non-linear feature lift that solves fine-scale parity at compute budgets where the feed-forward baselines fail | Supplementary Fig. 7 |
| [`kernel_rank_cg/`](kernel_rank_cg/) | Kernel rank, d95 (PCs for 95 % variance) and participation ratio of the reservoir feature matrix Ψc as the `m_z` field is spatially coarse-grained from 50×50 down to 4×4 | Supp. Table 2 / Table S9.1, Fig. S9.2 |

See each subfolder's `README.md` for the bundled inputs, the run
command, and the expected output.
