# `experiments/` — downstream analyses

Each subfolder is a self-contained analysis bundle for one figure (or
figure group) of the manuscript. They all sit on top of the SWRC simulator in
[`../simulator/`](../simulator/) — every analysis consumes an `m_z` cube
produced by that simulator and reports a classification or dimensionality
result. The bundled `.npz` caches are sufficient to regenerate the published
figures without the raw cubes.

| Folder | What it does | Manuscript element |
|--------|--------------|--------------------|
| [`xor_checkerboard/`](xor_checkerboard/) | SWR + MLP + CNN compared on an XOR-checkerboard task ladder (balanced accuracy vs FLOPs) | Supplementary Fig. 7 |
| [`kernel_rank_cg/`](kernel_rank_cg/) | Kernel rank, d95, participation ratio of the reservoir feature matrix vs spatial coarse-graining | Supp. Table 2 / Table S9.1, Fig. S9.2 |

See each subfolder's `README.md` for the analysis pipeline, bundled caches,
and step-by-step regeneration recipe.
