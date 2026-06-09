# `Demo/` — runnable demonstrations of the software

End-to-end demonstrations of the wave-based neuromorphic computing
pipeline. Two top-level groups: one application-focused demo (full
obstacle-classification workflow) and a `reservoir_metrics/` group
that quantifies intrinsic properties of the substrate. Each subfolder
is self-contained — every input it needs is bundled, running its
script regenerates the published figure(s).

| Folder | What it does | Manuscript element |
|--------|--------------|--------------------|
| [`2_3_5_Robotic_output_classification_task/`](2_3_5_Robotic_output_classification_task/) | Full obstacle-classification workflow — dataset generation, simulator-input mapping, expected simulator output, linear-SVM training + testing for the water-wave vs Raw SVM vs SWRC comparison | Fig. 2e |
| [`reservoir_metrics/xor_chequerboard/`](reservoir_metrics/xor_chequerboard/) | SWR + MLP + CNN compared on an XOR-chequerboard task ladder (balanced accuracy vs FLOPs) | Supplementary Fig. 7 |
| [`reservoir_metrics/kernel_rank_cg/`](reservoir_metrics/kernel_rank_cg/) | Kernel rank, d95, participation ratio of the reservoir feature matrix vs spatial coarse-graining | Supp. Table 2 / Table S9.1, Fig. S9.2 |

See each subfolder's `README.md` for the bundled inputs, the run
command, and the expected output.
