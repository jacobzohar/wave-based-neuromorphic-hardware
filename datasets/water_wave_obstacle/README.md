# Water-wave reservoir obstacle-distance datasets

Balanced random-dataset generators for the water-wave reservoir's
obstacle-classification task. Each sample is a 6-tuple of integer "distance
readings" from the robotic vehicle's six ultrasonic sensors, mapped (in the
physical experiment) to the wave initiation times that drive the six
actuators. Labels encode where obstacles are detected.

Used to produce the input data for the water-wave reservoir results in the
manuscript (Fig. 2e — accuracy vs training-set size; Fig. 5b — image-input
variant).

## Files

| File | Classes | Distance range | Manuscript role |
|------|--------:|----------------|-----------------|
| [`2_output_unique_data_generation.ipynb`](2_output_unique_data_generation.ipynb) | 2 | 1–6 | Binary obstacle / no-obstacle detection |
| [`3_output_unique_data_generation.ipynb`](3_output_unique_data_generation.ipynb) | 3 | 1–6 | Ternary: clear / obstacle-left / obstacle-right |
| [`5_output_unique_data_generation.ipynb`](5_output_unique_data_generation.ipynb) | 5 | 1–7 | 5-way: clear / near-left / near-right / mid-left / mid-right |

The 5-class variant uses an extended distance range (1–7 rather than 1–6)
because the near/mid split halves the resolution available per side; the
extra level restores enough dispersion to keep the classes well separated.

## Encoding

The six sensor readings are split into a left half (sensors 0, 1, 2) and a
right half (sensors 3, 4, 5). Within each variant, a "near" range of distance
values marks the presence of an obstacle on that side; the highest values
mark a clear field of view.

| Variant | Near range | Mid range | Far range |
|---------|------------|-----------|-----------|
| 2-class | 1, 2, 3 | — | 4, 5, 6 |
| 3-class | 1, 2, 3 | — | 4, 5, 6 |
| 5-class | 1, 2 | 3, 4 | 5, 6, 7 |

The 3- and 5-class generators enforce side-exclusivity: a sample is only
emitted if at most one half contains "near"/"mid" values, so each non-clear
class corresponds unambiguously to one side of the vehicle.

## Usage

```python
import sys, importlib.util

spec = importlib.util.spec_from_file_location(
    "gen", "datasets/water_wave_obstacle/3_output_unique_data_generation.ipynb")
# For convenience the same logic also runs directly inside Jupyter:
#   jupyter notebook datasets/water_wave_obstacle/3_output_unique_data_generation.ipynb

# Inside the notebook (or after exec'ing its single cell):
data, labels = generate_balanced_dataset(2000)   # 2000 must divide by n_classes
# data   : list of 2000 six-int lists
# labels : list of 2000 ints in {0, …, n_classes-1}, balanced
```

All three notebooks expose the same two-function interface:

- `determine_label(sample) -> int` — pure function from a 6-tuple to its class label.
- `generate_balanced_dataset(n_total) -> [data, labels]` — rejection-samples
  unique 6-tuples until the dataset is full and class-balanced. `n_total`
  must be divisible by the number of classes.

The generators are deterministic up to `random.seed(...)` if you set one
before calling `generate_balanced_dataset`.

## Reproducibility note

The notebooks are the canonical generators used to assemble the training and
testing splits for the water-wave reservoir results. Once a dataset is
generated, it is fed into the water-wave reservoir setup described in the
Methods section: the six distance levels are mapped to wave initiation times,
the six actuators are driven accordingly, and the wave-network response is
read out by the six sensor lines and trained against the corresponding label
vector.
