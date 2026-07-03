# Track-a-Tree: Cell Lineage Construction by Simulated Annealing

Track-a-Tree reconstructs cell lineage trees from time-lapse fluorescence microscopy by globally optimizing a cost function over all parent-daughter assignments using simulated annealing. Unlike greedy frame-by-frame tracking, the global optimization correctly handles the case where multiple divisions occur between consecutive frames — a common situation in early embryo imaging at low frame rates.

The method is applied to two model systems:
- **Preimplantation mouse embryos** (8→16, 16→32, and 32→64 cell stages)
- **C. elegans embryos** (early cleavage stages)

For each organism, a dataset of nuclear features (centroids, volumes, shape descriptors, intensities) and a ground-truth lineage graph are included so you can run the tutorial immediately.

---

## Installation

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate lineage-sa
```

---

## Tutorial

Open `Notebooks/lineage_tutorial.ipynb` in Jupyter and follow the instructions there. The notebook walks through loading features, running simulated annealing, evaluating the result against the ground truth, and visualizing the reconstructed lineage for both the mouse and worm datasets.

```bash
jupyter notebook Notebooks/lineage_tutorial.ipynb
```

---

## Code Structure

```
ComputeLineage/
  sim_anneal.py          # SA engine: Tracks, anneal(), tracksCost(), tracksShuffle()
  NewCostFunction.py     # Default cost function: MyCostParams, GeneralCostFunc
  sim_anneal_utils.py    # Utilities: tracksInit, OutputGraph, Eval, GetMotherFromNode
  ComputeLineage.py      # CLI: run SA on a config file + frame range
  EvaluateLineage.py     # Compare a saved sim_graph against ground truth
  EvaluateCost.py        # Compute per-term cost breakdown

CheckResults/
  check_splits.py        # Print split statistics for a saved result
  plot_lineage.py        # Plot the lineage tree

Config/
  220827_stack1_config.yaml   # Mouse embryo configuration
  Worm_config.yaml            # C. elegans configuration

Data/
  Mouse/220827_stack1/        # Mouse features, ground truth, precomputed results
  Worm/                       # Worm features, ground truth, precomputed results
```

### Running from the command line

```bash
cd ComputeLineage
python ComputeLineage.py -c 220827_stack1 -s 14 -e 32 --phase 16to32
```

`-c` is the config name (without `_config.yaml`), `-s`/`-e` are the feature-file start/end frames, and `--phase` selects the SA parameter block from the config file.

---

## Cost Function

The total cost is a sum of local edge costs plus a global balance penalty:

- **Centroid distance** — penalizes large displacement from mother to daughter(s)
- **Volume ratio** — penalizes departure from expected daughter/mother volume fractions
- **Division angle** — penalizes deviation from the expected opening angle (~156° in mouse)
- **Symmetry** — penalizes asymmetric divisions (|vol₁ − vol₂| / mean)
- **Aspect ratio** — rewards elongated mothers at division
- **Mean / std intensity** — penalizes intensity discontinuities
- **Balanced forest** (`splitWt`) — a global penalty that enforces the constraint that each cell divides at most once per stage, essential for correct reconstruction at low frame rates

All weights are set in the config YAML under the `simulated_annealing` key and passed to the SA engine via `MyCostParams`.

---

## Writing a Custom Cost Function

The SA engine in `sim_anneal.py` accepts a `daughterCostFunc` argument to both `anneal()` and `tracksCost()`. This callable receives a mother nucleus and its candidate daughter(s) and must return a scalar cost.

The default implementation lives in `NewCostFunction.py`. The pattern is a factory function that closes over a parameter object:

```python
from sim_anneal import Nucleus   # type hint; Nucleus has .centroid, .volume, .label, etc.

def MyCostFunc(params):
    """Return the callable that the SA engine will call."""
    def cost(mother, d1, d2=None):
        """
        mother, d1, d2: Nucleus objects (d2 is None for 1-to-1 assignments).
        Return a scalar >= 0; lower is better.
        """
        c = 0.0
        # 1-to-1 link
        dist = sum((a - b)**2 for a, b in zip(mother.centroid, d1.centroid))**0.5
        c += params.centWt * dist**2

        # division link
        if d2 is not None:
            vol_ratio = (d1.volume + d2.volume) / (mother.volume + 1e-6)
            c += params.volWt * (vol_ratio - params.volSplitMult)**2

        return c
    return cost

# Usage:
tracks = sim_anneal.anneal(
    tracks, epochs=2000,
    daughterCostFunc=MyCostFunc(my_params),
    splitWt=8000,
)
```

Study `NewCostFunction.py:GeneralCostFunc` for the full set of terms and `sim_anneal.py:tracksCost` for the exact calling convention. The cost function is called independently for each mother-daughter edge, so it must not maintain state across calls.

---

## Using Custom Features

Features are stored in `Features.json` — a dict of named arrays, each indexed by `[frame][label]`:

```json
{
  "centroids":       [[...], [...], ...],
  "volumes":         [[...], [...], ...],
  "aspectRatios":    [[...], [...], ...],
  "meanIntensities": [[...], [...], ...]
}
```

Each `Nucleus` object inside `Tracks` exposes these as attributes (`.centroid`, `.volume`, `.solidity`, etc.) after `tracksInit()` populates them.

**To add a new feature:**

1. **Add it to `Features.json`** — append a key whose value is a 2-D list with the same `[frame][label]` shape as the existing arrays.

2. **Load it in `tracksInit`** (`sim_anneal_utils.py`) — add one line in the frame-label loop:
   ```python
   nucleus.my_feature = data['my_feature'][iframe][label_idx]
   ```

3. **Use it in your cost function** — access `mother.my_feature`, `d1.my_feature`, etc. inside your `daughterCostFunc`.

The worm and mouse datasets include centroids, volumes, solidity, aspect ratios, and mean/std intensities. The paper also describes an optional extended pipeline that incorporates CNN classifier probabilities for uncertain divisions; `MyCostParams` has a `probs` field for this.

---

## Citation

If you use this code, please cite:

> Nunley, Grover, Denberg, Avdeeva, Watters, Shvartsman, Posfai, Brown.
> *Track-a-Tree: lineage construction during early embryonic development.*
