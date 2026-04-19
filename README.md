# DoubleF

DoubleF is a phase association and earthquake location program based on adaptive Sobol sampling. It searches event latitude, longitude, depth, and origin time in a unified framework and progressively refines the search region around the best-scoring candidates.

![Iterative search process](src/doublef/img/plot.png)

## Highlights

- Adaptive Sobol sampling: efficient search in a 4-D source parameter space without an exhaustive grid.
- Unified association and location: phase association and hypocenter estimation are solved together.
- Flexible scoring: phase count, residual, probability, and distance weighting can be combined in different ways.
- GPU acceleration: the main tensor computations can run on CUDA when available.
- Stable output: final phase reports are aligned automatically and written in a compact format.

## Installation

```bash
conda create -n doublef python=3.9
conda activate doublef
pip install .
```

If you want GPU execution, install a PyTorch build that matches your CUDA runtime before installing DoubleF.

## Command Line

Run DoubleF with a YAML config file:

```bash
doublef example/example.yaml
```

Show program information:

```bash
doublef
```

## Example

The repository includes a runnable example in `example/`:

- `example/example.yaml`: example configuration
- `example/example.csv`: example pick file
- `example/example_stations.csv`: station file
- `example/tt/example.nd`: velocity model

The public example uses `recompute-travel-time: true`, so the first run will build travel-time tables in `example/tt/`. After the tables are built, set it to `false` to avoid repeated travel-time computation.

## Input Files

### Pick CSV

Required columns:

- `network`
- `station`
- `phasetype`
- `Time`
- `Probability`
- `Amplitude`

### Station CSV

Required columns:

- `network`
- `station`
- `latitude`
- `longitude`
- `elevation`

### Velocity Model

DoubleF uses a 1-D velocity model file for travel-time calculation.

## Configuration

DoubleF is controlled by a YAML file. A complete example is provided in [`example/example.yaml`](example/example.yaml).

In normal use, most parameters do not strongly change the final result. You usually do not need to tune everything.

The two sections that deserve the most attention are:

- `tolerance`
- `output`

These are strongly related to real data quality, station coverage, pick uncertainty, noise level, and the practical goal of the analysis.

For larger search regions, broader study areas, or larger initial uncertainty, increasing `number-of-samples` is often beneficial. In many smaller and denser problems, the default sampling is already adequate.

For more detailed parameter descriptions, refer to the comments in the YAML file.

## Output Files

Each run creates an experiment directory under `output.output-directory`, for example:

```text
results/000-example/
```

Typical files:

- `example.log`: full log file
- `Config.yaml`: normalized effective configuration used for the run
- `example.phase`: final association and location report

## Phase Report Format

The phase report does not contain explanatory header lines.

Each event consists of:

- one event line
- multiple associated pick lines

### Event Line

```text
# YYYY MM DD HH MM SS LAT LON DEP MAG ERR_LAT ERR_LON ERR_DEP ERR_TIME RMS P S BOTH SUM ID
```

Important note:

- `ERR_LAT`, `ERR_LON`, `ERR_DEP`, and `ERR_TIME` are not formal location errors.
- They describe the spread of the final iteration search result around the selected solution.
- Large values may indicate that the result is less stable, the sample count is insufficient, the quantile is too broad, or the solution space remains diffuse.
- They should be interpreted as indicators of solution concentration, not as strict uncertainty estimates.

### Pick Line

```text
NET STATION DIST PICK_TIME PROB PHASE RESIDUAL MAG AMP
```