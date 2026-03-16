
# DoubleF
#### A Fast and Flexible Phase Association and Earthquake Location Method Using Adaptive Sobol Sampling
1. We define a search space over the source parameters and initialize it with a quasi-uniform set of samples generated using a Sobol sequence, which provides an efficient and systematic exploration of high-dimensional parameter spaces with low discrepancy. 
2. The objective function is evaluated at each sampled point, and the search region is subsequently refined based on the quantile range of the top-value samples.
3.  This process is implemented iteratively. In each iteration, the search space is progressively narrowed to focus on regions most likely to contain the global optimum. 
4.  The iteration continues until a predefined convergence criterion is met. 
5.  Finally, the sample with the highest objective function value is selected, and the phases associated within an acceptable residual threshold are considered the optimal association for the event.

![Iterative search process](src/doublef/img/plot.png)

## Environment Setup and Installation

A dedicated Conda environment is recommended.

### Create a Conda environment

```bash
conda create -n doublef python=3.9
conda activate doublef
pip install doublef
```

If you prefer to install from source inside the environment:

```bash
conda create -n doublef python=3.9
conda activate doublef
git clone https://github.com/Lonngfei/DoubleF.git
cd DoubleF
pip install .
```


### GPU support

If you plan to run DoubleF on a GPU, make sure that your **PyTorch** version matches your **CUDA** version.

Please refer to the official PyTorch installation guide: [https://pytorch.org/get-started/previous-versions](https://pytorch.org/get-started/previous-versions).

### Confirm CUDA availability

Run the following in Python:

```python
import torch
print(torch.cuda.is_available())
```

* `True`: CUDA is available and PyTorch can use your GPU.
* `False`: Check the NVIDIA driver and CUDA-PyTorch compatibility.

### Show program information

```bash
doublef
```

This prints the program name,  version, and basic usage information.

## Input Files

Before running DoubleF, make sure the required input files are correctly prepared.

Typical inputs include:

```text
Picks/YYYYMMDD.csv       # Daily pick files
TravelTime/mymodel.nd    # Velocity model used for travel-time calculation
example.config           # Configuration file
```

The exact directory structure can be adjusted in the configuration file.

## Configuration

DoubleF is controlled through a configuration file such as `example.config`.

Most parameters do not need frequent modification. In most cases, only a few settings require special attention.

### 1. Velocity model and travel-time table

Set `cal_tt = True` when using a new velocity model.

This is usually required only once to generate the travel-time tables.

After the tables have been generated, set: `cal_tt = False`, so that the program loads the existing tables and skips recalculation.

### 2. Sampling parameters

In most applications, the default sampling settings are sufficient.

If the nearest-station distance is larger than **0.6°**, increasing the number of samples may improve the results.

### 3. Score calculation

DoubleF provides several alternative objective functions.

In most cases, the choice among these objective functions does not significantly affect the final results.

Users who are familiar with the method may further customize the scoring strategy if needed.

A custom objective function can be implemented by modifying: ``weight.py``, ``batch_weight.py`` in the source code.

### 4. Output settings

Set the output directory and related options according to your needs.

DoubleF automatically writes:

* logs
* configuration records
* phase association results

to the specified output path.

### 5. Memory and speed

#### `max_batch_size`

This parameter only affects computational efficiency and does **not** affect the final results. 

In general, a larger value may improve speed, but this is not always the case.  

Once the computation reaches saturation, further increasing `max_batch_size` may provide little or no additional speedup, while leading to higher memory usage.

### 6. Visualization

Visualization is usually recommended to be turned off during normal runs. 

It should only be enabled when intermediate inspection, debugging, or result checking is needed.


## Running the Program

Once the input files and configuration file are ready, run:

```bash
doublef example.config
```

If the installation and configuration are correct, DoubleF will start processing and write logs and results to the output directory.


## Output Phase File Format

A typical output phase file has the following format:

```text
# Year Month Day Hour Minute Second Latitude Longitude Depth Magnitude ErrHorizontal ErrVertical ErrTime RMS NumP NumS NumBoth NumSum ID
NET Station Distance PhaseTime Probability PhaseType Residual ML Mag Amplitude
```

### Notes

* `ErrHorizontal`, `ErrVertical`, and `ErrTime` do not necessarily represent the true location uncertainty.
*  These values describe the spatial dispersion of candidate solutions that are associated with the same nearby location. Specifically, they quantify the statistical deviation (e.g., mean or standard deviation) of these candidate locations relative to the final solution.
*  If only a single candidate solution exists, the dispersion cannot be computed and the value is reported as **NaN**.
*  When these values are unusually large or reported as NaN, the corresponding results should be interpreted with caution.
