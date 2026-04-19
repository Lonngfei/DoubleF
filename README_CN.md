# DoubleF

DoubleF 是一个基于自适应 Sobol 采样的地震相位关联与震源定位程序。它在统一框架中搜索事件的纬度、经度、深度和发震时刻，并围绕评分最高的候选结果逐步收缩搜索范围。

![迭代搜索过程](src/doublef/img/plot.png)

## 亮点

- 自适应 Sobol 采样：在四维震源参数空间中高效搜索，无需穷举网格。
- 关联与定位一体化：相位关联和震源位置估计同时求解。
- 灵活评分：相位数、残差、概率和距离加权可以以不同方式组合。
- GPU 加速：主要张量计算在可用时可运行在 CUDA 上。
- 稳定输出：最终相位报告自动对齐，并以紧凑格式写出。

## 安装

```bash
conda create -n doublef python=3.9
conda activate doublef
pip install .
```

如果希望使用 GPU，请在安装 DoubleF 之前先安装与本机 CUDA 运行时匹配的 PyTorch。

## 命令行

使用 YAML 配置文件运行 DoubleF：

```bash
doublef example/example.yaml
```

显示程序信息：

```bash
doublef
```

## 示例

仓库在 `example/` 中提供了一个可直接运行的示例：

- `example/example.yaml`：示例配置
- `example/example.csv`：示例 pick 文件
- `example/example_stations.csv`：台站文件
- `example/tt/example.nd`：速度模型

公开示例使用 `recompute-travel-time: true`，因此第一次运行会在 `example/tt/` 中建立走时表。建立完成后，请将其设置为 `false`，以避免重复计算走时表。

## 输入文件

### Pick CSV

必需列：

- `network`
- `station`
- `phasetype`
- `Time`
- `Probability`
- `Amplitude`

### Station CSV

必需列：

- `network`
- `station`
- `latitude`
- `longitude`
- `elevation`

### 速度模型

DoubleF 使用一维速度模型文件进行走时计算。

## 配置

DoubleF 通过 YAML 文件控制。完整示例见 [`example/example.yaml`](example/example.yaml)。

在正常使用中，大多数参数不会强烈改变最终结果。通常不需要一开始就调整所有参数。

最值得关注的两个部分是：

- `tolerance`
- `output`

这两部分与实际数据质量、台站覆盖、拾取不确定性、噪声水平以及分析的实际目标密切相关。

对于更大的搜索区域、更宽的研究范围或更大的初始不确定性，适当增加 `number-of-samples` 往往更有帮助。在许多更小、更密集的问题中，默认采样通常已经足够。

更详细的参数说明请参考 YAML 文件中的注释。

## 输出文件

每次运行都会在 `output.output-directory` 下创建一个实验目录，例如：

```text
results/000-example/
```

常见文件：

- `example.log`：完整日志文件
- `Config.yaml`：本次运行使用的归一化有效配置
- `example.phase`：最终关联与定位结果报告

## 相位报告格式

相位报告顶部不包含说明性标题行。

每个事件由以下部分组成：

- 一行事件信息
- 多行关联到的 pick 信息

### 事件行

```text
# YYYY MM DD HH MM SS LAT LON DEP MAG ERR_LAT ERR_LON ERR_DEP ERR_TIME RMS P S BOTH SUM ID
```

重要说明：

- `ERR_LAT`、`ERR_LON`、`ERR_DEP` 和 `ERR_TIME` 不是正式的定位误差。
- 它们描述的是最终一次迭代搜索结果在所选解附近的离散范围。
- 较大的数值可能表示结果稳定性较差、采样数不足、`quantile` 过宽，或者解空间仍然较为分散。
- 应将它们理解为解集中程度的指示，而不是严格的不确定性估计。

### Pick 行

```text
NET STATION DIST PICK_TIME PROB PHASE RESIDUAL MAG AMP
```
