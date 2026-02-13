# MAsNMF

Modularized symmetric NMtri-F.

$\min_{} \| \mathbf{X} - \mathbf{WHW}^\top \| _F^2 - \text{tr}(\hat{\mathbf{W}}^\top\mathbf{B}\hat{\mathbf{W}}) + \text{dist}(\mathbf{W},\hat{\mathbf{W}})$
where $\mathbf{B}$ is modularity matrix.

### # updating rules

$$
    \begin{aligned}
        \mathbf{W}_{ir} &\leftarrow \mathbf{W}_{ir} \cdot \left(
\frac{(
\mathbf{XWH}^\top + \mathbf{X}^\top\mathbf{WH} + \mu\hat{\mathbf{W}}
)_{ir}}{(
\mathbf{WHW}^\top\mathbf{WH}^\top+\mathbf{WH}^\top\mathbf{W}^\top\mathbf{WH}+\mu
)_{ir}}
        \right) ^ {\frac{1}{4}}\ , \\
        \mathbf{H}_{rs} &\leftarrow \mathbf{H}_{rs} \cdot\frac{(
        \mathbf{W}^\top\mathbf{XW}
        )_{rs}}{(
        \mathbf{W}^\top\mathbf{WHW}^\top\mathbf{W}
        )_{rs}}\ , \\
        \hat{\mathbf{W}}_{js} &\leftarrow \hat{\mathbf{W}}_{js} \cdot \left(\frac{(
         \mathbf{A}\hat{\mathbf{W}} + \mathbf{A}^\top\hat{\mathbf{W}} + 2\mu \mathbf{W}
        )_{js}}{(
         \mathbf{B}_1\hat{\mathbf{W}} + \mathbf{B}_1^\top\hat{\mathbf{W}} + 2\mu \hat{\mathbf{W}}
        )_{js}}\right)^\frac{1}{2}\ , \\
        \hat{\mathbf{W}}_{js} &\leftarrow \hat{\mathbf{W}}_{js} /
        \sum_{l=1}^k \hat{\mathbf{W}}_{jl}\ .
    \end{aligned}
$$

## Installation

``` bash
conda create -n msnmf python=3.9
conda activate msnmf
```

``` bash
pip install -r requirements.txt
```

## Citation

If you find this code useful, please cite:
