# MAsNMF

Modularized symmetric NMtri-F.

$$
\begin{aligned}
    \min \|\mathbf{X} - \mathbf{WHW}^\mathrm{T}\|_F^2 - \text{tr}(\hat{\mathbf{W}}{}^\mathrm{T}\mathbf{B}\hat{\mathbf{W}}) + \text{dist}(\mathbf{W},\hat{\mathbf{W}})
\end{aligned}
$$
where $\mathbf{B}$ is modularity matrix.

### # updating rules

$$
    \begin{aligned}
        \mathbf{W}_{ir} &\leftarrow \mathbf{W}_{ir} \cdot \left(
\frac{(
\mathbf{XWH}^\mathrm{T} + \mathbf{X}^\mathrm{T}\mathbf{WH} + \mu\hat{\mathbf{W}}
)_{ir}}{(
\mathbf{WHW}^\mathrm{T}\mathbf{WH}^\mathrm{T}+\mathbf{WH}^\mathrm{T}\mathbf{W}^\mathrm{T}\mathbf{WH}+\mu
)_{ir}}
        \right) ^ {\frac{1}{4}}\ , \\
        \mathbf{H}_{rs} &\leftarrow \mathbf{H}_{rs} \cdot\frac{(
        \mathbf{W}^\mathrm{T}\mathbf{XW}
        )_{rs}}{(
        \mathbf{W}^\mathrm{T}\mathbf{WHW}^\mathrm{T}\mathbf{W}
        )_{rs}}\ , \\
        \hat{\mathbf{W}}_{js} &\leftarrow \hat{\mathbf{W}}_{js} \cdot \left(\frac{(
         \mathbf{A}\hat{\mathbf{W}} + \mathbf{A}^\mathrm{T}\hat{\mathbf{W}} + 2\mu \mathbf{W}
        )_{js}}{(
         \mathbf{B}_1\hat{\mathbf{W}} + \mathbf{B}_1^\mathrm{T}\hat{\mathbf{W}} + 2\mu \hat{\mathbf{W}}
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
