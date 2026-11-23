# HighPrecisionLinearAlgebra.jl

[![Julia 1.12+](https://badgen.net/static/Julia/1.12%2B/007ec6?icon=https%3A%2F%2Fraw.githubusercontent.com%2FJuliaLang%2Fjulia-logo-graphics%2Fmaster%2Fimages%2Fjulia-dots.svg)](https://julialang.org/downloads/)
[![CI](https://github.com/JuliaFewBody/HighPrecisionLinearAlgebra.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaFewBody/HighPrecisionLinearAlgebra.jl/actions/workflows/CI.yml?query=branch%3Amain)

Parallel dense linear algebra in extended precision (`BigFloat`, and `Float64x2`/`Float64x4` from MultiFloats.jl).

- The trailing updates of Cholesky factorizations and triangular solves use the Ozaki scheme: the operands are split
  into integers (slices, or residues modulo primes) whose products are computed exactly by Float64 BLAS, and the result
  is rounded once.
- Extended-precision inner loops use in-place MPFR operations that do not allocate, so they scale with threads.
- `SymPacked` stores a symmetric `BigFloat` matrix in about 1/3 of the memory of a `Matrix{BigFloat}`.

## Functions

| | |
|---|---|
| Cholesky | `chol_lower_ozaki!`, `chol_lower!`, `chol_lower_persist!` |
| Triangular solves | `llt_solve!`, `lt_solve!`, `ltri_solve!`, `trsm_lower_ozaki!` |
| Eigenvalues | `lowest_eig_si` (lowest eigenpair of H x = E S x), `eigen_mt`, `eigen_proj` |
| Storage | `SymPacked`, `PackedMatrix` |
| MPFR | `mpfr_add!`, `mpfr_sub!`, `mpfr_mul!`, `mpfr_div!`, `mpfr_fma!`, `mpfr_sqrt!`, … |

Functions ending in `!` overwrite their arguments in place; a `BigFloat` matrix passed to them must hold distinct
objects (use `fresh`).

## Quick Start

Requires [Julia](https://julialang.org/install/) 1.12 or later.

```julia
import Pkg; Pkg.add(url="https://github.com/JuliaFewBody/HighPrecisionLinearAlgebra.jl.git")
using HighPrecisionLinearAlgebra, LinearAlgebra
setprecision(BigFloat, 320)
X = rand(BigFloat, 300, 300) .- 0.5
S = SymPacked(X * X' / 300 + I)
chol_lower_ozaki!(S)                       # S now holds the Cholesky factor L
b = [BigFloat(1) for _ in 1:300]
x = llt_solve!(b, S)                       # solves L Lᵀ x = b in place
```

Start Julia with threads (`julia -t 16`).

## Development

```shell
git clone https://github.com/JuliaFewBody/HighPrecisionLinearAlgebra.jl.git
cd HighPrecisionLinearAlgebra.jl
julia --project=. --startup-file=no -e "using Pkg; Pkg.test()"
```

## Acknowledgments

This package is built on an initial project template generated using [Julia Package Factory](https://github.com/JuliaPackageFactory).
