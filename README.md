# HighPrecisionLinearAlgebra.jl

[![Julia 1.12+](https://badgen.net/static/Julia/1.12%2B/007ec6?icon=https%3A%2F%2Fraw.githubusercontent.com%2FJuliaLang%2Fjulia-logo-graphics%2Fmaster%2Fimages%2Fjulia-dots.svg)](https://julialang.org/downloads/)
[![CI](https://github.com/JuliaFewBody/HighPrecisionLinearAlgebra.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaFewBody/HighPrecisionLinearAlgebra.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24542848.svg)](https://doi.org/10.5281/zenodo.24542848)

Parallel dense linear algebra in extended precision (`BigFloat`, and `Float64x2`/`Float64x4` from MultiFloats.jl).

- Matrix products, the Cholesky, LU and QR factorizations and triangular solves put their cubic work into Ozaki-scheme
  products: the operands are scaled to integers, the integers are reduced modulo several primes, the residue products
  are computed exactly by Float64 BLAS, and the result is reconstructed by the Chinese remainder theorem and rounded once.
- Symmetric (generalized) eigendecompositions and singular value decompositions refine Float64 LAPACK results with the
  iterations of Ogita and Aishima, which consist of matrix products; clustered and graded spectra are handled.
- Extended-precision inner loops use in-place MPFR operations that do not allocate, so they scale with threads.
- `SymPacked` stores a symmetric `BigFloat` matrix in about 1/3 of the memory of a `Matrix{BigFloat}`.

## Functions

| | |
|---|---|
| Matrix products | `gemm_ozaki!`, `mul_ozaki`, `gemm_mt!` |
| Cholesky | `chol_lower_ozaki!`, `chol_lower!`, `chol_lower_persist!` |
| LU and QR | `lu_ozaki!`, `lu_solve!`, `qr_ozaki!`, `qr_mulQ!`, `qr_Q` |
| Triangular solves | `llt_solve!`, `lt_solve!`, `ltri_solve!`, `trsm_lower_ozaki!`, `trsm_lower_unit!`, `trsm_upper!` |
| Eigenvalues | `eigen_refine` (A x = λ x and A x = λ B x), `lowest_eig_si` (lowest eigenpair of H x = E S x), `eigen_mt`, `eigen_proj` |
| Singular values | `svd_refine` |
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

A = fresh(X)
λ, V, info = eigen_refine(A + A')          # symmetric eigendecomposition
U, σ, W, info = svd_refine(A)              # singular value decomposition
```

Start Julia with threads (`julia -t 16`).

## Performance status

The Cholesky factorization, the triangular solves and the eigensolvers used for generalized eigenvalue problems are
tuned. The matrix product, LU, QR, `eigen_refine` and `svd_refine` are complete and tested for accuracy but not tuned
further. Known limitations:

- Every residue product pays a reconstruction cost proportional to the size of its output, independent of the inner
  dimension. Products with short inner dimensions gain little; below 32 they fall back to `gemm_mt!`. LU, QR and their
  triangular solves, which issue many such products, gain least at small sizes: at n = 500 the `Float64x4` LU is no
  faster than element-wise multi-word kernels.
- QR: the narrowest levels of the recursive panel factorization (two to four columns) and the generation of the
  Householder vectors run serially. The `BigFloat` QR speeds up only about threefold from 1 to 10 threads.
- The residue products run Float64 BLAS with one thread per task; there is no INT8 (AMX) or GPU back-end.

## Development

```shell
git clone https://github.com/JuliaFewBody/HighPrecisionLinearAlgebra.jl.git
cd HighPrecisionLinearAlgebra.jl
julia --project=. --startup-file=no -e "using Pkg; Pkg.test()"
```

## Acknowledgments

This package is built on an initial project template generated using [Julia Package Factory](https://github.com/JuliaPackageFactory).
