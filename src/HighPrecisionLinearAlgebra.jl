"""
    HighPrecisionLinearAlgebra

Parallel dense linear algebra in extended precision (`BigFloat`, MultiFloats' `Float64x2`/`Float64x4`).
The trailing updates of Cholesky and triangular solves use the Ozaki scheme (exact Float64 dgemms of integer slices or
residues), and the extended-precision inner loops use in-place MPFR operations that do not allocate.

- Cholesky: `chol_lower!`, `chol_lower_persist!`, `chol_lower_ozaki!`; solves `llt_solve!`, `lt_solve!`, `ltri_solve!`, `trsm_lower_ozaki!`
- Eigenvalues: `lowest_eig_si` (lowest pair of H x = E S x by shifted inverse iteration), `eigen_mt`, `eigen_proj`
- Storage: `SymPacked` (packed symmetric BigFloat, about 1/3 of the memory), `PackedMatrix`
- In-place MPFR operations: `mpfr_add!`, `mpfr_mul!`, `mpfr_fma!`, …
"""
module HighPrecisionLinearAlgebra

using LinearAlgebra
using MultiFloats

include("mpfr.jl")
include("dense.jl")
include("packed.jl")
include("ozaki.jl")
include("crt.jl")
include("eigen_mt.jl")
include("lowest.jl")

export chol_lower!, chol_lower_persist!, chol_lower_ozaki!, llt_solve!, lt_solve!, ltri_solve!, trsm_lower_ozaki!, copy_leading!
export eigen_mt, eigen_proj, xtHx_mt
export lowest_eig_si, shifted_lower, shifted_lower!, add_lowrank_lower!
export SymPacked, PackedMatrix, sympacked_elbytes, symv!, diagonal_scale!, fresh, release_task_refs
export mpfr_ptr, mpfr_add!, mpfr_sub!, mpfr_mul!, mpfr_mul_si!, mpfr_mul_2si!, mpfr_set!, mpfr_powi!, mpfr_exp!, mpfr_div!,
       mpfr_neg!, mpfr_fma!, mpfr_sqrt!

end
