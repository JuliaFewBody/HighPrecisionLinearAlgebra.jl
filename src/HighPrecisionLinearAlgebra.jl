"""
    HighPrecisionLinearAlgebra

Parallel dense linear algebra in extended precision (`BigFloat`, MultiFloats' `Float64x2`/`Float64x4`).
Matrix products, Cholesky, LU, QR and triangular solves put their cubic work into Ozaki-scheme products (exact Float64
dgemms of integer residues modulo primes); eigen- and singular value decompositions refine Float64 LAPACK results by
Ogita–Aishima iterations. Extended-precision inner loops use in-place MPFR operations that do not allocate.

- Products: `gemm_ozaki!`, `mul_ozaki`, `gemm_mt!`
- Cholesky: `chol_lower!`, `chol_lower_persist!`, `chol_lower_ozaki!`; solves `llt_solve!`, `lt_solve!`, `ltri_solve!`, `trsm_lower_ozaki!`
- LU and QR: `lu_ozaki!`, `lu_solve!`, `trsm_lower_unit!`, `trsm_upper!`, `qr_ozaki!`, `qr_mulQ!`, `qr_Q`
- Eigenvalues: `eigen_refine`, `lowest_eig_si` (lowest pair of H x = E S x by shifted inverse iteration), `eigen_mt`, `eigen_proj`
- Singular values: `svd_refine`
- Storage: `SymPacked` (packed symmetric BigFloat, about 1/3 of the memory), `PackedMatrix`
- In-place MPFR operations: `mpfr_add!`, `mpfr_mul!`, `mpfr_fma!`, …
"""
module HighPrecisionLinearAlgebra
using Printf

using LinearAlgebra
using MultiFloats

include("mpfr.jl")
include("dense.jl")
include("packed.jl")
include("ozaki.jl")
include("crt.jl")
include("gemm.jl")
include("lu.jl")
include("qr.jl")
include("refine.jl")
include("eigen_mt.jl")
include("lowest.jl")

export chol_lower!, chol_lower_persist!, chol_lower_ozaki!, llt_solve!, lt_solve!, ltri_solve!, trsm_lower_ozaki!, copy_leading!
export gemm_ozaki!, gemm_mt!, mul_ozaki, lu_ozaki!, lu_solve!, trsm_lower_unit!, trsm_upper!, qr_ozaki!, qr_mulQ!, qr_Q, eigen_refine, svd_refine
export eigen_mt, eigen_proj, xtHx_mt
export lowest_eig_si, shifted_lower, shifted_lower!, add_lowrank_lower!
export SymPacked, PackedMatrix, sympacked_elbytes, symv!, diagonal_scale!, fresh, release_task_refs
export mpfr_ptr, mpfr_add!, mpfr_sub!, mpfr_mul!, mpfr_mul_si!, mpfr_mul_2si!, mpfr_set!, mpfr_powi!, mpfr_exp!, mpfr_div!,
       mpfr_neg!, mpfr_fma!, mpfr_sqrt!

end
