const _TRSM_LEAF = Ref(32)
# LU with partial pivoting and triangular solves. Columns are split recursively in half (Toledo), and the
# trailing updates are matrix products (Ozaki scheme). Only leaves of width `leaf` stay in extended precision
# (parallel over row blocks).

"|x| > |y| (no allocation for BigFloat)."
@inline _absgt(x, y) = abs(x) > abs(y)
@inline _absgt(x::BigFloat, y::BigFloat) = ccall((:mpfr_cmpabs, :libmpfr), Int32, (Ref{BigFloat}, Ref{BigFloat}), x, y) > 0
@inline _iszero_mp(x) = iszero(x)
@inline _iszero_mp(x::BigFloat) = _sgn(x) == 0 && !isnan(x)

# Divide column j below the diagonal by the pivot (rows `rows`)
function _lu_scale!(A::AbstractMatrix, rows, j)
    @inbounds d = inv(A[j, j])
    @inbounds for i in rows; A[i, j] *= d; end
end
function _lu_scale!(A::AbstractMatrix{BigFloat}, rows, j)
    @inbounds d = A[j, j]
    @inbounds for i in rows; x = A[i, j]; mpfr_div!(x, x, d); end
end
# A[rows, c] −= A[rows, j] A[j, c] for c ∈ cols
function _lu_rank1!(A::AbstractMatrix, rows, j, cols)
    @inbounds for c in cols
        a = A[j, c]
        for i in rows; A[i, c] -= A[i, j] * a; end
    end
end
function _lu_rank1!(A::AbstractMatrix{BigFloat}, rows, j, cols)
    na = BigFloat(; precision=_prec(A))
    @inbounds for c in cols
        mpfr_neg!(na, A[j, c])
        for i in rows; x = A[i, c]; mpfr_fma!(x, A[i, j], na, x); end
    end
end

"Factor columns c1:c2 over rows c1:m directly (right-looking, partial pivoting); swaps touch only columns c1:c2. Return the first zero pivot (0 if none)."
function _lu_leaf!(A::AbstractMatrix, c1::Int, c2::Int, ipiv::Vector{Int})
    m = size(A, 1); info = 0
    @inbounds for j in c1:c2
        j > m && break
        p = j
        for i in j+1:m; _absgt(A[i, j], A[p, j]) && (p = i); end
        ipiv[j] = p
        if p != j
            for c in c1:c2; A[j, c], A[p, c] = A[p, c], A[j, c]; end
        end
        if _iszero_mp(A[j, j])
            info == 0 && (info = j)
            continue
        end
        rows = j+1:m
        isempty(rows) && continue
        f = r -> (_lu_scale!(A, r, j); j < c2 && _lu_rank1!(A, r, j, j+1:c2))
        # one fork-join per column for the division and the update
        Threads.nthreads() == 1 || length(rows) * (c2 - j + 1) < 512 ? f(rows) : _foreach_chunk(f, rows)
    end
    info
end

"Apply the row swaps ipiv[r] (in order, or reversed) to columns cols."
function _swaprows!(A::AbstractMatrix, ipiv::Vector{Int}, r, cols; rev::Bool=false)
    isempty(cols) && return nothing
    f = cs -> begin
        @inbounds for j in (rev ? reverse(r) : r)
            p = ipiv[j]
            p == j && continue
            for c in cs; A[j, c], A[p, c] = A[p, c], A[j, c]; end
        end
    end
    Threads.nthreads() == 1 || length(cols) < 64 ? f(cols) : _foreach_chunk(f, cols)
    nothing
end

function _lu_rec!(A::AbstractMatrix, c1::Int, c2::Int, ipiv::Vector{Int}, leaf::Int)
    m = size(A, 1); n = c2 - c1 + 1
    n <= leaf && return _lu_leaf!(A, c1, c2, ipiv)
    h = c1 + n ÷ 2 - 1
    info = _lu_rec!(A, c1, h, ipiv, leaf)
    _swaprows!(A, ipiv, c1:h, h+1:c2)
    A12 = view(A, c1:h, h+1:c2)
    trsm_lower_unit!(A12, view(A, c1:h, c1:h))
    h < m && _gemm!(view(A, h+1:m, h+1:c2), view(A, h+1:m, c1:h), A12; sub=true)
    info2 = h < m ? _lu_rec!(A, h+1, c2, ipiv, leaf) : 0
    _swaprows!(A, ipiv, h+1:min(c2, m), c1:h)
    info != 0 ? info : info2
end

"""
    lu_ozaki!(A; leaf=16) -> (ipiv, info)

Overwrite A with its LU factorization P A = L U with partial pivoting (unit lower L below the diagonal, U on and
above it; the layout of LAPACK getrf). Row j was swapped with row ipiv[j]. info is the first zero pivot (0 if none).
"""
function lu_ozaki!(A::AbstractMatrix{T}; leaf::Int=16) where {T}
    m, n = size(A); k = min(m, n)
    ipiv = collect(1:k)
    k == 0 && return ipiv, 0
    info = _lu_rec!(A, 1, k, ipiv, leaf)
    if n > k                                        # wide: swaps and L⁻¹ on the remaining columns
        _swaprows!(A, ipiv, 1:k, k+1:n)
        trsm_lower_unit!(view(A, 1:k, k+1:n), view(A, 1:k, 1:k))
    end
    ipiv, info
end

"""
    lu_solve!(B, LU, ipiv) -> B

Solve A X = B with the factorization from `lu_ozaki!`, overwriting B (its elements must be distinct objects).
"""
function lu_solve!(B::AbstractVecOrMat{T}, LU::AbstractMatrix{T}, ipiv::Vector{Int}) where {T}
    Bm = B isa AbstractVector ? reshape(B, :, 1) : B
    _swaprows!(Bm, ipiv, 1:length(ipiv), 1:size(Bm, 2))
    trsm_lower_unit!(Bm, LU)
    trsm_upper!(Bm, LU)
    B
end

"""
    trsm_lower_unit!(B, L; leaf=32) -> B

B ← L⁻¹ B for unit lower triangular L (the diagonal and upper triangle are not read). L is split recursively and the
off-diagonal blocks are matrix products.
"""
function trsm_lower_unit!(B::AbstractMatrix{T}, L::AbstractMatrix{T}; leaf::Int=_TRSM_LEAF[]) where {T}
    n = size(L, 1)
    n == 0 && return B
    if n <= leaf
        f = cols -> _fwd_unit!(B, L, cols)
        Threads.nthreads() == 1 || n * n * size(B, 2) < _MIN_WORK ? f(axes(B, 2)) : _foreach_chunk(f, axes(B, 2))
        return B
    end
    h = n ÷ 2
    B1 = view(B, 1:h, :); B2 = view(B, h+1:n, :)
    trsm_lower_unit!(B1, view(L, 1:h, 1:h); leaf)
    _gemm!(B2, view(L, h+1:n, 1:h), B1; sub=true)
    trsm_lower_unit!(B2, view(L, h+1:n, h+1:n); leaf)
    B
end
function _fwd_unit!(B::AbstractMatrix, L, cols)
    n = size(L, 1)
    @inbounds for j in cols, k in 1:n
        b = B[k, j]
        for i in k+1:n; B[i, j] -= L[i, k] * b; end
    end
end
function _fwd_unit!(B::AbstractMatrix{BigFloat}, L::AbstractMatrix{BigFloat}, cols)
    n = size(L, 1); nb = BigFloat(; precision=_prec(B))
    @inbounds for j in cols, k in 1:n
        mpfr_neg!(nb, B[k, j])
        for i in k+1:n; x = B[i, j]; mpfr_fma!(x, L[i, k], nb, x); end
    end
end

"""
    trsm_upper!(B, U; leaf=32) -> B

B ← U⁻¹ B for upper triangular U (the lower triangle is not read).
"""
function trsm_upper!(B::AbstractMatrix{T}, U::AbstractMatrix{T}; leaf::Int=_TRSM_LEAF[]) where {T}
    n = size(U, 1)
    n == 0 && return B
    if n <= leaf
        f = cols -> _bwd_upper!(B, U, cols)
        Threads.nthreads() == 1 || n * n * size(B, 2) < _MIN_WORK ? f(axes(B, 2)) : _foreach_chunk(f, axes(B, 2))
        return B
    end
    h = n ÷ 2
    B1 = view(B, 1:h, :); B2 = view(B, h+1:n, :)
    trsm_upper!(B2, view(U, h+1:n, h+1:n); leaf)
    _gemm!(B1, view(U, 1:h, h+1:n), B2; sub=true)
    trsm_upper!(B1, view(U, 1:h, 1:h); leaf)
    B
end
function _bwd_upper!(B::AbstractMatrix, U, cols)
    n = size(U, 1)
    @inbounds for j in cols, k in n:-1:1
        b = B[k, j] / U[k, k]; B[k, j] = b
        for i in 1:k-1; B[i, j] -= U[i, k] * b; end
    end
end
function _bwd_upper!(B::AbstractMatrix{BigFloat}, U::AbstractMatrix{BigFloat}, cols)
    n = size(U, 1); nb = BigFloat(; precision=_prec(B))
    @inbounds for j in cols, k in n:-1:1
        x = B[k, j]; mpfr_div!(x, x, U[k, k]); mpfr_neg!(nb, x)
        for i in 1:k-1; y = B[i, j]; mpfr_fma!(y, U[i, k], nb, y); end
    end
end
