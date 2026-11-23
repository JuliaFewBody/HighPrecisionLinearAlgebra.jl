# Dense Cholesky, triangular solves and symmetric matrix-vector products, threaded over columns or rows.
# Generic element types use ordinary arithmetic; BigFloat methods use the in-place MPFR kernels.

const _MIN_WORK = 10_000        # below this many element operations, run serially

"Overwrite the lower triangle of A with its Cholesky factor (right-looking). Return false if A is not positive definite."
function chol_lower!(A::AbstractMatrix{T}) where {T}
    M = size(A, 1)
    @inbounds for k in 1:M
        _chol_pivot!(A, k, M) || return false
        if Threads.nthreads() == 1 || (M - k)^2 < _MIN_WORK
            for j in k+1:M; _chol_upd!(A, j, k, M); end
        else
            Threads.@threads for j in k+1:M; _chol_upd!(A, j, k, M); end
        end
    end
    true
end
@inline function _chol_upd!(A, j, k, M)
    @inbounds begin
        ajk = A[j, k]
        for i in j:M; A[i, j] -= A[i, k] * ajk; end
    end
end
@inline function _chol_upd!(A::AbstractMatrix{BigFloat}, j, k, M)
    @inbounds begin
        njk = BigFloat(; precision=_prec(A)); mpfr_neg!(njk, _el(A, j, k))
        for i in j:M
            aij = _el(A, i, j); mpfr_fma!(aij, _el(A, i, k), njk, aij)
        end
    end
end
"Take the square root of the pivot of column k and divide the column below it. Return false if the pivot is not positive."
@inline function _chol_pivot!(A::AbstractMatrix, k::Int, M::Int)
    @inbounds begin
        d = A[k, k]
        d > 0 || return false
        s = sqrt(d); A[k, k] = s; si = inv(s)
        for i in k+1:M; A[i, k] *= si; end
    end
    true
end
@inline function _chol_pivot!(A::AbstractMatrix{BigFloat}, k::Int, M::Int)
    @inbounds begin
        d = _el(A, k, k)
        _sgn(d) > 0 || return false
        mpfr_sqrt!(d, d)
        for i in k+1:M; aik = _el(A, i, k); mpfr_div!(aik, aik, d); end
    end
    true
end

"Solve L X = B column by column, overwriting B."
function ltri_solve!(B::AbstractMatrix{T}, L::AbstractMatrix{T}) where {T}
    M, N = size(B)
    if Threads.nthreads() == 1 || M * N < _MIN_WORK
        for j in 1:N; _ltri_col!(B, L, j, M); end
    else
        Threads.@threads for j in 1:N; _ltri_col!(B, L, j, M); end
    end
    B
end
@inline function _ltri_col!(B, L, j, M)
    @inbounds for k in 1:M
        bk = B[k, j] / L[k, k]; B[k, j] = bk
        for i in k+1:M; B[i, j] -= L[i, k] * bk; end
    end
end
@inline function _ltri_col!(B::AbstractMatrix{BigFloat}, L::AbstractMatrix{BigFloat}, j, M)
    @inbounds begin
        nbk = BigFloat(; precision=_prec(B))
        for k in 1:M
            bkj = B[k, j]
            mpfr_div!(bkj, bkj, _el(L, k, k)); mpfr_neg!(nbk, bkj)
            for i in k+1:M; mpfr_fma!(B[i, j], _el(L, i, k), nbk, B[i, j]); end
        end
    end
end

"""
    llt_solve!(b, L; nb=32) -> b

Solve L Lᵀ x = b in place (L is the lower Cholesky factor; the upper triangle is not read).
Rows are processed in blocks of nb; the forward sweep updates the remaining rows in parallel and the
backward sweep computes the block's dot products in parallel. The elements of b must be distinct objects.
"""
function llt_solve!(b::AbstractVector{T}, L::AbstractMatrix{T}; nb::Int=32) where {T}
    M = length(b)
    par = Threads.nthreads() > 1 && M >= 2nb
    blocks = [(k1, min(k1 + nb - 1, M)) for k1 in 1:nb:M]
    @inbounds for (k1, k2) in blocks                    # L y = b
        for k in k1:k2
            _divk!(b, L, k)
            k < k2 && _axpy_rows!(b, L, k+1:k2, k, k)
        end
        k2 < M || continue
        par ? _foreach_chunk(rows -> _axpy_rows!(b, L, rows, k1, k2), k2+1:M) : _axpy_rows!(b, L, k2+1:M, k1, k2)
    end
    _lt_backward!(b, L, blocks, par)                   # Lᵀ x = y
end
"""
    lt_solve!(b, L; nb=32) -> b

Solve Lᵀ x = b in place (the second half of `llt_solve!`).
"""
function lt_solve!(b::AbstractVector{T}, L::AbstractMatrix{T}; nb::Int=32) where {T}
    M = length(b)
    _lt_backward!(b, L, [(k1, min(k1 + nb - 1, M)) for k1 in 1:nb:M], Threads.nthreads() > 1 && M >= 2nb)
end
function _lt_backward!(b, L, blocks, par)
    M = length(b)
    @inbounds for (k1, k2) in Iterators.reverse(blocks)
        if k2 < M
            f = cols -> (for k in cols; _dot_sub!(b, L, k, k2+1:M); end)
            par ? _foreach_chunk(f, k1:k2) : f(k1:k2)
        end
        for k in k2:-1:k1
            k < k2 && _dot_sub!(b, L, k, k+1:k2)
            _divk!(b, L, k)
        end
    end
    b
end
@inline _divk!(b, L, k) = (@inbounds b[k] = b[k] / L[k, k]; nothing)
@inline function _axpy_rows!(b, L, rows, k1, k2)         # b[i] −= Σ_k L[i,k] b[k]
    @inbounds for k in k1:k2
        bk = b[k]
        for i in rows; b[i] -= L[i, k] * bk; end
    end
end
@inline function _dot_sub!(b, L, k, r)                   # b[k] −= Σ_i L[i,k] b[i]
    s = zero(eltype(b))
    @inbounds for i in r; s += L[i, k] * b[i]; end
    @inbounds b[k] -= s
    nothing
end
@inline _divk!(b::AbstractVector{BigFloat}, L::AbstractMatrix{BigFloat}, k) = (@inbounds mpfr_div!(b[k], b[k], _el(L, k, k)); nothing)
@inline function _axpy_rows!(b::AbstractVector{BigFloat}, L::AbstractMatrix{BigFloat}, rows, k1, k2)
    t = BigFloat(; precision=precision(b[k1]))
    @inbounds for k in k1:k2
        mpfr_neg!(t, b[k])
        for i in rows; mpfr_fma!(b[i], _el(L, i, k), t, b[i]); end
    end
end
@inline function _dot_sub!(b::AbstractVector{BigFloat}, L::AbstractMatrix{BigFloat}, k, r)
    s = BigFloat(0; precision=precision(b[k]))
    @inbounds for i in r; mpfr_fma!(s, _el(L, i, k), b[i], s); end
    @inbounds mpfr_sub!(b[k], b[k], s)
    nothing
end

"y = A x for symmetric A (rows in parallel)."
function symv!(y, A, x)
    M = size(A, 1)
    if Threads.nthreads() == 1 || M * M < _MIN_WORK
        for i in 1:M; y[i] = _row_dot(A, x, i, M); end
    else
        Threads.@threads for i in 1:M; y[i] = _row_dot(A, x, i, M); end
    end
    y
end
@inline function _row_dot(A, x, i, M)
    s = zero(eltype(x))
    @inbounds for k in 1:M; s += A[i, k] * x[k]; end
    s
end
@inline function _row_dot(A::AbstractMatrix{BigFloat}, x::AbstractVector{BigFloat}, i, M)
    s = BigFloat(0; precision=_prec(A))
    @inbounds for k in 1:M; mpfr_fma!(s, _el(A, i, k), x[k], s); end
    s
end

"A copy whose elements are distinct objects (a BigFloat matrix that may be overwritten in place)."
fresh(A::AbstractMatrix) = Matrix{eltype(A)}(copy(A))
fresh(A::AbstractMatrix{BigFloat}) = BigFloat[BigFloat(a; precision=precision(a)) + 0 for a in A]
fresh(v::AbstractVector) = Vector{eltype(v)}(copy(v))
fresh(v::AbstractVector{BigFloat}) = BigFloat[BigFloat(a; precision=precision(a)) + 0 for a in v]

"""
Split the lower-triangular columns 1:M (column j costs ∝ M − j + 1) into 4·nthreads contiguous blocks of equal work
and call f(block) in parallel. Splitting the columns by count gives the first thread about twice its share.
"""
function _foreach_tri_cols(f, M::Int)
    nt = Threads.nthreads()
    (nt == 1 || M < 2nt) && return (M >= 1 && f(1:M); nothing)
    nb = 4nt; next = Threads.Atomic{Int}(1)
    @sync for _ in 1:nt
        Threads.@spawn while (b = Threads.atomic_add!(next, 1)) <= nb
            c1, c2 = _wsplit(1, M, M, nb, b)
            c2 >= c1 && f(c1:c2)
        end
    end
    nothing
end
"Split r into nthreads chunks and call f(chunk) in parallel."
function _foreach_chunk(f, r::AbstractUnitRange)
    n = min(Threads.nthreads(), length(r))
    @sync for c in Iterators.partition(r, cld(length(r), n))
        Threads.@spawn f(c)
    end
end
"The t-th of nt contiguous ranges of lo:hi with equal Σ (M − j + 1)."
@inline function _wsplit(lo::Int, hi::Int, M::Int, nt::Int, t::Int)
    hi < lo && return (1, 0)
    u0 = M - lo + 1; u1 = M - hi + 1
    W = (u0 * (u0 + 1) - u1 * (u1 - 1)) / 2
    a = t == 1 ? lo : _wcut(u0, W, (t - 1) / nt, M) + 1
    b = t == nt ? hi : _wcut(u0, W, t / nt, M)
    (max(a, lo), min(b, hi))
end
@inline function _wcut(u0::Int, W::Float64, f::Float64, M::Int)
    x = max(u0 * (u0 + 1) - 2 * f * W, 0.0)
    round(Int, M - (1 + sqrt(1 + 4x)) / 2 + 1)
end

# ---------------------------------------------------------------------------------------------
# Cholesky with persistent workers: `chol_lower!` forks once per column; this version starts the tasks once and
# synchronizes them with a spinning barrier. The operation order is the same, so the result is bit-identical.
# ---------------------------------------------------------------------------------------------

"Sense-reversing barrier (spin, yield every spinlimit iterations)."
struct _Barrier
    n::Int
    cnt::Threads.Atomic{Int}
    gen::Threads.Atomic{Int}
end
_Barrier(n::Int) = _Barrier(n, Threads.Atomic{Int}(0), Threads.Atomic{Int}(0))
@inline function _bwait(b::_Barrier, spinlimit::Int=4096)
    g = b.gen[]
    if Threads.atomic_add!(b.cnt, 1) == b.n - 1
        b.cnt[] = 0
        Threads.atomic_add!(b.gen, 1)
    else
        spins = 0
        while b.gen[] == g
            spins += 1
            spins >= spinlimit ? (yield(); spins = 0) : ccall(:jl_cpu_pause, Cvoid, ())
        end
    end
    nothing
end

"""
    chol_lower_persist!(A; nt=Threads.nthreads(), spinlimit=4096) -> Bool

Same result as `chol_lower!` (bit-identical) using persistent tasks and a barrier.
"""
function chol_lower_persist!(A::AbstractMatrix{T}; nt::Int=Threads.nthreads(), spinlimit::Int=4096) where {T}
    M = size(A, 1)
    nt = clamp(nt, 1, Threads.nthreads())
    nt == 1 && return chol_lower!(A)
    bar = _Barrier(nt); bad = Threads.Atomic{Bool}(false)
    @sync for t in 1:nt
        Threads.@spawn _persist_worker!(A, M, t, nt, bar, bad, spinlimit)
    end
    !bad[]
end
function _persist_worker!(A::AbstractMatrix, M::Int, t::Int, nt::Int, bar::_Barrier, bad, spinlimit::Int)
    for k in 1:M
        t == 1 && (_chol_pivot!(A, k, M) || (bad[] = true))
        _bwait(bar, spinlimit)
        bad[] && return
        m = M - k
        if m > 0
            if m * m < _MIN_WORK
                t == 1 && for j in k+1:M; _chol_upd!(A, j, k, M); end
            else
                a, b = _wsplit(k + 1, M, M, nt, t)
                for j in a:b; _chol_upd!(A, j, k, M); end
            end
        end
        _bwait(bar, spinlimit)
    end
end

"""
    release_task_refs()

Drop each thread's reference to the last task it ran. Julia keeps that task (and the arrays its closure captured)
alive until the thread runs another task, so large matrices used inside `@threads` are not collected.
"""
release_task_refs() = (Threads.@threads :static for _ in 1:Threads.nthreads(); end; nothing)
