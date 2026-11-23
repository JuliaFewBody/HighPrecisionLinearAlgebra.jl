# Threaded direct symmetric eigensolver, bit-identical to GenericLinearAlgebra's `eigen(Symmetric)`.
#
# GenericLinearAlgebra spends most of the time accumulating eigenvectors: forming Q from the Householder
# reflectors and applying the QL Givens rotations to the vectors. The rotation angles depend only on the
# tridiagonal (d, e), so the QL sweep is run once to record the rotations, and the recording is then applied to
# row blocks of the vectors in parallel. Each row sees the same operations in the same order, so the result is
# bit-identical. Q is formed column-parallel, and the tridiagonalization parallelizes its symv over rows and its
# rank-2 update over columns.

using GenericLinearAlgebra: EigenQ, SymmetricTridiagonalFactorization, symtri!
using LinearAlgebra: SymTridiagonal, diag, givensAlgorithm

const _MT_MIN_WORK = 20_000

"Q B for the Householder Q of GenericLinearAlgebra, parallel over the columns of B (same expressions, bit-identical)."
function lmulQ_mt!(Q::EigenQ, B::StridedMatrix)
    m, n = size(B)
    size(Q, 1) == m || throw(DimensionMismatch("Q and B do not match"))
    nt = Threads.nthreads()
    @inbounds for k = length(Q.τ):-1:1
        if Q.uplo == 'L'
            h = view(Q.factors, (k + 1):m, k); rows = (k + 1):m; ih = 1
        else
            h = view(Q.factors, 1:(m - k), m - k + 1); rows = 1:(n - k); ih = length(h)
        end
        τ = Q.τ[k]
        tmp = h[ih]; h[ih] = 1
        if nt == 1 || length(rows) * n < _MT_MIN_WORK
            for j = 1:n; _applyH_col!(B, h, τ, rows, j); end
        else
            Threads.@threads for j = 1:n; _applyH_col!(B, h, τ, rows, j); end
        end
        h[ih] = tmp
    end
    B
end
@inline function _applyH_col!(B, h, τ, rows, j)
    @inbounds begin
        t = τ * (h' * view(B, rows, j))
        i0 = first(rows) - 1
        for i = 1:length(rows); B[i0 + i, j] -= h[i] * t; end
    end
end

"Recorded Givens rotations: rotation t acts on columns idx[t] and idx[t]+1."
struct RotLog{T}
    c::Vector{T}
    s::Vector{T}
    idx::Vector{Int}
end
RotLog{T}() where {T} = RotLog{T}(T[], T[], Int[])
Base.length(r::RotLog) = length(r.idx)
@inline _record!(r::RotLog, c, s, j) = (push!(r.c, c); push!(r.s, s); push!(r.idx, j); r)
Base.empty!(r::RotLog) = (empty!(r.c); empty!(r.s); empty!(r.idx); r)

"Apply the recorded rotations to V, parallel over row blocks."
function apply_rots!(V::Matrix, r::RotLog; nt::Int=Threads.nthreads())
    nrow = size(V, 1)
    (nrow == 0 || length(r) == 0) && return V
    if nt == 1 || nrow * length(r) < _MT_MIN_WORK
        _rot_rows!(V, r, 1:nrow)
    else
        chunk = cld(nrow, min(nt, nrow))
        Threads.@threads for lo in 1:chunk:nrow
            _rot_rows!(V, r, lo:min(lo + chunk - 1, nrow))
        end
    end
    V
end
@inline function _rot_rows!(V, r::RotLog, rows)
    @inbounds for t = 1:length(r.idx)
        c = r.c[t]; s = r.s[t]; j = r.idx[t]
        for i in rows
            v1 = V[i, j + 1]; v2 = V[i, j]
            V[i, j + 1] = c * v1 + s * v2
            V[i, j] = c * v2 - s * v1
        end
    end
    V
end

# GenericLinearAlgebra's eig2x2! and singleShiftQL! with the vector update replaced by recording (d, e unchanged).
function _eig2x2_rec!(d::StridedVector, e::StridedVector, j::Integer, r::RotLog)
    dj = d[j]; dj1 = d[j + 1]; ej = e[j]
    r1 = (dj + dj1) / 2
    r2 = hypot(dj - dj1, 2 * ej) / 2
    λ⁺ = r1 + r2; λ⁻ = r1 - r2
    d[j] = λ⁺; d[j + 1] = λ⁻; e[j] = 0
    if iszero(ej)
        c = one(λ⁺); s = zero(λ⁺)
    elseif abs(λ⁺ - dj) > abs(λ⁻ - dj)
        c = -ej / hypot(ej, λ⁺ - dj); s = sqrt(1 - c * c)
    else
        s = abs(ej) / hypot(ej, λ⁻ - dj); c = copysign(sqrt(1 - s * s), -ej)
    end
    _record!(r, c, s, j)
end
function _singleShiftQL_rec!(S::SymTridiagonal, shift::Number, istart::Integer, iend::Integer, r::RotLog)
    d, e = S.dv, S.ev
    γi = d[iend] - shift
    π_ = γi
    ci, si = one(eltype(S)), zero(eltype(S))
    @inbounds for i = (iend - 1):-1:istart
        ei = e[i]
        ci1 = ci; si1 = si
        ci, si, ζ = givensAlgorithm(π_, ei)
        i < iend - 1 && (e[i + 1] = si1 * ζ)
        di = d[i]
        γi1 = γi
        γi = ci * ci * (di - shift) - si * si * γi1
        d[i + 1] = γi1 + di - γi
        π_ = ci == 0 ? -ei * ci1 : γi / ci
        _record!(r, ci, si, i)
    end
    e[istart] = si * π_
    d[istart] = shift + γi
    S
end

"QL iteration of GenericLinearAlgebra's eigQL!, with the rotations recorded and applied to V in parallel batches."
function eigQL_mt!(S::SymTridiagonal{T}, V::Matrix; tol=eps(T), flush_cap::Int=0) where {T<:Real}
    d = S.dv; e = S.ev; n = length(d)
    size(V, 2) == n || throw(DimensionMismatch("V must have $n columns"))
    cap = flush_cap > 0 ? flush_cap : max(4n, 1024)
    r = RotLog{T}()
    blockstart = 1; blockend = n
    @inbounds while true
        for be = (blockstart + 1):n
            if abs(e[be - 1]) <= tol * sqrt(abs(d[be - 1])) * sqrt(abs(d[be]))
                blockend = be - 1
                break
            end
            blockend = n
        end
        if blockstart == blockend
            blockstart += 1
        elseif blockstart + 1 == blockend
            _eig2x2_rec!(d, e, blockstart, r)
            blockstart += 2
        else
            μ = (d[blockstart + 1] - d[blockstart]) / (2 * e[blockstart])
            μ = d[blockstart] - (e[blockstart] / (μ + copysign(hypot(μ, one(T)), μ)))
            _singleShiftQL_rec!(S, μ, blockstart, blockend, r)
        end
        length(r) >= cap && (apply_rots!(V, r); empty!(r))
        blockstart >= n && break
    end
    apply_rots!(V, r)
    d, V
end

"GenericLinearAlgebra's symtri! (upper) with the symv parallel over rows and the rank-2 update over columns."
function symtri_mt!(A::Symmetric{T}) where {T<:Real}
    A.uplo == 'U' || return symtri!(A)
    AS = parent(A)
    n = LinearAlgebra.checksquare(AS)
    τ = zeros(T, max(n - 1, 0))
    u = Vector{T}(undef, n)
    nt = Threads.nthreads()
    @inbounds for k = 1:(n - 2)
        nk = n - k
        x = view(AS, 1:nk, nk + 1)
        τk = LinearAlgebra.reflector!(view(AS, nk:-1:1, nk + 1))
        τ[k] = τk
        tmp = x[end]; x[end] = 1
        ũ = view(u, 1:nk)
        _par_or_serial(nt, nk * nk, 1:nk, rows -> _symv_rows!(ũ, AS, x, τk, nk, rows))
        ξ = τk' * (x' * ũ)
        if nt == 1 || nk * nk < _MT_MIN_WORK
            for j = 1:nk; _rank2_col!(AS, x, ũ, ξ, j); end
        else
            Threads.@threads :dynamic for j = 1:nk; _rank2_col!(AS, x, ũ, ξ, j); end
        end
        x[end] = tmp
    end
    SymmetricTridiagonalFactorization(EigenQ('U', AS, τ), SymTridiagonal(diag(AS), diag(AS, 1)))
end
_par_or_serial(nt, work, r, f) = nt == 1 || work < _MT_MIN_WORK ? f(r) : _foreach_chunk(f, r)
# ũ = τk Ã x for the symmetric Ã stored in the upper triangle; each ũ[i] accumulates over columns in order.
@inline function _symv_rows!(ũ, AS, x, τk, nk, rows)
    @inbounds begin
        for i in rows; ũ[i] = zero(eltype(ũ)); end
        for k = 1:nk
            b = τk * x[k]
            for i in rows; ũ[i] += (i <= k ? AS[i, k] : AS[k, i]) * b; end
        end
    end
end
@inline function _rank2_col!(AS, x, ũ, ξ, j)
    @inbounds begin
        xⱼ = x[j]; ũⱼ = ũ[j]; ξxⱼ = ξ * xⱼ
        for i = 1:j; AS[i, j] += x[i] * ξxⱼ' - x[i] * ũⱼ' - ũ[i] * xⱼ'; end
    end
end

"""
    eigen_mt(A::Symmetric) -> LinearAlgebra.Eigen

Same result as GenericLinearAlgebra's `eigen(A)` (bit-identical), threaded.
"""
function eigen_mt(A::Symmetric{T}; tol=eps(T), sortby=LinearAlgebra.eigsortby) where {T<:Real}
    F = symtri_mt!(Symmetric(copy(parent(A)), Symbol(A.uplo) === :U ? :U : :L))
    V = lmulQ_mt!(F.Q, Matrix{T}(I, size(F.diagonals, 1), size(F.diagonals, 1)))
    d, V = eigQL_mt!(F.diagonals, V; tol=tol)
    LinearAlgebra.Eigen(LinearAlgebra.sorteig!(d, V, sortby)...)
end
eigen_mt(A::AbstractMatrix; kw...) = eigen_mt(Symmetric(A); kw...)

# Qᵀ B (reflectors applied from the first), B is n×k with small k.
function _lmulQt!(Q::EigenQ, B::StridedMatrix)
    m, k = size(B)
    @inbounds for j = 1:length(Q.τ)
        if Q.uplo == 'L'
            h = view(Q.factors, (j + 1):m, j); i0 = j; ih = 1
        else
            h = view(Q.factors, 1:(m - j), m - j + 1); i0 = 0; ih = length(h)
        end
        τ = Q.τ[j]
        tmp = h[ih]; h[ih] = 1
        for c = 1:k
            t = τ * (h' * view(B, i0+1:i0+length(h), c))
            for i = 1:length(h); B[i0 + i, c] -= h[i] * t; end
        end
        h[ih] = tmp
    end
    B
end

"""
    eigen_proj(A::Symmetric, Y) -> (values, P)

Eigenvalues and the projections P = Eᵀ Y (E = `eigen_mt(A).vectors`, same order and signs) without forming E:
Q is applied as Qᵀ Y and the QL rotations act on a k×M matrix (k = columns of Y). O(kM²) instead of O(M³).
"""
function eigen_proj(A::Symmetric{T}, Y::AbstractMatrix; tol=eps(T)) where {T<:Real}
    size(Y, 1) == size(A, 1) || throw(DimensionMismatch("Y must have as many rows as A"))
    F = symtri_mt!(Symmetric(copy(parent(A)), Symbol(A.uplo) === :U ? :U : :L))
    W = Matrix(permutedims(_lmulQt!(F.Q, Matrix{T}(Y))))
    d, W = eigQL_mt!(F.diagonals, W; tol=tol)
    d, W = LinearAlgebra.sorteig!(d, W, LinearAlgebra.eigsortby)
    (values=d, P=Matrix(permutedims(W)))
end

"""
    xtHx_mt(X, H) -> X' * H * X

Computed as (XᵀH)X like Julia's three-argument `*`, threaded over output columns with the inner sums in increasing
order (bit-identical to `X' * H * X` for generic element types).
"""
function xtHx_mt(X::AbstractMatrix{T}, H::AbstractMatrix{T}) where {T}
    M, K = size(X)
    size(H) == (M, M) || throw(DimensionMismatch("H must be $(M)×$(M)"))
    Hf = H isa Symmetric ? Matrix(H) : H
    W = Matrix{T}(undef, K, M)                       # W = Xᵀ H
    _par_cols(j -> _dotcols!(W, X, Hf, j, true), M, K * M)
    A = Matrix{T}(undef, K, K)                       # A = W X
    _par_cols(j -> _dotcols!(A, W, X, j, false), K, K * K)
    A
end
# C[i, j] = Σ_k P[k, i] Q[k, j] (tP) or Σ_k P[i, k] Q[k, j], k increasing
function _dotcols!(C, P, Q, j, tP)
    T = eltype(C)
    @inbounds for i in axes(C, 1)
        s = zero(T)
        for k in axes(Q, 1); s += (tP ? P[k, i] : P[i, k]) * Q[k, j]; end
        C[i, j] = s
    end
end
function _par_cols(f, n, work)
    if Threads.nthreads() == 1 || work < 10_000
        foreach(f, 1:n)
    else
        Threads.@threads for j in 1:n; f(j); end
    end
end
