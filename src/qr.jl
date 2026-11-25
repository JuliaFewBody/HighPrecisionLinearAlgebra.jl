# Householder QR. Panels of width nb are factored recursively (Elmroth–Gustavson, the form of LAPACK geqrt3) and the
# block reflector I − V T Vᵀ is applied to the remaining columns with matrix products (Ozaki scheme).

"Explicit V (unit diagonal, zero above, entries below shared with A; read only)."
function _explicit_V(A::AbstractMatrix{T}) where {T}
    m, n = size(A)
    o = one(T); z = zero(T)
    V = Matrix{T}(undef, m, n)
    @inbounds for j in 1:n, i in 1:m
        V[i, j] = i > j ? A[i, j] : (i == j ? o : z)
    end
    V
end
function _explicit_V(A::AbstractMatrix{BigFloat})
    m, n = size(A); p = _prec(A)
    o = BigFloat(1; precision=p); z = BigFloat(0; precision=p)
    V = Matrix{BigFloat}(undef, m, n)
    @inbounds for j in 1:n, i in 1:m
        V[i, j] = i > j ? A[i, j] : (i == j ? o : z)
    end
    V
end
_ptranspose(A::AbstractMatrix) = permutedims(A)            # elements are shared (read only)
"Transpose of an upper triangular T (new matrix, zeros above the diagonal, elements shared)."
function _triu_t(T::AbstractMatrix{E}) where {E}
    n = size(T, 1); z = E === BigFloat ? BigFloat(0; precision=_prec(T)) : zero(E)
    [i >= j ? T[j, i] : z for i in 1:n, j in 1:n]
end
_triu_copy(T::AbstractMatrix{E}) where {E} = (n = size(T, 1); z = E === BigFloat ? BigFloat(0; precision=_prec(T)) : zero(E); [i <= j ? T[i, j] : z for i in 1:n, j in 1:n])

"""
Householder reflector of x (LAPACK larfg): H x = β e₁ with H = I − τ v vᵀ, v₁ = 1. Overwrite x[1] with β and x[2:end] with v[2:end]; return τ.
"""
function _larfg!(x::AbstractVector{T}) where {T}
    n = length(x)
    n <= 1 && return zero(T)
    ss = _sumsq(x, 2:n)
    iszero(ss) && return zero(T)
    α = x[1]
    β = -copysign(sqrt(α * α + ss), α)
    τ = (β - α) / β
    s = inv(α - β)
    _scale_range!(x, 2:n, s)
    x[1] = β
    τ
end
function _larfg!(x::AbstractVector{BigFloat})
    n = length(x); p = precision(x[1])
    n <= 1 && return BigFloat(0; precision=p)
    ss = _sumsq(x, 2:n)
    iszero(ss) && return BigFloat(0; precision=p)
    α = x[1]
    β = -copysign(sqrt(α * α + ss), α)
    τ = (β - α) / β
    s = inv(α - β)
    _scale_range!(x, 2:n, s)
    mpfr_set!(x[1], β)
    τ
end
function _sumsq(x::AbstractVector{T}, r) where {T}
    s = zero(T); @inbounds for i in r; s += x[i] * x[i]; end; s
end
function _sumsq(x::AbstractVector{BigFloat}, r)
    s = BigFloat(0; precision=precision(x[1])); @inbounds for i in r; mpfr_fma!(s, x[i], x[i], s); end; s
end
_scale_range!(x::AbstractVector, r, s) = (@inbounds for i in r; x[i] *= s; end; nothing)
_scale_range!(x::AbstractVector{BigFloat}, r, s::BigFloat) = (@inbounds for i in r; mpfr_mul!(x[i], x[i], s); end; nothing)

"""
Recursive QR of a panel A (m×n); write the upper triangular T (n×n, I − V T Vᵀ = H₁⋯Hₙ) into Tm (geqrt3).
A is overwritten by R (upper triangle) and V (below the diagonal; v₁ = 1 is implicit).
"""
function _geqrt3!(A::AbstractMatrix{E}, Tm::AbstractMatrix{E}) where {E}
    m, n = size(A)
    if n == 1
        τ = _larfg!(view(A, :, 1))
        _setel!(Tm, 1, 1, τ)
        return nothing
    end
    n1 = n ÷ 2
    A1 = view(A, :, 1:n1); A2 = view(A, :, n1+1:n)
    _geqrt3!(A1, view(Tm, 1:n1, 1:n1))
    V1 = _explicit_V(A1)
    # A2 ← (I − V1 T1 V1ᵀ)ᵀ A2 = A2 − V1 T1ᵀ (V1ᵀ A2)
    W = _zeros_like(A, n1, n - n1)
    _gemm!(W, _ptranspose(V1), A2)
    W2 = _zeros_like(A, n1, n - n1)
    _gemm!(W2, _triu_t(view(Tm, 1:n1, 1:n1)), W)
    _gemm!(A2, V1, W2; sub=true)
    _geqrt3!(view(A, n1+1:m, n1+1:n), view(Tm, n1+1:n, n1+1:n))
    # T12 = −T1 (V1ᵀ V2) T2 (V2 has rows n1+1:m)
    V2 = _explicit_V(view(A, n1+1:m, n1+1:n))
    X = _zeros_like(A, n1, n - n1)
    _gemm!(X, _ptranspose(view(V1, n1+1:m, :)), V2)
    Y = _zeros_like(A, n1, n - n1)
    _gemm!(Y, _triu_copy(view(Tm, 1:n1, 1:n1)), X)
    T12 = view(Tm, 1:n1, n1+1:n)
    _set_zero!(T12)
    _gemm!(T12, Y, _triu_copy(view(Tm, n1+1:n, n1+1:n)); sub=true)
    nothing
end
_setel!(A::AbstractMatrix, i, j, x) = (A[i, j] = x; nothing)
_setel!(A::AbstractMatrix{BigFloat}, i, j, x::BigFloat) = (A[i, j] = BigFloat(x; precision=precision(x)) + 0; nothing)
_set_zero!(A::AbstractMatrix{T}) where {T} = (A .= zero(T); nothing)
function _set_zero!(A::AbstractMatrix{BigFloat})
    p = isempty(A) ? precision(BigFloat) : _prec(A)
    @inbounds for j in axes(A, 2), i in axes(A, 1); A[i, j] = BigFloat(0; precision=p); end
    nothing
end

"""
C ← (I − V T Vᵀ)ᵀ C (trans=true) or (I − V T Vᵀ) C. V is explicit (m×k), T is k×k upper triangular.
"""
function _apply_block_reflector!(C::AbstractMatrix{E}, V::AbstractMatrix{E}, Tm::AbstractMatrix{E}; trans::Bool) where {E}
    k = size(V, 2); n = size(C, 2)
    (k == 0 || n == 0) && return C
    W = _zeros_like(C, k, n)
    _gemm!(W, _ptranspose(V), C)
    W2 = _zeros_like(C, k, n)
    _gemm!(W2, trans ? _triu_t(Tm) : _triu_copy(Tm), W)
    _gemm!(C, V, W2; sub=true)
    C
end

"""
    qr_ozaki!(A; nb=64) -> T

Overwrite A (m×n) with its Householder QR factorization: R in the upper triangle, reflector vectors V below it (LAPACK geqrt layout).
T (nb×min(m,n)) holds the triangular factors per block: for columns k1:k2, Q_b = I − V_b T_b V_bᵀ with T_b = T[1:k2−k1+1, k1:k2].
Q = Q₁ Q₂ ⋯. Both the panels and the updates use matrix products (Ozaki scheme).
"""
function qr_ozaki!(A::AbstractMatrix{E}; nb::Int=64) where {E}
    m, n = size(A); k = min(m, n)
    Tm = _zeros_like(A, min(nb, max(k, 1)), k)
    for k1 in 1:nb:k
        k2 = min(k1 + nb - 1, k); w = k2 - k1 + 1
        P = view(A, k1:m, k1:k2)
        Tb = view(Tm, 1:w, k1:k2)
        _geqrt3!(P, Tb)
        if k2 < n
            _apply_block_reflector!(view(A, k1:m, k2+1:n), _explicit_V(P), Tb; trans=true)
        end
    end
    Tm
end

"""
    qr_mulQ!(B, QR, T; trans=false) -> B

B ← Q B (Qᵀ B with `trans=true`) using the result of `qr_ozaki!`. B has m rows.
"""
function qr_mulQ!(B::AbstractMatrix{E}, A::AbstractMatrix{E}, Tm::AbstractMatrix{E}; trans::Bool=false) where {E}
    m, n = size(A); k = min(m, n); nb = size(Tm, 1)
    blocks = collect(1:nb:k)
    order = trans ? blocks : reverse(blocks)
    for k1 in order
        k2 = min(k1 + nb - 1, k); w = k2 - k1 + 1
        _apply_block_reflector!(view(B, k1:m, :), _explicit_V(view(A, k1:m, k1:k2)), view(Tm, 1:w, k1:k2); trans)
    end
    B
end

"""
    qr_Q(QR, T; thin=true) -> Q

Q as a matrix (m×min(m,n) if thin, else m×m).
"""
function qr_Q(A::AbstractMatrix{E}, Tm::AbstractMatrix{E}; thin::Bool=true) where {E}
    m, n = size(A); c = thin ? min(m, n) : m
    Q = _zeros_like(A, m, c)
    o = E === BigFloat ? BigFloat(1; precision=_prec(A)) : one(E)
    for i in 1:c
        E === BigFloat ? mpfr_set!(Q[i, i], o) : (Q[i, i] = o)
    end
    qr_mulQ!(Q, A, Tm)
end
