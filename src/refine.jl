# Symmetric eigendecomposition and SVD refined from Float64 LAPACK results to extended precision with matrix products
# only (Ogita–Aishima). Each iteration squares the error and costs about four matrix products (Ozaki scheme).
#
# Symmetric (generalized) problem A x = λ B x (B = I allowed): from X̂, R = I − X̂ᵀ B X̂, S = X̂ᵀ A X̂, λ̃ᵢ = sᵢᵢ / (1 − rᵢᵢ),
#   eᵢⱼ = (sᵢⱼ + λ̃ⱼ rᵢⱼ) / (λ̃ⱼ − λ̃ᵢ) if |λ̃ᵢ − λ̃ⱼ| > δ, otherwise (and on the diagonal) rᵢⱼ / 2; X ← X̂ (I + E).
#   δ = 2(‖S − D̃‖ + ‖A‖‖R‖). Eigenvalues closer than δ form clusters; after the update, each cluster is rotated by
#   solving the small problem on its columns (shifted and rescaled) with the same method.
# SVD A = U Σ Vᵀ (square): R = I − ÛᵀÛ, S = I − V̂ᵀV̂, T = ÛᵀAV̂, σ̃ᵢ = tᵢᵢ / (1 − (rᵢᵢ + sᵢᵢ)/2),
#   (fᵢⱼ, gᵢⱼ) solve −σⱼ f + σᵢ g = −tᵢⱼ − σⱼ rᵢⱼ, σᵢ f − σⱼ g = −tⱼᵢ − σⱼ sᵢⱼ; U ← Û(I + F), V ← V̂(I + G).
#   Rectangular matrices go through a QR factorization first.

_frob(A) = sqrt(sum(x -> Float64(x)^2, A; init=0.0))
_mT(A, B) = mul_ozaki(permutedims(A), B)                        # AᵀB (the transpose only shares elements)
_tofloat64(A) = Float64.(A)
_convert_like(::Type{T}, X::AbstractMatrix{Float64}, prec) where {T} =
    T === BigFloat ? BigFloat[BigFloat(x; precision=prec) for x in X] : T.(X)
_prec_of(A::AbstractMatrix{BigFloat}) = _prec(A)
_prec_of(A) = 0
_eps_of(A::AbstractMatrix{BigFloat}) = 2.0^(-_prec(A))
_eps_of(A::AbstractMatrix{T}) where {T} = Float64(eps(T))
"I − G as a new matrix."
_I_minus(G::AbstractMatrix) = (R = -G; for i in axes(R, 1); R[i, i] += 1; end; R)
_offdiag_frob(S) = sqrt(max(_frob(S)^2 - sum(i -> Float64(S[i, i])^2, axes(S, 1); init=0.0), 0.0))
_sym(C) = (C + permutedims(C)) / 2

"Index sets (size ≥ 2) of eigenvalues chained by gaps ≤ δ when sorted."
function _clusters(λ::AbstractVector, δ::Float64)
    isempty(λ) && return Vector{Vector{Int}}()
    p = sortperm(Float64.(λ)); cl = Vector{Vector{Int}}(); cur = [p[1]]
    for k in 2:length(p)
        if Float64(λ[p[k]]) - Float64(λ[p[k-1]]) <= δ
            push!(cur, p[k])
        else
            length(cur) > 1 && push!(cl, cur); cur = [p[k]]
        end
    end
    length(cur) > 1 && push!(cl, cur)
    cl
end
"λᵢ = xᵢᵀ A xᵢ / xᵢᵀ B xᵢ for the columns of X."
function _rayleigh!(λ, A, B, X)
    AX = mul_ozaki(A, X); BX = B === nothing ? X : mul_ozaki(B, X)
    Threads.@threads for i in axes(X, 2)
        s = zero(λ[1]); r = zero(s)
        for k in axes(X, 1); s += X[k, i] * AX[k, i]; r += X[k, i] * BX[k, i]; end
        λ[i] = s / r
    end
    λ
end
"Replace columns J of X by X_J W."
function _rotate_cols!(X, J, W)
    XJ = mul_ozaki(X[:, J], W)
    for (c, j) in enumerate(J), i in axes(X, 1); X[i, j] = XJ[i, c]; end
    X
end

"""
    eigen_refine(A[, B]; X0=nothing, maxit=30, verbose=false) -> (λ, X, info)

Eigendecomposition A X = B X Λ (Xᵀ B X = I) of symmetric A (and symmetric positive definite B) in the precision of the
element type (BigFloat, Float64x4, …). The start is Float64 LAPACK (or X0); each matrix-product iteration squares the error.
Eigenvalues that Float64 cannot separate are refined per cluster as a small problem. λ is ascending.
info = (iters, clusters = largest number of clusters handled, corr = size of the last correction).
"""
function eigen_refine(A::AbstractMatrix{T}, B::Union{Nothing,AbstractMatrix{T}}=nothing;
                      X0=nothing, maxit::Int=30, verbose::Bool=false, depth::Int=0) where {T}
    n = size(A, 1); prec = _prec_of(A); u = _eps_of(A)
    n == 0 && return (T[], _zeros_like(A, 0, 0), (iters=0, clusters=0, corr=0.0))
    nA = _frob(A)
    if X0 === nothing
        A64 = Symmetric(_tofloat64(A))
        F = B === nothing ? eigen(A64) : eigen(A64, Symmetric(_tofloat64(B)))
        X = _convert_like(T, F.vectors, prec)
    else
        X = fresh(X0)
    end
    λ = [zero(A[1, 1]) for _ in 1:n]; iters = 0; ncl = 0; corr = Inf; prev = Inf
    nA == 0 && return (λ, X, (iters=0, clusters=0, corr=0.0))
    for it in 1:maxit
        iters = it
        BX = B === nothing ? X : mul_ozaki(B, X)
        R = _I_minus(_mT(X, BX))
        S = _sym(_mT(X, mul_ozaki(A, X)))               # exactly symmetric, so that eᵢⱼ + eⱼᵢ = rᵢⱼ keeps X orthogonal
        for i in 1:n; λ[i] = S[i, i] / (1 - R[i, i]); end
        nR = _frob(R)
        δ = 2 * (_offdiag_frob(S) + nA * nR)
        E = similar(S)
        Threads.@threads for j in 1:n
            for i in 1:n
                E[i, j] = i != j && abs(Float64(λ[j]) - Float64(λ[i])) > δ ? (S[i, j] + λ[j] * R[i, j]) / (λ[j] - λ[i]) : R[i, j] / 2
            end
        end
        corr = maximum(x -> abs(Float64(x)), E)
        Xn = fresh(X); gemm_ozaki!(Xn, X, E); X = Xn
        cl = depth < 8 && δ > 64 * n * u * nA ? _clusters(λ, δ) : Vector{Vector{Int}}()   # no clusters at the rounding level
        for J in cl                                     # rotate within each cluster using its updated columns
            XJ = X[:, J]
            SJ = _mT(XJ, mul_ozaki(A, XJ)); BJ = _mT(XJ, B === nothing ? XJ : mul_ozaki(B, XJ))
            μ0 = sum(λ[J]) / length(J)
            C = SJ - μ0 * BJ
            sc = _frob(C)
            sc <= 16 * u * nA && continue               # degenerate: any basis will do
            μ, W, _ = eigen_refine(_sym(C * T(1 / sc)), _sym(BJ); maxit, depth=depth + 1)
            _rotate_cols!(X, J, W)
            for (c, j) in enumerate(J); λ[j] = μ0 + sc * μ[c]; end    # the columns of J are reordered with the sub-solution
        end
        ncl = max(ncl, length(cl))
        verbose && @printf("  iteration %d: max|E| = %.2e, ‖R‖ = %.2e, δ = %.2e, clusters %d\n", it, corr, nR, δ, length(cl))
        (isempty(cl) && corr^2 <= 64 * n * u || corr <= 64 * n * u) && break   # the error is now about corr², or corr is at the rounding level
        # corr no longer falls: it is rounding noise divided by eigenvalue gaps that cannot be resolved better
        it >= 3 && corr > prev / 16 && nR <= 64 * n * u && break
        prev = corr
    end
    _rayleigh!(λ, A, B, X)                                # final values from the final vectors
    p = sortperm(Float64.(λ))
    λ[p], X[:, p], (iters=iters, clusters=ncl, corr=corr)
end
eigen_refine(A::Symmetric; kw...) = eigen_refine(Matrix(A); kw...)

"""
    svd_refine(A; maxit=30, verbose=false) -> (U, σ, V, info)

A = U Σ Vᵀ (U is m×n for m ≥ n, σ descending) in the precision of the element type, refined from Float64 LAPACK with
matrix-product iterations that square the error. For m > n, the QR factorization (`qr_ozaki!`) reduces it to the SVD of R.
"""
function svd_refine(A::AbstractMatrix{T}; maxit::Int=30, verbose::Bool=false, depth::Int=0) where {T}
    m, n = size(A)
    if m < n
        V, σ, U, info = svd_refine(permutedims(A); maxit, verbose, depth)
        return U, σ, V, info
    end
    if m > n
        F = fresh(A); Tm = qr_ozaki!(F)
        UR, σ, V, info = svd_refine(_triu_copy(view(F, 1:n, 1:n)); maxit, verbose, depth)
        Uf = _zeros_like(A, m, n)
        for j in 1:n, i in 1:n; Uf[i, j] = UR[i, j]; end
        Uf = fresh(Uf)
        qr_mulQ!(Uf, F, Tm)
        return Uf, σ, V, info
    end
    prec = _prec_of(A); u = _eps_of(A)
    nA = _frob(A)
    F = svd(_tofloat64(A))
    U = _convert_like(T, F.U, prec); V = _convert_like(T, F.V, prec)
    σ = [zero(A[1, 1]) for _ in 1:n]; iters = 0; ncl = 0; corr = Inf
    for it in 1:maxit
        iters = it
        R = _I_minus(_mT(U, U)); S = _I_minus(_mT(V, V))
        Tt = _mT(U, mul_ozaki(A, V))
        for i in 1:n; σ[i] = Tt[i, i] / (1 - (R[i, i] + S[i, i]) / 2); end
        # Wider margin than for eigenvalues: small singular values have absolute errors ~ u‖A‖ in σ̃, and pairs with
        # gaps just above that would get O(1) corrections.
        δ = 32 * (_offdiag_frob(Tt) + nA * (_frob(R) + _frob(S)))
        Fm = similar(Tt); Gm = similar(Tt)
        Threads.@threads for j in 1:n
            for i in 1:n
                σi = σ[i]; σj = σ[j]
                if i != j && abs(Float64(σj) - Float64(σi)) > δ && abs(Float64(σj) + Float64(σi)) > δ
                    d = σj * σj - σi * σi
                    Fm[i, j] = (σj * Tt[i, j] + σj * σj * R[i, j] + σi * Tt[j, i] + σi * σj * S[i, j]) / d
                    Gm[i, j] = (σj * Tt[j, i] + σj * σj * S[i, j] + σi * Tt[i, j] + σi * σj * R[i, j]) / d
                else
                    Fm[i, j] = R[i, j] / 2; Gm[i, j] = S[i, j] / 2
                end
            end
        end
        corr = max(maximum(x -> abs(Float64(x)), Fm), maximum(x -> abs(Float64(x)), Gm))
        Un = fresh(U); gemm_ozaki!(Un, U, Fm); U = Un
        Vn = fresh(V); gemm_ozaki!(Vn, V, Gm); V = Vn
        # Clusters: with the updated columns, eigendecompose (T_JJᵀ T_JJ − μ0² V_JᵀV_J)/scale (W); V_J ← V_J W, U_J ← A V_J Σ_J⁻¹.
        # The next iteration restores the orthogonality of U and V quadratically.
        cl = depth < 8 && δ > 64 * n * u * nA ? _clusters(σ, δ) : Vector{Vector{Int}}()
        for J in cl
            if minimum(x -> abs(Float64(x)), σ[J]) < 0.5 * maximum(x -> abs(Float64(x)), σ[J])
                # Cluster near zero: rescaled T_JJ = U_Jᵀ A V_J separates its singular values (squaring would lose digits). Recurse.
                TJ = _mT(U[:, J], mul_ozaki(A, V[:, J]))
                sc = _frob(TJ)
                sc <= 16 * u * nA && continue
                Us, σs, Vs, _ = svd_refine(TJ * T(1 / sc); maxit, depth=depth + 1)
                _rotate_cols!(U, J, Us); _rotate_cols!(V, J, Vs)
                for (c, j) in enumerate(J); σ[j] = sc * σs[c]; end
                continue
            end
            VJ = V[:, J]; AVJ = mul_ozaki(A, VJ)
            G0 = _mT(AVJ, AVJ); BJ = _mT(VJ, VJ)
            μ0 = sum(σ[J]) / length(J)
            C = G0 - (μ0 * μ0) * BJ
            sc = _frob(C)
            sc <= 16 * u * nA^2 && continue
            μ, W, _ = eigen_refine(_sym(C * T(1 / sc)), _sym(BJ); maxit, depth=depth + 1)
            _rotate_cols!(V, J, W)
            UJ = mul_ozaki(AVJ, W)
            for (c, j) in enumerate(J)
                s_c = sqrt(max(μ0 * μ0 + μ[c] * sc, zero(μ0)))
                σ[j] = s_c
                iszero(s_c) && continue
                for i in 1:m; U[i, j] = UJ[i, c] / s_c; end
            end
        end
        ncl = max(ncl, length(cl))
        verbose && @printf("  iteration %d: max|F,G| = %.2e, δ = %.2e, clusters %d\n", it, corr, δ, length(cl))
        (isempty(cl) && corr^2 <= 64 * n * u || corr <= 64 * n * u) && break
    end
    AV = mul_ozaki(A, V)                                  # final values from the final vectors
    Threads.@threads for i in 1:n
        t = zero(σ[1]); uu = zero(t); vv = zero(t)
        for k in 1:m; t += U[k, i] * AV[k, i]; uu += U[k, i] * U[k, i]; end
        for k in 1:n; vv += V[k, i] * V[k, i]; end
        σ[i] = t / ((uu + vv) / 2)
    end
    for i in 1:n                                         # absorb negative singular values into U, sort descending
        if σ[i] < 0
            σ[i] = -σ[i]; for k in 1:m; U[k, i] = -U[k, i]; end
        end
    end
    p = sortperm(Float64.(σ); rev=true)
    U[:, p], σ[p], V[:, p], (iters=iters, clusters=ncl, corr=corr)
end
