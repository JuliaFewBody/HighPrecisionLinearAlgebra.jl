# Lowest eigenpair of H x = E S x by shifted inverse iteration (P = L⁻¹HL⁻ᵀ is never formed).

"Lower triangle of H − σS (upper triangle zero), columns in parallel."
shifted_lower(H::AbstractMatrix{T}, S::AbstractMatrix{T}, σ::T) where {T} = shifted_lower!(zeros(T, size(H)...), H, S, σ)
"Write the lower triangle of H − σS into A (the upper triangle is untouched)."
function shifted_lower!(A::AbstractMatrix{T}, H::AbstractMatrix{T}, S::AbstractMatrix{T}, σ::T) where {T}
    M = size(H, 1)
    _foreach_tri_cols(M) do cols
        @inbounds for j in cols, i in j:M
            A[i, j] = H[i, j] - σ * S[i, j]
        end
    end
    A
end

"""
    add_lowrank_lower!(A, V, c) -> A

Add c·V Vᵀ (V is M×k) to the lower triangle of A. Used to lift converged lower states in inverse iteration (`lift`).
"""
function add_lowrank_lower!(A::AbstractMatrix{T}, V::AbstractMatrix{T}, c) where {T}
    M, k = size(V)
    W = V .* T(c)
    _foreach_tri_cols(M) do cols
        @inbounds for j in cols, i in j:M
            s = A[i, j]
            for m in 1:k; s += W[i, m] * V[j, m]; end
            A[i, j] = s
        end
    end
    A
end
function add_lowrank_lower!(A::Union{SymPacked,AbstractMatrix{BigFloat}}, V::AbstractMatrix{BigFloat}, c)
    M, k = size(V); p = precision(V[1, 1])
    W = [BigFloat(V[i, m] * c; precision=p) for i in 1:M, m in 1:k]
    GC.@preserve A V W _foreach_tri_cols(M) do cols
        @inbounds for j in cols, i in j:M
            a = _el(A, i, j)
            for m in 1:k; mpfr_fma!(a, W[i, m], V[j, m], a); end
        end
    end
    A
end

"""
    lowest_eig_si(H, S, σ; x0, chol!, maxit=30, rounds=6, tol=10eps, work, lift, etol, etol_dx, prods)
        -> (E, x, nit, converged, σ) or nothing

Lowest eigenpair of H x = E S x (H, S symmetric, S positive definite) by shifted inverse iteration x ← (H − σS)⁻¹ S x.
σ must lie below the lowest eigenvalue so that H − σS is positive definite; otherwise `nothing` is returned and the
caller lowers σ. x is S-normalized and E is the Rayleigh quotient. Each iteration costs one S·vector product and one
triangular solve (xᵀHx follows from (H − σS) x_new = S x_old; only the final E uses H·vector directly).

Convergence: the change of x (largest component ratio) is below `tol` or stops shrinking (rounding floor); since the
error of E is the square of that of x, E is then at its floor too. With `etol`, iteration also stops from the second
step once the change of E is ≤ etol while x shrinks fast (change ≤ 1e-3 of the previous) and is ≤ `etol_dx`.
If no convergence within a round, σ is moved closer (E − gap/f, f = 10, 3, 1.5) and the factorization is redone.

- `chol!`: Cholesky that overwrites the lower triangle (`chol_lower!`, `chol_lower_ozaki!`, …).
- `work`: storage of H's shape (e.g. an unneeded factor) to hold H − σS without allocating another matrix.
- `prods::Ref`: receives (H x, S x) of the final x.
- `lift = (X, c)`: excited states. With X the converged lower eigenvectors (XᵀSX = I), B = H − σS + c·(SX)(SX)ᵀ moves
  them to E_m − σ + c; for c > σ − E_m, B is positive definite and the iteration converges to the lowest state not in X.
"""
function lowest_eig_si(H::AbstractMatrix{T}, S::AbstractMatrix{T}, σ::T; x0=nothing, chol! = chol_lower!,
                       maxit::Int=30, rounds::Int=6, tol=10eps(T), work=nothing, lift=nothing, etol=nothing, etol_dx=Inf, prods=nothing) where {T}
    M = size(H, 1)
    SX = lift === nothing ? nothing : reduce(hcat, [symv!(Vector{T}(undef, M), S, lift[1][:, m]) for m in axes(lift[1], 2)])
    fillA!(A, σ) = (shifted_lower!(A, H, S, σ); SX === nothing || add_lowrank_lower!(A, SX, lift[2]); A)
    A = work === nothing ? shifted_lower(H, S, σ) : shifted_lower!(work, H, S, σ)
    SX === nothing || add_lowrank_lower!(A, SX, lift[2])
    chol!(A) || return nothing
    x = x0 === nothing ? T[one(T) for _ in 1:M] : T[T(v) for v in x0]
    Sx = symv!(Vector{T}(undef, M), S, x)
    s = sqrt(dot(x, Sx)); x = x ./ s; Sx = Sx ./ s
    E = T(Inf); Eprev = T(Inf); conv = false; nit = 0
    for round in 1:rounds
        dxprev = Inf
        for _ in 1:maxit
            xold = x
            x = llt_solve!(fresh(Sx), A)                 # (H − σS) x_new = S x
            Sxn = symv!(Vector{T}(undef, M), S, x)
            xSx = dot(x, Sxn); E = σ + dot(x, Sx) / xSx  # x_newᵀ H x_new = σ x_newᵀ S x_new + x_newᵀ S x_old
            s = sqrt(xSx); x = x ./ s; Sx = Sxn ./ s
            nit += 1
            dx = Float64(maximum(abs, x - xold) / maximum(abs, x))
            # Do not compare the change of E with eps: with an ill-conditioned S the Rayleigh quotient fluctuates far above eps.
            conv = nit >= 2 && abs(E - Eprev) <= sqrt(sqrt(eps(T))) * abs(E) && (dx <= Float64(tol) || dx >= dxprev)
            conv |= etol !== nothing && nit >= 2 && abs(E - Eprev) <= etol && dx <= 1e-3 * dxprev && dx <= etol_dx
            conv && break
            Eprev = E; dxprev = dx
        end
        (conv || round == rounds) && break
        gap = E - σ
        ok = false
        for f in (10.0, 3.0, 1.5)                      # rebuild in the same storage
            σn = E - gap / f
            fillA!(A, σn)
            chol!(A) && (σ = σn; ok = true; break)
        end
        if !ok                                          # none factored: restore the old shift and stop
            fillA!(A, σ); chol!(A)
            break
        end
    end
    Hx = symv!(Vector{T}(undef, M), H, x)
    E = dot(x, Hx) / dot(x, Sx)
    prods === nothing || (prods[] = (Hx, Sx))
    (E, x, nit, conv, σ)
end
