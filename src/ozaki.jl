# Ozaki scheme: an extended-precision matrix product computed exactly as Float64 dgemms of integer slices.
#
# Column i of A gets a scale 2^{e_i} and is cut into NS slices, A_ki = Σ_s q^{(s)}_ki 2^{e_i − β s} with integers |q| ≤ 2^β.
# For each d = s + t the integer sum G_d = Σ_{s+t=d} Σ_k q^{(s)}_ki r^{(t)}_kj is one dgemm of stacked slices,
# exact as long as K·NS·2^{2β} < 2^53. Only s + t ≤ NS + 1 is kept; the dropped terms are below 2^{−βNS}·K·NS
# relative, and βNS ≥ bits + log2 K + 8.
#
# Cholesky and triangular solves factor narrow panels in extended precision and do the trailing updates
# (almost all the work) with these dgemms. The BLAS thread count is set separately from Julia's (BLAS.set_num_threads).

nbits_of(::Type{MultiFloat{Float64,N}}) where {N} = 53N
nbits_of(::Type{Float64}) = 53
@inline _el(A::AbstractMatrix{T}, i, j) where {T<:MultiFloat} = @inbounds A[i, j]
_nbits(A::AbstractMatrix{T}) where {T} = nbits_of(T)
_nbits(A::AbstractMatrix{BigFloat}) = _prec(A)
_nbits(A::AbstractMatrix{PElem}) = isempty(A) ? precision(BigFloat) : first(A).prec

"""
    oz_params(bits, K) -> (β, NS)

Slice width β and number of slices NS for inner products of length K: the largest β with
2β + ⌈log2 K⌉ + ⌈log2 NS⌉ ≤ 53, and βNS ≥ bits + ⌈log2 K⌉ + 8.
"""
function oz_params(bits::Int, K::Int)
    lk = ceil(Int, log2(max(K, 2)))
    for β in 26:-1:4
        NS = cld(bits + lk + 8, β)
        2β + lk + ceil(Int, log2(max(NS, 2))) <= 53 && return (β, NS)
    end
    error("Ozaki: K=$K is too large")
end

"""
    ozaki_slice!(F, R, e, A, K, NS, β)

Scale each column i of A[1:K, :] by e[i] and write its NS integer slices stacked: row (s−1)K + k of F is slice s,
and row (NS−s)K + k of R (unless R is nothing) is the same slice in reverse order.
"""
function ozaki_slice!(F::AbstractMatrix{Float64}, R, e::Vector{Int}, A::AbstractMatrix{T}, K::Int, NS::Int, β::Int) where {T}
    Threads.@threads for i in 1:size(A, 2)
        ei = _col_scale(A, i, K)
        e[i] = ei
        @inbounds for k in 1:K
            x = A[k, i]
            for s in 1:NS
                q = ei == typemin(Int) ? 0.0 : round(Float64(x * ldexp(1.0, β * s - ei)))
                _put_slice!(F, R, K, NS, s, k, i, q)
                ei == typemin(Int) || (x = x - q * ldexp(1.0, ei - β * s))
            end
        end
    end
    nothing
end
# BigFloat: N = round(x·2^{βNS−e}) as one integer (|N| < 2^{βNS}), then cut |N| into β-bit pieces with the sign.
function ozaki_slice!(F::AbstractMatrix{Float64}, R, e::Vector{Int}, A::AbstractMatrix{<:MPFRArg}, K::Int, NS::Int, β::Int)
    βNS = β * NS
    Threads.@threads for i in 1:size(A, 2)
        ei = _col_scale(A, i, K)
        e[i] = ei
        t = BigFloat(; precision=_nbits(A) + 64)
        z = Base.GMP.MPZ.realloc2(βNS + 128)
        @inbounds for k in 1:K
            if ei == typemin(Int)
                for s in 1:NS; _put_slice!(F, R, K, NS, s, k, i, 0.0); end
                continue
            end
            mpfr_mul_2si!(t, A[k, i], βNS - ei)
            ccall((:mpfr_get_z, :libmpfr), Int32, (Ref{BigInt}, Ref{BigFloat}, Int32), z, t, _RNDN)
            n = Int(abs(z.size)); sg = z.size < 0 ? -1.0 : 1.0
            for s in 1:NS
                _put_slice!(F, R, K, NS, s, k, i, sg * Float64(_getbits(z.d, n, β * (NS - s), β)))
            end
        end
    end
    nothing
end
"β bits (β ≤ 63) at bit position pos of the non-negative integer stored in the n limbs at d."
@inline function _getbits(d::Ptr{UInt64}, n::Int, pos::Int, β::Int)
    k = pos >> 6; sh = pos & 63
    k < n || return UInt64(0)
    v = unsafe_load(d, k + 1) >> sh
    sh + β > 64 && k + 1 < n && (v |= unsafe_load(d, k + 2) << (64 - sh))
    v & ((UInt64(1) << β) - 1)
end
"Scale of column i: e with max_k |A[k,i]| < 2^e (typemin(Int) if the column is zero)."
function _col_scale(A, i, K)
    mx = 0.0
    @inbounds for k in 1:K
        v = abs(_tod(A[k, i])); v > mx && (mx = v)
    end
    mx == 0.0 ? typemin(Int) : exponent(mx) + 1
end
_tod(x) = Float64(x)
_tod(x::MPFRArg) = _getd(x)
@inline function _put_slice!(F, R, K, NS, s, k, i, q)
    @inbounds F[(s-1)*K+k, i] = q
    R === nothing || (@inbounds R[(NS-s)*K+k, i] = q)
    nothing
end

"""
    ozaki_gemms!(G, FA, RB, K, M1, M2, NS; lower=false)

G[d−1][1:M1, 1:M2] = Σ_{s+t=d} SAₛᵀ SBₜ for d = 2..NS+1. FA holds A's slices in order and RB holds B's in reverse,
so all pairs (s, d−s) are one product of the first (d−1)K rows of FA and the last (d−1)K rows of RB.
`lower=true` computes only column blocks that contain i ≥ j (symmetric updates).
"""
function ozaki_gemms!(G, FA, RB, K::Int, M1::Int, M2::Int, NS::Int; lower::Bool=false, w::Int=256)
    for d in 2:NS+1
        n = d - 1
        a = view(FA, 1:n*K, 1:M1)
        b = view(RB, (NS-n)*K+1:NS*K, 1:M2)
        if lower
            for j0 in 1:w:M2
                jc = j0:min(j0 + w - 1, M2)
                BLAS.gemm!('T', 'N', 1.0, view(a, :, j0:M1), view(b, :, jc), 0.0, view(G[n], j0:M1, jc))
            end
        else
            BLAS.gemm!('T', 'N', 1.0, a, b, 0.0, view(G[n], 1:M1, 1:M2))
        end
    end
    nothing
end

"A[r, c] −= Σ_d G_d[i,j] 2^{ei+fj−βd}."
@inline function _ozaki_sub!(A::AbstractMatrix{T}, r, c, G, i, j, ei, fj, NS, β, tmp) where {T}
    (ei == typemin(Int) || fj == typemin(Int)) && return nothing
    acc = zero(T)
    @inbounds for d in 2:NS+1
        g = G[d-1][i, j]
        g == 0.0 && continue
        acc += T(g) * ldexp(1.0, ei + fj - β * d)
    end
    @inbounds A[r, c] -= acc
    nothing
end
_ozaki_tmp(::Type{T}, β, NS) where {T} = nothing

# BigFloat: carry Σ_d g_d 2^{−βd} (|g_d| < 2^53 integers) from the last slice up into one integer
# N = c·2^{βNS} + Σ_d r_d 2^{β(NS+1−d)} (0 ≤ r_d < 2^β), set it exactly with mpfr_set_z_2exp and subtract once.
struct _OzTmp
    z::BigInt
    t::BigFloat
    L::Vector{UInt64}
end
function _ozaki_tmp(::Type{BigFloat}, β, NS)
    nl = cld(β * NS + 64, 64) + 1
    _OzTmp(Base.GMP.MPZ.realloc2(64nl), BigFloat(; precision=64nl), zeros(UInt64, nl))
end
@inline function _orbits!(L::Vector{UInt64}, v::UInt64, pos::Int)
    k = pos >> 6 + 1; sh = pos & 63
    @inbounds L[k] |= v << sh
    sh > 0 && (@inbounds L[k+1] |= v >> (64 - sh))
    nothing
end
@inline function _ozaki_sub!(A::AbstractMatrix{BigFloat}, r, c, G, i, j, ei, fj, NS, β, tmp::_OzTmp)
    (ei == typemin(Int) || fj == typemin(Int)) && return nothing
    L = tmp.L; nl = length(L)
    fill!(L, 0)
    mask = (Int64(1) << β) - 1
    carry = Int64(0)
    @inbounds for d in NS+1:-1:2
        v = unsafe_trunc(Int64, G[d-1][i, j]) + carry
        _orbits!(L, UInt64(v & mask), β * (NS + 1 - d))
        carry = v >> β
    end
    βNS = β * NS
    neg = carry < 0
    if neg                                   # |N| = (|c| − 1)·2^{βNS} + (2^{βNS} − P)
        nz = false
        @inbounds for k in 1:nl; nz |= L[k] != 0; end
        if nz
            cy = UInt64(1)                   # 2^{βNS} − P = (~P + 1) truncated to βNS bits
            @inbounds for k in 1:nl
                x = ~L[k] + cy
                cy = (cy == 1 && x == 0) ? UInt64(1) : UInt64(0)
                L[k] = x
            end
            kb = βNS >> 6 + 1; sb = βNS & 63
            @inbounds L[kb] &= (UInt64(1) << sb) - 1
            @inbounds for k in kb+1:nl; L[k] = 0; end
            carry = -carry - 1
        else
            carry = -carry
        end
    end
    carry != 0 && _orbits!(L, UInt64(carry), βNS)
    n = nl
    @inbounds while n > 0 && L[n] == 0; n -= 1; end
    n == 0 && return nothing
    z = tmp.z
    @inbounds for k in 1:n; unsafe_store!(z.d, L[k], k); end
    z.size = neg ? -Cint(n) : Cint(n)
    t = tmp.t
    ccall((:mpfr_set_z_2exp, :libmpfr), Int32, (Ref{BigFloat}, Ref{BigInt}, Clong, Int32), t, z, ei + fj - β * (NS + 1), _RNDN)
    x = _el(A, r, c)
    mpfr_sub!(x, x, t)
    nothing
end

"Workspace of the slice scheme (slices F, R, scales e, integer sums G_d)."
struct _OzWS
    β::Int
    NS::Int
    F::Matrix{Float64}
    R::Matrix{Float64}
    e::Vector{Int}
    G::Vector{Matrix{Float64}}
end
function _OzWS(bits::Int, nb::Int, M::Int)
    β, NS = oz_params(bits, nb)
    _OzWS(β, NS, zeros(NS * nb, M), zeros(NS * nb, M), zeros(Int, M), [zeros(M, M) for _ in 1:NS])
end
_free!(ws) = nothing

"""
A[i, c] −= Σ_{p∈P} A[i,p] A[c,p] for i ∈ max(lo, rlo):M, c ∈ lo:lo+w−1, i ≥ c, with slice dgemms.
Rows lo:M are sliced once and their first w columns are reused as the B side.
"""
function _ozaki_syrk!(A::AbstractMatrix{T}, lo::Int, w::Int, P::UnitRange{Int}, ws::_OzWS; rlo::Int=lo) where {T}
    M = size(A, 1); Mr = M - lo + 1; K = length(P); β = ws.β; NS = ws.NS
    rl = max(rlo - lo + 1, 1); rl > Mr && return nothing
    ozaki_slice!(ws.F, ws.R, ws.e, _transpose_block(A, lo:M, P), K, NS, β)
    ozaki_gemms!(ws.G, ws.F, ws.R, K, Mr, w, NS; lower=true)
    e = ws.e; G = ws.G; nt = Threads.nthreads(); c0 = lo - 1
    if w >= 2nt                                   # column j has Mr − j + 1 rows, so deal columns round-robin
        @sync for t in 1:nt
            Threads.@spawn begin
                tmp = _ozaki_tmp(T, β, NS)
                for j in t:nt:w, i in max(j, rl):Mr
                    _ozaki_sub!(A, c0 + i, c0 + j, G, i, j, e[i], e[j], NS, β, tmp)
                end
            end
        end
    else                                          # few columns: split the rows
        _foreach_chunk(rl:Mr) do rows
            tmp = _ozaki_tmp(T, β, NS)
            for j in 1:w, i in max(j, first(rows)):last(rows)
                _ozaki_sub!(A, c0 + i, c0 + j, G, i, j, e[i], e[j], NS, β, tmp)
            end
        end
    end
    nothing
end

"""
    chol_lower_ozaki!(A; nb, leaf=16, ozmin=16, blas_threads, scheme=:auto, wc=128, done=0, split=0) -> Bool

Right-looking blocked Cholesky that overwrites the lower triangle with L (same contract as `chol_lower!`).
The trailing update A[lo:M, lo:M] −= L_p L_pᵀ is done with dgemms: `scheme=:crt` (default for BigFloat and MultiFloat) uses
residues modulo primes and the Chinese remainder theorem (crt.jl); `:slice` uses integer slices.
Each panel of width nb is split recursively; inner updates of width ≥ ozmin also use dgemms, narrower ones run in
extended precision in parallel row blocks. Only the leaf diagonal blocks (width leaf) are serial.
`wc` is the number of columns per residue update (the result does not depend on it).

Restart (`done = m`): if the leading m×m block holds the factor of A's leading m×m block (computed with the same nb,
leaf, ozmin) and rows m+1:M are untouched, only the rest is factored; the work is ∝ M³ − m³. The result is
bit-identical to `chol_lower_ozaki!(A; split=m)` on the whole matrix (panels do not cross column m).
"""
function chol_lower_ozaki!(A::AbstractMatrix{T}; nb::Union{Nothing,Int}=nothing, leaf::Int=16, ozmin::Int=16,
                           blas_threads::Int=_oz_blas_threads(), scheme::Symbol=:auto, wc::Int=128,
                           done::Int=0, split::Int=0) where {T}
    M = size(A, 1)
    0 <= done <= M || throw(ArgumentError("done must be in 0:M"))
    done == M && return true
    brk = max(done, split)
    crt = scheme === :crt || (scheme === :auto && (T === BigFloat || T <: MultiFloat))
    nb = max(1, min(something(nb, crt ? 256 : 128), M))
    ws = crt ? _CrtWS(_nbits(A), nb, M; wc=wc) : _OzWS(_nbits(A), nb, M)
    try
        _with_blas_threads(crt ? 1 : blas_threads) do         # the residue scheme runs one dgemm per prime in Julia tasks
            k1 = 1
            while k1 <= M
                k2 = min(k1 + nb - 1, M)
                k1 <= brk < k2 && (k2 = brk)
                if k2 <= done                      # factored columns: rows done+1:M only
                    _chol_panel_rows!(A, k1, k2, M, done + 1; leaf, ws, ozmin); ok = true
                else
                    ok = _chol_panel!(A, k1, k2, M; leaf, ws, ozmin)
                end
                ok || return false
                k2 < M && _ozaki_syrk!(A, k2 + 1, M - k2, k1:k2, ws; rlo=max(k2 + 1, done + 1))
                k1 = k2 + 1
            end
            true
        end
    finally
        _free!(ws)
    end
end

# BLAS threads spin between calls and compete with the Julia threads, so use half of them by default.
# The dgemm sums are exact, so the thread count does not change the result.
_oz_blas_threads() = max(1, Threads.nthreads() ÷ 2)
function _with_blas_threads(f, n::Int)
    old = BLAS.get_num_threads()
    n == old && return f()
    BLAS.set_num_threads(n)
    try
        f()
    finally
        BLAS.set_num_threads(old)
    end
end

"""
Factor panel k1:k2. A leaf (width ≤ leaf) is done right-looking on its diagonal block and in parallel row blocks
below it. A wider panel is split in half: factor the left half, subtract it from the right half (dgemm if wide
enough), factor the right half.
"""
function _chol_panel!(A::AbstractMatrix{T}, k1::Int, k2::Int, M::Int; leaf::Int=16, ws=nothing, ozmin::Int=16) where {T}
    if k2 - k1 + 1 > leaf
        m = (k1 + k2) ÷ 2
        _chol_panel!(A, k1, m, M; leaf, ws, ozmin) || return false
        if ws !== nothing && m - k1 + 1 >= ozmin
            _ozaki_syrk!(A, m + 1, k2 - m, k1:m, ws)
        else
            _par_rows(rows -> _panel_upd!(A, rows, m+1, k2, k1, m), m+1:M, (k2 - m) * (m - k1 + 1))
        end
        return _chol_panel!(A, m+1, k2, M; leaf, ws, ozmin)
    end
    @inbounds for k in k1:k2
        _chol_pivot!(A, k, k2) || return false
        for j in k+1:k2; _chol_upd!(A, j, k, k2); end
    end
    k2 < M && _par_rows(rows -> _panel_rows!(A, rows, k1, k2), k2+1:M, (k2 - k1 + 1)^2)
    true
end
"Rows rlo:M of an already factored panel k1:k2 (same recursion and operations as `_chol_panel!`, rows < rlo untouched)."
function _chol_panel_rows!(A::AbstractMatrix{T}, k1::Int, k2::Int, M::Int, rlo::Int; leaf::Int=16, ws=nothing, ozmin::Int=16) where {T}
    rlo > M && return nothing
    if k2 - k1 + 1 > leaf
        m = (k1 + k2) ÷ 2
        _chol_panel_rows!(A, k1, m, M, rlo; leaf, ws, ozmin)
        if ws !== nothing && m - k1 + 1 >= ozmin
            _ozaki_syrk!(A, m + 1, k2 - m, k1:m, ws; rlo=rlo)
        else
            _par_rows(rows -> _panel_upd!(A, rows, m+1, k2, k1, m), rlo:M, (k2 - m) * (m - k1 + 1))
        end
        return _chol_panel_rows!(A, m+1, k2, M, rlo; leaf, ws, ozmin)
    end
    _par_rows(rows -> _panel_rows!(A, rows, k1, k2), rlo:M, (k2 - k1 + 1)^2)
    nothing
end
"Call f on row chunks of r in parallel (serially if the work per row w is small)."
_par_rows(f, r, w) = Threads.nthreads() == 1 || length(r) * w < 2_MIN_WORK ? f(r) : _foreach_chunk(f, r)

# Subtract columns p1:p2 from columns c1:c2 (rows i ≥ c only).
function _panel_upd!(A::AbstractMatrix{T}, rows, c1, c2, p1, p2) where {T}
    @inbounds for c in c1:c2, p in p1:p2
        acp = A[c, p]
        for i in max(c, first(rows)):last(rows); A[i, c] -= A[i, p] * acp; end
    end
end
function _panel_upd!(A::AbstractMatrix{BigFloat}, rows, c1, c2, p1, p2)
    ncp = BigFloat(; precision=_prec(A))
    @inbounds for c in c1:c2, p in p1:p2
        mpfr_neg!(ncp, _el(A, c, p))
        for i in max(c, first(rows)):last(rows)
            x = _el(A, i, c); mpfr_fma!(x, _el(A, i, p), ncp, x)
        end
    end
end
# For each column c: subtract columns p < c in increasing p, then divide (same per-row order as right-looking).
function _panel_rows!(A::AbstractMatrix{T}, rows, k1, k2) where {T}
    @inbounds for c in k1:k2
        for p in k1:c-1
            acp = A[c, p]
            for i in rows; A[i, c] -= A[i, p] * acp; end
        end
        ic = inv(A[c, c])
        for i in rows; A[i, c] *= ic; end
    end
end
function _panel_rows!(A::AbstractMatrix{BigFloat}, rows, k1, k2)
    ncp = BigFloat(; precision=_prec(A))
    @inbounds for c in k1:k2
        for p in k1:c-1
            mpfr_neg!(ncp, _el(A, c, p))
            for i in rows; x = _el(A, i, c); mpfr_fma!(x, _el(A, i, p), ncp, x); end
        end
        d = _el(A, c, c)
        for i in rows; x = _el(A, i, c); mpfr_div!(x, x, d); end
    end
end

function _transpose_block(A::AbstractMatrix{T}, rows, cols) where {T}
    X = Matrix{T}(undef, length(cols), length(rows))
    @inbounds for (jj, j) in enumerate(rows), (ii, i) in enumerate(cols)
        X[ii, jj] = A[j, i]
    end
    X
end

"""
    trsm_lower_ozaki!(Y, L; nb=128, blas_threads) -> Y

Y ← L⁻¹Y (L lower triangular) by row panels: subtract the solved panels with slice dgemms, then solve the diagonal
block in extended precision. Each solved panel is sliced once and kept.
"""
function trsm_lower_ozaki!(Y::AbstractMatrix{T}, L::AbstractMatrix{T}; nb::Int=128, blas_threads::Int=_oz_blas_threads()) where {T}
    _with_blas_threads(() -> _trsm_lower_ozaki!(Y, L, nb), blas_threads)
end
function _trsm_lower_ozaki!(Y::AbstractMatrix{T}, L::AbstractMatrix{T}, nb::Int) where {T}
    M, N = size(Y)
    β, NS = oz_params(_nbits(L), nb)
    npan = cld(M, nb)
    RY = Vector{Matrix{Float64}}(undef, npan)
    eY = [zeros(Int, N) for _ in 1:npan]
    FL = zeros(NS * nb, nb)
    G = [zeros(nb, N) for _ in 1:NS]
    nt = Threads.nthreads()
    for p in 1:npan
        k1 = (p - 1) * nb + 1; k2 = min(k1 + nb - 1, M); w = k2 - k1 + 1
        for q in 1:p-1
            c1 = (q - 1) * nb + 1; c2 = min(c1 + nb - 1, M); K = c2 - c1 + 1
            eL = zeros(Int, w)
            ozaki_slice!(FL, nothing, eL, _transpose_block(L, k1:k2, c1:c2), K, NS, β)
            ozaki_gemms!(G, FL, RY[q], K, w, N, NS)
            eq = eY[q]
            @sync for t in 1:min(nt, N)
                Threads.@spawn begin
                    tmp = _ozaki_tmp(T, β, NS)
                    for j in t:nt:N, i in 1:w
                        _ozaki_sub!(Y, k1 + i - 1, j, G, i, j, eL[i], eq[j], NS, β, tmp)
                    end
                end
            end
        end
        Threads.@threads for j in 1:N
            _diag_block_solve!(Y, L, k1, k2, j)
        end
        RY[p] = zeros(NS * w, N)
        ozaki_slice!(zeros(NS * w, N), RY[p], eY[p], view(Y, k1:k2, 1:N), w, NS, β)
    end
    Y
end
"Forward substitution of the diagonal block: Y[k1:k2, j] ← L[k1:k2, k1:k2]⁻¹ Y[k1:k2, j]."
function _diag_block_solve!(Y::AbstractMatrix{T}, L::AbstractMatrix{T}, k1, k2, j) where {T}
    @inbounds for k in k1:k2
        yk = Y[k, j] / L[k, k]; Y[k, j] = yk
        for i in k+1:k2; Y[i, j] -= L[i, k] * yk; end
    end
end
function _diag_block_solve!(Y::AbstractMatrix{BigFloat}, L::AbstractMatrix{BigFloat}, k1, k2, j)
    @inbounds begin
        nyk = BigFloat(; precision=_prec(Y))
        for k in k1:k2
            ykj = Y[k, j]
            mpfr_div!(ykj, ykj, _el(L, k, k)); mpfr_neg!(nyk, ykj)
            for i in k+1:k2; mpfr_fma!(Y[i, j], _el(L, i, k), nyk, Y[i, j]); end
        end
    end
end
