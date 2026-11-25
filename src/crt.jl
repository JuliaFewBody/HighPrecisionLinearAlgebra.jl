# Residue (CRT) variant of the Ozaki scheme for BigFloat (Ozaki scheme II).
#
# Row i of A gets a scale 2^{e_i} and N_ki = round(A[i, p_k]·2^{B−e_i}) (|N| < 2^B). The integer Σ_k N_ki N_kj
# (|Z| ≤ K 2^{2B}) is recovered from its residues z_m = Σ_k (N_ki mod p_m)(N_kj mod p_m) mod p_m, one exact dgemm
# per prime p_m (K((p−1)/2)² < 2^53). With P = Π p_m > 4K 2^{2B}: Z = Σ z_m c_m − qP, c_m = (P/p_m)((P/p_m)⁻¹ mod p_m),
# q = round(Σ z_m c_m / P).
#
# The slice scheme needs NS(NS+1)/2 dgemms (136 at 320 bits, K = 128); this one needs one per prime (29).
# Reconstruction per element: residues → sums of 16-bit digits (exact in Float64, a dgemm per 64 elements) →
# carries → one mpfr_set_z_2exp. Only the rounding of the inputs to B = bits + ⌈log2 K⌉ + 8 bits is inexact.

const _CRT_PIECE = 16
const _CRT_RB = 64          # rows per reconstruction block
const _CRT_GRAB = 8         # reconstruction jobs taken per atomic increment

"Primes below 2^bp, largest first, until their product exceeds 2^need."
function _crt_primes(bp::Int, need::Int)
    ps = Int[]; lg = 0.0
    c = 2^bp - 1
    while lg < need
        isp = c > 1 && all(c % d != 0 for d in 2:isqrt(c))
        isp && (push!(ps, c); lg += log2(c))
        c -= 1
    end
    ps
end

"Constants of the residue scheme for `bits` bits and inner products of length ≤ K."
struct _CrtParams
    B::Int                    # bits of the input integers
    p::Vector{Float64}        # primes
    pinv::Vector{Float64}
    f::Vector{Float64}        # ((P/p_m)⁻¹ mod p_m)/p_m, so q = round(Σ z_m f_m)
    pw::Matrix{Float64}       # pw[k, m] = 2^{16(k−1)} mod p_m
    C::Matrix{Float64}        # C[k, m] = 16-bit digit k of c_m; C[k, nm+1] = digit k of P
    nin::Int                  # input digits
    nq::Int                   # output digits
end
function _CrtParams(bits::Int, K::Int)
    lk = ceil(Int, log2(max(K, 2)))
    B = bits + lk + 8
    bp = (53 - lk) ÷ 2 + 1
    while K * ((2^bp - 2) ÷ 2)^2 >= 2^53; bp -= 1; end
    ps = _crt_primes(bp, 2B + lk + 3)
    P = prod(BigInt, ps)
    nm = length(ps)
    nin = cld(B, _CRT_PIECE)
    nq = cld(ndigits(P; base=2) + 1, _CRT_PIECE)
    pw = [Float64(powermod(BigInt(2), _CRT_PIECE * (k - 1), p)) for k in 1:nin, p in ps]
    C = zeros(nq, nm + 1); f = zeros(nm)
    piece(x, k) = Float64((x >> (_CRT_PIECE * (k - 1))) & (2^_CRT_PIECE - 1))
    for (m, p) in enumerate(ps)
        Pm = P ÷ p; inv = invmod(Pm % p, p)
        f[m] = Float64(inv) / Float64(p)
        for k in 1:nq; C[k, m] = piece(Pm * inv, k); end
    end
    for k in 1:nq; C[k, nm+1] = piece(P, k); end
    _CrtParams(B, Float64.(ps), 1 ./ Float64.(ps), f, pw, C, nin, nq)
end

"Per-thread scratch for reconstruction."
struct _CrtTmp
    zb::Matrix{Float64}      # residues of a row block (and −q in column nm+1)
    acc::Matrix{Float64}     # 16-bit digit sums (zb · Cᵀ)
    s::Vector{Float64}
    big::BigInt
    t::BigFloat
end
function _CrtTmp(cp::_CrtParams, prec::Int)
    nm = length(cp.p); nl = cld(cp.nq, 4)
    _CrtTmp(zeros(_CRT_RB, nm + 1), zeros(_CRT_RB, cp.nq), zeros(_CRT_RB), Base.GMP.MPZ.realloc2(64nl), BigFloat(; precision=prec + 64))
end
"Per-thread scratch for computing residues (inner length ≤ nb)."
struct _ResTmp
    pcs::Vector{Float64}; PC::Matrix{Float64}; R::Matrix{Float64}; sg::Vector{Float64}
end
_ResTmp(cp::_CrtParams, nb::Int) = _ResTmp(zeros(cp.nin), zeros(nb, cp.nin), zeros(nb, length(cp.p)), zeros(nb))

"""
Workspace of the residue Cholesky: input residues F[k, i, m], product residues G[i, j, m], column block width wc,
and per-thread scratch reused for the whole factorization. F and G are malloc'ed and released by `_free!`.
"""
struct _CrtWS
    cp::_CrtParams
    F::Array{Float64,3}
    G::Array{Float64,3}
    e::Vector{Int}
    wc::Int
    bits::Int
    tmps::Vector{_CrtTmp}
    rts::Vector{_ResTmp}
end
function _CrtWS(bits::Int, nb::Int, M::Int; wc::Int=128)
    cp = _CrtParams(bits, nb); nm = length(cp.p); nt = Threads.nthreads()
    _CrtWS(cp, _calloc_array(Float64, nb, M, nm), _calloc_array(Float64, M, min(wc, M), nm), zeros(Int, M), min(wc, M), bits,
           [_CrtTmp(cp, bits) for _ in 1:nt], [_ResTmp(cp, nb) for _ in 1:nt])
end
_free!(ws::_CrtWS) = (_free_array!(ws.F); _free_array!(ws.G); nothing)

"A zero-filled array from calloc (not owned by the GC; release with `_free_array!`)."
function _calloc_array(::Type{T}, dims::Int...) where {T}
    n = prod(dims)
    n == 0 && return zeros(T, dims...)
    p = Libc.calloc(n, sizeof(T))
    p == C_NULL && throw(OutOfMemoryError())
    unsafe_wrap(Array, Ptr{T}(p), dims; own=false)
end
_free_array!(A::Array) = (isempty(A) || Libc.free(pointer(A)); nothing)

# x mod p as the symmetric representative in [−(p−1)/2, (p−1)/2] (|x| < 2^52)
@inline function _smod(x::Float64, p::Float64, pinv::Float64)
    r = muladd(-round(x * pinv), p, x)
    r > p / 2 ? r - p : r < -p / 2 ? r + p : r
end
# x mod p in [0, p)
@inline function _umod(x::Float64, p::Float64, pinv::Float64)
    r = muladd(-floor(x * pinv), p, x)
    r < 0 ? r + p : r >= p ? r - p : r
end

"MPFR exponent (x = 0.1… × 2^exp; zero, NaN and Inf have special values near −2^63)."
@inline _mpfr_exp(x::Union{BigFloat,Ptr{BigFloat}}) = GC.@preserve x unsafe_load(Ptr{Clong}(Ptr{UInt8}(_p(x)) + 16))
@inline _mpfr_exp(x::PElem) = _dec(unsafe_load(x.p))[2]
"(prec, sign, exponent, limb pointer) of an MPFR number; use inside GC.@preserve x."
@inline function _mpfr_parts(x::Union{BigFloat,Ptr{BigFloat}})
    p = Ptr{UInt8}(_p(x))
    (unsafe_load(Ptr{Clong}(p)), unsafe_load(Ptr{Cint}(p + 8)), unsafe_load(Ptr{Clong}(p + 16)), unsafe_load(Ptr{Ptr{UInt64}}(p + 24)))
end
@inline _mpfr_parts(x::PElem) = ((s, ex) = _dec(unsafe_load(x.p)); (x.prec, s, ex, x.p + 8))

"""
The 16-bit digits of N = round(|x| 2^{B−e}) read directly from the limbs into pcs[1:nin] (digit 1 may reach 2^16
after rounding). Return the sign (±1.0). Requires |x| < 2^e.
"""
@inline function _crt_pieces!(pcs::Vector{Float64}, x::MPFRArg, B::Int, e::Int, nin::Int)
    GC.@preserve x begin
        prec, sg, ex, d = _mpfr_parts(x)
        if ex < -(Int64(1) << 40)                                  # zero (or NaN, Inf)
            fill!(pcs, 0.0); return 1.0
        end
        nl = cld(prec, 64)
        r = 64nl - (ex + B - e)                                    # |x| 2^{B−e} = (mantissa integer) · 2^{−r}
        @inbounds for q in 1:nin
            pos = _CRT_PIECE * (q - 1) + r
            pcs[q] = pos >= 0 ? Float64(_getbits(d, nl, pos, _CRT_PIECE)) :
                     pos > -_CRT_PIECE ? Float64(_getbits(d, nl, 0, _CRT_PIECE + pos) << (-pos)) : 0.0
        end
        r >= 1 && (@inbounds pcs[1] += Float64(_getbits(d, nl, r - 1, 1)))   # round half up
        sg < 0 ? -1.0 : 1.0
    end
end

"""
F[k, i, m] = round(x(i, k)·2^{B−e[i]}) mod p_m (symmetric residue) for i ∈ 1:n, k ∈ 1:K, where x(i, k) = get(i, k) and
max_k |x(i,k)| < 2^{e[i]} (typemin if the row is zero). Residues come from the 16-bit digits times 2^{16(q−1)} mod p_m
(one small dgemm per row, sums < 2^45 are exact). Rows `skip` are left alone.
"""
function _gres!(F::AbstractArray{Float64,3}, e::AbstractVector{Int}, get::G, rows, K::Int, cp::_CrtParams, rts) where {G}
    n = min(length(rts), length(rows))
    n == 0 && return nothing
    @sync for (slot, rr) in enumerate(Iterators.partition(rows, cld(length(rows), n)))
        Threads.@spawn _gres_rows!(F, e, get, rr, K, cp, rts[slot], Val(get(first(rows), 1) isa MultiFloat))
    end
    nothing
end
"""
Residues of MultiFloat rows: each limb y = x_l 2^{B−e} is rounded to an integer on its own (the limbs do not overlap,
so the total differs from round(x 2^{B−e}) by at most N/2), and its 16-bit digits are added with its sign into the
digit slots; the residues are then one small dgemm per row, as for MPFR (|slot| < N·2^16, sums < 2^45 are exact).
"""
function _gres_rows!(F, e, get::G, rows, K, cp::_CrtParams, rt::_ResTmp, ::Val{true}) where {G}
    nm = length(cp.p); B = cp.B; nin = cp.nin
    PC = view(rt.PC, 1:K, :); R = view(rt.R, 1:K, :)
    @inbounds for i in rows
        ei = typemin(Int)
        for k in 1:K
            x0 = get(i, k)._limbs[1]
            iszero(x0) || isfinite(x0) && (ex = exponent(x0) + 2; ex > ei && (ei = ex))
        end
        e[i] = ei
        if ei == typemin(Int)
            for m in 1:nm, k in 1:K; F[k, i, m] = 0.0; end
            continue
        end
        for k in 1:K
            for q in 1:nin; PC[k, q] = 0.0; end
            for y0 in get(i, k)._limbs
                t = round(ldexp(y0, B - ei))
                iszero(t) && continue
                a = abs(t); sh = max(exponent(a) - 52, 0)
                mt = unsafe_trunc(UInt64, ldexp(a, -sh)); sg = t < 0 ? -1.0 : 1.0
                for q in sh >> 4:min((sh + 52) >> 4, nin - 1)
                    lo = 16q - sh
                    PC[k, q+1] += sg * Float64((lo >= 0 ? mt >> lo : mt << -lo) & 0xffff)
                end
            end
        end
        BLAS.gemm!('N', 'N', 1.0, PC, cp.pw, 0.0, R)
        for m in 1:nm
            pm = cp.p[m]; pi = cp.pinv[m]
            @simd for k in 1:K
                F[k, i, m] = _smod(R[k, m], pm, pi)
            end
        end
    end
    nothing
end
_gres_rows!(F, e, get::G, rows, K, cp::_CrtParams, rt::_ResTmp, ::Val{false}) where {G} = _gres_rows!(F, e, get, rows, K, cp, rt)
function _gres_rows!(F, e, get::G, rows, K, cp::_CrtParams, rt::_ResTmp) where {G}
    nm = length(cp.p); B = cp.B; nin = cp.nin
    pcs = rt.pcs; PC = view(rt.PC, 1:K, :); R = view(rt.R, 1:K, :); sg = rt.sg
    @inbounds for i in rows
        ei = typemin(Int)
        for k in 1:K
            ex = _mpfr_exp(get(i, k)); ex > ei && (ei = ex)
        end
        ei < -(Int64(1) << 40) && (ei = typemin(Int))
        e[i] = ei
        if ei == typemin(Int)
            for m in 1:nm, k in 1:K; F[k, i, m] = 0.0; end
            continue
        end
        for k in 1:K
            sg[k] = _crt_pieces!(pcs, get(i, k), B, ei, nin)
            for q in 1:nin; PC[k, q] = pcs[q]; end
        end
        BLAS.gemm!('N', 'N', 1.0, PC, cp.pw, 0.0, R)
        for m in 1:nm
            pm = cp.p[m]; pi = cp.pinv[m]
            @simd for k in 1:K
                F[k, i, m] = sg[k] * _smod(R[k, m], pm, pi)
            end
        end
    end
    nothing
end

"""
A[r0 + i, c] −= Z_i · 2^{e[i]+fj−2B} for i = 1..nr, where Z[i, :] holds the residues of Z_i (multiples of p allowed).
z_m and q first, then the 16-bit digit sums Σ z_m c_m − qP (exact, |·| < 2^46) for all nr rows in one dgemm.
"""
function _crt_sub_block!(A, r0, c, nr, e, fj, cp::_CrtParams, tmp::_CrtTmp, Z=tmp.zb)
    fj == typemin(Int) && return nothing
    nm = length(cp.p); nq = cp.nq; zb = tmp.zb; acc = tmp.acc
    s = tmp.s; fill!(s, 0.0)
    @inbounds for m in 1:nm
        pm = cp.p[m]; pi = cp.pinv[m]; fm = cp.f[m]
        @simd for i in 1:nr
            zm = _umod(Z[i, m], pm, pi)
            zb[i, m] = zm; s[i] = muladd(zm, fm, s[i])
        end
    end
    @inbounds for i in 1:nr; zb[i, nm+1] = -round(s[i]); end
    BLAS.gemm!('N', 'T', 1.0, view(zb, 1:nr, :), cp.C, 0.0, view(acc, 1:nr, :))
    nl = cld(nq, 4); top = nq - 4(nl - 1)
    # Normalize the digit sums to [0, 2^16) with carries (vectorized across rows). The final carry is 0 or −1 (|Z| < P/2).
    ai = reinterpret(Int64, acc); cyv = reinterpret(Int64, s)
    @inbounds begin
        @simd for i in 1:nr; cyv[i] = 0; end
        for k in 1:nq
            @simd for i in 1:nr
                v = unsafe_trunc(Int64, acc[i, k]) + cyv[i]
                ai[i, k] = v & 0xffff; cyv[i] = v >> 16
            end
        end
    end
    @inbounds for l in 1:nl                                         # pack 4 digits per 64-bit word (word l in column l)
        k = 4(l - 1)
        if k + 4 <= nq
            @simd for i in 1:nr; ai[i, l] = ai[i, k+1] | (ai[i, k+2] << 16) | (ai[i, k+3] << 32) | (ai[i, k+4] << 48); end
        else
            for i in 1:nr
                w = ai[i, k+1]
                for q in 1:nq-k-1; w |= ai[i, k+q+1] << (16q); end
                ai[i, l] = w
            end
        end
    end
    _crt_store!(A, r0, c, nr, e, fj, cp, tmp, ai, cyv, nl, top)
    nothing
end

"Two's complement words of row i → magnitude in d[1:nl]; return (number of words, negative?) (0 words if zero)."
@inline function _crt_words!(d, ai, cyv, i, nl, top)
    nz = UInt64(0)
    @inbounds for l in 1:nl; w = reinterpret(UInt64, ai[i, l]); unsafe_store!(d, w, l); nz |= w; end
    neg = @inbounds cyv[i] < 0
    if neg
        top < 4 && unsafe_store!(d, unsafe_load(d, nl) | ~((UInt64(1) << (16top)) - 1), nl)
        cy = UInt64(1)
        for l in 1:nl
            x = ~unsafe_load(d, l) + cy
            cy = (cy == 1 && x == 0) ? UInt64(1) : UInt64(0)
            unsafe_store!(d, x, l)
        end
    elseif nz == 0
        return 0, false
    end
    n = nl
    while n > 0 && unsafe_load(d, n) == 0; n -= 1; end
    n, neg
end

# MultiFloat: the integer from its top 53-bit fields (exact Float64 each, disjoint bits, truncated below
# nbits + 12 bits). The fields are already a non-overlapping expansion: the lowest ones are summed into the last limb
# and one renormalization gives the MultiFloat.
_nfields(::Type{MultiFloat{Float64,N}}) where {N} = cld(53N + 12, 53)
@inline function _crt_field(d, n, tb, f, sc)
    pos = tb - 53f
    pos <= -53 && return 0.0
    ldexp(Float64(pos >= 0 ? _getbits(d, n, pos, 53) : _getbits(d, n, 0, 53 + pos)), max(pos, 0) + sc)
end
function _crt_store!(A::AbstractMatrix{MultiFloat{Float64,N}}, r0, c, nr, e, fj, cp::_CrtParams, tmp::_CrtTmp, ai, cyv, nl, top) where {N}
    T = MultiFloat{Float64,N}; d = tmp.big.d
    @inbounds for i in 1:nr
        e[i] == typemin(Int) && continue
        n, neg = _crt_words!(d, ai, cyv, i, nl, top)
        n == 0 && continue
        sc = e[i] + fj - 2cp.B; tb = 64n - leading_zeros(unsafe_load(d, n))
        fs = ntuple(f -> _crt_field(d, n, tb, f, sc), Val(_nfields(T)))
        v = T(MultiFloats.renormalize(ntuple(l -> l < N ? fs[l] : +(fs[N:end]...), Val(N))))
        A[r0 + i, c] = neg ? A[r0 + i, c] + v : A[r0 + i, c] - v
    end
    nothing
end

function _crt_store!(A, r0, c, nr, e, fj, cp::_CrtParams, tmp::_CrtTmp, ai, cyv, nl, top)
    big = tmp.big; d = big.d; t = tmp.t
    @inbounds for i in 1:nr
        e[i] == typemin(Int) && continue
        nz = UInt64(0)
        for l in 1:nl; w = reinterpret(UInt64, ai[i, l]); unsafe_store!(d, w, l); nz |= w; end
        neg = cyv[i] < 0
        if neg                                                      # two's complement → magnitude
            top < 4 && unsafe_store!(d, unsafe_load(d, nl) | ~((UInt64(1) << (16top)) - 1), nl)
            cy = UInt64(1)
            for l in 1:nl
                x = ~unsafe_load(d, l) + cy
                cy = (cy == 1 && x == 0) ? UInt64(1) : UInt64(0)
                unsafe_store!(d, x, l)
            end
        elseif nz == 0
            continue
        end
        n = nl
        while n > 0 && unsafe_load(d, n) == 0; n -= 1; end
        n == 0 && continue
        big.size = neg ? -Cint(n) : Cint(n)
        ccall((:mpfr_set_z_2exp, :libmpfr), Int32, (Ref{BigFloat}, Ref{BigInt}, Clong, Int32), t, big, e[i] + fj - 2cp.B, _RNDN)
        x = _el(A, r0 + i, c)
        mpfr_sub!(x, x, t)
    end
    nothing
end

"""
A[i, c] −= Σ_{p∈P} A[i,p] A[c,p] (i ∈ max(lo, rlo):M, c ∈ lo:lo+w−1, i ≥ c) with the residue scheme, wc columns at a time:
one dgemm per (prime, row block) in Julia tasks, then reconstruction per (column, 64 rows). Each element depends only on
its inputs and scales, so the result does not depend on rlo.
"""
function _ozaki_syrk!(A::AbstractMatrix, lo::Int, w::Int, P::UnitRange{Int}, ws::_CrtWS; rlo::Int=lo)
    M = size(A, 1); Mr = M - lo + 1; cp = ws.cp; nm = length(cp.p)
    rl = max(rlo - lo + 1, 1)
    rl > Mr && return nothing
    c0 = lo - 1
    todo = rl - 1 <= w ? (1:Mr) : [1:w; rl:Mr]               # rows w+1:rl−1 are not updated and not needed
    _gres!(ws.F, ws.e, (i, k) -> _el(A, c0 + i, P[k]), todo, length(P), cp, ws.rts)
    K = length(P); F = ws.F; G = ws.G; e = ws.e
    nt = Threads.nthreads()
    for j0 in 1:ws.wc:w
        j1 = min(j0 + ws.wc - 1, w); nc = j1 - j0 + 1
        i0 = max(j0, rl); nr = Mr - i0 + 1                    # rows i0:Mr (row 1 of G is row i0)
        rbs = collect(Iterators.partition(1:nr, cld(nr, _gemm_split(nm, nt, nr))))
        @sync for m in 1:nm, rb in rbs
            Threads.@spawn BLAS.gemm!('T', 'N', 1.0, view(F, 1:K, i0 - 1 .+ rb, m), view(F, 1:K, j0:j1, m), 0.0, view(G, rb, 1:nc, m))
        end
        njobs = sum(j -> cld(Mr - max(j, rl) + 1, _CRT_RB), j0:j1)
        next = Threads.Atomic{Int}(1)
        @sync for slot in 1:min(nt, njobs)
            Threads.@spawn _crt_recon_worker!(A, G, e, cp, ws.tmps[slot], next, j0, j1, Mr, c0, rl, i0)
        end
    end
    nothing
end

"""
Number of row blocks r (1..8, ≥ 64 rows each) for nm × r equal dgemm tasks on nt threads: the smallest r with
efficiency nm·r / (nt·⌈nm·r/nt⌉) ≥ 0.97, else the most efficient.
"""
function _gemm_split(nm::Int, nt::Int, nr::Int)
    rmax = clamp(nr ÷ 64, 1, 8)
    eff(r) = nm * r / (nt * cld(nm * r, nt))
    for r in 1:rmax
        eff(r) >= 0.97 && return r
    end
    argmax(eff, 1:rmax)
end

# Job q (1-based) is (column j, row block starting at i1): columns j0..j1 in order, each with blocks s, s+64, … (s = max(j, rl)).
function _crt_recon_worker!(A, G, e, cp, tmp::_CrtTmp, next, j0, j1, Mr, c0, rl::Int=1, i0::Int=j0)
    j = j0; base = 0; s = max(j, rl); nj = cld(Mr - s + 1, _CRT_RB)
    while true
        q0 = Threads.atomic_add!(next, _CRT_GRAB)
        for q in q0:q0+_CRT_GRAB-1
            while q > base + nj
                base += nj; j += 1
                j > j1 && return nothing
                s = max(j, rl); nj = cld(Mr - s + 1, _CRT_RB)
            end
            i1 = s + _CRT_RB * (q - base - 1); i2 = min(i1 + _CRT_RB - 1, Mr); gj = j - j0 + 1
            _crt_sub_block!(A, c0 + i1 - 1, c0 + j, i2 - i1 + 1, view(e, i1:i2), e[j], cp, tmp,
                            view(G, i1 - i0 + 1:i2 - i0 + 1, gj, :))
        end
    end
end
