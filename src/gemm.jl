# General matrix product C ← C ± A B with the Ozaki scheme (residues modulo primes),
# and a plain threaded extended-precision version. LU, QR and the eigen/SVD refinement are built on these.
#
# Error: only the rounding of the inputs to ⌈log2 K⌉ + 8 extra bits (relative to the row/column maxima) is inexact;
# the integer sums are exact and the result is rounded once.

const _CRTP_CACHE = Dict{Tuple{Int,Int},_CrtParams}()
const _CRTP_LOCK = ReentrantLock()
"Residue-scheme constants, cached by (bits, K rounded up to a power of 2)."
_crt_params(bits::Int, K::Int) = (K2 = nextpow(2, max(K, 2)); lock(() -> get!(() -> _CrtParams(bits, K2), _CRTP_CACHE, (bits, K2)), _CRTP_LOCK))

"""
    gemm_ozaki!(C, A, B; sub=false, wc=256, scheme=:crt) -> C

C ← C + A B (C − A B with `sub=true`) with the Ozaki scheme. A is M×K, B is K×N; views are fine.
Both BigFloat and MultiFloat use residues modulo primes (`scheme=:slice` selects integer slices for MultiFloat). The elements of C must be distinct objects.
Each element's error is at most max_k|A_ik|·max_k|B_kj|·2^{−bits−6}. B is processed wc columns at a time.
"""
function gemm_ozaki!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T};
                     sub::Bool=false, wc::Int=256, scheme::Symbol=:crt) where {T<:Union{BigFloat,MultiFloat}}
    T <: MultiFloat && scheme === :slice && return _gemm_slice!(C, A, B; sub, wc)
    M1, K = size(A); K2, N = size(B)
    K == K2 && size(C) == (M1, N) || throw(DimensionMismatch("C $(size(C)) ≠ A $(size(A)) × B $(size(B))"))
    (M1 == 0 || N == 0 || K == 0) && return C
    bits = _nbits(C); cp = _crt_params(bits, K); nm = length(cp.p); nt = Threads.nthreads()
    rts = [_ResTmp(cp, K) for _ in 1:nt]; tmps = [_CrtTmp(cp, bits) for _ in 1:nt]
    wc = min(wc, N)
    FA = _calloc_array(Float64, K, M1, nm); FB = _calloc_array(Float64, K, wc, nm); G = _calloc_array(Float64, M1, wc, nm)
    eA = zeros(Int, M1); eB = zeros(Int, wc)
    try
        _with_blas_threads(1) do
            _gres!(FA, eA, (i, k) -> @inbounds(A[i, k]), 1:M1, K, cp, rts)
            rbs = collect(Iterators.partition(1:M1, cld(M1, _gemm_split(nm, nt, M1))))
            for j0 in 1:wc:N
                j1 = min(j0 + wc - 1, N); nc = j1 - j0 + 1
                _gres!(FB, eB, (j, k) -> @inbounds(B[k, j0 + j - 1]), 1:nc, K, cp, rts)
                if !sub                                    # reconstruction subtracts, so flip B's residues to add
                    @inbounds for m in 1:nm, j in 1:nc
                        @simd for k in 1:K; FB[k, j, m] = -FB[k, j, m]; end
                    end
                end
                @sync for m in 1:nm, rb in rbs
                    Threads.@spawn BLAS.gemm!('T', 'N', 1.0, view(FA, :, rb, m), view(FB, :, 1:nc, m), 0.0, view(G, rb, 1:nc, m))
                end
                nrj = cld(M1, _CRT_RB); njobs = nc * nrj
                next = Threads.Atomic{Int}(1)
                @sync for slot in 1:min(nt, njobs)
                    Threads.@spawn begin
                        tmp = tmps[slot]
                        while (q = Threads.atomic_add!(next, 1)) <= njobs
                            j = (q - 1) ÷ nrj + 1; i1 = ((q - 1) % nrj) * _CRT_RB + 1; i2 = min(i1 + _CRT_RB - 1, M1)
                            _crt_sub_block!(C, i1 - 1, j0 + j - 1, i2 - i1 + 1, view(eA, i1:i2), eB[j], cp, tmp, view(G, i1:i2, j, :))
                        end
                    end
                end
            end
        end
    finally
        _free_array!(FA); _free_array!(FB); _free_array!(G)
    end
    C
end
"Slice scheme for MultiFloat (NS(NS+1)/2 dgemms; kept for comparison with the residue scheme)."
function _gemm_slice!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T};
                      sub::Bool=false, wc::Int=256) where {T<:MultiFloat}
    M1, K = size(A); K2, N = size(B)
    K == K2 && size(C) == (M1, N) || throw(DimensionMismatch("C $(size(C)) ≠ A $(size(A)) × B $(size(B))"))
    (M1 == 0 || N == 0 || K == 0) && return C
    β, NS = oz_params(nbits_of(T), K)
    wc = min(wc, N)
    FA = zeros(NS * K, M1); eA = zeros(Int, M1)
    RB = zeros(NS * K, wc); FB = zeros(NS * K, wc); eB = zeros(Int, wc)
    G = [zeros(M1, wc) for _ in 1:NS]
    _with_blas_threads(max(1, Threads.nthreads())) do
        ozaki_slice!(FA, nothing, eA, _transpose_block(A, 1:M1, 1:K), K, NS, β)
        for j0 in 1:wc:N
            j1 = min(j0 + wc - 1, N); nc = j1 - j0 + 1
            ozaki_slice!(FB, RB, eB, view(B, :, j0:j1), K, NS, β)
            sub || (RB .= .-RB)
            ozaki_gemms!(G, FA, RB, K, M1, nc, NS)
            Threads.@threads for j in 1:nc
                for i in 1:M1
                    _ozaki_sub!(C, i, j0 + j - 1, G, i, j, eA[i], eB[j], NS, β, nothing)
                end
            end
        end
    end
    C
end

"""
    gemm_mt!(C, A, B; sub=false) -> C

C ← C ± A B in extended precision (BigFloat via mpfr_fma, no allocation). Threads split the columns of C, or the rows,
or (small C, long inner products) the inner dimension with partial sums.
"""
function gemm_mt!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}; sub::Bool=false) where {T}
    M1, K = size(A); N = size(B, 2)
    size(B, 1) == K && size(C) == (M1, N) || throw(DimensionMismatch("C $(size(C)) ≠ A $(size(A)) × B $(size(B))"))
    nt = Threads.nthreads()
    if nt == 1 || M1 * N * K < _MIN_WORK
        _gemm_cols!(C, A, B, 1:N, sub)
    elseif N >= nt
        _foreach_chunk(cols -> _gemm_cols!(C, A, B, cols, sub), 1:N)
    elseif M1 >= 4nt
        _foreach_chunk(rows -> _gemm_cols!(view(C, rows, :), view(A, rows, :), B, 1:N, sub), 1:M1)
    else
        parts = collect(Iterators.partition(1:K, cld(K, nt)))
        Ps = [_zeros_like(C, M1, N) for _ in parts]
        @sync for (q, ks) in enumerate(parts)
            Threads.@spawn _gemm_cols!(Ps[q], view(A, :, ks), view(B, ks, :), 1:N, false)
        end
        for P in Ps; _addto!(C, P, sub); end
    end
    C
end
function _gemm_cols!(C, A, B, cols, sub)
    @inbounds for j in cols, k in axes(A, 2)
        b = sub ? -B[k, j] : B[k, j]
        for i in axes(A, 1); C[i, j] += A[i, k] * b; end
    end
end
function _gemm_cols!(C::AbstractMatrix{BigFloat}, A::AbstractMatrix{BigFloat}, B::AbstractMatrix{BigFloat}, cols, sub)
    nb = BigFloat(; precision=_prec(C))
    @inbounds for j in cols, k in axes(A, 2)
        sub ? mpfr_neg!(nb, B[k, j]) : mpfr_set!(nb, B[k, j])
        for i in axes(A, 1); c = C[i, j]; mpfr_fma!(c, A[i, k], nb, c); end
    end
end
_addto!(C, P, sub) = (sub ? (C .-= P) : (C .+= P); nothing)
function _addto!(C::AbstractMatrix{BigFloat}, P::AbstractMatrix{BigFloat}, sub)
    @inbounds for j in axes(C, 2), i in axes(C, 1)
        c = C[i, j]; sub ? mpfr_sub!(c, c, P[i, j]) : mpfr_add!(c, c, P[i, j])
    end
    nothing
end

const _OZ_KMIN = 32       # reconstruction costs about 15 fma per element, so short inner products stay in gemm_mt!
"C ← C ± A B: `gemm_ozaki!` for supported types with long inner products, otherwise `gemm_mt!`."
function _gemm!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}; sub::Bool=false) where {T}
    if (T === BigFloat || T <: MultiFloat) && size(A, 2) >= _OZ_KMIN && size(A, 1) * size(B, 2) >= 1024
        gemm_ozaki!(C, A, B; sub)
    else
        gemm_mt!(C, A, B; sub)
    end
end
"A B as a new matrix (Ozaki scheme when it pays)."
mul_ozaki(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T} = _gemm!(_zeros_like(A, size(A, 1), size(B, 2)), A, B)
_zeros_like(A::AbstractMatrix{T}, m, n) where {T} = zeros(T, m, n)
_zeros_like(A::AbstractMatrix{BigFloat}, m, n) = (p = _prec(A); BigFloat[BigFloat(0; precision=p) for _ in 1:m, _ in 1:n])
