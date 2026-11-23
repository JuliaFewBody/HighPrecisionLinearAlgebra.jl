# Packed storage for BigFloat matrices.
#
# SymPacked: the lower triangle of a symmetric matrix in one Vector{UInt64}; each element is one word of sign and
# exponent followed by its limbs. No MPFR struct is stored (it is built on the stack per operation, see `PElem`),
# so the storage has no absolute addresses and no objects for the GC to trace. About 1/3 of the memory of a dense
# Matrix{BigFloat}. `_el` returns a `PElem`, so the BigFloat kernels run unchanged on it.
# PackedMatrix: a small dense BigFloat matrix with whole MPFR structs stored contiguously (an element pointer is a
# valid mpfr_t). Used for scratch.

"""
    SymPacked(M; precision=precision(BigFloat))
    SymPacked(A::AbstractMatrix{BigFloat})

A symmetric BigFloat matrix that stores only the lower triangle. `P[i, j]` with i < j returns `P[j, i]`
(as a new BigFloat; slow). The Cholesky routines overwrite the lower triangle with the factor, after which `P`
is the factor L for `llt_solve!`. Different elements may be written from different threads.
"""
mutable struct SymPacked <: AbstractMatrix{BigFloat}
    M::Int
    prec::Int
    stride::Int              # words per element (1 + limbs)
    buf::Vector{UInt64}
end
function SymPacked(M::Int; precision::Integer=precision(BigFloat))
    stride = _sym_stride(precision)
    SymPacked(M, Int(precision), stride, _zerobuf!(Vector{UInt64}(undef, stride * (M * (M + 1) ÷ 2))))
end
function SymPacked(A::AbstractMatrix{BigFloat})
    M = size(A, 1)
    P = SymPacked(M; precision=_prec(A))
    _foreach_tri_cols(M) do cols
        for j in cols, i in j:M; mpfr_set!(_el(P, i, j), _el(A, i, j)); end
    end
    P
end
_nlimbs(prec::Integer) = Int(ccall((:mpfr_custom_get_size, :libmpfr), Csize_t, (Clong,), prec)) ÷ 8
_sym_stride(prec::Integer) = 1 + _nlimbs(prec)
"Bytes per element of a SymPacked at the given precision."
sympacked_elbytes(prec::Integer=precision(BigFloat)) = 8 * _sym_stride(prec)

"Fill with zeros (+0 for SymPacked); in parallel for large buffers so pages land on the threads that use them."
function _zerobuf!(buf::Vector{UInt64})
    n = length(buf); base = pointer(buf)
    GC.@preserve buf begin
        if n < 1 << 18 || Threads.nthreads() == 1
            fill!(buf, 0)
        else
            @sync for c in Iterators.partition(0:n-1, cld(n, 4Threads.nthreads()))
                Threads.@spawn ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t), base + 8 * $(first(c)), 0, 8 * $(length(c)))
            end
        end
    end
    buf
end

# 0-based index of element (i, j), i ≥ j. Column j (length M − j + 1) is contiguous.
@inline _pidx(M, i, j) = (j - 1) * M - ((j - 1) * (j - 2)) ÷ 2 + (i - j)
@inline function _el(P::SymPacked, i, j)
    i < j && ((i, j) = (j, i))
    PElem(pointer(P.buf) + 8 * P.stride * _pidx(P.M, i, j), P.prec)
end
_prec(P::SymPacked) = P.prec

Base.size(P::SymPacked) = (P.M, P.M)
Base.IndexStyle(::Type{SymPacked}) = IndexCartesian()
function Base.getindex(P::SymPacked, i::Int, j::Int)
    @boundscheck checkbounds(P, i, j)
    x = BigFloat(; precision=P.prec)
    GC.@preserve P mpfr_set!(x, _el(P, i, j))
end
function Base.setindex!(P::SymPacked, v, i::Int, j::Int)
    @boundscheck checkbounds(P, i, j)
    x = v isa BigFloat ? v : BigFloat(v; precision=P.prec)
    GC.@preserve P mpfr_set!(_el(P, i, j), x)
    v
end
Base.copy(P::SymPacked) = SymPacked(P.M, P.prec, P.stride, _pcopyto!(Vector{UInt64}(undef, length(P.buf)), P.buf))
function Base.copyto!(P::SymPacked, Q::SymPacked)
    (P.M == Q.M && P.prec == Q.prec) || throw(DimensionMismatch("size or precision differs"))
    _pcopyto!(P.buf, Q.buf); P
end
"Copy in parallel chunks (pages of the new buffer are placed on the NUMA node of the thread that writes them)."
function _pcopyto!(dst::Vector{UInt64}, src::Vector{UInt64})
    n = length(src); nt = Threads.nthreads()
    (nt == 1 || n < (1 << 22)) && return copyto!(dst, src)
    @sync for c in Iterators.partition(1:n, cld(n, 4nt))
        Threads.@spawn copyto!(dst, first(c), src, first(c), length(c))
    end
    dst
end
fresh(P::SymPacked) = copy(P)
LinearAlgebra.tril!(P::SymPacked) = P
_transpose_block(P::SymPacked, rows, cols) = PElem[_el(P, j, i) for i in cols, j in rows]

"""
    copy_leading!(A, L) -> A

Copy the lower triangle of L (m×m) into the leading m×m block of A (M ≥ m, same precision); the rest of A is untouched.
Used to restart a Cholesky factorization: `copy_leading!(A, L); chol_lower_ozaki!(A; done=m)`.
"""
function copy_leading!(A::SymPacked, L::SymPacked)
    m = L.M
    (A.M >= m && A.prec == L.prec) || throw(DimensionMismatch("size or precision differs"))
    st = A.stride
    GC.@preserve A L begin
        pa = pointer(A.buf); pl = pointer(L.buf)
        _foreach_tri_cols(m) do cols
            for j in cols
                unsafe_copyto!(pa + 8 * st * _pidx(A.M, j, j), pl + 8 * st * _pidx(m, j, j), st * (m - j + 1))
            end
        end
    end
    A
end
function copy_leading!(A::AbstractMatrix{T}, L::AbstractMatrix{T}) where {T}
    m = size(L, 1)
    size(A, 1) >= m || throw(DimensionMismatch("A is smaller than L"))
    @inbounds for j in 1:m, i in j:m; A[i, j] = L[i, j]; end
    A
end
"Dense copy (both triangles, distinct elements)."
function Base.Matrix(P::SymPacked)
    A = Matrix{BigFloat}(undef, P.M, P.M)
    _foreach_tri_cols(P.M) do cols
        for j in cols, i in j:P.M
            A[i, j] = P[i, j]
            i > j && (A[j, i] = P[i, j])
        end
    end
    A
end

"""
    symv!(y, A::SymPacked, x) -> y

y = A x reading the packed lower triangle once, column by column. The columns are split into 64 blocks of equal work;
block b accumulates z_b[i] = Σ_{k ∈ b} A[i,k] x_k and y_i = Σ_b z_b[i] in a fixed order, so the result does not
depend on the number of threads. A dense symmetric `Matrix{BigFloat}` takes the same path (bit-identical).
"""
function symv!(y::AbstractVector{BigFloat}, A::Union{SymPacked,Matrix{BigFloat}}, x::AbstractVector{BigFloat})
    M = size(A, 1); p = _prec(A)
    nb = min(_SYMV_NBLK, M)
    cuts = [_wsplit(1, M, M, nb, b) for b in 1:nb]
    parts = [PackedMatrix(c2 < c1 ? 0 : M - c1 + 1, c2 < c1 ? 0 : 1; precision=p, malloc=true) for (c1, c2) in cuts]
    try
        next = Threads.Atomic{Int}(1)
        GC.@preserve A x parts @sync for _ in 1:min(Threads.nthreads(), nb)
            Threads.@spawn while (b = Threads.atomic_add!(next, 1)) <= nb
                c1, c2 = cuts[b]
                c2 < c1 || _symv_cols!(parts[b], A, x, c1, c2, p)
            end
        end
        GC.@preserve parts _foreach_chunk(1:M) do rows
            for i in rows
                acc = BigFloat(0; precision=p)
                for b in 1:nb
                    c1, c2 = cuts[b]
                    (c2 >= c1 && c1 <= i) && mpfr_add!(acc, acc, _el(parts[b], i - c1 + 1, 1))
                end
                y[i] = acc
            end
        end
    finally
        foreach(free!, parts)
    end
    y
end
const _SYMV_NBLK = 64
# z[i − c1 + 1] = Σ_{k ∈ c1:c2} A[i,k] x_k. A safepoint per column lets a GC started by another thread proceed.
function _symv_cols!(z, A, x, c1, c2, p)
    M = size(A, 1)
    s = BigFloat(0; precision=p)
    @inbounds for j in c1:c2
        GC.safepoint()
        xj = x[j]
        mpfr_mul!(s, _el(A, j, j), xj)
        for i in j+1:M
            a = _el(A, i, j); zi = _el(z, i - c1 + 1, 1)
            mpfr_fma!(zi, a, xj, zi)                       # row i, term k = j
            mpfr_fma!(s, a, x[i], s)                       # row j, term k = i
        end
        zj = _el(z, j - c1 + 1, 1); mpfr_add!(zj, zj, s)
    end
    z
end

"Write H − σS into A (all SymPacked; A's storage is reused). No allocation, columns in parallel."
function shifted_lower!(A::SymPacked, H::SymPacked, S::SymPacked, σ::BigFloat)
    M = H.M
    (A.M == M && A.prec == H.prec) || throw(DimensionMismatch("size or precision differs"))
    GC.@preserve H S A _foreach_tri_cols(M) do cols
        for j in cols, i in j:M
            a = _el(A, i, j)
            mpfr_mul!(a, σ, _el(S, i, j)); mpfr_sub!(a, _el(H, i, j), a)
        end
    end
    A
end
shifted_lower(H::SymPacked, S::SymPacked, σ::BigFloat) = shifted_lower!(SymPacked(H.M; precision=H.prec), H, S, σ)

"Scale H and S in place by D = diag(S)^{-1/2} (H_ij ← D_i H_ij D_j). Return D."
function diagonal_scale!(H::SymPacked, S::SymPacked)
    M = S.M
    D = [inv(sqrt(S[i, i])) for i in 1:M]
    GC.@preserve H S _foreach_tri_cols(M) do cols
        for j in cols, i in j:M, P in (S, H)
            x = _el(P, i, j); mpfr_mul!(x, D[i], x); mpfr_mul!(x, x, D[j])
        end
    end
    D
end

"""
    PackedMatrix(m, n; precision=precision(BigFloat), malloc=false)

A dense BigFloat matrix (column major) with the MPFR structs and limbs in one `Vector{UInt64}`; initialized to +0.
`_el(A, i, j)` is a `Ptr{BigFloat}` that can be passed to the `mpfr_*!` kernels. With `malloc=true` the storage is
outside the GC and must be released with `free!`.
"""
mutable struct PackedMatrix <: AbstractMatrix{BigFloat}
    m::Int
    n::Int
    prec::Int
    stride::Int
    buf::Vector{UInt64}
end
function PackedMatrix(m::Int, n::Int; precision::Integer=precision(BigFloat), malloc::Bool=false)
    stride = 4 + _nlimbs(precision); N = stride * m * n
    buf = !malloc || N == 0 ? Vector{UInt64}(undef, N) :
          ((p = Libc.malloc(8N)) == C_NULL ? throw(OutOfMemoryError()) : unsafe_wrap(Array, Ptr{UInt64}(p), N; own=false))
    PackedMatrix(m, n, Int(precision), stride, _initbuf!(buf, stride, Int(precision)))
end
"Release the storage of a `PackedMatrix(…; malloc=true)` (P becomes 0×0)."
free!(P::PackedMatrix) = (isempty(P.buf) || Libc.free(pointer(P.buf)); P.buf = UInt64[]; P.m = 0; P.n = 0; nothing)
"Write the MPFR headers (precision, +0, d pointing at the element's own limbs) and zero the limbs."
function _initbuf!(buf::Vector{UInt64}, stride::Int, prec::Int)
    n = length(buf) ÷ stride; base = pointer(buf)
    GC.@preserve buf begin
        if n < 1 << 15 || Threads.nthreads() == 1
            _initbuf_range!(base, stride, prec, 0, n - 1)
        else
            @sync for c in Iterators.partition(0:n-1, cld(n, 4Threads.nthreads()))
                Threads.@spawn _initbuf_range!(base, stride, prec, $(first(c)), $(last(c)))
            end
        end
    end
    buf
end
function _initbuf_range!(base::Ptr{UInt64}, stride::Int, prec::Int, lo::Int, hi::Int)
    for k in lo:hi
        p = base + 8 * stride * k
        unsafe_store!(Ptr{Clong}(p), Clong(prec))
        unsafe_store!(Ptr{UInt64}(p + 8), UInt64(1))             # sign = +1 (Cint) and padding
        unsafe_store!(Ptr{Clong}(p + 16), _EZERO)
        unsafe_store!(Ptr{Ptr{UInt64}}(p + 24), Ptr{UInt64}(p + 32))
        for l in 5:stride; unsafe_store!(Ptr{UInt64}(p), UInt64(0), l); end
    end
end
@inline _el(P::PackedMatrix, i, j) = Ptr{BigFloat}(pointer(P.buf) + 8 * P.stride * ((j - 1) * P.m + (i - 1)))
_prec(P::PackedMatrix) = P.prec
Base.size(P::PackedMatrix) = (P.m, P.n)
Base.IndexStyle(::Type{PackedMatrix}) = IndexCartesian()
function Base.getindex(P::PackedMatrix, i::Int, j::Int)
    @boundscheck checkbounds(P, i, j)
    x = BigFloat(; precision=P.prec)
    GC.@preserve P mpfr_set!(x, _el(P, i, j))
end
function Base.setindex!(P::PackedMatrix, v, i::Int, j::Int)
    @boundscheck checkbounds(P, i, j)
    x = v isa BigFloat ? v : BigFloat(v; precision=P.prec)
    GC.@preserve P mpfr_set!(_el(P, i, j), x)
    v
end
function Base.copy(P::PackedMatrix)                     # the copy's d pointers must point at its own limbs
    Q = PackedMatrix(P.m, P.n, P.prec, P.stride, copy(P.buf))
    GC.@preserve Q for k in 0:length(Q.buf)÷Q.stride-1
        p = pointer(Q.buf) + 8 * Q.stride * k
        unsafe_store!(Ptr{Ptr{UInt64}}(p + 24), Ptr{UInt64}(p + 32))
    end
    Q
end
