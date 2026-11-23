# In-place MPFR operations on BigFloat (no allocation).
#
# Ordinary BigFloat arithmetic allocates a new object per operation, and the GC becomes the
# bottleneck of multithreaded inner loops. The kernels below write into an existing number.
# `mpfr_fma!` rounds once, so it is not bit-identical to a multiply followed by an add.
# A matrix that is updated in place must hold distinct BigFloat objects (see `fresh`).
#
# Arguments may be a BigFloat, a `Ptr{BigFloat}` (an element of a `PackedMatrix`), or a `PElem`
# (an element of a `SymPacked`). Kernels fetch elements with `_el(A, i, j)`, so the same code
# runs on dense matrices and on the packed formats.

"""
An element of a `SymPacked`: `p` points to one word holding sign and exponent, followed by the limbs.
The MPFR struct is built on the stack for each call, so the storage holds no absolute addresses.
"""
struct PElem
    p::Ptr{UInt64}
    prec::Int
end
const MPFRArg = Union{BigFloat,Ptr{BigFloat},PElem}
@inline _p(x::BigFloat) = Base.unsafe_convert(Ref{BigFloat}, Base.cconvert(Ref{BigFloat}, x))
@inline _p(x::Ptr{BigFloat}) = x

# Same layout as MPFR's __mpfr_struct.
struct _MPFRHdr
    prec::Clong
    sign::Cint
    exp::Clong
    d::Ptr{UInt64}
end
# Sign and exponent in one word: (u << 1) | negative, with u = 0 (zero), 1 (NaN), 2 (Inf), or exponent + _EOFF.
# The zero word is +0, so zero-filled storage is a zero matrix.
const _EOFF = (Int64(1) << 61) + 3
const _EZERO = Base.MPFR.mpfr_special_exponent_zero
const _ENAN = Base.MPFR.mpfr_special_exponent_nan
const _EINF = Base.MPFR.mpfr_special_exponent_inf
@inline function _dec(w::UInt64)
    u = Int64(w >> 1); s = (w & 1) == 1 ? Cint(-1) : Cint(1)
    ex = u >= 3 ? u - _EOFF : u == 0 ? _EZERO : u == 1 ? _ENAN : _EINF
    s, ex
end
@inline function _enc(sign::Cint, ex::Clong)
    u = ex == _EZERO ? 0 : ex == _ENAN ? 1 : ex == _EINF ? 2 : ex + _EOFF
    (ex == _EZERO || ex == _ENAN || ex == _EINF || abs(ex) < (Int64(1) << 61)) || throw(OverflowError("exponent $ex out of range"))
    (UInt64(u) << 1) | UInt64(sign < 0)
end
@inline _hdr(x::PElem) = ((s, ex) = _dec(unsafe_load(x.p)); _MPFRHdr(x.prec, s, ex, x.p + 8))
@inline _mkref(x::PElem) = Ref(_hdr(x))
@inline _mkref(x) = nothing
@inline _ptr(x::PElem, r) = Ptr{BigFloat}(Base.unsafe_convert(Ptr{_MPFRHdr}, r))
@inline _ptr(x, ::Nothing) = _p(x)
@inline _alias(x::PElem, z::PElem) = x.p == z.p
@inline _alias(x, z) = false
@inline _wb!(z::PElem, r) = (h = r[]; unsafe_store!(z.p, _enc(h.sign, h.exp)); nothing)
@inline _wb!(z, r) = nothing

"""
Call `f(pz, px...)` with pointers to MPFR structs (`z` is the output). An input that is the same `PElem`
as `z` gets z's struct, so MPFR sees the aliasing. A `PElem` output has its sign and exponent written back.
"""
@inline function _mpcall(f::F, z, xs::Vararg{Any,N}) where {F,N}
    rz = _mkref(z); rs = map(_mkref, xs)
    GC.@preserve z xs rz rs begin
        pz = _ptr(z, rz)
        ps = ntuple(i -> _alias(xs[i], z) ? pz : _ptr(xs[i], rs[i]), Val(N))
        f(pz, ps...)
        _wb!(z, rz)
    end
    z
end
@inline _mpread(f::F, x) where {F} = (r = _mkref(x); GC.@preserve x r f(_ptr(x, r)))
@inline _el(A::AbstractMatrix{BigFloat}, i, j) = @inbounds A[i, j]
"Precision of the elements of A (used for scratch numbers)."
_prec(A::AbstractMatrix{BigFloat}) = isempty(A) ? precision(BigFloat) : precision(@inbounds A[1, 1])

const _RNDN = Int32(0)                 # round to nearest
const _PB = Ptr{BigFloat}
mpfr_add!(z::MPFRArg, x::MPFRArg, y::MPFRArg) =
    _mpcall((pz, px, py) -> ccall((:mpfr_add, :libmpfr), Int32, (_PB, _PB, _PB, Int32), pz, px, py, _RNDN), z, x, y)
mpfr_sub!(z::MPFRArg, x::MPFRArg, y::MPFRArg) =
    _mpcall((pz, px, py) -> ccall((:mpfr_sub, :libmpfr), Int32, (_PB, _PB, _PB, Int32), pz, px, py, _RNDN), z, x, y)
mpfr_mul!(z::MPFRArg, x::MPFRArg, y::MPFRArg) =
    _mpcall((pz, px, py) -> ccall((:mpfr_mul, :libmpfr), Int32, (_PB, _PB, _PB, Int32), pz, px, py, _RNDN), z, x, y)
mpfr_div!(z::MPFRArg, x::MPFRArg, y::MPFRArg) =
    _mpcall((pz, px, py) -> ccall((:mpfr_div, :libmpfr), Int32, (_PB, _PB, _PB, Int32), pz, px, py, _RNDN), z, x, y)
"z = x*y + w with a single rounding."
mpfr_fma!(z::MPFRArg, x::MPFRArg, y::MPFRArg, w::MPFRArg) =
    _mpcall((pz, px, py, pw) -> ccall((:mpfr_fma, :libmpfr), Int32, (_PB, _PB, _PB, _PB, Int32), pz, px, py, pw, _RNDN), z, x, y, w)
mpfr_neg!(z::MPFRArg, x::MPFRArg) = _mpcall((pz, px) -> ccall((:mpfr_neg, :libmpfr), Int32, (_PB, _PB, Int32), pz, px, _RNDN), z, x)
mpfr_set!(z::MPFRArg, x::MPFRArg) = _mpcall((pz, px) -> ccall((:mpfr_set, :libmpfr), Int32, (_PB, _PB, Int32), pz, px, _RNDN), z, x)
mpfr_sqrt!(z::MPFRArg, x::MPFRArg) = _mpcall((pz, px) -> ccall((:mpfr_sqrt, :libmpfr), Int32, (_PB, _PB, Int32), pz, px, _RNDN), z, x)
mpfr_mul_si!(z::MPFRArg, x::MPFRArg, n::Integer) =
    _mpcall((pz, px) -> ccall((:mpfr_mul_si, :libmpfr), Int32, (_PB, _PB, Clong, Int32), pz, px, n, _RNDN), z, x)
mpfr_mul_2si!(z::MPFRArg, x::MPFRArg, n::Integer) =
    _mpcall((pz, px) -> ccall((:mpfr_mul_2si, :libmpfr), Int32, (_PB, _PB, Clong, Int32), pz, px, n, _RNDN), z, x)
mpfr_exp!(z::MPFRArg, x::MPFRArg) = _mpcall((pz, px) -> ccall((:mpfr_exp, :libmpfr), Int32, (_PB, _PB, Int32), pz, px, _RNDN), z, x)
_sgn(x::MPFRArg) = _mpread(px -> ccall((:mpfr_sgn, :libmpfr), Int32, (_PB,), px), x)
_getd(x::MPFRArg) = _mpread(px -> ccall((:mpfr_get_d, :libmpfr), Float64, (_PB, Int32), px, _RNDN), x)
"Pointer to the MPFR struct of a BigFloat (use inside `GC.@preserve x`)."
mpfr_ptr(x::BigFloat) = _p(x)

"z = b^n (n ≥ 1) by binary powering; x is scratch (mpfr_pow_si allocates internally)."
function mpfr_powi!(z::BigFloat, b::BigFloat, n::Int, x::BigFloat)
    mpfr_set!(x, b); first = true
    while true
        if isodd(n)
            first ? mpfr_set!(z, x) : mpfr_mul!(z, z, x); first = false
        end
        n >>= 1
        n == 0 && return z
        mpfr_mul!(x, x, x)
    end
end
