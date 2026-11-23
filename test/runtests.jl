using Test
using HighPrecisionLinearAlgebra
using LinearAlgebra
using MultiFloats
using Random
using Serialization
using GenericLinearAlgebra

const HP = HighPrecisionLinearAlgebra
Random.seed!(1)
fro(X) = sqrt(sum(x -> Float64(x)^2, X))

@testset "Cholesky and triangular solves (Float64x4, BigFloat)" begin
    for (U, M, tol) in ((Float64x4, 200, 1e-60), (BigFloat, 200, 1e-120))
        U === BigFloat && setprecision(BigFloat, 424)
        X = U.(randn(M, M)) ./ M; S = X'X + I; H = U.(randn(M, M)); H = H + H'
        L1 = fresh(S); @test chol_lower!(L1); tril!(L1)
        L3 = fresh(S); @test chol_lower_persist!(L3); tril!(L3)
        @test L3 == L1
        nblas = BLAS.get_num_threads()
        L2 = fresh(S); @test chol_lower_ozaki!(L2; nb=64); tril!(L2)
        @test BLAS.get_num_threads() == nblas
        @test maximum(abs, L1 - L2) <= tol * maximum(abs, L1)
        L4 = fresh(S); @test chol_lower_ozaki!(L4; nb=128, leaf=8, ozmin=8); tril!(L4)
        @test maximum(abs, L1 - L4) <= tol * maximum(abs, L1)
        @test !chol_lower_ozaki!(fresh(H)) && !chol_lower_persist!(fresh(H))
        Y1 = fresh(H); ltri_solve!(Y1, L1); Y2 = fresh(H); trsm_lower_ozaki!(Y2, L2; nb=64)
        @test maximum(abs, Y1 - Y2) <= tol * maximum(abs, Y1)
        b = U.(randn(M)); x = fresh(b); llt_solve!(x, L1)
        @test maximum(abs, S * x - b) <= 1e3 * tol * maximum(abs, b) * M
        y = similar(b); symv!(y, H, b)
        @test maximum(abs, y - H * b) <= 10 * tol * maximum(abs, H * b) * M
    end
end

@testset "slice reconstruction is exact" begin
    setprecision(BigFloat, 320) do
        β, NS = HP.oz_params(320, 128); tmp = HP._ozaki_tmp(BigFloat, β, NS)
        for trial in 1:200
            g = [Float64(rand(-2^52:2^52)) for _ in 1:NS]
            trial % 3 == 0 && (g[2:end] .= 0)
            trial % 5 == 0 && (g .= -abs.(g))
            G = [fill(x, 1, 1) for x in g]
            x0 = BigFloat(rand()); A = fill(x0 + 0, 1, 1)
            ei, fj = rand(-40:40), rand(-40:40)
            HP._ozaki_sub!(A, 1, 1, G, 1, 1, ei, fj, NS, β, tmp)
            exact = sum(BigInt(Int64(g[d-1])) * BigInt(2)^(β * (NS + 1 - d)) for d in 2:NS+1)
            ref = BigFloat(setprecision(() -> x0 - BigFloat(exact) * BigFloat(2)^(ei + fj - β * (NS + 1)), BigFloat, 2000); precision=320)
            @test A[1, 1] == ref
        end
        K = 5; X = BigFloat.(randn(K, 3)) .* BigFloat(10) .^ rand(-30:30, K, 3); X[:, 3] .= 0
        F = zeros(NS * K, 3); R = zeros(NS * K, 3); e = zeros(Int, 3)
        HP.ozaki_slice!(F, R, e, X, K, NS, β)
        @test e[3] == typemin(Int) && all(iszero, F[:, 3])
        for i in 1:2, k in 1:K
            N = sum(BigInt(Int64(F[(s-1)*K+k, i])) * BigInt(2)^(β * (NS - s)) for s in 1:NS)
            @test N == round(BigInt, X[k, i] * BigFloat(2)^(β * NS - e[i]))
            @test all(F[(s-1)*K+k, i] == R[(NS-s)*K+k, i] for s in 1:NS)
        end
    end
end

@testset "residue scheme: Cholesky, digits and reconstruction" begin
    setprecision(BigFloat, 320) do
        M = 260
        X = BigFloat.(randn(M, M)) ./ M; S = fresh(X'X + I)
        L1 = fresh(S); chol_lower!(L1); tril!(L1)
        for nb in (64, 128)
            Ls = fresh(S); @test chol_lower_ozaki!(Ls; nb=nb, scheme=:slice); tril!(Ls)
            Lc = fresh(S); @test chol_lower_ozaki!(Lc; nb=nb, scheme=:crt); tril!(Lc)
            @test maximum(abs, Lc - L1) <= 1e-94 * maximum(abs, L1)
            Lw = fresh(S); chol_lower_ozaki!(Lw; nb=nb, wc=32); tril!(Lw)
            @test Lw == Lc                                         # exact integer sums: independent of wc
        end
        Lp = fresh(SymPacked(S)); chol_lower_ozaki!(Lp); Ld = fresh(S); chol_lower_ozaki!(Ld)
        @test all(Ld[i, j] == Lp[i, j] for j in 1:M for i in j:M)
        cp = HP._CrtParams(320, 128); tmp = HP._CrtTmp(cp, 320); nm = length(cp.p)
        @test prod(BigInt.(cp.p)) > 4 * 128 * big(2)^(2cp.B)
        @test 128 * ((maximum(cp.p) - 1) / 2)^2 < 2.0^53
        for trial in 1:300
            Z = rand(-(big(2)^(2cp.B + 7) - 1):(big(2)^(2cp.B + 7) - 1)) >> rand(0:650)
            trial % 7 == 0 && (Z = big(0)); trial % 11 == 0 && (Z = big(-1))
            for m in 1:nm; tmp.zb[1, m] = Float64(mod(Z, Int(cp.p[m])) - rand(0:3) * Int(cp.p[m])); end
            x0 = BigFloat(rand()); A = fill(x0 + 0, 1, 1); ei, fj = rand(-40:40), rand(-40:40)
            HP._crt_sub_block!(A, 0, 1, 1, [ei], fj, cp, tmp)
            ref = setprecision(() -> x0 - BigFloat(Z) * BigFloat(2)^(ei + fj - 2cp.B), BigFloat, 4000)
            @test abs(A[1, 1] - ref) <= 2eps(BigFloat(abs(ref); precision=320))
        end
    end
    for bits in (320, 424, 200)                                    # digits read directly from the MPFR mantissa
        setprecision(BigFloat, bits) do
            cp = HP._CrtParams(bits, 256); pcs = zeros(cp.nin); ok = true
            for trial in 1:1000
                e = rand(-60:60); x = BigFloat(rand() * (rand(Bool) ? 1 : -1)) * BigFloat(2)^(e - rand(0:300))
                trial % 13 == 0 && (x = BigFloat(0))
                trial % 17 == 0 && (x = prevfloat(BigFloat(2)^e))
                sg = HP._crt_pieces!(pcs, x, cp.B, e, cp.nin)
                N = sum(BigInt(pcs[q]) << (16(q - 1)) for q in 1:cp.nin)
                ok &= abs(BigFloat(N; precision=4000) - BigFloat(abs(x) * BigFloat(2)^(cp.B - e); precision=4000)) <= 0.5 && (x == 0 || sg == sign(x))
            end
            @test ok
        end
    end
    for (bits, K) in ((320, 128), (424, 256), (424, 100), (256, 64))   # nq mod 4 = 0, 1, 2, 3
        setprecision(BigFloat, bits) do
            cp = HP._CrtParams(bits, K); tmp = HP._CrtTmp(cp, bits); nm = length(cp.p); N = 64
            Zs = [rand(-(big(2)^(2cp.B + 7) - 1):(big(2)^(2cp.B + 7) - 1)) >> rand(0:2cp.B) for _ in 1:N]
            Zs[5] = 0; Zs[6] = -1; Zs[7] = 1
            for r in 1:N, m in 1:nm; tmp.zb[r, m] = Float64(mod(Zs[r], Int(cp.p[m])) + rand(-3:3) * Int(cp.p[m])); end
            e = rand(-40:40, N); e[9] = typemin(Int); fj = rand(-40:40)
            x0 = [BigFloat(rand()) for _ in 1:N]; A = reshape(x0 .+ 0, N, 1)
            HP._crt_sub_block!(A, 0, 1, N, e, fj, cp, tmp)
            ok = true
            for r in 1:N
                ref = e[r] == typemin(Int) ? x0[r] : setprecision(() -> x0[r] - BigFloat(Zs[r]) * BigFloat(2)^(e[r] + fj - 2cp.B), BigFloat, 4000)
                ok &= abs(A[r, 1] - ref) <= 2eps(BigFloat(abs(ref); precision=bits))
            end
            @test ok
        end
    end
end

@testset "Cholesky restart (done, split, copy_leading!)" begin
    setprecision(BigFloat, 256) do
        M = 260; m = 150
        X = BigFloat.(randn(M, M)) ./ M; A = fresh(X'X + I)
        for scheme in (:crt, :slice), nb in (64, 128)
            Lf = fresh(A); @test chol_lower_ozaki!(Lf; split=m, nb, scheme)
            L0 = fresh(A); @test chol_lower_ozaki!(L0; nb, scheme)
            Lm = fresh(A[1:m, 1:m]); @test chol_lower_ozaki!(Lm; nb, scheme)
            B = fresh(A); copy_leading!(B, Lm); @test chol_lower_ozaki!(B; done=m, nb, scheme)
            @test all(B[i, j] == Lf[i, j] for j in 1:M for i in j:M)
            @test maximum(abs(B[i, j] - L0[i, j]) for j in 1:M for i in j:M) <= 1e-70
        end
        P = SymPacked(A); Pm = SymPacked(A[1:m, 1:m]); @test chol_lower_ozaki!(Pm; nb=64)
        Pf = fresh(P); @test chol_lower_ozaki!(Pf; nb=64, split=m)
        Q = fresh(P); copy_leading!(Q, Pm); @test chol_lower_ozaki!(Q; nb=64, done=m)
        @test all(Q[i, j] == Pf[i, j] for j in 1:M for i in j:M)
        Lm = fresh(A[1:m, 1:m]); chol_lower_ozaki!(Lm; nb=64)
        C = fresh(A); C[M, M] = -one(BigFloat); copy_leading!(C, Lm)
        @test !chol_lower_ozaki!(C; nb=64, done=m)
    end
end

@testset "SymPacked" begin
    setprecision(BigFloat, 320) do
        M = 150
        X = BigFloat.(randn(M, M)) ./ M; S = fresh(X'X + I); H = BigFloat.(randn(M, M)) ./ 10; H = fresh(H + H')
        Sp = SymPacked(S); Hp = SymPacked(H)
        @test Matrix(Sp) == S && Sp[3, 7] == S[3, 7] && size(Sp) == (M, M)
        @test sizeof(Sp.buf) < Base.summarysize(S) / 2.5
        for f in (chol_lower!, chol_lower_persist!, A -> chol_lower_ozaki!(A; nb=64))
            L = fresh(S); f(L); Lp = fresh(Sp); f(Lp)
            @test all(L[i, j] == Lp[i, j] for j in 1:M for i in j:M)
        end
        L = fresh(S); chol_lower!(L); Lp = fresh(Sp); chol_lower!(Lp)
        b = BigFloat.(randn(M))
        @test llt_solve!(fresh(b), L) == llt_solve!(fresh(b), Lp)
        @test lt_solve!(fresh(b), L) == lt_solve!(fresh(b), Lp)
        @test symv!(similar(b), H, b) == symv!(similar(b), Hp, b)
        E1 = eigen(Symmetric(Float64.(H)), Symmetric(Float64.(S))).values[1]; σ = BigFloat(E1 - 1e-3)
        @test lowest_eig_si(H, S, σ)[1] == lowest_eig_si(Hp, Sp, σ)[1]
        io = IOBuffer(); serialize(io, Sp); seekstart(io); Sq = deserialize(io)
        @test Sq == Sp
        Sq[2, 1] = 5; @test Sp[2, 1] == S[2, 1] && Sq[1, 2] == 5
        Sc = copy(Sp); D = diagonal_scale!(fresh(Hp), Sc)
        @test Sc[5, 3] == D[5] * S[5, 3] * D[3]
    end
    setprecision(BigFloat, 576) do                                # sign and exponent word: special values, aliasing
        @test sympacked_elbytes(576) == 80 && sympacked_elbytes(320) == 48
        P = SymPacked(4)
        @test all(iszero(P[i, j]) && !signbit(P[i, j]) for j in 1:4 for i in j:4)
        for v in (BigFloat(-0.0), BigFloat(NaN), BigFloat(Inf), BigFloat(-Inf), BigFloat(1) / 3, -BigFloat(2)^-100000, BigFloat(7)^5000)
            P[2, 1] = v; w = P[2, 1]
            @test isequal(w, v) && signbit(w) == signbit(v)
        end
        a = BigFloat(5) / 7; y = BigFloat(3) / 11
        P[2, 2] = a; x = HP._el(P, 2, 2)
        z = copy(a); mpfr_fma!(z, z, y, z)
        mpfr_fma!(x, x, y, x); @test P[2, 2] == z
        mpfr_div!(x, x, x); @test P[2, 2] == 1
        @test_throws OverflowError (P[3, 3] = BigFloat(2)^(big(2)^61 + 5))
    end
end

@testset "PackedMatrix and allocation-free MPFR operations" begin
    setprecision(BigFloat, 320) do
        A = PackedMatrix(5, 3)
        @test size(A) == (5, 3) && A[4, 2] == 0
        for j in 1:3, i in 1:5; A[i, j] = BigFloat(i) / (j + 6); end
        B = copy(A); B[1, 1] = 7
        @test A[1, 1] == BigFloat(1) / 7 && B[1, 1] == 7 && B[5, 3] == A[5, 3]
        x = BigFloat(3) / 11; z = BigFloat(); w = BigFloat()
        g() = GC.@preserve A x z begin
            pz = mpfr_ptr(z)
            for j in 1:3, i in 1:5; mpfr_mul!(pz, HP._el(A, i, j), mpfr_ptr(x)); end
        end
        g(); @test @allocated(g()) == 0
        @test z == A[5, 3] * x
        y = BigFloat(5) / 11
        f() = (for _ in 1:100; mpfr_mul!(z, x, y); mpfr_add!(z, z, x); mpfr_powi!(z, x, 13, w); end)
        f(); @test @allocated(f()) == 0
        mpfr_powi!(z, x, 23, w); @test abs(z - x^23) <= 30eps(BigFloat) * x^23
    end
end

@testset "release_task_refs" begin
    function grab()
        X = zeros(UInt8, 200_000_000)
        Threads.@threads :static for t in 1:Threads.nthreads(); X[t] = 1; end
        nothing
    end
    grab(); release_task_refs(); GC.gc(); GC.gc()
    @test Base.gc_live_bytes() < 150_000_000
end

@testset "lowest eigenpair by shifted inverse iteration" begin
    for U in (Float64x4, BigFloat)
        U === BigFloat && setprecision(BigFloat, 320)
        M = 200
        X = U.(randn(M, M)) ./ M; S = X'X + I; H = U.(randn(M, M)) ./ 10; H = H + H'
        L = fresh(S); chol_lower!(L); tril!(L)
        Y = fresh(H); ltri_solve!(Y, L); Z = permutedims(Y); ltri_solve!(Z, L)
        P64 = Float64.(Z); E1 = eigen(Symmetric((P64 + P64') / 2)).values[1]
        r = lowest_eig_si(H, S, U(E1) - U(1e-3); maxit=200)
        @test r !== nothing && r[4]
        E, x = r
        @test abs(Float64(E) - E1) < 1e-10
        @test maximum(abs, H * x - E * (S * x)) < 1e4 * eps(U) * M
        @test abs(dot(x, S * x) - 1) < 1e3 * eps(U)
        @test lowest_eig_si(H, S, U(E1) + U(1e-3)) === nothing
        for kw in ((; maxit=200), (; maxit=3, rounds=4))
            r0 = lowest_eig_si(H, S, U(E1) - U(1e-1); kw...)
            r1 = lowest_eig_si(H, S, U(E1) - U(1e-1); work=fresh(L), kw...)
            @test r0[1] == r1[1] && r0[2] == r1[2] && r0[3] == r1[3] && r0[5] == r1[5]
        end
    end
    setprecision(BigFloat, 320) do                                 # etol and prods
        M = 150
        X = BigFloat.(randn(M, M)) ./ M; S = SymPacked(X'X + I); H = BigFloat.(randn(M, M)) ./ 10; H = SymPacked(H + H')
        r0 = lowest_eig_si(H, S, BigFloat(-10); maxit=200)
        σ = r0[1] - BigFloat(1e-30)
        a = lowest_eig_si(H, S, σ)
        pr = Ref{Any}(nothing)
        b = lowest_eig_si(H, S, σ; prods=pr)
        @test a[1] == b[1] && a[2] == b[2] && a[3] == b[3]
        Hx, Sx = pr[]
        @test Hx == symv!(Vector{BigFloat}(undef, M), H, b[2])
        @test maximum(abs, Sx - symv!(Vector{BigFloat}(undef, M), S, b[2])) < 1e-90
        c = lowest_eig_si(H, S, σ; etol=BigFloat(1e-40), etol_dx=1e-33)
        @test c[4] && c[3] <= 3 && c[3] < a[3]
        @test abs(c[1] - a[1]) < 1e-80
    end
    setprecision(BigFloat, 256) do                                 # excited states (lift)
        M = 40
        Q = [sin(BigFloat(3i + 7j) / 5) for i in 1:M, j in 1:M]; S = Q'Q / M + I; S = (S + S') / 2
        R = [cos(BigFloat(i * j) / 3) + BigFloat(i == j ? i : 0) / 4 for i in 1:M, j in 1:M]; H = (R + R') / 2
        L = cholesky(Symmetric(S)).L; P = L \ (L \ H)'; ev = eigen_mt(Symmetric((P + P') / 2)).values
        for (Hm, Sm) in ((H, S), (SymPacked(H), SymPacked(S)))
            X = zeros(BigFloat, M, 0)
            for k in 1:4
                σ = ev[k] - (ev[k+1] - ev[k]) / 10
                r = lowest_eig_si(Hm, Sm, σ; lift=k == 1 ? nothing : (X, 3 * (σ - ev[1]) + 1))
                @test r !== nothing && abs(r[1] - ev[k]) < 1e-70 && r[4]
                X = hcat(X, r[2])
            end
            @test lowest_eig_si(Hm, Sm, (ev[2] + ev[3]) / 2) === nothing
        end
    end
end

@testset "eigen_mt and eigen_proj" begin
    A = Float64x4.(randn(80, 80)); A = A + A'
    @test eigen_mt(Symmetric(A)).values == eigen(Symmetric(A)).values
    for uplo in (:U, :L)
        B = BigFloat.(randn(60, 60)); A = Symmetric(B + B', uplo); Y = BigFloat.(randn(60, 2))
        e = eigen_mt(A); r = eigen_proj(A, Y)
        @test r.values == e.values
        @test maximum(abs.(r.P .- e.vectors' * Y)) < 1e-60
    end
    X = BigFloat.(randn(30, 5)); Hs = BigFloat.(randn(30, 30)); Hs = Hs + Hs'
    @test xtHx_mt(X, Hs) == X' * Hs * X
end
