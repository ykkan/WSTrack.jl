export faddeeva
export FaddeevaAlg
export Weidemann
export Abrarov

using FFTW

# ============================================================
# Algorithm types
# ============================================================

abstract type FaddeevaAlg end

struct Weidemann{N} <: FaddeevaAlg end
Weidemann(n::Int) = Weidemann{n}()

struct Abrarov{N} <: FaddeevaAlg end
Abrarov(n::Int) = Abrarov{n}()


# ============================================================
# Default algorithm
#
# Change only this line to switch implementation:
#
#   const DefaultFaddeevaAlg = Weidemann{48}
#   const DefaultFaddeevaAlg = Abrarov{15}
# ============================================================

# const DefaultFaddeevaAlg = Weidemann{32}
const DefaultFaddeevaAlg = Abrarov{23}

# ============================================================
# Public API
# ============================================================

Base.@noinline faddeeva(z::Complex) =
    faddeeva(z, DefaultFaddeevaAlg())

@inline faddeeva(z::Complex{T}, ::Weidemann{N}) where {T,N} =
    faddeeva_weidemann(z, Val(N))

@inline faddeeva(z::Complex{T}, ::Abrarov{N}) where {T,N} =
    faddeeva_abrarov(z, Val(N))


# ============================================================
# Weideman implementation
# ============================================================

function __weidemann_coeff(n, type::Type=Float64)
    N = n
    M = 2N
    L = sqrt(N / sqrt(type(2)))

    theta_arr = [
        k * type(pi) / M
        for k in (-M + 1):(M - 1)
    ]

    t_arr = L .* tan.(theta_arr ./ 2)

    f_arr = [
        (L^2 + t^2) * exp(-t^2)
        for t in t_arr
    ]

    f_arr = vcat(zero(type), f_arr)

    a_arr =
        real.(fft(fftshift(f_arr))) / (2M)

    return a_arr[2:(N + 1)], L
end


@generated function faddeeva_weidemann(
    z::Complex{T},
    ::Val{N}
) where {T,N}

    a_arr, L = __weidemann_coeff(N, T)
    a_arr = reverse(a_arr)

    # Horner evaluation of polynomial
    ex = :($(a_arr[1]))

    for i in 2:N
        ex = :(muladd($ex, Z, $(a_arr[i])))
    end

    return quote
        # Map lower half-plane to upper half-plane
        c1 = imag(z) >= 0 ? 0 : 1
        c2 = 1 - 2c1

        z = c2 * z

        lmiz_inv = inv($L - im * z)
        Z = ($L + im * z) * lmiz_inv

        w = (
            2 * $ex * lmiz_inv +
            $(1 / sqrt(T(pi)))
        ) * lmiz_inv

        return c1 * 2 * exp(-z^2) + c2 * w
    end
end


# ============================================================
# Abrarov implementation
# ============================================================

@inline function __abrarov_coeff(
    n,
    t=12,
    type::Type=Float64
)
    return (
        2 * sqrt(type(pi)) / t *
        exp(-n^2 * type(pi)^2 / t^2)
    )
end


@generated function faddeeva_abrarov(
    z::Complex{T},
    ::Val{N}
) where {T,N}

    t = 12

    ep = :(
        $(__abrarov_coeff(0, t, T)) / 2 *
        (1 - eitz) / tz2
    )

    for i in 1:N

        sign_i = iseven(i) ? 1 : -1

        ep = :(
            $ep +
            $(__abrarov_coeff(i, t, T)) *
            (
                $sign_i * eitz - 1
            ) /
            (
                $(i^2 * T(pi)^2) - tz2
            )
        )
    end

    ep = :(
        $(im / sqrt(T(pi))) *
        ttz *
        $ep
    )

    return quote
        # Map lower half-plane to upper half-plane
        c1 = imag(z) >= 0 ? 0 : 1
        c2 = 1 - 2c1

        z = c2 * z

        tz = $t * z
        ttz = $t * tz
        tz2 = tz * tz

        eitz = exp(im * tz)

        w = $ep

        return c1 * 2 * exp(-z^2) + c2 * w
    end
end
