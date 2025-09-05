export faddeeva
export FaddeevaAlg
export Weidemann
export Abrarov

@inline faddeeva(z) = faddeeva(z, Weidemann(16))

abstract type FaddeevaAlg end

struct Weidemann{N} <: FaddeevaAlg end
Weidemann(n::Int) = Weidemann{n}()

struct Abrarov{N} <: FaddeevaAlg end
Abrarov(n::Int) = Abrarov{n}()

faddeeva(z::Complex{T}, alg::Weidemann{N}) where {T,N} = faddeeva_weidemann(z::Complex{T}, Val(N))
faddeeva(z::Complex{T}, alg::Abrarov{N}) where {T,N} = faddeeva_abrarov(z::Complex{T}, Val(N))

function __weidemann_coeff(n, type::Type=Float64)
  N = n
  M = 2*N
  L = sqrt(N/sqrt(type(2)))
  theta_arr = [k*type(pi)/M for k in (-M+1):(M-1)]
  t_arr = L*tan.(theta_arr/2)
  f_arr = [(L^2 + t^2)*exp(-t^2) for t in t_arr]
  f_arr = vcat(zero(type), f_arr)
  a_arr = real.(fft(fftshift(f_arr)))/(2*M)
  return a_arr[2:(N+1)], L
end

@generated function faddeeva_weidemann(z::Complex{T}, ::Val{N}=Val(16)) where {T,N}
  a_arr, L = __weidemann_coeff(N, T)
  a_arr = reverse(a_arr)
  ex = :($(a_arr[1]))
  for i in 2:N
    ex = :(muladd($ex, Z, $(a_arr[i])))
  end
  return quote
    c1 = imag(z) >= 0 ? 0 : 1
    c2 = 1 - 2*c1 # same as imag(z) >= 0 ? 1 : -1
    z = c2*z
    lmiz_inv = 1/($L - im*z)
    Z = ($L + im*z)*lmiz_inv
    return c1*2*exp(-z^2) + c2*(2*$ex*lmiz_inv + $(1/sqrt(T(pi))))*lmiz_inv
  end
end

__abrarov_coeff(n, t=12, type::Type=Float64) = 2*sqrt(type(pi))/t*exp(-n^2*type(pi)^2/t^2)  

@generated function faddeeva_abrarov(z::Complex{T}, ::Val{N}=Val(15)) where {T,N}
  t = 12
  ep = :( $(__abrarov_coeff(0,t,T))/2 * (1 - eitz)/tz2 )
  for i in 1:N
    ep = :($ep + $(__abrarov_coeff(i,t,T)) * ( $(iseven(i) ? 1 : -1) * eitz -1 )/( $(i^2*T(pi)^2) - tz2) )
  end
  ep = :( $(im/sqrt(T(pi))) * ttz * $ep )
  return quote
    c1 = imag(z) >= 0 ? 0 : 1
    c2 = 1 - 2*c1
    z = c2*z
    tz = $t * z
    ttz = $t * tz
    tz2 = tz*tz 
    eitz = exp(im * tz)
    return c1*2*exp(-z^2) + c2 * $ep
  end
end

