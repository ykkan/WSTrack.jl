_fadd_coef(n, t=12) = 2*sqrt(pi)/t*exp(-n^2*pi^2/t^2)  

@generated function faddeeva(z::Complex, ::Val{N}) where {N}
  t = 12
  ep = :( $(_fadd_coef(0))/2 * (1 - eitz)/tz2 )
  for i in 1:N
    ep = :($ep + $(_fadd_coef(i)) * ( $(iseven(i) ? 1 : -1) * eitz -1 )/( $(i^2*pi^2) - tz2) )
  end
  ep = :( $(im/sqrt(pi)) * ttz * $ep )
  return quote
    c1 = imag(z) >= 0 ? 0 : 1
    c2 = imag(z) >= 0 ? 1 : -1
    z = c2*z
    tz = $t * z
    ttz = $t * tz
    tz2 = tz*tz 
    eitz = exp(im * tz)
    return c1*2*exp(-z^2) + c2 * $ep
  end
end
