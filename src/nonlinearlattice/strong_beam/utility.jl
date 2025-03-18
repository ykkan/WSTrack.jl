# compute z-centroids for the `n` slices of a 
# strong-beam with bunch length `sig`. 
# Z-centroids are ordered from +z to -z
#
# `slicing_type` is used to specify the slicing algorithm
# - 1: equal charge, slice zcentroid determined by the mean location
# - 2: equal charge, slice zcentroid determined by 50% of the cumultative prob. 
function _zcentroids(n, sig::T, slicing_type::Int) where {T}
  zcentroids = zeros(T, n)
  if slicing_type == 1
    zcuts = [sqrt(2)*sig*erfinv(2.0*i/n - 1.0) for i in 0:n]
    for i in 1:n
      z1 = zcuts[i]
      z2 = zcuts[i+1] 
      Q1 = 0.5 + 0.5*erf(z1/sqrt(2)/sig)
      Q2 = 0.5 + 0.5*erf(z2/sqrt(2)/sig)
      zcentroids[i] = sig/sqrt(2*pi)/( Q2 - Q1) * ( exp(-z1^2/sig^2/2) - exp(-z2^2/sig^2/2) )
    end
    reverse!(zcentroids)
  elseif slicing_type == 2
    zcentroids = [ sqrt(2)*sig*erfinv((2*i-1-n)/n) for i in 1:n ]
    reverse!(zcentroids)
  end
  return zcentroids
end

function lboost(coord::SVector{6,T}, phi::T) where {T}
  x, px, y, py, z, pz = coord

  tg = tan(phi)
  ss = sin(phi)
  cs = cos(phi)

  h = 1 + pz - sqrt((1 + pz)^2 - px^2 -py^2)
  px_new = (px -h*tg)/cs  
  py_new = py/cs
  pz_new = pz - px*tg + h*tg^2

  ps_new = sqrt((1+pz_new^2) - px_new^2 - py_new^2)
  hx_new = px_new/ps_new
  hy_new = py_new/ps_new
  hz_new = 1 - (1+pz_new)/ps_new

  x_new = tg*z + (1 + hx_new*ss)*x 
  y_new = y + hy_new*ss*x
  z_new = z/cs + hz_new*ss*x

  return (x_new, px_new, y_new, py_new, z_new, pz_new)
end

function ilboost(x_new::T, px_new::T, y_new::T, py_new::T, z_new::T, pz_new::T, phi::T) where {T}
  ps_new = sqrt((1 + pz_new)^2 - px_new^2 - py_new^2) 
  hx_new = px_new/ps_new
  hy_new = py_new/ps_new
  hz_new = 1 - (pz_new + 1)/ps_new

  tg = tan(phi)
  ss = sin(phi)
  cs = cos(phi)

  det = 1/cs + (hx_new - hz_new*ss)*tg
  x = (x_new/cs - tg*z_new) / det
  y = ( -hy_new*tg*x_new  + (1/cs + tg*hx_new - hz_new*tg*ss)*y_new + hy_new*tg*ss*z_new ) / det
  z = ( -hz_new*ss*x_new + (1 + hx_new*ss)*z_new ) / det

  h_new = 1 + pz_new - sqrt((1 + pz_new)^2 - px_new^2 -py_new^2)
  h = h_new*cs^2
  px = px_new*cs + h*tg
  py = py_new*cs
  pz = pz_new + px*tg - h*tg^2
  
  return SVector{6,T}(x, px, y, py, z, pz)
end

# particle slice luminosity
# x, y relative to the slice centroid
function par_sl_luminosity(x::T, y::T, sigx::T, sigy::T) where {T}
  return exp( -x^2/2/sigx^2 - y^2/2/sigy^2 ) / (2*pi*sigx*sigy)
end
