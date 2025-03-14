export StrongBeam

######
# charge normalzied to e0
# mass normalized to electron mass
struct StrongBeam{T}
  npar::T
  q::T
  emmx::T
  emmy::T
  sigz::T
  betx::T
  bety::T
  cross_angle::T
  nslice::Int
  slice_centroids::Vector{SVector{3,T}}
end

function StrongBeam(;sp::ChargedSpecie{T}, npar::Number, 
        emmx::T, emmy::T,sigz::T,
        betx::T, bety::T, nslice::Int, slicing_type=1, 
        cross_angle::T=0.0, f_crab::T=1.0e38) where {T}

  z_centroids = _zcentroids(nslice, sigz, slicing_type)

  # generated crabbed slice centroids
  k_crab = 2*pi*f_crab/c0
  phi = cross_angle/2
  sl_centroids = Vector{SVector{3,T}}(undef, nslice) 
  for i in 1:nslice
    z = z_centroids[i]
    x = -tan(phi)*(sin(k_crab*z)/k_crab - z)
    sl_centroids[i] = SVector{3,T}(x, 0, z)
  end

  return StrongBeam(npar, sp.q, emmx, emmy, sigz, betx, bety, cross_angle, nslice, sl_centroids)
end

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

# cpu interaction methods
function interact!(beam::Beam{T}, elm::StrongBeam{T}) where {T}
  phi = elm.cross_angle/2
  # test macro particle
  t_q = beam.q
  t_p0 = beam.p0

  # bunch 
  b_q = elm.q
  b_npar = elm.npar
  b_emmx = elm.emmx
  b_emmy = elm.emmy
  b_sigz = elm.sigz
  b_betx = elm.betx
  b_bety = elm.bety
  b_nslice = elm.nslice
  b_sl_centroids = elm.slice_centroids
  b_Q_slice = b_q*b_npar/b_nslice

  # leading constant for beam-beam force from each slice
  A = t_q * b_Q_slice * e0 / (4*pi*epsilon0) / t_p0
  coords = beam.coords
  lumin = zero(T)
  Threads.@threads for i in 1:beam.npar
    x, px, y, py, z, pz = lboost(coords[i], phi)
    
    # collides with a sequence of slices 
    for k in 1:b_nslice
      x_slice, y_slice, z_slice = b_sl_centroids[k]
      s = (z - z_slice)/2.0
  
      # drift from ip to cp
      x = x + s*px
      y = y + s*py
      pz = pz - 0.25 * (px^2 + py^2)

      # kick at cp
      dpx, dpy, dpz = bbimpulses(x, y, s, A, b_emmx, b_emmy, b_betx, b_bety, x_slice, y_slice)
      px = px + dpx
      py = py + dpy
      pz = pz + dpz

      # dirft back from cp to ip
      x = x - s*px
      y = y - s*py
      pz = pz + 0.25 * (px^2 + py^2)

      lumin += luminosity(x, y, s, b_emmx, b_emmy, b_betx, b_bety, x_slice, y_slice)
    end
    coords[i] = ilboost(x, px, y, py, z, pz, phi)
  end

  lumin = abs(t_q) * abs(b_Q_slice) * lumin
  return lumin
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

# calculate the beam-beam impulses from a slice with the center (x0, y0)
function bbimpulses(x::T, y::T, s::T, A::T, emmx::T, emmy::T, betx::T, bety::T, x0::T=0, y0::T=0) where {T} 
  sigx = sqrt( emmx*betx*(1 + s^2/betx^2) ) 
  sigy = sqrt( emmy*bety*(1 + s^2/bety^2) ) 

  x = x - x0
  y = y - y0

  z1 = ( sigy/sigx*x + sigx/sigy*y*im )/sqrt(2*sigx^2 - 2*sigy^2)
  z2 = (x + y*im)/sqrt(2*sigx^2 - 2*sigy^2)
  ef = -sqrt(2*pi/(sigx^2 - sigy^2)) * ( faddeeva(z2, Val(15)) - faddeeva(z1, Val(15) )*exp(-x^2/(2*sigx^2) -y^2/(2*sigy^2)) )
  ux = ef.im
  uy = ef.re

  uxx = -(x*ux + y*uy)/(sigx^2-sigy^2) -
        2/(sigx^2-sigy^2)*( 1 - sigy/sigx*exp(-x^2/2/sigx^2 - y^2/2/sigy^2) )
  uyy =  (x*ux + y*uy)/(sigx^2-sigy^2) +
            2/(sigx^2-sigy^2)*( 1 - sigx/sigy*exp(-x^2/2/sigx^2 - y^2/2/sigy^2) )
  dsigxds = sqrt(emmx/betx^3)*s/sqrt(1 + s^2/betx^2)
  dsigyds = sqrt(emmy/bety^3)*s/sqrt(1 + s^2/bety^2)
  uz = 0.5 * (sigx*uxx*dsigxds + sigy*uyy*dsigyds)

  return (-A*ux, -A*uy, -A*uz)
end

function luminosity(x::T, y::T, s::T, emmx::T, emmy::T, betx::T, bety::T, x0::T=0, y0::T=0) where {T}
  sigx = sqrt( emmx*betx*(1 + s^2/betx^2) ) 
  sigy = sqrt( emmy*bety*(1 + s^2/bety^2) ) 
  lumin = exp( -(x-x0)^2/2/sigx^2 - (y-y0)^2/2/sigy^2 ) / (2*pi*sigx*sigy)
  return lumin
end

function interact!(beam::BeamGPU{T}, elm::StrongBeam{T}) where {T}
  cross_angle = elm.cross_angle

  # test macro particle
  t_npar = beam.npar
  t_coords = beam.coords
  t_q = beam.q
  t_p0 = beam.p0

  # bunch 
  b_q = elm.q
  b_npar = elm.npar
  b_emmx = elm.emmx
  b_emmy = elm.emmy
  b_sigz = elm.sigz
  b_betx = elm.betx
  b_bety = elm.bety
  b_nslice = elm.nslice
  global b_sl_centroids = CuArray(elm.slice_centroids)

  # luminosity 
  global total_luminosity = CuArray([zero(T)])

  nb = ceil(Int, t_npar/GLOBAL_BLOCK_SIZE)
  @cuda threads=GLOBAL_BLOCK_SIZE  blocks=nb  _gpu_interact_strongbeam!(t_coords, t_npar, t_q, t_p0, b_q, b_npar, b_emmx, b_emmy, b_sigz, b_betx, b_bety, b_nslice, b_sl_centroids, cross_angle, total_luminosity)

  return Array(total_luminosity)[1]
end

function _gpu_interact_strongbeam!(t_coords::CuDeviceVector{SVector{D,T},1}, t_npar::Int, t_q::T, t_p0::T, b_q::T, b_npar::T, b_emmx::T, b_emmy::T, b_sigz::T, b_betx::T, b_bety::T, b_nslice::Int, b_sl_centroids::CuDeviceVector{SVector{3,T},1}, cross_angle::T, luminosity_out::CuDeviceVector{T,1}) where {D,T}

  phi = cross_angle/2
  b_Q_slice = b_q*b_npar/b_nslice

  # leading constant for beam-beam force from each slice
  A = t_q * b_Q_slice * e0 / (4*pi*epsilon0) / t_p0

  tid = threadIdx().x
  bid = blockIdx().x
  block_size = blockDim().x 
  gid = tid + (bid - 1) * block_size

  local_luminosity = zero(T)
  if gid <= t_npar
    #@cuprint t_coords[gid][1] t_coords[gid][2] t_coords[gid][3] t_coords[gid][4] t_coords[gid][5] t_coords[gid][6] 
    #@cuprint "\n"
    x, px, y, py, z, pz = lboost(t_coords[gid], phi)
    
    # collides with a sequence of slices 
    for k in 1:b_nslice
      x_slice, y_slice, z_slice = b_sl_centroids[k]
      s = (z - z_slice)/2.0
  
      # drift from ip to cp
      x = x + s*px
      y = y + s*py
      pz = pz - 0.25 * (px^2 + py^2)

      # kick at cp
      dpx, dpy, dpz = bbimpulses(x, y, s, A, b_emmx, b_emmy, b_betx, b_bety, x_slice, y_slice)
      px = px + dpx
      py = py + dpy
      pz = pz + dpz

      # dirft back from cp to ip
      x = x - s*px
      y = y - s*py
      pz = pz + 0.25 * (px^2 + py^2)

      # luminosity
      local_luminosity = local_luminosity + luminosity(x, y, s, b_emmx, b_emmy, b_betx, b_bety, x_slice, y_slice)
    end
    t_coords[gid] = ilboost(x, px, y, py, z, pz, phi)
  end
  # blcok reduction for luminosity
  total_luminosity = abs(t_q) * abs(b_Q_slice) * CUDA.reduce_block(+, local_luminosity, zero(T), Val(true))
  if tid == 1
    @inbounds CUDA.@atomic luminosity_out[1] += total_luminosity
  end
  return nothing
end

