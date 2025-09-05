export StrongBeamDecoupled

######
# charge normalzied to e0
# mass normalized to electron mass
struct StrongBeamDecoupled{T}
  np::T
  q::T
  sigx::T
  sigpx::T
  sigy::T
  sigpy::T
  sigz::T
  cross_angle::T
  nslice::Int
  slice_centroids::Vector{SVector{3,T}}
end

function StrongBeamDecoupled(;sp::ChargedSpecie{T}, np::Number, 
        sigx::T, sigpx::T, sigy::T, sigpy::T, sigz::T,
        nslice::Int, slicing_type=1, 
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

  return StrongBeamDecoupled(np, sp.q, sigx, sigpx, sigy, sigpy, sigz, cross_angle, nslice, sl_centroids)
end


# calculate the beam-beam impulses from a slice with the center (x0, y0)
function par_sl_interaction_decoupled(x::T, y::T, s::T, A::T, sigx::T, sigpx::T, sigy::T, sigpy::T, phi::T, x0::T, y0::T, faddeeva_alg::FaddeevaAlg) where {T} 
  sigx_cp = sqrt(sigx^2 + (sigpx*s)^2) 
  sigy_cp = sqrt(sigy^2 + (sigpy*s)^2) 

  csphi = cos(phi)
  x = x - x0
  y = y - y0

  z1 = ( sigy_cp/sigx_cp*x + sigx_cp/sigy_cp*y*im )/sqrt(2*sigx_cp^2 - 2*sigy_cp^2)
  z2 = (x + y*im)/sqrt(2*sigx_cp^2 - 2*sigy_cp^2)
  ef = -sqrt(2*pi/(sigx_cp^2 - sigy_cp^2)) * ( faddeeva(z2, faddeeva_alg) - faddeeva(z1, faddeeva_alg)*exp(-x^2/(2*sigx_cp^2) -y^2/(2*sigy_cp^2)) )
  ux = ef.im
  uy = ef.re

  uxx = -(x*ux + y*uy)/(sigx_cp^2-sigy_cp^2) - 2/(sigx_cp^2-sigy_cp^2)*( 1 - sigy_cp/sigx_cp*exp(-x^2/2/sigx_cp^2 - y^2/2/sigy_cp^2) )
  uyy =  (x*ux + y*uy)/(sigx_cp^2-sigy_cp^2) + 2/(sigx_cp^2-sigy_cp^2)*( 1 - sigx_cp/sigy_cp*exp(-x^2/2/sigx_cp^2 - y^2/2/sigy_cp^2) )


  dsigxxds_cp = 2*(sigpx/csphi)^2*s
  dsigyyds_cp = 2*(sigpy/csphi)^2*s

  uz = 0.5 * (0.5*uxx*dsigxxds_cp + 0.5*uyy*dsigyyds_cp)

  lumin = par_sl_luminosity(x, y, sigx, sigy)
  return (-A*ux, -A*uy, -A*uz, lumin)
end


# CPU
function interact!(beam::Beam{T}, elm::StrongBeamDecoupled{T}, faddeeva_alg::FaddeevaAlg) where {T}
  phi = elm.cross_angle/2
  # test macro particle
  t_q = beam.q
  t_p0 = beam.p0

  # bunch 
  b_q = elm.q
  b_np = elm.np
  b_sigx = elm.sigx
  b_sigpx = elm.sigpx
  b_sigy = elm.sigy
  b_sigpy = elm.sigpy
  b_sigz = elm.sigz
  b_nslice = elm.nslice
  b_sl_centroids = elm.slice_centroids
  b_Q_slice = b_q*b_np/b_nslice

  # leading constant for beam-beam force from each slice
  A = t_q * b_Q_slice * e0 / (4*pi*epsilon0) / t_p0
  coords = beam.coords
  luminosity = zero(T)
  Threads.@threads for i in 1:beam.nmp
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
      dpx, dpy, dpz, lumin = par_sl_interaction_decoupled(x, y, s, A, b_sigx, b_sigpx, b_sigy, b_sigpy, phi, x_slice, y_slice, faddeeva_alg)

      px = px + dpx
      py = py + dpy
      pz = pz + dpz

      # dirft back from cp to ip
      x = x - s*px
      y = y - s*py
      pz = pz + 0.25 * (px^2 + py^2)

      luminosity += par_sl_luminosity(x, y, sigx, sigy)
    end
    coords[i] = ilboost(x, px, y, py, z, pz, phi)
  end

  luminosity = abs(t_q) * abs(b_Q_slice) * luminosity
  return luminosity
end


# GPU
function interact!(beam::BeamGPU{T}, elm::StrongBeamDecoupled{T}, faddeeva_alg::FaddeevaAlg=Abrarov(16)) where {T}
  cross_angle = elm.cross_angle

  # test macro particle
  t_nmp = beam.nmp
  t_coords = beam.coords
  t_q = beam.q
  t_p0 = beam.p0

  # bunch 
  b_q = elm.q
  b_np = elm.np
  b_sigx = elm.sigx
  b_sigpx = elm.sigpx
  b_sigy = elm.sigy
  b_sigpy = elm.sigpy
  b_sigz = elm.sigz
  b_nslice = elm.nslice
  global b_sl_centroids = CuArray(elm.slice_centroids)

  # luminosity 
  global total_luminosity = CuArray([zero(T)])

  nb = ceil(Int, t_nmp/GLOBAL_BLOCK_SIZE)
  @cuda threads=GLOBAL_BLOCK_SIZE  blocks=nb  _gpu_interact_strong_beam_decoupled!(t_coords, t_nmp, t_q, t_p0, b_q, b_np, b_sigx, b_sigpx, b_sigy, b_sigpy, b_nslice, b_sl_centroids, cross_angle, faddeeva_alg, total_luminosity)

  return Array(total_luminosity)[1]
end

function _gpu_interact_strong_beam_decoupled!(t_coords::CuDeviceVector{SVector{D,T},1}, t_nmp::Int, t_q::T, t_p0::T, b_q::T, b_np::T, b_sigx::T, b_sigpx::T, b_sigy::T, b_sigpy::T, b_nslice::Int, b_sl_centroids::CuDeviceVector{SVector{3,T},1}, cross_angle::T, faddeeva_alg::FaddeevaAlg, luminosity_out::CuDeviceVector{T,1}) where {D,T}

  phi = cross_angle/2
  b_Q_slice = b_q*b_np/b_nslice

  # leading constant for beam-beam force from each slice
  A = t_q * b_Q_slice * e0 / (4*pi*epsilon0) / t_p0

  tid = threadIdx().x
  bid = blockIdx().x
  block_size = blockDim().x 
  gid = tid + (bid - 1) * block_size

  local_luminosity = zero(T)
  if gid <= t_nmp
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
      dpx, dpy, dpz, lumin = par_sl_interaction_decoupled(x, y, s, A, b_sigx, b_sigpx, b_sigy, b_sigpy, phi, x_slice, y_slice, faddeeva_alg)
      px = px + dpx
      py = py + dpy
      pz = pz + dpz

      # dirft back from cp to ip
      x = x - s*px
      y = y - s*py
      pz = pz + 0.25 * (px^2 + py^2)

      # luminosity
      local_luminosity = local_luminosity + lumin
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

