export StrongBeamCoupled4D

######
# charge normalzied to e0
# mass normalized to electron mass
# sigs = [sig11..sig14, sig22...sig24, sig33, sig34, sig44]
struct StrongBeamCoupled4D{T}
  np::T
  q::T
  Sigma::SVector{10,T}
  sigz::T
  cross_angle::T
  nslice::Int
  slice_centroids::Vector{SVector{3,T}}
end

function StrongBeamCoupled4D(;sp::ChargedSpecie{T}, np::Number,
        Sigma::AbstractVector{T}, sigz::T, nslice::Int, slicing_type=1, 
        cross_angle::T, f_crab::T=1.0e38) where {T}
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
  return StrongBeamCoupled4D(np, sp.q, SVector{10,T}(Sigma), sigz, cross_angle, nslice, sl_centroids)
end


function par_sl_interaction_coupled4D(x::T, y::T, s::T, Amp::T, sig11::T, sig12::T, sig13::T, sig14::T, sig22::T, sig23::T, sig24::T, sig33::T, sig34::T, sig44::T, phi::T, x0::T=0, y0::T=0) where {T} 

  x = x - x0
  y = y - y0
  csphi = cos(phi)  

  # sigma matrix at cp in the boosted frame 
  sig11_cp = sig11 + 2*sig12/csphi*s + sig22/csphi^2*s^2
  sig33_cp = sig33 + 2*sig34/csphi*s + sig44/csphi^2*s^2
  sig13_cp = sig13 + (sig14 + sig23)/csphi*s + sig24/csphi^2*s^2
  dsig11_cp = 2*(sig12/csphi) + 2*sig22/csphi^2*s
  dsig33_cp = 2*(sig34/csphi) + 2*sig44/csphi^2*s
  dsig13_cp = (sig14 + sig23)/csphi + 2*sig24/csphi^2*s
  
  A = sig11_cp - sig33_cp
  B = 2*sig13_cp
  C = sig11_cp + sig33_cp
  D = sqrt(A^2 + B^2)
  dA = dsig11_cp - dsig33_cp
  dB = 2*dsig13_cp
  dC = dsig11_cp + dsig33_cp
  dD = (A*dA + B*dB)/D

  cs2th = A/D
  ss2th = B/D
  dcs2th = (dA*D - A*dD)/D^2
  dss2th = (dB*D - B*dD)/D^2

  csth = sqrt((1 + cs2th)/2)
  ssth = ss2th/csth/2 
  dcsth = dcs2th/csth/4
  dssth = (dss2th - 2*ssth*dcsth)/(2*csth)

  sig11_cp_bar = (C + D)/2
  sig33_cp_bar = (C - D)/2
  dsig11_cp_bar = (dC + dD)/2
  dsig33_cp_bar = (dC - dD)/2
  
  x_bar = x*csth + y*ssth
  y_bar = -x*ssth + y*csth
  dx_bar = x*dcsth + y*dssth
  dy_bar = -x*dssth + y*dcsth
  

  z1 = ( sqrt(sig33_cp_bar/sig11_cp_bar)*x_bar + sqrt(sig11_cp_bar/sig33_cp_bar)*y_bar*im )/sqrt( 2*(sig11_cp_bar - sig33_cp_bar) )
  z2 = (x_bar + y_bar*im)/sqrt( 2*(sig11_cp_bar - sig33_cp_bar) )
  ef = -sqrt(2*pi/(sig11_cp_bar - sig33_cp_bar)) * ( faddeeva(z2) - faddeeva(z1)*exp(-x_bar^2/sig11_cp_bar/2 -y_bar^2/sig33_cp_bar/2) )
  ux_bar = ef.im
  uy_bar = ef.re

  uxx_bar = -(x_bar*ux_bar + y_bar*uy_bar)/(sig11_cp_bar - sig33_cp_bar) -
  2/(sig11_cp_bar - sig33_cp_bar)*( 1 - sqrt(sig33_cp_bar/sig11_cp_bar)*exp(-x_bar^2/2/sig11_cp_bar - y_bar^2/2/sig33_cp_bar) )
  uyy_bar =  (x_bar*ux_bar + y_bar*uy_bar)/(sig11_cp_bar - sig33_cp_bar) +
  2/(sig11_cp_bar - sig33_cp_bar)*(1 - sqrt(sig11_cp_bar/sig33_cp_bar)*exp(-x_bar^2/2/sig11_cp_bar - y_bar^2/2/sig33_cp_bar) )
  uz = 0.5*(ux_bar*dx_bar + uy_bar*dy_bar + 0.5*uxx_bar*dsig11_cp_bar + 0.5*uyy_bar*dsig33_cp_bar)

  ux = ux_bar*csth - uy_bar*ssth
  uy = ux_bar*ssth + uy_bar*csth

  lumin = exp( -x_bar^2/2/sig11_cp_bar - y_bar^2/2/sig33_cp_bar ) / (2*pi*sqrt(sig11_cp_bar*sig33_cp_bar))
  
  return (-Amp*ux, -Amp*uy, -Amp*uz, lumin)
end

# CPU
function interact!(beam::Beam{T}, elm::StrongBeamCoupled4D{T}) where {T}
  phi = elm.cross_angle/2
  # test macro particle
  t_q = beam.q
  t_p0 = beam.p0

  # bunch 
  b_q = elm.q
  b_np = elm.np
 
  b_Sigma = elm.Sigma
  b_sig11 = b_Sigma[1] 
  b_sig12 = b_Sigma[2] 
  b_sig13 = b_Sigma[3] 
  b_sig14 = b_Sigma[4] 
  b_sig22 = b_Sigma[5] 
  b_sig23 = b_Sigma[6] 
  b_sig24 = b_Sigma[7] 
  b_sig33 = b_Sigma[8] 
  b_sig34 = b_Sigma[9] 
  b_sig44 = b_Sigma[10] 
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
      dpx, dpy, dpz, lumin = par_sl_interaction_coupled4D(x, y, s, A, sig11, sig12, sig13, sig14, sig22, sig23, sig24, sig33, sig34, sig44, phi, x_slice, y_slice) where {T} 
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
function interact!(beam::BeamGPU{T}, elm::StrongBeamCoupled4D{T}) where {T}
  cross_angle = elm.cross_angle

  # test macro particle
  t_nmp = beam.nmp
  t_coords = beam.coords
  t_q = beam.q
  t_p0 = beam.p0

  # bunch 
  b_q = elm.q
  b_np = elm.np
 
  b_Sigma = elm.Sigma
  b_sig11 = b_Sigma[1] 
  b_sig12 = b_Sigma[2] 
  b_sig13 = b_Sigma[3] 
  b_sig14 = b_Sigma[4] 
  b_sig22 = b_Sigma[5] 
  b_sig23 = b_Sigma[6] 
  b_sig24 = b_Sigma[7] 
  b_sig33 = b_Sigma[8] 
  b_sig34 = b_Sigma[9] 
  b_sig44 = b_Sigma[10] 
  b_sigz = elm.sigz
  b_nslice = elm.nslice
  global b_sl_centroids = CuArray(elm.slice_centroids)

  # luminosity 
  global total_luminosity = CuArray([zero(T)])

  nb = ceil(Int, t_nmp/GLOBAL_BLOCK_SIZE)
  @cuda threads=GLOBAL_BLOCK_SIZE  blocks=nb  _gpu_interact_strong_beam_coupled4D!(t_coords, t_nmp, t_q, t_p0, b_q, b_np, b_sig11, b_sig12, b_sig13, b_sig14, b_sig22, b_sig23, b_sig24, b_sig33, b_sig34, b_sig44, b_sigz, b_nslice, b_sl_centroids, cross_angle, total_luminosity)

  return Array(total_luminosity)[1]
end

function _gpu_interact_strong_beam_coupled4D!(t_coords::CuDeviceVector{SVector{D,T},1}, t_nmp::Int, t_q::T, t_p0::T, b_q::T, b_np::T,
        b_sig11::T, b_sig12::T, b_sig13::T, b_sig14::T, b_sig22::T, b_sig23::T, b_sig24::T, b_sig33::T, b_sig34::T, b_sig44::T, b_sigz::T, 
        b_nslice::Int, b_sl_centroids::CuDeviceVector{SVector{3,T},1}, 
        cross_angle::T, luminosity_out::CuDeviceVector{T,1}) where {D,T}

  phi = cross_angle/2
  b_Q_slice = b_q*b_np/b_nslice
  # leading constant for beam-beam force from each slice
  Amp = t_q * b_Q_slice * e0 / (4*pi*epsilon0) / t_p0

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
      dpx, dpy, dpz, lumin = par_sl_interaction_coupled4D(x, y, s, Amp, b_sig11, b_sig12, b_sig13, b_sig14, b_sig22, b_sig23, b_sig24, b_sig33, b_sig34, b_sig44, cross_angle/2, x_slice, y_slice)

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

