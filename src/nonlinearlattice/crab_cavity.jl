export CrabCavity

struct CrabCavity{T,N}
  kick_strength::T
  k0::T
  phi::T
  strength_factors::SVector{N,T}
end

# f: frequency (Hz)
# strength: theta_c / sqrt(beta_star * beta_cc)
# phi: phase (rad)
function CrabCavity(; f, kick_strength, strength_factors, phi)
  k0 = 2*pi*f/c0
  return CrabCavity(kick_strength, k0, phi, strength_factors)
end

function interact!(beam::Beam{T}, elm::CrabCavity{T,N}) where {T,N}
  kick_strength = elm.kick_strength
  k0 = elm.k0
  phi = elm.phi
  strength_factors = elm.strength_factors
  coords = beam.coords 
  Threads.@threads for i in 1:beam.npar
    x, px, y, py, z, pz = coords[i]
    for n in 1:N
      k = k0 * n
      phase = k*z + phi
      amp = strength_factors[n] * kick_strength
      px = px + amp * sin(phase)/k
      pz = pz + amp * cos(phase)*x
    end
    coords[i] = SVector{6,T}(x, px, y, py, z, pz)
  end
end

# gpu
function interact!(beam::BeamGPU{T}, elm::CrabCavity{T,N}) where {T,N}
  kick_strength = elm.kick_strength
  k0 = elm.k0
  phi = elm.phi
  strength_factors = elm.strength_factors
  npar = beam.npar
  coords = beam.coords
  nb = ceil(Int, npar/GLOBAL_BLOCK_SIZE)
  @cuda threads=GLOBAL_BLOCK_SIZE blocks=nb _gpu_interact_crab_cavity!(coords, npar, kick_strength, k0, phi, strength_factors)
end

function _gpu_interact_crab_cavity!(
        coords::CuDeviceVector{SVector{6,T},1}, npar::Int, 
        kick_strength::T, k0::T, phi::T, strength_factors::SVector{N,T}) where {N,T}

  tid = threadIdx().x
  bid = blockIdx().x
  block_size = blockDim().x 
  gid = tid + (bid - 1) * block_size
  if gid <= npar
    x, px, y, py, z, pz = coords[gid]
    for n in 1:N
      k = k0 * n
      phase = k*z + phi
      a = strength_factors[n] * kick_strength
      px = px + a*sin(phase)/k
      pz = pz + a*cos(phase)*x
    end
    coords[gid] = SVector{6,T}(x, px, y, py, z, pz)
  end
  return nothing
end
