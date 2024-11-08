export RFKick

Base.@kwdef struct RFKick{T}
  amp::T
  k::T
  phi::T
end

function interact!(beam::Beam{T}, elm::RFKick{T}) where {T}
  amp = elm.amp
  k = elm.k
  phi = elm.phi

  coords = beam.coords 
  Threads.@threads for i in 1:beam.npar
    x, px, y, py, z, pz = coords[i]
    pz_new = amp*sin(k*z + phi)
    coords[i] = SVector{6,T}(x, px, y, py, z, pz_new)
  end
end

# gpu
function interact!(beam::BeamGPU{T}, elm::RFKick{T}) where {T}
  amp = elm.amp
  k = elm.k
  phi = elm.phi

  npar = beam.npar
  coords = beam.coords
  nb = ceil(Int, npar/GLOBAL_BLOCK_SIZE)
  @cuda threads=GLOBAL_BLOCK_SIZE blocks=nb  _gpu_interact_rf_kick!(coords, npar, amp, k, phi)
end

function _gpu_interact_rf_kick!(
        coords::CuDeviceVector{SVector{6,T},1}, npar::Int, 
        amp::T, k::T, phi::T) where T

  tid = threadIdx().x
  bid = blockIdx().x
  block_size = blockDim().x 
  gid = tid + (bid - 1) * block_size
  if gid <= npar
    x, px, y, py, z, pz = coords[gid]
    pz_new = amp*sin(k*z + phi)
    coords[gid] = SVector{6,T}(x, px, y, py, z, pz_new)
  end 
  return nothing
end


