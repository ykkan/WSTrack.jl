export LinearMap

struct LinearMap{D,T}
  A::SMatrix{D,D,T}
  b::SVector{D,T}
end

function Base.:*(m1::LinearMap{D,T}, m2::LinearMap{D,T}) where {D,T}
  return LinearMap{D,T}(m1.A * m2.A, m1.A * m2.b + m2.b)
end

# cpu 
function interact!(beam::Beam, elm::LinearMap)
  nmp = beam.nmp
  coords = beam.coords
  A = elm.A
  b = elm.b
  Threads.@threads for i in 1:nmp
    coords[i] = A*coords[i] + b
  end
end

# gpu
function interact!(beam::BeamGPU, elm::LinearMap)
  nmp = beam.nmp
  coords = beam.coords
  nb = ceil(Int, nmp/GLOBAL_BLOCK_SIZE) 
  @cuda blocks=nb threads=GLOBAL_BLOCK_SIZE _gpu_interact_linearmap!(coords, nmp, elm.A, elm.b)
end

function _gpu_interact_linearmap!(
        coords::CuDeviceVector{SVector{D,T},1}, nmp::Int, A::SMatrix{D,D,T}, 
        b::SVector{D,T}) where {D,T}

  tid = threadIdx().x
  bid = blockIdx().x
  block_size = blockDim().x 
  gid = tid + (bid - 1) * block_size
  if gid <= nmp
    coords[gid] = A*coords[gid] + b
  end

  return nothing
end
