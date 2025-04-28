export RadiationDamp

struct RadiationDamp{T}
  lambdax::T
  lambday::T
  lambdaz::T
  sigx::T
  sigy::T
  sigz::T
  sigpx::T
  sigpy::T
  sigpz::T
end

function RadiationDamp(;emmx::T, emmy::T, emmz::T, betx::T, bety::T, betz::T, taux::T, tauy::T, tauz::T,alfx=0.0, alfy=0.0, alfz=0.0) where {T}
  gax = (1 + alfx^2)/betx
  gay = (1 + alfy^2)/bety
  gaz = (1 + alfz^2)/betz

  lambdax = exp(-1.0/taux)
  lambday = exp(-1.0/tauy)
  lambdaz = exp(-1.0/tauz)
  sigx = sqrt(emmx*betx)
  sigy = sqrt(emmy*bety)
  sigz = sqrt(emmz*betz)
  sigpx = sqrt(emmx*gax)
  sigpy = sqrt(emmy*gay)
  sigpz = sqrt(emmz*gaz)
  return RadiationDamp(lambdax, lambday, lambdaz, sigx, sigy, sigz, sigpx, sigpy,sigpz)
end

function interact!(beam::Beam{T}, elm::RadiationDamp{T}) where {T}
  lx = elm.lambdax 
  ly = elm.lambday
  lz = elm.lambdaz 
  sigx = elm.sigx
  sigy = elm.sigy
  sigz = elm.sigz
  sigpx = elm.sigpx
  sigpy = elm.sigpy
  sigpz = elm.sigpz

  slx = sqrt(1 - lx^2)
  sly = sqrt(1 - ly^2)
  slz = sqrt(1 - lz^2)

  nmp = beam.nmp
  coords = beam.coords
  d = Normal()
  Threads.@threads for i in 1:nmp
    x, px, y, py, z, pz = coords[i]
    x_new = lx*x + rand(d)*sigx*slx
    px_new = lx*px + rand(d)*sigpx*slx
    y_new = ly*y + rand(d)*sigy*sly
    py_new = ly*py + rand(d)*sigpy*sly
    z_new = lz*z + rand(d)*sigz*slz
    pz_new = lz*pz + rand(d)*sigpz*slz
    coords[i] = SVector{6,T}(x_new, px_new, y_new, py_new, z_new, pz_new)
  end
end

# gpu
function interact!(beam::BeamGPU{T}, elm::RadiationDamp{T}) where {T}
  lx = elm.lambdax 
  ly = elm.lambday
  lz = elm.lambdaz 
  sigx = elm.sigx
  sigy = elm.sigy
  sigz = elm.sigz
  sigpx = elm.sigpx
  sigpy = elm.sigpy
  sigpz = elm.sigpz

  slx = sqrt(1 - lx^2)
  sly = sqrt(1 - ly^2)
  slz = sqrt(1 - lz^2)

  nmp = beam.nmp
  coords = beam.coords
  nb = ceil(Int, nmp/GLOBAL_BLOCK_SIZE)
  @cuda threads=GLOBAL_BLOCK_SIZE blocks=nb _gpu_interact_radiation_damp!(coords, nmp, lx, ly, lz, slx, sly, slz, sigx, sigy, sigz, sigpx, sigpy, sigpz)
end

function _gpu_interact_radiation_damp!(coords::CuDeviceVector{SVector{6,T},1}, nmp::Int, lx::T, ly::T, lz::T, slx::T, sly::T, slz::T, sigx::T, sigy::T, sigz::T, sigpx::T, sigpy::T, sigpz::T) where {T}

  d = Normal()
  tid = threadIdx().x
  bid = blockIdx().x
  block_size = blockDim().x 
  gid = tid + (bid - 1) * block_size
  if gid <= nmp
    x, px, y, py, z, pz = coords[gid]
    x_new = lx*x + rand(d)*sigx*slx
    px_new = lx*px + rand(d)*sigpx*slx
    y_new = ly*y + rand(d)*sigy*sly
    py_new = ly*py + rand(d)*sigpy*sly
    z_new = lz*z + rand(d)*sigz*slz
    pz_new = lz*pz + rand(d)*sigpz*slz
    coords[gid] = SVector{6,T}(x_new, px_new, y_new, py_new, z_new, pz_new)
  end
  return nothing
end
