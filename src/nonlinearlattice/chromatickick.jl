export ChromaticKick

Base.@kwdef struct ChromaticKick{T}
  chromx::T
  chromy::T
  betx::T
  bety::T
  alfx::T
  alfy::T
  gamax::T
  gamay::T
end

function interact!(beam::Beam{T}, elm::ChromaticKick{T}) where {T}
  chromx =  elm.chromx
  chromy =  elm.chromy
  betx =  elm.betx
  bety =  elm.bety
  alfx =  elm.alfx
  alfy =  elm.alfy
  gamax =  elm.gamax
  gamay =  elm.gamay

  coords = beam.coords 
  Threads.@threads for i in 1:beam.npar
    x, px, y, py, z, pz = coords[i]
    phix = 2*pi*chromx*pz
    phiy = 2*pi*chromy*pz
    cx = cos(phix)
    cy = cos(phiy) 
    sx = sin(phix)
    sy = sin(phiy)
    jx = (betx*px^2 + 2*alfx*x*px + gamax*x^2)/2
    jy = (bety*py^2 + 2*alfy*y*py + gamay*y^2)/2

    x_new = x*(cx + alfx*sx) + px*betx*sx
    px_new = -x*gamax*sx + px*(cx - alfx*sx)
    y_new = y*(cy + alfy*sy) + py*bety*sy
    py_new = -y*gamax*sy + py*(cy - alfy*sy)
    z_new = z + 2*pi(chromx*jx + chromy*jy)
    pz_new = pz
    coords[i] = SVector{6,T}(x_new, px_new, y_new, py_new, z_new, pz_new)
  end
end

# gpu
function interact!(beam::BeamGPU{T}, elm::ChromaticKick{T}) where {T}
  chromx =  elm.chromx
  chromy =  elm.chromy
  betx =  elm.betx
  bety =  elm.bety
  alfx =  elm.alfx
  alfy =  elm.alfy
  gamax =  elm.gamax
  gamay =  elm.gamay

  npar = beam.npar
  coords = beam.coords
  nb = ceil(Int, npar/GLOBAL_BLOCK_SIZE)
  @cuda threads=GLOBAL_BLOCK_SIZE blocks=nb  _gpu_interact_chromatic_kick!(coords, npar, chromx, chromy, betx, bety, alfx, alfy, gamax, gamay)

end

function _gpu_interact_chromatic_kick!(
        coords::CuDeviceVector{SVector{6,T},1}, npar::Int, 
        chromx::T, chromy::T, betx::T, bety::T, 
        alfx::T, alfy::T, gamax::T, gamay::T) where {T}

  tid = threadIdx().x
  bid = blockIdx().x
  block_size = blockDim().x 
  gid = tid + (bid - 1) * block_size
  if gid <= npar
    x, px, y, py, z, pz = coords[gid]
    phix = 2*pi*chromx*pz
    phiy = 2*pi*chromy*pz
    cx = cos(phix)
    cy = cos(phiy) 
    sx = sin(phix)
    sy = sin(phiy)
    jx = (betx*px^2 + 2*alfx*x*px + gamax*x^2)/2
    jy = (bety*py^2 + 2*alfy*y*py + gamay*y^2)/2

    x_new = x*(cx + alfx*sx) + px*betx*sx
    px_new = -x*gamax*sx + px*(cx - alfx*sx)
    y_new = y*(cy + alfy*sy) + py*bety*sy
    py_new = -y*gamay*sy + py*(cy - alfy*sy)
    z_new = z + 2*pi*(chromx*jx + chromy*jy)
    pz_new = pz
    coords[gid] = SVector{6,T}(x_new, px_new, y_new, py_new, z_new, pz_new)
  end 
  return nothing
end


