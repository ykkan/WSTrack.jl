export Gaussian

Base.@kwdef struct Gaussian{T} <: Distribution{T}
  emmx::T
  emmy::T
  betx::T
  bety::T
  sigz::T
  sigpz::T
  mux::T=0.0
  muy::T=0.0
  muz::T=0.0
  mupx::T=0.0
  mupy::T=0.0
  mupz::T=0.0
  rseed::Int=123456789
end

function (dist::Gaussian{T})(npar::Int) where {T}
  emmx = dist.emmx
  emmy = dist.emmy
  betx = dist.betx
  bety = dist.bety
  sigz = dist.sigz
  sigpz = dist.sigpz
  mux = dist.mux
  muy = dist.muy
  muz = dist.muz
  mupx = dist.mupx
  mupy = dist.mupy
  mupz = dist.mupz
  
  gax = 1.0/betx
  gay = 1.0/bety
  alx = 0.0
  aly = 0.0
  cx1 = sqrt(emmx*betx) 
  cx2 = -sqrt(emmx/betx)*alx
  cx3 = sqrt(emmx/betx)
  cy1 = sqrt(emmy*bety) 
  cy2 = -sqrt(emmy/bety)*aly
  cy3 = sqrt(emmy/bety)

  d = Normal(0.0, 1.0)
  Random.seed!(dist.rseed)
  coords = Vector{SVector{6,T}}(undef, npar)
  for i in 1:npar
    vecx = rand(d,2)
    vecy = rand(d,2)
    vecz = rand(d,2)
    x = cx1*vecx[1] + mux
    y = cy1*vecy[1] + muy
    z = sigz*vecz[1] + muz
    px = (cx2*vecx[1] + cx3*vecx[2]) + mupx
    py = (cy2*vecy[1] + cy3*vecy[2]) + mupy
    pz = sigpz*vecz[2] + mupz
    coords[i] = SVector(x, px, y, py, z, pz)
  end
  return coords
end
