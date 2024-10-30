export Gaussian

struct Gaussian{T} <: Distribution{T}
  mu::AbstractVector{T}
  sig::AbstractPDMat{T}
  rseed::Int
end

function Gaussian(mu::AbstractVector{T}, sig::AbstractMatrix{T}, rseed=123456789) where {T}
  return Gaussian(mu, PDMat(sig), rseed)
end

function Gaussian(sig::AbstractMatrix{T}, rseed=123456789) where {T}
    return Gaussian(zeros(T,size(sig)[1]), sig, rseed)
end

function Gaussian(;emmx::T, emmy::T, emmz::T, betx::T, bety::T, betz::T, alfx=0.0, alfy=0.0, alfz=0.0, mu=zeros(6), rseed=123456789) where {T}
    gax = (1 + alfx^2)/betx
    gay = (1 + alfy^2)/bety
    gaz = (1 + alfz^2)/betz

    sig = [emmx*betx -emmx*alfx 0.0 0.0 0.0 0.0;
           -emmx*alfx emmx*gax 0.0 0.0 0.0 0.0;
           0.0 0.0 emmy*bety -emmy*alfy 0.0 0.0
           0.0 0.0 -emmy*alfy emmy*gay 0.0 0.0;  
           0.0 0.0 0.0 0.0 emmz*betz -emmz*alfz; 
           0.0 0.0 0.0 0.0 -emmz*alfz emmz*gaz]
    return Gaussian(mu, sig, rseed)
end

function (dist::Gaussian{T})(npar::Int) where {T}
  d = MvNormal(dist.mu, dist.sig)
  Random.seed!(dist.rseed)
  coords = Vector{SVector{6,T}}(undef, npar)
  for i in 1:npar
    x, px, y, py, z, pz = rand(d)
    coords[i] = SVector{6,T}(x, px, y, py, z, pz)
  end
  return coords
end
