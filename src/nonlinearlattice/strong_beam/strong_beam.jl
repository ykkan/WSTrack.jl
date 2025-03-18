include("utility.jl")
include("strong_beam_decoupled.jl")
include("strong_beam_coupled4D.jl")

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

