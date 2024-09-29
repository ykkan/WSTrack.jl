include("distribution/distribution.jl")

export Beam
export BeamGPU

# q: charge of a macro particle (e0)
# m: mass of a macro particle (eV)
# p0: momentum of a reference macro particle (eV/C)
# coords: ( x, px/P0, y, py/P0, z, (p-P0)/P0 ) x, y, z (m)
struct Beam{T} 
  npar::Int
  q::T
  m::T
  p0::T
  coords::Vector{SVector{6,T}}
end

function Beam(;sp::ChargedSpecie{T}, num_particle::T, energy::T, num_macro_particle::Int, dist::Distribution{T}) where {T}
  npar2nmpar = num_particle/num_macro_particle
  q = npar2nmpar*sp.q
  m = npar2nmpar*sp.m
  p0 = npar2nmpar*sqrt(energy^2 - sp.m^2)
  coords = dist(num_macro_particle)
  return Beam(num_macro_particle, q, m, p0, coords)
end

struct BeamGPU{T}
  npar::Int
  q::T
  m::T
  p0::T
  coords::CuVector{SVector{6,T}}
end

function BeamGPU(;sp::ChargedSpecie{T}, num_particle::T, energy::T, num_macro_particle::Int, dist::Distribution{T}) where {T}
  npar2nmpar = num_particle/num_macro_particle
  q = npar2nmpar*sp.q
  m = npar2nmpar*sp.m
  p0 = npar2nmpar*sqrt(energy^2 - sp.m^2)
  coords = CuArray(dist(num_macro_particle))
  return BeamGPU(num_macro_particle, q, m, p0, coords)
end
