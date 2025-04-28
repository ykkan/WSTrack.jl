include("distribution/distribution.jl")

export Beam
export BeamGPU


# q: charge of a macro particle (e0)
# m: mass of a macro particle (eV)
# p0: momentum of a reference macro particle (eV/C)
# coords: ( x, px/P0, y, py/P0, z, (p-P0)/P0 ) x, y, z (m)
struct Beam{T} 
  nmp::Int
  np2nmp::T
  q::T
  m::T
  p0::T
  coords::Vector{Coord{T}}
end

function Beam(;sp::ChargedSpecie{T}, num_particle::T, energy::T, num_macro_particle::Int, dist::Distribution{T}) where {T}
  np2nmp = num_particle/num_macro_particle
  q = np2nmp*sp.q
  m = np2nmp*sp.m
  p0 = np2nmp*sqrt(energy^2 - sp.m^2)
  coords = dist(num_macro_particle)
  return Beam(num_macro_particle, np2nmp, q, m, p0, coords)
end

struct BeamGPU{T}
  nmp::Int
  np2nmp::T
  q::T
  m::T
  p0::T
  coords::CuVector{Coord{T}}
end

function BeamGPU(beam::Beam{T}) where {T}
  nmp = beam.nmp
  np2nmp = beam.np2nmp
  q = beam.q
  m = beam.m
  p0 = beam.p0
  coords = beam.coords
  return BeamGPU(nmp, np2nmp, q, m, p0, CuArray(coords))
end

function BeamGPU(;sp::ChargedSpecie{T}, num_particle::T, energy::T, num_macro_particle::Int, dist::Distribution{T}) where {T}
  np2nmp = num_particle/num_macro_particle
  q = np2nmp*sp.q
  m = np2nmp*sp.m
  p0 = np2nmp*sqrt(energy^2 - sp.m^2)
  coords = CuArray(dist(num_macro_particle))
  return BeamGPU(num_macro_particle, np2nmp, q, m, p0, coords)
end
