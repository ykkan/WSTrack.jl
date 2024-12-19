function Base.count(beam::Beam{T}; filter=RectBound{T}()) where {T}
  coords = beam.coords
  return mapreduce(x -> filter(x) ? 1 : 0, +, coords)
end

function Base.count(beam::BeamGPU{T}; filter=RectBound{T}()) where {T}
  coords = beam.coords
  return mapreduce(x -> filter(x) ? 1 : 0, +, coords)
end
