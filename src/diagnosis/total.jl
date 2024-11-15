export total

function total(beam::Beam{T}, mask=RectMask{T}()) where {T}
  coords = beam.coords
  return mapreduce(x -> mask(x) ? 1 : 0, +, coords)
end

function total(beam::BeamGPU{T}, mask=RectMask{T}()) where {T}
  coords = beam.coords
  return mapreduce(x -> mask(x) ? 1 : 0, +, coords)
end
