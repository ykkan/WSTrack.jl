export covariance

function covariance(beam::Beam{T}, filter=RectBound{T}()) where {T}
  npar=beam.npar
  coords = beam.coords

  function filtered_cov(x::Coord{T}) where {T} 
    x =  filter(x) ? x : zero(Coord{T})
    return (x * x')
  end
  cov = mapreduce(filtered_cov, +, coords) / npar
  return cov
end

function covariance(beam::BeamGPU{T}, filter=RectBound{T}()) where {T}
  npar=beam.npar
  coords = beam.coords

  function filtered_cov(x::Coord{T}) where {T} 
    x =  filter(x) ? x : zero(Coord{T})
    return (x * x')
  end
  cov = mapreduce(filtered_cov, +, coords) / npar
  return cov
end
