export covariance

function covariance(beam::Beam{T}, mask=RectMask{T}()) where {T}
  npar=beam.npar
  coords = beam.coords

  function masked_cov(x::Coord{T}) where {T} 
    x =  mask(x) ? x : zero(Coord{T})
    return (x * x')
  end
  cov = mapreduce(masked_cov, +, coords) / npar
  return cov
end

function covariance(beam::BeamGPU{T}, mask=RectMask{T}()) where {T}
  npar=beam.npar
  coords = beam.coords

  function masked_cov(x::Coord{T}) where {T} 
    x =  mask(x) ? x : zero(Coord{T})
    return (x * x')
  end
  cov = mapreduce(masked_cov, +, coords) / npar
  return cov
end
