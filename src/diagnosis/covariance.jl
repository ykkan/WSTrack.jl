export covariance

function covariance(beam::Beam{T}, filter=RectBound{T}()) where {T}
  coords = beam.coords

  function filtered_cov(x::Coord{T}) where {T} 
    x =  filter(x) ? x : zero(Coord{T})
    return (x * x')
  end
  n = count(beam; filter=filter)
  covsum = mapreduce(filtered_cov, +, coords)
  return (n , covsum/n)
end

function covariance(beam::BeamGPU{T}, filter=RectBound{T}()) where {T}
  coords = beam.coords

  function filtered_cov(x::Coord{T}) where {T} 
    x =  filter(x) ? x : zero(Coord{T})
    return (x * x')
  end
  n = count(beam; filter=filter)
  covsum = mapreduce(filtered_cov, +, coords)
  return (n , covsum/n)
end
