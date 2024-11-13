export covariance

function covariance(beam::Beam{T}, bound=coordmax(T)) where {T}
  npar=beam.npar
  coords = beam.coords

  cov = mapreduce(x-> all( abs.(x) .< bound ) ? x * x' : zero(SMatrix{6, 6, T, 36}), +, coords) / npar
  return cov
end

function covariance(beam::BeamGPU{T}, bound=coordmax(T)) where {T}
  npar=beam.npar
  coords = beam.coords

  cov = mapreduce(x-> all( abs.(x) .< bound ) ? x * x' : zero(SMatrix{6, 6, T, 36}), +, coords) / npar
  return cov
end


