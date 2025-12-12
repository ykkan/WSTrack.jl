export IBSConstantRate

struct IBSConstantRate{T,FT<:Filter{T}}
  T_rev::T
  rate_x::T
  rate_y::T
  rate_z::T
  filter::FT
end

function IBSConstantRate(;rate_x::T, rate_y::T, rate_z::T, filter::Filter{T}=RectBound{Float64}()) where {T} 
  return IBSConstantRate(T_rev, rate_x, rate_y, rate_z, filter)  
end

function interact!(beam::BeamGPU{T}, ele::IBSConstantRate{T,FT}, nturn::Int=1) where {T,FT}
  filter = ele.filter
  nmp_alive, cov = covariance(beam, filter) 
  sigx = sqrt(cov[1,1])  
  sigpx =sqrt(cov[2,2])
  sigy = sqrt(cov[3,3])
  sigpy = sqrt(cov[4,4])
  sigz = sqrt(cov[5,5])
  sigpz = sqrt(cov[6,6])
  emmx = sqrt(cov[1,1]*cov[2,2] - cov[1,2]^2)
  emmy = sqrt(cov[3,3]*cov[4,4] - cov[3,4]^2)
 
  nmp = beam.nmp
  coords = beam.coords
  
  T_rev = ele.T_rev
  demmx2emmx = T_rev * ele.rate_x
  demmy2emmy = T_rev * ele.rate_y
  demmz2emmz = T_rev * ele.rate_z

  demmx2emmx_nturn = (1 + demmx2emmx)^nturn - 1
  demmy2emmy_nturn = (1 + demmy2emmy)^nturn - 1
  demmz2emmz_nturn = (1 + demmz2emmz)^nturn - 1

  w_sum = mapreduce(r -> filtered_w(r, filter, sigx, sigy, sigz), +, coords)

  nb = ceil(Int, nmp/GLOBAL_BLOCK_SIZE)
  @cuda threads=GLOBAL_BLOCK_SIZE blocks=nb _gpu_bruce_kick_xyzmod(coords, nmp, nmp_alive, demmx2emmx_nturn, demmy2emmy_nturn, demmz2emmz_nturn, w_sum, sigx, sigy, sigz, sigpx, sigpy, sigpz, filter)
  return nmp_alive, cov, demmx2emmx_nturn, demmy2emmy_nturn, demmz2emmz_nturn
end
