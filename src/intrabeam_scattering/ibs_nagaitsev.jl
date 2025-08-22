export IBSNagaitsev

struct IBSNagaitsev{T,SV<:AbstractVector{SVector{7,T}},FT<:Filter{T}}
  T_rev::T
  lattice_optics::SV
  filter::FT
end

function IBSNagaitsev(;T_rev::T, optics_filename::String, n::Int=4096, filter::Filter{T}=RectBound{Float64}()) where {T} 
  opt_mat = readdlm(optics_filename)
  funcs = opticsfuncs(opt_mat)
  opt = CuArray( optics_quadrature(funcs, gausslegendre(n)) )
  return IBSNagaitsev(T_rev, opt, filter)  
end

function opticsfuncs(data::Matrix{T}) where {T}
  n, m = size(data)
  s = data[:,1]
  smin, smax = s[1], s[end]
  @. s = (2*s - smax - smin)/(smax - smin)
  funcs = [AkimaInterpolation(data[:,j], s,) for j in 2:m] 
  return funcs
end

function optics_quadrature(funcs::AbstractVector, qd::Tuple{Vector{T},Vector{T}}) where {T}
  m = length(funcs) + 1
  node = qd[1] 
  w = qd[2]
  n = length(w)
  data = zeros(n, m)
  data[:,1] = w
  for j in 2:m
    data[:,j] .= funcs[j-1].(node)
  end
  output = zeros(SVector{7,T},n)
  for i in 1:n
    output[i] = SVector{7,T}(data[i,:]...)
  end
  return output
end

function filtered_w(coord::Coord{T}, filter::Filter{T}, sigz::T) where {T}
  val =  filter(coord) ? exp(-coord[5]^2/sigz^2/2) : zero(T)
  return val 
end

function interact!(beam::BeamGPU{T}, ele::IBSNagaitsev{T,SV,FT}) where {T,SV,FT}
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
  lattice_optics = ele.lattice_optics
 
  nmp = beam.nmp
  coords = beam.coords
  np2nmp = beam.np2nmp 
  np_alive = nmp_alive*np2nmp
  sp_q = beam.q / np2nmp
  sp_m = beam.m / np2nmp
  sp_p0 = beam.p0 / np2nmp
  gamma = (sp_p0/sp_m) + 1
  ibs_rate_x, ibs_rate_y, ibs_rate_z = ibs_rate_nagaitsev(sp_q, sp_m, np_alive, emmx, emmy, sigz, sigpz, gamma, lattice_optics)
  T_rev = ele.T_rev
  demmx2emmx = T_rev * ibs_rate_x
  demmy2emmy = T_rev * ibs_rate_y
  dsigpzsq2sigpzsq = T_rev * ibs_rate_z

  w_sum = mapreduce(r -> filtered_w(r, filter, sigz), +, coords)

  nb = ceil(Int, nmp/GLOBAL_BLOCK_SIZE)
  @cuda threads=GLOBAL_BLOCK_SIZE blocks=nb _gpu_bruce_kick(coords, nmp, nmp_alive, demmx2emmx, demmy2emmy, dsigpzsq2sigpzsq, w_sum, sigz, sigpx, sigpy, sigpz, filter)
  return nmp, cov, demmx2emmx, demmy2emmy, dsigpzsq2sigpzsq
end


function _gpu_bruce_kick(coords::CuDeviceVector{SVector{6,T},1}, nmp::Int, nmp_alive::Int, demmx2emmx::T, demmy2emmy::T, dsigpzsq2sigpzsq::T, w_sum::T, sigz::T, sigpx::T, sigpy::T, sigpz::T, filter::Filter{T}) where {T}
  d = Normal()
  tid = threadIdx().x
  bid = blockIdx().x
  block_size = blockDim().x 
  gid = tid + (bid - 1) * block_size
  A = sqrt(2*nmp/w_sum)
  if gid <= nmp
    coord = coords[gid]
    x, px, y, py, z, pz = coord
    w = filtered_w(coord, filter, sigz)
    px_new = px + rand(d)*sigpx*A*sqrt(w*demmx2emmx)
    py_new = py + rand(d)*sigpy*A*sqrt(w*demmy2emmy)
    pz_new = pz + rand(d)*sigpz*A*sqrt(w*dsigpzsq2sigpzsq)
    coords[gid] = SVector{6,T}(x, px_new, y, py_new, z, pz_new)
  end
  return nothing
end

function carlson_rd(x::T, y::T, z::T, n=10) where {T}
  total = zero(T)
  xn = x
  yn = y
  zn = z
  An = zero(T)
  Xn = zero(T)
  Yn = zero(T)
  Zn = zero(T)
  fn = one(T)
  for i in 1:n
    # step n 
    ln = sqrt(xn * yn) + sqrt(xn * zn) + sqrt(yn * zn)
    total = total + fn/(sqrt(zn)*(zn + ln))
    # step n + 1
    fn = fn / 4
    xn = (xn + ln)/4
    yn = (yn + ln)/4
    zn = (zn + ln)/4
    An = (An + ln)/4
    Xn = 1 - xn/An
    Yn = 1 - yn/An
    Zn = 1 - zn/An
   end
   E2 = Xn*Yn - 6*Zn^2
   E3 = (3*Xn*Yn - 8*Zn^2)*Zn
   E4 = 3*(Xn*Yn - Zn^2)*Zn^2
   E5 = Xn*Yn*Zn^3
   return 3*total + fn*(1 - 3*E2/14 + E3/6 + 9*E2^2/88 - 3*E4/22 - 9*E2*E3/52 + 3*E5/26)/(sqrt(An)*An)
end


# the domain is transformed to [-1, 1]
function _ibs_nagaitsev_partial_integral(rc::T, emmx::T, emmy::T, sigpz::T, gamma::T, w::T, betx::T, alfx::T, dx::T, dpx::T, bety::T, alfy::T) where {T}
  phi = dpx + alfx*dx/betx
  sigx = sqrt(dx^2*sigpz^2 + emmx*betx)
  sigy = sqrt(emmy*bety)
  ax = betx/emmx
  ay = bety/emmy
  as = ax*(dx^2/betx^2 + phi^2) + 1/sigpz^2 
  a1 = (ax + gamma^2*as)/2
  a2 = (ax - gamma^2*as)/2
  l1 = ay
  C = sqrt(a2^2 + (gamma*ax*phi)^2)
  l2 = a1 + C
  l3 = a1 - C
  
  r1 = carlson_rd(1/l2, 1/l3, 1/l1)
  r2 = carlson_rd(1/l3, 1/l1, 1/l2)
  r3 = 3*sqrt(l1*l2*l3) - r1 - r2
  R1 = r1/l1
  R2 = r2/l2
  R3 = r3/l3
  psi = -2*R1 + R2 + R3

  Sp = gamma^2/2*(2*R1 - R2*(1 - 3*a2/C) - R3*(1 + 3*a2/C))
  Sx = (2*R1 - R2*(1 + 3*a2/C) - R3*(1 - 3*a2/C))/2
  Sxp = 3*gamma^2*phi^2*ax/C*(R3 - R2)
 
  Lc = log(2*sigy*(sigpz^2 + gamma^2*(emmx/betx + emmy/bety))/rc)

  ix = w*Lc*( betx/(2*sigx*sigy))*(Sx + ((dx/betx)^2 + phi^2)*Sp + Sxp) * 0
  iy = w*Lc*( bety/(2*sigx*sigy)*psi ) * 0
  iz = w*Lc*( 1/(2*sigx*sigy)*Sp ) * 0
  @cuprint Sp, Sx, Sxp
  return SVector{3,T}(Sp, Sx, Sxp) #SVector{3,T}(Ix/emmx, Iy/emmy, Iz/sigpz^2)
end

function ibs_rate_nagaitsev(q::T, m::T, np::T, emmx::T, emmy::T, sigz::T, sigpz::T, gamma::T, optics) where {T}
  rc = e0/(4*pi*epsilon0)*q^2/m
  be = sqrt(1 - 1/gamma^2)
  A = np*rc^2*c0/(12*pi*be^3*gamma^5*sigz)
  I = mapreduce(x_vec -> _ibs_nagaitsev_partial_integral(rc, emmx, emmy, sigpz, gamma, x_vec...), +, optics)
  return A*I
end




