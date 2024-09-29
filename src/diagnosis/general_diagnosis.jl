export GeneralDiagnosis
export update!

Base.@kwdef mutable struct GeneralDiagnosis{T}
  every::Int = 1
  step::Int = 0
  sigmas::SVector{6,T} = SVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  emmitances::SVector{3,T} = SVector(0.0, 0.0, 0.0)
  io_emm::IOStream = open("emmitance.txt", "w")
  io_sig::IOStream = open("sigma.txt", "w")
end

function update!(diag::GeneralDiagnosis{T}; sigmas::SVector{6,T}, emmitances::SVector{3,T}) where {T}
  diag.sigmas = sigmas
  diag.emmitances = emmitances
  if diag.step % diag.every == 0
    @printf(diag.io_emm, "%d %.15e %.15e %.15e\n", diag.step, emmitances[1], emmitances[2], emmitances[3])
    @printf(diag.io_sig, "%d %.15e %.15e %.15e %.15e %.15e %.15e\n", diag.step, sigmas[1], sigmas[2], sigmas[3], sigmas[4], sigmas[5], sigmas[6])
  end
  diag.step += 1
end


function interact!(beam::Beam, diag::GeneralDiagnosis)
  npar=beam.npar
  coords = beam.coords
  #mux, mupx, muy, mupy, muz, mupz = mu
  mu2 = SVector(0.0,0.0,0.0,0.0,0.0,0.0)
  cov = SVector(0.0,0.0,0.0)
  for i in 1:npar
    vec = coords[i]
    mu2 = mu2 + vec.^2
    cov = cov + SVector(vec[1]*vec[2], vec[3]*vec[4], vec[5]*vec[6])
  end
  mu2 = mu2/npar
  cov = cov/npar

  sig = sqrt.(mu2)
  emm = sqrt.( SVector(mu2[1]*mu2[2]-cov[1]^2, mu2[3]*mu2[4] - cov[2]^2, mu2[5]*mu2[6] - cov[3]^2) )

  update!(diag; emmitances=emm, sigmas=sig)
end

function interact!(beam::BeamGPU, diag::GeneralDiagnosis)
  npar=beam.npar
  coords = beam.coords

  mu2 = mapreduce(x -> SVector(x[1]^2, x[2]^2, x[3]^2, x[4]^2, x[5]^2, x[6]^2), +, coords)
  cov = mapreduce(x -> SVector(x[1]*x[2], x[3]*x[4], x[5]*x[6]), +, coords)
  mu2 = mu2/npar
  cov = cov/npar

  sig = sqrt.(mu2)
  emm = SVector(sqrt(mu2[1]*mu2[2]-cov[1]^2), 
                  sqrt(mu2[3]*mu2[4] - cov[2]^2), 
                  sqrt(mu2[5]*mu2[6] - cov[3]^2))

  update!(diag; emmitances=emm, sigmas=sig)
end
