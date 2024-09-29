export LuminosityDiagnosis
export update!

Base.@kwdef mutable struct LuminosityDiagnosis{T}
  every::Int = 1
  step::Int = 1
  luminosity::T = 0.0
  io_lumin::IOStream = open("luminosity.txt", "w")
end

function update!(diag::LuminosityDiagnosis{T}; luminosity::T) where {T}
  diag.luminosity = luminosity
  if diag.step % diag.every == 0
    @printf(diag.io_lumin, "%d %.15e\n", diag.step, diag.luminosity)
  end
  diag.step += 1
end
