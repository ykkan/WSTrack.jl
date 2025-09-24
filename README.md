# WSTrack

[![Build Status](https://github.com/ykkan/WSTrack.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ykkan/WSTrack.jl/actions/workflows/CI.yml?query=branch%3Amain)

This package provides parallelized weak-strong beam-beam simulation solvers.

## Installation
The package can be installed using Julia's REPL
```julia
julia> import Pkg
julia> Pkg.add(url="https://github.com/ykkan/WSTrack.jl.git")
```
or with Pkg mode (hitting `]` in the command prompt)
```julia
pkg> add https://github.com/ykkan/WSTrack.jl.git
```

## Example - EIC CDR Simulation for Proton (GPU simulation)
```julia
using WSTrack
using StaticArrays
using CUDA

# create test particles (proton) on GPU 
nmpar = 102400
beam_gpu = BeamGPU(
            sp=PROTON, 
            num_particle=0.6881e11, 
            energy=275.0e9, 
            num_macro_particle=nmpar,
            dist=Gaussian(emmx=11.3e-9, emmy=1.0e-9, emmz=3.96e-5, betx=0.8, bety=0.072, betz=90.909)
           )

# crate beam-beam kick beam element (electron beam)
emmx = 20e-9
emmy = 1.3e-9
betx = 0.45
bety = 0.056
sb = StrongBeamDecoupled(np=1.7203e11, sp=ELECTRON,
      sigx=sqrt(emmx*betx), sigpx=sqrt(emmx/betx), sigy=sqrt(emmy*bety), sigpy=sqrt(emmy/bety), sigz=0.7e-2,
      nslice=1, cross_angle=25e-3, f_crab=200.0e6)

# define linear one-turn map element
oneturn = OneTurn(betx=0.8, bety=0.072, alx=0.0, aly=0.0, 
                  Qx=0.228, Qy=0.210, Qs=0.01,
                  sigz=6e-2, sigpz=6.6e-4)


# define crab cavity elements
ccstrength = -0.0003876287347981857
cc1 = CrabCavity(f=200.0e6, kick_strength= ccstrength, strength_factors=SVector(4.0/3.0, -1.0/3.0), phi=0.0)
cc2 = CrabCavity(f=200.0e6, kick_strength= ccstrength, strength_factors=SVector(4.0/3.0, -1.0/3.0), phi=0.0)


# define two transfermap elements between ip and crab cavity
backward = @SMatrix [0 -32.2490309931942 0 0 0 0;
              0.031008683647302117 0 0 0 0 0;
                0 0 1 0 0 0;
                0 0 0 1 0 0;
                0 0 0 0 1 0;
                0 0 0 0 0 1;]


forward = @SMatrix [0 32.2490309931942 0 0 0 0;
              -0.031008683647302117 0 0 0 0 0;
                0 0 1 0 0 0;
                0 0 0 1 0 0;
                0 0 0 0 1 0;
                0 0 0 0 0 1;]

b = @SVector [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

ccip_backward = LinearMap(backward, b)
ccip_forward = LinearMap(forward, b)

nmp, cov = covariance(beam_gpu)
cov_printer(cov)
for i in 1:1000000
  interact!(beam_gpu, ccip_backward)
  interact!(beam_gpu, cc1)
  interact!(beam_gpu, ccip_forward)
  lumin = interact!(beam_gpu, sb)
  interact!(beam_gpu, ccip_forward)
  interact!(beam_gpu, cc2)
  interact!(beam_gpu, ccip_backward)
  interact!(beam_gpu, oneturn)
  nmp, cov = covariance(beam_gpu)
  if i%30 == 0
    # print limin and covariance of particles
    println(lumin)
    println(cov)
  end
end
```

