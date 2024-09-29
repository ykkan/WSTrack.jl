export OneTurn

function OneTurn(;betx::T, bety::T, alx::T=0.0, aly::T=0.0, 
                  Qx::T, Qy::T, Qs::T,
                  sigz::T, sigpz::T) where {T}
  gax = (1 + alx^2)/betx
  gay = (1 + aly^2)/bety

  x11 = cos(2*pi*Qx) + alx*sin(2*pi*Qx)
  x12 = betx*sin(2*pi*Qx)
  x21 = -gax*sin(2*pi*Qx)
  x22 = cos(2*pi*Qx) - alx*sin(2*pi*Qx)

  y11 = cos(2*pi*Qy) + aly*sin(2*pi*Qy)
  y12 = bety*sin(2*pi*Qy)
  y21 = -gay*sin(2*pi*Qy)
  y22 = cos(2*pi*Qy) - alx*sin(2*pi*Qy)

  z11 = cos(2*pi*Qs)
  z12 = sigz/sigpz*sin(2*pi*Qs)
  z21 = -sigpz/sigz*sin(2*pi*Qs)
  z22 = cos(2*pi*Qs)
  A = @SMatrix [x11 x12 0 0 0 0;
                x21 x22 0 0 0 0;
                0 0 y11 y12 0 0;
                0 0 y21 y22 0 0;
                0 0 0 0 z11 z12;
                0 0 0 0 z21 z22;]

  b = @SVector [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  return LinearMap(A, b)
end
