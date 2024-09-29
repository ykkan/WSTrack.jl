export Drift

function Drift(;L=0.0) 
  A = @SMatrix [1 L 0 0 0 0;
                0 1 0 0 0 0;
                0 0 1 L 0 0;
                0 0 0 1 0 0;
                0 0 0 0 1 0;
                0 0 0 0 0 1;
                ]

  b = @SVector [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  return LinearMap(A, b)
end
