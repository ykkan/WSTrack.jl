export Polynomial
export TaylorMap


struct Polynomial{N,I,T} 
  n::I
  coeffs::Vector{T}
  exponents::Vector{NTuple{N,I}}
end

function Polynomial(coeffs, exponents)
  n = length(coeffs)
  if n != length(exponents)
    error("length of `coeff` and `expoents` mismatch")
  end
  Polynomial(n, coeffs, exponents)
end

function __polynomial_term_ex(coeff::T, exponent::NTuple{N,I}, variable::NTuple{N,Symbol}) where {N,I,T}
  if all(iszero, exponent)
    return :($coeff)
  else
    ex = :($coeff)
    for i in 1:N
      p = exponent[i]
      if p != 0
        var = variable[i]
        ex = :( $ex * $(var)^$p)
      end
    end
    return ex
  end
end

function __polynomial_ex(polynomial::Polynomial{N,I,T}, variable::NTuple{N,Symbol}) where {N,I,T}
  n = polynomial.n
  coeffs = polynomial.coeffs
  exponents = polynomial.exponents
  ex = :( $(__polynomial_term_ex(coeffs[1],exponents[1],variable)) ) 
  for i in 2:n
      ex = :( $ex + $(__polynomial_term_ex(coeffs[i],exponents[i],variable)) ) 
  end
  return ex
end

function __taylormap_ex(polynomials::NTuple{N,Polynomial{N,I,T}}, new_variable::NTuple{N,Symbol}, variable::NTuple{N,Symbol}) where {N,I,T}
  ex_lines = Vector{Expr}(undef,N)
  for i in 1:N
    rhs = :( $(__polynomial_ex(polynomials[i], variable)) )
    ex_lines[i] = :( $(new_variable[i]) = $rhs )
  end
  return Expr(:block, ex_lines...) 
end


###
#
__gen_symbol_seq(s::String, n) = ntuple(i -> Symbol(s, i), n)

function TaylorMap(polynomials::NTuple{N,Polynomial{N,I,T}}) where {N,I,T}
  if N != 6
    error("number of polynomials and variables should be both 6")
  end
  map_name = gensym(:TaylorMap)
  eval( :(struct $(map_name) end) )

  xvars = __gen_symbol_seq("x", N)
  yvars = __gen_symbol_seq("y", N)
  sv_ex = :(SVector{$N,$T})
  cpu_func = quote
    function interact!(beam::Beam{$T}, elm::$(map_name))
      nmp = beam.nmp
      coords = beam.coords
      Threads.@threads for i in 1:nmp
        $( Expr(:tuple, xvars...) ) = coords[i]
        $( __taylormap_ex(polynomials, yvars, xvars) )
        coords[i] = $( Expr(:call, sv_ex, yvars...) )
      end
    end
  end
  eval(cpu_func)

  gpu_internal_func_name = Symbol(:__gpu_interact_, map_name, :!)
  gpu_internal_func_body = quote
                              tid = threadIdx().x
                              bid = blockIdx().x
                              block_size = blockDim().x 
                              gid = tid + (bid - 1) * block_size
                              if gid <= nmp
                                $( Expr(:tuple, xvars...) ) = coords[gid]
                                $( __taylormap_ex(polynomials, yvars, xvars) )
                                coords[gid] = $( Expr(:call, sv_ex, yvars...) )
                              end
                              return nothing
                          end
  gpu_internal_func_def = Expr(:where, Expr(:call, gpu_internal_func_name, :(coords::CuDeviceVector{SVector{6,T},1}), :nmp), :T)
  gpu_internal_func = Expr(:function, gpu_internal_func_def, gpu_internal_func_body) 
  eval(gpu_internal_func)

  gpu_interface_func = quote
    function interact!(beam::BeamGPU{$T}, elm::$(map_name)) 
       nmp = beam.nmp
       coords = beam.coords

       nb = ceil(Int, nmp/GLOBAL_BLOCK_SIZE)
       @cuda threads=GLOBAL_BLOCK_SIZE  blocks=nb $(Expr(:call, gpu_internal_func_name, :coords, :nmp))
    end
  end
  eval(gpu_interface_func)
  
  return eval(:( $(map_name)() ))
end

