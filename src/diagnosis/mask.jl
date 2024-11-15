export RectMask
export EllipseMask 


Base.@kwdef struct RectMask{T}
  bound::Coord{T}=coordmax(T)
end

function (m::RectMask{T})(coord::Coord{T}) where {T}
  return all( abs.(coord) .< m.bound )
end


Base.@kwdef struct EllipseMask{T}
  bound::Coord{T}=coordmax(T)
end

function (m::EllipseMask{T})(coord::Coord{T}) where {T}
  return sum( (coord ./ bound).^2 ) <= one(T)
end


