export RectBound
export EllipseBound

abstract type Filter{T} end

Base.@kwdef struct RectBound{T} <: Filter{T}
  bound::Coord{T}=coordmax(T)
end

function (m::RectBound{T})(coord::Coord{T}) where {T}
  return all( abs.(coord) .< m.bound )
end


Base.@kwdef struct EllipseBound{T} <: Filter{T}
  bound::Coord{T}=coordmax(T)
end

function (m::EllipseBound{T})(coord::Coord{T}) where {T}
  return sum( (coord ./ bound).^2 ) <= one(T)
end


