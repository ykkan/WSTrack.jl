const Coord{T} = SVector{6,T}

function coordmax(type)
  return Coord{type}(fill(typemax(type), 6)...)
end
