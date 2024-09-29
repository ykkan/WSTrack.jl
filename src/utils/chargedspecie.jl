export ELECTRON
export POSITRON
export PROTON

struct ChargedSpecie{T}
  q::T
  m::T
end

const ELECTRON = ChargedSpecie(-1.0, 0.5109989461e6)
const POSITRON = ChargedSpecie(1.0, 938.272e6) 
const PROTON = ChargedSpecie(1.0, 938.272e6) 
