export Printer

struct BinaryPrinter
  fname::String
  io::IO
end

function (p::BinaryPrinter)(x)
  write(p.io, x) 
end

struct AsciiPrinter
  fname::String
  io::IO
end

function Printer(fname::String, format::String="binary")
  io = open(fname, "w")
  if format == "binary"
    return BinaryPrinter(fname, io)
  elseif format == "ascii"
    return BinaryPrinter(fname, io)
  else
    error("No support format for $(format)")
  end
end
