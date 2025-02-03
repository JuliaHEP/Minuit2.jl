module Minuit2


import Base.getindex
import Base.setindex!

using CxxWrap
import Libdl
@wrapmodule(()->"$(@__DIR__)/../deps/libjlMinuit2." * Libdl.dlext)

function __init__()
    @initcxx
end

end #module
