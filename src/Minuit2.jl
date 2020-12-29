module Minuit2
    using Minuit2_jll
    using CxxWrap
    using Minuit2_Julia_Wrapper_jll

    @wrapmodule(Minuit2_Julia_Wrapper_jll.minuit2wrap_path)
    function __init__()
        @initcxx
    end
end

