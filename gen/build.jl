using CxxWrap
#using WrapIt
using Minuit2_jll

#---Build the wrapper library----------------------------------------------------------------------
builddir = joinpath(@__DIR__, "build")
sourcedir = @__DIR__
cd(@__DIR__)
mkpath(builddir)

if Sys.isapple()
    #ENV["SDKROOT"]=readchomp(`xcrun --sdk macosx --show-sdk-path`)
    ENV["SDKROOT"]="/Library/Developer/CommandLineTools/SDKs/MacOSX14.sdk"
end

cxxwrap_prefix = CxxWrap.prefix_path()
minuit2_prefix = Minuit2_jll.artifact_dir
julia_prefix = dirname(Sys.BINDIR)

#---Generate the wrapper code----------------------------------------------------------------------
updatemode = ("--update" ∈ ARGS)
updatemode && println("Update mode")
wit = joinpath(@__DIR__, "Minuit2.wit")
witin = joinpath(@__DIR__, "Minuit2.wit.in")
open(wit, "w") do f
    for l in eachline(witin)
	println(f, replace(l, "@Minuit2_INCLUDE_DIR@" => "$minuit2_prefix/include",
                          "@Julia_INCLUDE_DIR@" => "$julia_prefix/include/julia",
                          "@JlCxx_INCLUDE_DIR@" => "$cxxwrap_prefix/include",
                          "@CxxWrap_VERSION@" => "$(pkgversion(CxxWrap))"))
    end
end

#WrapIt.install()
#rc = wrapit(wit, force=true, cmake=false, update=updatemode, verbosity=1)
rc = run(`wrapit $wit --force -v 1`).exitcode
if !isnothing(rc) && rc != 0
    println(stderr, "Failed to produce wrapper code with the wrapit function. Exited with code ", rc, ".")
    exit(rc)
end

cd(builddir)
run(`cmake -DCMAKE_BUILD_TYPE=Release
           -DCMAKE_CXX_STANDARD=17
           -DCMAKE_PREFIX_PATH=$cxxwrap_prefix\;$minuit2_prefix  $sourcedir`)
run(`cmake --build . --config Release --parallel 8`)
