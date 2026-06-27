using BenchmarkTools
using Distributions
using Minuit2
using Random
using SpecialFunctions: erf

const XMIN = 5.0
const XMAX = 7.0
const WIDTH = XMAX - XMIN
const INV_SQRT2PI = inv(sqrt(2 * pi))
const INV_SQRT2 = inv(sqrt(2.0))

const GEN_MU = 5.28
const GEN_SIGMA = 0.06
const GEN_TAU = 1.0
const GEN_LOG_FB = 1 / (1 / (1 - 0.7) - 1)
const INIT = [GEN_MU, GEN_SIGMA, GEN_TAU, GEN_LOG_FB]
const STEPS = [0.01, 0.005, 0.05, 0.05]
const LIMITS = [(XMIN, XMAX), (1e-4, 1.0), (1e-3, 20.0), (-10.0, 10.0)]
const NAMES = ("mu", "sigma", "tau", "log_fb")

normal_cdf01(z) = 0.5 * (1 + erf(z * INV_SQRT2))
normal_pdf01(z) = INV_SQRT2PI * exp(-0.5 * z * z)

function signal_norm(mu, sigma)
    alpha = (XMIN - mu) / sigma
    beta = (XMAX - mu) / sigma
    return normal_cdf01(beta) - normal_cdf01(alpha)
end

function signal_pdf(x, mu, sigma)
    z = (x - mu) / sigma
    return normal_pdf01(z) / (sigma * signal_norm(mu, sigma))
end

function background_pdf(x, tau)
    return exp((XMIN - x) / tau) / (tau * (1 - exp(-WIDTH / tau)))
end

function fill_toy!(rng::AbstractRNG, data::Vector{Float64}, mu, sigma, tau, log_fb)
    p_signal = inv(1 + exp(log_fb))
    bg_norm = 1 - exp(-WIDTH / tau)
    @inbounds for i in eachindex(data)
        if rand(rng) < p_signal
            x = randn(rng) * sigma + mu
            while x < XMIN || x > XMAX
                x = randn(rng) * sigma + mu
            end
            data[i] = x
        else
            data[i] = XMIN - tau * log1p(-rand(rng) * bg_norm)
        end
    end
    return data
end

function toy_data(n::Integer; seed=0x31)
    rng = Xoshiro(seed)
    data = Vector{Float64}(undef, n)
    return fill_toy!(rng, data, GEN_MU, GEN_SIGMA, GEN_TAU, GEN_LOG_FB)
end

function distributions_model(pars, a=XMIN, b=XMAX)
    prior = [1, exp(pars.log_fb)] ./ (1 + exp(pars.log_fb))
    sig = truncated(Normal(pars.sig.mu, pars.sig.sigma), a, b)
    bgd = truncated(Exponential(pars.bgd.tau), a, b)
    return MixtureModel([sig, bgd], prior)
end

function naive_nll(data, mu, sigma, tau, log_fb)
    pars = (; sig=(; mu, sigma), bgd=(; tau), log_fb)
    model = distributions_model(pars)
    return -sum(logpdf.(Ref(model), data))
end

function fast_nll(data, mu, sigma, tau, log_fb)
    sigma <= 0 && return Inf
    tau <= 0 && return Inf
    zsig = signal_norm(mu, sigma)
    zsig <= 0 && return Inf
    inv_sig_norm = inv(sigma * zsig)
    bg_norm = 1 - exp(-WIDTH / tau)
    bg_norm <= 0 && return Inf
    inv_bg_norm = inv(tau * bg_norm)
    bg_shift = inv(tau)
    w_bg = exp(log_fb) / (1 + exp(log_fb))
    w_sig = 1 - w_bg

    nll = 0.0
    @inbounds @simd for x in data
        z = (x - mu) / sigma
        sig = INV_SQRT2PI * exp(-0.5 * z * z) * inv_sig_norm
        bgd = exp((XMIN - x) * bg_shift) * inv_bg_norm
        nll -= log(w_sig * sig + w_bg * bgd + floatmin(Float64))
    end
    return nll
end

function fast_grad(data, mu, sigma, tau, log_fb)
    alpha = (XMIN - mu) / sigma
    beta = (XMAX - mu) / sigma
    phi_alpha = normal_pdf01(alpha)
    phi_beta = normal_pdf01(beta)
    zsig = normal_cdf01(beta) - normal_cdf01(alpha)
    zsig <= 0 && return (NaN, NaN, NaN, NaN)

    inv_sigma = inv(sigma)
    inv_sig_norm = inv(sigma * zsig)
    dlogz_dmu = (phi_alpha - phi_beta) * inv_sigma / zsig
    dlogz_dsigma = (alpha * phi_alpha - beta * phi_beta) * inv_sigma / zsig

    q = exp(-WIDTH / tau)
    bg_norm = 1 - q
    bg_norm <= 0 && return (NaN, NaN, NaN, NaN)
    inv_tau = inv(tau)
    inv_tau2 = inv_tau * inv_tau
    inv_bg_norm = inv(tau * bg_norm)
    bg_tail = WIDTH * q * inv_tau2 / bg_norm

    w_bg = exp(log_fb) / (1 + exp(log_fb))
    w_sig = 1 - w_bg
    w_prod = w_sig * w_bg

    g_mu = 0.0
    g_sigma = 0.0
    g_tau = 0.0
    g_log_fb = 0.0
    @inbounds for x in data
        z = (x - mu) * inv_sigma
        sig = INV_SQRT2PI * exp(-0.5 * z * z) * inv_sig_norm
        bgd = exp((XMIN - x) * inv_tau) * inv_bg_norm
        mix = w_sig * sig + w_bg * bgd + floatmin(Float64)

        dlogsig_dmu = z * inv_sigma - dlogz_dmu
        dlogsig_dsigma = (z * z - 1) * inv_sigma - dlogz_dsigma
        dlogbg_dtau = (x - XMIN) * inv_tau2 - inv_tau + bg_tail

        inv_mix = inv(mix)
        g_mu -= w_sig * sig * dlogsig_dmu * inv_mix
        g_sigma -= w_sig * sig * dlogsig_dsigma * inv_mix
        g_tau -= w_bg * bgd * dlogbg_dtau * inv_mix
        g_log_fb -= w_prod * (bgd - sig) * inv_mix
    end
    return (g_mu, g_sigma, g_tau, g_log_fb)
end

function logfb_step_from_fraction_step(log_fb, df)
    f = inv(1 + exp(log_fb))
    return abs(df / (f * (1 - f)))
end

function step_preset(name::Symbol)
    if name === :current
        return STEPS
    elseif name === :matched
        return [0.01, 0.001, 0.01, logfb_step_from_fraction_step(GEN_LOG_FB, 0.01)]
    elseif name === :small
        return [0.005, 0.0005, 0.005, logfb_step_from_fraction_step(GEN_LOG_FB, 0.005)]
    elseif name === :large
        return [0.02, 0.005, 0.05, logfb_step_from_fraction_step(GEN_LOG_FB, 0.02)]
    else
        throw(ArgumentError("unknown step preset: $name"))
    end
end

function fit_naive(data; tolerance=1e-6, strategy=1, ncall=0, steps=STEPS)
    f(mu, sigma, tau, log_fb) = naive_nll(data, mu, sigma, tau, log_fb)
    m = Minuit(f, INIT; error=steps, limits=LIMITS, names=NAMES, errordef=0.5, tolerance)
    migrad!(m, strategy; ncall)
    return m
end

function fit_fast(data; tolerance=1e-6, strategy=1, ncall=0, grad=false, steps=STEPS)
    f(mu, sigma, tau, log_fb) = fast_nll(data, mu, sigma, tau, log_fb)
    g = grad ? ((mu, sigma, tau, log_fb) -> fast_grad(data, mu, sigma, tau, log_fb)) : nothing
    m = Minuit(f, INIT; grad=g, error=steps, limits=LIMITS, names=NAMES, errordef=0.5, tolerance)
    migrad!(m, strategy; ncall)
    return m
end

mutable struct FastObjective
    data::Vector{Float64}
end

(obj::FastObjective)(mu, sigma, tau, log_fb) = fast_nll(obj.data, mu, sigma, tau, log_fb)

struct FastGradient
    objective::FastObjective
end

(grad::FastGradient)(mu, sigma, tau, log_fb) = fast_grad(grad.objective.data, mu, sigma, tau, log_fb)

mutable struct ToyFitRunner{R<:AbstractRNG}
    rng::R
    objective::FastObjective
    minuit::Minuit
    strategy::Int
    ncall::Int
    run_hesse::Bool
    hesse_strategy::Int
end

function ToyFitRunner(n::Integer; seed=0x31, grad=false, strategy=0, tolerance=0.2,
        ncall=0, steps=STEPS, run_hesse=false, hesse_strategy=strategy)
    rng = Xoshiro(seed)
    objective = FastObjective(Vector{Float64}(undef, n))
    gradient = grad ? FastGradient(objective) : nothing
    m = Minuit(objective, INIT; grad=gradient, error=steps, limits=LIMITS,
        names=NAMES, errordef=0.5, tolerance)
    return ToyFitRunner(rng, objective, m, strategy, ncall, run_hesse, hesse_strategy)
end

function reset_minuit!(m::Minuit)
    m.last_state = copy(m.init_state)
    m.app = nothing
    m.fmin = nothing
    m.mino = nothing
    return m
end

function run_toy!(runner::ToyFitRunner)
    fill_toy!(runner.rng, runner.objective.data, GEN_MU, GEN_SIGMA, GEN_TAU, GEN_LOG_FB)
    reset_minuit!(runner.minuit)
    migrad!(runner.minuit, runner.strategy; ncall=runner.ncall)
    runner.run_hesse && hesse!(runner.minuit; strategy=runner.hesse_strategy)
    return runner.minuit
end

function toy_naive(n; seed=0x31)
    data = toy_data(n; seed)
    return fit_naive(data)
end

function toy_fast(n; seed=0x31, grad=false, strategy=1, tolerance=1e-6, steps=STEPS)
    data = toy_data(n; seed)
    return fit_fast(data; grad, strategy, tolerance, steps)
end

function toy_fast_reuse(n; seed=0x31, grad=false, strategy=0, tolerance=0.2, steps=STEPS)
    runner = ToyFitRunner(n; seed, grad, strategy, tolerance, steps)
    return run_toy!(runner)
end

function summarize(label, trial, m; ngrad=m.ngrad)
    t_ms = median(trial).time / 1e6
    alloc_mb = median(trial).memory / 2.0^20
    return (; label, time_ms=t_ms, alloc_mb, nfcn=m.fmin.nfcn, ngrad, niter=m.niter,
        fval=m.fval, valid=m.is_valid, values=collect(m.values))
end

function run_one(label, f, n; seconds=3, samples=100)
    m = f(n)
    trial = @benchmark $f($n) seconds=seconds samples=samples evals=1
    return summarize(label, trial, m)
end

function run_reuse(label, n; grad=false, strategy=0, tolerance=0.2, steps=STEPS,
        run_hesse=false, hesse_strategy=strategy, seconds=3, samples=100)
    runner = ToyFitRunner(n; grad, strategy, tolerance, steps, run_hesse, hesse_strategy)
    trial = @benchmark run_toy!($runner) seconds=seconds samples=samples evals=1
    ngrad0 = runner.minuit.ngrad
    m = run_toy!(runner)
    ngrad = runner.minuit.ngrad - ngrad0
    return summarize(label, trial, m; ngrad)
end

function sweep_steps(; n=10_000, grad=false, samples=60, seconds=2)
    println("Step/tolerance/strategy sweep for N = ", n, ", grad = ", grad)
    println("Matched reference steps: mean=0.01, sigma=0.001, fraction=0.01, slope/scale=0.01")
    for preset in (:small, :matched, :current, :large)
        steps = step_preset(preset)
        println("\nsteps=", preset, " -> ", round.(steps; digits=6))
        for strategy in (0, 1, 2), tolerance in (1e-6, 1e-3, 0.1, 0.2)
            label = "strategy=$strategy tolerance=$tolerance"
            result = run_reuse(label, n; grad, strategy, tolerance, steps, seconds, samples)
            print_result(result)
        end
    end
    return nothing
end

function print_result(r)
    vals = join(round.(r.values; digits=6), ", ")
    println(r.label, ": ",
        round(r.time_ms; digits=3), " ms, ",
        round(r.alloc_mb; digits=3), " MiB, ",
        "nfcn=", r.nfcn, ", ngrad=", r.ngrad, ", niter=", r.niter,
        ", valid=", r.valid, ", fval=", round(r.fval; digits=3),
        ", values=[", vals, "]")
end

function main(args=ARGS)
    if !isempty(args) && args[1] == "sweep"
        n = length(args) >= 2 ? parse(Int, args[2]) : 10_000
        grad = length(args) >= 3 ? parse(Bool, args[3]) : false
        return sweep_steps(; n, grad)
    end
    ns = isempty(args) ? (1_000, 10_000, 100_000) : parse.(Int, args)
    println("Benchmarking generate + fit per toy")
    println("Historical RooFit SIMD reference: 1k=1.26 ms, 10k=8.22 ms, 100k=78.4 ms")
    for n in ns
        println("\nN = ", n)
        results = [
            run_one("naive_distributions", toy_naive, n; seconds=2, samples=20),
            run_one("fast_nograd", x -> toy_fast(x; grad=false), n; seconds=3, samples=100),
            run_one("fast_nograd_strategy0", x -> toy_fast(x; grad=false, strategy=0, tolerance=0.2), n; seconds=3, samples=100),
            run_reuse("fast_nograd_reuse", n; grad=false, strategy=0, tolerance=0.2, seconds=3, samples=100),
            run_reuse("fast_nograd_matched_settings_hesse", n; grad=false, strategy=2,
                tolerance=0.1, steps=step_preset(:matched), run_hesse=true,
                hesse_strategy=2, seconds=3, samples=100),
            run_one("fast_grad", x -> toy_fast(x; grad=true), n; seconds=3, samples=100),
            run_reuse("fast_grad_reuse", n; grad=true, strategy=0, tolerance=0.2, seconds=3, samples=100),
            run_reuse("fast_grad_matched_settings_hesse", n; grad=true, strategy=2,
                tolerance=0.1, steps=step_preset(:matched), run_hesse=true,
                hesse_strategy=2, seconds=3, samples=100),
        ]
        foreach(print_result, results)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
