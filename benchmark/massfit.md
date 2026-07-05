# Mass-Fit Benchmark Notes

## Motivation

Minuit2 is the de facto standard minimizer in high-energy physics. It sits at
the center of many likelihood-fit workflows, and RooFit uses Minuit2 as its
default numerical fitter for common unbinned maximum-likelihood fits. That
makes Minuit2 performance visible far beyond standalone minimization examples:
the cost of each likelihood evaluation, the number of function calls requested
by Migrad and Hesse, and the way a model is evaluated all directly affect
analysis turnaround time.

The benchmark here follows a small but representative mass-fit problem:

- observable range: `m in [5.0, 7.0]`
- signal model: Gaussian, `mb=5.28`, `sigma=0.06`
- background model: exponential, `alpha=-1.0`
- signal fraction: `fsig=0.3`
- fit target: unbinned negative log-likelihood

The point is not just to compare wrappers. The interesting question is where
the time is actually spent: RooFit model evaluation, generated RooFit code,
Julia probability abstractions, native Julia likelihood code, Minuit2 numerical
derivatives, supplied first derivatives, or post-fit error estimation.

## What Was Measured

The main comparison below uses only measurements executed locally on the same
machine.

| Row | Status | Source used | Events per toy | Toys per timing | Minuit settings |
|---|---|---|---:|---:|---|
| RooFit legacy | Executed locally | temporary 10k-only variant of upstream `benchmarking/roofit_massfit.cc` | 10000 | 100 | `Strategy(2)`, `Hesse(true)` |
| RooFit codegen | Executed locally | temporary 10k-only variant of upstream `benchmarking/roofit_massfit.cc` | 10000 | 100 | `Strategy(2)`, `Hesse(true)` |
| Julia optimized NLL, no gradient | Executed locally | `benchmark/toy_nll.jl` | 10000 | BenchmarkTools samples | `Strategy(2)`, `hesse!`, tolerance `0.1` |
| Julia optimized NLL, analytic gradient | Executed locally | `benchmark/toy_nll.jl` | 10000 | BenchmarkTools samples | `Strategy(2)`, `hesse!`, tolerance `0.1` |

The RooFit rows were compiled and run locally from temporary 10k-only variants
of the upstream mass-fit benchmark. Those local C++ build artifacts are not
part of this repository. The Julia rows use the same model ingredients, but
with two parameterization differences:

- the exponential background is represented by a positive scale `tau=1.0`,
  equivalent to RooFit's `alpha=-1.0` at the starting point;
- the signal fraction is represented internally by `log_fb`, the log
  background-to-signal ratio, to keep the Minuit parameter unconstrained.

For the matched rows, the Julia initial step sizes follow the RooFit benchmark
settings as closely as this parameterization allows.

| Concept | RooFit step | Julia matched step |
|---|---:|---:|
| Gaussian mean | `mb = 0.01` | `mu = 0.01` |
| Gaussian sigma | `sigma = 0.001` | `sigma = 0.001` |
| Exponential slope/scale | `alpha = 0.01` | `tau = 0.01` |
| Signal fraction | `fsig = 0.01` | equivalent `log_fb` step |

## Local Machine

| Component | Value |
|---|---|
| OS | Darwin kernel 24.5.0, arm64 |
| CPU reported by Julia | `8 x Apple M3` |
| Julia | 1.11.5, LLVM 16.0.6 |
| ROOT | 6.36.000 from Homebrew |

## Matched Local Results

These are local 10k-event measurements. All timings include toy generation plus
fitting.

| Implementation | Backend/objective | Minuit settings | Local time per toy |
|---|---|---|---:|
| RooFit | `legacy`, `NumCPU(16)` | `Strategy(2)`, `Hesse(true)` | 233.2 ms |
| RooFit | `codegen` | `Strategy(2)`, `Hesse(true)` | 52.4 ms |
| Julia | optimized scalar NLL, no analytic gradient | `Strategy(2)`, `hesse!`, tolerance `0.1` | 13.9 ms |
| Julia | optimized scalar NLL, analytic gradient supplied | `Strategy(2)`, `hesse!`, tolerance `0.1` | 14.3 ms |

The analytic gradient does not help in this matched configuration because the
post-fit Hesse step dominates the function-call count.

## RooFit Legacy and Codegen

`RooFit legacy` means RooFit's classic evaluator backend. In the local benchmark
it is run with `EvalBackend("legacy"), NumCPU(16)`, so it uses the traditional
RooFit evaluation machinery and 16 RooFit worker processes for each toy-study
fit.

`RooFit codegen` means RooFit's generated-code evaluator backend,
`EvalBackend("codegen")`. It avoids much of the legacy evaluator overhead and
is substantially faster locally, but it is still running the same
`Strategy(2), Hesse(true)` fit request.

The local RooFit timing difference is therefore an evaluator/backend effect,
not a change in statistical model or Minuit strategy.

| RooFit backend | Minuit settings | Local time per toy |
|---|---|---:|
| `legacy`, `NumCPU(16)` | `Strategy(2)`, `Hesse(true)` | 233.2 ms |
| `codegen` | `Strategy(2)`, `Hesse(true)` | 52.4 ms |

## Distributions.jl vs Native Evaluation

The largest Julia-side surprise is the overhead of building and evaluating the
likelihood through general-purpose probability objects in the hot loop. The
baseline implementation constructs the probability model through
`Distributions.jl` objects and evaluates `logpdf` through that abstraction. The
optimized implementation evaluates the same truncated Gaussian, truncated
exponential, and mixture density directly in one scalar loop.

Fixed-data evaluation at 10k events:

| Function | Meaning | Time per call | Allocation |
|---|---|---:|---:|
| `naive_nll` | `Distributions.jl` model construction plus `logpdf` broadcast | 1.35 ms | 1.77 MiB |
| `fast_nll` | native scalar NLL loop | 0.133 ms | 0 |
| `fast_grad` | native analytic gradient loop | 0.079 ms | 0 |

For a full fit, the same effect appears in the end-to-end timing:

| Row | Settings | Time per toy | FCN calls |
|---|---|---:|---:|
| `naive_distributions` | default Migrad, no supplied gradient | 124.6 ms | 96 |
| native NLL, no supplied gradient | default Migrad, no supplied gradient | 20.3 ms | 96 |

The first large win is therefore not a Minuit setting. It is removing repeated
distribution object construction, dynamic dispatch through generic `logpdf`,
broadcast allocation, and mixture abstraction overhead from the hot FCN path.
On this machine the direct evaluator is about 10x faster per NLL call and makes
the NLL allocation-free.

## Derivative Paths

There are several derivative paths involved in these measurements:

| Case | What supplies derivatives to Migrad? | What supplies post-fit Hesse? |
|---|---|---|
| Julia without `grad` | Minuit2 finite-difference gradients from FCN calls | Minuit2 numerical Hesse |
| Julia with `grad` | user-supplied analytic first derivative | Minuit2 numerical Hesse |
| RooFit legacy | RooFit model evaluation, Minuit2 finite-difference gradients | RooFit/Minuit2 post-fit Hesse |
| RooFit codegen | RooFit generated evaluator; RooFit reports an external gradient path | RooFit/Minuit2 post-fit Hesse |

In this context, Minuit2's internal derivative machinery is numerical
finite-difference machinery unless a gradient is explicitly supplied by the
caller. Minuit2 is not doing symbolic or source-level automatic differentiation
of the Julia likelihood.

Measured derivative impact for the native Julia evaluator at 10k:

| Configuration | Hesse? | Time per toy | FCN calls | Gradient calls |
|---|---|---:|---:|---:|
| `Strategy(2)`, matched steps, no supplied gradient | no | 10.9 ms | 83 | 0 |
| `Strategy(2)`, matched steps, analytic gradient | no | 11.0 ms | 83 | 2 |
| `Strategy(2)`, matched steps, no supplied gradient | yes | 14.0 ms | 106 | 0 |
| `Strategy(2)`, matched steps, analytic gradient | yes | 14.0 ms | 106 | 2 |
| `Strategy(0)`, relaxed tolerance, no supplied gradient | no | 7.0 ms | 45 | 0 |
| `Strategy(0)`, relaxed tolerance, analytic gradient | no | 3.25 ms | 21 | 3 |

The supplied analytic gradient matters most for the fast Migrad-only path,
where it roughly halves the number of FCN calls. Under the matched
`Strategy(2)+Hesse` settings, the total cost is dominated by numerical Hesse
and by Minuit's FCN evaluations, so the first-derivative callback has little
impact on total runtime.

## Remaining Optimization Room

After switching from `Distributions.jl` to the native evaluator, the 10k native
NLL call is about 0.13 ms and allocation-free. Toy generation is about
0.087 ms, so generation is not the bottleneck.

For the matched `Strategy(2)+Hesse` Julia rows, Minuit makes about 106 FCN
calls. At roughly 0.13 ms per native NLL call, the expected evaluator cost alone
is already about 14 ms, which matches the measured fit time. The remaining room
is mainly:

1. reduce the number of FCN evaluations by changing Minuit strategy, tolerance,
   or post-fit error calculation requirements;
2. make the native NLL loop faster through SIMD or parallel evaluation over the
   data sample;
3. provide more derivative information for the error calculation, if the
   Minuit2 API path can use it, because post-fit numerical Hesse dominates the
   matched comparison;
4. generate specialized evaluator code, in the same spirit as RooFit codegen
   and other generated-likelihood approaches, rather than routing hot
   likelihood calls through generic probability-model abstractions.

The fast-path Julia row, `Strategy(0)` with a supplied analytic gradient and no
post-fit Hesse, is already at about 3.25 ms per 10k toy on this machine. That
row should be compared only to RooFit configurations that also omit post-fit
Hesse and use similarly relaxed strategy settings.

## Commands Run

RooFit local variants were compiled from upstream `benchmarking/roofit_massfit.cc`
after changing the benchmark shape to one 10k sample size and three repeats of
100 toys. The executed local binaries were temporary artifacts named:

```sh
roofit_massfit_10k_legacy16
roofit_massfit_10k_codegen
```

Julia local benchmark:

```sh
julia --project=. benchmark/toy_nll.jl 10000
```

The Julia command printed the matched rows:

```text
fast_nograd_matched_settings_hesse: 13.935 ms
fast_grad_matched_settings_hesse: 14.25 ms
```

## Fast-Path Context

These rows were also executed locally, but they are not the same comparison as
the matched RooFit table because they omit post-fit Hesse and use `Strategy(0)`.
They are useful for understanding where the optimized Julia path can land when
only the best-fit point is required.

| Implementation | What it means | Minuit settings | Local time per toy |
|---|---|---|---:|
| Julia | optimized scalar NLL, reusing the Minuit object across toys | `Strategy(0)`, no Hesse, tolerance `0.2` | 7.0 ms |
| Julia | same optimized NLL with analytic gradient supplied | `Strategy(0)`, no Hesse, tolerance `0.2` | 3.5 ms |
