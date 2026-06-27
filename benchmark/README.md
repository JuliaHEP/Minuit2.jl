# Toy NLL Benchmark

Run the toy mass-fit benchmark:

```sh
julia --project=. benchmark/toy_nll.jl
```

Pass sample sizes as arguments to restrict the run:

```sh
julia --project=. benchmark/toy_nll.jl 10000
```

Scan initial step sizes, Minuit strategy, and tolerance:

```sh
julia --project=. benchmark/toy_nll.jl sweep 10000 false
julia --project=. benchmark/toy_nll.jl sweep 10000 true
```

The script reports generate-plus-fit time per toy. The relevant optimized rows
are:

- `fast_nograd_reuse`: optimized scalar NLL, `Strategy(0)`, no post-fit Hesse,
  with the Minuit object reused between toys.
- `fast_grad_reuse`: same optimized scalar NLL, with analytic gradients supplied.
- `fast_nograd_matched_settings_hesse`: optimized scalar NLL with matched
  initial steps, `Strategy(2)`, tolerance `0.1`, and post-fit `hesse!`.
- `fast_grad_matched_settings_hesse`: same matched settings, with analytic
  gradients supplied.

The `naive_distributions` row keeps the original `Distributions.jl`
construction pattern as the reproduction baseline.

See [`massfit.md`](massfit.md) for the local RooFit/Julia benchmark notes,
including RooFit legacy vs codegen, the impact of the `Distributions.jl`
baseline, derivative-path comparisons, and remaining optimization room.
