# Turing

![Status](https://img.shields.io/badge/status-demo-blue)
![Language](https://img.shields.io/badge/language-Mojo-orange)
![License](https://img.shields.io/badge/license-BSD--3--Clause-green)

`/turing` is a focused Mojo library for Bayesian probabilistic programming.

The goal is to keep large-system inference manageable when multiple contributors add model insights over time. The library intentionally stays small: typed score containers, stable update semantics, practical execution paths, and a compile seam that makes execution targets explicit.

## Design goals

- Keep probabilistic interfaces simple enough to review and evolve.
- Prevent incompatible signal composition before runtime.
- Preserve model math while changing execution target.
- Make integration points explicit for downstream applications.

![Turing](turing.png)

## HKT-inspired type design

Mojo does not currently expose full higher-kinded-type syntax, so `turing` uses an HKT-inspired pattern to preserve the same safety intent.

- `KindTag` is a type-level marker for semantic signal families.
- `BayesianScore[K]` binds prior/evidence/posterior state to one kind `K`.
- `HKTInferenceEngine[T]` binds inference behavior to one compute target `T`.

### Why this matters in probabilistic systems

Probabilistic pipelines are often assembled from many independent modules. If one module emits a trust signal and another expects a feature signal, wiring them together silently can create invalid posteriors. By expressing signal families in the type layer, invalid composition fails early during compile-time checking instead of becoming ranking drift in production.

This mirrors the practical benefit of HKTs in Scala/Haskell: constraints apply to type constructors, not only concrete values.

## Mojo compile-system rationale

`HKTInferenceEngine[T]` uses generic specialization so target behavior is known at compile time:

- `T` must satisfy `ComputeTarget`.
- `CPU` and `GPU` implement the same target interface.
- Engine methods are monomorphized for each instantiated target.

That provides a predictable boundary between model semantics and lowering behavior:

- `observe(...)` keeps Bayesian update semantics target-agnostic.
- `run_map(...)`, `run_mh(...)`, `run_smc(...)`, and `run_vi(...)` provide practical algorithm execution paths over the same typed model state.
- `target_label()` and `compile_kernel(...)` expose where target-specific compilation hooks live.
- Downstream repos can select `CPU` or `GPU` by type argument without rewriting model logic.

## Supported algorithm paths

- **MAP path (`run_map`)**
  - Uses deterministic confidence-scaled updates.
  - Useful for baseline ranking services, risk scoring, and low-variance online updates.
- **Metropolis-Hastings path (`run_mh`)**
  - Applies proposal updates with deterministic acceptance safeguards.
  - Useful when teams need bounded uncertainty exploration without introducing heavy runtime complexity.
- **Sequential Monte Carlo path (`run_smc`)**
  - Retains weighted evidence with minimum-confidence safeguards.
  - Useful for streaming operations where signals arrive continuously and confidence can fluctuate.
- **Variational inference path (`run_vi`)**
  - Uses discounted evidence and posterior blending for stable serving behavior.
  - Useful for high-throughput systems where smooth online adaptation is preferred over noisy jumps.

## Practical applications

The same pattern applies across domains where uncertainty-aware scoring matters:

- **Finance:** risk scoring, fraud suspicion priors, and confidence-weighted market signals.
- **Healthcare:** triage prioritization, readmission risk estimates, and noisy-signal fusion.
- **Operations:** demand forecasting, incident likelihood updates, and resource balancing.
- **Recommendations:** feed ranking, candidate reweighting, and exploration/exploitation control.

## Repository contents

- `src/turing/hkt_probprog.mojo`: core library types and inference engine.
- `src/turing/__init__.mojo`: canonical package API surface for downstream imports.
- `examples/inference_paths_demo.mojo`: practical MAP/MH/SMC/VI execution path walkthrough.
- `src/main.mojo`: minimal smoke demo for local verification.
- `examples/social_reco_demo.mojo`: well-documented library usage example.
- `tests/test_hkt_demo.mojo`: posterior update correctness test.
- `tests/test_inference_paths.mojo`: deterministic checks for algorithm path behavior.

## Example

```mojo
from turing import CPU, BayesianScore, FeatureSignal, GaussianPrior, HKTInferenceEngine

fn main() raises:
    let engine = HKTInferenceEngine[CPU]()
    var score = BayesianScore[FeatureSignal](
        GaussianPrior(mu=0.45, sigma=0.20),
        evidence_weight=0.0,
    )

    score = engine.observe(score, observed=0.82, confidence=0.75)
    print("target:", engine.target_label())
    print("posterior relevance:", score.posterior_mean())
    print(engine.compile_kernel(score))
```

## Quick start

```zsh
mojo -I src src/main.mojo
mojo -I src examples/inference_paths_demo.mojo
mojo -I src examples/social_reco_demo.mojo
mojo -I src tests/test_hkt_demo.mojo
mojo -I src tests/test_inference_paths.mojo
```
