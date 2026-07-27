# options-market-making-engine

Book-level brain for an options market making desk on Deribit: vol surface, risk aggregation, quoting/skew, and hedge orchestration. This repo does not touch a wire. It consumes market data and positions through plain Rust types and produces desired quotes / hedge orders through plain Rust types. Wiring those to an actual exchange is somebody else's crate.

## Status

All four crates done and tested: `vol-surface` (21/21), `book-risk` (12/12), `quoting-engine` (21/21), `hedge-orchestrator` (14/14). 68/68 across the workspace. This closes out the core book: surface → risk → quotes → hedges. Next real work is wiring, not new crates, see "Production dependencies" below.

## Crates

- **vol-surface** - raw SVI calibration per expiry, static (butterfly) and calendar no-arbitrage checks, interpolated surface queries.
- **book-risk** - Deribit inverse (coin-settled) option pricing and Greeks derived from Deribit's own published formula (not standard BSM, coin-denomination changes the delta/gamma math), position-level scaling, book aggregation bucketed by expiry. First-order + gamma only, vanna/volga not in yet.
- **quoting-engine** - Avellaneda-Stoikov run in implied-vol space instead of price space: reservation vol skewed off book-risk's per-expiry vega/gamma, AS optimal spread converted to bid/ask vol, both priced through book-risk's inverse-option pricer and rounded to Deribit's real tick schedule. Output (`QuoteEntry`) is shaped to map directly onto a Mass Quote `QuoteEntry` (Symbol/BidPx/OfferPx/BidSize/OfferSize). Toxicity is an external input slot (`toxicity_score: f64`), not computed here, that's [`game-theory-trading-strats`](https://github.com/tfrmma/game-theory-trading-strats)' job.
- **hedge-orchestrator** - delta hedge sizing against BTC-PERPETUAL derived directly from Deribit's documented inverse-contract PnL formula (exact, not approximated), a Whalley-Wilmott no-trade band around that hedge (approximated via the standard dollar-greek rescaling, flagged explicitly, see `no_trade_band.rs`), and a cost-benefit gate for hedging naked vega with other options using Deribit's real capped fee mechanism. Doesn't pick which option to hedge with, that's strike/tenor selection logic for a future pass.

## Production dependencies (not in this repo)

This engine assumes the following are already running and expects to be wired to them, it doesn't reimplement any of it:

- **Market data feed**: [`feedhandler-core-rs`](https://github.com/tfrmma/feedhandler-core-rs), extended with a Deribit-specific normalizer for the options + perp + index feeds.
- **Order execution**: [`oms-order-management-system`](https://github.com/tfrmma/oms-order-management-system) and [`sor-engine`](https://github.com/tfrmma/sor-engine) for routing and risk-checked execution, plus Deribit's Mass Quote endpoint directly for `quoting-engine`'s output.
- **Options pricing / Greeks**: [`options-pricing-engine-rs`](https://github.com/tfrmma/options-pricing-engine-rs) for anything beyond surface-fitting IV (this repo carries only the minimal Black-76 solver it needs internally for quote ingestion, don't use it as a general pricing library).
- **Flow toxicity**: [`game-theory-trading-strats`](https://github.com/tfrmma/game-theory-trading-strats)' VPIN/Kyle's lambda feeds `quoting-engine`'s `toxicity_score` input.
- **Delta hedge base**: [`gamma-scalper`](https://github.com/tfrmma/gamma-scalper) for the execution side of what `hedge-orchestrator` sizes.
- **Pre-production validation**: [`realistic-mm-backtester`](https://github.com/tfrmma/realistic-mm-backtester) for backtesting the quoting/hedge logic against realistic FIFO fills before anything touches live capital.

## Dev notes

- Rust 2021, minimal dependencies on purpose: `vol-surface` has none, `book-risk` depends only on `vol-surface` (path dep, reuses its `norm_cdf`/`norm_pdf` instead of duplicating them). Calibration is a from-scratch Nelder-Mead, no `argmin`/`nalgebra` for 5 free parameters.
- `cargo test --workspace` for unit tests, `cargo build --release` before benchmarking anything, this sandbox didn't have clippy available, run it before merging.
- `book-risk`'s Greeks are coin-denominated, derived from Deribit's own published inverse-option Black-Scholes formula (linked in `inverse_option.rs`), not textbook BSM. The division by the forward that coin-settlement implies changes the delta/gamma formulas, this is documented and cross-checked (put-call parity + finite differences) in that module rather than asserted.
- `quoting-engine`'s tick rounding mirrors Deribit's actual `tick_size`/`tick_size_steps` schema from `public/get_instrument`. Deliberately not modeled: Deribit's order-price bandwidth clamp against their portfolio margin risk matrix (a documented "minimum trading bandwidth constant" of 0.015). That bandwidth applies to a risk-matrix price-bucket move that isn't public, applying the 0.015 to the wrong base quantity would produce a plausible-looking but wrong number, so it's left undone rather than faked, see the comment in `tick.rs`.
- `hedge-orchestrator`'s perp delta-hedge sizing is exact, derived directly from Deribit's own documented inverse-contract PnL formula and cross-checked against finite differences. Its no-trade band is not exact, applying Whalley-Wilmott to a coin-denominated book requires rescaling to "dollar greeks" first (documented in `no_trade_band.rs`), which is standard practice but is an approximation, not a from-scratch quanto-corrected derivation. Its option-fee model uses Deribit's real capped-fee mechanism (min of a rate-based fee and a cap fraction of premium) but not a specific rate/cap number, public sources disagree with each other on the current values.
- Known gaps are marked `TODO` inline rather than tracked externally for now: no multi-start in the SVI calibration (bad initial guess can land in a local min), no handling for `T <= 0` in surface queries (caller filters expired/expiring contracts), IV solver's bisection fallback bounds are hand-tuned for crypto vol levels, `book-risk`'s theta is finite-differenced rather than closed-form (deliberate, it's off the per-tick hot path), `ForwardCurve` picks the nearest listed expiry rather than interpolating or erroring on a poor match, `quoting-engine`'s inventory skew works off per-expiry vega/gamma buckets rather than strike-localized gamma, and `hedge-orchestrator`'s vega hedge is a hedge-or-don't threshold, not strike/tenor selection for what to hedge with.
