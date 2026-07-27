// Minimal Black-76 pricing + IV solver, forward convention (matches how
// Deribit quotes options against the index/future). Just enough to turn a
// quoted price into a total-variance point for calibrate_slice. Full
// pricing/Greeks engine lives in options-pricing-engine-rs, don't duplicate
// that here.

use std::f64::consts::PI;

pub fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

// Abramowitz-Stegun rational approx, ~7.5e-8 accurate. Fine for IV
// inversion, wouldn't trust it for tail risk numbers.
pub fn norm_cdf(x: f64) -> f64 {
    let (b1, b2, b3, b4, b5) = (0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429);
    let p = 0.2316419;
    let c = 0.39894228;

    if x >= 0.0 {
        let t = 1.0 / (1.0 + p * x);
        1.0 - c * (-x * x / 2.0).exp() * t * (t * (t * (t * (t * b5 + b4) + b3) + b2) + b1)
    } else {
        1.0 - norm_cdf(-x)
    }
}

pub fn black76_call(forward: f64, strike: f64, vol: f64, t: f64) -> f64 {
    if vol <= 0.0 || t <= 0.0 {
        return (forward - strike).max(0.0);
    }
    let sqrt_t = t.sqrt();
    let d1 = ((forward / strike).ln() + 0.5 * vol * vol * t) / (vol * sqrt_t);
    let d2 = d1 - vol * sqrt_t;
    forward * norm_cdf(d1) - strike * norm_cdf(d2)
}

fn black76_vega(forward: f64, strike: f64, vol: f64, t: f64) -> f64 {
    if vol <= 0.0 || t <= 0.0 {
        return 0.0;
    }
    let sqrt_t = t.sqrt();
    let d1 = ((forward / strike).ln() + 0.5 * vol * vol * t) / (vol * sqrt_t);
    forward * norm_pdf(d1) * sqrt_t
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IvSolverError;

// Newton with a bisection fallback for the cases Newton doesn't like:
// deep OTM, near expiry, vega near zero. Bounds below are hand-picked for
// crypto vol levels, wouldn't reuse this as-is for FX or equity indices.
//
// TODO: bisection bounds (1e-4, 5.0) are hardcoded for BTC/ETH-ish vol
// regimes. Fine today, will bite if this ever prices something with vol
// consistently outside that range.
pub fn implied_vol_black76(price: f64, forward: f64, strike: f64, t: f64, initial_guess: f64) -> Result<f64, IvSolverError> {
    let intrinsic = (forward - strike).max(0.0);
    if price < intrinsic - 1e-10 {
        return Err(IvSolverError);
    }

    let mut vol = initial_guess.max(1e-4);
    for _ in 0..50 {
        let model_price = black76_call(forward, strike, vol, t);
        let vega = black76_vega(forward, strike, vol, t);
        if vega.abs() < 1e-12 {
            break;
        }
        let diff = model_price - price;
        if diff.abs() < 1e-8 {
            return Ok(vol);
        }
        let step = diff / vega;
        let next = vol - step;
        vol = if next > 1e-4 && next < 5.0 { next } else { (vol - step.signum() * vol * 0.5).max(1e-4) };
    }

    let (mut lo, mut hi) = (1e-4, 5.0);
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if black76_call(forward, strike, mid, t) > price {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let vol = 0.5 * (lo + hi);
    if vol.is_finite() {
        Ok(vol)
    } else {
        Err(IvSolverError)
    }
}

/// Converts a quoted price into the (log-moneyness, total-variance) point
/// calibrate_slice expects.
pub fn quote_to_variance_point(price: f64, forward: f64, strike: f64, t: f64, iv_guess: f64) -> Result<(f64, f64), IvSolverError> {
    let vol = implied_vol_black76(price, forward, strike, t, iv_guess)?;
    let k = (strike / forward).ln();
    Ok((k, vol * vol * t))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_price_to_vol_recovers_input_vol() {
        let forward = 65000.0;
        let strike = 68000.0;
        let t = 30.0 / 365.0;
        let true_vol = 0.62;
        let price = black76_call(forward, strike, true_vol, t);
        let recovered = implied_vol_black76(price, forward, strike, t, 0.5).unwrap();
        assert!((recovered - true_vol).abs() < 1e-4);
    }

    #[test]
    fn deep_otm_short_dated_still_converges() {
        let forward = 65000.0;
        let strike = 90000.0;
        let t = 3.0 / 365.0;
        let true_vol = 0.9;
        let price = black76_call(forward, strike, true_vol, t);
        let recovered = implied_vol_black76(price, forward, strike, t, 0.5).unwrap();
        assert!((recovered - true_vol).abs() < 5e-3);
    }

    #[test]
    fn price_below_intrinsic_is_rejected() {
        // forward 65000, strike 60000: intrinsic is 5000, quoting 100 is nonsense
        let result = implied_vol_black76(100.0, 65000.0, 60000.0, 30.0 / 365.0, 0.5);
        assert_eq!(result, Err(IvSolverError));
    }

    #[test]
    fn quote_to_variance_point_matches_manual_calc() {
        let forward = 65000.0;
        let strike = 70000.0;
        let t = 14.0 / 365.0;
        let true_vol = 0.55;
        let price = black76_call(forward, strike, true_vol, t);
        let (k, w) = quote_to_variance_point(price, forward, strike, t, 0.5).unwrap();
        assert!((k - (strike / forward).ln()).abs() < 1e-12);
        assert!((w - true_vol * true_vol * t).abs() < 1e-6);
    }
}
