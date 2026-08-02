// Deribit inverse (coin-settled) option: price and Greeks in coin terms.
//
// Deribit publishes their own Black-Scholes formula for inverse options:
// https://support.deribit.com/hc/en-us/articles/31424939096093-Inverse-Options
// C = X*N(d1) - K*N(d2)*e^(-RT), with R = ln(F/X)/T, X = index, F = forward
// (the corresponding future's mark price). The coin price shown on the
// order book is that USD price divided by X.
//
// Substitute e^(-RT) = X/F into C and divide by X and the X dependence
// cancels completely, leaving a plain forward-based formula in F, K, sigma,
// T alone:
//
//   call_coin = N(d1) - (K/F) * N(d2)
//   put_coin  = (K/F) * N(-d2) - N(-d1)
//
// with the usual d1 = (ln(F/K) + sigma^2 T / 2) / (sigma sqrt(T)), d2 = d1 - sigma sqrt(T).
//
// Worked the delta/gamma/vega out by hand from that (chain rule through the
// division by F, not just reusing standard BS Greeks), cross-checked
// against put-call parity (call - put = 1 - K/F, since a coin forward
// struck at K in USD is (F-K)/F in coin terms) and against finite
// differences in the test module below. Alexander & Imeraj (2021) call the
// formula above the "naive" inverse parametrization and derive a
// quanto-corrected version with an extra convexity term; this module
// implements what Deribit's own docs use, since matching the venue's
// actual mark price matters more than matching the more theoretically
// complete academic version.
//
// Sanity check against a number that's actually out in the wild: a
// Deribit/Laevitas writeup on "true" inverse delta for an ATM-ish BTC
// option quotes a coin delta of ~0.0000063 per $1 move, same order of
// magnitude this formula produces for an ATM strike (roughly 0.5/F).
// Naive USD-style deltas would be off by a factor of F from that.

use vol_surface::black_scholes::{norm_cdf, norm_pdf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionType {
    Call,
    Put,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct InverseGreeks {
    pub price_coin: f64,
    /// d(price)/dF: hedge ratio against the tradable future/perp, not the index.
    pub delta: f64,
    pub gamma: f64,
    /// per 1.0 change in sigma (i.e. per 100 vol points), divide by 100 for per-point.
    pub vega: f64,
    /// d(price)/d(calendar time), negative for a long option. Closed-form,
    /// see greeks() below.
    pub theta: f64,
    /// d(delta)/d(vol), equivalently d(vega)/dF, both derivations cross-checked below.
    pub vanna: f64,
    /// d(vega)/d(vol).
    pub volga: f64,
}

fn d1_d2(forward: f64, strike: f64, vol: f64, t: f64) -> (f64, f64) {
    let sqrt_t = t.sqrt();
    let d1 = ((forward / strike).ln() + 0.5 * vol * vol * t) / (vol * sqrt_t);
    (d1, d1 - vol * sqrt_t)
}

pub fn price_coin(option_type: OptionType, forward: f64, strike: f64, vol: f64, t: f64) -> f64 {
    if t <= 0.0 || vol <= 0.0 {
        // at/past expiry: intrinsic value, in coin terms that's (F-K)/F, not
        // (F-K), the /F is the whole point of this module existing
        return match option_type {
            OptionType::Call => ((forward - strike) / forward).max(0.0),
            OptionType::Put => ((strike - forward) / forward).max(0.0),
        };
    }
    let (d1, d2) = d1_d2(forward, strike, vol, t);
    match option_type {
        OptionType::Call => norm_cdf(d1) - (strike / forward) * norm_cdf(d2),
        OptionType::Put => (strike / forward) * norm_cdf(-d2) - norm_cdf(-d1),
    }
}

/// All Greeks are closed-form. Theta derivation: coin_call = N(d1) -
/// (K/F)N(d2), differentiate w.r.t. T holding F/K/vol fixed (the standard
/// Greek convention), the same n(d1) = (K/F)n(d2) identity used for
/// delta/gamma kills the A = ln(F/K) terms and leaves d(coin_call)/dT =
/// (K/F)*n(d2)*vol / (2*sqrt(T)). Repeating it for the put gives the
/// identical expression, which has to be true since call - put = 1 - K/F
/// doesn't depend on T at all, so their T-derivatives must match, that
/// parity check is what this was validated against before trusting it.
///
/// Vanna derived two independent ways (had to agree before this got
/// trusted): d(vega)/dF starting from vega = (K/F)*n(d2)*sqrt(T), and
/// d(delta)/dvol starting from delta = (K/F^2)*N(d2), using the standard
/// identity d(d2)/dvol = -d1/vol. Both collapse to the same closed form:
/// vanna = -(K/F^2) * n(d2) * d1 / vol. Same call/put value again, same
/// parity argument as vega (delta_call - delta_put = K/F^2 doesn't depend
/// on vol either).
///
/// Volga = d(vega)/dvol works out to vega * d1 * d2 / vol, which matches
/// the standard textbook Black-Scholes volga identity structurally (Haug's
/// formula collection has the same vega*d1*d2/sigma form), a useful
/// external check beyond just the finite-difference tests below.
pub fn greeks(
    option_type: OptionType,
    forward: f64,
    strike: f64,
    vol: f64,
    t: f64,
) -> InverseGreeks {
    let price_coin_now = price_coin(option_type, forward, strike, vol, t);

    if t <= 0.0 || vol <= 0.0 {
        return InverseGreeks {
            price_coin: price_coin_now,
            ..Default::default()
        };
    }

    let (d1, d2) = d1_d2(forward, strike, vol, t);
    let sqrt_t = t.sqrt();
    let k_over_f2 = strike / (forward * forward);
    let k_over_f3 = strike / (forward * forward * forward);

    let (delta, gamma) = match option_type {
        OptionType::Call => (
            k_over_f2 * norm_cdf(d2),
            k_over_f3 * (norm_pdf(d2) / (vol * sqrt_t) - 2.0 * norm_cdf(d2)),
        ),
        OptionType::Put => (
            -k_over_f2 * norm_cdf(-d2),
            k_over_f3 * (norm_pdf(d2) / (vol * sqrt_t) + 2.0 * norm_cdf(-d2)),
        ),
    };

    // same closed form for calls and puts, same put-call-parity reasoning as vega
    let vega = (strike / forward) * norm_pdf(d2) * sqrt_t;
    let theta = -(strike / forward) * norm_pdf(d2) * vol / (2.0 * sqrt_t);
    let vanna = -k_over_f2 * norm_pdf(d2) * d1 / vol;
    let volga = vega * d1 * d2 / vol;

    InverseGreeks {
        price_coin: price_coin_now,
        delta,
        gamma,
        vega,
        theta,
        vanna,
        volga,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fd_delta(option_type: OptionType, forward: f64, strike: f64, vol: f64, t: f64) -> f64 {
        let eps = forward * 1e-6;
        let up = price_coin(option_type, forward + eps, strike, vol, t);
        let down = price_coin(option_type, forward - eps, strike, vol, t);
        (up - down) / (2.0 * eps)
    }

    fn fd_gamma(option_type: OptionType, forward: f64, strike: f64, vol: f64, t: f64) -> f64 {
        let eps = forward * 1e-4; // gamma needs a wider bump than delta or noise dominates
        let up = price_coin(option_type, forward + eps, strike, vol, t);
        let mid = price_coin(option_type, forward, strike, vol, t);
        let down = price_coin(option_type, forward - eps, strike, vol, t);
        (up - 2.0 * mid + down) / (eps * eps)
    }

    fn fd_vega(option_type: OptionType, forward: f64, strike: f64, vol: f64, t: f64) -> f64 {
        let eps = 1e-6;
        let up = price_coin(option_type, forward, strike, vol + eps, t);
        let down = price_coin(option_type, forward, strike, vol - eps, t);
        (up - down) / (2.0 * eps)
    }

    fn fd_theta(option_type: OptionType, forward: f64, strike: f64, vol: f64, t: f64) -> f64 {
        let eps = t * 1e-6;
        let up = price_coin(option_type, forward, strike, vol, t + eps); // theta = -dV/dT
        let down = price_coin(option_type, forward, strike, vol, t - eps);
        -(up - down) / (2.0 * eps)
    }

    // vanna two independent ways: d(delta)/dvol and d(vega)/dF, both should
    // land on the same number if the closed form is right
    fn fd_vanna_via_delta(
        option_type: OptionType,
        forward: f64,
        strike: f64,
        vol: f64,
        t: f64,
    ) -> f64 {
        let eps = 1e-6;
        let up = greeks(option_type, forward, strike, vol + eps, t).delta;
        let down = greeks(option_type, forward, strike, vol - eps, t).delta;
        (up - down) / (2.0 * eps)
    }

    fn fd_vanna_via_vega(
        option_type: OptionType,
        forward: f64,
        strike: f64,
        vol: f64,
        t: f64,
    ) -> f64 {
        let eps = forward * 1e-6;
        let up = greeks(option_type, forward + eps, strike, vol, t).vega;
        let down = greeks(option_type, forward - eps, strike, vol, t).vega;
        (up - down) / (2.0 * eps)
    }

    fn fd_volga(option_type: OptionType, forward: f64, strike: f64, vol: f64, t: f64) -> f64 {
        let eps = 1e-6;
        let up = greeks(option_type, forward, strike, vol + eps, t).vega;
        let down = greeks(option_type, forward, strike, vol - eps, t).vega;
        (up - down) / (2.0 * eps)
    }

    #[test]
    fn put_call_parity_holds_in_coin_terms() {
        let (forward, strike, vol, t) = (65000.0, 68000.0, 0.6, 30.0 / 365.0);
        let call = price_coin(OptionType::Call, forward, strike, vol, t);
        let put = price_coin(OptionType::Put, forward, strike, vol, t);
        let expected = 1.0 - strike / forward;
        assert!(
            (call - put - expected).abs() < 1e-12,
            "call={call} put={put} expected={expected}"
        );
    }

    #[test]
    fn analytic_delta_matches_finite_difference() {
        for (forward, strike) in [(65000.0, 65000.0), (65000.0, 70000.0), (65000.0, 55000.0)] {
            let (vol, t) = (0.65, 14.0 / 365.0);
            for option_type in [OptionType::Call, OptionType::Put] {
                let analytic = greeks(option_type, forward, strike, vol, t).delta;
                let fd = fd_delta(option_type, forward, strike, vol, t);
                let rel_diff = (analytic - fd).abs() / fd.abs().max(1e-12);
                assert!(
                    rel_diff < 1e-4,
                    "{option_type:?} F={forward} K={strike} analytic={analytic} fd={fd}"
                );
            }
        }
    }

    #[test]
    fn analytic_gamma_matches_finite_difference() {
        for (forward, strike) in [(65000.0, 65000.0), (65000.0, 70000.0), (65000.0, 55000.0)] {
            let (vol, t) = (0.65, 14.0 / 365.0);
            for option_type in [OptionType::Call, OptionType::Put] {
                let analytic = greeks(option_type, forward, strike, vol, t).gamma;
                let fd = fd_gamma(option_type, forward, strike, vol, t);
                let rel_diff = (analytic - fd).abs() / fd.abs().max(1e-12);
                assert!(
                    rel_diff < 1e-2,
                    "{option_type:?} F={forward} K={strike} analytic={analytic} fd={fd}"
                );
            }
        }
    }

    #[test]
    fn analytic_vega_matches_finite_difference() {
        for option_type in [OptionType::Call, OptionType::Put] {
            let (forward, strike, vol, t) = (65000.0, 68000.0, 0.6, 30.0 / 365.0);
            let analytic = greeks(option_type, forward, strike, vol, t).vega;
            let fd = fd_vega(option_type, forward, strike, vol, t);
            let rel_diff = (analytic - fd).abs() / fd.abs().max(1e-12);
            assert!(
                rel_diff < 1e-4,
                "{option_type:?} analytic={analytic} fd={fd}"
            );
        }
    }

    #[test]
    fn coin_delta_is_orders_of_magnitude_smaller_than_a_direct_option_delta() {
        // ATM coin delta should sit near 0.5/F, not near 0.5 the way a
        // direct/USD-settled option's delta would.
        let forward = 65000.0;
        let d = greeks(OptionType::Call, forward, forward, 0.6, 30.0 / 365.0).delta;
        assert!(
            d > 0.0 && d < 1.0 / forward,
            "coin delta {d} should be well under 1/F"
        );
    }

    #[test]
    fn analytic_theta_matches_finite_difference() {
        for option_type in [OptionType::Call, OptionType::Put] {
            let (forward, strike, vol, t) = (65000.0, 68000.0, 0.6, 30.0 / 365.0);
            let analytic = greeks(option_type, forward, strike, vol, t).theta;
            let fd = fd_theta(option_type, forward, strike, vol, t);
            let rel_diff = (analytic - fd).abs() / fd.abs().max(1e-12);
            assert!(
                rel_diff < 1e-4,
                "{option_type:?} analytic={analytic} fd={fd}"
            );
        }
    }

    #[test]
    fn theta_is_negative_for_a_long_option() {
        let theta = greeks(OptionType::Call, 65000.0, 65000.0, 0.6, 14.0 / 365.0).theta;
        assert!(
            theta < 0.0,
            "long option should decay in value as T shrinks, got {theta}"
        );
    }

    #[test]
    fn call_and_put_theta_are_equal() {
        // same reasoning as vega: C - P = 1 - K/F doesn't depend on T
        let (forward, strike, vol, t) = (65000.0, 62000.0, 0.7, 45.0 / 365.0);
        let call_theta = greeks(OptionType::Call, forward, strike, vol, t).theta;
        let put_theta = greeks(OptionType::Put, forward, strike, vol, t).theta;
        assert!((call_theta - put_theta).abs() < 1e-10);
    }

    #[test]
    fn vanna_matches_finite_difference_both_ways() {
        for option_type in [OptionType::Call, OptionType::Put] {
            let (forward, strike, vol, t) = (65000.0, 68000.0, 0.6, 30.0 / 365.0);
            let analytic = greeks(option_type, forward, strike, vol, t).vanna;
            let via_delta = fd_vanna_via_delta(option_type, forward, strike, vol, t);
            let via_vega = fd_vanna_via_vega(option_type, forward, strike, vol, t);
            assert!(
                (analytic - via_delta).abs() / via_delta.abs().max(1e-12) < 1e-4,
                "{option_type:?} analytic={analytic} via_delta={via_delta}"
            );
            assert!(
                (analytic - via_vega).abs() / via_vega.abs().max(1e-12) < 1e-4,
                "{option_type:?} analytic={analytic} via_vega={via_vega}"
            );
        }
    }

    #[test]
    fn volga_matches_finite_difference() {
        for option_type in [OptionType::Call, OptionType::Put] {
            let (forward, strike, vol, t) = (65000.0, 68000.0, 0.6, 30.0 / 365.0);
            let analytic = greeks(option_type, forward, strike, vol, t).volga;
            let fd = fd_volga(option_type, forward, strike, vol, t);
            let rel_diff = (analytic - fd).abs() / fd.abs().max(1e-12);
            assert!(
                rel_diff < 1e-4,
                "{option_type:?} analytic={analytic} fd={fd}"
            );
        }
    }

    #[test]
    fn call_and_put_vanna_and_volga_are_equal() {
        // delta_call - delta_put = K/F^2 doesn't depend on vol, and vega is
        // already equal for both, so both cross-greeks inherit the same parity
        let (forward, strike, vol, t) = (65000.0, 62000.0, 0.7, 45.0 / 365.0);
        let call = greeks(OptionType::Call, forward, strike, vol, t);
        let put = greeks(OptionType::Put, forward, strike, vol, t);
        assert!((call.vanna - put.vanna).abs() < 1e-10);
        assert!((call.volga - put.volga).abs() < 1e-10);
    }

    #[test]
    fn vanna_and_volga_are_zero_at_or_past_expiry() {
        let g = greeks(OptionType::Call, 65000.0, 65000.0, 0.6, 0.0);
        assert_eq!(g.vanna, 0.0);
        assert_eq!(g.volga, 0.0);
    }

    #[test]
    fn call_and_put_vega_are_equal() {
        let (forward, strike, vol, t) = (65000.0, 62000.0, 0.7, 45.0 / 365.0);
        let call_vega = greeks(OptionType::Call, forward, strike, vol, t).vega;
        let put_vega = greeks(OptionType::Put, forward, strike, vol, t).vega;
        assert!((call_vega - put_vega).abs() < 1e-10);
    }

    #[test]
    fn at_expiry_price_is_intrinsic_in_coin_terms() {
        let forward = 65000.0;
        let itm_call = price_coin(OptionType::Call, forward, 60000.0, 0.6, 0.0);
        assert!((itm_call - (forward - 60000.0) / forward).abs() < 1e-12);
        let otm_call = price_coin(OptionType::Call, forward, 70000.0, 0.6, 0.0);
        assert_eq!(otm_call, 0.0);
    }
}
