// Whalley & Wilmott (1997): for CARA utility and small proportional
// transaction costs, the half-width of the no-rehedge delta region is
//
//   w = (3 * c * Gamma^2 * S / (2 * a))^(1/3)
//
// c = transaction cost rate, Gamma = position gamma, S = spot, a = CARA
// risk-aversion. Cross-checked against two independent sources (PFHedge's
// docs and the NTBN paper's eq. 20), the formula itself isn't in question.
//
// Applying it to this book is the approximate part: WW assumes wealth and
// the underlying move in the same currency, ours don't (BTC wealth, USD
// underlying, same quanto mismatch as inverse_option.rs). This rescales
// coin greeks to "dollar greeks" first (dollar_delta = coin_delta * F,
// dollar_gamma = coin_gamma * F^2), the standard practitioner move for
// per-absolute-move to per-percent-move sensitivities, same reason an ATM
// coin delta of ~1/F becomes the familiar ~0.5 once multiplied by F.
//
// TODO: this is a well-established approximation, not a from-scratch
// derivation, a rigorous quanto-corrected WW band for coin wealth is an
// open problem, not solved here.

#[derive(Debug, Clone, Copy)]
pub struct BandParams {
    pub transaction_cost_rate: f64,
    pub risk_aversion: f64,
}

pub fn dollar_delta(book_delta_coin: f64, forward: f64) -> f64 {
    book_delta_coin * forward
}

pub fn dollar_gamma(book_gamma_coin: f64, forward: f64) -> f64 {
    book_gamma_coin * forward * forward
}

pub fn half_width_dollar_delta(dollar_gamma: f64, forward: f64, params: &BandParams) -> f64 {
    let inner = 3.0 * params.transaction_cost_rate * dollar_gamma * dollar_gamma * forward / (2.0 * params.risk_aversion);
    inner.cbrt()
}

// WW's policy is "do nothing inside the band, trade only to the nearest edge when outside it",
// not "hedge straight back to target". None means already inside the band.
pub fn rehedge_target(current_dollar_delta: f64, target_dollar_delta: f64, half_width: f64) -> Option<f64> {
    let deviation = current_dollar_delta - target_dollar_delta;
    if deviation.abs() <= half_width {
        None
    } else if deviation > 0.0 {
        Some(target_dollar_delta + half_width)
    } else {
        Some(target_dollar_delta - half_width)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> BandParams {
        BandParams { transaction_cost_rate: 0.0005, risk_aversion: 1e-8 }
    }

    #[test]
    fn band_widens_with_bigger_gamma() {
        let forward = 65000.0;
        let small = half_width_dollar_delta(0.01, forward, &params());
        let big = half_width_dollar_delta(0.1, forward, &params());
        assert!(big > small);
    }

    #[test]
    fn band_narrows_with_higher_risk_aversion() {
        let forward = 65000.0;
        let gamma = 0.05;
        let relaxed = half_width_dollar_delta(gamma, forward, &BandParams { risk_aversion: 1e-9, ..params() });
        let averse = half_width_dollar_delta(gamma, forward, &BandParams { risk_aversion: 1e-7, ..params() });
        assert!(averse < relaxed, "more risk-averse should tolerate a narrower band");
    }

    #[test]
    fn staying_inside_the_band_needs_no_rehedge() {
        let target = 0.0;
        let half_width = 0.1;
        assert_eq!(rehedge_target(0.05, target, half_width), None);
        assert_eq!(rehedge_target(-0.05, target, half_width), None);
    }

    #[test]
    fn leaving_the_band_rehedges_to_the_edge_not_the_target() {
        let target = 0.0;
        let half_width = 0.1;
        // way above target, should snap to the upper edge, not all the way to 0
        assert_eq!(rehedge_target(0.5, target, half_width), Some(0.1));
        assert_eq!(rehedge_target(-0.5, target, half_width), Some(-0.1));
    }

    #[test]
    fn exactly_on_the_boundary_does_not_trigger_a_rehedge() {
        // boundary case, deviation == half_width exactly, should count as inside
        assert_eq!(rehedge_target(0.1, 0.0, 0.1), None);
    }
}
