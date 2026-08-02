// AS optimal spread: delta = gamma*sigma^2*(T-t) + (2/gamma)*ln(1+gamma/kappa),
// same vol-space substitution as reservation.rs, sigma is vol-of-vol here, not
// underlying vol. kappa and vol_of_vol are fit from real fill/IV data, not invented.

#[derive(Debug, Clone, Copy)]
pub struct SpreadParams {
    pub risk_aversion: f64,
    pub vol_of_vol: f64,
    pub horizon_years: f64,
    pub kappa: f64,
}

// TODO: kappa and vol_of_vol are external config here, this crate doesn't fit them
// from fill/IV history, that calibration has to happen upstream.
pub fn half_spread_vol(p: &SpreadParams) -> f64 {
    let inventory_term = p.risk_aversion * p.vol_of_vol * p.vol_of_vol * p.horizon_years;
    let market_term = (2.0 / p.risk_aversion) * (1.0 + p.risk_aversion / p.kappa).ln();
    0.5 * (inventory_term + market_term)
}

// toxicity score is external, VPIN/Kyle's lambda lives in game-theory-trading-strats
pub fn widen_for_toxicity(
    half_spread: f64,
    toxicity_score: f64,
    toxicity_widen_factor: f64,
) -> f64 {
    half_spread * (1.0 + toxicity_widen_factor * toxicity_score.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_params() -> SpreadParams {
        SpreadParams {
            risk_aversion: 0.3,
            vol_of_vol: 1.2,
            horizon_years: 1.0 / 365.0,
            kappa: 8.0,
        }
    }

    #[test]
    fn spread_is_positive_under_normal_params() {
        assert!(half_spread_vol(&base_params()) > 0.0);
    }

    #[test]
    fn longer_horizon_widens_the_spread() {
        let short = SpreadParams {
            horizon_years: 1.0 / 365.0,
            ..base_params()
        };
        let long = SpreadParams {
            horizon_years: 10.0 / 365.0,
            ..base_params()
        };
        assert!(half_spread_vol(&long) > half_spread_vol(&short));
    }

    #[test]
    fn higher_vol_of_vol_widens_the_spread() {
        let calm = SpreadParams {
            vol_of_vol: 0.5,
            ..base_params()
        };
        let choppy = SpreadParams {
            vol_of_vol: 2.0,
            ..base_params()
        };
        assert!(half_spread_vol(&choppy) > half_spread_vol(&calm));
    }

    #[test]
    fn zero_toxicity_leaves_the_spread_unchanged() {
        let half = half_spread_vol(&base_params());
        assert_eq!(widen_for_toxicity(half, 0.0, 3.0), half);
    }

    #[test]
    fn full_toxicity_widens_by_the_configured_factor() {
        let half = half_spread_vol(&base_params());
        let widened = widen_for_toxicity(half, 1.0, 2.0);
        assert!((widened - half * 3.0).abs() < 1e-12);
    }

    #[test]
    fn toxicity_score_is_clamped_to_zero_one() {
        let half = half_spread_vol(&base_params());
        let widened = widen_for_toxicity(half, 5.0, 2.0); // garbage input, should clamp not explode
        assert!((widened - half * 3.0).abs() < 1e-12);
    }
}
