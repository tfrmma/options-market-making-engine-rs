// AS's reservation price shifts mid by inventory * risk_aversion * variance * horizon.
// No single inventory number for an options book, so this skews implied vol instead
// of price, off per-expiry vega and localized gamma, each with its own coefficient
// since they're different risks: vega is realized-vs-implied over the full holding
// period and diffuse across a tenor, gamma is hedge slippage over a much shorter
// window and concentrated near the strike itself, a strike sitting on top of a big
// position shouldn't skew the same as one sitting in an empty part of the same wing.

#[derive(Debug, Clone, Copy)]
pub struct RiskAversion {
    pub vega: f64,
    pub gamma: f64,
}

pub fn reservation_vol(mid_vol: f64, bucket_vega: f64, local_gamma: f64, risk_aversion: RiskAversion) -> f64 {
    (mid_vol - risk_aversion.vega * bucket_vega - risk_aversion.gamma * local_gamma).max(1e-4)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ra() -> RiskAversion {
        RiskAversion { vega: 0.02, gamma: 0.01 }
    }

    #[test]
    fn zero_inventory_leaves_mid_untouched() {
        assert_eq!(reservation_vol(0.6, 0.0, 0.0, ra()), 0.6);
    }

    #[test]
    fn long_vega_pulls_reservation_vol_down() {
        let r = reservation_vol(0.6, 5.0, 0.0, ra());
        assert!(r < 0.6, "long vega should skew reservation vol down, got {r}");
    }

    #[test]
    fn short_vega_pushes_reservation_vol_up() {
        let r = reservation_vol(0.6, -5.0, 0.0, ra());
        assert!(r > 0.6, "short vega should skew reservation vol up, got {r}");
    }

    #[test]
    fn vega_and_gamma_skew_are_independently_scaled() {
        let r_vega = reservation_vol(0.6, 5.0, 0.0, ra());
        let r_gamma = reservation_vol(0.6, 0.0, 5.0, ra());
        // same magnitude of inventory, different coefficients, so different shift
        assert!((0.6 - r_vega) - (0.6 - r_gamma) > 1e-9);
    }

    #[test]
    fn never_returns_a_non_positive_vol_under_extreme_inventory() {
        assert!(reservation_vol(0.6, 1e6, 0.0, ra()) > 0.0);
    }
}
