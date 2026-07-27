// AS's reservation price shifts mid by inventory * risk_aversion * variance * horizon.
// No single inventory number for an options book, so this skews implied vol instead
// of price, off per-expiry vega and gamma, each with its own coefficient since they're
// different risks: vega is realized-vs-implied over the full holding period, gamma is
// hedge slippage over a much shorter window.

use book_risk::BookGreeks;

#[derive(Debug, Clone, Copy)]
pub struct RiskAversion {
    pub vega: f64,
    pub gamma: f64,
}

pub fn reservation_vol(mid_vol: f64, bucket: &BookGreeks, risk_aversion: RiskAversion) -> f64 {
    (mid_vol - risk_aversion.vega * bucket.vega - risk_aversion.gamma * bucket.gamma).max(1e-4)
}

// TODO: skew is per-expiry-bucket, not localized to nearby strikes (kernel-weighted
// gamma), book-risk would need to expose finer-than-bucket granularity for that.

#[cfg(test)]
mod tests {
    use super::*;

    fn ra() -> RiskAversion {
        RiskAversion { vega: 0.02, gamma: 0.01 }
    }

    #[test]
    fn zero_inventory_leaves_mid_untouched() {
        let bucket = BookGreeks::default();
        assert_eq!(reservation_vol(0.6, &bucket, ra()), 0.6);
    }

    #[test]
    fn long_vega_pulls_reservation_vol_down() {
        let bucket = BookGreeks { vega: 5.0, ..Default::default() };
        let r = reservation_vol(0.6, &bucket, ra());
        assert!(r < 0.6, "long vega should skew reservation vol down, got {r}");
    }

    #[test]
    fn short_vega_pushes_reservation_vol_up() {
        let bucket = BookGreeks { vega: -5.0, ..Default::default() };
        let r = reservation_vol(0.6, &bucket, ra());
        assert!(r > 0.6, "short vega should skew reservation vol up, got {r}");
    }

    #[test]
    fn vega_and_gamma_skew_are_independently_scaled() {
        let vega_only = BookGreeks { vega: 5.0, ..Default::default() };
        let gamma_only = BookGreeks { gamma: 5.0, ..Default::default() };
        let r_vega = reservation_vol(0.6, &vega_only, ra());
        let r_gamma = reservation_vol(0.6, &gamma_only, ra());
        // same magnitude of inventory, different coefficients, so different shift
        assert!((0.6 - r_vega) - (0.6 - r_gamma) > 1e-9);
    }

    #[test]
    fn never_returns_a_non_positive_vol_under_extreme_inventory() {
        let bucket = BookGreeks { vega: 1e6, ..Default::default() };
        assert!(reservation_vol(0.6, &bucket, ra()) > 0.0);
    }
}
