// Raw SVI slice, Gatheral (2004). No-arb conditions and g(k) from Gatheral &
// Jacquier (2014).

/// Raw SVI parameters for a single expiry slice.
///
/// Total implied variance as a function of log-moneyness `k = ln(K/F)`:
///
/// `w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RawSviParams {
    pub a: f64,
    pub b: f64,
    pub rho: f64,
    pub m: f64,
    pub sigma: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SviValidationError {
    NegativeB,
    RhoOutOfBounds,
    NonPositiveSigma,
    NegativeMinVariance,
    ButterflyArbitrage,
}

impl RawSviParams {
    /// Validate the static (single-slice) no-arbitrage constraints from
    /// Gatheral & Jacquier (2014), section 2:
    /// - b >= 0
    /// - |rho| < 1
    /// - sigma > 0
    /// - a + b * sigma * sqrt(1 - rho^2) >= 0  (minimum total variance >= 0)
    pub fn validate_static(&self) -> Result<(), SviValidationError> {
        if self.b < 0.0 {
            return Err(SviValidationError::NegativeB);
        }
        if self.rho.abs() >= 1.0 {
            return Err(SviValidationError::RhoOutOfBounds);
        }
        if self.sigma <= 0.0 {
            return Err(SviValidationError::NonPositiveSigma);
        }
        let min_variance = self.a + self.b * self.sigma * (1.0 - self.rho * self.rho).sqrt();
        if min_variance < -1e-12 {
            return Err(SviValidationError::NegativeMinVariance);
        }
        Ok(())
    }

    /// Total implied variance `w(k) = sigma_BS(k)^2 * T` at log-moneyness k.
    pub fn total_variance(&self, k: f64) -> f64 {
        let dk = k - self.m;
        self.a + self.b * (self.rho * dk + (dk * dk + self.sigma * self.sigma).sqrt())
    }

    /// Black-Scholes implied volatility at log-moneyness k for expiry T (years).
    pub fn implied_vol(&self, k: f64, t: f64) -> f64 {
        (self.total_variance(k) / t).max(0.0).sqrt()
    }

    /// First derivative dw/dk, needed for the butterfly-arbitrage check.
    fn dw_dk(&self, k: f64) -> f64 {
        let dk = k - self.m;
        self.b * (self.rho + dk / (dk * dk + self.sigma * self.sigma).sqrt())
    }

    /// Second derivative d^2w/dk^2.
    fn d2w_dk2(&self, k: f64) -> f64 {
        let dk = k - self.m;
        let denom = (dk * dk + self.sigma * self.sigma).powf(1.5);
        self.b * self.sigma * self.sigma / denom
    }

    /// Durrleman's butterfly-arbitrage function g(k) (Gatheral & Jacquier
    /// 2014, eq. 2.5). The slice is free of butterfly arbitrage on its
    /// domain iff g(k) >= 0 for all k, equivalent to the density implied
    /// by the smile staying non-negative everywhere.
    pub fn g(&self, k: f64) -> f64 {
        let w = self.total_variance(k);
        if w <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let wp = self.dw_dk(k);
        let wpp = self.d2w_dk2(k);
        let term1 = 1.0 - (k * wp) / (2.0 * w);
        let term2 = (wp * wp / 4.0) * (1.0 / w + 0.25);
        term1 * term1 - term2 + wpp / 2.0
    }

    /// Scan g(k) over a grid to flag butterfly arbitrage. This is a
    /// numerical sufficiency check, not a closed-form proof -- widen the
    /// grid/range for wide, low-liquidity wings.
    pub fn check_butterfly_arbitrage(
        &self,
        k_min: f64,
        k_max: f64,
        n: usize,
    ) -> Result<(), SviValidationError> {
        let step = (k_max - k_min) / (n as f64 - 1.0);
        for i in 0..n {
            let k = k_min + step * i as f64;
            if self.g(k) < -1e-8 {
                return Err(SviValidationError::ButterflyArbitrage);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_params() -> RawSviParams {
        // Roughly BTC-shaped short-dated smile: modest left skew, modest curvature.
        RawSviParams {
            a: 0.02,
            b: 0.18,
            rho: -0.35,
            m: 0.0,
            sigma: 0.15,
        }
    }

    #[test]
    fn validate_static_accepts_sane_params() {
        assert!(sample_params().validate_static().is_ok());
    }

    #[test]
    fn validate_static_rejects_negative_b() {
        let p = RawSviParams {
            b: -0.1,
            ..sample_params()
        };
        assert_eq!(p.validate_static(), Err(SviValidationError::NegativeB));
    }

    #[test]
    fn validate_static_rejects_rho_out_of_bounds() {
        let p = RawSviParams {
            rho: 1.2,
            ..sample_params()
        };
        assert_eq!(p.validate_static(), Err(SviValidationError::RhoOutOfBounds));
    }

    #[test]
    fn validate_static_rejects_negative_min_variance() {
        let p = RawSviParams {
            a: -1.0,
            b: 0.1,
            rho: 0.0,
            m: 0.0,
            sigma: 0.1,
        };
        assert_eq!(
            p.validate_static(),
            Err(SviValidationError::NegativeMinVariance)
        );
    }

    #[test]
    fn total_variance_is_symmetric_at_atm_when_rho_zero() {
        let p = RawSviParams {
            rho: 0.0,
            ..sample_params()
        };
        let w_up = p.total_variance(0.1);
        let w_down = p.total_variance(-0.1);
        assert!((w_up - w_down).abs() < 1e-12);
    }

    #[test]
    fn implied_vol_matches_sqrt_total_variance_over_t() {
        let p = sample_params();
        let t = 30.0 / 365.0;
        let k = 0.05;
        let expected = (p.total_variance(k) / t).sqrt();
        assert!((p.implied_vol(k, t) - expected).abs() < 1e-12);
    }

    #[test]
    fn no_butterfly_arbitrage_on_sane_smile() {
        let p = sample_params();
        assert!(p.check_butterfly_arbitrage(-2.0, 2.0, 200).is_ok());
    }

    #[test]
    fn extreme_curvature_triggers_butterfly_arbitrage_flag() {
        // b large relative to sigma forces curvature that violates g(k) >= 0.
        let p = RawSviParams {
            a: 0.0001,
            b: 5.0,
            rho: 0.0,
            m: 0.0,
            sigma: 0.01,
        };
        assert!(p.check_butterfly_arbitrage(-1.0, 1.0, 400).is_err());
    }
}
