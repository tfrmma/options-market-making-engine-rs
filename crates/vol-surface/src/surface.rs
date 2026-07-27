// Sorted set of calibrated SVI slices, one per listed expiry, with
// calendar-arb checking between them and interpolation in between.

use crate::svi::{RawSviParams, SviValidationError};

#[derive(Debug, Clone, Copy)]
pub struct Slice {
    pub expiry_years: f64,
    pub params: RawSviParams,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceError {
    EmptySurface,
    UnsortedExpiries,
    CalendarArbitrage { expiry_short: usize, expiry_long: usize },
    SliceInvalid(SviValidationError),
}

#[derive(Debug)]
pub struct VolSurface {
    slices: Vec<Slice>, // sorted ascending by expiry_years
}

// grid used for the calendar no-arb scan below, wide enough to catch
// crossings in the wings without being absurdly slow
const K_GRID_MIN: f64 = -2.0;
const K_GRID_MAX: f64 = 2.0;
const K_GRID_N: usize = 81;

impl VolSurface {
    /// Validates each slice individually, then checks that total variance is
    /// non-decreasing in T at every point on the log-moneyness grid between
    /// every consecutive pair of expiries (Gatheral & Jacquier 2014, sec. 4).
    pub fn build(mut slices: Vec<Slice>) -> Result<Self, SurfaceError> {
        if slices.is_empty() {
            return Err(SurfaceError::EmptySurface);
        }
        slices.sort_by(|a, b| a.expiry_years.partial_cmp(&b.expiry_years).unwrap());

        for (i, s) in slices.iter().enumerate() {
            s.params.validate_static().map_err(SurfaceError::SliceInvalid)?;
            if i > 0 && slices[i - 1].expiry_years >= s.expiry_years {
                return Err(SurfaceError::UnsortedExpiries);
            }
        }

        let step = (K_GRID_MAX - K_GRID_MIN) / (K_GRID_N as f64 - 1.0);
        for i in 1..slices.len() {
            for j in 0..K_GRID_N {
                let k = K_GRID_MIN + step * j as f64;
                let w_short = slices[i - 1].params.total_variance(k);
                let w_long = slices[i].params.total_variance(k);
                if w_long < w_short - 1e-9 {
                    return Err(SurfaceError::CalendarArbitrage { expiry_short: i - 1, expiry_long: i });
                }
            }
        }

        Ok(Self { slices })
    }

    pub fn slices(&self) -> &[Slice] {
        &self.slices
    }

    /// Total variance at arbitrary (k, t). Flat-extrapolates past the listed
    /// tenors, linearly interpolates in variance between them, which stays
    /// arb-free since it's a convex combination of two non-crossing curves.
    pub fn total_variance(&self, k: f64, t: f64) -> f64 {
        if t <= self.slices[0].expiry_years {
            return self.slices[0].params.total_variance(k);
        }
        let last = self.slices.len() - 1;
        if t >= self.slices[last].expiry_years {
            return self.slices[last].params.total_variance(k);
        }
        let i = self.slices.partition_point(|s| s.expiry_years < t);
        let lo = &self.slices[i - 1];
        let hi = &self.slices[i];
        let w_lo = lo.params.total_variance(k);
        let w_hi = hi.params.total_variance(k);
        let weight = (t - lo.expiry_years) / (hi.expiry_years - lo.expiry_years);
        w_lo + weight * (w_hi - w_lo)
    }

    // expired/expiring: total_variance/t would blow up or divide by <=0, 0.0
    // is the right answer, an expired option has no remaining implied vol to speak of
    pub fn implied_vol(&self, k: f64, t: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        (self.total_variance(k, t) / t).max(0.0).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::svi::RawSviParams;

    fn slice(expiry: f64, a: f64) -> Slice {
        Slice { expiry_years: expiry, params: RawSviParams { a, b: 0.15, rho: -0.3, m: 0.0, sigma: 0.15 } }
    }

    #[test]
    fn interpolated_variance_matches_endpoints_exactly() {
        let s = VolSurface::build(vec![slice(7.0 / 365.0, 0.01), slice(30.0 / 365.0, 0.03)]).unwrap();
        let w_short = s.total_variance(0.0, 7.0 / 365.0);
        let w_long = s.total_variance(0.0, 30.0 / 365.0);
        assert!((w_short - s.slices()[0].params.total_variance(0.0)).abs() < 1e-12);
        assert!((w_long - s.slices()[1].params.total_variance(0.0)).abs() < 1e-12);
    }

    #[test]
    fn interpolated_midpoint_is_between_the_two_endpoints() {
        let s = VolSurface::build(vec![slice(7.0 / 365.0, 0.01), slice(30.0 / 365.0, 0.03)]).unwrap();
        let mid_t = (7.0 + 30.0) / 2.0 / 365.0;
        let w_mid = s.total_variance(0.0, mid_t);
        let w_short = s.slices()[0].params.total_variance(0.0);
        let w_long = s.slices()[1].params.total_variance(0.0);
        assert!(w_mid > w_short && w_mid < w_long);
    }

    #[test]
    fn detects_calendar_arbitrage_when_later_slice_has_lower_variance() {
        let result = VolSurface::build(vec![slice(7.0 / 365.0, 0.05), slice(30.0 / 365.0, 0.005)]);
        assert!(matches!(result, Err(SurfaceError::CalendarArbitrage { .. })));
    }

    #[test]
    fn rejects_invalid_slice_before_checking_calendar_arb() {
        let bad = Slice { expiry_years: 7.0 / 365.0, params: RawSviParams { a: 0.0, b: -1.0, rho: 0.0, m: 0.0, sigma: 0.1 } };
        let result = VolSurface::build(vec![bad, slice(30.0 / 365.0, 0.03)]);
        assert!(matches!(result, Err(SurfaceError::SliceInvalid(_))));
    }

    #[test]
    fn flat_extrapolates_beyond_listed_tenors() {
        let s = VolSurface::build(vec![slice(7.0 / 365.0, 0.01), slice(30.0 / 365.0, 0.03)]).unwrap();
        let w_far = s.total_variance(0.0, 365.0 / 365.0);
        let w_last = s.slices()[1].params.total_variance(0.0);
        assert!((w_far - w_last).abs() < 1e-12);
    }

    #[test]
    fn empty_surface_is_rejected() {
        assert_eq!(VolSurface::build(vec![]).unwrap_err(), SurfaceError::EmptySurface);
    }

    #[test]
    fn implied_vol_is_zero_at_and_past_expiry_instead_of_blowing_up() {
        let s = VolSurface::build(vec![slice(7.0 / 365.0, 0.01), slice(30.0 / 365.0, 0.03)]).unwrap();
        assert_eq!(s.implied_vol(0.0, 0.0), 0.0);
        assert_eq!(s.implied_vol(0.0, -1.0), 0.0);
        assert!(s.implied_vol(0.0, 0.0).is_finite());
    }

    #[test]
    fn implied_vol_stays_finite_for_a_very_small_positive_t() {
        let s = VolSurface::build(vec![slice(7.0 / 365.0, 0.01), slice(30.0 / 365.0, 0.03)]).unwrap();
        let v = s.implied_vol(0.0, 1e-9);
        assert!(v.is_finite() && v >= 0.0, "v={v}");
    }
}
