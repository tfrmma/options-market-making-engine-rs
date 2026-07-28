// Aggregates positions into book-level Greeks, one forward per expiry, not one shared number for the whole book.

use std::collections::BTreeMap;
use std::ops::AddAssign;

use vol_surface::VolSurface;

use crate::inverse_option::InverseGreeks;
use crate::position::OptionPosition;

#[derive(Debug, Clone, Copy, Default)]
pub struct BookGreeks {
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
}

impl AddAssign<InverseGreeks> for BookGreeks {
    fn add_assign(&mut self, g: InverseGreeks) {
        self.delta += g.delta;
        self.gamma += g.gamma;
        self.vega += g.vega;
        self.theta += g.theta;
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ForwardMatch {
    Exact,
    Interpolated,
    /// expiry_years fell outside the listed range, flat-extrapolated from
    /// whichever end was closer. Forward curves have real term structure,
    /// silently flat-extrapolating an arbitrary distance is a much bigger
    /// assumption than vol-surface's flat vol extrapolation, so this is
    /// surfaced instead of hidden, the caller decides whether to trust it.
    Extrapolated { nearest_listed_expiry_years: f64 },
}

#[derive(Debug, Clone, Copy)]
pub struct ForwardLookup {
    pub forward: f64,
    pub match_kind: ForwardMatch,
}

pub struct ForwardCurve {
    expiries: Vec<f64>,
    forwards: Vec<f64>,
}

impl ForwardCurve {
    pub fn new(mut points: Vec<(f64, f64)>) -> Self {
        points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let expiries = points.iter().map(|p| p.0).collect();
        let forwards = points.iter().map(|p| p.1).collect();
        Self { expiries, forwards }
    }

    pub fn forward_for(&self, expiry_years: f64) -> ForwardLookup {
        let first = self.expiries[0];
        let last = *self.expiries.last().unwrap();

        if expiry_years <= first {
            let forward = self.forwards[0];
            return if (expiry_years - first).abs() < 1e-9 {
                ForwardLookup { forward, match_kind: ForwardMatch::Exact }
            } else {
                ForwardLookup { forward, match_kind: ForwardMatch::Extrapolated { nearest_listed_expiry_years: first } }
            };
        }
        if expiry_years >= last {
            let forward = *self.forwards.last().unwrap();
            return if (expiry_years - last).abs() < 1e-9 {
                ForwardLookup { forward, match_kind: ForwardMatch::Exact }
            } else {
                ForwardLookup { forward, match_kind: ForwardMatch::Extrapolated { nearest_listed_expiry_years: last } }
            };
        }

        // strictly between two listed expiries here, find the bracket and interpolate
        let hi = self.expiries.partition_point(|&e| e < expiry_years);
        let lo = hi - 1;
        if (self.expiries[hi] - expiry_years).abs() < 1e-9 {
            return ForwardLookup { forward: self.forwards[hi], match_kind: ForwardMatch::Exact };
        }
        let weight = (expiry_years - self.expiries[lo]) / (self.expiries[hi] - self.expiries[lo]);
        let forward = self.forwards[lo] + weight * (self.forwards[hi] - self.forwards[lo]);
        ForwardLookup { forward, match_kind: ForwardMatch::Interpolated }
    }
}

#[derive(Debug, Default)]
pub struct BookRisk {
    pub total: BookGreeks,
    pub by_expiry: BTreeMap<u64, BookGreeks>,
}

impl BookRisk {
    pub fn bucket_near(&self, expiry_years: f64) -> BookGreeks {
        self.by_expiry.get(&bucket_key(expiry_years)).copied().unwrap_or_default()
    }
}

// rounds to the nearest hour so float noise in T doesn't fragment one expiry into a dozen buckets
fn bucket_key(expiry_years: f64) -> u64 {
    (expiry_years * 365.0 * 24.0).round() as u64
}

pub fn aggregate(positions: &[OptionPosition], surface: &VolSurface, forwards: &ForwardCurve) -> BookRisk {
    let mut risk = BookRisk::default();

    for pos in positions {
        let strike = pos.instrument.strike;
        let t = pos.instrument.expiry_years;
        let forward = forwards.forward_for(t).forward;
        let log_moneyness = (strike / forward).ln();
        let vol = surface.implied_vol(log_moneyness, t);

        let g = pos.greeks(forward, vol);
        risk.total += g;
        *risk.by_expiry.entry(bucket_key(t)).or_default() += g;
    }

    risk
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inverse_option::OptionType;
    use crate::position::OptionKey;
    use vol_surface::{RawSviParams, Slice};

    fn flat_surface(a: f64) -> VolSurface {
        let params = RawSviParams { a, b: 0.15, rho: -0.3, m: 0.0, sigma: 0.15 };
        VolSurface::build(vec![
            Slice { expiry_years: 7.0 / 365.0, params },
            Slice { expiry_years: 60.0 / 365.0, params },
        ])
        .unwrap()
    }

    #[test]
    fn two_offsetting_positions_net_close_to_zero() {
        let surface = flat_surface(0.04);
        let forwards = ForwardCurve::new(vec![(14.0 / 365.0, 65000.0)]);
        let instrument = OptionKey { option_type: OptionType::Call, strike: 65000.0, expiry_years: 14.0 / 365.0 };
        let positions = vec![
            OptionPosition { instrument, size: 3.0 },
            OptionPosition { instrument, size: -3.0 },
        ];

        let risk = aggregate(&positions, &surface, &forwards);
        assert!(risk.total.delta.abs() < 1e-12);
        assert!(risk.total.vega.abs() < 1e-12);
    }

    #[test]
    fn positions_at_different_expiries_land_in_different_buckets() {
        let surface = flat_surface(0.04);
        let forwards = ForwardCurve::new(vec![(7.0 / 365.0, 65000.0), (60.0 / 365.0, 66000.0)]);
        let short_dated = OptionKey { option_type: OptionType::Call, strike: 65000.0, expiry_years: 7.0 / 365.0 };
        let long_dated = OptionKey { option_type: OptionType::Call, strike: 65000.0, expiry_years: 60.0 / 365.0 };
        let positions = vec![
            OptionPosition { instrument: short_dated, size: 1.0 },
            OptionPosition { instrument: long_dated, size: 1.0 },
        ];

        let risk = aggregate(&positions, &surface, &forwards);
        assert_eq!(risk.by_expiry.len(), 2);
        // book total is just the sum of the two buckets
        let bucket_sum: f64 = risk.by_expiry.values().map(|g| g.delta).sum();
        assert!((bucket_sum - risk.total.delta).abs() < 1e-12);
    }

    #[test]
    fn forward_curve_interpolates_between_listed_expiries() {
        let forwards = ForwardCurve::new(vec![(7.0 / 365.0, 65000.0), (60.0 / 365.0, 70000.0)]);
        let midpoint_t = (7.0 + 60.0) / 2.0 / 365.0;
        let lookup = forwards.forward_for(midpoint_t);
        assert!((lookup.forward - 67500.0).abs() < 1e-6, "forward={}", lookup.forward);
        assert_eq!(lookup.match_kind, ForwardMatch::Interpolated);
    }

    #[test]
    fn forward_curve_reports_exact_matches_as_exact_not_interpolated() {
        let forwards = ForwardCurve::new(vec![(7.0 / 365.0, 65000.0), (60.0 / 365.0, 70000.0)]);
        let lookup = forwards.forward_for(60.0 / 365.0);
        assert_eq!(lookup.forward, 70000.0);
        assert_eq!(lookup.match_kind, ForwardMatch::Exact);
    }

    #[test]
    fn forward_curve_flags_extrapolation_instead_of_hiding_it() {
        let forwards = ForwardCurve::new(vec![(7.0 / 365.0, 65000.0), (60.0 / 365.0, 70000.0)]);
        let lookup = forwards.forward_for(365.0 / 365.0); // way past the longest listed future
        assert_eq!(lookup.forward, 70000.0); // still flat-extrapolates, just doesn't hide that it did
        assert_eq!(lookup.match_kind, ForwardMatch::Extrapolated { nearest_listed_expiry_years: 60.0 / 365.0 });
    }

    #[test]
    fn single_listed_expiry_extrapolates_flat_in_both_directions() {
        let forwards = ForwardCurve::new(vec![(14.0 / 365.0, 65000.0)]);
        assert_eq!(forwards.forward_for(7.0 / 365.0).forward, 65000.0);
        assert_eq!(forwards.forward_for(30.0 / 365.0).forward, 65000.0);
        assert_eq!(forwards.forward_for(14.0 / 365.0).match_kind, ForwardMatch::Exact);
    }
}
