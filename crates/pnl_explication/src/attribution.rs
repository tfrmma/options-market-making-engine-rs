// Decomposes a position's realized PnL between two market snapshots into
// delta/gamma/vega/vanna/volga/theta contributions, using the Greeks
// measured at the start snapshot (standard practice, not "correct" in any
// deeper sense, a Taylor expansion around the end snapshot would give
// different numbers). Whatever the second-order expansion doesn't explain
// falls out as unexplained_pnl, which is where model misspecification, a
// stale surface, or a bad fill would actually show up.
//
// Second order in (forward, vol): delta/gamma/vega/vanna/volga. First order
// only in time: just theta*dt, no charm (d(delta)/dt) term. That's the
// conventional choice for PnL explain, time evolution is deterministic and
// smooth, not something you're trying to catch model risk in the way you
// are for F and vol, so the extra cross-term isn't worth the complexity here.

use book_risk::{greeks as compute_greeks, price_coin, OptionType};
use std::ops::AddAssign;

#[derive(Debug, Clone, Copy)]
pub struct MarketSnapshot {
    pub forward: f64,
    pub vol: f64,
    /// time to expiry remaining AT this snapshot, not elapsed time
    pub expiry_years: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Attribution {
    pub delta_pnl: f64,
    pub gamma_pnl: f64,
    pub vega_pnl: f64,
    pub vanna_pnl: f64,
    pub volga_pnl: f64,
    pub theta_pnl: f64,
    pub explained_pnl: f64,
    pub realized_pnl: f64,
    pub unexplained_pnl: f64,
}

impl AddAssign for Attribution {
    fn add_assign(&mut self, o: Attribution) {
        self.delta_pnl += o.delta_pnl;
        self.gamma_pnl += o.gamma_pnl;
        self.vega_pnl += o.vega_pnl;
        self.vanna_pnl += o.vanna_pnl;
        self.volga_pnl += o.volga_pnl;
        self.theta_pnl += o.theta_pnl;
        self.explained_pnl += o.explained_pnl;
        self.realized_pnl += o.realized_pnl;
        self.unexplained_pnl += o.unexplained_pnl;
    }
}

/// Attributes one position's PnL between two snapshots. `size` is signed
/// coins, same convention as book_risk::OptionPosition, multiplies straight
/// through every term.
pub fn attribute_position(
    option_type: OptionType,
    strike: f64,
    start: MarketSnapshot,
    end: MarketSnapshot,
    size: f64,
) -> Attribution {
    let g = compute_greeks(
        option_type,
        start.forward,
        strike,
        start.vol,
        start.expiry_years,
    );
    let start_price = price_coin(
        option_type,
        start.forward,
        strike,
        start.vol,
        start.expiry_years,
    );
    let end_price = price_coin(option_type, end.forward, strike, end.vol, end.expiry_years);

    let d_forward = end.forward - start.forward;
    let d_vol = end.vol - start.vol;
    let d_calendar_time = start.expiry_years - end.expiry_years; // positive as time passes forward

    let delta_pnl = g.delta * d_forward * size;
    let gamma_pnl = 0.5 * g.gamma * d_forward * d_forward * size;
    let vega_pnl = g.vega * d_vol * size;
    let vanna_pnl = g.vanna * d_forward * d_vol * size;
    let volga_pnl = 0.5 * g.volga * d_vol * d_vol * size;
    let theta_pnl = g.theta * d_calendar_time * size;

    let explained_pnl = delta_pnl + gamma_pnl + vega_pnl + vanna_pnl + volga_pnl + theta_pnl;
    let realized_pnl = (end_price - start_price) * size;

    Attribution {
        delta_pnl,
        gamma_pnl,
        vega_pnl,
        vanna_pnl,
        volga_pnl,
        theta_pnl,
        explained_pnl,
        realized_pnl,
        unexplained_pnl: realized_pnl - explained_pnl,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PositionMove {
    pub option_type: OptionType,
    pub strike: f64,
    pub size: f64,
    pub start: MarketSnapshot,
    pub end: MarketSnapshot,
}

/// Book-level attribution: just the sum of each position's own attribution,
/// each measured against its own Greeks, not the book's aggregate Greeks.
/// Summing per-position keeps this exact regardless of how the book's mix
/// of strikes/expiries changed shape between snapshots.
pub fn attribute_book(moves: &[PositionMove]) -> Attribution {
    let mut total = Attribution::default();
    for m in moves {
        total += attribute_position(m.option_type, m.strike, m.start, m.end, m.size);
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_snapshot() -> MarketSnapshot {
        MarketSnapshot {
            forward: 65000.0,
            vol: 0.6,
            expiry_years: 30.0 / 365.0,
        }
    }

    #[test]
    fn explained_plus_unexplained_always_equals_realized() {
        let start = base_snapshot();
        let end = MarketSnapshot {
            forward: 67000.0,
            vol: 0.55,
            expiry_years: 25.0 / 365.0,
        };
        let a = attribute_position(OptionType::Call, 65000.0, start, end, 3.0);
        assert!((a.explained_pnl + a.unexplained_pnl - a.realized_pnl).abs() < 1e-12);
    }

    #[test]
    fn unexplained_residual_shrinks_faster_than_the_move_size_for_forward_and_vol_moves() {
        // second order in (F, vol), so halving the move should shrink the
        // residual by roughly 1/8, well under the 1/2 a first-order scheme would give
        let start = base_snapshot();
        let big_end = MarketSnapshot {
            forward: 66000.0,
            vol: 0.62,
            expiry_years: start.expiry_years,
        };
        let small_end = MarketSnapshot {
            forward: 65500.0,
            vol: 0.61,
            expiry_years: start.expiry_years,
        };

        let big = attribute_position(OptionType::Call, 65000.0, start, big_end, 1.0);
        let small = attribute_position(OptionType::Call, 65000.0, start, small_end, 1.0);

        let ratio = small.unexplained_pnl.abs() / big.unexplained_pnl.abs();
        assert!(
            ratio < 0.5,
            "halving the move should shrink the residual by much more than half, ratio={ratio}"
        );
    }

    #[test]
    fn pure_time_decay_is_explained_almost_entirely_by_theta() {
        let start = base_snapshot();
        let end = MarketSnapshot {
            forward: start.forward,
            vol: start.vol,
            expiry_years: start.expiry_years - 1.0 / 365.0,
        };
        let a = attribute_position(OptionType::Call, 65000.0, start, end, 1.0);
        assert_eq!(a.delta_pnl, 0.0);
        assert_eq!(a.gamma_pnl, 0.0);
        assert_eq!(a.vega_pnl, 0.0);
        assert!(
            (a.theta_pnl - a.realized_pnl).abs() / a.realized_pnl.abs() < 0.05,
            "theta_pnl={} realized_pnl={}",
            a.theta_pnl,
            a.realized_pnl
        );
    }

    #[test]
    fn short_position_flips_the_sign_of_realized_pnl() {
        let start = base_snapshot();
        let end = MarketSnapshot {
            forward: 67000.0,
            vol: 0.6,
            expiry_years: start.expiry_years,
        };
        let long = attribute_position(OptionType::Call, 65000.0, start, end, 1.0);
        let short = attribute_position(OptionType::Call, 65000.0, start, end, -1.0);
        assert!((long.realized_pnl + short.realized_pnl).abs() < 1e-12);
    }

    #[test]
    fn book_attribution_matches_the_sum_of_individual_positions() {
        let start = base_snapshot();
        let end = MarketSnapshot {
            forward: 66500.0,
            vol: 0.58,
            expiry_years: 20.0 / 365.0,
        };

        let moves = vec![
            PositionMove {
                option_type: OptionType::Call,
                strike: 65000.0,
                size: 2.0,
                start,
                end,
            },
            PositionMove {
                option_type: OptionType::Put,
                strike: 60000.0,
                size: -1.5,
                start,
                end,
            },
        ];

        let book_total = attribute_book(&moves);

        let mut manual_total = Attribution::default();
        manual_total += attribute_position(OptionType::Call, 65000.0, start, end, 2.0);
        manual_total += attribute_position(OptionType::Put, 60000.0, start, end, -1.5);

        assert!((book_total.realized_pnl - manual_total.realized_pnl).abs() < 1e-9);
        assert!((book_total.unexplained_pnl - manual_total.unexplained_pnl).abs() < 1e-9);
    }
}
