// Mirrors the exact schema Deribit returns from public/get_instrument: a
// base tick_size plus tick_size_steps, {above_price, tick_size} pairs that
// raise the tick above certain price levels.
//
// TODO: doesn't model Deribit's order-price bandwidth clamp against their
// portfolio margin risk matrix (a "minimum trading bandwidth constant" of
// 0.015 per their docs). That bandwidth applies to a risk-matrix price-bucket
// move that isn't public, applying 0.015 to the wrong base quantity would
// produce a plausible-looking but wrong number, so left undone on purpose.

#[derive(Debug, Clone)]
pub struct TickStep {
    pub above_price: f64,
    pub tick_size: f64,
}

#[derive(Debug, Clone)]
pub struct TickSchedule {
    pub base_tick_size: f64,
    pub steps: Vec<TickStep>, // must be sorted ascending by above_price, not enforced
}

impl TickSchedule {
    // BTC/ETH default as of the July 2023 tick size change. Fallback only,
    // production should pull the live schedule off get_instrument, Deribit
    // has changed this before.
    pub fn deribit_btc_option_default() -> Self {
        Self {
            base_tick_size: 0.0001,
            steps: vec![TickStep {
                above_price: 0.005,
                tick_size: 0.0005,
            }],
        }
    }

    pub fn tick_for(&self, price: f64) -> f64 {
        self.steps
            .iter()
            .rev()
            .find(|s| price > s.above_price)
            .map(|s| s.tick_size)
            .unwrap_or(self.base_tick_size)
    }

    pub fn round_down(&self, price: f64) -> f64 {
        let tick = self.tick_for(price);
        // price/tick can land at e.g. 41.999999999999993 instead of 42.0
        // for perfectly on-tick prices, the epsilon below absorbs that
        // without being big enough to round a genuinely-off-tick price
        // onto the wrong tick.
        (price / tick + 1e-7).floor() * tick
    }

    pub fn round_up(&self, price: f64) -> f64 {
        let tick = self.tick_for(price);
        (price / tick - 1e-7).ceil() * tick
    }
}

/// Not Deribit's bandwidth mechanism (see the module TODO above, that one's
/// still unmodeled on purpose). This is a plain internal safety net: if a
/// bug upstream (bad vol, bad forward, whatever) produces a price wildly
/// off from theoretical, this stops it from going out rather than trusting
/// Deribit to catch it. `max_deviation` is a fraction, e.g. 0.5 means don't
/// let the quoted price sit more than 50% away from `theoretical_price`.
pub fn sanity_clamp(price: f64, theoretical_price: f64, max_deviation: f64) -> f64 {
    if theoretical_price <= 0.0 {
        return price;
    }
    let lo = theoretical_price * (1.0 - max_deviation);
    let hi = theoretical_price * (1.0 + max_deviation);
    price.clamp(lo.max(0.0), hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounds_within_the_base_tick_below_the_step() {
        let s = TickSchedule::deribit_btc_option_default();
        assert!((s.round_down(0.00347) - 0.0034).abs() < 1e-12);
        assert!((s.round_up(0.00341) - 0.0035).abs() < 1e-12);
    }

    #[test]
    fn switches_to_the_wider_tick_above_the_step() {
        let s = TickSchedule::deribit_btc_option_default();
        // above 0.005, tick is 0.0005, not 0.0001
        assert!((s.round_down(0.00734) - 0.0070).abs() < 1e-12);
        assert!((s.round_up(0.00701) - 0.0075).abs() < 1e-12);
    }

    #[test]
    fn price_exactly_at_the_step_boundary_uses_the_base_tick() {
        // Deribit's own wording is "if price <= 0.005 then tick is 0.0001",
        // the step only applies strictly above the boundary
        let s = TickSchedule::deribit_btc_option_default();
        assert!((s.tick_for(0.005) - 0.0001).abs() < 1e-15);
        assert!((s.tick_for(0.0050001) - 0.0005).abs() < 1e-15);
    }

    #[test]
    fn already_on_tick_round_trips_unchanged() {
        let s = TickSchedule::deribit_btc_option_default();
        assert!((s.round_down(0.0042) - 0.0042).abs() < 1e-12);
        assert!((s.round_up(0.0042) - 0.0042).abs() < 1e-12);
    }

    #[test]
    fn sanity_clamp_passes_through_a_reasonable_price() {
        assert_eq!(sanity_clamp(0.052, 0.05, 0.5), 0.052);
    }

    #[test]
    fn sanity_clamp_catches_a_wildly_off_price() {
        // 10x theoretical, way outside a 50% band
        let clamped = sanity_clamp(0.5, 0.05, 0.5);
        assert!((clamped - 0.075).abs() < 1e-12, "clamped={clamped}");
    }

    #[test]
    fn sanity_clamp_is_a_noop_when_theoretical_price_is_non_positive() {
        // no reference to clamp against, pass the price through rather than guess
        assert_eq!(sanity_clamp(0.5, 0.0, 0.5), 0.5);
    }
}
