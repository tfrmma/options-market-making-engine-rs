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
        Self { base_tick_size: 0.0001, steps: vec![TickStep { above_price: 0.005, tick_size: 0.0005 }] }
    }

    pub fn tick_for(&self, price: f64) -> f64 {
        self.steps.iter().rev().find(|s| price > s.above_price).map(|s| s.tick_size).unwrap_or(self.base_tick_size)
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
}
