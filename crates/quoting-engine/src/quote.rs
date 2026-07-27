// Ties reservation.rs + spread.rs + tick.rs into one bid/ask per instrument.
// Output fields mirror Deribit's Mass Quote QuoteEntry field-for-field.

use book_risk::{price_coin, BookRisk, OptionType};

use crate::reservation::{reservation_vol, RiskAversion};
use crate::spread::{half_spread_vol, widen_for_toxicity, SpreadParams};
use crate::tick::TickSchedule;

pub struct QuoteRequest<'a> {
    pub option_type: OptionType,
    pub strike: f64,
    pub expiry_years: f64,
    pub forward: f64,
    pub mid_vol: f64,
    pub book: &'a BookRisk,
    pub risk_aversion: RiskAversion,
    pub spread_params: SpreadParams,
    pub toxicity_score: f64,
    pub toxicity_widen_factor: f64,
    pub tick_schedule: &'a TickSchedule,
    pub base_size: f64,
    pub min_trade_amount: f64,
    // TODO: throttle floor isn't tied to a real MMP group's actual vega limit yet, wire that up.
    pub max_bucket_vega: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct QuoteEntry {
    pub option_type: OptionType,
    pub strike: f64,
    pub expiry_years: f64,
    pub bid_price: f64,
    pub ask_price: f64,
    pub bid_size: f64,
    pub ask_size: f64,
    pub bid_vol: f64,
    pub ask_vol: f64,
}

// None = don't quote this instrument right now, either the throttled size fell under
// min_trade_amount or the inputs were nonsense. Sending an odd-lot quote is worse than skipping it.
pub fn build_quote(req: &QuoteRequest) -> Option<QuoteEntry> {
    if req.mid_vol <= 0.0 || req.expiry_years <= 0.0 || req.forward <= 0.0 {
        return None;
    }

    let bucket = req.book.bucket_near(req.expiry_years);
    let r_vol = reservation_vol(req.mid_vol, &bucket, req.risk_aversion);
    let half = widen_for_toxicity(half_spread_vol(&req.spread_params), req.toxicity_score, req.toxicity_widen_factor);

    let bid_vol = (r_vol - half).max(1e-4);
    let ask_vol = r_vol + half;

    let bid_price_raw = price_coin(req.option_type, req.forward, req.strike, bid_vol, req.expiry_years);
    let ask_price_raw = price_coin(req.option_type, req.forward, req.strike, ask_vol, req.expiry_years);

    let bid_price = req.tick_schedule.round_down(bid_price_raw);
    let mut ask_price = req.tick_schedule.round_up(ask_price_raw);
    // rounding in opposite directions can occasionally collapse a thin
    // spread to zero width, never send a crossed or locked quote
    if ask_price <= bid_price {
        ask_price = bid_price + req.tick_schedule.tick_for(bid_price);
    }

    let size = quote_size(req.base_size, bucket.vega, req.max_bucket_vega);
    if size < req.min_trade_amount {
        return None;
    }

    Some(QuoteEntry {
        option_type: req.option_type,
        strike: req.strike,
        expiry_years: req.expiry_years,
        bid_price,
        ask_price,
        bid_size: size,
        ask_size: size,
        bid_vol,
        ask_vol,
    })
}

// linear throttle toward a floor, never zero, MMP is what actually pulls quotes under real stress
fn quote_size(base_size: f64, bucket_vega: f64, max_bucket_vega: f64) -> f64 {
    if max_bucket_vega <= 0.0 {
        return base_size;
    }
    let utilization = (bucket_vega.abs() / max_bucket_vega).min(1.0);
    let floor_fraction = 0.1;
    base_size * (1.0 - utilization * (1.0 - floor_fraction))
}

#[cfg(test)]
mod tests {
    use super::*;
    use book_risk::{aggregate, OptionKey, OptionPosition};
    use vol_surface::{RawSviParams, Slice, VolSurface};

    fn flat_surface() -> VolSurface {
        let params = RawSviParams { a: 0.04, b: 0.15, rho: -0.3, m: 0.0, sigma: 0.15 };
        VolSurface::build(vec![
            Slice { expiry_years: 7.0 / 365.0, params },
            Slice { expiry_years: 30.0 / 365.0, params },
        ])
        .unwrap()
    }

    fn base_request<'a>(book: &'a BookRisk, tick_schedule: &'a TickSchedule) -> QuoteRequest<'a> {
        QuoteRequest {
            option_type: OptionType::Call,
            strike: 65000.0,
            expiry_years: 14.0 / 365.0,
            forward: 65000.0,
            mid_vol: 0.6,
            book,
            risk_aversion: RiskAversion { vega: 0.02, gamma: 0.01 },
            spread_params: SpreadParams { risk_aversion: 0.3, vol_of_vol: 1.2, horizon_years: 1.0 / 365.0, kappa: 8.0 },
            toxicity_score: 0.0,
            toxicity_widen_factor: 2.0,
            tick_schedule,
            base_size: 10.0,
            min_trade_amount: 0.1,
            max_bucket_vega: 500.0,
        }
    }

    #[test]
    fn quote_never_crosses() {
        let book = BookRisk::default();
        let tick_schedule = TickSchedule::deribit_btc_option_default();
        let req = base_request(&book, &tick_schedule);
        let quote = build_quote(&req).expect("should quote with an empty book");
        assert!(quote.ask_price > quote.bid_price);
    }

    #[test]
    fn quote_prices_land_on_valid_ticks() {
        let book = BookRisk::default();
        let tick_schedule = TickSchedule::deribit_btc_option_default();
        let req = base_request(&book, &tick_schedule);
        let quote = build_quote(&req).unwrap();

        let bid_tick = tick_schedule.tick_for(quote.bid_price);
        let ask_tick = tick_schedule.tick_for(quote.ask_price);
        assert!((quote.bid_price / bid_tick).round() * bid_tick - quote.bid_price < 1e-9);
        assert!((quote.ask_price / ask_tick).round() * ask_tick - quote.ask_price < 1e-9);
    }

    #[test]
    fn heavy_long_vega_bucket_skews_quotes_lower_and_shrinks_size() {
        let tick_schedule = TickSchedule::deribit_btc_option_default();
        let empty_book = BookRisk::default();
        let empty_quote = build_quote(&base_request(&empty_book, &tick_schedule)).unwrap();

        // build a book that's heavily long vega in the 14d bucket by aggregating
        // a large short-option position (short options = negative vega... use a
        // long position instead to get positive/long vega exposure)
        let surface = flat_surface();
        let instrument = OptionKey { option_type: OptionType::Call, strike: 65000.0, expiry_years: 14.0 / 365.0 };
        let positions = vec![OptionPosition { instrument, size: 400.0 }];
        let forwards = book_risk::ForwardCurve::new(vec![(14.0 / 365.0, 65000.0)]);
        let loaded_book = aggregate(&positions, &surface, &forwards);

        let loaded_quote = build_quote(&base_request(&loaded_book, &tick_schedule)).unwrap();

        assert!(loaded_quote.bid_vol < empty_quote.bid_vol, "long vega book should quote a lower bid vol");
        assert!(loaded_quote.bid_size < empty_quote.bid_size, "long vega book should throttle size down");
    }

    #[test]
    fn size_below_min_trade_amount_skips_the_quote_entirely() {
        let book = BookRisk::default();
        let tick_schedule = TickSchedule::deribit_btc_option_default();
        let mut req = base_request(&book, &tick_schedule);
        req.min_trade_amount = 100.0; // above base_size, nothing should ever clear this
        assert!(build_quote(&req).is_none());
    }

    #[test]
    fn toxicity_widens_the_quoted_spread() {
        let book = BookRisk::default();
        let tick_schedule = TickSchedule::deribit_btc_option_default();
        let calm = build_quote(&base_request(&book, &tick_schedule)).unwrap();

        let mut toxic_req = base_request(&book, &tick_schedule);
        toxic_req.toxicity_score = 1.0;
        let toxic = build_quote(&toxic_req).unwrap();

        assert!(toxic.ask_price - toxic.bid_price > calm.ask_price - calm.bid_price);
    }

    #[test]
    fn nonsense_inputs_return_none_instead_of_garbage() {
        let book = BookRisk::default();
        let tick_schedule = TickSchedule::deribit_btc_option_default();
        let mut req = base_request(&book, &tick_schedule);
        req.mid_vol = 0.0;
        assert!(build_quote(&req).is_none());
    }
}
