// Every other test in this workspace exercises one crate at a time. This
// file is the only place that wires all five together, the same way a real
// caller would: build a surface, aggregate a book off it, quote an
// instrument against that book, size a delta hedge and evaluate a vega
// hedge off the same book, then attribute PnL for the positions that were
// in it. A type mismatch or a sign-convention drift between crates would
// slip past every unit test suite and still show up here.

use book_risk::{aggregate, price_coin, ForwardCurve, OptionKey, OptionPosition, OptionType};
use hedge_orchestrator::{
    dollar_delta, dollar_gamma, half_width_dollar_delta, rehedge_target, select_vega_hedge,
    BandParams, InversePerp, OptionFeeSchedule, VegaHedgeCandidate,
};
use pnl_explain::{attribute_book, MarketSnapshot, PositionMove};
use quoting_engine::{build_quote, QuoteRequest, RiskAversion, SpreadParams, TickSchedule};
use vol_surface::{RawSviParams, Slice, VolSurface};

fn fee_schedule() -> OptionFeeSchedule {
    OptionFeeSchedule {
        rate_of_underlying: 0.0004,
        cap_fraction_of_premium: 0.125,
    }
}

#[test]
fn full_pipeline_surface_to_quote_to_hedge_to_attribution() {
    // 1. surface: calibration itself is vol-surface's own job and already
    // tested there, build directly from known-good params here
    let params = RawSviParams {
        a: 0.03,
        b: 0.2,
        rho: -0.35,
        m: 0.0,
        sigma: 0.35,
    };
    let surface = VolSurface::build(vec![
        Slice {
            expiry_years: 7.0 / 365.0,
            params,
        },
        Slice {
            expiry_years: 30.0 / 365.0,
            params,
        },
    ])
    .unwrap();

    // 2. a small book: long calls, short puts, same expiry bucket
    let forward = 65000.0;
    let expiry = 14.0 / 365.0;
    let forwards = ForwardCurve::new(vec![(7.0 / 365.0, forward), (30.0 / 365.0, forward)]);

    let call_key = OptionKey {
        option_type: OptionType::Call,
        strike: 65000.0,
        expiry_years: expiry,
    };
    let put_key = OptionKey {
        option_type: OptionType::Put,
        strike: 58000.0,
        expiry_years: expiry,
    };
    let positions = vec![
        OptionPosition {
            instrument: call_key,
            size: 20.0,
        },
        OptionPosition {
            instrument: put_key,
            size: -10.0,
        },
    ];

    let book = aggregate(&positions, &surface, &forwards);
    assert!(
        book.total.delta != 0.0,
        "a directional mix of calls and puts should carry some delta"
    );
    assert!(book.total.vega != 0.0);

    // 3. quote a different strike in the same bucket, skewed off the book we just built
    let quote_strike = 66000.0;
    let mid_vol = surface.implied_vol((quote_strike / forward).ln(), expiry);
    let tick_schedule = TickSchedule::deribit_btc_option_default();
    let quote_req = QuoteRequest {
        option_type: OptionType::Call,
        strike: quote_strike,
        expiry_years: expiry,
        forward,
        mid_vol,
        book: &book,
        risk_aversion: RiskAversion {
            vega: 0.02,
            gamma: 0.01,
        },
        spread_params: SpreadParams {
            risk_aversion: 0.3,
            vol_of_vol: 1.2,
            horizon_years: 1.0 / 365.0,
            kappa: 8.0,
        },
        toxicity_score: 0.0,
        toxicity_widen_factor: 2.0,
        tick_schedule: &tick_schedule,
        base_size: 5.0,
        min_trade_amount: 0.1,
        max_bucket_vega: 1000.0,
        size_floor_fraction: 0.1,
        gamma_kernel_bandwidth: 0.1,
        max_price_deviation: 0.5,
    };
    let quote = build_quote(&quote_req)
        .expect("a reasonable book and reasonable params should always produce a quote");
    assert!(quote.ask_price > quote.bid_price);

    // 4. delta hedge: size a perp position against the book's coin delta,
    // then run the no-trade band decision on top of that
    let perp = InversePerp {
        contract_size: 10.0,
    };
    let hedge_contracts = perp.hedge_contracts_for_zero_delta(book.total.delta, forward);
    let residual_delta = book.total.delta + hedge_contracts * perp.coin_delta_per_contract(forward);
    assert!(
        residual_delta.abs() < 1e-9,
        "perp hedge should exactly cancel book delta at the current price"
    );

    let dollar_delta_now = dollar_delta(book.total.delta, forward);
    let dollar_gamma_now = dollar_gamma(book.total.gamma, forward);
    let band = BandParams {
        transaction_cost_rate: 0.0005,
        risk_aversion: 1e-8,
    };
    let half_width = half_width_dollar_delta(dollar_gamma_now, forward, &band);
    match rehedge_target(dollar_delta_now, 0.0, half_width) {
        None => {} // inside the band, a perfectly valid outcome
        Some(target) => assert!(
            target.abs() <= half_width + 1e-9,
            "a rehedge target should land at or inside the band edge"
        ),
    }

    // 5. vega hedge: same book vega feeding select_vega_hedge's candidate evaluation
    let candidate = VegaHedgeCandidate {
        option_type: OptionType::Call,
        strike: 65000.0,
        price_coin: price_coin(OptionType::Call, forward, 65000.0, mid_vol, expiry),
        vega_per_contract: 0.02,
        half_spread_vol: 0.02,
    };
    if let Some(selection) = select_vega_hedge(
        book.total.vega,
        1.2,
        7.0 / 365.0,
        0.05,
        1.0,
        &fee_schedule(),
        &[candidate],
    ) {
        assert!(
            selection.decision.should_hedge,
            "select_vega_hedge should only ever return a selection that clears its own bar"
        );
    }

    // 6. PnL attribution for the same two positions that built the book, after a market move
    let end_forward = forward + 500.0;
    let end_expiry = expiry - 1.0 / 365.0;

    let start_call = MarketSnapshot {
        forward,
        vol: surface.implied_vol((65000.0_f64 / forward).ln(), expiry),
        expiry_years: expiry,
    };
    let end_call = MarketSnapshot {
        forward: end_forward,
        vol: surface.implied_vol((65000.0_f64 / end_forward).ln(), end_expiry),
        expiry_years: end_expiry,
    };
    let start_put = MarketSnapshot {
        forward,
        vol: surface.implied_vol((58000.0_f64 / forward).ln(), expiry),
        expiry_years: expiry,
    };
    let end_put = MarketSnapshot {
        forward: end_forward,
        vol: surface.implied_vol((58000.0_f64 / end_forward).ln(), end_expiry),
        expiry_years: end_expiry,
    };

    let moves = vec![
        PositionMove {
            option_type: OptionType::Call,
            strike: 65000.0,
            size: 20.0,
            start: start_call,
            end: end_call,
        },
        PositionMove {
            option_type: OptionType::Put,
            strike: 58000.0,
            size: -10.0,
            start: start_put,
            end: end_put,
        },
    ];
    let attribution = attribute_book(&moves);
    assert!(
        (attribution.explained_pnl + attribution.unexplained_pnl - attribution.realized_pnl).abs()
            < 1e-9,
        "explained + unexplained must equal realized by construction"
    );
}

#[test]
fn empty_book_needs_no_hedge_anywhere_in_the_pipeline() {
    let params = RawSviParams {
        a: 0.03,
        b: 0.2,
        rho: -0.35,
        m: 0.0,
        sigma: 0.35,
    };
    let surface = VolSurface::build(vec![
        Slice {
            expiry_years: 7.0 / 365.0,
            params,
        },
        Slice {
            expiry_years: 30.0 / 365.0,
            params,
        },
    ])
    .unwrap();
    let forwards = ForwardCurve::new(vec![(7.0 / 365.0, 65000.0), (30.0 / 365.0, 65000.0)]);

    let book = aggregate(&[], &surface, &forwards);
    assert_eq!(book.total.delta, 0.0);
    assert_eq!(book.total.vega, 0.0);

    let perp = InversePerp {
        contract_size: 10.0,
    };
    assert_eq!(
        perp.hedge_contracts_for_zero_delta(book.total.delta, 65000.0),
        0.0
    );

    let band = BandParams {
        transaction_cost_rate: 0.0005,
        risk_aversion: 1e-8,
    };
    let half_width =
        half_width_dollar_delta(dollar_gamma(book.total.gamma, 65000.0), 65000.0, &band);
    assert_eq!(
        rehedge_target(dollar_delta(book.total.delta, 65000.0), 0.0, half_width),
        None
    );

    let candidate = VegaHedgeCandidate {
        option_type: OptionType::Call,
        strike: 65000.0,
        price_coin: 0.05,
        vega_per_contract: 0.02,
        half_spread_vol: 0.02,
    };
    assert!(select_vega_hedge(
        book.total.vega,
        1.2,
        7.0 / 365.0,
        0.05,
        1.0,
        &fee_schedule(),
        &[candidate]
    )
    .is_none());
}
