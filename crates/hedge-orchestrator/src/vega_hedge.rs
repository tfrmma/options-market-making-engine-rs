// Deribit caps option fees as min(rate * underlying notional, cap_fraction
// * option price), confirmed in their own Fees support article.
//
// TODO: rate and cap fraction aren't hardcoded, public sources disagree
// with each other (rate 0.03-0.04%, cap 12.5-20%, likely stale articles
// plus fee-tier differences), a desk running this should read its actual
// fee tier off the account API instead. Mechanism modeled, numbers are config.

#[derive(Debug, Clone, Copy)]
pub struct OptionFeeSchedule {
    pub rate_of_underlying: f64,
    pub cap_fraction_of_premium: f64,
}

impl OptionFeeSchedule {
    pub fn fee_coin(&self, contract_size_coin: f64, option_price_coin: f64) -> f64 {
        let uncapped = self.rate_of_underlying * contract_size_coin;
        uncapped.min(self.cap_fraction_of_premium * option_price_coin)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VegaHedgeDecision {
    pub carry_cost_coin: f64,
    pub hedge_cost_coin: f64,
    pub should_hedge: bool,
}

// same mean-variance shape as the AS inventory term in quoting-engine::spread,
// applied to a standing position instead of a quote width
fn carry_cost(vega_exposure_coin: f64, vol_of_vol: f64, horizon_years: f64, risk_aversion: f64) -> f64 {
    risk_aversion * (vega_exposure_coin * vol_of_vol).powi(2) * horizon_years
}

fn hedge_cost(
    hedge_size_contracts: f64,
    hedge_option_vega_per_contract: f64,
    half_spread_vol: f64,
    hedge_option_price_coin: f64,
    contract_size_coin: f64,
    fees: &OptionFeeSchedule,
) -> f64 {
    let spread_cost = hedge_size_contracts.abs() * hedge_option_vega_per_contract.abs() * half_spread_vol;
    let fee_cost = hedge_size_contracts.abs() * fees.fee_coin(contract_size_coin, hedge_option_price_coin);
    spread_cost + fee_cost
}

// TODO: this is a hedge-or-don't threshold, not strike/tenor selection,
// caller decides which option to hedge with.
pub fn evaluate(
    vega_exposure_coin: f64,
    vol_of_vol: f64,
    horizon_years: f64,
    risk_aversion: f64,
    hedge_size_contracts: f64,
    hedge_option_vega_per_contract: f64,
    half_spread_vol: f64,
    hedge_option_price_coin: f64,
    contract_size_coin: f64,
    fees: &OptionFeeSchedule,
) -> VegaHedgeDecision {
    let carry = carry_cost(vega_exposure_coin, vol_of_vol, horizon_years, risk_aversion);
    let hedge = hedge_cost(
        hedge_size_contracts,
        hedge_option_vega_per_contract,
        half_spread_vol,
        hedge_option_price_coin,
        contract_size_coin,
        fees,
    );
    VegaHedgeDecision { carry_cost_coin: carry, hedge_cost_coin: hedge, should_hedge: carry > hedge }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fees() -> OptionFeeSchedule {
        OptionFeeSchedule { rate_of_underlying: 0.0004, cap_fraction_of_premium: 0.125 }
    }

    #[test]
    fn fee_is_capped_for_a_cheap_deep_otm_option() {
        // uncapped would be 0.0004 BTC, but the option is only worth 0.0001 BTC,
        // so the cap (12.5% of premium) should bind instead
        let fee = fees().fee_coin(1.0, 0.0001);
        assert!((fee - 0.0000125).abs() < 1e-12, "fee={fee}");
    }

    #[test]
    fn fee_is_uncapped_for_a_normally_priced_option() {
        let fee = fees().fee_coin(1.0, 0.05);
        assert!((fee - 0.0004).abs() < 1e-12, "fee={fee}");
    }

    #[test]
    fn large_naked_vega_over_a_long_horizon_is_worth_hedging() {
        let decision = evaluate(
            50.0,  // large vega exposure
            1.2,   // vol-of-vol
            30.0 / 365.0, // sitting on it for a month
            0.05,  // risk aversion
            5.0,   // small hedge trade
            0.02,  // hedge instrument vega/contract
            0.05,  // half spread vol
            0.05,  // hedge option price
            1.0,
            &fees(),
        );
        assert!(decision.should_hedge, "carry={} hedge={}", decision.carry_cost_coin, decision.hedge_cost_coin);
    }

    #[test]
    fn tiny_vega_over_a_short_horizon_is_not_worth_hedging() {
        let decision = evaluate(
            0.5, 1.2, 1.0 / 365.0, 0.05, 5.0, 0.02, 0.05, 0.05, 1.0, &fees(),
        );
        assert!(!decision.should_hedge, "carry={} hedge={}", decision.carry_cost_coin, decision.hedge_cost_coin);
    }

    #[test]
    fn zero_vega_exposure_never_clears_the_hedge_cost() {
        let decision = evaluate(0.0, 1.2, 30.0 / 365.0, 0.05, 5.0, 0.02, 0.05, 0.05, 1.0, &fees());
        assert!(!decision.should_hedge);
        assert_eq!(decision.carry_cost_coin, 0.0);
    }
}
