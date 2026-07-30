// Deribit caps option fees as min(rate * underlying notional, cap_fraction
// * option price), confirmed in their own Fees support article.
//
// TODO: rate and cap fraction aren't hardcoded, public sources disagree
// with each other (rate 0.03-0.04%, cap 12.5-20%, likely stale articles
// plus fee-tier differences). This is correctly caller-supplied config, not
// a code gap, the actual gap is a production-wiring one: something needs to
// call Deribit's authenticated account API periodically and refresh this,
// which needs an HTTP client, request signing, and credential handling this
// pure-computation crate deliberately doesn't pull in.

use book_risk::OptionType;

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
#[derive(Debug, Clone, Copy)]
pub struct VegaCarryParams {
    pub vega_exposure_coin: f64,
    pub vol_of_vol: f64,
    pub horizon_years: f64,
    pub risk_aversion: f64,
}

fn carry_cost(p: &VegaCarryParams) -> f64 {
    p.risk_aversion * (p.vega_exposure_coin * p.vol_of_vol).powi(2) * p.horizon_years
}

fn hedge_cost(hedge_size_contracts: f64, candidate: &VegaHedgeCandidate, contract_size_coin: f64, fees: &OptionFeeSchedule) -> f64 {
    let spread_cost = hedge_size_contracts.abs() * candidate.vega_per_contract.abs() * candidate.half_spread_vol;
    let fee_cost = hedge_size_contracts.abs() * fees.fee_coin(contract_size_coin, candidate.price_coin);
    spread_cost + fee_cost
}

/// One option this crate could hedge naked vega with. Caller supplies the
/// candidate list (from their own instrument/order-book snapshot), this
/// crate doesn't have a route to Deribit's instrument list itself. Scoped
/// to same-bucket candidates for now, picking a different-expiry option
/// introduces calendar/basis risk that isn't modeled here.
#[derive(Debug, Clone, Copy)]
pub struct VegaHedgeCandidate {
    pub option_type: OptionType,
    pub strike: f64,
    pub price_coin: f64,
    pub vega_per_contract: f64,
    pub half_spread_vol: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct VegaHedgeSelection {
    pub candidate_index: usize,
    pub hedge_size_contracts: f64,
    pub decision: VegaHedgeDecision,
}

/// Sizes each candidate to exactly zero out vega_exposure_coin, evaluates
/// the cost-benefit for each, and picks whichever clears the bar at the
/// lowest hedge cost per unit of vega neutralized. None if nothing clears
/// it, or every candidate has ~zero vega and can't hedge anything.
pub fn select_vega_hedge(
    vega_exposure_coin: f64,
    vol_of_vol: f64,
    horizon_years: f64,
    risk_aversion: f64,
    contract_size_coin: f64,
    fees: &OptionFeeSchedule,
    candidates: &[VegaHedgeCandidate],
) -> Option<VegaHedgeSelection> {
    let carry = VegaCarryParams { vega_exposure_coin, vol_of_vol, horizon_years, risk_aversion };
    let mut best: Option<(VegaHedgeSelection, f64)> = None;

    for (i, c) in candidates.iter().enumerate() {
        if c.vega_per_contract.abs() < 1e-12 {
            continue;
        }
        let hedge_size_contracts = -vega_exposure_coin / c.vega_per_contract;
        let decision = evaluate(&carry, hedge_size_contracts, c, contract_size_coin, fees);
        if !decision.should_hedge {
            continue;
        }
        let cost_per_vega = decision.hedge_cost_coin / vega_exposure_coin.abs().max(1e-12);
        let selection = VegaHedgeSelection { candidate_index: i, hedge_size_contracts, decision };
        if best.as_ref().map_or(true, |(_, c)| cost_per_vega < *c) {
            best = Some((selection, cost_per_vega));
        }
    }

    best.map(|(selection, _)| selection)
}

pub fn evaluate(
    carry: &VegaCarryParams,
    hedge_size_contracts: f64,
    candidate: &VegaHedgeCandidate,
    contract_size_coin: f64,
    fees: &OptionFeeSchedule,
) -> VegaHedgeDecision {
    let carry_cost_coin = carry_cost(carry);
    let hedge_cost_coin = hedge_cost(hedge_size_contracts, candidate, contract_size_coin, fees);
    VegaHedgeDecision { carry_cost_coin, hedge_cost_coin, should_hedge: carry_cost_coin > hedge_cost_coin }
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
        let carry = VegaCarryParams { vega_exposure_coin: 50.0, vol_of_vol: 1.2, horizon_years: 30.0 / 365.0, risk_aversion: 0.05 };
        let candidate =
            VegaHedgeCandidate { option_type: OptionType::Call, strike: 65000.0, price_coin: 0.05, vega_per_contract: 0.02, half_spread_vol: 0.05 };
        let decision = evaluate(&carry, 5.0, &candidate, 1.0, &fees());
        assert!(decision.should_hedge, "carry={} hedge={}", decision.carry_cost_coin, decision.hedge_cost_coin);
    }

    #[test]
    fn tiny_vega_over_a_short_horizon_is_not_worth_hedging() {
        let carry = VegaCarryParams { vega_exposure_coin: 0.5, vol_of_vol: 1.2, horizon_years: 1.0 / 365.0, risk_aversion: 0.05 };
        let candidate =
            VegaHedgeCandidate { option_type: OptionType::Call, strike: 65000.0, price_coin: 0.05, vega_per_contract: 0.02, half_spread_vol: 0.05 };
        let decision = evaluate(&carry, 5.0, &candidate, 1.0, &fees());
        assert!(!decision.should_hedge, "carry={} hedge={}", decision.carry_cost_coin, decision.hedge_cost_coin);
    }

    #[test]
    fn zero_vega_exposure_never_clears_the_hedge_cost() {
        let carry = VegaCarryParams { vega_exposure_coin: 0.0, vol_of_vol: 1.2, horizon_years: 30.0 / 365.0, risk_aversion: 0.05 };
        let candidate =
            VegaHedgeCandidate { option_type: OptionType::Call, strike: 65000.0, price_coin: 0.05, vega_per_contract: 0.02, half_spread_vol: 0.05 };
        let decision = evaluate(&carry, 5.0, &candidate, 1.0, &fees());
        assert!(!decision.should_hedge);
        assert_eq!(decision.carry_cost_coin, 0.0);
    }

    #[test]
    fn selects_the_cheaper_of_two_candidates_that_both_clear_the_bar() {
        let cheap = VegaHedgeCandidate {
            option_type: OptionType::Call,
            strike: 65000.0,
            price_coin: 0.05,
            vega_per_contract: 0.02,
            half_spread_vol: 0.01, // tight spread, cheap to cross
        };
        let expensive = VegaHedgeCandidate {
            option_type: OptionType::Call,
            strike: 90000.0,
            price_coin: 0.01,
            vega_per_contract: 0.005, // needs way more contracts for the same vega
            half_spread_vol: 0.08,    // wide spread, expensive to cross
        };

        let selection = select_vega_hedge(50.0, 1.2, 30.0 / 365.0, 0.05, 1.0, &fees(), &[expensive, cheap]).unwrap();
        assert_eq!(selection.candidate_index, 1, "should pick the cheap candidate at index 1, not the expensive one at index 0");
    }

    #[test]
    fn hedge_size_sign_offsets_the_vega_exposure() {
        let candidate =
            VegaHedgeCandidate { option_type: OptionType::Call, strike: 65000.0, price_coin: 0.05, vega_per_contract: 0.02, half_spread_vol: 0.01 };
        let selection = select_vega_hedge(50.0, 1.2, 30.0 / 365.0, 0.05, 1.0, &fees(), &[candidate]).unwrap();
        let residual = 50.0 + selection.hedge_size_contracts * candidate.vega_per_contract;
        assert!(residual.abs() < 1e-9, "residual={residual}");
    }

    #[test]
    fn skips_candidates_with_negligible_vega() {
        let dead = VegaHedgeCandidate { option_type: OptionType::Call, strike: 65000.0, price_coin: 0.05, vega_per_contract: 0.0, half_spread_vol: 0.01 };
        assert!(select_vega_hedge(50.0, 1.2, 30.0 / 365.0, 0.05, 1.0, &fees(), &[dead]).is_none());
    }

    #[test]
    fn returns_none_when_no_candidate_clears_the_bar() {
        let tiny_exposure_but_wide_spread = VegaHedgeCandidate {
            option_type: OptionType::Call,
            strike: 65000.0,
            price_coin: 0.05,
            vega_per_contract: 0.02,
            half_spread_vol: 0.5, // absurdly wide, not worth crossing for this little exposure
        };
        let selection = select_vega_hedge(0.3, 1.2, 1.0 / 365.0, 0.05, 1.0, &fees(), &[tiny_exposure_but_wide_spread]);
        assert!(selection.is_none());
    }
}
