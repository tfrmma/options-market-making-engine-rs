// Deribit's own inverse futures/perpetual PnL formula (see Inverse Futures
// support article): for N contracts of size `contract_size` USD each,
// entered at F_entry:
//
//   PnL_coin(F) = N * contract_size * (1/F_entry - 1/F)
//
// Differentiating at the current price (i.e. treating "now" as the entry
// point, since a Greek is about marginal risk from here, not sunk cost
// basis) gives the instantaneous coin-delta contributed by one contract:
//
//   d(PnL_coin)/dF = N * contract_size / F^2
//
// so one contract's coin-delta at price F is contract_size / F^2. This
// isn't an approximation, it falls straight out of the documented formula,
// cross-checked against finite differences below the same way
// book-risk::inverse_option was.

#[derive(Debug, Clone, Copy)]
pub struct InversePerp {
    /// USD notional per contract, 10 for BTC-PERPETUAL, 1 for ETH-PERPETUAL
    /// per Deribit's contract specs. Pass it in rather than assuming, this
    /// has changed before.
    pub contract_size: f64,
}

impl InversePerp {
    pub fn coin_delta_per_contract(&self, forward: f64) -> f64 {
        self.contract_size / (forward * forward)
    }

    /// Contracts needed to bring the book's coin-delta to exactly zero at
    /// the current price. Sign follows the usual convention: positive book
    /// delta (net long) needs a negative (short) perp position to offset.
    pub fn hedge_contracts_for_zero_delta(&self, book_delta_coin: f64, forward: f64) -> f64 {
        -book_delta_coin * forward * forward / self.contract_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fd_coin_delta_per_contract(contract_size: f64, forward: f64) -> f64 {
        // PnL_coin(F) = contract_size * (1/entry - 1/F), entry = forward (today's price)
        let eps = forward * 1e-6;
        let pnl_up = contract_size * (1.0 / forward - 1.0 / (forward + eps));
        let pnl_down = contract_size * (1.0 / forward - 1.0 / (forward - eps));
        (pnl_up - pnl_down) / (2.0 * eps)
    }

    #[test]
    fn coin_delta_per_contract_matches_finite_difference_of_deribits_own_pnl_formula() {
        let perp = InversePerp {
            contract_size: 10.0,
        };
        let forward = 65000.0;
        let analytic = perp.coin_delta_per_contract(forward);
        let fd = fd_coin_delta_per_contract(perp.contract_size, forward);
        assert!(
            (analytic - fd).abs() / fd < 1e-6,
            "analytic={analytic} fd={fd}"
        );
    }

    #[test]
    fn hedging_a_long_book_requires_a_short_perp_position() {
        let perp = InversePerp {
            contract_size: 10.0,
        };
        let n = perp.hedge_contracts_for_zero_delta(0.5, 65000.0);
        assert!(
            n < 0.0,
            "long book delta should need a short hedge, got {n}"
        );
    }

    #[test]
    fn hedge_size_exactly_cancels_book_delta_at_current_price() {
        let perp = InversePerp {
            contract_size: 10.0,
        };
        let forward = 65000.0;
        let book_delta = 0.35;
        let n = perp.hedge_contracts_for_zero_delta(book_delta, forward);
        let hedge_delta = n * perp.coin_delta_per_contract(forward);
        assert!(
            (book_delta + hedge_delta).abs() < 1e-9,
            "book + hedge delta should net to zero"
        );
    }

    #[test]
    fn zero_book_delta_needs_no_hedge() {
        let perp = InversePerp {
            contract_size: 10.0,
        };
        assert_eq!(perp.hedge_contracts_for_zero_delta(0.0, 65000.0), 0.0);
    }
}
