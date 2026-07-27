// Deribit inverse options: 1 coin per contract, so `size` is already in coins.

use crate::inverse_option::{greeks, InverseGreeks, OptionType};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OptionKey {
    pub option_type: OptionType,
    pub strike: f64,
    pub expiry_years: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct OptionPosition {
    pub instrument: OptionKey,
    pub size: f64,
}

impl OptionPosition {
    pub fn greeks(&self, forward: f64, vol: f64) -> InverseGreeks {
        let g = greeks(self.instrument.option_type, forward, self.instrument.strike, vol, self.instrument.expiry_years);
        InverseGreeks {
            price_coin: g.price_coin * self.size,
            delta: g.delta * self.size,
            gamma: g.gamma * self.size,
            vega: g.vega * self.size,
            theta: g.theta * self.size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_position_flips_the_sign_of_long_greeks() {
        let instrument = OptionKey { option_type: OptionType::Call, strike: 65000.0, expiry_years: 14.0 / 365.0 };
        let long = OptionPosition { instrument, size: 1.0 };
        let short = OptionPosition { instrument, size: -1.0 };

        let (forward, vol) = (65000.0, 0.6);
        let long_g = long.greeks(forward, vol);
        let short_g = short.greeks(forward, vol);

        assert!((long_g.delta + short_g.delta).abs() < 1e-15);
        assert!((long_g.gamma + short_g.gamma).abs() < 1e-15);
    }

    #[test]
    fn double_size_doubles_greeks() {
        let instrument = OptionKey { option_type: OptionType::Put, strike: 60000.0, expiry_years: 30.0 / 365.0 };
        let one = OptionPosition { instrument, size: 1.0 };
        let two = OptionPosition { instrument, size: 2.0 };

        let (forward, vol) = (65000.0, 0.65);
        let g1 = one.greeks(forward, vol);
        let g2 = two.greeks(forward, vol);

        assert!((g2.delta - 2.0 * g1.delta).abs() < 1e-15);
        assert!((g2.vega - 2.0 * g1.vega).abs() < 1e-15);
    }
}
