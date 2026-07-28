// Real estimators for SpreadParams::vol_of_vol and SpreadParams::kappa.
// This repo has no market-data pipeline of its own (that's feedhandler-core-rs's
// job), so these take a plain observation set as an argument instead of reaching
// out to fetch anything, the calibration logic lives here, real data feeds in from upstream.

/// Annualized vol-of-vol from a time series of ATM IV samples taken at even
/// intervals. Standard realized-vol estimator: stdev of log returns,
/// annualized by sqrt(samples per year).
pub fn estimate_vol_of_vol(iv_samples: &[f64], samples_per_year: f64) -> Option<f64> {
    if iv_samples.len() < 2 || iv_samples.iter().any(|&v| v <= 0.0) {
        return None;
    }
    let log_returns: Vec<f64> = iv_samples.windows(2).map(|w| (w[1] / w[0]).ln()).collect();
    let n = log_returns.len() as f64;
    let mean = log_returns.iter().sum::<f64>() / n;
    let variance = log_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    Some(variance.sqrt() * samples_per_year.sqrt())
}

#[derive(Debug, Clone, Copy)]
pub struct FillObservation {
    pub distance_from_mid_vol: f64,
    /// fills / quotes at this distance, must be > 0 to contribute (a zero
    /// fill rate has no finite log and gets dropped rather than blowing up the fit)
    pub fill_rate: f64,
}

/// Fits kappa from lambda(delta) = A * exp(-kappa * delta) via log-linear
/// least squares: ln(fill_rate) = ln(A) - kappa * delta. Needs at least two
/// distinct distances with a positive fill rate, and a genuinely decaying
/// relationship, a kappa <= 0 would mean fill rate increasing with distance
/// from mid, which isn't the regime this formula is valid in.
pub fn estimate_kappa(observations: &[FillObservation]) -> Option<f64> {
    let points: Vec<(f64, f64)> =
        observations.iter().filter(|o| o.fill_rate > 0.0).map(|o| (o.distance_from_mid_vol, o.fill_rate.ln())).collect();
    if points.len() < 2 {
        return None;
    }

    let n = points.len() as f64;
    let sum_x: f64 = points.iter().map(|(x, _)| x).sum();
    let sum_y: f64 = points.iter().map(|(_, y)| y).sum();
    let sum_xy: f64 = points.iter().map(|(x, y)| x * y).sum();
    let sum_xx: f64 = points.iter().map(|(x, _)| x * x).sum();
    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-12 {
        return None; // all observations at the same distance, nothing to fit a slope to
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let kappa = -slope;
    if kappa > 0.0 {
        Some(kappa)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vol_of_vol_matches_hand_computed_value_on_a_known_series() {
        // log returns exactly [0.01, -0.01, 0.01, -0.01] by construction
        let iv = [0.5, 0.5 * 0.01_f64.exp(), 0.5, 0.5 * 0.01_f64.exp(), 0.5];
        let result = estimate_vol_of_vol(&iv, 1.0).unwrap();
        let expected = (0.0004_f64 / 3.0).sqrt(); // variance = sum((r-0)^2)/(n-1) = 4*0.0001/3
        assert!((result - expected).abs() < 1e-9, "result={result} expected={expected}");
    }

    #[test]
    fn vol_of_vol_scales_with_sqrt_samples_per_year() {
        let iv = [0.5, 0.5 * 1.02, 0.5 * 0.99, 0.5 * 1.01];
        let daily = estimate_vol_of_vol(&iv, 365.0).unwrap();
        let weekly = estimate_vol_of_vol(&iv, 52.0).unwrap();
        assert!(daily > weekly, "more samples per year should annualize to a bigger number");
    }

    #[test]
    fn vol_of_vol_rejects_too_few_or_invalid_samples() {
        assert!(estimate_vol_of_vol(&[0.5], 365.0).is_none());
        assert!(estimate_vol_of_vol(&[0.5, -0.1], 365.0).is_none());
    }

    #[test]
    fn kappa_recovers_the_true_decay_from_synthetic_fill_data() {
        let true_kappa = 6.0;
        let a = 0.4;
        let observations: Vec<FillObservation> = [0.0, 0.02, 0.05, 0.08, 0.12, 0.2]
            .iter()
            .map(|&d| FillObservation { distance_from_mid_vol: d, fill_rate: a * (-true_kappa * d).exp() })
            .collect();

        let fitted = estimate_kappa(&observations).unwrap();
        assert!((fitted - true_kappa).abs() < 1e-6, "fitted={fitted}");
    }

    #[test]
    fn kappa_rejects_a_non_decaying_relationship() {
        // fill rate increasing with distance, not the regime this formula models
        let observations =
            vec![FillObservation { distance_from_mid_vol: 0.0, fill_rate: 0.1 }, FillObservation { distance_from_mid_vol: 0.1, fill_rate: 0.5 }];
        assert!(estimate_kappa(&observations).is_none());
    }

    #[test]
    fn kappa_needs_at_least_two_usable_points() {
        let observations = vec![FillObservation { distance_from_mid_vol: 0.0, fill_rate: 0.1 }];
        assert!(estimate_kappa(&observations).is_none());
        let with_a_zero = vec![
            FillObservation { distance_from_mid_vol: 0.0, fill_rate: 0.1 },
            FillObservation { distance_from_mid_vol: 0.1, fill_rate: 0.0 }, // dropped, ln(0) is undefined
        ];
        assert!(estimate_kappa(&with_a_zero).is_none());
    }
}
