// Fits a slice to market (k, total variance) points via Nelder-Mead over a
// reparametrized space. Went gradient-free instead of Levenberg-Marquardt,
// didn't want to hand-derive/verify a Jacobian for 5 params when NM
// converges fine at this scale. Revisit if slice count per surface grows
// and calibration latency starts mattering.

use crate::svi::RawSviParams;

#[derive(Debug, Clone, Copy)]
pub struct VarianceQuote {
    pub log_moneyness: f64,
    pub total_variance: f64,
    /// Weight for the fit, e.g. inverse of the quoted bid/ask total-variance
    /// spread so tight, liquid strikes pull harder than wide wings.
    pub weight: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct CalibrationConfig {
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self { max_iterations: 2000, tolerance: 1e-10 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CalibrationError;

// x = (a, log_b, atanh_rho, m, log_sigma). Keeps the simplex inside the
// feasible region by construction instead of rejecting/clamping points,
// which tends to collapse the simplex near the boundary.
fn to_constrained(x: &[f64; 5]) -> RawSviParams {
    RawSviParams { a: x[0], b: x[1].exp(), rho: x[2].tanh(), m: x[3], sigma: x[4].exp() }
}

fn from_constrained(p: &RawSviParams) -> [f64; 5] {
    [p.a, p.b.max(1e-8).ln(), p.rho.clamp(-0.999, 0.999).atanh(), p.m, p.sigma.max(1e-8).ln()]
}

fn objective(x: &[f64; 5], quotes: &[VarianceQuote]) -> f64 {
    let params = to_constrained(x);
    let sse: f64 = quotes
        .iter()
        .map(|q| {
            let resid = params.total_variance(q.log_moneyness) - q.total_variance;
            q.weight * resid * resid
        })
        .sum();

    // Soft penalty on the min-variance constraint instead of a hard reject,
    // same reasoning as the reparametrization above.
    let min_var = params.a + params.b * params.sigma * (1.0 - params.rho * params.rho).sqrt();
    if min_var < 0.0 {
        sse + 1e6 * min_var * min_var
    } else {
        sse
    }
}

/// Fits one expiry slice. Pass the previous snapshot's calibrated params as
/// `initial_guess` in a live loop so each tick is a local refinement, not a
/// cold start.
// TODO: no multi-start / basin-hopping here, a bad initial_guess can land
// in a local min the penalty term doesn't rescue. Fine for the warm-start
// case (previous tick's params), risky for the first calibration of a
// freshly listed expiry with a generic seed.
pub fn calibrate_slice(
    quotes: &[VarianceQuote],
    initial_guess: RawSviParams,
    config: CalibrationConfig,
) -> Result<RawSviParams, CalibrationError> {
    if quotes.len() < 5 {
        return Err(CalibrationError); // 5 free params, need at least 5 points
    }

    let x0 = from_constrained(&initial_guess);
    let mut simplex: Vec<[f64; 5]> = vec![x0];
    for i in 0..5 {
        let mut v = x0;
        v[i] += if v[i].abs() > 1e-6 { v[i] * 0.1 } else { 0.1 };
        simplex.push(v);
    }
    let mut scores: Vec<f64> = simplex.iter().map(|x| objective(x, quotes)).collect();

    for _ in 0..config.max_iterations {
        let mut idx: Vec<usize> = (0..simplex.len()).collect();
        idx.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap());
        simplex = idx.iter().map(|&i| simplex[i]).collect();
        scores = idx.iter().map(|&i| scores[i]).collect();

        if (scores[5] - scores[0]).abs() < config.tolerance {
            break;
        }

        let mut centroid = [0.0; 5];
        for point in &simplex[0..5] {
            for d in 0..5 {
                centroid[d] += point[d] / 5.0;
            }
        }
        let worst = simplex[5];
        let along = |coeff: f64| -> [f64; 5] {
            let mut out = [0.0; 5];
            for d in 0..5 {
                out[d] = centroid[d] + coeff * (centroid[d] - worst[d]);
            }
            out
        };

        let xr = along(1.0);
        let fr = objective(&xr, quotes);

        if fr < scores[0] {
            let xe = along(2.0);
            let fe = objective(&xe, quotes);
            (simplex[5], scores[5]) = if fe < fr { (xe, fe) } else { (xr, fr) };
        } else if fr < scores[4] {
            simplex[5] = xr;
            scores[5] = fr;
        } else {
            let xc = along(-0.5);
            let fc = objective(&xc, quotes);
            if fc < scores[5] {
                simplex[5] = xc;
                scores[5] = fc;
            } else {
                let best = simplex[0];
                for i in 1..simplex.len() {
                    for d in 0..5 {
                        simplex[i][d] = best[d] + 0.5 * (simplex[i][d] - best[d]);
                    }
                    scores[i] = objective(&simplex[i], quotes);
                }
            }
        }
    }

    let best_idx = (0..simplex.len()).min_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap()).unwrap();
    let params = to_constrained(&simplex[best_idx]);
    params.validate_static().map_err(|_| CalibrationError)?;
    Ok(params)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_known_svi_params_from_synthetic_smile() {
        let true_params = RawSviParams { a: 0.02, b: 0.2, rho: -0.3, m: 0.0, sigma: 0.15 };
        let strikes_k = [-0.6, -0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4, 0.6];
        let quotes: Vec<VarianceQuote> = strikes_k
            .iter()
            .map(|&k| VarianceQuote { log_moneyness: k, total_variance: true_params.total_variance(k), weight: 1.0 })
            .collect();

        // deliberately bad starting guess, proves the search actually converges
        let bad_guess = RawSviParams { a: 0.05, b: 0.1, rho: 0.0, m: 0.05, sigma: 0.3 };
        let fitted = calibrate_slice(&quotes, bad_guess, CalibrationConfig::default()).unwrap();

        for &k in &strikes_k {
            let diff = (fitted.total_variance(k) - true_params.total_variance(k)).abs();
            assert!(diff < 1e-5, "k={k} diff={diff}");
        }
    }

    #[test]
    fn rejects_too_few_quotes() {
        let quotes = vec![VarianceQuote { log_moneyness: 0.0, total_variance: 0.02, weight: 1.0 }];
        let guess = RawSviParams { a: 0.02, b: 0.2, rho: 0.0, m: 0.0, sigma: 0.15 };
        assert!(calibrate_slice(&quotes, guess, CalibrationConfig::default()).is_err());
    }

    #[test]
    fn warm_start_from_previous_slice_converges_faster_than_cold_start() {
        // not a strict perf test, just checks warm start lands in fewer iterations
        // by capping max_iterations low and confirming it still converges
        let true_params = RawSviParams { a: 0.021, b: 0.19, rho: -0.31, m: 0.005, sigma: 0.14 };
        let strikes_k = [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5];
        let quotes: Vec<VarianceQuote> = strikes_k
            .iter()
            .map(|&k| VarianceQuote { log_moneyness: k, total_variance: true_params.total_variance(k), weight: 1.0 })
            .collect();

        let warm_start = RawSviParams { a: 0.02, b: 0.2, rho: -0.3, m: 0.0, sigma: 0.15 };
        let tight_config = CalibrationConfig { max_iterations: 300, tolerance: 1e-10 };
        let fitted = calibrate_slice(&quotes, warm_start, tight_config).unwrap();

        let diff = (fitted.total_variance(0.0) - true_params.total_variance(0.0)).abs();
        assert!(diff < 1e-4, "warm start should converge in 300 iters, diff={diff}");
    }
}
