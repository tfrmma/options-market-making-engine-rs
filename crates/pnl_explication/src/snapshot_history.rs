// In-memory history of MarketSnapshots per instrument, so attribution can
// pull "state as of two points in time" instead of the caller wiring that
// up separately every time. Keyed by Deribit's own instrument name
// convention (e.g. "BTC-29AUG25-65000-C"), not by (option_type, strike,
// expiry_years): expiry_years inside MarketSnapshot decays with every new
// snapshot of the same instrument, it's not a stable identity, the
// instrument name is.
//
// No persistence, no eviction beyond retain_since, this is a buffer, not a
// database, a long-running process needs to call retain_since periodically
// or this grows forever.

use std::collections::BTreeMap;

use book_risk::OptionType;

use crate::attribution::{attribute_position, Attribution, MarketSnapshot};

#[derive(Debug, Default)]
pub struct SnapshotHistory {
    by_instrument: BTreeMap<String, BTreeMap<i64, MarketSnapshot>>,
}

impl SnapshotHistory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, instrument_name: &str, timestamp: i64, snapshot: MarketSnapshot) {
        self.by_instrument.entry(instrument_name.to_string()).or_default().insert(timestamp, snapshot);
    }

    /// Exact snapshot at this timestamp, None if nothing was recorded there.
    pub fn at(&self, instrument_name: &str, timestamp: i64) -> Option<MarketSnapshot> {
        self.by_instrument.get(instrument_name)?.get(&timestamp).copied()
    }

    /// Most recent snapshot at or before `timestamp`, for "what was the
    /// state as of this time" when you don't have an exact match.
    pub fn latest_at_or_before(&self, instrument_name: &str, timestamp: i64) -> Option<MarketSnapshot> {
        self.by_instrument.get(instrument_name)?.range(..=timestamp).next_back().map(|(_, s)| *s)
    }

    /// All (timestamp, snapshot) pairs for an instrument, sorted ascending.
    pub fn history_for(&self, instrument_name: &str) -> Vec<(i64, MarketSnapshot)> {
        self.by_instrument.get(instrument_name).map(|m| m.iter().map(|(&t, &s)| (t, s)).collect()).unwrap_or_default()
    }

    /// Drops every entry older than `cutoff_timestamp`, across all
    /// instruments, and drops instruments left with no entries at all.
    pub fn retain_since(&mut self, cutoff_timestamp: i64) {
        for history in self.by_instrument.values_mut() {
            history.retain(|&ts, _| ts >= cutoff_timestamp);
        }
        self.by_instrument.retain(|_, history| !history.is_empty());
    }
}

/// Pulls the exact start/end snapshots for an instrument out of the history
/// and runs attribute_position on them. None if either timestamp wasn't recorded.
pub fn attribute_from_history(
    history: &SnapshotHistory,
    instrument_name: &str,
    option_type: OptionType,
    strike: f64,
    size: f64,
    start_timestamp: i64,
    end_timestamp: i64,
) -> Option<Attribution> {
    let start = history.at(instrument_name, start_timestamp)?;
    let end = history.at(instrument_name, end_timestamp)?;
    Some(attribute_position(option_type, strike, start, end, size))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(forward: f64) -> MarketSnapshot {
        MarketSnapshot { forward, vol: 0.6, expiry_years: 30.0 / 365.0 }
    }

    #[test]
    fn records_and_retrieves_an_exact_snapshot() {
        let mut h = SnapshotHistory::new();
        h.record("BTC-29AUG25-65000-C", 1000, snap(65000.0));
        let s = h.at("BTC-29AUG25-65000-C", 1000).unwrap();
        assert_eq!(s.forward, 65000.0);
    }

    #[test]
    fn missing_timestamp_returns_none() {
        let h = SnapshotHistory::new();
        assert!(h.at("BTC-29AUG25-65000-C", 1000).is_none());
    }

    #[test]
    fn latest_at_or_before_finds_the_nearest_prior_entry() {
        let mut h = SnapshotHistory::new();
        h.record("BTC-29AUG25-65000-C", 1000, snap(65000.0));
        h.record("BTC-29AUG25-65000-C", 2000, snap(66000.0));
        h.record("BTC-29AUG25-65000-C", 3000, snap(67000.0));

        let s = h.latest_at_or_before("BTC-29AUG25-65000-C", 2500).unwrap();
        assert_eq!(s.forward, 66000.0); // nearest prior, not nearest overall
    }

    #[test]
    fn latest_at_or_before_returns_none_when_nothing_recorded_that_early() {
        let mut h = SnapshotHistory::new();
        h.record("BTC-29AUG25-65000-C", 1000, snap(65000.0));
        assert!(h.latest_at_or_before("BTC-29AUG25-65000-C", 500).is_none());
    }

    #[test]
    fn different_instruments_do_not_leak_into_each_others_history() {
        let mut h = SnapshotHistory::new();
        h.record("BTC-29AUG25-65000-C", 1000, snap(65000.0));
        h.record("BTC-29AUG25-70000-C", 1000, snap(65000.0));
        assert!(h.at("BTC-29AUG25-65000-C", 2000).is_none());
        assert_eq!(h.history_for("BTC-29AUG25-70000-C").len(), 1);
    }

    #[test]
    fn attribute_from_history_matches_calling_attribute_position_directly() {
        let mut h = SnapshotHistory::new();
        let start = snap(65000.0);
        let end = snap(67000.0);
        h.record("BTC-29AUG25-65000-C", 1000, start);
        h.record("BTC-29AUG25-65000-C", 2000, end);

        let via_history = attribute_from_history(&h, "BTC-29AUG25-65000-C", OptionType::Call, 65000.0, 2.0, 1000, 2000).unwrap();
        let direct = attribute_position(OptionType::Call, 65000.0, start, end, 2.0);
        assert!((via_history.realized_pnl - direct.realized_pnl).abs() < 1e-12);
    }

    #[test]
    fn attribute_from_history_is_none_when_a_timestamp_is_missing() {
        let mut h = SnapshotHistory::new();
        h.record("BTC-29AUG25-65000-C", 1000, snap(65000.0));
        // no entry at 2000
        assert!(attribute_from_history(&h, "BTC-29AUG25-65000-C", OptionType::Call, 65000.0, 1.0, 1000, 2000).is_none());
    }

    #[test]
    fn retain_since_drops_old_entries_and_keeps_recent_ones() {
        let mut h = SnapshotHistory::new();
        h.record("BTC-29AUG25-65000-C", 1000, snap(65000.0));
        h.record("BTC-29AUG25-65000-C", 2000, snap(66000.0));
        h.record("BTC-29AUG25-65000-C", 3000, snap(67000.0));

        h.retain_since(2000);
        assert!(h.at("BTC-29AUG25-65000-C", 1000).is_none());
        assert!(h.at("BTC-29AUG25-65000-C", 2000).is_some());
        assert!(h.at("BTC-29AUG25-65000-C", 3000).is_some());
    }

    #[test]
    fn retain_since_drops_instruments_left_with_nothing() {
        let mut h = SnapshotHistory::new();
        h.record("BTC-29AUG25-65000-C", 1000, snap(65000.0));
        h.retain_since(5000); // past every recorded timestamp
        assert_eq!(h.history_for("BTC-29AUG25-65000-C").len(), 0);
    }
}
