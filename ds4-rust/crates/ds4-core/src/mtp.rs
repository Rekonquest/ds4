// DS4 (DwarfStar) — multi-token prediction (MTP).
//
// Mirrors the public surface of `ds4_mtp_*` in `ds4.c`. The full
// speculative-decoding implementation depends on the live
// transformer weights and so lives behind the engine. For v1 we ship
// the configuration struct + an empty draft-state container.

/// MTP configuration — draft tokens and rejection margin.
#[derive(Debug, Clone)]
pub struct Ds4MtpConfig {
    pub draft_tokens: usize,
    pub margin: f32,
}

impl Default for Ds4MtpConfig {
    fn default() -> Self {
        Ds4MtpConfig {
            draft_tokens: 0,
            margin: 0.0,
        }
    }
}

/// Speculative-decoding state. Holds the most recent K draft tokens
/// and the live acceptance counter.
#[derive(Debug, Clone)]
pub struct Ds4Mtp {
    /// Disambiguator so we don't accidentally clash with the
    /// public-only shadow module.
    _priv: (),
    config: Ds4MtpConfig,
    drafts: Vec<u32>,
    accepted_run: usize,
}

impl Ds4Mtp {
    pub fn new() -> Self {
        Ds4Mtp {
            _priv: (),
            config: Ds4MtpConfig::default(),
            drafts: Vec::new(),
            accepted_run: 0,
        }
    }

    pub fn with_config(cfg: Ds4MtpConfig) -> Self {
        let drafts = Vec::with_capacity(cfg.draft_tokens);
        Ds4Mtp {
            _priv: (),
            config: cfg,
            drafts,
            accepted_run: 0,
        }
    }

    pub fn config(&self) -> &Ds4MtpConfig {
        &self.config
    }
    pub fn accepted_run(&self) -> usize {
        self.accepted_run
    }
    pub fn drafts(&self) -> &[u32] {
        &self.drafts
    }

    pub fn reset(&mut self) {
        self.drafts.clear();
        self.accepted_run = 0;
    }
}

impl Default for Ds4Mtp {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_empty() {
        let m = Ds4Mtp::new();
        assert_eq!(m.config().draft_tokens, 0);
        assert_eq!(m.accepted_run(), 0);
        assert!(m.drafts().is_empty());
    }

    #[test]
    fn reset_clears() {
        let mut m = Ds4Mtp::new();
        m.reset();
        assert_eq!(m.accepted_run(), 0);
    }
}
