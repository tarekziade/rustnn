//! Debug utilities for rustnn
//!
//! Provides centralized debug logging controlled by the RUSTNN_DEBUG environment variable.
//! Set RUSTNN_DEBUG=1 or RUSTNN_DEBUG=true to enable debug output.

use std::env;
use std::sync::OnceLock;

static DEBUG_ENABLED: OnceLock<bool> = OnceLock::new();

/// Check if debug mode is enabled via RUSTNN_DEBUG environment variable
#[inline]
pub fn debug_enabled() -> bool {
    *DEBUG_ENABLED.get_or_init(|| {
        env::var("RUSTNN_DEBUG")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Debug print macro that only outputs when RUSTNN_DEBUG is enabled
#[macro_export]
macro_rules! debug_print {
    ($($arg:tt)*) => {
        if $crate::debug::debug_enabled() {
            eprintln!($($arg)*);
        }
    };
}
