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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debug_enabled_default() {
        // Without RUSTNN_DEBUG set, should be false
        // Note: This test assumes RUSTNN_DEBUG is not set in test environment
        // If it is set, the result will be different
        let _ = debug_enabled();
        // Just verify it doesn't panic
    }

    #[test]
    fn test_debug_enabled_consistency() {
        // debug_enabled() should return the same value on subsequent calls
        // (OnceLock guarantees initialization happens only once)
        let first = debug_enabled();
        let second = debug_enabled();
        assert_eq!(
            first, second,
            "debug_enabled() should return consistent values"
        );
    }

    #[test]
    fn test_debug_print_macro_compiles() {
        // Test that the macro compiles and executes without panic
        debug_print!("Test message");
        debug_print!("Test with arg: {}", 42);
        debug_print!("Multiple args: {} {}", "hello", "world");
    }

    #[test]
    fn test_debug_print_macro_with_formatting() {
        // Test various formatting patterns
        debug_print!("Simple string");
        debug_print!("Number: {}", 123);
        debug_print!("Multiple: {} {} {}", 1, 2, 3);
        debug_print!("Named: {value}", value = 42);
        debug_print!("Debug: {:?}", vec![1, 2, 3]);
    }
}
