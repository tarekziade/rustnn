#!/bin/bash
# Sync expected-failures file with current WPT results.
# Usage: ./scripts/update_expected_failures.sh <backend>
#
# 1. Empty the expected-failures file (every failure becomes visible).
# 2. Run WPT to collect actual failures.
# 3. Extract failures from the log, rebuild expected_failures.txt
#
# If no expected-failures file exists, one is created from the failures.

set -e

usage() {
    cat >&2 <<'EOF'
Usage: scripts/update_expected_failures.sh <backend>

  <backend>    onnx | trtx | litert | coreml

EOF
    exit 2
}

BACKEND="${1:-}"
[ -z "$BACKEND" ] && usage

case "$BACKEND" in
    onnx)   MAKE_TARGET="test-wpt"       ;;
    trtx)   MAKE_TARGET="test-wpt-trtx"  ;;
    litert) MAKE_TARGET="test-wpt-litert" ;;
    coreml) MAKE_TARGET="test-wpt-coreml" ;;
    *)      usage ;;
esac

PROJECT_DIR="$(dirname "$(cd "$(dirname "$0")" && pwd)")"
EXPECTED="$PROJECT_DIR/tests/wpt_conformance/${BACKEND}_expected_failures.txt"
BACKUP="${EXPECTED}.bak"
FAILURES="/tmp/wpt_${BACKEND}_failures.txt"
WPT_LOG="/tmp/wpt_${BACKEND}_sync.log"

_cleanup()   { rm -f "$FAILURES" "$WPT_LOG"; }
_restore()   { [ -f "$BACKUP" ] && [ ! -s "$EXPECTED" ] && cp "$BACKUP" "$EXPECTED"; }
_on_exit()   { _restore; _cleanup; }

trap _on_exit EXIT
trap '_on_exit; exit 130' INT TERM

cd "$PROJECT_DIR"

# ---- Run WPT ----

if [ -f "$EXPECTED" ]; then
    cp "$EXPECTED" "$BACKUP"
    echo "" > "$EXPECTED"
fi

set +e
make "$MAKE_TARGET" 2>&1 | tee "$WPT_LOG" || true
set -e

# ---- Extract failures from log ----

sed -nE "s/^[[:space:]]{4,}(${BACKEND}::[^[:space:]]+).*/\1/p" "$WPT_LOG" | sort -u > "$FAILURES"
num_failing=$(wc -l < "$FAILURES")

# ---- Rebuild (or create) expected-failures.txt ----

if [ -f "$EXPECTED" ]; then

    old_count=$(grep "^${BACKEND}::" "$BACKUP" 2>/dev/null | wc -l)

    grep '^#' "$BACKUP" > "$EXPECTED" 2>/dev/null || true
    cat "$FAILURES" >> "$EXPECTED"
    sort -u -o "$EXPECTED" "$EXPECTED"

    new_count=$(grep "^${BACKEND}::" "$EXPECTED" 2>/dev/null | wc -l)
    added_entries=$(comm -13 <(sort "$BACKUP" 2>/dev/null) "$EXPECTED" | grep "^${BACKEND}::" || true)
    removed_entries=$(comm -23 <(sort "$BACKUP" 2>/dev/null) "$EXPECTED" | grep "^${BACKEND}::" || true)
    added=$(printf '%s' "$added_entries" | grep -c '^'"${BACKEND}::" || true)
    removed=$(printf '%s' "$removed_entries" | grep -c '^'"${BACKEND}::" || true)

    echo ""
    echo "=== Rebuilding expected failures ==="
    echo "Current failures: $num_failing"
    echo "Synced: $old_count -> $new_count entries (+$added added, -$removed removed)"

    [ -n "$added_entries" ]   && { echo ""; echo "Added:";   echo "$added_entries"   | sed 's/^/  /'; }
    [ -n "$removed_entries" ] && { echo ""; echo "Removed:"; echo "$removed_entries" | sed 's/^/  /'; }
    echo ""
    echo "Backup: $BACKUP"

elif [ "$num_failing" -gt 0 ]; then

    cp "$FAILURES" "$EXPECTED"
    echo ""
    echo "=== Created expected-failures file ==="
    echo "Current failures: $num_failing"
    echo "New file: $EXPECTED"

else

    echo ""
    echo "No failures found and no expected-failures file exists."
    echo "See full output in: $WPT_LOG"

fi

trap - EXIT
_cleanup
