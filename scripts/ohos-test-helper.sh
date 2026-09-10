#!/bin/bash
# CANN OHOS device helpers — push and test the Rust device test binary.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ -z "$CANN_DDK" ]; then
    echo "Error: CANN_DDK is not set."
    echo "  export CANN_DDK=/path/to/CANN-Kit-next/ddk/"
    exit 1
fi

DDK_LIB="${CANN_DDK}/ai_ddk_lib/lib64"

BINARY_DIR="${PROJECT_DIR}/target/aarch64-unknown-linux-ohos/release/deps"
# libtest gives the test binary a hashed name; pick the most recently built.
BINARY="$(ls -t "${BINARY_DIR}"/test_cann_execution-* 2>/dev/null | head -1)"
DEVICE_BIN="test_cann_execution"

# ── Helpers ────────────────────────────────────────────────────────────

ok()  { echo "  [OK] $*"; }
fail(){ echo "  [FAIL] $*"; exit 1; }

# ── Push to device ─────────────────────────────────────────────────────

push_to_device() {
    local target="${1:-/data/local/tmp/cann-test}"

    if ! command -v hdc &>/dev/null; then
        echo "Error: hdc not found. Install Huawei DevEco Device Tool."
        exit 1
    fi

    echo "=== Pushing to ${target} ==="
    hdc shell "mkdir -p ${target}"

    if [ -n "$BINARY" ] && [ -f "$BINARY" ]; then
        echo "  ${BINARY##*/} -> ${target}/${DEVICE_BIN}"
        hdc file send "$BINARY" "${target}/${DEVICE_BIN}"
    else
        echo "  WARN: test_cann_execution binary not found. Run 'make cann-device-test' first."
    fi

    for lib in libhiai.so libhiai_ir.so libhiai_ir_build.so libhiai_ir_build_aipp.so; do
        local src="${DDK_LIB}/${lib}"
        if [ -f "$src" ]; then
            echo "  ${lib} -> ${target}/"
            hdc file send "$src" "${target}/"
        fi
    done

    hdc shell "chmod +x ${target}/${DEVICE_BIN}"
    ok "pushed"
}

# ── Test on device ─────────────────────────────────────────────────────

test_on_device() {
    local target="${1:-/data/local/tmp/cann-test}"
    push_to_device "$target"
    echo ""
    echo "=== Running on device ==="
    hdc shell "cd ${target} && LD_LIBRARY_PATH=. ./${DEVICE_BIN} --nocapture --test-threads=1"
}

# ── Main ───────────────────────────────────────────────────────────────

TARGET="${1:-test}"

case "$TARGET" in
    push) push_to_device "$2" ;;
    test) test_on_device "$2" ;;
    *)
        echo "Usage: $0 [push|test] [target-dir]"
        echo ""
        echo "  push    Transfer files to OHOS device via hdc"
        echo "  test    Push + execute test_cann_execution on device"
        echo ""
        echo "  Default target-dir: /data/local/tmp/cann-test"
        exit 1
        ;;
esac
