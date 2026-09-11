# WPT Test Guide

The in-repo WPT harness runs upstream [WebNN conformance tests](https://github.com/web-platform-tests/wpt/tree/master/webnn/conformance_tests) against rustnn's WebNN API (`MLGraphBuilder` + `MLContext::dispatch`). Tests are loaded live from `.https.any.js` files via a Node.js bridge — there is no checked-in JSON snapshot of the corpus.

**Entry point:** `tests/run_wpt_conformance.rs`  
**Harness modules:** `tests/wpt_conformance/`

## Prerequisites

- **Node.js** on `PATH` (used by `scripts/wpt_bridge/dump_corpus.mjs`)
- **WPT corpus** in `.cache/wpt` (fetched automatically on first run, or manually via `make fetch-wpt`)
- **ONNX Runtime** for the default `onnx` backend (`make onnxruntime-download` is a dependency of `make test-wpt`)
- **TensorRT** (optional) for the `trtx` backend — requires `trtx-runtime` feature and a working GPU; unavailable backends are skipped at startup

## Quick start

```bash
# Fetch WPT corpus (optional — first run auto-fetches if missing)
make fetch-wpt

# Full suite on ONNX CPU (~2482 cases, ~15–25 s)
make test-wpt

# Filter by operation name
make test-wpt-op OP=relu

# Full suite on TensorRT (skips if GPU unavailable)
make test-wpt-trtx
```

Always use `--test-threads 1` for WPT runs. Parallel execution is not validated for `MLContext` thread safety.

## How it works

```
WPT .https.any.js files
        │
        ▼
  dump_corpus.mjs (Node bridge)
        │
        ▼
  WptCorpus JSON  ──►  MLGraphBuilder  ──►  MLContext::dispatch
        │                                              │
        ▼                                              ▼
  tolerance check  ◄──────────────────────  actual outputs
        │
        ▼
  pass / fail (libtest)
```

1. **Load corpus** — One Node.js invocation parses all conformance `.https.any.js` files and returns JSON (`WptCorpus`).
2. **Build graph** — `wpt_execute_graph.rs` replays each case through `MLGraphBuilder` method calls.
3. **Execute** — `MLContext::dispatch` runs the graph on the selected backend.
4. **Validate** — `tolerance.rs` compares actual vs expected outputs (ULP, ATOL, or RTOL depending on operation and per-case overrides).

Trial names follow the pattern `{backend}::{operation}::{sanitized_test_name}`, for example:

```
onnx::relu::relu_float32_2D_tensor
trtx::clamp::clamp_uint64_1D_tensor_with_bigint_max
```

## Make targets

| Target | Description |
|--------|-------------|
| `make fetch-wpt` | Download/update WPT corpus into `.cache/wpt` |
| `make test-wpt` | Full suite, ONNX CPU backend |
| `make test-wpt-trtx` | Full suite, TensorRT backend |
| `make test-wpt-op OP=<name>` | Filter trials by operation (e.g. `OP=add`, `OP=dequantize`) |
| `make test-wpt-report` | Full ONNX run; writes JSON/HTML report even on failures |

Equivalent `cargo` invocations:

```bash
# ONNX CPU
cargo test --test run_wpt_conformance --features onnx-runtime -- --test-threads 1

# TensorRT only
WPT_BACKEND=trtx cargo test --test run_wpt_conformance \
  --features "onnx-runtime,trtx-runtime" -- trtx --test-threads 1

# Filter trials (libtest substring match on trial name)
cargo test --test run_wpt_conformance --features onnx-runtime -- relu --test-threads 1
```

## Backend selection

Set `WPT_BACKEND` to limit which backends register trials:

| Value | Backend | Notes |
|-------|---------|-------|
| `onnx` (default when unset) | ONNX Runtime CPU | `MLPowerPreference::Default`, `accelerated=false` |
| `trtx` | TensorRT | `MLPowerPreference::HighPerformance`, `accelerated=true` |

Aliases: `ort`, `cpu`, `tensorrt`, `trt` are also accepted.

When `WPT_BACKEND` is unset, all **available** backends register trials. Unavailable backends (e.g. TRTX without a GPU) are skipped with a log message — they do not count as skips in the summary.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WPT_DIR` | `.cache/wpt` | Path to WPT checkout |
| `WPT_BACKEND` | (all available) | Limit backend: `onnx` or `trtx` |
| `WPT_REPORT_JSON` | (none; `reports/wpt-conformance.json` when `CI` is set) | Write structured pass/fail JSON report |
| `WPT_REPORT_HTML` | (derived from JSON path) | HTML report path; set to empty string to disable |
| `WPT_AUDIT` | (off) | Enable per-pass error metrics collection (see [Audit mode](#audit-mode)) |
| `WPT_AUDIT_JSON` | `reports/wpt-trtx-audit.json` | Output path for audit JSON |

## Tolerance

Validation logic is in `tests/wpt_conformance/tolerance.rs`, aligned with pywebnn's `wpt_assert.py`.

- **ULP** — default for most float operations; per-operation defaults in `default_tolerances()` (e.g. `relu` = 0, `sigmoid` = 34, `conv2d` = rtol 1e-3).
- **ATOL** — used for some float16 trig operations.
- **RTOL** — used for conv2d/conv_transpose2d when no per-case ULP override is present.
- **Integer** — exact match by default; per-case tolerance from WPT JSON when specified.

WPT per-case tolerance overrides take precedence. For ULP, the applied tolerance is `max(wpt_value, merged_ulp_minimum(operation, graph_operators))`.

On failure, the harness prints inputs, expected/actual slices, and n-dimensional tensor dumps for triage.

## Audit mode

Green tests are not always numerically tight. Enable audit mode to record **max ULP**, **max absolute error**, and **max relative error** for every passing test, and flag cases that only pass because of wide tolerance.

```bash
# Run with audit (example: TensorRT backend)
WPT_BACKEND=trtx WPT_AUDIT=1 \
  cargo test --test run_wpt_conformance \
  --features "onnx-runtime,trtx-runtime" -- trtx --test-threads 1

# Summarize the audit JSON
python scripts/analyze_wpt_audit.py reports/wpt-trtx-audit.json
```

At the end of the run:

```
[WPT audit] 2482 passed, 143 flagged -> reports/wpt-trtx-audit.json
```

### Audit JSON fields

Each entry in `cases[]`:

| Field | Description |
|-------|-------------|
| `testName` | WPT test case name |
| `fileName` | Source `.https.any.js` file |
| `operation` | Primary operation under test |
| `toleranceKind` | `Ulp`, `Atol`, or `Rtol` |
| `toleranceValue` | Applied tolerance |
| `tightUlpMinimum` | Minimum ULP across graph operators (strict floor) |
| `maxUlp` | Peak ULP distance vs reference (float outputs) |
| `maxAbs` | Peak absolute error |
| `maxRtol` | Peak relative error |
| `maxIntDiff` | Peak integer difference (integer outputs) |
| `slackRatio` | `max_error / tolerance` (how much of the budget is used) |
| `flagged` | `true` if the case warrants review |
| `flagReasons` | Why it was flagged |

### Flag reasons

| Reason | Meaning |
|--------|---------|
| `max_ulp N exceeds tight minimum M` | Would fail at the strict per-operator ULP floor; passes only because applied tolerance is wider |
| `wide ULP tolerance N` | Applied tolerance ≥ 1000 ULP (policy flag; result may still be exact) |
| `uses X% of tolerance budget` | `slackRatio ≥ 0.5` — close to the edge |

**Note:** WPT cases that specify ULP tolerance also allow an absolute floor (`|actual − expected| ≤ 2e-6` for float32). A test can show enormous ULP but still pass via this abs floor. Check `maxUlp > toleranceValue` in the audit JSON to catch these.

## Structured reports

`make test-wpt-report` (or `WPT_REPORT_JSON=...`) writes a JSON conformance report compatible with `scripts/wpt_bridge/render_conformance_html.mjs`. HTML is generated automatically unless `WPT_REPORT_HTML=""`.

Report schema: per-file summaries, per-case status (`pass`/`fail`/`skip`), duration, and error messages.

## Module layout

| Module | Role |
|--------|------|
| `run_wpt_conformance.rs` | libtest entry point, trial registration |
| `wpt_js_loader.rs` | Node bridge, corpus loading, trial naming |
| `wpt_execute_graph.rs` | Graph build + dispatch via WebNN API |
| `wpt_backend.rs` | Backend selection (`WptBackend`, `WPT_BACKEND`) |
| `wpt_context_pool.rs` | Optional per-thread `MLContext` reuse |
| `wpt_config.rs` | Compile-time options (`REUSE_ML_CONTEXT`) |
| `tolerance.rs` | ULP/ATOL/RTOL validation |
| `expected_failures.rs` | Loads `{backend}_expected_failures.txt` allowlists |
| `wpt_audit.rs` | Per-pass error metrics (`WPT_AUDIT`) |
| `wpt_report.rs` | Structured JSON/HTML reports (`WPT_REPORT_JSON`) |
| `wpt_types.rs` | Corpus JSON types |
| `wpt_tensor.rs` | Tensor packing and dtype conversion |

## Failure tracking

A test is either passing or failing:

- **Passing** → a PASS snapshot (`tests/snapshots/run_wpt_conformance__{backend}_{test}.snap`).
  Used by `onnx`, `trtx`, and `litert`. CoreML produces no snapshots.
- **Failing** → listed in `tests/wpt_conformance/{backend}_expected_failures.txt`
  (one `{backend}::{operation}::{name}` per line). Used by `coreml` and `litert`.
  The test still runs, but its failure is non-fatal.

Failures are never snapshotted. Regenerate with:

```bash
make wpt-sync-onnx     # ONNX PASS snapshots
make wpt-sync-litert   # LiteRT PASS snapshots + expected-failures
make wpt-sync-coreml   # macOS: coreml expected-failures
make wpt-sync-trtx     # TensorRT PASS snapshots (requires a GPU)
```

A scheduled workflow (`.github/workflows/snapshot-sync.yml`) runs these and opens a PR.

## Troubleshooting

**`Node.js is required for WPT conformance tests`** — Install Node.js and ensure it is on `PATH`.

**`no WPT conformance cases loaded`** — Run `make fetch-wpt` or set `WPT_DIR` to a valid WPT checkout.

**TRTX backend skipped** — TensorRT is not available (no GPU, missing drivers, or `trtx-runtime` feature not enabled). Only `onnx` trials will run.

**Parse warnings (`file_errors`)** — Some WPT files may fail to parse; warnings are logged but do not fail the run.

**Slow TRTX runs** — TensorRT engine compilation per trial makes debug builds slow (~25–30 min for 2482 cases). Use `make test-wpt-op OP=<op>` to iterate on a single operation.
