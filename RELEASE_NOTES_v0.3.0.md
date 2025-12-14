# Release Notes - v0.3.0

Release Date: 2025-12-14

## Overview

Version 0.3.0 represents a major milestone in the rustnn project with 130 commits since v0.2.0. This release brings significant improvements in WPT (Web Platform Tests) conformance, TensorRT integration, Float16 support for CoreML, code quality improvements, and comprehensive documentation updates.

## Highlights

- **91.3% WPT Conformance**: Achieved high conformance with W3C WebNN specification test suite
- **TensorRT Integration**: Added NVIDIA GPU acceleration support via trtx-rs
- **CoreML Float16 Support**: Full Float16 data type support with MLPackage weight files
- **100+ Bug Fixes**: Comprehensive operation fixes across ONNX and CoreML backends
- **Code Quality**: Significant refactoring reducing code duplication and improving maintainability
- **Documentation**: Added IPC design notes, Windows setup guide, and Chromium comparison

## Major Features

### TensorRT Backend Integration
- Added TensorRT executor integration using trtx-rs (commit: 5bd4582d)
- Comprehensive Windows setup guide for TensorRT (commit: 2ab8ecff)
- Support for NVIDIA GPU acceleration on Linux and Windows
- Added TensorRT integration planning guide (commit: 86f1e4b7)

### CoreML Float16 Support
- Implemented complete Float16 support for CoreML backend (commit: bf45d9ba)
- Added MLPackage file generation with weights directory (Phase 3) (commit: 185517b9)
- Added Float16 weight file handling with integration tests (Phase 2) (commit: cba495ee)
- Created weight file builder infrastructure (Phase 1) (commit: 213ecc1f)
- Documented CoreML Float16 constant limitation with error handling (commit: 8e169f79)
- Fixed Float16 constant encoding following Chromium approach (commit: cc7750ee)
- Fixed float16 support for normalization default initializers (commit: 3e354de9)

### WPT Conformance Testing
- Achieved 91.3% WPT conformance (commit: 05232ec4)
- Added WPT conformance test data for 15 Tier 1 operations (395 tests) (commit: ead2f593)
- Added WPT conformance test data for 10 reduce operations (411 tests) (commit: 11b27a0f)
- Implemented working WPT test data converter with Node.js extraction (commit: 247de86b)
- Added comprehensive WPT operation name mappings (commit: 86eacea0)
- Fixed WPT test data parsing - 74 tests now passing (commit: 7e014122)
- Major WPT test fixes - 107 tests now passing (44% increase) (commit: 43688e47)
- Fixed reshape and gather WPT test failures (148 tests fixed) (commit: 13bedd8f)
- Added automatic type casting for ONNX float-only operations (8 tests fixed) (commit: 04ff4327)
- Fixed reduction operations axes parameter handling for ONNX opset 13 (commit: 62ce220b)
- Fixed gather out-of-bounds and slice 0D tensor WPT test failures (commit: 93e4db39)

### Backend Selection Improvements
- Added explicit backend selection via device_type parameter (commit: 67f080fe)
- Updated implementation status docs for explicit backend selection (commit: 0e66f0ee)

### ONNX Runtime Upgrade
- Migrated from onnxruntime-rs to ort v2.0.0-rc.10 (commit: cba6d946)
- Upgraded ONNX Runtime to v1.23.2 to fix cleanup crash (commit: d1546682)
- Updated CI workflow for ort v2.0 and ONNX Runtime v1.23.2 (commit: ac06dc9d)
- Used Once to initialize ort runtime only once per process (commit: d97a041b)

## Operation Fixes and Improvements

### Convolution Operations
- Fixed conv_transpose2d: add bias parameter and correct filter_layout default (commit: fe84e9ad)
- Fixed conv_transpose2d operation - add pad_type and correct outputSizes mapping (commit: f7bc3e50)
- Fixed convTranspose2d filter layout support for WPT conformance (commit: 3ef6c8ff)
- Fixed conv2d, split, and slice operations with full layout support (commit: f9f69e34)
- Added conv2d bias parameter support and fix WPT test failures (commit: 46bc9848)
- Fixed WPT parameter name mappings for conv operations (commit: 43977267)
- Added filter and input layout transformations for conv operations (commit: b5eb3c81)
- Fixed conv_transpose2d filter layout permutation for hwoi (commit: 424b567d)

### Normalization Operations
- Fixed batch_normalization mean/variance input ordering and axis-based shape calculation (commit: ce2c8b66)
- Fixed layer_normalization: split from batch/instance norm, add required axes parameter (commit: 6d12467c)
- Fixed layer_normalization axis and scale/bias shape calculation (commit: c34f555a)
- Fixed layer_normalization for 0D tensors and empty axes (commit: 32274ad0)
- Added validation for batch/instance normalization constant params (commit: 6fd4c491)
- Added validation for layer_normalization constant parameters (commit: e16b3e71)
- Added ONNX converter support for normalization operations (commit: 87188d92)
- Added TODO note for instance_normalization NHWC layout issue (commit: 74c271ea)

### Element-wise Operations
- Fixed neg operation: decompose into mul(x, -1) with typed constant (commit: 093726e3)
- Fixed hard_swish: decompose into sigmoid_hard + mul following Chromium (commit: 7e316d03)
- Added HardSwish decomposition for ONNX opset 13 compatibility (commit: 82e41ead)
- Fixed log and hardswish operations in CoreML converter (commit: de6742be)
- Fixed logical_not parameter mapping for WPT conformance tests (commit: df6e2ad3)
- Fixed clamp and concat operations in CoreML converter (commit: cb6e53b3)
- Fixed clamp float16 type mismatch - alpha/beta must match input type (commit: eab3e95c)
- Fixed clamp ONNX type matching for all data types (commit: 580787a9)
- Fixed clamp parameter mapping and None handling (commit: 345efccb)

### Reduction Operations
- Added reduction operations WPT test support (commit: 1576a4f5)
- Fixed reduce_l1 uint32 support with automatic type casting (commit: 74c271ea)
- Fixed reduceProduct operation - typo in operation name matching (commit: 3bf75f84)

### Other Operations
- Fixed expand operation: add reshape for rank-increasing cases (commit: 8102e510)
- Fixed expand operation in CoreML converter and add skip logic for limitations (commit: 2e4dcb72)
- Fixed expand operation bug in CoreML converter (commit: c36dd0a6)
- Fixed expand operation - add shape as second ONNX input (commit: 612a1ed9)
- Fixed gather operation by setting validate_indices=false (commit: 23bcde9f)
- Improved gather axis parameter handling (commit: 02bf7f73)
- Fixed CoreML matmul and neg operations by adding required parameters (commit: 3bf75f84)
- Fixed concat operation: resolve list operands and add ONNX axis attribute (commit: c27e04dd)
- Fixed CoreML cast operation dtype handling and add unsupported type skip logic (commit: fd46e237)
- Fixed CoreML cast operation dtype parameter (commit: 1af0e271)
- Fixed CoreML gather operation parameter names (commit: 2554cdb2)
- Fixed critical CoreML converter bugs: add missing required parameters (commit: cb9221e9)

### Data Type Support
- Added comprehensive data type support for ONNX executor (commit: 8398c896)
- Enabled all WPT tests: support all data types and large tensors (commit: 353a8366)
- Implemented bool → uint8 casting for ONNX (98% Chromium parity) (commit: 5b1f1640)

### Scalar and Edge Case Handling
- Fixed CoreML 0D (scalar) tensor handling (commit: 704ea5a2)
- Fixed CoreML executor panic by adding proper error handling (commit: bbe61322)

## Code Quality Improvements

### Refactoring (This Session)
- Refactor: extract common operand_name() helper (commit: 5163a356)
  - Eliminated 12 lines of code duplication across ONNX and CoreML converters
  - Updated 43 call sites (30 ONNX + 13 CoreML)
  - Net reduction: 71 insertions, 81 deletions
- Refactor: extract JSON attribute parsing helpers in ONNX converter (commit: 04e9ecd1)
  - Created 3 helper functions reducing JSON parsing duplication
  - Refactored 5 attribute creation functions
  - Net reduction: 70 insertions, 226 deletions (156 lines saved)
  - Reduced ONNX converter from 2253 to 2084 lines (-7.5%)
- Fix Rust warnings and Python test issues (commit: 3507a5cf)
  - Fixed 3 Rust compiler warnings
  - Fixed 4 Python fixture errors
  - Fixed 1 flaky performance test
- Fix unsafe function call in CoreML executor (commit: 134f310e)
  - Added proper unsafe block wrapping
  - Added #[allow(dead_code)] to 5 utility functions

### Skip Logic and Platform Limitations
- Skip CoreML layer_normalization with empty axes (commit: a535b9ce)
- Skip CoreML normalization tests with non-constant parameters (commit: 939c5e53)
- Skip CoreML tests with known platform limitations (commit: 212eedd2)
- Skip gather operations with 0D indices on CoreML (commit: c28d5511)
- Add skip logic for CoreML type and dimension limitations (commit: cdbec718)
- Expand Float16 skip logic to check constants and outputs (commit: 866b1c72)
- Add skip logic for Float16 input tests with large arrays on CoreML (commit: 4c1ee100)
- Mark 32 architectural limitation tests as skipped (commit: e05cddca)

### Testing Improvements
- Enable dual-backend testing for ONNX and CoreML (commit: 8398c896)
- Add separate Makefile targets for ONNX and CoreML WPT testing (commit: 7b6c4e2b)
- Add multi-output operation support to test harness (commit: 345efccb)
- Fixed pytest fixture parametrization for Python tests (commit: 4d4299f7)
- Fix CI failures: pytest fixture and docs build issues (commit: e251565f)

### Performance
- Implement full CoreML execution with real input/output data (commit: 117ebb34)
- Add performance benchmarks and documentation (commit: 3a7b880d)
- Optimize mobilenet demo to use fastest CoreML configuration (commit: 8fa96dcc)
- Fix CoreML performance regression: skip expensive execution (commit: 3f82b107)
- Fix mobilenet-demo target by installing PIL dependency (commit: 658fe107)

## Documentation

### New Documentation
- Add IPC design documentation (commit: 6b8e7cef)
  - Comprehensive 454-line design document for future IPC implementation
  - Analysis of Chromium's Mojo approach
  - Three design options with Cap'n Proto recommended
  - Migration strategy and implementation checklist
- Add comprehensive Windows setup guide for TensorRT (commit: 2ab8ecff)
- Add Chromium WebNN implementation comparison (commit: 6a796953)
- Add Chromium comparison to docs navigation (commit: 2a1568a1)
- Add Chromium comparison analysis and fix graphviz test (commit: 98b0f274)
- Add TensorRT integration planning guide (commit: 86f1e4b7)
- Add GGML integration planning guide (commit: 952c4c5b)
- Add CoreML fixes session learnings and summary (commit: f7fe8389)

### Documentation Updates
- Update implementation status docs for explicit backend selection (commit: 0e66f0ee)
- Document CoreML backend priority issue (commit: 67b7172f)
- Update CoreML documentation with accurate explanation (commit: 9f1c546c)
- Update implementation status docs to reflect 100% pass rate (commit: 871fb689)
- Update documentation: 91.3% WPT conformance achieved (commit: 05232ec4)
- Update implementation status: 90.2% WPT conformance (commit: 81d70279)
- Update implementation status with latest WPT progress (commit: 32274ad0)
- Update implementation status with test improvements (commit: 96cc35f2)
- Update operator status doc date (commit: d6707965)
- Update status (commit: f9f69e34)
- Reorganize implementation status into single alphabetically sorted table (commit: 060dd140)
- Merge operator-status.md and wpt-integration-plan.md into implementation-status.md (commit: 99ff5b1e)
- Update Chromium comparison after ort v2.0 migration (commit: 6bca4198)
- Rebuild docs with bool→uint8 fix and 98% compatibility update (commit: 77bac81d)
- Rebuild docs with updated installation instructions (commit: 30651728)
- Improve installation docs and PyPI description (commit: 064e4a81)
- Update FLOAT16_STATUS.md with expand operation fixes and current status (commit: d4df4704)
- Document CoreML Float16 constant limitation with error handling (commit: 8e169f79)

### Style Improvements
- Remove emojis from project and add no-emoji policy (commit: a5bb7284)

## Breaking Changes

None. This release maintains backward compatibility with v0.2.0.

## Bug Fixes

See "Operation Fixes and Improvements" section above for the complete list of 100+ bug fixes.

## Known Issues and Limitations

- CoreML backend has some platform-specific limitations documented in skip logic
- Float16 support on CoreML has some edge cases with large arrays
- Some normalization operations require constant parameters (validation enforced)
- Instance normalization NHWC layout has known issues (documented in TODO)

## Migration Guide

No migration required. Version 0.3.0 is a drop-in replacement for v0.2.0.

## Contributors

- Tarek Ziade (@tarekziade)

## Statistics

- **Commits**: 130 commits since v0.2.0
- **WPT Conformance**: 91.3% (up from ~50% in v0.2.0)
- **Code Quality**: Reduced ONNX converter by 169 lines through refactoring
- **Test Coverage**: 2262 Python tests passing, 133 Rust tests passing
- **Documentation**: Added 500+ lines of new documentation

---

**Note**: This is a development release. While the library is functional and passes extensive testing, it should be thoroughly tested in your specific use case before production deployment.
