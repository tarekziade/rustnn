# WebNN Static Shapes Limitation

**Date:** 2026-01-09
**Status:** Documented limitation, not a bug

---

## Summary

WebNN models use **static shapes** compiled at conversion time and cannot handle truly dynamic shapes like ONNX models. This affects variable-length sequence models (e.g., LLMs with different prompt lengths).

---

## Impact

### For All Models
- ✅ **Bug Fix Applied:** Output ordering now ensures "logits" is always at index 0
- ✅ **Bug Fix Applied:** Empty KV cache inputs (past_sequence_length=0) automatically skipped
- ✅ **Converter Working:** WebNN→ONNX conversion faithfully reproduces graph structure

### For Variable-Length Models (LLMs)
- ⚠️ **Limitation:** WebNN models compiled for specific sequence length (e.g., 128 tokens)
- ⚠️ **Requirement:** Inputs must be padded to match compilation length exactly
- ⚠️ **Result:** Incorrect outputs if actual length differs from compiled length

---

## Technical Details

### Root Cause

WebNN specification requires:
1. All dynamic dimensions resolved to static values at conversion time
2. Shape calculations constant-folded for operations like Expand/Reshape
3. This creates "baked" models with fixed input shapes

### Example: SmolLM-135M

**Conversion Parameters:**
```json
{
  "past_sequence_length": 0,
  "past_sequence_length + 1": 128,
  "batch_size": 1,
  "sequence_length": 128
}
```

**What Gets Baked:**
- 218 Shape→Gather operations folded to constants
- Position embeddings computed for indices 0-127
- Attention masks expected shape [1, 128]
- All reshape operations use fixed dimensions

**Runtime Behavior:**
- ✅ 128-token input: Works correctly (matches compilation)
- ❌ 5-token input: Uses position 0-127 embeddings (wrong!)
- ❌ 200-token input: Cannot fit, shape mismatch

---

## Investigation Results

### Bug #1: Output Ordering - FIXED ✅

**Problem:** Logits at output index 60 instead of 0

**Fix:** `src/converters/onnx.rs` lines 783-807
```rust
// Sort outputs: "logits" first, then alphabetically by name
let mut sorted_outputs: Vec<u32> = graph.output_operands.clone();
sorted_outputs.sort_by_key(|&id| {
    let operand = graph.operand(id);
    let name = operand.and_then(|op| op.name.as_deref()).unwrap_or("");
    if name == "logits" {
        (0, String::new())  // Priority 0 = first
    } else {
        (1, name.to_string())  // Priority 1 = alphabetical
    }
});
```

**Benefit:** All models now have predictable output ordering

### Bug #2: Logit Corruption - LIMITATION DOCUMENTED ⚠️

**Initial Hypothesis:** ONNX→WebNN conversion corrupts model
**Actual Finding:** WebNN static shape limitation

**Fix Attempts:**
1. Disabled Gather folding in shape_inference.rs → Still folded
2. Found second location in convert.rs → Disabled both
3. Result: Conversion fails - Expand needs constant shapes
4. Conclusion: **Folding is mandatory for WebNN**

**Evidence:**
- Original ONNX: Dynamic shapes, works with any sequence length
- WebNN model: Static shapes, compiled for 128 tokens
- Converted ONNX: Preserves static compilation from WebNN

---

## Recommendations

### For Users

**When to use WebNN:**
- ✅ Fixed-size inputs (images, fixed-length sequences)
- ✅ Deployment scenarios with known input shapes
- ✅ Cross-platform inference with consistent shapes

**When to use ONNX:**
- ✅ Variable-length inputs (chatbot prompts, translations)
- ✅ Development and validation
- ✅ Models requiring dynamic batching

### For Developers

**Testing WebNN→ONNX Conversion:**
```python
# Correct: Use padding matching original WebNN compilation
input_ids = np.pad(tokens, (0, 128 - len(tokens)), constant_values=pad_token)

# Wrong: Use actual token count without padding
input_ids = np.array(tokens)  # Shape doesn't match compilation
```

**Documentation to Add:**
```python
def convert_to_onnx(self, graph, output_path):
    """
    Convert WebNN graph to ONNX format.

    WARNING: WebNN models use static shapes compiled at conversion time.
    The resulting ONNX model preserves these static shapes and will not
    handle variable-length inputs correctly.

    For models compiled with sequence_length=128, inputs must be padded
    to exactly 128 tokens. Use the original dynamic ONNX model for
    variable-length inference.
    """
```

---

## Files Modified

### Source Code (Ready to Commit)
- ✅ `src/converters/onnx.rs` - Output ordering fix + empty KV cache handling
- ✅ `src/python/context.rs` - Empty KV cache handling in compute methods

### Documentation (Keep for Reference)
- ✅ `docs/webnn_static_shapes_limitation.md` - This document
- ✅ `docs/investigation_complete.md` - Full investigation summary
- ✅ `docs/bug2_investigation_progress.md` - Detailed investigation steps

### Test Scripts (Keep for Debugging)
- ✅ `scripts/compare_onnx_graphs.py` - Graph comparison tool
- ✅ `examples/test_converted_onnx.py` - Test with proper padding

---

## Conclusion

**Rustnn Status:**
- ✅ WebNN→ONNX converter working correctly
- ✅ Output ordering fixed for all models
- ✅ Empty KV cache handling for LLM models
- ✅ Not responsible for static shape compilation

**WebNN Specification:**
- ⚠️ Static shapes only (no dynamic dimensions)
- ⚠️ Requires constant folding for shape operations
- ⚠️ Creates compiled models for specific input sizes

**Next Steps:**
1. Commit Bug #1 fix and empty KV cache handling
2. Document WebNN limitations in user-facing docs
3. Update test expectations for WebNN→ONNX round-trip
4. Consider warning users about static shape limitations

---

**Investigation conducted:** 2026-01-09
**Time invested:** ~4 hours
**External repos analyzed:** webnn-wg (no changes needed)
