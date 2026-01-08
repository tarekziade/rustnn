# Tensor API Refactoring Plan: Device-First by Default

## Goal

Align `create_tensor()` API with WebNN spec: device-resident tensors by default, with explicit opt-in for host access.

## Current State vs WebNN Spec

### WebNN Spec Behavior

```typescript
// WebNN JavaScript API
dictionary MLTensorDescriptor : MLOperandDescriptor {
  boolean readable = false;   // ← Device-resident by default
  boolean writable = false;   // ← Device-resident by default
};

// Creating a device-resident tensor (default)
const tensor = await context.createTensor({
  dataType: "float32",
  shape: [2, 3]
  // readable: false, writable: false (defaults)
});

// Creating a host-accessible tensor (explicit)
const hostTensor = await context.createTensor({
  dataType: "float32",
  shape: [2, 3],
  readable: true,
  writable: true
});
```

### Current rustnn Behavior (WRONG)

```python
# Current API - defaults to HOST-backed (backwards!)
tensor = context.create_tensor([2, 3], "float32")
# ↑ readable=True, writable=True by default

# Separate method for device tensors
device_tensor = context.create_device_tensor(graph, [2, 3], "float32")
```

**Problem**: This inverts the spec's intent. The spec assumes device-resident by default for performance.

## Proposed Refactoring

### New API Design

```python
# 1. Spec-compliant: device-resident by default
tensor = context.create_tensor([2, 3], "float32")
# ↑ readable=False, writable=False → device-resident (MLDeviceTensor)
# This is the PRIMARY API - use for production code

# 2. Spec-compliant: host-accessible when needed
host_tensor = context.create_tensor([2, 3], "float32", readable=True, writable=True)
# ↑ readable=True, writable=True → host-backed (MLTensor)
# Use when you need to inspect/debug tensor contents

# 3. Non-spec convenience: explicit host tensor
host_tensor = context.create_host_tensor([2, 3], "float32")
# ↑ Always host-backed, always readable/writable
# Syntactic sugar for create_tensor(..., readable=True, writable=True)
# Use for prototyping, debugging, or when you know you need host access

# 4. DEPRECATED: create_device_tensor() removed
# Use create_tensor() with default parameters instead
```

### Implementation Strategy

#### Phase 1: Add create_host_tensor() (Non-Breaking)

```python
# src/python/context.rs

#[pymethods]
impl PyMLContext {
    /// Convenience method for creating host-backed tensors (non-spec)
    ///
    /// This is equivalent to:
    ///   create_tensor(shape, data_type, readable=True, writable=True)
    ///
    /// Use this for:
    /// - Debugging and inspecting tensor contents
    /// - Quick prototyping
    /// - When you know you'll need host access
    ///
    /// For production code, prefer create_tensor() with explicit flags.
    #[pyo3(signature = (shape, data_type))]
    fn create_host_tensor(
        &self,
        shape: Vec<u32>,
        data_type: &str,
    ) -> PyResult<PyMLTensor> {
        // Just call create_tensor with host flags
        self.create_tensor(shape, data_type, true, true, false)
    }
}
```

#### Phase 2: Change create_tensor() Defaults (BREAKING)

```rust
// src/python/context.rs

#[pymethods]
impl PyMLContext {
    /// Create a tensor (device-resident by default, per WebNN spec)
    ///
    /// Following W3C WebNN specification:
    /// https://www.w3.org/TR/webnn/#dom-mlcontext-createtensor
    ///
    /// Args:
    ///     shape: Shape of the tensor
    ///     data_type: Data type string (e.g., "float32")
    ///     readable: If True, tensor can be read back to CPU (default: False)
    ///     writable: If True, tensor can be written from CPU (default: False)
    ///     exportable_to_gpu: If True, tensor can be exported as GPU texture (default: False)
    ///
    /// Returns:
    ///     MLTensor or MLDeviceTensor depending on flags
    ///
    /// Note:
    ///     Device-resident tensors (readable=False, writable=False) offer best
    ///     performance for iterative workloads like KV cache in transformers.
    #[pyo3(signature = (shape, data_type, readable=false, writable=false, exportable_to_gpu=false))]
    fn create_tensor(
        &self,
        shape: Vec<u32>,
        data_type: &str,
        readable: bool,
        writable: bool,
        exportable_to_gpu: bool,
    ) -> PyResult<PyObject> {  // ← Returns PyObject (could be MLTensor or MLDeviceTensor)
        use pyo3::Python;

        let py = Python::acquire_gil().python();

        // If host-accessible, create MLTensor
        if readable || writable {
            let tensor = self.create_mltensor_impl(shape, data_type, readable, writable, exportable_to_gpu)?;
            return Ok(tensor.into_py(py));
        }

        // Otherwise, create device tensor (requires backend support)
        match self.backend {
            Backend::OnnxCpu | Backend::OnnxGpu | Backend::TensorRT | Backend::CoreML => {
                // Need graph for device tensor creation - this is a problem!
                // See "Open Question" section below
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "Device tensors require graph parameter. Use create_tensor_for_graph() instead."
                ));
            }
            Backend::None => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "No backend available for device tensors"
                ));
            }
        }
    }
}
```

#### Phase 3: Remove create_device_tensor() (BREAKING)

```python
# Mark as deprecated first, then remove in next major version

@deprecated("Use create_tensor() instead - it creates device tensors by default")
def create_device_tensor(self, graph, shape, data_type, device=None):
    # Call new API
    return self.create_tensor_for_graph(graph, shape, data_type)
```

## Open Questions & Challenges

### Challenge 1: Device Tensors Require Graph Context

**Problem**: Creating device tensors requires a compiled graph (for session management).

Current `create_device_tensor()` signature:
```python
tensor = context.create_device_tensor(graph, shape, data_type)
                                     ^^^^^^ required!
```

But WebNN spec's `createTensor()` doesn't take a graph parameter:
```javascript
const tensor = await context.createTensor(descriptor);
                                         ^^^^^^^^^^^^ just descriptor
```

**Possible Solutions**:

**Option A: Lazy device tensor creation**
- `create_tensor()` always returns a "shell" tensor initially
- First `dispatch()` call materializes device storage
- Pro: Matches spec exactly
- Con: Complex lifetime management

**Option B: Require graph parameter for device tensors**
```python
# Host tensor (no graph needed)
host_tensor = context.create_tensor([2, 3], "float32", readable=True)

# Device tensor (graph required)
device_tensor = context.create_tensor_for_graph(graph, [2, 3], "float32")
```
- Pro: Explicit and clear
- Con: Deviates from spec

**Option C: Context holds "current graph"**
```python
context.set_current_graph(graph)  # Set context state
tensor = context.create_tensor([2, 3], "float32")  # Uses current graph
```
- Pro: Matches spec signature
- Con: Implicit state is error-prone

**Recommendation**: Option B with eventual migration to Option A

```python
# Phase 1: Explicit graph parameter (clear semantics)
device_tensor = context.create_tensor_for_graph(graph, [2, 3], "float32")

# Phase 2 (future): Lazy materialization (spec-compliant)
tensor = context.create_tensor([2, 3], "float32")  # Shell tensor
context.dispatch(graph, {"x": tensor}, {"y": tensor})  # Materialize on first use
```

### Challenge 2: Backward Compatibility

**Impact Analysis**:
- `create_tensor()` used in 27 places across tests
- Changing defaults from `readable=True` to `readable=False` is BREAKING
- All existing code would need updates

**Migration Strategy**:

**Step 1**: Add deprecation warnings (v0.x)
```python
# Warn when using old defaults
def create_tensor(self, shape, data_type, readable=None, writable=None, ...):
    if readable is None and writable is None:
        warnings.warn(
            "create_tensor() defaults will change in v1.0: "
            "readable=False, writable=False (device-resident). "
            "Use create_host_tensor() for host-backed tensors.",
            DeprecationWarning
        )
        readable = True  # Old default
        writable = True
```

**Step 2**: Update all examples and tests (v0.x)
```python
# Before
tensor = context.create_tensor([2, 3], "float32")

# After - explicit about host access
tensor = context.create_host_tensor([2, 3], "float32")
```

**Step 3**: Change defaults (v1.0 - breaking release)
```python
# New defaults match spec
def create_tensor(self, shape, data_type, readable=False, writable=False, ...):
    ...
```

## Implementation Plan

### Phase 1: Non-Breaking Additions (Current Version)

1. ✅ Add `create_host_tensor()` convenience method
2. ✅ Add `create_tensor_for_graph()` for device tensors
3. ✅ Add deprecation warnings to `create_device_tensor()`
4. ✅ Update documentation to recommend new APIs
5. ✅ Update all examples to use new APIs

**Result**: Three APIs coexist:
- `create_tensor()` - old behavior (readable=True by default) ⚠️ DEPRECATED DEFAULTS
- `create_host_tensor()` - new convenience (always host)
- `create_tensor_for_graph()` - device tensors (new)

### Phase 2: Breaking Changes (v1.0)

1. ❌ Change `create_tensor()` defaults to `readable=False, writable=False`
2. ❌ Remove `create_device_tensor()` entirely
3. ❌ Make `create_tensor()` automatically choose host vs device based on flags
4. ❌ Update migration guide

**Result**: Two APIs:
- `create_tensor()` - spec-compliant (device by default)
- `create_host_tensor()` - convenience (always host)

### Phase 3: Future Optimization (v2.0)

1. ❌ Implement lazy device tensor materialization
2. ❌ Remove `create_tensor_for_graph()` in favor of lazy approach
3. ❌ Make `create_tensor()` match spec signature exactly

**Result**: One spec-compliant API:
- `create_tensor(descriptor)` - matches WebNN spec perfectly

## Testing Strategy

### Update Existing Tests

```python
# tests/test_device_tensors.py

def test_device_tensor_is_default():
    """Verify create_tensor() defaults to device-resident"""
    context = get_context()

    # Default behavior: device-resident
    tensor = context.create_tensor([2, 3], "float32")
    assert isinstance(tensor, webnn.MLDeviceTensor)
    assert not tensor.readable
    assert not tensor.writable

def test_host_tensor_explicit():
    """Verify host tensors require explicit flags"""
    context = get_context()

    # Explicit host access
    tensor = context.create_tensor([2, 3], "float32", readable=True, writable=True)
    assert isinstance(tensor, webnn.MLTensor)
    assert tensor.readable
    assert tensor.writable

def test_host_tensor_convenience():
    """Verify create_host_tensor() convenience method"""
    context = get_context()

    # Convenience method
    tensor = context.create_host_tensor([2, 3], "float32")
    assert isinstance(tensor, webnn.MLTensor)
    assert tensor.readable
    assert tensor.writable
```

### Add Migration Tests

```python
# tests/test_migration.py

def test_old_api_with_warning():
    """Verify deprecated APIs emit warnings"""
    context = get_context()

    with pytest.warns(DeprecationWarning, match="create_device_tensor"):
        tensor = context.create_device_tensor(graph, [2, 3], "float32")
```

## Documentation Updates

### API Reference

```markdown
### MLContext.create_tensor()

**Spec-compliant**: Creates a tensor (device-resident by default)

```python
tensor = context.create_tensor(
    shape=[2, 3],
    data_type="float32",
    readable=False,    # Default: device-resident
    writable=False     # Default: device-resident
)
```

**Parameters**:
- `shape`: Tensor dimensions
- `data_type`: One of "float32", "float16", "int32", etc.
- `readable`: Allow reading to host (default: False)
- `writable`: Allow writing from host (default: False)

**Returns**: MLDeviceTensor (device) or MLTensor (host)

**Note**: Device-resident tensors offer best performance for iterative workloads.

---

### MLContext.create_host_tensor()

**Convenience**: Always creates host-backed tensor

```python
tensor = context.create_host_tensor(
    shape=[2, 3],
    data_type="float32"
)
# Equivalent to: create_tensor(..., readable=True, writable=True)
```

Use this for:
- Debugging and inspection
- Prototyping
- When you know you need host access

**Returns**: MLTensor (always host-backed, readable and writable)
```

## Benefits of This Refactoring

1. **Spec Compliance**: Matches WebNN JavaScript API behavior
2. **Performance**: Device-resident tensors by default (optimal)
3. **Clarity**: Explicit about host vs device allocation
4. **Flexibility**: Descriptor flags control behavior precisely
5. **Future-Proof**: Aligns with WebNN evolution

## Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking existing code | High | Multi-phase rollout with warnings |
| Confusion about APIs | Medium | Clear documentation and examples |
| Device tensor lifecycle | High | Start with explicit graph parameter |
| Backend inconsistency | Medium | Graceful fallback to host tensors |

## Timeline

| Phase | Version | Timeline | Breaking? |
|-------|---------|----------|-----------|
| Phase 1: Add new APIs | v0.4 | 1-2 weeks | No ✅ |
| Phase 2: Breaking changes | v1.0 | 1-2 months | Yes ⚠️ |
| Phase 3: Lazy materialization | v2.0 | Future | Maybe ⚠️ |

## Success Criteria

1. ✅ All examples use new API
2. ✅ Tests pass with new defaults
3. ✅ Migration guide complete
4. ✅ No performance regression
5. ✅ Matches WebNN spec behavior
