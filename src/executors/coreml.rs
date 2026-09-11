//! Minimal CoreML execution bridge for macOS.
//! Loads a `.mlmodel`, compiles it if needed, and runs a zeroed inference
//! using CoreML's Objective-C API.

#![cfg(feature = "coreml-runtime")]

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::io::Write;
use std::os::raw::{c_char, c_void};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::Arc;
use std::sync::mpsc;

use block::ConcreteBlock;
use objc::rc::autoreleasepool;
use objc::runtime::{Class, Object};
use objc::{class, msg_send, sel, sel_impl};

use crate::error::GraphError;
use crate::graph::{DataType, Dimension, OperandDescriptor, get_static_or_max_size};
use crate::runtime_checks::{RuntimeShapeState, TensorKind, validate_shape_data_length};

// Link against the system frameworks we use.
#[cfg(target_os = "macos")]
#[link(name = "Foundation", kind = "framework")]
unsafe extern "C" {}
#[cfg(target_os = "macos")]
#[link(name = "CoreML", kind = "framework")]
unsafe extern "C" {}

// Objective-C++ exception firewall (src/executors/coreml_shim.mm).
// Return codes: 0 = success, 1 = NSError, 2 = NSException, 3 = C++ exception.
#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn rustnn_coreml_compile(
        model_url: *mut Object,
        out_url: *mut *mut Object,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
    fn rustnn_coreml_load(
        compiled_url: *mut Object,
        configuration: *mut Object,
        out_model: *mut *mut Object,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
    fn rustnn_coreml_predict(
        model: *mut Object,
        features: *mut Object,
        out_provider: *mut *mut Object,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
}

// Shims to check compilation on Linux
#[cfg(not(target_os = "macos"))]
pub unsafe extern "C" fn rustnn_coreml_compile(
    _model_url: *mut Object,
    _out_url: *mut *mut Object,
    _error: *mut c_char,
    _error_length: usize,
) -> i32 {
    1
}
#[cfg(not(target_os = "macos"))]
pub unsafe extern "C" fn rustnn_coreml_load(
    _compiled_url: *mut Object,
    _configuration: *mut Object,
    _out_model: *mut *mut Object,
    _error: *mut c_char,
    _error_length: usize,
) -> i32 {
    1
}
#[cfg(not(target_os = "macos"))]
pub unsafe extern "C" fn rustnn_coreml_predict(
    _model: *mut Object,
    _features: *mut Object,
    _out_provider: *mut *mut Object,
    _error: *mut c_char,
    _error_length: usize,
) -> i32 {
    1
}

/// Releases an owned (+1) Objective-C object when dropped (including on early
/// `return`/`?`), balancing any +1 ownership we hold: the retain transferred by
/// the shim's `__bridge_retained` casts, or objects we created via
/// `new`/`alloc`+`init`.
struct ReleaseOnDrop(*mut Object);

impl ReleaseOnDrop {
    /// Take the pointer back out without releasing, handing ownership (+1)
    /// to the caller.
    fn into_inner(self) -> *mut Object {
        let ptr = self.0;
        std::mem::forget(self);
        ptr
    }
}

impl Drop for ReleaseOnDrop {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.0, release];
        }
    }
}

fn shim_error_to_string(buffer: &[u8]) -> String {
    let end = buffer
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(buffer.len());
    String::from_utf8_lossy(&buffer[..end]).into_owned()
}

#[derive(Debug, Clone)]
pub struct CoremlInput {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct CoremlOutput {
    pub name: String,
    pub shape: Vec<i64>,
    pub data_type_code: i64,
    pub data: Vec<f32>, // Output data converted to f32 for consistency
}

#[derive(Debug, Clone)]
pub struct CoremlRunAttempt {
    pub compute_unit: &'static str,
    pub result: Result<Vec<CoremlOutput>, String>,
}

pub fn run_coreml_zeroed(
    model_bytes: &[u8],
    inputs: &HashMap<String, OperandDescriptor>,
) -> Result<Vec<CoremlRunAttempt>, GraphError> {
    run_coreml_zeroed_cached(model_bytes, inputs, None)
}

pub fn run_coreml_zeroed_cached(
    model_bytes: &[u8],
    inputs: &HashMap<String, OperandDescriptor>,
    compiled_path: Option<&Path>,
) -> Result<Vec<CoremlRunAttempt>, GraphError> {
    run_coreml_zeroed_cached_with_weights(model_bytes, None, inputs, compiled_path)
}

/// Run CoreML inference with zeroed inputs and optional weight file
pub fn run_coreml_zeroed_cached_with_weights(
    model_bytes: &[u8],
    weights_data: Option<&[u8]>,
    inputs: &HashMap<String, OperandDescriptor>,
    compiled_path: Option<&Path>,
) -> Result<Vec<CoremlRunAttempt>, GraphError> {
    autoreleasepool(|| {
        run_impl_zeroed_with_weights(model_bytes, weights_data, inputs, compiled_path)
    })
}

/// Run CoreML inference with actual input data
pub fn run_coreml_with_inputs(
    model_bytes: &[u8],
    inputs: Vec<CoremlInput>,
) -> Result<Vec<CoremlRunAttempt>, GraphError> {
    run_coreml_with_inputs_with_weights(model_bytes, None, inputs)
}

/// Run CoreML inference with actual input data and optional weight file
pub fn run_coreml_with_inputs_with_weights(
    model_bytes: &[u8],
    weights_data: Option<&[u8]>,
    inputs: Vec<CoremlInput>,
) -> Result<Vec<CoremlRunAttempt>, GraphError> {
    autoreleasepool(|| {
        run_impl_with_inputs_with_weights(model_bytes, weights_data, inputs, None, None, None)
    })
}

/// Run CoreML inference with actual input data and model caching
pub fn run_coreml_with_inputs_cached(
    model_bytes: &[u8],
    inputs: Vec<CoremlInput>,
    cache_path: Option<&Path>,
) -> Result<Vec<CoremlRunAttempt>, GraphError> {
    autoreleasepool(|| {
        run_impl_with_inputs_with_weights(model_bytes, None, inputs, cache_path, None, None)
    })
}

/// Run CoreML inference with runtime descriptor checks for dynamic dimensions.
pub fn run_coreml_with_inputs_checked(
    model_bytes: &[u8],
    inputs: Vec<CoremlInput>,
    input_descriptors: &HashMap<String, OperandDescriptor>,
    output_descriptors: &HashMap<String, OperandDescriptor>,
) -> Result<Vec<CoremlRunAttempt>, GraphError> {
    autoreleasepool(|| {
        run_impl_with_inputs_with_weights(
            model_bytes,
            None,
            inputs,
            None,
            Some(input_descriptors),
            Some(output_descriptors),
        )
    })
}

// ---------------------------------------------------------------------------
// Byte-oriented execution path for the unified WebNN IDL API (src/backends/coreml.rs).
//
// Unlike the f32-centric `CoremlInput`/`CoremlOutput` helpers above, this path
// works in raw bytes keyed by feature name plus an `OperandDescriptor`, mirroring
// the ONNX Runtime backend (`src/backends/ort.rs`). The model is compiled and
// loaded once (`compile_model`) and reused across dispatches (`run_coreml_bytes`).
// ---------------------------------------------------------------------------

/// A CoreML model that has been compiled and loaded once, ready for repeated dispatch.
///
/// Owns a retained `MLModel` and the in-memory CoreML asset backing it. All are
/// released when the value is dropped. This type is intentionally not `Send`/`Sync`:
/// `MLGraph`/`MLContext` are single-threaded, matching CoreML's usage model.
pub(crate) struct CompiledCoremlModel {
    /// Retained `MLModel` Objective-C object.
    model: *mut Object,
    /// Compute unit the model was successfully loaded with (diagnostic only).
    compute_unit: &'static str,
    backing: CoremlModelBacking,
}

enum CoremlModelBacking {
    InMemory {
        /// Retained `MLModelAsset`. CoreML may refer to it after loading.
        asset: *mut Object,
        /// Retained model specification data backing `asset`.
        specification_data: *mut Object,
        /// Retained external weights data backing `asset`, if any.
        weights_data: Option<*mut Object>,
    },
    OnDisk {
        compiled_dir: PathBuf,
        /// Owns the temporary `.mlmodel`/`.mlpackage` source; removed on drop. Held
        /// purely as a drop guard (never read directly).
        #[allow(dead_code)]
        temp_model: Option<TempModelSource>,
    },
}

impl std::fmt::Debug for CompiledCoremlModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledCoremlModel")
            .field("compute_unit", &self.compute_unit)
            .finish()
    }
}

impl Drop for CompiledCoremlModel {
    fn drop(&mut self) {
        if !self.model.is_null() {
            unsafe {
                let _: () = msg_send![self.model, release];
            }
        }
        match &self.backing {
            CoremlModelBacking::InMemory {
                asset,
                specification_data,
                weights_data,
            } => unsafe {
                let _: () = msg_send![*asset, release];
                let _: () = msg_send![*specification_data, release];
                if let Some(weights_data) = weights_data {
                    let _: () = msg_send![*weights_data, release];
                }
            },
            CoremlModelBacking::OnDisk { compiled_dir, .. } => {
                // The compiled `.mlmodelc` is produced by CoreML (not a tempfile handle),
                // so remove it explicitly. `temp_model`, if any, cleans itself up on drop.
                let _ = std::fs::remove_dir_all(compiled_dir);
            }
        }
    }
}

/// Raw-byte input for [`run_coreml_bytes`]: a tensor's bytes plus its descriptor.
pub(crate) struct CoremlByteInput<'a> {
    pub(crate) data: &'a [u8],
    pub(crate) descriptor: &'a OperandDescriptor,
}

/// Map a [`DeviceType`] to an `MLComputeUnits` raw value.
///
/// Apple's `MLComputeUnits`: `cpuOnly = 0`, `cpuAndGPU = 1`, `all = 2`, `cpuAndNeuralEngine = 3`.
fn compute_unit_for_device(
    device_type: crate::backend_selection::DeviceType,
) -> (i64, &'static str) {
    match device_type {
        crate::backend_selection::DeviceType::Npu => (3, "CPU_AND_NE"),
        crate::backend_selection::DeviceType::Gpu => (1, "CPU_AND_GPU"),
        crate::backend_selection::DeviceType::Cpu => (0, "CPU_ONLY"),
    }
}

/// Load a CoreML model directly from protobuf bytes and retain it for repeated
/// dispatch, falling back to CPU-only if the preferred compute units fail.
pub(crate) fn compile_model(
    model_bytes: Vec<u8>,
    weights_data: Option<Vec<u8>>,
    device_type: crate::backend_selection::DeviceType,
    use_in_memory_asset: bool,
) -> Result<CompiledCoremlModel, GraphError> {
    // Owned so the in-memory path can hand the buffers to NSData without
    // copying (weight blobs reach hundreds of MB); the Arcs keep them valid
    // for the URL fallback below even after the NSData objects are released.
    let model_bytes = Arc::new(model_bytes);
    let weights_data = weights_data.map(Arc::new);
    if !use_in_memory_asset {
        return compile_model_from_url(
            &model_bytes,
            weights_data.as_deref().map(Vec::as_slice),
            device_type,
        );
    }
    let memory_result = autoreleasepool(|| unsafe {
        let (asset, specification_data, retained_weights_data) =
            create_in_memory_model_asset(&model_bytes, weights_data.as_ref())?;

        let (preferred_code, preferred_name) = compute_unit_for_device(device_type);
        let mut candidates: Vec<(i64, &'static str)> = vec![(preferred_code, preferred_name)];
        if preferred_code != 0 {
            candidates.push((0, "CPU_ONLY"));
        }

        let mut last_error = String::from("MLModel load failed");
        for (code, name) in candidates {
            let config: *mut Object = msg_send![class!(MLModelConfiguration), new];
            let _config_guard = ReleaseOnDrop(config);
            let () = msg_send![config, setComputeUnits: code];
            match load_model_asset(asset, config) {
                Ok(model) => {
                    return Ok(CompiledCoremlModel {
                        model,
                        compute_unit: name,
                        backing: CoremlModelBacking::InMemory {
                            asset,
                            specification_data,
                            weights_data: retained_weights_data,
                        },
                    });
                }
                Err(reason) => last_error = reason,
            }
        }

        let _: () = msg_send![asset, release];
        let _: () = msg_send![specification_data, release];
        if let Some(weights_data) = retained_weights_data {
            let _: () = msg_send![weights_data, release];
        }
        Err(GraphError::CoremlRuntimeFailed { reason: last_error })
    });
    memory_result.or_else(|memory_error| {
        compile_model_from_url(
            &model_bytes,
            weights_data.as_deref().map(Vec::as_slice),
            device_type,
        )
        .map_err(|url_error| GraphError::CoremlRuntimeFailed {
            reason: format!(
                "in-memory model load failed ({memory_error}); URL fallback failed ({url_error})"
            ),
        })
    })
}

fn compile_model_from_url(
    model_bytes: &[u8],
    weights_data: Option<&[u8]>,
    device_type: crate::backend_selection::DeviceType,
) -> Result<CompiledCoremlModel, GraphError> {
    autoreleasepool(|| unsafe {
        let (compiled_url, compiled_dir, temp_model) =
            prepare_compiled_model_with_weights(model_bytes, weights_data, None)?;
        // Owned (+1) by us; released when this function returns on any path.
        let _compiled_url_guard = ReleaseOnDrop(compiled_url);
        let (preferred_code, preferred_name) = compute_unit_for_device(device_type);
        let mut candidates = vec![(preferred_code, preferred_name)];
        if preferred_code != 0 {
            candidates.push((0, "CPU_ONLY"));
        }

        let mut last_error = String::from("MLModel load failed");
        for (code, name) in candidates {
            let config: *mut Object = msg_send![class!(MLModelConfiguration), new];
            let _config_guard = ReleaseOnDrop(config);
            let () = msg_send![config, setComputeUnits: code];
            let mut model: *mut Object = ptr::null_mut();
            let mut error = [0u8; 1024];
            let status = rustnn_coreml_load(
                compiled_url,
                config,
                &mut model,
                error.as_mut_ptr().cast(),
                error.len(),
            );
            if status != 0 || model.is_null() {
                last_error = format!("MLModel load failed: {}", shim_error_to_string(&error));
                continue;
            }
            // The shim returns an owned (+1) model; `CompiledCoremlModel`'s
            // Drop releases it.
            return Ok(CompiledCoremlModel {
                model,
                compute_unit: name,
                backing: CoremlModelBacking::OnDisk {
                    compiled_dir,
                    temp_model,
                },
            });
        }

        let _ = std::fs::remove_dir_all(&compiled_dir);
        // `temp_model` removes its temp path when dropped at end of scope.
        drop(temp_model);
        Err(GraphError::CoremlRuntimeFailed { reason: last_error })
    })
}

/// Wrap an `Arc`-owned buffer in an `NSData` WITHOUT copying. The NSData's
/// deallocator drops an `Arc` clone, so the bytes stay valid for as long as
/// EITHER the caller's `Arc` or the `NSData` lives — however long CoreML
/// internally retains the latter. Returns an owned (+1) object.
unsafe fn nsdata_from_arc(buffer: &Arc<Vec<u8>>) -> Result<*mut Object, GraphError> {
    let bytes = buffer.as_ptr() as *mut c_void;
    let length = buffer.len();
    let raw = Arc::into_raw(Arc::clone(buffer)) as usize;
    // NSData calls the deallocator exactly once, when its storage is freed.
    let deallocator = ConcreteBlock::new(move |_bytes: *mut c_void, _length: usize| {
        drop(unsafe { Arc::from_raw(raw as *const Vec<u8>) });
    })
    .copy();
    let data: *mut Object = msg_send![class!(NSData), alloc];
    let data: *mut Object = msg_send![data,
        initWithBytesNoCopy: bytes
        length: length
        deallocator: &*deallocator];
    if data.is_null() {
        // Never observed in practice; leak the Arc clone rather than risk a
        // double-free if the failed init already consumed the deallocator.
        return Err(GraphError::CoremlRuntimeFailed {
            reason: "failed to create NSData for CoreML model bytes".to_string(),
        });
    }
    Ok(data)
}

/// Create an `MLModelAsset` whose specification and optional external weight blob
/// are entirely memory-backed (zero-copy views over the given buffers). The
/// returned Objective-C objects are retained and must remain alive for at least
/// as long as the loaded `MLModel`.
unsafe fn create_in_memory_model_asset(
    model_bytes: &Arc<Vec<u8>>,
    weights: Option<&Arc<Vec<u8>>>,
) -> Result<(*mut Object, *mut Object, Option<*mut Object>), GraphError> {
    let Some(asset_class) = Class::get("MLModelAsset") else {
        return Err(GraphError::CoremlRuntimeFailed {
            reason: "in-memory CoreML model loading requires macOS 15 or newer".to_string(),
        });
    };

    let specification_data: *mut Object = unsafe { nsdata_from_arc(model_bytes)? };

    let mut error: *mut Object = ptr::null_mut();
    let (asset, weights_data): (*mut Object, Option<*mut Object>) = match weights {
        Some(weights) => {
            let weights_data: *mut Object = match unsafe { nsdata_from_arc(weights) } {
                Ok(data) => data,
                Err(err) => {
                    let _: () = msg_send![specification_data, release];
                    return Err(err);
                }
            };

            // BlobFileValue stores `@model_path/weights/weights.bin`. For an
            // in-memory asset CoreML expects the path relative to `@model_path`.
            let relative_path = unsafe { nsstring_from_str("weights/weights.bin")? };
            let blob_url: *mut Object = msg_send![class!(NSURL), fileURLWithPath: relative_path];
            let mapping: *mut Object = msg_send![class!(NSDictionary),
                dictionaryWithObject: weights_data forKey: blob_url];
            let asset: *mut Object = msg_send![asset_class,
                modelAssetWithSpecificationData: specification_data
                blobMapping: mapping
                error: &mut error];
            (asset, Some(weights_data))
        }
        None => {
            let asset: *mut Object = msg_send![asset_class,
                modelAssetWithSpecificationData: specification_data
                error: &mut error];
            (asset, None)
        }
    };

    if asset.is_null() {
        let reason = unsafe { ns_error_to_string(error, "MLModelAsset creation failed") };
        let _: () = msg_send![specification_data, release];
        if let Some(weights_data) = weights_data {
            let _: () = msg_send![weights_data, release];
        }
        return Err(GraphError::CoremlRuntimeFailed { reason });
    }
    let _: *mut Object = msg_send![asset, retain];
    Ok((asset, specification_data, weights_data))
}

/// Bridge CoreML's completion-handler API to this backend's synchronous graph
/// compilation contract. The callback retains the model before its autorelease
/// scope ends and transfers that ownership to the caller.
unsafe fn load_model_asset(
    asset: *mut Object,
    configuration: *mut Object,
) -> Result<*mut Object, String> {
    let (sender, receiver) = mpsc::sync_channel(1);
    let completion = ConcreteBlock::new(move |model: *mut Object, error: *mut Object| {
        let result = if model.is_null() {
            Err(unsafe { ns_error_to_string(error, "MLModelAsset load failed") })
        } else {
            unsafe {
                let _: *mut Object = msg_send![model, retain];
            }
            Ok(model as usize)
        };
        let _ = sender.send(result);
    })
    .copy();

    let (): () = msg_send![class!(MLModel),
        loadModelAsset: asset
        configuration: configuration
        completionHandler: &*completion];

    receiver
        .recv()
        .map_err(|_| "MLModelAsset load callback was dropped".to_string())?
        .map(|model| model as *mut Object)
}

/// Run a compiled CoreML model with raw-byte inputs, returning raw-byte outputs by name.
pub(crate) fn run_coreml_bytes(
    model: &CompiledCoremlModel,
    inputs: &HashMap<String, CoremlByteInput<'_>>,
    output_descriptors: &HashMap<String, OperandDescriptor>,
) -> Result<HashMap<String, Vec<u8>>, GraphError> {
    autoreleasepool(|| unsafe {
        let dict: *mut Object = msg_send![class!(NSMutableDictionary), dictionary];

        // Query the model's declared input types so the MLMultiArray we build matches
        // exactly what CoreML expects. Using our own dtype codes here is unsafe: an
        // array created with a code CoreML does not recognize (e.g. a bare `16` for
        // Float16) is treated as Float32, so CoreML reads past our 2-bytes-per-element
        // buffer -- garbage for small tensors, an out-of-bounds crash for large ones.
        let model_description: *mut Object = msg_send![model.model, modelDescription];
        let input_descs: *mut Object = msg_send![model_description, inputDescriptionsByName];

        for (name, input) in inputs {
            let key = nsstring_from_str(name)?;
            let mut shape_i64: Vec<i64> = input
                .descriptor
                .static_or_max_shape()
                .iter()
                .map(|&d| i64::from(d))
                .collect();
            if shape_i64.is_empty() {
                // Scalars are represented as a single-element 1-D array.
                shape_i64.push(1);
            }

            // Prefer the model's own data type code; fall back to our mapping only when
            // the model exposes no constraint for this input.
            let code = model_input_dtype_code(input_descs, key)
                .map_or_else(|| map_dtype(input.descriptor.data_type), Ok)?;
            let array = create_multi_array(&shape_i64, code)?;
            fill_multiarray_from_bytes(array, input.data, input.descriptor.data_type, code)?;
            let feature_value: *mut Object =
                msg_send![class!(MLFeatureValue), featureValueWithMultiArray: array];
            let () = msg_send![dict, setObject: feature_value forKey: key];
        }

        let mut create_error: *mut Object = ptr::null_mut();
        let provider_alloc: *mut Object = msg_send![class!(MLDictionaryFeatureProvider), alloc];
        let provider: *mut Object =
            msg_send![provider_alloc, initWithDictionary: dict error: &mut create_error];
        if provider.is_null() {
            return Err(GraphError::CoremlRuntimeFailed {
                reason: ns_error_to_string(create_error, "MLDictionaryFeatureProvider init failed"),
            });
        }
        let _provider_guard = ReleaseOnDrop(provider);

        let mut output_provider: *mut Object = ptr::null_mut();
        let mut error = [0u8; 1024];
        let status = rustnn_coreml_predict(
            model.model,
            provider,
            &mut output_provider,
            error.as_mut_ptr().cast(),
            error.len(),
        );
        if status != 0 || output_provider.is_null() {
            return Err(GraphError::CoremlRuntimeFailed {
                reason: format!("prediction failed: {}", shim_error_to_string(&error)),
            });
        }
        // `rustnn_coreml_predict` (coreml_shim.mm) hands back `output_provider`
        // already retained (`__bridge_retained`) -- we own this reference and
        // must release it once we're done extracting outputs.
        let _output_provider_guard = ReleaseOnDrop(output_provider);

        let mut result = HashMap::with_capacity(output_descriptors.len());
        for (name, descriptor) in output_descriptors {
            let key = nsstring_from_str(name)?;
            let value: *mut Object = msg_send![output_provider, featureValueForName: key];
            if value.is_null() {
                return Err(GraphError::CoremlRuntimeFailed {
                    reason: format!("model did not produce output `{name}`"),
                });
            }
            let array: *mut Object = msg_send![value, multiArrayValue];
            if array.is_null() {
                return Err(GraphError::CoremlRuntimeFailed {
                    reason: format!("output `{name}` is not a MLMultiArray"),
                });
            }
            let bytes = extract_multiarray_bytes(array, descriptor)?;
            result.insert(name.clone(), bytes);
        }
        Ok(result)
    })
}

/// Copy raw bytes into a freshly created `MLMultiArray`. The array must have been
/// created with the data type matching `dtype`, so the byte layout is identical.
unsafe fn fill_multiarray_from_bytes(
    array: *mut Object,
    src: &[u8],
    dtype: DataType,
    array_code: i32,
) -> Result<(), GraphError> {
    let count_obj: isize = msg_send![array, count];
    let count = usize::try_from(count_obj).map_err(|_| GraphError::CoremlRuntimeFailed {
        reason: format!("invalid element count: {count_obj}"),
    })?;
    // int4/uint4 inputs arrive packed two-per-byte and are exposed as float32 at the
    // model boundary; unpack the nibbles and write them into the float32 array.
    if matches!(dtype, DataType::Int4 | DataType::Uint4) {
        if count == 0 {
            return Ok(());
        }
        let floats: Vec<f32> = if matches!(dtype, DataType::Int4) {
            crate::graph::unpack_int4(src, count)
                .into_iter()
                .map(|v| v as f32)
                .collect()
        } else {
            crate::graph::unpack_uint4(src, count)
                .into_iter()
                .map(|v| v as f32)
                .collect()
        };
        let ptr: *mut c_void = msg_send![array, dataPointer];
        if ptr.is_null() {
            return Err(GraphError::CoremlRuntimeFailed {
                reason: format!("MLMultiArray has no backing buffer for data type {dtype:?}"),
            });
        }
        let dst = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, count * 4) };
        for (i, f) in floats.iter().enumerate() {
            dst[i * 4..i * 4 + 4].copy_from_slice(&f.to_le_bytes());
        }
        return Ok(());
    }
    let elem = dtype.bytes_per_element();
    let expected = count.saturating_mul(elem);
    if src.len() != expected {
        return Err(GraphError::CoremlRuntimeFailed {
            reason: format!(
                "input byte length mismatch: expected {expected} bytes ({count} elements), got {}",
                src.len()
            ),
        });
    }
    // When the model boundary promotes the input type (e.g. a WebNN uint8 boolean
    // condition exposed as Float32 by CoreML), convert the source bytes to the
    // required element size before writing.
    let canonical_code = normalize_dtype_code(array_code);
    // A different element size always needs conversion. So does a same-width
    // dtype mismatch: uint32 (4 bytes) promoted into an fp32 (4 bytes) array must
    // be converted by value, not copied as raw bits.
    let same_width_mismatch =
        canonical_code == 32 && !matches!(dtype, DataType::Float32 | DataType::Int32);
    if let Some(array_elem) = ml_dtype_code_element_size(canonical_code)
        && (array_elem != elem || same_width_mismatch)
    {
        if count == 0 {
            return Ok(());
        }
        let ptr: *mut c_void = msg_send![array, dataPointer];
        if ptr.is_null() {
            return Err(GraphError::CoremlRuntimeFailed {
                reason: format!("MLMultiArray has no backing buffer for data type {dtype:?}"),
            });
        }
        // Convert src bytes (dtype) → array bytes (canonical_code).
        let converted = convert_input_bytes(src, dtype, canonical_code, count);
        let dst = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, count * array_elem) };
        dst.copy_from_slice(&converted);
        return Ok(());
    }
    if expected == 0 {
        return Ok(());
    }
    let ptr: *mut c_void = msg_send![array, dataPointer];
    if ptr.is_null() {
        return Err(GraphError::CoremlRuntimeFailed {
            reason: format!("MLMultiArray has no backing buffer for data type {dtype:?}"),
        });
    }
    unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), ptr as *mut u8, expected) };
    Ok(())
}

/// Convert input bytes from `src_dtype` to a buffer compatible with `target_code`.
/// Used when CoreML promotes an input type (e.g. uint8 boolean → float32).
fn convert_input_bytes(src: &[u8], src_dtype: DataType, target_code: i32, count: usize) -> Vec<u8> {
    match (src_dtype, target_code) {
        (DataType::Uint8, 32) => {
            // uint8/bool → float32: 0→0.0, 1→1.0 (also used for boolean condition inputs)
            let mut out = Vec::with_capacity(count * 4);
            for &b in src.iter().take(count) {
                out.extend_from_slice(&(b as f32).to_le_bytes());
            }
            out
        }
        (DataType::Int8, 32) => {
            // int8 → float32: reinterpret the byte as signed so negatives are preserved
            // (a raw `b as f32` would turn -128 into 128.0).
            let mut out = Vec::with_capacity(count * 4);
            for &b in src.iter().take(count) {
                out.extend_from_slice(&((b as i8) as f32).to_le_bytes());
            }
            out
        }
        (DataType::Int32, 32) => {
            // int32 as float32 bits (reinterpret, same size — shouldn't normally happen)
            src.to_vec()
        }
        (DataType::Uint32, 32) => {
            // uint32 → float32 by value (the interface promotes uint32 to float32).
            let src_u32 = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u32, count) };
            let mut out = Vec::with_capacity(count * 4);
            for &v in src_u32 {
                out.extend_from_slice(&(v as f32).to_le_bytes());
            }
            out
        }
        (DataType::Uint64, 32) => {
            // uint64 → float32 by value (the interface promotes uint64 to float32).
            let src_u64 = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u64, count) };
            let mut out = Vec::with_capacity(count * 4);
            for &v in src_u64 {
                out.extend_from_slice(&(v as f32).to_le_bytes());
            }
            out
        }
        (DataType::Float32, 16) => {
            // float32 → float16
            let src_f32 = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, count) };
            let mut out = Vec::with_capacity(count * 2);
            for &v in src_f32 {
                out.extend_from_slice(&half::f16::from_f32(v).to_bits().to_le_bytes());
            }
            out
        }
        (DataType::Float16, 32) => {
            // float16 → float32
            let src_f16 = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u16, count) };
            let mut out = Vec::with_capacity(count * 4);
            for &bits in src_f16 {
                out.extend_from_slice(&half::f16::from_bits(bits).to_f32().to_le_bytes());
            }
            out
        }
        (DataType::Int64, 32) => {
            // int64 → float32: used for int64 inputs promoted to float32 by CoreML
            let src_i64 = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const i64, count) };
            let mut out = Vec::with_capacity(count * 4);
            for &v in src_i64 {
                out.extend_from_slice(&(v as f32).to_le_bytes());
            }
            out
        }
        _ => {
            // Fallback: pass bytes as-is (may be wrong but avoids panic)
            src.to_vec()
        }
    }
}

/// Extract a `MLMultiArray` into raw bytes laid out per the output descriptor's dtype.
///
/// Handles non-contiguous layouts (e.g. 64-byte aligned outputs from the Apple
/// Neural Engine) by gathering elements according to the array's strides.
/// Also handles dtype mismatches: CoreML may return a different dtype than requested
/// (e.g. float32 for a uint8-declared output from a comparison op). In that case the
/// data is converted element-by-element to the expected dtype.
unsafe fn extract_multiarray_bytes(
    array: *mut Object,
    descriptor: &OperandDescriptor,
) -> Result<Vec<u8>, GraphError> {
    let count_obj: isize = msg_send![array, count];
    let count = usize::try_from(count_obj).map_err(|_| GraphError::CoremlRuntimeFailed {
        reason: format!("invalid element count: {count_obj}"),
    })?;

    // Query the actual data type code from the MLMultiArray — CoreML may promote
    // the dtype (e.g. uint8 cast result returned as float32).
    let actual_dtype_code: i64 = msg_send![array, dataType];
    let actual_elem = ml_dtype_code_element_size(actual_dtype_code as i32).unwrap_or(4);

    // int4/uint4 outputs are produced as int32 (the proxy). Read the int32 values and
    // re-pack them two-per-byte into the sub-byte layout the host tensor expects.
    if matches!(descriptor.data_type, DataType::Int4 | DataType::Uint4) {
        let ptr: *const u8 = {
            let p: *mut c_void = msg_send![array, dataPointer];
            if p.is_null() {
                return Err(GraphError::CoremlRuntimeFailed {
                    reason: "output MLMultiArray has no backing buffer for int4/uint4".to_string(),
                });
            }
            p as *const u8
        };
        let shape_ns: *mut Object = msg_send![array, shape];
        let shape = unsafe { nsarray_to_i64_vec(shape_ns)? };
        let strides_ns: *mut Object = msg_send![array, strides];
        let strides = unsafe { nsarray_to_i64_vec(strides_ns)? };
        let raw = if is_contiguous(&shape, &strides) {
            unsafe { std::slice::from_raw_parts(ptr, count.saturating_mul(actual_elem)) }.to_vec()
        } else {
            unsafe { gather_strided_bytes(ptr, &shape, &strides, actual_elem) }
        };
        // Float codes (Float32 = 32/0x10020, Float16 = 16/0x10010) must be read as
        // floats; int codes (Int32 = 3/0x20020) as integers. NOTE: 0x20020 (131104) is
        // Int32, not Float32 — do not route it through normalize_dtype_code here.
        let is_float = matches!(actual_dtype_code as i32, 32 | 65568 | 16 | 65552);
        let ints: Vec<i32> = if is_float {
            raw.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()) as i32)
                .collect()
        } else {
            raw.chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        let packed = if matches!(descriptor.data_type, DataType::Int4) {
            crate::graph::pack_int4(&ints)
        } else {
            let u: Vec<u8> = ints.iter().map(|&v| v as u8).collect();
            crate::graph::pack_uint4(&u)
        };
        return Ok(packed);
    }

    let expected_elem = descriptor.data_type.bytes_per_element();

    let ptr: *const u8 = {
        let p: *mut c_void = msg_send![array, dataPointer];
        if p.is_null() {
            return Err(GraphError::CoremlRuntimeFailed {
                reason: format!(
                    "output MLMultiArray has no backing buffer for data type {:?}",
                    descriptor.data_type
                ),
            });
        }
        p as *const u8
    };

    let shape_nsarray: *mut Object = msg_send![array, shape];
    let shape = unsafe { nsarray_to_i64_vec(shape_nsarray)? };
    let strides_nsarray: *mut Object = msg_send![array, strides];
    let strides = unsafe { nsarray_to_i64_vec(strides_nsarray)? };

    // Read raw bytes using the ACTUAL element size from the MLMultiArray.
    let actual_bytes = if is_contiguous(&shape, &strides) {
        let total = count.saturating_mul(actual_elem);
        let slice = unsafe { std::slice::from_raw_parts(ptr, total) };
        slice.to_vec()
    } else {
        unsafe { gather_strided_bytes(ptr, &shape, &strides, actual_elem) }
    };

    if actual_elem == expected_elem {
        return Ok(actual_bytes);
    }

    // Dtype mismatch: convert from the actual dtype to the expected dtype.
    Ok(convert_multiarray_bytes(
        actual_bytes,
        actual_dtype_code as i32,
        descriptor.data_type,
    ))
}

/// Normalize non-standard MLMultiArrayDataType codes to canonical ones.
/// CoreML sometimes returns vendor-specific codes (e.g. 65568 for Float32).
fn normalize_dtype_code(code: i32) -> i32 {
    match code {
        65600 | 4 => 4,            // Int64 / Double → treat as Int64
        65568 | 131104 | 32 => 32, // Float32 variants
        65552 | 16 => 16,          // Float16 variants
        3 => 3,                    // Int32
        1 => 1,                    // Int8
        _ => code,
    }
}

/// Convert a byte buffer from `actual_code` dtype to the `target` dtype.
/// Used when CoreML promotes an output dtype (e.g. uint8 boolean result → float32).
fn convert_multiarray_bytes(actual_bytes: Vec<u8>, actual_code: i32, target: DataType) -> Vec<u8> {
    let canonical_code = normalize_dtype_code(actual_code);
    let actual_elem = ml_dtype_code_element_size(canonical_code).unwrap_or(4);
    let count = actual_bytes.len().checked_div(actual_elem).unwrap_or(0);

    match canonical_code {
        32 => {
            // Source: Float32
            let src =
                unsafe { std::slice::from_raw_parts(actual_bytes.as_ptr() as *const f32, count) };
            match target {
                DataType::Uint8 => src.iter().map(|&v| v as u8).collect(),
                // f32 -> int8: go through i8 so negatives keep their bit pattern
                // (`v as u8` saturates a negative float to 0).
                DataType::Int8 => src.iter().map(|&v| (v as i8) as u8).collect(),
                DataType::Int32 | DataType::Uint32 => {
                    let mut out = Vec::with_capacity(count * 4);
                    for &v in src {
                        out.extend_from_slice(&(v as i32).to_le_bytes());
                    }
                    out
                }
                DataType::Float16 => {
                    let mut out = Vec::with_capacity(count * 2);
                    for &v in src {
                        out.extend_from_slice(&half::f16::from_f32(v).to_bits().to_le_bytes());
                    }
                    out
                }
                _ => actual_bytes,
            }
        }
        16 => {
            // Source: Float16
            let src =
                unsafe { std::slice::from_raw_parts(actual_bytes.as_ptr() as *const u16, count) };
            match target {
                DataType::Uint8 => src
                    .iter()
                    .map(|&bits| half::f16::from_bits(bits).to_f32() as u8)
                    .collect(),
                // f16 -> int8: go through i8 so negatives keep their bit pattern.
                DataType::Int8 => src
                    .iter()
                    .map(|&bits| (half::f16::from_bits(bits).to_f32() as i8) as u8)
                    .collect(),
                DataType::Float32 => {
                    let mut out = Vec::with_capacity(count * 4);
                    for &bits in src {
                        out.extend_from_slice(&half::f16::from_bits(bits).to_f32().to_le_bytes());
                    }
                    out
                }
                _ => actual_bytes,
            }
        }
        3 => {
            // Source: Int32
            let src =
                unsafe { std::slice::from_raw_parts(actual_bytes.as_ptr() as *const i32, count) };
            match target {
                DataType::Float32 => {
                    let mut out = Vec::with_capacity(count * 4);
                    for &v in src {
                        out.extend_from_slice(&(v as f32).to_le_bytes());
                    }
                    out
                }
                DataType::Uint8 | DataType::Int8 => src.iter().map(|&v| v as u8).collect(),
                _ => actual_bytes,
            }
        }
        1 => {
            // Source: Int8 — compatible byte layout with Uint8
            actual_bytes
        }
        4 => {
            // Source: Int64
            let src =
                unsafe { std::slice::from_raw_parts(actual_bytes.as_ptr() as *const i64, count) };
            match target {
                DataType::Int32 | DataType::Uint32 => {
                    let mut out = Vec::with_capacity(count * 4);
                    for &v in src {
                        out.extend_from_slice(&(v as i32).to_le_bytes());
                    }
                    out
                }
                DataType::Float32 => {
                    let mut out = Vec::with_capacity(count * 4);
                    for &v in src {
                        out.extend_from_slice(&(v as f32).to_le_bytes());
                    }
                    out
                }
                DataType::Uint8 | DataType::Int8 => src.iter().map(|&v| v as u8).collect(),
                _ => actual_bytes,
            }
        }
        _ => actual_bytes,
    }
}

/// Whether `strides` (in elements) describe a C-contiguous layout for `shape`.
fn is_contiguous(shape: &[i64], strides: &[i64]) -> bool {
    if shape.len() != strides.len() {
        return false;
    }
    let mut expected = 1i64;
    for i in (0..shape.len()).rev() {
        if strides[i] != expected {
            return false;
        }
        expected *= shape[i].max(1);
    }
    true
}

/// Gather a strided `MLMultiArray` into a contiguous C-order byte buffer.
unsafe fn gather_strided_bytes(
    ptr: *const u8,
    shape: &[i64],
    strides: &[i64],
    elem: usize,
) -> Vec<u8> {
    let count: i64 = shape.iter().copied().map(|d| d.max(1)).product();
    let count = count.max(0) as usize;
    let mut out = Vec::with_capacity(count * elem);
    let mut idx = vec![0i64; shape.len()];
    for _ in 0..count {
        let mut offset_elems = 0i64;
        for d in 0..shape.len() {
            offset_elems += idx[d] * strides[d];
        }
        let byte_off = offset_elems as usize * elem;
        let slice = unsafe { std::slice::from_raw_parts(ptr.add(byte_off), elem) };
        out.extend_from_slice(slice);
        for d in (0..shape.len()).rev() {
            idx[d] += 1;
            if idx[d] < shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    out
}

#[allow(dead_code)]
fn run_impl_zeroed(
    model_bytes: &[u8],
    inputs: &HashMap<String, OperandDescriptor>,
    compiled_path: Option<&Path>,
) -> Result<Vec<CoremlRunAttempt>, GraphError> {
    run_impl_zeroed_with_weights(model_bytes, None, inputs, compiled_path)
}

fn run_impl_zeroed_with_weights(
    model_bytes: &[u8],
    weights_data: Option<&[u8]>,
    inputs: &HashMap<String, OperandDescriptor>,
    compiled_path: Option<&Path>,
) -> Result<Vec<CoremlRunAttempt>, GraphError> {
    unsafe {
        let (compiled_url, compiled_path_buf, temp_mlmodel) =
            prepare_compiled_model_with_weights(model_bytes, weights_data, compiled_path)?;
        // Owned (+1) by us; released when this function returns on any path.
        let _compiled_url_guard = ReleaseOnDrop(compiled_url);

        // Try only Neural Engine + GPU (best performance on Apple Silicon)
        // Fallback to ALL if that fails
        let targets = [
            (3i64, "CPU_AND_NE"), // CPU + Neural Engine (best for Apple Silicon)
            (2i64, "ALL"),        // All available compute units
            (0i64, "CPU_ONLY"),   // Guaranteed-to-load last resort
        ];
        let mut attempts = Vec::new();

        for (code, name) in targets {
            let config: *mut Object = msg_send![class!(MLModelConfiguration), new];
            let _config_guard = ReleaseOnDrop(config);
            let () = msg_send![config, setComputeUnits: code];
            let mut model: *mut Object = ptr::null_mut();
            let mut error = [0u8; 1024];
            let status = rustnn_coreml_load(
                compiled_url,
                config,
                &mut model,
                error.as_mut_ptr().cast(),
                error.len(),
            );
            if status != 0 || model.is_null() {
                attempts.push(CoremlRunAttempt {
                    compute_unit: name,
                    result: Err(format!(
                        "MLModel load failed: {}",
                        shim_error_to_string(&error)
                    )),
                });
                continue;
            }
            // Owned (+1) by us for this attempt; released at end of iteration.
            let _model_guard = ReleaseOnDrop(model);
            let model_description: *mut Object = msg_send![model, modelDescription];
            let input_descs: *mut Object = msg_send![model_description, inputDescriptionsByName];

            let dict: *mut Object = msg_send![class!(NSMutableDictionary), dictionary];
            let mut feature_err: Option<String> = None;
            for (name, descriptor) in inputs {
                let key = nsstring_from_str(name)?;
                let desc_obj: *mut Object = msg_send![input_descs, objectForKey: key];
                let (shape, data_type_code) = if desc_obj.is_null() {
                    (
                        coerce_shape(&descriptor.shape),
                        map_dtype(descriptor.data_type)?,
                    )
                } else {
                    let constraint_obj: *mut Object = msg_send![desc_obj, multiArrayConstraint];
                    if constraint_obj.is_null() {
                        (
                            coerce_shape(&descriptor.shape),
                            map_dtype(descriptor.data_type)?,
                        )
                    } else {
                        let shape_obj: *mut Object = msg_send![constraint_obj, shape];
                        let ml_data_type: i64 = msg_send![constraint_obj, dataType];
                        (nsarray_to_i64_vec(shape_obj)?, ml_data_type as i32)
                    }
                };

                let array = match create_multi_array(&shape, data_type_code) {
                    Ok(arr) => arr,
                    Err(err) => {
                        feature_err = Some(err.to_string());
                        break;
                    }
                };
                let fill_kind = data_type_from_code(data_type_code).unwrap_or(descriptor.data_type);
                if let Err(err) = fill_zero(array, fill_kind, &shape) {
                    feature_err = Some(err.to_string());
                    break;
                }
                let feature_value: *mut Object =
                    msg_send![class!(MLFeatureValue), featureValueWithMultiArray: array];
                let () = msg_send![dict, setObject: feature_value forKey: key];
            }

            if let Some(reason) = feature_err {
                attempts.push(CoremlRunAttempt {
                    compute_unit: name,
                    result: Err(reason),
                });
                continue;
            }

            let mut create_error: *mut Object = ptr::null_mut();
            let provider_alloc: *mut Object = msg_send![class!(MLDictionaryFeatureProvider), alloc];
            let provider: *mut Object =
                msg_send![provider_alloc, initWithDictionary: dict error: &mut create_error];
            if provider.is_null() {
                attempts.push(CoremlRunAttempt {
                    compute_unit: name,
                    result: Err(ns_error_to_string(
                        create_error,
                        "MLDictionaryFeatureProvider init failed",
                    )),
                });
                continue;
            }
            let _provider_guard = ReleaseOnDrop(provider);

            let mut output_provider: *mut Object = ptr::null_mut();
            let mut error = [0u8; 1024];
            let status = rustnn_coreml_predict(
                model,
                provider,
                &mut output_provider,
                error.as_mut_ptr().cast(),
                error.len(),
            );
            if status != 0 || output_provider.is_null() {
                attempts.push(CoremlRunAttempt {
                    compute_unit: name,
                    result: Err(format!(
                        "prediction failed: {}",
                        shim_error_to_string(&error)
                    )),
                });
                continue;
            }
            // The shim returns a retained provider; release it after collecting.
            let _output_provider_guard = ReleaseOnDrop(output_provider);

            match collect_outputs(output_provider) {
                Ok(outputs) => attempts.push(CoremlRunAttempt {
                    compute_unit: name,
                    result: Ok(outputs),
                }),
                Err(err) => attempts.push(CoremlRunAttempt {
                    compute_unit: name,
                    result: Err(err.to_string()),
                }),
            }
        }

        // The temporary source is removed when its handle drops.
        drop(temp_mlmodel);
        if compiled_path.is_none() {
            let _ = std::fs::remove_dir_all(&compiled_path_buf);
        }
        Ok(attempts)
    }
}

#[allow(dead_code)]
fn run_impl_with_inputs(
    model_bytes: &[u8],
    inputs: Vec<CoremlInput>,
    cache_path: Option<&Path>,
) -> Result<Vec<CoremlRunAttempt>, GraphError> {
    run_impl_with_inputs_with_weights(model_bytes, None, inputs, cache_path, None, None)
}

fn run_impl_with_inputs_with_weights(
    model_bytes: &[u8],
    weights_data: Option<&[u8]>,
    inputs: Vec<CoremlInput>,
    cache_path: Option<&Path>,
    input_descriptors: Option<&HashMap<String, OperandDescriptor>>,
    output_descriptors: Option<&HashMap<String, OperandDescriptor>>,
) -> Result<Vec<CoremlRunAttempt>, GraphError> {
    let mut runtime_shape_state = RuntimeShapeState::new();
    let mut actual_input_shapes = HashMap::new();
    for input in &inputs {
        validate_shape_data_length(&input.name, &input.shape, input.data.len())?;
        actual_input_shapes.insert(input.name.clone(), input.shape.clone());
    }
    if let Some(descriptors) = input_descriptors {
        runtime_shape_state.validate_named_shapes(
            &actual_input_shapes,
            descriptors,
            TensorKind::Input,
        )?;
    }

    unsafe {
        let (compiled_url, compiled_path_buf, temp_mlmodel) =
            prepare_compiled_model_with_weights(model_bytes, weights_data, cache_path)?;
        // Owned (+1) by us; released when this function returns on any path.
        let _compiled_url_guard = ReleaseOnDrop(compiled_url);

        // Try only Neural Engine + GPU (best performance on Apple Silicon)
        // Fallback to ALL if that fails
        let targets = [
            (3i64, "CPU_AND_NE"), // CPU + Neural Engine (best for Apple Silicon)
            (2i64, "ALL"),        // All available compute units
            (0i64, "CPU_ONLY"),   // Guaranteed-to-load last resort
        ];
        let mut attempts = Vec::new();

        for (code, name) in targets {
            let config: *mut Object = msg_send![class!(MLModelConfiguration), new];
            let _config_guard = ReleaseOnDrop(config);
            let () = msg_send![config, setComputeUnits: code];
            let mut model: *mut Object = ptr::null_mut();
            let mut error = [0u8; 1024];
            let status = rustnn_coreml_load(
                compiled_url,
                config,
                &mut model,
                error.as_mut_ptr().cast(),
                error.len(),
            );
            if status != 0 || model.is_null() {
                attempts.push(CoremlRunAttempt {
                    compute_unit: name,
                    result: Err(format!(
                        "MLModel load failed: {}",
                        shim_error_to_string(&error)
                    )),
                });
                continue;
            }
            // Owned (+1) by us for this attempt; released at end of iteration.
            let _model_guard = ReleaseOnDrop(model);

            // Get model input descriptions to query expected data types
            let model_description: *mut Object = msg_send![model, modelDescription];
            let input_descs: *mut Object = msg_send![model_description, inputDescriptionsByName];

            let dict: *mut Object = msg_send![class!(NSMutableDictionary), dictionary];
            let mut feature_err: Option<String> = None;

            // Create input features with actual data
            for input in &inputs {
                let key = nsstring_from_str(&input.name)?;
                let shape_i64: Vec<i64> = input.shape.iter().map(|&s| s as i64).collect();

                // Query model's expected data type for this input
                // Following Chromium's approach: match the model's expected type to avoid conversion errors
                let desc_obj: *mut Object = msg_send![input_descs, objectForKey: key];
                let data_type_code = if desc_obj.is_null() {
                    // No model info - default to Float32
                    32
                } else {
                    let constraint_obj: *mut Object = msg_send![desc_obj, multiArrayConstraint];
                    if constraint_obj.is_null() {
                        // No constraint - default to Float32
                        32
                    } else {
                        let ml_data_type: i64 = msg_send![constraint_obj, dataType];
                        ml_data_type as i32
                    }
                };

                // Create MLMultiArray with the model's expected data type
                let array = match create_multi_array(&shape_i64, data_type_code) {
                    Ok(arr) => arr,
                    Err(err) => {
                        feature_err = Some(err.to_string());
                        break;
                    }
                };

                // Fill with actual data, converting to the target type if needed
                if let Err(err) =
                    fill_data_with_type_conversion(array, &input.data, &shape_i64, data_type_code)
                {
                    feature_err = Some(err.to_string());
                    break;
                }

                let feature_value: *mut Object =
                    msg_send![class!(MLFeatureValue), featureValueWithMultiArray: array];
                let () = msg_send![dict, setObject: feature_value forKey: key];
            }

            if let Some(reason) = feature_err {
                attempts.push(CoremlRunAttempt {
                    compute_unit: name,
                    result: Err(reason),
                });
                continue;
            }

            let mut create_error: *mut Object = ptr::null_mut();
            let provider_alloc: *mut Object = msg_send![class!(MLDictionaryFeatureProvider), alloc];
            let provider: *mut Object =
                msg_send![provider_alloc, initWithDictionary: dict error: &mut create_error];
            if provider.is_null() {
                attempts.push(CoremlRunAttempt {
                    compute_unit: name,
                    result: Err(ns_error_to_string(
                        create_error,
                        "MLDictionaryFeatureProvider init failed",
                    )),
                });
                continue;
            }
            let _provider_guard = ReleaseOnDrop(provider);

            let mut output_provider: *mut Object = ptr::null_mut();
            let mut error = [0u8; 1024];
            let status = rustnn_coreml_predict(
                model,
                provider,
                &mut output_provider,
                error.as_mut_ptr().cast(),
                error.len(),
            );
            if status != 0 || output_provider.is_null() {
                attempts.push(CoremlRunAttempt {
                    compute_unit: name,
                    result: Err(format!(
                        "prediction failed: {}",
                        shim_error_to_string(&error)
                    )),
                });
                continue;
            }
            // The shim returns a retained provider; release it after collecting.
            let _output_provider_guard = ReleaseOnDrop(output_provider);

            match collect_outputs(output_provider) {
                Ok(outputs) => attempts.push(CoremlRunAttempt {
                    compute_unit: name,
                    result: Ok(outputs),
                }),
                Err(err) => attempts.push(CoremlRunAttempt {
                    compute_unit: name,
                    result: Err(err.to_string()),
                }),
            }
        }

        // The temporary source is removed when its handle drops.
        drop(temp_mlmodel);
        // Only delete compiled model if not cached
        if cache_path.is_none() {
            let _ = std::fs::remove_dir_all(&compiled_path_buf);
        }

        if let Some(descriptors) = output_descriptors {
            for attempt in &attempts {
                if let Ok(outputs) = &attempt.result {
                    let mut actual_output_shapes = HashMap::new();
                    for output in outputs {
                        let mut shape = Vec::with_capacity(output.shape.len());
                        for &dim in &output.shape {
                            let dim = usize::try_from(dim).map_err(|_| {
                                GraphError::CoremlRuntimeFailed {
                                    reason: format!(
                                        "output `{}` has invalid negative dimension {}",
                                        output.name, dim
                                    ),
                                }
                            })?;
                            shape.push(dim);
                        }
                        actual_output_shapes.insert(output.name.clone(), shape);
                    }
                    runtime_shape_state.validate_named_shapes(
                        &actual_output_shapes,
                        descriptors,
                        TensorKind::Output,
                    )?;
                }
            }
        }

        Ok(attempts)
    }
}

unsafe fn collect_outputs(provider: *mut Object) -> Result<Vec<CoremlOutput>, GraphError> {
    let feature_names: *mut Object = msg_send![provider, featureNames];
    let names_array: *mut Object = msg_send![feature_names, allObjects];
    let count: usize = msg_send![names_array, count];

    let mut outputs = Vec::new();
    for idx in 0..count {
        let name_obj: *mut Object = msg_send![names_array, objectAtIndex: idx];
        let rust_name = unsafe { nsstring_to_string(name_obj) };
        let value: *mut Object = msg_send![provider, featureValueForName: name_obj];
        let array: *mut Object = msg_send![value, multiArrayValue];
        if array.is_null() {
            return Err(GraphError::CoremlRuntimeFailed {
                reason: format!("output `{}` is not a MLMultiArray", rust_name),
            });
        }
        let data_type: i64 = msg_send![array, dataType];
        let shape_nsarray: *mut Object = msg_send![array, shape];
        let shape = unsafe { nsarray_to_i64_vec(shape_nsarray)? };

        // Extract actual data from MLMultiArray
        let data = unsafe { extract_mlmultiarray_data(array, data_type, &shape)? };

        outputs.push(CoremlOutput {
            name: rust_name,
            shape,
            data_type_code: data_type,
            data,
        });
    }
    Ok(outputs)
}

unsafe fn extract_mlmultiarray_data(
    array: *mut Object,
    data_type: i64,
    _shape: &[i64],
) -> Result<Vec<f32>, GraphError> {
    let count_obj: isize = msg_send![array, count];
    let count = usize::try_from(count_obj).map_err(|_| GraphError::CoremlRuntimeFailed {
        reason: format!("invalid element count: {}", count_obj),
    })?;

    let ptr: *mut std::os::raw::c_void = msg_send![array, dataPointer];

    // Convert to f32 regardless of source type
    let data = match data_type as i32 {
        32 | 65568 | 65552 => {
            // Float32 - codes: 32 (standard), 65568 (0x10020), 65552 (0x10010)
            // CoreML sometimes returns non-standard type codes for Float32
            let slice = unsafe { std::slice::from_raw_parts(ptr as *const f32, count) };
            slice.to_vec()
        }
        16 => {
            // Float16 - Must handle 64-byte aligned non-contiguous data from ANE
            // Following Chromium's approach: when Float16 executes on Apple Neural Engine,
            // outputs are 64-byte aligned and may be non-contiguous
            // Reference: chromium/src/+/5a3727be66 - Handle non-contiguous CoreML predictions

            // Get strides to detect non-contiguous data
            let strides_nsarray: *mut Object = msg_send![array, strides];
            let stride_count: usize = msg_send![strides_nsarray, count];

            if stride_count > 0 {
                // Get first stride value (bytes between elements)
                let stride_obj: *mut Object = msg_send![strides_nsarray, objectAtIndex: 0];
                let stride_value: isize = msg_send![stride_obj, integerValue];
                let stride_bytes = stride_value as usize;

                // If stride != 2 (size of f16), data is non-contiguous
                if stride_bytes != 2 {
                    // Non-contiguous: iterate with stride
                    let base_ptr = ptr as *const u8;
                    let mut result = Vec::with_capacity(count);
                    for i in 0..count {
                        let offset = i * stride_bytes;
                        let f16_ptr = unsafe { base_ptr.add(offset) as *const u16 };
                        let bits = unsafe { *f16_ptr };
                        result.push(half::f16::from_bits(bits).to_f32());
                    }
                    return Ok(result);
                }
            }

            // Contiguous data: use simple slice
            let slice = unsafe { std::slice::from_raw_parts(ptr as *const u16, count) };
            slice
                .iter()
                .map(|&bits| half::f16::from_bits(bits).to_f32())
                .collect()
        }
        3 => {
            // Int32
            let slice = unsafe { std::slice::from_raw_parts(ptr as *const i32, count) };
            slice.iter().map(|&x| x as f32).collect()
        }
        1 => {
            // Int8
            let slice = unsafe { std::slice::from_raw_parts(ptr as *const i8, count) };
            slice.iter().map(|&x| x as f32).collect()
        }
        _ => {
            // Try treating unknown types as Float32 (most common output type)
            // This is a fallback for non-standard CoreML type codes
            let slice = unsafe { std::slice::from_raw_parts(ptr as *const f32, count) };
            slice.to_vec()
        }
    };

    Ok(data)
}

#[allow(dead_code)]
unsafe fn prepare_compiled_model(
    model_bytes: &[u8],
    cached_compiled: Option<&Path>,
) -> Result<(*mut Object, PathBuf, Option<TempModelSource>), GraphError> {
    unsafe { prepare_compiled_model_with_weights(model_bytes, None, cached_compiled) }
}

/// Compile a model to a `.mlmodelc`, optionally persisting it to `cached_compiled`.
///
/// On success the returned NSURL is owned (+1); the caller must release it
/// (e.g. via [`ReleaseOnDrop`]) once the model has been loaded.
pub(crate) unsafe fn prepare_compiled_model_with_weights(
    model_bytes: &[u8],
    weights_data: Option<&[u8]>,
    cached_compiled: Option<&Path>,
) -> Result<(*mut Object, PathBuf, Option<TempModelSource>), GraphError> {
    let temp_mlmodel = write_temp_model_with_weights(model_bytes, weights_data)?;
    let url = unsafe { nsurl_from_path(temp_mlmodel.path())? };
    let mut compiled_url: *mut Object = ptr::null_mut();
    let mut error = [0u8; 1024];
    let status = unsafe {
        rustnn_coreml_compile(
            url,
            &mut compiled_url,
            error.as_mut_ptr().cast(),
            error.len(),
        )
    };
    if status != 0 || compiled_url.is_null() {
        return Err(GraphError::CoremlRuntimeFailed {
            reason: format!("MLModel compile failed: {}", shim_error_to_string(&error)),
        });
    }
    // The shim returns an owned (+1) URL; the guard releases it on every path
    // that doesn't hand it back to the caller.
    let compiled_url_guard = ReleaseOnDrop(compiled_url);

    let compiled_path_obj: *mut Object = msg_send![compiled_url, path];
    let compiled_src_path = PathBuf::from(unsafe { nsstring_to_string(compiled_path_obj) });

    if let Some(path) = cached_compiled {
        if path.exists() {
            let _ = std::fs::remove_dir_all(path);
        }
        let copy_result = copy_dir_recursively(&compiled_src_path, path);
        // Drop CoreML's temp .mlmodelc either way; the non-cached paths delete
        // it via their callers.
        let _ = std::fs::remove_dir_all(&compiled_src_path);
        if let Err(err) = copy_result {
            return Err(GraphError::CoremlRuntimeFailed {
                reason: format!("failed to persist compiled model: {}", err),
            });
        }
        let persisted_url = unsafe { nsurl_from_path(path)? };
        // `nsurl_from_path` returns an autoreleased URL; retain it so both
        // branches hand the caller an owned (+1) reference.
        let _: *mut Object = msg_send![persisted_url, retain];
        return Ok((persisted_url, path.to_path_buf(), Some(temp_mlmodel)));
    }

    Ok((
        compiled_url_guard.into_inner(),
        compiled_src_path,
        Some(temp_mlmodel),
    ))
}

#[allow(dead_code)]
fn write_temp_model(model_bytes: &[u8]) -> Result<TempModelSource, GraphError> {
    write_temp_model_with_weights(model_bytes, None)
}

/// An on-disk model source (`.mlmodel` file or `.mlpackage` directory) that CoreML
/// compiles from. The underlying temp path is created with a random, `O_EXCL`-guarded
/// name via `tempfile` and is deleted when this handle drops (RAII), so there is no
/// cross-process collision or symlink/TOCTOU exposure and cleanup survives panics.
pub(crate) enum TempModelSource {
    /// A `.mlpackage` directory (used when the model has external weights).
    Package(tempfile::TempDir),
    /// A single `.mlmodel` file.
    Model(tempfile::TempPath),
}

impl TempModelSource {
    fn path(&self) -> &Path {
        match self {
            TempModelSource::Package(dir) => dir.path(),
            TempModelSource::Model(path) => path,
        }
    }
}

/// Write a CoreML model to a temporary path, creating an `.mlpackage` when weights are
/// present. The returned [`TempModelSource`] owns the temp path and removes it on drop.
fn write_temp_model_with_weights(
    model_bytes: &[u8],
    weights_data: Option<&[u8]>,
) -> Result<TempModelSource, GraphError> {
    let map_io = |err: std::io::Error| GraphError::CoremlRuntimeFailed {
        reason: format!("failed to write temporary CoreML model: {err}"),
    };

    if let Some(weights) = weights_data {
        // Create .mlpackage directory structure with weights. The random directory
        // name keeps the `.mlpackage` suffix so CoreML recognizes it as a package.
        let package = tempfile::Builder::new()
            .prefix("rustnn_coreml_")
            .suffix(".mlpackage")
            .tempdir()
            .map_err(map_io)?;
        let package_path = package.path();
        let data_dir = package_path.join("Data").join("com.apple.CoreML");
        let weights_dir = data_dir.join("weights");

        // Create directories
        std::fs::create_dir_all(&weights_dir)
            .map_err(|err| GraphError::export(&weights_dir, err))?;

        // Write model.mlmodel (protobuf)
        let model_path = data_dir.join("model.mlmodel");
        std::fs::write(&model_path, model_bytes)
            .map_err(|err| GraphError::export(&model_path, err))?;

        // Write weights/weights.bin
        let weights_path = weights_dir.join("weights.bin");
        std::fs::write(&weights_path, weights)
            .map_err(|err| GraphError::export(&weights_path, err))?;

        // Write the package Manifest.json. CoreML refuses to load an .mlpackage
        // ("A valid manifest does not exist") without this root-level file
        // pointing at the model spec inside Data/.
        let manifest_path = package_path.join("Manifest.json");
        let model_id = "00000000-0000-0000-0000-0000000000AA";
        let weights_id = "00000000-0000-0000-0000-0000000000BB";
        let manifest = format!(
            r#"{{
  "fileFormatVersion": "1.0.0",
  "itemInfoEntries": {{
    "{model_id}": {{
      "author": "com.apple.CoreML",
      "description": "CoreML Model Specification",
      "name": "model.mlmodel",
      "path": "com.apple.CoreML/model.mlmodel"
    }},
    "{weights_id}": {{
      "author": "com.apple.CoreML",
      "description": "CoreML Model Weights",
      "name": "weights",
      "path": "com.apple.CoreML/weights"
    }}
  }},
  "rootModelIdentifier": "{model_id}"
}}
"#
        );
        std::fs::write(&manifest_path, manifest)
            .map_err(|err| GraphError::export(&manifest_path, err))?;

        Ok(TempModelSource::Package(package))
    } else {
        // No weights: write a single .mlmodel file with a random, exclusively-created
        // name. Convert to a closed `TempPath` so CoreML can open it by path.
        let mut file = tempfile::Builder::new()
            .prefix("rustnn_coreml_")
            .suffix(".mlmodel")
            .tempfile()
            .map_err(map_io)?;
        file.write_all(model_bytes).map_err(map_io)?;
        file.flush().map_err(map_io)?;
        Ok(TempModelSource::Model(file.into_temp_path()))
    }
}

fn coerce_shape(shape: &[Dimension]) -> Vec<i64> {
    let mut dims: Vec<i64> = shape
        .iter()
        .map(|d| i64::from(get_static_or_max_size(d)))
        .collect();
    match dims.len() {
        0 => vec![1],
        1 => dims,
        2 => {
            let mut with_batch = vec![1];
            with_batch.append(&mut dims);
            with_batch
        }
        3 => dims,
        _ => {
            let prod: i64 = dims.iter().product();
            vec![prod]
        }
    }
}

fn element_count(shape: &[i64]) -> Option<usize> {
    let mut count: i64 = 1;
    for dim in shape {
        count = count.checked_mul(*dim)?;
    }
    usize::try_from(count).ok()
}

fn map_dtype(data_type: DataType) -> Result<i32, GraphError> {
    // `MLMultiArrayDataType` enum values from Apple docs.
    let code = match data_type {
        DataType::Int4 | DataType::Uint4 => {
            return Err(GraphError::ConversionFailed {
                format: "coreml".to_string(),
                reason: "int4/uint4 tensors are not supported by CoreML runtime".to_string(),
            });
        }
        DataType::Float32 => 32, // MLMultiArrayDataTypeFloat32
        DataType::Float16 => 16, // MLMultiArrayDataTypeFloat16
        DataType::Int32 => 3,    // MLMultiArrayDataTypeInt32
        DataType::Int64 => 4,    // Closest available type
        DataType::Int8 => 1,     // MLMultiArrayDataTypeInt8
        DataType::Uint8 => 1,    // closest available signed byte type
        DataType::Uint32 => 3,   // closest available signed int type
        DataType::Uint64 => 4,   // closest available signed int type
    };
    Ok(code)
}

/// Element size in bytes for an `MLMultiArrayDataType` raw code, if known.
///
/// Covers Apple's canonical `0x1_0000`/`0x2_0000`-flagged values as well as the
/// legacy bare codes still used by [`map_dtype`].
fn ml_dtype_code_element_size(code: i32) -> Option<usize> {
    match code {
        65600 | 4 => Some(8),               // Double / Int64
        65568 | 131104 | 32 | 3 => Some(4), // Float32 / Int32
        65552 | 16 => Some(2),              // Float16
        1 => Some(1),                       // Int8
        _ => None,
    }
}

/// Query the model's declared `MLMultiArrayDataType` code for the input named `key`.
/// Returns `None` when the model exposes no multi-array constraint for that input.
unsafe fn model_input_dtype_code(input_descs: *mut Object, key: *mut Object) -> Option<i32> {
    if input_descs.is_null() {
        return None;
    }
    let desc_obj: *mut Object = msg_send![input_descs, objectForKey: key];
    if desc_obj.is_null() {
        return None;
    }
    let constraint_obj: *mut Object = msg_send![desc_obj, multiArrayConstraint];
    if constraint_obj.is_null() {
        return None;
    }
    let ml_data_type: i64 = msg_send![constraint_obj, dataType];
    Some(ml_data_type as i32)
}

fn data_type_from_code(code: i32) -> Option<DataType> {
    match code {
        32 => Some(DataType::Float32),
        16 => Some(DataType::Float16),
        3 => Some(DataType::Int32),
        1 => Some(DataType::Int8),
        _ => None,
    }
}

unsafe fn nsstring_from_str(value: &str) -> Result<*mut Object, GraphError> {
    let c_string = CString::new(value).map_err(|err| GraphError::CoremlRuntimeFailed {
        reason: format!("failed to build NSString: {err}"),
    })?;
    let obj: *mut Object = msg_send![class!(NSString), stringWithUTF8String: c_string.as_ptr()];
    Ok(obj)
}

unsafe fn nsurl_from_path(path: &Path) -> Result<*mut Object, GraphError> {
    let path_str = path
        .to_str()
        .ok_or_else(|| GraphError::CoremlRuntimeFailed {
            reason: format!("invalid path for CoreML model: {}", path.display()),
        })?;
    let ns_path = unsafe { nsstring_from_str(path_str)? };
    let url: *mut Object = msg_send![class!(NSURL), fileURLWithPath: ns_path];
    Ok(url)
}

unsafe fn create_multi_array(shape: &[i64], data_type: i32) -> Result<*mut Object, GraphError> {
    let numbers: Vec<*mut Object> = shape
        .iter()
        .map(|dim| {
            let number: *mut Object = msg_send![class!(NSNumber), numberWithLongLong: *dim];
            number
        })
        .collect();
    let nsarray: *mut Object =
        msg_send![class!(NSArray), arrayWithObjects: numbers.as_ptr() count: numbers.len()];
    let mut error: *mut Object = ptr::null_mut();
    let alloc: *mut Object = msg_send![class!(MLMultiArray), alloc];
    let array: *mut Object =
        msg_send![alloc, initWithShape: nsarray dataType: data_type error: &mut error];
    if array.is_null() {
        return Err(GraphError::CoremlRuntimeFailed {
            reason: unsafe { ns_error_to_string(error, "MLMultiArray init failed") },
        });
    }
    // Hand callers a pool-backed reference; MLFeatureValue retains the array
    // for as long as the feature dictionary needs it.
    let array: *mut Object = msg_send![array, autorelease];
    Ok(array)
}

unsafe fn fill_zero(
    array: *mut Object,
    data_type: DataType,
    shape: &[i64],
) -> Result<(), GraphError> {
    // Prefer the runtime-reported element count to avoid mismatches with coerced shapes.
    let count_obj: isize = msg_send![array, count];
    let count_from_runtime: Option<usize> = usize::try_from(count_obj).ok();
    let count_from_shape = element_count(shape);
    let Some(count) = count_from_runtime.or(count_from_shape) else {
        return Err(GraphError::CoremlRuntimeFailed {
            reason: format!("shape {:?} overflows element count", shape),
        });
    };
    let ptr: *mut c_void = msg_send![array, dataPointer];
    match data_type {
        DataType::Int4 | DataType::Uint4 => {
            return Err(GraphError::CoremlRuntimeFailed {
                reason: "int4/uint4 tensors are not supported by CoreML runtime".to_string(),
            });
        }
        DataType::Float32 => {
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut f32, count) };
            for v in slice.iter_mut() {
                *v = 0.0;
            }
        }
        DataType::Float16 => {
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u16, count) };
            for v in slice.iter_mut() {
                *v = 0;
            }
        }
        DataType::Int32 | DataType::Uint32 => {
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut i32, count) };
            for v in slice.iter_mut() {
                *v = 0;
            }
        }
        DataType::Int64 | DataType::Uint64 => {
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut i64, count) };
            for v in slice.iter_mut() {
                *v = 0;
            }
        }
        DataType::Int8 | DataType::Uint8 => {
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut i8, count) };
            for v in slice.iter_mut() {
                *v = 0;
            }
        }
    }
    Ok(())
}

#[allow(dead_code)]
unsafe fn fill_data(array: *mut Object, data: &[f32], _shape: &[i64]) -> Result<(), GraphError> {
    let count_obj: isize = msg_send![array, count];
    let count = usize::try_from(count_obj).map_err(|_| GraphError::CoremlRuntimeFailed {
        reason: format!("invalid element count: {}", count_obj),
    })?;

    if data.len() != count {
        return Err(GraphError::CoremlRuntimeFailed {
            reason: format!(
                "data size mismatch: expected {} elements but got {}",
                count,
                data.len()
            ),
        });
    }

    let ptr: *mut c_void = msg_send![array, dataPointer];
    let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut f32, count) };
    slice.copy_from_slice(data);

    Ok(())
}

/// Fill MLMultiArray with data, converting f32 input to target type if needed
/// Following Chromium's approach: match the model's expected data type
unsafe fn fill_data_with_type_conversion(
    array: *mut Object,
    data: &[f32],
    _shape: &[i64],
    data_type_code: i32,
) -> Result<(), GraphError> {
    let count_obj: isize = msg_send![array, count];
    let count = usize::try_from(count_obj).map_err(|_| GraphError::CoremlRuntimeFailed {
        reason: format!("invalid element count: {}", count_obj),
    })?;

    if data.len() != count {
        return Err(GraphError::CoremlRuntimeFailed {
            reason: format!(
                "data size mismatch: expected {} elements but got {}",
                count,
                data.len()
            ),
        });
    }

    let ptr: *mut c_void = msg_send![array, dataPointer];

    // Convert f32 data to target type based on data_type_code
    match data_type_code {
        32 | 65568 | 65552 => {
            // Float32 - codes: 32 (standard), 65568 (0x10020), 65552 (0x10010)
            // CoreML sometimes returns non-standard type codes for Float32
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut f32, count) };
            slice.copy_from_slice(data);
        }
        16 => {
            // Float16 - convert f32 to f16
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u16, count) };
            for (i, &val) in data.iter().enumerate() {
                slice[i] = half::f16::from_f32(val).to_bits();
            }
        }
        3 => {
            // Int32 - convert f32 to i32
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut i32, count) };
            for (i, &val) in data.iter().enumerate() {
                slice[i] = val as i32;
            }
        }
        1 => {
            // Int8 - convert f32 to i8
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut i8, count) };
            for (i, &val) in data.iter().enumerate() {
                slice[i] = val as i8;
            }
        }
        _ => {
            // Fallback: try treating unknown types as Float32 (most common output type)
            // This is a fallback for non-standard CoreML type codes
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut f32, count) };
            slice.copy_from_slice(data);
        }
    }

    Ok(())
}

unsafe fn nsarray_to_i64_vec(array: *mut Object) -> Result<Vec<i64>, GraphError> {
    let count: usize = msg_send![array, count];
    let mut result = Vec::with_capacity(count);
    for idx in 0..count {
        let obj: *mut Object = msg_send![array, objectAtIndex: idx];
        let value: i64 = msg_send![obj, longLongValue];
        result.push(value);
    }
    Ok(result)
}

unsafe fn nsstring_to_string(obj: *mut Object) -> String {
    let c_str: *const c_char = msg_send![obj, UTF8String];
    if c_str.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(c_str).to_string_lossy().into_owned() }
}

unsafe fn ns_error_to_string(error: *mut Object, default: &str) -> String {
    if error.is_null() {
        return default.to_string();
    }
    let desc: *mut Object = msg_send![error, localizedDescription];
    if desc.is_null() {
        return default.to_string();
    }
    unsafe { nsstring_to_string(desc) }
}

fn copy_dir_recursively(src: &Path, dst: &Path) -> std::io::Result<()> {
    if dst.exists() {
        std::fs::remove_dir_all(dst)?;
    }
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let dst_path = dst.join(entry.file_name());
        if file_type.is_dir() {
            copy_dir_recursively(&entry.path(), &dst_path)?;
        } else {
            std::fs::copy(entry.path(), dst_path)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod temp_model_tests {
    use super::{TempModelSource, write_temp_model_with_weights};

    #[test]
    fn bare_model_writes_mlmodel_file_and_cleans_up_on_drop() {
        let bytes = b"fake-mlmodel-protobuf";
        let source = write_temp_model_with_weights(bytes, None).expect("write temp model");
        let path = source.path().to_path_buf();

        assert!(matches!(&source, TempModelSource::Model(_)));
        assert!(path.is_file(), "expected a file at {path:?}");
        assert_eq!(path.extension().and_then(|e| e.to_str()), Some("mlmodel"));
        assert!(
            path.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("rustnn_coreml_")),
            "unexpected temp name: {path:?}"
        );
        assert_eq!(std::fs::read(&path).unwrap(), bytes);

        // Dropping the handle removes the temp file (RAII cleanup).
        drop(source);
        assert!(
            !path.exists(),
            "temp file should be removed on drop: {path:?}"
        );
    }

    #[test]
    fn model_with_weights_writes_mlpackage_and_cleans_up_on_drop() {
        let model = b"fake-mlmodel";
        let weights = b"\x00\x01\x02\x03weights-blob";
        let source =
            write_temp_model_with_weights(model, Some(weights)).expect("write temp package");
        let dir = source.path().to_path_buf();

        assert!(matches!(&source, TempModelSource::Package(_)));
        assert!(dir.is_dir(), "expected an .mlpackage dir at {dir:?}");
        assert_eq!(dir.extension().and_then(|e| e.to_str()), Some("mlpackage"));

        let model_path = dir.join("Data/com.apple.CoreML/model.mlmodel");
        let weights_path = dir.join("Data/com.apple.CoreML/weights/weights.bin");
        let manifest_path = dir.join("Manifest.json");
        assert_eq!(std::fs::read(&model_path).unwrap(), model);
        assert_eq!(std::fs::read(&weights_path).unwrap(), weights);
        let manifest = std::fs::read_to_string(&manifest_path).unwrap();
        assert!(manifest.contains("rootModelIdentifier"));
        assert!(manifest.contains("com.apple.CoreML/model.mlmodel"));

        // Dropping the handle removes the whole package directory (RAII cleanup).
        drop(source);
        assert!(
            !dir.exists(),
            "temp package should be removed on drop: {dir:?}"
        );
    }

    #[test]
    fn temp_paths_are_unique_across_calls() {
        // Randomized names must not collide even for identical content compiled
        // back-to-back (the old millisecond+counter scheme could).
        let a = write_temp_model_with_weights(b"same", None).unwrap();
        let b = write_temp_model_with_weights(b"same", None).unwrap();
        assert_ne!(a.path(), b.path(), "temp names must be unique");
    }
}
