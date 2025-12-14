//! Minimal CoreML execution bridge for macOS.
//! Loads a `.mlmodel`, compiles it if needed, and runs a zeroed inference
//! using CoreML's Objective-C API.

#![cfg(all(target_os = "macos", feature = "coreml-runtime"))]
#![allow(unexpected_cfgs)]

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::path::{Path, PathBuf};
use std::ptr;
use std::time::{SystemTime, UNIX_EPOCH};

use objc::rc::autoreleasepool;
use objc::runtime::Object;
use objc::{class, msg_send, sel, sel_impl};

use crate::error::GraphError;
use crate::graph::{DataType, OperandDescriptor};

// Link against the system frameworks we use.
#[link(name = "Foundation", kind = "framework")]
unsafe extern "C" {}
#[link(name = "CoreML", kind = "framework")]
unsafe extern "C" {}

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
    autoreleasepool(|| run_impl_with_inputs_with_weights(model_bytes, weights_data, inputs, None))
}

/// Run CoreML inference with actual input data and model caching
pub fn run_coreml_with_inputs_cached(
    model_bytes: &[u8],
    inputs: Vec<CoremlInput>,
    cache_path: Option<&Path>,
) -> Result<Vec<CoremlRunAttempt>, GraphError> {
    autoreleasepool(|| run_impl_with_inputs_with_weights(model_bytes, None, inputs, cache_path))
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

        // Try only Neural Engine + GPU (best performance on Apple Silicon)
        // Fallback to ALL if that fails
        let targets = [
            (3i64, "CPU_AND_NE"), // Neural Engine + GPU (best for Apple Silicon)
            (0i64, "ALL"),        // Fallback to all available compute units
        ];
        let mut attempts = Vec::new();

        for (code, name) in targets {
            let config: *mut Object = msg_send![class!(MLModelConfiguration), new];
            let () = msg_send![config, setComputeUnits: code];
            let mut error: *mut Object = ptr::null_mut();
            let model: *mut Object = msg_send![class!(MLModel), modelWithContentsOfURL: compiled_url configuration: config error: &mut error];
            if model.is_null() {
                attempts.push(CoremlRunAttempt {
                    compute_unit: name,
                    result: Err(ns_error_to_string(error, "MLModel load failed")),
                });
                continue;
            }

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

            let mut predict_error: *mut Object = ptr::null_mut();
            let output_provider: *mut Object =
                msg_send![model, predictionFromFeatures: provider error: &mut predict_error];
            if output_provider.is_null() {
                attempts.push(CoremlRunAttempt {
                    compute_unit: name,
                    result: Err(ns_error_to_string(predict_error, "prediction failed")),
                });
                continue;
            }

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

        if let Some(tmp) = temp_mlmodel {
            let _ = std::fs::remove_file(&tmp);
        }
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
    run_impl_with_inputs_with_weights(model_bytes, None, inputs, cache_path)
}

fn run_impl_with_inputs_with_weights(
    model_bytes: &[u8],
    weights_data: Option<&[u8]>,
    inputs: Vec<CoremlInput>,
    cache_path: Option<&Path>,
) -> Result<Vec<CoremlRunAttempt>, GraphError> {
    unsafe {
        let (compiled_url, compiled_path_buf, temp_mlmodel) =
            prepare_compiled_model_with_weights(model_bytes, weights_data, cache_path)?;

        // Try only Neural Engine + GPU (best performance on Apple Silicon)
        // Fallback to ALL if that fails
        let targets = [
            (3i64, "CPU_AND_NE"), // Neural Engine + GPU (best for Apple Silicon)
            (0i64, "ALL"),        // Fallback to all available compute units
        ];
        let mut attempts = Vec::new();

        for (code, name) in targets {
            let config: *mut Object = msg_send![class!(MLModelConfiguration), new];
            let () = msg_send![config, setComputeUnits: code];
            let mut error: *mut Object = ptr::null_mut();
            let model: *mut Object = msg_send![class!(MLModel), modelWithContentsOfURL: compiled_url configuration: config error: &mut error];
            if model.is_null() {
                attempts.push(CoremlRunAttempt {
                    compute_unit: name,
                    result: Err(ns_error_to_string(error, "MLModel load failed")),
                });
                continue;
            }

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

            let mut predict_error: *mut Object = ptr::null_mut();
            let output_provider: *mut Object =
                msg_send![model, predictionFromFeatures: provider error: &mut predict_error];
            if output_provider.is_null() {
                attempts.push(CoremlRunAttempt {
                    compute_unit: name,
                    result: Err(ns_error_to_string(predict_error, "prediction failed")),
                });
                continue;
            }

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

        if let Some(tmp) = temp_mlmodel {
            let _ = std::fs::remove_file(&tmp);
        }
        // Only delete compiled model if not cached
        if cache_path.is_none() {
            let _ = std::fs::remove_dir_all(&compiled_path_buf);
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
) -> Result<(*mut Object, PathBuf, Option<PathBuf>), GraphError> {
    unsafe { prepare_compiled_model_with_weights(model_bytes, None, cached_compiled) }
}

unsafe fn prepare_compiled_model_with_weights(
    model_bytes: &[u8],
    weights_data: Option<&[u8]>,
    cached_compiled: Option<&Path>,
) -> Result<(*mut Object, PathBuf, Option<PathBuf>), GraphError> {
    let temp_mlmodel = write_temp_model_with_weights(model_bytes, weights_data)?;
    let url = unsafe { nsurl_from_path(&temp_mlmodel)? };
    let mut compile_error: *mut Object = ptr::null_mut();
    let compiled_url: *mut Object =
        msg_send![class!(MLModel), compileModelAtURL: url error: &mut compile_error];
    if compiled_url.is_null() {
        return Err(GraphError::CoremlRuntimeFailed {
            reason: unsafe { ns_error_to_string(compile_error, "MLModel compile failed") },
        });
    }

    let compiled_path_obj: *mut Object = msg_send![compiled_url, path];
    let compiled_src_path = PathBuf::from(unsafe { nsstring_to_string(compiled_path_obj) });

    if let Some(path) = cached_compiled {
        if path.exists() {
            let _ = std::fs::remove_dir_all(path);
        }
        if let Err(err) = copy_dir_recursively(&compiled_src_path, path) {
            return Err(GraphError::CoremlRuntimeFailed {
                reason: format!("failed to persist compiled model: {}", err),
            });
        }
        let persisted_url = unsafe { nsurl_from_path(path)? };
        return Ok((persisted_url, path.to_path_buf(), Some(temp_mlmodel)));
    }

    Ok((compiled_url, compiled_src_path, Some(temp_mlmodel)))
}

#[allow(dead_code)]
fn write_temp_model(model_bytes: &[u8]) -> Result<PathBuf, GraphError> {
    write_temp_model_with_weights(model_bytes, None)
}

/// Write a CoreML model to a temporary file, optionally creating an .mlpackage with weights
fn write_temp_model_with_weights(
    model_bytes: &[u8],
    weights_data: Option<&[u8]>,
) -> Result<PathBuf, GraphError> {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();

    if let Some(weights) = weights_data {
        // Create .mlpackage directory structure with weights
        let package_path = std::env::temp_dir().join(format!("rustnn_coreml_{ts}.mlpackage"));
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

        Ok(package_path)
    } else {
        // No weights: write single .mlmodel file as before
        let path = std::env::temp_dir().join(format!("rustnn_coreml_{ts}.mlmodel"));
        std::fs::write(&path, model_bytes).map_err(|err| GraphError::export(&path, err))?;
        Ok(path)
    }
}

fn coerce_shape(shape: &[u32]) -> Vec<i64> {
    let mut dims: Vec<i64> = shape.iter().map(|d| *d as i64).collect();
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
        DataType::Float32 => 32, // MLMultiArrayDataTypeFloat32
        DataType::Float16 => 16, // MLMultiArrayDataTypeFloat16
        DataType::Int32 => 3,    // MLMultiArrayDataTypeInt32
        DataType::Int64 => 4,    // Closest available type
        DataType::Int8 => 1,     // MLMultiArrayDataTypeInt8
        DataType::Uint8 => 1,    // closest available signed byte type
        DataType::Uint32 => 3,   // closest available signed int type
    };
    Ok(code)
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
        DataType::Int64 => {
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
