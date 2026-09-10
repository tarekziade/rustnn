/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Shubham Gupta <shubhamg13.work@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! CANN device test -- exercises the full rustnn CANN pipeline on a real
//! OHOS device with Kirin NPU.
//!
//! These tests only run on the OpenHarmony target (they are compiled out
//! everywhere else) and require the `cann-runtime` feature plus `CANN_DDK`.
//! Use `make cann-device-test` to cross-compile, push, and run them on device.

#[cfg(all(feature = "cann-runtime", target_env = "ohos"))]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::{LazyLock, Mutex};

    use rustnn::backend_selection::Backend;
    use rustnn::mlcontext::{
        MLContext, MLContextOptions, MLOperand, MLOperandDescriptor, MLPowerPreference,
        MLTensorDescriptor,
    };
    use rustnn::mlgraphbuilder::MLGraphBuilder;
    use rustnn::operator_enums::MLOperandDataType;
    use rustnn::operator_options::{
        MLConv2dOptions, MLDimension, MLPool2dOptions, MLReduceOptions, MLResample2dOptions,
        MLTransposeOptions,
    };

    /// Single shared context, created lazily on first use and reused by every
    /// test (context creation is independent of model load, so one context is
    /// enough). Wrapped in a `Mutex` so tests can take `&mut` access from the
    /// static while remaining `Sync` for the libtest harness.
    static CONTEXT: LazyLock<Mutex<MLContext<'static>>> = LazyLock::new(|| {
        let options = MLContextOptions::new(MLPowerPreference::Default, true)
            .with_rustnn_backend_hint(Backend::Cann);
        let context = MLContext::create(&options).expect("failed to create CANN context");
        Mutex::new(context)
    });

    #[test]
    fn test_add() {
        let mut context = CONTEXT.lock().unwrap();

        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let desc_2x2 = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![2, 2]);
        let a: MLOperand = builder.input("a", &desc_2x2).unwrap();
        let b: MLOperand = builder.input("b", &desc_2x2).unwrap();
        let sum: MLOperand = builder.add(a, b).unwrap();
        let mut graph = builder.build(&BTreeMap::from([("sum", sum)])).unwrap();

        let tdesc = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2, 2])
            .to_writable()
            .to_readable();
        let tensor_a = context.create_tensor(&tdesc).unwrap();
        let tensor_b = context.create_tensor(&tdesc).unwrap();
        let tensor_out = context.create_tensor(&tdesc).unwrap();

        context
            .write_tensor(&tensor_a, &vec![1.0f32, 2.0, 3.0, 4.0])
            .unwrap();
        context
            .write_tensor(&tensor_b, &vec![5.0f32, 6.0, 7.0, 8.0])
            .unwrap();
        context
            .dispatch(
                &mut graph,
                &BTreeMap::from([("a", &tensor_a), ("b", &tensor_b)]),
                &BTreeMap::from([("sum", &tensor_out)]),
            )
            .unwrap();

        let mut out_data = vec![0.0f32; 4];
        context.read_tensor(&tensor_out, &mut out_data).unwrap();
        println!("  Add([1,2,3,4], [5,6,7,8]) = {:?}", out_data);
        assert_eq!(out_data, [6.0f32, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_sub() {
        let mut context = CONTEXT.lock().unwrap();

        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let desc_2x2 = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![2, 2]);
        let a: MLOperand = builder.input("a", &desc_2x2).unwrap();
        let b: MLOperand = builder.input("b", &desc_2x2).unwrap();
        let output: MLOperand = builder.sub(a, b).unwrap();
        let mut graph = builder.build(&BTreeMap::from([("y", output)])).unwrap();

        let tdesc = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2, 2])
            .to_writable()
            .to_readable();
        let tensor_a = context.create_tensor(&tdesc).unwrap();
        let tensor_b = context.create_tensor(&tdesc).unwrap();
        let tensor_out = context.create_tensor(&tdesc).unwrap();

        context
            .write_tensor(&tensor_a, &vec![5.0f32, 6.0, 7.0, 8.0])
            .unwrap();
        context
            .write_tensor(&tensor_b, &vec![1.0f32, 2.0, 3.0, 4.0])
            .unwrap();
        context
            .dispatch(
                &mut graph,
                &BTreeMap::from([("a", &tensor_a), ("b", &tensor_b)]),
                &BTreeMap::from([("y", &tensor_out)]),
            )
            .unwrap();

        let mut out_data = vec![0.0f32; 4];
        context.read_tensor(&tensor_out, &mut out_data).unwrap();
        println!("  Sub([5,6,7,8], [1,2,3,4]) = {:?}", out_data);
        assert_eq!(out_data, [4.0f32, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_mul() {
        let mut context = CONTEXT.lock().unwrap();

        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let desc_2x2 = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![2, 2]);
        let a: MLOperand = builder.input("a", &desc_2x2).unwrap();
        let b: MLOperand = builder.input("b", &desc_2x2).unwrap();
        let output: MLOperand = builder.mul(a, b).unwrap();
        let mut graph = builder.build(&BTreeMap::from([("y", output)])).unwrap();

        let tdesc = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2, 2])
            .to_writable()
            .to_readable();
        let tensor_a = context.create_tensor(&tdesc).unwrap();
        let tensor_b = context.create_tensor(&tdesc).unwrap();
        let tensor_out = context.create_tensor(&tdesc).unwrap();

        context
            .write_tensor(&tensor_a, &vec![1.0f32, 2.0, 3.0, 4.0])
            .unwrap();
        context
            .write_tensor(&tensor_b, &vec![2.0f32, 2.0, 2.0, 2.0])
            .unwrap();
        context
            .dispatch(
                &mut graph,
                &BTreeMap::from([("a", &tensor_a), ("b", &tensor_b)]),
                &BTreeMap::from([("y", &tensor_out)]),
            )
            .unwrap();

        let mut out_data = vec![0.0f32; 4];
        context.read_tensor(&tensor_out, &mut out_data).unwrap();
        println!("  Mul([1,2,3,4], [2,2,2,2]) = {:?}", out_data);
        assert_eq!(out_data, [2.0f32, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_conv2d() {
        let mut context = CONTEXT.lock().unwrap();

        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let input_desc = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![1, 1, 3, 3]);
        let filter_desc = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![1, 1, 2, 2]);
        let input: MLOperand = builder.input("x", &input_desc).unwrap();
        let filter: MLOperand = builder
            .constant_from_vec(&filter_desc, vec![1.0f32, 0.0, 0.0, 1.0])
            .unwrap();
        let output: MLOperand = builder.conv2d(input, filter).unwrap();
        let mut graph = builder.build(&BTreeMap::from([("y", output)])).unwrap();

        let tdesc_input = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![1, 1, 3, 3])
            .to_writable()
            .to_readable();
        let tdesc_output =
            MLTensorDescriptor::new(MLOperandDataType::Float32, vec![1, 1, 2, 2]).to_readable();
        let tensor_input = context.create_tensor(&tdesc_input).unwrap();
        let tensor_output = context.create_tensor(&tdesc_output).unwrap();

        context
            .write_tensor(
                &tensor_input,
                &vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            )
            .unwrap();
        context
            .dispatch(
                &mut graph,
                &BTreeMap::from([("x", &tensor_input)]),
                &BTreeMap::from([("y", &tensor_output)]),
            )
            .unwrap();

        let mut out_data = vec![0.0f32; 4];
        context.read_tensor(&tensor_output, &mut out_data).unwrap();
        println!("  Conv2d(3x3, 2x2 identity) = {:?}", out_data);
        assert_eq!(out_data, [6.0f32, 8.0, 12.0, 14.0]);
    }

    #[test]
    fn test_max_pool2d() {
        let mut context = CONTEXT.lock().unwrap();

        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let input_desc = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![1, 1, 4, 4]);
        let input: MLOperand = builder.input("x", &input_desc).unwrap();
        let pool_options = MLPool2dOptions {
            window_dimensions: Some(vec![2, 2]),
            strides: vec![2, 2],
            ..MLPool2dOptions::default()
        };
        let output: MLOperand = builder
            .max_pool2d_with_options(input, pool_options)
            .unwrap();
        let mut graph = builder.build(&BTreeMap::from([("y", output)])).unwrap();

        let tdesc_input = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![1, 1, 4, 4])
            .to_writable()
            .to_readable();
        let tdesc_output =
            MLTensorDescriptor::new(MLOperandDataType::Float32, vec![1, 1, 2, 2]).to_readable();
        let tensor_input = context.create_tensor(&tdesc_input).unwrap();
        let tensor_output = context.create_tensor(&tdesc_output).unwrap();

        let input_data = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ];
        context.write_tensor(&tensor_input, &input_data).unwrap();
        context
            .dispatch(
                &mut graph,
                &BTreeMap::from([("x", &tensor_input)]),
                &BTreeMap::from([("y", &tensor_output)]),
            )
            .unwrap();

        let mut out_data = vec![0.0f32; 4];
        context.read_tensor(&tensor_output, &mut out_data).unwrap();
        println!("  MaxPool2d(4x4, 2x2, stride 2) = {:?}", out_data);
        assert_eq!(out_data, [6.0f32, 8.0, 14.0, 16.0]);
    }

    #[test]
    fn test_concat() {
        let mut context = CONTEXT.lock().unwrap();

        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let desc_2 = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![2]);
        let a: MLOperand = builder.input("a", &desc_2).unwrap();
        let b: MLOperand = builder.input("b", &desc_2).unwrap();
        let output: MLOperand = builder.concat(&[a, b], 0).unwrap();
        let mut graph = builder.build(&BTreeMap::from([("y", output)])).unwrap();

        let tdesc_in = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2])
            .to_writable()
            .to_readable();
        let tdesc_out = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![4]).to_readable();
        let tensor_a = context.create_tensor(&tdesc_in).unwrap();
        let tensor_b = context.create_tensor(&tdesc_in).unwrap();
        let tensor_out = context.create_tensor(&tdesc_out).unwrap();

        context.write_tensor(&tensor_a, &vec![1.0f32, 2.0]).unwrap();
        context.write_tensor(&tensor_b, &vec![3.0f32, 4.0]).unwrap();
        context
            .dispatch(
                &mut graph,
                &BTreeMap::from([("a", &tensor_a), ("b", &tensor_b)]),
                &BTreeMap::from([("y", &tensor_out)]),
            )
            .unwrap();

        let mut out_data = vec![0.0f32; 4];
        context.read_tensor(&tensor_out, &mut out_data).unwrap();
        println!("  Concat([1,2], [3,4], axis=0) = {:?}", out_data);
        assert_eq!(out_data, [1.0f32, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_reshape() {
        let mut context = CONTEXT.lock().unwrap();

        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let desc_4 = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![4]);
        let input: MLOperand = builder.input("x", &desc_4).unwrap();
        let output: MLOperand = builder
            .reshape(input, vec![MLDimension::Static(2), MLDimension::Static(2)])
            .unwrap();
        let mut graph = builder.build(&BTreeMap::from([("y", output)])).unwrap();

        let tdesc_in = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![4])
            .to_writable()
            .to_readable();
        let tdesc_out =
            MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2, 2]).to_readable();
        let tensor_in = context.create_tensor(&tdesc_in).unwrap();
        let tensor_out = context.create_tensor(&tdesc_out).unwrap();

        context
            .write_tensor(&tensor_in, &vec![1.0f32, 2.0, 3.0, 4.0])
            .unwrap();
        context
            .dispatch(
                &mut graph,
                &BTreeMap::from([("x", &tensor_in)]),
                &BTreeMap::from([("y", &tensor_out)]),
            )
            .unwrap();

        let mut out_data = vec![0.0f32; 4];
        context.read_tensor(&tensor_out, &mut out_data).unwrap();
        println!("  Reshape([1,2,3,4], [2,2]) = {:?}", out_data);
        assert_eq!(out_data, [1.0f32, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_resample2d_nearest() {
        let mut context = CONTEXT.lock().unwrap();

        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let input_desc = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![1, 1, 1, 1]);
        let input: MLOperand = builder.input("x", &input_desc).unwrap();
        let resample_opts = MLResample2dOptions {
            mode: "nearest-neighbor".to_string(),
            sizes: Some(vec![2, 2]),
            ..MLResample2dOptions::default()
        };
        let output: MLOperand = builder
            .resample2d_with_options(input, resample_opts)
            .unwrap();
        let mut graph = builder.build(&BTreeMap::from([("y", output)])).unwrap();

        let tdesc_in = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![1, 1, 1, 1])
            .to_writable()
            .to_readable();
        let tdesc_out =
            MLTensorDescriptor::new(MLOperandDataType::Float32, vec![1, 1, 2, 2]).to_readable();
        let tensor_in = context.create_tensor(&tdesc_in).unwrap();
        let tensor_out = context.create_tensor(&tdesc_out).unwrap();

        context.write_tensor(&tensor_in, &vec![5.0f32]).unwrap();
        context
            .dispatch(
                &mut graph,
                &BTreeMap::from([("x", &tensor_in)]),
                &BTreeMap::from([("y", &tensor_out)]),
            )
            .unwrap();

        let mut out_data = vec![0.0f32; 4];
        context.read_tensor(&tensor_out, &mut out_data).unwrap();
        println!("  Resample2d(1x1 -> 2x2, nearest) = {:?}", out_data);
        assert_eq!(out_data, [5.0f32, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_sigmoid() {
        let mut context = CONTEXT.lock().unwrap();

        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let desc_1 = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![1]);
        let input: MLOperand = builder.input("x", &desc_1).unwrap();
        let output: MLOperand = builder.sigmoid(input).unwrap();
        let mut graph = builder.build(&BTreeMap::from([("y", output)])).unwrap();

        let tdesc = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![1])
            .to_writable()
            .to_readable();
        let tensor_in = context.create_tensor(&tdesc).unwrap();
        let tensor_out = context.create_tensor(&tdesc).unwrap();

        context.write_tensor(&tensor_in, &vec![0.0f32]).unwrap();
        context
            .dispatch(
                &mut graph,
                &BTreeMap::from([("x", &tensor_in)]),
                &BTreeMap::from([("y", &tensor_out)]),
            )
            .unwrap();

        let mut out_data = vec![0.0f32; 1];
        context.read_tensor(&tensor_out, &mut out_data).unwrap();
        println!("  Sigmoid([0]) = {:?}", out_data);
        assert!(
            (out_data[0] - 0.5).abs() < 1e-2,
            "sigmoid(0) should be ~0.5, got {}",
            out_data[0]
        );
    }

    #[test]
    fn test_slice() {
        let mut context = CONTEXT.lock().unwrap();

        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let desc_4 = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![4]);
        let input: MLOperand = builder.input("x", &desc_4).unwrap();
        let output: MLOperand = builder
            .slice(input, &[1], &[MLDimension::Static(2)])
            .unwrap();
        let mut graph = builder.build(&BTreeMap::from([("y", output)])).unwrap();

        let tdesc_in = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![4])
            .to_writable()
            .to_readable();
        let tdesc_out = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2]).to_readable();
        let tensor_in = context.create_tensor(&tdesc_in).unwrap();
        let tensor_out = context.create_tensor(&tdesc_out).unwrap();

        context
            .write_tensor(&tensor_in, &vec![1.0f32, 2.0, 3.0, 4.0])
            .unwrap();
        context
            .dispatch(
                &mut graph,
                &BTreeMap::from([("x", &tensor_in)]),
                &BTreeMap::from([("y", &tensor_out)]),
            )
            .unwrap();

        let mut out_data = vec![0.0f32; 2];
        context.read_tensor(&tensor_out, &mut out_data).unwrap();
        println!("  Slice([1,2,3,4], start=1, size=2) = {:?}", out_data);
        assert_eq!(out_data, [2.0f32, 3.0]);
    }

    #[test]
    fn test_softmax() {
        let mut context = CONTEXT.lock().unwrap();

        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let desc_2 = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![2]);
        let input: MLOperand = builder.input("x", &desc_2).unwrap();
        let output: MLOperand = builder.softmax(input, 0).unwrap();
        let mut graph = builder.build(&BTreeMap::from([("y", output)])).unwrap();

        let tdesc = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2])
            .to_writable()
            .to_readable();
        let tensor_in = context.create_tensor(&tdesc).unwrap();
        let tensor_out = context.create_tensor(&tdesc).unwrap();

        context
            .write_tensor(&tensor_in, &vec![1.0f32, 1.0])
            .unwrap();
        context
            .dispatch(
                &mut graph,
                &BTreeMap::from([("x", &tensor_in)]),
                &BTreeMap::from([("y", &tensor_out)]),
            )
            .unwrap();

        let mut out_data = vec![0.0f32; 2];
        context.read_tensor(&tensor_out, &mut out_data).unwrap();
        println!("  Softmax([1,1], axis=0) = {:?}", out_data);
        assert!(
            (out_data[0] - 0.5).abs() < 1e-2 && (out_data[1] - 0.5).abs() < 1e-2,
            "softmax([1,1]) should be ~[0.5, 0.5], got {:?}",
            out_data
        );
    }

    #[test]
    fn test_split() {
        let mut context = CONTEXT.lock().unwrap();

        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let desc_4 = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![4]);
        let input: MLOperand = builder.input("x", &desc_4).unwrap();
        let outputs: Vec<MLOperand> = builder.split(input, &[2, 2]).unwrap();
        let mut graph = builder
            .build(&BTreeMap::from([("y0", outputs[0]), ("y1", outputs[1])]))
            .unwrap();

        let tdesc_in = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![4])
            .to_writable()
            .to_readable();
        let tdesc_out = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2]).to_readable();
        let tensor_in = context.create_tensor(&tdesc_in).unwrap();
        let tensor_out0 = context.create_tensor(&tdesc_out).unwrap();
        let tensor_out1 = context.create_tensor(&tdesc_out).unwrap();

        context
            .write_tensor(&tensor_in, &vec![1.0f32, 2.0, 3.0, 4.0])
            .unwrap();
        context
            .dispatch(
                &mut graph,
                &BTreeMap::from([("x", &tensor_in)]),
                &BTreeMap::from([("y0", &tensor_out0), ("y1", &tensor_out1)]),
            )
            .unwrap();

        let mut out0 = vec![0.0f32; 2];
        let mut out1 = vec![0.0f32; 2];
        context.read_tensor(&tensor_out0, &mut out0).unwrap();
        context.read_tensor(&tensor_out1, &mut out1).unwrap();
        println!("  Split([1,2,3,4], [2,2]) = {:?}, {:?}", out0, out1);
        assert_eq!(out0, [1.0f32, 2.0]);
        assert_eq!(out1, [3.0f32, 4.0]);
    }

    #[test]
    fn test_transpose() {
        let mut context = CONTEXT.lock().unwrap();

        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let desc_2x2 = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![2, 2]);
        let input: MLOperand = builder.input("x", &desc_2x2).unwrap();
        let transpose_opts = MLTransposeOptions {
            permutation: vec![1, 0],
            ..MLTransposeOptions::default()
        };
        let output: MLOperand = builder
            .transpose_with_options(input, transpose_opts)
            .unwrap();
        let mut graph = builder.build(&BTreeMap::from([("y", output)])).unwrap();

        let tdesc = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2, 2])
            .to_writable()
            .to_readable();
        let tensor_in = context.create_tensor(&tdesc).unwrap();
        let tensor_out = context.create_tensor(&tdesc).unwrap();

        context
            .write_tensor(&tensor_in, &vec![1.0f32, 2.0, 3.0, 4.0])
            .unwrap();
        context
            .dispatch(
                &mut graph,
                &BTreeMap::from([("x", &tensor_in)]),
                &BTreeMap::from([("y", &tensor_out)]),
            )
            .unwrap();

        let mut out_data = vec![0.0f32; 4];
        context.read_tensor(&tensor_out, &mut out_data).unwrap();
        println!("  Transpose([[1,2],[3,4]]) = {:?}", out_data);
        assert_eq!(out_data, [1.0f32, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_div() {
        let mut context = CONTEXT.lock().unwrap();

        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let desc_2x2 = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![2, 2]);
        let a: MLOperand = builder.input("a", &desc_2x2).unwrap();
        let b: MLOperand = builder.input("b", &desc_2x2).unwrap();
        let output: MLOperand = builder.div(a, b).unwrap();
        let mut graph = builder.build(&BTreeMap::from([("y", output)])).unwrap();

        let tdesc = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2, 2])
            .to_writable()
            .to_readable();
        let tensor_a = context.create_tensor(&tdesc).unwrap();
        let tensor_b = context.create_tensor(&tdesc).unwrap();
        let tensor_out = context.create_tensor(&tdesc).unwrap();

        context
            .write_tensor(&tensor_a, &vec![6.0f32, 8.0, 10.0, 12.0])
            .unwrap();
        context
            .write_tensor(&tensor_b, &vec![2.0f32, 2.0, 2.0, 2.0])
            .unwrap();
        context
            .dispatch(
                &mut graph,
                &BTreeMap::from([("a", &tensor_a), ("b", &tensor_b)]),
                &BTreeMap::from([("y", &tensor_out)]),
            )
            .unwrap();

        let mut out_data = vec![0.0f32; 4];
        context.read_tensor(&tensor_out, &mut out_data).unwrap();
        println!("  Div([6,8,10,12], [2,2,2,2]) = {:?}", out_data);
        let expected = [3.0f32, 4.0, 5.0, 6.0];
        assert!(
            out_data
                .iter()
                .zip(expected.iter())
                .all(|(got, want)| (got - want).abs() < 2e-2),
            "Div precision: got {out_data:?}, expected {expected:?}"
        );
    }

    #[test]
    fn test_cast() {
        let mut context = CONTEXT.lock().unwrap();

        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let desc_2x2 = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![2, 2]);
        let input: MLOperand = builder.input("x", &desc_2x2).unwrap();
        let output: MLOperand = builder.cast(input, MLOperandDataType::Int32).unwrap();
        let mut graph = builder.build(&BTreeMap::from([("y", output)])).unwrap();

        let tdesc_in = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2, 2])
            .to_writable()
            .to_readable();
        let tdesc_out = MLTensorDescriptor::new(MLOperandDataType::Int32, vec![2, 2]).to_readable();
        let tensor_in = context.create_tensor(&tdesc_in).unwrap();
        let tensor_out = context.create_tensor(&tdesc_out).unwrap();

        context
            .write_tensor(&tensor_in, &vec![1.0f32, 2.0, 3.0, 4.0])
            .unwrap();
        context
            .dispatch(
                &mut graph,
                &BTreeMap::from([("x", &tensor_in)]),
                &BTreeMap::from([("y", &tensor_out)]),
            )
            .unwrap();

        let mut out_data = vec![0i32; 4];
        context.read_tensor(&tensor_out, &mut out_data).unwrap();
        println!("  Cast([1,2,3,4] float -> int32) = {:?}", out_data);
        assert_eq!(out_data, [1i32, 2, 3, 4]);
    }

    #[test]
    fn test_reduce_sum_axis0() {
        let mut context = CONTEXT.lock().unwrap();

        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let desc_2x2 = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![2, 2]);
        let input: MLOperand = builder.input("x", &desc_2x2).unwrap();
        let reduce_opts = MLReduceOptions {
            axes: Some(vec![0]),
            ..Default::default()
        };
        let output: MLOperand = builder.reduce_sum_with_options(input, reduce_opts).unwrap();
        let mut graph = builder.build(&BTreeMap::from([("y", output)])).unwrap();

        let tdesc_in = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2, 2])
            .to_writable()
            .to_readable();
        let tdesc_out = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2]).to_readable();
        let tensor_in = context.create_tensor(&tdesc_in).unwrap();
        let tensor_out = context.create_tensor(&tdesc_out).unwrap();

        context
            .write_tensor(&tensor_in, &vec![1.0f32, 2.0, 3.0, 4.0])
            .unwrap();
        context
            .dispatch(
                &mut graph,
                &BTreeMap::from([("x", &tensor_in)]),
                &BTreeMap::from([("y", &tensor_out)]),
            )
            .unwrap();

        let mut out_data = vec![0.0f32; 2];
        context.read_tensor(&tensor_out, &mut out_data).unwrap();
        println!("  ReduceSum([[1,2],[3,4]], axis=0) = {:?}", out_data);
        assert_eq!(out_data, [4.0f32, 6.0]);
    }

    #[test]
    fn test_prelu() {
        let mut context = CONTEXT.lock().unwrap();

        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let desc_2x2 = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![2, 2]);
        let input: MLOperand = builder.input("x", &desc_2x2).unwrap();
        let slope: MLOperand = builder
            .constant_from_vec(&desc_2x2, vec![0.5f32, 0.5, 0.5, 0.5])
            .unwrap();
        let output: MLOperand = builder.prelu(input, slope).unwrap();
        let mut graph = builder.build(&BTreeMap::from([("y", output)])).unwrap();

        let tdesc = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2, 2])
            .to_writable()
            .to_readable();
        let tensor_in = context.create_tensor(&tdesc).unwrap();
        let tensor_out = context.create_tensor(&tdesc).unwrap();

        context
            .write_tensor(&tensor_in, &vec![-1.0f32, 2.0, -3.0, 4.0])
            .unwrap();
        context
            .dispatch(
                &mut graph,
                &BTreeMap::from([("x", &tensor_in)]),
                &BTreeMap::from([("y", &tensor_out)]),
            )
            .unwrap();

        let mut out_data = vec![0.0f32; 4];
        context.read_tensor(&tensor_out, &mut out_data).unwrap();
        println!("  PRelu([-1,2,-3,4], slope=0.5) = {:?}", out_data);
        let expected = [-0.5f32, 2.0, -1.5, 4.0];
        assert!(
            out_data
                .iter()
                .zip(expected.iter())
                .all(|(got, want)| (got - want).abs() < 1e-2),
            "PRelu precision: got {out_data:?}, expected {expected:?}"
        );
    }

    #[test]
    fn test_split_add_downstream() {
        let mut context = CONTEXT.lock().unwrap();

        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let desc_4 = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![4]);
        let desc_2 = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![2]);
        let input: MLOperand = builder.input("x", &desc_4).unwrap();
        let splits: Vec<MLOperand> = builder.split(input, &[2, 2]).unwrap();
        let c0: MLOperand = builder
            .constant_from_vec(&desc_2, vec![10.0f32, 10.0])
            .unwrap();
        let c1: MLOperand = builder
            .constant_from_vec(&desc_2, vec![100.0f32, 100.0])
            .unwrap();
        let o0: MLOperand = builder.add(splits[0], c0).unwrap();
        let o1: MLOperand = builder.add(splits[1], c1).unwrap();
        let mut graph = builder
            .build(&BTreeMap::from([("o0", o0), ("o1", o1)]))
            .unwrap();

        let tdesc_in = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![4])
            .to_writable()
            .to_readable();
        let tdesc_out = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2]).to_readable();
        let tensor_in = context.create_tensor(&tdesc_in).unwrap();
        let tensor_o0 = context.create_tensor(&tdesc_out).unwrap();
        let tensor_o1 = context.create_tensor(&tdesc_out).unwrap();

        context
            .write_tensor(&tensor_in, &vec![1.0f32, 2.0, 3.0, 4.0])
            .unwrap();
        context
            .dispatch(
                &mut graph,
                &BTreeMap::from([("x", &tensor_in)]),
                &BTreeMap::from([("o0", &tensor_o0), ("o1", &tensor_o1)]),
            )
            .unwrap();

        let mut out0 = vec![0.0f32; 2];
        let mut out1 = vec![0.0f32; 2];
        context.read_tensor(&tensor_o0, &mut out0).unwrap();
        context.read_tensor(&tensor_o1, &mut out1).unwrap();
        println!("  Split->Add: o0 = {:?}, o1 = {:?}", out0, out1);
        assert_eq!(out0, [11.0f32, 12.0]);
        assert_eq!(out1, [103.0f32, 104.0]);
    }

    #[test]
    fn test_mul_broadcast() {
        let mut context = CONTEXT.lock().unwrap();

        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let desc_a = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![2, 3]);
        let desc_b = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![3]);
        let a: MLOperand = builder.input("a", &desc_a).unwrap();
        let b: MLOperand = builder.input("b", &desc_b).unwrap();
        let output: MLOperand = builder.mul(a, b).unwrap();
        let mut graph = builder.build(&BTreeMap::from([("y", output)])).unwrap();

        let tdesc_a = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2, 3])
            .to_writable()
            .to_readable();
        let tdesc_b = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![3])
            .to_writable()
            .to_readable();
        let tdesc_out =
            MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2, 3]).to_readable();
        let tensor_a = context.create_tensor(&tdesc_a).unwrap();
        let tensor_b = context.create_tensor(&tdesc_b).unwrap();
        let tensor_out = context.create_tensor(&tdesc_out).unwrap();

        context
            .write_tensor(&tensor_a, &vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        context
            .write_tensor(&tensor_b, &vec![10.0f32, 100.0, 1000.0])
            .unwrap();
        context
            .dispatch(
                &mut graph,
                &BTreeMap::from([("a", &tensor_a), ("b", &tensor_b)]),
                &BTreeMap::from([("y", &tensor_out)]),
            )
            .unwrap();

        let mut out_data = vec![0.0f32; 6];
        context.read_tensor(&tensor_out, &mut out_data).unwrap();
        println!("  Mul([[1,2,3],[4,5,6]], [10,100,1000]) = {:?}", out_data);
        assert_eq!(out_data, [10.0f32, 200.0, 3000.0, 40.0, 500.0, 6000.0]);
    }

    #[test]
    fn test_reduce_sum_axis1_keepdims() {
        let mut context = CONTEXT.lock().unwrap();

        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let desc_2x2 = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![2, 2]);
        let input: MLOperand = builder.input("x", &desc_2x2).unwrap();
        let reduce_opts = MLReduceOptions {
            axes: Some(vec![1]),
            keep_dimensions: true,
            ..Default::default()
        };
        let output: MLOperand = builder.reduce_sum_with_options(input, reduce_opts).unwrap();
        let mut graph = builder.build(&BTreeMap::from([("y", output)])).unwrap();

        let tdesc_in = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2, 2])
            .to_writable()
            .to_readable();
        let tdesc_out =
            MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2, 1]).to_readable();
        let tensor_in = context.create_tensor(&tdesc_in).unwrap();
        let tensor_out = context.create_tensor(&tdesc_out).unwrap();

        context
            .write_tensor(&tensor_in, &vec![1.0f32, 2.0, 3.0, 4.0])
            .unwrap();
        context
            .dispatch(
                &mut graph,
                &BTreeMap::from([("x", &tensor_in)]),
                &BTreeMap::from([("y", &tensor_out)]),
            )
            .unwrap();

        let mut out_data = vec![0.0f32; 2];
        context.read_tensor(&tensor_out, &mut out_data).unwrap();
        println!(
            "  ReduceSum([[1,2],[3,4]], axis=1, keepDims) = {:?}",
            out_data
        );
        assert_eq!(out_data, [3.0f32, 7.0]);
    }

    #[test]
    fn test_softmax_axis1() {
        let mut context = CONTEXT.lock().unwrap();

        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let desc_2x3 = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![2, 3]);
        let input: MLOperand = builder.input("x", &desc_2x3).unwrap();
        let output: MLOperand = builder.softmax(input, 1).unwrap();
        let mut graph = builder.build(&BTreeMap::from([("y", output)])).unwrap();

        let tdesc_in = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2, 3])
            .to_writable()
            .to_readable();
        let tdesc_out =
            MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2, 3]).to_readable();
        let tensor_in = context.create_tensor(&tdesc_in).unwrap();
        let tensor_out = context.create_tensor(&tdesc_out).unwrap();

        context
            .write_tensor(&tensor_in, &vec![1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0])
            .unwrap();
        context
            .dispatch(
                &mut graph,
                &BTreeMap::from([("x", &tensor_in)]),
                &BTreeMap::from([("y", &tensor_out)]),
            )
            .unwrap();

        let mut out_data = vec![0.0f32; 6];
        context.read_tensor(&tensor_out, &mut out_data).unwrap();
        println!("  Softmax([[1,2,3],[1,2,3]], axis=1) = {:?}", out_data);
        let expected = [0.0900f32, 0.2447, 0.6652, 0.0900, 0.2447, 0.6652];
        assert!(
            out_data
                .iter()
                .zip(expected.iter())
                .all(|(got, want)| (got - want).abs() < 1e-2),
            "Softmax axis 1 precision: got {out_data:?}, expected {expected:?}"
        );
    }

    #[test]
    fn test_conv2d_stride2() {
        let mut context = CONTEXT.lock().unwrap();

        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let input_desc = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![1, 1, 4, 4]);
        let filter_desc = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![1, 1, 2, 2]);
        let input: MLOperand = builder.input("x", &input_desc).unwrap();
        let filter: MLOperand = builder
            .constant_from_vec(&filter_desc, vec![1.0f32, 0.0, 0.0, 1.0])
            .unwrap();
        let conv_opts = MLConv2dOptions {
            strides: vec![2, 2],
            ..Default::default()
        };
        let output: MLOperand = builder
            .conv2_with_options(input, filter, conv_opts)
            .unwrap();
        let mut graph = builder.build(&BTreeMap::from([("y", output)])).unwrap();

        let tdesc_input = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![1, 1, 4, 4])
            .to_writable()
            .to_readable();
        let tdesc_output =
            MLTensorDescriptor::new(MLOperandDataType::Float32, vec![1, 1, 2, 2]).to_readable();
        let tensor_input = context.create_tensor(&tdesc_input).unwrap();
        let tensor_output = context.create_tensor(&tdesc_output).unwrap();

        context
            .write_tensor(
                &tensor_input,
                &vec![
                    1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                    15.0, 16.0,
                ],
            )
            .unwrap();
        context
            .dispatch(
                &mut graph,
                &BTreeMap::from([("x", &tensor_input)]),
                &BTreeMap::from([("y", &tensor_output)]),
            )
            .unwrap();

        let mut out_data = vec![0.0f32; 4];
        context.read_tensor(&tensor_output, &mut out_data).unwrap();
        println!("  Conv2d(4x4, identity, stride 2) = {:?}", out_data);
        assert_eq!(out_data, [7.0f32, 11.0, 23.0, 27.0]);
    }
}
