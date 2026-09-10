#![allow(dead_code, unused_variables)]

use log::{debug, info};

use crate::GraphInfo;
use crate::OperandDescriptor;
pub use crate::backend_selection::{Backend, BackendDevice, DeviceType};
#[cfg(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))]
use crate::backends::trtx::TrtxGraph;
use crate::error::Error;
use crate::error::Result;
use crate::graph::{DataType, Dimension, Operand, get_static_or_max_size};
use crate::mlgraphbuilder::get_operand;
use crate::runtime_checks::{RuntimeShapeState, TensorKind};

use crate::backends::cann::CannContext;
use crate::backends::coreml::CoremlContext;
use crate::backends::litert::LiteRtContext;
use crate::backends::ort::OrtContext;
use crate::backends::trtx::TrtxContext;
use std::collections::BTreeMap;
use std::{collections::HashMap, fmt::Display, marker::PhantomData};

pub use crate::mlcontextoptions::{
    CoremlOptions, LiteRtOptions, MLContextOptions, MLPowerPreference, OrtOptions, RustNNOptions,
    TrtxOptions,
};

/// <https://www.w3.org/TR/webnn/#typedefdef-mlnamedtensors>
pub type MLNamedTensors<'names> = BTreeMap<&'names str, &'names MLTensor>;
/// <https://www.w3.org/TR/webnn/#typedefdef-mlnamedoperands>
pub type MLNamedOperands<'names> = BTreeMap<&'names str, MLOperand>;

pub use crate::mlgraphbuilder::MLGraphBuilder;
use crate::{
    backend_selection::{select_backend, select_backend_by_gpu},
    operator_enums::MLOperandDataType,
};

// Backend traits

pub(crate) trait ListDevices {
    // TODO: should probably be a Result or just be an empty Vec when something is not working the
    fn list_devices() -> Vec<BackendDevice>;
}

// could make public later if interface stabilized
pub(crate) trait MLBackendContext<'context>: std::fmt::Debug + Send + Sync {
    fn accelerated(&self) -> bool;
    fn create_builder<'builder>(
        &mut self,
    ) -> Result<Box<dyn MLBackendBuilder<'context, 'builder> + 'builder>>
    where
        'context: 'builder;
    fn create_tensor(&mut self, descriptor: &MLTensorDescriptor) -> Result<MLTensor>;
    fn rustnn_resize_tensor(&mut self, tensor: &mut MLTensor, new_shape: &[u64]) -> Result<()>;
    fn rustnn_set_tensor_capacity(
        &mut self,
        tensor: &mut MLTensor,
        max_shape: &[u64],
    ) -> Result<()>;
    fn create_constant_tensor(
        &mut self,
        descriptor: &MLTensorDescriptor,
        input_data: &[u8],
    ) -> Result<MLTensor>;
    /*async*/
    fn read_tensor(&mut self, tensor: &MLTensor, array: &mut [u8]) -> Result<()>;
    /*async*/
    fn write_tensor(&mut self, tensor: &MLTensor, array: &[u8]) -> Result<()>;
    fn dispatch(
        &mut self,
        graph: &mut MLGraph,
        inputs: &MLNamedTensors,
        outputs: &MLNamedTensors,
    ) -> Result<()>;
}

pub(crate) trait MLBackendBuilder<'context, 'builder>: std::fmt::Debug + Send {
    /*async*/
    fn build(&mut self, graph: GraphInfo) -> Result<MLGraph<'context>>;
}

// can be made a Box<dyn better_any::Tid<'context> + 'context> for dynamic dispatch
// dyn Any does not work since Any requires 'static
#[derive(Debug)]
pub(crate) enum MLBackendGraph<'context> {
    #[cfg(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))]
    TrtxEngine(TrtxGraph<'context>),
    #[cfg(feature = "onnx-runtime")]
    OnnxSession(
        crate::backends::ort::OrtGraph,
        std::marker::PhantomData<&'context ()>,
    ),
    #[cfg(feature = "coreml-runtime")]
    CoremlModel(crate::backends::coreml::CoremlGraph),
    #[cfg(feature = "litert-runtime")]
    LiteRtGraph(crate::backends::litert::LiteRtGraph),
    #[cfg(any(feature = "cann-runtime", feature = "cann-runtime-mock"))]
    CannEngine(crate::backends::cann::CannGraph),
    PhantomData(PhantomData<&'context u8>),
}

impl<'context> MLBackendGraph<'context> {
    #[cfg(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))]
    pub(crate) fn as_trtx_engine_mut(&mut self) -> Option<&mut TrtxGraph<'context>> {
        if let Self::TrtxEngine(v) = self {
            Some(v)
        } else {
            None
        }
    }

    #[cfg(feature = "onnx-runtime")]
    pub(crate) fn as_onnx_session_mut(&mut self) -> Option<&mut crate::backends::ort::OrtGraph> {
        match self {
            Self::OnnxSession(g, _) => Some(g),
            _ => None,
        }
    }

    #[cfg(feature = "coreml-runtime")]
    pub(crate) fn as_coreml_model(&self) -> Option<&crate::backends::coreml::CoremlGraph> {
        if let Self::CoremlModel(v) = self {
            Some(v)
        } else {
            None
        }
    }
}

// types for MLContext

// aka WebGpuDevice
#[derive(Debug)]
pub struct GpuDevice {}

#[derive(Debug)]
pub struct MLContextLostInfo {
    message: String,
}

impl MLContextLostInfo {
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl Display for MLContextLostInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

/// <https://www.w3.org/TR/webnn/#api-mltensor>
#[derive(Debug, Clone)]
pub struct MLTensor {
    pub(crate) id: usize,
    pub(crate) constant: bool,
    /// internal slots as per <https://www.w3.org/TR/webnn/#api-mltensor>
    pub(crate) descriptor: MLTensorDescriptor,
    //context: &'context MLContext, // todo, omit context?
    //// pending promises, need to be canceled when tensor is destroyed
}

impl MLTensor {
    pub fn data_type(&self) -> MLOperandDataType {
        self.descriptor.data_type
    }
    pub fn shape(&self) -> &[u64] {
        &self.descriptor.shape
    }
    pub fn readable(&self) -> bool {
        self.descriptor.readable
    }
    pub fn writable(&self) -> bool {
        self.descriptor.writable
    }
    pub fn constant(&self) -> bool {
        self.constant
    }
    // TODO: or replace by Rust's drop?
    pub fn destoy(&self) {
        todo!() // destroying needs to cancel pending promises
    }
    pub fn destroyed(&self) -> bool {
        todo!() // JS has a isDestroyed method
    }

    pub fn rustnn_required_bytes(&self) -> usize {
        self.descriptor.rustnn_required_bytes()
    }

    pub(crate) fn descriptor(&self) -> &MLTensorDescriptor {
        &self.descriptor
    }
}

#[derive(Debug)]
pub struct MLGraph<'context> {
    pub(crate) backend: MLBackendGraph<'context>,

    // inputs/outputs of compiled graph
    pub input_descriptors: HashMap<String, OperandDescriptor>,
    pub output_descriptors: HashMap<String, OperandDescriptor>,
}

impl<'context> MLGraph<'context> {
    pub(crate) fn new(backend: MLBackendGraph<'context>, graph_info: &GraphInfo) -> Result<Self> {
        let (input_descriptors, output_descriptors) = graph_info
            .io_binding_maps()
            .map_err(|e| Error::GraphBuildError { source: e.into() })?;
        Ok(Self {
            backend,
            input_descriptors,
            output_descriptors,
        })
    }

    fn operand_descriptors(
        operands: &HashMap<String, Operand>,
    ) -> HashMap<String, OperandDescriptor> {
        operands
            .iter()
            .map(|(name, op)| (name.clone(), op.descriptor.clone()))
            .collect()
    }

    fn verify_dispatch_bindings(
        &mut self,
        inputs: &MLNamedTensors,
        outputs: &MLNamedTensors,
    ) -> Result<()> {
        let input_shapes: HashMap<String, Vec<usize>> = inputs
            .iter()
            .map(|(&name, tensor)| {
                (
                    name.to_string(),
                    tensor.shape().iter().map(|&d| d as usize).collect(),
                )
            })
            .collect();
        let output_shapes: HashMap<String, Vec<usize>> = outputs
            .iter()
            .map(|(&name, tensor)| {
                (
                    name.to_string(),
                    tensor.shape().iter().map(|&d| d as usize).collect(),
                )
            })
            .collect();

        let mut runtime_shape_state = RuntimeShapeState::new();
        runtime_shape_state
            .validate_named_shapes(&input_shapes, &self.input_descriptors, TensorKind::Input)
            .map_err(|e| Error::GraphDispatchError { source: e.into() })?;
        runtime_shape_state
            .validate_named_shapes(&output_shapes, &self.output_descriptors, TensorKind::Output)
            .map_err(|e| Error::GraphDispatchError { source: e.into() })?;

        for (&name, tensor) in inputs {
            let expected = self.input_descriptors.get(name).expect("validated above");
            if DataType::from(tensor.data_type()) != expected.data_type {
                return Err(Error::GraphDispatchError {
                    source: format!(
                        "input '{name}' data type mismatch (expected {:?}, got {:?})",
                        expected.data_type,
                        tensor.data_type()
                    )
                    .into(),
                });
            }
        }

        for (&name, tensor) in outputs {
            let expected = self.output_descriptors.get(name).expect("validated above");
            if DataType::from(tensor.data_type()) != expected.data_type {
                return Err(Error::GraphDispatchError {
                    source: format!(
                        "output '{name}' data type mismatch (expected {:?}, got {:?})",
                        expected.data_type,
                        tensor.data_type()
                    )
                    .into(),
                });
            }
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct MLOpSupportLimits {}

// https://www.w3.org/TR/webnn/#dictdef-mloperanddescriptor
#[derive(Debug, Eq, PartialEq, Default, Clone)]
pub struct MLOperandDescriptor {
    data_type: MLOperandDataType,
    shape: Vec<u64>, // TODO: this is u64 instead of WebNN's u32. u32 is screaming for problems on desktop
}

impl From<&MLOperandDescriptor> for OperandDescriptor {
    fn from(val: &MLOperandDescriptor) -> Self {
        OperandDescriptor {
            data_type: val.data_type.into(),
            shape: val
                .shape
                .iter()
                .map(|s| Dimension::Static(*s as u32))
                .collect(),
            pending_permutation: Default::default(),
        }
    }
}

impl MLOperandDescriptor {
    pub fn new(data_type: MLOperandDataType, shape: Vec<u64>) -> Self {
        Self { data_type, shape }
    }

    pub fn data_type(&self) -> MLOperandDataType {
        self.data_type
    }

    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    pub fn set_data_type(&mut self, data_type: MLOperandDataType) {
        self.data_type = data_type;
    }

    pub fn set_shape(&mut self, shape: Vec<u64>) {
        self.shape = shape;
    }

    pub(crate) fn rustnn_required_bytes(&self) -> usize {
        let elements = (self.shape().iter().copied().product::<u64>() as usize).max(1);
        self.data_type().rustnn_storage_byte_length(elements).max(1)
    }
}

/// https://www.w3.org/TR/webnn/#dictdef-mltensordescriptor
#[derive(Debug, Eq, PartialEq, Default, Clone)]
pub struct MLTensorDescriptor {
    operand_descriptor: MLOperandDescriptor,
    readable: bool,
    writable: bool,
}

#[derive(Debug, Eq, PartialEq, Default, Copy, Clone, Hash)]
pub struct MLOperand {
    pub(crate) id: usize,
}

// TODO: actually, WebNN requires shape, data_type directly on MLOperand
// would require MLOperand==Operand and we give the user &MLOperand or Rc<MLOperand>
impl MLOperand {
    pub fn shape(self, graph: &GraphInfo) -> Result<Vec<u64>> {
        let operand = get_operand(self, graph)?;

        Ok(operand
            .descriptor
            .shape
            .iter()
            .map(|d| get_static_or_max_size(d) as u64)
            .collect())
    }

    pub fn data_type(self, graph: &GraphInfo) -> Result<MLOperandDataType> {
        let operand = get_operand(self, graph)?;

        Ok(operand.descriptor.data_type.try_into()?)
    }
}

impl From<u32> for MLOperand {
    fn from(value: u32) -> Self {
        Self { id: value as usize }
    }
}

impl std::ops::Deref for MLTensorDescriptor {
    type Target = MLOperandDescriptor;

    fn deref(&self) -> &Self::Target {
        &self.operand_descriptor
    }
}

impl std::ops::DerefMut for MLTensorDescriptor {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.operand_descriptor
    }
}

impl MLTensorDescriptor {
    pub fn new(data_type: MLOperandDataType, shape: Vec<u64>) -> Self {
        Self {
            operand_descriptor: MLOperandDescriptor { data_type, shape },
            writable: false,
            readable: false,
        }
    }
    pub fn from_operand_descriptor(operand_descriptor: &MLOperandDescriptor) -> Self {
        Self {
            operand_descriptor: operand_descriptor.clone(),
            writable: false,
            readable: false,
        }
    }
    pub fn readable(&self) -> bool {
        self.readable
    }

    pub fn writable(&self) -> bool {
        self.writable
    }

    pub fn set_writable(&mut self, writable: bool) {
        self.writable = writable;
    }

    pub fn set_readable(&mut self, readable: bool) {
        self.readable = readable;
    }

    pub fn operand_descriptor(&self) -> &MLOperandDescriptor {
        &self.operand_descriptor
    }

    pub fn set_operand_descriptor(&mut self, operand_descriptor: MLOperandDescriptor) {
        self.operand_descriptor = operand_descriptor;
    }

    pub fn to_writable(&self) -> Self {
        let mut copy = self.clone();
        copy.writable = true;
        copy
    }

    pub fn to_readable(&self) -> Self {
        let mut copy = self.clone();
        copy.readable = true;
        copy
    }
}

// TODO: this is wrong. must be 'context and `for <'builder>` to be valid for each builder lifetime (multiple children!)
#[derive(Debug)]
pub struct MLContext<'context> {
    pub(crate) backend: Box<dyn MLBackendContext<'context> + 'context>,
    pub(crate) device: BackendDevice,
}

impl<'context> MLContext<'context> {
    // those are methods on `create_context`
    //pub async
    pub fn create(options: &MLContextOptions) -> Result<Self> {
        let device = select_backend(options)
            .inspect_err(|e| log::warn!("Error selecting backend: {e:?}"))?;
        info!("Backend selected: {device:?}");
        let backend: Box<dyn MLBackendContext<'context> + 'context> = match device {
            crate::backend_selection::BackendDevice::Onnx { ep_device_idx, .. } => Box::new(
                OrtContext::new_from_ep_idx(ep_device_idx, Some(&options.rustnn_options))?,
            ),
            crate::backend_selection::BackendDevice::Trtx { cuda_device_idx } => Box::new(
                TrtxContext::new(cuda_device_idx, Some(&options.rustnn_options))
                    .map_err(|e| Error::ContextCreationError { source: e.into() })?,
            ),
            crate::backend_selection::BackendDevice::Coreml { device_type } => Box::new(
                CoremlContext::new_from_device_type(device_type, Some(&options.rustnn_options))?,
            ),
            crate::backend_selection::BackendDevice::LiteRt { device_type } => Box::new(
                LiteRtContext::new_from_device_type(device_type, Some(&options.rustnn_options))?,
            ),
            crate::backend_selection::BackendDevice::Cann { device_type } => Box::new(
                CannContext::new_from_device_type(device_type, Some(&options.rustnn_options))?,
            ),
        };
        Ok(Self { backend, device })
    }

    #[expect(unreachable_code)]
    pub async fn create_from_gpu_device(gpu_device: &GpuDevice) -> Result<Self> {
        let device = select_backend_by_gpu(gpu_device)?;
        let backend = match device {
            crate::backend_selection::BackendDevice::Onnx { .. } => todo!(),
            crate::backend_selection::BackendDevice::Trtx { cuda_device_idx } => todo!(),
            crate::backend_selection::BackendDevice::Coreml { device_type } => todo!(),
            crate::backend_selection::BackendDevice::LiteRt { .. } => todo!(),
            crate::backend_selection::BackendDevice::Cann { .. } => todo!(),
        };
        Ok(Self { backend, device })
    }
    pub fn accelerated(&self) -> bool {
        self.backend.accelerated()
    }

    pub async fn lost(&self) -> MLContextLostInfo {
        todo!()
    }

    pub async fn create_constant_tensor(
        &mut self,
        descriptor: &MLOperandDescriptor,
        input_data: &[u8], // with owned variant?
    ) -> MLTensor {
        todo!()
    }

    // async
    pub fn create_tensor(&mut self, descriptor: &MLTensorDescriptor) -> Result<MLTensor> {
        self.backend.create_tensor(descriptor)
    }

    // omit destroy()? We're not JS, objects can be destroyed via drop, we could do destroy stuff in Drop
    pub fn destroy(self) {
        todo!()
    }

    pub fn rustnn_backend(&self) -> Backend {
        self.device.backend()
    }

    pub fn rustnn_device(&self) -> BackendDevice {
        self.device
    }

    pub fn rustnn_device_type(&self) -> DeviceType {
        self.device.device_type()
    }

    pub fn dispatch(
        &mut self,
        graph: &mut MLGraph,
        inputs: &MLNamedTensors,
        outputs: &MLNamedTensors,
    ) -> crate::error::Result<()> {
        debug!("Dispatch {graph:?}, inputs={inputs:?}, outputs={outputs:?}");
        //https://www.w3.org/TR/webnn/#dom-mlcontext-dispatch
        // spec: 4. If allTensors contains any duplicate items, then throw a TypeError.
        let mut all_tensor_ids = HashMap::new();
        for (&name, &tensor) in inputs.iter().chain(outputs.iter()) {
            if let Some(other_name) = all_tensor_ids.insert(name, tensor.id) {
                return Err(Error::DuplicateTensorBinding {
                    aliased_tensor: tensor.clone(),
                    first_binding: other_name.to_string(),
                    other_binding: name.to_string(),
                });
            }
        }

        graph.verify_dispatch_bindings(inputs, outputs)?;
        self.backend.dispatch(graph, inputs, outputs)
    }

    pub fn op_support_limits(&self) -> MLOpSupportLimits {
        todo!()
    }

    //async
    pub fn read_tensor<T: bytemuck::Pod>(
        &mut self,
        tensor: &MLTensor,
        array: &mut [T],
    ) -> Result<()> {
        debug!(
            "Read {} bytes from tensor {tensor:?}",
            std::mem::size_of_val(array)
        );
        if !tensor.readable() {
            return Err(Error::ReadToNonReadableTensor {
                tensor: tensor.clone(),
            });
        }
        if tensor.rustnn_required_bytes() != std::mem::size_of_val(array) {
            return Err(Error::WrongReadSize {
                read_size: std::mem::size_of_val(array),
                required_size: tensor.rustnn_required_bytes(),
                tensor: tensor.clone(),
            });
        }
        self.backend
            .read_tensor(tensor, bytemuck::cast_slice_mut(array))
    }

    //async
    pub fn write_tensor<T: bytemuck::Pod>(&mut self, tensor: &MLTensor, array: &[T]) -> Result<()> {
        debug!(
            "Write {} bytes to tensor {tensor:?}",
            std::mem::size_of_val(array)
        );
        if !tensor.writable() {
            return Err(Error::WriteToNonWritableTensor {
                tensor: tensor.clone(),
            });
        }
        if tensor.rustnn_required_bytes() != std::mem::size_of_val(array) {
            return Err(Error::WrongWriteSize {
                write_size: std::mem::size_of_val(array),
                required_size: tensor.rustnn_required_bytes(),
                tensor: tensor.clone(),
            });
        }
        self.backend
            .write_tensor(tensor, bytemuck::cast_slice(array))
    }

    pub fn rustnn_resize_tensor(&mut self, tensor: &mut MLTensor, new_shape: &[u64]) -> Result<()> {
        self.backend.rustnn_resize_tensor(tensor, new_shape)
    }

    pub fn rustnn_set_tensor_capacity(
        &mut self,
        tensor: &mut MLTensor,
        max_shape: &[u64],
    ) -> Result<()> {
        self.backend.rustnn_set_tensor_capacity(tensor, max_shape)
    }
}

#[cfg(test)]
mod test {
    use crate::{mlcontext::*, mlgraphbuilder::MLGraphBuilder, webnn_json::from_graph_json};

    fn create_add_graph_context_and_graph() -> Option<(MLContext<'static>, MLGraph<'static>)> {
        let contents = r#"
webnn_graph "sample_graph" v1 {
  inputs {
    lhs: f32[2, 2];
  }

  consts {
    rhs: f32[2, 2] @scalar(1.0);
  }

  nodes {
    sum = add(lhs, rhs);
  }

  outputs { sum; }
}"#;

        let _ = pretty_env_logger::try_init();
        let sanitized = crate::loader::sanitize_webnn_identifiers(contents);
        let graph_json = webnn_graph::parser::parse_wg_text(&sanitized).unwrap();
        let graph_info = from_graph_json(&graph_json).unwrap();

        let context = MLContext::create(&MLContextOptions::new(MLPowerPreference::Default, true));
        if matches!(context, Err(crate::error::Error::NoBackendAvailable { .. })) {
            return None;
        };

        let mut context = context.unwrap();
        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let graph = builder.build_graph_info(graph_info).unwrap();
        drop(builder);

        Some((context, graph))
    }

    fn rw_tensor_desc(shape: Vec<u64>) -> MLTensorDescriptor {
        MLTensorDescriptor::new(crate::operator_enums::MLOperandDataType::Float32, shape)
            .to_writable()
            .to_readable()
    }

    #[test]
    fn test_tensor_desc() {
        let default_operand_desc = MLOperandDescriptor::default();
        let mut default_tensor_desc = MLTensorDescriptor::default();
        assert_eq!(default_tensor_desc.shape(), default_operand_desc.shape());
        assert_eq!(
            default_tensor_desc.data_type(),
            default_operand_desc.data_type()
        );
        assert_eq!(default_tensor_desc.data_type(), MLOperandDataType::Float32);
        assert!(!default_tensor_desc.writable());
        assert!(!default_tensor_desc.readable());
        default_tensor_desc.set_writable(true);
        assert!(default_tensor_desc.writable());
        default_tensor_desc.set_writable(true);
        assert!(default_tensor_desc.writable());

        let desc = MLTensorDescriptor::new(MLOperandDataType::Float16, vec![3, 4]);
        let op_desc = MLOperandDescriptor::new(MLOperandDataType::Float16, vec![3, 4]);
        assert_eq!(*desc.operand_descriptor(), op_desc);
    }

    #[test]
    fn test_backend_hint() {
        let _ = pretty_env_logger::try_init();
        let context = MLContext::create(
            &MLContextOptions::new(MLPowerPreference::Default, true)
                .with_rustnn_backend_hint(Backend::Onnx),
        );
        if matches!(
            context,
            Err(crate::error::Error::NoBackendAvailableForBackendHint { .. })
        ) {
            return;
        };
        assert_eq!(context.unwrap().rustnn_backend(), Backend::Onnx);
    }
    #[test]
    fn test_create_context() {
        let _ = pretty_env_logger::try_init();
        let context = MLContext::create(&MLContextOptions::new(MLPowerPreference::Default, true));
        if matches!(context, Err(crate::error::Error::NoBackendAvailable { .. })) {
            return;
        };

        let mut context = context.unwrap();
        dbg!(&context);
        let mut desc = MLTensorDescriptor::new(
            crate::operator_enums::MLOperandDataType::Float32,
            [2, 2].to_vec(),
        );
        desc.set_readable(true);
        desc.set_writable(true);
        let tensor = context.create_tensor(&desc).unwrap();

        let upload = vec![1.0f32, 2., 3., 4.];
        let mut download = vec![0.0f32; 4];
        context.write_tensor(&tensor, &upload).unwrap();
        context.read_tensor(&tensor, &mut download).unwrap();
        assert_eq!(&upload, &download);
    }

    #[test]
    fn test_dispatch() {
        let Some((mut context, mut graph)) = create_add_graph_context_and_graph() else {
            return;
        };
        dbg!(&context);
        let desc = rw_tensor_desc([2, 2].to_vec());

        let tensor = context.create_tensor(&desc).unwrap();
        let mut inputs = MLNamedTensors::new();
        inputs.insert("lhs", &tensor);
        let mut outputs = MLNamedTensors::new();
        outputs.insert("sum", &tensor);

        let upload = vec![1.0f32, 2., 3., 4.];
        let upload_f64 = vec![1.0, 2., 3., 4.];
        let mut download = vec![0.0f32; 4];
        context.write_tensor(&tensor, &upload_f64).unwrap_err();
        context.write_tensor(&tensor, &upload).unwrap();
        context.dispatch(&mut graph, &inputs, &outputs).unwrap();
        context.read_tensor(&tensor, &mut download).unwrap();
        assert_eq!(&vec![2.0f32, 3., 4., 5.], &download);
    }

    #[cfg(feature = "trtx-runtime")]
    #[test]
    fn test_trtx_cuda_graph_replay() {
        let Some((mut context, mut graph)) = create_add_graph_context_and_graph() else {
            return;
        };
        if context.rustnn_backend() != Backend::Trtx {
            return;
        }

        let desc = rw_tensor_desc([2, 2].to_vec());
        let input_tensor = context.create_tensor(&desc).unwrap();
        let output_tensor = context.create_tensor(&desc).unwrap();
        let inputs = MLNamedTensors::from([("lhs", &input_tensor)]);
        let outputs = MLNamedTensors::from([("sum", &output_tensor)]);

        for upload in [
            vec![1.0f32, 2., 3., 4.],
            vec![5.0f32, 6., 7., 8.],
            vec![9.0f32, 10., 11., 12.],
        ] {
            context.write_tensor(&input_tensor, &upload).unwrap();
            context.dispatch(&mut graph, &inputs, &outputs).unwrap();

            let mut download = vec![0.0f32; 4];
            context.read_tensor(&output_tensor, &mut download).unwrap();
            assert_eq!(
                upload.iter().map(|value| value + 1.0).collect::<Vec<_>>(),
                download
            );
        }
    }

    #[test]
    fn test_dispatch_invalid_input_name_error_message() {
        let Some((mut context, mut graph)) = create_add_graph_context_and_graph() else {
            return;
        };

        let desc = rw_tensor_desc([2, 2].to_vec());
        let tensor = context.create_tensor(&desc).unwrap();
        let mut inputs = MLNamedTensors::new();
        inputs.insert("invalid_input", &tensor);
        let mut outputs = MLNamedTensors::new();
        outputs.insert("sum", &tensor);

        let err = context.dispatch(&mut graph, &inputs, &outputs).unwrap_err();
        std::assert_matches!(
            err,
            crate::error::Error::GraphDispatchError { source }
                if source.to_string() == "missing runtime input tensor `lhs`"
        );
    }

    #[test]
    fn test_dispatch_invalid_input_shape_error_message() {
        let Some((mut context, mut graph)) = create_add_graph_context_and_graph() else {
            return;
        };

        let desc = rw_tensor_desc([2, 3].to_vec());
        let tensor = context.create_tensor(&desc).unwrap();
        let mut inputs = MLNamedTensors::new();
        inputs.insert("lhs", &tensor);
        let mut outputs = MLNamedTensors::new();
        outputs.insert("sum", &tensor);

        let err = context.dispatch(&mut graph, &inputs, &outputs).unwrap_err();
        std::assert_matches!(
            err,
            crate::error::Error::GraphDispatchError { source }
                if source.to_string()
                    == "runtime input tensor `lhs` dimension 1 mismatch (expected 2, got 3)"
        );
    }

    #[test]
    fn test_dispatch_invalid_output_shape_error_message() {
        let Some((mut context, mut graph)) = create_add_graph_context_and_graph() else {
            return;
        };

        let in_desc = rw_tensor_desc([2, 2].to_vec());
        let out_desc = rw_tensor_desc([2, 3].to_vec());
        let input_tensor = context.create_tensor(&in_desc).unwrap();
        let output_tensor = context.create_tensor(&out_desc).unwrap();
        let mut inputs = MLNamedTensors::new();
        inputs.insert("lhs", &input_tensor);
        let mut outputs = MLNamedTensors::new();
        outputs.insert("sum", &output_tensor);

        let err = context.dispatch(&mut graph, &inputs, &outputs).unwrap_err();
        std::assert_matches!(
            err,
            crate::error::Error::GraphDispatchError { source }
                if source.to_string()
                    == "runtime output tensor `sum` dimension 1 mismatch (expected 2, got 3)"
        );
    }
}
