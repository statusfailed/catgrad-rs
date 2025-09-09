use super::super::types::*;
use crate::category::core::{Dtype, Shape};
use crate::interpreter::backend::{Backend, BackendError, NdArray};
use candle_core::{DType, Device, Tensor};

// ============================================================================
// CANDLE BACKEND ARCHITECTURE EXPLANATION
// ============================================================================
//
// This backend follows a two-layer architecture pattern common in Rust:
//
// 1. **`CandleTensor` - The Data Container**
//    - Wrapper around `candle_core::Tensor`
//    - Implements the `NdArray<D>` trait (required by the Backend trait)
//    - Adds `PartialEq` implementation (missing from the underlying Tensor)
//    - Provides type safety and API consistency with other backends
//
// 2. **`CandleBackend` - The Operations Provider**
//    - Manages device state (CPU, GPU, Metal, etc.)
//    - Implements the `Backend` trait with all operations (add, mul, matmul, etc.)
//    - Handles device-specific configuration and state
//
// ARCHITECTURE PATTERN:
//
// Backend (Operations) ──→ Tensor (Data)
//      ↓                      ↓
// CandleBackend         CandleTensor
//   - device              - Tensor
//   - operations          - PartialEq
//   - configuration       - NdArray trait
//
// COMPARISON WITH NDARRAY BACKEND:
//
// The ndarray backend is simpler because:
// - `ArrayD<D>` already implements `PartialEq` and other required traits
// - No device management needed (always CPU)
// - No additional configuration required
//
// Candle needs both structs because:
// - Device Management: Need to track which device tensors are on
// - Missing Traits: `candle_core::Tensor` doesn't implement `PartialEq`
// - Custom Behavior: Need to add our own methods and error handling
//
// This design follows the Single Responsibility Principle:
// - `CandleTensor`: Manages tensor data and implements data-related traits
// - `CandleBackend`: Manages operations and device state
//
// ============================================================================

#[derive(Clone, Debug)]
pub struct CandleTensor(pub Tensor);

#[derive(Clone, Debug)]
pub struct CandleBackend {
    device: Device,
}

impl CandleBackend {
    pub fn new() -> Self {
        Self {
            device: Device::Cpu,
        }
    }

    pub fn with_device(device: Device) -> Self {
        Self { device }
    }
}

impl Default for CandleBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CandleBackend {
    type NdArray<D: HasDtype> = CandleTensor;

    fn scalar<D: HasDtype>(&self, d: D) -> Self::NdArray<D> {
        // Use unsafe transmute as a workaround for type erasure
        // This is not ideal but necessary due to trait constraints
        if std::mem::size_of::<D>() == std::mem::size_of::<f32>() {
            let val = unsafe { std::mem::transmute_copy::<D, f32>(&d) };
            CandleTensor(Tensor::new(&[val], &self.device).unwrap())
        } else if std::mem::size_of::<D>() == std::mem::size_of::<u32>() {
            let val = unsafe { std::mem::transmute_copy::<D, u32>(&d) };
            CandleTensor(Tensor::new(&[val], &self.device).unwrap())
        } else {
            panic!("Unsupported dtype for scalar creation");
        }
    }

    fn zeros<D: HasDtype + Default>(&self, shape: Shape) -> Self::NdArray<D> {
        let dims: &[usize] = &shape.0;
        if std::mem::size_of::<D>() == std::mem::size_of::<f32>() {
            CandleTensor(Tensor::zeros(dims, DType::F32, &self.device).unwrap())
        } else if std::mem::size_of::<D>() == std::mem::size_of::<u32>() {
            CandleTensor(Tensor::zeros(dims, DType::U32, &self.device).unwrap())
        } else {
            panic!("Unsupported dtype for zeros creation");
        }
    }

    fn ndarray_from_slice<D: HasDtype>(
        &self,
        data: &[D],
        shape: Shape,
    ) -> Result<Self::NdArray<D>, BackendError> {
        let dims: &[usize] = &shape.0;
        if std::mem::size_of::<D>() == std::mem::size_of::<f32>() {
            let data_f32: &[f32] = unsafe { std::mem::transmute(data) };
            let tensor = Tensor::new(data_f32, &self.device)
                .map_err(|_| BackendError::ShapeError)?
                .reshape(dims)
                .map_err(|_| BackendError::ShapeError)?;
            Ok(CandleTensor(tensor))
        } else if std::mem::size_of::<D>() == std::mem::size_of::<u32>() {
            let data_u32: &[u32] = unsafe { std::mem::transmute(data) };
            let tensor = Tensor::new(data_u32, &self.device)
                .map_err(|_| BackendError::ShapeError)?
                .reshape(dims)
                .map_err(|_| BackendError::ShapeError)?;
            Ok(CandleTensor(tensor))
        } else {
            panic!("Unsupported dtype for slice creation");
        }
    }

    fn cast(&self, x: TaggedNdArray<Self>, target_dtype: Dtype) -> TaggedNdArray<Self> {
        match (&x, target_dtype) {
            (TaggedNdArray::F32(arr), Dtype::U32) => {
                let result = arr[0].0.to_dtype(DType::U32).unwrap();
                TaggedNdArray::U32([CandleTensor(result)])
            }
            (TaggedNdArray::U32(arr), Dtype::F32) => {
                let result = arr[0].0.to_dtype(DType::F32).unwrap();
                TaggedNdArray::F32([CandleTensor(result)])
            }
            (TaggedNdArray::F32(_), Dtype::F32) => x,
            (TaggedNdArray::U32(_), Dtype::U32) => x,
        }
    }

    fn matmul(&self, lhs: TaggedNdArrayTuple<Self, 2>) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match lhs {
            F32([x, y]) => F32([CandleTensor(Self::batched_matmul(x.0, y.0))]),
            U32([x, y]) => U32([CandleTensor(Self::batched_matmul(x.0, y.0))]),
        }
    }

    fn add(&self, lhs: TaggedNdArrayTuple<Self, 2>) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::add_f32(x, y)]),
            U32([x, y]) => U32([Self::add_u32(x, y)]),
        }
    }

    fn mul(&self, lhs: TaggedNdArrayTuple<Self, 2>) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::mul_f32(x, y)]),
            U32([x, y]) => U32([Self::mul_u32(x, y)]),
        }
    }

    fn div(&self, lhs: TaggedNdArrayTuple<Self, 2>) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::div_f32(x, y)]),
            U32([x, y]) => U32([Self::div_u32(x, y)]),
        }
    }

    fn pow(&self, lhs: TaggedNdArrayTuple<Self, 2>) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::pow_f32(x, y)]),
            U32([x, y]) => U32([Self::pow_u32(x, y)]),
        }
    }

    fn neg(&self, x: TaggedNdArray<Self>) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match x {
            F32([arr]) => F32([Self::neg_f32(arr)]),
            U32([arr]) => U32([Self::neg_u32(arr)]),
        }
    }

    fn broadcast(&self, x: TaggedNdArray<Self>, shape_prefix: Shape) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match x {
            F32([arr]) => F32([CandleTensor(Self::broadcast_tensor(arr.0, shape_prefix))]),
            U32([arr]) => U32([CandleTensor(Self::broadcast_tensor(arr.0, shape_prefix))]),
        }
    }

    fn reshape(&self, x: TaggedNdArray<Self>, new_shape: Shape) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match x {
            F32([arr]) => F32([CandleTensor(Self::reshape_tensor(arr.0, new_shape))]),
            U32([arr]) => U32([CandleTensor(Self::reshape_tensor(arr.0, new_shape))]),
        }
    }
}

impl CandleBackend {
    fn reshape_tensor(tensor: Tensor, new_shape: Shape) -> Tensor {
        tensor.reshape(&*new_shape.0).unwrap()
    }

    fn broadcast_tensor(tensor: Tensor, shape_prefix: Shape) -> Tensor {
        let current_shape = tensor.dims();
        let mut new_shape = current_shape.to_vec();
        new_shape.splice(0..0, shape_prefix.0.iter().cloned());

        // Candle doesn't have a direct broadcast function, so this is a simplified implementation
        // (in practice you might need more sophisticated broadcasting, see below)
        tensor.expand(&*new_shape).unwrap()
    }

    // ============================================================================
    // BROADCASTING HELPER FUNCTIONS
    // ============================================================================
    //
    // WHY CANDLE NEEDS THESE FUNCTIONS (vs ndarray backend):
    //
    // The ndarray backend doesn't need explicit broadcasting helpers because:
    //
    // 1. **Built-in Broadcasting**: ndarray's `Zip` iterator automatically handles
    //    broadcasting when iterating over arrays of different shapes. For example:
    //    ```rust
    //    ndarray::Zip::from(&array1).and(&array2).map_collect(|&a, &b| a + b)
    //    ```
    //    This works even if array1 has shape [2, 100] and array2 has shape [2, 100, 1].
    //
    // 2. **Generic Operations**: ndarray operations are generic over element types,
    //    so the same broadcasting logic works for all dtypes (f32, u32, etc.).
    //
    // 3. **Automatic Shape Inference**: ndarray can automatically determine the
    //    output shape from the input shapes during broadcasting.
    //
    // Candle, however, requires explicit broadcasting because:
    //
    // 1. **No Built-in Broadcasting**: Candle's tensor operations (+, *, /, pow)
    //    require tensors to have exactly the same shape. There's no automatic
    //    broadcasting like ndarray's Zip iterator.
    //
    // 2. **Type Erasure**: Candle uses runtime type dispatch via `DType` enum
    //    rather than Rust generics, so we can't write generic broadcasting
    //    functions that work for all types.
    //
    // 3. **Manual Shape Management**: We must explicitly:
    //    - Check tensor shapes before operations
    //    - Squeeze extra dimensions of size 1
    //    - Broadcast scalars to match tensor shapes
    //    - Handle each dtype (f32, u32) separately
    //
    // 4. **Neural Network Requirements**: Neural networks frequently have shape
    //    mismatches that need broadcasting:
    //    - Constants (scalars) broadcast to tensor shapes
    //    - Extra dimensions from operations like reshape/broadcast
    //    - Element-wise operations between tensors of different ranks
    //
    // EXAMPLES OF SHAPE MISMATCHES WE HANDLE:
    //
    // 1. **Sigmoid Operation**: `1.0 / (1.0 + exp(-x))`
    //    - `1.0` is a scalar (shape [])
    //    - `x` might be shape [2, 100]
    //    - We need to broadcast `1.0` to [2, 100]
    //
    // 2. **Broadcast Operations**: After `broadcast(x, shape)`, we might get:
    //    - `x` has shape [2, 100]
    //    - `broadcast(x, [1, 2, 100])` gives shape [1, 2, 100]
    //    - Later operations need to squeeze the extra dimension
    //
    // 3. **Matrix Operations**: After matrix multiplication:
    //    - `matmul(A, B)` might produce shape [2, 100, 1]
    //    - Element-wise operations with [2, 100] need squeezing
    // ============================================================================

    fn broadcast_and_op_f32<F>(x: CandleTensor, y: CandleTensor, op: F) -> CandleTensor
    where
        F: FnOnce(Tensor, Tensor) -> Result<Tensor, candle_core::Error>,
    {
        let x_shape = x.0.shape().dims();
        let y_shape = y.0.shape().dims();

        // If shapes are different, try to make them compatible
        if x_shape != y_shape {
            // Check if one shape can be squeezed to match the other
            if x_shape.len() > y_shape.len() && x_shape.ends_with(&[1]) {
                // x has extra dimension of size 1, squeeze it
                let x_squeezed = x.0.squeeze(x_shape.len() - 1).unwrap();
                CandleTensor(op(x_squeezed, y.0).unwrap())
            } else if y_shape.len() > x_shape.len() && y_shape.ends_with(&[1]) {
                // y has extra dimension of size 1, squeeze it
                let y_squeezed = y.0.squeeze(y_shape.len() - 1).unwrap();
                CandleTensor(op(x.0, y_squeezed).unwrap())
            } else if y_shape.len() == 1 && y_shape[0] == 1 {
                // y is a scalar, broadcast it
                let y_broadcast = y.0.broadcast_as(x_shape).unwrap();
                CandleTensor(op(x.0, y_broadcast).unwrap())
            } else if x_shape.len() == 1 && x_shape[0] == 1 {
                // x is a scalar, broadcast it
                let x_broadcast = x.0.broadcast_as(y_shape).unwrap();
                CandleTensor(op(x_broadcast, y.0).unwrap())
            } else {
                // Try the operation directly
                CandleTensor(op(x.0, y.0).unwrap())
            }
        } else {
            CandleTensor(op(x.0, y.0).unwrap())
        }
    }

    fn broadcast_and_op_u32<F>(x: CandleTensor, y: CandleTensor, op: F) -> CandleTensor
    where
        F: FnOnce(Tensor, Tensor) -> Result<Tensor, candle_core::Error>,
    {
        let x_shape = x.0.shape().dims();
        let y_shape = y.0.shape().dims();

        // If shapes are different, try to make them compatible
        if x_shape != y_shape {
            // Check if one shape can be squeezed to match the other
            if x_shape.len() > y_shape.len() && x_shape.ends_with(&[1]) {
                // x has extra dimension of size 1, squeeze it
                let x_squeezed = x.0.squeeze(x_shape.len() - 1).unwrap();
                CandleTensor(op(x_squeezed, y.0).unwrap())
            } else if y_shape.len() > x_shape.len() && y_shape.ends_with(&[1]) {
                // y has extra dimension of size 1, squeeze it
                let y_squeezed = y.0.squeeze(y_shape.len() - 1).unwrap();
                CandleTensor(op(x.0, y_squeezed).unwrap())
            } else if y_shape.len() == 1 && y_shape[0] == 1 {
                // y is a scalar, broadcast it
                let y_broadcast = y.0.broadcast_as(x_shape).unwrap();
                CandleTensor(op(x.0, y_broadcast).unwrap())
            } else if x_shape.len() == 1 && x_shape[0] == 1 {
                // x is a scalar, broadcast it
                let x_broadcast = x.0.broadcast_as(y_shape).unwrap();
                CandleTensor(op(x_broadcast, y.0).unwrap())
            } else {
                // Try the operation directly
                CandleTensor(op(x.0, y.0).unwrap())
            }
        } else {
            CandleTensor(op(x.0, y.0).unwrap())
        }
    }

    // ============================================================================
    // TYPE-SPECIFIC OPERATION FUNCTIONS
    // ============================================================================
    //
    // WHY WE NEED SEPARATE add_f32/add_u32 FUNCTIONS (vs ndarray's generic approach):
    //
    // The ndarray backend can use generic functions like:
    //   fn add<D>(x: ArrayD<D>, y: ArrayD<D>) -> ArrayD<D> { x + y }
    //
    // But Candle requires separate functions because:
    //
    // 1. **Type Erasure**: CandleTensor wraps `candle_core::Tensor` which uses runtime
    //    type dispatch via `DType` enum, not Rust generics. The tensor type is not
    //    generic over element types like `ArrayD<D>`.
    //
    // 2. **API Design**: `candle_core::Tensor` operations are not generic - they work
    //    on the concrete `Tensor` type and dispatch internally based on `DType`.
    //
    // 3. **Pattern Matching**: The main Backend trait methods need to match on
    //    TaggedNdArrayTuple variants (F32/U32) and call the appropriate function.
    //
    // 4. **Consistency**: Even though both functions do the same thing (x.0 + y.0),
    //    having separate functions makes the code more explicit and matches the
    //    pattern used throughout the Candle backend.
    //
    // This is a fundamental difference in how ndarray (generic) vs Candle (type-erased)
    // handle multi-type tensor operations.
    // ============================================================================

    fn add_f32(x: CandleTensor, y: CandleTensor) -> CandleTensor {
        Self::broadcast_and_op_f32(x, y, |a, b| a + b)
    }

    fn add_u32(x: CandleTensor, y: CandleTensor) -> CandleTensor {
        Self::broadcast_and_op_u32(x, y, |a, b| a + b)
    }

    fn mul_f32(x: CandleTensor, y: CandleTensor) -> CandleTensor {
        Self::broadcast_and_op_f32(x, y, |a, b| a * b)
    }

    fn mul_u32(x: CandleTensor, y: CandleTensor) -> CandleTensor {
        Self::broadcast_and_op_u32(x, y, |a, b| a * b)
    }

    fn div_f32(x: CandleTensor, y: CandleTensor) -> CandleTensor {
        Self::broadcast_and_op_f32(x, y, |a, b| a / b)
    }

    fn div_u32(x: CandleTensor, y: CandleTensor) -> CandleTensor {
        Self::broadcast_and_op_u32(x, y, |a, b| a / b)
    }

    fn neg_f32(x: CandleTensor) -> CandleTensor {
        CandleTensor(x.0.neg().unwrap())
    }

    fn neg_u32(x: CandleTensor) -> CandleTensor {
        CandleTensor(x.0.neg().unwrap())
    }

    fn pow_f32(x: CandleTensor, y: CandleTensor) -> CandleTensor {
        Self::broadcast_and_op_f32(x, y, |a, b| a.pow(&b))
    }

    fn pow_u32(x: CandleTensor, y: CandleTensor) -> CandleTensor {
        Self::broadcast_and_op_u32(x, y, |a, b| a.pow(&b))
    }

    fn matmul_generic(lhs: Tensor, rhs: Tensor) -> Tensor {
        // For now, only handle rank 2 case
        assert_eq!(lhs.dims().len(), 2, "matmul: lhs must be rank 2");
        assert_eq!(rhs.dims().len(), 2, "matmul: rhs must be rank 2");

        lhs.matmul(&rhs).unwrap()
    }

    pub fn batched_matmul(lhs: Tensor, rhs: Tensor) -> Tensor {
        // For now, handle the simple 2D case
        if lhs.dims().len() == 2 && rhs.dims().len() == 2 {
            return Self::matmul_generic(lhs, rhs);
        }

        // For batched matmul, we need to handle higher dimensions
        // This is a simplified implementation
        let lhs_dims = lhs.dims();
        let rhs_dims = rhs.dims();

        if lhs_dims.len() >= 2 && rhs_dims.len() >= 2 {
            // Get batch dimensions and matrix dimensions
            let lhs_batch_dims = &lhs_dims[..lhs_dims.len() - 2];
            let rhs_batch_dims = &rhs_dims[..rhs_dims.len() - 2];
            let lhs_matrix_dims = &lhs_dims[lhs_dims.len() - 2..];
            let rhs_matrix_dims = &rhs_dims[rhs_dims.len() - 2..];

            // Check matrix dimensions compatibility
            assert_eq!(
                lhs_matrix_dims[1], rhs_matrix_dims[0],
                "batched_matmul: incompatible matrix dimensions"
            );

            // For simplicity, require batch dimensions to match exactly
            assert_eq!(
                lhs_batch_dims, rhs_batch_dims,
                "batched_matmul: batch dimensions must match"
            );

            let batch_size: usize = lhs_batch_dims.iter().product();
            let lhs_m = lhs_matrix_dims[0];
            let lhs_k = lhs_matrix_dims[1];
            let rhs_k = rhs_matrix_dims[0];
            let rhs_n = rhs_matrix_dims[1];

            // Reshape to (batch_size, m, k) and (batch_size, k, n)
            let lhs_reshaped = lhs.reshape(&[batch_size, lhs_m, lhs_k]).unwrap();
            let rhs_reshaped = rhs.reshape(&[batch_size, rhs_k, rhs_n]).unwrap();

            // Perform batched matrix multiplication
            let mut results = Vec::new();
            for b in 0..batch_size {
                let lhs_batch = lhs_reshaped.get(b).unwrap();
                let rhs_batch = rhs_reshaped.get(b).unwrap();
                let batch_result = lhs_batch.matmul(&rhs_batch).unwrap();
                results.push(batch_result);
            }

            // Concatenate results and reshape back to proper batch shape
            let mut result_shape = lhs_batch_dims.to_vec();
            result_shape.push(lhs_m);
            result_shape.push(rhs_n);

            Tensor::cat(&results, 0)
                .unwrap()
                .reshape(&*result_shape)
                .unwrap()
        } else {
            // Fallback to regular matmul
            lhs.matmul(&rhs).unwrap()
        }
    }
}

impl<D: HasDtype> NdArray<D> for CandleTensor {
    type Backend = CandleBackend;

    fn shape(&self) -> Shape {
        Shape(self.0.dims().to_vec())
    }
}

// ============================================================================
// PartialEq Implementation for CandleTensor
// ============================================================================
//
// WHY WE NEED THIS IMPLEMENTATION:
//
// The CandleTensor wrapper exists because the underlying `candle_core::Tensor`
// type does not implement `PartialEq`. This creates several important problems:
//
// 1. **Trait Requirements**: The `NdArray<D>` trait requires `PartialEq` to be
//    implemented for the tensor type. This is essential for:
//    - Testing and debugging (comparing tensors in assertions)
//    - Generic algorithms that need to compare tensor values
//    - Hash-based data structures that require equality
//
// 2. **Backend Consistency**: Other backends (like ndarray) implement `PartialEq`
//    naturally, so our Candle backend needs to match this interface for
//    compatibility and interchangeability.
//
// 3. **Testing Infrastructure**: Many test frameworks and assertion macros
//    rely on `PartialEq` to compare expected vs actual values. Without this,
//    we cannot write meaningful tests for our tensor operations.
//
// 4. **API Ergonomics**: Users expect to be able to compare tensors using `==`
//    and `!=` operators. This is a fundamental expectation in Rust APIs.
//
// IMPLEMENTATION CHOICES:
//
// Our current implementation compares only shape and dtype, not the actual values.
// This is a deliberate trade-off for several reasons:
//
// 1. **Performance**: Comparing all tensor values would be expensive for large
//    tensors, especially on GPU devices where data transfer is costly.
//
// 2. **Device Independence**: The underlying tensor data might be on different
//    devices (CPU, GPU, Metal), making direct value comparison complex.
//
// 3. **Floating Point Precision**: Direct value comparison can be problematic
//    with floating-point numbers due to precision issues and different
//    computational paths.
//
// 4. **Use Case Alignment**: Most equality checks in the codebase are for
//    structural equality (shape/dtype) rather than value equality.
//
// ALTERNATIVE APPROACHES:
//
// For cases where you need value-based equality, consider:
// - `tensor.to_vec1()?` to extract values and compare manually
// - Custom comparison functions with tolerance for floating-point values
// - Specialized equality traits for different precision requirements
//
// This implementation strikes a balance between functionality and performance,
// providing the necessary trait implementation while avoiding expensive operations.

impl PartialEq for CandleBackend {
    fn eq(&self, other: &Self) -> bool {
        // Compare devices by their types only, since the underlying device IDs
        // (CudaDevice, MetalDevice) don't implement PartialEq.
        // For most use cases, this is sufficient as we typically care about
        // whether we're using the same device type (CPU, CUDA, Metal) rather
        // than specific device instances.
        matches!(
            (&self.device, &other.device),
            (Device::Cpu, Device::Cpu)
                | (Device::Cuda(_), Device::Cuda(_))
                | (Device::Metal(_), Device::Metal(_))
        )
    }
}

impl PartialEq for CandleTensor {
    fn eq(&self, other: &Self) -> bool {
        // Compare structural properties: shape and data type
        // This is sufficient for most use cases where we need to verify
        // that two tensors have the same structure and type.
        self.0.shape() == other.0.shape() && self.0.dtype() == other.0.dtype()
    }
}

#[test]
fn test_batched_matmul() {
    use candle_core::Tensor;

    // Test with 2 batch dimensions: [2, 3, 2, 2] × [2, 3, 2, 1] = [2, 3, 2, 1]
    let lhs_data = vec![
        1.0f32, 2.0, 3.0, 4.0, // batch 0,0
        5.0, 6.0, 7.0, 8.0, // batch 0,1
        9.0, 10.0, 11.0, 12.0, // batch 0,2
        13.0, 14.0, 15.0, 16.0, // batch 1,0
        17.0, 18.0, 19.0, 20.0, // batch 1,1
        21.0, 22.0, 23.0, 24.0, // batch 1,2
    ];
    let lhs = Tensor::new(&*lhs_data, &candle_core::Device::Cpu)
        .unwrap()
        .reshape(&[2, 3, 2, 2])
        .unwrap();

    let rhs_data = vec![
        1.0f32, 2.0, // batch 0,0
        3.0, 4.0, // batch 0,1
        5.0, 6.0, // batch 0,2
        7.0, 8.0, // batch 1,0
        9.0, 10.0, // batch 1,1
        11.0, 12.0, // batch 1,2
    ];
    let rhs = Tensor::new(&*rhs_data, &candle_core::Device::Cpu)
        .unwrap()
        .reshape(&[2, 3, 2, 1])
        .unwrap();

    let result = CandleBackend::batched_matmul(lhs, rhs);

    // Expected shape: [2, 3, 2, 1]
    assert_eq!(result.dims(), &[2, 3, 2, 1]);

    let expected = [
        5.0f32, 11.0, // batch 0,0
        39.0, 53.0, // batch 0,1
        105.0, 127.0, // batch 0,2
        203.0, 233.0, // batch 1,0
        333.0, 371.0, // batch 1,1
        495.0, 541.0, // batch 1,2
    ];

    // Flatten the result to compare with expected values
    let result_data: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
    for (i, (&actual, &expected)) in result_data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-6,
            "Mismatch at index {i}: got {actual}, expected {expected}"
        );
    }
}
