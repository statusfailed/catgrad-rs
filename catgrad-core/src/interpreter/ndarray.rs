use crate::category::core::Shape;

#[derive(PartialEq, Debug, Clone)]
pub struct NdArray<T>(pub ndarray::Array<T, ndarray::IxDyn>);

#[derive(PartialEq, Debug, Clone)]
pub enum TaggedNdArray {
    F32(NdArray<f32>),
    U32(NdArray<u32>),
}

impl TaggedNdArray {
    /// Create from a slice with shape
    pub fn from_slice<T: IntoTagged>(data: &[T], shape: &[usize]) -> TaggedNdArray {
        let arr =
            ndarray::Array::from_shape_vec(shape, data.to_vec()).expect("Invalid shape for data");
        T::into_tagged(NdArray(arr.into_dyn()))
    }

    /// Create a scalar (0-dimensional array)
    pub fn scalar<T: IntoTagged>(value: T) -> TaggedNdArray {
        let arr = ndarray::Array::from_elem(ndarray::IxDyn(&[]), value);
        T::into_tagged(NdArray(arr))
    }

    /// Create from existing ndarray
    pub fn from_array<T: IntoTagged>(arr: NdArray<T>) -> TaggedNdArray {
        T::into_tagged(NdArray(arr.0))
    }

    /// Get the shape of the array
    pub fn shape(&self) -> Shape {
        match self {
            TaggedNdArray::F32(arr) => Shape(arr.shape().to_vec()),
            TaggedNdArray::U32(arr) => Shape(arr.shape().to_vec()),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Reduce some boilerplate for TaggedNdArray

/// Trait for types that can be used in TaggedNdArray
pub trait IntoTagged: Clone + PartialEq + std::fmt::Debug {
    fn into_tagged(arr: NdArray<Self>) -> TaggedNdArray;
}

impl IntoTagged for f32 {
    fn into_tagged(arr: NdArray<Self>) -> TaggedNdArray {
        TaggedNdArray::F32(arr)
    }
}

impl IntoTagged for u32 {
    fn into_tagged(arr: NdArray<Self>) -> TaggedNdArray {
        TaggedNdArray::U32(arr)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Passing through traits from ndarray

// Make sure T implements the right traits
fn add_arrays<T>(mut a: ndarray::ArrayD<T>, b: &ndarray::ArrayD<T>) -> ndarray::ArrayD<T>
where
    T: Clone + std::ops::AddAssign<T> + ndarray::ScalarOperand,
{
    a += b;
    a
}

impl<T> std::ops::Add for NdArray<T>
where
    T: Clone + ndarray::LinalgScalar + ndarray::ScalarOperand + std::ops::AddAssign<T>,
{
    type Output = NdArray<T>;

    fn add(self, other: NdArray<T>) -> NdArray<T> {
        NdArray(add_arrays(self.0, &other.0))
    }
}

impl<T> std::ops::Add for &NdArray<T>
where
    T: Clone + ndarray::LinalgScalar,
{
    type Output = NdArray<T>;

    fn add(self, other: &NdArray<T>) -> NdArray<T> {
        NdArray(&self.0 + &other.0)
    }
}

impl<T> std::ops::Mul for &NdArray<T>
where
    T: Clone + ndarray::LinalgScalar,
{
    type Output = NdArray<T>;

    fn mul(self, other: &NdArray<T>) -> NdArray<T> {
        NdArray(&self.0 * &other.0)
    }
}

impl<T> std::ops::Sub for &NdArray<T>
where
    T: Clone + ndarray::LinalgScalar,
{
    type Output = NdArray<T>;

    fn sub(self, other: &NdArray<T>) -> NdArray<T> {
        NdArray(&self.0 - &other.0)
    }
}

impl<T> std::ops::Div for &NdArray<T>
where
    T: Clone + ndarray::LinalgScalar,
{
    type Output = NdArray<T>;

    fn div(self, other: &NdArray<T>) -> NdArray<T> {
        NdArray(&self.0 / &other.0)
    }
}

impl<T> NdArray<T> {
    pub fn new(arr: ndarray::ArrayD<T>) -> Self {
        NdArray(arr)
    }

    pub fn shape(&self) -> &[usize] {
        self.0.shape()
    }

    pub fn ndim(&self) -> usize {
        self.0.ndim()
    }
}

impl<T> std::ops::Index<&[usize]> for NdArray<T> {
    type Output = T;

    fn index(&self, index: &[usize]) -> &Self::Output {
        &self.0[index]
    }
}
