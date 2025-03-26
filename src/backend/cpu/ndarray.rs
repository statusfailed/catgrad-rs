use crate::core::object::*;

use half::f16;
use num_traits::Zero;

/// N-dimensional arrays of elements T with a given shape.
#[derive(PartialEq, Debug, Clone)]
pub struct NdArray<T> {
    pub data: Vec<T>, // raw data of the array
    pub shape: Shape, // shape information (erasable?)
}

/// Immutable slice into an NdArray
pub struct NdArraySlice<'a, T> {
    pub data: &'a [T],
    pub shape: Shape,
}

/// Mutable slice into an NdArray
pub struct NdArrayMutSlice<'a, T> {
    pub data: &'a mut [T],
    pub shape: Shape,
}

impl<T> NdArray<T> {
    pub fn new(data: Vec<T>, shape: Shape) -> Self {
        assert_eq!(
            data.len(),
            shape.size(),
            "Data length must match shape size"
        );
        NdArray { data, shape }
    }

    pub fn is_empty(&self) -> bool {
        self.shape.size() == 0
    }

    pub fn len(&self) -> usize {
        self.shape.size()
    }

    /// Compute slice indices for fixed first n dimension.
    /// Returns (start_index, slice_shape)
    fn calculate_slice_info(&self, indices: &[usize]) -> (usize, Shape) {
        if indices.len() > self.shape.0.len() {
            panic!("Too many indices provided");
        }

        // Check that indices are within bounds
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape.0[i] {
                panic!(
                    "Index {} out of bounds for dimension {} with size {}",
                    idx, i, self.shape.0[i]
                );
            }
        }

        // If no indices provided, use the whole array
        if indices.is_empty() {
            return (0, self.shape.clone());
        }

        // Compute the start offset in the flattened array
        let mut start_index = 0;
        let mut stride = 1;

        // Start with the innermost dimension
        for i in (0..self.shape.0.len()).rev() {
            if i >= indices.len() {
                // This dimension is part of the slice, we start at index 0
                // No need to add anything to start_index
            } else {
                // This dimension is fixed by the provided indices
                start_index += indices[i] * stride;
            }
            // Update stride for the next (more significant) dimension
            stride *= self.shape.0[i];
        }

        // Compute the shape of the resulting slice
        let slice_shape = Shape(self.shape.0[indices.len()..].to_vec());

        (start_index, slice_shape)
    }

    /// Creates an immutable slice from the array by fixing the first n dimensions
    ///
    /// # Example
    /// ```
    /// use catgrad::backend::cpu::ndarray::NdArray;
    /// use catgrad::core::object::Shape;
    /// use num_traits::Zero;
    ///
    /// let array: NdArray<f32> = NdArray::from_shape(Shape(vec![2, 3, 4, 5]));
    /// let slice = array.slice(&[0, 1]); // Slice with shape [4, 5]
    /// ```
    pub fn slice<'a>(&'a self, indices: &[usize]) -> NdArraySlice<'a, T> {
        let (start_index, slice_shape) = self.calculate_slice_info(indices);
        let slice_len = slice_shape.size();

        // Create slice from data
        let data_slice = &self.data[start_index..(start_index + slice_len)];

        NdArraySlice {
            data: data_slice,
            shape: slice_shape,
        }
    }

    /// Creates a mutable slice from the array by fixing the first n dimensions
    ///
    /// # Example
    /// ```
    /// use catgrad::backend::cpu::ndarray::NdArray;
    /// use catgrad::core::object::Shape;
    /// use num_traits::Zero;
    ///
    /// let mut array: NdArray<f32> = NdArray::from_shape(Shape(vec![2, 3, 4, 5]));
    /// let slice = array.slice_mut(&[0, 1]); // Slice with shape [4, 5]
    /// ```
    pub fn slice_mut<'a>(&'a mut self, indices: &[usize]) -> NdArrayMutSlice<'a, T> {
        let (start_index, slice_shape) = self.calculate_slice_info(indices);
        let slice_len = slice_shape.size();

        // Create slice from data
        let data_slice = &mut self.data[start_index..(start_index + slice_len)];

        NdArrayMutSlice {
            data: data_slice,
            shape: slice_shape,
        }
    }
}

impl<T: Clone + Zero> NdArray<T> {
    pub fn from_shape(shape: Shape) -> Self {
        // TODO: don't really need to initialize to zero; is there a better way here? bytemuck?
        NdArray::new(vec![T::zero(); shape.size()], shape)
    }
    pub fn fill(&mut self, value: T) {
        for i in 0..self.data.len() {
            self.data[i] = value.clone();
        }
    }
}

/// A disjoint union of typed arrays
/// intuition: the "values" assigned to each node in a [`Term`].
#[derive(PartialEq, Debug, Clone)]
pub enum TaggedNdArray {
    F16(NdArray<f16>),
    F32(NdArray<f32>),
    I32(NdArray<i32>),
}

impl TaggedNdArray {
    pub fn is_empty(&self) -> bool {
        match self {
            TaggedNdArray::F16(vec) => vec.is_empty(),
            TaggedNdArray::F32(vec) => vec.is_empty(),
            TaggedNdArray::I32(vec) => vec.is_empty(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            TaggedNdArray::F16(vec) => vec.len(),
            TaggedNdArray::F32(vec) => vec.len(),
            TaggedNdArray::I32(vec) => vec.len(),
        }
    }

    pub fn from_type(t: &NdArrayType) -> Self {
        match t.dtype {
            Dtype::F16 => TaggedNdArray::F16(NdArray::from_shape(t.shape.clone())),
            Dtype::F32 => TaggedNdArray::F32(NdArray::from_shape(t.shape.clone())),
            Dtype::I32 => TaggedNdArray::I32(NdArray::from_shape(t.shape.clone())),
        }
    }
}

impl From<NdArray<f16>> for TaggedNdArray {
    fn from(value: NdArray<f16>) -> Self {
        TaggedNdArray::F16(value)
    }
}
impl From<NdArray<f32>> for TaggedNdArray {
    fn from(value: NdArray<f32>) -> Self {
        TaggedNdArray::F32(value)
    }
}

impl From<NdArray<i32>> for TaggedNdArray {
    fn from(value: NdArray<i32>) -> Self {
        TaggedNdArray::I32(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice() {
        // Create a 2x3x4 array filled with zeros
        let mut array = NdArray::new(vec![0.0; 24], Shape(vec![2, 3, 4]));

        // Fill the array with a simple pattern for testing
        for i in 0..24 {
            array.data[i] = i as f32;
        }

        // Calculate the expected linear indices for a 2x3x4 array with row-major ordering
        // For [0,1,...] we should get indices [4,5,6,7]

        // Get a slice at [0, 1, ...]
        let slice = array.slice(&[0, 1]);

        // Check the shape of the slice
        assert_eq!(slice.shape, Shape(vec![4]));

        // Check the data in the slice
        assert_eq!(slice.data, &[4.0, 5.0, 6.0, 7.0]);

        // Test with different indices [1, 2, ...]
        let slice2 = array.slice(&[1, 2]);
        assert_eq!(slice2.shape, Shape(vec![4]));
        assert_eq!(slice2.data, &[20.0, 21.0, 22.0, 23.0]);

        // Test slicing with no indices (should return the whole array)
        let slice_all = array.slice(&[]);
        assert_eq!(slice_all.shape, Shape(vec![2, 3, 4]));
        assert_eq!(slice_all.data.len(), 24);

        // Test slicing a 4D array
        let mut array4d = NdArray::new(vec![0.0; 120], Shape(vec![2, 3, 4, 5]));

        for i in 0..120 {
            array4d.data[i] = i as f32;
        }

        let slice4d = array4d.slice(&[0, 0]);
        assert_eq!(slice4d.shape, Shape(vec![4, 5]));
        assert_eq!(slice4d.data.len(), 20);

        // Check the data in the 4D slice
        // In a 2x3x4x5 array, [0,0,...] should give the first 20 elements
        for i in 0..20 {
            assert_eq!(slice4d.data[i], i as f32);
        }
    }

    #[test]
    fn test_slice_mut() {
        // Create a 2x3x4 array filled with zeros
        let mut array = NdArray::new(vec![0.0; 24], Shape(vec![2, 3, 4]));

        // Fill the array with a simple pattern for testing
        for i in 0..24 {
            array.data[i] = i as f32;
        }

        // Calculate the expected linear indices for a 2x3x4 array with row-major ordering
        // For [0,1,...] we should get indices [4,5,6,7]

        // Get a slice at [0, 1, ...]
        let slice = array.slice_mut(&[0, 1]);

        // Check the shape of the slice
        assert_eq!(slice.shape, Shape(vec![4]));

        // Check the data in the slice
        assert_eq!(slice.data, &mut [4.0, 5.0, 6.0, 7.0]);

        // Modify the slice and check that it affects the original array
        slice.data[2] = 99.0;
        assert_eq!(array.data[6], 99.0);

        // Test with different indices [1, 2, ...]
        let slice2 = array.slice_mut(&[1, 2]);
        assert_eq!(slice2.shape, Shape(vec![4]));
        assert_eq!(slice2.data, &mut [20.0, 21.0, 22.0, 23.0]);

        // Test slicing with no indices (should return the whole array)
        let slice_all = array.slice_mut(&[]);
        assert_eq!(slice_all.shape, Shape(vec![2, 3, 4]));
        assert_eq!(slice_all.data.len(), 24);

        // Test slicing a 4D array
        let mut array4d = NdArray::new(vec![0.0; 120], Shape(vec![2, 3, 4, 5]));

        for i in 0..120 {
            array4d.data[i] = i as f32;
        }

        let slice4d = array4d.slice_mut(&[0, 0]);
        assert_eq!(slice4d.shape, Shape(vec![4, 5]));
        assert_eq!(slice4d.data.len(), 20);

        // Check the data in the 4D slice
        // In a 2x3x4x5 array, [0,0,...] should give the first 20 elements
        for i in 0..20 {
            assert_eq!(slice4d.data[i], i as f32);
        }
    }
}
