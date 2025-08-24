use crate::category::core::Shape;
use crate::interpreter::backend::{Backend, DType, NdArray};
use ndarray::{ArrayD, IxDyn};

#[derive(Clone, Debug, PartialEq)]
pub struct NdArrayBackend;

impl Backend for NdArrayBackend {
    type NdArray<D: DType> = ArrayD<D>;

    fn scalar<D: DType>(d: D) -> Self::NdArray<D> {
        ArrayD::from_elem(IxDyn(&[]), d)
    }

    fn zeros<D: DType + Default>(&self, shape: Shape) -> Self::NdArray<D> {
        let dims: Vec<usize> = shape.0;
        ArrayD::from_elem(IxDyn(&dims), D::default())
    }

    fn ndarray_from_slice<D: DType>(&self, data: &[D], shape: Shape) -> Self::NdArray<D> {
        let dims: Vec<usize> = shape.0;
        ArrayD::from_shape_vec(IxDyn(&dims), data.to_vec()).unwrap()
    }

    fn matmul_f32(lhs: Self::NdArray<f32>, rhs: Self::NdArray<f32>) -> Self::NdArray<f32> {
        Self::batched_matmul(lhs, rhs)
    }

    fn matmul_u32(lhs: Self::NdArray<u32>, rhs: Self::NdArray<u32>) -> Self::NdArray<u32> {
        Self::batched_matmul(lhs, rhs)
    }
}

impl NdArrayBackend {
    fn matmul_generic<D>(lhs: ArrayD<D>, rhs: ArrayD<D>) -> ArrayD<D>
    where
        D: DType + ndarray::LinalgScalar,
    {
        // For now, only handle rank 2 case
        assert_eq!(lhs.ndim(), 2, "matmul: self must be rank 2");
        assert_eq!(rhs.ndim(), 2, "matmul: rhs must be rank 2");

        // Convert ArrayD to Array2 for dot product
        // NOTE: ndarray needs to know we have 2d arrays statically to use .dot():
        // https://stackoverflow.com/questions/79035190/
        let self_2d = lhs.into_dimensionality::<ndarray::Ix2>().unwrap();
        let rhs_2d = rhs.into_dimensionality::<ndarray::Ix2>().unwrap();

        // Perform matrix multiplication and convert back to ArrayD
        self_2d.dot(&rhs_2d).into_dyn()
    }

    pub fn batched_matmul<D>(lhs: ArrayD<D>, rhs: ArrayD<D>) -> ArrayD<D>
    where
        D: DType + ndarray::LinalgScalar,
    {
        assert!(
            lhs.ndim() >= 2,
            "batched_matmul: lhs must be at least rank 2"
        );
        assert!(
            rhs.ndim() >= 2,
            "batched_matmul: rhs must be at least rank 2"
        );

        let lhs_shape = lhs.shape().to_vec();
        let rhs_shape = rhs.shape().to_vec();

        if lhs.ndim() == 2 && rhs.ndim() == 2 {
            // Regular matrix multiplication
            return Self::matmul_generic(lhs, rhs);
        }

        // Get the batch dimensions and matrix dimensions
        let lhs_batch_dims = &lhs_shape[..lhs_shape.len() - 2];
        let rhs_batch_dims = &rhs_shape[..rhs_shape.len() - 2];
        let lhs_matrix_dims = &lhs_shape[lhs_shape.len() - 2..];
        let rhs_matrix_dims = &rhs_shape[rhs_shape.len() - 2..];

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
        let lhs_reshaped = lhs.to_shape((batch_size, lhs_m, lhs_k)).unwrap();
        let rhs_reshaped = rhs.to_shape((batch_size, rhs_k, rhs_n)).unwrap();

        // Perform batched matrix multiplication
        let mut result_data = Vec::with_capacity(batch_size * lhs_m * rhs_n);

        for b in 0..batch_size {
            let lhs_batch = lhs_reshaped.slice(ndarray::s![b, .., ..]).to_owned();
            let rhs_batch = rhs_reshaped.slice(ndarray::s![b, .., ..]).to_owned();
            let batch_result = Self::matmul_generic(lhs_batch.into_dyn(), rhs_batch.into_dyn());
            result_data.extend_from_slice(batch_result.as_slice().unwrap());
        }

        // Reshape result back to proper batch shape
        let mut result_shape = lhs_batch_dims.to_vec();
        result_shape.push(lhs_m);
        result_shape.push(rhs_n);

        ArrayD::from_shape_vec(ndarray::IxDyn(&result_shape), result_data).unwrap()
    }
}

impl<D: DType> NdArray<D> for ArrayD<D> {
    type Backend = NdArrayBackend;

    fn shape(&self) -> Shape {
        Shape(self.shape().to_vec())
    }

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
}

#[test]
fn test_batched_matmul() {
    use ndarray::ArrayD;

    // Test with 2 batch dimensions: [2, 3, 2, 2] × [2, 3, 2, 1] = [2, 3, 2, 1]
    let lhs_data = vec![
        1.0f32, 2.0, 3.0, 4.0, // batch 0,0
        5.0, 6.0, 7.0, 8.0, // batch 0,1
        9.0, 10.0, 11.0, 12.0, // batch 0,2
        13.0, 14.0, 15.0, 16.0, // batch 1,0
        17.0, 18.0, 19.0, 20.0, // batch 1,1
        21.0, 22.0, 23.0, 24.0, // batch 1,2
    ];
    let lhs = ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 3, 2, 2]), lhs_data).unwrap();

    let rhs_data = vec![
        1.0f32, 2.0, // batch 0,0
        3.0, 4.0, // batch 0,1
        5.0, 6.0, // batch 0,2
        7.0, 8.0, // batch 1,0
        9.0, 10.0, // batch 1,1
        11.0, 12.0, // batch 1,2
    ];
    let rhs = ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 3, 2, 1]), rhs_data).unwrap();

    let result = NdArrayBackend::batched_matmul(lhs, rhs);

    // Expected shape: [2, 3, 2, 1]
    assert_eq!(result.shape(), &[2, 3, 2, 1]);

    let expected = [
        5.0f32, 11.0, // batch 0,0
        39.0, 53.0, // batch 0,1
        105.0, 127.0, // batch 0,2
        203.0, 233.0, // batch 1,0
        333.0, 371.0, // batch 1,1
        495.0, 541.0, // batch 1,2
    ];

    let result_flat = result.as_slice().unwrap();
    for (i, (&actual, &expected)) in result_flat.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            actual, expected,
            "Mismatch at index {i}: got {actual}, expected {expected}"
        );
    }
}
