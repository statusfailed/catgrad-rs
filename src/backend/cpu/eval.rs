use super::ndarray::*;
use crate::backend::cpu::kernel;
use crate::core::{Operation, Term};
use half::f16;
use std::collections::HashMap;
use Operation::*;
use TaggedNdArray::*;

// TODO: this convenience method should live in open_hypergraphs
use open_hypergraphs::layer::*;
use open_hypergraphs::prelude::*;
fn layered_operations(f: &Term) -> Vec<Vec<usize>> {
    let (order, _unvisited) = layer(f);
    // TODO: check not unvisited.any(|x| x == 1).
    let c = converse(&IndexedCoproduct::elements(order));
    c.into_iter().map(|x| x.table.0).collect()
}

/// For each node in the term `f`, allocate an `NdArray` whose size corresponds to the given
/// shape.
/// NOTE: some operations (like broadcasting) don't actually need to allocate a new array; they are
/// really just "renamings" of the same underlying data. So this allocation strategy should be
/// improved in future.
fn allocate(f: &Term) -> Vec<TaggedNdArray> {
    // Loop over all nodes in the term, allocate an array according to the size/dtype of its
    // labeled NdArrayType.
    let mut result = Vec::with_capacity(f.h.w.len());

    for t in f.h.w.0.iter() {
        let t = TaggedNdArray::from_type(t);
        result.push(t);
    }

    result
}

/// Evaluator state for a single term.
pub struct EvalState {
    term: Term,
    data: Vec<TaggedNdArray>,
    parameters: Option<HashMap<String, TaggedNdArray>>,
}

impl EvalState {
    /// Preallocate arrays for each node in a term
    pub fn new(f: Term) -> Self {
        Self {
            data: allocate(&f),
            term: f,
            parameters: None,
        }
    }

    pub fn set_parameters(&mut self, parameters: HashMap<String, TaggedNdArray>) {
        self.parameters = Some(parameters);
    }

    fn apply_binary_operation(
        &mut self,
        sources: &[usize],
        targets: &[usize],
        operation: &Operation,
    ) {
        let (i, j) = (sources[0], sources[1]);
        let k = targets[0];

        match self.data[..].get_disjoint_mut([i, j, k]) {
            Ok([F16(a), F16(b), F16(c)]) => {
                let op: Box<dyn kernel::BinOp<f16>> = match operation {
                    Add(_) => Box::new(kernel::AddOp),
                    Sub(_) => Box::new(kernel::SubOp),
                    Mul(_) => Box::new(kernel::MulOp),
                    Div(_) => Box::new(kernel::DivOp),
                    Pow(_) => Box::new(kernel::PowOp),
                    MatrixMultiply { .. } => Box::new(kernel::MatMulOp),
                    _ => panic!("invalid operation"),
                };

                op.apply(&*a, &*b, c);
            }
            Ok([F32(a), F32(b), F32(c)]) => {
                let op: Box<dyn kernel::BinOp<f32>> = match operation {
                    Add(_) => Box::new(kernel::AddOp),
                    Sub(_) => Box::new(kernel::SubOp),
                    Mul(_) => Box::new(kernel::MulOp),
                    Div(_) => Box::new(kernel::DivOp),
                    Pow(_) => Box::new(kernel::PowOp),
                    MatrixMultiply { .. } => Box::new(kernel::MatMulOp),
                    _ => panic!("invalid operation"),
                };

                op.apply(&*a, &*b, c);
            }
            Ok([I32(a), I32(b), I32(c)]) => {
                let op: Box<dyn kernel::BinOp<i32>> = match operation {
                    Add(_) => Box::new(kernel::AddOp),
                    Sub(_) => Box::new(kernel::SubOp),
                    Mul(_) => Box::new(kernel::MulOp),
                    Div(_) => Box::new(kernel::DivOp),
                    Pow(_) => Box::new(kernel::PowOp),
                    _ => panic!("invalid operation"),
                };

                op.apply(&*a, &*b, c);
            }
            t => panic!("invalid type: {t:?}"),
        }
    }

    fn apply_unary_operation(
        &mut self,
        sources: &[usize],
        targets: &[usize],
        operation: &Operation,
    ) {
        match self.data[..].get_disjoint_mut([sources[0], targets[0]]) {
            Ok([F16(a), F16(b)]) => {
                let op: Box<dyn kernel::UnaryOp<f16>> = match operation {
                    Negate(_) => Box::new(kernel::NegOp),
                    Reshape { x: _, shape } => Box::new(kernel::ReshapeOp {
                        shape: shape.clone(),
                    }),
                    Broadcast { n, x: _ } => Box::new(kernel::BroadcastOp { n: n.clone() }),
                    Transpose { x: _, dim0, dim1 } => Box::new(kernel::TransposeOp {
                        dim0: *dim0,
                        dim1: *dim1,
                    }),
                    Max => Box::new(kernel::MaxOp),
                    Sum => Box::new(kernel::SumOp),
                    _ => panic!("invalid operation"),
                };

                op.apply(&*a, b);
            }
            Ok([F32(a), F32(b)]) => {
                let op: Box<dyn kernel::UnaryOp<f32>> = match operation {
                    Negate(_) => Box::new(kernel::NegOp),
                    Reshape { x: _, shape } => Box::new(kernel::ReshapeOp {
                        shape: shape.clone(),
                    }),
                    Broadcast { n, x: _ } => Box::new(kernel::BroadcastOp { n: n.clone() }),
                    Transpose { x: _, dim0, dim1 } => Box::new(kernel::TransposeOp {
                        dim0: *dim0,
                        dim1: *dim1,
                    }),
                    Max => Box::new(kernel::MaxOp),
                    Sum => Box::new(kernel::SumOp),
                    _ => panic!("invalid operation"),
                };

                op.apply(&*a, b);
            }
            Ok([I32(a), I32(b)]) => {
                let op: Box<dyn kernel::UnaryOp<i32>> = match operation {
                    Negate(_) => Box::new(kernel::NegOp),
                    Reshape { x: _, shape } => Box::new(kernel::ReshapeOp {
                        shape: shape.clone(),
                    }),
                    Broadcast { n, x: _ } => Box::new(kernel::BroadcastOp { n: n.clone() }),
                    Transpose { x: _, dim0, dim1 } => Box::new(kernel::TransposeOp {
                        dim0: *dim0,
                        dim1: *dim1,
                    }),
                    Max => Box::new(kernel::MaxOp),
                    Sum => Box::new(kernel::SumOp),
                    _ => panic!("invalid operation"),
                };

                op.apply(&*a, b);
            }
            t => panic!("invalid type: {t:?}"),
        }
    }

    /// Apply an operation to specified sources and target arrays in self.data.
    pub fn apply(&mut self, op: &Operation, sources: &[usize], targets: &[usize]) {
        if op.clone().validate().is_none() {
            panic!("invalid operation");
        }
        match op {
            Add(_) | Sub(_) | Mul(_) | Div(_) | Pow(_) | MatrixMultiply { .. } => {
                self.apply_binary_operation(sources, targets, op);
            }
            Sum | Max | Negate(_) | Reshape { .. } | Broadcast { .. } | Transpose { .. } => {
                self.apply_unary_operation(sources, targets, op);
            }
            Copy(_) => match self.data[..].get_disjoint_mut([sources[0], targets[0], targets[1]]) {
                Ok([F32(a), F32(b), F32(c)]) => {
                    b.copy_from(a);
                    c.copy_from(a);
                }
                Ok([F16(a), F16(b), F16(c)]) => {
                    b.copy_from(a);
                    c.copy_from(a);
                }
                Ok([I32(a), I32(b), I32(c)]) => {
                    b.copy_from(a);
                    c.copy_from(a);
                }
                _ => panic!("invalid types"),
            },
            Const { x: _, k } => match self.data.get_mut(targets[0]) {
                Some(F16(a)) => a.fill(f16::from_f32(*k)),
                Some(F32(a)) => a.fill(*k),
                Some(I32(a)) => a.fill(*k as i32),
                _ => panic!("invalid type"),
            },
            Parameter { x: _, name } => {
                // TODO:
                // - The matching here is very ugly and incomplete
                // - The parameters are being copied instead of referenced.
                if let Some(parameters) = self.parameters.as_mut() {
                    match self.data.get_mut(targets[0]) {
                        Some(F32(a)) => {
                            let p = parameters.get(name);
                            if let Some(TaggedNdArray::F32(x)) = p {
                                a.copy_from(x);
                            } else {
                                panic!("Parameters loaded, parameter '{name}'::F32 not found.")
                            }
                        }
                        _ => panic!("Invalid type for parameter '{name}'"),
                    }
                } else {
                    panic!("Parameters not loaded, requested parameter '{name}'");
                }
            }
        }
    }

    /// mutably evaluate self with args, returning a reference to output arrays.
    pub fn eval_with(&mut self, args: Vec<TaggedNdArray>) -> Vec<&TaggedNdArray> {
        let sources = &self.term.s.table;

        assert_eq!(
            args.len(),
            sources.len(),
            "Expected {} arguments but got {}.",
            sources.len(),
            args.len()
        );
        for (i, arg) in args.iter().enumerate() {
            self.data[sources[i]] = arg.clone();
        }

        self.eval()
    }

    /// mutably evaluate self, returning a reference to output arrays.
    pub fn eval(&mut self) -> Vec<&TaggedNdArray> {
        // unpack the segmented array of sources into a vec of vecs.
        // TODO: this clones the value - provide non-cloning iter, or one that returns slices?
        #[rustfmt::skip]
        let sources: Vec<Vec<usize>> = self.term.h.s.clone().into_iter().map(|x| x.table.0).collect();
        #[rustfmt::skip]
        let targets: Vec<Vec<usize>> = self.term.h.t.clone().into_iter().map(|x| x.table.0).collect();

        for ops in layered_operations(&self.term).iter() {
            // each layer has any number of ops. TODO: evaluate these in parallel!
            for i in ops {
                let op = self.term.h.x.0[*i].clone();
                self.apply(&op, &sources[*i], &targets[*i]);
            }
        }

        // Return result array ptrs
        self.term.t.table.0.iter().map(|i| &self.data[*i]).collect()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::core::{Dtype, NdArrayType, Operation, Shape};

    fn test_unarynop_generic<T>(op_type: Operation, x_data: Vec<T>, expected_data: Vec<T>)
    where
        TaggedNdArray: From<NdArray<T>>,
    {
        let f = op_type.term();

        let x = NdArray::new(x_data, Shape(vec![2, 2]));
        let expected = NdArray::new(expected_data, Shape(vec![2, 2]));

        let mut state = EvalState::new(f);

        let [actual] = state.eval_with(vec![x.into()])[..] else {
            panic!("unexpected coarity at eval time")
        };

        let tagged: TaggedNdArray = expected.into();
        assert_eq!(&tagged, actual);
    }

    #[test]
    fn test_neg() {
        test_unarynop_generic::<f16>(
            Operation::Negate(NdArrayType {
                shape: Shape(vec![2, 2]),
                dtype: Dtype::F16,
            }),
            vec![1.0, 2.0, 3.0, 4.0]
                .iter()
                .map(|&x| f16::from_f32(x))
                .collect(),
            vec![-1.0, -2.0, -3.0, -4.0]
                .iter()
                .map(|&x| f16::from_f32(x))
                .collect(),
        );
        test_unarynop_generic::<f32>(
            Operation::Negate(NdArrayType {
                shape: Shape(vec![2, 2]),
                dtype: Dtype::F32,
            }),
            vec![1.0, 2.0, 3.0, 4.0],
            vec![-1.0, -2.0, -3.0, -4.0],
        );
        test_unarynop_generic::<i32>(
            Operation::Negate(NdArrayType {
                shape: Shape(vec![2, 2]),
                dtype: Dtype::I32,
            }),
            vec![1, 2, 3, 4],
            vec![-1, -2, -3, -4],
        );
    }

    fn test_binop_generic<T>(
        op_type: Operation,
        x_data: Vec<T>,
        y_data: Vec<T>,
        expected_data: Vec<T>,
    ) where
        TaggedNdArray: From<NdArray<T>>,
    {
        let f = op_type.term();
        let x = NdArray::new(x_data, Shape(vec![2, 2]));
        let y = NdArray::new(y_data, Shape(vec![2, 2]));
        let expected = NdArray::new(expected_data, Shape(vec![2, 2]));

        let mut state = EvalState::new(f);

        let [actual] = state.eval_with(vec![x.into(), y.into()])[..] else {
            panic!("unexpected coarity at eval time")
        };

        let tagged: TaggedNdArray = expected.into();
        assert_eq!(&tagged, actual);
    }

    #[test]
    #[should_panic]
    fn test_eval_with_argcount() {
        let f = Operation::Add(NdArrayType {
            shape: Shape(vec![2, 2]),
            dtype: Dtype::F32,
        })
        .term();

        let x = NdArray::new(vec![0.; 4], Shape(vec![2, 2]));
        let mut state = EvalState::new(f);

        // Passing a single argument into a binary
        let [_] = state.eval_with(vec![x.into()])[..] else {
            panic!("unexpected coarity at eval time")
        };
    }

    #[test]
    fn test_add() {
        test_binop_generic::<f16>(
            Operation::Add(NdArrayType {
                shape: Shape(vec![2, 2]),
                dtype: Dtype::F16,
            }),
            vec![1.0, 2.0, 3.0, 4.0]
                .iter()
                .map(|&x| f16::from_f32(x))
                .collect(),
            vec![10.0, 20.0, 30.0, 40.0]
                .iter()
                .map(|&x| f16::from_f32(x))
                .collect(),
            vec![11.0, 22.0, 33.0, 44.0]
                .iter()
                .map(|&x| f16::from_f32(x))
                .collect(),
        );

        test_binop_generic::<f32>(
            Operation::Add(NdArrayType {
                shape: Shape(vec![2, 2]),
                dtype: Dtype::F32,
            }),
            vec![1.0, 2.0, 3.0, 4.0],
            vec![10.0, 20.0, 30.0, 40.0],
            vec![11.0, 22.0, 33.0, 44.0],
        );

        // Test for I32
        test_binop_generic::<i32>(
            Operation::Add(NdArrayType {
                shape: Shape(vec![2, 2]),
                dtype: Dtype::I32,
            }),
            vec![1, 2, 3, 4],
            vec![10, 20, 30, 40],
            vec![11, 22, 33, 44],
        );
    }

    #[test]
    fn test_sub() {
        // Test subtraction with F32
        test_binop_generic::<f32>(
            Operation::Sub(NdArrayType {
                shape: Shape(vec![2, 2]),
                dtype: Dtype::F32,
            }),
            vec![10.0, 20.0, 30.0, 40.0],
            vec![1.0, 2.0, 3.0, 4.0],
            vec![9.0, 18.0, 27.0, 36.0],
        );

        // Test subtraction with I32
        test_binop_generic::<i32>(
            Operation::Sub(NdArrayType {
                shape: Shape(vec![2, 2]),
                dtype: Dtype::I32,
            }),
            vec![10, 20, 30, 40],
            vec![1, 2, 3, 4],
            vec![9, 18, 27, 36],
        );
    }

    #[test]
    fn test_mul() {
        // Test multiplication with F32
        test_binop_generic::<f32>(
            Operation::Mul(NdArrayType {
                shape: Shape(vec![2, 2]),
                dtype: Dtype::F32,
            }),
            vec![2.0, 3.0, 4.0, 5.0],
            vec![10.0, 20.0, 30.0, 40.0],
            vec![20.0, 60.0, 120.0, 200.0],
        );

        // Test multiplication with I32
        test_binop_generic::<i32>(
            Operation::Mul(NdArrayType {
                shape: Shape(vec![2, 2]),
                dtype: Dtype::I32,
            }),
            vec![2, 3, 4, 5],
            vec![10, 20, 30, 40],
            vec![20, 60, 120, 200],
        );
    }

    #[test]
    fn test_div() {
        // Test division with F32
        test_binop_generic::<f32>(
            Operation::Div(NdArrayType {
                shape: Shape(vec![2, 2]),
                dtype: Dtype::F32,
            }),
            vec![2.0, 4.0, 6.0, 8.0],
            vec![2.0, 2.0, 2.0, 2.0],
            vec![1.0, 2.0, 3.0, 4.0],
        );

        // Test division with I32
        test_binop_generic::<i32>(
            Operation::Div(NdArrayType {
                shape: Shape(vec![2, 2]),
                dtype: Dtype::I32,
            }),
            vec![2, 4, 6, 8],
            vec![2, 2, 2, 2],
            vec![1, 2, 3, 4],
        );
    }

    #[test]
    fn test_pow() {
        // Test raising to a power with F32
        test_binop_generic::<f32>(
            Operation::Pow(NdArrayType {
                shape: Shape(vec![2, 2]),
                dtype: Dtype::F32,
            }),
            vec![2.0, 4.0, 6.0, 8.0],
            vec![2.0, 2.0, 2.0, 2.0],
            vec![4.0, 16.0, 36.0, 64.0],
        );

        // Test raising to a power with F16
        test_binop_generic::<f16>(
            Operation::Pow(NdArrayType {
                shape: Shape(vec![2, 2]),
                dtype: Dtype::F16,
            }),
            vec![2.0, 4.0, 6.0, 8.0]
                .iter()
                .map(|&x| f16::from_f32(x))
                .collect(),
            vec![2.0, 2.0, 2.0, 2.0]
                .iter()
                .map(|&x| f16::from_f32(x))
                .collect(),
            vec![4.0, 16.0, 36.0, 64.0]
                .iter()
                .map(|&x| f16::from_f32(x))
                .collect(),
        );

        // Test raising to a power with I32
        test_binop_generic::<i32>(
            Operation::Pow(NdArrayType {
                shape: Shape(vec![2, 2]),
                dtype: Dtype::I32,
            }),
            vec![2, 4, 6, 8],
            vec![2, 2, 2, 2],
            vec![4, 16, 36, 64],
        );
    }

    #[test]
    fn test_matmul() {
        let f = Operation::MatrixMultiply {
            n: Shape::empty(),
            a: 1,
            b: 2,
            c: 3,
            dtype: Dtype::F32,
        }
        .term();

        // a (1×2) matrix
        let x = NdArray::new(vec![2., 4.], Shape(vec![1, 2]));

        // a (2×3) matrix
        let m = NdArray::new(vec![1., 2., 3., 4., 5., 6.], Shape(vec![2, 3]));

        // result should be a 1×3 result
        let mut expected = NdArray::new(vec![0.; 3], Shape(vec![1, 3]));

        kernel::batch_matmul::<f32>(&x, &m, &mut expected);

        let mut state = EvalState::new(f);

        let [actual] = state.eval_with(vec![x.into(), m.into()])[..] else {
            panic!("unexpected coarity at eval time")
        };

        let tagged: TaggedNdArray = expected.into();
        assert_eq!(&tagged, actual);
    }

    #[test]
    fn test_matmul_transposed() {
        let f = Operation::MatrixMultiply {
            n: Shape::empty(),
            a: 1,
            b: 2,
            c: 3,
            dtype: Dtype::F32,
        }
        .term();

        // a (1×2) matrix
        let x = NdArray::new(vec![2., 4.], Shape(vec![1, 2]));

        // a (2×3) matrix
        let m = NdArray::new(vec![1., 2., 3., 4., 5., 6.], Shape(vec![2, 3]));

        // Transposed equivalent of m
        let mut mt = NdArray::new(vec![1., 4., 2., 5., 3., 6.], Shape(vec![3, 2]));
        mt.shape = Shape(vec![2, 3]);
        mt.strides = vec![1, 2];

        // result should be a 1×3 result
        let mut expected = NdArray::new(vec![0.; 3], Shape(vec![1, 3]));

        kernel::batch_matmul::<f32>(&x, &m, &mut expected);

        let mut state = EvalState::new(f);

        let [actual] = state.eval_with(vec![x.into(), mt.into()])[..] else {
            panic!("unexpected coarity at eval time")
        };

        let tagged: TaggedNdArray = expected.into();
        assert_eq!(&tagged, actual);
    }

    #[test]
    fn test_const() {
        let f = Operation::Const {
            x: NdArrayType {
                shape: Shape(vec![4, 3]),
                dtype: Dtype::F32,
            },
            k: 2.3,
        }
        .term();

        let expected = NdArray::new(vec![2.3; 12], Shape(vec![4, 3]));

        let mut state = EvalState::new(f);

        let [actual] = state.eval()[..] else {
            panic!("unexpected coarity at eval time")
        };

        assert_eq!(actual, &expected.into());
    }

    #[test]
    fn test_parameter() {
        let typ = NdArrayType {
            shape: Shape(vec![2, 2]),
            dtype: Dtype::F32,
        };

        let param_a = Operation::Parameter {
            x: typ.clone(),
            name: "param_a".to_string(),
        }
        .term();

        let param_b = Operation::Parameter {
            x: typ.clone(),
            name: "param_b".to_string(),
        }
        .term();

        let add = Operation::Add(NdArrayType {
            shape: Shape(vec![2, 2]),
            dtype: Dtype::F32,
        })
        .term();

        let a = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], Shape(vec![2, 2]));
        let b = NdArray::new(vec![-1.0, 2.0, -3.0, 4.0], Shape(vec![2, 2]));

        let mut parameters = HashMap::new();
        parameters.insert("param_a".to_string(), a.into());
        parameters.insert("param_b".to_string(), b.into());

        let expected = NdArray::new(vec![0., 4., 0., 8.], Shape(vec![2, 2]));

        let f = (&(&param_a | &param_b) >> &add).unwrap();
        let mut state = EvalState::new(f);
        state.set_parameters(parameters);

        let [actual] = state.eval()[..] else {
            panic!("unexpected coarity at eval time")
        };

        assert_eq!(actual, &expected.into());
    }

    #[test]
    #[should_panic]
    fn test_missing_parameter() {
        let typ = NdArrayType {
            shape: Shape(vec![2, 2]),
            dtype: Dtype::F16,
        };

        let param_a = Operation::Parameter {
            x: typ.clone(),
            name: "param_a".to_string(),
        }
        .term();

        let mut state = EvalState::new(param_a);
        let parameters = HashMap::new();
        state.set_parameters(parameters);

        let [_] = state.eval()[..] else {
            panic!("unexpected coarity at eval time")
        };
    }

    #[test]
    #[should_panic]
    fn test_missing_parameters() {
        let typ = NdArrayType {
            shape: Shape(vec![2, 2]),
            dtype: Dtype::F32,
        };

        let param_a = Operation::Parameter {
            x: typ.clone(),
            name: "param_a".to_string(),
        }
        .term();

        let mut state = EvalState::new(param_a);

        let [_] = state.eval()[..] else {
            panic!("unexpected coarity at eval time")
        };
    }

    #[test]
    fn test_reshape() {
        let f = Operation::Reshape {
            x: NdArrayType {
                shape: Shape(vec![4, 3]),
                dtype: Dtype::I32,
            },
            shape: Shape(vec![2, 6]),
        }
        .term();

        let x = NdArray::new((0..12).collect(), Shape(vec![4, 3]));

        let expected = NdArray::new((0..12).collect(), Shape(vec![2, 6]));

        let mut state = EvalState::new(f);

        let [actual] = state.eval_with(vec![x.into()])[..] else {
            panic!("unexpected coarity at eval time")
        };

        assert_eq!(actual, &expected.into());
    }

    #[test]
    fn test_broadcast() {
        let f = Operation::Broadcast {
            x: NdArrayType {
                shape: Shape(vec![2, 3]),
                dtype: Dtype::I32,
            },
            n: Shape(vec![2, 1]),
        }
        .term();

        let x = NdArray::new((30..36).collect(), Shape(vec![2, 3]));

        let mut state = EvalState::new(f);

        let [actual] = state.eval_with(vec![x.into()])[..] else {
            panic!("unexpected coarity at eval time")
        };

        // check that array is broadcasted across two new dimensions
        if let TaggedNdArray::I32(actual) = actual {
            assert_eq!(actual.shape.0, &[2, 1, 2, 3]);
            assert_eq!(actual.strides, [0, 0, 3, 1]);

            assert_eq!(actual[&[0, 0, 0, 0]], 30);
            assert_eq!(actual[&[1, 0, 0, 0]], 30);
            assert_eq!(actual[&[0, 0, 0, 2]], 32);
            assert_eq!(actual[&[1, 0, 0, 2]], 32);
            assert_eq!(actual[&[0, 0, 1, 2]], 35);
            assert_eq!(actual[&[1, 0, 1, 2]], 35);
        }
    }
    #[test]
    fn test_transpose() {
        let f = Operation::Transpose {
            x: NdArrayType {
                shape: Shape(vec![2, 3]),
                dtype: Dtype::F32,
            },
            dim0: 0,
            dim1: 1,
        }
        .term();

        // Create a 2x3 matrix
        let input = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape(vec![2, 3]));

        let mut state = EvalState::new(f);
        let [actual] = state.eval_with(vec![input.into()])[..] else {
            panic!("unexpected coarity at eval time")
        };
        match actual {
            TaggedNdArray::F32(array) => {
                assert_eq!(array.shape, Shape(vec![3, 2]));
                assert_eq!(array[&[0, 0]], 1.0);
                assert_eq!(array[&[0, 1]], 4.0);
                assert_eq!(array[&[1, 0]], 2.0);
                assert_eq!(array[&[1, 1]], 5.0);
                assert_eq!(array[&[2, 0]], 3.0);
                assert_eq!(array[&[2, 1]], 6.0);
            }
            _ => panic!("Expected F32 array"),
        }
    }

    #[test]
    fn test_copy_from() {
        let f = Operation::Copy(NdArrayType {
            shape: Shape(vec![2, 2]),
            dtype: Dtype::F32,
        })
        .term();

        let x = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], Shape(vec![2, 2]));

        let tagged: TaggedNdArray = x.into();

        let mut state = EvalState::new(f);

        let [copy1, copy2] = state.eval_with(vec![tagged.clone()])[..] else {
            panic!("unexpected coarity at eval time")
        };

        assert_eq!(copy1, &tagged);
        assert_eq!(copy2, &tagged);
    }

    #[test]
    fn test_max() {
        let f = Operation::max(NdArrayType {
            shape: Shape(vec![2, 2]),
            dtype: Dtype::F32,
        });

        let x = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], Shape(vec![2, 2]));

        let expected = NdArray::new(vec![2.0, 4.0], Shape(vec![2]));

        let mut state = EvalState::new(f);

        let [actual] = state.eval_with(vec![x.into()])[..] else {
            panic!("unexpected coarity at eval time")
        };

        assert_eq!(actual, &expected.into());
    }

    #[test]
    fn test_sum() {
        let f = Operation::sum(NdArrayType {
            shape: Shape(vec![2, 3]),
            dtype: Dtype::F32,
        });

        let x = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape(vec![2, 3]));

        let expected = NdArray::new(vec![6.0, 15.0], Shape(vec![2]));

        let mut state = EvalState::new(f);

        let [actual] = state.eval_with(vec![x.into()])[..] else {
            panic!("unexpected coarity at eval time")
        };

        assert_eq!(actual, &expected.into());
    }
}
