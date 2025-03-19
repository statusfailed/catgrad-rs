use super::ndarray::*;
use crate::backend::cpu::kernel;
use crate::core::{Dtype, Operation, Term};
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
}

impl EvalState {
    /// Preallocate arrays for each node in a term
    pub fn new(f: Term) -> Self {
        Self {
            data: allocate(&f),
            term: f,
        }
    }

    /// Apply an operation to specified sources and target arrays in self.data.
    pub fn apply(&mut self, op: &Operation, sources: &[usize], targets: &[usize]) {
        use Operation::*;
        match op {
            MatrixMultiply {
                dtype: Dtype::F32, ..
            } => {
                let (i, j) = (sources[0], sources[1]);
                let k = targets[0];
                if let Ok([F32(f), F32(g), F32(h)]) = self.data[..].get_disjoint_mut([i, j, k]) {
                    kernel::batch_matmul(f, g, h);
                } else {
                    panic!("invalid types!");
                }
            }

            Add(_) => {
                let (i, j) = (sources[0], sources[1]);
                let k = targets[0];

                if let Ok([F32(a), F32(b), F32(c)]) = self.data[..].get_disjoint_mut([i, j, k]) {
                    for i in 0..a.data.len() {
                        c.data[i] = a.data[i] + b.data[i];
                    }
                } else {
                    panic!("invalid types!");
                }
            }

            // this should be ruled out by typechecking
            op => {
                panic!("unknown operation {:?}", op);
            }
        }
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

    #[test]
    fn test_add() {
        let f = Operation::Add(NdArrayType {
            shape: Shape(vec![2, 2]),
            dtype: Dtype::F32,
        })
        .term();

        let x = NdArray {
            data: vec![1., 2., 3., 4.],
            shape: Shape(vec![2, 2]),
        };
        let y = NdArray {
            data: vec![10., 20., 30., 40.],
            shape: Shape(vec![2, 2]),
        };
        let expected = NdArray {
            data: vec![11., 22., 33., 44.],
            shape: Shape(vec![2, 2]),
        };

        let mut state = EvalState::new(f);

        // TODO: fix hack - API for EvalState?
        state.data[0] = x.into();
        state.data[1] = y.into();

        let [actual] = state.eval()[..] else {
            panic!("unexpected coarity at eval time")
        };

        let tagged: TaggedNdArray = expected.into();
        assert_eq!(&tagged, actual);
    }

    #[test]
    fn test_mat_mul() {
        let f = Operation::MatrixMultiply {
            n: Shape::empty(),
            a: 1,
            b: 2,
            c: 3,
            dtype: Dtype::F32,
        }
        .term();

        // a (1×2) matrix
        let x = NdArray {
            data: vec![2., 4.],
            shape: Shape(vec![1, 2]),
        };

        // a (2×3) matrix
        let m = NdArray {
            data: vec![1., 2., 3., 4., 5., 6.],
            shape: Shape(vec![2, 3]),
        };

        // result should be a 1×3 result
        let mut expected = NdArray {
            data: vec![0.; 3],
            shape: Shape(vec![1, 3]),
        };

        kernel::batch_matmul::<f32>(&x, &m, &mut expected);

        let mut state = EvalState::new(f);

        // TODO: fix hack - API for EvalState?
        state.data[0] = x.into();
        state.data[1] = m.into();

        let [actual] = state.eval()[..] else {
            panic!("unexpected coarity at eval time")
        };

        let tagged: TaggedNdArray = expected.into();
        assert_eq!(&tagged, actual);
    }
}
