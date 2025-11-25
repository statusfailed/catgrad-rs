//! Test that operations can be lowered to text and mlir-opt will accept them.
use catgrad::prelude::*;
use catgrad::stdlib::ops::IntoDtypeVar;
use catgrad::typecheck::*;
use catgrad::{category::lang, stdlib::ops};

use catgrad_mlir::compile::CompiledModel;

use open_hypergraphs::lax::{
    OpenHypergraph,
    var::{BuildResult, build},
};

//use crate::lower::preprocess;

use std::cell::RefCell;
use std::rc::Rc;

fn run_test(term: TypedTerm) {
    let symbol = path(vec!["test", "term"]).unwrap();

    // Make environment
    let mut env = stdlib();
    env.definitions.extend([(symbol.clone(), term)]);

    let params = typecheck::Parameters::from([]);

    // TODO: CompiledModel::new crashes on failure- fix!
    let _ = CompiledModel::new(&env, &params, symbol);
}

#[test]
fn test_shape_op() {
    // shape of *input* tensor
    let x_type = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        shape: ShapeExpr::Shape(vec![3.into(), 1.into(), 4.into()]),
        dtype: Dtype::F32.into(),
    }));
    run_test(
        build_typed_term(
            [x_type],
            [shape_type(3)], // rank 3
            |builder, [x]| vec![ops::shape(builder, x)],
        )
        .unwrap(),
    );
}

#[test]
fn test_cast_u32_f32() {
    let s = vec![3, 1, 4];
    run_test(
        build_typed_term(
            [tensor_type(&s, Dtype::U32)],
            [tensor_type(&s, Dtype::F32)],
            |builder, [x]| {
                let d = Dtype::F32.to_nat(builder);
                vec![ops::cast(builder, x, d)]
            },
        )
        .unwrap(),
    );
}

#[test]
fn test_arange() {
    let s = vec![10];
    run_test(
        build_typed_term([], [tensor_type(&s, Dtype::U32)], |builder, []| {
            vec![ops::arange(builder, 10)]
        })
        .unwrap(),
    );
}

#[test]
fn test_shape_pack() {
    run_test(
        build_typed_term([], [shape_type(3)], |builder, []| {
            let n1 = 2.to_nat(builder);
            let n2 = 3.to_nat(builder);
            let n3 = 4.to_nat(builder);
            vec![ops::pack(builder, [n1, n2, n3])]
        })
        .unwrap(),
    );
}

#[test]
fn test_tensor_index() {
    let input_shape = vec![4, 3]; // 4x3 tensor
    let indices_shape = vec![2]; // 2 indices
    let output_shape = vec![2, 3]; // result: 2x3 tensor (indexed along dim 0)

    run_test(
        build_typed_term(
            [
                tensor_type(&input_shape, Dtype::F32),
                tensor_type(&indices_shape, Dtype::U32),
            ],
            [tensor_type(&output_shape, Dtype::F32)],
            |builder, [input_tensor, indices_tensor]| {
                let dim = 0.to_nat(builder); // Index along dimension 0
                vec![ops::index(builder, dim, indices_tensor, input_tensor)]
            },
        )
        .unwrap(),
    );
}

#[test]
fn test_tensor_transpose() {
    let input_shape = vec![2, 3, 4]; // 2x3x4 tensor
    let output_shape = vec![2, 4, 3]; // transposed: 2x4x3 (dims 1 and 2 swapped)

    run_test(
        build_typed_term(
            [tensor_type(&input_shape, Dtype::F32)],
            [tensor_type(&output_shape, Dtype::F32)],
            |builder, [x]| vec![ops::transpose(builder, 1, 2, x)],
        )
        .unwrap(),
    );
}

////////////////////////////////////////////////////////////////////////////////
// Type helpers

fn tensor_type(shape: &[usize], dtype: Dtype) -> Type {
    Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        shape: shape.to_owned().into(),
        dtype: dtype.into(),
    }))
}

fn shape_type(rank: usize) -> Type {
    use catgrad::typecheck::{NatExpr, ShapeExpr};
    Type::Shape(ShapeExpr::Shape(vec![NatExpr::Constant(rank)]))
}

////////////////////////////////////////////////////////////////////////////////
// Building terms

// Forget type details (for constructing terms)
fn type_to_kind(t: Type) -> lang::Object {
    match t {
        Type::Nat(_) => lang::Object::Nat,
        Type::Dtype(_) => lang::Object::Dtype,
        Type::Shape(_) => lang::Object::Shape,
        Type::Type(_) => lang::Object::NdArrayType,
        Type::Tensor(_) => lang::Object::Tensor,
    }
}

fn build_typed_term<const ARITY: usize, const COARITY: usize, F>(
    source_type: [Type; ARITY],
    target_type: [Type; COARITY],
    f: F,
) -> Option<TypedTerm>
where
    F: Fn(&Rc<RefCell<OpenHypergraph<lang::Object, lang::Operation>>>, [Var; ARITY]) -> Vec<Var>,
{
    let source_kind = source_type.clone().map(type_to_kind);
    let term = build_typed(source_kind, f).ok()?;
    Some(TypedTerm {
        term,
        source_type: source_type.to_vec(),
        target_type: target_type.to_vec(),
    })
}

fn build_typed<const ARITY: usize, F>(
    source_types: [lang::Object; ARITY],
    f: F,
) -> BuildResult<lang::Object, lang::Operation>
where
    F: Fn(&Rc<RefCell<OpenHypergraph<lang::Object, lang::Operation>>>, [Var; ARITY]) -> Vec<Var>,
{
    use std::array;
    build(move |state| {
        // use from_fn to avoid having to clone source_types
        let sources: [Var; ARITY] =
            array::from_fn(|i| Var::new(state.clone(), source_types[i].clone()));

        let sources_vec = sources.iter().cloned().collect();
        let targets = f(state, sources);
        (sources_vec, targets)
    })
}
