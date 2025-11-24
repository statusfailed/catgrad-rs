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
    // TODO: don't actually write the .so if we can avoid it!
    // Might need to split `compile` implementation into two procedures
    let output_so = "./test.so".into();

    let symbol = path(vec!["test", "term"]).unwrap();

    // Make environment
    let mut env = stdlib();
    env.definitions.extend([(symbol.clone(), term)]);

    let params = typecheck::Parameters::from([]);

    // TODO: CompiledModel::new crashes on failure- fix!
    let _ = CompiledModel::new(&env, &params, symbol, output_so);
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

////////////////////////////////////////////////////////////////////////////////
// Type helpers

fn tensor_type(shape: &Vec<usize>, dtype: Dtype) -> Type {
    Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        shape: shape.clone().into(),
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
