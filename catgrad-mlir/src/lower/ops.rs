use catgrad::ssa::SSA;
use catgrad::{
    category::lang,
    typecheck::{DtypeExpr, NatExpr, NdArrayType, ShapeExpr, Type, TypeExpr},
};

use super::grammar;
use super::util::*;

// NOTE: this just usese arith.constant false to return a bool.
// We don't actually put shape information into mlir
pub fn shape(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Assignment> {
    // typechecked ssa should never break this invariant
    assert!(ssa.sources.len() == 1);
    assert!(ssa.targets.len() == 1);
    let _target_type = core_type_to_mlir(&ssa.targets[0].1);

    vec![
        grammar::Expr::Constant(grammar::Constant {
            name: "arith.constant".to_string(),
            value: Some("false".to_string()),
            ty: None,
        })
        .into_assignment(ssa),
    ]
}

pub fn broadcast(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Statement> {
    assert!(ssa.sources.len() == 2);
    assert!(ssa.targets.len() == 1);

    // Irrefutable matches because typecheck should catch errors
    let Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        shape: ShapeExpr::Shape(in_shape),
        ..
    })) = &ssa.sources[0].1
    else {
        panic!()
    };

    let Type::Shape(ShapeExpr::Shape(out_shape)) = &ssa.sources[1].1 else {
        panic!()
    };

    let tensor_id = grammar::Identifier(ssa.sources[0].0.0);
    let target_id = grammar::Identifier(ssa.targets[0].0.0);

    // Generate indexing maps for broadcasting
    let out_rank = out_shape.len();

    // Make affine map for *broadcasted* tensor
    //      (d_0..d_{out_rank}) -> (trailing dims, with 1s replaced with literal 0)
    // NOTE: we need to handle
    let input_dims: Vec<_> = in_shape
        .iter()
        .map(|nat| match nat {
            NatExpr::Constant(n) => Some(*n),
            _ => None,
        })
        .collect();

    let input_affine_map = make_affine_map(input_dims, out_rank);

    // make affine map for *output* tensor
    //      (d_0..d_{out_rank}) → (d_0..d_{out_rank})
    let output_affine_map = make_affine_map(vec![None; out_rank], out_rank);

    let attrs = format!(
        "indexing_maps = [{}, {}], iterator_types = [{}]",
        input_affine_map,
        output_affine_map,
        vec!["\"parallel\""; out_rank].join(", "),
    );

    let tensor_type = core_type_to_mlir(&ssa.sources[0].1);
    let target_type = core_type_to_mlir(&ssa.targets[0].1);

    let empty_tensor = grammar::Statement::Custom(format!(
        "  %v{}_out = tensor.empty() : {}",
        target_id.0, target_type
    ));

    let linalg_generic = grammar::Statement::Custom(format!(
        "  {} = linalg.generic {{{}}} ins({} : {}) outs(%v{}_out : {}) {{\n^bb0(%in: f32, %out: f32):\n  linalg.yield %in : f32\n}} -> {}",
        target_id, attrs, tensor_id, tensor_type, target_id.0, target_type, target_type
    ));

    vec![empty_tensor, linalg_generic]
}

fn make_affine_map(dims: Vec<Option<usize>>, out_rank: usize) -> String {
    // Assume dim names are d{i}
    let params: Vec<String> = (0..out_rank).map(|i| format!("d{}", i)).collect();
    let outputs: Vec<String> = dims
        .iter()
        .enumerate()
        .map(|(i, d)| match d {
            Some(1) => "0".to_string(),
            _ => format!("d{i}"),
        })
        .collect();

    format!(
        "affine_map<({}) -> ({})>",
        params.join(", "),
        outputs.join(", ")
    )
}

////////////////////////////////////////////////////////////////////////////////
// NOTE: below here are essentially unfinished!

// TODO: FIXME: this just uses arith.sitofp, which is not always correct
pub fn cast(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Assignment> {
    assert!(ssa.sources.len() == 2);
    assert!(ssa.targets.len() == 1);

    let tensor_id = grammar::Identifier(ssa.sources[0].0.0);
    let source_type = core_type_to_mlir(&ssa.sources[0].1);

    // Extract dtype from second source
    let Type::Dtype(catgrad::typecheck::DtypeExpr::Constant(target_dtype)) = &ssa.sources[1].1
    else {
        panic!("cast dtype must be a constant")
    };

    // Get source *shape*
    let Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        shape: _source_shape,
        dtype: DtypeExpr::Constant(source_dtype),
    })) = &ssa.sources[0].1
    else {
        panic!("cast source must be a tensor");
    };

    let target_type = core_type_to_mlir(&ssa.targets[0].1);

    let op_name = match (source_dtype, target_dtype) {
        (lang::Dtype::F32, lang::Dtype::F32) => panic!("Invalid cast F32 → F32"),
        (lang::Dtype::F32, lang::Dtype::U32) => todo!("Cast F32 → U32"),
        (lang::Dtype::U32, lang::Dtype::F32) => "arith.sitofp",
        (lang::Dtype::U32, lang::Dtype::U32) => panic!("Invalid cast U32 → U32"),
    };

    vec![
        grammar::Expr::Custom(format!(
            "{} {} : {} to {}",
            op_name, tensor_id, source_type, target_type
        ))
        .into_assignment(ssa),
    ]
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise ops

fn elementwise(op_name: &str, ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Assignment> {
    let operands: Vec<grammar::Identifier> = ssa
        .sources
        .iter()
        .map(|(source_node, _)| grammar::Identifier(source_node.0))
        .collect();
    let result_type = core_type_to_mlir(&ssa.targets[0].1);

    vec![
        grammar::Expr::Elementwise(grammar::Elementwise {
            name: op_name.to_string(),
            operands,
            ty: result_type,
        })
        .into_assignment(ssa),
    ]
}

pub fn neg(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Assignment> {
    // typechecked ssa should never break this invariant
    assert!(ssa.sources.len() == 1);
    assert!(ssa.targets.len() == 1);
    elementwise("arith.negf", ssa)
}

pub fn add(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Assignment> {
    assert!(ssa.sources.len() == 2);
    assert!(ssa.targets.len() == 1);
    elementwise("arith.addf", ssa)
}

pub fn pow(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Assignment> {
    assert!(ssa.sources.len() == 2);
    assert!(ssa.targets.len() == 1);
    elementwise("math.powf", ssa)
}

pub fn div(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Assignment> {
    assert!(ssa.sources.len() == 2);
    assert!(ssa.targets.len() == 1);
    elementwise("arith.divf", ssa)
}
