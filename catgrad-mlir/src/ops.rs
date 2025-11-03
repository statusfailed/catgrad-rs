use catgrad::ssa::SSA;
use catgrad::{
    category::lang,
    typecheck::{NatExpr, NdArrayType, ShapeExpr, Type, TypeExpr},
};

use super::grammar;
use super::util::*;

pub fn copy(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Assignment> {
    // typechecked ssa should never break this invariant
    assert!(ssa.sources.len() == 1, "copy ssa had more than 1 source");
    let source_id = grammar::Identifier(ssa.sources[0].0.0);

    ssa.targets
        .iter()
        .map(|(target_node, _)| grammar::Assignment {
            result: vec![grammar::Identifier(target_node.0)],
            expr: grammar::Expr::Identifier(source_id.clone()),
        })
        .collect()
}

pub fn shape(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Assignment> {
    // typechecked ssa should never break this invariant
    assert!(ssa.sources.len() == 1);
    assert!(ssa.targets.len() == 1);
    let source_id = grammar::Identifier(ssa.sources[0].0.0);
    let target_id = grammar::Identifier(ssa.targets[0].0.0);
    vec![grammar::Assignment {
        result: vec![target_id],
        expr: grammar::Expr::Identifier(source_id),
    }]
}

pub fn neg(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Assignment> {
    // typechecked ssa should never break this invariant
    assert!(ssa.sources.len() == 1);
    assert!(ssa.targets.len() == 1);
    let source_id = grammar::Identifier(ssa.sources[0].0.0);
    let target_id = grammar::Identifier(ssa.targets[0].0.0);
    let source_type = core_type_to_mlir(&ssa.sources[0].1);

    vec![grammar::Assignment {
        result: vec![target_id],
        expr: grammar::Expr::Operation(grammar::Operation {
            name: "arith.negf".to_string(),
            ins: vec![grammar::TypedIdentifier {
                id: source_id,
                ty: source_type.clone(),
            }],
            outs: vec![],
            return_types: vec![source_type],
            attrs: None,
            inner_block: None,
        }),
    }]
}

pub fn broadcast(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Assignment> {
    assert!(ssa.sources.len() == 2);
    assert!(ssa.targets.len() == 1);

    println!("ssa: {:?}", ssa.sources);

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
    let shape_id = grammar::Identifier(ssa.sources[1].0.0);
    let target_id = grammar::Identifier(ssa.targets[0].0.0);

    // Generate indexing maps for broadcasting
    let out_rank = out_shape.len();

    // Make affine map for *broadcasted* tensor
    //      (d_0..d_{out_rank}) -> (trailing dims, with 1s replaced with literal 0)
    // NOTE: we need to handle
    let input_dims: Vec<_> = in_shape
        .into_iter()
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

    vec![grammar::Assignment {
        result: vec![target_id],
        expr: grammar::Expr::Operation(grammar::Operation {
            name: "linalg.generic".to_string(),
            ins: vec![grammar::TypedIdentifier {
                id: tensor_id,
                ty: tensor_type,
            }],
            outs: vec![grammar::TypedIdentifier {
                id: shape_id,
                ty: target_type.clone(),
            }],
            return_types: vec![target_type],
            attrs: Some(attrs),
            inner_block: Some("^bb0(%in: f32, %out: f32):\n  linalg.yield %in : f32".to_string()),
        }),
    }]
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
