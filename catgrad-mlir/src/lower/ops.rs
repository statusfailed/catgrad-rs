use catgrad::ssa::SSA;
use catgrad::{
    category::lang,
    typecheck::{DtypeExpr, NatExpr, NdArrayType, ShapeExpr, Type, TypeExpr},
};

use super::grammar;
use super::util::*;

// Represent a `Shape` as MLIR `tensor<rank x index>`.
//
// Generates assignments like
// ```
//  %shape_dim0 = tensor.dim %input, %c0 : tensor<?x?x?xf32>
//  %shape_dim1 = tensor.dim %input, %c1 : tensor<?x?x?xf32>
//  %shape_dim2 = tensor.dim %input, %c2 : tensor<?x?x?xf32>
//  %shape = tensor.from_elements %id_dim0, %id_dim1, %id_dim2 : tensor<3 x index>
// ```
//
// Where `shape` is the SSA node id of the result.
pub fn shape(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Assignment> {
    assert!(ssa.sources.len() == 1);
    assert!(ssa.targets.len() == 1);

    // Extract the shape from the input tensor type
    let Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        shape: ShapeExpr::Shape(shape_dims),
        ..
    })) = &ssa.sources[0].1
    else {
        panic!("shape operation requires tensor input")
    };

    let target_type = core_type_to_mlir(&ssa.targets[0].1);

    // For static shapes, create tensor.from_elements with constants
    let dim_values: Vec<String> = shape_dims
        .iter()
        .map(|dim| match dim {
            NatExpr::Constant(n) => format!("arith.constant {} : index", n),
            _ => panic!("shape operation requires constant dimensions for now"),
        })
        .collect();

    vec![
        grammar::Expr::Custom(format!(
            "tensor.from_elements {} : {}",
            dim_values.join(", "),
            target_type
        ))
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

pub fn shape_pack(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Assignment> {
    // Pack k Nats into a shape: Nat^k → Type
    // Creates a tensor<k x index> from k natural number inputs
    assert!(ssa.targets.len() == 1);

    let target_type = core_type_to_mlir(&ssa.targets[0].1);

    // Get the input identifiers (natural numbers)
    let input_ids: Vec<String> = ssa
        .sources
        .iter()
        .map(|(source_node, _)| format!("%v{}", source_node.0))
        .collect();

    vec![
        grammar::Expr::Custom(format!(
            "tensor.from_elements {} : {}",
            input_ids.join(", "),
            target_type
        ))
        .into_assignment(ssa),
    ]
}

// Index : Sources × Dim × Indexes → Values
//
// Lowers to something like this (in this example, dim = 0)
// ```
// func.func @example(
//   %src: tensor<?x?x?xf32>,
//   %idx: tensor<?xindex>
// ) -> tensor<?x?x?xf32> {
//   // Constants for each dimension
//   %c0 = arith.constant 0 : index
//   %c1 = arith.constant 1 : index
//   %c2 = arith.constant 2 : index
//
//   // Size of indexes tensor
//   %n = tensor.dim %idx, %c0 : tensor<?xindex>
//
//   // Dims of input tensor
//   %d1 = tensor.dim %src, %c1 : tensor<?x?x?xf32>
//   %d2 = tensor.dim %src, %c2 : tensor<?x?x?xf32>
//
//   // Result tensor
//   %empty = tensor.empty(%n, %d1, %d2) : tensor<?x?x?xf32>
//
//   %res = linalg.generic
//     {
//       indexing_maps = [
//         affine_map<(d0, d1, d2) -> (d0)>,
//         affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//       ],
//       iterator_types = ["parallel", "parallel", "parallel"]
//     }
//     ins(%idx : tensor<?xindex>)
//     outs(%empty : tensor<?x?x?xf32>) {
//       ^bb0(%idx_elem: index, %out_elem: f32):
//         %idx_1 = linalg.index 1 : index
//         %idx_2 = linalg.index 2 : index
//         %val = tensor.extract %src[%idx_elem, %idx_1, %idx_2] : tensor<?x?x?xf32>
//         linalg.yield %val : f32
//     } -> tensor<?x?x?xf32>
//   return %res : tensor<?x?x?xf32>
// }
// ```
pub fn tensor_index(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Statement> {
    assert!(ssa.sources.len() == 3);
    assert!(ssa.targets.len() == 1);

    // Verify that dim is statically known to be 0
    let Type::Nat(NatExpr::Constant(0)) = &ssa.sources[1].1 else {
        panic!("tensor.index currently only supports dimension 0")
    };

    let source_type = core_type_to_mlir(&ssa.sources[0].1);
    let index_type = core_type_to_mlir(&ssa.sources[2].1);
    let target_type = core_type_to_mlir(&ssa.targets[0].1);
    // Extract target shape to determine which dimensions are statically known
    let Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        shape: ShapeExpr::Shape(target_shape_dims),
        ..
    })) = &ssa.targets[0].1
    else {
        panic!("target must be a tensor")
    };

    // List of statements to be returned
    let mut statements = vec![];

    // Extract the rank from the source tensor type
    let Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        shape: ShapeExpr::Shape(shape_dims),
        ..
    })) = &ssa.sources[0].1
    else {
        panic!("tensor_index operation requires tensor input")
    };

    let rank = shape_dims.len();

    // Base name prefix used for intermediate variables that don't appear in the original hypergraph
    let source_ssa = grammar::Identifier(ssa.sources[0].0.0);
    let base = source_ssa.clone();
    let index_ssa = grammar::Identifier(ssa.sources[2].0.0);

    let target_id = grammar::Identifier(ssa.targets[0].0.0);

    // Generate constant statements for each dimension
    //  %c0 = arith.constant 0 : index
    for i in 0..rank {
        statements.push(grammar::Statement::Custom(format!(
            "  {base}_c{i} = arith.constant {i} : index"
        )));
    }

    // Generate size of indexes tensor
    //  %n = tensor.dim %idx, %c0 : tensor<?xindex>
    statements.push(grammar::Statement::Custom(format!(
        "  {base}_n = tensor.dim {index_ssa}, {base}_c0 : {index_type}",
    )));

    // Generate dimension extraction statements for dims 1..rank
    //  %d1 = tensor.dim %src, %c1 : tensor<?x?x?xf32>
    let src_id = grammar::Identifier(ssa.sources[0].0.0);
    for i in 1..rank {
        statements.push(grammar::Statement::Custom(format!(
            "  {base}_d{i} = tensor.dim {source_ssa}, {base}_c{i} : {source_type}",
        )));
    }

    // Generate empty result tensor - only include dynamic dimensions
    let empty_expr = to_empty_expr(&base, &target_type, target_shape_dims);

    // Generate affine map inputs: d0, d1, ..., d_{rank-1}
    let parallel_stmts = (0..rank)
        .map(|_| "\"parallel\"")
        .collect::<Vec<_>>()
        .join(", ");
    let affine_map_inputs = (0..rank)
        .map(|i| format!("d{i}"))
        .collect::<Vec<_>>()
        .join(", ");

    // Generate index variables: %idx_elem, %idx_1, %idx_2, ...
    let mut index_vars = vec!["%idx_elem_index".to_string()];
    for i in 1..rank {
        index_vars.push(format!("%idx_{i}"));
    }
    let index_vars_str = index_vars.join(", ");

    // Generate index assignment statements for dimensions 1..rank
    //  %idx_1 = linalg.index 1 : index
    let index_assignments = (1..rank)
        .map(|i| format!("              %idx_{i} = linalg.index {i} : index"))
        .collect::<Vec<_>>()
        .join("\n");

    let linalg_generic = grammar::Statement::Custom(format!(
        r#"
        {base}_empty = {empty_expr}
        {target_id} = linalg.generic
          {{
            indexing_maps = [
              affine_map<({affine_map_inputs}) -> (d0)>,                    // idx
              affine_map<({affine_map_inputs}) -> ({affine_map_inputs})>    // output
            ],
            iterator_types = [{parallel_stmts}]
          }}
          ins({index_ssa} : {index_type})
          outs({base}_empty : {target_type}) {{
            ^bb0(%idx_elem: i32, %out_elem: f32):
              %idx_elem_index = arith.index_cast %idx_elem : i32 to index
              {index_assignments}
              %val = tensor.extract {src_id}[{index_vars_str}] : {source_type}
              linalg.yield %val : f32
          }} -> {target_type}
    "#
    ));

    statements.push(linalg_generic);
    statements
}

/// Generate a `tensor.empty` expression from a list of NatExpr representing the dims.
fn to_empty_expr(
    base: &grammar::Identifier,
    target_type: &grammar::Type,
    dims: &[NatExpr],
) -> String {
    let rank = dims.len();

    let is_known_dimension: Vec<bool> = dims
        .iter()
        .map(|dim| matches!(dim, NatExpr::Constant(_)))
        .collect();

    let mut dim_args = vec![];
    if !is_known_dimension[0] {
        dim_args.push(format!("{base}_n"));
    }
    for i in 1..rank {
        if !is_known_dimension[i] {
            dim_args.push(format!("{base}_d{i}"));
        }
    }
    let empty_expr = format!("tensor.empty({}) : {target_type}", dim_args.join(", "));
    empty_expr
}

// Transpose : Tensor × Nat × Nat → Tensor
// Transposes two dimensions (assumed to be statically known as NatExpr::Constant) in the input tensor.
// Example:
// ```
// %transposed = linalg.transpose ins(%0 : tensor<?x?x?x?xf32>)
//                                outs(%init : tensor<?x?x?x?xf32>)
//                                permutation = [0, 2, 1, 3]
// ```
pub fn tensor_transpose(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Statement> {
    assert!(ssa.sources.len() == 3);
    assert!(ssa.targets.len() == 1);

    // Extract the two dimension indices (must be constants)
    let Type::Nat(NatExpr::Constant(dim1)) = &ssa.sources[1].1 else {
        panic!("transpose dim1 must be a constant")
    };
    let Type::Nat(NatExpr::Constant(dim2)) = &ssa.sources[2].1 else {
        panic!("transpose dim2 must be a constant")
    };

    // Get tensor type info
    let Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        shape: ShapeExpr::Shape(shape_dims),
        ..
    })) = &ssa.sources[0].1
    else {
        panic!("transpose operation requires tensor input")
    };

    let rank = shape_dims.len();

    let source_type = core_type_to_mlir(&ssa.sources[0].1);
    let target_type = core_type_to_mlir(&ssa.targets[0].1);

    let tensor_id = grammar::Identifier(ssa.sources[0].0.0);
    let target_id = grammar::Identifier(ssa.targets[0].0.0);

    // Create permutation array: identity with dim1 and dim2 swapped
    let mut permutation: Vec<usize> = (0..rank).collect();
    permutation.swap(*dim1, *dim2);

    // Get target shape for empty tensor creation
    let Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        shape: ShapeExpr::Shape(target_shape_dims),
        ..
    })) = &ssa.targets[0].1
    else {
        panic!("target must be a tensor")
    };

    let mut statements = vec![];

    // Generate dimension extraction for dynamic dimensions
    let base = grammar::Identifier(ssa.sources[0].0.0);
    for (i, dim) in target_shape_dims.iter().enumerate() {
        if !matches!(dim, NatExpr::Constant(_)) {
            // This is a dynamic dimension, extract it
            statements.push(grammar::Statement::Custom(format!(
                "  {base}_c{i} = arith.constant {i} : index"
            )));
            statements.push(grammar::Statement::Custom(format!(
                "  {base}_d{i} = tensor.dim {tensor_id}, {base}_c{i} : {source_type}"
            )));
        }
    }

    // Create empty tensor for the result
    let empty_expr = to_empty_expr(&base, &target_type, target_shape_dims);

    // Generate the transpose operation
    let permutation_str = permutation
        .iter()
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    let transpose_stmt = grammar::Statement::Custom(format!(
        "  {base}_empty = {empty_expr}\n  {target_id} = linalg.transpose ins({tensor_id} : {source_type}) outs({base}_empty : {target_type}) permutation = [{permutation_str}]"
    ));

    statements.push(transpose_stmt);
    statements
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
        (lang::Dtype::U32, lang::Dtype::F32) => "arith.uitofp",
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

pub fn arange(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Assignment> {
    assert!(ssa.sources.len() == 1);
    assert!(ssa.targets.len() == 1);

    let target_type = core_type_to_mlir(&ssa.targets[0].1);

    vec![
        grammar::Expr::Custom(format!(
            "tensor.generate {{\n^bb0(%i0: index):\n  %idx = arith.index_cast %i0 : index to i32\n  tensor.yield %idx : i32\n}} : {}",
            target_type
        ))
        .into_assignment(ssa),
    ]
}
