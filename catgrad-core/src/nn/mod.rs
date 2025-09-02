use std::collections::HashMap;

use crate::category::lang::*;
use crate::util::build_typed;

use open_hypergraphs::lax::var;

pub fn sigmoid_term() -> Term {
    build_typed([Object::Tensor], |graph, [x]| {
        let c1 = constant_f32(graph, 1.0);
        let s = shape(graph, x.clone());
        let c1 = broadcast(graph, c1, s);
        let r = c1.clone() / (c1 + exp(graph, -x));
        vec![r]
    })
    .expect("impossible")
}

pub fn sigmoid_source() -> Term {
    build_typed([Object::NdArrayType], |_graph, [t]| vec![t]).expect("impossible")
}

pub fn sigmoid_target() -> Term {
    build_typed([Object::NdArrayType], |_graph, [t]| vec![t]).expect("impossible")
}

pub fn sigmoid(builder: &Builder, x: Var) -> Var {
    var::fn_operation(
        builder,
        &[x],
        Object::Tensor,
        Operation::Definition(path(vec!["nn", "sigmoid"])),
    )
}

pub fn exp_term() -> Term {
    build_typed([Object::Tensor], |graph, [x]| {
        let e = constant_f32(graph, std::f32::consts::E);
        let s = shape(graph, x.clone());
        let e = broadcast(graph, e, s);
        vec![pow(graph, e, x)]
    })
    .expect("impossible")
}

pub fn exp_source() -> Term {
    build_typed([Object::NdArrayType], |_graph, [t]| vec![t]).expect("impossible")
}

pub fn exp_target() -> Term {
    build_typed([Object::NdArrayType], |_graph, [t]| vec![t]).expect("impossible")
}

pub fn exp(builder: &Builder, x: Var) -> Var {
    var::fn_operation(
        builder,
        &[x],
        Object::Tensor,
        Operation::Definition(path(vec!["nn", "exp"])),
    )
}

pub fn stdlib() -> Environment {
    let operations = HashMap::from([
        (
            path(vec!["nn", "sigmoid"]),
            OperationDefinition {
                term: sigmoid_term(),
                source_type: sigmoid_source(),
                target_type: sigmoid_target(),
            },
        ),
        (
            path(vec!["nn", "exp"]),
            OperationDefinition {
                term: exp_term(),
                source_type: exp_source(),
                target_type: exp_target(),
            },
        ),
    ]);

    Environment { operations }
}
