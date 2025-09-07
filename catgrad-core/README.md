<div style="text-align: center;" align="center">

[![catgrad][catgrad_img]][catgrad_link]

[![Crate][crate_img]][crate_link]
[![License][license_img]][license_file]
[![Documentation][docs_img]][docs_link]

## a categorical deep learning compiler <!-- omit in toc -->

</div>

1. [What is catgrad?](#what-is-catgrad)
1. [Using catgrad](#using-catgrad)
    1. [Defining models](#defining-models)
    1. [Typechecking](#typechecking)
    1. [Running models](#running-models)
    1. [Visualising models](#visualising-models) (coming soon!)
    1. [Serverless runtime](#serverless-runtime) (coming soon!)
1. [Feature Flags](#feature-flags)
1. [Roadmap](#roadmap)
1. [Category theory](#category-theory)

## What is catgrad?

catgrad is a *deep learning **compiler*** using category theory to statically
compile models into their forward and backwards passes.
This means both your inference *and* training loop compile to static code and
run without needing a deep learning framework (not even catgrad!)

A model in catgrad is an [Open Hypergraph](https://docs.rs/open-hypergraphs/)
meaning that models are completely symbolic mathematical expressions.
Catgrad's autodiff algorithm is then a source-to-source transformation of this
symbolic representation, requiring no explicit tracking of backwards passes at
runtime: training and inference are simply two different open hypergraphs.

This also means that developing additional backends for catgrad is easy: there
is no need to worry about the backwards pass in Tensor representations, since
it is handled at the *syntactic* level: a catgrad backend is an NdArray
implementation.

Catgrad's syntactic representation powers the [Hellas
Network](https://hellas.ai/): a network for trustlessly buying and selling
tensor compute in real-time.
The Network uses Catgrad as a serverless runtime: models are serialized and
sent to remote nodes for computation.

## Using Catgrad

There are a few ways to use the catgrad rust crate:

1. [Defining models](#defining-models)
1. [Running models](#running-models)
1. [Visualising models](#visualising-models) (coming soon!)
1. [Serverless runtime](#serverless-runtime) (coming soon!)

See [./examples/hidden.rs] for an end-to-end example of defining, running, and
visualising a model's structure.
We'll briefly outline the important parts here.

### Defining Models

Construct a model by declaring a type and implementing [`crate::stdlib::Def`] for it.
The most important method is `def`, which defines your model structure:

```rust
struct ReLU;
impl Def<1, 1> for ReLU {
    // TODO: example?
    fn def() {
    }
};
```

In catgrad, models, layers, and modules are all the same: OpenHypergraphs.
Implementing `Def` will get you a `term() -> OpenHypergrpah` method which you can call to get an instance of your model:

```rust
// TODO
struct TODO;
```

TODO: read more - the [Var interface of Open Hypergraphs](https://docs.rs/open-hypergraphs/latest/open_hypergraphs/lax/var/index.html).

### Typechecking

TODO

### Running Models

Models are just *syntax*; to run them you need a backend.
Catgrad provides an `Interpreter` with pluggable backends- the `ndarray` backend is enabled for developers

The `Interpreter` class 

```rust
// Choose an array backend
let model = MyModel.term();
let backend = NdArrayBackend;
let env = stdlib(); // load standard definitions / layers
let model_params = load_param_data(&backend); // user code - e.g., from safetensors
let interpreter = Interpreter::new(backend, env, model_params);
let inputs = load_inputs(); // get from user/tokenizer/etc.
interpreter.run(model, todo());
```

### Visualising models

TODO

### Serverless runtime

Coming soon!

## Feature flags

Backends are feature-flagged (TODO: list them)

## Roadmap

Some features on the roadmap:

- Rewriting & optimization
- Deterministic backend
- Serverless runtime features - package model/weights + run with no env/etc.
- Implement autodiff (as in our original [python implementation](..)).
- Explicit (in-language) control flow
- More models
- Type checker improvements
- Ongoing language changes

## Category theory

- Lang & core: categories as "dialects", functors as passes
    - TODO: actually implement this way!
    - Forget as a pass
    - Optimization as rewriting
- Open Hypergraphs and Symmetric Monoidal Categories
- Links to papers

<!-- Badges and Logo -->
[crate_link]: https://crates.io/crates/catgrad "Crate listing"
[crate_img]: https://img.shields.io/crates/v/catgrad.svg?style=for-the-badge&color=f46623 "Crate badge"
[docs_link]: https://docs.rs/catgrad/latest/catgrad "Crate documentation"
[docs_img]: https://img.shields.io/docsrs/catgrad/latest.svg?style=for-the-badge "Documentation badge"
[license_file]: https://github.com/hellas-ai/catgrad/blob/master/LICENSE "Project license"
[license_img]: https://img.shields.io/crates/l/catgrad.svg?style=for-the-badge "License badge"

[catgrad_link]: https://catgrad.com
[catgrad_img]: ../catgrad.svg
