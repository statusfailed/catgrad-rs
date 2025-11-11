func.func @term(%v0 : tensor<3x1x4xf32>) -> tensor<3x1x4xf32> {
  // Declaration(Path([PathComponent("tensor"), PathComponent("shape")]))
  %v1 = arith.constant false
  // Literal(F32(1.0))
    %v2_scalar = arith.constant 1.0 : f32
    %v2 = tensor.from_elements %v2_scalar : tensor<f32>
  // Declaration(Path([PathComponent("tensor"), PathComponent("dtype")]))
  // Declaration(Path([PathComponent("tensor"), PathComponent("neg")]))
  %v5 = arith.negf %v0 : tensor<3x1x4xf32>
  // Literal(F32(2.7182817))
    %v10_scalar = arith.constant 2.7182817 : f32
    %v10 = tensor.from_elements %v10_scalar : tensor<f32>
  // Declaration(Path([PathComponent("tensor"), PathComponent("broadcast")]))
    %v3_out = tensor.empty() : tensor<3x1x4xf32>
    %v3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%v2 : tensor<f32>) outs(%v3_out : tensor<3x1x4xf32>) {
^bb0(%in: f32, %out: f32):
  linalg.yield %in : f32
} -> tensor<3x1x4xf32>
  // Declaration(Path([PathComponent("tensor"), PathComponent("shape")]))
  %v9 = arith.constant false
  // Declaration(Path([PathComponent("tensor"), PathComponent("dtype")]))
  // Declaration(Path([PathComponent("tensor"), PathComponent("broadcast")]))
    %v11_out = tensor.empty() : tensor<3x1x4xf32>
    %v11 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%v10 : tensor<f32>) outs(%v11_out : tensor<3x1x4xf32>) {
^bb0(%in: f32, %out: f32):
  linalg.yield %in : f32
} -> tensor<3x1x4xf32>
  // Declaration(Path([PathComponent("tensor"), PathComponent("pow")]))
  %v6 = math.powf %v11, %v5 : tensor<3x1x4xf32>
  // Declaration(Path([PathComponent("tensor"), PathComponent("add")]))
  %v7 = arith.addf %v3, %v6 : tensor<3x1x4xf32>
  // Declaration(Path([PathComponent("tensor"), PathComponent("div")]))
  %v8 = arith.divf %v3, %v7 : tensor<3x1x4xf32>
  return %v8 : tensor<3x1x4xf32>
}
