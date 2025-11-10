func.func @negate_f32(%arg0: tensor<3x1x4xf32>) -> (tensor<3x1x4xf32>, tensor<3x1x4xf32>) {
  %0 = arith.negf %arg0 : tensor<3x1x4xf32>
  return %0, %arg0 : tensor<3x1x4xf32>, tensor<3x1x4xf32>
}
