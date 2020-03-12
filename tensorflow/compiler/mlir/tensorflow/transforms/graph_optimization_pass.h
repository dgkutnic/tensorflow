/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_GRAPH_OPTIMIZATION_PASS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_GRAPH_OPTIMIZATION_PASS_H_

#include "tensorflow/compiler/mlir/mlir_graph_optimization_pass.h"

namespace tensorflow {

// Bundle generic MLIR graph optimization passes (some derived from TF Grappler
// graph optimizers) into a single MLIR optimization pass.
class MlirGraphOptimizationPass : public MlirOptimizationPass {
 public:
  llvm::StringRef name() const override { return "graph_optimization"; }

  bool IsEnabled(const ConfigProto& config_proto) const override {
    return config_proto.experimental().enable_mlir_graph_optimization();
  }

  Status Run(const ConfigProto& config_proto, mlir::ModuleOp module) override;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_GRAPH_OPTIMIZATION_PASS_H_
