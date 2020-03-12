/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EDSL_TRANSLATE_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EDSL_TRANSLATE_H_

#include <string>

#include "mlir/IR/Module.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/op_or_arg_name_mapper.h"

namespace tflite {

// Translates the given MLIR `module` into a FlatBuffer and stores the
// serialized flatbuffer into the string. This uses OpOrArgLocNameMapper to
// convert location of the op to name in flatbuffer. Returns true if translation
// fails, otherwise returns false.
bool MlirToEDSLTranslateFunction(mlir::ModuleOp module,
                                       std::string* serialized_flatbuffer);

// Same as the above but with a custom op name mapper.
bool MlirToEDSLTranslateFunction(
    mlir::ModuleOp module, std::string* serialized_flatbuffer,
    tensorflow::OpOrArgNameMapper* op_or_arg_name_mapper);
}  // namespace tflite

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_FLATBUFFER_TRANSLATE_H_
