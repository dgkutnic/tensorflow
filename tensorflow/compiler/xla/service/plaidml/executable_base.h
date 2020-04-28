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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_PLAIDML_EXECUTABLE_BASE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_PLAIDML_EXECUTABLE_BASE_H_

#include <memory>

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/dynamic_dimension_inference.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_execution_profile.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "plaidml/edsl/edsl.h"

using ::plaidml::edsl::Program;

namespace xla {
namespace plaidml {

// PlaidML Executable Base
class PlaidMLExecutableBase : public Executable {
 public:
  explicit PlaidMLExecutableBase(
      std::shared_ptr<Program> plaidml_program) : Executable(nullptr, nullptr, nullptr), program_(plaidml_program) {};

  StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments,
      HloExecutionProfile* hlo_execution_profile);

  Program& program() const { return *program_; }
  std::shared_ptr<Program> shared_program() const { return program_; }

  const bool has_program() const { return program_ != nullptr; }

  /*
  Most of these describe things that the PlaidML Binder does
  TODO: Add binder to protected vars, add binder functions to this executable
  const HloModuleConfig& module_config() const { return hlo_module_->config(); }

  // The shape (including layout) that results from this execution. This is the
  // shape of the DeviceMemoryBase result value in ExecuteOnStream above.
  const Shape& result_shape() const {
    return hlo_module_->config().entry_computation_layout().result_shape();
  }

  // Returns the size of the executable in bytes. Returns -1 if this query is
  // not supported by the executable.
  //
  // Does not include the size of used libraries (e.g. cuDNN, Eigen, etc.).
  virtual int64 SizeOfGeneratedCodeInBytes();

  // Dumping helpers.
  void set_hlo_proto(std::unique_ptr<xla::HloProto> hlo_proto) {
    hlo_proto_ = std::move(hlo_proto);
  }
  bool dumping_snapshot() const { return hlo_proto_ != nullptr; }
  HloProto const* hlo_proto() const { return hlo_proto_.get(); }
  */

 protected:
  // PlaidML program this was compiled from.
  const std::shared_ptr<Program> program_;

  // Execution count, used to generate a unique filename for each dumped
  // execution.
  int64 execution_count_ = 0;

  virtual StatusOr<Literal> Evaluate(
      const HloComputation& computation,
      absl::Span<const Literal> arg_literals) = 0;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PlaidMLExecutableBase);
};

}  // namespace plaidml
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_PLAIDML_EXECUTABLE_BASE_H_
