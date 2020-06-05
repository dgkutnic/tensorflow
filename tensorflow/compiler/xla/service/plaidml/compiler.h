/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_PLAIDML_COMPILER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_PLAIDML_COMPILER_H_

#include <memory>
#include <unordered_map>
#include <vector>

#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/executable.h"
//#include "tensorflow/compiler/xla/service/plaidml/executable.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/plaidml/platform_id.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/stream_executor/stream_executor.h"
#include "plaidml/edsl/edsl.h"
#include "plaidml/exec/exec.h"
#include "plaidml/op/op.h"
#include "pmlc/target/x86/passes.h"

using ::plaidml::edsl::Program;
using ::plaidml::DType;

namespace xla {
namespace plaidml {

class PlaidMLCompiler : public Compiler {
 public:
  PlaidMLCompiler() {
    VLOG(1) << "Initializing PlaidMLCompiler";
    ::plaidml::init();
    ::plaidml::edsl::init();
    ::plaidml::op::init();
    ::plaidml::exec::init();
    ::pmlc::target::x86::registerPassPipeline();
  }
  ~PlaidMLCompiler() {}

  std::unordered_map<xla::HloComputation*, std::string> function_map_;
  std::unordered_map<xla::PrimitiveType, std::string> cpp_dtype_map_ = {
    {xla::PRED, "bool"},
    {xla::S8, "int8_t"},
    {xla::S16, "int16_t"},
    {xla::S32, "int32_t"},
    {xla::S64, "int64_t"},
    {xla::U8, "uint8_t"},
    {xla::U16, "uint16_t"},
    {xla::U32, "uint32_t"},
    {xla::U64, "uint64_t"},
    {xla::F32, "float"},
    {xla::F64, "double"}
  };
  std::unordered_map<xla::PrimitiveType, DType> pml_dtype_map_ = {
    {xla::PRED, DType::BOOLEAN},
    {xla::S8, DType::INT8},
    {xla::S16, DType::INT16},
    {xla::S32, DType::INT32},
    {xla::S64, DType::INT64},
    {xla::U8, DType::UINT8},
    {xla::U16, DType::UINT16},
    {xla::U32, DType::UINT32},
    {xla::U64, DType::UINT64},
    {xla::F16, DType::FLOAT16},
    {xla::F32, DType::FLOAT32},
    {xla::F64, DType::FLOAT32}
  };

  std::string HumanString(const Shape& shape);

  StatusOr<std::unique_ptr<Program>> ProgramFromHloModule (
      std::unique_ptr<HloModule> hlo_module);

  StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> hlo_module, se::StreamExecutor* stream_exec,
      se::DeviceMemoryAllocator* device_allocator);
//  StatusOr<std::unique_ptr<xla::plaidml::PlaidMLExecutable>> RunBackend(
  StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule> hlo_module, se::StreamExecutor* stream_exec,
      se::DeviceMemoryAllocator* device_allocator);
//  StatusOr<std::vector<std::unique_ptr<xla::plaidml::PlaidMLExecutable>>> Compile(
  StatusOr<std::vector<std::unique_ptr<Executable>>> Compile(
      std::unique_ptr<HloModuleGroup> module_group,
      std::vector<std::vector<se::StreamExecutor*>> stream_exec,
      se::DeviceMemoryAllocator* device_allocator);

  StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                     const AotCompilationOptions& aot_options);

  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const;

  se::Platform::Id PlatformId() const;

  std::string DEVICE_ID = "llvm_cpu.0";

  const std::vector<char> translator_dictionary = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'};

  std::string legalize_ids(int32_t id) {
    std::string legalized_id = "";
    int32_t ic = id;
    do {
      int32_t i = ic % 10;
      legalized_id = translator_dictionary[i] + legalized_id;
      ic /= 10;
    } while (ic > 0);
    return legalized_id;
  }

  std::string legalize_computation_name(const std::string& cname) {
   std::string result;
   for (int i = 0; i < cname.size(); i++) {
      if (cname[i] == '.') {
        result += "_";
      } else {
        result += cname[i];
      }
    }
    return result;
  }

 private:
  Status RunHloOptimization(HloModule* hlo_module);

  TF_DISALLOW_COPY_AND_ASSIGN(PlaidMLCompiler);
};

}  // namespace plaidml
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_PLAIDML_COMPILER_H_
