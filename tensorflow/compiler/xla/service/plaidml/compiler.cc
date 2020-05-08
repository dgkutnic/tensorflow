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

#include "tensorflow/compiler/xla/service/plaidml/compiler.h"

#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/cholesky_expander.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/service/dynamic_index_splitter.h"
#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_subcomputation_unification.h"
#include "tensorflow/compiler/xla/service/plaidml/executable.h"
#include "tensorflow/compiler/xla/service/layout_assignment.h"
#include "tensorflow/compiler/xla/service/map_inliner.h"
#include "tensorflow/compiler/xla/service/reshape_mover.h"
#include "tensorflow/compiler/xla/service/triangular_solve_expander.h"
#include "tensorflow/compiler/xla/service/while_loop_simplifier.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"
#include "plaidml/edsl/edsl.h"
#include "plaidml/op/op.h"
#include "plaidml/exec/exec.h"

using ::plaidml::edsl::Placeholder;
using ::plaidml::edsl::Program;
using ::plaidml::edsl::ProgramBuilder;
using ::plaidml::edsl::Tensor;
using ::plaidml::edsl::TensorDim;
using ::plaidml::edsl::TensorIndex;
using ::plaidml::edsl::TensorOutput;

using ::plaidml::DType;

namespace xla {
namespace plaidml {

namespace {

// Handles custom_call ops during evaluation by routing them through the global
// CPU registry used by other CPU-based backends.
StatusOr<Literal> HandleEvaluatorCustomCall(
    HloInstruction* custom_call, absl::Span<const Literal*> operands) {
  // Find the target C function in the global registry.
  auto* registry = CustomCallTargetRegistry::Global();
  void* target_fn = registry->Lookup(custom_call->custom_call_target(), "Host");
  if (!target_fn) {
    return NotFound("Custom call target '%s' was not registered",
                    custom_call->custom_call_target());
  }

  // Populate pointers to operand and output literal data.
  std::vector<const void*> operand_data;
  operand_data.reserve(operands.size());
  for (const auto* literal : operands) {
    operand_data.push_back(literal->untyped_data());
  }
  auto output = Literal::CreateFromShape(custom_call->shape());
  void* output_data = output.untyped_data();

  // Call the target function matching the C ABI used by the CPU backends.
  auto* typed_fn = reinterpret_cast<void (*)(void*, const void**)>(target_fn);
  (*typed_fn)(output_data, operand_data.data());

  return std::move(output);
}

}  // namespace

Status PlaidMLCompiler::RunHloOptimization(HloModule* hlo_module) {
  HloPassPipeline pipeline("PlaidML");

  pipeline.AddPass<DynamicIndexSplitter>();
  pipeline.AddPass<CholeskyExpander>();
  pipeline.AddPass<TriangularSolveExpander>();
  pipeline.AddPass<LayoutAssignment>(
      hlo_module->mutable_entry_computation_layout(),
      LayoutAssignment::InstructionCanChangeLayout);

  return pipeline.Run(hlo_module).status();
}

StatusOr<std::unique_ptr<HloModule>> PlaidMLCompiler::RunHloPasses(
    std::unique_ptr<HloModule> hlo_module, se::StreamExecutor* /*stream_exec*/,
    se::DeviceMemoryAllocator* /*device_allocator*/) {
  VLOG(1) << "Run hlo passes on graph " << hlo_module->name();
  TF_RETURN_IF_ERROR(RunHloOptimization(hlo_module.get()));
  return std::move(hlo_module);
}

// Remove after testing
Tensor Dot(const Tensor& X, const Tensor& Y) {
  TensorDim I, J, K;
  TensorIndex i("i"), j("j"), k("k");
  X.bind_dims(I, K);
  Y.bind_dims(K, J);
  auto R = TensorOutput(I, J);
  R(i, j) += X(i, k) * Y(k, j);
  return R;
}

std::unique_ptr<Program> makeProgram(const std::string& name, const std::vector<Tensor>& outputs) {
  //auto program = ProgramBuilder(name, outputs).compile();
  auto program = absl::make_unique<Program>(ProgramBuilder(name, outputs));
  std::cout << program << std::endl;
  return std::move(program);
}

// Translate HLO Module to EDSL
std::unique_ptr<Program> PlaidMLCompiler::ProgramFromHloModule (
    HloModule* hlo_module) {

  VLOG(1) << "ProgramFromHloModule begin";

  for (auto* computation : hlo_module->computations()) {
    for (auto* instruction : computation->instructions()) {
      switch (instruction->opcode()) {
        // Unary ops.
        case HloOpcode::kAbs:
        case HloOpcode::kRoundNearestAfz:
        case HloOpcode::kBitcast:
        case HloOpcode::kCeil:
        case HloOpcode::kClz:
        case HloOpcode::kCopy:
        case HloOpcode::kCopyStart:
        case HloOpcode::kCopyDone:
        case HloOpcode::kCos:
        case HloOpcode::kExp:
        case HloOpcode::kExpm1:
        case HloOpcode::kImag:
        case HloOpcode::kIsFinite:
        case HloOpcode::kFloor:
        case HloOpcode::kLog:
        case HloOpcode::kLog1p:
        case HloOpcode::kNot:
        case HloOpcode::kNegate:
        case HloOpcode::kPopulationCount:
        case HloOpcode::kReal:
        case HloOpcode::kRsqrt:
        case HloOpcode::kSign:
        case HloOpcode::kSin:
        case HloOpcode::kSqrt:
        case HloOpcode::kTanh: {
          // Parse operands.
        }
        // Perhaps add a message here that the op isn't implemented for the plaidml backend (yet)
        default:
          break;
      }
    }
  }

  auto A = Placeholder(DType::FLOAT32, {8, 16});
  auto B = Placeholder(DType::FLOAT32, {16, 32});
  auto C = Dot(A, B);
  auto program = makeProgram("dot", {C});
  return std::move(program);
}

//StatusOr<std::unique_ptr<xla::plaidml::PlaidMLExecutable>> PlaidMLCompiler::RunBackend(
StatusOr<std::unique_ptr<Executable>> PlaidMLCompiler::RunBackend(
    std::unique_ptr<HloModule> hlo_module, se::StreamExecutor* stream_exec,
    se::DeviceMemoryAllocator* /*device_allocator*/) {
  TF_RET_CHECK(stream_exec != nullptr);

  VLOG(1) << "Run backend PLAIDML " << hlo_module->name();

  
  TF_ASSIGN_OR_RETURN(DynamicDimensionInference dynamic_dimension_inference,
                      DynamicDimensionInference::Run(hlo_module.get()));

  auto evaluator = absl::make_unique<HloEvaluator>();
  evaluator->set_use_fast_path(
      hlo_module->config().debug_options().xla_hlo_evaluator_use_fast_path());
  evaluator->set_custom_call_handler(HandleEvaluatorCustomCall);
  

  auto program = ProgramFromHloModule(hlo_module.get());

  // Create executable from the PlaidML Program.
  
  /*
  std::unique_ptr<Executable> executable =
      absl::make_unique<Executable>(
          std::move(hlo_module), std::move(evaluator),
          std::move(dynamic_dimension_inference));
  */

  std::unique_ptr<Executable> executable =
      absl::make_unique<PlaidMLExecutable>(
          std::move(hlo_module), std::move(evaluator), std::move(program),
          std::move(dynamic_dimension_inference));

  return std::move(executable);
}

//StatusOr<std::vector<std::unique_ptr<xla::plaidml::PlaidMLExecutable>>> PlaidMLCompiler::Compile(
StatusOr<std::vector<std::unique_ptr<Executable>>> PlaidMLCompiler::Compile(
    std::unique_ptr<HloModuleGroup> module_group,
    std::vector<std::vector<se::StreamExecutor*>> stream_exec,
    se::DeviceMemoryAllocator* device_allocator) {
  if (module_group->empty()) {
    //return std::vector<std::unique_ptr<Executable>>();
  }
  if (module_group->size() > 1) {
    return tensorflow::errors::Unimplemented(
        "Compilation of multiple HLO modules is not supported on PlaidML.");
  }
  if (stream_exec.size() != 1 || stream_exec[0].size() != 1) {
    return tensorflow::errors::Unimplemented(
        "Unexpected number of StreamExecutor's.");
  }
  auto hlo_modules = module_group->ConsumeModules();
  TF_ASSIGN_OR_RETURN(auto module,
                      RunHloPasses(std::move(hlo_modules[0]), stream_exec[0][0],
                                   device_allocator));
  TF_ASSIGN_OR_RETURN(
      auto executable,
      RunBackend(std::move(module), stream_exec[0][0], device_allocator));
  std::vector<std::unique_ptr<Executable>> ret;
  ret.push_back(std::move(executable));
  return std::move(ret);
}

StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
PlaidMLCompiler::CompileAheadOfTime(
    std::unique_ptr<HloModuleGroup> module_group,
    const AotCompilationOptions& aot_options) {
  return tensorflow::errors::InvalidArgument(
      "AOT compilation not supported on PlaidML");
}

se::Platform::Id PlaidMLCompiler::PlatformId() const {
  return se::plaidml::kXlaPlaidMLPlatformId;
}

HloCostAnalysis::ShapeSizeFunction PlaidMLCompiler::ShapeSizeBytesFunction()
    const {
  //return PlaidMLExecutable::ShapeSizeBytes;
  return nullptr;
}

static bool InitModule() {
  xla::Compiler::RegisterCompilerFactory(
      se::plaidml::kXlaPlaidMLPlatformId, []() {
        return absl::make_unique<xla::plaidml::PlaidMLCompiler>();
      });
  xla::ComputationPlacer::RegisterComputationPlacer(
      se::plaidml::kXlaPlaidMLPlatformId,
      []() { return absl::make_unique<xla::ComputationPlacer>(); });
  return true;
}

static bool module_initialized = InitModule();

}  // namespace plaidml
}  // namespace xla
