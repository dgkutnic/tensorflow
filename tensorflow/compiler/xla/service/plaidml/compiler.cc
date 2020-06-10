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
#include <unordered_map>

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
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_subcomputation_unification.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/plaidml/executable.h"
#include "tensorflow/compiler/xla/service/layout_assignment.h"
#include "tensorflow/compiler/xla/service/map_inliner.h"
#include "tensorflow/compiler/xla/service/reshape_mover.h"
#include "tensorflow/compiler/xla/service/triangular_solve_expander.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/service/while_loop_simplifier.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"
#include "plaidml/edsl/edsl.h"
#include "plaidml/op/op.h"
#include "plaidml/exec/exec.h"

using ::plaidml::Buffer;
using ::plaidml::TensorShape;
using ::plaidml::edsl::LogicalShape;
using ::plaidml::edsl::Placeholder;
using ::plaidml::edsl::Program;
using ::plaidml::edsl::ProgramBuilder;
using ::plaidml::edsl::Tensor;
using ::plaidml::edsl::TensorDim;
using ::plaidml::edsl::TensorIndex;
using ::plaidml::edsl::TensorOutput;
using ::plaidml::edsl::Value;

using ::plaidml::DType;
namespace plaidml_op = ::plaidml::op;
namespace m = xla::match;

Buffer makeBuffer(const TensorShape& shape, const void* data) {
  const auto& curDevice = ::plaidml::Settings::get("PLAIDML_DEVICE");
  Buffer buffer(curDevice, shape);
  buffer.copy_from(data);
  return buffer;
}

namespace xla {
namespace plaidml {

namespace {

// TODO: Is this needed?
// Handles custom_call ops during evaluation by routing them through the global
// registry used by other backends.
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

// TODO: Figure out the right passes to run here
Status PlaidMLCompiler::RunHloOptimization(HloModule* hlo_module) {
  HloPassPipeline pipeline("PlaidML");

  /*
  pipeline.AddPass<DynamicIndexSplitter>();
  pipeline.AddPass<CholeskyExpander>();
  pipeline.AddPass<TriangularSolveExpander>();
  pipeline.AddPass<LayoutAssignment>(
      hlo_module->mutable_entry_computation_layout(),
      LayoutAssignment::InstructionCanChangeLayout);
  */
  pipeline.AddPass<FlattenCallGraph>();
  //pipeline.AddPass<WhileLoopSimplifier>();
  return pipeline.Run(hlo_module).status();
}

StatusOr<std::unique_ptr<HloModule>> PlaidMLCompiler::RunHloPasses(
    std::unique_ptr<HloModule> hlo_module, se::StreamExecutor* /*stream_exec*/,
    se::DeviceMemoryAllocator* /*device_allocator*/) {
  VLOG(1) << "Run hlo passes on graph " << hlo_module->name();
  TF_RETURN_IF_ERROR(RunHloOptimization(hlo_module.get()));
  return std::move(hlo_module);
}

// TODO: Remove after testing
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
  VLOG(1) << "makeProgram begin";
  auto program = absl::make_unique<Program>(ProgramBuilder(name, outputs));
  //std::cout << *program << std::endl;
  VLOG(1) << "Generated program:\n" << (*program).str();
  return std::move(program);
}

/* static */ std::string PlaidMLCompiler::HumanString(const Shape& shape) {
  if (shape.IsTuple()) {
    string text = "{";
    const char* prefix = "";
    for (const Shape& elem_shape : shape.tuple_shapes()) {
      absl::StrAppend(&text, prefix, HumanString(elem_shape));
      prefix = ", ";
    }
    text += "}";
    return text;
  }
  std::vector<std::string> dim_elements;
  for (int i = 0; i < shape.dimensions_size(); ++i) {
    if (shape.is_dynamic_dimension(i)) {
      dim_elements.push_back(absl::StrCat("<=", shape.dimensions(i)));
    } else {
      dim_elements.push_back(absl::StrCat(shape.dimensions(i)));
    }
  }
  return absl::StrCat("{", absl::StrJoin(dim_elements, ","), "}");
}

// Translate HLO Module to EDSL
StatusOr<std::unique_ptr<Program>> PlaidMLCompiler::ProgramFromHloModule (
    std::unique_ptr<HloModule> hlo_module) {

  VLOG(2) << "ORIGINAL HLO MODULE:\n" << hlo_module->ToString();

  std::unordered_map<int, Tensor> instr_map;
  std::unordered_map<int, Value> tuple_instr_map;
  std::unordered_map<std::string, Value> fn_returns;

  auto computation_is_addition = [](HloComputation* c) {
    return c->instruction_count() == 3 &&
    Match(c->root_instruction(), m::Add(m::Parameter(), m::Parameter()));
  };

  auto computation_is_multiplication = [](HloComputation* c) {
    return c->instruction_count() == 3 &&
    Match(c->root_instruction(), m::Multiply(m::Parameter(), m::Parameter()));
  };

  auto computation_is_maximum = [](HloComputation* c) {
    return c->instruction_count() == 3 &&
    Match(c->root_instruction(), m::Maximum(m::Parameter(), m::Parameter()));
  };

  // TODO: may be unnecessary because TF has the kParameter opcode which instantiates Placeholder creation.
  // std::vector<Tensor> inputs;
  std::string program_str_cpp;

  VLOG(1) << "ProgramFromHloModule begin";

  VLOG(1) << "result_shape" << hlo_module->result_shape().ToString();

  if (hlo_module->has_entry_computation()) {
    auto entry_computation = hlo_module->entry_computation();
    VLOG(1) << "PlaidML Entry Computation " + entry_computation->name() + ": num parameters " << entry_computation->num_parameters() << " returns " << entry_computation->root_instruction()->shape().ToString(); 
    /*
    for (int i = 0; i < entry_computation->num_parameters(); i++) {
      auto param = entry_computation->parameter_instruction(i);
      auto pshape = param->shape();
      std::vector<int64_t> param_shape;
      auto param_dtype = pml_dtype_map_[pshape.element_type()];
      for (int j = 0; j < pshape.dimensions_size(); j++) {
        param_shape.push_back(pshape.dimensions(j));
      }
      inputs.push_back(Placeholder(param_dtype, param_shape));
    }
    */
  }

  std::string tabs = "";
  for (auto* computation : hlo_module->computations()) {
    VLOG(2) << "Computation name" << computation->name() << " num_parameters " << computation->num_parameters() << " returns " << computation->root_instruction()->shape().ToString();
    // TODO: Verify that the computation return type should actuually be Tensor, or Value, or something else...
    // TODO: Add parameters
    // TODO: Replace periods in computation name with underscores to make it a valid function name
    auto function_name = legalize_computation_name(computation->name());
    auto root_instr = computation->root_instruction();
    auto root_instr_id = root_instr->unique_id();
    auto root_instr_shape = root_instr->shape();
/*
    std::string return_type;
    if (root_instr_shape.IsArray()) {
      return_type = "Tensor";
    } else {
      return_type = "Value";
    }
    program_str_cpp += tabs + return_type + " " + function_name + "() {\n";
    tabs += "  ";
*/
    for (auto* instruction : computation->instructions()) {
      VLOG(2) << xla::HloOpcodeString(instruction->opcode()) << " name " << instruction->name() << " id " << instruction->unique_id() << " num_operands " << instruction->operand_count();
      VLOG(2) << instruction->OperandsToString(HloPrintOptions());
      VLOG(2) << "instruction returns " << instruction->shape().ToString();
      auto cur_instr_name = instruction->name();
      auto cur_instr_id = instruction->unique_id();
      auto num_operands = instruction->operand_count();
      std::vector<int> operand_ids;
      for (auto i = 0; i < num_operands; i++) {
        int operand_id = instruction->operand(i)->unique_id();
        VLOG(2) << "visiting operand " << operand_id;
        operand_ids.push_back(operand_id);
      }
      // result shape
      auto shape = instruction->shape();
      std::vector<int64_t> dims;
      for (int j = 0; j < shape.dimensions_size(); j++) {
        dims.push_back(shape.dimensions(j));
      }
      auto type = pml_dtype_map_[shape.element_type()];
      //std::string type_cpp = cpp_dtype_map_[type];
      //std::string type_plaidml = pml_dtype_map_[type];
      // TODO: validate that all these general parameters are correct before constructing them into a larger program.
      switch (instruction->opcode()) {
        case HloOpcode::kConstant: {
          auto tshape = TensorShape(type, dims);
          auto lshape = LogicalShape(type, dims);
          const Literal& literal = instruction->literal();
          auto buf = makeBuffer(tshape, literal.untyped_data());
          auto op = Constant(lshape, buf);
          instr_map.insert(std::make_pair(cur_instr_id, op));
          // Create constant buffers, etc.
          //program_str_cpp += tabs + "std::vector<int64_t> shape" + unique_name + " = " + dims + ";\n";
          //program_str_cpp += tabs + "std::vector<"+ type_cpp + "> " + unique_name + "_vals = " + instruction->OperandsToString(HloPrintOptions()) + ";\n";
          //program_str_cpp += tabs + "auto buffer" + unique_name + " = makeBuffer(TensorShape(" + type_plaidml + ", shape"  + unique_name + "), " + unique_name + "_vals);\n";
          //program_str_cpp += tabs + "auto " + unique_name + " = Constant(LogicalShape("+ type_plaidml + ", shape"  + unique_name + "), buffer" + unique_name + ", \"" + unique_name +"\");\n";
          break;
        }
        case HloOpcode::kAdd: {
          // Tensor elementwise addition
          auto op = instr_map[operand_ids[0]] + instr_map[operand_ids[1]];
          instr_map.insert(std::make_pair(cur_instr_id, op));
          //program_str_cpp += tabs + "auto " + unique_name + " = " + operand_names[0] + " + " + operand_names[1] + ";\n";
          break;
        }
        case HloOpcode::kMultiply: {
          // Tensor elementwise multiplication
          auto op = instr_map[operand_ids[0]] * instr_map[operand_ids[1]];
          instr_map.insert(std::make_pair(cur_instr_id, op));
          //program_str_cpp += tabs + "auto " + unique_name + " = " + operand_names[0] + " * " + operand_names[1] + ";\n";
          break;
        }
        case HloOpcode::kSubtract: {
          // Tensor elementwse subtraction
          auto op = instr_map[operand_ids[0]] - instr_map[operand_ids[1]];
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kDivide: {
          // Tensor elementwise division
          auto op = instr_map[operand_ids[0]] / instr_map[operand_ids[1]];
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kExp: {
          // Tensor elementwise exp
          auto op = ::plaidml::edsl::exp(instr_map[operand_ids[0]]);
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kMaximum: {
          // Tensor elementwise maximum
          auto op = plaidml_op::maximum(instr_map[operand_ids[0]], instr_map[operand_ids[1]]);
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kMinimum: {
          // Tensor elementwise minimum
          auto op = plaidml_op::minimum(instr_map[operand_ids[0]], instr_map[operand_ids[1]]);
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kConvolution: {
          // Tensor convolution operation
          // TODO: make sure optional operands are passed correctly
          auto op = plaidml_op::convolution(instr_map[operand_ids[0]], instr_map[operand_ids[1]]);
          // padding defaults to valid, test same padding and manual padding 
          // auto padding_config = instruction->padding_config();
          op = op.autopad_mode(plaidml_op::AutoPadMode::VALID);
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kDot: {
          // Tensor dot operation
          auto op = plaidml_op::dot(instr_map[operand_ids[0]], instr_map[operand_ids[1]]);
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kReduce: {
          // with reduce, an axis is specified, so figure out which computation and pass that in to the op lib
          VLOG(2) << "begin processing reduce";
          HloReduceInstruction* reduce = Cast<HloReduceInstruction>(instruction);
          auto reduction_dims = reduce->dimensions();
          std::vector<int> axes;
          for (auto d : reduction_dims) {
            axes.push_back(d);
            VLOG(2) <<  "Adding axis " << d << " to reduce";
          }
          Tensor op;
          auto applied_computation = instruction->to_apply();
          if (computation_is_maximum(applied_computation)) {
            VLOG(2) << "Reached condition: max with reduce";
            op = plaidml_op::max(instr_map[operand_ids[0]], ::plaidml::edsl::make_tuple(axes));
          } else if (computation_is_addition(applied_computation)) {
            VLOG(2) << "Reached condition: add with reduce";
            op = plaidml_op::sum(instr_map[operand_ids[0]], ::plaidml::edsl::make_tuple(axes));
          }
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kReduceWindow: {
          // This is some sort of aggregation. If the aggregation op is max, then it's a max pool
          VLOG(2) << "begin processing reduce-window";
          Tensor op;
          // Keras default pool mode and pad mode
          plaidml_op::PoolMode pm = plaidml_op::PoolMode::AVG;
          plaidml_op::AutoPadMode am = plaidml_op::AutoPadMode::VALID;
          // TODO: Window dimensions also store strides, if strides are nondefault.
          std::vector<int> default_strides = {1, 1};
          // This window is in NHWC format: for example, a window of 1x2x2x1 translates to {2, 2} in the pooling op
          auto raw_window = instruction->window();
          std::vector<int> processed_window;
          /*
          for (auto d : raw_window.dimensions()) {
            if (!window_util::IsTrivialWindowDimension(d)) {
              processed_window.push_back(d.size());
              VLOG(2) << "added dimension " << d.size() << " to pool";
            }
          }
          */
          // IsTrivialWindowDimension doesn't work as expected on a test case. I'll hard code these values for now and re-visit.
          processed_window.push_back(raw_window.dimensions()[1].size());
          processed_window.push_back(raw_window.dimensions()[2].size());
          auto applied_computation = instruction->to_apply();
          if (computation_is_maximum(applied_computation)) {
            VLOG(2) << "Reached condition: max pooling";
            pm = plaidml_op::PoolMode::MAX;
          }
          // TODO: Add conditions for avg pooling, sum pooling, etc.
          op = plaidml_op::pool(instr_map[operand_ids[0]], pm, processed_window, default_strides, am, {}, plaidml_op::TensorLayout::NXC);
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kBroadcast: {
          // Broadcasting is handled implicitly by PlaidML, so this is a no-op, UNLESS the broadcast doesn't have numpy semantics, then we must perform a repeat
          auto op = instr_map[operand_ids[0]];
          auto operand_shape = instruction->operand(0)->shape();
          auto bcast_dims = instruction->dimensions();
          // Check if numpy style broadcasting semantics exist
          // Constants can always be broadcasted
          // Assume that two tensors with the same rank can be broadcasted
          // Dims can be prepended, cannot be appended
          if (operand_shape.rank() && operand_shape.rank() != dims.size()) {
            if (bcast_dims.size() == operand_shape.rank() && std::find(bcast_dims.begin(), bcast_dims.end(), dims.size() - 1) == bcast_dims.end()) {
              // Broadcasting all dims, need to add 1 to the reshape
              std::vector<int64_t> reshape_dims = dims;
              reshape_dims[reshape_dims.size() - 1] = 1;
              op = ::plaidml::edsl::reshape(op, reshape_dims);
            }
          }
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kConvert: {
          // Equivalent to cast
          auto op = ::plaidml::edsl::cast(instr_map[operand_ids[0]], type);
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kParameter: {
          // Tensor inputs, create a placeholder
          auto op = Placeholder(type, dims);
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kReshape: {
          // Tensor reshape operation
          auto op = ::plaidml::edsl::reshape(instr_map[operand_ids[0]], dims);
          instr_map.insert(std::make_pair(cur_instr_id, op));
          //program_str_cpp += tabs + "std::vector<int64_t> shape" + unique_name + " = " + dims + ";\n";
          //program_str_cpp += tabs + "auto " + unique_name + " = reshape(" + operand_names[0] + ", shape" + unique_name + ");\n";
          break;
        }
        case HloOpcode::kTranspose: {
          // Tensor transpose operation
          // TODO: test correctness, see if the axes operand is needed
          auto op = plaidml_op::transpose(instr_map[operand_ids[0]]);
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kTuple: {
          // a kTuple operation is kind of like make_tuple in EDSL
          std::vector<Value> tup_operands;
          for (int i = 0; i < num_operands; i++) {
            tup_operands.push_back(Value(instr_map[operand_ids[i]]));
          }
          auto op = ::plaidml::edsl::make_tuple(tup_operands);
          tuple_instr_map.insert(std::make_pair(cur_instr_id, op));
          /*
          program_str_cpp += tabs + "auto " + unique_name + " = make_tuple(";
          for (int i = 0; i < operand_names.size(); i++) {
            program_str_cpp += operand_names[i];
            if (i != operand_names.size() - 1) {
              program_str_cpp += ", ";
            }
          } 
          program_str_cpp += ");\n";
          */
          break;
        }
        case HloOpcode::kGetTupleElement: {
          // a kGetTupleElement operation is like taking an element in a Value and interpreting it as a tensor, int, etc.
          // TODO: Handle return type
          auto tindex = instruction->tuple_index();
          //program_str_cpp += tabs + "auto " + unique_name + " = " + operand_names[0] + ".as_tuple()[" + tindex + "].as_tensor();\n";
          auto op = tuple_instr_map[operand_ids[0]].as_tuple()[tindex].as_tensor();
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kCall: {
          // Call another EDSL function
          auto computation_to_apply = instruction->to_apply();
          auto computation_name = legalize_computation_name(computation_to_apply->name());
          auto op = fn_returns[computation_name];
          instr_map.insert(std::make_pair(cur_instr_id, op.as_tensor()));
          // TODO: check parameters, return types, etc.
          //program_str_cpp += tabs + "auto " + unique_name + " = " + computation_name + "();\n";
          break;
        }
        // TODO: Unary ops.
        case HloOpcode::kAbs:
        case HloOpcode::kRoundNearestAfz:
        case HloOpcode::kBitcast:
        case HloOpcode::kCeil:
        case HloOpcode::kClz:
        case HloOpcode::kCopy:
        case HloOpcode::kCopyStart:
        case HloOpcode::kCopyDone:
        case HloOpcode::kCos:
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
          VLOG(2) << "Unimplemented unary op " << cur_instr_name  << " has been called here\n";
          break;
        }
        // Binary ops.
        case HloOpcode::kAtan2:
        case HloOpcode::kComplex:
        case HloOpcode::kPower:
        case HloOpcode::kRemainder:
        case HloOpcode::kAnd:
        case HloOpcode::kOr:
        case HloOpcode::kXor:
        case HloOpcode::kShiftLeft:
        case HloOpcode::kShiftRightArithmetic:
        case HloOpcode::kShiftRightLogical: {
          // Parse operands.
          VLOG(2) << "Unimplemented binary op " << cur_instr_name  << " has been called here\n";
          break;
        }
        // TODO: special instructions
        default:
          VLOG(2) << "Unknown op " << cur_instr_name << " (opcode " << HloOpcodeString(instruction->opcode()) << ") has been called here\n";
          break;
      }
    }
    /*
    program_str_cpp += tabs + "return " + root_instr_id_str + ";\n";
    tabs.pop_back();
    tabs.pop_back();
    program_str_cpp += tabs + "}\n";
    */
    if (root_instr_shape.IsArray()) {
      fn_returns.insert(std::make_pair(function_name, instr_map[root_instr_id]));
    } else {
      fn_returns.insert(std::make_pair(function_name, tuple_instr_map[root_instr_id]));
    }
  }

  Tensor output; 

  if (hlo_module->has_entry_computation()) {
    //auto entry_computation_name = legalize_computation_name(hlo_module->entry_computation()->name());
    //program_str_cpp += tabs + "auto program_result = " + entry_computation_name + "();\n";
    output = fn_returns[legalize_computation_name(hlo_module->entry_computation()->name())].as_tensor();
  }
 
  //VLOG(1) << "Program string:\n" << program_str_cpp;
 
  // TODO: Remove this after testing. Hard-coded until program generation works correctly 
  //auto A = Placeholder(DType::FLOAT32, {8, 16});
  //auto B = Placeholder(DType::FLOAT32, {16, 32});
  //auto C = Dot(A, B);
  auto program = makeProgram("hlo_module", {output});
  VLOG(1) << "ProgramFromHloModule complete";
  return std::move(program);
}

//StatusOr<std::unique_ptr<xla::plaidml::PlaidMLExecutable>> PlaidMLCompiler::RunBackend(
StatusOr<std::unique_ptr<Executable>> PlaidMLCompiler::RunBackend(
    std::unique_ptr<HloModule> hlo_module, se::StreamExecutor* stream_exec,
    se::DeviceMemoryAllocator* device_allocator) {
  TF_RET_CHECK(stream_exec != nullptr);

  VLOG(1) << "Run backend PLAIDML " << hlo_module->name();

  /* TODO: Add passes
  TF_ASSIGN_OR_RETURN(auto module,
                      RunHloPasses(std::move(hlo_module), stream_exec,
                                   device_allocator));
  */
  
  TF_ASSIGN_OR_RETURN(DynamicDimensionInference dynamic_dimension_inference,
                      DynamicDimensionInference::Run(hlo_module.get()));

  auto evaluator = absl::make_unique<HloEvaluator>();
  evaluator->set_use_fast_path(
      hlo_module->config().debug_options().xla_hlo_evaluator_use_fast_path());
  evaluator->set_custom_call_handler(HandleEvaluatorCustomCall);
  

  TF_ASSIGN_OR_RETURN(auto program, ProgramFromHloModule(std::move(hlo_module)));

  // Create executable from the PlaidML Program.
  
  /*
  std::unique_ptr<Executable> executable =
      absl::make_unique<Executable>(
          std::move(hlo_module), std::move(evaluator),
          std::move(dynamic_dimension_inference));
  */

  VLOG(1) << "Creating executable from PlaidML Program";

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
