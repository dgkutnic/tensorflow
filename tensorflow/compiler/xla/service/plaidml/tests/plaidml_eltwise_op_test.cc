// Tests that show HLO Module conversion to PlaidML Program.

#include <algorithm>
#include <string>

#include <gtest/gtest.h>

#include "absl/strings/str_cat.h"
//#include "tensorflow/compiler/xla/service/cpu/cpu_compiler.h"
#include "tensorflow/compiler/xla/service/plaidml/compiler.h"
#include "tensorflow/compiler/xla/service/plaidml/tests/plaidml_codegen_test.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_evaluator.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace plaidml {
namespace {

struct EltwiseTestSpec {
  PrimitiveType primitive_type;
  string filecheck_lines;
};

string EltwiseTestSpecToString(const ::testing::TestParamInfo<EltwiseTestSpec>& info) {
  return PrimitiveType_Name(info.param.primitive_type);
}

class PlaidMLEltwiseOperationTest
    : public PlaidMLCodegenTest,
      public ::testing::WithParamInterface<EltwiseTestSpec> {
 protected:
  Status CompileAndCheck(std::unique_ptr<HloComputation> entry_computation,
                       const string& filecheck_lines) {

    std::unique_ptr<HloModule> hlo_module = CreateNewVerifiedModule();
    hlo_module->AddEntryComputation(std::move(entry_computation));

    auto program = CompileToProgram(std::move(hlo_module));

    VLOG(2) << "Program:\n" << program->str();

    StatusOr<bool> fc_result = RunFileCheck(program->str(), filecheck_lines);

    //TF_ASSERT_OK(fc_result.status());
    EXPECT_TRUE(fc_result.ValueOrDie());

    VLOG(2) << "Evaluating results";

    // Create TensorShape
    // ::plaidml::TensorShape(pml_dtype_map_[shape.element_type()], shape_here);
    // makeBuffer

    std::vector<float> input_vec = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<float> expected = {2, 4, 6, 8, 10, 12, 14, 16, 18};

/*

    auto cpu_compiler = cpu::CpuCompiler().RunBackend(std::move(hlo_module),
                                   backend().default_stream_executor(),
                                   //device_allocator=nullptr);

    std::initializer_list<std::initializer_list<float>> input_vec = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    std::initializer_list<std::initializer_list<float>> expected  = {{2, 4, 6}, {8, 10, 12}, {14, 16, 18}};

    VLOG(2) << "Creating inputs";

    std::vector<Literal> input_args = {LiteralUtil::CreateR2<float>(input_vec), LiteralUtil::CreateR2<float>(input_vec)};

    HloEvaluator eval;

    VLOG(2) << "Calling Evaluator";

    StatusOr<Literal> result_literal = eval.Evaluate(*hlo_module->entry_computation(), input_args);
*/

    //CHECK_EQ(result_literal, LiteralUtil::CreateR2<float>(expected));

    return Status::OK();

  }
};

TEST_P(PlaidMLEltwiseOperationTest, EltwiseAddOp) {
  HloComputation::Builder builder(TestName());
  EltwiseTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(
      HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateBinary(param_shape, HloOpcode::kAdd, lhs, rhs));
  CompileAndCheck(builder.Build(), spec.filecheck_lines);
}

/*

TEST_P(PlaidMLEltwiseOperationTest, EltwiseSubOp) {
  HloComputation::Builder builder(TestName());
  EltwiseTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(
      HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateBinary(param_shape, HloOpcode::kSubtract, lhs, rhs));
  CompileAndCheck(builder.Build(), spec.filecheck_lines);
}

TEST_P(PlaidMLEltwiseOperationTest, EltwiseMulOp) {
  HloComputation::Builder builder(TestName());
  EltwiseTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(
      HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateBinary(param_shape, HloOpcode::kMultiply, lhs, rhs));
  CompileAndCheck(builder.Build(), spec.filecheck_lines);
}

TEST_P(PlaidMLEltwiseOperationTest, EltwiseDivOp) {
  HloComputation::Builder builder(TestName());
  EltwiseTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(
      HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateBinary(param_shape, HloOpcode::kDivide, lhs, rhs));
  CompileAndCheck(builder.Build(), spec.filecheck_lines);
}
*/

std::vector<EltwiseTestSpec> GetEltwiseTestCases() {
  std::vector<EltwiseTestSpec> result;
// TODO: reenable F16 when it is ready
//  result.push_back(
//      {F16, R"(CHECK: func @hlo_module(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32>)"});
  result.push_back(
      {F32, R"(CHECK: func @hlo_module(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32>)"});
  result.push_back(
      {F64, R"(CHECK: func @hlo_module(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32>)"});
  return result;
}


/**/
// TODO: INSTANTIATE_TEST_CASE_P was deprecated in favor for INSTANTIATE_TEST_SUITE_P, but the version of gtest that bazel links in is looking for INSTANTIATE_TEST_CASE_P right now.
INSTANTIATE_TEST_CASE_P(All,
                         PlaidMLEltwiseOperationTest,
                         ::testing::ValuesIn(GetEltwiseTestCases()),
                         EltwiseTestSpecToString);
/**/
}  // namespace
}  // namespace plaidml
}  // namespace xla
