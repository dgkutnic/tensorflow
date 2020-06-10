// Tests that show HLO Module conversion to PlaidML Program.

#include <algorithm>
#include <string>

#include <gtest/gtest.h>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/service/plaidml/compiler.h"
#include "tensorflow/compiler/xla/service/plaidml/tests/plaidml_codegen_test.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace plaidml {
namespace {

struct TrivialModelTestSpec {
  PrimitiveType primitive_type;
  string filecheck_lines;
};

string TrivialModelTestSpecToString(const ::testing::TestParamInfo<TrivialModelTestSpec>& info) {
  return PrimitiveType_Name(info.param.primitive_type);
}

class PlaidMLTrivialModelOperationTest
    : public PlaidMLCodegenTest,
      public ::testing::WithParamInterface<TrivialModelTestSpec> {
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

    return Status::OK();

  }
};

TEST_P(PlaidMLTrivialModelOperationTest, SimpleTrivialModel) {
  HloComputation::Builder builder(TestName());
  TrivialModelTestSpec spec = GetParam();

  auto param_0_shape = ShapeUtil::MakeShape(spec.primitive_type, {1, 3});
  auto param_1_shape = ShapeUtil::MakeShape(spec.primitive_type, {1, 3});

  HloInstruction* lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_0_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(
      HloInstruction::CreateParameter(1, param_1_shape, "input"));

  // %constant.32 = u32[] constant(1), metadata={op_type="Size" op_name="loss/input_1_loss/num_elements"}
  //builder.AddInstruction(HloInstruction::CreateConstant(LiteralUtil::CreateR0<unsigned int>(1)));
  // %reshape.3 = f32[1,3]{1,0} reshape(f32[1,3]{1,0} %arg0.1)
  auto reshape_3 = builder.AddInstruction(HloInstruction::CreateReshape(lhs->shape(), lhs));
  // %reshape.4 = f32[1,3]{1,0} reshape(f32[1,3]{1,0} %arg1.2)
  auto reshape_4 = builder.AddInstruction(HloInstruction::CreateReshape(rhs->shape(), rhs));
  // %subtract.5 = f32[1,3]{1,0} subtract(f32[1,3]{1,0} %reshape.3, f32[1,3]{1,0} %reshape.4), metadata={op_type="SquaredDifference" op_name="loss/input_1_loss/SquaredDifference"}
  auto subtract_5 = builder.AddInstruction(HloInstruction::CreateBinary(reshape_3->shape(), HloOpcode::kSubtract, reshape_3, reshape_4));
  // %multiply.6 = f32[1,3]{1,0} multiply(f32[1,3]{1,0} %subtract.5, f32[1,3]{1,0} %subtract.5), metadata={op_type="SquaredDifference" op_name="loss/input_1_loss/SquaredDifference"}
  auto multiply_6 = builder.AddInstruction(HloInstruction::CreateBinary(subtract_5->shape(), HloOpcode::kMultiply, subtract_5, subtract_5));
  // %convert.7 = f32[1,3]{1,0} convert(f32[1,3]{1,0} %multiply.6), metadata={op_type="Mean" op_name="loss/input_1_loss/Mean"}
  auto convert_7 = builder.AddInstruction(HloInstruction::CreateConvert(multiply_6->shape(), multiply_6));
  // %constant.8 = f32[] constant(0), metadata={op_type="Mean" op_name="loss/input_1_loss/Mean"}
  auto constant_8 = builder.AddInstruction(HloInstruction::CreateConstant(multiply_6->shape(), multiply_6));
  CompileAndCheck(builder.Build(), spec.filecheck_lines);
}

std::vector<TrivialModelTestSpec> GetTrivialModelTestCases() {
  std::vector<TrivialModelTestSpec> result;
  result.push_back(
      {F32, R"(CHECK: func @hlo_module(%arg0: tensor<1x3xf32>, %arg1: tensor<1x3xf32>) -> tensor<1x3xf32>)"});
  result.push_back(
      {F64, R"(CHECK: func @hlo_module(%arg0: tensor<1x3xf32>, %arg1: tensor<1x3xf32>) -> tensor<1x3xf32>)"});
  return result;
}

/**/
// TODO: INSTANTIATE_TEST_CASE_P was deprecated in favor for INSTANTIATE_TEST_SUITE_P, but the version of gtest that bazel links in is looking for INSTANTIATE_TEST_CASE_P right now.
INSTANTIATE_TEST_CASE_P(All,
                         PlaidMLTrivialModelOperationTest,
                         ::testing::ValuesIn(GetTrivialModelTestCases()),
                         TrivialModelTestSpecToString);
/**/
}  // namespace
}  // namespace plaidml
}  // namespace xla
