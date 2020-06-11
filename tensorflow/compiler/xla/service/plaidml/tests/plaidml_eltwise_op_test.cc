// Tests that show HLO Module conversion to PlaidML Program.

#include <algorithm>
#include <string>
#include <map>
#include <variant>

#include <gtest/gtest.h>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/service/plaidml/compiler.h"
#include "tensorflow/compiler/xla/service/plaidml/tests/plaidml_codegen_test.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_evaluator.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "plaidml/testenv.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

using ::plaidml::edsl::TensorBuffers;

namespace xla {
namespace plaidml {
namespace {

using TestCaseVal = std::vector<std::vector<float>>;
using TestCasePairs = std::map<TestCaseVal, TestCaseVal>;

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
                         const string& filecheck_lines,
                         const TestCasePairs& testcase_pairs) {

    HloModuleConfig cfg;

    std::unique_ptr<HloModule> hlo_module = absl::make_unique<HloModule>("module", cfg);
    hlo_module->AddEntryComputation(std::move(entry_computation));

    auto program = CompileToProgram(std::move(hlo_module));

    VLOG(2) << "Program:\n" << program->str();

    StatusOr<bool> fc_result = RunFileCheck(program->str(), filecheck_lines);

    //TF_ASSERT_OK(fc_result.status());
    EXPECT_TRUE(fc_result.ValueOrDie());

    VLOG(2) << "Evaluating results";

    //std::vector<float> input_vec = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    //std::vector<float> expected = {2, 4, 6, 8, 10, 12, 14, 16, 18};

    for (auto pair : testcase_pairs) {

      TensorBuffers inp;
      TensorBuffers exp;

      auto program_inputs = program->inputs();

      for (auto i = 0; i < program_inputs.size(); i++) {
        inp.insert(std::make_pair(program_inputs[i].tensor, pair.first[i]));
      }

      auto program_outputs = program->outputs();

      for (auto i = 0; i < program_outputs.size(); i++) {
        exp.insert(std::make_pair(program_outputs[i].tensor, pair.second[i]));
      }

      checkProgram(*program, inp, exp);

    }

    return Status::OK();

  }
};

TEST_P(PlaidMLEltwiseOperationTest, EltwiseAddOp) {
  std::vector<float> input_val = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> expected_val = {2, 4, 6, 8, 10, 12, 14, 16, 18};

  TestCaseVal inputs = {input_val, input_val};
  TestCaseVal results = {expected_val};
  TestCasePairs testcase_pairs = {{inputs, results}};

  HloComputation::Builder builder("EltwiseAddOp");
  EltwiseTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(
      HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateBinary(param_shape, HloOpcode::kAdd, lhs, rhs));
  CompileAndCheck(builder.Build(), spec.filecheck_lines, testcase_pairs);
}


TEST_P(PlaidMLEltwiseOperationTest, EltwiseSubOp) {
  std::vector<float> input_val = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> expected_val = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  TestCaseVal inputs = {input_val, input_val};
  TestCaseVal results = {expected_val};
  TestCasePairs testcase_pairs = {{inputs, results}};

  HloComputation::Builder builder("EltwiseSubOp");
  EltwiseTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(
      HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateBinary(param_shape, HloOpcode::kSubtract, lhs, rhs));
  CompileAndCheck(builder.Build(), spec.filecheck_lines, testcase_pairs);
}

TEST_P(PlaidMLEltwiseOperationTest, EltwiseMulOp) {
  std::vector<float> input_val = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> expected_val = {1, 4, 9, 16, 25, 36, 49, 64, 81};

  TestCaseVal inputs = {input_val, input_val};
  TestCaseVal results = {expected_val};
  TestCasePairs testcase_pairs = {{inputs, results}};

  HloComputation::Builder builder("EltwiseMulOp");
  EltwiseTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(
      HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateBinary(param_shape, HloOpcode::kMultiply, lhs, rhs));
  CompileAndCheck(builder.Build(), spec.filecheck_lines, testcase_pairs);
}

TEST_P(PlaidMLEltwiseOperationTest, EltwiseDivOp) {
  std::vector<float> input_val = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> expected_val = {1, 1, 1, 1, 1, 1, 1, 1, 1};

  TestCaseVal inputs = {input_val, input_val};
  TestCaseVal results = {expected_val};
  TestCasePairs testcase_pairs = {{inputs, results}};

  HloComputation::Builder builder("EltwiseDivOp");
  EltwiseTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3});

  HloInstruction* lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(
      HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(HloInstruction::CreateBinary(param_shape, HloOpcode::kDivide, lhs, rhs));
  CompileAndCheck(builder.Build(), spec.filecheck_lines, testcase_pairs);
}

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
