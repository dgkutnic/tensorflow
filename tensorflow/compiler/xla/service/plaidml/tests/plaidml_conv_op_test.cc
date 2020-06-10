// Tests that show HLO Module conversion to PlaidML Program.

#include <algorithm>
#include <string>

#include <gtest/gtest.h>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/service/plaidml/compiler.h"
#include "tensorflow/compiler/xla/service/plaidml/tests/plaidml_codegen_test.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace plaidml {
namespace {

struct DotTestSpec {
  PrimitiveType primitive_type;
  string filecheck_lines;
};

string DotTestSpecToString(const ::testing::TestParamInfo<DotTestSpec>& info) {
  return PrimitiveType_Name(info.param.primitive_type);
}

class PlaidMLDotOperationTest
    : public PlaidMLCodegenTest,
      public ::testing::WithParamInterface<DotTestSpec> {
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

TEST_P(PlaidMLDotOperationTest, SimpleDotOp) {
  HloComputation::Builder builder(TestName());
  DotTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {128, 128});

  HloInstruction* lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(
      HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(CreateCanonicalDot(param_shape, lhs, rhs));
  CompileAndCheck(builder.Build(), spec.filecheck_lines);
}

TEST_P(PlaidMLDotOperationTest, DotTransposeOp) {
  HloComputation::Builder builder(TestName());
  DotTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {128, 128});

  HloInstruction* lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(
      HloInstruction::CreateParameter(1, param_shape, "input"));
  HloInstruction* lhs_transposed = builder.AddInstruction(
      HloInstruction::CreateTranspose(param_shape, lhs, {1, 0}));

  builder.AddInstruction(CreateCanonicalDot(param_shape, lhs_transposed, rhs));
  CompileAndCheck(builder.Build(), spec.filecheck_lines);
}

std::vector<DotTestSpec> GetDotTestCases() {
  std::vector<DotTestSpec> result;
// TODO: reenable F16 when it is ready
//  result.push_back(
//      {F16, R"(CHECK: func @hlo_module(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32>)"});
  result.push_back(
      {F32, R"(CHECK: func @hlo_module(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32>)"});
  result.push_back(
      {F64, R"(CHECK: func @hlo_module(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32>)"});
  return result;
}

/**/
// TODO: INSTANTIATE_TEST_CASE_P was deprecated in favor for INSTANTIATE_TEST_SUITE_P, but the version of gtest that bazel links in is looking for INSTANTIATE_TEST_CASE_P right now.
INSTANTIATE_TEST_CASE_P(All,
                         PlaidMLDotOperationTest,
                         ::testing::ValuesIn(GetDotTestCases()),
                         DotTestSpecToString);
/**/
}  // namespace
}  // namespace plaidml
}  // namespace xla
