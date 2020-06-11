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

struct ConvTestSpec {
  PrimitiveType primitive_type;
  string filecheck_lines;
};

string ConvTestSpecToString(const ::testing::TestParamInfo<ConvTestSpec>& info) {
  return PrimitiveType_Name(info.param.primitive_type);
}

class PlaidMLConvOperationTest
    : public PlaidMLCodegenTest,
      public ::testing::WithParamInterface<ConvTestSpec> {
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

TEST_P(PlaidMLConvOperationTest, SimpleConvOp) {
  HloComputation::Builder builder(TestName());
  ConvTestSpec spec = GetParam();

  auto input_shape = ShapeUtil::MakeShape(spec.primitive_type, {1, 224, 224, 1});
  auto kernel_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3, 1, 32});

  HloInstruction* input = builder.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "input"));
  HloInstruction* kernel = builder.AddInstruction(
      HloInstruction::CreateParameter(1, kernel_shape, "input"));

  auto conv_shape = Shape();
  conv_shape.set_element_type(spec.primitive_type);
  conv_shape.add_dimensions(1);
  conv_shape.add_dimensions(222);
  conv_shape.add_dimensions(222);
  conv_shape.add_dimensions(32);
  Window conv_window;
  WindowDimension* conv_dim_1 = conv_window.add_dimensions();
  conv_dim_1->set_size(3);
  conv_dim_1->set_padding_low(0);
  conv_dim_1->set_padding_high(0);
  conv_dim_1->set_stride(1);
  conv_dim_1->set_window_dilation(1);
  conv_dim_1->set_base_dilation(1);
  conv_dim_1->set_window_reversal(false);
  WindowDimension* conv_dim_2 = conv_window.add_dimensions();
  conv_dim_2->set_size(3);
  conv_dim_2->set_padding_low(0);
  conv_dim_2->set_padding_high(0);
  conv_dim_2->set_stride(1);
  conv_dim_2->set_window_dilation(1);
  conv_dim_2->set_base_dilation(1);
  conv_dim_2->set_window_reversal(false);
  ConvolutionDimensionNumbers conv_dnums;
  conv_dnums.set_input_batch_dimension(0);
  conv_dnums.add_input_spatial_dimensions(1);
  conv_dnums.add_input_spatial_dimensions(2);
  conv_dnums.set_input_feature_dimension(3);
  conv_dnums.add_kernel_spatial_dimensions(0);
  conv_dnums.add_kernel_spatial_dimensions(1);
  conv_dnums.set_kernel_input_feature_dimension(2);
  conv_dnums.set_kernel_output_feature_dimension(3);
  conv_dnums.set_output_batch_dimension(0);
  conv_dnums.add_output_spatial_dimensions(1);
  conv_dnums.add_output_spatial_dimensions(2);
  conv_dnums.set_output_feature_dimension(3);
  PrecisionConfig conv_pc;
  conv_pc.mutable_operand_precision()->Resize(
      /*new_size=*/2, PrecisionConfig::DEFAULT);
  auto convolution_19 = builder.AddInstruction(HloAllGatherInstruction::CreateConvolve(conv_shape, input, kernel, 1, 1, conv_window, conv_dnums, conv_pc));

  CompileAndCheck(builder.Build(), spec.filecheck_lines);
}

std::vector<ConvTestSpec> GetConvTestCases() {
  std::vector<ConvTestSpec> result;
// TODO: reenable F16 when it is ready
//  result.push_back(
//      {F16, R"(CHECK: func @hlo_module(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32>)"});
  result.push_back(
      {F32, R"(CHECK: func @hlo_module(%arg0: tensor<3x3x1x32xf32>, %arg1: tensor<1x224x224x1xf32>) -> tensor<1x222x222x32xf32>)"});
  result.push_back(
      {F64, R"(CHECK: func @hlo_module(%arg0: tensor<3x3x1x32xf32>, %arg1: tensor<1x224x224x1xf32>) -> tensor<1x222x222x32xf32>)"});
  return result;
}

/**/
// TODO: INSTANTIATE_TEST_CASE_P was deprecated in favor for INSTANTIATE_TEST_SUITE_P, but the version of gtest that bazel links in is looking for INSTANTIATE_TEST_CASE_P right now.
INSTANTIATE_TEST_CASE_P(All,
                         PlaidMLConvOperationTest,
                         ::testing::ValuesIn(GetConvTestCases()),
                         ConvTestSpecToString);
/**/
}  // namespace
}  // namespace plaidml
}  // namespace xla
