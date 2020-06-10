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

struct MnistCnnTestSpec {
  PrimitiveType primitive_type;
  string filecheck_lines;
};

string MnistCnnTestSpecToString(const ::testing::TestParamInfo<MnistCnnTestSpec>& info) {
  return PrimitiveType_Name(info.param.primitive_type);
}

class PlaidMLMnistCnnOperationTest
    : public PlaidMLCodegenTest,
      public ::testing::WithParamInterface<MnistCnnTestSpec> {
 protected:
  Status CompileAndCheck(std::unique_ptr<HloModule> hlo_module,
                       const string& filecheck_lines) {

    auto program = CompileToProgram(std::move(hlo_module));

    StatusOr<bool> fc_result = RunFileCheck(program->str(), filecheck_lines);

    //TF_ASSERT_OK(fc_result.status());
    EXPECT_TRUE(fc_result.ValueOrDie());

    return Status::OK();

  }
};

TEST_P(PlaidMLMnistCnnOperationTest, SimpleMnistCnn) {
  MnistCnnTestSpec spec = GetParam();

  std::unique_ptr<HloModule> hlo_module = CreateNewVerifiedModule();

  auto scalar_shape = ShapeUtil::MakeShape(spec.primitive_type, {});

  HloComputation::Builder max_builder(TestName() + ".max");

  HloInstruction* max_lhs = max_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "input"));
  HloInstruction* max_rhs = max_builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "input"));

  auto max_builder_body = max_builder.AddInstruction(HloInstruction::CreateBinary(scalar_shape, HloOpcode::kMaximum, max_lhs, max_rhs));

  auto max_computation = hlo_module->AddEmbeddedComputation(max_builder.Build());

  HloComputation::Builder add_builder(TestName() + ".add");

  HloInstruction* add_lhs = add_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "input"));
  HloInstruction* add_rhs = add_builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "input"));

  auto add_builder_body = add_builder.AddInstruction(HloInstruction::CreateBinary(scalar_shape, HloOpcode::kAdd, add_lhs, add_rhs));

  auto add_computation = hlo_module->AddEmbeddedComputation(add_builder.Build());

  HloComputation::Builder builder(TestName());

  auto param_0_shape = ShapeUtil::MakeShape(spec.primitive_type, {1, 224, 224, 1});
  auto param_1_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3, 1, 32});
  auto param_2_shape = ShapeUtil::MakeShape(spec.primitive_type, {32});
  auto param_3_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3, 32, 64});
  auto param_4_shape = ShapeUtil::MakeShape(spec.primitive_type, {64});
  auto param_5_shape = ShapeUtil::MakeShape(spec.primitive_type, {64, 128});
  auto param_6_shape = ShapeUtil::MakeShape(spec.primitive_type, {128});
  auto param_7_shape = ShapeUtil::MakeShape(spec.primitive_type, {128, 100});
  auto param_8_shape = ShapeUtil::MakeShape(spec.primitive_type, {100});

  HloInstruction* I = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_0_shape, "input"));
  HloInstruction* K1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, param_1_shape, "input"));
  HloInstruction* B1 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, param_2_shape, "input"));
  HloInstruction* K2 = builder.AddInstruction(
      HloInstruction::CreateParameter(3, param_3_shape, "input"));
  HloInstruction* B2 = builder.AddInstruction(
      HloInstruction::CreateParameter(4, param_4_shape, "input"));
  HloInstruction* K3 = builder.AddInstruction(
      HloInstruction::CreateParameter(5, param_5_shape, "input"));
  HloInstruction* B3 = builder.AddInstruction(
      HloInstruction::CreateParameter(6, param_6_shape, "input"));
  HloInstruction* K4 = builder.AddInstruction(
      HloInstruction::CreateParameter(7, param_7_shape, "input"));
  HloInstruction* B4 = builder.AddInstruction(
      HloInstruction::CreateParameter(8, param_8_shape, "input"));

  auto reshape_18 = builder.AddInstruction(HloInstruction::CreateReshape(B4->shape(), B4));

  auto shape_50 = Shape();
  shape_50.set_element_type(spec.primitive_type);
  shape_50.add_dimensions(1);
  shape_50.add_dimensions(219);
  shape_50.add_dimensions(219);
  shape_50.add_dimensions(100);
  auto broadcast_50 = builder.AddInstruction(HloInstruction::CreateBroadcast(shape_50, reshape_18, {3}));

  auto constant_44 = builder.AddInstruction(HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0)));

  auto shape_45 = Shape();
  shape_45.set_element_type(spec.primitive_type);
  shape_45.add_dimensions(1);
  shape_45.add_dimensions(219);
  shape_45.add_dimensions(219);
  shape_45.add_dimensions(128);
  auto broadcast_45 = builder.AddInstruction(HloInstruction::CreateBroadcast(shape_45, constant_44, {}));

  auto reshape_16 = builder.AddInstruction(HloInstruction::CreateReshape(B3->shape(), B3));

  auto shape_42 = Shape();
  shape_42.set_element_type(spec.primitive_type);
  shape_42.add_dimensions(1);
  shape_42.add_dimensions(219);
  shape_42.add_dimensions(219);
  shape_42.add_dimensions(128);
  auto broadcast_42 = builder.AddInstruction(HloInstruction::CreateBroadcast(shape_42, reshape_16, {3}));

  auto constant_34 = builder.AddInstruction(HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0)));

  auto shape_35 = Shape();
  shape_35.set_element_type(spec.primitive_type);
  shape_35.add_dimensions(1);
  shape_35.add_dimensions(219);
  shape_35.add_dimensions(219);
  shape_35.add_dimensions(64);
  auto broadcast_35 = builder.AddInstruction(HloInstruction::CreateBroadcast(shape_35, constant_34, {}));

  auto constant_22 = builder.AddInstruction(HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0)));

  auto shape_23 = Shape();
  shape_23.set_element_type(spec.primitive_type);
  shape_23.add_dimensions(1);
  shape_23.add_dimensions(222);
  shape_23.add_dimensions(222);
  shape_23.add_dimensions(32);
  auto broadcast_23 = builder.AddInstruction(HloInstruction::CreateBroadcast(shape_23, constant_22, {}));

  auto reshape_10 = builder.AddInstruction(HloInstruction::CreateReshape(I->shape(), I));

  auto reshape_11 = builder.AddInstruction(HloInstruction::CreateReshape(K1->shape(), K1));

  auto shape_19 = Shape();
  shape_19.set_element_type(spec.primitive_type);
  shape_19.add_dimensions(1);
  shape_19.add_dimensions(222);
  shape_19.add_dimensions(222);
  shape_19.add_dimensions(32);
  Window window_19;
  WindowDimension* dim_19_1 = window_19.add_dimensions();
  dim_19_1->set_size(3);
  dim_19_1->set_padding_low(0);
  dim_19_1->set_padding_high(0);
  dim_19_1->set_stride(1);
  dim_19_1->set_window_dilation(1);
  dim_19_1->set_base_dilation(1);
  dim_19_1->set_window_reversal(false);
  WindowDimension* dim_19_2 = window_19.add_dimensions();
  dim_19_2->set_size(3);
  dim_19_2->set_padding_low(0);
  dim_19_2->set_padding_high(0);
  dim_19_2->set_stride(1);
  dim_19_2->set_window_dilation(1);
  dim_19_2->set_base_dilation(1);
  dim_19_2->set_window_reversal(false);
  ConvolutionDimensionNumbers dnums_19;
  dnums_19.set_input_batch_dimension(0);
  dnums_19.add_input_spatial_dimensions(1);
  dnums_19.add_input_spatial_dimensions(2);
  dnums_19.set_input_feature_dimension(3);
  dnums_19.add_kernel_spatial_dimensions(0);
  dnums_19.add_kernel_spatial_dimensions(1);
  dnums_19.set_kernel_input_feature_dimension(2);
  dnums_19.set_kernel_output_feature_dimension(3);
  dnums_19.set_output_batch_dimension(0);
  dnums_19.add_output_spatial_dimensions(1);
  dnums_19.add_output_spatial_dimensions(2);
  dnums_19.set_output_feature_dimension(3);
  PrecisionConfig precision_config_19;
  precision_config_19.mutable_operand_precision()->Resize(
      /*new_size=*/2, PrecisionConfig::DEFAULT);
  auto convolution_19 = builder.AddInstruction(HloAllGatherInstruction::CreateConvolve(shape_19, reshape_10, reshape_11, 1, 1, window_19, dnums_19, precision_config_19));

  auto reshape_12 = builder.AddInstruction(HloInstruction::CreateReshape(B1->shape(), B1));

  auto shape_20 = Shape();
  shape_20.set_element_type(spec.primitive_type);
  shape_20.add_dimensions(1);
  shape_20.add_dimensions(222);
  shape_20.add_dimensions(222);
  shape_20.add_dimensions(32);
  auto broadcast_20 = builder.AddInstruction(HloInstruction::CreateBroadcast(shape_20, reshape_12, {3}));

  auto add_21 = builder.AddInstruction(HloInstruction::CreateBinary(convolution_19->shape(), HloOpcode::kAdd, convolution_19, broadcast_20));

  auto maximum_24 = builder.AddInstruction(HloInstruction::CreateBinary(broadcast_23->shape(), HloOpcode::kMaximum, broadcast_23, add_21));

  auto reshape_13 = builder.AddInstruction(HloInstruction::CreateReshape(K2->shape(), K2));

  auto shape_25 = Shape();
  shape_25.set_element_type(spec.primitive_type);
  shape_25.add_dimensions(1);
  shape_25.add_dimensions(220);
  shape_25.add_dimensions(220);
  shape_25.add_dimensions(64);
  Window window_25;
  WindowDimension* dim_25_1 = window_25.add_dimensions();
  dim_25_1->set_size(3);
  dim_25_1->set_padding_low(0);
  dim_25_1->set_padding_high(0);
  dim_25_1->set_stride(1);
  dim_25_1->set_window_dilation(1);
  dim_25_1->set_base_dilation(1);
  dim_25_1->set_window_reversal(false);
  WindowDimension* dim_25_2 = window_25.add_dimensions();
  dim_25_2->set_size(3);
  dim_25_2->set_padding_low(0);
  dim_25_2->set_padding_high(0);
  dim_25_2->set_stride(1);
  dim_25_2->set_window_dilation(1);
  dim_25_2->set_base_dilation(1);
  dim_25_2->set_window_reversal(false);
  ConvolutionDimensionNumbers dnums_25;
  dnums_25.set_input_batch_dimension(0);
  dnums_25.add_input_spatial_dimensions(1);
  dnums_25.add_input_spatial_dimensions(2);
  dnums_25.set_input_feature_dimension(3);
  dnums_25.add_kernel_spatial_dimensions(0);
  dnums_25.add_kernel_spatial_dimensions(1);
  dnums_25.set_kernel_input_feature_dimension(2);
  dnums_25.set_kernel_output_feature_dimension(3);
  dnums_25.set_output_batch_dimension(0);
  dnums_25.add_output_spatial_dimensions(1);
  dnums_25.add_output_spatial_dimensions(2);
  dnums_25.set_output_feature_dimension(3);
  PrecisionConfig precision_config_25;
  precision_config_25.mutable_operand_precision()->Resize(
      /*new_size=*/2, PrecisionConfig::DEFAULT);
  auto convolution_25 = builder.AddInstruction(HloAllGatherInstruction::CreateConvolve(shape_25, maximum_24, reshape_13, 1, 1, window_25, dnums_25, precision_config_25));

  auto reshape_14 = builder.AddInstruction(HloInstruction::CreateReshape(B2->shape(), B2));

  auto shape_26 = Shape();
  shape_26.set_element_type(spec.primitive_type);
  shape_26.add_dimensions(1);
  shape_26.add_dimensions(220);
  shape_26.add_dimensions(220);
  shape_26.add_dimensions(64);
  auto broadcast_26 = builder.AddInstruction(HloInstruction::CreateBroadcast(shape_26, reshape_14, {3}));

  auto add_27 = builder.AddInstruction(HloInstruction::CreateBinary(convolution_25->shape(), HloOpcode::kAdd, convolution_25, broadcast_26));


  // Max pool

   auto constant_28 = builder.AddInstruction(HloInstruction::CreateConstant(LiteralUtil::MinValue(spec.primitive_type)));

   auto shape_33 = Shape();
   shape_33.set_element_type(spec.primitive_type);
   shape_33.add_dimensions(1);
   shape_33.add_dimensions(219);
   shape_33.add_dimensions(219);
   shape_33.add_dimensions(64);
   Window window_33;
   WindowDimension* dim_33_1 = window_33.add_dimensions();
   dim_33_1->set_size(1);
   dim_33_1->set_padding_low(0);
   dim_33_1->set_padding_high(0);
   dim_33_1->set_stride(1);
   dim_33_1->set_window_dilation(1);
   dim_33_1->set_base_dilation(1);
   dim_33_1->set_window_reversal(false);
   WindowDimension* dim_33_2 = window_33.add_dimensions();
   dim_33_2->set_size(2);
   dim_33_2->set_padding_low(0);
   dim_33_2->set_padding_high(0);
   dim_33_2->set_stride(1);
   dim_33_2->set_window_dilation(1);
   dim_33_2->set_base_dilation(1);
   dim_33_2->set_window_reversal(false);
   WindowDimension* dim_33_3 = window_33.add_dimensions();
   dim_33_3->set_size(2);
   dim_33_3->set_padding_low(0);
   dim_33_3->set_padding_high(0);
   dim_33_3->set_stride(1);
   dim_33_3->set_window_dilation(1);
   dim_33_3->set_base_dilation(1);
   dim_33_3->set_window_reversal(false);
   WindowDimension* dim_33_4 = window_33.add_dimensions();
   dim_33_4->set_size(1);
   dim_33_4->set_padding_low(0);
   dim_33_4->set_padding_high(0);
   dim_33_4->set_stride(1);
   dim_33_4->set_window_dilation(1);
   dim_33_4->set_base_dilation(1);
   dim_33_4->set_window_reversal(false);
   auto reduce_window_33 = builder.AddInstruction(HloInstruction::CreateReduceWindow(shape_33, add_27, constant_28, window_33, max_computation));  

  auto maximum_36 = builder.AddInstruction(HloInstruction::CreateBinary(broadcast_35->shape(), HloOpcode::kMaximum, broadcast_35, reduce_window_33));

  auto shape_39 = Shape();
  shape_39.set_element_type(spec.primitive_type);
  shape_39.add_dimensions(47961);
  shape_39.add_dimensions(64);
  auto reshape_39 = builder.AddInstruction(HloInstruction::CreateReshape(shape_39, maximum_36));

  auto reshape_15 = builder.AddInstruction(HloInstruction::CreateReshape(K3->shape(), K3));
  auto reshape_37 = builder.AddInstruction(HloInstruction::CreateReshape(reshape_15->shape(), reshape_15));

  auto shape_40 = Shape();
  shape_40.set_element_type(spec.primitive_type);
  shape_40.add_dimensions(47961);
  shape_40.add_dimensions(128);
  DotDimensionNumbers dnums_40;
  dnums_40.add_lhs_contracting_dimensions(1);
  dnums_40.add_rhs_contracting_dimensions(0);
  PrecisionConfig precision_config_40;
  precision_config_40.mutable_operand_precision()->Resize(
      /*new_size=*/2, PrecisionConfig::DEFAULT);
  auto dot_40 = builder.AddInstruction(HloInstruction::CreateDot(shape_40, reshape_39, reshape_37, dnums_40, precision_config_40));

  auto shape_41 = Shape();
  shape_41.set_element_type(spec.primitive_type);
  shape_41.add_dimensions(1);
  shape_41.add_dimensions(219);
  shape_41.add_dimensions(219);
  shape_41.add_dimensions(128);
  auto reshape_41 = builder.AddInstruction(HloInstruction::CreateReshape(shape_41, dot_40));

  auto add_43 = builder.AddInstruction(HloInstruction::CreateBinary(broadcast_42->shape(), HloOpcode::kAdd, broadcast_42, reshape_41));

  auto maximum_46 = builder.AddInstruction(HloInstruction::CreateBinary(broadcast_45->shape(), HloOpcode::kMaximum, broadcast_45, add_43));

  auto shape_47 = Shape();
  shape_47.set_element_type(spec.primitive_type);
  shape_47.add_dimensions(47961);
  shape_47.add_dimensions(128);
  auto reshape_47 = builder.AddInstruction(HloInstruction::CreateReshape(shape_47, maximum_46));

  auto reshape_17 = builder.AddInstruction(HloInstruction::CreateReshape(K4->shape(), K4));
  auto reshape_38 = builder.AddInstruction(HloInstruction::CreateReshape(reshape_17->shape(), reshape_17));

  auto shape_48 = Shape();
  shape_48.set_element_type(spec.primitive_type);
  shape_48.add_dimensions(47961);
  shape_48.add_dimensions(100);
  DotDimensionNumbers dnums_48;
  dnums_48.add_lhs_contracting_dimensions(1);
  dnums_48.add_rhs_contracting_dimensions(0);
  PrecisionConfig precision_config_48;
  precision_config_48.mutable_operand_precision()->Resize(
      /*new_size=*/2, PrecisionConfig::DEFAULT);
  auto dot_48 = builder.AddInstruction(HloInstruction::CreateDot(shape_48, reshape_47, reshape_38, dnums_48, precision_config_48));

  auto shape_49 = Shape();
  shape_49.set_element_type(spec.primitive_type);
  shape_49.add_dimensions(1);
  shape_49.add_dimensions(219);
  shape_49.add_dimensions(219);
  shape_49.add_dimensions(100);
  auto reshape_49 = builder.AddInstruction(HloInstruction::CreateReshape(shape_49, dot_48));

  auto add_51 = builder.AddInstruction(HloInstruction::CreateBinary(broadcast_50->shape(), HloOpcode::kAdd, broadcast_50, reshape_49));

  auto constant_52 = builder.AddInstruction(HloInstruction::CreateConstant(LiteralUtil::MinValue(spec.primitive_type)));

  auto shape_57 = Shape();
  shape_57.set_element_type(spec.primitive_type);
  shape_57.add_dimensions(1);
  shape_57.add_dimensions(219);
  shape_57.add_dimensions(219);
  absl::Span<const int64> dims_57 = {3};
  auto reduce_57 = builder.AddInstruction(HloInstruction::CreateReduce(shape_57, add_51, constant_52, dims_57, max_computation));

  auto shape_58 = Shape();
  shape_58.set_element_type(spec.primitive_type);
  shape_58.add_dimensions(1);
  shape_58.add_dimensions(219);
  shape_58.add_dimensions(219);
  shape_58.add_dimensions(100);
  auto broadcast_58 = builder.AddInstruction(HloInstruction::CreateBroadcast(shape_58, reduce_57, {0, 1, 2}));

  auto subtract_59 = builder.AddInstruction(HloInstruction::CreateBinary(add_51->shape(), HloOpcode::kSubtract, add_51, broadcast_58)); 

  auto exponential_60 = builder.AddInstruction(HloInstruction::CreateUnary(subtract_59->shape(), HloOpcode::kExp, subtract_59));

  auto convert_61 = builder.AddInstruction(HloInstruction::CreateConvert(exponential_60->shape(), exponential_60));

  auto constant_62 = builder.AddInstruction(HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0))); 

  auto shape_67 = Shape();
  shape_67.set_element_type(spec.primitive_type);
  shape_67.add_dimensions(1);
  shape_67.add_dimensions(219);
  shape_67.add_dimensions(219);
  absl::Span<const int64> dims_67 = {3};
  auto reduce_67 = builder.AddInstruction(HloInstruction::CreateReduce(shape_67, convert_61, constant_62, dims_67, add_computation));

  auto convert_68 = builder.AddInstruction(HloInstruction::CreateConvert(reduce_67->shape(), reduce_67));

  auto shape_69 = Shape();
  shape_69.set_element_type(spec.primitive_type);
  shape_69.add_dimensions(1);
  shape_69.add_dimensions(219);
  shape_69.add_dimensions(219);
  shape_69.add_dimensions(100);
  auto broadcast_69 = builder.AddInstruction(HloInstruction::CreateBroadcast(shape_69, convert_68, {0, 1, 2}));

  auto subtract_70 = builder.AddInstruction(HloInstruction::CreateBinary(exponential_60->shape(), HloOpcode::kDivide, exponential_60, broadcast_69));

  auto reshape_71 = builder.AddInstruction(HloInstruction::CreateReshape(subtract_70->shape(), subtract_70));

  hlo_module->AddEntryComputation(builder.Build());

  CompileAndCheck(std::move(hlo_module), spec.filecheck_lines);
}

std::vector<MnistCnnTestSpec> GetMnistCnnTestCases() {
  std::vector<MnistCnnTestSpec> result;
  result.push_back(
      {F32, R"(CHECK: func @hlo_module(%arg0: tensor<128x100xf32>, %arg1: tensor<f32>, %arg2: tensor<64x128xf32>, %arg3: tensor<f32>, %arg4: tensor<64xf32>, %arg5: tensor<3x3x32x64xf32>, %arg6: tensor<f32>, %arg7: tensor<32xf32>, %arg8: tensor<3x3x1x32xf32>, %arg9: tensor<1x224x224x1xf32>, %arg10: tensor<128xf32>, %arg11: tensor<100xf32>) -> tensor<1x219x219x100xf32>)"});
  //result.push_back(
  //    {F64, R"(CHECK: func @hlo_module(%arg0: tensor<1x3xf32>, %arg1: tensor<1x3xf32>) -> tensor<1x3xf32>)"});
  return result;
}

/**/
// TODO: INSTANTIATE_TEST_CASE_P was deprecated in favor for INSTANTIATE_TEST_SUITE_P, but the version of gtest that bazel links in is looking for INSTANTIATE_TEST_CASE_P right now.
INSTANTIATE_TEST_CASE_P(All,
                         PlaidMLMnistCnnOperationTest,
                         ::testing::ValuesIn(GetMnistCnnTestCases()),
                         MnistCnnTestSpecToString);
/**/
}  // namespace
}  // namespace plaidml
}  // namespace xla
