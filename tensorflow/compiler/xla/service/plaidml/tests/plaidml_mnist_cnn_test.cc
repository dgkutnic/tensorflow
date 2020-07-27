// Tests that show HLO Module conversion to PlaidML Program.

#include <algorithm>
#include <string>

#include <gtest/gtest.h>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/service/plaidml/compiler.h"
#include "tensorflow/compiler/xla/service/plaidml/tests/plaidml_codegen_test.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/tests/verified_hlo_module.h"
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

  HloModuleConfig cfg;

  //std::unique_ptr<HloModule> hlo_module = absl::make_unique<HloModule>("module", cfg);

  std::unique_ptr<VerifiedHloModule> hlo_module = absl::make_unique<VerifiedHloModule>(
      "module", cfg, false, false, nullptr);

std::string hlo_text = R"(HloModule cluster_0__XlaCompiledKernel_true__XlaHasReferenceVars_false__XlaNumConstantArgs_6__XlaNumResourceArgs_0_.74

%max_F32.29 (lhs.30: f32[], rhs.31: f32[]) -> f32[] {
  %lhs.30 = f32[] parameter(0)
  %rhs.31 = f32[] parameter(1)
  ROOT %maximum.32 = f32[] maximum(f32[] %lhs.30, f32[] %rhs.31)
}

%max_float_.53 (x.54: f32[], y.55: f32[]) -> f32[] {
  %x.54 = f32[] parameter(0)
  %y.55 = f32[] parameter(1)
  ROOT %maximum.56 = f32[] maximum(f32[] %x.54, f32[] %y.55)
}

%add_float_.63 (x.64: f32[], y.65: f32[]) -> f32[] {
  %x.64 = f32[] parameter(0)
  %y.65 = f32[] parameter(1)
  ROOT %add.66 = f32[] add(f32[] %x.64, f32[] %y.65)
}

ENTRY %cluster_0__XlaCompiledKernel_true__XlaHasReferenceVars_false__XlaNumConstantArgs_6__XlaNumResourceArgs_0_.74 (arg0.1: f32[1,224,224,1], arg1.2: f32[3,3,1,32], arg2.3: f32[32], arg3.4: f32[3,3,32,64], arg4.5: f32[64], arg5.6: f32[64,128], arg6.7: f32[128], arg7.8: f32[128,100], arg8.9: f32[100]) -> f32[1,219,219,100] {
  %arg8.9 = f32[100]{0} parameter(8), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.18 = f32[100]{0} reshape(f32[100]{0} %arg8.9)
  %broadcast.50 = f32[1,219,219,100]{3,2,1,0} broadcast(f32[100]{0} %reshape.18), dimensions={3}, metadata={op_type="AddV2" op_name="add_3"}
  %constant.44 = f32[] constant(0), metadata={op_type="Relu" op_name="Relu_2"}
  %broadcast.45 = f32[1,219,219,128]{3,2,1,0} broadcast(f32[] %constant.44), dimensions={}, metadata={op_type="Relu" op_name="Relu_2"}
  %arg6.7 = f32[128]{0} parameter(6), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.16 = f32[128]{0} reshape(f32[128]{0} %arg6.7)
  %broadcast.42 = f32[1,219,219,128]{3,2,1,0} broadcast(f32[128]{0} %reshape.16), dimensions={3}, metadata={op_type="AddV2" op_name="add_2"}
  %constant.34 = f32[] constant(0), metadata={op_type="Relu" op_name="Relu_1"}
  %broadcast.35 = f32[1,219,219,64]{3,2,1,0} broadcast(f32[] %constant.34), dimensions={}, metadata={op_type="Relu" op_name="Relu_1"}
  %constant.22 = f32[] constant(0), metadata={op_type="Relu" op_name="Relu"}
  %broadcast.23 = f32[1,222,222,32]{3,2,1,0} broadcast(f32[] %constant.22), dimensions={}, metadata={op_type="Relu" op_name="Relu"}
  %arg0.1 = f32[1,224,224,1]{3,2,1,0} parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.10 = f32[1,224,224,1]{3,2,1,0} reshape(f32[1,224,224,1]{3,2,1,0} %arg0.1)
  %arg1.2 = f32[3,3,1,32]{3,2,1,0} parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.11 = f32[3,3,1,32]{3,2,1,0} reshape(f32[3,3,1,32]{3,2,1,0} %arg1.2)
  %convolution.19 = f32[1,222,222,32]{3,2,1,0} convolution(f32[1,224,224,1]{3,2,1,0} %reshape.10, f32[3,3,1,32]{3,2,1,0} %reshape.11), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="Conv2D"}
  %arg2.3 = f32[32]{0} parameter(2), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.12 = f32[32]{0} reshape(f32[32]{0} %arg2.3)
  %broadcast.20 = f32[1,222,222,32]{3,2,1,0} broadcast(f32[32]{0} %reshape.12), dimensions={3}, metadata={op_type="AddV2" op_name="add"}
  %add.21 = f32[1,222,222,32]{3,2,1,0} add(f32[1,222,222,32]{3,2,1,0} %convolution.19, f32[1,222,222,32]{3,2,1,0} %broadcast.20), metadata={op_type="AddV2" op_name="add"}
  %maximum.24 = f32[1,222,222,32]{3,2,1,0} maximum(f32[1,222,222,32]{3,2,1,0} %broadcast.23, f32[1,222,222,32]{3,2,1,0} %add.21), metadata={op_type="Relu" op_name="Relu"}
  %arg3.4 = f32[3,3,32,64]{3,2,1,0} parameter(3), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.13 = f32[3,3,32,64]{3,2,1,0} reshape(f32[3,3,32,64]{3,2,1,0} %arg3.4)
  %convolution.25 = f32[1,220,220,64]{3,2,1,0} convolution(f32[1,222,222,32]{3,2,1,0} %maximum.24, f32[3,3,32,64]{3,2,1,0} %reshape.13), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="Conv2D_1"}
  %arg4.5 = f32[64]{0} parameter(4), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.14 = f32[64]{0} reshape(f32[64]{0} %arg4.5)
  %broadcast.26 = f32[1,220,220,64]{3,2,1,0} broadcast(f32[64]{0} %reshape.14), dimensions={3}, metadata={op_type="AddV2" op_name="add_1"}
  %add.27 = f32[1,220,220,64]{3,2,1,0} add(f32[1,220,220,64]{3,2,1,0} %convolution.25, f32[1,220,220,64]{3,2,1,0} %broadcast.26), metadata={op_type="AddV2" op_name="add_1"}
  %constant.28 = f32[] constant(-inf), metadata={op_type="MaxPool" op_name="MaxPool2d"}
  %reduce-window.33 = f32[1,219,219,64]{3,2,1,0} reduce-window(f32[1,220,220,64]{3,2,1,0} %add.27, f32[] %constant.28), window={size=1x2x2x1}, to_apply=%max_F32.29, metadata={op_type="MaxPool" op_name="MaxPool2d"}
  %maximum.36 = f32[1,219,219,64]{3,2,1,0} maximum(f32[1,219,219,64]{3,2,1,0} %broadcast.35, f32[1,219,219,64]{3,2,1,0} %reduce-window.33), metadata={op_type="Relu" op_name="Relu_1"}
  %reshape.39 = f32[47961,64]{1,0} reshape(f32[1,219,219,64]{3,2,1,0} %maximum.36), inferred_dimension=0, metadata={op_type="Reshape" op_name="Reshape"}
  %arg5.6 = f32[64,128]{1,0} parameter(5), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.15 = f32[64,128]{1,0} reshape(f32[64,128]{1,0} %arg5.6)
  %reshape.37 = f32[64,128]{1,0} reshape(f32[64,128]{1,0} %reshape.15), inferred_dimension=1, metadata={op_type="Reshape" op_name="Reshape_1"}
  %dot.40 = f32[47961,128]{1,0} dot(f32[47961,64]{1,0} %reshape.39, f32[64,128]{1,0} %reshape.37), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="MatMul"}
  %reshape.41 = f32[1,219,219,128]{3,2,1,0} reshape(f32[47961,128]{1,0} %dot.40), metadata={op_type="Reshape" op_name="Reshape_2"}
  %add.43 = f32[1,219,219,128]{3,2,1,0} add(f32[1,219,219,128]{3,2,1,0} %broadcast.42, f32[1,219,219,128]{3,2,1,0} %reshape.41), metadata={op_type="AddV2" op_name="add_2"}
  %maximum.46 = f32[1,219,219,128]{3,2,1,0} maximum(f32[1,219,219,128]{3,2,1,0} %broadcast.45, f32[1,219,219,128]{3,2,1,0} %add.43), metadata={op_type="Relu" op_name="Relu_2"}
  %reshape.47 = f32[47961,128]{1,0} reshape(f32[1,219,219,128]{3,2,1,0} %maximum.46), inferred_dimension=0, metadata={op_type="Reshape" op_name="Reshape_3"}
  %arg7.8 = f32[128,100]{1,0} parameter(7), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.17 = f32[128,100]{1,0} reshape(f32[128,100]{1,0} %arg7.8)
  %reshape.38 = f32[128,100]{1,0} reshape(f32[128,100]{1,0} %reshape.17), inferred_dimension=1, metadata={op_type="Reshape" op_name="Reshape_4"}
  %dot.48 = f32[47961,100]{1,0} dot(f32[47961,128]{1,0} %reshape.47, f32[128,100]{1,0} %reshape.38), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="MatMul_1"}
  %reshape.49 = f32[1,219,219,100]{3,2,1,0} reshape(f32[47961,100]{1,0} %dot.48), metadata={op_type="Reshape" op_name="Reshape_5"}
  %add.51 = f32[1,219,219,100]{3,2,1,0} add(f32[1,219,219,100]{3,2,1,0} %broadcast.50, f32[1,219,219,100]{3,2,1,0} %reshape.49), metadata={op_type="AddV2" op_name="add_3"}
  %constant.52 = f32[] constant(-inf), metadata={op_type="Softmax" op_name="Softmax"}
  %reduce.57 = f32[1,219,219]{2,1,0} reduce(f32[1,219,219,100]{3,2,1,0} %add.51, f32[] %constant.52), dimensions={3}, to_apply=%max_float_.53, metadata={op_type="Softmax" op_name="Softmax"}
  %broadcast.58 = f32[1,219,219,100]{3,2,1,0} broadcast(f32[1,219,219]{2,1,0} %reduce.57), dimensions={0,1,2}, metadata={op_type="Softmax" op_name="Softmax"}
  %subtract.59 = f32[1,219,219,100]{3,2,1,0} subtract(f32[1,219,219,100]{3,2,1,0} %add.51, f32[1,219,219,100]{3,2,1,0} %broadcast.58), metadata={op_type="Softmax" op_name="Softmax"}
  %exponential.60 = f32[1,219,219,100]{3,2,1,0} exponential(f32[1,219,219,100]{3,2,1,0} %subtract.59), metadata={op_type="Softmax" op_name="Softmax"}
  %convert.61 = f32[1,219,219,100]{3,2,1,0} convert(f32[1,219,219,100]{3,2,1,0} %exponential.60), metadata={op_type="Softmax" op_name="Softmax"}
  %constant.62 = f32[] constant(0), metadata={op_type="Softmax" op_name="Softmax"}
  %reduce.67 = f32[1,219,219]{2,1,0} reduce(f32[1,219,219,100]{3,2,1,0} %convert.61, f32[] %constant.62), dimensions={3}, to_apply=%add_float_.63, metadata={op_type="Softmax" op_name="Softmax"}
  %convert.68 = f32[1,219,219]{2,1,0} convert(f32[1,219,219]{2,1,0} %reduce.67), metadata={op_type="Softmax" op_name="Softmax"}
  %broadcast.69 = f32[1,219,219,100]{3,2,1,0} broadcast(f32[1,219,219]{2,1,0} %convert.68), dimensions={0,1,2}, metadata={op_type="Softmax" op_name="Softmax"}
  %divide.70 = f32[1,219,219,100]{3,2,1,0} divide(f32[1,219,219,100]{3,2,1,0} %exponential.60, f32[1,219,219,100]{3,2,1,0} %broadcast.69), metadata={op_type="Softmax" op_name="Softmax"}
  %reshape.71 = f32[1,219,219,100]{3,2,1,0} reshape(f32[1,219,219,100]{3,2,1,0} %divide.70), metadata={op_name="XLA_Retvals"}
  %tuple.72 = (f32[1,219,219,100]{3,2,1,0}) tuple(f32[1,219,219,100]{3,2,1,0} %reshape.71), metadata={op_name="XLA_Retvals"}
  ROOT %get-tuple-element.73 = f32[1,219,219,100]{3,2,1,0} get-tuple-element((f32[1,219,219,100]{3,2,1,0}) %tuple.72), index=0, metadata={op_name="XLA_Retvals"}
}
)";

  hlo_module->ParseHloStringAndVerifyModule(hlo_text); 

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
