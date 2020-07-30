// Tests that show HLO Module conversion to PlaidML Program.

#include <algorithm>
#include <string>

#include <gtest/gtest.h>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/service/plaidml/compiler.h"
#include "tensorflow/compiler/xla/service/plaidml/tests/plaidml_codegen_test.h"
#include "tensorflow/compiler/xla/service/plaidml/tests/i3d_pretrained_inputs_and_weights.h"
#include "tensorflow/compiler/xla/service/plaidml/tests/i3d_3b_output.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/tests/verified_hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "plaidml/testenv.h"

using ::plaidml::edsl::TensorBuffers;

namespace xla {
namespace plaidml {
namespace {

using TestCaseVal = std::vector<std::vector<float>>;
using TestCasePairs = std::map<TestCaseVal, TestCaseVal>;

struct I3DTestSpec {
  PrimitiveType primitive_type;
  string filecheck_lines;
};

string I3DTestSpecToString(const ::testing::TestParamInfo<I3DTestSpec>& info) {
  return PrimitiveType_Name(info.param.primitive_type);
}

class PlaidMLI3DOperationTest
    : public PlaidMLCodegenTest,
      public ::testing::WithParamInterface<I3DTestSpec> {
 protected:
  Status CompileAndCheck(std::unique_ptr<HloModule> hlo_module,
                       const string& filecheck_lines,
                       const TestCasePairs& testcase_pairs) {

    auto program = CompileToProgram(std::move(hlo_module));

    StatusOr<bool> fc_result = RunFileCheck(program->str(), filecheck_lines);

    //TF_ASSERT_OK(fc_result.status());
    EXPECT_TRUE(fc_result.ValueOrDie());
    
    VLOG(0) << "Evaluating results";

    for (auto pair : testcase_pairs) {

      TensorBuffers inp;
      TensorBuffers exp;

      auto program_inputs = program->inputs();
      auto tcp_inputs = pair.first;

      if (tcp_inputs.size() != program_inputs.size()) {
        VLOG(1) << "Found mismatch in input sizes: tcp " << tcp_inputs.size() << " program " << program_inputs.size();
      }

      for (auto i = 0; i < program_inputs.size(); i++) {
        VLOG(1) << "Adding TestCaseInput " << i;
        inp.insert(std::make_pair(program_inputs[i].tensor, pair.first[i]));
      }

      auto program_outputs = program->outputs();
      auto tcp_outputs = pair.second;

      if (tcp_outputs.size() != program_outputs.size()) {
        VLOG(1) << "Found mismatch in output sizes: tcp " << tcp_outputs.size() << " program " << program_outputs.size();
      }

      for (auto i = 0; i < program_outputs.size(); i++) {
        VLOG(1) << "Adding TestCaseOutput " << i;
        exp.insert(std::make_pair(program_outputs[i].tensor, pair.second[i]));
      }

      VLOG(0) << "Calling checkProgram";

      checkProgram(*program, inp, exp);

    }

    return Status::OK();

  }
};

TEST_P(PlaidMLI3DOperationTest, SimpleI3D) {
  std::vector<float> input_tensor(4816896, 1.0);

  TestCaseVal I3D_WeightsInputs = {{0}, ::weights::RGB_inception_i3d_Mixed_3b_Branch_0_Conv3d_0a_1x1_conv_3d_w, {0}, ::weights::RGB_inception_i3d_Conv3d_2c_3x3_conv_3d_w, //
                                   {0}, ::weights::RGB_inception_i3d_Conv3d_2b_1x1_conv_3d_w, {0}, ::weights::RGB_inception_i3d_Conv3d_1a_7x7_conv_3d_w, input_tensor, //
				   {0.001}, ::weights::RGB_inception_i3d_Conv3d_1a_7x7_batch_norm_beta, {0.001}, ::weights::RGB_inception_i3d_Conv3d_2b_1x1_batch_norm_beta, //
				   {0.001}, ::weights::RGB_inception_i3d_Conv3d_2c_3x3_batch_norm_beta, {0.001}, ::weights::RGB_inception_i3d_Mixed_3b_Branch_0_Conv3d_0a_1x1_batch_norm_beta};

  TestCaseVal I3D_Output = ::outputs::I3D_Outputs;

  TestCasePairs testcase_pairs = {{I3D_WeightsInputs, I3D_Output}};

  I3DTestSpec spec = GetParam();

  HloModuleConfig cfg;

  //std::unique_ptr<HloModule> hlo_module = absl::make_unique<HloModule>("module", cfg);

  std::unique_ptr<VerifiedHloModule> hlo_module = absl::make_unique<VerifiedHloModule>(
      "module", cfg, false, false, nullptr);

std::string hlo_text = R"(HloModule cluster_1__XlaCompiledKernel_true__XlaNumConstantArgs_8__XlaNumResourceArgs_8_.266

%RGB_inception_i3d_Conv3d_1a_7x7_batch_norm_normalize_moments_mean-reduction.15 (x.16: f32[], y.17: f32[]) -> f32[] {
  %x.16 = f32[] parameter(0)
  %y.17 = f32[] parameter(1)
  ROOT %add.18 = f32[] add(f32[] %x.16, f32[] %y.17)
}

%RGB_inception_i3d_Conv3d_1a_7x7_batch_norm_normalize_moments_variance-reduction.39 (x.40: f32[], y.41: f32[]) -> f32[] {
  %x.40 = f32[] parameter(0)
  %y.41 = f32[] parameter(1)
  ROOT %add.42 = f32[] add(f32[] %x.40, f32[] %y.41)
}

%max_F32.69 (lhs.70: f32[], rhs.71: f32[]) -> f32[] {
  %lhs.70 = f32[] parameter(0)
  %rhs.71 = f32[] parameter(1)
  ROOT %maximum.72 = f32[] maximum(f32[] %lhs.70, f32[] %rhs.71)
}

%RGB_inception_i3d_Conv3d_2b_1x1_batch_norm_normalize_moments_mean-reduction.81 (x.82: f32[], y.83: f32[]) -> f32[] {
  %x.82 = f32[] parameter(0)
  %y.83 = f32[] parameter(1)
  ROOT %add.84 = f32[] add(f32[] %x.82, f32[] %y.83)
}

%RGB_inception_i3d_Conv3d_2b_1x1_batch_norm_normalize_moments_variance-reduction.105 (x.106: f32[], y.107: f32[]) -> f32[] {
  %x.106 = f32[] parameter(0)
  %y.107 = f32[] parameter(1)
  ROOT %add.108 = f32[] add(f32[] %x.106, f32[] %y.107)
}

%RGB_inception_i3d_Conv3d_2c_3x3_batch_norm_normalize_moments_mean-reduction.141 (x.142: f32[], y.143: f32[]) -> f32[] {
  %x.142 = f32[] parameter(0)
  %y.143 = f32[] parameter(1)
  ROOT %add.144 = f32[] add(f32[] %x.142, f32[] %y.143)
}

%RGB_inception_i3d_Conv3d_2c_3x3_batch_norm_normalize_moments_variance-reduction.165 (x.166: f32[], y.167: f32[]) -> f32[] {
  %x.166 = f32[] parameter(0)
  %y.167 = f32[] parameter(1)
  ROOT %add.168 = f32[] add(f32[] %x.166, f32[] %y.167)
}

%max_F32.195 (lhs.196: f32[], rhs.197: f32[]) -> f32[] {
  %lhs.196 = f32[] parameter(0)
  %rhs.197 = f32[] parameter(1)
  ROOT %maximum.198 = f32[] maximum(f32[] %lhs.196, f32[] %rhs.197)
}

%RGB_inception_i3d_Mixed_3b_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.207 (x.208: f32[], y.209: f32[]) -> f32[] {
  %x.208 = f32[] parameter(0)
  %y.209 = f32[] parameter(1)
  ROOT %add.210 = f32[] add(f32[] %x.208, f32[] %y.209)
}

%RGB_inception_i3d_Mixed_3b_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.231 (x.232: f32[], y.233: f32[]) -> f32[] {
  %x.232 = f32[] parameter(0)
  %y.233 = f32[] parameter(1)
  ROOT %add.234 = f32[] add(f32[] %x.232, f32[] %y.233)
}

ENTRY %cluster_1__XlaCompiledKernel_true__XlaNumConstantArgs_8__XlaNumResourceArgs_8_.266 (arg0.1: f32[1,32,224,224,3], arg1.2: f32[1,1,1,1,64], arg2.3: f32[7,7,7,3,64], arg3.4: f32[1,1,1,1,64], arg4.5: f32[1,1,1,1,64], arg5.6: f32[1,1,1,1,192], arg6.7: f32[1,1,1,64,64], arg7.8: f32[3,3,3,64,192], arg8.9: f32[1,1,1,192,64]) -> f32[1,16,28,28,64] {
  %constant.260 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/Relu"}
  %broadcast.261 = f32[1,16,28,28,64]{4,3,2,1,0} broadcast(f32[] %constant.260), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/Relu"}
  %constant.248 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.249 = f32[1,1,1,1,64]{4,3,2,1,0} broadcast(f32[] %constant.248), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %constant.200 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Conv3d_2c_3x3/Relu"}
  %broadcast.201 = f32[1,16,28,28,192]{4,3,2,1,0} broadcast(f32[] %constant.200), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Conv3d_2c_3x3/Relu"}
  %constant.134 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Conv3d_2b_1x1/Relu"}
  %broadcast.135 = f32[1,16,56,56,64]{4,3,2,1,0} broadcast(f32[] %constant.134), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Conv3d_2b_1x1/Relu"}
  %constant.74 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Conv3d_1a_7x7/Relu"}
  %broadcast.75 = f32[1,16,56,56,64]{4,3,2,1,0} broadcast(f32[] %constant.74), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Conv3d_1a_7x7/Relu"}
  %arg0.1 = f32[1,32,224,224,3]{4,3,2,1,0} parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.10 = f32[1,32,224,224,3]{4,3,2,1,0} reshape(f32[1,32,224,224,3]{4,3,2,1,0} %arg0.1)
  %arg2.3 = f32[7,7,7,3,64]{4,3,2,1,0} parameter(2), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.11 = f32[1,16,112,112,64]{4,3,2,1,0} convolution(f32[1,32,224,224,3]{4,3,2,1,0} %reshape.10, f32[7,7,7,3,64]{4,3,2,1,0} %arg2.3), window={size=7x7x7 stride=2x2x2 pad=2_3x2_3x2_3}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Conv3d_1a_7x7/conv_3d/convolution"}
  %convert.12 = f32[1,16,112,112,64]{4,3,2,1,0} convert(f32[1,16,112,112,64]{4,3,2,1,0} %convolution.11), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/mean"}
  %constant.13 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/mean"}
  %convert.14 = f32[] convert(f32[] %constant.13), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/mean"}
  %reduce.19 = f32[64]{0} reduce(f32[1,16,112,112,64]{4,3,2,1,0} %convert.12, f32[] %convert.14), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Conv3d_1a_7x7_batch_norm_normalize_moments_mean-reduction.15, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/mean"}
  %get-dimension-size.20 = s32[] get-dimension-size(f32[1,16,112,112,64]{4,3,2,1,0} %convert.12), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/mean"}
  %get-dimension-size.21 = s32[] get-dimension-size(f32[1,16,112,112,64]{4,3,2,1,0} %convert.12), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/mean"}
  %multiply.22 = s32[] multiply(s32[] %get-dimension-size.20, s32[] %get-dimension-size.21), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/mean"}
  %get-dimension-size.23 = s32[] get-dimension-size(f32[1,16,112,112,64]{4,3,2,1,0} %convert.12), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/mean"}
  %multiply.24 = s32[] multiply(s32[] %multiply.22, s32[] %get-dimension-size.23), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/mean"}
  %get-dimension-size.25 = s32[] get-dimension-size(f32[1,16,112,112,64]{4,3,2,1,0} %convert.12), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/mean"}
  %multiply.26 = s32[] multiply(s32[] %multiply.24, s32[] %get-dimension-size.25), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/mean"}
  %convert.27 = f32[] convert(s32[] %multiply.26), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/mean"}
  %broadcast.28 = f32[64]{0} broadcast(f32[] %convert.27), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/mean"}
  %divide.29 = f32[64]{0} divide(f32[64]{0} %reduce.19, f32[64]{0} %broadcast.28), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/mean"}
  %convert.30 = f32[64]{0} convert(f32[64]{0} %divide.29), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/mean"}
  %reshape.31 = f32[1,1,1,1,64]{4,3,2,1,0} reshape(f32[64]{0} %convert.30), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/mean"}
  %reshape.32 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %reshape.31), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.33 = f32[1,16,112,112,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.32), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.34 = f32[1,16,112,112,64]{4,3,2,1,0} subtract(f32[1,16,112,112,64]{4,3,2,1,0} %broadcast.33, f32[1,16,112,112,64]{4,3,2,1,0} %convolution.11), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.35 = f32[1,16,112,112,64]{4,3,2,1,0} multiply(f32[1,16,112,112,64]{4,3,2,1,0} %subtract.34, f32[1,16,112,112,64]{4,3,2,1,0} %subtract.34), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/SquaredDifference"}
  %convert.36 = f32[1,16,112,112,64]{4,3,2,1,0} convert(f32[1,16,112,112,64]{4,3,2,1,0} %multiply.35), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/variance"}
  %constant.37 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/variance"}
  %convert.38 = f32[] convert(f32[] %constant.37), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/variance"}
  %reduce.43 = f32[64]{0} reduce(f32[1,16,112,112,64]{4,3,2,1,0} %convert.36, f32[] %convert.38), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Conv3d_1a_7x7_batch_norm_normalize_moments_variance-reduction.39, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/variance"}
  %get-dimension-size.44 = s32[] get-dimension-size(f32[1,16,112,112,64]{4,3,2,1,0} %convert.36), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/variance"}
  %get-dimension-size.45 = s32[] get-dimension-size(f32[1,16,112,112,64]{4,3,2,1,0} %convert.36), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/variance"}
  %multiply.46 = s32[] multiply(s32[] %get-dimension-size.44, s32[] %get-dimension-size.45), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/variance"}
  %get-dimension-size.47 = s32[] get-dimension-size(f32[1,16,112,112,64]{4,3,2,1,0} %convert.36), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/variance"}
  %multiply.48 = s32[] multiply(s32[] %multiply.46, s32[] %get-dimension-size.47), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/variance"}
  %get-dimension-size.49 = s32[] get-dimension-size(f32[1,16,112,112,64]{4,3,2,1,0} %convert.36), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/variance"}
  %multiply.50 = s32[] multiply(s32[] %multiply.48, s32[] %get-dimension-size.49), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/variance"}
  %convert.51 = f32[] convert(s32[] %multiply.50), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/variance"}
  %broadcast.52 = f32[64]{0} broadcast(f32[] %convert.51), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/variance"}
  %divide.53 = f32[64]{0} divide(f32[64]{0} %reduce.43, f32[64]{0} %broadcast.52), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/variance"}
  %convert.54 = f32[64]{0} convert(f32[64]{0} %divide.53), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/variance"}
  %reshape.55 = f32[1,1,1,1,64]{4,3,2,1,0} reshape(f32[64]{0} %convert.54), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/variance"}
  %constant.56 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/add"}
  %broadcast.57 = f32[1,1,1,1,64]{4,3,2,1,0} broadcast(f32[] %constant.56), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/add"}
  %add.58 = f32[1,1,1,1,64]{4,3,2,1,0} add(f32[1,1,1,1,64]{4,3,2,1,0} %reshape.55, f32[1,1,1,1,64]{4,3,2,1,0} %broadcast.57), metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/add"}
  %rsqrt.59 = f32[1,1,1,1,64]{4,3,2,1,0} rsqrt(f32[1,1,1,1,64]{4,3,2,1,0} %add.58), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/Rsqrt"}
  %reshape.60 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.59), metadata={op_type="Mul" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/mul"}
  %broadcast.61 = f32[1,16,112,112,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.60), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/mul"}
  %multiply.62 = f32[1,16,112,112,64]{4,3,2,1,0} multiply(f32[1,16,112,112,64]{4,3,2,1,0} %broadcast.61, f32[1,16,112,112,64]{4,3,2,1,0} %convolution.11), metadata={op_type="Mul" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/mul"}
  %arg4.5 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(4), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.63 = f32[1,1,1,1,64]{4,3,2,1,0} multiply(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.59, f32[1,1,1,1,64]{4,3,2,1,0} %reshape.31), metadata={op_type="Mul" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/mul_1"}
  %subtract.64 = f32[1,1,1,1,64]{4,3,2,1,0} subtract(f32[1,1,1,1,64]{4,3,2,1,0} %arg4.5, f32[1,1,1,1,64]{4,3,2,1,0} %multiply.63), metadata={op_type="Sub" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/sub"}
  %reshape.65 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %subtract.64), metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/add_1"}
  %broadcast.66 = f32[1,16,112,112,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.65), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/add_1"}
  %add.67 = f32[1,16,112,112,64]{4,3,2,1,0} add(f32[1,16,112,112,64]{4,3,2,1,0} %multiply.62, f32[1,16,112,112,64]{4,3,2,1,0} %broadcast.66), metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/add_1"}
  %constant.68 = f32[] constant(-inf), metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/MaxPool3d_2a_3x3"}
  %reduce-window.73 = f32[1,16,56,56,64]{4,3,2,1,0} reduce-window(f32[1,16,112,112,64]{4,3,2,1,0} %add.67, f32[] %constant.68), window={size=1x1x3x3x1 stride=1x1x2x2x1 pad=0_0x0_0x0_1x0_1x0_0}, to_apply=%max_F32.69, metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/MaxPool3d_2a_3x3"}
  %maximum.76 = f32[1,16,56,56,64]{4,3,2,1,0} maximum(f32[1,16,56,56,64]{4,3,2,1,0} %broadcast.75, f32[1,16,56,56,64]{4,3,2,1,0} %reduce-window.73), metadata={op_type="Relu" op_name="RGB/inception_i3d/Conv3d_1a_7x7/Relu"}
  %arg6.7 = f32[1,1,1,64,64]{4,3,2,1,0} parameter(6), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.77 = f32[1,16,56,56,64]{4,3,2,1,0} convolution(f32[1,16,56,56,64]{4,3,2,1,0} %maximum.76, f32[1,1,1,64,64]{4,3,2,1,0} %arg6.7), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Conv3d_2b_1x1/conv_3d/convolution"}
  %convert.78 = f32[1,16,56,56,64]{4,3,2,1,0} convert(f32[1,16,56,56,64]{4,3,2,1,0} %convolution.77), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/mean"}
  %constant.79 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/mean"}
  %convert.80 = f32[] convert(f32[] %constant.79), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/mean"}
  %reduce.85 = f32[64]{0} reduce(f32[1,16,56,56,64]{4,3,2,1,0} %convert.78, f32[] %convert.80), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Conv3d_2b_1x1_batch_norm_normalize_moments_mean-reduction.81, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.86 = s32[] get-dimension-size(f32[1,16,56,56,64]{4,3,2,1,0} %convert.78), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.87 = s32[] get-dimension-size(f32[1,16,56,56,64]{4,3,2,1,0} %convert.78), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/mean"}
  %multiply.88 = s32[] multiply(s32[] %get-dimension-size.86, s32[] %get-dimension-size.87), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.89 = s32[] get-dimension-size(f32[1,16,56,56,64]{4,3,2,1,0} %convert.78), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/mean"}
  %multiply.90 = s32[] multiply(s32[] %multiply.88, s32[] %get-dimension-size.89), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.91 = s32[] get-dimension-size(f32[1,16,56,56,64]{4,3,2,1,0} %convert.78), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/mean"}
  %multiply.92 = s32[] multiply(s32[] %multiply.90, s32[] %get-dimension-size.91), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/mean"}
  %convert.93 = f32[] convert(s32[] %multiply.92), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.94 = f32[64]{0} broadcast(f32[] %convert.93), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/mean"}
  %divide.95 = f32[64]{0} divide(f32[64]{0} %reduce.85, f32[64]{0} %broadcast.94), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/mean"}
  %convert.96 = f32[64]{0} convert(f32[64]{0} %divide.95), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/mean"}
  %reshape.97 = f32[1,1,1,1,64]{4,3,2,1,0} reshape(f32[64]{0} %convert.96), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/mean"}
  %reshape.98 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %reshape.97), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.99 = f32[1,16,56,56,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.98), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.100 = f32[1,16,56,56,64]{4,3,2,1,0} subtract(f32[1,16,56,56,64]{4,3,2,1,0} %broadcast.99, f32[1,16,56,56,64]{4,3,2,1,0} %convolution.77), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.101 = f32[1,16,56,56,64]{4,3,2,1,0} multiply(f32[1,16,56,56,64]{4,3,2,1,0} %subtract.100, f32[1,16,56,56,64]{4,3,2,1,0} %subtract.100), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.102 = f32[1,16,56,56,64]{4,3,2,1,0} convert(f32[1,16,56,56,64]{4,3,2,1,0} %multiply.101), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/variance"}
  %constant.103 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/variance"}
  %convert.104 = f32[] convert(f32[] %constant.103), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/variance"}
  %reduce.109 = f32[64]{0} reduce(f32[1,16,56,56,64]{4,3,2,1,0} %convert.102, f32[] %convert.104), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Conv3d_2b_1x1_batch_norm_normalize_moments_variance-reduction.105, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.110 = s32[] get-dimension-size(f32[1,16,56,56,64]{4,3,2,1,0} %convert.102), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.111 = s32[] get-dimension-size(f32[1,16,56,56,64]{4,3,2,1,0} %convert.102), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/variance"}
  %multiply.112 = s32[] multiply(s32[] %get-dimension-size.110, s32[] %get-dimension-size.111), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.113 = s32[] get-dimension-size(f32[1,16,56,56,64]{4,3,2,1,0} %convert.102), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/variance"}
  %multiply.114 = s32[] multiply(s32[] %multiply.112, s32[] %get-dimension-size.113), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.115 = s32[] get-dimension-size(f32[1,16,56,56,64]{4,3,2,1,0} %convert.102), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/variance"}
  %multiply.116 = s32[] multiply(s32[] %multiply.114, s32[] %get-dimension-size.115), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/variance"}
  %convert.117 = f32[] convert(s32[] %multiply.116), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.118 = f32[64]{0} broadcast(f32[] %convert.117), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/variance"}
  %divide.119 = f32[64]{0} divide(f32[64]{0} %reduce.109, f32[64]{0} %broadcast.118), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/variance"}
  %convert.120 = f32[64]{0} convert(f32[64]{0} %divide.119), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/variance"}
  %reshape.121 = f32[1,1,1,1,64]{4,3,2,1,0} reshape(f32[64]{0} %convert.120), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/variance"}
  %constant.122 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/batch_norm/add"}
  %broadcast.123 = f32[1,1,1,1,64]{4,3,2,1,0} broadcast(f32[] %constant.122), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/batch_norm/add"}
  %add.124 = f32[1,1,1,1,64]{4,3,2,1,0} add(f32[1,1,1,1,64]{4,3,2,1,0} %reshape.121, f32[1,1,1,1,64]{4,3,2,1,0} %broadcast.123), metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/batch_norm/add"}
  %rsqrt.125 = f32[1,1,1,1,64]{4,3,2,1,0} rsqrt(f32[1,1,1,1,64]{4,3,2,1,0} %add.124), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.126 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.125), metadata={op_type="Mul" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/batch_norm/mul"}
  %broadcast.127 = f32[1,16,56,56,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.126), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/batch_norm/mul"}
  %multiply.128 = f32[1,16,56,56,64]{4,3,2,1,0} multiply(f32[1,16,56,56,64]{4,3,2,1,0} %broadcast.127, f32[1,16,56,56,64]{4,3,2,1,0} %convolution.77), metadata={op_type="Mul" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/batch_norm/mul"}
  %arg3.4 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(3), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.129 = f32[1,1,1,1,64]{4,3,2,1,0} multiply(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.125, f32[1,1,1,1,64]{4,3,2,1,0} %reshape.97), metadata={op_type="Mul" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.130 = f32[1,1,1,1,64]{4,3,2,1,0} subtract(f32[1,1,1,1,64]{4,3,2,1,0} %arg3.4, f32[1,1,1,1,64]{4,3,2,1,0} %multiply.129), metadata={op_type="Sub" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/batch_norm/sub"}
  %reshape.131 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %subtract.130), metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.132 = f32[1,16,56,56,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.131), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/batch_norm/add_1"}
  %add.133 = f32[1,16,56,56,64]{4,3,2,1,0} add(f32[1,16,56,56,64]{4,3,2,1,0} %multiply.128, f32[1,16,56,56,64]{4,3,2,1,0} %broadcast.132), metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/batch_norm/add_1"}
  %maximum.136 = f32[1,16,56,56,64]{4,3,2,1,0} maximum(f32[1,16,56,56,64]{4,3,2,1,0} %broadcast.135, f32[1,16,56,56,64]{4,3,2,1,0} %add.133), metadata={op_type="Relu" op_name="RGB/inception_i3d/Conv3d_2b_1x1/Relu"}
  %arg7.8 = f32[3,3,3,64,192]{4,3,2,1,0} parameter(7), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.137 = f32[1,16,56,56,192]{4,3,2,1,0} convolution(f32[1,16,56,56,64]{4,3,2,1,0} %maximum.136, f32[3,3,3,64,192]{4,3,2,1,0} %arg7.8), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Conv3d_2c_3x3/conv_3d/convolution"}
  %convert.138 = f32[1,16,56,56,192]{4,3,2,1,0} convert(f32[1,16,56,56,192]{4,3,2,1,0} %convolution.137), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/mean"}
  %constant.139 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/mean"}
  %convert.140 = f32[] convert(f32[] %constant.139), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/mean"}
  %reduce.145 = f32[192]{0} reduce(f32[1,16,56,56,192]{4,3,2,1,0} %convert.138, f32[] %convert.140), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Conv3d_2c_3x3_batch_norm_normalize_moments_mean-reduction.141, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.146 = s32[] get-dimension-size(f32[1,16,56,56,192]{4,3,2,1,0} %convert.138), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.147 = s32[] get-dimension-size(f32[1,16,56,56,192]{4,3,2,1,0} %convert.138), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/mean"}
  %multiply.148 = s32[] multiply(s32[] %get-dimension-size.146, s32[] %get-dimension-size.147), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.149 = s32[] get-dimension-size(f32[1,16,56,56,192]{4,3,2,1,0} %convert.138), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/mean"}
  %multiply.150 = s32[] multiply(s32[] %multiply.148, s32[] %get-dimension-size.149), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.151 = s32[] get-dimension-size(f32[1,16,56,56,192]{4,3,2,1,0} %convert.138), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/mean"}
  %multiply.152 = s32[] multiply(s32[] %multiply.150, s32[] %get-dimension-size.151), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/mean"}
  %convert.153 = f32[] convert(s32[] %multiply.152), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/mean"}
  %broadcast.154 = f32[192]{0} broadcast(f32[] %convert.153), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/mean"}
  %divide.155 = f32[192]{0} divide(f32[192]{0} %reduce.145, f32[192]{0} %broadcast.154), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/mean"}
  %convert.156 = f32[192]{0} convert(f32[192]{0} %divide.155), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/mean"}
  %reshape.157 = f32[1,1,1,1,192]{4,3,2,1,0} reshape(f32[192]{0} %convert.156), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/mean"}
  %reshape.158 = f32[1,192]{1,0} reshape(f32[1,1,1,1,192]{4,3,2,1,0} %reshape.157), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.159 = f32[1,16,56,56,192]{4,3,2,1,0} broadcast(f32[1,192]{1,0} %reshape.158), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.160 = f32[1,16,56,56,192]{4,3,2,1,0} subtract(f32[1,16,56,56,192]{4,3,2,1,0} %broadcast.159, f32[1,16,56,56,192]{4,3,2,1,0} %convolution.137), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.161 = f32[1,16,56,56,192]{4,3,2,1,0} multiply(f32[1,16,56,56,192]{4,3,2,1,0} %subtract.160, f32[1,16,56,56,192]{4,3,2,1,0} %subtract.160), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %convert.162 = f32[1,16,56,56,192]{4,3,2,1,0} convert(f32[1,16,56,56,192]{4,3,2,1,0} %multiply.161), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/variance"}
  %constant.163 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/variance"}
  %convert.164 = f32[] convert(f32[] %constant.163), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/variance"}
  %reduce.169 = f32[192]{0} reduce(f32[1,16,56,56,192]{4,3,2,1,0} %convert.162, f32[] %convert.164), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Conv3d_2c_3x3_batch_norm_normalize_moments_variance-reduction.165, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.170 = s32[] get-dimension-size(f32[1,16,56,56,192]{4,3,2,1,0} %convert.162), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.171 = s32[] get-dimension-size(f32[1,16,56,56,192]{4,3,2,1,0} %convert.162), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/variance"}
  %multiply.172 = s32[] multiply(s32[] %get-dimension-size.170, s32[] %get-dimension-size.171), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.173 = s32[] get-dimension-size(f32[1,16,56,56,192]{4,3,2,1,0} %convert.162), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/variance"}
  %multiply.174 = s32[] multiply(s32[] %multiply.172, s32[] %get-dimension-size.173), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.175 = s32[] get-dimension-size(f32[1,16,56,56,192]{4,3,2,1,0} %convert.162), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/variance"}
  %multiply.176 = s32[] multiply(s32[] %multiply.174, s32[] %get-dimension-size.175), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/variance"}
  %convert.177 = f32[] convert(s32[] %multiply.176), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/variance"}
  %broadcast.178 = f32[192]{0} broadcast(f32[] %convert.177), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/variance"}
  %divide.179 = f32[192]{0} divide(f32[192]{0} %reduce.169, f32[192]{0} %broadcast.178), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/variance"}
  %convert.180 = f32[192]{0} convert(f32[192]{0} %divide.179), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/variance"}
  %reshape.181 = f32[1,1,1,1,192]{4,3,2,1,0} reshape(f32[192]{0} %convert.180), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/variance"}
  %constant.182 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/batch_norm/add"}
  %broadcast.183 = f32[1,1,1,1,192]{4,3,2,1,0} broadcast(f32[] %constant.182), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/batch_norm/add"}
  %add.184 = f32[1,1,1,1,192]{4,3,2,1,0} add(f32[1,1,1,1,192]{4,3,2,1,0} %reshape.181, f32[1,1,1,1,192]{4,3,2,1,0} %broadcast.183), metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/batch_norm/add"}
  %rsqrt.185 = f32[1,1,1,1,192]{4,3,2,1,0} rsqrt(f32[1,1,1,1,192]{4,3,2,1,0} %add.184), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.186 = f32[1,192]{1,0} reshape(f32[1,1,1,1,192]{4,3,2,1,0} %rsqrt.185), metadata={op_type="Mul" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/batch_norm/mul"}
  %broadcast.187 = f32[1,16,56,56,192]{4,3,2,1,0} broadcast(f32[1,192]{1,0} %reshape.186), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/batch_norm/mul"}
  %multiply.188 = f32[1,16,56,56,192]{4,3,2,1,0} multiply(f32[1,16,56,56,192]{4,3,2,1,0} %broadcast.187, f32[1,16,56,56,192]{4,3,2,1,0} %convolution.137), metadata={op_type="Mul" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/batch_norm/mul"}
  %arg5.6 = f32[1,1,1,1,192]{4,3,2,1,0} parameter(5), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.189 = f32[1,1,1,1,192]{4,3,2,1,0} multiply(f32[1,1,1,1,192]{4,3,2,1,0} %rsqrt.185, f32[1,1,1,1,192]{4,3,2,1,0} %reshape.157), metadata={op_type="Mul" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.190 = f32[1,1,1,1,192]{4,3,2,1,0} subtract(f32[1,1,1,1,192]{4,3,2,1,0} %arg5.6, f32[1,1,1,1,192]{4,3,2,1,0} %multiply.189), metadata={op_type="Sub" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/batch_norm/sub"}
  %reshape.191 = f32[1,192]{1,0} reshape(f32[1,1,1,1,192]{4,3,2,1,0} %subtract.190), metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.192 = f32[1,16,56,56,192]{4,3,2,1,0} broadcast(f32[1,192]{1,0} %reshape.191), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/batch_norm/add_1"}
  %add.193 = f32[1,16,56,56,192]{4,3,2,1,0} add(f32[1,16,56,56,192]{4,3,2,1,0} %multiply.188, f32[1,16,56,56,192]{4,3,2,1,0} %broadcast.192), metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/batch_norm/add_1"}
  %constant.194 = f32[] constant(-inf), metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/MaxPool3d_3a_3x3"}
  %reduce-window.199 = f32[1,16,28,28,192]{4,3,2,1,0} reduce-window(f32[1,16,56,56,192]{4,3,2,1,0} %add.193, f32[] %constant.194), window={size=1x1x3x3x1 stride=1x1x2x2x1 pad=0_0x0_0x0_1x0_1x0_0}, to_apply=%max_F32.195, metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/MaxPool3d_3a_3x3"}
  %maximum.202 = f32[1,16,28,28,192]{4,3,2,1,0} maximum(f32[1,16,28,28,192]{4,3,2,1,0} %broadcast.201, f32[1,16,28,28,192]{4,3,2,1,0} %reduce-window.199), metadata={op_type="Relu" op_name="RGB/inception_i3d/Conv3d_2c_3x3/Relu"}
  %arg8.9 = f32[1,1,1,192,64]{4,3,2,1,0} parameter(8), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.203 = f32[1,16,28,28,64]{4,3,2,1,0} convolution(f32[1,16,28,28,192]{4,3,2,1,0} %maximum.202, f32[1,1,1,192,64]{4,3,2,1,0} %arg8.9), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/conv_3d/convolution"}
  %convert.204 = f32[1,16,28,28,64]{4,3,2,1,0} convert(f32[1,16,28,28,64]{4,3,2,1,0} %convolution.203), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %constant.205 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.206 = f32[] convert(f32[] %constant.205), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reduce.211 = f32[64]{0} reduce(f32[1,16,28,28,64]{4,3,2,1,0} %convert.204, f32[] %convert.206), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_3b_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.207, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.212 = s32[] get-dimension-size(f32[1,16,28,28,64]{4,3,2,1,0} %convert.204), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.213 = s32[] get-dimension-size(f32[1,16,28,28,64]{4,3,2,1,0} %convert.204), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.214 = s32[] multiply(s32[] %get-dimension-size.212, s32[] %get-dimension-size.213), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.215 = s32[] get-dimension-size(f32[1,16,28,28,64]{4,3,2,1,0} %convert.204), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.216 = s32[] multiply(s32[] %multiply.214, s32[] %get-dimension-size.215), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.217 = s32[] get-dimension-size(f32[1,16,28,28,64]{4,3,2,1,0} %convert.204), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.218 = s32[] multiply(s32[] %multiply.216, s32[] %get-dimension-size.217), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.219 = f32[] convert(s32[] %multiply.218), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.220 = f32[64]{0} broadcast(f32[] %convert.219), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %divide.221 = f32[64]{0} divide(f32[64]{0} %reduce.211, f32[64]{0} %broadcast.220), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.222 = f32[64]{0} convert(f32[64]{0} %divide.221), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.223 = f32[1,1,1,1,64]{4,3,2,1,0} reshape(f32[64]{0} %convert.222), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.224 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %reshape.223), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.225 = f32[1,16,28,28,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.224), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.226 = f32[1,16,28,28,64]{4,3,2,1,0} subtract(f32[1,16,28,28,64]{4,3,2,1,0} %broadcast.225, f32[1,16,28,28,64]{4,3,2,1,0} %convolution.203), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.227 = f32[1,16,28,28,64]{4,3,2,1,0} multiply(f32[1,16,28,28,64]{4,3,2,1,0} %subtract.226, f32[1,16,28,28,64]{4,3,2,1,0} %subtract.226), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.228 = f32[1,16,28,28,64]{4,3,2,1,0} convert(f32[1,16,28,28,64]{4,3,2,1,0} %multiply.227), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %constant.229 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.230 = f32[] convert(f32[] %constant.229), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reduce.235 = f32[64]{0} reduce(f32[1,16,28,28,64]{4,3,2,1,0} %convert.228, f32[] %convert.230), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_3b_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.231, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.236 = s32[] get-dimension-size(f32[1,16,28,28,64]{4,3,2,1,0} %convert.228), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.237 = s32[] get-dimension-size(f32[1,16,28,28,64]{4,3,2,1,0} %convert.228), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.238 = s32[] multiply(s32[] %get-dimension-size.236, s32[] %get-dimension-size.237), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.239 = s32[] get-dimension-size(f32[1,16,28,28,64]{4,3,2,1,0} %convert.228), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.240 = s32[] multiply(s32[] %multiply.238, s32[] %get-dimension-size.239), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.241 = s32[] get-dimension-size(f32[1,16,28,28,64]{4,3,2,1,0} %convert.228), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.242 = s32[] multiply(s32[] %multiply.240, s32[] %get-dimension-size.241), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.243 = f32[] convert(s32[] %multiply.242), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.244 = f32[64]{0} broadcast(f32[] %convert.243), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %divide.245 = f32[64]{0} divide(f32[64]{0} %reduce.235, f32[64]{0} %broadcast.244), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.246 = f32[64]{0} convert(f32[64]{0} %divide.245), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reshape.247 = f32[1,1,1,1,64]{4,3,2,1,0} reshape(f32[64]{0} %convert.246), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %add.250 = f32[1,1,1,1,64]{4,3,2,1,0} add(f32[1,1,1,1,64]{4,3,2,1,0} %broadcast.249, f32[1,1,1,1,64]{4,3,2,1,0} %reshape.247), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.251 = f32[1,1,1,1,64]{4,3,2,1,0} rsqrt(f32[1,1,1,1,64]{4,3,2,1,0} %add.250), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.252 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.251), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.253 = f32[1,16,28,28,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.252), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %multiply.254 = f32[1,16,28,28,64]{4,3,2,1,0} multiply(f32[1,16,28,28,64]{4,3,2,1,0} %broadcast.253, f32[1,16,28,28,64]{4,3,2,1,0} %convolution.203), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg1.2 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.255 = f32[1,1,1,1,64]{4,3,2,1,0} multiply(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.251, f32[1,1,1,1,64]{4,3,2,1,0} %reshape.223), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.256 = f32[1,1,1,1,64]{4,3,2,1,0} subtract(f32[1,1,1,1,64]{4,3,2,1,0} %arg1.2, f32[1,1,1,1,64]{4,3,2,1,0} %multiply.255), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.257 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %subtract.256), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.258 = f32[1,16,28,28,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.257), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.259 = f32[1,16,28,28,64]{4,3,2,1,0} add(f32[1,16,28,28,64]{4,3,2,1,0} %multiply.254, f32[1,16,28,28,64]{4,3,2,1,0} %broadcast.258), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %maximum.262 = f32[1,16,28,28,64]{4,3,2,1,0} maximum(f32[1,16,28,28,64]{4,3,2,1,0} %broadcast.261, f32[1,16,28,28,64]{4,3,2,1,0} %add.259), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/Relu"}
  %reshape.263 = f32[1,16,28,28,64]{4,3,2,1,0} reshape(f32[1,16,28,28,64]{4,3,2,1,0} %maximum.262), metadata={op_name="XLA_Retvals"}
  %tuple.264 = (f32[1,16,28,28,64]{4,3,2,1,0}) tuple(f32[1,16,28,28,64]{4,3,2,1,0} %reshape.263), metadata={op_name="XLA_Retvals"}
  ROOT %get-tuple-element.265 = f32[1,16,28,28,64]{4,3,2,1,0} get-tuple-element((f32[1,16,28,28,64]{4,3,2,1,0}) %tuple.264), index=0, metadata={op_name="XLA_Retvals"}
}
)";

  hlo_module->ParseHloStringAndVerifyModule(hlo_text); 

  CompileAndCheck(std::move(hlo_module), spec.filecheck_lines, testcase_pairs);
}

std::vector<I3DTestSpec> GetI3DTestCases() {
  std::vector<I3DTestSpec> result;
  result.push_back(
      {F32, R"(CHECK: func @hlo_module)"});
  return result;
}

/**/
// TODO: INSTANTIATE_TEST_CASE_P was deprecated in favor for INSTANTIATE_TEST_SUITE_P, but the version of gtest that bazel links in is looking for INSTANTIATE_TEST_CASE_P right now.
INSTANTIATE_TEST_CASE_P(All,
                         PlaidMLI3DOperationTest,
                         ::testing::ValuesIn(GetI3DTestCases()),
                         I3DTestSpecToString);
/**/
}  // namespace
}  // namespace plaidml
}  // namespace xla
