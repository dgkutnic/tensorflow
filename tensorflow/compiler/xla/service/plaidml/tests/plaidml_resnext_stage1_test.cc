// Tests that show HLO Module conversion to PlaidML Program.

#include <algorithm>
#include <string>

#include <gtest/gtest.h>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/service/plaidml/compiler.h"
#include "tensorflow/compiler/xla/service/plaidml/tests/plaidml_codegen_test.h"
#include "tensorflow/compiler/xla/service/plaidml/tests/resnext50_pretrained_inputs_and_weights.h"
#include "tensorflow/compiler/xla/service/plaidml/tests/resnext50_stage1_output.h"
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

struct ResNeXTTestSpec {
  PrimitiveType primitive_type;
  string filecheck_lines;
};

string ResNeXTTestSpecToString(const ::testing::TestParamInfo<ResNeXTTestSpec>& info) {
  return PrimitiveType_Name(info.param.primitive_type);
}

class PlaidMLResNeXTOperationTest
    : public PlaidMLCodegenTest,
      public ::testing::WithParamInterface<ResNeXTTestSpec> {
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

      VLOG(0) << "checkProgram complete";
    }

    return Status::OK();

  }
};

TEST_P(PlaidMLResNeXTOperationTest, SimpleResNeXT) {

  TestCaseVal ResNeXt50_WeightsInputs = {
    {0},
    ::weights::stage1_unit3_bn3_mean,
    ::weights::stage1_unit3_bn3_scale,
    ::weights::stage1_unit3_bn3_var,
    {2e-05},
    ::weights::stage1_unit3_bn3_bias,
    ::weights::stage1_unit3_conv3_weight,
    {0},
    ::weights::stage1_unit3_bn2_mean,
    ::weights::stage1_unit3_bn2_scale,
    ::weights::stage1_unit3_bn2_var,
    {2e-05},
    ::weights::stage1_unit3_bn2_bias,
    ::weights::stage1_unit3_conv2_weight,
    {0},
    ::weights::stage1_unit3_bn1_mean,
    ::weights::stage1_unit3_bn1_scale,
    ::weights::stage1_unit3_bn1_var,
    {2e-05},
    ::weights::stage1_unit3_bn1_bias,
    ::weights::stage1_unit3_conv1_weight,
    {0},
    ::weights::stage1_unit2_bn3_mean,
    ::weights::stage1_unit2_bn3_scale,
    ::weights::stage1_unit2_bn3_var,
    {2e-05},
    ::weights::stage1_unit2_bn3_bias,
    ::weights::stage1_unit2_conv3_weight,
    {0},
    ::weights::stage1_unit2_bn2_mean,
    ::weights::stage1_unit2_bn2_scale,
    ::weights::stage1_unit2_bn2_var,
    {2e-05},
    ::weights::stage1_unit2_bn2_bias,
    ::weights::stage1_unit2_conv2_weight,
    {0},
    ::weights::stage1_unit2_bn1_mean,
    ::weights::stage1_unit2_bn1_scale,
    ::weights::stage1_unit2_bn1_var,
    {2e-05},
    ::weights::stage1_unit2_bn1_bias,
    ::weights::stage1_unit2_conv1_weight,
    {0},
    ::weights::stage1_unit1_sc_bn_mean,
    ::weights::stage1_unit1_sc_bn_scale,
    ::weights::stage1_unit1_sc_bn_var,
    {2e-05},
    ::weights::stage1_unit1_sc_bn_bias,
    ::weights::stage1_unit1_sc_weight,
    {0},
    ::weights::bn0_mean,
    ::weights::bn0_scale,
    {2e-05},
    ::weights::bn0_var,
    ::weights::bn0_bias,
    ::weights::conv0_weight,
    ::weights::bn_data_mean,
    ::weights::bn_data_var,
    {2e-05},
    ::weights::bn_data_bias,
    ::weights::input_tensor,
    ::weights::stage1_unit1_bn3_mean,
    ::weights::stage1_unit1_bn3_scale,
    ::weights::stage1_unit1_bn3_var,
    {2e-05},
    ::weights::stage1_unit1_bn3_bias,
    ::weights::stage1_unit1_conv3_weight,
    {0},
    ::weights::stage1_unit1_bn2_mean,
    ::weights::stage1_unit1_bn2_scale,
    ::weights::stage1_unit1_bn2_var,
    {2e-05},
    ::weights::stage1_unit1_bn2_bias,
    ::weights::stage1_unit1_conv2_weight,
    {0},
    ::weights::stage1_unit1_bn1_mean,
    ::weights::stage1_unit1_bn1_scale,
    ::weights::stage1_unit1_bn1_var,
    {2e-05},
    ::weights::stage1_unit1_bn1_bias,
    ::weights::stage1_unit1_conv1_weight
  };

  TestCaseVal ResNeXt50_Outputs = ::outputs::ResNext50_Outputs; 

  TestCasePairs testcase_pairs = {{ResNeXt50_WeightsInputs, ResNeXt50_Outputs}}; 

  ResNeXTTestSpec spec = GetParam();

  HloModuleConfig cfg;

  std::unique_ptr<VerifiedHloModule> hlo_module = absl::make_unique<VerifiedHloModule>(
      "module", cfg, false, false, nullptr);

std::string hlo_text = R"(
  HloModule cluster_0__XlaCompiledKernel_true__XlaNumConstantArgs_14__XlaNumResourceArgs_0_.601

%max_F32.149 (lhs.150: f32[], rhs.151: f32[]) -> f32[] {
  %lhs.150 = f32[] parameter(0)
  %rhs.151 = f32[] parameter(1)
  ROOT %maximum.152 = f32[] maximum(f32[] %lhs.150, f32[] %rhs.151)
}

ENTRY %cluster_0__XlaCompiledKernel_true__XlaNumConstantArgs_14__XlaNumResourceArgs_0_.601 (arg0.1: f32[128], arg1.2: f32[3], arg2.3: f32[64], arg3.4: f32[128], arg4.5: f32[3,3,4,128], arg5.6: f32[256], arg6.7: f32[256], arg7.8: f32[3,3,4,128], arg8.9: f32[128], arg9.10: f32[256], arg10.11: f32[128], arg11.12: f32[3,3,4,128], arg12.13: f32[128], arg13.14: f32[128], arg14.15: f32[256], arg15.16: f32[128], arg16.17: f32[3], arg17.18: f32[64], arg18.19: f32[128], arg19.20: f32[256], arg20.21: f32[256], arg21.22: f32[128], arg22.23: f32[256], arg23.24: f32[128], arg24.25: f32[128], arg25.26: f32[128], arg26.27: f32[256], arg27.28: f32[128], arg28.29: f32[3], arg29.30: f32[64], arg30.31: f32[128], arg31.32: f32[256], arg32.33: f32[256], arg33.34: f32[128], arg34.35: f32[256], arg35.36: f32[128], arg36.37: f32[128], arg37.38: f32[128], arg38.39: f32[256], arg39.40: f32[128], arg40.41: f32[64], arg41.42: f32[128], arg42.43: f32[256], arg43.44: f32[256], arg44.45: f32[128], arg45.46: f32[256], arg46.47: f32[128], arg47.48: f32[128], arg48.49: f32[128], arg49.50: f32[256], arg50.51: f32[7,7,3,64], arg51.52: f32[1,1,64,128], arg52.53: f32[1,1,64,256], arg53.54: f32[1,1,128,256], arg54.55: f32[1,1,256,128], arg55.56: f32[1,1,128,256], arg56.57: f32[1,1,256,128], arg57.58: f32[1,1,128,256], arg58.59: f32[1,224,224,3]) -> f32[1,56,56,256] {
  %constant.595 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit3_relu"}
  %broadcast.596 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[] %constant.595), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit3_relu"}
  %constant.451 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit2_relu"}
  %broadcast.452 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[] %constant.451), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit2_relu"}
  %constant.307 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit1_relu"}
  %broadcast.308 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[] %constant.307), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit1_relu"}
  %constant.168 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit1_bn3/add"}
  %broadcast.169 = f32[256]{0} broadcast(f32[] %constant.168), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit1_bn3/add"}
  %arg9.10 = f32[256]{0} parameter(9), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.69 = f32[256]{0} reshape(f32[256]{0} %arg9.10)
  %add.170 = f32[256]{0} add(f32[256]{0} %broadcast.169, f32[256]{0} %reshape.69), metadata={op_type="AddV2" op_name="stage1_unit1_bn3/add"}
  %rsqrt.171 = f32[256]{0} rsqrt(f32[256]{0} %add.170), metadata={op_type="Rsqrt" op_name="stage1_unit1_bn3/Rsqrt"}
  %arg22.23 = f32[256]{0} parameter(22), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.82 = f32[256]{0} reshape(f32[256]{0} %arg22.23)
  %multiply.172 = f32[256]{0} multiply(f32[256]{0} %rsqrt.171, f32[256]{0} %reshape.82), metadata={op_type="Mul" op_name="stage1_unit1_bn3/mul"}
  %broadcast.290 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.172), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit1_bn3/mul_1"}
  %constant.286 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit1_relu2"}
  %broadcast.287 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[] %constant.286), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit1_relu2"}
  %constant.180 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit1_relu1"}
  %broadcast.181 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[] %constant.180), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit1_relu1"}
  %constant.154 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit1_bn1/add"}
  %broadcast.155 = f32[128]{0} broadcast(f32[] %constant.154), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit1_bn1/add"}
  %arg3.4 = f32[128]{0} parameter(3), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.63 = f32[128]{0} reshape(f32[128]{0} %arg3.4)
  %add.156 = f32[128]{0} add(f32[128]{0} %broadcast.155, f32[128]{0} %reshape.63), metadata={op_type="AddV2" op_name="stage1_unit1_bn1/add"}
  %rsqrt.157 = f32[128]{0} rsqrt(f32[128]{0} %add.156), metadata={op_type="Rsqrt" op_name="stage1_unit1_bn1/Rsqrt"}
  %arg18.19 = f32[128]{0} parameter(18), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.78 = f32[128]{0} reshape(f32[128]{0} %arg18.19)
  %multiply.158 = f32[128]{0} multiply(f32[128]{0} %rsqrt.157, f32[128]{0} %reshape.78), metadata={op_type="Mul" op_name="stage1_unit1_bn1/mul"}
  %broadcast.176 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %multiply.158), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit1_bn1/mul_1"}
  %constant.143 = f32[] constant(0), metadata={op_type="Relu" op_name="relu0"}
  %broadcast.144 = f32[1,112,112,64]{3,2,1,0} broadcast(f32[] %constant.143), dimensions={}, metadata={op_type="Relu" op_name="relu0"}
  %arg2.3 = f32[64]{0} parameter(2), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.62 = f32[64]{0} reshape(f32[64]{0} %arg2.3)
  %constant.119 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="bn0/add"}
  %broadcast.120 = f32[64]{0} broadcast(f32[] %constant.119), dimensions={}, metadata={op_type="AddV2" op_name="bn0/add"}
  %add.121 = f32[64]{0} add(f32[64]{0} %reshape.62, f32[64]{0} %broadcast.120), metadata={op_type="AddV2" op_name="bn0/add"}
  %rsqrt.122 = f32[64]{0} rsqrt(f32[64]{0} %add.121), metadata={op_type="Rsqrt" op_name="bn0/Rsqrt"}
  %arg17.18 = f32[64]{0} parameter(17), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.77 = f32[64]{0} reshape(f32[64]{0} %arg17.18)
  %multiply.123 = f32[64]{0} multiply(f32[64]{0} %rsqrt.122, f32[64]{0} %reshape.77), metadata={op_type="Mul" op_name="bn0/mul"}
  %broadcast.139 = f32[1,112,112,64]{3,2,1,0} broadcast(f32[64]{0} %multiply.123), dimensions={3}, metadata={op_type="Mul" op_name="bn0/mul_1"}
  %constant.126 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="bn_data/add"}
  %broadcast.127 = f32[3]{0} broadcast(f32[] %constant.126), dimensions={}, metadata={op_type="AddV2" op_name="bn_data/add"}
  %arg1.2 = f32[3]{0} parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.61 = f32[3]{0} reshape(f32[3]{0} %arg1.2)
  %add.128 = f32[3]{0} add(f32[3]{0} %broadcast.127, f32[3]{0} %reshape.61), metadata={op_type="AddV2" op_name="bn_data/add"}
  %rsqrt.129 = f32[3]{0} rsqrt(f32[3]{0} %add.128), metadata={op_type="Rsqrt" op_name="bn_data/Rsqrt"}
  %broadcast.130 = f32[1,224,224,3]{3,2,1,0} broadcast(f32[3]{0} %rsqrt.129), dimensions={3}, metadata={op_type="Mul" op_name="bn_data/mul"}
  %arg58.59 = f32[1,224,224,3]{3,2,1,0} parameter(58), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.118 = f32[1,224,224,3]{3,2,1,0} reshape(f32[1,224,224,3]{3,2,1,0} %arg58.59)
  %multiply.131 = f32[1,224,224,3]{3,2,1,0} multiply(f32[1,224,224,3]{3,2,1,0} %broadcast.130, f32[1,224,224,3]{3,2,1,0} %reshape.118), metadata={op_type="Mul" op_name="bn_data/mul"}
  %arg28.29 = f32[3]{0} parameter(28), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.88 = f32[3]{0} reshape(f32[3]{0} %arg28.29)
  %arg16.17 = f32[3]{0} parameter(16), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.76 = f32[3]{0} reshape(f32[3]{0} %arg16.17)
  %multiply.132 = f32[3]{0} multiply(f32[3]{0} %rsqrt.129, f32[3]{0} %reshape.76), metadata={op_type="Mul" op_name="bn_data/mul_1"}
  %subtract.133 = f32[3]{0} subtract(f32[3]{0} %reshape.88, f32[3]{0} %multiply.132), metadata={op_type="Sub" op_name="bn_data/sub"}
  %broadcast.134 = f32[1,224,224,3]{3,2,1,0} broadcast(f32[3]{0} %subtract.133), dimensions={3}, metadata={op_type="AddV2" op_name="bn_data/add_1"}
  %add.135 = f32[1,224,224,3]{3,2,1,0} add(f32[1,224,224,3]{3,2,1,0} %multiply.131, f32[1,224,224,3]{3,2,1,0} %broadcast.134), metadata={op_type="AddV2" op_name="bn_data/add_1"}
  %constant.136 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad"}
  %pad.137 = f32[1,230,230,3]{3,2,1,0} pad(f32[1,224,224,3]{3,2,1,0} %add.135, f32[] %constant.136), padding=0_0x3_3x3_3x0_0, metadata={op_type="Pad" op_name="Pad"}
  %arg50.51 = f32[7,7,3,64]{3,2,1,0} parameter(50), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.110 = f32[7,7,3,64]{3,2,1,0} reshape(f32[7,7,3,64]{3,2,1,0} %arg50.51)
  %convolution.138 = f32[1,112,112,64]{3,2,1,0} convolution(f32[1,230,230,3]{3,2,1,0} %pad.137, f32[7,7,3,64]{3,2,1,0} %reshape.110), window={size=7x7 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="conv0"}
  %multiply.140 = f32[1,112,112,64]{3,2,1,0} multiply(f32[1,112,112,64]{3,2,1,0} %broadcast.139, f32[1,112,112,64]{3,2,1,0} %convolution.138), metadata={op_type="Mul" op_name="bn0/mul_1"}
  %arg40.41 = f32[64]{0} parameter(40), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.100 = f32[64]{0} reshape(f32[64]{0} %arg40.41)
  %arg29.30 = f32[64]{0} parameter(29), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.89 = f32[64]{0} reshape(f32[64]{0} %arg29.30)
  %multiply.124 = f32[64]{0} multiply(f32[64]{0} %multiply.123, f32[64]{0} %reshape.89), metadata={op_type="Mul" op_name="bn0/mul_2"}
  %subtract.125 = f32[64]{0} subtract(f32[64]{0} %reshape.100, f32[64]{0} %multiply.124), metadata={op_type="Sub" op_name="bn0/sub"}
  %broadcast.141 = f32[1,112,112,64]{3,2,1,0} broadcast(f32[64]{0} %subtract.125), dimensions={3}, metadata={op_type="AddV2" op_name="bn0/add_1"}
  %add.142 = f32[1,112,112,64]{3,2,1,0} add(f32[1,112,112,64]{3,2,1,0} %multiply.140, f32[1,112,112,64]{3,2,1,0} %broadcast.141), metadata={op_type="AddV2" op_name="bn0/add_1"}
  %maximum.145 = f32[1,112,112,64]{3,2,1,0} maximum(f32[1,112,112,64]{3,2,1,0} %broadcast.144, f32[1,112,112,64]{3,2,1,0} %add.142), metadata={op_type="Relu" op_name="relu0"}
  %constant.146 = f32[] constant(-inf), metadata={op_type="PadV2" op_name="PadV2"}
  %pad.147 = f32[1,114,114,64]{3,2,1,0} pad(f32[1,112,112,64]{3,2,1,0} %maximum.145, f32[] %constant.146), padding=0_0x1_1x1_1x0_0, metadata={op_type="PadV2" op_name="PadV2"}
  %constant.148 = f32[] constant(-inf), metadata={op_type="MaxPool" op_name="pooling0"}
  %reduce-window.153 = f32[1,56,56,64]{3,2,1,0} reduce-window(f32[1,114,114,64]{3,2,1,0} %pad.147, f32[] %constant.148), window={size=1x3x3x1 stride=1x2x2x1}, to_apply=%max_F32.149, metadata={op_type="MaxPool" op_name="pooling0"}
  %arg51.52 = f32[1,1,64,128]{3,2,1,0} parameter(51), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.111 = f32[1,1,64,128]{3,2,1,0} reshape(f32[1,1,64,128]{3,2,1,0} %arg51.52)
  %convolution.175 = f32[1,56,56,128]{3,2,1,0} convolution(f32[1,56,56,64]{3,2,1,0} %reduce-window.153, f32[1,1,64,128]{3,2,1,0} %reshape.111), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv1"}
  %multiply.177 = f32[1,56,56,128]{3,2,1,0} multiply(f32[1,56,56,128]{3,2,1,0} %broadcast.176, f32[1,56,56,128]{3,2,1,0} %convolution.175), metadata={op_type="Mul" op_name="stage1_unit1_bn1/mul_1"}
  %arg41.42 = f32[128]{0} parameter(41), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.101 = f32[128]{0} reshape(f32[128]{0} %arg41.42)
  %arg30.31 = f32[128]{0} parameter(30), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.90 = f32[128]{0} reshape(f32[128]{0} %arg30.31)
  %multiply.159 = f32[128]{0} multiply(f32[128]{0} %multiply.158, f32[128]{0} %reshape.90), metadata={op_type="Mul" op_name="stage1_unit1_bn1/mul_2"}
  %subtract.160 = f32[128]{0} subtract(f32[128]{0} %reshape.101, f32[128]{0} %multiply.159), metadata={op_type="Sub" op_name="stage1_unit1_bn1/sub"}
  %broadcast.178 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %subtract.160), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit1_bn1/add_1"}
  %add.179 = f32[1,56,56,128]{3,2,1,0} add(f32[1,56,56,128]{3,2,1,0} %multiply.177, f32[1,56,56,128]{3,2,1,0} %broadcast.178), metadata={op_type="AddV2" op_name="stage1_unit1_bn1/add_1"}
  %maximum.182 = f32[1,56,56,128]{3,2,1,0} maximum(f32[1,56,56,128]{3,2,1,0} %broadcast.181, f32[1,56,56,128]{3,2,1,0} %add.179), metadata={op_type="Relu" op_name="stage1_unit1_relu1"}
  %constant.183 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_1"}
  %pad.184 = f32[1,58,58,128]{3,2,1,0} pad(f32[1,56,56,128]{3,2,1,0} %maximum.182, f32[] %constant.183), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_1"}
  %slice.185 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.184), slice={[0:1], [0:58], [0:58], [0:4]}, metadata={op_type="Split" op_name="split_1"}
  %arg7.8 = f32[3,3,4,128]{3,2,1,0} parameter(7), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.67 = f32[3,3,4,128]{3,2,1,0} reshape(f32[3,3,4,128]{3,2,1,0} %arg7.8)
  %slice.217 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.67), slice={[0:3], [0:3], [0:4], [0:4]}, metadata={op_type="Split" op_name="split"}
  %convolution.249 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.185, f32[3,3,4,4]{3,2,1,0} %slice.217), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2"}
  %slice.186 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.184), slice={[0:1], [0:58], [0:58], [4:8]}, metadata={op_type="Split" op_name="split_1"}
  %slice.218 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.67), slice={[0:3], [0:3], [0:4], [4:8]}, metadata={op_type="Split" op_name="split"}
  %convolution.250 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.186, f32[3,3,4,4]{3,2,1,0} %slice.218), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_1"}
  %slice.187 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.184), slice={[0:1], [0:58], [0:58], [8:12]}, metadata={op_type="Split" op_name="split_1"}
  %slice.219 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.67), slice={[0:3], [0:3], [0:4], [8:12]}, metadata={op_type="Split" op_name="split"}
  %convolution.261 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.187, f32[3,3,4,4]{3,2,1,0} %slice.219), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_2"}
  %slice.188 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.184), slice={[0:1], [0:58], [0:58], [12:16]}, metadata={op_type="Split" op_name="split_1"}
  %slice.220 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.67), slice={[0:3], [0:3], [0:4], [12:16]}, metadata={op_type="Split" op_name="split"}
  %convolution.272 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.188, f32[3,3,4,4]{3,2,1,0} %slice.220), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_3"}
  %slice.189 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.184), slice={[0:1], [0:58], [0:58], [16:20]}, metadata={op_type="Split" op_name="split_1"}
  %slice.221 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.67), slice={[0:3], [0:3], [0:4], [16:20]}, metadata={op_type="Split" op_name="split"}
  %convolution.275 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.189, f32[3,3,4,4]{3,2,1,0} %slice.221), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_4"}
  %slice.190 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.184), slice={[0:1], [0:58], [0:58], [20:24]}, metadata={op_type="Split" op_name="split_1"}
  %slice.222 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.67), slice={[0:3], [0:3], [0:4], [20:24]}, metadata={op_type="Split" op_name="split"}
  %convolution.276 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.190, f32[3,3,4,4]{3,2,1,0} %slice.222), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_5"}
  %slice.191 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.184), slice={[0:1], [0:58], [0:58], [24:28]}, metadata={op_type="Split" op_name="split_1"}
  %slice.223 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.67), slice={[0:3], [0:3], [0:4], [24:28]}, metadata={op_type="Split" op_name="split"}
  %convolution.277 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.191, f32[3,3,4,4]{3,2,1,0} %slice.223), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_6"}
  %slice.192 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.184), slice={[0:1], [0:58], [0:58], [28:32]}, metadata={op_type="Split" op_name="split_1"}
  %slice.224 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.67), slice={[0:3], [0:3], [0:4], [28:32]}, metadata={op_type="Split" op_name="split"}
  %convolution.278 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.192, f32[3,3,4,4]{3,2,1,0} %slice.224), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_7"}
  %slice.193 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.184), slice={[0:1], [0:58], [0:58], [32:36]}, metadata={op_type="Split" op_name="split_1"}
  %slice.225 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.67), slice={[0:3], [0:3], [0:4], [32:36]}, metadata={op_type="Split" op_name="split"}
  %convolution.279 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.193, f32[3,3,4,4]{3,2,1,0} %slice.225), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_8"}
  %slice.194 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.184), slice={[0:1], [0:58], [0:58], [36:40]}, metadata={op_type="Split" op_name="split_1"}
  %slice.226 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.67), slice={[0:3], [0:3], [0:4], [36:40]}, metadata={op_type="Split" op_name="split"}
  %convolution.280 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.194, f32[3,3,4,4]{3,2,1,0} %slice.226), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_9"}
  %slice.195 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.184), slice={[0:1], [0:58], [0:58], [40:44]}, metadata={op_type="Split" op_name="split_1"}
  %slice.227 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.67), slice={[0:3], [0:3], [0:4], [40:44]}, metadata={op_type="Split" op_name="split"}
  %convolution.251 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.195, f32[3,3,4,4]{3,2,1,0} %slice.227), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_10"}
  %slice.196 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.184), slice={[0:1], [0:58], [0:58], [44:48]}, metadata={op_type="Split" op_name="split_1"}
  %slice.228 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.67), slice={[0:3], [0:3], [0:4], [44:48]}, metadata={op_type="Split" op_name="split"}
  %convolution.252 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.196, f32[3,3,4,4]{3,2,1,0} %slice.228), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_11"}
  %slice.197 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.184), slice={[0:1], [0:58], [0:58], [48:52]}, metadata={op_type="Split" op_name="split_1"}
  %slice.229 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.67), slice={[0:3], [0:3], [0:4], [48:52]}, metadata={op_type="Split" op_name="split"}
  %convolution.253 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.197, f32[3,3,4,4]{3,2,1,0} %slice.229), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_12"}
  %slice.198 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.184), slice={[0:1], [0:58], [0:58], [52:56]}, metadata={op_type="Split" op_name="split_1"}
  %slice.230 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.67), slice={[0:3], [0:3], [0:4], [52:56]}, metadata={op_type="Split" op_name="split"}
  %convolution.254 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.198, f32[3,3,4,4]{3,2,1,0} %slice.230), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_13"}
  %slice.199 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.184), slice={[0:1], [0:58], [0:58], [56:60]}, metadata={op_type="Split" op_name="split_1"}
  %slice.231 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.67), slice={[0:3], [0:3], [0:4], [56:60]}, metadata={op_type="Split" op_name="split"}
  %convolution.255 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.199, f32[3,3,4,4]{3,2,1,0} %slice.231), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_14"}
  %slice.200 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.184), slice={[0:1], [0:58], [0:58], [60:64]}, metadata={op_type="Split" op_name="split_1"}
  %slice.232 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.67), slice={[0:3], [0:3], [0:4], [60:64]}, metadata={op_type="Split" op_name="split"}
  %convolution.256 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.200, f32[3,3,4,4]{3,2,1,0} %slice.232), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_15"}
  %slice.201 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.184), slice={[0:1], [0:58], [0:58], [64:68]}, metadata={op_type="Split" op_name="split_1"}
  %slice.233 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.67), slice={[0:3], [0:3], [0:4], [64:68]}, metadata={op_type="Split" op_name="split"}
  %convolution.257 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.201, f32[3,3,4,4]{3,2,1,0} %slice.233), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_16"}
  %slice.202 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.184), slice={[0:1], [0:58], [0:58], [68:72]}, metadata={op_type="Split" op_name="split_1"}
  %slice.234 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.67), slice={[0:3], [0:3], [0:4], [68:72]}, metadata={op_type="Split" op_name="split"}
  %convolution.258 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.202, f32[3,3,4,4]{3,2,1,0} %slice.234), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_17"}
  %slice.203 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.184), slice={[0:1], [0:58], [0:58], [72:76]}, metadata={op_type="Split" op_name="split_1"}
  %slice.235 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.67), slice={[0:3], [0:3], [0:4], [72:76]}, metadata={op_type="Split" op_name="split"}
  %convolution.259 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.203, f32[3,3,4,4]{3,2,1,0} %slice.235), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_18"}
  %slice.204 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.184), slice={[0:1], [0:58], [0:58], [76:80]}, metadata={op_type="Split" op_name="split_1"}
  %slice.236 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.67), slice={[0:3], [0:3], [0:4], [76:80]}, metadata={op_type="Split" op_name="split"}
  %convolution.260 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.204, f32[3,3,4,4]{3,2,1,0} %slice.236), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_19"}
  %slice.205 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.184), slice={[0:1], [0:58], [0:58], [80:84]}, metadata={op_type="Split" op_name="split_1"}
  %slice.237 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.67), slice={[0:3], [0:3], [0:4], [80:84]}, metadata={op_type="Split" op_name="split"}
  %convolution.262 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.205, f32[3,3,4,4]{3,2,1,0} %slice.237), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_20"}
  %slice.206 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.184), slice={[0:1], [0:58], [0:58], [84:88]}, metadata={op_type="Split" op_name="split_1"}
  %slice.238 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.67), slice={[0:3], [0:3], [0:4], [84:88]}, metadata={op_type="Split" op_name="split"}
  %convolution.263 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.206, f32[3,3,4,4]{3,2,1,0} %slice.238), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_21"}
  %slice.207 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.184), slice={[0:1], [0:58], [0:58], [88:92]}, metadata={op_type="Split" op_name="split_1"}
  %slice.239 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.67), slice={[0:3], [0:3], [0:4], [88:92]}, metadata={op_type="Split" op_name="split"}
  %convolution.264 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.207, f32[3,3,4,4]{3,2,1,0} %slice.239), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_22"}
  %slice.208 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.184), slice={[0:1], [0:58], [0:58], [92:96]}, metadata={op_type="Split" op_name="split_1"}
  %slice.240 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.67), slice={[0:3], [0:3], [0:4], [92:96]}, metadata={op_type="Split" op_name="split"}
  %convolution.265 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.208, f32[3,3,4,4]{3,2,1,0} %slice.240), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_23"}
  %slice.209 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.184), slice={[0:1], [0:58], [0:58], [96:100]}, metadata={op_type="Split" op_name="split_1"}
  %slice.241 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.67), slice={[0:3], [0:3], [0:4], [96:100]}, metadata={op_type="Split" op_name="split"}
  %convolution.266 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.209, f32[3,3,4,4]{3,2,1,0} %slice.241), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_24"}
  %slice.210 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.184), slice={[0:1], [0:58], [0:58], [100:104]}, metadata={op_type="Split" op_name="split_1"}
  %slice.242 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.67), slice={[0:3], [0:3], [0:4], [100:104]}, metadata={op_type="Split" op_name="split"}
  %convolution.267 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.210, f32[3,3,4,4]{3,2,1,0} %slice.242), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_25"}
  %slice.211 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.184), slice={[0:1], [0:58], [0:58], [104:108]}, metadata={op_type="Split" op_name="split_1"}
  %slice.243 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.67), slice={[0:3], [0:3], [0:4], [104:108]}, metadata={op_type="Split" op_name="split"}
  %convolution.268 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.211, f32[3,3,4,4]{3,2,1,0} %slice.243), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_26"}
  %slice.212 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.184), slice={[0:1], [0:58], [0:58], [108:112]}, metadata={op_type="Split" op_name="split_1"}
  %slice.244 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.67), slice={[0:3], [0:3], [0:4], [108:112]}, metadata={op_type="Split" op_name="split"}
  %convolution.269 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.212, f32[3,3,4,4]{3,2,1,0} %slice.244), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_27"}
  %slice.213 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.184), slice={[0:1], [0:58], [0:58], [112:116]}, metadata={op_type="Split" op_name="split_1"}
  %slice.245 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.67), slice={[0:3], [0:3], [0:4], [112:116]}, metadata={op_type="Split" op_name="split"}
  %convolution.270 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.213, f32[3,3,4,4]{3,2,1,0} %slice.245), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_28"}
  %slice.214 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.184), slice={[0:1], [0:58], [0:58], [116:120]}, metadata={op_type="Split" op_name="split_1"}
  %slice.246 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.67), slice={[0:3], [0:3], [0:4], [116:120]}, metadata={op_type="Split" op_name="split"}
  %convolution.271 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.214, f32[3,3,4,4]{3,2,1,0} %slice.246), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_29"}
  %slice.215 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.184), slice={[0:1], [0:58], [0:58], [120:124]}, metadata={op_type="Split" op_name="split_1"}
  %slice.247 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.67), slice={[0:3], [0:3], [0:4], [120:124]}, metadata={op_type="Split" op_name="split"}
  %convolution.273 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.215, f32[3,3,4,4]{3,2,1,0} %slice.247), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_30"}
  %slice.216 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.184), slice={[0:1], [0:58], [0:58], [124:128]}, metadata={op_type="Split" op_name="split_1"}
  %slice.248 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.67), slice={[0:3], [0:3], [0:4], [124:128]}, metadata={op_type="Split" op_name="split"}
  %convolution.274 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.216, f32[3,3,4,4]{3,2,1,0} %slice.248), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_31"}
  %concatenate.281 = f32[1,56,56,128]{3,2,1,0} concatenate(f32[1,56,56,4]{3,2,1,0} %convolution.249, f32[1,56,56,4]{3,2,1,0} %convolution.250, f32[1,56,56,4]{3,2,1,0} %convolution.261, f32[1,56,56,4]{3,2,1,0} %convolution.272, f32[1,56,56,4]{3,2,1,0} %convolution.275, f32[1,56,56,4]{3,2,1,0} %convolution.276, f32[1,56,56,4]{3,2,1,0} %convolution.277, f32[1,56,56,4]{3,2,1,0} %convolution.278, f32[1,56,56,4]{3,2,1,0} %convolution.279, f32[1,56,56,4]{3,2,1,0} %convolution.280, f32[1,56,56,4]{3,2,1,0} %convolution.251, f32[1,56,56,4]{3,2,1,0} %convolution.252, f32[1,56,56,4]{3,2,1,0} %convolution.253, f32[1,56,56,4]{3,2,1,0} %convolution.254, f32[1,56,56,4]{3,2,1,0} %convolution.255, f32[1,56,56,4]{3,2,1,0} %convolution.256, f32[1,56,56,4]{3,2,1,0} %convolution.257, f32[1,56,56,4]{3,2,1,0} %convolution.258, f32[1,56,56,4]{3,2,1,0} %convolution.259, f32[1,56,56,4]{3,2,1,0} %convolution.260, f32[1,56,56,4]{3,2,1,0} %convolution.262, f32[1,56,56,4]{3,2,1,0} %convolution.263, f32[1,56,56,4]{3,2,1,0} %convolution.264, f32[1,56,56,4]{3,2,1,0} %convolution.265, f32[1,56,56,4]{3,2,1,0} %convolution.266, f32[1,56,56,4]{3,2,1,0} %convolution.267, f32[1,56,56,4]{3,2,1,0} %convolution.268, f32[1,56,56,4]{3,2,1,0} %convolution.269, f32[1,56,56,4]{3,2,1,0} %convolution.270, f32[1,56,56,4]{3,2,1,0} %convolution.271, f32[1,56,56,4]{3,2,1,0} %convolution.273, f32[1,56,56,4]{3,2,1,0} %convolution.274), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat"}
  %constant.161 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit1_bn2/add"}
  %broadcast.162 = f32[128]{0} broadcast(f32[] %constant.161), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit1_bn2/add"}
  %arg8.9 = f32[128]{0} parameter(8), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.68 = f32[128]{0} reshape(f32[128]{0} %arg8.9)
  %add.163 = f32[128]{0} add(f32[128]{0} %broadcast.162, f32[128]{0} %reshape.68), metadata={op_type="AddV2" op_name="stage1_unit1_bn2/add"}
  %rsqrt.164 = f32[128]{0} rsqrt(f32[128]{0} %add.163), metadata={op_type="Rsqrt" op_name="stage1_unit1_bn2/Rsqrt"}
  %arg21.22 = f32[128]{0} parameter(21), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.81 = f32[128]{0} reshape(f32[128]{0} %arg21.22)
  %multiply.165 = f32[128]{0} multiply(f32[128]{0} %rsqrt.164, f32[128]{0} %reshape.81), metadata={op_type="Mul" op_name="stage1_unit1_bn2/mul"}
  %broadcast.282 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %multiply.165), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit1_bn2/mul_1"}
  %multiply.283 = f32[1,56,56,128]{3,2,1,0} multiply(f32[1,56,56,128]{3,2,1,0} %concatenate.281, f32[1,56,56,128]{3,2,1,0} %broadcast.282), metadata={op_type="Mul" op_name="stage1_unit1_bn2/mul_1"}
  %arg44.45 = f32[128]{0} parameter(44), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.104 = f32[128]{0} reshape(f32[128]{0} %arg44.45)
  %arg33.34 = f32[128]{0} parameter(33), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.93 = f32[128]{0} reshape(f32[128]{0} %arg33.34)
  %multiply.166 = f32[128]{0} multiply(f32[128]{0} %multiply.165, f32[128]{0} %reshape.93), metadata={op_type="Mul" op_name="stage1_unit1_bn2/mul_2"}
  %subtract.167 = f32[128]{0} subtract(f32[128]{0} %reshape.104, f32[128]{0} %multiply.166), metadata={op_type="Sub" op_name="stage1_unit1_bn2/sub"}
  %broadcast.284 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %subtract.167), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit1_bn2/add_1"}
  %add.285 = f32[1,56,56,128]{3,2,1,0} add(f32[1,56,56,128]{3,2,1,0} %multiply.283, f32[1,56,56,128]{3,2,1,0} %broadcast.284), metadata={op_type="AddV2" op_name="stage1_unit1_bn2/add_1"}
  %maximum.288 = f32[1,56,56,128]{3,2,1,0} maximum(f32[1,56,56,128]{3,2,1,0} %broadcast.287, f32[1,56,56,128]{3,2,1,0} %add.285), metadata={op_type="Relu" op_name="stage1_unit1_relu2"}
  %arg53.54 = f32[1,1,128,256]{3,2,1,0} parameter(53), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.113 = f32[1,1,128,256]{3,2,1,0} reshape(f32[1,1,128,256]{3,2,1,0} %arg53.54)
  %convolution.289 = f32[1,56,56,256]{3,2,1,0} convolution(f32[1,56,56,128]{3,2,1,0} %maximum.288, f32[1,1,128,256]{3,2,1,0} %reshape.113), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv3"}
  %multiply.291 = f32[1,56,56,256]{3,2,1,0} multiply(f32[1,56,56,256]{3,2,1,0} %broadcast.290, f32[1,56,56,256]{3,2,1,0} %convolution.289), metadata={op_type="Mul" op_name="stage1_unit1_bn3/mul_1"}
  %arg45.46 = f32[256]{0} parameter(45), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.105 = f32[256]{0} reshape(f32[256]{0} %arg45.46)
  %arg34.35 = f32[256]{0} parameter(34), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.94 = f32[256]{0} reshape(f32[256]{0} %arg34.35)
  %multiply.173 = f32[256]{0} multiply(f32[256]{0} %multiply.172, f32[256]{0} %reshape.94), metadata={op_type="Mul" op_name="stage1_unit1_bn3/mul_2"}
  %subtract.174 = f32[256]{0} subtract(f32[256]{0} %reshape.105, f32[256]{0} %multiply.173), metadata={op_type="Sub" op_name="stage1_unit1_bn3/sub"}
  %broadcast.292 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.174), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit1_bn3/add_1"}
  %add.293 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %multiply.291, f32[1,56,56,256]{3,2,1,0} %broadcast.292), metadata={op_type="AddV2" op_name="stage1_unit1_bn3/add_1"}
  %arg52.53 = f32[1,1,64,256]{3,2,1,0} parameter(52), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.112 = f32[1,1,64,256]{3,2,1,0} reshape(f32[1,1,64,256]{3,2,1,0} %arg52.53)
  %convolution.301 = f32[1,56,56,256]{3,2,1,0} convolution(f32[1,56,56,64]{3,2,1,0} %reduce-window.153, f32[1,1,64,256]{3,2,1,0} %reshape.112), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_sc"}
  %constant.294 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit1_sc_bn/add"}
  %broadcast.295 = f32[256]{0} broadcast(f32[] %constant.294), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit1_sc_bn/add"}
  %arg6.7 = f32[256]{0} parameter(6), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.66 = f32[256]{0} reshape(f32[256]{0} %arg6.7)
  %add.296 = f32[256]{0} add(f32[256]{0} %broadcast.295, f32[256]{0} %reshape.66), metadata={op_type="AddV2" op_name="stage1_unit1_sc_bn/add"}
  %rsqrt.297 = f32[256]{0} rsqrt(f32[256]{0} %add.296), metadata={op_type="Rsqrt" op_name="stage1_unit1_sc_bn/Rsqrt"}
  %arg20.21 = f32[256]{0} parameter(20), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.80 = f32[256]{0} reshape(f32[256]{0} %arg20.21)
  %multiply.298 = f32[256]{0} multiply(f32[256]{0} %rsqrt.297, f32[256]{0} %reshape.80), metadata={op_type="Mul" op_name="stage1_unit1_sc_bn/mul"}
  %broadcast.302 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.298), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit1_sc_bn/mul_1"}
  %multiply.303 = f32[1,56,56,256]{3,2,1,0} multiply(f32[1,56,56,256]{3,2,1,0} %convolution.301, f32[1,56,56,256]{3,2,1,0} %broadcast.302), metadata={op_type="Mul" op_name="stage1_unit1_sc_bn/mul_1"}
  %arg43.44 = f32[256]{0} parameter(43), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.103 = f32[256]{0} reshape(f32[256]{0} %arg43.44)
  %arg32.33 = f32[256]{0} parameter(32), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.92 = f32[256]{0} reshape(f32[256]{0} %arg32.33)
  %multiply.299 = f32[256]{0} multiply(f32[256]{0} %multiply.298, f32[256]{0} %reshape.92), metadata={op_type="Mul" op_name="stage1_unit1_sc_bn/mul_2"}
  %subtract.300 = f32[256]{0} subtract(f32[256]{0} %reshape.103, f32[256]{0} %multiply.299), metadata={op_type="Sub" op_name="stage1_unit1_sc_bn/sub"}
  %broadcast.304 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.300), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit1_sc_bn/add_1"}
  %add.305 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %multiply.303, f32[1,56,56,256]{3,2,1,0} %broadcast.304), metadata={op_type="AddV2" op_name="stage1_unit1_sc_bn/add_1"}
  %add.306 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %add.293, f32[1,56,56,256]{3,2,1,0} %add.305), metadata={op_type="AddV2" op_name="add"}
  %maximum.309 = f32[1,56,56,256]{3,2,1,0} maximum(f32[1,56,56,256]{3,2,1,0} %broadcast.308, f32[1,56,56,256]{3,2,1,0} %add.306), metadata={op_type="Relu" op_name="stage1_unit1_relu"}
  %constant.324 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit2_bn3/add"}
  %broadcast.325 = f32[256]{0} broadcast(f32[] %constant.324), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit2_bn3/add"}
  %arg14.15 = f32[256]{0} parameter(14), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.74 = f32[256]{0} reshape(f32[256]{0} %arg14.15)
  %add.326 = f32[256]{0} add(f32[256]{0} %broadcast.325, f32[256]{0} %reshape.74), metadata={op_type="AddV2" op_name="stage1_unit2_bn3/add"}
  %rsqrt.327 = f32[256]{0} rsqrt(f32[256]{0} %add.326), metadata={op_type="Rsqrt" op_name="stage1_unit2_bn3/Rsqrt"}
  %arg26.27 = f32[256]{0} parameter(26), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.86 = f32[256]{0} reshape(f32[256]{0} %arg26.27)
  %multiply.328 = f32[256]{0} multiply(f32[256]{0} %rsqrt.327, f32[256]{0} %reshape.86), metadata={op_type="Mul" op_name="stage1_unit2_bn3/mul"}
  %broadcast.446 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.328), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit2_bn3/mul_1"}
  %constant.442 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit2_relu2"}
  %broadcast.443 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[] %constant.442), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit2_relu2"}
  %constant.336 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit2_relu1"}
  %broadcast.337 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[] %constant.336), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit2_relu1"}
  %constant.310 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit2_bn1/add"}
  %broadcast.311 = f32[128]{0} broadcast(f32[] %constant.310), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit2_bn1/add"}
  %arg10.11 = f32[128]{0} parameter(10), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.70 = f32[128]{0} reshape(f32[128]{0} %arg10.11)
  %add.312 = f32[128]{0} add(f32[128]{0} %broadcast.311, f32[128]{0} %reshape.70), metadata={op_type="AddV2" op_name="stage1_unit2_bn1/add"}
  %rsqrt.313 = f32[128]{0} rsqrt(f32[128]{0} %add.312), metadata={op_type="Rsqrt" op_name="stage1_unit2_bn1/Rsqrt"}
  %arg23.24 = f32[128]{0} parameter(23), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.83 = f32[128]{0} reshape(f32[128]{0} %arg23.24)
  %multiply.314 = f32[128]{0} multiply(f32[128]{0} %rsqrt.313, f32[128]{0} %reshape.83), metadata={op_type="Mul" op_name="stage1_unit2_bn1/mul"}
  %broadcast.332 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %multiply.314), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit2_bn1/mul_1"}
  %arg54.55 = f32[1,1,256,128]{3,2,1,0} parameter(54), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.114 = f32[1,1,256,128]{3,2,1,0} reshape(f32[1,1,256,128]{3,2,1,0} %arg54.55)
  %convolution.331 = f32[1,56,56,128]{3,2,1,0} convolution(f32[1,56,56,256]{3,2,1,0} %maximum.309, f32[1,1,256,128]{3,2,1,0} %reshape.114), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv1"}
  %multiply.333 = f32[1,56,56,128]{3,2,1,0} multiply(f32[1,56,56,128]{3,2,1,0} %broadcast.332, f32[1,56,56,128]{3,2,1,0} %convolution.331), metadata={op_type="Mul" op_name="stage1_unit2_bn1/mul_1"}
  %arg46.47 = f32[128]{0} parameter(46), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.106 = f32[128]{0} reshape(f32[128]{0} %arg46.47)
  %arg35.36 = f32[128]{0} parameter(35), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.95 = f32[128]{0} reshape(f32[128]{0} %arg35.36)
  %multiply.315 = f32[128]{0} multiply(f32[128]{0} %multiply.314, f32[128]{0} %reshape.95), metadata={op_type="Mul" op_name="stage1_unit2_bn1/mul_2"}
  %subtract.316 = f32[128]{0} subtract(f32[128]{0} %reshape.106, f32[128]{0} %multiply.315), metadata={op_type="Sub" op_name="stage1_unit2_bn1/sub"}
  %broadcast.334 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %subtract.316), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit2_bn1/add_1"}
  %add.335 = f32[1,56,56,128]{3,2,1,0} add(f32[1,56,56,128]{3,2,1,0} %multiply.333, f32[1,56,56,128]{3,2,1,0} %broadcast.334), metadata={op_type="AddV2" op_name="stage1_unit2_bn1/add_1"}
  %maximum.338 = f32[1,56,56,128]{3,2,1,0} maximum(f32[1,56,56,128]{3,2,1,0} %broadcast.337, f32[1,56,56,128]{3,2,1,0} %add.335), metadata={op_type="Relu" op_name="stage1_unit2_relu1"}
  %constant.339 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_2"}
  %pad.340 = f32[1,58,58,128]{3,2,1,0} pad(f32[1,56,56,128]{3,2,1,0} %maximum.338, f32[] %constant.339), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_2"}
  %slice.341 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.340), slice={[0:1], [0:58], [0:58], [0:4]}, metadata={op_type="Split" op_name="split_3"}
  %arg11.12 = f32[3,3,4,128]{3,2,1,0} parameter(11), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.71 = f32[3,3,4,128]{3,2,1,0} reshape(f32[3,3,4,128]{3,2,1,0} %arg11.12)
  %slice.373 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.71), slice={[0:3], [0:3], [0:4], [0:4]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.405 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.341, f32[3,3,4,4]{3,2,1,0} %slice.373), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2"}
  %slice.342 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.340), slice={[0:1], [0:58], [0:58], [4:8]}, metadata={op_type="Split" op_name="split_3"}
  %slice.374 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.71), slice={[0:3], [0:3], [0:4], [4:8]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.406 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.342, f32[3,3,4,4]{3,2,1,0} %slice.374), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_1"}
  %slice.343 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.340), slice={[0:1], [0:58], [0:58], [8:12]}, metadata={op_type="Split" op_name="split_3"}
  %slice.375 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.71), slice={[0:3], [0:3], [0:4], [8:12]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.417 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.343, f32[3,3,4,4]{3,2,1,0} %slice.375), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_2"}
  %slice.344 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.340), slice={[0:1], [0:58], [0:58], [12:16]}, metadata={op_type="Split" op_name="split_3"}
  %slice.376 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.71), slice={[0:3], [0:3], [0:4], [12:16]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.428 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.344, f32[3,3,4,4]{3,2,1,0} %slice.376), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_3"}
  %slice.345 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.340), slice={[0:1], [0:58], [0:58], [16:20]}, metadata={op_type="Split" op_name="split_3"}
  %slice.377 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.71), slice={[0:3], [0:3], [0:4], [16:20]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.431 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.345, f32[3,3,4,4]{3,2,1,0} %slice.377), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_4"}
  %slice.346 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.340), slice={[0:1], [0:58], [0:58], [20:24]}, metadata={op_type="Split" op_name="split_3"}
  %slice.378 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.71), slice={[0:3], [0:3], [0:4], [20:24]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.432 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.346, f32[3,3,4,4]{3,2,1,0} %slice.378), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_5"}
  %slice.347 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.340), slice={[0:1], [0:58], [0:58], [24:28]}, metadata={op_type="Split" op_name="split_3"}
  %slice.379 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.71), slice={[0:3], [0:3], [0:4], [24:28]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.433 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.347, f32[3,3,4,4]{3,2,1,0} %slice.379), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_6"}
  %slice.348 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.340), slice={[0:1], [0:58], [0:58], [28:32]}, metadata={op_type="Split" op_name="split_3"}
  %slice.380 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.71), slice={[0:3], [0:3], [0:4], [28:32]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.434 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.348, f32[3,3,4,4]{3,2,1,0} %slice.380), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_7"}
  %slice.349 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.340), slice={[0:1], [0:58], [0:58], [32:36]}, metadata={op_type="Split" op_name="split_3"}
  %slice.381 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.71), slice={[0:3], [0:3], [0:4], [32:36]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.435 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.349, f32[3,3,4,4]{3,2,1,0} %slice.381), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_8"}
  %slice.350 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.340), slice={[0:1], [0:58], [0:58], [36:40]}, metadata={op_type="Split" op_name="split_3"}
  %slice.382 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.71), slice={[0:3], [0:3], [0:4], [36:40]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.436 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.350, f32[3,3,4,4]{3,2,1,0} %slice.382), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_9"}
  %slice.351 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.340), slice={[0:1], [0:58], [0:58], [40:44]}, metadata={op_type="Split" op_name="split_3"}
  %slice.383 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.71), slice={[0:3], [0:3], [0:4], [40:44]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.407 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.351, f32[3,3,4,4]{3,2,1,0} %slice.383), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_10"}
  %slice.352 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.340), slice={[0:1], [0:58], [0:58], [44:48]}, metadata={op_type="Split" op_name="split_3"}
  %slice.384 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.71), slice={[0:3], [0:3], [0:4], [44:48]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.408 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.352, f32[3,3,4,4]{3,2,1,0} %slice.384), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_11"}
  %slice.353 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.340), slice={[0:1], [0:58], [0:58], [48:52]}, metadata={op_type="Split" op_name="split_3"}
  %slice.385 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.71), slice={[0:3], [0:3], [0:4], [48:52]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.409 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.353, f32[3,3,4,4]{3,2,1,0} %slice.385), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_12"}
  %slice.354 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.340), slice={[0:1], [0:58], [0:58], [52:56]}, metadata={op_type="Split" op_name="split_3"}
  %slice.386 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.71), slice={[0:3], [0:3], [0:4], [52:56]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.410 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.354, f32[3,3,4,4]{3,2,1,0} %slice.386), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_13"}
  %slice.355 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.340), slice={[0:1], [0:58], [0:58], [56:60]}, metadata={op_type="Split" op_name="split_3"}
  %slice.387 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.71), slice={[0:3], [0:3], [0:4], [56:60]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.411 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.355, f32[3,3,4,4]{3,2,1,0} %slice.387), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_14"}
  %slice.356 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.340), slice={[0:1], [0:58], [0:58], [60:64]}, metadata={op_type="Split" op_name="split_3"}
  %slice.388 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.71), slice={[0:3], [0:3], [0:4], [60:64]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.412 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.356, f32[3,3,4,4]{3,2,1,0} %slice.388), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_15"}
  %slice.357 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.340), slice={[0:1], [0:58], [0:58], [64:68]}, metadata={op_type="Split" op_name="split_3"}
  %slice.389 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.71), slice={[0:3], [0:3], [0:4], [64:68]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.413 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.357, f32[3,3,4,4]{3,2,1,0} %slice.389), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_16"}
  %slice.358 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.340), slice={[0:1], [0:58], [0:58], [68:72]}, metadata={op_type="Split" op_name="split_3"}
  %slice.390 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.71), slice={[0:3], [0:3], [0:4], [68:72]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.414 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.358, f32[3,3,4,4]{3,2,1,0} %slice.390), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_17"}
  %slice.359 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.340), slice={[0:1], [0:58], [0:58], [72:76]}, metadata={op_type="Split" op_name="split_3"}
  %slice.391 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.71), slice={[0:3], [0:3], [0:4], [72:76]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.415 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.359, f32[3,3,4,4]{3,2,1,0} %slice.391), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_18"}
  %slice.360 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.340), slice={[0:1], [0:58], [0:58], [76:80]}, metadata={op_type="Split" op_name="split_3"}
  %slice.392 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.71), slice={[0:3], [0:3], [0:4], [76:80]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.416 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.360, f32[3,3,4,4]{3,2,1,0} %slice.392), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_19"}
  %slice.361 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.340), slice={[0:1], [0:58], [0:58], [80:84]}, metadata={op_type="Split" op_name="split_3"}
  %slice.393 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.71), slice={[0:3], [0:3], [0:4], [80:84]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.418 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.361, f32[3,3,4,4]{3,2,1,0} %slice.393), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_20"}
  %slice.362 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.340), slice={[0:1], [0:58], [0:58], [84:88]}, metadata={op_type="Split" op_name="split_3"}
  %slice.394 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.71), slice={[0:3], [0:3], [0:4], [84:88]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.419 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.362, f32[3,3,4,4]{3,2,1,0} %slice.394), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_21"}
  %slice.363 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.340), slice={[0:1], [0:58], [0:58], [88:92]}, metadata={op_type="Split" op_name="split_3"}
  %slice.395 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.71), slice={[0:3], [0:3], [0:4], [88:92]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.420 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.363, f32[3,3,4,4]{3,2,1,0} %slice.395), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_22"}
  %slice.364 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.340), slice={[0:1], [0:58], [0:58], [92:96]}, metadata={op_type="Split" op_name="split_3"}
  %slice.396 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.71), slice={[0:3], [0:3], [0:4], [92:96]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.421 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.364, f32[3,3,4,4]{3,2,1,0} %slice.396), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_23"}
  %slice.365 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.340), slice={[0:1], [0:58], [0:58], [96:100]}, metadata={op_type="Split" op_name="split_3"}
  %slice.397 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.71), slice={[0:3], [0:3], [0:4], [96:100]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.422 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.365, f32[3,3,4,4]{3,2,1,0} %slice.397), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_24"}
  %slice.366 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.340), slice={[0:1], [0:58], [0:58], [100:104]}, metadata={op_type="Split" op_name="split_3"}
  %slice.398 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.71), slice={[0:3], [0:3], [0:4], [100:104]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.423 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.366, f32[3,3,4,4]{3,2,1,0} %slice.398), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_25"}
  %slice.367 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.340), slice={[0:1], [0:58], [0:58], [104:108]}, metadata={op_type="Split" op_name="split_3"}
  %slice.399 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.71), slice={[0:3], [0:3], [0:4], [104:108]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.424 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.367, f32[3,3,4,4]{3,2,1,0} %slice.399), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_26"}
  %slice.368 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.340), slice={[0:1], [0:58], [0:58], [108:112]}, metadata={op_type="Split" op_name="split_3"}
  %slice.400 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.71), slice={[0:3], [0:3], [0:4], [108:112]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.425 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.368, f32[3,3,4,4]{3,2,1,0} %slice.400), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_27"}
  %slice.369 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.340), slice={[0:1], [0:58], [0:58], [112:116]}, metadata={op_type="Split" op_name="split_3"}
  %slice.401 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.71), slice={[0:3], [0:3], [0:4], [112:116]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.426 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.369, f32[3,3,4,4]{3,2,1,0} %slice.401), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_28"}
  %slice.370 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.340), slice={[0:1], [0:58], [0:58], [116:120]}, metadata={op_type="Split" op_name="split_3"}
  %slice.402 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.71), slice={[0:3], [0:3], [0:4], [116:120]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.427 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.370, f32[3,3,4,4]{3,2,1,0} %slice.402), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_29"}
  %slice.371 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.340), slice={[0:1], [0:58], [0:58], [120:124]}, metadata={op_type="Split" op_name="split_3"}
  %slice.403 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.71), slice={[0:3], [0:3], [0:4], [120:124]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.429 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.371, f32[3,3,4,4]{3,2,1,0} %slice.403), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_30"}
  %slice.372 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.340), slice={[0:1], [0:58], [0:58], [124:128]}, metadata={op_type="Split" op_name="split_3"}
  %slice.404 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.71), slice={[0:3], [0:3], [0:4], [124:128]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.430 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.372, f32[3,3,4,4]{3,2,1,0} %slice.404), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_31"}
  %concatenate.437 = f32[1,56,56,128]{3,2,1,0} concatenate(f32[1,56,56,4]{3,2,1,0} %convolution.405, f32[1,56,56,4]{3,2,1,0} %convolution.406, f32[1,56,56,4]{3,2,1,0} %convolution.417, f32[1,56,56,4]{3,2,1,0} %convolution.428, f32[1,56,56,4]{3,2,1,0} %convolution.431, f32[1,56,56,4]{3,2,1,0} %convolution.432, f32[1,56,56,4]{3,2,1,0} %convolution.433, f32[1,56,56,4]{3,2,1,0} %convolution.434, f32[1,56,56,4]{3,2,1,0} %convolution.435, f32[1,56,56,4]{3,2,1,0} %convolution.436, f32[1,56,56,4]{3,2,1,0} %convolution.407, f32[1,56,56,4]{3,2,1,0} %convolution.408, f32[1,56,56,4]{3,2,1,0} %convolution.409, f32[1,56,56,4]{3,2,1,0} %convolution.410, f32[1,56,56,4]{3,2,1,0} %convolution.411, f32[1,56,56,4]{3,2,1,0} %convolution.412, f32[1,56,56,4]{3,2,1,0} %convolution.413, f32[1,56,56,4]{3,2,1,0} %convolution.414, f32[1,56,56,4]{3,2,1,0} %convolution.415, f32[1,56,56,4]{3,2,1,0} %convolution.416, f32[1,56,56,4]{3,2,1,0} %convolution.418, f32[1,56,56,4]{3,2,1,0} %convolution.419, f32[1,56,56,4]{3,2,1,0} %convolution.420, f32[1,56,56,4]{3,2,1,0} %convolution.421, f32[1,56,56,4]{3,2,1,0} %convolution.422, f32[1,56,56,4]{3,2,1,0} %convolution.423, f32[1,56,56,4]{3,2,1,0} %convolution.424, f32[1,56,56,4]{3,2,1,0} %convolution.425, f32[1,56,56,4]{3,2,1,0} %convolution.426, f32[1,56,56,4]{3,2,1,0} %convolution.427, f32[1,56,56,4]{3,2,1,0} %convolution.429, f32[1,56,56,4]{3,2,1,0} %convolution.430), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_1"}
  %constant.317 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit2_bn2/add"}
  %broadcast.318 = f32[128]{0} broadcast(f32[] %constant.317), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit2_bn2/add"}
  %arg13.14 = f32[128]{0} parameter(13), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.73 = f32[128]{0} reshape(f32[128]{0} %arg13.14)
  %add.319 = f32[128]{0} add(f32[128]{0} %broadcast.318, f32[128]{0} %reshape.73), metadata={op_type="AddV2" op_name="stage1_unit2_bn2/add"}
  %rsqrt.320 = f32[128]{0} rsqrt(f32[128]{0} %add.319), metadata={op_type="Rsqrt" op_name="stage1_unit2_bn2/Rsqrt"}
  %arg25.26 = f32[128]{0} parameter(25), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.85 = f32[128]{0} reshape(f32[128]{0} %arg25.26)
  %multiply.321 = f32[128]{0} multiply(f32[128]{0} %rsqrt.320, f32[128]{0} %reshape.85), metadata={op_type="Mul" op_name="stage1_unit2_bn2/mul"}
  %broadcast.438 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %multiply.321), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit2_bn2/mul_1"}
  %multiply.439 = f32[1,56,56,128]{3,2,1,0} multiply(f32[1,56,56,128]{3,2,1,0} %concatenate.437, f32[1,56,56,128]{3,2,1,0} %broadcast.438), metadata={op_type="Mul" op_name="stage1_unit2_bn2/mul_1"}
  %arg48.49 = f32[128]{0} parameter(48), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.108 = f32[128]{0} reshape(f32[128]{0} %arg48.49)
  %arg37.38 = f32[128]{0} parameter(37), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.97 = f32[128]{0} reshape(f32[128]{0} %arg37.38)
  %multiply.322 = f32[128]{0} multiply(f32[128]{0} %multiply.321, f32[128]{0} %reshape.97), metadata={op_type="Mul" op_name="stage1_unit2_bn2/mul_2"}
  %subtract.323 = f32[128]{0} subtract(f32[128]{0} %reshape.108, f32[128]{0} %multiply.322), metadata={op_type="Sub" op_name="stage1_unit2_bn2/sub"}
  %broadcast.440 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %subtract.323), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit2_bn2/add_1"}
  %add.441 = f32[1,56,56,128]{3,2,1,0} add(f32[1,56,56,128]{3,2,1,0} %multiply.439, f32[1,56,56,128]{3,2,1,0} %broadcast.440), metadata={op_type="AddV2" op_name="stage1_unit2_bn2/add_1"}
  %maximum.444 = f32[1,56,56,128]{3,2,1,0} maximum(f32[1,56,56,128]{3,2,1,0} %broadcast.443, f32[1,56,56,128]{3,2,1,0} %add.441), metadata={op_type="Relu" op_name="stage1_unit2_relu2"}
  %arg55.56 = f32[1,1,128,256]{3,2,1,0} parameter(55), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.115 = f32[1,1,128,256]{3,2,1,0} reshape(f32[1,1,128,256]{3,2,1,0} %arg55.56)
  %convolution.445 = f32[1,56,56,256]{3,2,1,0} convolution(f32[1,56,56,128]{3,2,1,0} %maximum.444, f32[1,1,128,256]{3,2,1,0} %reshape.115), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv3"}
  %multiply.447 = f32[1,56,56,256]{3,2,1,0} multiply(f32[1,56,56,256]{3,2,1,0} %broadcast.446, f32[1,56,56,256]{3,2,1,0} %convolution.445), metadata={op_type="Mul" op_name="stage1_unit2_bn3/mul_1"}
  %arg49.50 = f32[256]{0} parameter(49), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.109 = f32[256]{0} reshape(f32[256]{0} %arg49.50)
  %arg38.39 = f32[256]{0} parameter(38), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.98 = f32[256]{0} reshape(f32[256]{0} %arg38.39)
  %multiply.329 = f32[256]{0} multiply(f32[256]{0} %multiply.328, f32[256]{0} %reshape.98), metadata={op_type="Mul" op_name="stage1_unit2_bn3/mul_2"}
  %subtract.330 = f32[256]{0} subtract(f32[256]{0} %reshape.109, f32[256]{0} %multiply.329), metadata={op_type="Sub" op_name="stage1_unit2_bn3/sub"}
  %broadcast.448 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.330), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit2_bn3/add_1"}
  %add.449 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %multiply.447, f32[1,56,56,256]{3,2,1,0} %broadcast.448), metadata={op_type="AddV2" op_name="stage1_unit2_bn3/add_1"}
  %add.450 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %maximum.309, f32[1,56,56,256]{3,2,1,0} %add.449), metadata={op_type="AddV2" op_name="add_1"}
  %maximum.453 = f32[1,56,56,256]{3,2,1,0} maximum(f32[1,56,56,256]{3,2,1,0} %broadcast.452, f32[1,56,56,256]{3,2,1,0} %add.450), metadata={op_type="Relu" op_name="stage1_unit2_relu"}
  %constant.468 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit3_bn3/add"}
  %broadcast.469 = f32[256]{0} broadcast(f32[] %constant.468), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit3_bn3/add"}
  %arg5.6 = f32[256]{0} parameter(5), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.65 = f32[256]{0} reshape(f32[256]{0} %arg5.6)
  %add.470 = f32[256]{0} add(f32[256]{0} %broadcast.469, f32[256]{0} %reshape.65), metadata={op_type="AddV2" op_name="stage1_unit3_bn3/add"}
  %rsqrt.471 = f32[256]{0} rsqrt(f32[256]{0} %add.470), metadata={op_type="Rsqrt" op_name="stage1_unit3_bn3/Rsqrt"}
  %arg19.20 = f32[256]{0} parameter(19), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.79 = f32[256]{0} reshape(f32[256]{0} %arg19.20)
  %multiply.472 = f32[256]{0} multiply(f32[256]{0} %rsqrt.471, f32[256]{0} %reshape.79), metadata={op_type="Mul" op_name="stage1_unit3_bn3/mul"}
  %broadcast.590 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.472), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit3_bn3/mul_1"}
  %constant.586 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit3_relu2"}
  %broadcast.587 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[] %constant.586), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit3_relu2"}
  %constant.480 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit3_relu1"}
  %broadcast.481 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[] %constant.480), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit3_relu1"}
  %constant.454 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit3_bn1/add"}
  %broadcast.455 = f32[128]{0} broadcast(f32[] %constant.454), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit3_bn1/add"}
  %arg0.1 = f32[128]{0} parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.60 = f32[128]{0} reshape(f32[128]{0} %arg0.1)
  %add.456 = f32[128]{0} add(f32[128]{0} %broadcast.455, f32[128]{0} %reshape.60), metadata={op_type="AddV2" op_name="stage1_unit3_bn1/add"}
  %rsqrt.457 = f32[128]{0} rsqrt(f32[128]{0} %add.456), metadata={op_type="Rsqrt" op_name="stage1_unit3_bn1/Rsqrt"}
  %arg15.16 = f32[128]{0} parameter(15), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.75 = f32[128]{0} reshape(f32[128]{0} %arg15.16)
  %multiply.458 = f32[128]{0} multiply(f32[128]{0} %rsqrt.457, f32[128]{0} %reshape.75), metadata={op_type="Mul" op_name="stage1_unit3_bn1/mul"}
  %broadcast.476 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %multiply.458), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit3_bn1/mul_1"}
  %arg56.57 = f32[1,1,256,128]{3,2,1,0} parameter(56), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.116 = f32[1,1,256,128]{3,2,1,0} reshape(f32[1,1,256,128]{3,2,1,0} %arg56.57)
  %convolution.475 = f32[1,56,56,128]{3,2,1,0} convolution(f32[1,56,56,256]{3,2,1,0} %maximum.453, f32[1,1,256,128]{3,2,1,0} %reshape.116), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv1"}
  %multiply.477 = f32[1,56,56,128]{3,2,1,0} multiply(f32[1,56,56,128]{3,2,1,0} %broadcast.476, f32[1,56,56,128]{3,2,1,0} %convolution.475), metadata={op_type="Mul" op_name="stage1_unit3_bn1/mul_1"}
  %arg39.40 = f32[128]{0} parameter(39), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.99 = f32[128]{0} reshape(f32[128]{0} %arg39.40)
  %arg27.28 = f32[128]{0} parameter(27), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.87 = f32[128]{0} reshape(f32[128]{0} %arg27.28)
  %multiply.459 = f32[128]{0} multiply(f32[128]{0} %multiply.458, f32[128]{0} %reshape.87), metadata={op_type="Mul" op_name="stage1_unit3_bn1/mul_2"}
  %subtract.460 = f32[128]{0} subtract(f32[128]{0} %reshape.99, f32[128]{0} %multiply.459), metadata={op_type="Sub" op_name="stage1_unit3_bn1/sub"}
  %broadcast.478 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %subtract.460), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit3_bn1/add_1"}
  %add.479 = f32[1,56,56,128]{3,2,1,0} add(f32[1,56,56,128]{3,2,1,0} %multiply.477, f32[1,56,56,128]{3,2,1,0} %broadcast.478), metadata={op_type="AddV2" op_name="stage1_unit3_bn1/add_1"}
  %maximum.482 = f32[1,56,56,128]{3,2,1,0} maximum(f32[1,56,56,128]{3,2,1,0} %broadcast.481, f32[1,56,56,128]{3,2,1,0} %add.479), metadata={op_type="Relu" op_name="stage1_unit3_relu1"}
  %constant.483 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_3"}
  %pad.484 = f32[1,58,58,128]{3,2,1,0} pad(f32[1,56,56,128]{3,2,1,0} %maximum.482, f32[] %constant.483), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_3"}
  %slice.485 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.484), slice={[0:1], [0:58], [0:58], [0:4]}, metadata={op_type="Split" op_name="split_5"}
  %arg4.5 = f32[3,3,4,128]{3,2,1,0} parameter(4), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.64 = f32[3,3,4,128]{3,2,1,0} reshape(f32[3,3,4,128]{3,2,1,0} %arg4.5)
  %slice.517 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.64), slice={[0:3], [0:3], [0:4], [0:4]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.549 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.485, f32[3,3,4,4]{3,2,1,0} %slice.517), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2"}
  %slice.486 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.484), slice={[0:1], [0:58], [0:58], [4:8]}, metadata={op_type="Split" op_name="split_5"}
  %slice.518 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.64), slice={[0:3], [0:3], [0:4], [4:8]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.550 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.486, f32[3,3,4,4]{3,2,1,0} %slice.518), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_1"}
  %slice.487 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.484), slice={[0:1], [0:58], [0:58], [8:12]}, metadata={op_type="Split" op_name="split_5"}
  %slice.519 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.64), slice={[0:3], [0:3], [0:4], [8:12]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.561 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.487, f32[3,3,4,4]{3,2,1,0} %slice.519), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_2"}
  %slice.488 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.484), slice={[0:1], [0:58], [0:58], [12:16]}, metadata={op_type="Split" op_name="split_5"}
  %slice.520 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.64), slice={[0:3], [0:3], [0:4], [12:16]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.572 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.488, f32[3,3,4,4]{3,2,1,0} %slice.520), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_3"}
  %slice.489 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.484), slice={[0:1], [0:58], [0:58], [16:20]}, metadata={op_type="Split" op_name="split_5"}
  %slice.521 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.64), slice={[0:3], [0:3], [0:4], [16:20]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.575 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.489, f32[3,3,4,4]{3,2,1,0} %slice.521), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_4"}
  %slice.490 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.484), slice={[0:1], [0:58], [0:58], [20:24]}, metadata={op_type="Split" op_name="split_5"}
  %slice.522 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.64), slice={[0:3], [0:3], [0:4], [20:24]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.576 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.490, f32[3,3,4,4]{3,2,1,0} %slice.522), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_5"}
  %slice.491 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.484), slice={[0:1], [0:58], [0:58], [24:28]}, metadata={op_type="Split" op_name="split_5"}
  %slice.523 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.64), slice={[0:3], [0:3], [0:4], [24:28]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.577 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.491, f32[3,3,4,4]{3,2,1,0} %slice.523), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_6"}
  %slice.492 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.484), slice={[0:1], [0:58], [0:58], [28:32]}, metadata={op_type="Split" op_name="split_5"}
  %slice.524 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.64), slice={[0:3], [0:3], [0:4], [28:32]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.578 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.492, f32[3,3,4,4]{3,2,1,0} %slice.524), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_7"}
  %slice.493 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.484), slice={[0:1], [0:58], [0:58], [32:36]}, metadata={op_type="Split" op_name="split_5"}
  %slice.525 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.64), slice={[0:3], [0:3], [0:4], [32:36]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.579 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.493, f32[3,3,4,4]{3,2,1,0} %slice.525), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_8"}
  %slice.494 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.484), slice={[0:1], [0:58], [0:58], [36:40]}, metadata={op_type="Split" op_name="split_5"}
  %slice.526 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.64), slice={[0:3], [0:3], [0:4], [36:40]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.580 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.494, f32[3,3,4,4]{3,2,1,0} %slice.526), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_9"}
  %slice.495 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.484), slice={[0:1], [0:58], [0:58], [40:44]}, metadata={op_type="Split" op_name="split_5"}
  %slice.527 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.64), slice={[0:3], [0:3], [0:4], [40:44]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.551 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.495, f32[3,3,4,4]{3,2,1,0} %slice.527), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_10"}
  %slice.496 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.484), slice={[0:1], [0:58], [0:58], [44:48]}, metadata={op_type="Split" op_name="split_5"}
  %slice.528 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.64), slice={[0:3], [0:3], [0:4], [44:48]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.552 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.496, f32[3,3,4,4]{3,2,1,0} %slice.528), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_11"}
  %slice.497 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.484), slice={[0:1], [0:58], [0:58], [48:52]}, metadata={op_type="Split" op_name="split_5"}
  %slice.529 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.64), slice={[0:3], [0:3], [0:4], [48:52]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.553 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.497, f32[3,3,4,4]{3,2,1,0} %slice.529), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_12"}
  %slice.498 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.484), slice={[0:1], [0:58], [0:58], [52:56]}, metadata={op_type="Split" op_name="split_5"}
  %slice.530 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.64), slice={[0:3], [0:3], [0:4], [52:56]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.554 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.498, f32[3,3,4,4]{3,2,1,0} %slice.530), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_13"}
  %slice.499 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.484), slice={[0:1], [0:58], [0:58], [56:60]}, metadata={op_type="Split" op_name="split_5"}
  %slice.531 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.64), slice={[0:3], [0:3], [0:4], [56:60]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.555 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.499, f32[3,3,4,4]{3,2,1,0} %slice.531), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_14"}
  %slice.500 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.484), slice={[0:1], [0:58], [0:58], [60:64]}, metadata={op_type="Split" op_name="split_5"}
  %slice.532 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.64), slice={[0:3], [0:3], [0:4], [60:64]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.556 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.500, f32[3,3,4,4]{3,2,1,0} %slice.532), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_15"}
  %slice.501 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.484), slice={[0:1], [0:58], [0:58], [64:68]}, metadata={op_type="Split" op_name="split_5"}
  %slice.533 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.64), slice={[0:3], [0:3], [0:4], [64:68]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.557 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.501, f32[3,3,4,4]{3,2,1,0} %slice.533), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_16"}
  %slice.502 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.484), slice={[0:1], [0:58], [0:58], [68:72]}, metadata={op_type="Split" op_name="split_5"}
  %slice.534 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.64), slice={[0:3], [0:3], [0:4], [68:72]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.558 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.502, f32[3,3,4,4]{3,2,1,0} %slice.534), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_17"}
  %slice.503 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.484), slice={[0:1], [0:58], [0:58], [72:76]}, metadata={op_type="Split" op_name="split_5"}
  %slice.535 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.64), slice={[0:3], [0:3], [0:4], [72:76]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.559 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.503, f32[3,3,4,4]{3,2,1,0} %slice.535), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_18"}
  %slice.504 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.484), slice={[0:1], [0:58], [0:58], [76:80]}, metadata={op_type="Split" op_name="split_5"}
  %slice.536 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.64), slice={[0:3], [0:3], [0:4], [76:80]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.560 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.504, f32[3,3,4,4]{3,2,1,0} %slice.536), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_19"}
  %slice.505 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.484), slice={[0:1], [0:58], [0:58], [80:84]}, metadata={op_type="Split" op_name="split_5"}
  %slice.537 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.64), slice={[0:3], [0:3], [0:4], [80:84]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.562 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.505, f32[3,3,4,4]{3,2,1,0} %slice.537), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_20"}
  %slice.506 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.484), slice={[0:1], [0:58], [0:58], [84:88]}, metadata={op_type="Split" op_name="split_5"}
  %slice.538 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.64), slice={[0:3], [0:3], [0:4], [84:88]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.563 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.506, f32[3,3,4,4]{3,2,1,0} %slice.538), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_21"}
  %slice.507 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.484), slice={[0:1], [0:58], [0:58], [88:92]}, metadata={op_type="Split" op_name="split_5"}
  %slice.539 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.64), slice={[0:3], [0:3], [0:4], [88:92]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.564 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.507, f32[3,3,4,4]{3,2,1,0} %slice.539), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_22"}
  %slice.508 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.484), slice={[0:1], [0:58], [0:58], [92:96]}, metadata={op_type="Split" op_name="split_5"}
  %slice.540 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.64), slice={[0:3], [0:3], [0:4], [92:96]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.565 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.508, f32[3,3,4,4]{3,2,1,0} %slice.540), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_23"}
  %slice.509 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.484), slice={[0:1], [0:58], [0:58], [96:100]}, metadata={op_type="Split" op_name="split_5"}
  %slice.541 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.64), slice={[0:3], [0:3], [0:4], [96:100]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.566 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.509, f32[3,3,4,4]{3,2,1,0} %slice.541), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_24"}
  %slice.510 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.484), slice={[0:1], [0:58], [0:58], [100:104]}, metadata={op_type="Split" op_name="split_5"}
  %slice.542 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.64), slice={[0:3], [0:3], [0:4], [100:104]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.567 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.510, f32[3,3,4,4]{3,2,1,0} %slice.542), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_25"}
  %slice.511 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.484), slice={[0:1], [0:58], [0:58], [104:108]}, metadata={op_type="Split" op_name="split_5"}
  %slice.543 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.64), slice={[0:3], [0:3], [0:4], [104:108]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.568 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.511, f32[3,3,4,4]{3,2,1,0} %slice.543), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_26"}
  %slice.512 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.484), slice={[0:1], [0:58], [0:58], [108:112]}, metadata={op_type="Split" op_name="split_5"}
  %slice.544 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.64), slice={[0:3], [0:3], [0:4], [108:112]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.569 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.512, f32[3,3,4,4]{3,2,1,0} %slice.544), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_27"}
  %slice.513 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.484), slice={[0:1], [0:58], [0:58], [112:116]}, metadata={op_type="Split" op_name="split_5"}
  %slice.545 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.64), slice={[0:3], [0:3], [0:4], [112:116]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.570 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.513, f32[3,3,4,4]{3,2,1,0} %slice.545), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_28"}
  %slice.514 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.484), slice={[0:1], [0:58], [0:58], [116:120]}, metadata={op_type="Split" op_name="split_5"}
  %slice.546 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.64), slice={[0:3], [0:3], [0:4], [116:120]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.571 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.514, f32[3,3,4,4]{3,2,1,0} %slice.546), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_29"}
  %slice.515 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.484), slice={[0:1], [0:58], [0:58], [120:124]}, metadata={op_type="Split" op_name="split_5"}
  %slice.547 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.64), slice={[0:3], [0:3], [0:4], [120:124]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.573 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.515, f32[3,3,4,4]{3,2,1,0} %slice.547), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_30"}
  %slice.516 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.484), slice={[0:1], [0:58], [0:58], [124:128]}, metadata={op_type="Split" op_name="split_5"}
  %slice.548 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.64), slice={[0:3], [0:3], [0:4], [124:128]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.574 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.516, f32[3,3,4,4]{3,2,1,0} %slice.548), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_31"}
  %concatenate.581 = f32[1,56,56,128]{3,2,1,0} concatenate(f32[1,56,56,4]{3,2,1,0} %convolution.549, f32[1,56,56,4]{3,2,1,0} %convolution.550, f32[1,56,56,4]{3,2,1,0} %convolution.561, f32[1,56,56,4]{3,2,1,0} %convolution.572, f32[1,56,56,4]{3,2,1,0} %convolution.575, f32[1,56,56,4]{3,2,1,0} %convolution.576, f32[1,56,56,4]{3,2,1,0} %convolution.577, f32[1,56,56,4]{3,2,1,0} %convolution.578, f32[1,56,56,4]{3,2,1,0} %convolution.579, f32[1,56,56,4]{3,2,1,0} %convolution.580, f32[1,56,56,4]{3,2,1,0} %convolution.551, f32[1,56,56,4]{3,2,1,0} %convolution.552, f32[1,56,56,4]{3,2,1,0} %convolution.553, f32[1,56,56,4]{3,2,1,0} %convolution.554, f32[1,56,56,4]{3,2,1,0} %convolution.555, f32[1,56,56,4]{3,2,1,0} %convolution.556, f32[1,56,56,4]{3,2,1,0} %convolution.557, f32[1,56,56,4]{3,2,1,0} %convolution.558, f32[1,56,56,4]{3,2,1,0} %convolution.559, f32[1,56,56,4]{3,2,1,0} %convolution.560, f32[1,56,56,4]{3,2,1,0} %convolution.562, f32[1,56,56,4]{3,2,1,0} %convolution.563, f32[1,56,56,4]{3,2,1,0} %convolution.564, f32[1,56,56,4]{3,2,1,0} %convolution.565, f32[1,56,56,4]{3,2,1,0} %convolution.566, f32[1,56,56,4]{3,2,1,0} %convolution.567, f32[1,56,56,4]{3,2,1,0} %convolution.568, f32[1,56,56,4]{3,2,1,0} %convolution.569, f32[1,56,56,4]{3,2,1,0} %convolution.570, f32[1,56,56,4]{3,2,1,0} %convolution.571, f32[1,56,56,4]{3,2,1,0} %convolution.573, f32[1,56,56,4]{3,2,1,0} %convolution.574), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_2"}
  %constant.461 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit3_bn2/add"}
  %broadcast.462 = f32[128]{0} broadcast(f32[] %constant.461), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit3_bn2/add"}
  %arg12.13 = f32[128]{0} parameter(12), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.72 = f32[128]{0} reshape(f32[128]{0} %arg12.13)
  %add.463 = f32[128]{0} add(f32[128]{0} %broadcast.462, f32[128]{0} %reshape.72), metadata={op_type="AddV2" op_name="stage1_unit3_bn2/add"}
  %rsqrt.464 = f32[128]{0} rsqrt(f32[128]{0} %add.463), metadata={op_type="Rsqrt" op_name="stage1_unit3_bn2/Rsqrt"}
  %arg24.25 = f32[128]{0} parameter(24), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.84 = f32[128]{0} reshape(f32[128]{0} %arg24.25)
  %multiply.465 = f32[128]{0} multiply(f32[128]{0} %rsqrt.464, f32[128]{0} %reshape.84), metadata={op_type="Mul" op_name="stage1_unit3_bn2/mul"}
  %broadcast.582 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %multiply.465), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit3_bn2/mul_1"}
  %multiply.583 = f32[1,56,56,128]{3,2,1,0} multiply(f32[1,56,56,128]{3,2,1,0} %concatenate.581, f32[1,56,56,128]{3,2,1,0} %broadcast.582), metadata={op_type="Mul" op_name="stage1_unit3_bn2/mul_1"}
  %arg47.48 = f32[128]{0} parameter(47), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.107 = f32[128]{0} reshape(f32[128]{0} %arg47.48)
  %arg36.37 = f32[128]{0} parameter(36), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.96 = f32[128]{0} reshape(f32[128]{0} %arg36.37)
  %multiply.466 = f32[128]{0} multiply(f32[128]{0} %multiply.465, f32[128]{0} %reshape.96), metadata={op_type="Mul" op_name="stage1_unit3_bn2/mul_2"}
  %subtract.467 = f32[128]{0} subtract(f32[128]{0} %reshape.107, f32[128]{0} %multiply.466), metadata={op_type="Sub" op_name="stage1_unit3_bn2/sub"}
  %broadcast.584 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %subtract.467), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit3_bn2/add_1"}
  %add.585 = f32[1,56,56,128]{3,2,1,0} add(f32[1,56,56,128]{3,2,1,0} %multiply.583, f32[1,56,56,128]{3,2,1,0} %broadcast.584), metadata={op_type="AddV2" op_name="stage1_unit3_bn2/add_1"}
  %maximum.588 = f32[1,56,56,128]{3,2,1,0} maximum(f32[1,56,56,128]{3,2,1,0} %broadcast.587, f32[1,56,56,128]{3,2,1,0} %add.585), metadata={op_type="Relu" op_name="stage1_unit3_relu2"}
  %arg57.58 = f32[1,1,128,256]{3,2,1,0} parameter(57), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.117 = f32[1,1,128,256]{3,2,1,0} reshape(f32[1,1,128,256]{3,2,1,0} %arg57.58)
  %convolution.589 = f32[1,56,56,256]{3,2,1,0} convolution(f32[1,56,56,128]{3,2,1,0} %maximum.588, f32[1,1,128,256]{3,2,1,0} %reshape.117), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv3"}
  %multiply.591 = f32[1,56,56,256]{3,2,1,0} multiply(f32[1,56,56,256]{3,2,1,0} %broadcast.590, f32[1,56,56,256]{3,2,1,0} %convolution.589), metadata={op_type="Mul" op_name="stage1_unit3_bn3/mul_1"}
  %arg42.43 = f32[256]{0} parameter(42), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.102 = f32[256]{0} reshape(f32[256]{0} %arg42.43)
  %arg31.32 = f32[256]{0} parameter(31), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.91 = f32[256]{0} reshape(f32[256]{0} %arg31.32)
  %multiply.473 = f32[256]{0} multiply(f32[256]{0} %multiply.472, f32[256]{0} %reshape.91), metadata={op_type="Mul" op_name="stage1_unit3_bn3/mul_2"}
  %subtract.474 = f32[256]{0} subtract(f32[256]{0} %reshape.102, f32[256]{0} %multiply.473), metadata={op_type="Sub" op_name="stage1_unit3_bn3/sub"}
  %broadcast.592 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.474), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit3_bn3/add_1"}
  %add.593 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %multiply.591, f32[1,56,56,256]{3,2,1,0} %broadcast.592), metadata={op_type="AddV2" op_name="stage1_unit3_bn3/add_1"}
  %add.594 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %maximum.453, f32[1,56,56,256]{3,2,1,0} %add.593), metadata={op_type="AddV2" op_name="add_2"}
  %maximum.597 = f32[1,56,56,256]{3,2,1,0} maximum(f32[1,56,56,256]{3,2,1,0} %broadcast.596, f32[1,56,56,256]{3,2,1,0} %add.594), metadata={op_type="Relu" op_name="stage1_unit3_relu"}
  %reshape.598 = f32[1,56,56,256]{3,2,1,0} reshape(f32[1,56,56,256]{3,2,1,0} %maximum.597), metadata={op_name="XLA_Retvals"}
  %tuple.599 = (f32[1,56,56,256]{3,2,1,0}) tuple(f32[1,56,56,256]{3,2,1,0} %reshape.598), metadata={op_name="XLA_Retvals"}
  ROOT %get-tuple-element.600 = f32[1,56,56,256]{3,2,1,0} get-tuple-element((f32[1,56,56,256]{3,2,1,0}) %tuple.599), index=0, metadata={op_name="XLA_Retvals"}
}
)";

  hlo_module->ParseHloStringAndVerifyModule(hlo_text); 

  CompileAndCheck(std::move(hlo_module), spec.filecheck_lines, testcase_pairs);
}

std::vector<ResNeXTTestSpec> GetResNeXTTestCases() {
  std::vector<ResNeXTTestSpec> result;
  result.push_back(
      {F32, R"(CHECK: func @hlo_module)"});
  return result;
}

/**/
// TODO: INSTANTIATE_TEST_CASE_P was deprecated in favor for INSTANTIATE_TEST_SUITE_P, but the version of gtest that bazel links in is looking for INSTANTIATE_TEST_CASE_P right now.
INSTANTIATE_TEST_CASE_P(All,
                         PlaidMLResNeXTOperationTest,
                         ::testing::ValuesIn(GetResNeXTTestCases()),
                         ResNeXTTestSpecToString);
/**/
}  // namespace
}  // namespace plaidml
}  // namespace xla
