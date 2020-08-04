// Tests that show HLO Module conversion to PlaidML Program.

#include <algorithm>
#include <string>

#include <gtest/gtest.h>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/service/plaidml/compiler.h"
#include "tensorflow/compiler/xla/service/plaidml/tests/plaidml_codegen_test.h"
#include "tensorflow/compiler/xla/service/plaidml/tests/resnext50_pretrained_inputs_and_weights.h"
#include "tensorflow/compiler/xla/service/plaidml/tests/resnext50_stage2_output.h"
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
    {0}, // stage2_unit4_relu
    ::weights::stage2_unit4_bn3_mean, //
    ::weights::stage2_unit4_bn3_scale, //
    ::weights::stage2_unit4_bn3_var, //
    {2e-05}, // stage2_unit4_bn3/add
    ::weights::stage2_unit4_bn3_bias, //
    ::weights::stage2_unit4_conv3_weight, //
    {0}, // stage2_unit4_relu2
    ::weights::stage2_unit4_bn2_mean, //
    ::weights::stage2_unit4_bn2_scale, //
    ::weights::stage2_unit4_bn2_var, //
    {2e-05}, // stage2_unit4_bn2/add
    ::weights::stage2_unit4_bn2_bias, //
    ::weights::stage2_unit4_conv2_weight, //
    {0}, // stage2_unit4_relu1
    ::weights::stage2_unit4_bn1_mean, //
    ::weights::stage2_unit4_bn1_scale, //
    ::weights::stage2_unit4_bn1_var, //
    {2e-05}, // stage2_unit4_bn1/add
    ::weights::stage2_unit4_bn1_bias, //
    ::weights::stage2_unit4_conv1_weight, //
    {0}, // stage2_unit3_relu
    ::weights::stage2_unit3_bn3_mean, //
    ::weights::stage2_unit3_bn3_scale, //
    ::weights::stage2_unit3_bn3_var, //
    {2e-05}, // stage2_unit3_bn3/add
    ::weights::stage2_unit3_bn3_bias, //
    ::weights::stage2_unit3_conv3_weight, //
    {0}, // stage2_unit3_relu2
    ::weights::stage2_unit3_bn2_mean, //
    ::weights::stage2_unit3_bn2_scale, //
    ::weights::stage2_unit3_bn2_var, //
    {2e-05}, // stage2_unit3_bn2/add
    ::weights::stage2_unit3_bn2_bias, //
    ::weights::stage2_unit3_conv2_weight, //
    {0}, // stage2_unit3_relu1
    ::weights::stage2_unit3_bn1_mean, //
    ::weights::stage2_unit3_bn1_scale, //
    ::weights::stage2_unit3_bn1_var, //
    {2e-05}, // stage2_unit3_bn1/add
    ::weights::stage2_unit3_bn1_bias, //
    ::weights::stage2_unit3_conv1_weight, //
    {0}, // stage2_unit2_relu
    ::weights::stage2_unit2_bn3_mean, //
    ::weights::stage2_unit2_bn3_scale, //
    ::weights::stage2_unit2_bn3_var, //
    {2e-05}, // stage2_unit2_bn3/add
    ::weights::stage2_unit2_bn3_bias, //
    ::weights::stage2_unit2_conv3_weight, //
    {0}, // stage2_unit2_relu2
    ::weights::stage2_unit2_bn2_mean, //
    ::weights::stage2_unit2_bn2_scale, //
    ::weights::stage2_unit2_bn2_var, //
    {2e-05}, // stage2_unit2_bn2/add
    ::weights::stage2_unit2_bn2_bias, //
    ::weights::stage2_unit2_conv2_weight, //
    {0}, // stage2_unit2_relu1
    ::weights::stage2_unit2_bn1_mean, //
    ::weights::stage2_unit2_bn1_scale, //
    ::weights::stage2_unit2_bn1_var, //
    {2e-05}, // stage2_unit2_bn1/add
    ::weights::stage2_unit2_bn1_bias, //
    ::weights::stage2_unit2_conv1_weight, //
    {0}, // stage2_unit1_relu
    ::weights::stage2_unit1_sc_bn_mean, //
    ::weights::stage2_unit1_sc_bn_scale, //
    ::weights::stage2_unit1_sc_bn_var, //
    {2e-05}, // stage2_unit1_sc_bn/add
    ::weights::stage2_unit1_sc_bn_bias, //
    ::weights::stage2_unit1_sc_weight, //
    {0}, // stage1_unit3_relu 
    ::weights::stage1_unit3_bn3_mean,
    ::weights::stage1_unit3_bn3_scale,
    ::weights::stage1_unit3_bn3_var,
    {2e-05}, // stage1_unit3_bn3/add
    ::weights::stage1_unit3_bn3_bias,
    ::weights::stage1_unit3_conv3_weight,
    {0}, // stage1_unit3_relu2
    ::weights::stage1_unit3_bn2_mean,
    ::weights::stage1_unit3_bn2_scale,
    ::weights::stage1_unit3_bn2_var,
    {2e-05}, // stage1_unit3_bn2/add
    ::weights::stage1_unit3_bn2_bias,
    ::weights::stage1_unit3_conv2_weight,
    {0}, // stage1_unit3_relu1
    ::weights::stage1_unit3_bn1_mean,
    ::weights::stage1_unit3_bn1_scale,
    ::weights::stage1_unit3_bn1_var,
    {2e-05}, // stage1_unit3_bn1/add
    ::weights::stage1_unit3_bn1_bias,
    ::weights::stage1_unit3_conv1_weight,
    {0}, // stage1_unit2_relu
    ::weights::stage1_unit2_bn3_mean,
    ::weights::stage1_unit2_bn3_scale,
    ::weights::stage1_unit2_bn3_var,
    {2e-05}, // stage1_unit2_bn3/add
    ::weights::stage1_unit2_bn3_bias,
    ::weights::stage1_unit2_conv3_weight,
    {0}, // stage1_unit2_relu2
    ::weights::stage1_unit2_bn2_mean,
    ::weights::stage1_unit2_bn2_scale,
    ::weights::stage1_unit2_bn2_var,
    {2e-05}, // stage1_unit2_bn2/add
    ::weights::stage1_unit2_bn2_bias,
    ::weights::stage1_unit2_conv2_weight,
    {0}, // stage1_unit2_relu1
    ::weights::stage1_unit2_bn1_mean,
    ::weights::stage1_unit2_bn1_scale,
    ::weights::stage1_unit2_bn1_var,
    {2e-05}, // stage1_unit2_bn1/add
    ::weights::stage1_unit2_bn1_bias,
    ::weights::stage1_unit2_conv1_weight,
    {0}, // stage1_unit1_relu
    ::weights::stage1_unit1_sc_bn_mean,
    ::weights::stage1_unit1_sc_bn_scale,
    ::weights::stage1_unit1_sc_bn_var,
    {2e-05}, // stage1_unit1_sc_bn/add
    ::weights::stage1_unit1_sc_bn_bias,
    ::weights::stage1_unit1_sc_weight,
    {0}, // relu0
    ::weights::bn0_mean,
    ::weights::bn0_scale,
    {2e-05}, // bn0/add
    ::weights::bn0_var,
    ::weights::bn0_bias,
    ::weights::conv0_weight,
    ::weights::bn_data_mean,
    ::weights::bn_data_var,
    {2e-05}, // bn_data/add
    ::weights::bn_data_bias,
    ::weights::input_tensor,
    ::weights::stage1_unit1_bn3_mean,
    ::weights::stage1_unit1_bn3_scale,
    ::weights::stage1_unit1_bn3_var,
    {2e-05}, // stage1_unit1_bn3/add
    ::weights::stage1_unit1_bn3_bias,
    ::weights::stage1_unit1_conv3_weight,
    {0}, // stage1_unit1_relu2
    ::weights::stage1_unit1_bn2_mean,
    ::weights::stage1_unit1_bn2_scale,
    ::weights::stage1_unit1_bn2_var,
    {2e-05}, // stage1_unit1_bn2/add
    ::weights::stage1_unit1_bn2_bias,
    ::weights::stage1_unit1_conv2_weight,
    {0}, // stage1_unit1_relu1
    ::weights::stage1_unit1_bn1_mean,
    ::weights::stage1_unit1_bn1_scale,
    ::weights::stage1_unit1_bn1_var,
    {2e-05}, // stage1_unit1_bn1/add
    ::weights::stage1_unit1_bn1_bias,
    ::weights::stage1_unit1_conv1_weight,
    ::weights::stage2_unit1_bn3_mean, //
    ::weights::stage2_unit1_bn3_scale, //
    ::weights::stage2_unit1_bn3_var, //
    {2e-05}, // stage2_unit1_bn3/add
    ::weights::stage2_unit1_bn3_bias, //
    ::weights::stage2_unit1_conv3_weight, //
    {0}, // stage2_unit1_relu2
    ::weights::stage2_unit1_bn2_mean, //
    ::weights::stage2_unit1_bn2_scale, //
    ::weights::stage2_unit1_bn2_var, //
    {2e-05}, // stage2_unit1_bn2/add
    ::weights::stage2_unit1_bn2_bias, //
    ::weights::stage2_unit1_conv2_weight, //
    {0}, // stage2_unit1_relu1
    ::weights::stage2_unit1_bn1_mean, //
    ::weights::stage2_unit1_bn1_scale, //
    ::weights::stage2_unit1_bn1_var, //
    {2e-05}, // stage2_unit1_bn1/add
    ::weights::stage2_unit1_bn1_bias, //
    ::weights::stage2_unit1_conv1_weight
  };

  TestCaseVal ResNeXt50_Outputs = ::outputs::ResNext50_Outputs; 

  TestCasePairs testcase_pairs = {{ResNeXt50_WeightsInputs, ResNeXt50_Outputs}}; 

  ResNeXTTestSpec spec = GetParam();

  HloModuleConfig cfg;

  std::unique_ptr<VerifiedHloModule> hlo_module = absl::make_unique<VerifiedHloModule>(
      "module", cfg, false, false, nullptr);

std::string hlo_text = R"(
HloModule cluster_0__XlaCompiledKernel_true__XlaNumConstantArgs_30__XlaNumResourceArgs_0_.1319

%max_F32.279 (lhs.280: f32[], rhs.281: f32[]) -> f32[] {
  %lhs.280 = f32[] parameter(0)
  %rhs.281 = f32[] parameter(1)
  ROOT %maximum.282 = f32[] maximum(f32[] %lhs.280, f32[] %rhs.281)
}

ENTRY %cluster_0__XlaCompiledKernel_true__XlaNumConstantArgs_30__XlaNumResourceArgs_0_.1319 (arg0.1: f32[3], arg1.2: f32[256], arg2.3: f32[64], arg3.4: f32[128], arg4.5: f32[3,3,8,256], arg5.6: f32[256], arg6.7: f32[256], arg7.8: f32[3,3,4,128], arg8.9: f32[128], arg9.10: f32[256], arg10.11: f32[128], arg11.12: f32[3,3,4,128], arg12.13: f32[128], arg13.14: f32[256], arg14.15: f32[256], arg15.16: f32[128], arg16.17: f32[3,3,4,128], arg17.18: f32[128], arg18.19: f32[512], arg19.20: f32[256], arg20.21: f32[256], arg21.22: f32[512], arg22.23: f32[256], arg23.24: f32[3,3,8,256], arg24.25: f32[256], arg25.26: f32[512], arg26.27: f32[3,3,8,256], arg27.28: f32[512], arg28.29: f32[256], arg29.30: f32[3,3,8,256], arg30.31: f32[256], arg31.32: f32[512], arg32.33: f32[3], arg33.34: f32[256], arg34.35: f32[64], arg35.36: f32[128], arg36.37: f32[256], arg37.38: f32[256], arg38.39: f32[128], arg39.40: f32[256], arg40.41: f32[128], arg41.42: f32[128], arg42.43: f32[256], arg43.44: f32[256], arg44.45: f32[128], arg45.46: f32[128], arg46.47: f32[512], arg47.48: f32[256], arg48.49: f32[256], arg49.50: f32[512], arg50.51: f32[256], arg51.52: f32[256], arg52.53: f32[512], arg53.54: f32[512], arg54.55: f32[256], arg55.56: f32[256], arg56.57: f32[512], arg57.58: f32[3], arg58.59: f32[256], arg59.60: f32[64], arg60.61: f32[128], arg61.62: f32[256], arg62.63: f32[256], arg63.64: f32[128], arg64.65: f32[256], arg65.66: f32[128], arg66.67: f32[128], arg67.68: f32[256], arg68.69: f32[256], arg69.70: f32[128], arg70.71: f32[128], arg71.72: f32[512], arg72.73: f32[256], arg73.74: f32[256], arg74.75: f32[512], arg75.76: f32[256], arg76.77: f32[256], arg77.78: f32[512], arg78.79: f32[512], arg79.80: f32[256], arg80.81: f32[256], arg81.82: f32[512], arg82.83: f32[256], arg83.84: f32[64], arg84.85: f32[128], arg85.86: f32[256], arg86.87: f32[256], arg87.88: f32[128], arg88.89: f32[256], arg89.90: f32[128], arg90.91: f32[128], arg91.92: f32[256], arg92.93: f32[256], arg93.94: f32[128], arg94.95: f32[128], arg95.96: f32[512], arg96.97: f32[256], arg97.98: f32[256], arg98.99: f32[512], arg99.100: f32[256], arg100.101: f32[256], arg101.102: f32[512], arg102.103: f32[512], arg103.104: f32[256], arg104.105: f32[256], arg105.106: f32[512], arg106.107: f32[7,7,3,64], arg107.108: f32[1,1,64,128], arg108.109: f32[1,1,64,256], arg109.110: f32[1,1,128,256], arg110.111: f32[1,1,256,128], arg111.112: f32[1,1,128,256], arg112.113: f32[1,1,256,128], arg113.114: f32[1,1,128,256], arg114.115: f32[1,1,256,256], arg115.116: f32[1,1,256,512], arg116.117: f32[1,1,256,512], arg117.118: f32[1,1,512,256], arg118.119: f32[1,1,256,512], arg119.120: f32[1,1,512,256], arg120.121: f32[1,1,256,512], arg121.122: f32[1,1,512,256], arg122.123: f32[1,1,256,512], arg123.124: f32[1,224,224,3]) -> f32[1,28,28,512] {
  %constant.1313 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit4_relu"}
  %broadcast.1314 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[] %constant.1313), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit4_relu"}
  %constant.1169 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit3_relu"}
  %broadcast.1170 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[] %constant.1169), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit3_relu"}
  %constant.1025 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit2_relu"}
  %broadcast.1026 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[] %constant.1025), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit2_relu"}
  %constant.881 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit1_relu"}
  %broadcast.882 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[] %constant.881), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit1_relu"}
  %constant.742 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit1_bn3/add"}
  %broadcast.743 = f32[512]{0} broadcast(f32[] %constant.742), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit1_bn3/add"}
  %arg25.26 = f32[512]{0} parameter(25), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.150 = f32[512]{0} reshape(f32[512]{0} %arg25.26)
  %add.744 = f32[512]{0} add(f32[512]{0} %broadcast.743, f32[512]{0} %reshape.150), metadata={op_type="AddV2" op_name="stage2_unit1_bn3/add"}
  %rsqrt.745 = f32[512]{0} rsqrt(f32[512]{0} %add.744), metadata={op_type="Rsqrt" op_name="stage2_unit1_bn3/Rsqrt"}
  %arg52.53 = f32[512]{0} parameter(52), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.177 = f32[512]{0} reshape(f32[512]{0} %arg52.53)
  %multiply.746 = f32[512]{0} multiply(f32[512]{0} %rsqrt.745, f32[512]{0} %reshape.177), metadata={op_type="Mul" op_name="stage2_unit1_bn3/mul"}
  %broadcast.864 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.746), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit1_bn3/mul_1"}
  %constant.860 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit1_relu2"}
  %broadcast.861 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[] %constant.860), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit1_relu2"}
  %constant.754 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit1_relu1"}
  %broadcast.755 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[] %constant.754), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit1_relu1"}
  %constant.728 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit1_bn1/add"}
  %broadcast.729 = f32[256]{0} broadcast(f32[] %constant.728), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit1_bn1/add"}
  %arg20.21 = f32[256]{0} parameter(20), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.145 = f32[256]{0} reshape(f32[256]{0} %arg20.21)
  %add.730 = f32[256]{0} add(f32[256]{0} %broadcast.729, f32[256]{0} %reshape.145), metadata={op_type="AddV2" op_name="stage2_unit1_bn1/add"}
  %rsqrt.731 = f32[256]{0} rsqrt(f32[256]{0} %add.730), metadata={op_type="Rsqrt" op_name="stage2_unit1_bn1/Rsqrt"}
  %arg48.49 = f32[256]{0} parameter(48), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.173 = f32[256]{0} reshape(f32[256]{0} %arg48.49)
  %multiply.732 = f32[256]{0} multiply(f32[256]{0} %rsqrt.731, f32[256]{0} %reshape.173), metadata={op_type="Mul" op_name="stage2_unit1_bn1/mul"}
  %broadcast.750 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.732), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit1_bn1/mul_1"}
  %constant.725 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit3_relu"}
  %broadcast.726 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[] %constant.725), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit3_relu"}
  %constant.581 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit2_relu"}
  %broadcast.582 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[] %constant.581), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit2_relu"}
  %constant.437 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit1_relu"}
  %broadcast.438 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[] %constant.437), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit1_relu"}
  %constant.298 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit1_bn3/add"}
  %broadcast.299 = f32[256]{0} broadcast(f32[] %constant.298), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit1_bn3/add"}
  %arg9.10 = f32[256]{0} parameter(9), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.134 = f32[256]{0} reshape(f32[256]{0} %arg9.10)
  %add.300 = f32[256]{0} add(f32[256]{0} %broadcast.299, f32[256]{0} %reshape.134), metadata={op_type="AddV2" op_name="stage1_unit1_bn3/add"}
  %rsqrt.301 = f32[256]{0} rsqrt(f32[256]{0} %add.300), metadata={op_type="Rsqrt" op_name="stage1_unit1_bn3/Rsqrt"}
  %arg39.40 = f32[256]{0} parameter(39), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.164 = f32[256]{0} reshape(f32[256]{0} %arg39.40)
  %multiply.302 = f32[256]{0} multiply(f32[256]{0} %rsqrt.301, f32[256]{0} %reshape.164), metadata={op_type="Mul" op_name="stage1_unit1_bn3/mul"}
  %broadcast.420 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.302), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit1_bn3/mul_1"}
  %constant.416 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit1_relu2"}
  %broadcast.417 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[] %constant.416), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit1_relu2"}
  %constant.310 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit1_relu1"}
  %broadcast.311 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[] %constant.310), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit1_relu1"}
  %constant.284 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit1_bn1/add"}
  %broadcast.285 = f32[128]{0} broadcast(f32[] %constant.284), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit1_bn1/add"}
  %arg3.4 = f32[128]{0} parameter(3), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.128 = f32[128]{0} reshape(f32[128]{0} %arg3.4)
  %add.286 = f32[128]{0} add(f32[128]{0} %broadcast.285, f32[128]{0} %reshape.128), metadata={op_type="AddV2" op_name="stage1_unit1_bn1/add"}
  %rsqrt.287 = f32[128]{0} rsqrt(f32[128]{0} %add.286), metadata={op_type="Rsqrt" op_name="stage1_unit1_bn1/Rsqrt"}
  %arg35.36 = f32[128]{0} parameter(35), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.160 = f32[128]{0} reshape(f32[128]{0} %arg35.36)
  %multiply.288 = f32[128]{0} multiply(f32[128]{0} %rsqrt.287, f32[128]{0} %reshape.160), metadata={op_type="Mul" op_name="stage1_unit1_bn1/mul"}
  %broadcast.306 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %multiply.288), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit1_bn1/mul_1"}
  %constant.273 = f32[] constant(0), metadata={op_type="Relu" op_name="relu0"}
  %broadcast.274 = f32[1,112,112,64]{3,2,1,0} broadcast(f32[] %constant.273), dimensions={}, metadata={op_type="Relu" op_name="relu0"}
  %arg2.3 = f32[64]{0} parameter(2), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.127 = f32[64]{0} reshape(f32[64]{0} %arg2.3)
  %constant.249 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="bn0/add"}
  %broadcast.250 = f32[64]{0} broadcast(f32[] %constant.249), dimensions={}, metadata={op_type="AddV2" op_name="bn0/add"}
  %add.251 = f32[64]{0} add(f32[64]{0} %reshape.127, f32[64]{0} %broadcast.250), metadata={op_type="AddV2" op_name="bn0/add"}
  %rsqrt.252 = f32[64]{0} rsqrt(f32[64]{0} %add.251), metadata={op_type="Rsqrt" op_name="bn0/Rsqrt"}
  %arg34.35 = f32[64]{0} parameter(34), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.159 = f32[64]{0} reshape(f32[64]{0} %arg34.35)
  %multiply.253 = f32[64]{0} multiply(f32[64]{0} %rsqrt.252, f32[64]{0} %reshape.159), metadata={op_type="Mul" op_name="bn0/mul"}
  %broadcast.269 = f32[1,112,112,64]{3,2,1,0} broadcast(f32[64]{0} %multiply.253), dimensions={3}, metadata={op_type="Mul" op_name="bn0/mul_1"}
  %constant.256 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="bn_data/add"}
  %broadcast.257 = f32[3]{0} broadcast(f32[] %constant.256), dimensions={}, metadata={op_type="AddV2" op_name="bn_data/add"}
  %arg0.1 = f32[3]{0} parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.125 = f32[3]{0} reshape(f32[3]{0} %arg0.1)
  %add.258 = f32[3]{0} add(f32[3]{0} %broadcast.257, f32[3]{0} %reshape.125), metadata={op_type="AddV2" op_name="bn_data/add"}
  %rsqrt.259 = f32[3]{0} rsqrt(f32[3]{0} %add.258), metadata={op_type="Rsqrt" op_name="bn_data/Rsqrt"}
  %broadcast.260 = f32[1,224,224,3]{3,2,1,0} broadcast(f32[3]{0} %rsqrt.259), dimensions={3}, metadata={op_type="Mul" op_name="bn_data/mul"}
  %arg123.124 = f32[1,224,224,3]{3,2,1,0} parameter(123), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.248 = f32[1,224,224,3]{3,2,1,0} reshape(f32[1,224,224,3]{3,2,1,0} %arg123.124)
  %multiply.261 = f32[1,224,224,3]{3,2,1,0} multiply(f32[1,224,224,3]{3,2,1,0} %broadcast.260, f32[1,224,224,3]{3,2,1,0} %reshape.248), metadata={op_type="Mul" op_name="bn_data/mul"}
  %arg57.58 = f32[3]{0} parameter(57), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.182 = f32[3]{0} reshape(f32[3]{0} %arg57.58)
  %arg32.33 = f32[3]{0} parameter(32), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.157 = f32[3]{0} reshape(f32[3]{0} %arg32.33)
  %multiply.262 = f32[3]{0} multiply(f32[3]{0} %rsqrt.259, f32[3]{0} %reshape.157), metadata={op_type="Mul" op_name="bn_data/mul_1"}
  %subtract.263 = f32[3]{0} subtract(f32[3]{0} %reshape.182, f32[3]{0} %multiply.262), metadata={op_type="Sub" op_name="bn_data/sub"}
  %broadcast.264 = f32[1,224,224,3]{3,2,1,0} broadcast(f32[3]{0} %subtract.263), dimensions={3}, metadata={op_type="AddV2" op_name="bn_data/add_1"}
  %add.265 = f32[1,224,224,3]{3,2,1,0} add(f32[1,224,224,3]{3,2,1,0} %multiply.261, f32[1,224,224,3]{3,2,1,0} %broadcast.264), metadata={op_type="AddV2" op_name="bn_data/add_1"}
  %constant.266 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad"}
  %pad.267 = f32[1,230,230,3]{3,2,1,0} pad(f32[1,224,224,3]{3,2,1,0} %add.265, f32[] %constant.266), padding=0_0x3_3x3_3x0_0, metadata={op_type="Pad" op_name="Pad"}
  %arg106.107 = f32[7,7,3,64]{3,2,1,0} parameter(106), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.231 = f32[7,7,3,64]{3,2,1,0} reshape(f32[7,7,3,64]{3,2,1,0} %arg106.107)
  %convolution.268 = f32[1,112,112,64]{3,2,1,0} convolution(f32[1,230,230,3]{3,2,1,0} %pad.267, f32[7,7,3,64]{3,2,1,0} %reshape.231), window={size=7x7 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="conv0"}
  %multiply.270 = f32[1,112,112,64]{3,2,1,0} multiply(f32[1,112,112,64]{3,2,1,0} %broadcast.269, f32[1,112,112,64]{3,2,1,0} %convolution.268), metadata={op_type="Mul" op_name="bn0/mul_1"}
  %arg83.84 = f32[64]{0} parameter(83), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.208 = f32[64]{0} reshape(f32[64]{0} %arg83.84)
  %arg59.60 = f32[64]{0} parameter(59), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.184 = f32[64]{0} reshape(f32[64]{0} %arg59.60)
  %multiply.254 = f32[64]{0} multiply(f32[64]{0} %multiply.253, f32[64]{0} %reshape.184), metadata={op_type="Mul" op_name="bn0/mul_2"}
  %subtract.255 = f32[64]{0} subtract(f32[64]{0} %reshape.208, f32[64]{0} %multiply.254), metadata={op_type="Sub" op_name="bn0/sub"}
  %broadcast.271 = f32[1,112,112,64]{3,2,1,0} broadcast(f32[64]{0} %subtract.255), dimensions={3}, metadata={op_type="AddV2" op_name="bn0/add_1"}
  %add.272 = f32[1,112,112,64]{3,2,1,0} add(f32[1,112,112,64]{3,2,1,0} %multiply.270, f32[1,112,112,64]{3,2,1,0} %broadcast.271), metadata={op_type="AddV2" op_name="bn0/add_1"}
  %maximum.275 = f32[1,112,112,64]{3,2,1,0} maximum(f32[1,112,112,64]{3,2,1,0} %broadcast.274, f32[1,112,112,64]{3,2,1,0} %add.272), metadata={op_type="Relu" op_name="relu0"}
  %constant.276 = f32[] constant(-inf), metadata={op_type="PadV2" op_name="PadV2"}
  %pad.277 = f32[1,114,114,64]{3,2,1,0} pad(f32[1,112,112,64]{3,2,1,0} %maximum.275, f32[] %constant.276), padding=0_0x1_1x1_1x0_0, metadata={op_type="PadV2" op_name="PadV2"}
  %constant.278 = f32[] constant(-inf), metadata={op_type="MaxPool" op_name="pooling0"}
  %reduce-window.283 = f32[1,56,56,64]{3,2,1,0} reduce-window(f32[1,114,114,64]{3,2,1,0} %pad.277, f32[] %constant.278), window={size=1x3x3x1 stride=1x2x2x1}, to_apply=%max_F32.279, metadata={op_type="MaxPool" op_name="pooling0"}
  %arg107.108 = f32[1,1,64,128]{3,2,1,0} parameter(107), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.232 = f32[1,1,64,128]{3,2,1,0} reshape(f32[1,1,64,128]{3,2,1,0} %arg107.108)
  %convolution.305 = f32[1,56,56,128]{3,2,1,0} convolution(f32[1,56,56,64]{3,2,1,0} %reduce-window.283, f32[1,1,64,128]{3,2,1,0} %reshape.232), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv1"}
  %multiply.307 = f32[1,56,56,128]{3,2,1,0} multiply(f32[1,56,56,128]{3,2,1,0} %broadcast.306, f32[1,56,56,128]{3,2,1,0} %convolution.305), metadata={op_type="Mul" op_name="stage1_unit1_bn1/mul_1"}
  %arg84.85 = f32[128]{0} parameter(84), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.209 = f32[128]{0} reshape(f32[128]{0} %arg84.85)
  %arg60.61 = f32[128]{0} parameter(60), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.185 = f32[128]{0} reshape(f32[128]{0} %arg60.61)
  %multiply.289 = f32[128]{0} multiply(f32[128]{0} %multiply.288, f32[128]{0} %reshape.185), metadata={op_type="Mul" op_name="stage1_unit1_bn1/mul_2"}
  %subtract.290 = f32[128]{0} subtract(f32[128]{0} %reshape.209, f32[128]{0} %multiply.289), metadata={op_type="Sub" op_name="stage1_unit1_bn1/sub"}
  %broadcast.308 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %subtract.290), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit1_bn1/add_1"}
  %add.309 = f32[1,56,56,128]{3,2,1,0} add(f32[1,56,56,128]{3,2,1,0} %multiply.307, f32[1,56,56,128]{3,2,1,0} %broadcast.308), metadata={op_type="AddV2" op_name="stage1_unit1_bn1/add_1"}
  %maximum.312 = f32[1,56,56,128]{3,2,1,0} maximum(f32[1,56,56,128]{3,2,1,0} %broadcast.311, f32[1,56,56,128]{3,2,1,0} %add.309), metadata={op_type="Relu" op_name="stage1_unit1_relu1"}
  %constant.313 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_1"}
  %pad.314 = f32[1,58,58,128]{3,2,1,0} pad(f32[1,56,56,128]{3,2,1,0} %maximum.312, f32[] %constant.313), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_1"}
  %slice.315 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.314), slice={[0:1], [0:58], [0:58], [0:4]}, metadata={op_type="Split" op_name="split_1"}
  %arg7.8 = f32[3,3,4,128]{3,2,1,0} parameter(7), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.132 = f32[3,3,4,128]{3,2,1,0} reshape(f32[3,3,4,128]{3,2,1,0} %arg7.8)
  %slice.347 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.132), slice={[0:3], [0:3], [0:4], [0:4]}, metadata={op_type="Split" op_name="split"}
  %convolution.379 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.315, f32[3,3,4,4]{3,2,1,0} %slice.347), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2"}
  %slice.316 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.314), slice={[0:1], [0:58], [0:58], [4:8]}, metadata={op_type="Split" op_name="split_1"}
  %slice.348 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.132), slice={[0:3], [0:3], [0:4], [4:8]}, metadata={op_type="Split" op_name="split"}
  %convolution.380 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.316, f32[3,3,4,4]{3,2,1,0} %slice.348), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_1"}
  %slice.317 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.314), slice={[0:1], [0:58], [0:58], [8:12]}, metadata={op_type="Split" op_name="split_1"}
  %slice.349 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.132), slice={[0:3], [0:3], [0:4], [8:12]}, metadata={op_type="Split" op_name="split"}
  %convolution.391 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.317, f32[3,3,4,4]{3,2,1,0} %slice.349), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_2"}
  %slice.318 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.314), slice={[0:1], [0:58], [0:58], [12:16]}, metadata={op_type="Split" op_name="split_1"}
  %slice.350 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.132), slice={[0:3], [0:3], [0:4], [12:16]}, metadata={op_type="Split" op_name="split"}
  %convolution.402 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.318, f32[3,3,4,4]{3,2,1,0} %slice.350), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_3"}
  %slice.319 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.314), slice={[0:1], [0:58], [0:58], [16:20]}, metadata={op_type="Split" op_name="split_1"}
  %slice.351 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.132), slice={[0:3], [0:3], [0:4], [16:20]}, metadata={op_type="Split" op_name="split"}
  %convolution.405 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.319, f32[3,3,4,4]{3,2,1,0} %slice.351), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_4"}
  %slice.320 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.314), slice={[0:1], [0:58], [0:58], [20:24]}, metadata={op_type="Split" op_name="split_1"}
  %slice.352 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.132), slice={[0:3], [0:3], [0:4], [20:24]}, metadata={op_type="Split" op_name="split"}
  %convolution.406 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.320, f32[3,3,4,4]{3,2,1,0} %slice.352), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_5"}
  %slice.321 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.314), slice={[0:1], [0:58], [0:58], [24:28]}, metadata={op_type="Split" op_name="split_1"}
  %slice.353 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.132), slice={[0:3], [0:3], [0:4], [24:28]}, metadata={op_type="Split" op_name="split"}
  %convolution.407 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.321, f32[3,3,4,4]{3,2,1,0} %slice.353), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_6"}
  %slice.322 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.314), slice={[0:1], [0:58], [0:58], [28:32]}, metadata={op_type="Split" op_name="split_1"}
  %slice.354 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.132), slice={[0:3], [0:3], [0:4], [28:32]}, metadata={op_type="Split" op_name="split"}
  %convolution.408 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.322, f32[3,3,4,4]{3,2,1,0} %slice.354), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_7"}
  %slice.323 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.314), slice={[0:1], [0:58], [0:58], [32:36]}, metadata={op_type="Split" op_name="split_1"}
  %slice.355 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.132), slice={[0:3], [0:3], [0:4], [32:36]}, metadata={op_type="Split" op_name="split"}
  %convolution.409 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.323, f32[3,3,4,4]{3,2,1,0} %slice.355), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_8"}
  %slice.324 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.314), slice={[0:1], [0:58], [0:58], [36:40]}, metadata={op_type="Split" op_name="split_1"}
  %slice.356 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.132), slice={[0:3], [0:3], [0:4], [36:40]}, metadata={op_type="Split" op_name="split"}
  %convolution.410 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.324, f32[3,3,4,4]{3,2,1,0} %slice.356), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_9"}
  %slice.325 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.314), slice={[0:1], [0:58], [0:58], [40:44]}, metadata={op_type="Split" op_name="split_1"}
  %slice.357 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.132), slice={[0:3], [0:3], [0:4], [40:44]}, metadata={op_type="Split" op_name="split"}
  %convolution.381 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.325, f32[3,3,4,4]{3,2,1,0} %slice.357), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_10"}
  %slice.326 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.314), slice={[0:1], [0:58], [0:58], [44:48]}, metadata={op_type="Split" op_name="split_1"}
  %slice.358 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.132), slice={[0:3], [0:3], [0:4], [44:48]}, metadata={op_type="Split" op_name="split"}
  %convolution.382 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.326, f32[3,3,4,4]{3,2,1,0} %slice.358), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_11"}
  %slice.327 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.314), slice={[0:1], [0:58], [0:58], [48:52]}, metadata={op_type="Split" op_name="split_1"}
  %slice.359 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.132), slice={[0:3], [0:3], [0:4], [48:52]}, metadata={op_type="Split" op_name="split"}
  %convolution.383 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.327, f32[3,3,4,4]{3,2,1,0} %slice.359), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_12"}
  %slice.328 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.314), slice={[0:1], [0:58], [0:58], [52:56]}, metadata={op_type="Split" op_name="split_1"}
  %slice.360 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.132), slice={[0:3], [0:3], [0:4], [52:56]}, metadata={op_type="Split" op_name="split"}
  %convolution.384 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.328, f32[3,3,4,4]{3,2,1,0} %slice.360), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_13"}
  %slice.329 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.314), slice={[0:1], [0:58], [0:58], [56:60]}, metadata={op_type="Split" op_name="split_1"}
  %slice.361 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.132), slice={[0:3], [0:3], [0:4], [56:60]}, metadata={op_type="Split" op_name="split"}
  %convolution.385 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.329, f32[3,3,4,4]{3,2,1,0} %slice.361), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_14"}
  %slice.330 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.314), slice={[0:1], [0:58], [0:58], [60:64]}, metadata={op_type="Split" op_name="split_1"}
  %slice.362 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.132), slice={[0:3], [0:3], [0:4], [60:64]}, metadata={op_type="Split" op_name="split"}
  %convolution.386 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.330, f32[3,3,4,4]{3,2,1,0} %slice.362), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_15"}
  %slice.331 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.314), slice={[0:1], [0:58], [0:58], [64:68]}, metadata={op_type="Split" op_name="split_1"}
  %slice.363 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.132), slice={[0:3], [0:3], [0:4], [64:68]}, metadata={op_type="Split" op_name="split"}
  %convolution.387 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.331, f32[3,3,4,4]{3,2,1,0} %slice.363), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_16"}
  %slice.332 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.314), slice={[0:1], [0:58], [0:58], [68:72]}, metadata={op_type="Split" op_name="split_1"}
  %slice.364 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.132), slice={[0:3], [0:3], [0:4], [68:72]}, metadata={op_type="Split" op_name="split"}
  %convolution.388 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.332, f32[3,3,4,4]{3,2,1,0} %slice.364), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_17"}
  %slice.333 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.314), slice={[0:1], [0:58], [0:58], [72:76]}, metadata={op_type="Split" op_name="split_1"}
  %slice.365 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.132), slice={[0:3], [0:3], [0:4], [72:76]}, metadata={op_type="Split" op_name="split"}
  %convolution.389 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.333, f32[3,3,4,4]{3,2,1,0} %slice.365), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_18"}
  %slice.334 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.314), slice={[0:1], [0:58], [0:58], [76:80]}, metadata={op_type="Split" op_name="split_1"}
  %slice.366 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.132), slice={[0:3], [0:3], [0:4], [76:80]}, metadata={op_type="Split" op_name="split"}
  %convolution.390 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.334, f32[3,3,4,4]{3,2,1,0} %slice.366), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_19"}
  %slice.335 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.314), slice={[0:1], [0:58], [0:58], [80:84]}, metadata={op_type="Split" op_name="split_1"}
  %slice.367 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.132), slice={[0:3], [0:3], [0:4], [80:84]}, metadata={op_type="Split" op_name="split"}
  %convolution.392 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.335, f32[3,3,4,4]{3,2,1,0} %slice.367), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_20"}
  %slice.336 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.314), slice={[0:1], [0:58], [0:58], [84:88]}, metadata={op_type="Split" op_name="split_1"}
  %slice.368 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.132), slice={[0:3], [0:3], [0:4], [84:88]}, metadata={op_type="Split" op_name="split"}
  %convolution.393 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.336, f32[3,3,4,4]{3,2,1,0} %slice.368), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_21"}
  %slice.337 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.314), slice={[0:1], [0:58], [0:58], [88:92]}, metadata={op_type="Split" op_name="split_1"}
  %slice.369 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.132), slice={[0:3], [0:3], [0:4], [88:92]}, metadata={op_type="Split" op_name="split"}
  %convolution.394 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.337, f32[3,3,4,4]{3,2,1,0} %slice.369), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_22"}
  %slice.338 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.314), slice={[0:1], [0:58], [0:58], [92:96]}, metadata={op_type="Split" op_name="split_1"}
  %slice.370 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.132), slice={[0:3], [0:3], [0:4], [92:96]}, metadata={op_type="Split" op_name="split"}
  %convolution.395 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.338, f32[3,3,4,4]{3,2,1,0} %slice.370), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_23"}
  %slice.339 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.314), slice={[0:1], [0:58], [0:58], [96:100]}, metadata={op_type="Split" op_name="split_1"}
  %slice.371 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.132), slice={[0:3], [0:3], [0:4], [96:100]}, metadata={op_type="Split" op_name="split"}
  %convolution.396 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.339, f32[3,3,4,4]{3,2,1,0} %slice.371), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_24"}
  %slice.340 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.314), slice={[0:1], [0:58], [0:58], [100:104]}, metadata={op_type="Split" op_name="split_1"}
  %slice.372 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.132), slice={[0:3], [0:3], [0:4], [100:104]}, metadata={op_type="Split" op_name="split"}
  %convolution.397 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.340, f32[3,3,4,4]{3,2,1,0} %slice.372), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_25"}
  %slice.341 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.314), slice={[0:1], [0:58], [0:58], [104:108]}, metadata={op_type="Split" op_name="split_1"}
  %slice.373 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.132), slice={[0:3], [0:3], [0:4], [104:108]}, metadata={op_type="Split" op_name="split"}
  %convolution.398 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.341, f32[3,3,4,4]{3,2,1,0} %slice.373), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_26"}
  %slice.342 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.314), slice={[0:1], [0:58], [0:58], [108:112]}, metadata={op_type="Split" op_name="split_1"}
  %slice.374 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.132), slice={[0:3], [0:3], [0:4], [108:112]}, metadata={op_type="Split" op_name="split"}
  %convolution.399 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.342, f32[3,3,4,4]{3,2,1,0} %slice.374), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_27"}
  %slice.343 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.314), slice={[0:1], [0:58], [0:58], [112:116]}, metadata={op_type="Split" op_name="split_1"}
  %slice.375 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.132), slice={[0:3], [0:3], [0:4], [112:116]}, metadata={op_type="Split" op_name="split"}
  %convolution.400 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.343, f32[3,3,4,4]{3,2,1,0} %slice.375), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_28"}
  %slice.344 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.314), slice={[0:1], [0:58], [0:58], [116:120]}, metadata={op_type="Split" op_name="split_1"}
  %slice.376 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.132), slice={[0:3], [0:3], [0:4], [116:120]}, metadata={op_type="Split" op_name="split"}
  %convolution.401 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.344, f32[3,3,4,4]{3,2,1,0} %slice.376), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_29"}
  %slice.345 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.314), slice={[0:1], [0:58], [0:58], [120:124]}, metadata={op_type="Split" op_name="split_1"}
  %slice.377 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.132), slice={[0:3], [0:3], [0:4], [120:124]}, metadata={op_type="Split" op_name="split"}
  %convolution.403 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.345, f32[3,3,4,4]{3,2,1,0} %slice.377), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_30"}
  %slice.346 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.314), slice={[0:1], [0:58], [0:58], [124:128]}, metadata={op_type="Split" op_name="split_1"}
  %slice.378 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.132), slice={[0:3], [0:3], [0:4], [124:128]}, metadata={op_type="Split" op_name="split"}
  %convolution.404 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.346, f32[3,3,4,4]{3,2,1,0} %slice.378), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_31"}
  %concatenate.411 = f32[1,56,56,128]{3,2,1,0} concatenate(f32[1,56,56,4]{3,2,1,0} %convolution.379, f32[1,56,56,4]{3,2,1,0} %convolution.380, f32[1,56,56,4]{3,2,1,0} %convolution.391, f32[1,56,56,4]{3,2,1,0} %convolution.402, f32[1,56,56,4]{3,2,1,0} %convolution.405, f32[1,56,56,4]{3,2,1,0} %convolution.406, f32[1,56,56,4]{3,2,1,0} %convolution.407, f32[1,56,56,4]{3,2,1,0} %convolution.408, f32[1,56,56,4]{3,2,1,0} %convolution.409, f32[1,56,56,4]{3,2,1,0} %convolution.410, f32[1,56,56,4]{3,2,1,0} %convolution.381, f32[1,56,56,4]{3,2,1,0} %convolution.382, f32[1,56,56,4]{3,2,1,0} %convolution.383, f32[1,56,56,4]{3,2,1,0} %convolution.384, f32[1,56,56,4]{3,2,1,0} %convolution.385, f32[1,56,56,4]{3,2,1,0} %convolution.386, f32[1,56,56,4]{3,2,1,0} %convolution.387, f32[1,56,56,4]{3,2,1,0} %convolution.388, f32[1,56,56,4]{3,2,1,0} %convolution.389, f32[1,56,56,4]{3,2,1,0} %convolution.390, f32[1,56,56,4]{3,2,1,0} %convolution.392, f32[1,56,56,4]{3,2,1,0} %convolution.393, f32[1,56,56,4]{3,2,1,0} %convolution.394, f32[1,56,56,4]{3,2,1,0} %convolution.395, f32[1,56,56,4]{3,2,1,0} %convolution.396, f32[1,56,56,4]{3,2,1,0} %convolution.397, f32[1,56,56,4]{3,2,1,0} %convolution.398, f32[1,56,56,4]{3,2,1,0} %convolution.399, f32[1,56,56,4]{3,2,1,0} %convolution.400, f32[1,56,56,4]{3,2,1,0} %convolution.401, f32[1,56,56,4]{3,2,1,0} %convolution.403, f32[1,56,56,4]{3,2,1,0} %convolution.404), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat"}
  %constant.291 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit1_bn2/add"}
  %broadcast.292 = f32[128]{0} broadcast(f32[] %constant.291), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit1_bn2/add"}
  %arg8.9 = f32[128]{0} parameter(8), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.133 = f32[128]{0} reshape(f32[128]{0} %arg8.9)
  %add.293 = f32[128]{0} add(f32[128]{0} %broadcast.292, f32[128]{0} %reshape.133), metadata={op_type="AddV2" op_name="stage1_unit1_bn2/add"}
  %rsqrt.294 = f32[128]{0} rsqrt(f32[128]{0} %add.293), metadata={op_type="Rsqrt" op_name="stage1_unit1_bn2/Rsqrt"}
  %arg38.39 = f32[128]{0} parameter(38), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.163 = f32[128]{0} reshape(f32[128]{0} %arg38.39)
  %multiply.295 = f32[128]{0} multiply(f32[128]{0} %rsqrt.294, f32[128]{0} %reshape.163), metadata={op_type="Mul" op_name="stage1_unit1_bn2/mul"}
  %broadcast.412 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %multiply.295), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit1_bn2/mul_1"}
  %multiply.413 = f32[1,56,56,128]{3,2,1,0} multiply(f32[1,56,56,128]{3,2,1,0} %concatenate.411, f32[1,56,56,128]{3,2,1,0} %broadcast.412), metadata={op_type="Mul" op_name="stage1_unit1_bn2/mul_1"}
  %arg87.88 = f32[128]{0} parameter(87), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.212 = f32[128]{0} reshape(f32[128]{0} %arg87.88)
  %arg63.64 = f32[128]{0} parameter(63), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.188 = f32[128]{0} reshape(f32[128]{0} %arg63.64)
  %multiply.296 = f32[128]{0} multiply(f32[128]{0} %multiply.295, f32[128]{0} %reshape.188), metadata={op_type="Mul" op_name="stage1_unit1_bn2/mul_2"}
  %subtract.297 = f32[128]{0} subtract(f32[128]{0} %reshape.212, f32[128]{0} %multiply.296), metadata={op_type="Sub" op_name="stage1_unit1_bn2/sub"}
  %broadcast.414 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %subtract.297), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit1_bn2/add_1"}
  %add.415 = f32[1,56,56,128]{3,2,1,0} add(f32[1,56,56,128]{3,2,1,0} %multiply.413, f32[1,56,56,128]{3,2,1,0} %broadcast.414), metadata={op_type="AddV2" op_name="stage1_unit1_bn2/add_1"}
  %maximum.418 = f32[1,56,56,128]{3,2,1,0} maximum(f32[1,56,56,128]{3,2,1,0} %broadcast.417, f32[1,56,56,128]{3,2,1,0} %add.415), metadata={op_type="Relu" op_name="stage1_unit1_relu2"}
  %arg109.110 = f32[1,1,128,256]{3,2,1,0} parameter(109), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.234 = f32[1,1,128,256]{3,2,1,0} reshape(f32[1,1,128,256]{3,2,1,0} %arg109.110)
  %convolution.419 = f32[1,56,56,256]{3,2,1,0} convolution(f32[1,56,56,128]{3,2,1,0} %maximum.418, f32[1,1,128,256]{3,2,1,0} %reshape.234), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv3"}
  %multiply.421 = f32[1,56,56,256]{3,2,1,0} multiply(f32[1,56,56,256]{3,2,1,0} %broadcast.420, f32[1,56,56,256]{3,2,1,0} %convolution.419), metadata={op_type="Mul" op_name="stage1_unit1_bn3/mul_1"}
  %arg88.89 = f32[256]{0} parameter(88), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.213 = f32[256]{0} reshape(f32[256]{0} %arg88.89)
  %arg64.65 = f32[256]{0} parameter(64), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.189 = f32[256]{0} reshape(f32[256]{0} %arg64.65)
  %multiply.303 = f32[256]{0} multiply(f32[256]{0} %multiply.302, f32[256]{0} %reshape.189), metadata={op_type="Mul" op_name="stage1_unit1_bn3/mul_2"}
  %subtract.304 = f32[256]{0} subtract(f32[256]{0} %reshape.213, f32[256]{0} %multiply.303), metadata={op_type="Sub" op_name="stage1_unit1_bn3/sub"}
  %broadcast.422 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.304), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit1_bn3/add_1"}
  %add.423 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %multiply.421, f32[1,56,56,256]{3,2,1,0} %broadcast.422), metadata={op_type="AddV2" op_name="stage1_unit1_bn3/add_1"}
  %arg108.109 = f32[1,1,64,256]{3,2,1,0} parameter(108), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.233 = f32[1,1,64,256]{3,2,1,0} reshape(f32[1,1,64,256]{3,2,1,0} %arg108.109)
  %convolution.431 = f32[1,56,56,256]{3,2,1,0} convolution(f32[1,56,56,64]{3,2,1,0} %reduce-window.283, f32[1,1,64,256]{3,2,1,0} %reshape.233), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_sc"}
  %constant.424 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit1_sc_bn/add"}
  %broadcast.425 = f32[256]{0} broadcast(f32[] %constant.424), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit1_sc_bn/add"}
  %arg5.6 = f32[256]{0} parameter(5), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.130 = f32[256]{0} reshape(f32[256]{0} %arg5.6)
  %add.426 = f32[256]{0} add(f32[256]{0} %broadcast.425, f32[256]{0} %reshape.130), metadata={op_type="AddV2" op_name="stage1_unit1_sc_bn/add"}
  %rsqrt.427 = f32[256]{0} rsqrt(f32[256]{0} %add.426), metadata={op_type="Rsqrt" op_name="stage1_unit1_sc_bn/Rsqrt"}
  %arg36.37 = f32[256]{0} parameter(36), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.161 = f32[256]{0} reshape(f32[256]{0} %arg36.37)
  %multiply.428 = f32[256]{0} multiply(f32[256]{0} %rsqrt.427, f32[256]{0} %reshape.161), metadata={op_type="Mul" op_name="stage1_unit1_sc_bn/mul"}
  %broadcast.432 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.428), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit1_sc_bn/mul_1"}
  %multiply.433 = f32[1,56,56,256]{3,2,1,0} multiply(f32[1,56,56,256]{3,2,1,0} %convolution.431, f32[1,56,56,256]{3,2,1,0} %broadcast.432), metadata={op_type="Mul" op_name="stage1_unit1_sc_bn/mul_1"}
  %arg85.86 = f32[256]{0} parameter(85), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.210 = f32[256]{0} reshape(f32[256]{0} %arg85.86)
  %arg61.62 = f32[256]{0} parameter(61), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.186 = f32[256]{0} reshape(f32[256]{0} %arg61.62)
  %multiply.429 = f32[256]{0} multiply(f32[256]{0} %multiply.428, f32[256]{0} %reshape.186), metadata={op_type="Mul" op_name="stage1_unit1_sc_bn/mul_2"}
  %subtract.430 = f32[256]{0} subtract(f32[256]{0} %reshape.210, f32[256]{0} %multiply.429), metadata={op_type="Sub" op_name="stage1_unit1_sc_bn/sub"}
  %broadcast.434 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.430), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit1_sc_bn/add_1"}
  %add.435 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %multiply.433, f32[1,56,56,256]{3,2,1,0} %broadcast.434), metadata={op_type="AddV2" op_name="stage1_unit1_sc_bn/add_1"}
  %add.436 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %add.423, f32[1,56,56,256]{3,2,1,0} %add.435), metadata={op_type="AddV2" op_name="add"}
  %maximum.439 = f32[1,56,56,256]{3,2,1,0} maximum(f32[1,56,56,256]{3,2,1,0} %broadcast.438, f32[1,56,56,256]{3,2,1,0} %add.436), metadata={op_type="Relu" op_name="stage1_unit1_relu"}
  %constant.454 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit2_bn3/add"}
  %broadcast.455 = f32[256]{0} broadcast(f32[] %constant.454), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit2_bn3/add"}
  %arg14.15 = f32[256]{0} parameter(14), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.139 = f32[256]{0} reshape(f32[256]{0} %arg14.15)
  %add.456 = f32[256]{0} add(f32[256]{0} %broadcast.455, f32[256]{0} %reshape.139), metadata={op_type="AddV2" op_name="stage1_unit2_bn3/add"}
  %rsqrt.457 = f32[256]{0} rsqrt(f32[256]{0} %add.456), metadata={op_type="Rsqrt" op_name="stage1_unit2_bn3/Rsqrt"}
  %arg43.44 = f32[256]{0} parameter(43), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.168 = f32[256]{0} reshape(f32[256]{0} %arg43.44)
  %multiply.458 = f32[256]{0} multiply(f32[256]{0} %rsqrt.457, f32[256]{0} %reshape.168), metadata={op_type="Mul" op_name="stage1_unit2_bn3/mul"}
  %broadcast.576 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.458), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit2_bn3/mul_1"}
  %constant.572 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit2_relu2"}
  %broadcast.573 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[] %constant.572), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit2_relu2"}
  %constant.466 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit2_relu1"}
  %broadcast.467 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[] %constant.466), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit2_relu1"}
  %constant.440 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit2_bn1/add"}
  %broadcast.441 = f32[128]{0} broadcast(f32[] %constant.440), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit2_bn1/add"}
  %arg10.11 = f32[128]{0} parameter(10), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.135 = f32[128]{0} reshape(f32[128]{0} %arg10.11)
  %add.442 = f32[128]{0} add(f32[128]{0} %broadcast.441, f32[128]{0} %reshape.135), metadata={op_type="AddV2" op_name="stage1_unit2_bn1/add"}
  %rsqrt.443 = f32[128]{0} rsqrt(f32[128]{0} %add.442), metadata={op_type="Rsqrt" op_name="stage1_unit2_bn1/Rsqrt"}
  %arg40.41 = f32[128]{0} parameter(40), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.165 = f32[128]{0} reshape(f32[128]{0} %arg40.41)
  %multiply.444 = f32[128]{0} multiply(f32[128]{0} %rsqrt.443, f32[128]{0} %reshape.165), metadata={op_type="Mul" op_name="stage1_unit2_bn1/mul"}
  %broadcast.462 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %multiply.444), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit2_bn1/mul_1"}
  %arg110.111 = f32[1,1,256,128]{3,2,1,0} parameter(110), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.235 = f32[1,1,256,128]{3,2,1,0} reshape(f32[1,1,256,128]{3,2,1,0} %arg110.111)
  %convolution.461 = f32[1,56,56,128]{3,2,1,0} convolution(f32[1,56,56,256]{3,2,1,0} %maximum.439, f32[1,1,256,128]{3,2,1,0} %reshape.235), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv1"}
  %multiply.463 = f32[1,56,56,128]{3,2,1,0} multiply(f32[1,56,56,128]{3,2,1,0} %broadcast.462, f32[1,56,56,128]{3,2,1,0} %convolution.461), metadata={op_type="Mul" op_name="stage1_unit2_bn1/mul_1"}
  %arg89.90 = f32[128]{0} parameter(89), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.214 = f32[128]{0} reshape(f32[128]{0} %arg89.90)
  %arg65.66 = f32[128]{0} parameter(65), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.190 = f32[128]{0} reshape(f32[128]{0} %arg65.66)
  %multiply.445 = f32[128]{0} multiply(f32[128]{0} %multiply.444, f32[128]{0} %reshape.190), metadata={op_type="Mul" op_name="stage1_unit2_bn1/mul_2"}
  %subtract.446 = f32[128]{0} subtract(f32[128]{0} %reshape.214, f32[128]{0} %multiply.445), metadata={op_type="Sub" op_name="stage1_unit2_bn1/sub"}
  %broadcast.464 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %subtract.446), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit2_bn1/add_1"}
  %add.465 = f32[1,56,56,128]{3,2,1,0} add(f32[1,56,56,128]{3,2,1,0} %multiply.463, f32[1,56,56,128]{3,2,1,0} %broadcast.464), metadata={op_type="AddV2" op_name="stage1_unit2_bn1/add_1"}
  %maximum.468 = f32[1,56,56,128]{3,2,1,0} maximum(f32[1,56,56,128]{3,2,1,0} %broadcast.467, f32[1,56,56,128]{3,2,1,0} %add.465), metadata={op_type="Relu" op_name="stage1_unit2_relu1"}
  %constant.469 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_2"}
  %pad.470 = f32[1,58,58,128]{3,2,1,0} pad(f32[1,56,56,128]{3,2,1,0} %maximum.468, f32[] %constant.469), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_2"}
  %slice.471 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.470), slice={[0:1], [0:58], [0:58], [0:4]}, metadata={op_type="Split" op_name="split_3"}
  %arg11.12 = f32[3,3,4,128]{3,2,1,0} parameter(11), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.136 = f32[3,3,4,128]{3,2,1,0} reshape(f32[3,3,4,128]{3,2,1,0} %arg11.12)
  %slice.503 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.136), slice={[0:3], [0:3], [0:4], [0:4]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.535 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.471, f32[3,3,4,4]{3,2,1,0} %slice.503), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2"}
  %slice.472 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.470), slice={[0:1], [0:58], [0:58], [4:8]}, metadata={op_type="Split" op_name="split_3"}
  %slice.504 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.136), slice={[0:3], [0:3], [0:4], [4:8]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.536 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.472, f32[3,3,4,4]{3,2,1,0} %slice.504), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_1"}
  %slice.473 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.470), slice={[0:1], [0:58], [0:58], [8:12]}, metadata={op_type="Split" op_name="split_3"}
  %slice.505 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.136), slice={[0:3], [0:3], [0:4], [8:12]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.547 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.473, f32[3,3,4,4]{3,2,1,0} %slice.505), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_2"}
  %slice.474 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.470), slice={[0:1], [0:58], [0:58], [12:16]}, metadata={op_type="Split" op_name="split_3"}
  %slice.506 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.136), slice={[0:3], [0:3], [0:4], [12:16]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.558 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.474, f32[3,3,4,4]{3,2,1,0} %slice.506), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_3"}
  %slice.475 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.470), slice={[0:1], [0:58], [0:58], [16:20]}, metadata={op_type="Split" op_name="split_3"}
  %slice.507 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.136), slice={[0:3], [0:3], [0:4], [16:20]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.561 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.475, f32[3,3,4,4]{3,2,1,0} %slice.507), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_4"}
  %slice.476 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.470), slice={[0:1], [0:58], [0:58], [20:24]}, metadata={op_type="Split" op_name="split_3"}
  %slice.508 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.136), slice={[0:3], [0:3], [0:4], [20:24]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.562 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.476, f32[3,3,4,4]{3,2,1,0} %slice.508), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_5"}
  %slice.477 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.470), slice={[0:1], [0:58], [0:58], [24:28]}, metadata={op_type="Split" op_name="split_3"}
  %slice.509 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.136), slice={[0:3], [0:3], [0:4], [24:28]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.563 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.477, f32[3,3,4,4]{3,2,1,0} %slice.509), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_6"}
  %slice.478 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.470), slice={[0:1], [0:58], [0:58], [28:32]}, metadata={op_type="Split" op_name="split_3"}
  %slice.510 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.136), slice={[0:3], [0:3], [0:4], [28:32]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.564 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.478, f32[3,3,4,4]{3,2,1,0} %slice.510), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_7"}
  %slice.479 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.470), slice={[0:1], [0:58], [0:58], [32:36]}, metadata={op_type="Split" op_name="split_3"}
  %slice.511 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.136), slice={[0:3], [0:3], [0:4], [32:36]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.565 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.479, f32[3,3,4,4]{3,2,1,0} %slice.511), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_8"}
  %slice.480 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.470), slice={[0:1], [0:58], [0:58], [36:40]}, metadata={op_type="Split" op_name="split_3"}
  %slice.512 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.136), slice={[0:3], [0:3], [0:4], [36:40]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.566 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.480, f32[3,3,4,4]{3,2,1,0} %slice.512), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_9"}
  %slice.481 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.470), slice={[0:1], [0:58], [0:58], [40:44]}, metadata={op_type="Split" op_name="split_3"}
  %slice.513 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.136), slice={[0:3], [0:3], [0:4], [40:44]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.537 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.481, f32[3,3,4,4]{3,2,1,0} %slice.513), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_10"}
  %slice.482 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.470), slice={[0:1], [0:58], [0:58], [44:48]}, metadata={op_type="Split" op_name="split_3"}
  %slice.514 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.136), slice={[0:3], [0:3], [0:4], [44:48]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.538 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.482, f32[3,3,4,4]{3,2,1,0} %slice.514), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_11"}
  %slice.483 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.470), slice={[0:1], [0:58], [0:58], [48:52]}, metadata={op_type="Split" op_name="split_3"}
  %slice.515 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.136), slice={[0:3], [0:3], [0:4], [48:52]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.539 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.483, f32[3,3,4,4]{3,2,1,0} %slice.515), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_12"}
  %slice.484 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.470), slice={[0:1], [0:58], [0:58], [52:56]}, metadata={op_type="Split" op_name="split_3"}
  %slice.516 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.136), slice={[0:3], [0:3], [0:4], [52:56]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.540 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.484, f32[3,3,4,4]{3,2,1,0} %slice.516), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_13"}
  %slice.485 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.470), slice={[0:1], [0:58], [0:58], [56:60]}, metadata={op_type="Split" op_name="split_3"}
  %slice.517 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.136), slice={[0:3], [0:3], [0:4], [56:60]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.541 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.485, f32[3,3,4,4]{3,2,1,0} %slice.517), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_14"}
  %slice.486 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.470), slice={[0:1], [0:58], [0:58], [60:64]}, metadata={op_type="Split" op_name="split_3"}
  %slice.518 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.136), slice={[0:3], [0:3], [0:4], [60:64]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.542 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.486, f32[3,3,4,4]{3,2,1,0} %slice.518), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_15"}
  %slice.487 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.470), slice={[0:1], [0:58], [0:58], [64:68]}, metadata={op_type="Split" op_name="split_3"}
  %slice.519 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.136), slice={[0:3], [0:3], [0:4], [64:68]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.543 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.487, f32[3,3,4,4]{3,2,1,0} %slice.519), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_16"}
  %slice.488 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.470), slice={[0:1], [0:58], [0:58], [68:72]}, metadata={op_type="Split" op_name="split_3"}
  %slice.520 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.136), slice={[0:3], [0:3], [0:4], [68:72]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.544 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.488, f32[3,3,4,4]{3,2,1,0} %slice.520), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_17"}
  %slice.489 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.470), slice={[0:1], [0:58], [0:58], [72:76]}, metadata={op_type="Split" op_name="split_3"}
  %slice.521 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.136), slice={[0:3], [0:3], [0:4], [72:76]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.545 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.489, f32[3,3,4,4]{3,2,1,0} %slice.521), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_18"}
  %slice.490 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.470), slice={[0:1], [0:58], [0:58], [76:80]}, metadata={op_type="Split" op_name="split_3"}
  %slice.522 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.136), slice={[0:3], [0:3], [0:4], [76:80]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.546 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.490, f32[3,3,4,4]{3,2,1,0} %slice.522), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_19"}
  %slice.491 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.470), slice={[0:1], [0:58], [0:58], [80:84]}, metadata={op_type="Split" op_name="split_3"}
  %slice.523 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.136), slice={[0:3], [0:3], [0:4], [80:84]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.548 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.491, f32[3,3,4,4]{3,2,1,0} %slice.523), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_20"}
  %slice.492 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.470), slice={[0:1], [0:58], [0:58], [84:88]}, metadata={op_type="Split" op_name="split_3"}
  %slice.524 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.136), slice={[0:3], [0:3], [0:4], [84:88]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.549 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.492, f32[3,3,4,4]{3,2,1,0} %slice.524), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_21"}
  %slice.493 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.470), slice={[0:1], [0:58], [0:58], [88:92]}, metadata={op_type="Split" op_name="split_3"}
  %slice.525 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.136), slice={[0:3], [0:3], [0:4], [88:92]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.550 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.493, f32[3,3,4,4]{3,2,1,0} %slice.525), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_22"}
  %slice.494 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.470), slice={[0:1], [0:58], [0:58], [92:96]}, metadata={op_type="Split" op_name="split_3"}
  %slice.526 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.136), slice={[0:3], [0:3], [0:4], [92:96]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.551 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.494, f32[3,3,4,4]{3,2,1,0} %slice.526), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_23"}
  %slice.495 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.470), slice={[0:1], [0:58], [0:58], [96:100]}, metadata={op_type="Split" op_name="split_3"}
  %slice.527 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.136), slice={[0:3], [0:3], [0:4], [96:100]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.552 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.495, f32[3,3,4,4]{3,2,1,0} %slice.527), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_24"}
  %slice.496 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.470), slice={[0:1], [0:58], [0:58], [100:104]}, metadata={op_type="Split" op_name="split_3"}
  %slice.528 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.136), slice={[0:3], [0:3], [0:4], [100:104]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.553 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.496, f32[3,3,4,4]{3,2,1,0} %slice.528), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_25"}
  %slice.497 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.470), slice={[0:1], [0:58], [0:58], [104:108]}, metadata={op_type="Split" op_name="split_3"}
  %slice.529 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.136), slice={[0:3], [0:3], [0:4], [104:108]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.554 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.497, f32[3,3,4,4]{3,2,1,0} %slice.529), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_26"}
  %slice.498 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.470), slice={[0:1], [0:58], [0:58], [108:112]}, metadata={op_type="Split" op_name="split_3"}
  %slice.530 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.136), slice={[0:3], [0:3], [0:4], [108:112]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.555 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.498, f32[3,3,4,4]{3,2,1,0} %slice.530), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_27"}
  %slice.499 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.470), slice={[0:1], [0:58], [0:58], [112:116]}, metadata={op_type="Split" op_name="split_3"}
  %slice.531 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.136), slice={[0:3], [0:3], [0:4], [112:116]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.556 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.499, f32[3,3,4,4]{3,2,1,0} %slice.531), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_28"}
  %slice.500 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.470), slice={[0:1], [0:58], [0:58], [116:120]}, metadata={op_type="Split" op_name="split_3"}
  %slice.532 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.136), slice={[0:3], [0:3], [0:4], [116:120]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.557 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.500, f32[3,3,4,4]{3,2,1,0} %slice.532), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_29"}
  %slice.501 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.470), slice={[0:1], [0:58], [0:58], [120:124]}, metadata={op_type="Split" op_name="split_3"}
  %slice.533 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.136), slice={[0:3], [0:3], [0:4], [120:124]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.559 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.501, f32[3,3,4,4]{3,2,1,0} %slice.533), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_30"}
  %slice.502 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.470), slice={[0:1], [0:58], [0:58], [124:128]}, metadata={op_type="Split" op_name="split_3"}
  %slice.534 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.136), slice={[0:3], [0:3], [0:4], [124:128]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.560 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.502, f32[3,3,4,4]{3,2,1,0} %slice.534), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_31"}
  %concatenate.567 = f32[1,56,56,128]{3,2,1,0} concatenate(f32[1,56,56,4]{3,2,1,0} %convolution.535, f32[1,56,56,4]{3,2,1,0} %convolution.536, f32[1,56,56,4]{3,2,1,0} %convolution.547, f32[1,56,56,4]{3,2,1,0} %convolution.558, f32[1,56,56,4]{3,2,1,0} %convolution.561, f32[1,56,56,4]{3,2,1,0} %convolution.562, f32[1,56,56,4]{3,2,1,0} %convolution.563, f32[1,56,56,4]{3,2,1,0} %convolution.564, f32[1,56,56,4]{3,2,1,0} %convolution.565, f32[1,56,56,4]{3,2,1,0} %convolution.566, f32[1,56,56,4]{3,2,1,0} %convolution.537, f32[1,56,56,4]{3,2,1,0} %convolution.538, f32[1,56,56,4]{3,2,1,0} %convolution.539, f32[1,56,56,4]{3,2,1,0} %convolution.540, f32[1,56,56,4]{3,2,1,0} %convolution.541, f32[1,56,56,4]{3,2,1,0} %convolution.542, f32[1,56,56,4]{3,2,1,0} %convolution.543, f32[1,56,56,4]{3,2,1,0} %convolution.544, f32[1,56,56,4]{3,2,1,0} %convolution.545, f32[1,56,56,4]{3,2,1,0} %convolution.546, f32[1,56,56,4]{3,2,1,0} %convolution.548, f32[1,56,56,4]{3,2,1,0} %convolution.549, f32[1,56,56,4]{3,2,1,0} %convolution.550, f32[1,56,56,4]{3,2,1,0} %convolution.551, f32[1,56,56,4]{3,2,1,0} %convolution.552, f32[1,56,56,4]{3,2,1,0} %convolution.553, f32[1,56,56,4]{3,2,1,0} %convolution.554, f32[1,56,56,4]{3,2,1,0} %convolution.555, f32[1,56,56,4]{3,2,1,0} %convolution.556, f32[1,56,56,4]{3,2,1,0} %convolution.557, f32[1,56,56,4]{3,2,1,0} %convolution.559, f32[1,56,56,4]{3,2,1,0} %convolution.560), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_1"}
  %constant.447 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit2_bn2/add"}
  %broadcast.448 = f32[128]{0} broadcast(f32[] %constant.447), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit2_bn2/add"}
  %arg12.13 = f32[128]{0} parameter(12), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.137 = f32[128]{0} reshape(f32[128]{0} %arg12.13)
  %add.449 = f32[128]{0} add(f32[128]{0} %broadcast.448, f32[128]{0} %reshape.137), metadata={op_type="AddV2" op_name="stage1_unit2_bn2/add"}
  %rsqrt.450 = f32[128]{0} rsqrt(f32[128]{0} %add.449), metadata={op_type="Rsqrt" op_name="stage1_unit2_bn2/Rsqrt"}
  %arg41.42 = f32[128]{0} parameter(41), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.166 = f32[128]{0} reshape(f32[128]{0} %arg41.42)
  %multiply.451 = f32[128]{0} multiply(f32[128]{0} %rsqrt.450, f32[128]{0} %reshape.166), metadata={op_type="Mul" op_name="stage1_unit2_bn2/mul"}
  %broadcast.568 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %multiply.451), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit2_bn2/mul_1"}
  %multiply.569 = f32[1,56,56,128]{3,2,1,0} multiply(f32[1,56,56,128]{3,2,1,0} %concatenate.567, f32[1,56,56,128]{3,2,1,0} %broadcast.568), metadata={op_type="Mul" op_name="stage1_unit2_bn2/mul_1"}
  %arg90.91 = f32[128]{0} parameter(90), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.215 = f32[128]{0} reshape(f32[128]{0} %arg90.91)
  %arg66.67 = f32[128]{0} parameter(66), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.191 = f32[128]{0} reshape(f32[128]{0} %arg66.67)
  %multiply.452 = f32[128]{0} multiply(f32[128]{0} %multiply.451, f32[128]{0} %reshape.191), metadata={op_type="Mul" op_name="stage1_unit2_bn2/mul_2"}
  %subtract.453 = f32[128]{0} subtract(f32[128]{0} %reshape.215, f32[128]{0} %multiply.452), metadata={op_type="Sub" op_name="stage1_unit2_bn2/sub"}
  %broadcast.570 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %subtract.453), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit2_bn2/add_1"}
  %add.571 = f32[1,56,56,128]{3,2,1,0} add(f32[1,56,56,128]{3,2,1,0} %multiply.569, f32[1,56,56,128]{3,2,1,0} %broadcast.570), metadata={op_type="AddV2" op_name="stage1_unit2_bn2/add_1"}
  %maximum.574 = f32[1,56,56,128]{3,2,1,0} maximum(f32[1,56,56,128]{3,2,1,0} %broadcast.573, f32[1,56,56,128]{3,2,1,0} %add.571), metadata={op_type="Relu" op_name="stage1_unit2_relu2"}
  %arg111.112 = f32[1,1,128,256]{3,2,1,0} parameter(111), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.236 = f32[1,1,128,256]{3,2,1,0} reshape(f32[1,1,128,256]{3,2,1,0} %arg111.112)
  %convolution.575 = f32[1,56,56,256]{3,2,1,0} convolution(f32[1,56,56,128]{3,2,1,0} %maximum.574, f32[1,1,128,256]{3,2,1,0} %reshape.236), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv3"}
  %multiply.577 = f32[1,56,56,256]{3,2,1,0} multiply(f32[1,56,56,256]{3,2,1,0} %broadcast.576, f32[1,56,56,256]{3,2,1,0} %convolution.575), metadata={op_type="Mul" op_name="stage1_unit2_bn3/mul_1"}
  %arg92.93 = f32[256]{0} parameter(92), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.217 = f32[256]{0} reshape(f32[256]{0} %arg92.93)
  %arg68.69 = f32[256]{0} parameter(68), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.193 = f32[256]{0} reshape(f32[256]{0} %arg68.69)
  %multiply.459 = f32[256]{0} multiply(f32[256]{0} %multiply.458, f32[256]{0} %reshape.193), metadata={op_type="Mul" op_name="stage1_unit2_bn3/mul_2"}
  %subtract.460 = f32[256]{0} subtract(f32[256]{0} %reshape.217, f32[256]{0} %multiply.459), metadata={op_type="Sub" op_name="stage1_unit2_bn3/sub"}
  %broadcast.578 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.460), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit2_bn3/add_1"}
  %add.579 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %multiply.577, f32[1,56,56,256]{3,2,1,0} %broadcast.578), metadata={op_type="AddV2" op_name="stage1_unit2_bn3/add_1"}
  %add.580 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %maximum.439, f32[1,56,56,256]{3,2,1,0} %add.579), metadata={op_type="AddV2" op_name="add_1"}
  %maximum.583 = f32[1,56,56,256]{3,2,1,0} maximum(f32[1,56,56,256]{3,2,1,0} %broadcast.582, f32[1,56,56,256]{3,2,1,0} %add.580), metadata={op_type="Relu" op_name="stage1_unit2_relu"}
  %constant.598 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit3_bn3/add"}
  %broadcast.599 = f32[256]{0} broadcast(f32[] %constant.598), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit3_bn3/add"}
  %arg19.20 = f32[256]{0} parameter(19), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.144 = f32[256]{0} reshape(f32[256]{0} %arg19.20)
  %add.600 = f32[256]{0} add(f32[256]{0} %broadcast.599, f32[256]{0} %reshape.144), metadata={op_type="AddV2" op_name="stage1_unit3_bn3/add"}
  %rsqrt.601 = f32[256]{0} rsqrt(f32[256]{0} %add.600), metadata={op_type="Rsqrt" op_name="stage1_unit3_bn3/Rsqrt"}
  %arg47.48 = f32[256]{0} parameter(47), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.172 = f32[256]{0} reshape(f32[256]{0} %arg47.48)
  %multiply.602 = f32[256]{0} multiply(f32[256]{0} %rsqrt.601, f32[256]{0} %reshape.172), metadata={op_type="Mul" op_name="stage1_unit3_bn3/mul"}
  %broadcast.720 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.602), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit3_bn3/mul_1"}
  %constant.716 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit3_relu2"}
  %broadcast.717 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[] %constant.716), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit3_relu2"}
  %constant.610 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit3_relu1"}
  %broadcast.611 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[] %constant.610), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit3_relu1"}
  %constant.584 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit3_bn1/add"}
  %broadcast.585 = f32[128]{0} broadcast(f32[] %constant.584), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit3_bn1/add"}
  %arg15.16 = f32[128]{0} parameter(15), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.140 = f32[128]{0} reshape(f32[128]{0} %arg15.16)
  %add.586 = f32[128]{0} add(f32[128]{0} %broadcast.585, f32[128]{0} %reshape.140), metadata={op_type="AddV2" op_name="stage1_unit3_bn1/add"}
  %rsqrt.587 = f32[128]{0} rsqrt(f32[128]{0} %add.586), metadata={op_type="Rsqrt" op_name="stage1_unit3_bn1/Rsqrt"}
  %arg44.45 = f32[128]{0} parameter(44), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.169 = f32[128]{0} reshape(f32[128]{0} %arg44.45)
  %multiply.588 = f32[128]{0} multiply(f32[128]{0} %rsqrt.587, f32[128]{0} %reshape.169), metadata={op_type="Mul" op_name="stage1_unit3_bn1/mul"}
  %broadcast.606 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %multiply.588), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit3_bn1/mul_1"}
  %arg112.113 = f32[1,1,256,128]{3,2,1,0} parameter(112), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.237 = f32[1,1,256,128]{3,2,1,0} reshape(f32[1,1,256,128]{3,2,1,0} %arg112.113)
  %convolution.605 = f32[1,56,56,128]{3,2,1,0} convolution(f32[1,56,56,256]{3,2,1,0} %maximum.583, f32[1,1,256,128]{3,2,1,0} %reshape.237), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv1"}
  %multiply.607 = f32[1,56,56,128]{3,2,1,0} multiply(f32[1,56,56,128]{3,2,1,0} %broadcast.606, f32[1,56,56,128]{3,2,1,0} %convolution.605), metadata={op_type="Mul" op_name="stage1_unit3_bn1/mul_1"}
  %arg93.94 = f32[128]{0} parameter(93), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.218 = f32[128]{0} reshape(f32[128]{0} %arg93.94)
  %arg69.70 = f32[128]{0} parameter(69), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.194 = f32[128]{0} reshape(f32[128]{0} %arg69.70)
  %multiply.589 = f32[128]{0} multiply(f32[128]{0} %multiply.588, f32[128]{0} %reshape.194), metadata={op_type="Mul" op_name="stage1_unit3_bn1/mul_2"}
  %subtract.590 = f32[128]{0} subtract(f32[128]{0} %reshape.218, f32[128]{0} %multiply.589), metadata={op_type="Sub" op_name="stage1_unit3_bn1/sub"}
  %broadcast.608 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %subtract.590), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit3_bn1/add_1"}
  %add.609 = f32[1,56,56,128]{3,2,1,0} add(f32[1,56,56,128]{3,2,1,0} %multiply.607, f32[1,56,56,128]{3,2,1,0} %broadcast.608), metadata={op_type="AddV2" op_name="stage1_unit3_bn1/add_1"}
  %maximum.612 = f32[1,56,56,128]{3,2,1,0} maximum(f32[1,56,56,128]{3,2,1,0} %broadcast.611, f32[1,56,56,128]{3,2,1,0} %add.609), metadata={op_type="Relu" op_name="stage1_unit3_relu1"}
  %constant.613 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_3"}
  %pad.614 = f32[1,58,58,128]{3,2,1,0} pad(f32[1,56,56,128]{3,2,1,0} %maximum.612, f32[] %constant.613), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_3"}
  %slice.615 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.614), slice={[0:1], [0:58], [0:58], [0:4]}, metadata={op_type="Split" op_name="split_5"}
  %arg16.17 = f32[3,3,4,128]{3,2,1,0} parameter(16), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.141 = f32[3,3,4,128]{3,2,1,0} reshape(f32[3,3,4,128]{3,2,1,0} %arg16.17)
  %slice.647 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.141), slice={[0:3], [0:3], [0:4], [0:4]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.679 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.615, f32[3,3,4,4]{3,2,1,0} %slice.647), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2"}
  %slice.616 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.614), slice={[0:1], [0:58], [0:58], [4:8]}, metadata={op_type="Split" op_name="split_5"}
  %slice.648 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.141), slice={[0:3], [0:3], [0:4], [4:8]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.680 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.616, f32[3,3,4,4]{3,2,1,0} %slice.648), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_1"}
  %slice.617 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.614), slice={[0:1], [0:58], [0:58], [8:12]}, metadata={op_type="Split" op_name="split_5"}
  %slice.649 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.141), slice={[0:3], [0:3], [0:4], [8:12]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.691 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.617, f32[3,3,4,4]{3,2,1,0} %slice.649), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_2"}
  %slice.618 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.614), slice={[0:1], [0:58], [0:58], [12:16]}, metadata={op_type="Split" op_name="split_5"}
  %slice.650 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.141), slice={[0:3], [0:3], [0:4], [12:16]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.702 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.618, f32[3,3,4,4]{3,2,1,0} %slice.650), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_3"}
  %slice.619 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.614), slice={[0:1], [0:58], [0:58], [16:20]}, metadata={op_type="Split" op_name="split_5"}
  %slice.651 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.141), slice={[0:3], [0:3], [0:4], [16:20]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.705 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.619, f32[3,3,4,4]{3,2,1,0} %slice.651), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_4"}
  %slice.620 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.614), slice={[0:1], [0:58], [0:58], [20:24]}, metadata={op_type="Split" op_name="split_5"}
  %slice.652 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.141), slice={[0:3], [0:3], [0:4], [20:24]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.706 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.620, f32[3,3,4,4]{3,2,1,0} %slice.652), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_5"}
  %slice.621 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.614), slice={[0:1], [0:58], [0:58], [24:28]}, metadata={op_type="Split" op_name="split_5"}
  %slice.653 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.141), slice={[0:3], [0:3], [0:4], [24:28]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.707 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.621, f32[3,3,4,4]{3,2,1,0} %slice.653), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_6"}
  %slice.622 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.614), slice={[0:1], [0:58], [0:58], [28:32]}, metadata={op_type="Split" op_name="split_5"}
  %slice.654 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.141), slice={[0:3], [0:3], [0:4], [28:32]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.708 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.622, f32[3,3,4,4]{3,2,1,0} %slice.654), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_7"}
  %slice.623 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.614), slice={[0:1], [0:58], [0:58], [32:36]}, metadata={op_type="Split" op_name="split_5"}
  %slice.655 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.141), slice={[0:3], [0:3], [0:4], [32:36]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.709 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.623, f32[3,3,4,4]{3,2,1,0} %slice.655), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_8"}
  %slice.624 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.614), slice={[0:1], [0:58], [0:58], [36:40]}, metadata={op_type="Split" op_name="split_5"}
  %slice.656 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.141), slice={[0:3], [0:3], [0:4], [36:40]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.710 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.624, f32[3,3,4,4]{3,2,1,0} %slice.656), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_9"}
  %slice.625 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.614), slice={[0:1], [0:58], [0:58], [40:44]}, metadata={op_type="Split" op_name="split_5"}
  %slice.657 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.141), slice={[0:3], [0:3], [0:4], [40:44]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.681 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.625, f32[3,3,4,4]{3,2,1,0} %slice.657), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_10"}
  %slice.626 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.614), slice={[0:1], [0:58], [0:58], [44:48]}, metadata={op_type="Split" op_name="split_5"}
  %slice.658 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.141), slice={[0:3], [0:3], [0:4], [44:48]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.682 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.626, f32[3,3,4,4]{3,2,1,0} %slice.658), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_11"}
  %slice.627 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.614), slice={[0:1], [0:58], [0:58], [48:52]}, metadata={op_type="Split" op_name="split_5"}
  %slice.659 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.141), slice={[0:3], [0:3], [0:4], [48:52]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.683 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.627, f32[3,3,4,4]{3,2,1,0} %slice.659), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_12"}
  %slice.628 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.614), slice={[0:1], [0:58], [0:58], [52:56]}, metadata={op_type="Split" op_name="split_5"}
  %slice.660 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.141), slice={[0:3], [0:3], [0:4], [52:56]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.684 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.628, f32[3,3,4,4]{3,2,1,0} %slice.660), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_13"}
  %slice.629 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.614), slice={[0:1], [0:58], [0:58], [56:60]}, metadata={op_type="Split" op_name="split_5"}
  %slice.661 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.141), slice={[0:3], [0:3], [0:4], [56:60]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.685 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.629, f32[3,3,4,4]{3,2,1,0} %slice.661), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_14"}
  %slice.630 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.614), slice={[0:1], [0:58], [0:58], [60:64]}, metadata={op_type="Split" op_name="split_5"}
  %slice.662 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.141), slice={[0:3], [0:3], [0:4], [60:64]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.686 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.630, f32[3,3,4,4]{3,2,1,0} %slice.662), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_15"}
  %slice.631 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.614), slice={[0:1], [0:58], [0:58], [64:68]}, metadata={op_type="Split" op_name="split_5"}
  %slice.663 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.141), slice={[0:3], [0:3], [0:4], [64:68]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.687 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.631, f32[3,3,4,4]{3,2,1,0} %slice.663), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_16"}
  %slice.632 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.614), slice={[0:1], [0:58], [0:58], [68:72]}, metadata={op_type="Split" op_name="split_5"}
  %slice.664 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.141), slice={[0:3], [0:3], [0:4], [68:72]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.688 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.632, f32[3,3,4,4]{3,2,1,0} %slice.664), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_17"}
  %slice.633 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.614), slice={[0:1], [0:58], [0:58], [72:76]}, metadata={op_type="Split" op_name="split_5"}
  %slice.665 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.141), slice={[0:3], [0:3], [0:4], [72:76]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.689 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.633, f32[3,3,4,4]{3,2,1,0} %slice.665), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_18"}
  %slice.634 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.614), slice={[0:1], [0:58], [0:58], [76:80]}, metadata={op_type="Split" op_name="split_5"}
  %slice.666 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.141), slice={[0:3], [0:3], [0:4], [76:80]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.690 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.634, f32[3,3,4,4]{3,2,1,0} %slice.666), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_19"}
  %slice.635 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.614), slice={[0:1], [0:58], [0:58], [80:84]}, metadata={op_type="Split" op_name="split_5"}
  %slice.667 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.141), slice={[0:3], [0:3], [0:4], [80:84]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.692 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.635, f32[3,3,4,4]{3,2,1,0} %slice.667), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_20"}
  %slice.636 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.614), slice={[0:1], [0:58], [0:58], [84:88]}, metadata={op_type="Split" op_name="split_5"}
  %slice.668 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.141), slice={[0:3], [0:3], [0:4], [84:88]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.693 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.636, f32[3,3,4,4]{3,2,1,0} %slice.668), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_21"}
  %slice.637 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.614), slice={[0:1], [0:58], [0:58], [88:92]}, metadata={op_type="Split" op_name="split_5"}
  %slice.669 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.141), slice={[0:3], [0:3], [0:4], [88:92]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.694 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.637, f32[3,3,4,4]{3,2,1,0} %slice.669), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_22"}
  %slice.638 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.614), slice={[0:1], [0:58], [0:58], [92:96]}, metadata={op_type="Split" op_name="split_5"}
  %slice.670 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.141), slice={[0:3], [0:3], [0:4], [92:96]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.695 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.638, f32[3,3,4,4]{3,2,1,0} %slice.670), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_23"}
  %slice.639 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.614), slice={[0:1], [0:58], [0:58], [96:100]}, metadata={op_type="Split" op_name="split_5"}
  %slice.671 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.141), slice={[0:3], [0:3], [0:4], [96:100]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.696 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.639, f32[3,3,4,4]{3,2,1,0} %slice.671), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_24"}
  %slice.640 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.614), slice={[0:1], [0:58], [0:58], [100:104]}, metadata={op_type="Split" op_name="split_5"}
  %slice.672 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.141), slice={[0:3], [0:3], [0:4], [100:104]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.697 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.640, f32[3,3,4,4]{3,2,1,0} %slice.672), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_25"}
  %slice.641 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.614), slice={[0:1], [0:58], [0:58], [104:108]}, metadata={op_type="Split" op_name="split_5"}
  %slice.673 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.141), slice={[0:3], [0:3], [0:4], [104:108]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.698 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.641, f32[3,3,4,4]{3,2,1,0} %slice.673), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_26"}
  %slice.642 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.614), slice={[0:1], [0:58], [0:58], [108:112]}, metadata={op_type="Split" op_name="split_5"}
  %slice.674 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.141), slice={[0:3], [0:3], [0:4], [108:112]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.699 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.642, f32[3,3,4,4]{3,2,1,0} %slice.674), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_27"}
  %slice.643 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.614), slice={[0:1], [0:58], [0:58], [112:116]}, metadata={op_type="Split" op_name="split_5"}
  %slice.675 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.141), slice={[0:3], [0:3], [0:4], [112:116]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.700 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.643, f32[3,3,4,4]{3,2,1,0} %slice.675), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_28"}
  %slice.644 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.614), slice={[0:1], [0:58], [0:58], [116:120]}, metadata={op_type="Split" op_name="split_5"}
  %slice.676 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.141), slice={[0:3], [0:3], [0:4], [116:120]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.701 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.644, f32[3,3,4,4]{3,2,1,0} %slice.676), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_29"}
  %slice.645 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.614), slice={[0:1], [0:58], [0:58], [120:124]}, metadata={op_type="Split" op_name="split_5"}
  %slice.677 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.141), slice={[0:3], [0:3], [0:4], [120:124]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.703 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.645, f32[3,3,4,4]{3,2,1,0} %slice.677), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_30"}
  %slice.646 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.614), slice={[0:1], [0:58], [0:58], [124:128]}, metadata={op_type="Split" op_name="split_5"}
  %slice.678 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.141), slice={[0:3], [0:3], [0:4], [124:128]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.704 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.646, f32[3,3,4,4]{3,2,1,0} %slice.678), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_31"}
  %concatenate.711 = f32[1,56,56,128]{3,2,1,0} concatenate(f32[1,56,56,4]{3,2,1,0} %convolution.679, f32[1,56,56,4]{3,2,1,0} %convolution.680, f32[1,56,56,4]{3,2,1,0} %convolution.691, f32[1,56,56,4]{3,2,1,0} %convolution.702, f32[1,56,56,4]{3,2,1,0} %convolution.705, f32[1,56,56,4]{3,2,1,0} %convolution.706, f32[1,56,56,4]{3,2,1,0} %convolution.707, f32[1,56,56,4]{3,2,1,0} %convolution.708, f32[1,56,56,4]{3,2,1,0} %convolution.709, f32[1,56,56,4]{3,2,1,0} %convolution.710, f32[1,56,56,4]{3,2,1,0} %convolution.681, f32[1,56,56,4]{3,2,1,0} %convolution.682, f32[1,56,56,4]{3,2,1,0} %convolution.683, f32[1,56,56,4]{3,2,1,0} %convolution.684, f32[1,56,56,4]{3,2,1,0} %convolution.685, f32[1,56,56,4]{3,2,1,0} %convolution.686, f32[1,56,56,4]{3,2,1,0} %convolution.687, f32[1,56,56,4]{3,2,1,0} %convolution.688, f32[1,56,56,4]{3,2,1,0} %convolution.689, f32[1,56,56,4]{3,2,1,0} %convolution.690, f32[1,56,56,4]{3,2,1,0} %convolution.692, f32[1,56,56,4]{3,2,1,0} %convolution.693, f32[1,56,56,4]{3,2,1,0} %convolution.694, f32[1,56,56,4]{3,2,1,0} %convolution.695, f32[1,56,56,4]{3,2,1,0} %convolution.696, f32[1,56,56,4]{3,2,1,0} %convolution.697, f32[1,56,56,4]{3,2,1,0} %convolution.698, f32[1,56,56,4]{3,2,1,0} %convolution.699, f32[1,56,56,4]{3,2,1,0} %convolution.700, f32[1,56,56,4]{3,2,1,0} %convolution.701, f32[1,56,56,4]{3,2,1,0} %convolution.703, f32[1,56,56,4]{3,2,1,0} %convolution.704), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_2"}
  %constant.591 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit3_bn2/add"}
  %broadcast.592 = f32[128]{0} broadcast(f32[] %constant.591), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit3_bn2/add"}
  %arg17.18 = f32[128]{0} parameter(17), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.142 = f32[128]{0} reshape(f32[128]{0} %arg17.18)
  %add.593 = f32[128]{0} add(f32[128]{0} %broadcast.592, f32[128]{0} %reshape.142), metadata={op_type="AddV2" op_name="stage1_unit3_bn2/add"}
  %rsqrt.594 = f32[128]{0} rsqrt(f32[128]{0} %add.593), metadata={op_type="Rsqrt" op_name="stage1_unit3_bn2/Rsqrt"}
  %arg45.46 = f32[128]{0} parameter(45), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.170 = f32[128]{0} reshape(f32[128]{0} %arg45.46)
  %multiply.595 = f32[128]{0} multiply(f32[128]{0} %rsqrt.594, f32[128]{0} %reshape.170), metadata={op_type="Mul" op_name="stage1_unit3_bn2/mul"}
  %broadcast.712 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %multiply.595), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit3_bn2/mul_1"}
  %multiply.713 = f32[1,56,56,128]{3,2,1,0} multiply(f32[1,56,56,128]{3,2,1,0} %concatenate.711, f32[1,56,56,128]{3,2,1,0} %broadcast.712), metadata={op_type="Mul" op_name="stage1_unit3_bn2/mul_1"}
  %arg94.95 = f32[128]{0} parameter(94), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.219 = f32[128]{0} reshape(f32[128]{0} %arg94.95)
  %arg70.71 = f32[128]{0} parameter(70), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.195 = f32[128]{0} reshape(f32[128]{0} %arg70.71)
  %multiply.596 = f32[128]{0} multiply(f32[128]{0} %multiply.595, f32[128]{0} %reshape.195), metadata={op_type="Mul" op_name="stage1_unit3_bn2/mul_2"}
  %subtract.597 = f32[128]{0} subtract(f32[128]{0} %reshape.219, f32[128]{0} %multiply.596), metadata={op_type="Sub" op_name="stage1_unit3_bn2/sub"}
  %broadcast.714 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %subtract.597), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit3_bn2/add_1"}
  %add.715 = f32[1,56,56,128]{3,2,1,0} add(f32[1,56,56,128]{3,2,1,0} %multiply.713, f32[1,56,56,128]{3,2,1,0} %broadcast.714), metadata={op_type="AddV2" op_name="stage1_unit3_bn2/add_1"}
  %maximum.718 = f32[1,56,56,128]{3,2,1,0} maximum(f32[1,56,56,128]{3,2,1,0} %broadcast.717, f32[1,56,56,128]{3,2,1,0} %add.715), metadata={op_type="Relu" op_name="stage1_unit3_relu2"}
  %arg113.114 = f32[1,1,128,256]{3,2,1,0} parameter(113), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.238 = f32[1,1,128,256]{3,2,1,0} reshape(f32[1,1,128,256]{3,2,1,0} %arg113.114)
  %convolution.719 = f32[1,56,56,256]{3,2,1,0} convolution(f32[1,56,56,128]{3,2,1,0} %maximum.718, f32[1,1,128,256]{3,2,1,0} %reshape.238), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv3"}
  %multiply.721 = f32[1,56,56,256]{3,2,1,0} multiply(f32[1,56,56,256]{3,2,1,0} %broadcast.720, f32[1,56,56,256]{3,2,1,0} %convolution.719), metadata={op_type="Mul" op_name="stage1_unit3_bn3/mul_1"}
  %arg96.97 = f32[256]{0} parameter(96), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.221 = f32[256]{0} reshape(f32[256]{0} %arg96.97)
  %arg72.73 = f32[256]{0} parameter(72), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.197 = f32[256]{0} reshape(f32[256]{0} %arg72.73)
  %multiply.603 = f32[256]{0} multiply(f32[256]{0} %multiply.602, f32[256]{0} %reshape.197), metadata={op_type="Mul" op_name="stage1_unit3_bn3/mul_2"}
  %subtract.604 = f32[256]{0} subtract(f32[256]{0} %reshape.221, f32[256]{0} %multiply.603), metadata={op_type="Sub" op_name="stage1_unit3_bn3/sub"}
  %broadcast.722 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.604), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit3_bn3/add_1"}
  %add.723 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %multiply.721, f32[1,56,56,256]{3,2,1,0} %broadcast.722), metadata={op_type="AddV2" op_name="stage1_unit3_bn3/add_1"}
  %add.724 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %maximum.583, f32[1,56,56,256]{3,2,1,0} %add.723), metadata={op_type="AddV2" op_name="add_2"}
  %maximum.727 = f32[1,56,56,256]{3,2,1,0} maximum(f32[1,56,56,256]{3,2,1,0} %broadcast.726, f32[1,56,56,256]{3,2,1,0} %add.724), metadata={op_type="Relu" op_name="stage1_unit3_relu"}
  %arg114.115 = f32[1,1,256,256]{3,2,1,0} parameter(114), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.239 = f32[1,1,256,256]{3,2,1,0} reshape(f32[1,1,256,256]{3,2,1,0} %arg114.115)
  %convolution.749 = f32[1,56,56,256]{3,2,1,0} convolution(f32[1,56,56,256]{3,2,1,0} %maximum.727, f32[1,1,256,256]{3,2,1,0} %reshape.239), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv1"}
  %multiply.751 = f32[1,56,56,256]{3,2,1,0} multiply(f32[1,56,56,256]{3,2,1,0} %broadcast.750, f32[1,56,56,256]{3,2,1,0} %convolution.749), metadata={op_type="Mul" op_name="stage2_unit1_bn1/mul_1"}
  %arg97.98 = f32[256]{0} parameter(97), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.222 = f32[256]{0} reshape(f32[256]{0} %arg97.98)
  %arg73.74 = f32[256]{0} parameter(73), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.198 = f32[256]{0} reshape(f32[256]{0} %arg73.74)
  %multiply.733 = f32[256]{0} multiply(f32[256]{0} %multiply.732, f32[256]{0} %reshape.198), metadata={op_type="Mul" op_name="stage2_unit1_bn1/mul_2"}
  %subtract.734 = f32[256]{0} subtract(f32[256]{0} %reshape.222, f32[256]{0} %multiply.733), metadata={op_type="Sub" op_name="stage2_unit1_bn1/sub"}
  %broadcast.752 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.734), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit1_bn1/add_1"}
  %add.753 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %multiply.751, f32[1,56,56,256]{3,2,1,0} %broadcast.752), metadata={op_type="AddV2" op_name="stage2_unit1_bn1/add_1"}
  %maximum.756 = f32[1,56,56,256]{3,2,1,0} maximum(f32[1,56,56,256]{3,2,1,0} %broadcast.755, f32[1,56,56,256]{3,2,1,0} %add.753), metadata={op_type="Relu" op_name="stage2_unit1_relu1"}
  %constant.757 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_4"}
  %pad.758 = f32[1,58,58,256]{3,2,1,0} pad(f32[1,56,56,256]{3,2,1,0} %maximum.756, f32[] %constant.757), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_4"}
  %slice.759 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.758), slice={[0:1], [0:58], [0:58], [0:8]}, metadata={op_type="Split" op_name="split_7"}
  %arg23.24 = f32[3,3,8,256]{3,2,1,0} parameter(23), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.148 = f32[3,3,8,256]{3,2,1,0} reshape(f32[3,3,8,256]{3,2,1,0} %arg23.24)
  %slice.791 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.148), slice={[0:3], [0:3], [0:8], [0:8]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.823 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.759, f32[3,3,8,8]{3,2,1,0} %slice.791), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2"}
  %slice.760 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.758), slice={[0:1], [0:58], [0:58], [8:16]}, metadata={op_type="Split" op_name="split_7"}
  %slice.792 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.148), slice={[0:3], [0:3], [0:8], [8:16]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.824 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.760, f32[3,3,8,8]{3,2,1,0} %slice.792), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_1"}
  %slice.761 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.758), slice={[0:1], [0:58], [0:58], [16:24]}, metadata={op_type="Split" op_name="split_7"}
  %slice.793 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.148), slice={[0:3], [0:3], [0:8], [16:24]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.835 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.761, f32[3,3,8,8]{3,2,1,0} %slice.793), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_2"}
  %slice.762 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.758), slice={[0:1], [0:58], [0:58], [24:32]}, metadata={op_type="Split" op_name="split_7"}
  %slice.794 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.148), slice={[0:3], [0:3], [0:8], [24:32]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.846 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.762, f32[3,3,8,8]{3,2,1,0} %slice.794), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_3"}
  %slice.763 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.758), slice={[0:1], [0:58], [0:58], [32:40]}, metadata={op_type="Split" op_name="split_7"}
  %slice.795 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.148), slice={[0:3], [0:3], [0:8], [32:40]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.849 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.763, f32[3,3,8,8]{3,2,1,0} %slice.795), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_4"}
  %slice.764 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.758), slice={[0:1], [0:58], [0:58], [40:48]}, metadata={op_type="Split" op_name="split_7"}
  %slice.796 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.148), slice={[0:3], [0:3], [0:8], [40:48]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.850 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.764, f32[3,3,8,8]{3,2,1,0} %slice.796), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_5"}
  %slice.765 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.758), slice={[0:1], [0:58], [0:58], [48:56]}, metadata={op_type="Split" op_name="split_7"}
  %slice.797 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.148), slice={[0:3], [0:3], [0:8], [48:56]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.851 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.765, f32[3,3,8,8]{3,2,1,0} %slice.797), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_6"}
  %slice.766 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.758), slice={[0:1], [0:58], [0:58], [56:64]}, metadata={op_type="Split" op_name="split_7"}
  %slice.798 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.148), slice={[0:3], [0:3], [0:8], [56:64]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.852 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.766, f32[3,3,8,8]{3,2,1,0} %slice.798), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_7"}
  %slice.767 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.758), slice={[0:1], [0:58], [0:58], [64:72]}, metadata={op_type="Split" op_name="split_7"}
  %slice.799 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.148), slice={[0:3], [0:3], [0:8], [64:72]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.853 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.767, f32[3,3,8,8]{3,2,1,0} %slice.799), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_8"}
  %slice.768 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.758), slice={[0:1], [0:58], [0:58], [72:80]}, metadata={op_type="Split" op_name="split_7"}
  %slice.800 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.148), slice={[0:3], [0:3], [0:8], [72:80]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.854 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.768, f32[3,3,8,8]{3,2,1,0} %slice.800), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_9"}
  %slice.769 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.758), slice={[0:1], [0:58], [0:58], [80:88]}, metadata={op_type="Split" op_name="split_7"}
  %slice.801 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.148), slice={[0:3], [0:3], [0:8], [80:88]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.825 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.769, f32[3,3,8,8]{3,2,1,0} %slice.801), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_10"}
  %slice.770 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.758), slice={[0:1], [0:58], [0:58], [88:96]}, metadata={op_type="Split" op_name="split_7"}
  %slice.802 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.148), slice={[0:3], [0:3], [0:8], [88:96]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.826 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.770, f32[3,3,8,8]{3,2,1,0} %slice.802), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_11"}
  %slice.771 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.758), slice={[0:1], [0:58], [0:58], [96:104]}, metadata={op_type="Split" op_name="split_7"}
  %slice.803 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.148), slice={[0:3], [0:3], [0:8], [96:104]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.827 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.771, f32[3,3,8,8]{3,2,1,0} %slice.803), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_12"}
  %slice.772 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.758), slice={[0:1], [0:58], [0:58], [104:112]}, metadata={op_type="Split" op_name="split_7"}
  %slice.804 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.148), slice={[0:3], [0:3], [0:8], [104:112]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.828 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.772, f32[3,3,8,8]{3,2,1,0} %slice.804), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_13"}
  %slice.773 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.758), slice={[0:1], [0:58], [0:58], [112:120]}, metadata={op_type="Split" op_name="split_7"}
  %slice.805 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.148), slice={[0:3], [0:3], [0:8], [112:120]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.829 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.773, f32[3,3,8,8]{3,2,1,0} %slice.805), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_14"}
  %slice.774 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.758), slice={[0:1], [0:58], [0:58], [120:128]}, metadata={op_type="Split" op_name="split_7"}
  %slice.806 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.148), slice={[0:3], [0:3], [0:8], [120:128]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.830 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.774, f32[3,3,8,8]{3,2,1,0} %slice.806), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_15"}
  %slice.775 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.758), slice={[0:1], [0:58], [0:58], [128:136]}, metadata={op_type="Split" op_name="split_7"}
  %slice.807 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.148), slice={[0:3], [0:3], [0:8], [128:136]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.831 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.775, f32[3,3,8,8]{3,2,1,0} %slice.807), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_16"}
  %slice.776 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.758), slice={[0:1], [0:58], [0:58], [136:144]}, metadata={op_type="Split" op_name="split_7"}
  %slice.808 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.148), slice={[0:3], [0:3], [0:8], [136:144]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.832 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.776, f32[3,3,8,8]{3,2,1,0} %slice.808), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_17"}
  %slice.777 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.758), slice={[0:1], [0:58], [0:58], [144:152]}, metadata={op_type="Split" op_name="split_7"}
  %slice.809 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.148), slice={[0:3], [0:3], [0:8], [144:152]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.833 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.777, f32[3,3,8,8]{3,2,1,0} %slice.809), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_18"}
  %slice.778 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.758), slice={[0:1], [0:58], [0:58], [152:160]}, metadata={op_type="Split" op_name="split_7"}
  %slice.810 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.148), slice={[0:3], [0:3], [0:8], [152:160]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.834 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.778, f32[3,3,8,8]{3,2,1,0} %slice.810), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_19"}
  %slice.779 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.758), slice={[0:1], [0:58], [0:58], [160:168]}, metadata={op_type="Split" op_name="split_7"}
  %slice.811 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.148), slice={[0:3], [0:3], [0:8], [160:168]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.836 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.779, f32[3,3,8,8]{3,2,1,0} %slice.811), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_20"}
  %slice.780 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.758), slice={[0:1], [0:58], [0:58], [168:176]}, metadata={op_type="Split" op_name="split_7"}
  %slice.812 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.148), slice={[0:3], [0:3], [0:8], [168:176]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.837 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.780, f32[3,3,8,8]{3,2,1,0} %slice.812), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_21"}
  %slice.781 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.758), slice={[0:1], [0:58], [0:58], [176:184]}, metadata={op_type="Split" op_name="split_7"}
  %slice.813 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.148), slice={[0:3], [0:3], [0:8], [176:184]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.838 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.781, f32[3,3,8,8]{3,2,1,0} %slice.813), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_22"}
  %slice.782 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.758), slice={[0:1], [0:58], [0:58], [184:192]}, metadata={op_type="Split" op_name="split_7"}
  %slice.814 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.148), slice={[0:3], [0:3], [0:8], [184:192]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.839 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.782, f32[3,3,8,8]{3,2,1,0} %slice.814), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_23"}
  %slice.783 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.758), slice={[0:1], [0:58], [0:58], [192:200]}, metadata={op_type="Split" op_name="split_7"}
  %slice.815 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.148), slice={[0:3], [0:3], [0:8], [192:200]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.840 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.783, f32[3,3,8,8]{3,2,1,0} %slice.815), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_24"}
  %slice.784 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.758), slice={[0:1], [0:58], [0:58], [200:208]}, metadata={op_type="Split" op_name="split_7"}
  %slice.816 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.148), slice={[0:3], [0:3], [0:8], [200:208]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.841 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.784, f32[3,3,8,8]{3,2,1,0} %slice.816), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_25"}
  %slice.785 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.758), slice={[0:1], [0:58], [0:58], [208:216]}, metadata={op_type="Split" op_name="split_7"}
  %slice.817 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.148), slice={[0:3], [0:3], [0:8], [208:216]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.842 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.785, f32[3,3,8,8]{3,2,1,0} %slice.817), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_26"}
  %slice.786 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.758), slice={[0:1], [0:58], [0:58], [216:224]}, metadata={op_type="Split" op_name="split_7"}
  %slice.818 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.148), slice={[0:3], [0:3], [0:8], [216:224]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.843 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.786, f32[3,3,8,8]{3,2,1,0} %slice.818), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_27"}
  %slice.787 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.758), slice={[0:1], [0:58], [0:58], [224:232]}, metadata={op_type="Split" op_name="split_7"}
  %slice.819 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.148), slice={[0:3], [0:3], [0:8], [224:232]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.844 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.787, f32[3,3,8,8]{3,2,1,0} %slice.819), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_28"}
  %slice.788 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.758), slice={[0:1], [0:58], [0:58], [232:240]}, metadata={op_type="Split" op_name="split_7"}
  %slice.820 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.148), slice={[0:3], [0:3], [0:8], [232:240]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.845 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.788, f32[3,3,8,8]{3,2,1,0} %slice.820), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_29"}
  %slice.789 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.758), slice={[0:1], [0:58], [0:58], [240:248]}, metadata={op_type="Split" op_name="split_7"}
  %slice.821 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.148), slice={[0:3], [0:3], [0:8], [240:248]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.847 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.789, f32[3,3,8,8]{3,2,1,0} %slice.821), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_30"}
  %slice.790 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.758), slice={[0:1], [0:58], [0:58], [248:256]}, metadata={op_type="Split" op_name="split_7"}
  %slice.822 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.148), slice={[0:3], [0:3], [0:8], [248:256]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.848 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.790, f32[3,3,8,8]{3,2,1,0} %slice.822), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_31"}
  %concatenate.855 = f32[1,28,28,256]{3,2,1,0} concatenate(f32[1,28,28,8]{3,2,1,0} %convolution.823, f32[1,28,28,8]{3,2,1,0} %convolution.824, f32[1,28,28,8]{3,2,1,0} %convolution.835, f32[1,28,28,8]{3,2,1,0} %convolution.846, f32[1,28,28,8]{3,2,1,0} %convolution.849, f32[1,28,28,8]{3,2,1,0} %convolution.850, f32[1,28,28,8]{3,2,1,0} %convolution.851, f32[1,28,28,8]{3,2,1,0} %convolution.852, f32[1,28,28,8]{3,2,1,0} %convolution.853, f32[1,28,28,8]{3,2,1,0} %convolution.854, f32[1,28,28,8]{3,2,1,0} %convolution.825, f32[1,28,28,8]{3,2,1,0} %convolution.826, f32[1,28,28,8]{3,2,1,0} %convolution.827, f32[1,28,28,8]{3,2,1,0} %convolution.828, f32[1,28,28,8]{3,2,1,0} %convolution.829, f32[1,28,28,8]{3,2,1,0} %convolution.830, f32[1,28,28,8]{3,2,1,0} %convolution.831, f32[1,28,28,8]{3,2,1,0} %convolution.832, f32[1,28,28,8]{3,2,1,0} %convolution.833, f32[1,28,28,8]{3,2,1,0} %convolution.834, f32[1,28,28,8]{3,2,1,0} %convolution.836, f32[1,28,28,8]{3,2,1,0} %convolution.837, f32[1,28,28,8]{3,2,1,0} %convolution.838, f32[1,28,28,8]{3,2,1,0} %convolution.839, f32[1,28,28,8]{3,2,1,0} %convolution.840, f32[1,28,28,8]{3,2,1,0} %convolution.841, f32[1,28,28,8]{3,2,1,0} %convolution.842, f32[1,28,28,8]{3,2,1,0} %convolution.843, f32[1,28,28,8]{3,2,1,0} %convolution.844, f32[1,28,28,8]{3,2,1,0} %convolution.845, f32[1,28,28,8]{3,2,1,0} %convolution.847, f32[1,28,28,8]{3,2,1,0} %convolution.848), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_3"}
  %constant.735 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit1_bn2/add"}
  %broadcast.736 = f32[256]{0} broadcast(f32[] %constant.735), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit1_bn2/add"}
  %arg24.25 = f32[256]{0} parameter(24), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.149 = f32[256]{0} reshape(f32[256]{0} %arg24.25)
  %add.737 = f32[256]{0} add(f32[256]{0} %broadcast.736, f32[256]{0} %reshape.149), metadata={op_type="AddV2" op_name="stage2_unit1_bn2/add"}
  %rsqrt.738 = f32[256]{0} rsqrt(f32[256]{0} %add.737), metadata={op_type="Rsqrt" op_name="stage2_unit1_bn2/Rsqrt"}
  %arg51.52 = f32[256]{0} parameter(51), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.176 = f32[256]{0} reshape(f32[256]{0} %arg51.52)
  %multiply.739 = f32[256]{0} multiply(f32[256]{0} %rsqrt.738, f32[256]{0} %reshape.176), metadata={op_type="Mul" op_name="stage2_unit1_bn2/mul"}
  %broadcast.856 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.739), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit1_bn2/mul_1"}
  %multiply.857 = f32[1,28,28,256]{3,2,1,0} multiply(f32[1,28,28,256]{3,2,1,0} %concatenate.855, f32[1,28,28,256]{3,2,1,0} %broadcast.856), metadata={op_type="Mul" op_name="stage2_unit1_bn2/mul_1"}
  %arg100.101 = f32[256]{0} parameter(100), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.225 = f32[256]{0} reshape(f32[256]{0} %arg100.101)
  %arg76.77 = f32[256]{0} parameter(76), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.201 = f32[256]{0} reshape(f32[256]{0} %arg76.77)
  %multiply.740 = f32[256]{0} multiply(f32[256]{0} %multiply.739, f32[256]{0} %reshape.201), metadata={op_type="Mul" op_name="stage2_unit1_bn2/mul_2"}
  %subtract.741 = f32[256]{0} subtract(f32[256]{0} %reshape.225, f32[256]{0} %multiply.740), metadata={op_type="Sub" op_name="stage2_unit1_bn2/sub"}
  %broadcast.858 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.741), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit1_bn2/add_1"}
  %add.859 = f32[1,28,28,256]{3,2,1,0} add(f32[1,28,28,256]{3,2,1,0} %multiply.857, f32[1,28,28,256]{3,2,1,0} %broadcast.858), metadata={op_type="AddV2" op_name="stage2_unit1_bn2/add_1"}
  %maximum.862 = f32[1,28,28,256]{3,2,1,0} maximum(f32[1,28,28,256]{3,2,1,0} %broadcast.861, f32[1,28,28,256]{3,2,1,0} %add.859), metadata={op_type="Relu" op_name="stage2_unit1_relu2"}
  %arg116.117 = f32[1,1,256,512]{3,2,1,0} parameter(116), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.241 = f32[1,1,256,512]{3,2,1,0} reshape(f32[1,1,256,512]{3,2,1,0} %arg116.117)
  %convolution.863 = f32[1,28,28,512]{3,2,1,0} convolution(f32[1,28,28,256]{3,2,1,0} %maximum.862, f32[1,1,256,512]{3,2,1,0} %reshape.241), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv3"}
  %multiply.865 = f32[1,28,28,512]{3,2,1,0} multiply(f32[1,28,28,512]{3,2,1,0} %broadcast.864, f32[1,28,28,512]{3,2,1,0} %convolution.863), metadata={op_type="Mul" op_name="stage2_unit1_bn3/mul_1"}
  %arg101.102 = f32[512]{0} parameter(101), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.226 = f32[512]{0} reshape(f32[512]{0} %arg101.102)
  %arg77.78 = f32[512]{0} parameter(77), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.202 = f32[512]{0} reshape(f32[512]{0} %arg77.78)
  %multiply.747 = f32[512]{0} multiply(f32[512]{0} %multiply.746, f32[512]{0} %reshape.202), metadata={op_type="Mul" op_name="stage2_unit1_bn3/mul_2"}
  %subtract.748 = f32[512]{0} subtract(f32[512]{0} %reshape.226, f32[512]{0} %multiply.747), metadata={op_type="Sub" op_name="stage2_unit1_bn3/sub"}
  %broadcast.866 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.748), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit1_bn3/add_1"}
  %add.867 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %multiply.865, f32[1,28,28,512]{3,2,1,0} %broadcast.866), metadata={op_type="AddV2" op_name="stage2_unit1_bn3/add_1"}
  %arg115.116 = f32[1,1,256,512]{3,2,1,0} parameter(115), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.240 = f32[1,1,256,512]{3,2,1,0} reshape(f32[1,1,256,512]{3,2,1,0} %arg115.116)
  %convolution.875 = f32[1,28,28,512]{3,2,1,0} convolution(f32[1,56,56,256]{3,2,1,0} %maximum.727, f32[1,1,256,512]{3,2,1,0} %reshape.240), window={size=1x1 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_sc"}
  %constant.868 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit1_sc_bn/add"}
  %broadcast.869 = f32[512]{0} broadcast(f32[] %constant.868), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit1_sc_bn/add"}
  %arg21.22 = f32[512]{0} parameter(21), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.146 = f32[512]{0} reshape(f32[512]{0} %arg21.22)
  %add.870 = f32[512]{0} add(f32[512]{0} %broadcast.869, f32[512]{0} %reshape.146), metadata={op_type="AddV2" op_name="stage2_unit1_sc_bn/add"}
  %rsqrt.871 = f32[512]{0} rsqrt(f32[512]{0} %add.870), metadata={op_type="Rsqrt" op_name="stage2_unit1_sc_bn/Rsqrt"}
  %arg49.50 = f32[512]{0} parameter(49), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.174 = f32[512]{0} reshape(f32[512]{0} %arg49.50)
  %multiply.872 = f32[512]{0} multiply(f32[512]{0} %rsqrt.871, f32[512]{0} %reshape.174), metadata={op_type="Mul" op_name="stage2_unit1_sc_bn/mul"}
  %broadcast.876 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.872), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit1_sc_bn/mul_1"}
  %multiply.877 = f32[1,28,28,512]{3,2,1,0} multiply(f32[1,28,28,512]{3,2,1,0} %convolution.875, f32[1,28,28,512]{3,2,1,0} %broadcast.876), metadata={op_type="Mul" op_name="stage2_unit1_sc_bn/mul_1"}
  %arg98.99 = f32[512]{0} parameter(98), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.223 = f32[512]{0} reshape(f32[512]{0} %arg98.99)
  %arg74.75 = f32[512]{0} parameter(74), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.199 = f32[512]{0} reshape(f32[512]{0} %arg74.75)
  %multiply.873 = f32[512]{0} multiply(f32[512]{0} %multiply.872, f32[512]{0} %reshape.199), metadata={op_type="Mul" op_name="stage2_unit1_sc_bn/mul_2"}
  %subtract.874 = f32[512]{0} subtract(f32[512]{0} %reshape.223, f32[512]{0} %multiply.873), metadata={op_type="Sub" op_name="stage2_unit1_sc_bn/sub"}
  %broadcast.878 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.874), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit1_sc_bn/add_1"}
  %add.879 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %multiply.877, f32[1,28,28,512]{3,2,1,0} %broadcast.878), metadata={op_type="AddV2" op_name="stage2_unit1_sc_bn/add_1"}
  %add.880 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %add.867, f32[1,28,28,512]{3,2,1,0} %add.879), metadata={op_type="AddV2" op_name="add_3"}
  %maximum.883 = f32[1,28,28,512]{3,2,1,0} maximum(f32[1,28,28,512]{3,2,1,0} %broadcast.882, f32[1,28,28,512]{3,2,1,0} %add.880), metadata={op_type="Relu" op_name="stage2_unit1_relu"}
  %constant.898 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit2_bn3/add"}
  %broadcast.899 = f32[512]{0} broadcast(f32[] %constant.898), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit2_bn3/add"}
  %arg31.32 = f32[512]{0} parameter(31), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.156 = f32[512]{0} reshape(f32[512]{0} %arg31.32)
  %add.900 = f32[512]{0} add(f32[512]{0} %broadcast.899, f32[512]{0} %reshape.156), metadata={op_type="AddV2" op_name="stage2_unit2_bn3/add"}
  %rsqrt.901 = f32[512]{0} rsqrt(f32[512]{0} %add.900), metadata={op_type="Rsqrt" op_name="stage2_unit2_bn3/Rsqrt"}
  %arg56.57 = f32[512]{0} parameter(56), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.181 = f32[512]{0} reshape(f32[512]{0} %arg56.57)
  %multiply.902 = f32[512]{0} multiply(f32[512]{0} %rsqrt.901, f32[512]{0} %reshape.181), metadata={op_type="Mul" op_name="stage2_unit2_bn3/mul"}
  %broadcast.1020 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.902), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit2_bn3/mul_1"}
  %constant.1016 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit2_relu2"}
  %broadcast.1017 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[] %constant.1016), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit2_relu2"}
  %constant.910 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit2_relu1"}
  %broadcast.911 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[] %constant.910), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit2_relu1"}
  %constant.884 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit2_bn1/add"}
  %broadcast.885 = f32[256]{0} broadcast(f32[] %constant.884), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit2_bn1/add"}
  %arg28.29 = f32[256]{0} parameter(28), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.153 = f32[256]{0} reshape(f32[256]{0} %arg28.29)
  %add.886 = f32[256]{0} add(f32[256]{0} %broadcast.885, f32[256]{0} %reshape.153), metadata={op_type="AddV2" op_name="stage2_unit2_bn1/add"}
  %rsqrt.887 = f32[256]{0} rsqrt(f32[256]{0} %add.886), metadata={op_type="Rsqrt" op_name="stage2_unit2_bn1/Rsqrt"}
  %arg54.55 = f32[256]{0} parameter(54), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.179 = f32[256]{0} reshape(f32[256]{0} %arg54.55)
  %multiply.888 = f32[256]{0} multiply(f32[256]{0} %rsqrt.887, f32[256]{0} %reshape.179), metadata={op_type="Mul" op_name="stage2_unit2_bn1/mul"}
  %broadcast.906 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.888), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit2_bn1/mul_1"}
  %arg117.118 = f32[1,1,512,256]{3,2,1,0} parameter(117), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.242 = f32[1,1,512,256]{3,2,1,0} reshape(f32[1,1,512,256]{3,2,1,0} %arg117.118)
  %convolution.905 = f32[1,28,28,256]{3,2,1,0} convolution(f32[1,28,28,512]{3,2,1,0} %maximum.883, f32[1,1,512,256]{3,2,1,0} %reshape.242), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv1"}
  %multiply.907 = f32[1,28,28,256]{3,2,1,0} multiply(f32[1,28,28,256]{3,2,1,0} %broadcast.906, f32[1,28,28,256]{3,2,1,0} %convolution.905), metadata={op_type="Mul" op_name="stage2_unit2_bn1/mul_1"}
  %arg103.104 = f32[256]{0} parameter(103), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.228 = f32[256]{0} reshape(f32[256]{0} %arg103.104)
  %arg79.80 = f32[256]{0} parameter(79), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.204 = f32[256]{0} reshape(f32[256]{0} %arg79.80)
  %multiply.889 = f32[256]{0} multiply(f32[256]{0} %multiply.888, f32[256]{0} %reshape.204), metadata={op_type="Mul" op_name="stage2_unit2_bn1/mul_2"}
  %subtract.890 = f32[256]{0} subtract(f32[256]{0} %reshape.228, f32[256]{0} %multiply.889), metadata={op_type="Sub" op_name="stage2_unit2_bn1/sub"}
  %broadcast.908 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.890), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit2_bn1/add_1"}
  %add.909 = f32[1,28,28,256]{3,2,1,0} add(f32[1,28,28,256]{3,2,1,0} %multiply.907, f32[1,28,28,256]{3,2,1,0} %broadcast.908), metadata={op_type="AddV2" op_name="stage2_unit2_bn1/add_1"}
  %maximum.912 = f32[1,28,28,256]{3,2,1,0} maximum(f32[1,28,28,256]{3,2,1,0} %broadcast.911, f32[1,28,28,256]{3,2,1,0} %add.909), metadata={op_type="Relu" op_name="stage2_unit2_relu1"}
  %constant.913 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_5"}
  %pad.914 = f32[1,30,30,256]{3,2,1,0} pad(f32[1,28,28,256]{3,2,1,0} %maximum.912, f32[] %constant.913), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_5"}
  %slice.915 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.914), slice={[0:1], [0:30], [0:30], [0:8]}, metadata={op_type="Split" op_name="split_9"}
  %arg29.30 = f32[3,3,8,256]{3,2,1,0} parameter(29), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.154 = f32[3,3,8,256]{3,2,1,0} reshape(f32[3,3,8,256]{3,2,1,0} %arg29.30)
  %slice.947 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.154), slice={[0:3], [0:3], [0:8], [0:8]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.979 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.915, f32[3,3,8,8]{3,2,1,0} %slice.947), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2"}
  %slice.916 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.914), slice={[0:1], [0:30], [0:30], [8:16]}, metadata={op_type="Split" op_name="split_9"}
  %slice.948 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.154), slice={[0:3], [0:3], [0:8], [8:16]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.980 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.916, f32[3,3,8,8]{3,2,1,0} %slice.948), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_1"}
  %slice.917 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.914), slice={[0:1], [0:30], [0:30], [16:24]}, metadata={op_type="Split" op_name="split_9"}
  %slice.949 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.154), slice={[0:3], [0:3], [0:8], [16:24]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.991 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.917, f32[3,3,8,8]{3,2,1,0} %slice.949), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_2"}
  %slice.918 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.914), slice={[0:1], [0:30], [0:30], [24:32]}, metadata={op_type="Split" op_name="split_9"}
  %slice.950 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.154), slice={[0:3], [0:3], [0:8], [24:32]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1002 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.918, f32[3,3,8,8]{3,2,1,0} %slice.950), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_3"}
  %slice.919 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.914), slice={[0:1], [0:30], [0:30], [32:40]}, metadata={op_type="Split" op_name="split_9"}
  %slice.951 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.154), slice={[0:3], [0:3], [0:8], [32:40]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1005 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.919, f32[3,3,8,8]{3,2,1,0} %slice.951), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_4"}
  %slice.920 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.914), slice={[0:1], [0:30], [0:30], [40:48]}, metadata={op_type="Split" op_name="split_9"}
  %slice.952 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.154), slice={[0:3], [0:3], [0:8], [40:48]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1006 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.920, f32[3,3,8,8]{3,2,1,0} %slice.952), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_5"}
  %slice.921 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.914), slice={[0:1], [0:30], [0:30], [48:56]}, metadata={op_type="Split" op_name="split_9"}
  %slice.953 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.154), slice={[0:3], [0:3], [0:8], [48:56]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1007 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.921, f32[3,3,8,8]{3,2,1,0} %slice.953), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_6"}
  %slice.922 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.914), slice={[0:1], [0:30], [0:30], [56:64]}, metadata={op_type="Split" op_name="split_9"}
  %slice.954 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.154), slice={[0:3], [0:3], [0:8], [56:64]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1008 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.922, f32[3,3,8,8]{3,2,1,0} %slice.954), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_7"}
  %slice.923 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.914), slice={[0:1], [0:30], [0:30], [64:72]}, metadata={op_type="Split" op_name="split_9"}
  %slice.955 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.154), slice={[0:3], [0:3], [0:8], [64:72]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1009 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.923, f32[3,3,8,8]{3,2,1,0} %slice.955), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_8"}
  %slice.924 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.914), slice={[0:1], [0:30], [0:30], [72:80]}, metadata={op_type="Split" op_name="split_9"}
  %slice.956 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.154), slice={[0:3], [0:3], [0:8], [72:80]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1010 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.924, f32[3,3,8,8]{3,2,1,0} %slice.956), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_9"}
  %slice.925 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.914), slice={[0:1], [0:30], [0:30], [80:88]}, metadata={op_type="Split" op_name="split_9"}
  %slice.957 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.154), slice={[0:3], [0:3], [0:8], [80:88]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.981 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.925, f32[3,3,8,8]{3,2,1,0} %slice.957), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_10"}
  %slice.926 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.914), slice={[0:1], [0:30], [0:30], [88:96]}, metadata={op_type="Split" op_name="split_9"}
  %slice.958 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.154), slice={[0:3], [0:3], [0:8], [88:96]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.982 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.926, f32[3,3,8,8]{3,2,1,0} %slice.958), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_11"}
  %slice.927 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.914), slice={[0:1], [0:30], [0:30], [96:104]}, metadata={op_type="Split" op_name="split_9"}
  %slice.959 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.154), slice={[0:3], [0:3], [0:8], [96:104]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.983 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.927, f32[3,3,8,8]{3,2,1,0} %slice.959), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_12"}
  %slice.928 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.914), slice={[0:1], [0:30], [0:30], [104:112]}, metadata={op_type="Split" op_name="split_9"}
  %slice.960 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.154), slice={[0:3], [0:3], [0:8], [104:112]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.984 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.928, f32[3,3,8,8]{3,2,1,0} %slice.960), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_13"}
  %slice.929 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.914), slice={[0:1], [0:30], [0:30], [112:120]}, metadata={op_type="Split" op_name="split_9"}
  %slice.961 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.154), slice={[0:3], [0:3], [0:8], [112:120]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.985 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.929, f32[3,3,8,8]{3,2,1,0} %slice.961), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_14"}
  %slice.930 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.914), slice={[0:1], [0:30], [0:30], [120:128]}, metadata={op_type="Split" op_name="split_9"}
  %slice.962 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.154), slice={[0:3], [0:3], [0:8], [120:128]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.986 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.930, f32[3,3,8,8]{3,2,1,0} %slice.962), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_15"}
  %slice.931 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.914), slice={[0:1], [0:30], [0:30], [128:136]}, metadata={op_type="Split" op_name="split_9"}
  %slice.963 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.154), slice={[0:3], [0:3], [0:8], [128:136]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.987 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.931, f32[3,3,8,8]{3,2,1,0} %slice.963), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_16"}
  %slice.932 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.914), slice={[0:1], [0:30], [0:30], [136:144]}, metadata={op_type="Split" op_name="split_9"}
  %slice.964 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.154), slice={[0:3], [0:3], [0:8], [136:144]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.988 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.932, f32[3,3,8,8]{3,2,1,0} %slice.964), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_17"}
  %slice.933 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.914), slice={[0:1], [0:30], [0:30], [144:152]}, metadata={op_type="Split" op_name="split_9"}
  %slice.965 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.154), slice={[0:3], [0:3], [0:8], [144:152]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.989 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.933, f32[3,3,8,8]{3,2,1,0} %slice.965), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_18"}
  %slice.934 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.914), slice={[0:1], [0:30], [0:30], [152:160]}, metadata={op_type="Split" op_name="split_9"}
  %slice.966 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.154), slice={[0:3], [0:3], [0:8], [152:160]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.990 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.934, f32[3,3,8,8]{3,2,1,0} %slice.966), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_19"}
  %slice.935 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.914), slice={[0:1], [0:30], [0:30], [160:168]}, metadata={op_type="Split" op_name="split_9"}
  %slice.967 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.154), slice={[0:3], [0:3], [0:8], [160:168]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.992 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.935, f32[3,3,8,8]{3,2,1,0} %slice.967), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_20"}
  %slice.936 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.914), slice={[0:1], [0:30], [0:30], [168:176]}, metadata={op_type="Split" op_name="split_9"}
  %slice.968 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.154), slice={[0:3], [0:3], [0:8], [168:176]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.993 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.936, f32[3,3,8,8]{3,2,1,0} %slice.968), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_21"}
  %slice.937 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.914), slice={[0:1], [0:30], [0:30], [176:184]}, metadata={op_type="Split" op_name="split_9"}
  %slice.969 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.154), slice={[0:3], [0:3], [0:8], [176:184]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.994 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.937, f32[3,3,8,8]{3,2,1,0} %slice.969), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_22"}
  %slice.938 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.914), slice={[0:1], [0:30], [0:30], [184:192]}, metadata={op_type="Split" op_name="split_9"}
  %slice.970 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.154), slice={[0:3], [0:3], [0:8], [184:192]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.995 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.938, f32[3,3,8,8]{3,2,1,0} %slice.970), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_23"}
  %slice.939 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.914), slice={[0:1], [0:30], [0:30], [192:200]}, metadata={op_type="Split" op_name="split_9"}
  %slice.971 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.154), slice={[0:3], [0:3], [0:8], [192:200]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.996 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.939, f32[3,3,8,8]{3,2,1,0} %slice.971), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_24"}
  %slice.940 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.914), slice={[0:1], [0:30], [0:30], [200:208]}, metadata={op_type="Split" op_name="split_9"}
  %slice.972 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.154), slice={[0:3], [0:3], [0:8], [200:208]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.997 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.940, f32[3,3,8,8]{3,2,1,0} %slice.972), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_25"}
  %slice.941 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.914), slice={[0:1], [0:30], [0:30], [208:216]}, metadata={op_type="Split" op_name="split_9"}
  %slice.973 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.154), slice={[0:3], [0:3], [0:8], [208:216]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.998 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.941, f32[3,3,8,8]{3,2,1,0} %slice.973), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_26"}
  %slice.942 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.914), slice={[0:1], [0:30], [0:30], [216:224]}, metadata={op_type="Split" op_name="split_9"}
  %slice.974 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.154), slice={[0:3], [0:3], [0:8], [216:224]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.999 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.942, f32[3,3,8,8]{3,2,1,0} %slice.974), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_27"}
  %slice.943 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.914), slice={[0:1], [0:30], [0:30], [224:232]}, metadata={op_type="Split" op_name="split_9"}
  %slice.975 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.154), slice={[0:3], [0:3], [0:8], [224:232]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1000 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.943, f32[3,3,8,8]{3,2,1,0} %slice.975), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_28"}
  %slice.944 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.914), slice={[0:1], [0:30], [0:30], [232:240]}, metadata={op_type="Split" op_name="split_9"}
  %slice.976 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.154), slice={[0:3], [0:3], [0:8], [232:240]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1001 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.944, f32[3,3,8,8]{3,2,1,0} %slice.976), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_29"}
  %slice.945 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.914), slice={[0:1], [0:30], [0:30], [240:248]}, metadata={op_type="Split" op_name="split_9"}
  %slice.977 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.154), slice={[0:3], [0:3], [0:8], [240:248]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1003 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.945, f32[3,3,8,8]{3,2,1,0} %slice.977), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_30"}
  %slice.946 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.914), slice={[0:1], [0:30], [0:30], [248:256]}, metadata={op_type="Split" op_name="split_9"}
  %slice.978 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.154), slice={[0:3], [0:3], [0:8], [248:256]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1004 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.946, f32[3,3,8,8]{3,2,1,0} %slice.978), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_31"}
  %concatenate.1011 = f32[1,28,28,256]{3,2,1,0} concatenate(f32[1,28,28,8]{3,2,1,0} %convolution.979, f32[1,28,28,8]{3,2,1,0} %convolution.980, f32[1,28,28,8]{3,2,1,0} %convolution.991, f32[1,28,28,8]{3,2,1,0} %convolution.1002, f32[1,28,28,8]{3,2,1,0} %convolution.1005, f32[1,28,28,8]{3,2,1,0} %convolution.1006, f32[1,28,28,8]{3,2,1,0} %convolution.1007, f32[1,28,28,8]{3,2,1,0} %convolution.1008, f32[1,28,28,8]{3,2,1,0} %convolution.1009, f32[1,28,28,8]{3,2,1,0} %convolution.1010, f32[1,28,28,8]{3,2,1,0} %convolution.981, f32[1,28,28,8]{3,2,1,0} %convolution.982, f32[1,28,28,8]{3,2,1,0} %convolution.983, f32[1,28,28,8]{3,2,1,0} %convolution.984, f32[1,28,28,8]{3,2,1,0} %convolution.985, f32[1,28,28,8]{3,2,1,0} %convolution.986, f32[1,28,28,8]{3,2,1,0} %convolution.987, f32[1,28,28,8]{3,2,1,0} %convolution.988, f32[1,28,28,8]{3,2,1,0} %convolution.989, f32[1,28,28,8]{3,2,1,0} %convolution.990, f32[1,28,28,8]{3,2,1,0} %convolution.992, f32[1,28,28,8]{3,2,1,0} %convolution.993, f32[1,28,28,8]{3,2,1,0} %convolution.994, f32[1,28,28,8]{3,2,1,0} %convolution.995, f32[1,28,28,8]{3,2,1,0} %convolution.996, f32[1,28,28,8]{3,2,1,0} %convolution.997, f32[1,28,28,8]{3,2,1,0} %convolution.998, f32[1,28,28,8]{3,2,1,0} %convolution.999, f32[1,28,28,8]{3,2,1,0} %convolution.1000, f32[1,28,28,8]{3,2,1,0} %convolution.1001, f32[1,28,28,8]{3,2,1,0} %convolution.1003, f32[1,28,28,8]{3,2,1,0} %convolution.1004), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_4"}
  %constant.891 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit2_bn2/add"}
  %broadcast.892 = f32[256]{0} broadcast(f32[] %constant.891), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit2_bn2/add"}
  %arg30.31 = f32[256]{0} parameter(30), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.155 = f32[256]{0} reshape(f32[256]{0} %arg30.31)
  %add.893 = f32[256]{0} add(f32[256]{0} %broadcast.892, f32[256]{0} %reshape.155), metadata={op_type="AddV2" op_name="stage2_unit2_bn2/add"}
  %rsqrt.894 = f32[256]{0} rsqrt(f32[256]{0} %add.893), metadata={op_type="Rsqrt" op_name="stage2_unit2_bn2/Rsqrt"}
  %arg55.56 = f32[256]{0} parameter(55), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.180 = f32[256]{0} reshape(f32[256]{0} %arg55.56)
  %multiply.895 = f32[256]{0} multiply(f32[256]{0} %rsqrt.894, f32[256]{0} %reshape.180), metadata={op_type="Mul" op_name="stage2_unit2_bn2/mul"}
  %broadcast.1012 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.895), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit2_bn2/mul_1"}
  %multiply.1013 = f32[1,28,28,256]{3,2,1,0} multiply(f32[1,28,28,256]{3,2,1,0} %concatenate.1011, f32[1,28,28,256]{3,2,1,0} %broadcast.1012), metadata={op_type="Mul" op_name="stage2_unit2_bn2/mul_1"}
  %arg104.105 = f32[256]{0} parameter(104), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.229 = f32[256]{0} reshape(f32[256]{0} %arg104.105)
  %arg80.81 = f32[256]{0} parameter(80), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.205 = f32[256]{0} reshape(f32[256]{0} %arg80.81)
  %multiply.896 = f32[256]{0} multiply(f32[256]{0} %multiply.895, f32[256]{0} %reshape.205), metadata={op_type="Mul" op_name="stage2_unit2_bn2/mul_2"}
  %subtract.897 = f32[256]{0} subtract(f32[256]{0} %reshape.229, f32[256]{0} %multiply.896), metadata={op_type="Sub" op_name="stage2_unit2_bn2/sub"}
  %broadcast.1014 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.897), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit2_bn2/add_1"}
  %add.1015 = f32[1,28,28,256]{3,2,1,0} add(f32[1,28,28,256]{3,2,1,0} %multiply.1013, f32[1,28,28,256]{3,2,1,0} %broadcast.1014), metadata={op_type="AddV2" op_name="stage2_unit2_bn2/add_1"}
  %maximum.1018 = f32[1,28,28,256]{3,2,1,0} maximum(f32[1,28,28,256]{3,2,1,0} %broadcast.1017, f32[1,28,28,256]{3,2,1,0} %add.1015), metadata={op_type="Relu" op_name="stage2_unit2_relu2"}
  %arg118.119 = f32[1,1,256,512]{3,2,1,0} parameter(118), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.243 = f32[1,1,256,512]{3,2,1,0} reshape(f32[1,1,256,512]{3,2,1,0} %arg118.119)
  %convolution.1019 = f32[1,28,28,512]{3,2,1,0} convolution(f32[1,28,28,256]{3,2,1,0} %maximum.1018, f32[1,1,256,512]{3,2,1,0} %reshape.243), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv3"}
  %multiply.1021 = f32[1,28,28,512]{3,2,1,0} multiply(f32[1,28,28,512]{3,2,1,0} %broadcast.1020, f32[1,28,28,512]{3,2,1,0} %convolution.1019), metadata={op_type="Mul" op_name="stage2_unit2_bn3/mul_1"}
  %arg105.106 = f32[512]{0} parameter(105), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.230 = f32[512]{0} reshape(f32[512]{0} %arg105.106)
  %arg81.82 = f32[512]{0} parameter(81), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.206 = f32[512]{0} reshape(f32[512]{0} %arg81.82)
  %multiply.903 = f32[512]{0} multiply(f32[512]{0} %multiply.902, f32[512]{0} %reshape.206), metadata={op_type="Mul" op_name="stage2_unit2_bn3/mul_2"}
  %subtract.904 = f32[512]{0} subtract(f32[512]{0} %reshape.230, f32[512]{0} %multiply.903), metadata={op_type="Sub" op_name="stage2_unit2_bn3/sub"}
  %broadcast.1022 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.904), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit2_bn3/add_1"}
  %add.1023 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %multiply.1021, f32[1,28,28,512]{3,2,1,0} %broadcast.1022), metadata={op_type="AddV2" op_name="stage2_unit2_bn3/add_1"}
  %add.1024 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %maximum.883, f32[1,28,28,512]{3,2,1,0} %add.1023), metadata={op_type="AddV2" op_name="add_4"}
  %maximum.1027 = f32[1,28,28,512]{3,2,1,0} maximum(f32[1,28,28,512]{3,2,1,0} %broadcast.1026, f32[1,28,28,512]{3,2,1,0} %add.1024), metadata={op_type="Relu" op_name="stage2_unit2_relu"}
  %constant.1042 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit3_bn3/add"}
  %broadcast.1043 = f32[512]{0} broadcast(f32[] %constant.1042), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit3_bn3/add"}
  %arg18.19 = f32[512]{0} parameter(18), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.143 = f32[512]{0} reshape(f32[512]{0} %arg18.19)
  %add.1044 = f32[512]{0} add(f32[512]{0} %broadcast.1043, f32[512]{0} %reshape.143), metadata={op_type="AddV2" op_name="stage2_unit3_bn3/add"}
  %rsqrt.1045 = f32[512]{0} rsqrt(f32[512]{0} %add.1044), metadata={op_type="Rsqrt" op_name="stage2_unit3_bn3/Rsqrt"}
  %arg46.47 = f32[512]{0} parameter(46), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.171 = f32[512]{0} reshape(f32[512]{0} %arg46.47)
  %multiply.1046 = f32[512]{0} multiply(f32[512]{0} %rsqrt.1045, f32[512]{0} %reshape.171), metadata={op_type="Mul" op_name="stage2_unit3_bn3/mul"}
  %broadcast.1164 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.1046), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit3_bn3/mul_1"}
  %constant.1160 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit3_relu2"}
  %broadcast.1161 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[] %constant.1160), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit3_relu2"}
  %constant.1054 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit3_relu1"}
  %broadcast.1055 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[] %constant.1054), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit3_relu1"}
  %constant.1028 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit3_bn1/add"}
  %broadcast.1029 = f32[256]{0} broadcast(f32[] %constant.1028), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit3_bn1/add"}
  %arg1.2 = f32[256]{0} parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.126 = f32[256]{0} reshape(f32[256]{0} %arg1.2)
  %add.1030 = f32[256]{0} add(f32[256]{0} %broadcast.1029, f32[256]{0} %reshape.126), metadata={op_type="AddV2" op_name="stage2_unit3_bn1/add"}
  %rsqrt.1031 = f32[256]{0} rsqrt(f32[256]{0} %add.1030), metadata={op_type="Rsqrt" op_name="stage2_unit3_bn1/Rsqrt"}
  %arg33.34 = f32[256]{0} parameter(33), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.158 = f32[256]{0} reshape(f32[256]{0} %arg33.34)
  %multiply.1032 = f32[256]{0} multiply(f32[256]{0} %rsqrt.1031, f32[256]{0} %reshape.158), metadata={op_type="Mul" op_name="stage2_unit3_bn1/mul"}
  %broadcast.1050 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.1032), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit3_bn1/mul_1"}
  %arg119.120 = f32[1,1,512,256]{3,2,1,0} parameter(119), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.244 = f32[1,1,512,256]{3,2,1,0} reshape(f32[1,1,512,256]{3,2,1,0} %arg119.120)
  %convolution.1049 = f32[1,28,28,256]{3,2,1,0} convolution(f32[1,28,28,512]{3,2,1,0} %maximum.1027, f32[1,1,512,256]{3,2,1,0} %reshape.244), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv1"}
  %multiply.1051 = f32[1,28,28,256]{3,2,1,0} multiply(f32[1,28,28,256]{3,2,1,0} %broadcast.1050, f32[1,28,28,256]{3,2,1,0} %convolution.1049), metadata={op_type="Mul" op_name="stage2_unit3_bn1/mul_1"}
  %arg82.83 = f32[256]{0} parameter(82), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.207 = f32[256]{0} reshape(f32[256]{0} %arg82.83)
  %arg58.59 = f32[256]{0} parameter(58), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.183 = f32[256]{0} reshape(f32[256]{0} %arg58.59)
  %multiply.1033 = f32[256]{0} multiply(f32[256]{0} %multiply.1032, f32[256]{0} %reshape.183), metadata={op_type="Mul" op_name="stage2_unit3_bn1/mul_2"}
  %subtract.1034 = f32[256]{0} subtract(f32[256]{0} %reshape.207, f32[256]{0} %multiply.1033), metadata={op_type="Sub" op_name="stage2_unit3_bn1/sub"}
  %broadcast.1052 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.1034), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit3_bn1/add_1"}
  %add.1053 = f32[1,28,28,256]{3,2,1,0} add(f32[1,28,28,256]{3,2,1,0} %multiply.1051, f32[1,28,28,256]{3,2,1,0} %broadcast.1052), metadata={op_type="AddV2" op_name="stage2_unit3_bn1/add_1"}
  %maximum.1056 = f32[1,28,28,256]{3,2,1,0} maximum(f32[1,28,28,256]{3,2,1,0} %broadcast.1055, f32[1,28,28,256]{3,2,1,0} %add.1053), metadata={op_type="Relu" op_name="stage2_unit3_relu1"}
  %constant.1057 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_6"}
  %pad.1058 = f32[1,30,30,256]{3,2,1,0} pad(f32[1,28,28,256]{3,2,1,0} %maximum.1056, f32[] %constant.1057), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_6"}
  %slice.1059 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1058), slice={[0:1], [0:30], [0:30], [0:8]}, metadata={op_type="Split" op_name="split_11"}
  %arg4.5 = f32[3,3,8,256]{3,2,1,0} parameter(4), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.129 = f32[3,3,8,256]{3,2,1,0} reshape(f32[3,3,8,256]{3,2,1,0} %arg4.5)
  %slice.1091 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.129), slice={[0:3], [0:3], [0:8], [0:8]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1123 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1059, f32[3,3,8,8]{3,2,1,0} %slice.1091), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2"}
  %slice.1060 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1058), slice={[0:1], [0:30], [0:30], [8:16]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1092 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.129), slice={[0:3], [0:3], [0:8], [8:16]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1124 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1060, f32[3,3,8,8]{3,2,1,0} %slice.1092), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_1"}
  %slice.1061 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1058), slice={[0:1], [0:30], [0:30], [16:24]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1093 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.129), slice={[0:3], [0:3], [0:8], [16:24]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1135 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1061, f32[3,3,8,8]{3,2,1,0} %slice.1093), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_2"}
  %slice.1062 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1058), slice={[0:1], [0:30], [0:30], [24:32]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1094 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.129), slice={[0:3], [0:3], [0:8], [24:32]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1146 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1062, f32[3,3,8,8]{3,2,1,0} %slice.1094), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_3"}
  %slice.1063 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1058), slice={[0:1], [0:30], [0:30], [32:40]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1095 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.129), slice={[0:3], [0:3], [0:8], [32:40]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1149 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1063, f32[3,3,8,8]{3,2,1,0} %slice.1095), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_4"}
  %slice.1064 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1058), slice={[0:1], [0:30], [0:30], [40:48]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1096 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.129), slice={[0:3], [0:3], [0:8], [40:48]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1150 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1064, f32[3,3,8,8]{3,2,1,0} %slice.1096), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_5"}
  %slice.1065 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1058), slice={[0:1], [0:30], [0:30], [48:56]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1097 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.129), slice={[0:3], [0:3], [0:8], [48:56]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1151 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1065, f32[3,3,8,8]{3,2,1,0} %slice.1097), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_6"}
  %slice.1066 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1058), slice={[0:1], [0:30], [0:30], [56:64]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1098 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.129), slice={[0:3], [0:3], [0:8], [56:64]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1152 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1066, f32[3,3,8,8]{3,2,1,0} %slice.1098), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_7"}
  %slice.1067 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1058), slice={[0:1], [0:30], [0:30], [64:72]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1099 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.129), slice={[0:3], [0:3], [0:8], [64:72]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1153 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1067, f32[3,3,8,8]{3,2,1,0} %slice.1099), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_8"}
  %slice.1068 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1058), slice={[0:1], [0:30], [0:30], [72:80]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1100 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.129), slice={[0:3], [0:3], [0:8], [72:80]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1154 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1068, f32[3,3,8,8]{3,2,1,0} %slice.1100), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_9"}
  %slice.1069 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1058), slice={[0:1], [0:30], [0:30], [80:88]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1101 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.129), slice={[0:3], [0:3], [0:8], [80:88]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1125 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1069, f32[3,3,8,8]{3,2,1,0} %slice.1101), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_10"}
  %slice.1070 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1058), slice={[0:1], [0:30], [0:30], [88:96]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1102 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.129), slice={[0:3], [0:3], [0:8], [88:96]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1126 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1070, f32[3,3,8,8]{3,2,1,0} %slice.1102), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_11"}
  %slice.1071 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1058), slice={[0:1], [0:30], [0:30], [96:104]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1103 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.129), slice={[0:3], [0:3], [0:8], [96:104]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1127 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1071, f32[3,3,8,8]{3,2,1,0} %slice.1103), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_12"}
  %slice.1072 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1058), slice={[0:1], [0:30], [0:30], [104:112]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1104 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.129), slice={[0:3], [0:3], [0:8], [104:112]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1128 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1072, f32[3,3,8,8]{3,2,1,0} %slice.1104), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_13"}
  %slice.1073 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1058), slice={[0:1], [0:30], [0:30], [112:120]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1105 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.129), slice={[0:3], [0:3], [0:8], [112:120]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1129 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1073, f32[3,3,8,8]{3,2,1,0} %slice.1105), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_14"}
  %slice.1074 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1058), slice={[0:1], [0:30], [0:30], [120:128]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1106 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.129), slice={[0:3], [0:3], [0:8], [120:128]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1130 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1074, f32[3,3,8,8]{3,2,1,0} %slice.1106), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_15"}
  %slice.1075 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1058), slice={[0:1], [0:30], [0:30], [128:136]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1107 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.129), slice={[0:3], [0:3], [0:8], [128:136]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1131 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1075, f32[3,3,8,8]{3,2,1,0} %slice.1107), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_16"}
  %slice.1076 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1058), slice={[0:1], [0:30], [0:30], [136:144]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1108 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.129), slice={[0:3], [0:3], [0:8], [136:144]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1132 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1076, f32[3,3,8,8]{3,2,1,0} %slice.1108), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_17"}
  %slice.1077 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1058), slice={[0:1], [0:30], [0:30], [144:152]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1109 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.129), slice={[0:3], [0:3], [0:8], [144:152]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1133 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1077, f32[3,3,8,8]{3,2,1,0} %slice.1109), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_18"}
  %slice.1078 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1058), slice={[0:1], [0:30], [0:30], [152:160]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1110 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.129), slice={[0:3], [0:3], [0:8], [152:160]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1134 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1078, f32[3,3,8,8]{3,2,1,0} %slice.1110), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_19"}
  %slice.1079 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1058), slice={[0:1], [0:30], [0:30], [160:168]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1111 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.129), slice={[0:3], [0:3], [0:8], [160:168]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1136 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1079, f32[3,3,8,8]{3,2,1,0} %slice.1111), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_20"}
  %slice.1080 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1058), slice={[0:1], [0:30], [0:30], [168:176]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1112 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.129), slice={[0:3], [0:3], [0:8], [168:176]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1137 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1080, f32[3,3,8,8]{3,2,1,0} %slice.1112), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_21"}
  %slice.1081 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1058), slice={[0:1], [0:30], [0:30], [176:184]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1113 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.129), slice={[0:3], [0:3], [0:8], [176:184]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1138 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1081, f32[3,3,8,8]{3,2,1,0} %slice.1113), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_22"}
  %slice.1082 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1058), slice={[0:1], [0:30], [0:30], [184:192]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1114 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.129), slice={[0:3], [0:3], [0:8], [184:192]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1139 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1082, f32[3,3,8,8]{3,2,1,0} %slice.1114), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_23"}
  %slice.1083 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1058), slice={[0:1], [0:30], [0:30], [192:200]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1115 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.129), slice={[0:3], [0:3], [0:8], [192:200]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1140 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1083, f32[3,3,8,8]{3,2,1,0} %slice.1115), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_24"}
  %slice.1084 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1058), slice={[0:1], [0:30], [0:30], [200:208]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1116 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.129), slice={[0:3], [0:3], [0:8], [200:208]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1141 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1084, f32[3,3,8,8]{3,2,1,0} %slice.1116), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_25"}
  %slice.1085 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1058), slice={[0:1], [0:30], [0:30], [208:216]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1117 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.129), slice={[0:3], [0:3], [0:8], [208:216]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1142 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1085, f32[3,3,8,8]{3,2,1,0} %slice.1117), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_26"}
  %slice.1086 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1058), slice={[0:1], [0:30], [0:30], [216:224]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1118 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.129), slice={[0:3], [0:3], [0:8], [216:224]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1143 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1086, f32[3,3,8,8]{3,2,1,0} %slice.1118), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_27"}
  %slice.1087 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1058), slice={[0:1], [0:30], [0:30], [224:232]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1119 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.129), slice={[0:3], [0:3], [0:8], [224:232]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1144 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1087, f32[3,3,8,8]{3,2,1,0} %slice.1119), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_28"}
  %slice.1088 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1058), slice={[0:1], [0:30], [0:30], [232:240]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1120 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.129), slice={[0:3], [0:3], [0:8], [232:240]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1145 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1088, f32[3,3,8,8]{3,2,1,0} %slice.1120), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_29"}
  %slice.1089 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1058), slice={[0:1], [0:30], [0:30], [240:248]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1121 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.129), slice={[0:3], [0:3], [0:8], [240:248]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1147 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1089, f32[3,3,8,8]{3,2,1,0} %slice.1121), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_30"}
  %slice.1090 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1058), slice={[0:1], [0:30], [0:30], [248:256]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1122 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.129), slice={[0:3], [0:3], [0:8], [248:256]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1148 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1090, f32[3,3,8,8]{3,2,1,0} %slice.1122), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_31"}
  %concatenate.1155 = f32[1,28,28,256]{3,2,1,0} concatenate(f32[1,28,28,8]{3,2,1,0} %convolution.1123, f32[1,28,28,8]{3,2,1,0} %convolution.1124, f32[1,28,28,8]{3,2,1,0} %convolution.1135, f32[1,28,28,8]{3,2,1,0} %convolution.1146, f32[1,28,28,8]{3,2,1,0} %convolution.1149, f32[1,28,28,8]{3,2,1,0} %convolution.1150, f32[1,28,28,8]{3,2,1,0} %convolution.1151, f32[1,28,28,8]{3,2,1,0} %convolution.1152, f32[1,28,28,8]{3,2,1,0} %convolution.1153, f32[1,28,28,8]{3,2,1,0} %convolution.1154, f32[1,28,28,8]{3,2,1,0} %convolution.1125, f32[1,28,28,8]{3,2,1,0} %convolution.1126, f32[1,28,28,8]{3,2,1,0} %convolution.1127, f32[1,28,28,8]{3,2,1,0} %convolution.1128, f32[1,28,28,8]{3,2,1,0} %convolution.1129, f32[1,28,28,8]{3,2,1,0} %convolution.1130, f32[1,28,28,8]{3,2,1,0} %convolution.1131, f32[1,28,28,8]{3,2,1,0} %convolution.1132, f32[1,28,28,8]{3,2,1,0} %convolution.1133, f32[1,28,28,8]{3,2,1,0} %convolution.1134, f32[1,28,28,8]{3,2,1,0} %convolution.1136, f32[1,28,28,8]{3,2,1,0} %convolution.1137, f32[1,28,28,8]{3,2,1,0} %convolution.1138, f32[1,28,28,8]{3,2,1,0} %convolution.1139, f32[1,28,28,8]{3,2,1,0} %convolution.1140, f32[1,28,28,8]{3,2,1,0} %convolution.1141, f32[1,28,28,8]{3,2,1,0} %convolution.1142, f32[1,28,28,8]{3,2,1,0} %convolution.1143, f32[1,28,28,8]{3,2,1,0} %convolution.1144, f32[1,28,28,8]{3,2,1,0} %convolution.1145, f32[1,28,28,8]{3,2,1,0} %convolution.1147, f32[1,28,28,8]{3,2,1,0} %convolution.1148), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_5"}
  %constant.1035 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit3_bn2/add"}
  %broadcast.1036 = f32[256]{0} broadcast(f32[] %constant.1035), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit3_bn2/add"}
  %arg13.14 = f32[256]{0} parameter(13), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.138 = f32[256]{0} reshape(f32[256]{0} %arg13.14)
  %add.1037 = f32[256]{0} add(f32[256]{0} %broadcast.1036, f32[256]{0} %reshape.138), metadata={op_type="AddV2" op_name="stage2_unit3_bn2/add"}
  %rsqrt.1038 = f32[256]{0} rsqrt(f32[256]{0} %add.1037), metadata={op_type="Rsqrt" op_name="stage2_unit3_bn2/Rsqrt"}
  %arg42.43 = f32[256]{0} parameter(42), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.167 = f32[256]{0} reshape(f32[256]{0} %arg42.43)
  %multiply.1039 = f32[256]{0} multiply(f32[256]{0} %rsqrt.1038, f32[256]{0} %reshape.167), metadata={op_type="Mul" op_name="stage2_unit3_bn2/mul"}
  %broadcast.1156 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.1039), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit3_bn2/mul_1"}
  %multiply.1157 = f32[1,28,28,256]{3,2,1,0} multiply(f32[1,28,28,256]{3,2,1,0} %concatenate.1155, f32[1,28,28,256]{3,2,1,0} %broadcast.1156), metadata={op_type="Mul" op_name="stage2_unit3_bn2/mul_1"}
  %arg91.92 = f32[256]{0} parameter(91), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.216 = f32[256]{0} reshape(f32[256]{0} %arg91.92)
  %arg67.68 = f32[256]{0} parameter(67), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.192 = f32[256]{0} reshape(f32[256]{0} %arg67.68)
  %multiply.1040 = f32[256]{0} multiply(f32[256]{0} %multiply.1039, f32[256]{0} %reshape.192), metadata={op_type="Mul" op_name="stage2_unit3_bn2/mul_2"}
  %subtract.1041 = f32[256]{0} subtract(f32[256]{0} %reshape.216, f32[256]{0} %multiply.1040), metadata={op_type="Sub" op_name="stage2_unit3_bn2/sub"}
  %broadcast.1158 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.1041), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit3_bn2/add_1"}
  %add.1159 = f32[1,28,28,256]{3,2,1,0} add(f32[1,28,28,256]{3,2,1,0} %multiply.1157, f32[1,28,28,256]{3,2,1,0} %broadcast.1158), metadata={op_type="AddV2" op_name="stage2_unit3_bn2/add_1"}
  %maximum.1162 = f32[1,28,28,256]{3,2,1,0} maximum(f32[1,28,28,256]{3,2,1,0} %broadcast.1161, f32[1,28,28,256]{3,2,1,0} %add.1159), metadata={op_type="Relu" op_name="stage2_unit3_relu2"}
  %arg120.121 = f32[1,1,256,512]{3,2,1,0} parameter(120), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.245 = f32[1,1,256,512]{3,2,1,0} reshape(f32[1,1,256,512]{3,2,1,0} %arg120.121)
  %convolution.1163 = f32[1,28,28,512]{3,2,1,0} convolution(f32[1,28,28,256]{3,2,1,0} %maximum.1162, f32[1,1,256,512]{3,2,1,0} %reshape.245), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv3"}
  %multiply.1165 = f32[1,28,28,512]{3,2,1,0} multiply(f32[1,28,28,512]{3,2,1,0} %broadcast.1164, f32[1,28,28,512]{3,2,1,0} %convolution.1163), metadata={op_type="Mul" op_name="stage2_unit3_bn3/mul_1"}
  %arg95.96 = f32[512]{0} parameter(95), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.220 = f32[512]{0} reshape(f32[512]{0} %arg95.96)
  %arg71.72 = f32[512]{0} parameter(71), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.196 = f32[512]{0} reshape(f32[512]{0} %arg71.72)
  %multiply.1047 = f32[512]{0} multiply(f32[512]{0} %multiply.1046, f32[512]{0} %reshape.196), metadata={op_type="Mul" op_name="stage2_unit3_bn3/mul_2"}
  %subtract.1048 = f32[512]{0} subtract(f32[512]{0} %reshape.220, f32[512]{0} %multiply.1047), metadata={op_type="Sub" op_name="stage2_unit3_bn3/sub"}
  %broadcast.1166 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.1048), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit3_bn3/add_1"}
  %add.1167 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %multiply.1165, f32[1,28,28,512]{3,2,1,0} %broadcast.1166), metadata={op_type="AddV2" op_name="stage2_unit3_bn3/add_1"}
  %add.1168 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %maximum.1027, f32[1,28,28,512]{3,2,1,0} %add.1167), metadata={op_type="AddV2" op_name="add_5"}
  %maximum.1171 = f32[1,28,28,512]{3,2,1,0} maximum(f32[1,28,28,512]{3,2,1,0} %broadcast.1170, f32[1,28,28,512]{3,2,1,0} %add.1168), metadata={op_type="Relu" op_name="stage2_unit3_relu"}
  %constant.1186 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit4_bn3/add"}
  %broadcast.1187 = f32[512]{0} broadcast(f32[] %constant.1186), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit4_bn3/add"}
  %arg27.28 = f32[512]{0} parameter(27), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.152 = f32[512]{0} reshape(f32[512]{0} %arg27.28)
  %add.1188 = f32[512]{0} add(f32[512]{0} %broadcast.1187, f32[512]{0} %reshape.152), metadata={op_type="AddV2" op_name="stage2_unit4_bn3/add"}
  %rsqrt.1189 = f32[512]{0} rsqrt(f32[512]{0} %add.1188), metadata={op_type="Rsqrt" op_name="stage2_unit4_bn3/Rsqrt"}
  %arg53.54 = f32[512]{0} parameter(53), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.178 = f32[512]{0} reshape(f32[512]{0} %arg53.54)
  %multiply.1190 = f32[512]{0} multiply(f32[512]{0} %rsqrt.1189, f32[512]{0} %reshape.178), metadata={op_type="Mul" op_name="stage2_unit4_bn3/mul"}
  %broadcast.1308 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.1190), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit4_bn3/mul_1"}
  %constant.1304 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit4_relu2"}
  %broadcast.1305 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[] %constant.1304), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit4_relu2"}
  %constant.1198 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit4_relu1"}
  %broadcast.1199 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[] %constant.1198), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit4_relu1"}
  %constant.1172 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit4_bn1/add"}
  %broadcast.1173 = f32[256]{0} broadcast(f32[] %constant.1172), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit4_bn1/add"}
  %arg22.23 = f32[256]{0} parameter(22), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.147 = f32[256]{0} reshape(f32[256]{0} %arg22.23)
  %add.1174 = f32[256]{0} add(f32[256]{0} %broadcast.1173, f32[256]{0} %reshape.147), metadata={op_type="AddV2" op_name="stage2_unit4_bn1/add"}
  %rsqrt.1175 = f32[256]{0} rsqrt(f32[256]{0} %add.1174), metadata={op_type="Rsqrt" op_name="stage2_unit4_bn1/Rsqrt"}
  %arg50.51 = f32[256]{0} parameter(50), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.175 = f32[256]{0} reshape(f32[256]{0} %arg50.51)
  %multiply.1176 = f32[256]{0} multiply(f32[256]{0} %rsqrt.1175, f32[256]{0} %reshape.175), metadata={op_type="Mul" op_name="stage2_unit4_bn1/mul"}
  %broadcast.1194 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.1176), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit4_bn1/mul_1"}
  %arg121.122 = f32[1,1,512,256]{3,2,1,0} parameter(121), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.246 = f32[1,1,512,256]{3,2,1,0} reshape(f32[1,1,512,256]{3,2,1,0} %arg121.122)
  %convolution.1193 = f32[1,28,28,256]{3,2,1,0} convolution(f32[1,28,28,512]{3,2,1,0} %maximum.1171, f32[1,1,512,256]{3,2,1,0} %reshape.246), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv1"}
  %multiply.1195 = f32[1,28,28,256]{3,2,1,0} multiply(f32[1,28,28,256]{3,2,1,0} %broadcast.1194, f32[1,28,28,256]{3,2,1,0} %convolution.1193), metadata={op_type="Mul" op_name="stage2_unit4_bn1/mul_1"}
  %arg99.100 = f32[256]{0} parameter(99), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.224 = f32[256]{0} reshape(f32[256]{0} %arg99.100)
  %arg75.76 = f32[256]{0} parameter(75), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.200 = f32[256]{0} reshape(f32[256]{0} %arg75.76)
  %multiply.1177 = f32[256]{0} multiply(f32[256]{0} %multiply.1176, f32[256]{0} %reshape.200), metadata={op_type="Mul" op_name="stage2_unit4_bn1/mul_2"}
  %subtract.1178 = f32[256]{0} subtract(f32[256]{0} %reshape.224, f32[256]{0} %multiply.1177), metadata={op_type="Sub" op_name="stage2_unit4_bn1/sub"}
  %broadcast.1196 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.1178), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit4_bn1/add_1"}
  %add.1197 = f32[1,28,28,256]{3,2,1,0} add(f32[1,28,28,256]{3,2,1,0} %multiply.1195, f32[1,28,28,256]{3,2,1,0} %broadcast.1196), metadata={op_type="AddV2" op_name="stage2_unit4_bn1/add_1"}
  %maximum.1200 = f32[1,28,28,256]{3,2,1,0} maximum(f32[1,28,28,256]{3,2,1,0} %broadcast.1199, f32[1,28,28,256]{3,2,1,0} %add.1197), metadata={op_type="Relu" op_name="stage2_unit4_relu1"}
  %constant.1201 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_7"}
  %pad.1202 = f32[1,30,30,256]{3,2,1,0} pad(f32[1,28,28,256]{3,2,1,0} %maximum.1200, f32[] %constant.1201), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_7"}
  %slice.1203 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1202), slice={[0:1], [0:30], [0:30], [0:8]}, metadata={op_type="Split" op_name="split_13"}
  %arg26.27 = f32[3,3,8,256]{3,2,1,0} parameter(26), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.151 = f32[3,3,8,256]{3,2,1,0} reshape(f32[3,3,8,256]{3,2,1,0} %arg26.27)
  %slice.1235 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.151), slice={[0:3], [0:3], [0:8], [0:8]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1267 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1203, f32[3,3,8,8]{3,2,1,0} %slice.1235), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2"}
  %slice.1204 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1202), slice={[0:1], [0:30], [0:30], [8:16]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1236 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.151), slice={[0:3], [0:3], [0:8], [8:16]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1268 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1204, f32[3,3,8,8]{3,2,1,0} %slice.1236), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_1"}
  %slice.1205 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1202), slice={[0:1], [0:30], [0:30], [16:24]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1237 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.151), slice={[0:3], [0:3], [0:8], [16:24]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1279 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1205, f32[3,3,8,8]{3,2,1,0} %slice.1237), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_2"}
  %slice.1206 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1202), slice={[0:1], [0:30], [0:30], [24:32]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1238 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.151), slice={[0:3], [0:3], [0:8], [24:32]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1290 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1206, f32[3,3,8,8]{3,2,1,0} %slice.1238), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_3"}
  %slice.1207 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1202), slice={[0:1], [0:30], [0:30], [32:40]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1239 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.151), slice={[0:3], [0:3], [0:8], [32:40]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1293 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1207, f32[3,3,8,8]{3,2,1,0} %slice.1239), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_4"}
  %slice.1208 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1202), slice={[0:1], [0:30], [0:30], [40:48]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1240 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.151), slice={[0:3], [0:3], [0:8], [40:48]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1294 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1208, f32[3,3,8,8]{3,2,1,0} %slice.1240), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_5"}
  %slice.1209 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1202), slice={[0:1], [0:30], [0:30], [48:56]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1241 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.151), slice={[0:3], [0:3], [0:8], [48:56]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1295 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1209, f32[3,3,8,8]{3,2,1,0} %slice.1241), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_6"}
  %slice.1210 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1202), slice={[0:1], [0:30], [0:30], [56:64]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1242 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.151), slice={[0:3], [0:3], [0:8], [56:64]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1296 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1210, f32[3,3,8,8]{3,2,1,0} %slice.1242), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_7"}
  %slice.1211 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1202), slice={[0:1], [0:30], [0:30], [64:72]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1243 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.151), slice={[0:3], [0:3], [0:8], [64:72]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1297 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1211, f32[3,3,8,8]{3,2,1,0} %slice.1243), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_8"}
  %slice.1212 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1202), slice={[0:1], [0:30], [0:30], [72:80]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1244 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.151), slice={[0:3], [0:3], [0:8], [72:80]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1298 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1212, f32[3,3,8,8]{3,2,1,0} %slice.1244), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_9"}
  %slice.1213 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1202), slice={[0:1], [0:30], [0:30], [80:88]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1245 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.151), slice={[0:3], [0:3], [0:8], [80:88]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1269 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1213, f32[3,3,8,8]{3,2,1,0} %slice.1245), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_10"}
  %slice.1214 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1202), slice={[0:1], [0:30], [0:30], [88:96]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1246 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.151), slice={[0:3], [0:3], [0:8], [88:96]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1270 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1214, f32[3,3,8,8]{3,2,1,0} %slice.1246), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_11"}
  %slice.1215 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1202), slice={[0:1], [0:30], [0:30], [96:104]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1247 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.151), slice={[0:3], [0:3], [0:8], [96:104]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1271 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1215, f32[3,3,8,8]{3,2,1,0} %slice.1247), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_12"}
  %slice.1216 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1202), slice={[0:1], [0:30], [0:30], [104:112]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1248 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.151), slice={[0:3], [0:3], [0:8], [104:112]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1272 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1216, f32[3,3,8,8]{3,2,1,0} %slice.1248), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_13"}
  %slice.1217 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1202), slice={[0:1], [0:30], [0:30], [112:120]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1249 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.151), slice={[0:3], [0:3], [0:8], [112:120]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1273 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1217, f32[3,3,8,8]{3,2,1,0} %slice.1249), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_14"}
  %slice.1218 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1202), slice={[0:1], [0:30], [0:30], [120:128]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1250 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.151), slice={[0:3], [0:3], [0:8], [120:128]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1274 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1218, f32[3,3,8,8]{3,2,1,0} %slice.1250), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_15"}
  %slice.1219 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1202), slice={[0:1], [0:30], [0:30], [128:136]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1251 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.151), slice={[0:3], [0:3], [0:8], [128:136]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1275 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1219, f32[3,3,8,8]{3,2,1,0} %slice.1251), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_16"}
  %slice.1220 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1202), slice={[0:1], [0:30], [0:30], [136:144]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1252 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.151), slice={[0:3], [0:3], [0:8], [136:144]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1276 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1220, f32[3,3,8,8]{3,2,1,0} %slice.1252), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_17"}
  %slice.1221 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1202), slice={[0:1], [0:30], [0:30], [144:152]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1253 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.151), slice={[0:3], [0:3], [0:8], [144:152]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1277 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1221, f32[3,3,8,8]{3,2,1,0} %slice.1253), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_18"}
  %slice.1222 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1202), slice={[0:1], [0:30], [0:30], [152:160]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1254 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.151), slice={[0:3], [0:3], [0:8], [152:160]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1278 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1222, f32[3,3,8,8]{3,2,1,0} %slice.1254), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_19"}
  %slice.1223 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1202), slice={[0:1], [0:30], [0:30], [160:168]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1255 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.151), slice={[0:3], [0:3], [0:8], [160:168]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1280 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1223, f32[3,3,8,8]{3,2,1,0} %slice.1255), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_20"}
  %slice.1224 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1202), slice={[0:1], [0:30], [0:30], [168:176]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1256 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.151), slice={[0:3], [0:3], [0:8], [168:176]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1281 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1224, f32[3,3,8,8]{3,2,1,0} %slice.1256), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_21"}
  %slice.1225 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1202), slice={[0:1], [0:30], [0:30], [176:184]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1257 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.151), slice={[0:3], [0:3], [0:8], [176:184]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1282 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1225, f32[3,3,8,8]{3,2,1,0} %slice.1257), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_22"}
  %slice.1226 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1202), slice={[0:1], [0:30], [0:30], [184:192]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1258 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.151), slice={[0:3], [0:3], [0:8], [184:192]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1283 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1226, f32[3,3,8,8]{3,2,1,0} %slice.1258), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_23"}
  %slice.1227 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1202), slice={[0:1], [0:30], [0:30], [192:200]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1259 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.151), slice={[0:3], [0:3], [0:8], [192:200]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1284 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1227, f32[3,3,8,8]{3,2,1,0} %slice.1259), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_24"}
  %slice.1228 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1202), slice={[0:1], [0:30], [0:30], [200:208]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1260 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.151), slice={[0:3], [0:3], [0:8], [200:208]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1285 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1228, f32[3,3,8,8]{3,2,1,0} %slice.1260), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_25"}
  %slice.1229 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1202), slice={[0:1], [0:30], [0:30], [208:216]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1261 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.151), slice={[0:3], [0:3], [0:8], [208:216]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1286 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1229, f32[3,3,8,8]{3,2,1,0} %slice.1261), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_26"}
  %slice.1230 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1202), slice={[0:1], [0:30], [0:30], [216:224]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1262 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.151), slice={[0:3], [0:3], [0:8], [216:224]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1287 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1230, f32[3,3,8,8]{3,2,1,0} %slice.1262), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_27"}
  %slice.1231 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1202), slice={[0:1], [0:30], [0:30], [224:232]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1263 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.151), slice={[0:3], [0:3], [0:8], [224:232]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1288 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1231, f32[3,3,8,8]{3,2,1,0} %slice.1263), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_28"}
  %slice.1232 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1202), slice={[0:1], [0:30], [0:30], [232:240]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1264 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.151), slice={[0:3], [0:3], [0:8], [232:240]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1289 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1232, f32[3,3,8,8]{3,2,1,0} %slice.1264), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_29"}
  %slice.1233 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1202), slice={[0:1], [0:30], [0:30], [240:248]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1265 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.151), slice={[0:3], [0:3], [0:8], [240:248]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1291 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1233, f32[3,3,8,8]{3,2,1,0} %slice.1265), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_30"}
  %slice.1234 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1202), slice={[0:1], [0:30], [0:30], [248:256]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1266 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.151), slice={[0:3], [0:3], [0:8], [248:256]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1292 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1234, f32[3,3,8,8]{3,2,1,0} %slice.1266), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_31"}
  %concatenate.1299 = f32[1,28,28,256]{3,2,1,0} concatenate(f32[1,28,28,8]{3,2,1,0} %convolution.1267, f32[1,28,28,8]{3,2,1,0} %convolution.1268, f32[1,28,28,8]{3,2,1,0} %convolution.1279, f32[1,28,28,8]{3,2,1,0} %convolution.1290, f32[1,28,28,8]{3,2,1,0} %convolution.1293, f32[1,28,28,8]{3,2,1,0} %convolution.1294, f32[1,28,28,8]{3,2,1,0} %convolution.1295, f32[1,28,28,8]{3,2,1,0} %convolution.1296, f32[1,28,28,8]{3,2,1,0} %convolution.1297, f32[1,28,28,8]{3,2,1,0} %convolution.1298, f32[1,28,28,8]{3,2,1,0} %convolution.1269, f32[1,28,28,8]{3,2,1,0} %convolution.1270, f32[1,28,28,8]{3,2,1,0} %convolution.1271, f32[1,28,28,8]{3,2,1,0} %convolution.1272, f32[1,28,28,8]{3,2,1,0} %convolution.1273, f32[1,28,28,8]{3,2,1,0} %convolution.1274, f32[1,28,28,8]{3,2,1,0} %convolution.1275, f32[1,28,28,8]{3,2,1,0} %convolution.1276, f32[1,28,28,8]{3,2,1,0} %convolution.1277, f32[1,28,28,8]{3,2,1,0} %convolution.1278, f32[1,28,28,8]{3,2,1,0} %convolution.1280, f32[1,28,28,8]{3,2,1,0} %convolution.1281, f32[1,28,28,8]{3,2,1,0} %convolution.1282, f32[1,28,28,8]{3,2,1,0} %convolution.1283, f32[1,28,28,8]{3,2,1,0} %convolution.1284, f32[1,28,28,8]{3,2,1,0} %convolution.1285, f32[1,28,28,8]{3,2,1,0} %convolution.1286, f32[1,28,28,8]{3,2,1,0} %convolution.1287, f32[1,28,28,8]{3,2,1,0} %convolution.1288, f32[1,28,28,8]{3,2,1,0} %convolution.1289, f32[1,28,28,8]{3,2,1,0} %convolution.1291, f32[1,28,28,8]{3,2,1,0} %convolution.1292), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_6"}
  %constant.1179 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit4_bn2/add"}
  %broadcast.1180 = f32[256]{0} broadcast(f32[] %constant.1179), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit4_bn2/add"}
  %arg6.7 = f32[256]{0} parameter(6), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.131 = f32[256]{0} reshape(f32[256]{0} %arg6.7)
  %add.1181 = f32[256]{0} add(f32[256]{0} %broadcast.1180, f32[256]{0} %reshape.131), metadata={op_type="AddV2" op_name="stage2_unit4_bn2/add"}
  %rsqrt.1182 = f32[256]{0} rsqrt(f32[256]{0} %add.1181), metadata={op_type="Rsqrt" op_name="stage2_unit4_bn2/Rsqrt"}
  %arg37.38 = f32[256]{0} parameter(37), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.162 = f32[256]{0} reshape(f32[256]{0} %arg37.38)
  %multiply.1183 = f32[256]{0} multiply(f32[256]{0} %rsqrt.1182, f32[256]{0} %reshape.162), metadata={op_type="Mul" op_name="stage2_unit4_bn2/mul"}
  %broadcast.1300 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.1183), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit4_bn2/mul_1"}
  %multiply.1301 = f32[1,28,28,256]{3,2,1,0} multiply(f32[1,28,28,256]{3,2,1,0} %concatenate.1299, f32[1,28,28,256]{3,2,1,0} %broadcast.1300), metadata={op_type="Mul" op_name="stage2_unit4_bn2/mul_1"}
  %arg86.87 = f32[256]{0} parameter(86), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.211 = f32[256]{0} reshape(f32[256]{0} %arg86.87)
  %arg62.63 = f32[256]{0} parameter(62), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.187 = f32[256]{0} reshape(f32[256]{0} %arg62.63)
  %multiply.1184 = f32[256]{0} multiply(f32[256]{0} %multiply.1183, f32[256]{0} %reshape.187), metadata={op_type="Mul" op_name="stage2_unit4_bn2/mul_2"}
  %subtract.1185 = f32[256]{0} subtract(f32[256]{0} %reshape.211, f32[256]{0} %multiply.1184), metadata={op_type="Sub" op_name="stage2_unit4_bn2/sub"}
  %broadcast.1302 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.1185), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit4_bn2/add_1"}
  %add.1303 = f32[1,28,28,256]{3,2,1,0} add(f32[1,28,28,256]{3,2,1,0} %multiply.1301, f32[1,28,28,256]{3,2,1,0} %broadcast.1302), metadata={op_type="AddV2" op_name="stage2_unit4_bn2/add_1"}
  %maximum.1306 = f32[1,28,28,256]{3,2,1,0} maximum(f32[1,28,28,256]{3,2,1,0} %broadcast.1305, f32[1,28,28,256]{3,2,1,0} %add.1303), metadata={op_type="Relu" op_name="stage2_unit4_relu2"}
  %arg122.123 = f32[1,1,256,512]{3,2,1,0} parameter(122), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.247 = f32[1,1,256,512]{3,2,1,0} reshape(f32[1,1,256,512]{3,2,1,0} %arg122.123)
  %convolution.1307 = f32[1,28,28,512]{3,2,1,0} convolution(f32[1,28,28,256]{3,2,1,0} %maximum.1306, f32[1,1,256,512]{3,2,1,0} %reshape.247), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv3"}
  %multiply.1309 = f32[1,28,28,512]{3,2,1,0} multiply(f32[1,28,28,512]{3,2,1,0} %broadcast.1308, f32[1,28,28,512]{3,2,1,0} %convolution.1307), metadata={op_type="Mul" op_name="stage2_unit4_bn3/mul_1"}
  %arg102.103 = f32[512]{0} parameter(102), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.227 = f32[512]{0} reshape(f32[512]{0} %arg102.103)
  %arg78.79 = f32[512]{0} parameter(78), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.203 = f32[512]{0} reshape(f32[512]{0} %arg78.79)
  %multiply.1191 = f32[512]{0} multiply(f32[512]{0} %multiply.1190, f32[512]{0} %reshape.203), metadata={op_type="Mul" op_name="stage2_unit4_bn3/mul_2"}
  %subtract.1192 = f32[512]{0} subtract(f32[512]{0} %reshape.227, f32[512]{0} %multiply.1191), metadata={op_type="Sub" op_name="stage2_unit4_bn3/sub"}
  %broadcast.1310 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.1192), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit4_bn3/add_1"}
  %add.1311 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %multiply.1309, f32[1,28,28,512]{3,2,1,0} %broadcast.1310), metadata={op_type="AddV2" op_name="stage2_unit4_bn3/add_1"}
  %add.1312 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %maximum.1171, f32[1,28,28,512]{3,2,1,0} %add.1311), metadata={op_type="AddV2" op_name="add_6"}
  %maximum.1315 = f32[1,28,28,512]{3,2,1,0} maximum(f32[1,28,28,512]{3,2,1,0} %broadcast.1314, f32[1,28,28,512]{3,2,1,0} %add.1312), metadata={op_type="Relu" op_name="stage2_unit4_relu"}
  %reshape.1316 = f32[1,28,28,512]{3,2,1,0} reshape(f32[1,28,28,512]{3,2,1,0} %maximum.1315), metadata={op_name="XLA_Retvals"}
  %tuple.1317 = (f32[1,28,28,512]{3,2,1,0}) tuple(f32[1,28,28,512]{3,2,1,0} %reshape.1316), metadata={op_name="XLA_Retvals"}
  ROOT %get-tuple-element.1318 = f32[1,28,28,512]{3,2,1,0} get-tuple-element((f32[1,28,28,512]{3,2,1,0}) %tuple.1317), index=0, metadata={op_name="XLA_Retvals"}
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
