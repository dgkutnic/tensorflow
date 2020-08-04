// Tests that show HLO Module conversion to PlaidML Program.

#include <algorithm>
#include <string>

#include <gtest/gtest.h>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/service/plaidml/compiler.h"
#include "tensorflow/compiler/xla/service/plaidml/tests/plaidml_codegen_test.h"
#include "tensorflow/compiler/xla/service/plaidml/tests/resnext50_pretrained_inputs_and_weights.h"
#include "tensorflow/compiler/xla/service/plaidml/tests/resnext50_stage3_output.h"
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
    {0}, // stage3_unit6_relu
    ::weights::stage3_unit6_bn3_mean, //
    ::weights::stage3_unit6_bn3_scale, //
    ::weights::stage3_unit6_bn3_var, //
    {2e-05}, // stage3_unit6_bn3/add
    ::weights::stage3_unit6_bn3_bias, //
    ::weights::stage3_unit6_conv3_weight, //
    {0}, // stage3_unit6_relu2
    ::weights::stage3_unit6_bn2_mean, //
    ::weights::stage3_unit6_bn2_scale, //
    ::weights::stage3_unit6_bn2_var, //
    {2e-05}, // stage3_unit6_bn2/add
    ::weights::stage3_unit6_bn2_bias, //
    ::weights::stage3_unit6_conv2_weight, //
    {0}, // stage3_unit6_relu1
    ::weights::stage3_unit6_bn1_mean, //
    ::weights::stage3_unit6_bn1_scale, //
    ::weights::stage3_unit6_bn1_var, //
    {2e-05}, // stage3_unit6_bn1/add
    ::weights::stage3_unit6_bn1_bias, //
    ::weights::stage3_unit6_conv1_weight, //
    {0}, // stage3_unit5_relu
    ::weights::stage3_unit5_bn3_mean, //
    ::weights::stage3_unit5_bn3_scale, //
    ::weights::stage3_unit5_bn3_var, //
    {2e-05}, // stage3_unit5_bn3/add
    ::weights::stage3_unit5_bn3_bias, //
    ::weights::stage3_unit5_conv3_weight, //
    {0}, // stage3_unit5_relu2
    ::weights::stage3_unit5_bn2_mean, //
    ::weights::stage3_unit5_bn2_scale, //
    ::weights::stage3_unit5_bn2_var, //
    {2e-05}, // stage3_unit5_bn2/add
    ::weights::stage3_unit5_bn2_bias, //
    ::weights::stage3_unit5_conv2_weight, //
    {0}, // stage3_unit5_relu1
    ::weights::stage3_unit5_bn1_mean, //
    ::weights::stage3_unit5_bn1_scale, //
    ::weights::stage3_unit5_bn1_var, //
    {2e-05}, // stage3_unit5_bn1/add
    ::weights::stage3_unit5_bn1_bias, //
    ::weights::stage3_unit5_conv1_weight, //
    {0}, // stage3_unit4_relu
    ::weights::stage3_unit4_bn3_mean, //
    ::weights::stage3_unit4_bn3_scale, //
    ::weights::stage3_unit4_bn3_var, //
    {2e-05}, // stage3_unit4_bn3/add
    ::weights::stage3_unit4_bn3_bias, //
    ::weights::stage3_unit4_conv3_weight, //
    {0}, // stage3_unit4_relu2
    ::weights::stage3_unit4_bn2_mean, //
    ::weights::stage3_unit4_bn2_scale, //
    ::weights::stage3_unit4_bn2_var, //
    {2e-05}, // stage3_unit4_bn2/add
    ::weights::stage3_unit4_bn2_bias, //
    ::weights::stage3_unit4_conv2_weight, //
    {0}, // stage3_unit4_relu1
    ::weights::stage3_unit4_bn1_mean, //
    ::weights::stage3_unit4_bn1_scale, //
    ::weights::stage3_unit4_bn1_var, //
    {2e-05}, // stage3_unit4_bn1/add
    ::weights::stage3_unit4_bn1_bias, //
    ::weights::stage3_unit4_conv1_weight, //
    {0}, // stage3_unit3_relu
    ::weights::stage3_unit3_bn3_mean, //
    ::weights::stage3_unit3_bn3_scale, //
    ::weights::stage3_unit3_bn3_var, //
    {2e-05}, // stage3_unit3_bn3/add
    ::weights::stage3_unit3_bn3_bias, //
    ::weights::stage3_unit3_conv3_weight, //
    {0}, // stage3_unit3_relu2
    ::weights::stage3_unit3_bn2_mean, //
    ::weights::stage3_unit3_bn2_scale, //
    ::weights::stage3_unit3_bn2_var, //
    {2e-05}, // stage3_unit3_bn2/add
    ::weights::stage3_unit3_bn2_bias, //
    ::weights::stage3_unit3_conv2_weight, //
    {0}, // stage3_unit3_relu1
    ::weights::stage3_unit3_bn1_mean, //
    ::weights::stage3_unit3_bn1_scale, //
    ::weights::stage3_unit3_bn1_var, //
    {2e-05}, // stage3_unit3_bn1/add
    ::weights::stage3_unit3_bn1_bias, //
    ::weights::stage3_unit3_conv1_weight, //
    {0}, // stage3_unit2_relu
    ::weights::stage3_unit2_bn3_mean, //
    ::weights::stage3_unit2_bn3_scale, //
    ::weights::stage3_unit2_bn3_var, //
    {2e-05}, // stage3_unit2_bn3/add
    ::weights::stage3_unit2_bn3_bias, //
    ::weights::stage3_unit2_conv3_weight, //
    {0}, // stage3_unit2_relu2
    ::weights::stage3_unit2_bn2_mean, //
    ::weights::stage3_unit2_bn2_scale, //
    ::weights::stage3_unit2_bn2_var, //
    {2e-05}, // stage3_unit2_bn2/add
    ::weights::stage3_unit2_bn2_bias, //
    ::weights::stage3_unit2_conv2_weight, //
    {0}, // stage3_unit2_relu1
    ::weights::stage3_unit2_bn1_mean, //
    ::weights::stage3_unit2_bn1_scale, //
    ::weights::stage3_unit2_bn1_var, //
    {2e-05}, // stage3_unit2_bn1/add
    ::weights::stage3_unit2_bn1_bias, //
    ::weights::stage3_unit2_conv1_weight, //
    {0}, // stage3_unit1_relu
    ::weights::stage3_unit1_sc_bn_mean, //
    ::weights::stage3_unit1_sc_bn_scale, //
    ::weights::stage3_unit1_sc_bn_var, //
    {2e-05}, // stage3_unit1_sc_bn/add
    ::weights::stage3_unit1_sc_bn_bias, //
    ::weights::stage3_unit1_sc_weight, //
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
    ::weights::stage2_unit1_conv1_weight, //
    ::weights::stage3_unit1_bn3_mean, //
    ::weights::stage3_unit1_bn3_scale, //
    ::weights::stage3_unit1_bn3_var, //
    {2e-05}, // stage3_unit1_bn3/add
    ::weights::stage3_unit1_bn3_bias, //
    ::weights::stage3_unit1_conv3_weight, //
    {0}, // stage3_unit1_relu2
    ::weights::stage3_unit1_bn2_mean, //
    ::weights::stage3_unit1_bn2_scale, //
    ::weights::stage3_unit1_bn2_var, //
    {2e-05}, // stage3_unit1_bn2/add
    ::weights::stage3_unit1_bn2_bias, //
    ::weights::stage3_unit1_conv2_weight, //
    {0}, // stage3_unit1_relu1
    ::weights::stage3_unit1_bn1_mean, //
    ::weights::stage3_unit1_bn1_scale, //
    ::weights::stage3_unit1_bn1_var, //
    {2e-05}, // stage3_unit1_bn1/add
    ::weights::stage3_unit1_bn1_bias, //
    ::weights::stage3_unit1_conv1_weight, //
  };

  TestCaseVal ResNeXt50_Outputs = ::outputs::ResNext50_Outputs; 

  TestCasePairs testcase_pairs = {{ResNeXt50_WeightsInputs, ResNeXt50_Outputs}}; 

  ResNeXTTestSpec spec = GetParam();

  HloModuleConfig cfg;

  std::unique_ptr<VerifiedHloModule> hlo_module = absl::make_unique<VerifiedHloModule>(
      "module", cfg, false, false, nullptr);

std::string hlo_text = R"(
HloModule cluster_0__XlaCompiledKernel_true__XlaNumConstantArgs_54__XlaNumResourceArgs_0_.2385

%max_F32.469 (lhs.470: f32[], rhs.471: f32[]) -> f32[] {
  %lhs.470 = f32[] parameter(0)
  %rhs.471 = f32[] parameter(1)
  ROOT %maximum.472 = f32[] maximum(f32[] %lhs.470, f32[] %rhs.471)
}

ENTRY %cluster_0__XlaCompiledKernel_true__XlaNumConstantArgs_54__XlaNumResourceArgs_0_.2385 (arg0.1: f32[3], arg1.2: f32[64], arg2.3: f32[512], arg3.4: f32[128], arg4.5: f32[256], arg5.6: f32[3,3,4,128], arg6.7: f32[128], arg7.8: f32[1024], arg8.9: f32[3,3,16,512], arg9.10: f32[1024], arg10.11: f32[256], arg11.12: f32[128], arg12.13: f32[3,3,4,128], arg13.14: f32[128], arg14.15: f32[512], arg15.16: f32[256], arg16.17: f32[128], arg17.18: f32[3,3,16,512], arg18.19: f32[3,3,4,128], arg19.20: f32[128], arg20.21: f32[256], arg21.22: f32[256], arg22.23: f32[512], arg23.24: f32[3,3,8,256], arg24.25: f32[512], arg25.26: f32[256], arg26.27: f32[512], arg27.28: f32[256], arg28.29: f32[3,3,8,256], arg29.30: f32[1024], arg30.31: f32[256], arg31.32: f32[512], arg32.33: f32[256], arg33.34: f32[3,3,8,256], arg34.35: f32[512], arg35.36: f32[256], arg36.37: f32[512], arg37.38: f32[3,3,16,512], arg38.39: f32[256], arg39.40: f32[512], arg40.41: f32[3,3,8,256], arg41.42: f32[256], arg42.43: f32[512], arg43.44: f32[512], arg44.45: f32[1024], arg45.46: f32[512], arg46.47: f32[3,3,16,512], arg47.48: f32[512], arg48.49: f32[1024], arg49.50: f32[512], arg50.51: f32[1024], arg51.52: f32[3,3,16,512], arg52.53: f32[512], arg53.54: f32[1024], arg54.55: f32[512], arg55.56: f32[512], arg56.57: f32[3,3,16,512], arg57.58: f32[3], arg58.59: f32[64], arg59.60: f32[512], arg60.61: f32[128], arg61.62: f32[256], arg62.63: f32[128], arg63.64: f32[1024], arg64.65: f32[1024], arg65.66: f32[256], arg66.67: f32[128], arg67.68: f32[128], arg68.69: f32[512], arg69.70: f32[256], arg70.71: f32[128], arg71.72: f32[128], arg72.73: f32[256], arg73.74: f32[256], arg74.75: f32[512], arg75.76: f32[512], arg76.77: f32[256], arg77.78: f32[512], arg78.79: f32[256], arg79.80: f32[1024], arg80.81: f32[256], arg81.82: f32[512], arg82.83: f32[256], arg83.84: f32[512], arg84.85: f32[256], arg85.86: f32[512], arg86.87: f32[256], arg87.88: f32[512], arg88.89: f32[256], arg89.90: f32[512], arg90.91: f32[512], arg91.92: f32[1024], arg92.93: f32[512], arg93.94: f32[512], arg94.95: f32[1024], arg95.96: f32[512], arg96.97: f32[1024], arg97.98: f32[512], arg98.99: f32[1024], arg99.100: f32[512], arg100.101: f32[512], arg101.102: f32[3], arg102.103: f32[64], arg103.104: f32[512], arg104.105: f32[128], arg105.106: f32[256], arg106.107: f32[128], arg107.108: f32[1024], arg108.109: f32[1024], arg109.110: f32[256], arg110.111: f32[128], arg111.112: f32[128], arg112.113: f32[512], arg113.114: f32[256], arg114.115: f32[128], arg115.116: f32[128], arg116.117: f32[256], arg117.118: f32[256], arg118.119: f32[512], arg119.120: f32[512], arg120.121: f32[256], arg121.122: f32[512], arg122.123: f32[256], arg123.124: f32[1024], arg124.125: f32[256], arg125.126: f32[512], arg126.127: f32[256], arg127.128: f32[512], arg128.129: f32[256], arg129.130: f32[512], arg130.131: f32[256], arg131.132: f32[512], arg132.133: f32[256], arg133.134: f32[512], arg134.135: f32[512], arg135.136: f32[1024], arg136.137: f32[512], arg137.138: f32[512], arg138.139: f32[1024], arg139.140: f32[512], arg140.141: f32[1024], arg141.142: f32[512], arg142.143: f32[1024], arg143.144: f32[512], arg144.145: f32[512], arg145.146: f32[64], arg146.147: f32[512], arg147.148: f32[128], arg148.149: f32[256], arg149.150: f32[128], arg150.151: f32[1024], arg151.152: f32[1024], arg152.153: f32[256], arg153.154: f32[128], arg154.155: f32[128], arg155.156: f32[512], arg156.157: f32[256], arg157.158: f32[128], arg158.159: f32[128], arg159.160: f32[256], arg160.161: f32[256], arg161.162: f32[512], arg162.163: f32[512], arg163.164: f32[256], arg164.165: f32[512], arg165.166: f32[256], arg166.167: f32[1024], arg167.168: f32[256], arg168.169: f32[512], arg169.170: f32[256], arg170.171: f32[512], arg171.172: f32[256], arg172.173: f32[512], arg173.174: f32[256], arg174.175: f32[512], arg175.176: f32[256], arg176.177: f32[512], arg177.178: f32[512], arg178.179: f32[1024], arg179.180: f32[512], arg180.181: f32[512], arg181.182: f32[1024], arg182.183: f32[512], arg183.184: f32[1024], arg184.185: f32[512], arg185.186: f32[1024], arg186.187: f32[512], arg187.188: f32[512], arg188.189: f32[7,7,3,64], arg189.190: f32[1,1,64,128], arg190.191: f32[1,1,64,256], arg191.192: f32[1,1,128,256], arg192.193: f32[1,1,256,128], arg193.194: f32[1,1,128,256], arg194.195: f32[1,1,256,128], arg195.196: f32[1,1,128,256], arg196.197: f32[1,1,256,256], arg197.198: f32[1,1,256,512], arg198.199: f32[1,1,256,512], arg199.200: f32[1,1,512,256], arg200.201: f32[1,1,256,512], arg201.202: f32[1,1,512,256], arg202.203: f32[1,1,256,512], arg203.204: f32[1,1,512,256], arg204.205: f32[1,1,256,512], arg205.206: f32[1,1,512,512], arg206.207: f32[1,1,512,1024], arg207.208: f32[1,1,512,1024], arg208.209: f32[1,1,1024,512], arg209.210: f32[1,1,512,1024], arg210.211: f32[1,1,1024,512], arg211.212: f32[1,1,512,1024], arg212.213: f32[1,1,1024,512], arg213.214: f32[1,1,512,1024], arg214.215: f32[1,1,1024,512], arg215.216: f32[1,1,512,1024], arg216.217: f32[1,1,1024,512], arg217.218: f32[1,1,512,1024], arg218.219: f32[1,224,224,3]) -> f32[1,14,14,1024] {
  %constant.2379 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit6_relu"}
  %broadcast.2380 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[] %constant.2379), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit6_relu"}
  %constant.2235 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit5_relu"}
  %broadcast.2236 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[] %constant.2235), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit5_relu"}
  %constant.2091 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit4_relu"}
  %broadcast.2092 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[] %constant.2091), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit4_relu"}
  %constant.1947 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit3_relu"}
  %broadcast.1948 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[] %constant.1947), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit3_relu"}
  %constant.1803 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit2_relu"}
  %broadcast.1804 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[] %constant.1803), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit2_relu"}
  %constant.1659 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit1_relu"}
  %broadcast.1660 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[] %constant.1659), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit1_relu"}
  %constant.1520 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit1_bn3/add"}
  %broadcast.1521 = f32[1024]{0} broadcast(f32[] %constant.1520), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit1_bn3/add"}
  %arg48.49 = f32[1024]{0} parameter(48), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.268 = f32[1024]{0} reshape(f32[1024]{0} %arg48.49)
  %add.1522 = f32[1024]{0} add(f32[1024]{0} %broadcast.1521, f32[1024]{0} %reshape.268), metadata={op_type="AddV2" op_name="stage3_unit1_bn3/add"}
  %rsqrt.1523 = f32[1024]{0} rsqrt(f32[1024]{0} %add.1522), metadata={op_type="Rsqrt" op_name="stage3_unit1_bn3/Rsqrt"}
  %arg94.95 = f32[1024]{0} parameter(94), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.314 = f32[1024]{0} reshape(f32[1024]{0} %arg94.95)
  %multiply.1524 = f32[1024]{0} multiply(f32[1024]{0} %rsqrt.1523, f32[1024]{0} %reshape.314), metadata={op_type="Mul" op_name="stage3_unit1_bn3/mul"}
  %broadcast.1642 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %multiply.1524), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit1_bn3/mul_1"}
  %constant.1638 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit1_relu2"}
  %broadcast.1639 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[] %constant.1638), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit1_relu2"}
  %constant.1532 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit1_relu1"}
  %broadcast.1533 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[] %constant.1532), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit1_relu1"}
  %constant.1506 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit1_bn1/add"}
  %broadcast.1507 = f32[512]{0} broadcast(f32[] %constant.1506), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit1_bn1/add"}
  %arg43.44 = f32[512]{0} parameter(43), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.263 = f32[512]{0} reshape(f32[512]{0} %arg43.44)
  %add.1508 = f32[512]{0} add(f32[512]{0} %broadcast.1507, f32[512]{0} %reshape.263), metadata={op_type="AddV2" op_name="stage3_unit1_bn1/add"}
  %rsqrt.1509 = f32[512]{0} rsqrt(f32[512]{0} %add.1508), metadata={op_type="Rsqrt" op_name="stage3_unit1_bn1/Rsqrt"}
  %arg90.91 = f32[512]{0} parameter(90), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.310 = f32[512]{0} reshape(f32[512]{0} %arg90.91)
  %multiply.1510 = f32[512]{0} multiply(f32[512]{0} %rsqrt.1509, f32[512]{0} %reshape.310), metadata={op_type="Mul" op_name="stage3_unit1_bn1/mul"}
  %broadcast.1528 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.1510), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit1_bn1/mul_1"}
  %constant.1503 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit4_relu"}
  %broadcast.1504 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[] %constant.1503), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit4_relu"}
  %constant.1359 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit3_relu"}
  %broadcast.1360 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[] %constant.1359), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit3_relu"}
  %constant.1215 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit2_relu"}
  %broadcast.1216 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[] %constant.1215), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit2_relu"}
  %constant.1071 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit1_relu"}
  %broadcast.1072 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[] %constant.1071), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit1_relu"}
  %constant.932 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit1_bn3/add"}
  %broadcast.933 = f32[512]{0} broadcast(f32[] %constant.932), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit1_bn3/add"}
  %arg26.27 = f32[512]{0} parameter(26), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.246 = f32[512]{0} reshape(f32[512]{0} %arg26.27)
  %add.934 = f32[512]{0} add(f32[512]{0} %broadcast.933, f32[512]{0} %reshape.246), metadata={op_type="AddV2" op_name="stage2_unit1_bn3/add"}
  %rsqrt.935 = f32[512]{0} rsqrt(f32[512]{0} %add.934), metadata={op_type="Rsqrt" op_name="stage2_unit1_bn3/Rsqrt"}
  %arg77.78 = f32[512]{0} parameter(77), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.297 = f32[512]{0} reshape(f32[512]{0} %arg77.78)
  %multiply.936 = f32[512]{0} multiply(f32[512]{0} %rsqrt.935, f32[512]{0} %reshape.297), metadata={op_type="Mul" op_name="stage2_unit1_bn3/mul"}
  %broadcast.1054 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.936), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit1_bn3/mul_1"}
  %constant.1050 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit1_relu2"}
  %broadcast.1051 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[] %constant.1050), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit1_relu2"}
  %constant.944 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit1_relu1"}
  %broadcast.945 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[] %constant.944), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit1_relu1"}
  %constant.918 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit1_bn1/add"}
  %broadcast.919 = f32[256]{0} broadcast(f32[] %constant.918), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit1_bn1/add"}
  %arg21.22 = f32[256]{0} parameter(21), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.241 = f32[256]{0} reshape(f32[256]{0} %arg21.22)
  %add.920 = f32[256]{0} add(f32[256]{0} %broadcast.919, f32[256]{0} %reshape.241), metadata={op_type="AddV2" op_name="stage2_unit1_bn1/add"}
  %rsqrt.921 = f32[256]{0} rsqrt(f32[256]{0} %add.920), metadata={op_type="Rsqrt" op_name="stage2_unit1_bn1/Rsqrt"}
  %arg73.74 = f32[256]{0} parameter(73), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.293 = f32[256]{0} reshape(f32[256]{0} %arg73.74)
  %multiply.922 = f32[256]{0} multiply(f32[256]{0} %rsqrt.921, f32[256]{0} %reshape.293), metadata={op_type="Mul" op_name="stage2_unit1_bn1/mul"}
  %broadcast.940 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.922), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit1_bn1/mul_1"}
  %constant.915 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit3_relu"}
  %broadcast.916 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[] %constant.915), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit3_relu"}
  %constant.771 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit2_relu"}
  %broadcast.772 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[] %constant.771), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit2_relu"}
  %constant.627 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit1_relu"}
  %broadcast.628 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[] %constant.627), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit1_relu"}
  %constant.488 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit1_bn3/add"}
  %broadcast.489 = f32[256]{0} broadcast(f32[] %constant.488), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit1_bn3/add"}
  %arg10.11 = f32[256]{0} parameter(10), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.230 = f32[256]{0} reshape(f32[256]{0} %arg10.11)
  %add.490 = f32[256]{0} add(f32[256]{0} %broadcast.489, f32[256]{0} %reshape.230), metadata={op_type="AddV2" op_name="stage1_unit1_bn3/add"}
  %rsqrt.491 = f32[256]{0} rsqrt(f32[256]{0} %add.490), metadata={op_type="Rsqrt" op_name="stage1_unit1_bn3/Rsqrt"}
  %arg65.66 = f32[256]{0} parameter(65), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.285 = f32[256]{0} reshape(f32[256]{0} %arg65.66)
  %multiply.492 = f32[256]{0} multiply(f32[256]{0} %rsqrt.491, f32[256]{0} %reshape.285), metadata={op_type="Mul" op_name="stage1_unit1_bn3/mul"}
  %broadcast.610 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.492), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit1_bn3/mul_1"}
  %constant.606 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit1_relu2"}
  %broadcast.607 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[] %constant.606), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit1_relu2"}
  %constant.500 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit1_relu1"}
  %broadcast.501 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[] %constant.500), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit1_relu1"}
  %constant.474 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit1_bn1/add"}
  %broadcast.475 = f32[128]{0} broadcast(f32[] %constant.474), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit1_bn1/add"}
  %arg3.4 = f32[128]{0} parameter(3), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.223 = f32[128]{0} reshape(f32[128]{0} %arg3.4)
  %add.476 = f32[128]{0} add(f32[128]{0} %broadcast.475, f32[128]{0} %reshape.223), metadata={op_type="AddV2" op_name="stage1_unit1_bn1/add"}
  %rsqrt.477 = f32[128]{0} rsqrt(f32[128]{0} %add.476), metadata={op_type="Rsqrt" op_name="stage1_unit1_bn1/Rsqrt"}
  %arg60.61 = f32[128]{0} parameter(60), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.280 = f32[128]{0} reshape(f32[128]{0} %arg60.61)
  %multiply.478 = f32[128]{0} multiply(f32[128]{0} %rsqrt.477, f32[128]{0} %reshape.280), metadata={op_type="Mul" op_name="stage1_unit1_bn1/mul"}
  %broadcast.496 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %multiply.478), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit1_bn1/mul_1"}
  %constant.463 = f32[] constant(0), metadata={op_type="Relu" op_name="relu0"}
  %broadcast.464 = f32[1,112,112,64]{3,2,1,0} broadcast(f32[] %constant.463), dimensions={}, metadata={op_type="Relu" op_name="relu0"}
  %arg1.2 = f32[64]{0} parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.221 = f32[64]{0} reshape(f32[64]{0} %arg1.2)
  %constant.439 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="bn0/add"}
  %broadcast.440 = f32[64]{0} broadcast(f32[] %constant.439), dimensions={}, metadata={op_type="AddV2" op_name="bn0/add"}
  %add.441 = f32[64]{0} add(f32[64]{0} %reshape.221, f32[64]{0} %broadcast.440), metadata={op_type="AddV2" op_name="bn0/add"}
  %rsqrt.442 = f32[64]{0} rsqrt(f32[64]{0} %add.441), metadata={op_type="Rsqrt" op_name="bn0/Rsqrt"}
  %arg58.59 = f32[64]{0} parameter(58), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.278 = f32[64]{0} reshape(f32[64]{0} %arg58.59)
  %multiply.443 = f32[64]{0} multiply(f32[64]{0} %rsqrt.442, f32[64]{0} %reshape.278), metadata={op_type="Mul" op_name="bn0/mul"}
  %broadcast.459 = f32[1,112,112,64]{3,2,1,0} broadcast(f32[64]{0} %multiply.443), dimensions={3}, metadata={op_type="Mul" op_name="bn0/mul_1"}
  %constant.446 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="bn_data/add"}
  %broadcast.447 = f32[3]{0} broadcast(f32[] %constant.446), dimensions={}, metadata={op_type="AddV2" op_name="bn_data/add"}
  %arg0.1 = f32[3]{0} parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.220 = f32[3]{0} reshape(f32[3]{0} %arg0.1)
  %add.448 = f32[3]{0} add(f32[3]{0} %broadcast.447, f32[3]{0} %reshape.220), metadata={op_type="AddV2" op_name="bn_data/add"}
  %rsqrt.449 = f32[3]{0} rsqrt(f32[3]{0} %add.448), metadata={op_type="Rsqrt" op_name="bn_data/Rsqrt"}
  %broadcast.450 = f32[1,224,224,3]{3,2,1,0} broadcast(f32[3]{0} %rsqrt.449), dimensions={3}, metadata={op_type="Mul" op_name="bn_data/mul"}
  %arg218.219 = f32[1,224,224,3]{3,2,1,0} parameter(218), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.438 = f32[1,224,224,3]{3,2,1,0} reshape(f32[1,224,224,3]{3,2,1,0} %arg218.219)
  %multiply.451 = f32[1,224,224,3]{3,2,1,0} multiply(f32[1,224,224,3]{3,2,1,0} %broadcast.450, f32[1,224,224,3]{3,2,1,0} %reshape.438), metadata={op_type="Mul" op_name="bn_data/mul"}
  %arg101.102 = f32[3]{0} parameter(101), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.321 = f32[3]{0} reshape(f32[3]{0} %arg101.102)
  %arg57.58 = f32[3]{0} parameter(57), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.277 = f32[3]{0} reshape(f32[3]{0} %arg57.58)
  %multiply.452 = f32[3]{0} multiply(f32[3]{0} %rsqrt.449, f32[3]{0} %reshape.277), metadata={op_type="Mul" op_name="bn_data/mul_1"}
  %subtract.453 = f32[3]{0} subtract(f32[3]{0} %reshape.321, f32[3]{0} %multiply.452), metadata={op_type="Sub" op_name="bn_data/sub"}
  %broadcast.454 = f32[1,224,224,3]{3,2,1,0} broadcast(f32[3]{0} %subtract.453), dimensions={3}, metadata={op_type="AddV2" op_name="bn_data/add_1"}
  %add.455 = f32[1,224,224,3]{3,2,1,0} add(f32[1,224,224,3]{3,2,1,0} %multiply.451, f32[1,224,224,3]{3,2,1,0} %broadcast.454), metadata={op_type="AddV2" op_name="bn_data/add_1"}
  %constant.456 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad"}
  %pad.457 = f32[1,230,230,3]{3,2,1,0} pad(f32[1,224,224,3]{3,2,1,0} %add.455, f32[] %constant.456), padding=0_0x3_3x3_3x0_0, metadata={op_type="Pad" op_name="Pad"}
  %arg188.189 = f32[7,7,3,64]{3,2,1,0} parameter(188), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.408 = f32[7,7,3,64]{3,2,1,0} reshape(f32[7,7,3,64]{3,2,1,0} %arg188.189)
  %convolution.458 = f32[1,112,112,64]{3,2,1,0} convolution(f32[1,230,230,3]{3,2,1,0} %pad.457, f32[7,7,3,64]{3,2,1,0} %reshape.408), window={size=7x7 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="conv0"}
  %multiply.460 = f32[1,112,112,64]{3,2,1,0} multiply(f32[1,112,112,64]{3,2,1,0} %broadcast.459, f32[1,112,112,64]{3,2,1,0} %convolution.458), metadata={op_type="Mul" op_name="bn0/mul_1"}
  %arg145.146 = f32[64]{0} parameter(145), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.365 = f32[64]{0} reshape(f32[64]{0} %arg145.146)
  %arg102.103 = f32[64]{0} parameter(102), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.322 = f32[64]{0} reshape(f32[64]{0} %arg102.103)
  %multiply.444 = f32[64]{0} multiply(f32[64]{0} %multiply.443, f32[64]{0} %reshape.322), metadata={op_type="Mul" op_name="bn0/mul_2"}
  %subtract.445 = f32[64]{0} subtract(f32[64]{0} %reshape.365, f32[64]{0} %multiply.444), metadata={op_type="Sub" op_name="bn0/sub"}
  %broadcast.461 = f32[1,112,112,64]{3,2,1,0} broadcast(f32[64]{0} %subtract.445), dimensions={3}, metadata={op_type="AddV2" op_name="bn0/add_1"}
  %add.462 = f32[1,112,112,64]{3,2,1,0} add(f32[1,112,112,64]{3,2,1,0} %multiply.460, f32[1,112,112,64]{3,2,1,0} %broadcast.461), metadata={op_type="AddV2" op_name="bn0/add_1"}
  %maximum.465 = f32[1,112,112,64]{3,2,1,0} maximum(f32[1,112,112,64]{3,2,1,0} %broadcast.464, f32[1,112,112,64]{3,2,1,0} %add.462), metadata={op_type="Relu" op_name="relu0"}
  %constant.466 = f32[] constant(-inf), metadata={op_type="PadV2" op_name="PadV2"}
  %pad.467 = f32[1,114,114,64]{3,2,1,0} pad(f32[1,112,112,64]{3,2,1,0} %maximum.465, f32[] %constant.466), padding=0_0x1_1x1_1x0_0, metadata={op_type="PadV2" op_name="PadV2"}
  %constant.468 = f32[] constant(-inf), metadata={op_type="MaxPool" op_name="pooling0"}
  %reduce-window.473 = f32[1,56,56,64]{3,2,1,0} reduce-window(f32[1,114,114,64]{3,2,1,0} %pad.467, f32[] %constant.468), window={size=1x3x3x1 stride=1x2x2x1}, to_apply=%max_F32.469, metadata={op_type="MaxPool" op_name="pooling0"}
  %arg189.190 = f32[1,1,64,128]{3,2,1,0} parameter(189), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.409 = f32[1,1,64,128]{3,2,1,0} reshape(f32[1,1,64,128]{3,2,1,0} %arg189.190)
  %convolution.495 = f32[1,56,56,128]{3,2,1,0} convolution(f32[1,56,56,64]{3,2,1,0} %reduce-window.473, f32[1,1,64,128]{3,2,1,0} %reshape.409), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv1"}
  %multiply.497 = f32[1,56,56,128]{3,2,1,0} multiply(f32[1,56,56,128]{3,2,1,0} %broadcast.496, f32[1,56,56,128]{3,2,1,0} %convolution.495), metadata={op_type="Mul" op_name="stage1_unit1_bn1/mul_1"}
  %arg147.148 = f32[128]{0} parameter(147), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.367 = f32[128]{0} reshape(f32[128]{0} %arg147.148)
  %arg104.105 = f32[128]{0} parameter(104), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.324 = f32[128]{0} reshape(f32[128]{0} %arg104.105)
  %multiply.479 = f32[128]{0} multiply(f32[128]{0} %multiply.478, f32[128]{0} %reshape.324), metadata={op_type="Mul" op_name="stage1_unit1_bn1/mul_2"}
  %subtract.480 = f32[128]{0} subtract(f32[128]{0} %reshape.367, f32[128]{0} %multiply.479), metadata={op_type="Sub" op_name="stage1_unit1_bn1/sub"}
  %broadcast.498 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %subtract.480), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit1_bn1/add_1"}
  %add.499 = f32[1,56,56,128]{3,2,1,0} add(f32[1,56,56,128]{3,2,1,0} %multiply.497, f32[1,56,56,128]{3,2,1,0} %broadcast.498), metadata={op_type="AddV2" op_name="stage1_unit1_bn1/add_1"}
  %maximum.502 = f32[1,56,56,128]{3,2,1,0} maximum(f32[1,56,56,128]{3,2,1,0} %broadcast.501, f32[1,56,56,128]{3,2,1,0} %add.499), metadata={op_type="Relu" op_name="stage1_unit1_relu1"}
  %constant.503 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_1"}
  %pad.504 = f32[1,58,58,128]{3,2,1,0} pad(f32[1,56,56,128]{3,2,1,0} %maximum.502, f32[] %constant.503), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_1"}
  %slice.505 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.504), slice={[0:1], [0:58], [0:58], [0:4]}, metadata={op_type="Split" op_name="split_1"}
  %arg5.6 = f32[3,3,4,128]{3,2,1,0} parameter(5), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.225 = f32[3,3,4,128]{3,2,1,0} reshape(f32[3,3,4,128]{3,2,1,0} %arg5.6)
  %slice.537 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.225), slice={[0:3], [0:3], [0:4], [0:4]}, metadata={op_type="Split" op_name="split"}
  %convolution.569 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.505, f32[3,3,4,4]{3,2,1,0} %slice.537), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2"}
  %slice.506 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.504), slice={[0:1], [0:58], [0:58], [4:8]}, metadata={op_type="Split" op_name="split_1"}
  %slice.538 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.225), slice={[0:3], [0:3], [0:4], [4:8]}, metadata={op_type="Split" op_name="split"}
  %convolution.570 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.506, f32[3,3,4,4]{3,2,1,0} %slice.538), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_1"}
  %slice.507 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.504), slice={[0:1], [0:58], [0:58], [8:12]}, metadata={op_type="Split" op_name="split_1"}
  %slice.539 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.225), slice={[0:3], [0:3], [0:4], [8:12]}, metadata={op_type="Split" op_name="split"}
  %convolution.581 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.507, f32[3,3,4,4]{3,2,1,0} %slice.539), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_2"}
  %slice.508 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.504), slice={[0:1], [0:58], [0:58], [12:16]}, metadata={op_type="Split" op_name="split_1"}
  %slice.540 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.225), slice={[0:3], [0:3], [0:4], [12:16]}, metadata={op_type="Split" op_name="split"}
  %convolution.592 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.508, f32[3,3,4,4]{3,2,1,0} %slice.540), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_3"}
  %slice.509 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.504), slice={[0:1], [0:58], [0:58], [16:20]}, metadata={op_type="Split" op_name="split_1"}
  %slice.541 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.225), slice={[0:3], [0:3], [0:4], [16:20]}, metadata={op_type="Split" op_name="split"}
  %convolution.595 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.509, f32[3,3,4,4]{3,2,1,0} %slice.541), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_4"}
  %slice.510 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.504), slice={[0:1], [0:58], [0:58], [20:24]}, metadata={op_type="Split" op_name="split_1"}
  %slice.542 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.225), slice={[0:3], [0:3], [0:4], [20:24]}, metadata={op_type="Split" op_name="split"}
  %convolution.596 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.510, f32[3,3,4,4]{3,2,1,0} %slice.542), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_5"}
  %slice.511 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.504), slice={[0:1], [0:58], [0:58], [24:28]}, metadata={op_type="Split" op_name="split_1"}
  %slice.543 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.225), slice={[0:3], [0:3], [0:4], [24:28]}, metadata={op_type="Split" op_name="split"}
  %convolution.597 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.511, f32[3,3,4,4]{3,2,1,0} %slice.543), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_6"}
  %slice.512 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.504), slice={[0:1], [0:58], [0:58], [28:32]}, metadata={op_type="Split" op_name="split_1"}
  %slice.544 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.225), slice={[0:3], [0:3], [0:4], [28:32]}, metadata={op_type="Split" op_name="split"}
  %convolution.598 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.512, f32[3,3,4,4]{3,2,1,0} %slice.544), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_7"}
  %slice.513 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.504), slice={[0:1], [0:58], [0:58], [32:36]}, metadata={op_type="Split" op_name="split_1"}
  %slice.545 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.225), slice={[0:3], [0:3], [0:4], [32:36]}, metadata={op_type="Split" op_name="split"}
  %convolution.599 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.513, f32[3,3,4,4]{3,2,1,0} %slice.545), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_8"}
  %slice.514 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.504), slice={[0:1], [0:58], [0:58], [36:40]}, metadata={op_type="Split" op_name="split_1"}
  %slice.546 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.225), slice={[0:3], [0:3], [0:4], [36:40]}, metadata={op_type="Split" op_name="split"}
  %convolution.600 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.514, f32[3,3,4,4]{3,2,1,0} %slice.546), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_9"}
  %slice.515 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.504), slice={[0:1], [0:58], [0:58], [40:44]}, metadata={op_type="Split" op_name="split_1"}
  %slice.547 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.225), slice={[0:3], [0:3], [0:4], [40:44]}, metadata={op_type="Split" op_name="split"}
  %convolution.571 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.515, f32[3,3,4,4]{3,2,1,0} %slice.547), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_10"}
  %slice.516 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.504), slice={[0:1], [0:58], [0:58], [44:48]}, metadata={op_type="Split" op_name="split_1"}
  %slice.548 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.225), slice={[0:3], [0:3], [0:4], [44:48]}, metadata={op_type="Split" op_name="split"}
  %convolution.572 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.516, f32[3,3,4,4]{3,2,1,0} %slice.548), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_11"}
  %slice.517 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.504), slice={[0:1], [0:58], [0:58], [48:52]}, metadata={op_type="Split" op_name="split_1"}
  %slice.549 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.225), slice={[0:3], [0:3], [0:4], [48:52]}, metadata={op_type="Split" op_name="split"}
  %convolution.573 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.517, f32[3,3,4,4]{3,2,1,0} %slice.549), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_12"}
  %slice.518 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.504), slice={[0:1], [0:58], [0:58], [52:56]}, metadata={op_type="Split" op_name="split_1"}
  %slice.550 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.225), slice={[0:3], [0:3], [0:4], [52:56]}, metadata={op_type="Split" op_name="split"}
  %convolution.574 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.518, f32[3,3,4,4]{3,2,1,0} %slice.550), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_13"}
  %slice.519 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.504), slice={[0:1], [0:58], [0:58], [56:60]}, metadata={op_type="Split" op_name="split_1"}
  %slice.551 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.225), slice={[0:3], [0:3], [0:4], [56:60]}, metadata={op_type="Split" op_name="split"}
  %convolution.575 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.519, f32[3,3,4,4]{3,2,1,0} %slice.551), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_14"}
  %slice.520 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.504), slice={[0:1], [0:58], [0:58], [60:64]}, metadata={op_type="Split" op_name="split_1"}
  %slice.552 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.225), slice={[0:3], [0:3], [0:4], [60:64]}, metadata={op_type="Split" op_name="split"}
  %convolution.576 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.520, f32[3,3,4,4]{3,2,1,0} %slice.552), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_15"}
  %slice.521 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.504), slice={[0:1], [0:58], [0:58], [64:68]}, metadata={op_type="Split" op_name="split_1"}
  %slice.553 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.225), slice={[0:3], [0:3], [0:4], [64:68]}, metadata={op_type="Split" op_name="split"}
  %convolution.577 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.521, f32[3,3,4,4]{3,2,1,0} %slice.553), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_16"}
  %slice.522 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.504), slice={[0:1], [0:58], [0:58], [68:72]}, metadata={op_type="Split" op_name="split_1"}
  %slice.554 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.225), slice={[0:3], [0:3], [0:4], [68:72]}, metadata={op_type="Split" op_name="split"}
  %convolution.578 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.522, f32[3,3,4,4]{3,2,1,0} %slice.554), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_17"}
  %slice.523 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.504), slice={[0:1], [0:58], [0:58], [72:76]}, metadata={op_type="Split" op_name="split_1"}
  %slice.555 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.225), slice={[0:3], [0:3], [0:4], [72:76]}, metadata={op_type="Split" op_name="split"}
  %convolution.579 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.523, f32[3,3,4,4]{3,2,1,0} %slice.555), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_18"}
  %slice.524 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.504), slice={[0:1], [0:58], [0:58], [76:80]}, metadata={op_type="Split" op_name="split_1"}
  %slice.556 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.225), slice={[0:3], [0:3], [0:4], [76:80]}, metadata={op_type="Split" op_name="split"}
  %convolution.580 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.524, f32[3,3,4,4]{3,2,1,0} %slice.556), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_19"}
  %slice.525 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.504), slice={[0:1], [0:58], [0:58], [80:84]}, metadata={op_type="Split" op_name="split_1"}
  %slice.557 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.225), slice={[0:3], [0:3], [0:4], [80:84]}, metadata={op_type="Split" op_name="split"}
  %convolution.582 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.525, f32[3,3,4,4]{3,2,1,0} %slice.557), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_20"}
  %slice.526 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.504), slice={[0:1], [0:58], [0:58], [84:88]}, metadata={op_type="Split" op_name="split_1"}
  %slice.558 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.225), slice={[0:3], [0:3], [0:4], [84:88]}, metadata={op_type="Split" op_name="split"}
  %convolution.583 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.526, f32[3,3,4,4]{3,2,1,0} %slice.558), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_21"}
  %slice.527 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.504), slice={[0:1], [0:58], [0:58], [88:92]}, metadata={op_type="Split" op_name="split_1"}
  %slice.559 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.225), slice={[0:3], [0:3], [0:4], [88:92]}, metadata={op_type="Split" op_name="split"}
  %convolution.584 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.527, f32[3,3,4,4]{3,2,1,0} %slice.559), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_22"}
  %slice.528 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.504), slice={[0:1], [0:58], [0:58], [92:96]}, metadata={op_type="Split" op_name="split_1"}
  %slice.560 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.225), slice={[0:3], [0:3], [0:4], [92:96]}, metadata={op_type="Split" op_name="split"}
  %convolution.585 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.528, f32[3,3,4,4]{3,2,1,0} %slice.560), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_23"}
  %slice.529 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.504), slice={[0:1], [0:58], [0:58], [96:100]}, metadata={op_type="Split" op_name="split_1"}
  %slice.561 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.225), slice={[0:3], [0:3], [0:4], [96:100]}, metadata={op_type="Split" op_name="split"}
  %convolution.586 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.529, f32[3,3,4,4]{3,2,1,0} %slice.561), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_24"}
  %slice.530 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.504), slice={[0:1], [0:58], [0:58], [100:104]}, metadata={op_type="Split" op_name="split_1"}
  %slice.562 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.225), slice={[0:3], [0:3], [0:4], [100:104]}, metadata={op_type="Split" op_name="split"}
  %convolution.587 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.530, f32[3,3,4,4]{3,2,1,0} %slice.562), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_25"}
  %slice.531 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.504), slice={[0:1], [0:58], [0:58], [104:108]}, metadata={op_type="Split" op_name="split_1"}
  %slice.563 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.225), slice={[0:3], [0:3], [0:4], [104:108]}, metadata={op_type="Split" op_name="split"}
  %convolution.588 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.531, f32[3,3,4,4]{3,2,1,0} %slice.563), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_26"}
  %slice.532 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.504), slice={[0:1], [0:58], [0:58], [108:112]}, metadata={op_type="Split" op_name="split_1"}
  %slice.564 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.225), slice={[0:3], [0:3], [0:4], [108:112]}, metadata={op_type="Split" op_name="split"}
  %convolution.589 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.532, f32[3,3,4,4]{3,2,1,0} %slice.564), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_27"}
  %slice.533 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.504), slice={[0:1], [0:58], [0:58], [112:116]}, metadata={op_type="Split" op_name="split_1"}
  %slice.565 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.225), slice={[0:3], [0:3], [0:4], [112:116]}, metadata={op_type="Split" op_name="split"}
  %convolution.590 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.533, f32[3,3,4,4]{3,2,1,0} %slice.565), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_28"}
  %slice.534 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.504), slice={[0:1], [0:58], [0:58], [116:120]}, metadata={op_type="Split" op_name="split_1"}
  %slice.566 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.225), slice={[0:3], [0:3], [0:4], [116:120]}, metadata={op_type="Split" op_name="split"}
  %convolution.591 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.534, f32[3,3,4,4]{3,2,1,0} %slice.566), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_29"}
  %slice.535 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.504), slice={[0:1], [0:58], [0:58], [120:124]}, metadata={op_type="Split" op_name="split_1"}
  %slice.567 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.225), slice={[0:3], [0:3], [0:4], [120:124]}, metadata={op_type="Split" op_name="split"}
  %convolution.593 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.535, f32[3,3,4,4]{3,2,1,0} %slice.567), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_30"}
  %slice.536 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.504), slice={[0:1], [0:58], [0:58], [124:128]}, metadata={op_type="Split" op_name="split_1"}
  %slice.568 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.225), slice={[0:3], [0:3], [0:4], [124:128]}, metadata={op_type="Split" op_name="split"}
  %convolution.594 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.536, f32[3,3,4,4]{3,2,1,0} %slice.568), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_31"}
  %concatenate.601 = f32[1,56,56,128]{3,2,1,0} concatenate(f32[1,56,56,4]{3,2,1,0} %convolution.569, f32[1,56,56,4]{3,2,1,0} %convolution.570, f32[1,56,56,4]{3,2,1,0} %convolution.581, f32[1,56,56,4]{3,2,1,0} %convolution.592, f32[1,56,56,4]{3,2,1,0} %convolution.595, f32[1,56,56,4]{3,2,1,0} %convolution.596, f32[1,56,56,4]{3,2,1,0} %convolution.597, f32[1,56,56,4]{3,2,1,0} %convolution.598, f32[1,56,56,4]{3,2,1,0} %convolution.599, f32[1,56,56,4]{3,2,1,0} %convolution.600, f32[1,56,56,4]{3,2,1,0} %convolution.571, f32[1,56,56,4]{3,2,1,0} %convolution.572, f32[1,56,56,4]{3,2,1,0} %convolution.573, f32[1,56,56,4]{3,2,1,0} %convolution.574, f32[1,56,56,4]{3,2,1,0} %convolution.575, f32[1,56,56,4]{3,2,1,0} %convolution.576, f32[1,56,56,4]{3,2,1,0} %convolution.577, f32[1,56,56,4]{3,2,1,0} %convolution.578, f32[1,56,56,4]{3,2,1,0} %convolution.579, f32[1,56,56,4]{3,2,1,0} %convolution.580, f32[1,56,56,4]{3,2,1,0} %convolution.582, f32[1,56,56,4]{3,2,1,0} %convolution.583, f32[1,56,56,4]{3,2,1,0} %convolution.584, f32[1,56,56,4]{3,2,1,0} %convolution.585, f32[1,56,56,4]{3,2,1,0} %convolution.586, f32[1,56,56,4]{3,2,1,0} %convolution.587, f32[1,56,56,4]{3,2,1,0} %convolution.588, f32[1,56,56,4]{3,2,1,0} %convolution.589, f32[1,56,56,4]{3,2,1,0} %convolution.590, f32[1,56,56,4]{3,2,1,0} %convolution.591, f32[1,56,56,4]{3,2,1,0} %convolution.593, f32[1,56,56,4]{3,2,1,0} %convolution.594), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat"}
  %constant.481 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit1_bn2/add"}
  %broadcast.482 = f32[128]{0} broadcast(f32[] %constant.481), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit1_bn2/add"}
  %arg6.7 = f32[128]{0} parameter(6), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.226 = f32[128]{0} reshape(f32[128]{0} %arg6.7)
  %add.483 = f32[128]{0} add(f32[128]{0} %broadcast.482, f32[128]{0} %reshape.226), metadata={op_type="AddV2" op_name="stage1_unit1_bn2/add"}
  %rsqrt.484 = f32[128]{0} rsqrt(f32[128]{0} %add.483), metadata={op_type="Rsqrt" op_name="stage1_unit1_bn2/Rsqrt"}
  %arg62.63 = f32[128]{0} parameter(62), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.282 = f32[128]{0} reshape(f32[128]{0} %arg62.63)
  %multiply.485 = f32[128]{0} multiply(f32[128]{0} %rsqrt.484, f32[128]{0} %reshape.282), metadata={op_type="Mul" op_name="stage1_unit1_bn2/mul"}
  %broadcast.602 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %multiply.485), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit1_bn2/mul_1"}
  %multiply.603 = f32[1,56,56,128]{3,2,1,0} multiply(f32[1,56,56,128]{3,2,1,0} %concatenate.601, f32[1,56,56,128]{3,2,1,0} %broadcast.602), metadata={op_type="Mul" op_name="stage1_unit1_bn2/mul_1"}
  %arg149.150 = f32[128]{0} parameter(149), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.369 = f32[128]{0} reshape(f32[128]{0} %arg149.150)
  %arg106.107 = f32[128]{0} parameter(106), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.326 = f32[128]{0} reshape(f32[128]{0} %arg106.107)
  %multiply.486 = f32[128]{0} multiply(f32[128]{0} %multiply.485, f32[128]{0} %reshape.326), metadata={op_type="Mul" op_name="stage1_unit1_bn2/mul_2"}
  %subtract.487 = f32[128]{0} subtract(f32[128]{0} %reshape.369, f32[128]{0} %multiply.486), metadata={op_type="Sub" op_name="stage1_unit1_bn2/sub"}
  %broadcast.604 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %subtract.487), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit1_bn2/add_1"}
  %add.605 = f32[1,56,56,128]{3,2,1,0} add(f32[1,56,56,128]{3,2,1,0} %multiply.603, f32[1,56,56,128]{3,2,1,0} %broadcast.604), metadata={op_type="AddV2" op_name="stage1_unit1_bn2/add_1"}
  %maximum.608 = f32[1,56,56,128]{3,2,1,0} maximum(f32[1,56,56,128]{3,2,1,0} %broadcast.607, f32[1,56,56,128]{3,2,1,0} %add.605), metadata={op_type="Relu" op_name="stage1_unit1_relu2"}
  %arg191.192 = f32[1,1,128,256]{3,2,1,0} parameter(191), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.411 = f32[1,1,128,256]{3,2,1,0} reshape(f32[1,1,128,256]{3,2,1,0} %arg191.192)
  %convolution.609 = f32[1,56,56,256]{3,2,1,0} convolution(f32[1,56,56,128]{3,2,1,0} %maximum.608, f32[1,1,128,256]{3,2,1,0} %reshape.411), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv3"}
  %multiply.611 = f32[1,56,56,256]{3,2,1,0} multiply(f32[1,56,56,256]{3,2,1,0} %broadcast.610, f32[1,56,56,256]{3,2,1,0} %convolution.609), metadata={op_type="Mul" op_name="stage1_unit1_bn3/mul_1"}
  %arg152.153 = f32[256]{0} parameter(152), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.372 = f32[256]{0} reshape(f32[256]{0} %arg152.153)
  %arg109.110 = f32[256]{0} parameter(109), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.329 = f32[256]{0} reshape(f32[256]{0} %arg109.110)
  %multiply.493 = f32[256]{0} multiply(f32[256]{0} %multiply.492, f32[256]{0} %reshape.329), metadata={op_type="Mul" op_name="stage1_unit1_bn3/mul_2"}
  %subtract.494 = f32[256]{0} subtract(f32[256]{0} %reshape.372, f32[256]{0} %multiply.493), metadata={op_type="Sub" op_name="stage1_unit1_bn3/sub"}
  %broadcast.612 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.494), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit1_bn3/add_1"}
  %add.613 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %multiply.611, f32[1,56,56,256]{3,2,1,0} %broadcast.612), metadata={op_type="AddV2" op_name="stage1_unit1_bn3/add_1"}
  %arg190.191 = f32[1,1,64,256]{3,2,1,0} parameter(190), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.410 = f32[1,1,64,256]{3,2,1,0} reshape(f32[1,1,64,256]{3,2,1,0} %arg190.191)
  %convolution.621 = f32[1,56,56,256]{3,2,1,0} convolution(f32[1,56,56,64]{3,2,1,0} %reduce-window.473, f32[1,1,64,256]{3,2,1,0} %reshape.410), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_sc"}
  %constant.614 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit1_sc_bn/add"}
  %broadcast.615 = f32[256]{0} broadcast(f32[] %constant.614), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit1_sc_bn/add"}
  %arg4.5 = f32[256]{0} parameter(4), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.224 = f32[256]{0} reshape(f32[256]{0} %arg4.5)
  %add.616 = f32[256]{0} add(f32[256]{0} %broadcast.615, f32[256]{0} %reshape.224), metadata={op_type="AddV2" op_name="stage1_unit1_sc_bn/add"}
  %rsqrt.617 = f32[256]{0} rsqrt(f32[256]{0} %add.616), metadata={op_type="Rsqrt" op_name="stage1_unit1_sc_bn/Rsqrt"}
  %arg61.62 = f32[256]{0} parameter(61), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.281 = f32[256]{0} reshape(f32[256]{0} %arg61.62)
  %multiply.618 = f32[256]{0} multiply(f32[256]{0} %rsqrt.617, f32[256]{0} %reshape.281), metadata={op_type="Mul" op_name="stage1_unit1_sc_bn/mul"}
  %broadcast.622 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.618), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit1_sc_bn/mul_1"}
  %multiply.623 = f32[1,56,56,256]{3,2,1,0} multiply(f32[1,56,56,256]{3,2,1,0} %convolution.621, f32[1,56,56,256]{3,2,1,0} %broadcast.622), metadata={op_type="Mul" op_name="stage1_unit1_sc_bn/mul_1"}
  %arg148.149 = f32[256]{0} parameter(148), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.368 = f32[256]{0} reshape(f32[256]{0} %arg148.149)
  %arg105.106 = f32[256]{0} parameter(105), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.325 = f32[256]{0} reshape(f32[256]{0} %arg105.106)
  %multiply.619 = f32[256]{0} multiply(f32[256]{0} %multiply.618, f32[256]{0} %reshape.325), metadata={op_type="Mul" op_name="stage1_unit1_sc_bn/mul_2"}
  %subtract.620 = f32[256]{0} subtract(f32[256]{0} %reshape.368, f32[256]{0} %multiply.619), metadata={op_type="Sub" op_name="stage1_unit1_sc_bn/sub"}
  %broadcast.624 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.620), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit1_sc_bn/add_1"}
  %add.625 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %multiply.623, f32[1,56,56,256]{3,2,1,0} %broadcast.624), metadata={op_type="AddV2" op_name="stage1_unit1_sc_bn/add_1"}
  %add.626 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %add.613, f32[1,56,56,256]{3,2,1,0} %add.625), metadata={op_type="AddV2" op_name="add"}
  %maximum.629 = f32[1,56,56,256]{3,2,1,0} maximum(f32[1,56,56,256]{3,2,1,0} %broadcast.628, f32[1,56,56,256]{3,2,1,0} %add.626), metadata={op_type="Relu" op_name="stage1_unit1_relu"}
  %constant.644 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit2_bn3/add"}
  %broadcast.645 = f32[256]{0} broadcast(f32[] %constant.644), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit2_bn3/add"}
  %arg15.16 = f32[256]{0} parameter(15), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.235 = f32[256]{0} reshape(f32[256]{0} %arg15.16)
  %add.646 = f32[256]{0} add(f32[256]{0} %broadcast.645, f32[256]{0} %reshape.235), metadata={op_type="AddV2" op_name="stage1_unit2_bn3/add"}
  %rsqrt.647 = f32[256]{0} rsqrt(f32[256]{0} %add.646), metadata={op_type="Rsqrt" op_name="stage1_unit2_bn3/Rsqrt"}
  %arg69.70 = f32[256]{0} parameter(69), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.289 = f32[256]{0} reshape(f32[256]{0} %arg69.70)
  %multiply.648 = f32[256]{0} multiply(f32[256]{0} %rsqrt.647, f32[256]{0} %reshape.289), metadata={op_type="Mul" op_name="stage1_unit2_bn3/mul"}
  %broadcast.766 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.648), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit2_bn3/mul_1"}
  %constant.762 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit2_relu2"}
  %broadcast.763 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[] %constant.762), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit2_relu2"}
  %constant.656 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit2_relu1"}
  %broadcast.657 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[] %constant.656), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit2_relu1"}
  %constant.630 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit2_bn1/add"}
  %broadcast.631 = f32[128]{0} broadcast(f32[] %constant.630), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit2_bn1/add"}
  %arg11.12 = f32[128]{0} parameter(11), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.231 = f32[128]{0} reshape(f32[128]{0} %arg11.12)
  %add.632 = f32[128]{0} add(f32[128]{0} %broadcast.631, f32[128]{0} %reshape.231), metadata={op_type="AddV2" op_name="stage1_unit2_bn1/add"}
  %rsqrt.633 = f32[128]{0} rsqrt(f32[128]{0} %add.632), metadata={op_type="Rsqrt" op_name="stage1_unit2_bn1/Rsqrt"}
  %arg66.67 = f32[128]{0} parameter(66), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.286 = f32[128]{0} reshape(f32[128]{0} %arg66.67)
  %multiply.634 = f32[128]{0} multiply(f32[128]{0} %rsqrt.633, f32[128]{0} %reshape.286), metadata={op_type="Mul" op_name="stage1_unit2_bn1/mul"}
  %broadcast.652 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %multiply.634), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit2_bn1/mul_1"}
  %arg192.193 = f32[1,1,256,128]{3,2,1,0} parameter(192), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.412 = f32[1,1,256,128]{3,2,1,0} reshape(f32[1,1,256,128]{3,2,1,0} %arg192.193)
  %convolution.651 = f32[1,56,56,128]{3,2,1,0} convolution(f32[1,56,56,256]{3,2,1,0} %maximum.629, f32[1,1,256,128]{3,2,1,0} %reshape.412), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv1"}
  %multiply.653 = f32[1,56,56,128]{3,2,1,0} multiply(f32[1,56,56,128]{3,2,1,0} %broadcast.652, f32[1,56,56,128]{3,2,1,0} %convolution.651), metadata={op_type="Mul" op_name="stage1_unit2_bn1/mul_1"}
  %arg153.154 = f32[128]{0} parameter(153), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.373 = f32[128]{0} reshape(f32[128]{0} %arg153.154)
  %arg110.111 = f32[128]{0} parameter(110), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.330 = f32[128]{0} reshape(f32[128]{0} %arg110.111)
  %multiply.635 = f32[128]{0} multiply(f32[128]{0} %multiply.634, f32[128]{0} %reshape.330), metadata={op_type="Mul" op_name="stage1_unit2_bn1/mul_2"}
  %subtract.636 = f32[128]{0} subtract(f32[128]{0} %reshape.373, f32[128]{0} %multiply.635), metadata={op_type="Sub" op_name="stage1_unit2_bn1/sub"}
  %broadcast.654 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %subtract.636), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit2_bn1/add_1"}
  %add.655 = f32[1,56,56,128]{3,2,1,0} add(f32[1,56,56,128]{3,2,1,0} %multiply.653, f32[1,56,56,128]{3,2,1,0} %broadcast.654), metadata={op_type="AddV2" op_name="stage1_unit2_bn1/add_1"}
  %maximum.658 = f32[1,56,56,128]{3,2,1,0} maximum(f32[1,56,56,128]{3,2,1,0} %broadcast.657, f32[1,56,56,128]{3,2,1,0} %add.655), metadata={op_type="Relu" op_name="stage1_unit2_relu1"}
  %constant.659 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_2"}
  %pad.660 = f32[1,58,58,128]{3,2,1,0} pad(f32[1,56,56,128]{3,2,1,0} %maximum.658, f32[] %constant.659), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_2"}
  %slice.661 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.660), slice={[0:1], [0:58], [0:58], [0:4]}, metadata={op_type="Split" op_name="split_3"}
  %arg12.13 = f32[3,3,4,128]{3,2,1,0} parameter(12), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.232 = f32[3,3,4,128]{3,2,1,0} reshape(f32[3,3,4,128]{3,2,1,0} %arg12.13)
  %slice.693 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.232), slice={[0:3], [0:3], [0:4], [0:4]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.725 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.661, f32[3,3,4,4]{3,2,1,0} %slice.693), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2"}
  %slice.662 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.660), slice={[0:1], [0:58], [0:58], [4:8]}, metadata={op_type="Split" op_name="split_3"}
  %slice.694 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.232), slice={[0:3], [0:3], [0:4], [4:8]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.726 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.662, f32[3,3,4,4]{3,2,1,0} %slice.694), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_1"}
  %slice.663 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.660), slice={[0:1], [0:58], [0:58], [8:12]}, metadata={op_type="Split" op_name="split_3"}
  %slice.695 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.232), slice={[0:3], [0:3], [0:4], [8:12]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.737 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.663, f32[3,3,4,4]{3,2,1,0} %slice.695), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_2"}
  %slice.664 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.660), slice={[0:1], [0:58], [0:58], [12:16]}, metadata={op_type="Split" op_name="split_3"}
  %slice.696 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.232), slice={[0:3], [0:3], [0:4], [12:16]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.748 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.664, f32[3,3,4,4]{3,2,1,0} %slice.696), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_3"}
  %slice.665 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.660), slice={[0:1], [0:58], [0:58], [16:20]}, metadata={op_type="Split" op_name="split_3"}
  %slice.697 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.232), slice={[0:3], [0:3], [0:4], [16:20]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.751 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.665, f32[3,3,4,4]{3,2,1,0} %slice.697), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_4"}
  %slice.666 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.660), slice={[0:1], [0:58], [0:58], [20:24]}, metadata={op_type="Split" op_name="split_3"}
  %slice.698 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.232), slice={[0:3], [0:3], [0:4], [20:24]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.752 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.666, f32[3,3,4,4]{3,2,1,0} %slice.698), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_5"}
  %slice.667 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.660), slice={[0:1], [0:58], [0:58], [24:28]}, metadata={op_type="Split" op_name="split_3"}
  %slice.699 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.232), slice={[0:3], [0:3], [0:4], [24:28]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.753 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.667, f32[3,3,4,4]{3,2,1,0} %slice.699), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_6"}
  %slice.668 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.660), slice={[0:1], [0:58], [0:58], [28:32]}, metadata={op_type="Split" op_name="split_3"}
  %slice.700 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.232), slice={[0:3], [0:3], [0:4], [28:32]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.754 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.668, f32[3,3,4,4]{3,2,1,0} %slice.700), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_7"}
  %slice.669 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.660), slice={[0:1], [0:58], [0:58], [32:36]}, metadata={op_type="Split" op_name="split_3"}
  %slice.701 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.232), slice={[0:3], [0:3], [0:4], [32:36]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.755 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.669, f32[3,3,4,4]{3,2,1,0} %slice.701), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_8"}
  %slice.670 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.660), slice={[0:1], [0:58], [0:58], [36:40]}, metadata={op_type="Split" op_name="split_3"}
  %slice.702 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.232), slice={[0:3], [0:3], [0:4], [36:40]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.756 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.670, f32[3,3,4,4]{3,2,1,0} %slice.702), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_9"}
  %slice.671 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.660), slice={[0:1], [0:58], [0:58], [40:44]}, metadata={op_type="Split" op_name="split_3"}
  %slice.703 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.232), slice={[0:3], [0:3], [0:4], [40:44]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.727 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.671, f32[3,3,4,4]{3,2,1,0} %slice.703), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_10"}
  %slice.672 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.660), slice={[0:1], [0:58], [0:58], [44:48]}, metadata={op_type="Split" op_name="split_3"}
  %slice.704 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.232), slice={[0:3], [0:3], [0:4], [44:48]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.728 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.672, f32[3,3,4,4]{3,2,1,0} %slice.704), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_11"}
  %slice.673 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.660), slice={[0:1], [0:58], [0:58], [48:52]}, metadata={op_type="Split" op_name="split_3"}
  %slice.705 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.232), slice={[0:3], [0:3], [0:4], [48:52]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.729 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.673, f32[3,3,4,4]{3,2,1,0} %slice.705), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_12"}
  %slice.674 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.660), slice={[0:1], [0:58], [0:58], [52:56]}, metadata={op_type="Split" op_name="split_3"}
  %slice.706 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.232), slice={[0:3], [0:3], [0:4], [52:56]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.730 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.674, f32[3,3,4,4]{3,2,1,0} %slice.706), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_13"}
  %slice.675 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.660), slice={[0:1], [0:58], [0:58], [56:60]}, metadata={op_type="Split" op_name="split_3"}
  %slice.707 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.232), slice={[0:3], [0:3], [0:4], [56:60]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.731 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.675, f32[3,3,4,4]{3,2,1,0} %slice.707), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_14"}
  %slice.676 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.660), slice={[0:1], [0:58], [0:58], [60:64]}, metadata={op_type="Split" op_name="split_3"}
  %slice.708 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.232), slice={[0:3], [0:3], [0:4], [60:64]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.732 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.676, f32[3,3,4,4]{3,2,1,0} %slice.708), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_15"}
  %slice.677 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.660), slice={[0:1], [0:58], [0:58], [64:68]}, metadata={op_type="Split" op_name="split_3"}
  %slice.709 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.232), slice={[0:3], [0:3], [0:4], [64:68]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.733 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.677, f32[3,3,4,4]{3,2,1,0} %slice.709), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_16"}
  %slice.678 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.660), slice={[0:1], [0:58], [0:58], [68:72]}, metadata={op_type="Split" op_name="split_3"}
  %slice.710 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.232), slice={[0:3], [0:3], [0:4], [68:72]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.734 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.678, f32[3,3,4,4]{3,2,1,0} %slice.710), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_17"}
  %slice.679 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.660), slice={[0:1], [0:58], [0:58], [72:76]}, metadata={op_type="Split" op_name="split_3"}
  %slice.711 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.232), slice={[0:3], [0:3], [0:4], [72:76]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.735 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.679, f32[3,3,4,4]{3,2,1,0} %slice.711), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_18"}
  %slice.680 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.660), slice={[0:1], [0:58], [0:58], [76:80]}, metadata={op_type="Split" op_name="split_3"}
  %slice.712 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.232), slice={[0:3], [0:3], [0:4], [76:80]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.736 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.680, f32[3,3,4,4]{3,2,1,0} %slice.712), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_19"}
  %slice.681 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.660), slice={[0:1], [0:58], [0:58], [80:84]}, metadata={op_type="Split" op_name="split_3"}
  %slice.713 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.232), slice={[0:3], [0:3], [0:4], [80:84]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.738 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.681, f32[3,3,4,4]{3,2,1,0} %slice.713), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_20"}
  %slice.682 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.660), slice={[0:1], [0:58], [0:58], [84:88]}, metadata={op_type="Split" op_name="split_3"}
  %slice.714 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.232), slice={[0:3], [0:3], [0:4], [84:88]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.739 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.682, f32[3,3,4,4]{3,2,1,0} %slice.714), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_21"}
  %slice.683 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.660), slice={[0:1], [0:58], [0:58], [88:92]}, metadata={op_type="Split" op_name="split_3"}
  %slice.715 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.232), slice={[0:3], [0:3], [0:4], [88:92]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.740 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.683, f32[3,3,4,4]{3,2,1,0} %slice.715), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_22"}
  %slice.684 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.660), slice={[0:1], [0:58], [0:58], [92:96]}, metadata={op_type="Split" op_name="split_3"}
  %slice.716 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.232), slice={[0:3], [0:3], [0:4], [92:96]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.741 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.684, f32[3,3,4,4]{3,2,1,0} %slice.716), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_23"}
  %slice.685 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.660), slice={[0:1], [0:58], [0:58], [96:100]}, metadata={op_type="Split" op_name="split_3"}
  %slice.717 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.232), slice={[0:3], [0:3], [0:4], [96:100]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.742 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.685, f32[3,3,4,4]{3,2,1,0} %slice.717), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_24"}
  %slice.686 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.660), slice={[0:1], [0:58], [0:58], [100:104]}, metadata={op_type="Split" op_name="split_3"}
  %slice.718 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.232), slice={[0:3], [0:3], [0:4], [100:104]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.743 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.686, f32[3,3,4,4]{3,2,1,0} %slice.718), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_25"}
  %slice.687 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.660), slice={[0:1], [0:58], [0:58], [104:108]}, metadata={op_type="Split" op_name="split_3"}
  %slice.719 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.232), slice={[0:3], [0:3], [0:4], [104:108]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.744 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.687, f32[3,3,4,4]{3,2,1,0} %slice.719), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_26"}
  %slice.688 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.660), slice={[0:1], [0:58], [0:58], [108:112]}, metadata={op_type="Split" op_name="split_3"}
  %slice.720 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.232), slice={[0:3], [0:3], [0:4], [108:112]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.745 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.688, f32[3,3,4,4]{3,2,1,0} %slice.720), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_27"}
  %slice.689 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.660), slice={[0:1], [0:58], [0:58], [112:116]}, metadata={op_type="Split" op_name="split_3"}
  %slice.721 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.232), slice={[0:3], [0:3], [0:4], [112:116]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.746 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.689, f32[3,3,4,4]{3,2,1,0} %slice.721), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_28"}
  %slice.690 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.660), slice={[0:1], [0:58], [0:58], [116:120]}, metadata={op_type="Split" op_name="split_3"}
  %slice.722 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.232), slice={[0:3], [0:3], [0:4], [116:120]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.747 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.690, f32[3,3,4,4]{3,2,1,0} %slice.722), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_29"}
  %slice.691 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.660), slice={[0:1], [0:58], [0:58], [120:124]}, metadata={op_type="Split" op_name="split_3"}
  %slice.723 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.232), slice={[0:3], [0:3], [0:4], [120:124]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.749 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.691, f32[3,3,4,4]{3,2,1,0} %slice.723), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_30"}
  %slice.692 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.660), slice={[0:1], [0:58], [0:58], [124:128]}, metadata={op_type="Split" op_name="split_3"}
  %slice.724 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.232), slice={[0:3], [0:3], [0:4], [124:128]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.750 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.692, f32[3,3,4,4]{3,2,1,0} %slice.724), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_31"}
  %concatenate.757 = f32[1,56,56,128]{3,2,1,0} concatenate(f32[1,56,56,4]{3,2,1,0} %convolution.725, f32[1,56,56,4]{3,2,1,0} %convolution.726, f32[1,56,56,4]{3,2,1,0} %convolution.737, f32[1,56,56,4]{3,2,1,0} %convolution.748, f32[1,56,56,4]{3,2,1,0} %convolution.751, f32[1,56,56,4]{3,2,1,0} %convolution.752, f32[1,56,56,4]{3,2,1,0} %convolution.753, f32[1,56,56,4]{3,2,1,0} %convolution.754, f32[1,56,56,4]{3,2,1,0} %convolution.755, f32[1,56,56,4]{3,2,1,0} %convolution.756, f32[1,56,56,4]{3,2,1,0} %convolution.727, f32[1,56,56,4]{3,2,1,0} %convolution.728, f32[1,56,56,4]{3,2,1,0} %convolution.729, f32[1,56,56,4]{3,2,1,0} %convolution.730, f32[1,56,56,4]{3,2,1,0} %convolution.731, f32[1,56,56,4]{3,2,1,0} %convolution.732, f32[1,56,56,4]{3,2,1,0} %convolution.733, f32[1,56,56,4]{3,2,1,0} %convolution.734, f32[1,56,56,4]{3,2,1,0} %convolution.735, f32[1,56,56,4]{3,2,1,0} %convolution.736, f32[1,56,56,4]{3,2,1,0} %convolution.738, f32[1,56,56,4]{3,2,1,0} %convolution.739, f32[1,56,56,4]{3,2,1,0} %convolution.740, f32[1,56,56,4]{3,2,1,0} %convolution.741, f32[1,56,56,4]{3,2,1,0} %convolution.742, f32[1,56,56,4]{3,2,1,0} %convolution.743, f32[1,56,56,4]{3,2,1,0} %convolution.744, f32[1,56,56,4]{3,2,1,0} %convolution.745, f32[1,56,56,4]{3,2,1,0} %convolution.746, f32[1,56,56,4]{3,2,1,0} %convolution.747, f32[1,56,56,4]{3,2,1,0} %convolution.749, f32[1,56,56,4]{3,2,1,0} %convolution.750), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_1"}
  %constant.637 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit2_bn2/add"}
  %broadcast.638 = f32[128]{0} broadcast(f32[] %constant.637), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit2_bn2/add"}
  %arg13.14 = f32[128]{0} parameter(13), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.233 = f32[128]{0} reshape(f32[128]{0} %arg13.14)
  %add.639 = f32[128]{0} add(f32[128]{0} %broadcast.638, f32[128]{0} %reshape.233), metadata={op_type="AddV2" op_name="stage1_unit2_bn2/add"}
  %rsqrt.640 = f32[128]{0} rsqrt(f32[128]{0} %add.639), metadata={op_type="Rsqrt" op_name="stage1_unit2_bn2/Rsqrt"}
  %arg67.68 = f32[128]{0} parameter(67), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.287 = f32[128]{0} reshape(f32[128]{0} %arg67.68)
  %multiply.641 = f32[128]{0} multiply(f32[128]{0} %rsqrt.640, f32[128]{0} %reshape.287), metadata={op_type="Mul" op_name="stage1_unit2_bn2/mul"}
  %broadcast.758 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %multiply.641), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit2_bn2/mul_1"}
  %multiply.759 = f32[1,56,56,128]{3,2,1,0} multiply(f32[1,56,56,128]{3,2,1,0} %concatenate.757, f32[1,56,56,128]{3,2,1,0} %broadcast.758), metadata={op_type="Mul" op_name="stage1_unit2_bn2/mul_1"}
  %arg154.155 = f32[128]{0} parameter(154), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.374 = f32[128]{0} reshape(f32[128]{0} %arg154.155)
  %arg111.112 = f32[128]{0} parameter(111), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.331 = f32[128]{0} reshape(f32[128]{0} %arg111.112)
  %multiply.642 = f32[128]{0} multiply(f32[128]{0} %multiply.641, f32[128]{0} %reshape.331), metadata={op_type="Mul" op_name="stage1_unit2_bn2/mul_2"}
  %subtract.643 = f32[128]{0} subtract(f32[128]{0} %reshape.374, f32[128]{0} %multiply.642), metadata={op_type="Sub" op_name="stage1_unit2_bn2/sub"}
  %broadcast.760 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %subtract.643), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit2_bn2/add_1"}
  %add.761 = f32[1,56,56,128]{3,2,1,0} add(f32[1,56,56,128]{3,2,1,0} %multiply.759, f32[1,56,56,128]{3,2,1,0} %broadcast.760), metadata={op_type="AddV2" op_name="stage1_unit2_bn2/add_1"}
  %maximum.764 = f32[1,56,56,128]{3,2,1,0} maximum(f32[1,56,56,128]{3,2,1,0} %broadcast.763, f32[1,56,56,128]{3,2,1,0} %add.761), metadata={op_type="Relu" op_name="stage1_unit2_relu2"}
  %arg193.194 = f32[1,1,128,256]{3,2,1,0} parameter(193), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.413 = f32[1,1,128,256]{3,2,1,0} reshape(f32[1,1,128,256]{3,2,1,0} %arg193.194)
  %convolution.765 = f32[1,56,56,256]{3,2,1,0} convolution(f32[1,56,56,128]{3,2,1,0} %maximum.764, f32[1,1,128,256]{3,2,1,0} %reshape.413), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv3"}
  %multiply.767 = f32[1,56,56,256]{3,2,1,0} multiply(f32[1,56,56,256]{3,2,1,0} %broadcast.766, f32[1,56,56,256]{3,2,1,0} %convolution.765), metadata={op_type="Mul" op_name="stage1_unit2_bn3/mul_1"}
  %arg156.157 = f32[256]{0} parameter(156), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.376 = f32[256]{0} reshape(f32[256]{0} %arg156.157)
  %arg113.114 = f32[256]{0} parameter(113), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.333 = f32[256]{0} reshape(f32[256]{0} %arg113.114)
  %multiply.649 = f32[256]{0} multiply(f32[256]{0} %multiply.648, f32[256]{0} %reshape.333), metadata={op_type="Mul" op_name="stage1_unit2_bn3/mul_2"}
  %subtract.650 = f32[256]{0} subtract(f32[256]{0} %reshape.376, f32[256]{0} %multiply.649), metadata={op_type="Sub" op_name="stage1_unit2_bn3/sub"}
  %broadcast.768 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.650), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit2_bn3/add_1"}
  %add.769 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %multiply.767, f32[1,56,56,256]{3,2,1,0} %broadcast.768), metadata={op_type="AddV2" op_name="stage1_unit2_bn3/add_1"}
  %add.770 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %maximum.629, f32[1,56,56,256]{3,2,1,0} %add.769), metadata={op_type="AddV2" op_name="add_1"}
  %maximum.773 = f32[1,56,56,256]{3,2,1,0} maximum(f32[1,56,56,256]{3,2,1,0} %broadcast.772, f32[1,56,56,256]{3,2,1,0} %add.770), metadata={op_type="Relu" op_name="stage1_unit2_relu"}
  %constant.788 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit3_bn3/add"}
  %broadcast.789 = f32[256]{0} broadcast(f32[] %constant.788), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit3_bn3/add"}
  %arg20.21 = f32[256]{0} parameter(20), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.240 = f32[256]{0} reshape(f32[256]{0} %arg20.21)
  %add.790 = f32[256]{0} add(f32[256]{0} %broadcast.789, f32[256]{0} %reshape.240), metadata={op_type="AddV2" op_name="stage1_unit3_bn3/add"}
  %rsqrt.791 = f32[256]{0} rsqrt(f32[256]{0} %add.790), metadata={op_type="Rsqrt" op_name="stage1_unit3_bn3/Rsqrt"}
  %arg72.73 = f32[256]{0} parameter(72), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.292 = f32[256]{0} reshape(f32[256]{0} %arg72.73)
  %multiply.792 = f32[256]{0} multiply(f32[256]{0} %rsqrt.791, f32[256]{0} %reshape.292), metadata={op_type="Mul" op_name="stage1_unit3_bn3/mul"}
  %broadcast.910 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.792), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit3_bn3/mul_1"}
  %constant.906 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit3_relu2"}
  %broadcast.907 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[] %constant.906), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit3_relu2"}
  %constant.800 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit3_relu1"}
  %broadcast.801 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[] %constant.800), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit3_relu1"}
  %constant.774 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit3_bn1/add"}
  %broadcast.775 = f32[128]{0} broadcast(f32[] %constant.774), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit3_bn1/add"}
  %arg16.17 = f32[128]{0} parameter(16), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.236 = f32[128]{0} reshape(f32[128]{0} %arg16.17)
  %add.776 = f32[128]{0} add(f32[128]{0} %broadcast.775, f32[128]{0} %reshape.236), metadata={op_type="AddV2" op_name="stage1_unit3_bn1/add"}
  %rsqrt.777 = f32[128]{0} rsqrt(f32[128]{0} %add.776), metadata={op_type="Rsqrt" op_name="stage1_unit3_bn1/Rsqrt"}
  %arg70.71 = f32[128]{0} parameter(70), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.290 = f32[128]{0} reshape(f32[128]{0} %arg70.71)
  %multiply.778 = f32[128]{0} multiply(f32[128]{0} %rsqrt.777, f32[128]{0} %reshape.290), metadata={op_type="Mul" op_name="stage1_unit3_bn1/mul"}
  %broadcast.796 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %multiply.778), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit3_bn1/mul_1"}
  %arg194.195 = f32[1,1,256,128]{3,2,1,0} parameter(194), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.414 = f32[1,1,256,128]{3,2,1,0} reshape(f32[1,1,256,128]{3,2,1,0} %arg194.195)
  %convolution.795 = f32[1,56,56,128]{3,2,1,0} convolution(f32[1,56,56,256]{3,2,1,0} %maximum.773, f32[1,1,256,128]{3,2,1,0} %reshape.414), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv1"}
  %multiply.797 = f32[1,56,56,128]{3,2,1,0} multiply(f32[1,56,56,128]{3,2,1,0} %broadcast.796, f32[1,56,56,128]{3,2,1,0} %convolution.795), metadata={op_type="Mul" op_name="stage1_unit3_bn1/mul_1"}
  %arg157.158 = f32[128]{0} parameter(157), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.377 = f32[128]{0} reshape(f32[128]{0} %arg157.158)
  %arg114.115 = f32[128]{0} parameter(114), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.334 = f32[128]{0} reshape(f32[128]{0} %arg114.115)
  %multiply.779 = f32[128]{0} multiply(f32[128]{0} %multiply.778, f32[128]{0} %reshape.334), metadata={op_type="Mul" op_name="stage1_unit3_bn1/mul_2"}
  %subtract.780 = f32[128]{0} subtract(f32[128]{0} %reshape.377, f32[128]{0} %multiply.779), metadata={op_type="Sub" op_name="stage1_unit3_bn1/sub"}
  %broadcast.798 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %subtract.780), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit3_bn1/add_1"}
  %add.799 = f32[1,56,56,128]{3,2,1,0} add(f32[1,56,56,128]{3,2,1,0} %multiply.797, f32[1,56,56,128]{3,2,1,0} %broadcast.798), metadata={op_type="AddV2" op_name="stage1_unit3_bn1/add_1"}
  %maximum.802 = f32[1,56,56,128]{3,2,1,0} maximum(f32[1,56,56,128]{3,2,1,0} %broadcast.801, f32[1,56,56,128]{3,2,1,0} %add.799), metadata={op_type="Relu" op_name="stage1_unit3_relu1"}
  %constant.803 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_3"}
  %pad.804 = f32[1,58,58,128]{3,2,1,0} pad(f32[1,56,56,128]{3,2,1,0} %maximum.802, f32[] %constant.803), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_3"}
  %slice.805 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.804), slice={[0:1], [0:58], [0:58], [0:4]}, metadata={op_type="Split" op_name="split_5"}
  %arg18.19 = f32[3,3,4,128]{3,2,1,0} parameter(18), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.238 = f32[3,3,4,128]{3,2,1,0} reshape(f32[3,3,4,128]{3,2,1,0} %arg18.19)
  %slice.837 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.238), slice={[0:3], [0:3], [0:4], [0:4]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.869 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.805, f32[3,3,4,4]{3,2,1,0} %slice.837), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2"}
  %slice.806 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.804), slice={[0:1], [0:58], [0:58], [4:8]}, metadata={op_type="Split" op_name="split_5"}
  %slice.838 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.238), slice={[0:3], [0:3], [0:4], [4:8]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.870 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.806, f32[3,3,4,4]{3,2,1,0} %slice.838), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_1"}
  %slice.807 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.804), slice={[0:1], [0:58], [0:58], [8:12]}, metadata={op_type="Split" op_name="split_5"}
  %slice.839 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.238), slice={[0:3], [0:3], [0:4], [8:12]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.881 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.807, f32[3,3,4,4]{3,2,1,0} %slice.839), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_2"}
  %slice.808 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.804), slice={[0:1], [0:58], [0:58], [12:16]}, metadata={op_type="Split" op_name="split_5"}
  %slice.840 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.238), slice={[0:3], [0:3], [0:4], [12:16]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.892 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.808, f32[3,3,4,4]{3,2,1,0} %slice.840), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_3"}
  %slice.809 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.804), slice={[0:1], [0:58], [0:58], [16:20]}, metadata={op_type="Split" op_name="split_5"}
  %slice.841 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.238), slice={[0:3], [0:3], [0:4], [16:20]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.895 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.809, f32[3,3,4,4]{3,2,1,0} %slice.841), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_4"}
  %slice.810 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.804), slice={[0:1], [0:58], [0:58], [20:24]}, metadata={op_type="Split" op_name="split_5"}
  %slice.842 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.238), slice={[0:3], [0:3], [0:4], [20:24]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.896 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.810, f32[3,3,4,4]{3,2,1,0} %slice.842), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_5"}
  %slice.811 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.804), slice={[0:1], [0:58], [0:58], [24:28]}, metadata={op_type="Split" op_name="split_5"}
  %slice.843 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.238), slice={[0:3], [0:3], [0:4], [24:28]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.897 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.811, f32[3,3,4,4]{3,2,1,0} %slice.843), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_6"}
  %slice.812 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.804), slice={[0:1], [0:58], [0:58], [28:32]}, metadata={op_type="Split" op_name="split_5"}
  %slice.844 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.238), slice={[0:3], [0:3], [0:4], [28:32]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.898 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.812, f32[3,3,4,4]{3,2,1,0} %slice.844), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_7"}
  %slice.813 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.804), slice={[0:1], [0:58], [0:58], [32:36]}, metadata={op_type="Split" op_name="split_5"}
  %slice.845 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.238), slice={[0:3], [0:3], [0:4], [32:36]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.899 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.813, f32[3,3,4,4]{3,2,1,0} %slice.845), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_8"}
  %slice.814 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.804), slice={[0:1], [0:58], [0:58], [36:40]}, metadata={op_type="Split" op_name="split_5"}
  %slice.846 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.238), slice={[0:3], [0:3], [0:4], [36:40]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.900 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.814, f32[3,3,4,4]{3,2,1,0} %slice.846), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_9"}
  %slice.815 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.804), slice={[0:1], [0:58], [0:58], [40:44]}, metadata={op_type="Split" op_name="split_5"}
  %slice.847 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.238), slice={[0:3], [0:3], [0:4], [40:44]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.871 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.815, f32[3,3,4,4]{3,2,1,0} %slice.847), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_10"}
  %slice.816 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.804), slice={[0:1], [0:58], [0:58], [44:48]}, metadata={op_type="Split" op_name="split_5"}
  %slice.848 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.238), slice={[0:3], [0:3], [0:4], [44:48]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.872 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.816, f32[3,3,4,4]{3,2,1,0} %slice.848), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_11"}
  %slice.817 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.804), slice={[0:1], [0:58], [0:58], [48:52]}, metadata={op_type="Split" op_name="split_5"}
  %slice.849 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.238), slice={[0:3], [0:3], [0:4], [48:52]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.873 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.817, f32[3,3,4,4]{3,2,1,0} %slice.849), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_12"}
  %slice.818 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.804), slice={[0:1], [0:58], [0:58], [52:56]}, metadata={op_type="Split" op_name="split_5"}
  %slice.850 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.238), slice={[0:3], [0:3], [0:4], [52:56]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.874 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.818, f32[3,3,4,4]{3,2,1,0} %slice.850), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_13"}
  %slice.819 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.804), slice={[0:1], [0:58], [0:58], [56:60]}, metadata={op_type="Split" op_name="split_5"}
  %slice.851 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.238), slice={[0:3], [0:3], [0:4], [56:60]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.875 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.819, f32[3,3,4,4]{3,2,1,0} %slice.851), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_14"}
  %slice.820 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.804), slice={[0:1], [0:58], [0:58], [60:64]}, metadata={op_type="Split" op_name="split_5"}
  %slice.852 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.238), slice={[0:3], [0:3], [0:4], [60:64]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.876 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.820, f32[3,3,4,4]{3,2,1,0} %slice.852), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_15"}
  %slice.821 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.804), slice={[0:1], [0:58], [0:58], [64:68]}, metadata={op_type="Split" op_name="split_5"}
  %slice.853 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.238), slice={[0:3], [0:3], [0:4], [64:68]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.877 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.821, f32[3,3,4,4]{3,2,1,0} %slice.853), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_16"}
  %slice.822 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.804), slice={[0:1], [0:58], [0:58], [68:72]}, metadata={op_type="Split" op_name="split_5"}
  %slice.854 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.238), slice={[0:3], [0:3], [0:4], [68:72]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.878 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.822, f32[3,3,4,4]{3,2,1,0} %slice.854), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_17"}
  %slice.823 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.804), slice={[0:1], [0:58], [0:58], [72:76]}, metadata={op_type="Split" op_name="split_5"}
  %slice.855 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.238), slice={[0:3], [0:3], [0:4], [72:76]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.879 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.823, f32[3,3,4,4]{3,2,1,0} %slice.855), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_18"}
  %slice.824 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.804), slice={[0:1], [0:58], [0:58], [76:80]}, metadata={op_type="Split" op_name="split_5"}
  %slice.856 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.238), slice={[0:3], [0:3], [0:4], [76:80]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.880 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.824, f32[3,3,4,4]{3,2,1,0} %slice.856), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_19"}
  %slice.825 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.804), slice={[0:1], [0:58], [0:58], [80:84]}, metadata={op_type="Split" op_name="split_5"}
  %slice.857 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.238), slice={[0:3], [0:3], [0:4], [80:84]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.882 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.825, f32[3,3,4,4]{3,2,1,0} %slice.857), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_20"}
  %slice.826 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.804), slice={[0:1], [0:58], [0:58], [84:88]}, metadata={op_type="Split" op_name="split_5"}
  %slice.858 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.238), slice={[0:3], [0:3], [0:4], [84:88]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.883 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.826, f32[3,3,4,4]{3,2,1,0} %slice.858), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_21"}
  %slice.827 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.804), slice={[0:1], [0:58], [0:58], [88:92]}, metadata={op_type="Split" op_name="split_5"}
  %slice.859 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.238), slice={[0:3], [0:3], [0:4], [88:92]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.884 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.827, f32[3,3,4,4]{3,2,1,0} %slice.859), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_22"}
  %slice.828 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.804), slice={[0:1], [0:58], [0:58], [92:96]}, metadata={op_type="Split" op_name="split_5"}
  %slice.860 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.238), slice={[0:3], [0:3], [0:4], [92:96]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.885 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.828, f32[3,3,4,4]{3,2,1,0} %slice.860), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_23"}
  %slice.829 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.804), slice={[0:1], [0:58], [0:58], [96:100]}, metadata={op_type="Split" op_name="split_5"}
  %slice.861 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.238), slice={[0:3], [0:3], [0:4], [96:100]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.886 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.829, f32[3,3,4,4]{3,2,1,0} %slice.861), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_24"}
  %slice.830 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.804), slice={[0:1], [0:58], [0:58], [100:104]}, metadata={op_type="Split" op_name="split_5"}
  %slice.862 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.238), slice={[0:3], [0:3], [0:4], [100:104]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.887 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.830, f32[3,3,4,4]{3,2,1,0} %slice.862), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_25"}
  %slice.831 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.804), slice={[0:1], [0:58], [0:58], [104:108]}, metadata={op_type="Split" op_name="split_5"}
  %slice.863 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.238), slice={[0:3], [0:3], [0:4], [104:108]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.888 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.831, f32[3,3,4,4]{3,2,1,0} %slice.863), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_26"}
  %slice.832 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.804), slice={[0:1], [0:58], [0:58], [108:112]}, metadata={op_type="Split" op_name="split_5"}
  %slice.864 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.238), slice={[0:3], [0:3], [0:4], [108:112]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.889 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.832, f32[3,3,4,4]{3,2,1,0} %slice.864), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_27"}
  %slice.833 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.804), slice={[0:1], [0:58], [0:58], [112:116]}, metadata={op_type="Split" op_name="split_5"}
  %slice.865 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.238), slice={[0:3], [0:3], [0:4], [112:116]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.890 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.833, f32[3,3,4,4]{3,2,1,0} %slice.865), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_28"}
  %slice.834 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.804), slice={[0:1], [0:58], [0:58], [116:120]}, metadata={op_type="Split" op_name="split_5"}
  %slice.866 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.238), slice={[0:3], [0:3], [0:4], [116:120]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.891 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.834, f32[3,3,4,4]{3,2,1,0} %slice.866), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_29"}
  %slice.835 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.804), slice={[0:1], [0:58], [0:58], [120:124]}, metadata={op_type="Split" op_name="split_5"}
  %slice.867 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.238), slice={[0:3], [0:3], [0:4], [120:124]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.893 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.835, f32[3,3,4,4]{3,2,1,0} %slice.867), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_30"}
  %slice.836 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.804), slice={[0:1], [0:58], [0:58], [124:128]}, metadata={op_type="Split" op_name="split_5"}
  %slice.868 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.238), slice={[0:3], [0:3], [0:4], [124:128]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.894 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.836, f32[3,3,4,4]{3,2,1,0} %slice.868), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_31"}
  %concatenate.901 = f32[1,56,56,128]{3,2,1,0} concatenate(f32[1,56,56,4]{3,2,1,0} %convolution.869, f32[1,56,56,4]{3,2,1,0} %convolution.870, f32[1,56,56,4]{3,2,1,0} %convolution.881, f32[1,56,56,4]{3,2,1,0} %convolution.892, f32[1,56,56,4]{3,2,1,0} %convolution.895, f32[1,56,56,4]{3,2,1,0} %convolution.896, f32[1,56,56,4]{3,2,1,0} %convolution.897, f32[1,56,56,4]{3,2,1,0} %convolution.898, f32[1,56,56,4]{3,2,1,0} %convolution.899, f32[1,56,56,4]{3,2,1,0} %convolution.900, f32[1,56,56,4]{3,2,1,0} %convolution.871, f32[1,56,56,4]{3,2,1,0} %convolution.872, f32[1,56,56,4]{3,2,1,0} %convolution.873, f32[1,56,56,4]{3,2,1,0} %convolution.874, f32[1,56,56,4]{3,2,1,0} %convolution.875, f32[1,56,56,4]{3,2,1,0} %convolution.876, f32[1,56,56,4]{3,2,1,0} %convolution.877, f32[1,56,56,4]{3,2,1,0} %convolution.878, f32[1,56,56,4]{3,2,1,0} %convolution.879, f32[1,56,56,4]{3,2,1,0} %convolution.880, f32[1,56,56,4]{3,2,1,0} %convolution.882, f32[1,56,56,4]{3,2,1,0} %convolution.883, f32[1,56,56,4]{3,2,1,0} %convolution.884, f32[1,56,56,4]{3,2,1,0} %convolution.885, f32[1,56,56,4]{3,2,1,0} %convolution.886, f32[1,56,56,4]{3,2,1,0} %convolution.887, f32[1,56,56,4]{3,2,1,0} %convolution.888, f32[1,56,56,4]{3,2,1,0} %convolution.889, f32[1,56,56,4]{3,2,1,0} %convolution.890, f32[1,56,56,4]{3,2,1,0} %convolution.891, f32[1,56,56,4]{3,2,1,0} %convolution.893, f32[1,56,56,4]{3,2,1,0} %convolution.894), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_2"}
  %constant.781 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit3_bn2/add"}
  %broadcast.782 = f32[128]{0} broadcast(f32[] %constant.781), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit3_bn2/add"}
  %arg19.20 = f32[128]{0} parameter(19), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.239 = f32[128]{0} reshape(f32[128]{0} %arg19.20)
  %add.783 = f32[128]{0} add(f32[128]{0} %broadcast.782, f32[128]{0} %reshape.239), metadata={op_type="AddV2" op_name="stage1_unit3_bn2/add"}
  %rsqrt.784 = f32[128]{0} rsqrt(f32[128]{0} %add.783), metadata={op_type="Rsqrt" op_name="stage1_unit3_bn2/Rsqrt"}
  %arg71.72 = f32[128]{0} parameter(71), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.291 = f32[128]{0} reshape(f32[128]{0} %arg71.72)
  %multiply.785 = f32[128]{0} multiply(f32[128]{0} %rsqrt.784, f32[128]{0} %reshape.291), metadata={op_type="Mul" op_name="stage1_unit3_bn2/mul"}
  %broadcast.902 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %multiply.785), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit3_bn2/mul_1"}
  %multiply.903 = f32[1,56,56,128]{3,2,1,0} multiply(f32[1,56,56,128]{3,2,1,0} %concatenate.901, f32[1,56,56,128]{3,2,1,0} %broadcast.902), metadata={op_type="Mul" op_name="stage1_unit3_bn2/mul_1"}
  %arg158.159 = f32[128]{0} parameter(158), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.378 = f32[128]{0} reshape(f32[128]{0} %arg158.159)
  %arg115.116 = f32[128]{0} parameter(115), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.335 = f32[128]{0} reshape(f32[128]{0} %arg115.116)
  %multiply.786 = f32[128]{0} multiply(f32[128]{0} %multiply.785, f32[128]{0} %reshape.335), metadata={op_type="Mul" op_name="stage1_unit3_bn2/mul_2"}
  %subtract.787 = f32[128]{0} subtract(f32[128]{0} %reshape.378, f32[128]{0} %multiply.786), metadata={op_type="Sub" op_name="stage1_unit3_bn2/sub"}
  %broadcast.904 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %subtract.787), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit3_bn2/add_1"}
  %add.905 = f32[1,56,56,128]{3,2,1,0} add(f32[1,56,56,128]{3,2,1,0} %multiply.903, f32[1,56,56,128]{3,2,1,0} %broadcast.904), metadata={op_type="AddV2" op_name="stage1_unit3_bn2/add_1"}
  %maximum.908 = f32[1,56,56,128]{3,2,1,0} maximum(f32[1,56,56,128]{3,2,1,0} %broadcast.907, f32[1,56,56,128]{3,2,1,0} %add.905), metadata={op_type="Relu" op_name="stage1_unit3_relu2"}
  %arg195.196 = f32[1,1,128,256]{3,2,1,0} parameter(195), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.415 = f32[1,1,128,256]{3,2,1,0} reshape(f32[1,1,128,256]{3,2,1,0} %arg195.196)
  %convolution.909 = f32[1,56,56,256]{3,2,1,0} convolution(f32[1,56,56,128]{3,2,1,0} %maximum.908, f32[1,1,128,256]{3,2,1,0} %reshape.415), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv3"}
  %multiply.911 = f32[1,56,56,256]{3,2,1,0} multiply(f32[1,56,56,256]{3,2,1,0} %broadcast.910, f32[1,56,56,256]{3,2,1,0} %convolution.909), metadata={op_type="Mul" op_name="stage1_unit3_bn3/mul_1"}
  %arg159.160 = f32[256]{0} parameter(159), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.379 = f32[256]{0} reshape(f32[256]{0} %arg159.160)
  %arg116.117 = f32[256]{0} parameter(116), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.336 = f32[256]{0} reshape(f32[256]{0} %arg116.117)
  %multiply.793 = f32[256]{0} multiply(f32[256]{0} %multiply.792, f32[256]{0} %reshape.336), metadata={op_type="Mul" op_name="stage1_unit3_bn3/mul_2"}
  %subtract.794 = f32[256]{0} subtract(f32[256]{0} %reshape.379, f32[256]{0} %multiply.793), metadata={op_type="Sub" op_name="stage1_unit3_bn3/sub"}
  %broadcast.912 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.794), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit3_bn3/add_1"}
  %add.913 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %multiply.911, f32[1,56,56,256]{3,2,1,0} %broadcast.912), metadata={op_type="AddV2" op_name="stage1_unit3_bn3/add_1"}
  %add.914 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %maximum.773, f32[1,56,56,256]{3,2,1,0} %add.913), metadata={op_type="AddV2" op_name="add_2"}
  %maximum.917 = f32[1,56,56,256]{3,2,1,0} maximum(f32[1,56,56,256]{3,2,1,0} %broadcast.916, f32[1,56,56,256]{3,2,1,0} %add.914), metadata={op_type="Relu" op_name="stage1_unit3_relu"}
  %arg196.197 = f32[1,1,256,256]{3,2,1,0} parameter(196), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.416 = f32[1,1,256,256]{3,2,1,0} reshape(f32[1,1,256,256]{3,2,1,0} %arg196.197)
  %convolution.939 = f32[1,56,56,256]{3,2,1,0} convolution(f32[1,56,56,256]{3,2,1,0} %maximum.917, f32[1,1,256,256]{3,2,1,0} %reshape.416), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv1"}
  %multiply.941 = f32[1,56,56,256]{3,2,1,0} multiply(f32[1,56,56,256]{3,2,1,0} %broadcast.940, f32[1,56,56,256]{3,2,1,0} %convolution.939), metadata={op_type="Mul" op_name="stage2_unit1_bn1/mul_1"}
  %arg160.161 = f32[256]{0} parameter(160), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.380 = f32[256]{0} reshape(f32[256]{0} %arg160.161)
  %arg117.118 = f32[256]{0} parameter(117), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.337 = f32[256]{0} reshape(f32[256]{0} %arg117.118)
  %multiply.923 = f32[256]{0} multiply(f32[256]{0} %multiply.922, f32[256]{0} %reshape.337), metadata={op_type="Mul" op_name="stage2_unit1_bn1/mul_2"}
  %subtract.924 = f32[256]{0} subtract(f32[256]{0} %reshape.380, f32[256]{0} %multiply.923), metadata={op_type="Sub" op_name="stage2_unit1_bn1/sub"}
  %broadcast.942 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.924), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit1_bn1/add_1"}
  %add.943 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %multiply.941, f32[1,56,56,256]{3,2,1,0} %broadcast.942), metadata={op_type="AddV2" op_name="stage2_unit1_bn1/add_1"}
  %maximum.946 = f32[1,56,56,256]{3,2,1,0} maximum(f32[1,56,56,256]{3,2,1,0} %broadcast.945, f32[1,56,56,256]{3,2,1,0} %add.943), metadata={op_type="Relu" op_name="stage2_unit1_relu1"}
  %constant.947 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_4"}
  %pad.948 = f32[1,58,58,256]{3,2,1,0} pad(f32[1,56,56,256]{3,2,1,0} %maximum.946, f32[] %constant.947), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_4"}
  %slice.949 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.948), slice={[0:1], [0:58], [0:58], [0:8]}, metadata={op_type="Split" op_name="split_7"}
  %arg23.24 = f32[3,3,8,256]{3,2,1,0} parameter(23), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.243 = f32[3,3,8,256]{3,2,1,0} reshape(f32[3,3,8,256]{3,2,1,0} %arg23.24)
  %slice.981 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.243), slice={[0:3], [0:3], [0:8], [0:8]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1013 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.949, f32[3,3,8,8]{3,2,1,0} %slice.981), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2"}
  %slice.950 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.948), slice={[0:1], [0:58], [0:58], [8:16]}, metadata={op_type="Split" op_name="split_7"}
  %slice.982 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.243), slice={[0:3], [0:3], [0:8], [8:16]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1014 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.950, f32[3,3,8,8]{3,2,1,0} %slice.982), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_1"}
  %slice.951 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.948), slice={[0:1], [0:58], [0:58], [16:24]}, metadata={op_type="Split" op_name="split_7"}
  %slice.983 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.243), slice={[0:3], [0:3], [0:8], [16:24]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1025 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.951, f32[3,3,8,8]{3,2,1,0} %slice.983), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_2"}
  %slice.952 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.948), slice={[0:1], [0:58], [0:58], [24:32]}, metadata={op_type="Split" op_name="split_7"}
  %slice.984 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.243), slice={[0:3], [0:3], [0:8], [24:32]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1036 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.952, f32[3,3,8,8]{3,2,1,0} %slice.984), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_3"}
  %slice.953 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.948), slice={[0:1], [0:58], [0:58], [32:40]}, metadata={op_type="Split" op_name="split_7"}
  %slice.985 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.243), slice={[0:3], [0:3], [0:8], [32:40]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1039 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.953, f32[3,3,8,8]{3,2,1,0} %slice.985), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_4"}
  %slice.954 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.948), slice={[0:1], [0:58], [0:58], [40:48]}, metadata={op_type="Split" op_name="split_7"}
  %slice.986 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.243), slice={[0:3], [0:3], [0:8], [40:48]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1040 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.954, f32[3,3,8,8]{3,2,1,0} %slice.986), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_5"}
  %slice.955 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.948), slice={[0:1], [0:58], [0:58], [48:56]}, metadata={op_type="Split" op_name="split_7"}
  %slice.987 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.243), slice={[0:3], [0:3], [0:8], [48:56]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1041 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.955, f32[3,3,8,8]{3,2,1,0} %slice.987), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_6"}
  %slice.956 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.948), slice={[0:1], [0:58], [0:58], [56:64]}, metadata={op_type="Split" op_name="split_7"}
  %slice.988 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.243), slice={[0:3], [0:3], [0:8], [56:64]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1042 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.956, f32[3,3,8,8]{3,2,1,0} %slice.988), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_7"}
  %slice.957 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.948), slice={[0:1], [0:58], [0:58], [64:72]}, metadata={op_type="Split" op_name="split_7"}
  %slice.989 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.243), slice={[0:3], [0:3], [0:8], [64:72]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1043 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.957, f32[3,3,8,8]{3,2,1,0} %slice.989), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_8"}
  %slice.958 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.948), slice={[0:1], [0:58], [0:58], [72:80]}, metadata={op_type="Split" op_name="split_7"}
  %slice.990 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.243), slice={[0:3], [0:3], [0:8], [72:80]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1044 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.958, f32[3,3,8,8]{3,2,1,0} %slice.990), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_9"}
  %slice.959 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.948), slice={[0:1], [0:58], [0:58], [80:88]}, metadata={op_type="Split" op_name="split_7"}
  %slice.991 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.243), slice={[0:3], [0:3], [0:8], [80:88]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1015 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.959, f32[3,3,8,8]{3,2,1,0} %slice.991), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_10"}
  %slice.960 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.948), slice={[0:1], [0:58], [0:58], [88:96]}, metadata={op_type="Split" op_name="split_7"}
  %slice.992 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.243), slice={[0:3], [0:3], [0:8], [88:96]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1016 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.960, f32[3,3,8,8]{3,2,1,0} %slice.992), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_11"}
  %slice.961 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.948), slice={[0:1], [0:58], [0:58], [96:104]}, metadata={op_type="Split" op_name="split_7"}
  %slice.993 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.243), slice={[0:3], [0:3], [0:8], [96:104]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1017 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.961, f32[3,3,8,8]{3,2,1,0} %slice.993), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_12"}
  %slice.962 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.948), slice={[0:1], [0:58], [0:58], [104:112]}, metadata={op_type="Split" op_name="split_7"}
  %slice.994 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.243), slice={[0:3], [0:3], [0:8], [104:112]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1018 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.962, f32[3,3,8,8]{3,2,1,0} %slice.994), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_13"}
  %slice.963 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.948), slice={[0:1], [0:58], [0:58], [112:120]}, metadata={op_type="Split" op_name="split_7"}
  %slice.995 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.243), slice={[0:3], [0:3], [0:8], [112:120]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1019 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.963, f32[3,3,8,8]{3,2,1,0} %slice.995), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_14"}
  %slice.964 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.948), slice={[0:1], [0:58], [0:58], [120:128]}, metadata={op_type="Split" op_name="split_7"}
  %slice.996 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.243), slice={[0:3], [0:3], [0:8], [120:128]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1020 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.964, f32[3,3,8,8]{3,2,1,0} %slice.996), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_15"}
  %slice.965 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.948), slice={[0:1], [0:58], [0:58], [128:136]}, metadata={op_type="Split" op_name="split_7"}
  %slice.997 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.243), slice={[0:3], [0:3], [0:8], [128:136]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1021 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.965, f32[3,3,8,8]{3,2,1,0} %slice.997), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_16"}
  %slice.966 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.948), slice={[0:1], [0:58], [0:58], [136:144]}, metadata={op_type="Split" op_name="split_7"}
  %slice.998 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.243), slice={[0:3], [0:3], [0:8], [136:144]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1022 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.966, f32[3,3,8,8]{3,2,1,0} %slice.998), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_17"}
  %slice.967 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.948), slice={[0:1], [0:58], [0:58], [144:152]}, metadata={op_type="Split" op_name="split_7"}
  %slice.999 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.243), slice={[0:3], [0:3], [0:8], [144:152]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1023 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.967, f32[3,3,8,8]{3,2,1,0} %slice.999), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_18"}
  %slice.968 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.948), slice={[0:1], [0:58], [0:58], [152:160]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1000 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.243), slice={[0:3], [0:3], [0:8], [152:160]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1024 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.968, f32[3,3,8,8]{3,2,1,0} %slice.1000), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_19"}
  %slice.969 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.948), slice={[0:1], [0:58], [0:58], [160:168]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1001 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.243), slice={[0:3], [0:3], [0:8], [160:168]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1026 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.969, f32[3,3,8,8]{3,2,1,0} %slice.1001), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_20"}
  %slice.970 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.948), slice={[0:1], [0:58], [0:58], [168:176]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1002 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.243), slice={[0:3], [0:3], [0:8], [168:176]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1027 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.970, f32[3,3,8,8]{3,2,1,0} %slice.1002), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_21"}
  %slice.971 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.948), slice={[0:1], [0:58], [0:58], [176:184]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1003 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.243), slice={[0:3], [0:3], [0:8], [176:184]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1028 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.971, f32[3,3,8,8]{3,2,1,0} %slice.1003), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_22"}
  %slice.972 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.948), slice={[0:1], [0:58], [0:58], [184:192]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1004 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.243), slice={[0:3], [0:3], [0:8], [184:192]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1029 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.972, f32[3,3,8,8]{3,2,1,0} %slice.1004), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_23"}
  %slice.973 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.948), slice={[0:1], [0:58], [0:58], [192:200]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1005 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.243), slice={[0:3], [0:3], [0:8], [192:200]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1030 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.973, f32[3,3,8,8]{3,2,1,0} %slice.1005), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_24"}
  %slice.974 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.948), slice={[0:1], [0:58], [0:58], [200:208]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1006 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.243), slice={[0:3], [0:3], [0:8], [200:208]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1031 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.974, f32[3,3,8,8]{3,2,1,0} %slice.1006), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_25"}
  %slice.975 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.948), slice={[0:1], [0:58], [0:58], [208:216]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1007 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.243), slice={[0:3], [0:3], [0:8], [208:216]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1032 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.975, f32[3,3,8,8]{3,2,1,0} %slice.1007), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_26"}
  %slice.976 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.948), slice={[0:1], [0:58], [0:58], [216:224]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1008 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.243), slice={[0:3], [0:3], [0:8], [216:224]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1033 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.976, f32[3,3,8,8]{3,2,1,0} %slice.1008), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_27"}
  %slice.977 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.948), slice={[0:1], [0:58], [0:58], [224:232]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1009 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.243), slice={[0:3], [0:3], [0:8], [224:232]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1034 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.977, f32[3,3,8,8]{3,2,1,0} %slice.1009), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_28"}
  %slice.978 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.948), slice={[0:1], [0:58], [0:58], [232:240]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1010 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.243), slice={[0:3], [0:3], [0:8], [232:240]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1035 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.978, f32[3,3,8,8]{3,2,1,0} %slice.1010), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_29"}
  %slice.979 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.948), slice={[0:1], [0:58], [0:58], [240:248]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1011 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.243), slice={[0:3], [0:3], [0:8], [240:248]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1037 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.979, f32[3,3,8,8]{3,2,1,0} %slice.1011), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_30"}
  %slice.980 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.948), slice={[0:1], [0:58], [0:58], [248:256]}, metadata={op_type="Split" op_name="split_7"}
  %slice.1012 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.243), slice={[0:3], [0:3], [0:8], [248:256]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1038 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.980, f32[3,3,8,8]{3,2,1,0} %slice.1012), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_31"}
  %concatenate.1045 = f32[1,28,28,256]{3,2,1,0} concatenate(f32[1,28,28,8]{3,2,1,0} %convolution.1013, f32[1,28,28,8]{3,2,1,0} %convolution.1014, f32[1,28,28,8]{3,2,1,0} %convolution.1025, f32[1,28,28,8]{3,2,1,0} %convolution.1036, f32[1,28,28,8]{3,2,1,0} %convolution.1039, f32[1,28,28,8]{3,2,1,0} %convolution.1040, f32[1,28,28,8]{3,2,1,0} %convolution.1041, f32[1,28,28,8]{3,2,1,0} %convolution.1042, f32[1,28,28,8]{3,2,1,0} %convolution.1043, f32[1,28,28,8]{3,2,1,0} %convolution.1044, f32[1,28,28,8]{3,2,1,0} %convolution.1015, f32[1,28,28,8]{3,2,1,0} %convolution.1016, f32[1,28,28,8]{3,2,1,0} %convolution.1017, f32[1,28,28,8]{3,2,1,0} %convolution.1018, f32[1,28,28,8]{3,2,1,0} %convolution.1019, f32[1,28,28,8]{3,2,1,0} %convolution.1020, f32[1,28,28,8]{3,2,1,0} %convolution.1021, f32[1,28,28,8]{3,2,1,0} %convolution.1022, f32[1,28,28,8]{3,2,1,0} %convolution.1023, f32[1,28,28,8]{3,2,1,0} %convolution.1024, f32[1,28,28,8]{3,2,1,0} %convolution.1026, f32[1,28,28,8]{3,2,1,0} %convolution.1027, f32[1,28,28,8]{3,2,1,0} %convolution.1028, f32[1,28,28,8]{3,2,1,0} %convolution.1029, f32[1,28,28,8]{3,2,1,0} %convolution.1030, f32[1,28,28,8]{3,2,1,0} %convolution.1031, f32[1,28,28,8]{3,2,1,0} %convolution.1032, f32[1,28,28,8]{3,2,1,0} %convolution.1033, f32[1,28,28,8]{3,2,1,0} %convolution.1034, f32[1,28,28,8]{3,2,1,0} %convolution.1035, f32[1,28,28,8]{3,2,1,0} %convolution.1037, f32[1,28,28,8]{3,2,1,0} %convolution.1038), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_3"}
  %constant.925 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit1_bn2/add"}
  %broadcast.926 = f32[256]{0} broadcast(f32[] %constant.925), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit1_bn2/add"}
  %arg25.26 = f32[256]{0} parameter(25), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.245 = f32[256]{0} reshape(f32[256]{0} %arg25.26)
  %add.927 = f32[256]{0} add(f32[256]{0} %broadcast.926, f32[256]{0} %reshape.245), metadata={op_type="AddV2" op_name="stage2_unit1_bn2/add"}
  %rsqrt.928 = f32[256]{0} rsqrt(f32[256]{0} %add.927), metadata={op_type="Rsqrt" op_name="stage2_unit1_bn2/Rsqrt"}
  %arg76.77 = f32[256]{0} parameter(76), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.296 = f32[256]{0} reshape(f32[256]{0} %arg76.77)
  %multiply.929 = f32[256]{0} multiply(f32[256]{0} %rsqrt.928, f32[256]{0} %reshape.296), metadata={op_type="Mul" op_name="stage2_unit1_bn2/mul"}
  %broadcast.1046 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.929), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit1_bn2/mul_1"}
  %multiply.1047 = f32[1,28,28,256]{3,2,1,0} multiply(f32[1,28,28,256]{3,2,1,0} %concatenate.1045, f32[1,28,28,256]{3,2,1,0} %broadcast.1046), metadata={op_type="Mul" op_name="stage2_unit1_bn2/mul_1"}
  %arg163.164 = f32[256]{0} parameter(163), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.383 = f32[256]{0} reshape(f32[256]{0} %arg163.164)
  %arg120.121 = f32[256]{0} parameter(120), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.340 = f32[256]{0} reshape(f32[256]{0} %arg120.121)
  %multiply.930 = f32[256]{0} multiply(f32[256]{0} %multiply.929, f32[256]{0} %reshape.340), metadata={op_type="Mul" op_name="stage2_unit1_bn2/mul_2"}
  %subtract.931 = f32[256]{0} subtract(f32[256]{0} %reshape.383, f32[256]{0} %multiply.930), metadata={op_type="Sub" op_name="stage2_unit1_bn2/sub"}
  %broadcast.1048 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.931), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit1_bn2/add_1"}
  %add.1049 = f32[1,28,28,256]{3,2,1,0} add(f32[1,28,28,256]{3,2,1,0} %multiply.1047, f32[1,28,28,256]{3,2,1,0} %broadcast.1048), metadata={op_type="AddV2" op_name="stage2_unit1_bn2/add_1"}
  %maximum.1052 = f32[1,28,28,256]{3,2,1,0} maximum(f32[1,28,28,256]{3,2,1,0} %broadcast.1051, f32[1,28,28,256]{3,2,1,0} %add.1049), metadata={op_type="Relu" op_name="stage2_unit1_relu2"}
  %arg198.199 = f32[1,1,256,512]{3,2,1,0} parameter(198), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.418 = f32[1,1,256,512]{3,2,1,0} reshape(f32[1,1,256,512]{3,2,1,0} %arg198.199)
  %convolution.1053 = f32[1,28,28,512]{3,2,1,0} convolution(f32[1,28,28,256]{3,2,1,0} %maximum.1052, f32[1,1,256,512]{3,2,1,0} %reshape.418), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv3"}
  %multiply.1055 = f32[1,28,28,512]{3,2,1,0} multiply(f32[1,28,28,512]{3,2,1,0} %broadcast.1054, f32[1,28,28,512]{3,2,1,0} %convolution.1053), metadata={op_type="Mul" op_name="stage2_unit1_bn3/mul_1"}
  %arg164.165 = f32[512]{0} parameter(164), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.384 = f32[512]{0} reshape(f32[512]{0} %arg164.165)
  %arg121.122 = f32[512]{0} parameter(121), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.341 = f32[512]{0} reshape(f32[512]{0} %arg121.122)
  %multiply.937 = f32[512]{0} multiply(f32[512]{0} %multiply.936, f32[512]{0} %reshape.341), metadata={op_type="Mul" op_name="stage2_unit1_bn3/mul_2"}
  %subtract.938 = f32[512]{0} subtract(f32[512]{0} %reshape.384, f32[512]{0} %multiply.937), metadata={op_type="Sub" op_name="stage2_unit1_bn3/sub"}
  %broadcast.1056 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.938), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit1_bn3/add_1"}
  %add.1057 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %multiply.1055, f32[1,28,28,512]{3,2,1,0} %broadcast.1056), metadata={op_type="AddV2" op_name="stage2_unit1_bn3/add_1"}
  %arg197.198 = f32[1,1,256,512]{3,2,1,0} parameter(197), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.417 = f32[1,1,256,512]{3,2,1,0} reshape(f32[1,1,256,512]{3,2,1,0} %arg197.198)
  %convolution.1065 = f32[1,28,28,512]{3,2,1,0} convolution(f32[1,56,56,256]{3,2,1,0} %maximum.917, f32[1,1,256,512]{3,2,1,0} %reshape.417), window={size=1x1 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_sc"}
  %constant.1058 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit1_sc_bn/add"}
  %broadcast.1059 = f32[512]{0} broadcast(f32[] %constant.1058), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit1_sc_bn/add"}
  %arg22.23 = f32[512]{0} parameter(22), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.242 = f32[512]{0} reshape(f32[512]{0} %arg22.23)
  %add.1060 = f32[512]{0} add(f32[512]{0} %broadcast.1059, f32[512]{0} %reshape.242), metadata={op_type="AddV2" op_name="stage2_unit1_sc_bn/add"}
  %rsqrt.1061 = f32[512]{0} rsqrt(f32[512]{0} %add.1060), metadata={op_type="Rsqrt" op_name="stage2_unit1_sc_bn/Rsqrt"}
  %arg74.75 = f32[512]{0} parameter(74), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.294 = f32[512]{0} reshape(f32[512]{0} %arg74.75)
  %multiply.1062 = f32[512]{0} multiply(f32[512]{0} %rsqrt.1061, f32[512]{0} %reshape.294), metadata={op_type="Mul" op_name="stage2_unit1_sc_bn/mul"}
  %broadcast.1066 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.1062), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit1_sc_bn/mul_1"}
  %multiply.1067 = f32[1,28,28,512]{3,2,1,0} multiply(f32[1,28,28,512]{3,2,1,0} %convolution.1065, f32[1,28,28,512]{3,2,1,0} %broadcast.1066), metadata={op_type="Mul" op_name="stage2_unit1_sc_bn/mul_1"}
  %arg161.162 = f32[512]{0} parameter(161), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.381 = f32[512]{0} reshape(f32[512]{0} %arg161.162)
  %arg118.119 = f32[512]{0} parameter(118), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.338 = f32[512]{0} reshape(f32[512]{0} %arg118.119)
  %multiply.1063 = f32[512]{0} multiply(f32[512]{0} %multiply.1062, f32[512]{0} %reshape.338), metadata={op_type="Mul" op_name="stage2_unit1_sc_bn/mul_2"}
  %subtract.1064 = f32[512]{0} subtract(f32[512]{0} %reshape.381, f32[512]{0} %multiply.1063), metadata={op_type="Sub" op_name="stage2_unit1_sc_bn/sub"}
  %broadcast.1068 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.1064), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit1_sc_bn/add_1"}
  %add.1069 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %multiply.1067, f32[1,28,28,512]{3,2,1,0} %broadcast.1068), metadata={op_type="AddV2" op_name="stage2_unit1_sc_bn/add_1"}
  %add.1070 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %add.1057, f32[1,28,28,512]{3,2,1,0} %add.1069), metadata={op_type="AddV2" op_name="add_3"}
  %maximum.1073 = f32[1,28,28,512]{3,2,1,0} maximum(f32[1,28,28,512]{3,2,1,0} %broadcast.1072, f32[1,28,28,512]{3,2,1,0} %add.1070), metadata={op_type="Relu" op_name="stage2_unit1_relu"}
  %constant.1088 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit2_bn3/add"}
  %broadcast.1089 = f32[512]{0} broadcast(f32[] %constant.1088), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit2_bn3/add"}
  %arg31.32 = f32[512]{0} parameter(31), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.251 = f32[512]{0} reshape(f32[512]{0} %arg31.32)
  %add.1090 = f32[512]{0} add(f32[512]{0} %broadcast.1089, f32[512]{0} %reshape.251), metadata={op_type="AddV2" op_name="stage2_unit2_bn3/add"}
  %rsqrt.1091 = f32[512]{0} rsqrt(f32[512]{0} %add.1090), metadata={op_type="Rsqrt" op_name="stage2_unit2_bn3/Rsqrt"}
  %arg81.82 = f32[512]{0} parameter(81), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.301 = f32[512]{0} reshape(f32[512]{0} %arg81.82)
  %multiply.1092 = f32[512]{0} multiply(f32[512]{0} %rsqrt.1091, f32[512]{0} %reshape.301), metadata={op_type="Mul" op_name="stage2_unit2_bn3/mul"}
  %broadcast.1210 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.1092), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit2_bn3/mul_1"}
  %constant.1206 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit2_relu2"}
  %broadcast.1207 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[] %constant.1206), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit2_relu2"}
  %constant.1100 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit2_relu1"}
  %broadcast.1101 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[] %constant.1100), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit2_relu1"}
  %constant.1074 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit2_bn1/add"}
  %broadcast.1075 = f32[256]{0} broadcast(f32[] %constant.1074), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit2_bn1/add"}
  %arg27.28 = f32[256]{0} parameter(27), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.247 = f32[256]{0} reshape(f32[256]{0} %arg27.28)
  %add.1076 = f32[256]{0} add(f32[256]{0} %broadcast.1075, f32[256]{0} %reshape.247), metadata={op_type="AddV2" op_name="stage2_unit2_bn1/add"}
  %rsqrt.1077 = f32[256]{0} rsqrt(f32[256]{0} %add.1076), metadata={op_type="Rsqrt" op_name="stage2_unit2_bn1/Rsqrt"}
  %arg78.79 = f32[256]{0} parameter(78), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.298 = f32[256]{0} reshape(f32[256]{0} %arg78.79)
  %multiply.1078 = f32[256]{0} multiply(f32[256]{0} %rsqrt.1077, f32[256]{0} %reshape.298), metadata={op_type="Mul" op_name="stage2_unit2_bn1/mul"}
  %broadcast.1096 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.1078), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit2_bn1/mul_1"}
  %arg199.200 = f32[1,1,512,256]{3,2,1,0} parameter(199), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.419 = f32[1,1,512,256]{3,2,1,0} reshape(f32[1,1,512,256]{3,2,1,0} %arg199.200)
  %convolution.1095 = f32[1,28,28,256]{3,2,1,0} convolution(f32[1,28,28,512]{3,2,1,0} %maximum.1073, f32[1,1,512,256]{3,2,1,0} %reshape.419), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv1"}
  %multiply.1097 = f32[1,28,28,256]{3,2,1,0} multiply(f32[1,28,28,256]{3,2,1,0} %broadcast.1096, f32[1,28,28,256]{3,2,1,0} %convolution.1095), metadata={op_type="Mul" op_name="stage2_unit2_bn1/mul_1"}
  %arg165.166 = f32[256]{0} parameter(165), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.385 = f32[256]{0} reshape(f32[256]{0} %arg165.166)
  %arg122.123 = f32[256]{0} parameter(122), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.342 = f32[256]{0} reshape(f32[256]{0} %arg122.123)
  %multiply.1079 = f32[256]{0} multiply(f32[256]{0} %multiply.1078, f32[256]{0} %reshape.342), metadata={op_type="Mul" op_name="stage2_unit2_bn1/mul_2"}
  %subtract.1080 = f32[256]{0} subtract(f32[256]{0} %reshape.385, f32[256]{0} %multiply.1079), metadata={op_type="Sub" op_name="stage2_unit2_bn1/sub"}
  %broadcast.1098 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.1080), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit2_bn1/add_1"}
  %add.1099 = f32[1,28,28,256]{3,2,1,0} add(f32[1,28,28,256]{3,2,1,0} %multiply.1097, f32[1,28,28,256]{3,2,1,0} %broadcast.1098), metadata={op_type="AddV2" op_name="stage2_unit2_bn1/add_1"}
  %maximum.1102 = f32[1,28,28,256]{3,2,1,0} maximum(f32[1,28,28,256]{3,2,1,0} %broadcast.1101, f32[1,28,28,256]{3,2,1,0} %add.1099), metadata={op_type="Relu" op_name="stage2_unit2_relu1"}
  %constant.1103 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_5"}
  %pad.1104 = f32[1,30,30,256]{3,2,1,0} pad(f32[1,28,28,256]{3,2,1,0} %maximum.1102, f32[] %constant.1103), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_5"}
  %slice.1105 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1104), slice={[0:1], [0:30], [0:30], [0:8]}, metadata={op_type="Split" op_name="split_9"}
  %arg28.29 = f32[3,3,8,256]{3,2,1,0} parameter(28), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.248 = f32[3,3,8,256]{3,2,1,0} reshape(f32[3,3,8,256]{3,2,1,0} %arg28.29)
  %slice.1137 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.248), slice={[0:3], [0:3], [0:8], [0:8]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1169 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1105, f32[3,3,8,8]{3,2,1,0} %slice.1137), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2"}
  %slice.1106 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1104), slice={[0:1], [0:30], [0:30], [8:16]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1138 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.248), slice={[0:3], [0:3], [0:8], [8:16]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1170 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1106, f32[3,3,8,8]{3,2,1,0} %slice.1138), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_1"}
  %slice.1107 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1104), slice={[0:1], [0:30], [0:30], [16:24]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1139 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.248), slice={[0:3], [0:3], [0:8], [16:24]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1181 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1107, f32[3,3,8,8]{3,2,1,0} %slice.1139), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_2"}
  %slice.1108 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1104), slice={[0:1], [0:30], [0:30], [24:32]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1140 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.248), slice={[0:3], [0:3], [0:8], [24:32]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1192 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1108, f32[3,3,8,8]{3,2,1,0} %slice.1140), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_3"}
  %slice.1109 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1104), slice={[0:1], [0:30], [0:30], [32:40]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1141 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.248), slice={[0:3], [0:3], [0:8], [32:40]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1195 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1109, f32[3,3,8,8]{3,2,1,0} %slice.1141), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_4"}
  %slice.1110 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1104), slice={[0:1], [0:30], [0:30], [40:48]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1142 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.248), slice={[0:3], [0:3], [0:8], [40:48]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1196 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1110, f32[3,3,8,8]{3,2,1,0} %slice.1142), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_5"}
  %slice.1111 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1104), slice={[0:1], [0:30], [0:30], [48:56]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1143 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.248), slice={[0:3], [0:3], [0:8], [48:56]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1197 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1111, f32[3,3,8,8]{3,2,1,0} %slice.1143), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_6"}
  %slice.1112 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1104), slice={[0:1], [0:30], [0:30], [56:64]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1144 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.248), slice={[0:3], [0:3], [0:8], [56:64]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1198 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1112, f32[3,3,8,8]{3,2,1,0} %slice.1144), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_7"}
  %slice.1113 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1104), slice={[0:1], [0:30], [0:30], [64:72]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1145 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.248), slice={[0:3], [0:3], [0:8], [64:72]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1199 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1113, f32[3,3,8,8]{3,2,1,0} %slice.1145), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_8"}
  %slice.1114 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1104), slice={[0:1], [0:30], [0:30], [72:80]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1146 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.248), slice={[0:3], [0:3], [0:8], [72:80]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1200 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1114, f32[3,3,8,8]{3,2,1,0} %slice.1146), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_9"}
  %slice.1115 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1104), slice={[0:1], [0:30], [0:30], [80:88]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1147 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.248), slice={[0:3], [0:3], [0:8], [80:88]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1171 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1115, f32[3,3,8,8]{3,2,1,0} %slice.1147), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_10"}
  %slice.1116 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1104), slice={[0:1], [0:30], [0:30], [88:96]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1148 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.248), slice={[0:3], [0:3], [0:8], [88:96]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1172 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1116, f32[3,3,8,8]{3,2,1,0} %slice.1148), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_11"}
  %slice.1117 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1104), slice={[0:1], [0:30], [0:30], [96:104]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1149 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.248), slice={[0:3], [0:3], [0:8], [96:104]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1173 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1117, f32[3,3,8,8]{3,2,1,0} %slice.1149), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_12"}
  %slice.1118 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1104), slice={[0:1], [0:30], [0:30], [104:112]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1150 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.248), slice={[0:3], [0:3], [0:8], [104:112]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1174 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1118, f32[3,3,8,8]{3,2,1,0} %slice.1150), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_13"}
  %slice.1119 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1104), slice={[0:1], [0:30], [0:30], [112:120]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1151 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.248), slice={[0:3], [0:3], [0:8], [112:120]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1175 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1119, f32[3,3,8,8]{3,2,1,0} %slice.1151), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_14"}
  %slice.1120 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1104), slice={[0:1], [0:30], [0:30], [120:128]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1152 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.248), slice={[0:3], [0:3], [0:8], [120:128]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1176 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1120, f32[3,3,8,8]{3,2,1,0} %slice.1152), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_15"}
  %slice.1121 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1104), slice={[0:1], [0:30], [0:30], [128:136]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1153 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.248), slice={[0:3], [0:3], [0:8], [128:136]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1177 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1121, f32[3,3,8,8]{3,2,1,0} %slice.1153), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_16"}
  %slice.1122 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1104), slice={[0:1], [0:30], [0:30], [136:144]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1154 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.248), slice={[0:3], [0:3], [0:8], [136:144]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1178 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1122, f32[3,3,8,8]{3,2,1,0} %slice.1154), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_17"}
  %slice.1123 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1104), slice={[0:1], [0:30], [0:30], [144:152]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1155 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.248), slice={[0:3], [0:3], [0:8], [144:152]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1179 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1123, f32[3,3,8,8]{3,2,1,0} %slice.1155), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_18"}
  %slice.1124 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1104), slice={[0:1], [0:30], [0:30], [152:160]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1156 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.248), slice={[0:3], [0:3], [0:8], [152:160]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1180 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1124, f32[3,3,8,8]{3,2,1,0} %slice.1156), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_19"}
  %slice.1125 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1104), slice={[0:1], [0:30], [0:30], [160:168]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1157 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.248), slice={[0:3], [0:3], [0:8], [160:168]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1182 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1125, f32[3,3,8,8]{3,2,1,0} %slice.1157), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_20"}
  %slice.1126 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1104), slice={[0:1], [0:30], [0:30], [168:176]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1158 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.248), slice={[0:3], [0:3], [0:8], [168:176]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1183 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1126, f32[3,3,8,8]{3,2,1,0} %slice.1158), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_21"}
  %slice.1127 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1104), slice={[0:1], [0:30], [0:30], [176:184]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1159 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.248), slice={[0:3], [0:3], [0:8], [176:184]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1184 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1127, f32[3,3,8,8]{3,2,1,0} %slice.1159), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_22"}
  %slice.1128 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1104), slice={[0:1], [0:30], [0:30], [184:192]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1160 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.248), slice={[0:3], [0:3], [0:8], [184:192]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1185 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1128, f32[3,3,8,8]{3,2,1,0} %slice.1160), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_23"}
  %slice.1129 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1104), slice={[0:1], [0:30], [0:30], [192:200]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1161 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.248), slice={[0:3], [0:3], [0:8], [192:200]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1186 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1129, f32[3,3,8,8]{3,2,1,0} %slice.1161), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_24"}
  %slice.1130 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1104), slice={[0:1], [0:30], [0:30], [200:208]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1162 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.248), slice={[0:3], [0:3], [0:8], [200:208]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1187 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1130, f32[3,3,8,8]{3,2,1,0} %slice.1162), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_25"}
  %slice.1131 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1104), slice={[0:1], [0:30], [0:30], [208:216]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1163 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.248), slice={[0:3], [0:3], [0:8], [208:216]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1188 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1131, f32[3,3,8,8]{3,2,1,0} %slice.1163), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_26"}
  %slice.1132 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1104), slice={[0:1], [0:30], [0:30], [216:224]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1164 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.248), slice={[0:3], [0:3], [0:8], [216:224]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1189 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1132, f32[3,3,8,8]{3,2,1,0} %slice.1164), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_27"}
  %slice.1133 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1104), slice={[0:1], [0:30], [0:30], [224:232]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1165 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.248), slice={[0:3], [0:3], [0:8], [224:232]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1190 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1133, f32[3,3,8,8]{3,2,1,0} %slice.1165), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_28"}
  %slice.1134 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1104), slice={[0:1], [0:30], [0:30], [232:240]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1166 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.248), slice={[0:3], [0:3], [0:8], [232:240]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1191 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1134, f32[3,3,8,8]{3,2,1,0} %slice.1166), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_29"}
  %slice.1135 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1104), slice={[0:1], [0:30], [0:30], [240:248]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1167 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.248), slice={[0:3], [0:3], [0:8], [240:248]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1193 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1135, f32[3,3,8,8]{3,2,1,0} %slice.1167), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_30"}
  %slice.1136 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1104), slice={[0:1], [0:30], [0:30], [248:256]}, metadata={op_type="Split" op_name="split_9"}
  %slice.1168 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.248), slice={[0:3], [0:3], [0:8], [248:256]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1194 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1136, f32[3,3,8,8]{3,2,1,0} %slice.1168), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_31"}
  %concatenate.1201 = f32[1,28,28,256]{3,2,1,0} concatenate(f32[1,28,28,8]{3,2,1,0} %convolution.1169, f32[1,28,28,8]{3,2,1,0} %convolution.1170, f32[1,28,28,8]{3,2,1,0} %convolution.1181, f32[1,28,28,8]{3,2,1,0} %convolution.1192, f32[1,28,28,8]{3,2,1,0} %convolution.1195, f32[1,28,28,8]{3,2,1,0} %convolution.1196, f32[1,28,28,8]{3,2,1,0} %convolution.1197, f32[1,28,28,8]{3,2,1,0} %convolution.1198, f32[1,28,28,8]{3,2,1,0} %convolution.1199, f32[1,28,28,8]{3,2,1,0} %convolution.1200, f32[1,28,28,8]{3,2,1,0} %convolution.1171, f32[1,28,28,8]{3,2,1,0} %convolution.1172, f32[1,28,28,8]{3,2,1,0} %convolution.1173, f32[1,28,28,8]{3,2,1,0} %convolution.1174, f32[1,28,28,8]{3,2,1,0} %convolution.1175, f32[1,28,28,8]{3,2,1,0} %convolution.1176, f32[1,28,28,8]{3,2,1,0} %convolution.1177, f32[1,28,28,8]{3,2,1,0} %convolution.1178, f32[1,28,28,8]{3,2,1,0} %convolution.1179, f32[1,28,28,8]{3,2,1,0} %convolution.1180, f32[1,28,28,8]{3,2,1,0} %convolution.1182, f32[1,28,28,8]{3,2,1,0} %convolution.1183, f32[1,28,28,8]{3,2,1,0} %convolution.1184, f32[1,28,28,8]{3,2,1,0} %convolution.1185, f32[1,28,28,8]{3,2,1,0} %convolution.1186, f32[1,28,28,8]{3,2,1,0} %convolution.1187, f32[1,28,28,8]{3,2,1,0} %convolution.1188, f32[1,28,28,8]{3,2,1,0} %convolution.1189, f32[1,28,28,8]{3,2,1,0} %convolution.1190, f32[1,28,28,8]{3,2,1,0} %convolution.1191, f32[1,28,28,8]{3,2,1,0} %convolution.1193, f32[1,28,28,8]{3,2,1,0} %convolution.1194), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_4"}
  %constant.1081 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit2_bn2/add"}
  %broadcast.1082 = f32[256]{0} broadcast(f32[] %constant.1081), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit2_bn2/add"}
  %arg30.31 = f32[256]{0} parameter(30), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.250 = f32[256]{0} reshape(f32[256]{0} %arg30.31)
  %add.1083 = f32[256]{0} add(f32[256]{0} %broadcast.1082, f32[256]{0} %reshape.250), metadata={op_type="AddV2" op_name="stage2_unit2_bn2/add"}
  %rsqrt.1084 = f32[256]{0} rsqrt(f32[256]{0} %add.1083), metadata={op_type="Rsqrt" op_name="stage2_unit2_bn2/Rsqrt"}
  %arg80.81 = f32[256]{0} parameter(80), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.300 = f32[256]{0} reshape(f32[256]{0} %arg80.81)
  %multiply.1085 = f32[256]{0} multiply(f32[256]{0} %rsqrt.1084, f32[256]{0} %reshape.300), metadata={op_type="Mul" op_name="stage2_unit2_bn2/mul"}
  %broadcast.1202 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.1085), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit2_bn2/mul_1"}
  %multiply.1203 = f32[1,28,28,256]{3,2,1,0} multiply(f32[1,28,28,256]{3,2,1,0} %concatenate.1201, f32[1,28,28,256]{3,2,1,0} %broadcast.1202), metadata={op_type="Mul" op_name="stage2_unit2_bn2/mul_1"}
  %arg167.168 = f32[256]{0} parameter(167), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.387 = f32[256]{0} reshape(f32[256]{0} %arg167.168)
  %arg124.125 = f32[256]{0} parameter(124), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.344 = f32[256]{0} reshape(f32[256]{0} %arg124.125)
  %multiply.1086 = f32[256]{0} multiply(f32[256]{0} %multiply.1085, f32[256]{0} %reshape.344), metadata={op_type="Mul" op_name="stage2_unit2_bn2/mul_2"}
  %subtract.1087 = f32[256]{0} subtract(f32[256]{0} %reshape.387, f32[256]{0} %multiply.1086), metadata={op_type="Sub" op_name="stage2_unit2_bn2/sub"}
  %broadcast.1204 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.1087), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit2_bn2/add_1"}
  %add.1205 = f32[1,28,28,256]{3,2,1,0} add(f32[1,28,28,256]{3,2,1,0} %multiply.1203, f32[1,28,28,256]{3,2,1,0} %broadcast.1204), metadata={op_type="AddV2" op_name="stage2_unit2_bn2/add_1"}
  %maximum.1208 = f32[1,28,28,256]{3,2,1,0} maximum(f32[1,28,28,256]{3,2,1,0} %broadcast.1207, f32[1,28,28,256]{3,2,1,0} %add.1205), metadata={op_type="Relu" op_name="stage2_unit2_relu2"}
  %arg200.201 = f32[1,1,256,512]{3,2,1,0} parameter(200), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.420 = f32[1,1,256,512]{3,2,1,0} reshape(f32[1,1,256,512]{3,2,1,0} %arg200.201)
  %convolution.1209 = f32[1,28,28,512]{3,2,1,0} convolution(f32[1,28,28,256]{3,2,1,0} %maximum.1208, f32[1,1,256,512]{3,2,1,0} %reshape.420), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv3"}
  %multiply.1211 = f32[1,28,28,512]{3,2,1,0} multiply(f32[1,28,28,512]{3,2,1,0} %broadcast.1210, f32[1,28,28,512]{3,2,1,0} %convolution.1209), metadata={op_type="Mul" op_name="stage2_unit2_bn3/mul_1"}
  %arg168.169 = f32[512]{0} parameter(168), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.388 = f32[512]{0} reshape(f32[512]{0} %arg168.169)
  %arg125.126 = f32[512]{0} parameter(125), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.345 = f32[512]{0} reshape(f32[512]{0} %arg125.126)
  %multiply.1093 = f32[512]{0} multiply(f32[512]{0} %multiply.1092, f32[512]{0} %reshape.345), metadata={op_type="Mul" op_name="stage2_unit2_bn3/mul_2"}
  %subtract.1094 = f32[512]{0} subtract(f32[512]{0} %reshape.388, f32[512]{0} %multiply.1093), metadata={op_type="Sub" op_name="stage2_unit2_bn3/sub"}
  %broadcast.1212 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.1094), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit2_bn3/add_1"}
  %add.1213 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %multiply.1211, f32[1,28,28,512]{3,2,1,0} %broadcast.1212), metadata={op_type="AddV2" op_name="stage2_unit2_bn3/add_1"}
  %add.1214 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %maximum.1073, f32[1,28,28,512]{3,2,1,0} %add.1213), metadata={op_type="AddV2" op_name="add_4"}
  %maximum.1217 = f32[1,28,28,512]{3,2,1,0} maximum(f32[1,28,28,512]{3,2,1,0} %broadcast.1216, f32[1,28,28,512]{3,2,1,0} %add.1214), metadata={op_type="Relu" op_name="stage2_unit2_relu"}
  %constant.1232 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit3_bn3/add"}
  %broadcast.1233 = f32[512]{0} broadcast(f32[] %constant.1232), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit3_bn3/add"}
  %arg36.37 = f32[512]{0} parameter(36), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.256 = f32[512]{0} reshape(f32[512]{0} %arg36.37)
  %add.1234 = f32[512]{0} add(f32[512]{0} %broadcast.1233, f32[512]{0} %reshape.256), metadata={op_type="AddV2" op_name="stage2_unit3_bn3/add"}
  %rsqrt.1235 = f32[512]{0} rsqrt(f32[512]{0} %add.1234), metadata={op_type="Rsqrt" op_name="stage2_unit3_bn3/Rsqrt"}
  %arg85.86 = f32[512]{0} parameter(85), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.305 = f32[512]{0} reshape(f32[512]{0} %arg85.86)
  %multiply.1236 = f32[512]{0} multiply(f32[512]{0} %rsqrt.1235, f32[512]{0} %reshape.305), metadata={op_type="Mul" op_name="stage2_unit3_bn3/mul"}
  %broadcast.1354 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.1236), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit3_bn3/mul_1"}
  %constant.1350 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit3_relu2"}
  %broadcast.1351 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[] %constant.1350), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit3_relu2"}
  %constant.1244 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit3_relu1"}
  %broadcast.1245 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[] %constant.1244), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit3_relu1"}
  %constant.1218 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit3_bn1/add"}
  %broadcast.1219 = f32[256]{0} broadcast(f32[] %constant.1218), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit3_bn1/add"}
  %arg32.33 = f32[256]{0} parameter(32), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.252 = f32[256]{0} reshape(f32[256]{0} %arg32.33)
  %add.1220 = f32[256]{0} add(f32[256]{0} %broadcast.1219, f32[256]{0} %reshape.252), metadata={op_type="AddV2" op_name="stage2_unit3_bn1/add"}
  %rsqrt.1221 = f32[256]{0} rsqrt(f32[256]{0} %add.1220), metadata={op_type="Rsqrt" op_name="stage2_unit3_bn1/Rsqrt"}
  %arg82.83 = f32[256]{0} parameter(82), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.302 = f32[256]{0} reshape(f32[256]{0} %arg82.83)
  %multiply.1222 = f32[256]{0} multiply(f32[256]{0} %rsqrt.1221, f32[256]{0} %reshape.302), metadata={op_type="Mul" op_name="stage2_unit3_bn1/mul"}
  %broadcast.1240 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.1222), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit3_bn1/mul_1"}
  %arg201.202 = f32[1,1,512,256]{3,2,1,0} parameter(201), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.421 = f32[1,1,512,256]{3,2,1,0} reshape(f32[1,1,512,256]{3,2,1,0} %arg201.202)
  %convolution.1239 = f32[1,28,28,256]{3,2,1,0} convolution(f32[1,28,28,512]{3,2,1,0} %maximum.1217, f32[1,1,512,256]{3,2,1,0} %reshape.421), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv1"}
  %multiply.1241 = f32[1,28,28,256]{3,2,1,0} multiply(f32[1,28,28,256]{3,2,1,0} %broadcast.1240, f32[1,28,28,256]{3,2,1,0} %convolution.1239), metadata={op_type="Mul" op_name="stage2_unit3_bn1/mul_1"}
  %arg169.170 = f32[256]{0} parameter(169), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.389 = f32[256]{0} reshape(f32[256]{0} %arg169.170)
  %arg126.127 = f32[256]{0} parameter(126), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.346 = f32[256]{0} reshape(f32[256]{0} %arg126.127)
  %multiply.1223 = f32[256]{0} multiply(f32[256]{0} %multiply.1222, f32[256]{0} %reshape.346), metadata={op_type="Mul" op_name="stage2_unit3_bn1/mul_2"}
  %subtract.1224 = f32[256]{0} subtract(f32[256]{0} %reshape.389, f32[256]{0} %multiply.1223), metadata={op_type="Sub" op_name="stage2_unit3_bn1/sub"}
  %broadcast.1242 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.1224), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit3_bn1/add_1"}
  %add.1243 = f32[1,28,28,256]{3,2,1,0} add(f32[1,28,28,256]{3,2,1,0} %multiply.1241, f32[1,28,28,256]{3,2,1,0} %broadcast.1242), metadata={op_type="AddV2" op_name="stage2_unit3_bn1/add_1"}
  %maximum.1246 = f32[1,28,28,256]{3,2,1,0} maximum(f32[1,28,28,256]{3,2,1,0} %broadcast.1245, f32[1,28,28,256]{3,2,1,0} %add.1243), metadata={op_type="Relu" op_name="stage2_unit3_relu1"}
  %constant.1247 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_6"}
  %pad.1248 = f32[1,30,30,256]{3,2,1,0} pad(f32[1,28,28,256]{3,2,1,0} %maximum.1246, f32[] %constant.1247), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_6"}
  %slice.1249 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1248), slice={[0:1], [0:30], [0:30], [0:8]}, metadata={op_type="Split" op_name="split_11"}
  %arg33.34 = f32[3,3,8,256]{3,2,1,0} parameter(33), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.253 = f32[3,3,8,256]{3,2,1,0} reshape(f32[3,3,8,256]{3,2,1,0} %arg33.34)
  %slice.1281 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.253), slice={[0:3], [0:3], [0:8], [0:8]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1313 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1249, f32[3,3,8,8]{3,2,1,0} %slice.1281), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2"}
  %slice.1250 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1248), slice={[0:1], [0:30], [0:30], [8:16]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1282 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.253), slice={[0:3], [0:3], [0:8], [8:16]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1314 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1250, f32[3,3,8,8]{3,2,1,0} %slice.1282), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_1"}
  %slice.1251 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1248), slice={[0:1], [0:30], [0:30], [16:24]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1283 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.253), slice={[0:3], [0:3], [0:8], [16:24]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1325 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1251, f32[3,3,8,8]{3,2,1,0} %slice.1283), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_2"}
  %slice.1252 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1248), slice={[0:1], [0:30], [0:30], [24:32]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1284 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.253), slice={[0:3], [0:3], [0:8], [24:32]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1336 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1252, f32[3,3,8,8]{3,2,1,0} %slice.1284), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_3"}
  %slice.1253 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1248), slice={[0:1], [0:30], [0:30], [32:40]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1285 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.253), slice={[0:3], [0:3], [0:8], [32:40]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1339 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1253, f32[3,3,8,8]{3,2,1,0} %slice.1285), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_4"}
  %slice.1254 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1248), slice={[0:1], [0:30], [0:30], [40:48]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1286 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.253), slice={[0:3], [0:3], [0:8], [40:48]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1340 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1254, f32[3,3,8,8]{3,2,1,0} %slice.1286), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_5"}
  %slice.1255 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1248), slice={[0:1], [0:30], [0:30], [48:56]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1287 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.253), slice={[0:3], [0:3], [0:8], [48:56]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1341 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1255, f32[3,3,8,8]{3,2,1,0} %slice.1287), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_6"}
  %slice.1256 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1248), slice={[0:1], [0:30], [0:30], [56:64]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1288 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.253), slice={[0:3], [0:3], [0:8], [56:64]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1342 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1256, f32[3,3,8,8]{3,2,1,0} %slice.1288), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_7"}
  %slice.1257 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1248), slice={[0:1], [0:30], [0:30], [64:72]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1289 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.253), slice={[0:3], [0:3], [0:8], [64:72]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1343 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1257, f32[3,3,8,8]{3,2,1,0} %slice.1289), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_8"}
  %slice.1258 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1248), slice={[0:1], [0:30], [0:30], [72:80]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1290 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.253), slice={[0:3], [0:3], [0:8], [72:80]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1344 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1258, f32[3,3,8,8]{3,2,1,0} %slice.1290), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_9"}
  %slice.1259 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1248), slice={[0:1], [0:30], [0:30], [80:88]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1291 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.253), slice={[0:3], [0:3], [0:8], [80:88]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1315 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1259, f32[3,3,8,8]{3,2,1,0} %slice.1291), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_10"}
  %slice.1260 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1248), slice={[0:1], [0:30], [0:30], [88:96]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1292 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.253), slice={[0:3], [0:3], [0:8], [88:96]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1316 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1260, f32[3,3,8,8]{3,2,1,0} %slice.1292), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_11"}
  %slice.1261 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1248), slice={[0:1], [0:30], [0:30], [96:104]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1293 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.253), slice={[0:3], [0:3], [0:8], [96:104]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1317 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1261, f32[3,3,8,8]{3,2,1,0} %slice.1293), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_12"}
  %slice.1262 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1248), slice={[0:1], [0:30], [0:30], [104:112]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1294 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.253), slice={[0:3], [0:3], [0:8], [104:112]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1318 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1262, f32[3,3,8,8]{3,2,1,0} %slice.1294), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_13"}
  %slice.1263 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1248), slice={[0:1], [0:30], [0:30], [112:120]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1295 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.253), slice={[0:3], [0:3], [0:8], [112:120]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1319 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1263, f32[3,3,8,8]{3,2,1,0} %slice.1295), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_14"}
  %slice.1264 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1248), slice={[0:1], [0:30], [0:30], [120:128]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1296 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.253), slice={[0:3], [0:3], [0:8], [120:128]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1320 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1264, f32[3,3,8,8]{3,2,1,0} %slice.1296), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_15"}
  %slice.1265 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1248), slice={[0:1], [0:30], [0:30], [128:136]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1297 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.253), slice={[0:3], [0:3], [0:8], [128:136]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1321 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1265, f32[3,3,8,8]{3,2,1,0} %slice.1297), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_16"}
  %slice.1266 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1248), slice={[0:1], [0:30], [0:30], [136:144]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1298 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.253), slice={[0:3], [0:3], [0:8], [136:144]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1322 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1266, f32[3,3,8,8]{3,2,1,0} %slice.1298), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_17"}
  %slice.1267 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1248), slice={[0:1], [0:30], [0:30], [144:152]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1299 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.253), slice={[0:3], [0:3], [0:8], [144:152]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1323 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1267, f32[3,3,8,8]{3,2,1,0} %slice.1299), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_18"}
  %slice.1268 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1248), slice={[0:1], [0:30], [0:30], [152:160]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1300 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.253), slice={[0:3], [0:3], [0:8], [152:160]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1324 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1268, f32[3,3,8,8]{3,2,1,0} %slice.1300), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_19"}
  %slice.1269 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1248), slice={[0:1], [0:30], [0:30], [160:168]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1301 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.253), slice={[0:3], [0:3], [0:8], [160:168]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1326 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1269, f32[3,3,8,8]{3,2,1,0} %slice.1301), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_20"}
  %slice.1270 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1248), slice={[0:1], [0:30], [0:30], [168:176]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1302 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.253), slice={[0:3], [0:3], [0:8], [168:176]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1327 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1270, f32[3,3,8,8]{3,2,1,0} %slice.1302), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_21"}
  %slice.1271 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1248), slice={[0:1], [0:30], [0:30], [176:184]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1303 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.253), slice={[0:3], [0:3], [0:8], [176:184]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1328 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1271, f32[3,3,8,8]{3,2,1,0} %slice.1303), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_22"}
  %slice.1272 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1248), slice={[0:1], [0:30], [0:30], [184:192]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1304 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.253), slice={[0:3], [0:3], [0:8], [184:192]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1329 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1272, f32[3,3,8,8]{3,2,1,0} %slice.1304), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_23"}
  %slice.1273 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1248), slice={[0:1], [0:30], [0:30], [192:200]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1305 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.253), slice={[0:3], [0:3], [0:8], [192:200]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1330 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1273, f32[3,3,8,8]{3,2,1,0} %slice.1305), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_24"}
  %slice.1274 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1248), slice={[0:1], [0:30], [0:30], [200:208]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1306 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.253), slice={[0:3], [0:3], [0:8], [200:208]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1331 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1274, f32[3,3,8,8]{3,2,1,0} %slice.1306), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_25"}
  %slice.1275 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1248), slice={[0:1], [0:30], [0:30], [208:216]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1307 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.253), slice={[0:3], [0:3], [0:8], [208:216]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1332 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1275, f32[3,3,8,8]{3,2,1,0} %slice.1307), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_26"}
  %slice.1276 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1248), slice={[0:1], [0:30], [0:30], [216:224]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1308 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.253), slice={[0:3], [0:3], [0:8], [216:224]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1333 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1276, f32[3,3,8,8]{3,2,1,0} %slice.1308), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_27"}
  %slice.1277 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1248), slice={[0:1], [0:30], [0:30], [224:232]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1309 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.253), slice={[0:3], [0:3], [0:8], [224:232]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1334 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1277, f32[3,3,8,8]{3,2,1,0} %slice.1309), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_28"}
  %slice.1278 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1248), slice={[0:1], [0:30], [0:30], [232:240]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1310 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.253), slice={[0:3], [0:3], [0:8], [232:240]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1335 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1278, f32[3,3,8,8]{3,2,1,0} %slice.1310), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_29"}
  %slice.1279 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1248), slice={[0:1], [0:30], [0:30], [240:248]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1311 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.253), slice={[0:3], [0:3], [0:8], [240:248]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1337 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1279, f32[3,3,8,8]{3,2,1,0} %slice.1311), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_30"}
  %slice.1280 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1248), slice={[0:1], [0:30], [0:30], [248:256]}, metadata={op_type="Split" op_name="split_11"}
  %slice.1312 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.253), slice={[0:3], [0:3], [0:8], [248:256]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1338 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1280, f32[3,3,8,8]{3,2,1,0} %slice.1312), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_31"}
  %concatenate.1345 = f32[1,28,28,256]{3,2,1,0} concatenate(f32[1,28,28,8]{3,2,1,0} %convolution.1313, f32[1,28,28,8]{3,2,1,0} %convolution.1314, f32[1,28,28,8]{3,2,1,0} %convolution.1325, f32[1,28,28,8]{3,2,1,0} %convolution.1336, f32[1,28,28,8]{3,2,1,0} %convolution.1339, f32[1,28,28,8]{3,2,1,0} %convolution.1340, f32[1,28,28,8]{3,2,1,0} %convolution.1341, f32[1,28,28,8]{3,2,1,0} %convolution.1342, f32[1,28,28,8]{3,2,1,0} %convolution.1343, f32[1,28,28,8]{3,2,1,0} %convolution.1344, f32[1,28,28,8]{3,2,1,0} %convolution.1315, f32[1,28,28,8]{3,2,1,0} %convolution.1316, f32[1,28,28,8]{3,2,1,0} %convolution.1317, f32[1,28,28,8]{3,2,1,0} %convolution.1318, f32[1,28,28,8]{3,2,1,0} %convolution.1319, f32[1,28,28,8]{3,2,1,0} %convolution.1320, f32[1,28,28,8]{3,2,1,0} %convolution.1321, f32[1,28,28,8]{3,2,1,0} %convolution.1322, f32[1,28,28,8]{3,2,1,0} %convolution.1323, f32[1,28,28,8]{3,2,1,0} %convolution.1324, f32[1,28,28,8]{3,2,1,0} %convolution.1326, f32[1,28,28,8]{3,2,1,0} %convolution.1327, f32[1,28,28,8]{3,2,1,0} %convolution.1328, f32[1,28,28,8]{3,2,1,0} %convolution.1329, f32[1,28,28,8]{3,2,1,0} %convolution.1330, f32[1,28,28,8]{3,2,1,0} %convolution.1331, f32[1,28,28,8]{3,2,1,0} %convolution.1332, f32[1,28,28,8]{3,2,1,0} %convolution.1333, f32[1,28,28,8]{3,2,1,0} %convolution.1334, f32[1,28,28,8]{3,2,1,0} %convolution.1335, f32[1,28,28,8]{3,2,1,0} %convolution.1337, f32[1,28,28,8]{3,2,1,0} %convolution.1338), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_5"}
  %constant.1225 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit3_bn2/add"}
  %broadcast.1226 = f32[256]{0} broadcast(f32[] %constant.1225), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit3_bn2/add"}
  %arg35.36 = f32[256]{0} parameter(35), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.255 = f32[256]{0} reshape(f32[256]{0} %arg35.36)
  %add.1227 = f32[256]{0} add(f32[256]{0} %broadcast.1226, f32[256]{0} %reshape.255), metadata={op_type="AddV2" op_name="stage2_unit3_bn2/add"}
  %rsqrt.1228 = f32[256]{0} rsqrt(f32[256]{0} %add.1227), metadata={op_type="Rsqrt" op_name="stage2_unit3_bn2/Rsqrt"}
  %arg84.85 = f32[256]{0} parameter(84), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.304 = f32[256]{0} reshape(f32[256]{0} %arg84.85)
  %multiply.1229 = f32[256]{0} multiply(f32[256]{0} %rsqrt.1228, f32[256]{0} %reshape.304), metadata={op_type="Mul" op_name="stage2_unit3_bn2/mul"}
  %broadcast.1346 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.1229), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit3_bn2/mul_1"}
  %multiply.1347 = f32[1,28,28,256]{3,2,1,0} multiply(f32[1,28,28,256]{3,2,1,0} %concatenate.1345, f32[1,28,28,256]{3,2,1,0} %broadcast.1346), metadata={op_type="Mul" op_name="stage2_unit3_bn2/mul_1"}
  %arg171.172 = f32[256]{0} parameter(171), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.391 = f32[256]{0} reshape(f32[256]{0} %arg171.172)
  %arg128.129 = f32[256]{0} parameter(128), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.348 = f32[256]{0} reshape(f32[256]{0} %arg128.129)
  %multiply.1230 = f32[256]{0} multiply(f32[256]{0} %multiply.1229, f32[256]{0} %reshape.348), metadata={op_type="Mul" op_name="stage2_unit3_bn2/mul_2"}
  %subtract.1231 = f32[256]{0} subtract(f32[256]{0} %reshape.391, f32[256]{0} %multiply.1230), metadata={op_type="Sub" op_name="stage2_unit3_bn2/sub"}
  %broadcast.1348 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.1231), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit3_bn2/add_1"}
  %add.1349 = f32[1,28,28,256]{3,2,1,0} add(f32[1,28,28,256]{3,2,1,0} %multiply.1347, f32[1,28,28,256]{3,2,1,0} %broadcast.1348), metadata={op_type="AddV2" op_name="stage2_unit3_bn2/add_1"}
  %maximum.1352 = f32[1,28,28,256]{3,2,1,0} maximum(f32[1,28,28,256]{3,2,1,0} %broadcast.1351, f32[1,28,28,256]{3,2,1,0} %add.1349), metadata={op_type="Relu" op_name="stage2_unit3_relu2"}
  %arg202.203 = f32[1,1,256,512]{3,2,1,0} parameter(202), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.422 = f32[1,1,256,512]{3,2,1,0} reshape(f32[1,1,256,512]{3,2,1,0} %arg202.203)
  %convolution.1353 = f32[1,28,28,512]{3,2,1,0} convolution(f32[1,28,28,256]{3,2,1,0} %maximum.1352, f32[1,1,256,512]{3,2,1,0} %reshape.422), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv3"}
  %multiply.1355 = f32[1,28,28,512]{3,2,1,0} multiply(f32[1,28,28,512]{3,2,1,0} %broadcast.1354, f32[1,28,28,512]{3,2,1,0} %convolution.1353), metadata={op_type="Mul" op_name="stage2_unit3_bn3/mul_1"}
  %arg172.173 = f32[512]{0} parameter(172), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.392 = f32[512]{0} reshape(f32[512]{0} %arg172.173)
  %arg129.130 = f32[512]{0} parameter(129), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.349 = f32[512]{0} reshape(f32[512]{0} %arg129.130)
  %multiply.1237 = f32[512]{0} multiply(f32[512]{0} %multiply.1236, f32[512]{0} %reshape.349), metadata={op_type="Mul" op_name="stage2_unit3_bn3/mul_2"}
  %subtract.1238 = f32[512]{0} subtract(f32[512]{0} %reshape.392, f32[512]{0} %multiply.1237), metadata={op_type="Sub" op_name="stage2_unit3_bn3/sub"}
  %broadcast.1356 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.1238), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit3_bn3/add_1"}
  %add.1357 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %multiply.1355, f32[1,28,28,512]{3,2,1,0} %broadcast.1356), metadata={op_type="AddV2" op_name="stage2_unit3_bn3/add_1"}
  %add.1358 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %maximum.1217, f32[1,28,28,512]{3,2,1,0} %add.1357), metadata={op_type="AddV2" op_name="add_5"}
  %maximum.1361 = f32[1,28,28,512]{3,2,1,0} maximum(f32[1,28,28,512]{3,2,1,0} %broadcast.1360, f32[1,28,28,512]{3,2,1,0} %add.1358), metadata={op_type="Relu" op_name="stage2_unit3_relu"}
  %constant.1376 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit4_bn3/add"}
  %broadcast.1377 = f32[512]{0} broadcast(f32[] %constant.1376), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit4_bn3/add"}
  %arg42.43 = f32[512]{0} parameter(42), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.262 = f32[512]{0} reshape(f32[512]{0} %arg42.43)
  %add.1378 = f32[512]{0} add(f32[512]{0} %broadcast.1377, f32[512]{0} %reshape.262), metadata={op_type="AddV2" op_name="stage2_unit4_bn3/add"}
  %rsqrt.1379 = f32[512]{0} rsqrt(f32[512]{0} %add.1378), metadata={op_type="Rsqrt" op_name="stage2_unit4_bn3/Rsqrt"}
  %arg89.90 = f32[512]{0} parameter(89), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.309 = f32[512]{0} reshape(f32[512]{0} %arg89.90)
  %multiply.1380 = f32[512]{0} multiply(f32[512]{0} %rsqrt.1379, f32[512]{0} %reshape.309), metadata={op_type="Mul" op_name="stage2_unit4_bn3/mul"}
  %broadcast.1498 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.1380), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit4_bn3/mul_1"}
  %constant.1494 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit4_relu2"}
  %broadcast.1495 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[] %constant.1494), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit4_relu2"}
  %constant.1388 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit4_relu1"}
  %broadcast.1389 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[] %constant.1388), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit4_relu1"}
  %constant.1362 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit4_bn1/add"}
  %broadcast.1363 = f32[256]{0} broadcast(f32[] %constant.1362), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit4_bn1/add"}
  %arg38.39 = f32[256]{0} parameter(38), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.258 = f32[256]{0} reshape(f32[256]{0} %arg38.39)
  %add.1364 = f32[256]{0} add(f32[256]{0} %broadcast.1363, f32[256]{0} %reshape.258), metadata={op_type="AddV2" op_name="stage2_unit4_bn1/add"}
  %rsqrt.1365 = f32[256]{0} rsqrt(f32[256]{0} %add.1364), metadata={op_type="Rsqrt" op_name="stage2_unit4_bn1/Rsqrt"}
  %arg86.87 = f32[256]{0} parameter(86), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.306 = f32[256]{0} reshape(f32[256]{0} %arg86.87)
  %multiply.1366 = f32[256]{0} multiply(f32[256]{0} %rsqrt.1365, f32[256]{0} %reshape.306), metadata={op_type="Mul" op_name="stage2_unit4_bn1/mul"}
  %broadcast.1384 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.1366), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit4_bn1/mul_1"}
  %arg203.204 = f32[1,1,512,256]{3,2,1,0} parameter(203), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.423 = f32[1,1,512,256]{3,2,1,0} reshape(f32[1,1,512,256]{3,2,1,0} %arg203.204)
  %convolution.1383 = f32[1,28,28,256]{3,2,1,0} convolution(f32[1,28,28,512]{3,2,1,0} %maximum.1361, f32[1,1,512,256]{3,2,1,0} %reshape.423), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv1"}
  %multiply.1385 = f32[1,28,28,256]{3,2,1,0} multiply(f32[1,28,28,256]{3,2,1,0} %broadcast.1384, f32[1,28,28,256]{3,2,1,0} %convolution.1383), metadata={op_type="Mul" op_name="stage2_unit4_bn1/mul_1"}
  %arg173.174 = f32[256]{0} parameter(173), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.393 = f32[256]{0} reshape(f32[256]{0} %arg173.174)
  %arg130.131 = f32[256]{0} parameter(130), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.350 = f32[256]{0} reshape(f32[256]{0} %arg130.131)
  %multiply.1367 = f32[256]{0} multiply(f32[256]{0} %multiply.1366, f32[256]{0} %reshape.350), metadata={op_type="Mul" op_name="stage2_unit4_bn1/mul_2"}
  %subtract.1368 = f32[256]{0} subtract(f32[256]{0} %reshape.393, f32[256]{0} %multiply.1367), metadata={op_type="Sub" op_name="stage2_unit4_bn1/sub"}
  %broadcast.1386 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.1368), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit4_bn1/add_1"}
  %add.1387 = f32[1,28,28,256]{3,2,1,0} add(f32[1,28,28,256]{3,2,1,0} %multiply.1385, f32[1,28,28,256]{3,2,1,0} %broadcast.1386), metadata={op_type="AddV2" op_name="stage2_unit4_bn1/add_1"}
  %maximum.1390 = f32[1,28,28,256]{3,2,1,0} maximum(f32[1,28,28,256]{3,2,1,0} %broadcast.1389, f32[1,28,28,256]{3,2,1,0} %add.1387), metadata={op_type="Relu" op_name="stage2_unit4_relu1"}
  %constant.1391 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_7"}
  %pad.1392 = f32[1,30,30,256]{3,2,1,0} pad(f32[1,28,28,256]{3,2,1,0} %maximum.1390, f32[] %constant.1391), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_7"}
  %slice.1393 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1392), slice={[0:1], [0:30], [0:30], [0:8]}, metadata={op_type="Split" op_name="split_13"}
  %arg40.41 = f32[3,3,8,256]{3,2,1,0} parameter(40), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.260 = f32[3,3,8,256]{3,2,1,0} reshape(f32[3,3,8,256]{3,2,1,0} %arg40.41)
  %slice.1425 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.260), slice={[0:3], [0:3], [0:8], [0:8]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1457 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1393, f32[3,3,8,8]{3,2,1,0} %slice.1425), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2"}
  %slice.1394 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1392), slice={[0:1], [0:30], [0:30], [8:16]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1426 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.260), slice={[0:3], [0:3], [0:8], [8:16]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1458 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1394, f32[3,3,8,8]{3,2,1,0} %slice.1426), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_1"}
  %slice.1395 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1392), slice={[0:1], [0:30], [0:30], [16:24]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1427 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.260), slice={[0:3], [0:3], [0:8], [16:24]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1469 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1395, f32[3,3,8,8]{3,2,1,0} %slice.1427), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_2"}
  %slice.1396 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1392), slice={[0:1], [0:30], [0:30], [24:32]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1428 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.260), slice={[0:3], [0:3], [0:8], [24:32]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1480 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1396, f32[3,3,8,8]{3,2,1,0} %slice.1428), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_3"}
  %slice.1397 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1392), slice={[0:1], [0:30], [0:30], [32:40]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1429 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.260), slice={[0:3], [0:3], [0:8], [32:40]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1483 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1397, f32[3,3,8,8]{3,2,1,0} %slice.1429), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_4"}
  %slice.1398 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1392), slice={[0:1], [0:30], [0:30], [40:48]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1430 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.260), slice={[0:3], [0:3], [0:8], [40:48]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1484 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1398, f32[3,3,8,8]{3,2,1,0} %slice.1430), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_5"}
  %slice.1399 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1392), slice={[0:1], [0:30], [0:30], [48:56]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1431 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.260), slice={[0:3], [0:3], [0:8], [48:56]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1485 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1399, f32[3,3,8,8]{3,2,1,0} %slice.1431), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_6"}
  %slice.1400 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1392), slice={[0:1], [0:30], [0:30], [56:64]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1432 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.260), slice={[0:3], [0:3], [0:8], [56:64]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1486 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1400, f32[3,3,8,8]{3,2,1,0} %slice.1432), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_7"}
  %slice.1401 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1392), slice={[0:1], [0:30], [0:30], [64:72]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1433 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.260), slice={[0:3], [0:3], [0:8], [64:72]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1487 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1401, f32[3,3,8,8]{3,2,1,0} %slice.1433), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_8"}
  %slice.1402 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1392), slice={[0:1], [0:30], [0:30], [72:80]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1434 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.260), slice={[0:3], [0:3], [0:8], [72:80]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1488 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1402, f32[3,3,8,8]{3,2,1,0} %slice.1434), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_9"}
  %slice.1403 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1392), slice={[0:1], [0:30], [0:30], [80:88]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1435 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.260), slice={[0:3], [0:3], [0:8], [80:88]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1459 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1403, f32[3,3,8,8]{3,2,1,0} %slice.1435), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_10"}
  %slice.1404 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1392), slice={[0:1], [0:30], [0:30], [88:96]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1436 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.260), slice={[0:3], [0:3], [0:8], [88:96]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1460 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1404, f32[3,3,8,8]{3,2,1,0} %slice.1436), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_11"}
  %slice.1405 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1392), slice={[0:1], [0:30], [0:30], [96:104]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1437 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.260), slice={[0:3], [0:3], [0:8], [96:104]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1461 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1405, f32[3,3,8,8]{3,2,1,0} %slice.1437), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_12"}
  %slice.1406 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1392), slice={[0:1], [0:30], [0:30], [104:112]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1438 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.260), slice={[0:3], [0:3], [0:8], [104:112]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1462 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1406, f32[3,3,8,8]{3,2,1,0} %slice.1438), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_13"}
  %slice.1407 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1392), slice={[0:1], [0:30], [0:30], [112:120]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1439 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.260), slice={[0:3], [0:3], [0:8], [112:120]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1463 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1407, f32[3,3,8,8]{3,2,1,0} %slice.1439), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_14"}
  %slice.1408 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1392), slice={[0:1], [0:30], [0:30], [120:128]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1440 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.260), slice={[0:3], [0:3], [0:8], [120:128]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1464 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1408, f32[3,3,8,8]{3,2,1,0} %slice.1440), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_15"}
  %slice.1409 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1392), slice={[0:1], [0:30], [0:30], [128:136]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1441 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.260), slice={[0:3], [0:3], [0:8], [128:136]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1465 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1409, f32[3,3,8,8]{3,2,1,0} %slice.1441), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_16"}
  %slice.1410 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1392), slice={[0:1], [0:30], [0:30], [136:144]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1442 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.260), slice={[0:3], [0:3], [0:8], [136:144]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1466 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1410, f32[3,3,8,8]{3,2,1,0} %slice.1442), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_17"}
  %slice.1411 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1392), slice={[0:1], [0:30], [0:30], [144:152]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1443 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.260), slice={[0:3], [0:3], [0:8], [144:152]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1467 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1411, f32[3,3,8,8]{3,2,1,0} %slice.1443), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_18"}
  %slice.1412 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1392), slice={[0:1], [0:30], [0:30], [152:160]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1444 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.260), slice={[0:3], [0:3], [0:8], [152:160]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1468 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1412, f32[3,3,8,8]{3,2,1,0} %slice.1444), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_19"}
  %slice.1413 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1392), slice={[0:1], [0:30], [0:30], [160:168]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1445 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.260), slice={[0:3], [0:3], [0:8], [160:168]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1470 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1413, f32[3,3,8,8]{3,2,1,0} %slice.1445), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_20"}
  %slice.1414 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1392), slice={[0:1], [0:30], [0:30], [168:176]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1446 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.260), slice={[0:3], [0:3], [0:8], [168:176]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1471 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1414, f32[3,3,8,8]{3,2,1,0} %slice.1446), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_21"}
  %slice.1415 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1392), slice={[0:1], [0:30], [0:30], [176:184]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1447 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.260), slice={[0:3], [0:3], [0:8], [176:184]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1472 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1415, f32[3,3,8,8]{3,2,1,0} %slice.1447), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_22"}
  %slice.1416 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1392), slice={[0:1], [0:30], [0:30], [184:192]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1448 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.260), slice={[0:3], [0:3], [0:8], [184:192]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1473 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1416, f32[3,3,8,8]{3,2,1,0} %slice.1448), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_23"}
  %slice.1417 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1392), slice={[0:1], [0:30], [0:30], [192:200]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1449 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.260), slice={[0:3], [0:3], [0:8], [192:200]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1474 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1417, f32[3,3,8,8]{3,2,1,0} %slice.1449), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_24"}
  %slice.1418 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1392), slice={[0:1], [0:30], [0:30], [200:208]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1450 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.260), slice={[0:3], [0:3], [0:8], [200:208]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1475 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1418, f32[3,3,8,8]{3,2,1,0} %slice.1450), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_25"}
  %slice.1419 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1392), slice={[0:1], [0:30], [0:30], [208:216]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1451 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.260), slice={[0:3], [0:3], [0:8], [208:216]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1476 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1419, f32[3,3,8,8]{3,2,1,0} %slice.1451), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_26"}
  %slice.1420 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1392), slice={[0:1], [0:30], [0:30], [216:224]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1452 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.260), slice={[0:3], [0:3], [0:8], [216:224]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1477 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1420, f32[3,3,8,8]{3,2,1,0} %slice.1452), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_27"}
  %slice.1421 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1392), slice={[0:1], [0:30], [0:30], [224:232]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1453 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.260), slice={[0:3], [0:3], [0:8], [224:232]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1478 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1421, f32[3,3,8,8]{3,2,1,0} %slice.1453), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_28"}
  %slice.1422 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1392), slice={[0:1], [0:30], [0:30], [232:240]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1454 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.260), slice={[0:3], [0:3], [0:8], [232:240]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1479 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1422, f32[3,3,8,8]{3,2,1,0} %slice.1454), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_29"}
  %slice.1423 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1392), slice={[0:1], [0:30], [0:30], [240:248]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1455 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.260), slice={[0:3], [0:3], [0:8], [240:248]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1481 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1423, f32[3,3,8,8]{3,2,1,0} %slice.1455), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_30"}
  %slice.1424 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1392), slice={[0:1], [0:30], [0:30], [248:256]}, metadata={op_type="Split" op_name="split_13"}
  %slice.1456 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.260), slice={[0:3], [0:3], [0:8], [248:256]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1482 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1424, f32[3,3,8,8]{3,2,1,0} %slice.1456), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_31"}
  %concatenate.1489 = f32[1,28,28,256]{3,2,1,0} concatenate(f32[1,28,28,8]{3,2,1,0} %convolution.1457, f32[1,28,28,8]{3,2,1,0} %convolution.1458, f32[1,28,28,8]{3,2,1,0} %convolution.1469, f32[1,28,28,8]{3,2,1,0} %convolution.1480, f32[1,28,28,8]{3,2,1,0} %convolution.1483, f32[1,28,28,8]{3,2,1,0} %convolution.1484, f32[1,28,28,8]{3,2,1,0} %convolution.1485, f32[1,28,28,8]{3,2,1,0} %convolution.1486, f32[1,28,28,8]{3,2,1,0} %convolution.1487, f32[1,28,28,8]{3,2,1,0} %convolution.1488, f32[1,28,28,8]{3,2,1,0} %convolution.1459, f32[1,28,28,8]{3,2,1,0} %convolution.1460, f32[1,28,28,8]{3,2,1,0} %convolution.1461, f32[1,28,28,8]{3,2,1,0} %convolution.1462, f32[1,28,28,8]{3,2,1,0} %convolution.1463, f32[1,28,28,8]{3,2,1,0} %convolution.1464, f32[1,28,28,8]{3,2,1,0} %convolution.1465, f32[1,28,28,8]{3,2,1,0} %convolution.1466, f32[1,28,28,8]{3,2,1,0} %convolution.1467, f32[1,28,28,8]{3,2,1,0} %convolution.1468, f32[1,28,28,8]{3,2,1,0} %convolution.1470, f32[1,28,28,8]{3,2,1,0} %convolution.1471, f32[1,28,28,8]{3,2,1,0} %convolution.1472, f32[1,28,28,8]{3,2,1,0} %convolution.1473, f32[1,28,28,8]{3,2,1,0} %convolution.1474, f32[1,28,28,8]{3,2,1,0} %convolution.1475, f32[1,28,28,8]{3,2,1,0} %convolution.1476, f32[1,28,28,8]{3,2,1,0} %convolution.1477, f32[1,28,28,8]{3,2,1,0} %convolution.1478, f32[1,28,28,8]{3,2,1,0} %convolution.1479, f32[1,28,28,8]{3,2,1,0} %convolution.1481, f32[1,28,28,8]{3,2,1,0} %convolution.1482), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_6"}
  %constant.1369 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit4_bn2/add"}
  %broadcast.1370 = f32[256]{0} broadcast(f32[] %constant.1369), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit4_bn2/add"}
  %arg41.42 = f32[256]{0} parameter(41), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.261 = f32[256]{0} reshape(f32[256]{0} %arg41.42)
  %add.1371 = f32[256]{0} add(f32[256]{0} %broadcast.1370, f32[256]{0} %reshape.261), metadata={op_type="AddV2" op_name="stage2_unit4_bn2/add"}
  %rsqrt.1372 = f32[256]{0} rsqrt(f32[256]{0} %add.1371), metadata={op_type="Rsqrt" op_name="stage2_unit4_bn2/Rsqrt"}
  %arg88.89 = f32[256]{0} parameter(88), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.308 = f32[256]{0} reshape(f32[256]{0} %arg88.89)
  %multiply.1373 = f32[256]{0} multiply(f32[256]{0} %rsqrt.1372, f32[256]{0} %reshape.308), metadata={op_type="Mul" op_name="stage2_unit4_bn2/mul"}
  %broadcast.1490 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.1373), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit4_bn2/mul_1"}
  %multiply.1491 = f32[1,28,28,256]{3,2,1,0} multiply(f32[1,28,28,256]{3,2,1,0} %concatenate.1489, f32[1,28,28,256]{3,2,1,0} %broadcast.1490), metadata={op_type="Mul" op_name="stage2_unit4_bn2/mul_1"}
  %arg175.176 = f32[256]{0} parameter(175), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.395 = f32[256]{0} reshape(f32[256]{0} %arg175.176)
  %arg132.133 = f32[256]{0} parameter(132), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.352 = f32[256]{0} reshape(f32[256]{0} %arg132.133)
  %multiply.1374 = f32[256]{0} multiply(f32[256]{0} %multiply.1373, f32[256]{0} %reshape.352), metadata={op_type="Mul" op_name="stage2_unit4_bn2/mul_2"}
  %subtract.1375 = f32[256]{0} subtract(f32[256]{0} %reshape.395, f32[256]{0} %multiply.1374), metadata={op_type="Sub" op_name="stage2_unit4_bn2/sub"}
  %broadcast.1492 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.1375), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit4_bn2/add_1"}
  %add.1493 = f32[1,28,28,256]{3,2,1,0} add(f32[1,28,28,256]{3,2,1,0} %multiply.1491, f32[1,28,28,256]{3,2,1,0} %broadcast.1492), metadata={op_type="AddV2" op_name="stage2_unit4_bn2/add_1"}
  %maximum.1496 = f32[1,28,28,256]{3,2,1,0} maximum(f32[1,28,28,256]{3,2,1,0} %broadcast.1495, f32[1,28,28,256]{3,2,1,0} %add.1493), metadata={op_type="Relu" op_name="stage2_unit4_relu2"}
  %arg204.205 = f32[1,1,256,512]{3,2,1,0} parameter(204), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.424 = f32[1,1,256,512]{3,2,1,0} reshape(f32[1,1,256,512]{3,2,1,0} %arg204.205)
  %convolution.1497 = f32[1,28,28,512]{3,2,1,0} convolution(f32[1,28,28,256]{3,2,1,0} %maximum.1496, f32[1,1,256,512]{3,2,1,0} %reshape.424), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv3"}
  %multiply.1499 = f32[1,28,28,512]{3,2,1,0} multiply(f32[1,28,28,512]{3,2,1,0} %broadcast.1498, f32[1,28,28,512]{3,2,1,0} %convolution.1497), metadata={op_type="Mul" op_name="stage2_unit4_bn3/mul_1"}
  %arg176.177 = f32[512]{0} parameter(176), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.396 = f32[512]{0} reshape(f32[512]{0} %arg176.177)
  %arg133.134 = f32[512]{0} parameter(133), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.353 = f32[512]{0} reshape(f32[512]{0} %arg133.134)
  %multiply.1381 = f32[512]{0} multiply(f32[512]{0} %multiply.1380, f32[512]{0} %reshape.353), metadata={op_type="Mul" op_name="stage2_unit4_bn3/mul_2"}
  %subtract.1382 = f32[512]{0} subtract(f32[512]{0} %reshape.396, f32[512]{0} %multiply.1381), metadata={op_type="Sub" op_name="stage2_unit4_bn3/sub"}
  %broadcast.1500 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.1382), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit4_bn3/add_1"}
  %add.1501 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %multiply.1499, f32[1,28,28,512]{3,2,1,0} %broadcast.1500), metadata={op_type="AddV2" op_name="stage2_unit4_bn3/add_1"}
  %add.1502 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %maximum.1361, f32[1,28,28,512]{3,2,1,0} %add.1501), metadata={op_type="AddV2" op_name="add_6"}
  %maximum.1505 = f32[1,28,28,512]{3,2,1,0} maximum(f32[1,28,28,512]{3,2,1,0} %broadcast.1504, f32[1,28,28,512]{3,2,1,0} %add.1502), metadata={op_type="Relu" op_name="stage2_unit4_relu"}
  %arg205.206 = f32[1,1,512,512]{3,2,1,0} parameter(205), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.425 = f32[1,1,512,512]{3,2,1,0} reshape(f32[1,1,512,512]{3,2,1,0} %arg205.206)
  %convolution.1527 = f32[1,28,28,512]{3,2,1,0} convolution(f32[1,28,28,512]{3,2,1,0} %maximum.1505, f32[1,1,512,512]{3,2,1,0} %reshape.425), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv1"}
  %multiply.1529 = f32[1,28,28,512]{3,2,1,0} multiply(f32[1,28,28,512]{3,2,1,0} %broadcast.1528, f32[1,28,28,512]{3,2,1,0} %convolution.1527), metadata={op_type="Mul" op_name="stage3_unit1_bn1/mul_1"}
  %arg177.178 = f32[512]{0} parameter(177), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.397 = f32[512]{0} reshape(f32[512]{0} %arg177.178)
  %arg134.135 = f32[512]{0} parameter(134), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.354 = f32[512]{0} reshape(f32[512]{0} %arg134.135)
  %multiply.1511 = f32[512]{0} multiply(f32[512]{0} %multiply.1510, f32[512]{0} %reshape.354), metadata={op_type="Mul" op_name="stage3_unit1_bn1/mul_2"}
  %subtract.1512 = f32[512]{0} subtract(f32[512]{0} %reshape.397, f32[512]{0} %multiply.1511), metadata={op_type="Sub" op_name="stage3_unit1_bn1/sub"}
  %broadcast.1530 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.1512), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit1_bn1/add_1"}
  %add.1531 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %multiply.1529, f32[1,28,28,512]{3,2,1,0} %broadcast.1530), metadata={op_type="AddV2" op_name="stage3_unit1_bn1/add_1"}
  %maximum.1534 = f32[1,28,28,512]{3,2,1,0} maximum(f32[1,28,28,512]{3,2,1,0} %broadcast.1533, f32[1,28,28,512]{3,2,1,0} %add.1531), metadata={op_type="Relu" op_name="stage3_unit1_relu1"}
  %constant.1535 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_8"}
  %pad.1536 = f32[1,30,30,512]{3,2,1,0} pad(f32[1,28,28,512]{3,2,1,0} %maximum.1534, f32[] %constant.1535), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_8"}
  %slice.1537 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1536), slice={[0:1], [0:30], [0:30], [0:16]}, metadata={op_type="Split" op_name="split_15"}
  %arg46.47 = f32[3,3,16,512]{3,2,1,0} parameter(46), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.266 = f32[3,3,16,512]{3,2,1,0} reshape(f32[3,3,16,512]{3,2,1,0} %arg46.47)
  %slice.1569 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.266), slice={[0:3], [0:3], [0:16], [0:16]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1601 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1537, f32[3,3,16,16]{3,2,1,0} %slice.1569), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2"}
  %slice.1538 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1536), slice={[0:1], [0:30], [0:30], [16:32]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1570 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.266), slice={[0:3], [0:3], [0:16], [16:32]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1602 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1538, f32[3,3,16,16]{3,2,1,0} %slice.1570), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_1"}
  %slice.1539 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1536), slice={[0:1], [0:30], [0:30], [32:48]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1571 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.266), slice={[0:3], [0:3], [0:16], [32:48]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1613 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1539, f32[3,3,16,16]{3,2,1,0} %slice.1571), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_2"}
  %slice.1540 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1536), slice={[0:1], [0:30], [0:30], [48:64]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1572 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.266), slice={[0:3], [0:3], [0:16], [48:64]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1624 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1540, f32[3,3,16,16]{3,2,1,0} %slice.1572), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_3"}
  %slice.1541 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1536), slice={[0:1], [0:30], [0:30], [64:80]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1573 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.266), slice={[0:3], [0:3], [0:16], [64:80]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1627 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1541, f32[3,3,16,16]{3,2,1,0} %slice.1573), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_4"}
  %slice.1542 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1536), slice={[0:1], [0:30], [0:30], [80:96]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1574 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.266), slice={[0:3], [0:3], [0:16], [80:96]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1628 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1542, f32[3,3,16,16]{3,2,1,0} %slice.1574), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_5"}
  %slice.1543 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1536), slice={[0:1], [0:30], [0:30], [96:112]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1575 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.266), slice={[0:3], [0:3], [0:16], [96:112]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1629 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1543, f32[3,3,16,16]{3,2,1,0} %slice.1575), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_6"}
  %slice.1544 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1536), slice={[0:1], [0:30], [0:30], [112:128]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1576 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.266), slice={[0:3], [0:3], [0:16], [112:128]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1630 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1544, f32[3,3,16,16]{3,2,1,0} %slice.1576), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_7"}
  %slice.1545 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1536), slice={[0:1], [0:30], [0:30], [128:144]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1577 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.266), slice={[0:3], [0:3], [0:16], [128:144]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1631 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1545, f32[3,3,16,16]{3,2,1,0} %slice.1577), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_8"}
  %slice.1546 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1536), slice={[0:1], [0:30], [0:30], [144:160]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1578 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.266), slice={[0:3], [0:3], [0:16], [144:160]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1632 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1546, f32[3,3,16,16]{3,2,1,0} %slice.1578), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_9"}
  %slice.1547 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1536), slice={[0:1], [0:30], [0:30], [160:176]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1579 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.266), slice={[0:3], [0:3], [0:16], [160:176]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1603 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1547, f32[3,3,16,16]{3,2,1,0} %slice.1579), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_10"}
  %slice.1548 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1536), slice={[0:1], [0:30], [0:30], [176:192]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1580 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.266), slice={[0:3], [0:3], [0:16], [176:192]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1604 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1548, f32[3,3,16,16]{3,2,1,0} %slice.1580), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_11"}
  %slice.1549 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1536), slice={[0:1], [0:30], [0:30], [192:208]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1581 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.266), slice={[0:3], [0:3], [0:16], [192:208]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1605 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1549, f32[3,3,16,16]{3,2,1,0} %slice.1581), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_12"}
  %slice.1550 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1536), slice={[0:1], [0:30], [0:30], [208:224]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1582 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.266), slice={[0:3], [0:3], [0:16], [208:224]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1606 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1550, f32[3,3,16,16]{3,2,1,0} %slice.1582), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_13"}
  %slice.1551 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1536), slice={[0:1], [0:30], [0:30], [224:240]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1583 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.266), slice={[0:3], [0:3], [0:16], [224:240]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1607 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1551, f32[3,3,16,16]{3,2,1,0} %slice.1583), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_14"}
  %slice.1552 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1536), slice={[0:1], [0:30], [0:30], [240:256]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1584 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.266), slice={[0:3], [0:3], [0:16], [240:256]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1608 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1552, f32[3,3,16,16]{3,2,1,0} %slice.1584), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_15"}
  %slice.1553 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1536), slice={[0:1], [0:30], [0:30], [256:272]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1585 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.266), slice={[0:3], [0:3], [0:16], [256:272]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1609 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1553, f32[3,3,16,16]{3,2,1,0} %slice.1585), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_16"}
  %slice.1554 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1536), slice={[0:1], [0:30], [0:30], [272:288]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1586 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.266), slice={[0:3], [0:3], [0:16], [272:288]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1610 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1554, f32[3,3,16,16]{3,2,1,0} %slice.1586), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_17"}
  %slice.1555 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1536), slice={[0:1], [0:30], [0:30], [288:304]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1587 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.266), slice={[0:3], [0:3], [0:16], [288:304]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1611 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1555, f32[3,3,16,16]{3,2,1,0} %slice.1587), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_18"}
  %slice.1556 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1536), slice={[0:1], [0:30], [0:30], [304:320]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1588 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.266), slice={[0:3], [0:3], [0:16], [304:320]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1612 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1556, f32[3,3,16,16]{3,2,1,0} %slice.1588), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_19"}
  %slice.1557 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1536), slice={[0:1], [0:30], [0:30], [320:336]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1589 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.266), slice={[0:3], [0:3], [0:16], [320:336]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1614 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1557, f32[3,3,16,16]{3,2,1,0} %slice.1589), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_20"}
  %slice.1558 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1536), slice={[0:1], [0:30], [0:30], [336:352]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1590 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.266), slice={[0:3], [0:3], [0:16], [336:352]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1615 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1558, f32[3,3,16,16]{3,2,1,0} %slice.1590), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_21"}
  %slice.1559 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1536), slice={[0:1], [0:30], [0:30], [352:368]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1591 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.266), slice={[0:3], [0:3], [0:16], [352:368]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1616 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1559, f32[3,3,16,16]{3,2,1,0} %slice.1591), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_22"}
  %slice.1560 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1536), slice={[0:1], [0:30], [0:30], [368:384]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1592 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.266), slice={[0:3], [0:3], [0:16], [368:384]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1617 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1560, f32[3,3,16,16]{3,2,1,0} %slice.1592), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_23"}
  %slice.1561 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1536), slice={[0:1], [0:30], [0:30], [384:400]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1593 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.266), slice={[0:3], [0:3], [0:16], [384:400]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1618 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1561, f32[3,3,16,16]{3,2,1,0} %slice.1593), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_24"}
  %slice.1562 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1536), slice={[0:1], [0:30], [0:30], [400:416]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1594 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.266), slice={[0:3], [0:3], [0:16], [400:416]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1619 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1562, f32[3,3,16,16]{3,2,1,0} %slice.1594), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_25"}
  %slice.1563 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1536), slice={[0:1], [0:30], [0:30], [416:432]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1595 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.266), slice={[0:3], [0:3], [0:16], [416:432]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1620 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1563, f32[3,3,16,16]{3,2,1,0} %slice.1595), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_26"}
  %slice.1564 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1536), slice={[0:1], [0:30], [0:30], [432:448]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1596 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.266), slice={[0:3], [0:3], [0:16], [432:448]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1621 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1564, f32[3,3,16,16]{3,2,1,0} %slice.1596), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_27"}
  %slice.1565 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1536), slice={[0:1], [0:30], [0:30], [448:464]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1597 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.266), slice={[0:3], [0:3], [0:16], [448:464]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1622 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1565, f32[3,3,16,16]{3,2,1,0} %slice.1597), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_28"}
  %slice.1566 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1536), slice={[0:1], [0:30], [0:30], [464:480]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1598 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.266), slice={[0:3], [0:3], [0:16], [464:480]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1623 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1566, f32[3,3,16,16]{3,2,1,0} %slice.1598), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_29"}
  %slice.1567 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1536), slice={[0:1], [0:30], [0:30], [480:496]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1599 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.266), slice={[0:3], [0:3], [0:16], [480:496]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1625 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1567, f32[3,3,16,16]{3,2,1,0} %slice.1599), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_30"}
  %slice.1568 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1536), slice={[0:1], [0:30], [0:30], [496:512]}, metadata={op_type="Split" op_name="split_15"}
  %slice.1600 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.266), slice={[0:3], [0:3], [0:16], [496:512]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1626 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1568, f32[3,3,16,16]{3,2,1,0} %slice.1600), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_31"}
  %concatenate.1633 = f32[1,14,14,512]{3,2,1,0} concatenate(f32[1,14,14,16]{3,2,1,0} %convolution.1601, f32[1,14,14,16]{3,2,1,0} %convolution.1602, f32[1,14,14,16]{3,2,1,0} %convolution.1613, f32[1,14,14,16]{3,2,1,0} %convolution.1624, f32[1,14,14,16]{3,2,1,0} %convolution.1627, f32[1,14,14,16]{3,2,1,0} %convolution.1628, f32[1,14,14,16]{3,2,1,0} %convolution.1629, f32[1,14,14,16]{3,2,1,0} %convolution.1630, f32[1,14,14,16]{3,2,1,0} %convolution.1631, f32[1,14,14,16]{3,2,1,0} %convolution.1632, f32[1,14,14,16]{3,2,1,0} %convolution.1603, f32[1,14,14,16]{3,2,1,0} %convolution.1604, f32[1,14,14,16]{3,2,1,0} %convolution.1605, f32[1,14,14,16]{3,2,1,0} %convolution.1606, f32[1,14,14,16]{3,2,1,0} %convolution.1607, f32[1,14,14,16]{3,2,1,0} %convolution.1608, f32[1,14,14,16]{3,2,1,0} %convolution.1609, f32[1,14,14,16]{3,2,1,0} %convolution.1610, f32[1,14,14,16]{3,2,1,0} %convolution.1611, f32[1,14,14,16]{3,2,1,0} %convolution.1612, f32[1,14,14,16]{3,2,1,0} %convolution.1614, f32[1,14,14,16]{3,2,1,0} %convolution.1615, f32[1,14,14,16]{3,2,1,0} %convolution.1616, f32[1,14,14,16]{3,2,1,0} %convolution.1617, f32[1,14,14,16]{3,2,1,0} %convolution.1618, f32[1,14,14,16]{3,2,1,0} %convolution.1619, f32[1,14,14,16]{3,2,1,0} %convolution.1620, f32[1,14,14,16]{3,2,1,0} %convolution.1621, f32[1,14,14,16]{3,2,1,0} %convolution.1622, f32[1,14,14,16]{3,2,1,0} %convolution.1623, f32[1,14,14,16]{3,2,1,0} %convolution.1625, f32[1,14,14,16]{3,2,1,0} %convolution.1626), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_7"}
  %constant.1513 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit1_bn2/add"}
  %broadcast.1514 = f32[512]{0} broadcast(f32[] %constant.1513), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit1_bn2/add"}
  %arg47.48 = f32[512]{0} parameter(47), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.267 = f32[512]{0} reshape(f32[512]{0} %arg47.48)
  %add.1515 = f32[512]{0} add(f32[512]{0} %broadcast.1514, f32[512]{0} %reshape.267), metadata={op_type="AddV2" op_name="stage3_unit1_bn2/add"}
  %rsqrt.1516 = f32[512]{0} rsqrt(f32[512]{0} %add.1515), metadata={op_type="Rsqrt" op_name="stage3_unit1_bn2/Rsqrt"}
  %arg93.94 = f32[512]{0} parameter(93), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.313 = f32[512]{0} reshape(f32[512]{0} %arg93.94)
  %multiply.1517 = f32[512]{0} multiply(f32[512]{0} %rsqrt.1516, f32[512]{0} %reshape.313), metadata={op_type="Mul" op_name="stage3_unit1_bn2/mul"}
  %broadcast.1634 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.1517), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit1_bn2/mul_1"}
  %multiply.1635 = f32[1,14,14,512]{3,2,1,0} multiply(f32[1,14,14,512]{3,2,1,0} %concatenate.1633, f32[1,14,14,512]{3,2,1,0} %broadcast.1634), metadata={op_type="Mul" op_name="stage3_unit1_bn2/mul_1"}
  %arg180.181 = f32[512]{0} parameter(180), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.400 = f32[512]{0} reshape(f32[512]{0} %arg180.181)
  %arg137.138 = f32[512]{0} parameter(137), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.357 = f32[512]{0} reshape(f32[512]{0} %arg137.138)
  %multiply.1518 = f32[512]{0} multiply(f32[512]{0} %multiply.1517, f32[512]{0} %reshape.357), metadata={op_type="Mul" op_name="stage3_unit1_bn2/mul_2"}
  %subtract.1519 = f32[512]{0} subtract(f32[512]{0} %reshape.400, f32[512]{0} %multiply.1518), metadata={op_type="Sub" op_name="stage3_unit1_bn2/sub"}
  %broadcast.1636 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.1519), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit1_bn2/add_1"}
  %add.1637 = f32[1,14,14,512]{3,2,1,0} add(f32[1,14,14,512]{3,2,1,0} %multiply.1635, f32[1,14,14,512]{3,2,1,0} %broadcast.1636), metadata={op_type="AddV2" op_name="stage3_unit1_bn2/add_1"}
  %maximum.1640 = f32[1,14,14,512]{3,2,1,0} maximum(f32[1,14,14,512]{3,2,1,0} %broadcast.1639, f32[1,14,14,512]{3,2,1,0} %add.1637), metadata={op_type="Relu" op_name="stage3_unit1_relu2"}
  %arg207.208 = f32[1,1,512,1024]{3,2,1,0} parameter(207), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.427 = f32[1,1,512,1024]{3,2,1,0} reshape(f32[1,1,512,1024]{3,2,1,0} %arg207.208)
  %convolution.1641 = f32[1,14,14,1024]{3,2,1,0} convolution(f32[1,14,14,512]{3,2,1,0} %maximum.1640, f32[1,1,512,1024]{3,2,1,0} %reshape.427), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv3"}
  %multiply.1643 = f32[1,14,14,1024]{3,2,1,0} multiply(f32[1,14,14,1024]{3,2,1,0} %broadcast.1642, f32[1,14,14,1024]{3,2,1,0} %convolution.1641), metadata={op_type="Mul" op_name="stage3_unit1_bn3/mul_1"}
  %arg181.182 = f32[1024]{0} parameter(181), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.401 = f32[1024]{0} reshape(f32[1024]{0} %arg181.182)
  %arg138.139 = f32[1024]{0} parameter(138), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.358 = f32[1024]{0} reshape(f32[1024]{0} %arg138.139)
  %multiply.1525 = f32[1024]{0} multiply(f32[1024]{0} %multiply.1524, f32[1024]{0} %reshape.358), metadata={op_type="Mul" op_name="stage3_unit1_bn3/mul_2"}
  %subtract.1526 = f32[1024]{0} subtract(f32[1024]{0} %reshape.401, f32[1024]{0} %multiply.1525), metadata={op_type="Sub" op_name="stage3_unit1_bn3/sub"}
  %broadcast.1644 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %subtract.1526), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit1_bn3/add_1"}
  %add.1645 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %multiply.1643, f32[1,14,14,1024]{3,2,1,0} %broadcast.1644), metadata={op_type="AddV2" op_name="stage3_unit1_bn3/add_1"}
  %arg206.207 = f32[1,1,512,1024]{3,2,1,0} parameter(206), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.426 = f32[1,1,512,1024]{3,2,1,0} reshape(f32[1,1,512,1024]{3,2,1,0} %arg206.207)
  %convolution.1653 = f32[1,14,14,1024]{3,2,1,0} convolution(f32[1,28,28,512]{3,2,1,0} %maximum.1505, f32[1,1,512,1024]{3,2,1,0} %reshape.426), window={size=1x1 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_sc"}
  %constant.1646 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit1_sc_bn/add"}
  %broadcast.1647 = f32[1024]{0} broadcast(f32[] %constant.1646), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit1_sc_bn/add"}
  %arg44.45 = f32[1024]{0} parameter(44), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.264 = f32[1024]{0} reshape(f32[1024]{0} %arg44.45)
  %add.1648 = f32[1024]{0} add(f32[1024]{0} %broadcast.1647, f32[1024]{0} %reshape.264), metadata={op_type="AddV2" op_name="stage3_unit1_sc_bn/add"}
  %rsqrt.1649 = f32[1024]{0} rsqrt(f32[1024]{0} %add.1648), metadata={op_type="Rsqrt" op_name="stage3_unit1_sc_bn/Rsqrt"}
  %arg91.92 = f32[1024]{0} parameter(91), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.311 = f32[1024]{0} reshape(f32[1024]{0} %arg91.92)
  %multiply.1650 = f32[1024]{0} multiply(f32[1024]{0} %rsqrt.1649, f32[1024]{0} %reshape.311), metadata={op_type="Mul" op_name="stage3_unit1_sc_bn/mul"}
  %broadcast.1654 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %multiply.1650), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit1_sc_bn/mul_1"}
  %multiply.1655 = f32[1,14,14,1024]{3,2,1,0} multiply(f32[1,14,14,1024]{3,2,1,0} %convolution.1653, f32[1,14,14,1024]{3,2,1,0} %broadcast.1654), metadata={op_type="Mul" op_name="stage3_unit1_sc_bn/mul_1"}
  %arg178.179 = f32[1024]{0} parameter(178), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.398 = f32[1024]{0} reshape(f32[1024]{0} %arg178.179)
  %arg135.136 = f32[1024]{0} parameter(135), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.355 = f32[1024]{0} reshape(f32[1024]{0} %arg135.136)
  %multiply.1651 = f32[1024]{0} multiply(f32[1024]{0} %multiply.1650, f32[1024]{0} %reshape.355), metadata={op_type="Mul" op_name="stage3_unit1_sc_bn/mul_2"}
  %subtract.1652 = f32[1024]{0} subtract(f32[1024]{0} %reshape.398, f32[1024]{0} %multiply.1651), metadata={op_type="Sub" op_name="stage3_unit1_sc_bn/sub"}
  %broadcast.1656 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %subtract.1652), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit1_sc_bn/add_1"}
  %add.1657 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %multiply.1655, f32[1,14,14,1024]{3,2,1,0} %broadcast.1656), metadata={op_type="AddV2" op_name="stage3_unit1_sc_bn/add_1"}
  %add.1658 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %add.1645, f32[1,14,14,1024]{3,2,1,0} %add.1657), metadata={op_type="AddV2" op_name="add_7"}
  %maximum.1661 = f32[1,14,14,1024]{3,2,1,0} maximum(f32[1,14,14,1024]{3,2,1,0} %broadcast.1660, f32[1,14,14,1024]{3,2,1,0} %add.1658), metadata={op_type="Relu" op_name="stage3_unit1_relu"}
  %constant.1676 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit2_bn3/add"}
  %broadcast.1677 = f32[1024]{0} broadcast(f32[] %constant.1676), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit2_bn3/add"}
  %arg53.54 = f32[1024]{0} parameter(53), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.273 = f32[1024]{0} reshape(f32[1024]{0} %arg53.54)
  %add.1678 = f32[1024]{0} add(f32[1024]{0} %broadcast.1677, f32[1024]{0} %reshape.273), metadata={op_type="AddV2" op_name="stage3_unit2_bn3/add"}
  %rsqrt.1679 = f32[1024]{0} rsqrt(f32[1024]{0} %add.1678), metadata={op_type="Rsqrt" op_name="stage3_unit2_bn3/Rsqrt"}
  %arg98.99 = f32[1024]{0} parameter(98), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.318 = f32[1024]{0} reshape(f32[1024]{0} %arg98.99)
  %multiply.1680 = f32[1024]{0} multiply(f32[1024]{0} %rsqrt.1679, f32[1024]{0} %reshape.318), metadata={op_type="Mul" op_name="stage3_unit2_bn3/mul"}
  %broadcast.1798 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %multiply.1680), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit2_bn3/mul_1"}
  %constant.1794 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit2_relu2"}
  %broadcast.1795 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[] %constant.1794), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit2_relu2"}
  %constant.1688 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit2_relu1"}
  %broadcast.1689 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[] %constant.1688), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit2_relu1"}
  %constant.1662 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit2_bn1/add"}
  %broadcast.1663 = f32[512]{0} broadcast(f32[] %constant.1662), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit2_bn1/add"}
  %arg49.50 = f32[512]{0} parameter(49), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.269 = f32[512]{0} reshape(f32[512]{0} %arg49.50)
  %add.1664 = f32[512]{0} add(f32[512]{0} %broadcast.1663, f32[512]{0} %reshape.269), metadata={op_type="AddV2" op_name="stage3_unit2_bn1/add"}
  %rsqrt.1665 = f32[512]{0} rsqrt(f32[512]{0} %add.1664), metadata={op_type="Rsqrt" op_name="stage3_unit2_bn1/Rsqrt"}
  %arg95.96 = f32[512]{0} parameter(95), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.315 = f32[512]{0} reshape(f32[512]{0} %arg95.96)
  %multiply.1666 = f32[512]{0} multiply(f32[512]{0} %rsqrt.1665, f32[512]{0} %reshape.315), metadata={op_type="Mul" op_name="stage3_unit2_bn1/mul"}
  %broadcast.1684 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.1666), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit2_bn1/mul_1"}
  %arg208.209 = f32[1,1,1024,512]{3,2,1,0} parameter(208), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.428 = f32[1,1,1024,512]{3,2,1,0} reshape(f32[1,1,1024,512]{3,2,1,0} %arg208.209)
  %convolution.1683 = f32[1,14,14,512]{3,2,1,0} convolution(f32[1,14,14,1024]{3,2,1,0} %maximum.1661, f32[1,1,1024,512]{3,2,1,0} %reshape.428), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv1"}
  %multiply.1685 = f32[1,14,14,512]{3,2,1,0} multiply(f32[1,14,14,512]{3,2,1,0} %broadcast.1684, f32[1,14,14,512]{3,2,1,0} %convolution.1683), metadata={op_type="Mul" op_name="stage3_unit2_bn1/mul_1"}
  %arg182.183 = f32[512]{0} parameter(182), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.402 = f32[512]{0} reshape(f32[512]{0} %arg182.183)
  %arg139.140 = f32[512]{0} parameter(139), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.359 = f32[512]{0} reshape(f32[512]{0} %arg139.140)
  %multiply.1667 = f32[512]{0} multiply(f32[512]{0} %multiply.1666, f32[512]{0} %reshape.359), metadata={op_type="Mul" op_name="stage3_unit2_bn1/mul_2"}
  %subtract.1668 = f32[512]{0} subtract(f32[512]{0} %reshape.402, f32[512]{0} %multiply.1667), metadata={op_type="Sub" op_name="stage3_unit2_bn1/sub"}
  %broadcast.1686 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.1668), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit2_bn1/add_1"}
  %add.1687 = f32[1,14,14,512]{3,2,1,0} add(f32[1,14,14,512]{3,2,1,0} %multiply.1685, f32[1,14,14,512]{3,2,1,0} %broadcast.1686), metadata={op_type="AddV2" op_name="stage3_unit2_bn1/add_1"}
  %maximum.1690 = f32[1,14,14,512]{3,2,1,0} maximum(f32[1,14,14,512]{3,2,1,0} %broadcast.1689, f32[1,14,14,512]{3,2,1,0} %add.1687), metadata={op_type="Relu" op_name="stage3_unit2_relu1"}
  %constant.1691 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_9"}
  %pad.1692 = f32[1,16,16,512]{3,2,1,0} pad(f32[1,14,14,512]{3,2,1,0} %maximum.1690, f32[] %constant.1691), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_9"}
  %slice.1693 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1692), slice={[0:1], [0:16], [0:16], [0:16]}, metadata={op_type="Split" op_name="split_17"}
  %arg51.52 = f32[3,3,16,512]{3,2,1,0} parameter(51), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.271 = f32[3,3,16,512]{3,2,1,0} reshape(f32[3,3,16,512]{3,2,1,0} %arg51.52)
  %slice.1725 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.271), slice={[0:3], [0:3], [0:16], [0:16]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1757 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1693, f32[3,3,16,16]{3,2,1,0} %slice.1725), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2"}
  %slice.1694 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1692), slice={[0:1], [0:16], [0:16], [16:32]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1726 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.271), slice={[0:3], [0:3], [0:16], [16:32]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1758 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1694, f32[3,3,16,16]{3,2,1,0} %slice.1726), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_1"}
  %slice.1695 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1692), slice={[0:1], [0:16], [0:16], [32:48]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1727 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.271), slice={[0:3], [0:3], [0:16], [32:48]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1769 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1695, f32[3,3,16,16]{3,2,1,0} %slice.1727), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_2"}
  %slice.1696 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1692), slice={[0:1], [0:16], [0:16], [48:64]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1728 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.271), slice={[0:3], [0:3], [0:16], [48:64]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1780 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1696, f32[3,3,16,16]{3,2,1,0} %slice.1728), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_3"}
  %slice.1697 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1692), slice={[0:1], [0:16], [0:16], [64:80]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1729 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.271), slice={[0:3], [0:3], [0:16], [64:80]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1783 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1697, f32[3,3,16,16]{3,2,1,0} %slice.1729), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_4"}
  %slice.1698 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1692), slice={[0:1], [0:16], [0:16], [80:96]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1730 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.271), slice={[0:3], [0:3], [0:16], [80:96]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1784 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1698, f32[3,3,16,16]{3,2,1,0} %slice.1730), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_5"}
  %slice.1699 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1692), slice={[0:1], [0:16], [0:16], [96:112]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1731 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.271), slice={[0:3], [0:3], [0:16], [96:112]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1785 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1699, f32[3,3,16,16]{3,2,1,0} %slice.1731), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_6"}
  %slice.1700 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1692), slice={[0:1], [0:16], [0:16], [112:128]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1732 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.271), slice={[0:3], [0:3], [0:16], [112:128]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1786 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1700, f32[3,3,16,16]{3,2,1,0} %slice.1732), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_7"}
  %slice.1701 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1692), slice={[0:1], [0:16], [0:16], [128:144]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1733 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.271), slice={[0:3], [0:3], [0:16], [128:144]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1787 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1701, f32[3,3,16,16]{3,2,1,0} %slice.1733), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_8"}
  %slice.1702 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1692), slice={[0:1], [0:16], [0:16], [144:160]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1734 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.271), slice={[0:3], [0:3], [0:16], [144:160]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1788 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1702, f32[3,3,16,16]{3,2,1,0} %slice.1734), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_9"}
  %slice.1703 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1692), slice={[0:1], [0:16], [0:16], [160:176]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1735 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.271), slice={[0:3], [0:3], [0:16], [160:176]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1759 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1703, f32[3,3,16,16]{3,2,1,0} %slice.1735), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_10"}
  %slice.1704 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1692), slice={[0:1], [0:16], [0:16], [176:192]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1736 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.271), slice={[0:3], [0:3], [0:16], [176:192]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1760 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1704, f32[3,3,16,16]{3,2,1,0} %slice.1736), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_11"}
  %slice.1705 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1692), slice={[0:1], [0:16], [0:16], [192:208]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1737 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.271), slice={[0:3], [0:3], [0:16], [192:208]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1761 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1705, f32[3,3,16,16]{3,2,1,0} %slice.1737), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_12"}
  %slice.1706 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1692), slice={[0:1], [0:16], [0:16], [208:224]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1738 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.271), slice={[0:3], [0:3], [0:16], [208:224]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1762 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1706, f32[3,3,16,16]{3,2,1,0} %slice.1738), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_13"}
  %slice.1707 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1692), slice={[0:1], [0:16], [0:16], [224:240]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1739 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.271), slice={[0:3], [0:3], [0:16], [224:240]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1763 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1707, f32[3,3,16,16]{3,2,1,0} %slice.1739), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_14"}
  %slice.1708 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1692), slice={[0:1], [0:16], [0:16], [240:256]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1740 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.271), slice={[0:3], [0:3], [0:16], [240:256]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1764 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1708, f32[3,3,16,16]{3,2,1,0} %slice.1740), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_15"}
  %slice.1709 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1692), slice={[0:1], [0:16], [0:16], [256:272]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1741 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.271), slice={[0:3], [0:3], [0:16], [256:272]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1765 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1709, f32[3,3,16,16]{3,2,1,0} %slice.1741), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_16"}
  %slice.1710 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1692), slice={[0:1], [0:16], [0:16], [272:288]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1742 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.271), slice={[0:3], [0:3], [0:16], [272:288]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1766 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1710, f32[3,3,16,16]{3,2,1,0} %slice.1742), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_17"}
  %slice.1711 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1692), slice={[0:1], [0:16], [0:16], [288:304]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1743 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.271), slice={[0:3], [0:3], [0:16], [288:304]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1767 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1711, f32[3,3,16,16]{3,2,1,0} %slice.1743), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_18"}
  %slice.1712 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1692), slice={[0:1], [0:16], [0:16], [304:320]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1744 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.271), slice={[0:3], [0:3], [0:16], [304:320]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1768 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1712, f32[3,3,16,16]{3,2,1,0} %slice.1744), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_19"}
  %slice.1713 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1692), slice={[0:1], [0:16], [0:16], [320:336]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1745 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.271), slice={[0:3], [0:3], [0:16], [320:336]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1770 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1713, f32[3,3,16,16]{3,2,1,0} %slice.1745), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_20"}
  %slice.1714 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1692), slice={[0:1], [0:16], [0:16], [336:352]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1746 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.271), slice={[0:3], [0:3], [0:16], [336:352]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1771 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1714, f32[3,3,16,16]{3,2,1,0} %slice.1746), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_21"}
  %slice.1715 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1692), slice={[0:1], [0:16], [0:16], [352:368]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1747 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.271), slice={[0:3], [0:3], [0:16], [352:368]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1772 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1715, f32[3,3,16,16]{3,2,1,0} %slice.1747), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_22"}
  %slice.1716 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1692), slice={[0:1], [0:16], [0:16], [368:384]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1748 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.271), slice={[0:3], [0:3], [0:16], [368:384]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1773 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1716, f32[3,3,16,16]{3,2,1,0} %slice.1748), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_23"}
  %slice.1717 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1692), slice={[0:1], [0:16], [0:16], [384:400]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1749 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.271), slice={[0:3], [0:3], [0:16], [384:400]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1774 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1717, f32[3,3,16,16]{3,2,1,0} %slice.1749), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_24"}
  %slice.1718 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1692), slice={[0:1], [0:16], [0:16], [400:416]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1750 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.271), slice={[0:3], [0:3], [0:16], [400:416]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1775 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1718, f32[3,3,16,16]{3,2,1,0} %slice.1750), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_25"}
  %slice.1719 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1692), slice={[0:1], [0:16], [0:16], [416:432]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1751 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.271), slice={[0:3], [0:3], [0:16], [416:432]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1776 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1719, f32[3,3,16,16]{3,2,1,0} %slice.1751), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_26"}
  %slice.1720 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1692), slice={[0:1], [0:16], [0:16], [432:448]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1752 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.271), slice={[0:3], [0:3], [0:16], [432:448]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1777 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1720, f32[3,3,16,16]{3,2,1,0} %slice.1752), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_27"}
  %slice.1721 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1692), slice={[0:1], [0:16], [0:16], [448:464]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1753 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.271), slice={[0:3], [0:3], [0:16], [448:464]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1778 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1721, f32[3,3,16,16]{3,2,1,0} %slice.1753), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_28"}
  %slice.1722 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1692), slice={[0:1], [0:16], [0:16], [464:480]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1754 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.271), slice={[0:3], [0:3], [0:16], [464:480]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1779 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1722, f32[3,3,16,16]{3,2,1,0} %slice.1754), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_29"}
  %slice.1723 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1692), slice={[0:1], [0:16], [0:16], [480:496]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1755 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.271), slice={[0:3], [0:3], [0:16], [480:496]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1781 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1723, f32[3,3,16,16]{3,2,1,0} %slice.1755), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_30"}
  %slice.1724 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1692), slice={[0:1], [0:16], [0:16], [496:512]}, metadata={op_type="Split" op_name="split_17"}
  %slice.1756 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.271), slice={[0:3], [0:3], [0:16], [496:512]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.1782 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1724, f32[3,3,16,16]{3,2,1,0} %slice.1756), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_31"}
  %concatenate.1789 = f32[1,14,14,512]{3,2,1,0} concatenate(f32[1,14,14,16]{3,2,1,0} %convolution.1757, f32[1,14,14,16]{3,2,1,0} %convolution.1758, f32[1,14,14,16]{3,2,1,0} %convolution.1769, f32[1,14,14,16]{3,2,1,0} %convolution.1780, f32[1,14,14,16]{3,2,1,0} %convolution.1783, f32[1,14,14,16]{3,2,1,0} %convolution.1784, f32[1,14,14,16]{3,2,1,0} %convolution.1785, f32[1,14,14,16]{3,2,1,0} %convolution.1786, f32[1,14,14,16]{3,2,1,0} %convolution.1787, f32[1,14,14,16]{3,2,1,0} %convolution.1788, f32[1,14,14,16]{3,2,1,0} %convolution.1759, f32[1,14,14,16]{3,2,1,0} %convolution.1760, f32[1,14,14,16]{3,2,1,0} %convolution.1761, f32[1,14,14,16]{3,2,1,0} %convolution.1762, f32[1,14,14,16]{3,2,1,0} %convolution.1763, f32[1,14,14,16]{3,2,1,0} %convolution.1764, f32[1,14,14,16]{3,2,1,0} %convolution.1765, f32[1,14,14,16]{3,2,1,0} %convolution.1766, f32[1,14,14,16]{3,2,1,0} %convolution.1767, f32[1,14,14,16]{3,2,1,0} %convolution.1768, f32[1,14,14,16]{3,2,1,0} %convolution.1770, f32[1,14,14,16]{3,2,1,0} %convolution.1771, f32[1,14,14,16]{3,2,1,0} %convolution.1772, f32[1,14,14,16]{3,2,1,0} %convolution.1773, f32[1,14,14,16]{3,2,1,0} %convolution.1774, f32[1,14,14,16]{3,2,1,0} %convolution.1775, f32[1,14,14,16]{3,2,1,0} %convolution.1776, f32[1,14,14,16]{3,2,1,0} %convolution.1777, f32[1,14,14,16]{3,2,1,0} %convolution.1778, f32[1,14,14,16]{3,2,1,0} %convolution.1779, f32[1,14,14,16]{3,2,1,0} %convolution.1781, f32[1,14,14,16]{3,2,1,0} %convolution.1782), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_8"}
  %constant.1669 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit2_bn2/add"}
  %broadcast.1670 = f32[512]{0} broadcast(f32[] %constant.1669), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit2_bn2/add"}
  %arg52.53 = f32[512]{0} parameter(52), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.272 = f32[512]{0} reshape(f32[512]{0} %arg52.53)
  %add.1671 = f32[512]{0} add(f32[512]{0} %broadcast.1670, f32[512]{0} %reshape.272), metadata={op_type="AddV2" op_name="stage3_unit2_bn2/add"}
  %rsqrt.1672 = f32[512]{0} rsqrt(f32[512]{0} %add.1671), metadata={op_type="Rsqrt" op_name="stage3_unit2_bn2/Rsqrt"}
  %arg97.98 = f32[512]{0} parameter(97), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.317 = f32[512]{0} reshape(f32[512]{0} %arg97.98)
  %multiply.1673 = f32[512]{0} multiply(f32[512]{0} %rsqrt.1672, f32[512]{0} %reshape.317), metadata={op_type="Mul" op_name="stage3_unit2_bn2/mul"}
  %broadcast.1790 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.1673), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit2_bn2/mul_1"}
  %multiply.1791 = f32[1,14,14,512]{3,2,1,0} multiply(f32[1,14,14,512]{3,2,1,0} %concatenate.1789, f32[1,14,14,512]{3,2,1,0} %broadcast.1790), metadata={op_type="Mul" op_name="stage3_unit2_bn2/mul_1"}
  %arg184.185 = f32[512]{0} parameter(184), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.404 = f32[512]{0} reshape(f32[512]{0} %arg184.185)
  %arg141.142 = f32[512]{0} parameter(141), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.361 = f32[512]{0} reshape(f32[512]{0} %arg141.142)
  %multiply.1674 = f32[512]{0} multiply(f32[512]{0} %multiply.1673, f32[512]{0} %reshape.361), metadata={op_type="Mul" op_name="stage3_unit2_bn2/mul_2"}
  %subtract.1675 = f32[512]{0} subtract(f32[512]{0} %reshape.404, f32[512]{0} %multiply.1674), metadata={op_type="Sub" op_name="stage3_unit2_bn2/sub"}
  %broadcast.1792 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.1675), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit2_bn2/add_1"}
  %add.1793 = f32[1,14,14,512]{3,2,1,0} add(f32[1,14,14,512]{3,2,1,0} %multiply.1791, f32[1,14,14,512]{3,2,1,0} %broadcast.1792), metadata={op_type="AddV2" op_name="stage3_unit2_bn2/add_1"}
  %maximum.1796 = f32[1,14,14,512]{3,2,1,0} maximum(f32[1,14,14,512]{3,2,1,0} %broadcast.1795, f32[1,14,14,512]{3,2,1,0} %add.1793), metadata={op_type="Relu" op_name="stage3_unit2_relu2"}
  %arg209.210 = f32[1,1,512,1024]{3,2,1,0} parameter(209), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.429 = f32[1,1,512,1024]{3,2,1,0} reshape(f32[1,1,512,1024]{3,2,1,0} %arg209.210)
  %convolution.1797 = f32[1,14,14,1024]{3,2,1,0} convolution(f32[1,14,14,512]{3,2,1,0} %maximum.1796, f32[1,1,512,1024]{3,2,1,0} %reshape.429), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv3"}
  %multiply.1799 = f32[1,14,14,1024]{3,2,1,0} multiply(f32[1,14,14,1024]{3,2,1,0} %broadcast.1798, f32[1,14,14,1024]{3,2,1,0} %convolution.1797), metadata={op_type="Mul" op_name="stage3_unit2_bn3/mul_1"}
  %arg185.186 = f32[1024]{0} parameter(185), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.405 = f32[1024]{0} reshape(f32[1024]{0} %arg185.186)
  %arg142.143 = f32[1024]{0} parameter(142), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.362 = f32[1024]{0} reshape(f32[1024]{0} %arg142.143)
  %multiply.1681 = f32[1024]{0} multiply(f32[1024]{0} %multiply.1680, f32[1024]{0} %reshape.362), metadata={op_type="Mul" op_name="stage3_unit2_bn3/mul_2"}
  %subtract.1682 = f32[1024]{0} subtract(f32[1024]{0} %reshape.405, f32[1024]{0} %multiply.1681), metadata={op_type="Sub" op_name="stage3_unit2_bn3/sub"}
  %broadcast.1800 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %subtract.1682), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit2_bn3/add_1"}
  %add.1801 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %multiply.1799, f32[1,14,14,1024]{3,2,1,0} %broadcast.1800), metadata={op_type="AddV2" op_name="stage3_unit2_bn3/add_1"}
  %add.1802 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %maximum.1661, f32[1,14,14,1024]{3,2,1,0} %add.1801), metadata={op_type="AddV2" op_name="add_8"}
  %maximum.1805 = f32[1,14,14,1024]{3,2,1,0} maximum(f32[1,14,14,1024]{3,2,1,0} %broadcast.1804, f32[1,14,14,1024]{3,2,1,0} %add.1802), metadata={op_type="Relu" op_name="stage3_unit2_relu"}
  %constant.1820 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit3_bn3/add"}
  %broadcast.1821 = f32[1024]{0} broadcast(f32[] %constant.1820), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit3_bn3/add"}
  %arg7.8 = f32[1024]{0} parameter(7), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.227 = f32[1024]{0} reshape(f32[1024]{0} %arg7.8)
  %add.1822 = f32[1024]{0} add(f32[1024]{0} %broadcast.1821, f32[1024]{0} %reshape.227), metadata={op_type="AddV2" op_name="stage3_unit3_bn3/add"}
  %rsqrt.1823 = f32[1024]{0} rsqrt(f32[1024]{0} %add.1822), metadata={op_type="Rsqrt" op_name="stage3_unit3_bn3/Rsqrt"}
  %arg63.64 = f32[1024]{0} parameter(63), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.283 = f32[1024]{0} reshape(f32[1024]{0} %arg63.64)
  %multiply.1824 = f32[1024]{0} multiply(f32[1024]{0} %rsqrt.1823, f32[1024]{0} %reshape.283), metadata={op_type="Mul" op_name="stage3_unit3_bn3/mul"}
  %broadcast.1942 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %multiply.1824), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit3_bn3/mul_1"}
  %constant.1938 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit3_relu2"}
  %broadcast.1939 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[] %constant.1938), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit3_relu2"}
  %constant.1832 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit3_relu1"}
  %broadcast.1833 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[] %constant.1832), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit3_relu1"}
  %constant.1806 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit3_bn1/add"}
  %broadcast.1807 = f32[512]{0} broadcast(f32[] %constant.1806), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit3_bn1/add"}
  %arg54.55 = f32[512]{0} parameter(54), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.274 = f32[512]{0} reshape(f32[512]{0} %arg54.55)
  %add.1808 = f32[512]{0} add(f32[512]{0} %broadcast.1807, f32[512]{0} %reshape.274), metadata={op_type="AddV2" op_name="stage3_unit3_bn1/add"}
  %rsqrt.1809 = f32[512]{0} rsqrt(f32[512]{0} %add.1808), metadata={op_type="Rsqrt" op_name="stage3_unit3_bn1/Rsqrt"}
  %arg99.100 = f32[512]{0} parameter(99), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.319 = f32[512]{0} reshape(f32[512]{0} %arg99.100)
  %multiply.1810 = f32[512]{0} multiply(f32[512]{0} %rsqrt.1809, f32[512]{0} %reshape.319), metadata={op_type="Mul" op_name="stage3_unit3_bn1/mul"}
  %broadcast.1828 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.1810), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit3_bn1/mul_1"}
  %arg210.211 = f32[1,1,1024,512]{3,2,1,0} parameter(210), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.430 = f32[1,1,1024,512]{3,2,1,0} reshape(f32[1,1,1024,512]{3,2,1,0} %arg210.211)
  %convolution.1827 = f32[1,14,14,512]{3,2,1,0} convolution(f32[1,14,14,1024]{3,2,1,0} %maximum.1805, f32[1,1,1024,512]{3,2,1,0} %reshape.430), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv1"}
  %multiply.1829 = f32[1,14,14,512]{3,2,1,0} multiply(f32[1,14,14,512]{3,2,1,0} %broadcast.1828, f32[1,14,14,512]{3,2,1,0} %convolution.1827), metadata={op_type="Mul" op_name="stage3_unit3_bn1/mul_1"}
  %arg186.187 = f32[512]{0} parameter(186), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.406 = f32[512]{0} reshape(f32[512]{0} %arg186.187)
  %arg143.144 = f32[512]{0} parameter(143), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.363 = f32[512]{0} reshape(f32[512]{0} %arg143.144)
  %multiply.1811 = f32[512]{0} multiply(f32[512]{0} %multiply.1810, f32[512]{0} %reshape.363), metadata={op_type="Mul" op_name="stage3_unit3_bn1/mul_2"}
  %subtract.1812 = f32[512]{0} subtract(f32[512]{0} %reshape.406, f32[512]{0} %multiply.1811), metadata={op_type="Sub" op_name="stage3_unit3_bn1/sub"}
  %broadcast.1830 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.1812), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit3_bn1/add_1"}
  %add.1831 = f32[1,14,14,512]{3,2,1,0} add(f32[1,14,14,512]{3,2,1,0} %multiply.1829, f32[1,14,14,512]{3,2,1,0} %broadcast.1830), metadata={op_type="AddV2" op_name="stage3_unit3_bn1/add_1"}
  %maximum.1834 = f32[1,14,14,512]{3,2,1,0} maximum(f32[1,14,14,512]{3,2,1,0} %broadcast.1833, f32[1,14,14,512]{3,2,1,0} %add.1831), metadata={op_type="Relu" op_name="stage3_unit3_relu1"}
  %constant.1835 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_10"}
  %pad.1836 = f32[1,16,16,512]{3,2,1,0} pad(f32[1,14,14,512]{3,2,1,0} %maximum.1834, f32[] %constant.1835), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_10"}
  %slice.1837 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1836), slice={[0:1], [0:16], [0:16], [0:16]}, metadata={op_type="Split" op_name="split_19"}
  %arg56.57 = f32[3,3,16,512]{3,2,1,0} parameter(56), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.276 = f32[3,3,16,512]{3,2,1,0} reshape(f32[3,3,16,512]{3,2,1,0} %arg56.57)
  %slice.1869 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:16], [0:16]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.1901 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1837, f32[3,3,16,16]{3,2,1,0} %slice.1869), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2"}
  %slice.1838 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1836), slice={[0:1], [0:16], [0:16], [16:32]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1870 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:16], [16:32]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.1902 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1838, f32[3,3,16,16]{3,2,1,0} %slice.1870), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_1"}
  %slice.1839 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1836), slice={[0:1], [0:16], [0:16], [32:48]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1871 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:16], [32:48]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.1913 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1839, f32[3,3,16,16]{3,2,1,0} %slice.1871), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_2"}
  %slice.1840 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1836), slice={[0:1], [0:16], [0:16], [48:64]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1872 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:16], [48:64]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.1924 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1840, f32[3,3,16,16]{3,2,1,0} %slice.1872), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_3"}
  %slice.1841 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1836), slice={[0:1], [0:16], [0:16], [64:80]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1873 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:16], [64:80]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.1927 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1841, f32[3,3,16,16]{3,2,1,0} %slice.1873), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_4"}
  %slice.1842 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1836), slice={[0:1], [0:16], [0:16], [80:96]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1874 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:16], [80:96]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.1928 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1842, f32[3,3,16,16]{3,2,1,0} %slice.1874), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_5"}
  %slice.1843 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1836), slice={[0:1], [0:16], [0:16], [96:112]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1875 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:16], [96:112]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.1929 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1843, f32[3,3,16,16]{3,2,1,0} %slice.1875), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_6"}
  %slice.1844 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1836), slice={[0:1], [0:16], [0:16], [112:128]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1876 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:16], [112:128]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.1930 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1844, f32[3,3,16,16]{3,2,1,0} %slice.1876), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_7"}
  %slice.1845 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1836), slice={[0:1], [0:16], [0:16], [128:144]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1877 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:16], [128:144]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.1931 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1845, f32[3,3,16,16]{3,2,1,0} %slice.1877), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_8"}
  %slice.1846 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1836), slice={[0:1], [0:16], [0:16], [144:160]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1878 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:16], [144:160]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.1932 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1846, f32[3,3,16,16]{3,2,1,0} %slice.1878), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_9"}
  %slice.1847 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1836), slice={[0:1], [0:16], [0:16], [160:176]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1879 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:16], [160:176]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.1903 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1847, f32[3,3,16,16]{3,2,1,0} %slice.1879), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_10"}
  %slice.1848 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1836), slice={[0:1], [0:16], [0:16], [176:192]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1880 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:16], [176:192]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.1904 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1848, f32[3,3,16,16]{3,2,1,0} %slice.1880), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_11"}
  %slice.1849 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1836), slice={[0:1], [0:16], [0:16], [192:208]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1881 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:16], [192:208]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.1905 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1849, f32[3,3,16,16]{3,2,1,0} %slice.1881), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_12"}
  %slice.1850 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1836), slice={[0:1], [0:16], [0:16], [208:224]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1882 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:16], [208:224]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.1906 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1850, f32[3,3,16,16]{3,2,1,0} %slice.1882), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_13"}
  %slice.1851 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1836), slice={[0:1], [0:16], [0:16], [224:240]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1883 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:16], [224:240]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.1907 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1851, f32[3,3,16,16]{3,2,1,0} %slice.1883), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_14"}
  %slice.1852 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1836), slice={[0:1], [0:16], [0:16], [240:256]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1884 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:16], [240:256]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.1908 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1852, f32[3,3,16,16]{3,2,1,0} %slice.1884), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_15"}
  %slice.1853 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1836), slice={[0:1], [0:16], [0:16], [256:272]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1885 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:16], [256:272]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.1909 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1853, f32[3,3,16,16]{3,2,1,0} %slice.1885), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_16"}
  %slice.1854 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1836), slice={[0:1], [0:16], [0:16], [272:288]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1886 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:16], [272:288]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.1910 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1854, f32[3,3,16,16]{3,2,1,0} %slice.1886), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_17"}
  %slice.1855 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1836), slice={[0:1], [0:16], [0:16], [288:304]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1887 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:16], [288:304]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.1911 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1855, f32[3,3,16,16]{3,2,1,0} %slice.1887), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_18"}
  %slice.1856 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1836), slice={[0:1], [0:16], [0:16], [304:320]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1888 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:16], [304:320]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.1912 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1856, f32[3,3,16,16]{3,2,1,0} %slice.1888), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_19"}
  %slice.1857 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1836), slice={[0:1], [0:16], [0:16], [320:336]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1889 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:16], [320:336]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.1914 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1857, f32[3,3,16,16]{3,2,1,0} %slice.1889), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_20"}
  %slice.1858 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1836), slice={[0:1], [0:16], [0:16], [336:352]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1890 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:16], [336:352]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.1915 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1858, f32[3,3,16,16]{3,2,1,0} %slice.1890), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_21"}
  %slice.1859 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1836), slice={[0:1], [0:16], [0:16], [352:368]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1891 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:16], [352:368]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.1916 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1859, f32[3,3,16,16]{3,2,1,0} %slice.1891), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_22"}
  %slice.1860 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1836), slice={[0:1], [0:16], [0:16], [368:384]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1892 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:16], [368:384]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.1917 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1860, f32[3,3,16,16]{3,2,1,0} %slice.1892), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_23"}
  %slice.1861 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1836), slice={[0:1], [0:16], [0:16], [384:400]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1893 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:16], [384:400]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.1918 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1861, f32[3,3,16,16]{3,2,1,0} %slice.1893), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_24"}
  %slice.1862 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1836), slice={[0:1], [0:16], [0:16], [400:416]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1894 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:16], [400:416]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.1919 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1862, f32[3,3,16,16]{3,2,1,0} %slice.1894), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_25"}
  %slice.1863 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1836), slice={[0:1], [0:16], [0:16], [416:432]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1895 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:16], [416:432]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.1920 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1863, f32[3,3,16,16]{3,2,1,0} %slice.1895), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_26"}
  %slice.1864 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1836), slice={[0:1], [0:16], [0:16], [432:448]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1896 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:16], [432:448]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.1921 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1864, f32[3,3,16,16]{3,2,1,0} %slice.1896), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_27"}
  %slice.1865 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1836), slice={[0:1], [0:16], [0:16], [448:464]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1897 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:16], [448:464]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.1922 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1865, f32[3,3,16,16]{3,2,1,0} %slice.1897), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_28"}
  %slice.1866 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1836), slice={[0:1], [0:16], [0:16], [464:480]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1898 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:16], [464:480]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.1923 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1866, f32[3,3,16,16]{3,2,1,0} %slice.1898), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_29"}
  %slice.1867 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1836), slice={[0:1], [0:16], [0:16], [480:496]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1899 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:16], [480:496]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.1925 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1867, f32[3,3,16,16]{3,2,1,0} %slice.1899), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_30"}
  %slice.1868 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1836), slice={[0:1], [0:16], [0:16], [496:512]}, metadata={op_type="Split" op_name="split_19"}
  %slice.1900 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:16], [496:512]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.1926 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1868, f32[3,3,16,16]{3,2,1,0} %slice.1900), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_31"}
  %concatenate.1933 = f32[1,14,14,512]{3,2,1,0} concatenate(f32[1,14,14,16]{3,2,1,0} %convolution.1901, f32[1,14,14,16]{3,2,1,0} %convolution.1902, f32[1,14,14,16]{3,2,1,0} %convolution.1913, f32[1,14,14,16]{3,2,1,0} %convolution.1924, f32[1,14,14,16]{3,2,1,0} %convolution.1927, f32[1,14,14,16]{3,2,1,0} %convolution.1928, f32[1,14,14,16]{3,2,1,0} %convolution.1929, f32[1,14,14,16]{3,2,1,0} %convolution.1930, f32[1,14,14,16]{3,2,1,0} %convolution.1931, f32[1,14,14,16]{3,2,1,0} %convolution.1932, f32[1,14,14,16]{3,2,1,0} %convolution.1903, f32[1,14,14,16]{3,2,1,0} %convolution.1904, f32[1,14,14,16]{3,2,1,0} %convolution.1905, f32[1,14,14,16]{3,2,1,0} %convolution.1906, f32[1,14,14,16]{3,2,1,0} %convolution.1907, f32[1,14,14,16]{3,2,1,0} %convolution.1908, f32[1,14,14,16]{3,2,1,0} %convolution.1909, f32[1,14,14,16]{3,2,1,0} %convolution.1910, f32[1,14,14,16]{3,2,1,0} %convolution.1911, f32[1,14,14,16]{3,2,1,0} %convolution.1912, f32[1,14,14,16]{3,2,1,0} %convolution.1914, f32[1,14,14,16]{3,2,1,0} %convolution.1915, f32[1,14,14,16]{3,2,1,0} %convolution.1916, f32[1,14,14,16]{3,2,1,0} %convolution.1917, f32[1,14,14,16]{3,2,1,0} %convolution.1918, f32[1,14,14,16]{3,2,1,0} %convolution.1919, f32[1,14,14,16]{3,2,1,0} %convolution.1920, f32[1,14,14,16]{3,2,1,0} %convolution.1921, f32[1,14,14,16]{3,2,1,0} %convolution.1922, f32[1,14,14,16]{3,2,1,0} %convolution.1923, f32[1,14,14,16]{3,2,1,0} %convolution.1925, f32[1,14,14,16]{3,2,1,0} %convolution.1926), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_9"}
  %constant.1813 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit3_bn2/add"}
  %broadcast.1814 = f32[512]{0} broadcast(f32[] %constant.1813), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit3_bn2/add"}
  %arg2.3 = f32[512]{0} parameter(2), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.222 = f32[512]{0} reshape(f32[512]{0} %arg2.3)
  %add.1815 = f32[512]{0} add(f32[512]{0} %broadcast.1814, f32[512]{0} %reshape.222), metadata={op_type="AddV2" op_name="stage3_unit3_bn2/add"}
  %rsqrt.1816 = f32[512]{0} rsqrt(f32[512]{0} %add.1815), metadata={op_type="Rsqrt" op_name="stage3_unit3_bn2/Rsqrt"}
  %arg59.60 = f32[512]{0} parameter(59), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.279 = f32[512]{0} reshape(f32[512]{0} %arg59.60)
  %multiply.1817 = f32[512]{0} multiply(f32[512]{0} %rsqrt.1816, f32[512]{0} %reshape.279), metadata={op_type="Mul" op_name="stage3_unit3_bn2/mul"}
  %broadcast.1934 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.1817), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit3_bn2/mul_1"}
  %multiply.1935 = f32[1,14,14,512]{3,2,1,0} multiply(f32[1,14,14,512]{3,2,1,0} %concatenate.1933, f32[1,14,14,512]{3,2,1,0} %broadcast.1934), metadata={op_type="Mul" op_name="stage3_unit3_bn2/mul_1"}
  %arg146.147 = f32[512]{0} parameter(146), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.366 = f32[512]{0} reshape(f32[512]{0} %arg146.147)
  %arg103.104 = f32[512]{0} parameter(103), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.323 = f32[512]{0} reshape(f32[512]{0} %arg103.104)
  %multiply.1818 = f32[512]{0} multiply(f32[512]{0} %multiply.1817, f32[512]{0} %reshape.323), metadata={op_type="Mul" op_name="stage3_unit3_bn2/mul_2"}
  %subtract.1819 = f32[512]{0} subtract(f32[512]{0} %reshape.366, f32[512]{0} %multiply.1818), metadata={op_type="Sub" op_name="stage3_unit3_bn2/sub"}
  %broadcast.1936 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.1819), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit3_bn2/add_1"}
  %add.1937 = f32[1,14,14,512]{3,2,1,0} add(f32[1,14,14,512]{3,2,1,0} %multiply.1935, f32[1,14,14,512]{3,2,1,0} %broadcast.1936), metadata={op_type="AddV2" op_name="stage3_unit3_bn2/add_1"}
  %maximum.1940 = f32[1,14,14,512]{3,2,1,0} maximum(f32[1,14,14,512]{3,2,1,0} %broadcast.1939, f32[1,14,14,512]{3,2,1,0} %add.1937), metadata={op_type="Relu" op_name="stage3_unit3_relu2"}
  %arg211.212 = f32[1,1,512,1024]{3,2,1,0} parameter(211), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.431 = f32[1,1,512,1024]{3,2,1,0} reshape(f32[1,1,512,1024]{3,2,1,0} %arg211.212)
  %convolution.1941 = f32[1,14,14,1024]{3,2,1,0} convolution(f32[1,14,14,512]{3,2,1,0} %maximum.1940, f32[1,1,512,1024]{3,2,1,0} %reshape.431), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv3"}
  %multiply.1943 = f32[1,14,14,1024]{3,2,1,0} multiply(f32[1,14,14,1024]{3,2,1,0} %broadcast.1942, f32[1,14,14,1024]{3,2,1,0} %convolution.1941), metadata={op_type="Mul" op_name="stage3_unit3_bn3/mul_1"}
  %arg150.151 = f32[1024]{0} parameter(150), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.370 = f32[1024]{0} reshape(f32[1024]{0} %arg150.151)
  %arg107.108 = f32[1024]{0} parameter(107), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.327 = f32[1024]{0} reshape(f32[1024]{0} %arg107.108)
  %multiply.1825 = f32[1024]{0} multiply(f32[1024]{0} %multiply.1824, f32[1024]{0} %reshape.327), metadata={op_type="Mul" op_name="stage3_unit3_bn3/mul_2"}
  %subtract.1826 = f32[1024]{0} subtract(f32[1024]{0} %reshape.370, f32[1024]{0} %multiply.1825), metadata={op_type="Sub" op_name="stage3_unit3_bn3/sub"}
  %broadcast.1944 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %subtract.1826), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit3_bn3/add_1"}
  %add.1945 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %multiply.1943, f32[1,14,14,1024]{3,2,1,0} %broadcast.1944), metadata={op_type="AddV2" op_name="stage3_unit3_bn3/add_1"}
  %add.1946 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %maximum.1805, f32[1,14,14,1024]{3,2,1,0} %add.1945), metadata={op_type="AddV2" op_name="add_9"}
  %maximum.1949 = f32[1,14,14,1024]{3,2,1,0} maximum(f32[1,14,14,1024]{3,2,1,0} %broadcast.1948, f32[1,14,14,1024]{3,2,1,0} %add.1946), metadata={op_type="Relu" op_name="stage3_unit3_relu"}
  %constant.1964 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit4_bn3/add"}
  %broadcast.1965 = f32[1024]{0} broadcast(f32[] %constant.1964), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit4_bn3/add"}
  %arg29.30 = f32[1024]{0} parameter(29), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.249 = f32[1024]{0} reshape(f32[1024]{0} %arg29.30)
  %add.1966 = f32[1024]{0} add(f32[1024]{0} %broadcast.1965, f32[1024]{0} %reshape.249), metadata={op_type="AddV2" op_name="stage3_unit4_bn3/add"}
  %rsqrt.1967 = f32[1024]{0} rsqrt(f32[1024]{0} %add.1966), metadata={op_type="Rsqrt" op_name="stage3_unit4_bn3/Rsqrt"}
  %arg79.80 = f32[1024]{0} parameter(79), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.299 = f32[1024]{0} reshape(f32[1024]{0} %arg79.80)
  %multiply.1968 = f32[1024]{0} multiply(f32[1024]{0} %rsqrt.1967, f32[1024]{0} %reshape.299), metadata={op_type="Mul" op_name="stage3_unit4_bn3/mul"}
  %broadcast.2086 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %multiply.1968), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit4_bn3/mul_1"}
  %constant.2082 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit4_relu2"}
  %broadcast.2083 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[] %constant.2082), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit4_relu2"}
  %constant.1976 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit4_relu1"}
  %broadcast.1977 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[] %constant.1976), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit4_relu1"}
  %constant.1950 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit4_bn1/add"}
  %broadcast.1951 = f32[512]{0} broadcast(f32[] %constant.1950), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit4_bn1/add"}
  %arg14.15 = f32[512]{0} parameter(14), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.234 = f32[512]{0} reshape(f32[512]{0} %arg14.15)
  %add.1952 = f32[512]{0} add(f32[512]{0} %broadcast.1951, f32[512]{0} %reshape.234), metadata={op_type="AddV2" op_name="stage3_unit4_bn1/add"}
  %rsqrt.1953 = f32[512]{0} rsqrt(f32[512]{0} %add.1952), metadata={op_type="Rsqrt" op_name="stage3_unit4_bn1/Rsqrt"}
  %arg68.69 = f32[512]{0} parameter(68), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.288 = f32[512]{0} reshape(f32[512]{0} %arg68.69)
  %multiply.1954 = f32[512]{0} multiply(f32[512]{0} %rsqrt.1953, f32[512]{0} %reshape.288), metadata={op_type="Mul" op_name="stage3_unit4_bn1/mul"}
  %broadcast.1972 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.1954), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit4_bn1/mul_1"}
  %arg212.213 = f32[1,1,1024,512]{3,2,1,0} parameter(212), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.432 = f32[1,1,1024,512]{3,2,1,0} reshape(f32[1,1,1024,512]{3,2,1,0} %arg212.213)
  %convolution.1971 = f32[1,14,14,512]{3,2,1,0} convolution(f32[1,14,14,1024]{3,2,1,0} %maximum.1949, f32[1,1,1024,512]{3,2,1,0} %reshape.432), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv1"}
  %multiply.1973 = f32[1,14,14,512]{3,2,1,0} multiply(f32[1,14,14,512]{3,2,1,0} %broadcast.1972, f32[1,14,14,512]{3,2,1,0} %convolution.1971), metadata={op_type="Mul" op_name="stage3_unit4_bn1/mul_1"}
  %arg155.156 = f32[512]{0} parameter(155), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.375 = f32[512]{0} reshape(f32[512]{0} %arg155.156)
  %arg112.113 = f32[512]{0} parameter(112), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.332 = f32[512]{0} reshape(f32[512]{0} %arg112.113)
  %multiply.1955 = f32[512]{0} multiply(f32[512]{0} %multiply.1954, f32[512]{0} %reshape.332), metadata={op_type="Mul" op_name="stage3_unit4_bn1/mul_2"}
  %subtract.1956 = f32[512]{0} subtract(f32[512]{0} %reshape.375, f32[512]{0} %multiply.1955), metadata={op_type="Sub" op_name="stage3_unit4_bn1/sub"}
  %broadcast.1974 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.1956), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit4_bn1/add_1"}
  %add.1975 = f32[1,14,14,512]{3,2,1,0} add(f32[1,14,14,512]{3,2,1,0} %multiply.1973, f32[1,14,14,512]{3,2,1,0} %broadcast.1974), metadata={op_type="AddV2" op_name="stage3_unit4_bn1/add_1"}
  %maximum.1978 = f32[1,14,14,512]{3,2,1,0} maximum(f32[1,14,14,512]{3,2,1,0} %broadcast.1977, f32[1,14,14,512]{3,2,1,0} %add.1975), metadata={op_type="Relu" op_name="stage3_unit4_relu1"}
  %constant.1979 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_11"}
  %pad.1980 = f32[1,16,16,512]{3,2,1,0} pad(f32[1,14,14,512]{3,2,1,0} %maximum.1978, f32[] %constant.1979), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_11"}
  %slice.1981 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1980), slice={[0:1], [0:16], [0:16], [0:16]}, metadata={op_type="Split" op_name="split_21"}
  %arg17.18 = f32[3,3,16,512]{3,2,1,0} parameter(17), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.237 = f32[3,3,16,512]{3,2,1,0} reshape(f32[3,3,16,512]{3,2,1,0} %arg17.18)
  %slice.2013 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.237), slice={[0:3], [0:3], [0:16], [0:16]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2045 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1981, f32[3,3,16,16]{3,2,1,0} %slice.2013), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2"}
  %slice.1982 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1980), slice={[0:1], [0:16], [0:16], [16:32]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2014 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.237), slice={[0:3], [0:3], [0:16], [16:32]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2046 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1982, f32[3,3,16,16]{3,2,1,0} %slice.2014), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_1"}
  %slice.1983 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1980), slice={[0:1], [0:16], [0:16], [32:48]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2015 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.237), slice={[0:3], [0:3], [0:16], [32:48]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2057 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1983, f32[3,3,16,16]{3,2,1,0} %slice.2015), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_2"}
  %slice.1984 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1980), slice={[0:1], [0:16], [0:16], [48:64]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2016 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.237), slice={[0:3], [0:3], [0:16], [48:64]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2068 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1984, f32[3,3,16,16]{3,2,1,0} %slice.2016), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_3"}
  %slice.1985 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1980), slice={[0:1], [0:16], [0:16], [64:80]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2017 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.237), slice={[0:3], [0:3], [0:16], [64:80]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2071 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1985, f32[3,3,16,16]{3,2,1,0} %slice.2017), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_4"}
  %slice.1986 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1980), slice={[0:1], [0:16], [0:16], [80:96]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2018 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.237), slice={[0:3], [0:3], [0:16], [80:96]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2072 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1986, f32[3,3,16,16]{3,2,1,0} %slice.2018), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_5"}
  %slice.1987 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1980), slice={[0:1], [0:16], [0:16], [96:112]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2019 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.237), slice={[0:3], [0:3], [0:16], [96:112]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2073 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1987, f32[3,3,16,16]{3,2,1,0} %slice.2019), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_6"}
  %slice.1988 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1980), slice={[0:1], [0:16], [0:16], [112:128]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2020 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.237), slice={[0:3], [0:3], [0:16], [112:128]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2074 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1988, f32[3,3,16,16]{3,2,1,0} %slice.2020), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_7"}
  %slice.1989 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1980), slice={[0:1], [0:16], [0:16], [128:144]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2021 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.237), slice={[0:3], [0:3], [0:16], [128:144]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2075 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1989, f32[3,3,16,16]{3,2,1,0} %slice.2021), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_8"}
  %slice.1990 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1980), slice={[0:1], [0:16], [0:16], [144:160]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2022 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.237), slice={[0:3], [0:3], [0:16], [144:160]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2076 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1990, f32[3,3,16,16]{3,2,1,0} %slice.2022), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_9"}
  %slice.1991 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1980), slice={[0:1], [0:16], [0:16], [160:176]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2023 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.237), slice={[0:3], [0:3], [0:16], [160:176]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2047 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1991, f32[3,3,16,16]{3,2,1,0} %slice.2023), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_10"}
  %slice.1992 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1980), slice={[0:1], [0:16], [0:16], [176:192]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2024 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.237), slice={[0:3], [0:3], [0:16], [176:192]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2048 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1992, f32[3,3,16,16]{3,2,1,0} %slice.2024), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_11"}
  %slice.1993 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1980), slice={[0:1], [0:16], [0:16], [192:208]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2025 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.237), slice={[0:3], [0:3], [0:16], [192:208]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2049 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1993, f32[3,3,16,16]{3,2,1,0} %slice.2025), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_12"}
  %slice.1994 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1980), slice={[0:1], [0:16], [0:16], [208:224]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2026 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.237), slice={[0:3], [0:3], [0:16], [208:224]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2050 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1994, f32[3,3,16,16]{3,2,1,0} %slice.2026), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_13"}
  %slice.1995 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1980), slice={[0:1], [0:16], [0:16], [224:240]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2027 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.237), slice={[0:3], [0:3], [0:16], [224:240]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2051 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1995, f32[3,3,16,16]{3,2,1,0} %slice.2027), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_14"}
  %slice.1996 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1980), slice={[0:1], [0:16], [0:16], [240:256]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2028 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.237), slice={[0:3], [0:3], [0:16], [240:256]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2052 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1996, f32[3,3,16,16]{3,2,1,0} %slice.2028), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_15"}
  %slice.1997 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1980), slice={[0:1], [0:16], [0:16], [256:272]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2029 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.237), slice={[0:3], [0:3], [0:16], [256:272]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2053 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1997, f32[3,3,16,16]{3,2,1,0} %slice.2029), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_16"}
  %slice.1998 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1980), slice={[0:1], [0:16], [0:16], [272:288]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2030 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.237), slice={[0:3], [0:3], [0:16], [272:288]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2054 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1998, f32[3,3,16,16]{3,2,1,0} %slice.2030), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_17"}
  %slice.1999 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1980), slice={[0:1], [0:16], [0:16], [288:304]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2031 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.237), slice={[0:3], [0:3], [0:16], [288:304]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2055 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.1999, f32[3,3,16,16]{3,2,1,0} %slice.2031), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_18"}
  %slice.2000 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1980), slice={[0:1], [0:16], [0:16], [304:320]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2032 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.237), slice={[0:3], [0:3], [0:16], [304:320]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2056 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2000, f32[3,3,16,16]{3,2,1,0} %slice.2032), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_19"}
  %slice.2001 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1980), slice={[0:1], [0:16], [0:16], [320:336]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2033 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.237), slice={[0:3], [0:3], [0:16], [320:336]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2058 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2001, f32[3,3,16,16]{3,2,1,0} %slice.2033), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_20"}
  %slice.2002 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1980), slice={[0:1], [0:16], [0:16], [336:352]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2034 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.237), slice={[0:3], [0:3], [0:16], [336:352]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2059 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2002, f32[3,3,16,16]{3,2,1,0} %slice.2034), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_21"}
  %slice.2003 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1980), slice={[0:1], [0:16], [0:16], [352:368]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2035 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.237), slice={[0:3], [0:3], [0:16], [352:368]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2060 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2003, f32[3,3,16,16]{3,2,1,0} %slice.2035), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_22"}
  %slice.2004 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1980), slice={[0:1], [0:16], [0:16], [368:384]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2036 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.237), slice={[0:3], [0:3], [0:16], [368:384]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2061 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2004, f32[3,3,16,16]{3,2,1,0} %slice.2036), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_23"}
  %slice.2005 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1980), slice={[0:1], [0:16], [0:16], [384:400]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2037 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.237), slice={[0:3], [0:3], [0:16], [384:400]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2062 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2005, f32[3,3,16,16]{3,2,1,0} %slice.2037), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_24"}
  %slice.2006 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1980), slice={[0:1], [0:16], [0:16], [400:416]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2038 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.237), slice={[0:3], [0:3], [0:16], [400:416]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2063 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2006, f32[3,3,16,16]{3,2,1,0} %slice.2038), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_25"}
  %slice.2007 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1980), slice={[0:1], [0:16], [0:16], [416:432]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2039 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.237), slice={[0:3], [0:3], [0:16], [416:432]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2064 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2007, f32[3,3,16,16]{3,2,1,0} %slice.2039), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_26"}
  %slice.2008 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1980), slice={[0:1], [0:16], [0:16], [432:448]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2040 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.237), slice={[0:3], [0:3], [0:16], [432:448]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2065 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2008, f32[3,3,16,16]{3,2,1,0} %slice.2040), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_27"}
  %slice.2009 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1980), slice={[0:1], [0:16], [0:16], [448:464]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2041 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.237), slice={[0:3], [0:3], [0:16], [448:464]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2066 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2009, f32[3,3,16,16]{3,2,1,0} %slice.2041), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_28"}
  %slice.2010 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1980), slice={[0:1], [0:16], [0:16], [464:480]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2042 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.237), slice={[0:3], [0:3], [0:16], [464:480]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2067 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2010, f32[3,3,16,16]{3,2,1,0} %slice.2042), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_29"}
  %slice.2011 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1980), slice={[0:1], [0:16], [0:16], [480:496]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2043 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.237), slice={[0:3], [0:3], [0:16], [480:496]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2069 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2011, f32[3,3,16,16]{3,2,1,0} %slice.2043), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_30"}
  %slice.2012 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.1980), slice={[0:1], [0:16], [0:16], [496:512]}, metadata={op_type="Split" op_name="split_21"}
  %slice.2044 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.237), slice={[0:3], [0:3], [0:16], [496:512]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2070 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2012, f32[3,3,16,16]{3,2,1,0} %slice.2044), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_31"}
  %concatenate.2077 = f32[1,14,14,512]{3,2,1,0} concatenate(f32[1,14,14,16]{3,2,1,0} %convolution.2045, f32[1,14,14,16]{3,2,1,0} %convolution.2046, f32[1,14,14,16]{3,2,1,0} %convolution.2057, f32[1,14,14,16]{3,2,1,0} %convolution.2068, f32[1,14,14,16]{3,2,1,0} %convolution.2071, f32[1,14,14,16]{3,2,1,0} %convolution.2072, f32[1,14,14,16]{3,2,1,0} %convolution.2073, f32[1,14,14,16]{3,2,1,0} %convolution.2074, f32[1,14,14,16]{3,2,1,0} %convolution.2075, f32[1,14,14,16]{3,2,1,0} %convolution.2076, f32[1,14,14,16]{3,2,1,0} %convolution.2047, f32[1,14,14,16]{3,2,1,0} %convolution.2048, f32[1,14,14,16]{3,2,1,0} %convolution.2049, f32[1,14,14,16]{3,2,1,0} %convolution.2050, f32[1,14,14,16]{3,2,1,0} %convolution.2051, f32[1,14,14,16]{3,2,1,0} %convolution.2052, f32[1,14,14,16]{3,2,1,0} %convolution.2053, f32[1,14,14,16]{3,2,1,0} %convolution.2054, f32[1,14,14,16]{3,2,1,0} %convolution.2055, f32[1,14,14,16]{3,2,1,0} %convolution.2056, f32[1,14,14,16]{3,2,1,0} %convolution.2058, f32[1,14,14,16]{3,2,1,0} %convolution.2059, f32[1,14,14,16]{3,2,1,0} %convolution.2060, f32[1,14,14,16]{3,2,1,0} %convolution.2061, f32[1,14,14,16]{3,2,1,0} %convolution.2062, f32[1,14,14,16]{3,2,1,0} %convolution.2063, f32[1,14,14,16]{3,2,1,0} %convolution.2064, f32[1,14,14,16]{3,2,1,0} %convolution.2065, f32[1,14,14,16]{3,2,1,0} %convolution.2066, f32[1,14,14,16]{3,2,1,0} %convolution.2067, f32[1,14,14,16]{3,2,1,0} %convolution.2069, f32[1,14,14,16]{3,2,1,0} %convolution.2070), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_10"}
  %constant.1957 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit4_bn2/add"}
  %broadcast.1958 = f32[512]{0} broadcast(f32[] %constant.1957), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit4_bn2/add"}
  %arg24.25 = f32[512]{0} parameter(24), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.244 = f32[512]{0} reshape(f32[512]{0} %arg24.25)
  %add.1959 = f32[512]{0} add(f32[512]{0} %broadcast.1958, f32[512]{0} %reshape.244), metadata={op_type="AddV2" op_name="stage3_unit4_bn2/add"}
  %rsqrt.1960 = f32[512]{0} rsqrt(f32[512]{0} %add.1959), metadata={op_type="Rsqrt" op_name="stage3_unit4_bn2/Rsqrt"}
  %arg75.76 = f32[512]{0} parameter(75), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.295 = f32[512]{0} reshape(f32[512]{0} %arg75.76)
  %multiply.1961 = f32[512]{0} multiply(f32[512]{0} %rsqrt.1960, f32[512]{0} %reshape.295), metadata={op_type="Mul" op_name="stage3_unit4_bn2/mul"}
  %broadcast.2078 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.1961), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit4_bn2/mul_1"}
  %multiply.2079 = f32[1,14,14,512]{3,2,1,0} multiply(f32[1,14,14,512]{3,2,1,0} %concatenate.2077, f32[1,14,14,512]{3,2,1,0} %broadcast.2078), metadata={op_type="Mul" op_name="stage3_unit4_bn2/mul_1"}
  %arg162.163 = f32[512]{0} parameter(162), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.382 = f32[512]{0} reshape(f32[512]{0} %arg162.163)
  %arg119.120 = f32[512]{0} parameter(119), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.339 = f32[512]{0} reshape(f32[512]{0} %arg119.120)
  %multiply.1962 = f32[512]{0} multiply(f32[512]{0} %multiply.1961, f32[512]{0} %reshape.339), metadata={op_type="Mul" op_name="stage3_unit4_bn2/mul_2"}
  %subtract.1963 = f32[512]{0} subtract(f32[512]{0} %reshape.382, f32[512]{0} %multiply.1962), metadata={op_type="Sub" op_name="stage3_unit4_bn2/sub"}
  %broadcast.2080 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.1963), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit4_bn2/add_1"}
  %add.2081 = f32[1,14,14,512]{3,2,1,0} add(f32[1,14,14,512]{3,2,1,0} %multiply.2079, f32[1,14,14,512]{3,2,1,0} %broadcast.2080), metadata={op_type="AddV2" op_name="stage3_unit4_bn2/add_1"}
  %maximum.2084 = f32[1,14,14,512]{3,2,1,0} maximum(f32[1,14,14,512]{3,2,1,0} %broadcast.2083, f32[1,14,14,512]{3,2,1,0} %add.2081), metadata={op_type="Relu" op_name="stage3_unit4_relu2"}
  %arg213.214 = f32[1,1,512,1024]{3,2,1,0} parameter(213), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.433 = f32[1,1,512,1024]{3,2,1,0} reshape(f32[1,1,512,1024]{3,2,1,0} %arg213.214)
  %convolution.2085 = f32[1,14,14,1024]{3,2,1,0} convolution(f32[1,14,14,512]{3,2,1,0} %maximum.2084, f32[1,1,512,1024]{3,2,1,0} %reshape.433), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv3"}
  %multiply.2087 = f32[1,14,14,1024]{3,2,1,0} multiply(f32[1,14,14,1024]{3,2,1,0} %broadcast.2086, f32[1,14,14,1024]{3,2,1,0} %convolution.2085), metadata={op_type="Mul" op_name="stage3_unit4_bn3/mul_1"}
  %arg166.167 = f32[1024]{0} parameter(166), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.386 = f32[1024]{0} reshape(f32[1024]{0} %arg166.167)
  %arg123.124 = f32[1024]{0} parameter(123), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.343 = f32[1024]{0} reshape(f32[1024]{0} %arg123.124)
  %multiply.1969 = f32[1024]{0} multiply(f32[1024]{0} %multiply.1968, f32[1024]{0} %reshape.343), metadata={op_type="Mul" op_name="stage3_unit4_bn3/mul_2"}
  %subtract.1970 = f32[1024]{0} subtract(f32[1024]{0} %reshape.386, f32[1024]{0} %multiply.1969), metadata={op_type="Sub" op_name="stage3_unit4_bn3/sub"}
  %broadcast.2088 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %subtract.1970), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit4_bn3/add_1"}
  %add.2089 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %multiply.2087, f32[1,14,14,1024]{3,2,1,0} %broadcast.2088), metadata={op_type="AddV2" op_name="stage3_unit4_bn3/add_1"}
  %add.2090 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %maximum.1949, f32[1,14,14,1024]{3,2,1,0} %add.2089), metadata={op_type="AddV2" op_name="add_10"}
  %maximum.2093 = f32[1,14,14,1024]{3,2,1,0} maximum(f32[1,14,14,1024]{3,2,1,0} %broadcast.2092, f32[1,14,14,1024]{3,2,1,0} %add.2090), metadata={op_type="Relu" op_name="stage3_unit4_relu"}
  %constant.2108 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit5_bn3/add"}
  %broadcast.2109 = f32[1024]{0} broadcast(f32[] %constant.2108), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit5_bn3/add"}
  %arg50.51 = f32[1024]{0} parameter(50), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.270 = f32[1024]{0} reshape(f32[1024]{0} %arg50.51)
  %add.2110 = f32[1024]{0} add(f32[1024]{0} %broadcast.2109, f32[1024]{0} %reshape.270), metadata={op_type="AddV2" op_name="stage3_unit5_bn3/add"}
  %rsqrt.2111 = f32[1024]{0} rsqrt(f32[1024]{0} %add.2110), metadata={op_type="Rsqrt" op_name="stage3_unit5_bn3/Rsqrt"}
  %arg96.97 = f32[1024]{0} parameter(96), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.316 = f32[1024]{0} reshape(f32[1024]{0} %arg96.97)
  %multiply.2112 = f32[1024]{0} multiply(f32[1024]{0} %rsqrt.2111, f32[1024]{0} %reshape.316), metadata={op_type="Mul" op_name="stage3_unit5_bn3/mul"}
  %broadcast.2230 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %multiply.2112), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit5_bn3/mul_1"}
  %constant.2226 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit5_relu2"}
  %broadcast.2227 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[] %constant.2226), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit5_relu2"}
  %constant.2120 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit5_relu1"}
  %broadcast.2121 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[] %constant.2120), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit5_relu1"}
  %constant.2094 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit5_bn1/add"}
  %broadcast.2095 = f32[512]{0} broadcast(f32[] %constant.2094), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit5_bn1/add"}
  %arg34.35 = f32[512]{0} parameter(34), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.254 = f32[512]{0} reshape(f32[512]{0} %arg34.35)
  %add.2096 = f32[512]{0} add(f32[512]{0} %broadcast.2095, f32[512]{0} %reshape.254), metadata={op_type="AddV2" op_name="stage3_unit5_bn1/add"}
  %rsqrt.2097 = f32[512]{0} rsqrt(f32[512]{0} %add.2096), metadata={op_type="Rsqrt" op_name="stage3_unit5_bn1/Rsqrt"}
  %arg83.84 = f32[512]{0} parameter(83), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.303 = f32[512]{0} reshape(f32[512]{0} %arg83.84)
  %multiply.2098 = f32[512]{0} multiply(f32[512]{0} %rsqrt.2097, f32[512]{0} %reshape.303), metadata={op_type="Mul" op_name="stage3_unit5_bn1/mul"}
  %broadcast.2116 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.2098), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit5_bn1/mul_1"}
  %arg214.215 = f32[1,1,1024,512]{3,2,1,0} parameter(214), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.434 = f32[1,1,1024,512]{3,2,1,0} reshape(f32[1,1,1024,512]{3,2,1,0} %arg214.215)
  %convolution.2115 = f32[1,14,14,512]{3,2,1,0} convolution(f32[1,14,14,1024]{3,2,1,0} %maximum.2093, f32[1,1,1024,512]{3,2,1,0} %reshape.434), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv1"}
  %multiply.2117 = f32[1,14,14,512]{3,2,1,0} multiply(f32[1,14,14,512]{3,2,1,0} %broadcast.2116, f32[1,14,14,512]{3,2,1,0} %convolution.2115), metadata={op_type="Mul" op_name="stage3_unit5_bn1/mul_1"}
  %arg170.171 = f32[512]{0} parameter(170), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.390 = f32[512]{0} reshape(f32[512]{0} %arg170.171)
  %arg127.128 = f32[512]{0} parameter(127), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.347 = f32[512]{0} reshape(f32[512]{0} %arg127.128)
  %multiply.2099 = f32[512]{0} multiply(f32[512]{0} %multiply.2098, f32[512]{0} %reshape.347), metadata={op_type="Mul" op_name="stage3_unit5_bn1/mul_2"}
  %subtract.2100 = f32[512]{0} subtract(f32[512]{0} %reshape.390, f32[512]{0} %multiply.2099), metadata={op_type="Sub" op_name="stage3_unit5_bn1/sub"}
  %broadcast.2118 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.2100), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit5_bn1/add_1"}
  %add.2119 = f32[1,14,14,512]{3,2,1,0} add(f32[1,14,14,512]{3,2,1,0} %multiply.2117, f32[1,14,14,512]{3,2,1,0} %broadcast.2118), metadata={op_type="AddV2" op_name="stage3_unit5_bn1/add_1"}
  %maximum.2122 = f32[1,14,14,512]{3,2,1,0} maximum(f32[1,14,14,512]{3,2,1,0} %broadcast.2121, f32[1,14,14,512]{3,2,1,0} %add.2119), metadata={op_type="Relu" op_name="stage3_unit5_relu1"}
  %constant.2123 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_12"}
  %pad.2124 = f32[1,16,16,512]{3,2,1,0} pad(f32[1,14,14,512]{3,2,1,0} %maximum.2122, f32[] %constant.2123), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_12"}
  %slice.2125 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2124), slice={[0:1], [0:16], [0:16], [0:16]}, metadata={op_type="Split" op_name="split_23"}
  %arg37.38 = f32[3,3,16,512]{3,2,1,0} parameter(37), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.257 = f32[3,3,16,512]{3,2,1,0} reshape(f32[3,3,16,512]{3,2,1,0} %arg37.38)
  %slice.2157 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.257), slice={[0:3], [0:3], [0:16], [0:16]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2189 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2125, f32[3,3,16,16]{3,2,1,0} %slice.2157), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2"}
  %slice.2126 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2124), slice={[0:1], [0:16], [0:16], [16:32]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2158 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.257), slice={[0:3], [0:3], [0:16], [16:32]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2190 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2126, f32[3,3,16,16]{3,2,1,0} %slice.2158), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_1"}
  %slice.2127 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2124), slice={[0:1], [0:16], [0:16], [32:48]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2159 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.257), slice={[0:3], [0:3], [0:16], [32:48]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2201 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2127, f32[3,3,16,16]{3,2,1,0} %slice.2159), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_2"}
  %slice.2128 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2124), slice={[0:1], [0:16], [0:16], [48:64]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2160 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.257), slice={[0:3], [0:3], [0:16], [48:64]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2212 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2128, f32[3,3,16,16]{3,2,1,0} %slice.2160), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_3"}
  %slice.2129 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2124), slice={[0:1], [0:16], [0:16], [64:80]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2161 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.257), slice={[0:3], [0:3], [0:16], [64:80]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2215 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2129, f32[3,3,16,16]{3,2,1,0} %slice.2161), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_4"}
  %slice.2130 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2124), slice={[0:1], [0:16], [0:16], [80:96]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2162 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.257), slice={[0:3], [0:3], [0:16], [80:96]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2216 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2130, f32[3,3,16,16]{3,2,1,0} %slice.2162), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_5"}
  %slice.2131 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2124), slice={[0:1], [0:16], [0:16], [96:112]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2163 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.257), slice={[0:3], [0:3], [0:16], [96:112]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2217 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2131, f32[3,3,16,16]{3,2,1,0} %slice.2163), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_6"}
  %slice.2132 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2124), slice={[0:1], [0:16], [0:16], [112:128]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2164 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.257), slice={[0:3], [0:3], [0:16], [112:128]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2218 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2132, f32[3,3,16,16]{3,2,1,0} %slice.2164), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_7"}
  %slice.2133 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2124), slice={[0:1], [0:16], [0:16], [128:144]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2165 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.257), slice={[0:3], [0:3], [0:16], [128:144]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2219 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2133, f32[3,3,16,16]{3,2,1,0} %slice.2165), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_8"}
  %slice.2134 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2124), slice={[0:1], [0:16], [0:16], [144:160]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2166 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.257), slice={[0:3], [0:3], [0:16], [144:160]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2220 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2134, f32[3,3,16,16]{3,2,1,0} %slice.2166), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_9"}
  %slice.2135 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2124), slice={[0:1], [0:16], [0:16], [160:176]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2167 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.257), slice={[0:3], [0:3], [0:16], [160:176]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2191 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2135, f32[3,3,16,16]{3,2,1,0} %slice.2167), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_10"}
  %slice.2136 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2124), slice={[0:1], [0:16], [0:16], [176:192]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2168 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.257), slice={[0:3], [0:3], [0:16], [176:192]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2192 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2136, f32[3,3,16,16]{3,2,1,0} %slice.2168), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_11"}
  %slice.2137 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2124), slice={[0:1], [0:16], [0:16], [192:208]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2169 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.257), slice={[0:3], [0:3], [0:16], [192:208]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2193 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2137, f32[3,3,16,16]{3,2,1,0} %slice.2169), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_12"}
  %slice.2138 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2124), slice={[0:1], [0:16], [0:16], [208:224]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2170 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.257), slice={[0:3], [0:3], [0:16], [208:224]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2194 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2138, f32[3,3,16,16]{3,2,1,0} %slice.2170), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_13"}
  %slice.2139 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2124), slice={[0:1], [0:16], [0:16], [224:240]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2171 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.257), slice={[0:3], [0:3], [0:16], [224:240]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2195 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2139, f32[3,3,16,16]{3,2,1,0} %slice.2171), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_14"}
  %slice.2140 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2124), slice={[0:1], [0:16], [0:16], [240:256]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2172 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.257), slice={[0:3], [0:3], [0:16], [240:256]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2196 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2140, f32[3,3,16,16]{3,2,1,0} %slice.2172), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_15"}
  %slice.2141 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2124), slice={[0:1], [0:16], [0:16], [256:272]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2173 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.257), slice={[0:3], [0:3], [0:16], [256:272]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2197 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2141, f32[3,3,16,16]{3,2,1,0} %slice.2173), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_16"}
  %slice.2142 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2124), slice={[0:1], [0:16], [0:16], [272:288]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2174 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.257), slice={[0:3], [0:3], [0:16], [272:288]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2198 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2142, f32[3,3,16,16]{3,2,1,0} %slice.2174), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_17"}
  %slice.2143 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2124), slice={[0:1], [0:16], [0:16], [288:304]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2175 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.257), slice={[0:3], [0:3], [0:16], [288:304]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2199 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2143, f32[3,3,16,16]{3,2,1,0} %slice.2175), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_18"}
  %slice.2144 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2124), slice={[0:1], [0:16], [0:16], [304:320]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2176 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.257), slice={[0:3], [0:3], [0:16], [304:320]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2200 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2144, f32[3,3,16,16]{3,2,1,0} %slice.2176), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_19"}
  %slice.2145 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2124), slice={[0:1], [0:16], [0:16], [320:336]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2177 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.257), slice={[0:3], [0:3], [0:16], [320:336]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2202 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2145, f32[3,3,16,16]{3,2,1,0} %slice.2177), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_20"}
  %slice.2146 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2124), slice={[0:1], [0:16], [0:16], [336:352]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2178 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.257), slice={[0:3], [0:3], [0:16], [336:352]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2203 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2146, f32[3,3,16,16]{3,2,1,0} %slice.2178), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_21"}
  %slice.2147 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2124), slice={[0:1], [0:16], [0:16], [352:368]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2179 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.257), slice={[0:3], [0:3], [0:16], [352:368]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2204 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2147, f32[3,3,16,16]{3,2,1,0} %slice.2179), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_22"}
  %slice.2148 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2124), slice={[0:1], [0:16], [0:16], [368:384]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2180 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.257), slice={[0:3], [0:3], [0:16], [368:384]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2205 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2148, f32[3,3,16,16]{3,2,1,0} %slice.2180), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_23"}
  %slice.2149 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2124), slice={[0:1], [0:16], [0:16], [384:400]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2181 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.257), slice={[0:3], [0:3], [0:16], [384:400]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2206 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2149, f32[3,3,16,16]{3,2,1,0} %slice.2181), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_24"}
  %slice.2150 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2124), slice={[0:1], [0:16], [0:16], [400:416]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2182 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.257), slice={[0:3], [0:3], [0:16], [400:416]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2207 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2150, f32[3,3,16,16]{3,2,1,0} %slice.2182), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_25"}
  %slice.2151 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2124), slice={[0:1], [0:16], [0:16], [416:432]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2183 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.257), slice={[0:3], [0:3], [0:16], [416:432]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2208 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2151, f32[3,3,16,16]{3,2,1,0} %slice.2183), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_26"}
  %slice.2152 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2124), slice={[0:1], [0:16], [0:16], [432:448]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2184 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.257), slice={[0:3], [0:3], [0:16], [432:448]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2209 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2152, f32[3,3,16,16]{3,2,1,0} %slice.2184), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_27"}
  %slice.2153 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2124), slice={[0:1], [0:16], [0:16], [448:464]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2185 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.257), slice={[0:3], [0:3], [0:16], [448:464]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2210 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2153, f32[3,3,16,16]{3,2,1,0} %slice.2185), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_28"}
  %slice.2154 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2124), slice={[0:1], [0:16], [0:16], [464:480]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2186 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.257), slice={[0:3], [0:3], [0:16], [464:480]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2211 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2154, f32[3,3,16,16]{3,2,1,0} %slice.2186), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_29"}
  %slice.2155 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2124), slice={[0:1], [0:16], [0:16], [480:496]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2187 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.257), slice={[0:3], [0:3], [0:16], [480:496]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2213 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2155, f32[3,3,16,16]{3,2,1,0} %slice.2187), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_30"}
  %slice.2156 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2124), slice={[0:1], [0:16], [0:16], [496:512]}, metadata={op_type="Split" op_name="split_23"}
  %slice.2188 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.257), slice={[0:3], [0:3], [0:16], [496:512]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2214 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2156, f32[3,3,16,16]{3,2,1,0} %slice.2188), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_31"}
  %concatenate.2221 = f32[1,14,14,512]{3,2,1,0} concatenate(f32[1,14,14,16]{3,2,1,0} %convolution.2189, f32[1,14,14,16]{3,2,1,0} %convolution.2190, f32[1,14,14,16]{3,2,1,0} %convolution.2201, f32[1,14,14,16]{3,2,1,0} %convolution.2212, f32[1,14,14,16]{3,2,1,0} %convolution.2215, f32[1,14,14,16]{3,2,1,0} %convolution.2216, f32[1,14,14,16]{3,2,1,0} %convolution.2217, f32[1,14,14,16]{3,2,1,0} %convolution.2218, f32[1,14,14,16]{3,2,1,0} %convolution.2219, f32[1,14,14,16]{3,2,1,0} %convolution.2220, f32[1,14,14,16]{3,2,1,0} %convolution.2191, f32[1,14,14,16]{3,2,1,0} %convolution.2192, f32[1,14,14,16]{3,2,1,0} %convolution.2193, f32[1,14,14,16]{3,2,1,0} %convolution.2194, f32[1,14,14,16]{3,2,1,0} %convolution.2195, f32[1,14,14,16]{3,2,1,0} %convolution.2196, f32[1,14,14,16]{3,2,1,0} %convolution.2197, f32[1,14,14,16]{3,2,1,0} %convolution.2198, f32[1,14,14,16]{3,2,1,0} %convolution.2199, f32[1,14,14,16]{3,2,1,0} %convolution.2200, f32[1,14,14,16]{3,2,1,0} %convolution.2202, f32[1,14,14,16]{3,2,1,0} %convolution.2203, f32[1,14,14,16]{3,2,1,0} %convolution.2204, f32[1,14,14,16]{3,2,1,0} %convolution.2205, f32[1,14,14,16]{3,2,1,0} %convolution.2206, f32[1,14,14,16]{3,2,1,0} %convolution.2207, f32[1,14,14,16]{3,2,1,0} %convolution.2208, f32[1,14,14,16]{3,2,1,0} %convolution.2209, f32[1,14,14,16]{3,2,1,0} %convolution.2210, f32[1,14,14,16]{3,2,1,0} %convolution.2211, f32[1,14,14,16]{3,2,1,0} %convolution.2213, f32[1,14,14,16]{3,2,1,0} %convolution.2214), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_11"}
  %constant.2101 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit5_bn2/add"}
  %broadcast.2102 = f32[512]{0} broadcast(f32[] %constant.2101), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit5_bn2/add"}
  %arg45.46 = f32[512]{0} parameter(45), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.265 = f32[512]{0} reshape(f32[512]{0} %arg45.46)
  %add.2103 = f32[512]{0} add(f32[512]{0} %broadcast.2102, f32[512]{0} %reshape.265), metadata={op_type="AddV2" op_name="stage3_unit5_bn2/add"}
  %rsqrt.2104 = f32[512]{0} rsqrt(f32[512]{0} %add.2103), metadata={op_type="Rsqrt" op_name="stage3_unit5_bn2/Rsqrt"}
  %arg92.93 = f32[512]{0} parameter(92), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.312 = f32[512]{0} reshape(f32[512]{0} %arg92.93)
  %multiply.2105 = f32[512]{0} multiply(f32[512]{0} %rsqrt.2104, f32[512]{0} %reshape.312), metadata={op_type="Mul" op_name="stage3_unit5_bn2/mul"}
  %broadcast.2222 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.2105), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit5_bn2/mul_1"}
  %multiply.2223 = f32[1,14,14,512]{3,2,1,0} multiply(f32[1,14,14,512]{3,2,1,0} %concatenate.2221, f32[1,14,14,512]{3,2,1,0} %broadcast.2222), metadata={op_type="Mul" op_name="stage3_unit5_bn2/mul_1"}
  %arg179.180 = f32[512]{0} parameter(179), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.399 = f32[512]{0} reshape(f32[512]{0} %arg179.180)
  %arg136.137 = f32[512]{0} parameter(136), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.356 = f32[512]{0} reshape(f32[512]{0} %arg136.137)
  %multiply.2106 = f32[512]{0} multiply(f32[512]{0} %multiply.2105, f32[512]{0} %reshape.356), metadata={op_type="Mul" op_name="stage3_unit5_bn2/mul_2"}
  %subtract.2107 = f32[512]{0} subtract(f32[512]{0} %reshape.399, f32[512]{0} %multiply.2106), metadata={op_type="Sub" op_name="stage3_unit5_bn2/sub"}
  %broadcast.2224 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.2107), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit5_bn2/add_1"}
  %add.2225 = f32[1,14,14,512]{3,2,1,0} add(f32[1,14,14,512]{3,2,1,0} %multiply.2223, f32[1,14,14,512]{3,2,1,0} %broadcast.2224), metadata={op_type="AddV2" op_name="stage3_unit5_bn2/add_1"}
  %maximum.2228 = f32[1,14,14,512]{3,2,1,0} maximum(f32[1,14,14,512]{3,2,1,0} %broadcast.2227, f32[1,14,14,512]{3,2,1,0} %add.2225), metadata={op_type="Relu" op_name="stage3_unit5_relu2"}
  %arg215.216 = f32[1,1,512,1024]{3,2,1,0} parameter(215), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.435 = f32[1,1,512,1024]{3,2,1,0} reshape(f32[1,1,512,1024]{3,2,1,0} %arg215.216)
  %convolution.2229 = f32[1,14,14,1024]{3,2,1,0} convolution(f32[1,14,14,512]{3,2,1,0} %maximum.2228, f32[1,1,512,1024]{3,2,1,0} %reshape.435), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv3"}
  %multiply.2231 = f32[1,14,14,1024]{3,2,1,0} multiply(f32[1,14,14,1024]{3,2,1,0} %broadcast.2230, f32[1,14,14,1024]{3,2,1,0} %convolution.2229), metadata={op_type="Mul" op_name="stage3_unit5_bn3/mul_1"}
  %arg183.184 = f32[1024]{0} parameter(183), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.403 = f32[1024]{0} reshape(f32[1024]{0} %arg183.184)
  %arg140.141 = f32[1024]{0} parameter(140), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.360 = f32[1024]{0} reshape(f32[1024]{0} %arg140.141)
  %multiply.2113 = f32[1024]{0} multiply(f32[1024]{0} %multiply.2112, f32[1024]{0} %reshape.360), metadata={op_type="Mul" op_name="stage3_unit5_bn3/mul_2"}
  %subtract.2114 = f32[1024]{0} subtract(f32[1024]{0} %reshape.403, f32[1024]{0} %multiply.2113), metadata={op_type="Sub" op_name="stage3_unit5_bn3/sub"}
  %broadcast.2232 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %subtract.2114), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit5_bn3/add_1"}
  %add.2233 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %multiply.2231, f32[1,14,14,1024]{3,2,1,0} %broadcast.2232), metadata={op_type="AddV2" op_name="stage3_unit5_bn3/add_1"}
  %add.2234 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %maximum.2093, f32[1,14,14,1024]{3,2,1,0} %add.2233), metadata={op_type="AddV2" op_name="add_11"}
  %maximum.2237 = f32[1,14,14,1024]{3,2,1,0} maximum(f32[1,14,14,1024]{3,2,1,0} %broadcast.2236, f32[1,14,14,1024]{3,2,1,0} %add.2234), metadata={op_type="Relu" op_name="stage3_unit5_relu"}
  %constant.2252 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit6_bn3/add"}
  %broadcast.2253 = f32[1024]{0} broadcast(f32[] %constant.2252), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit6_bn3/add"}
  %arg9.10 = f32[1024]{0} parameter(9), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.229 = f32[1024]{0} reshape(f32[1024]{0} %arg9.10)
  %add.2254 = f32[1024]{0} add(f32[1024]{0} %broadcast.2253, f32[1024]{0} %reshape.229), metadata={op_type="AddV2" op_name="stage3_unit6_bn3/add"}
  %rsqrt.2255 = f32[1024]{0} rsqrt(f32[1024]{0} %add.2254), metadata={op_type="Rsqrt" op_name="stage3_unit6_bn3/Rsqrt"}
  %arg64.65 = f32[1024]{0} parameter(64), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.284 = f32[1024]{0} reshape(f32[1024]{0} %arg64.65)
  %multiply.2256 = f32[1024]{0} multiply(f32[1024]{0} %rsqrt.2255, f32[1024]{0} %reshape.284), metadata={op_type="Mul" op_name="stage3_unit6_bn3/mul"}
  %broadcast.2374 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %multiply.2256), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit6_bn3/mul_1"}
  %constant.2370 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit6_relu2"}
  %broadcast.2371 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[] %constant.2370), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit6_relu2"}
  %constant.2264 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit6_relu1"}
  %broadcast.2265 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[] %constant.2264), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit6_relu1"}
  %constant.2238 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit6_bn1/add"}
  %broadcast.2239 = f32[512]{0} broadcast(f32[] %constant.2238), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit6_bn1/add"}
  %arg55.56 = f32[512]{0} parameter(55), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.275 = f32[512]{0} reshape(f32[512]{0} %arg55.56)
  %add.2240 = f32[512]{0} add(f32[512]{0} %broadcast.2239, f32[512]{0} %reshape.275), metadata={op_type="AddV2" op_name="stage3_unit6_bn1/add"}
  %rsqrt.2241 = f32[512]{0} rsqrt(f32[512]{0} %add.2240), metadata={op_type="Rsqrt" op_name="stage3_unit6_bn1/Rsqrt"}
  %arg100.101 = f32[512]{0} parameter(100), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.320 = f32[512]{0} reshape(f32[512]{0} %arg100.101)
  %multiply.2242 = f32[512]{0} multiply(f32[512]{0} %rsqrt.2241, f32[512]{0} %reshape.320), metadata={op_type="Mul" op_name="stage3_unit6_bn1/mul"}
  %broadcast.2260 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.2242), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit6_bn1/mul_1"}
  %arg216.217 = f32[1,1,1024,512]{3,2,1,0} parameter(216), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.436 = f32[1,1,1024,512]{3,2,1,0} reshape(f32[1,1,1024,512]{3,2,1,0} %arg216.217)
  %convolution.2259 = f32[1,14,14,512]{3,2,1,0} convolution(f32[1,14,14,1024]{3,2,1,0} %maximum.2237, f32[1,1,1024,512]{3,2,1,0} %reshape.436), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv1"}
  %multiply.2261 = f32[1,14,14,512]{3,2,1,0} multiply(f32[1,14,14,512]{3,2,1,0} %broadcast.2260, f32[1,14,14,512]{3,2,1,0} %convolution.2259), metadata={op_type="Mul" op_name="stage3_unit6_bn1/mul_1"}
  %arg187.188 = f32[512]{0} parameter(187), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.407 = f32[512]{0} reshape(f32[512]{0} %arg187.188)
  %arg144.145 = f32[512]{0} parameter(144), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.364 = f32[512]{0} reshape(f32[512]{0} %arg144.145)
  %multiply.2243 = f32[512]{0} multiply(f32[512]{0} %multiply.2242, f32[512]{0} %reshape.364), metadata={op_type="Mul" op_name="stage3_unit6_bn1/mul_2"}
  %subtract.2244 = f32[512]{0} subtract(f32[512]{0} %reshape.407, f32[512]{0} %multiply.2243), metadata={op_type="Sub" op_name="stage3_unit6_bn1/sub"}
  %broadcast.2262 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.2244), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit6_bn1/add_1"}
  %add.2263 = f32[1,14,14,512]{3,2,1,0} add(f32[1,14,14,512]{3,2,1,0} %multiply.2261, f32[1,14,14,512]{3,2,1,0} %broadcast.2262), metadata={op_type="AddV2" op_name="stage3_unit6_bn1/add_1"}
  %maximum.2266 = f32[1,14,14,512]{3,2,1,0} maximum(f32[1,14,14,512]{3,2,1,0} %broadcast.2265, f32[1,14,14,512]{3,2,1,0} %add.2263), metadata={op_type="Relu" op_name="stage3_unit6_relu1"}
  %constant.2267 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_13"}
  %pad.2268 = f32[1,16,16,512]{3,2,1,0} pad(f32[1,14,14,512]{3,2,1,0} %maximum.2266, f32[] %constant.2267), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_13"}
  %slice.2269 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2268), slice={[0:1], [0:16], [0:16], [0:16]}, metadata={op_type="Split" op_name="split_25"}
  %arg8.9 = f32[3,3,16,512]{3,2,1,0} parameter(8), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.228 = f32[3,3,16,512]{3,2,1,0} reshape(f32[3,3,16,512]{3,2,1,0} %arg8.9)
  %slice.2301 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.228), slice={[0:3], [0:3], [0:16], [0:16]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2333 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2269, f32[3,3,16,16]{3,2,1,0} %slice.2301), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2"}
  %slice.2270 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2268), slice={[0:1], [0:16], [0:16], [16:32]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2302 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.228), slice={[0:3], [0:3], [0:16], [16:32]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2334 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2270, f32[3,3,16,16]{3,2,1,0} %slice.2302), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_1"}
  %slice.2271 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2268), slice={[0:1], [0:16], [0:16], [32:48]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2303 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.228), slice={[0:3], [0:3], [0:16], [32:48]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2345 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2271, f32[3,3,16,16]{3,2,1,0} %slice.2303), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_2"}
  %slice.2272 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2268), slice={[0:1], [0:16], [0:16], [48:64]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2304 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.228), slice={[0:3], [0:3], [0:16], [48:64]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2356 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2272, f32[3,3,16,16]{3,2,1,0} %slice.2304), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_3"}
  %slice.2273 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2268), slice={[0:1], [0:16], [0:16], [64:80]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2305 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.228), slice={[0:3], [0:3], [0:16], [64:80]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2359 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2273, f32[3,3,16,16]{3,2,1,0} %slice.2305), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_4"}
  %slice.2274 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2268), slice={[0:1], [0:16], [0:16], [80:96]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2306 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.228), slice={[0:3], [0:3], [0:16], [80:96]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2360 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2274, f32[3,3,16,16]{3,2,1,0} %slice.2306), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_5"}
  %slice.2275 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2268), slice={[0:1], [0:16], [0:16], [96:112]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2307 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.228), slice={[0:3], [0:3], [0:16], [96:112]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2361 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2275, f32[3,3,16,16]{3,2,1,0} %slice.2307), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_6"}
  %slice.2276 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2268), slice={[0:1], [0:16], [0:16], [112:128]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2308 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.228), slice={[0:3], [0:3], [0:16], [112:128]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2362 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2276, f32[3,3,16,16]{3,2,1,0} %slice.2308), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_7"}
  %slice.2277 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2268), slice={[0:1], [0:16], [0:16], [128:144]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2309 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.228), slice={[0:3], [0:3], [0:16], [128:144]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2363 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2277, f32[3,3,16,16]{3,2,1,0} %slice.2309), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_8"}
  %slice.2278 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2268), slice={[0:1], [0:16], [0:16], [144:160]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2310 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.228), slice={[0:3], [0:3], [0:16], [144:160]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2364 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2278, f32[3,3,16,16]{3,2,1,0} %slice.2310), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_9"}
  %slice.2279 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2268), slice={[0:1], [0:16], [0:16], [160:176]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2311 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.228), slice={[0:3], [0:3], [0:16], [160:176]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2335 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2279, f32[3,3,16,16]{3,2,1,0} %slice.2311), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_10"}
  %slice.2280 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2268), slice={[0:1], [0:16], [0:16], [176:192]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2312 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.228), slice={[0:3], [0:3], [0:16], [176:192]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2336 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2280, f32[3,3,16,16]{3,2,1,0} %slice.2312), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_11"}
  %slice.2281 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2268), slice={[0:1], [0:16], [0:16], [192:208]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2313 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.228), slice={[0:3], [0:3], [0:16], [192:208]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2337 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2281, f32[3,3,16,16]{3,2,1,0} %slice.2313), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_12"}
  %slice.2282 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2268), slice={[0:1], [0:16], [0:16], [208:224]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2314 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.228), slice={[0:3], [0:3], [0:16], [208:224]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2338 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2282, f32[3,3,16,16]{3,2,1,0} %slice.2314), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_13"}
  %slice.2283 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2268), slice={[0:1], [0:16], [0:16], [224:240]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2315 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.228), slice={[0:3], [0:3], [0:16], [224:240]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2339 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2283, f32[3,3,16,16]{3,2,1,0} %slice.2315), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_14"}
  %slice.2284 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2268), slice={[0:1], [0:16], [0:16], [240:256]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2316 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.228), slice={[0:3], [0:3], [0:16], [240:256]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2340 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2284, f32[3,3,16,16]{3,2,1,0} %slice.2316), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_15"}
  %slice.2285 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2268), slice={[0:1], [0:16], [0:16], [256:272]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2317 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.228), slice={[0:3], [0:3], [0:16], [256:272]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2341 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2285, f32[3,3,16,16]{3,2,1,0} %slice.2317), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_16"}
  %slice.2286 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2268), slice={[0:1], [0:16], [0:16], [272:288]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2318 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.228), slice={[0:3], [0:3], [0:16], [272:288]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2342 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2286, f32[3,3,16,16]{3,2,1,0} %slice.2318), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_17"}
  %slice.2287 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2268), slice={[0:1], [0:16], [0:16], [288:304]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2319 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.228), slice={[0:3], [0:3], [0:16], [288:304]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2343 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2287, f32[3,3,16,16]{3,2,1,0} %slice.2319), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_18"}
  %slice.2288 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2268), slice={[0:1], [0:16], [0:16], [304:320]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2320 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.228), slice={[0:3], [0:3], [0:16], [304:320]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2344 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2288, f32[3,3,16,16]{3,2,1,0} %slice.2320), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_19"}
  %slice.2289 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2268), slice={[0:1], [0:16], [0:16], [320:336]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2321 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.228), slice={[0:3], [0:3], [0:16], [320:336]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2346 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2289, f32[3,3,16,16]{3,2,1,0} %slice.2321), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_20"}
  %slice.2290 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2268), slice={[0:1], [0:16], [0:16], [336:352]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2322 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.228), slice={[0:3], [0:3], [0:16], [336:352]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2347 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2290, f32[3,3,16,16]{3,2,1,0} %slice.2322), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_21"}
  %slice.2291 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2268), slice={[0:1], [0:16], [0:16], [352:368]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2323 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.228), slice={[0:3], [0:3], [0:16], [352:368]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2348 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2291, f32[3,3,16,16]{3,2,1,0} %slice.2323), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_22"}
  %slice.2292 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2268), slice={[0:1], [0:16], [0:16], [368:384]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2324 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.228), slice={[0:3], [0:3], [0:16], [368:384]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2349 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2292, f32[3,3,16,16]{3,2,1,0} %slice.2324), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_23"}
  %slice.2293 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2268), slice={[0:1], [0:16], [0:16], [384:400]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2325 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.228), slice={[0:3], [0:3], [0:16], [384:400]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2350 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2293, f32[3,3,16,16]{3,2,1,0} %slice.2325), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_24"}
  %slice.2294 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2268), slice={[0:1], [0:16], [0:16], [400:416]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2326 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.228), slice={[0:3], [0:3], [0:16], [400:416]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2351 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2294, f32[3,3,16,16]{3,2,1,0} %slice.2326), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_25"}
  %slice.2295 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2268), slice={[0:1], [0:16], [0:16], [416:432]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2327 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.228), slice={[0:3], [0:3], [0:16], [416:432]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2352 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2295, f32[3,3,16,16]{3,2,1,0} %slice.2327), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_26"}
  %slice.2296 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2268), slice={[0:1], [0:16], [0:16], [432:448]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2328 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.228), slice={[0:3], [0:3], [0:16], [432:448]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2353 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2296, f32[3,3,16,16]{3,2,1,0} %slice.2328), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_27"}
  %slice.2297 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2268), slice={[0:1], [0:16], [0:16], [448:464]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2329 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.228), slice={[0:3], [0:3], [0:16], [448:464]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2354 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2297, f32[3,3,16,16]{3,2,1,0} %slice.2329), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_28"}
  %slice.2298 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2268), slice={[0:1], [0:16], [0:16], [464:480]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2330 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.228), slice={[0:3], [0:3], [0:16], [464:480]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2355 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2298, f32[3,3,16,16]{3,2,1,0} %slice.2330), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_29"}
  %slice.2299 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2268), slice={[0:1], [0:16], [0:16], [480:496]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2331 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.228), slice={[0:3], [0:3], [0:16], [480:496]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2357 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2299, f32[3,3,16,16]{3,2,1,0} %slice.2331), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_30"}
  %slice.2300 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2268), slice={[0:1], [0:16], [0:16], [496:512]}, metadata={op_type="Split" op_name="split_25"}
  %slice.2332 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.228), slice={[0:3], [0:3], [0:16], [496:512]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2358 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2300, f32[3,3,16,16]{3,2,1,0} %slice.2332), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_31"}
  %concatenate.2365 = f32[1,14,14,512]{3,2,1,0} concatenate(f32[1,14,14,16]{3,2,1,0} %convolution.2333, f32[1,14,14,16]{3,2,1,0} %convolution.2334, f32[1,14,14,16]{3,2,1,0} %convolution.2345, f32[1,14,14,16]{3,2,1,0} %convolution.2356, f32[1,14,14,16]{3,2,1,0} %convolution.2359, f32[1,14,14,16]{3,2,1,0} %convolution.2360, f32[1,14,14,16]{3,2,1,0} %convolution.2361, f32[1,14,14,16]{3,2,1,0} %convolution.2362, f32[1,14,14,16]{3,2,1,0} %convolution.2363, f32[1,14,14,16]{3,2,1,0} %convolution.2364, f32[1,14,14,16]{3,2,1,0} %convolution.2335, f32[1,14,14,16]{3,2,1,0} %convolution.2336, f32[1,14,14,16]{3,2,1,0} %convolution.2337, f32[1,14,14,16]{3,2,1,0} %convolution.2338, f32[1,14,14,16]{3,2,1,0} %convolution.2339, f32[1,14,14,16]{3,2,1,0} %convolution.2340, f32[1,14,14,16]{3,2,1,0} %convolution.2341, f32[1,14,14,16]{3,2,1,0} %convolution.2342, f32[1,14,14,16]{3,2,1,0} %convolution.2343, f32[1,14,14,16]{3,2,1,0} %convolution.2344, f32[1,14,14,16]{3,2,1,0} %convolution.2346, f32[1,14,14,16]{3,2,1,0} %convolution.2347, f32[1,14,14,16]{3,2,1,0} %convolution.2348, f32[1,14,14,16]{3,2,1,0} %convolution.2349, f32[1,14,14,16]{3,2,1,0} %convolution.2350, f32[1,14,14,16]{3,2,1,0} %convolution.2351, f32[1,14,14,16]{3,2,1,0} %convolution.2352, f32[1,14,14,16]{3,2,1,0} %convolution.2353, f32[1,14,14,16]{3,2,1,0} %convolution.2354, f32[1,14,14,16]{3,2,1,0} %convolution.2355, f32[1,14,14,16]{3,2,1,0} %convolution.2357, f32[1,14,14,16]{3,2,1,0} %convolution.2358), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_12"}
  %constant.2245 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit6_bn2/add"}
  %broadcast.2246 = f32[512]{0} broadcast(f32[] %constant.2245), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit6_bn2/add"}
  %arg39.40 = f32[512]{0} parameter(39), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.259 = f32[512]{0} reshape(f32[512]{0} %arg39.40)
  %add.2247 = f32[512]{0} add(f32[512]{0} %broadcast.2246, f32[512]{0} %reshape.259), metadata={op_type="AddV2" op_name="stage3_unit6_bn2/add"}
  %rsqrt.2248 = f32[512]{0} rsqrt(f32[512]{0} %add.2247), metadata={op_type="Rsqrt" op_name="stage3_unit6_bn2/Rsqrt"}
  %arg87.88 = f32[512]{0} parameter(87), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.307 = f32[512]{0} reshape(f32[512]{0} %arg87.88)
  %multiply.2249 = f32[512]{0} multiply(f32[512]{0} %rsqrt.2248, f32[512]{0} %reshape.307), metadata={op_type="Mul" op_name="stage3_unit6_bn2/mul"}
  %broadcast.2366 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.2249), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit6_bn2/mul_1"}
  %multiply.2367 = f32[1,14,14,512]{3,2,1,0} multiply(f32[1,14,14,512]{3,2,1,0} %concatenate.2365, f32[1,14,14,512]{3,2,1,0} %broadcast.2366), metadata={op_type="Mul" op_name="stage3_unit6_bn2/mul_1"}
  %arg174.175 = f32[512]{0} parameter(174), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.394 = f32[512]{0} reshape(f32[512]{0} %arg174.175)
  %arg131.132 = f32[512]{0} parameter(131), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.351 = f32[512]{0} reshape(f32[512]{0} %arg131.132)
  %multiply.2250 = f32[512]{0} multiply(f32[512]{0} %multiply.2249, f32[512]{0} %reshape.351), metadata={op_type="Mul" op_name="stage3_unit6_bn2/mul_2"}
  %subtract.2251 = f32[512]{0} subtract(f32[512]{0} %reshape.394, f32[512]{0} %multiply.2250), metadata={op_type="Sub" op_name="stage3_unit6_bn2/sub"}
  %broadcast.2368 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.2251), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit6_bn2/add_1"}
  %add.2369 = f32[1,14,14,512]{3,2,1,0} add(f32[1,14,14,512]{3,2,1,0} %multiply.2367, f32[1,14,14,512]{3,2,1,0} %broadcast.2368), metadata={op_type="AddV2" op_name="stage3_unit6_bn2/add_1"}
  %maximum.2372 = f32[1,14,14,512]{3,2,1,0} maximum(f32[1,14,14,512]{3,2,1,0} %broadcast.2371, f32[1,14,14,512]{3,2,1,0} %add.2369), metadata={op_type="Relu" op_name="stage3_unit6_relu2"}
  %arg217.218 = f32[1,1,512,1024]{3,2,1,0} parameter(217), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.437 = f32[1,1,512,1024]{3,2,1,0} reshape(f32[1,1,512,1024]{3,2,1,0} %arg217.218)
  %convolution.2373 = f32[1,14,14,1024]{3,2,1,0} convolution(f32[1,14,14,512]{3,2,1,0} %maximum.2372, f32[1,1,512,1024]{3,2,1,0} %reshape.437), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv3"}
  %multiply.2375 = f32[1,14,14,1024]{3,2,1,0} multiply(f32[1,14,14,1024]{3,2,1,0} %broadcast.2374, f32[1,14,14,1024]{3,2,1,0} %convolution.2373), metadata={op_type="Mul" op_name="stage3_unit6_bn3/mul_1"}
  %arg151.152 = f32[1024]{0} parameter(151), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.371 = f32[1024]{0} reshape(f32[1024]{0} %arg151.152)
  %arg108.109 = f32[1024]{0} parameter(108), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.328 = f32[1024]{0} reshape(f32[1024]{0} %arg108.109)
  %multiply.2257 = f32[1024]{0} multiply(f32[1024]{0} %multiply.2256, f32[1024]{0} %reshape.328), metadata={op_type="Mul" op_name="stage3_unit6_bn3/mul_2"}
  %subtract.2258 = f32[1024]{0} subtract(f32[1024]{0} %reshape.371, f32[1024]{0} %multiply.2257), metadata={op_type="Sub" op_name="stage3_unit6_bn3/sub"}
  %broadcast.2376 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %subtract.2258), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit6_bn3/add_1"}
  %add.2377 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %multiply.2375, f32[1,14,14,1024]{3,2,1,0} %broadcast.2376), metadata={op_type="AddV2" op_name="stage3_unit6_bn3/add_1"}
  %add.2378 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %maximum.2237, f32[1,14,14,1024]{3,2,1,0} %add.2377), metadata={op_type="AddV2" op_name="add_12"}
  %maximum.2381 = f32[1,14,14,1024]{3,2,1,0} maximum(f32[1,14,14,1024]{3,2,1,0} %broadcast.2380, f32[1,14,14,1024]{3,2,1,0} %add.2378), metadata={op_type="Relu" op_name="stage3_unit6_relu"}
  %reshape.2382 = f32[1,14,14,1024]{3,2,1,0} reshape(f32[1,14,14,1024]{3,2,1,0} %maximum.2381), metadata={op_name="XLA_Retvals"}
  %tuple.2383 = (f32[1,14,14,1024]{3,2,1,0}) tuple(f32[1,14,14,1024]{3,2,1,0} %reshape.2382), metadata={op_name="XLA_Retvals"}
  ROOT %get-tuple-element.2384 = f32[1,14,14,1024]{3,2,1,0} get-tuple-element((f32[1,14,14,1024]{3,2,1,0}) %tuple.2383), index=0, metadata={op_name="XLA_Retvals"}
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
