// Tests that show HLO Module conversion to PlaidML Program.

#include <algorithm>
#include <string>

#include <gtest/gtest.h>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/service/plaidml/compiler.h"
#include "tensorflow/compiler/xla/service/plaidml/tests/plaidml_codegen_test.h"
#include "tensorflow/compiler/xla/service/plaidml/tests/i3d_pretrained_inputs_and_weights.h"
#include "tensorflow/compiler/xla/service/plaidml/tests/i3d_output.h"
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
  TestCaseVal I3D_WeightsInputs = {
    ::weights::RGB_inception_i3d_Logits_Conv3d_0c_1x1_conv_3d_w, // %arg0
    {98}, // %arg1: RGB/inception_i3d/Logits/AvgPool3D
    {0}, // %arg2: RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/Relu
    ::weights::RGB_inception_i3d_Mixed_5c_Branch_3_Conv3d_0b_1x1_conv_3d_w, // %arg3
    {0}, // %arg4: RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/Relu
    ::weights::RGB_inception_i3d_Mixed_5b_Branch_3_Conv3d_0b_1x1_conv_3d_w, // %arg5
    {0}, // %arg6: RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/Relu
    ::weights::RGB_inception_i3d_Mixed_4f_Branch_3_Conv3d_0b_1x1_conv_3d_w, // %arg7
    {0}, // %arg8: RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/Relu
    ::weights::RGB_inception_i3d_Mixed_4e_Branch_3_Conv3d_0b_1x1_conv_3d_w, // %arg9
    {0}, // %arg10: RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/Relu
    ::weights::RGB_inception_i3d_Mixed_4d_Branch_3_Conv3d_0b_1x1_conv_3d_w, // %arg11
    {0}, // %arg12: RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/Relu
    ::weights::RGB_inception_i3d_Mixed_4c_Branch_3_Conv3d_0b_1x1_conv_3d_w, // %arg13
    {0}, // %arg14: RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/Relu
    ::weights::RGB_inception_i3d_Mixed_4b_Branch_3_Conv3d_0b_1x1_conv_3d_w, // %arg15
    {0}, // %arg16: RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/Relu
    ::weights::RGB_inception_i3d_Mixed_3c_Branch_3_Conv3d_0b_1x1_conv_3d_w, // %arg17
    {0}, // %arg18: RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/Relu
    ::weights::RGB_inception_i3d_Mixed_3b_Branch_3_Conv3d_0b_1x1_conv_3d_w, // %arg19
    {0}, // %arg20: RGB/inception_i3d/Conv3d_2c_3x3/Relu
    ::weights::RGB_inception_i3d_Conv3d_2c_3x3_conv_3d_w, // %arg21
    {0}, // RGB/inception_i3d/Conv3d_2b_1x1/Relu
    ::weights::RGB_inception_i3d_Conv3d_2b_1x1_conv_3d_w, // %arg23
    {0}, // %arg24: RGB/inception_i3d/Conv3d_1a_7x7/Relu
    ::weights::RGB_inception_i3d_Conv3d_1a_7x7_conv_3d_w, // %arg25
    input_tensor, // %arg26
    {0.001}, // %arg27: RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Conv3d_1a_7x7_batch_norm_beta, // %arg28
    {0.001}, // %arg29: RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Conv3d_2b_1x1_batch_norm_beta, // %arg30
    {0.001}, // %arg31: RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Conv3d_2c_3x3_batch_norm_beta, // %arg32
    {0.001}, // %arg33: RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_3b_Branch_3_Conv3d_0b_1x1_batch_norm_beta, // %arg34
    ::weights::RGB_inception_i3d_Mixed_3b_Branch_0_Conv3d_0a_1x1_conv_3d_w, // %arg35
    {0.001}, // %arg36: RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_3b_Branch_0_Conv3d_0a_1x1_batch_norm_beta, // %arg37
    ::weights::RGB_inception_i3d_Mixed_3b_Branch_1_Conv3d_0b_3x3_conv_3d_w, // %arg38
    {0}, // %arg39: RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/Relu
    ::weights::RGB_inception_i3d_Mixed_3b_Branch_1_Conv3d_0a_1x1_conv_3d_w, // %arg40
    {0.001}, // %arg41: RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_3b_Branch_1_Conv3d_0a_1x1_batch_norm_beta, // %arg42
    {0.001}, // %arg43: RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_3b_Branch_1_Conv3d_0b_3x3_batch_norm_beta, // %arg44
    ::weights::RGB_inception_i3d_Mixed_3b_Branch_2_Conv3d_0b_3x3_conv_3d_w, // %arg45
    {0}, // %arg46: RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/Relu
    ::weights::RGB_inception_i3d_Mixed_3b_Branch_2_Conv3d_0a_1x1_conv_3d_w, // %arg47
    {0.001}, // %arg48: RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_3b_Branch_2_Conv3d_0a_1x1_batch_norm_beta, // %arg49
    {0.001}, // %arg50: RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_3b_Branch_2_Conv3d_0b_3x3_batch_norm_beta, // %arg51
    {0.001}, // %arg52: RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_3c_Branch_3_Conv3d_0b_1x1_batch_norm_beta, // %arg53
    ::weights::RGB_inception_i3d_Mixed_3c_Branch_0_Conv3d_0a_1x1_conv_3d_w, // %arg54
    {0.001}, // %arg55: RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_3c_Branch_0_Conv3d_0a_1x1_batch_norm_beta, // %arg56
    ::weights::RGB_inception_i3d_Mixed_3c_Branch_1_Conv3d_0b_3x3_conv_3d_w, // %arg57
    {0}, // %arg58: RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/Relu
    ::weights::RGB_inception_i3d_Mixed_3c_Branch_1_Conv3d_0a_1x1_conv_3d_w, // %arg59
    {0.001}, // RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_3c_Branch_1_Conv3d_0a_1x1_batch_norm_beta, // %arg61
    {0.001}, // RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_3c_Branch_1_Conv3d_0b_3x3_batch_norm_beta, // %arg63
    ::weights::RGB_inception_i3d_Mixed_3c_Branch_2_Conv3d_0b_3x3_conv_3d_w, // %arg64
    {0}, // %arg65: RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/Relu
    ::weights::RGB_inception_i3d_Mixed_3c_Branch_2_Conv3d_0a_1x1_conv_3d_w, // %arg66
    {0.001}, // %arg67: RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_3c_Branch_2_Conv3d_0a_1x1_batch_norm_beta, // %arg68
    {0.001}, // %arg69: RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_3c_Branch_2_Conv3d_0b_3x3_batch_norm_beta, // %arg70
    {0.001}, // %arg71: RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_4b_Branch_3_Conv3d_0b_1x1_batch_norm_beta, // %arg72
    ::weights::RGB_inception_i3d_Mixed_4b_Branch_0_Conv3d_0a_1x1_conv_3d_w, // %arg73
    {0.001}, // %arg74: RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_4b_Branch_0_Conv3d_0a_1x1_batch_norm_beta, // %arg75
    ::weights::RGB_inception_i3d_Mixed_4b_Branch_1_Conv3d_0b_3x3_conv_3d_w, // %arg76
    {0}, // %arg77: RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/Relu
    ::weights::RGB_inception_i3d_Mixed_4b_Branch_1_Conv3d_0a_1x1_conv_3d_w, // %arg78
    {0.001}, // %arg79: RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_4b_Branch_1_Conv3d_0a_1x1_batch_norm_beta, // %arg80
    {0.001}, // %arg81: RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_4b_Branch_1_Conv3d_0b_3x3_batch_norm_beta, // %arg82
    ::weights::RGB_inception_i3d_Mixed_4b_Branch_2_Conv3d_0b_3x3_conv_3d_w, // %arg83
    {0}, // %arg84: RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/Relu
    ::weights::RGB_inception_i3d_Mixed_4b_Branch_2_Conv3d_0a_1x1_conv_3d_w, // %arg85
    {0.001}, // %arg86: RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_4b_Branch_2_Conv3d_0a_1x1_batch_norm_beta, // %arg87
    {0.001}, // %arg88: RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_4b_Branch_2_Conv3d_0b_3x3_batch_norm_beta, // %arg89
    {0.001}, // %arg90: RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_4c_Branch_3_Conv3d_0b_1x1_batch_norm_beta, // %arg91
    ::weights::RGB_inception_i3d_Mixed_4c_Branch_0_Conv3d_0a_1x1_conv_3d_w, // %arg92
    {0.001}, // %arg93: RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_4c_Branch_0_Conv3d_0a_1x1_batch_norm_beta, // %arg94
    ::weights::RGB_inception_i3d_Mixed_4c_Branch_1_Conv3d_0b_3x3_conv_3d_w, // %arg95
    {0}, // %arg96: RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/Relu
    ::weights::RGB_inception_i3d_Mixed_4c_Branch_1_Conv3d_0a_1x1_conv_3d_w, // %arg97
    {0.001}, // %arg98: RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_4c_Branch_1_Conv3d_0a_1x1_batch_norm_beta, // %arg99
    {0.001}, // %arg100 RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_4c_Branch_1_Conv3d_0b_3x3_batch_norm_beta, // %arg101
    ::weights::RGB_inception_i3d_Mixed_4c_Branch_2_Conv3d_0b_3x3_conv_3d_w, // %arg102
    {0}, // %arg103: RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/Relu
    ::weights::RGB_inception_i3d_Mixed_4c_Branch_2_Conv3d_0a_1x1_conv_3d_w, // %arg104
    {0.001}, // %arg105: RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_4c_Branch_2_Conv3d_0a_1x1_batch_norm_beta, // %arg106
    {0.001}, // %arg107: RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_4c_Branch_2_Conv3d_0b_3x3_batch_norm_beta, // %arg108
    {0.001}, // %arg109: RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_4d_Branch_3_Conv3d_0b_1x1_batch_norm_beta, // %arg110
    ::weights::RGB_inception_i3d_Mixed_4d_Branch_0_Conv3d_0a_1x1_conv_3d_w, // %arg111
    {0.001}, // %arg112: RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_4d_Branch_0_Conv3d_0a_1x1_batch_norm_beta, // %arg113
    ::weights::RGB_inception_i3d_Mixed_4d_Branch_1_Conv3d_0b_3x3_conv_3d_w, // %arg114
    {0}, // %arg115: RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/Relu
    ::weights::RGB_inception_i3d_Mixed_4d_Branch_1_Conv3d_0a_1x1_conv_3d_w, // %arg116
    {0.001}, // %arg117: RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_4d_Branch_1_Conv3d_0a_1x1_batch_norm_beta, // %arg118
    {0.001}, // %arg119: RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_4d_Branch_1_Conv3d_0b_3x3_batch_norm_beta, // %arg120
    ::weights::RGB_inception_i3d_Mixed_4d_Branch_2_Conv3d_0b_3x3_conv_3d_w, // %arg121
    {0}, // %arg122: RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/Relu
    ::weights::RGB_inception_i3d_Mixed_4d_Branch_2_Conv3d_0a_1x1_conv_3d_w, // %arg123
    {0.001}, // %arg124: RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_4d_Branch_2_Conv3d_0a_1x1_batch_norm_beta, // %arg125
    {0.001}, // %arg126: RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_4d_Branch_2_Conv3d_0b_3x3_batch_norm_beta, // %arg127
    {0.001}, // %arg128: RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_4e_Branch_3_Conv3d_0b_1x1_batch_norm_beta, // %arg129
    ::weights::RGB_inception_i3d_Mixed_4e_Branch_0_Conv3d_0a_1x1_conv_3d_w, // %arg130
    {0.001}, // %arg131: RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_4e_Branch_0_Conv3d_0a_1x1_batch_norm_beta, // %arg132
    ::weights::RGB_inception_i3d_Mixed_4e_Branch_1_Conv3d_0b_3x3_conv_3d_w, // %arg133
    {0}, // %arg134: RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/Relu
    ::weights::RGB_inception_i3d_Mixed_4e_Branch_1_Conv3d_0a_1x1_conv_3d_w, // %arg135
    {0.001}, // %arg136: RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_4e_Branch_1_Conv3d_0a_1x1_batch_norm_beta, // %arg137
    {0.001}, // %arg138: RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_4e_Branch_1_Conv3d_0b_3x3_batch_norm_beta, // %arg139
    ::weights::RGB_inception_i3d_Mixed_4e_Branch_2_Conv3d_0b_3x3_conv_3d_w, // %arg140
    {0}, // %arg141: RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/Relu
    ::weights::RGB_inception_i3d_Mixed_4e_Branch_2_Conv3d_0a_1x1_conv_3d_w, // %arg142
    {0.001}, // %arg143: RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_4e_Branch_2_Conv3d_0a_1x1_batch_norm_beta, // %arg144
    {0.001}, // %arg145: RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_4e_Branch_2_Conv3d_0b_3x3_batch_norm_beta, // %arg146
    {0.001}, // %arg147: RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_4f_Branch_3_Conv3d_0b_1x1_batch_norm_beta, // %arg148
    ::weights::RGB_inception_i3d_Mixed_4f_Branch_0_Conv3d_0a_1x1_conv_3d_w, // %arg149
    {0.001}, // %arg150: RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_4f_Branch_0_Conv3d_0a_1x1_batch_norm_beta, // %arg151
    ::weights::RGB_inception_i3d_Mixed_4f_Branch_1_Conv3d_0b_3x3_conv_3d_w, // %arg152
    {0}, // %arg153: RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/Relu
    ::weights::RGB_inception_i3d_Mixed_4f_Branch_1_Conv3d_0a_1x1_conv_3d_w, // %arg154
    {0.001}, // %arg155: RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_4f_Branch_1_Conv3d_0a_1x1_batch_norm_beta, // %arg156
    {0.001}, // %arg157: RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_4f_Branch_1_Conv3d_0b_3x3_batch_norm_beta, // %arg158
    ::weights::RGB_inception_i3d_Mixed_4f_Branch_2_Conv3d_0b_3x3_conv_3d_w, // %arg159
    {0}, // %arg160: RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/Relu
    ::weights::RGB_inception_i3d_Mixed_4f_Branch_2_Conv3d_0a_1x1_conv_3d_w, // %arg161
    {0.001}, // %arg162: RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_4f_Branch_2_Conv3d_0a_1x1_batch_norm_beta, // %arg163
    {0.001}, // %arg164: RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_4f_Branch_2_Conv3d_0b_3x3_batch_norm_beta, // %arg165
    {0.001}, // %arg166: RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_5b_Branch_3_Conv3d_0b_1x1_batch_norm_beta, // %arg167
    ::weights::RGB_inception_i3d_Mixed_5b_Branch_0_Conv3d_0a_1x1_conv_3d_w, // %arg168
    {0.001}, // %arg169: RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_5b_Branch_0_Conv3d_0a_1x1_batch_norm_beta, // %arg170
    ::weights::RGB_inception_i3d_Mixed_5b_Branch_1_Conv3d_0b_3x3_conv_3d_w, // %arg171
    {0}, // %arg172: RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/Relu
    ::weights::RGB_inception_i3d_Mixed_5b_Branch_1_Conv3d_0a_1x1_conv_3d_w, // %arg173
    {0.001}, // %arg174: RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_5b_Branch_1_Conv3d_0a_1x1_batch_norm_beta, // %arg175
    {0.001}, // %arg176: RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_5b_Branch_1_Conv3d_0b_3x3_batch_norm_beta, // %arg177
    ::weights::RGB_inception_i3d_Mixed_5b_Branch_2_Conv3d_0a_3x3_conv_3d_w, // %arg178
    {0}, // %arg179: RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/Relu
    ::weights::RGB_inception_i3d_Mixed_5b_Branch_2_Conv3d_0a_1x1_conv_3d_w, // %arg180
    {0.001}, // %arg181: RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_5b_Branch_2_Conv3d_0a_1x1_batch_norm_beta, // %arg183
    {0.001}, // %arg183: RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_5b_Branch_2_Conv3d_0a_3x3_batch_norm_beta, // %arg184
    {0.001}, // %arg185: RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_5c_Branch_3_Conv3d_0b_1x1_batch_norm_beta, // %arg186
    ::weights::RGB_inception_i3d_Mixed_5c_Branch_0_Conv3d_0a_1x1_conv_3d_w, // %arg187
    {0.001}, // %arg188: RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_5c_Branch_0_Conv3d_0a_1x1_batch_norm_beta, // %arg189
    ::weights::RGB_inception_i3d_Mixed_5c_Branch_1_Conv3d_0b_3x3_conv_3d_w, // %arg190
    {0}, // %arg191: RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/Relu
    ::weights::RGB_inception_i3d_Mixed_5c_Branch_1_Conv3d_0a_1x1_conv_3d_w, // %arg192
    {0.001}, // %arg193: RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_5c_Branch_1_Conv3d_0a_1x1_batch_norm_beta, // %arg194
    {0.001}, // %arg195: RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_5c_Branch_1_Conv3d_0b_3x3_batch_norm_beta, // %arg196
    ::weights::RGB_inception_i3d_Mixed_5c_Branch_2_Conv3d_0b_3x3_conv_3d_w, // %arg197
    {0}, // %arg198: RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/Relu
    ::weights::RGB_inception_i3d_Mixed_5c_Branch_2_Conv3d_0a_1x1_conv_3d_w, // %arg199
    {0.001}, // %arg200: RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_5c_Branch_2_Conv3d_0a_1x1_batch_norm_beta, // %arg201
    {0.001}, // %arg202: RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add
    ::weights::RGB_inception_i3d_Mixed_5c_Branch_2_Conv3d_0b_3x3_batch_norm_beta, // %arg203
    ::weights::RGB_inception_i3d_Logits_Conv3d_0c_1x1_conv_3d_b // %arg204
  };

  TestCaseVal I3D_Output = ::outputs::I3D_Output;

TestCasePairs testcase_pairs = {{I3D_WeightsInputs, I3D_Output}};

  I3DTestSpec spec = GetParam();

  HloModuleConfig cfg;

  //std::unique_ptr<HloModule> hlo_module = absl::make_unique<HloModule>("module", cfg);

  std::unique_ptr<VerifiedHloModule> hlo_module = absl::make_unique<VerifiedHloModule>(
      "module", cfg, false, false, nullptr);

std::string hlo_text = R"(HloModule cluster_1__XlaCompiledKernel_true__XlaNumConstantArgs_125__XlaNumResourceArgs_116_.3580

%RGB_inception_i3d_Conv3d_1a_7x7_batch_norm_normalize_moments_mean-reduction.124 (x.125: f32[], y.126: f32[]) -> f32[] {
  %x.125 = f32[] parameter(0)
  %y.126 = f32[] parameter(1)
  ROOT %add.127 = f32[] add(f32[] %x.125, f32[] %y.126)
}

%RGB_inception_i3d_Conv3d_1a_7x7_batch_norm_normalize_moments_variance-reduction.148 (x.149: f32[], y.150: f32[]) -> f32[] {
  %x.149 = f32[] parameter(0)
  %y.150 = f32[] parameter(1)
  ROOT %add.151 = f32[] add(f32[] %x.149, f32[] %y.150)
}

%max_F32.178 (lhs.179: f32[], rhs.180: f32[]) -> f32[] {
  %lhs.179 = f32[] parameter(0)
  %rhs.180 = f32[] parameter(1)
  ROOT %maximum.181 = f32[] maximum(f32[] %lhs.179, f32[] %rhs.180)
}

%RGB_inception_i3d_Conv3d_2b_1x1_batch_norm_normalize_moments_mean-reduction.190 (x.191: f32[], y.192: f32[]) -> f32[] {
  %x.191 = f32[] parameter(0)
  %y.192 = f32[] parameter(1)
  ROOT %add.193 = f32[] add(f32[] %x.191, f32[] %y.192)
}

%RGB_inception_i3d_Conv3d_2b_1x1_batch_norm_normalize_moments_variance-reduction.214 (x.215: f32[], y.216: f32[]) -> f32[] {
  %x.215 = f32[] parameter(0)
  %y.216 = f32[] parameter(1)
  ROOT %add.217 = f32[] add(f32[] %x.215, f32[] %y.216)
}

%RGB_inception_i3d_Conv3d_2c_3x3_batch_norm_normalize_moments_mean-reduction.250 (x.251: f32[], y.252: f32[]) -> f32[] {
  %x.251 = f32[] parameter(0)
  %y.252 = f32[] parameter(1)
  ROOT %add.253 = f32[] add(f32[] %x.251, f32[] %y.252)
}

%RGB_inception_i3d_Conv3d_2c_3x3_batch_norm_normalize_moments_variance-reduction.274 (x.275: f32[], y.276: f32[]) -> f32[] {
  %x.275 = f32[] parameter(0)
  %y.276 = f32[] parameter(1)
  ROOT %add.277 = f32[] add(f32[] %x.275, f32[] %y.276)
}

%max_F32.304 (lhs.305: f32[], rhs.306: f32[]) -> f32[] {
  %lhs.305 = f32[] parameter(0)
  %rhs.306 = f32[] parameter(1)
  ROOT %maximum.307 = f32[] maximum(f32[] %lhs.305, f32[] %rhs.306)
}

%RGB_inception_i3d_Mixed_3b_Branch_1_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.317 (x.318: f32[], y.319: f32[]) -> f32[] {
  %x.318 = f32[] parameter(0)
  %y.319 = f32[] parameter(1)
  ROOT %add.320 = f32[] add(f32[] %x.318, f32[] %y.319)
}

%RGB_inception_i3d_Mixed_3b_Branch_1_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.341 (x.342: f32[], y.343: f32[]) -> f32[] {
  %x.342 = f32[] parameter(0)
  %y.343 = f32[] parameter(1)
  ROOT %add.344 = f32[] add(f32[] %x.342, f32[] %y.343)
}

%RGB_inception_i3d_Mixed_3b_Branch_1_Conv3d_0b_3x3_batch_norm_normalize_moments_mean-reduction.377 (x.378: f32[], y.379: f32[]) -> f32[] {
  %x.378 = f32[] parameter(0)
  %y.379 = f32[] parameter(1)
  ROOT %add.380 = f32[] add(f32[] %x.378, f32[] %y.379)
}

%RGB_inception_i3d_Mixed_3b_Branch_1_Conv3d_0b_3x3_batch_norm_normalize_moments_variance-reduction.401 (x.402: f32[], y.403: f32[]) -> f32[] {
  %x.402 = f32[] parameter(0)
  %y.403 = f32[] parameter(1)
  ROOT %add.404 = f32[] add(f32[] %x.402, f32[] %y.403)
}

%RGB_inception_i3d_Mixed_3b_Branch_2_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.434 (x.435: f32[], y.436: f32[]) -> f32[] {
  %x.435 = f32[] parameter(0)
  %y.436 = f32[] parameter(1)
  ROOT %add.437 = f32[] add(f32[] %x.435, f32[] %y.436)
}

%RGB_inception_i3d_Mixed_3b_Branch_2_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.458 (x.459: f32[], y.460: f32[]) -> f32[] {
  %x.459 = f32[] parameter(0)
  %y.460 = f32[] parameter(1)
  ROOT %add.461 = f32[] add(f32[] %x.459, f32[] %y.460)
}

%RGB_inception_i3d_Mixed_3b_Branch_2_Conv3d_0b_3x3_batch_norm_normalize_moments_mean-reduction.494 (x.495: f32[], y.496: f32[]) -> f32[] {
  %x.495 = f32[] parameter(0)
  %y.496 = f32[] parameter(1)
  ROOT %add.497 = f32[] add(f32[] %x.495, f32[] %y.496)
}

%RGB_inception_i3d_Mixed_3b_Branch_2_Conv3d_0b_3x3_batch_norm_normalize_moments_variance-reduction.518 (x.519: f32[], y.520: f32[]) -> f32[] {
  %x.519 = f32[] parameter(0)
  %y.520 = f32[] parameter(1)
  ROOT %add.521 = f32[] add(f32[] %x.519, f32[] %y.520)
}

%max_F32.548 (lhs.549: f32[], rhs.550: f32[]) -> f32[] {
  %lhs.549 = f32[] parameter(0)
  %rhs.550 = f32[] parameter(1)
  ROOT %maximum.551 = f32[] maximum(f32[] %lhs.549, f32[] %rhs.550)
}

%RGB_inception_i3d_Mixed_3b_Branch_3_Conv3d_0b_1x1_batch_norm_normalize_moments_mean-reduction.557 (x.558: f32[], y.559: f32[]) -> f32[] {
  %x.558 = f32[] parameter(0)
  %y.559 = f32[] parameter(1)
  ROOT %add.560 = f32[] add(f32[] %x.558, f32[] %y.559)
}

%RGB_inception_i3d_Mixed_3b_Branch_3_Conv3d_0b_1x1_batch_norm_normalize_moments_variance-reduction.581 (x.582: f32[], y.583: f32[]) -> f32[] {
  %x.582 = f32[] parameter(0)
  %y.583 = f32[] parameter(1)
  ROOT %add.584 = f32[] add(f32[] %x.582, f32[] %y.583)
}

%RGB_inception_i3d_Mixed_3b_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.613 (x.614: f32[], y.615: f32[]) -> f32[] {
  %x.614 = f32[] parameter(0)
  %y.615 = f32[] parameter(1)
  ROOT %add.616 = f32[] add(f32[] %x.614, f32[] %y.615)
}

%RGB_inception_i3d_Mixed_3b_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.637 (x.638: f32[], y.639: f32[]) -> f32[] {
  %x.638 = f32[] parameter(0)
  %y.639 = f32[] parameter(1)
  ROOT %add.640 = f32[] add(f32[] %x.638, f32[] %y.639)
}

%RGB_inception_i3d_Mixed_3c_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.674 (x.675: f32[], y.676: f32[]) -> f32[] {
  %x.675 = f32[] parameter(0)
  %y.676 = f32[] parameter(1)
  ROOT %add.677 = f32[] add(f32[] %x.675, f32[] %y.676)
}

%RGB_inception_i3d_Mixed_3c_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.698 (x.699: f32[], y.700: f32[]) -> f32[] {
  %x.699 = f32[] parameter(0)
  %y.700 = f32[] parameter(1)
  ROOT %add.701 = f32[] add(f32[] %x.699, f32[] %y.700)
}

%RGB_inception_i3d_Mixed_3c_Branch_1_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.731 (x.732: f32[], y.733: f32[]) -> f32[] {
  %x.732 = f32[] parameter(0)
  %y.733 = f32[] parameter(1)
  ROOT %add.734 = f32[] add(f32[] %x.732, f32[] %y.733)
}

%RGB_inception_i3d_Mixed_3c_Branch_1_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.755 (x.756: f32[], y.757: f32[]) -> f32[] {
  %x.756 = f32[] parameter(0)
  %y.757 = f32[] parameter(1)
  ROOT %add.758 = f32[] add(f32[] %x.756, f32[] %y.757)
}

%RGB_inception_i3d_Mixed_3c_Branch_1_Conv3d_0b_3x3_batch_norm_normalize_moments_mean-reduction.791 (x.792: f32[], y.793: f32[]) -> f32[] {
  %x.792 = f32[] parameter(0)
  %y.793 = f32[] parameter(1)
  ROOT %add.794 = f32[] add(f32[] %x.792, f32[] %y.793)
}

%RGB_inception_i3d_Mixed_3c_Branch_1_Conv3d_0b_3x3_batch_norm_normalize_moments_variance-reduction.815 (x.816: f32[], y.817: f32[]) -> f32[] {
  %x.816 = f32[] parameter(0)
  %y.817 = f32[] parameter(1)
  ROOT %add.818 = f32[] add(f32[] %x.816, f32[] %y.817)
}

%RGB_inception_i3d_Mixed_3c_Branch_2_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.848 (x.849: f32[], y.850: f32[]) -> f32[] {
  %x.849 = f32[] parameter(0)
  %y.850 = f32[] parameter(1)
  ROOT %add.851 = f32[] add(f32[] %x.849, f32[] %y.850)
}

%RGB_inception_i3d_Mixed_3c_Branch_2_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.872 (x.873: f32[], y.874: f32[]) -> f32[] {
  %x.873 = f32[] parameter(0)
  %y.874 = f32[] parameter(1)
  ROOT %add.875 = f32[] add(f32[] %x.873, f32[] %y.874)
}

%RGB_inception_i3d_Mixed_3c_Branch_2_Conv3d_0b_3x3_batch_norm_normalize_moments_mean-reduction.908 (x.909: f32[], y.910: f32[]) -> f32[] {
  %x.909 = f32[] parameter(0)
  %y.910 = f32[] parameter(1)
  ROOT %add.911 = f32[] add(f32[] %x.909, f32[] %y.910)
}

%RGB_inception_i3d_Mixed_3c_Branch_2_Conv3d_0b_3x3_batch_norm_normalize_moments_variance-reduction.932 (x.933: f32[], y.934: f32[]) -> f32[] {
  %x.933 = f32[] parameter(0)
  %y.934 = f32[] parameter(1)
  ROOT %add.935 = f32[] add(f32[] %x.933, f32[] %y.934)
}

%max_F32.962 (lhs.963: f32[], rhs.964: f32[]) -> f32[] {
  %lhs.963 = f32[] parameter(0)
  %rhs.964 = f32[] parameter(1)
  ROOT %maximum.965 = f32[] maximum(f32[] %lhs.963, f32[] %rhs.964)
}

%RGB_inception_i3d_Mixed_3c_Branch_3_Conv3d_0b_1x1_batch_norm_normalize_moments_mean-reduction.971 (x.972: f32[], y.973: f32[]) -> f32[] {
  %x.972 = f32[] parameter(0)
  %y.973 = f32[] parameter(1)
  ROOT %add.974 = f32[] add(f32[] %x.972, f32[] %y.973)
}

%RGB_inception_i3d_Mixed_3c_Branch_3_Conv3d_0b_1x1_batch_norm_normalize_moments_variance-reduction.995 (x.996: f32[], y.997: f32[]) -> f32[] {
  %x.996 = f32[] parameter(0)
  %y.997 = f32[] parameter(1)
  ROOT %add.998 = f32[] add(f32[] %x.996, f32[] %y.997)
}

%max_F32.1026 (lhs.1027: f32[], rhs.1028: f32[]) -> f32[] {
  %lhs.1027 = f32[] parameter(0)
  %rhs.1028 = f32[] parameter(1)
  ROOT %maximum.1029 = f32[] maximum(f32[] %lhs.1027, f32[] %rhs.1028)
}

%RGB_inception_i3d_Mixed_4b_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.1038 (x.1039: f32[], y.1040: f32[]) -> f32[] {
  %x.1039 = f32[] parameter(0)
  %y.1040 = f32[] parameter(1)
  ROOT %add.1041 = f32[] add(f32[] %x.1039, f32[] %y.1040)
}

%RGB_inception_i3d_Mixed_4b_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.1062 (x.1063: f32[], y.1064: f32[]) -> f32[] {
  %x.1063 = f32[] parameter(0)
  %y.1064 = f32[] parameter(1)
  ROOT %add.1065 = f32[] add(f32[] %x.1063, f32[] %y.1064)
}

%RGB_inception_i3d_Mixed_4b_Branch_1_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.1095 (x.1096: f32[], y.1097: f32[]) -> f32[] {
  %x.1096 = f32[] parameter(0)
  %y.1097 = f32[] parameter(1)
  ROOT %add.1098 = f32[] add(f32[] %x.1096, f32[] %y.1097)
}

%RGB_inception_i3d_Mixed_4b_Branch_1_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.1119 (x.1120: f32[], y.1121: f32[]) -> f32[] {
  %x.1120 = f32[] parameter(0)
  %y.1121 = f32[] parameter(1)
  ROOT %add.1122 = f32[] add(f32[] %x.1120, f32[] %y.1121)
}

%RGB_inception_i3d_Mixed_4b_Branch_1_Conv3d_0b_3x3_batch_norm_normalize_moments_mean-reduction.1155 (x.1156: f32[], y.1157: f32[]) -> f32[] {
  %x.1156 = f32[] parameter(0)
  %y.1157 = f32[] parameter(1)
  ROOT %add.1158 = f32[] add(f32[] %x.1156, f32[] %y.1157)
}

%RGB_inception_i3d_Mixed_4b_Branch_1_Conv3d_0b_3x3_batch_norm_normalize_moments_variance-reduction.1179 (x.1180: f32[], y.1181: f32[]) -> f32[] {
  %x.1180 = f32[] parameter(0)
  %y.1181 = f32[] parameter(1)
  ROOT %add.1182 = f32[] add(f32[] %x.1180, f32[] %y.1181)
}

%RGB_inception_i3d_Mixed_4b_Branch_2_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.1212 (x.1213: f32[], y.1214: f32[]) -> f32[] {
  %x.1213 = f32[] parameter(0)
  %y.1214 = f32[] parameter(1)
  ROOT %add.1215 = f32[] add(f32[] %x.1213, f32[] %y.1214)
}

%RGB_inception_i3d_Mixed_4b_Branch_2_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.1236 (x.1237: f32[], y.1238: f32[]) -> f32[] {
  %x.1237 = f32[] parameter(0)
  %y.1238 = f32[] parameter(1)
  ROOT %add.1239 = f32[] add(f32[] %x.1237, f32[] %y.1238)
}

%RGB_inception_i3d_Mixed_4b_Branch_2_Conv3d_0b_3x3_batch_norm_normalize_moments_mean-reduction.1272 (x.1273: f32[], y.1274: f32[]) -> f32[] {
  %x.1273 = f32[] parameter(0)
  %y.1274 = f32[] parameter(1)
  ROOT %add.1275 = f32[] add(f32[] %x.1273, f32[] %y.1274)
}

%RGB_inception_i3d_Mixed_4b_Branch_2_Conv3d_0b_3x3_batch_norm_normalize_moments_variance-reduction.1296 (x.1297: f32[], y.1298: f32[]) -> f32[] {
  %x.1297 = f32[] parameter(0)
  %y.1298 = f32[] parameter(1)
  ROOT %add.1299 = f32[] add(f32[] %x.1297, f32[] %y.1298)
}

%max_F32.1326 (lhs.1327: f32[], rhs.1328: f32[]) -> f32[] {
  %lhs.1327 = f32[] parameter(0)
  %rhs.1328 = f32[] parameter(1)
  ROOT %maximum.1329 = f32[] maximum(f32[] %lhs.1327, f32[] %rhs.1328)
}

%RGB_inception_i3d_Mixed_4b_Branch_3_Conv3d_0b_1x1_batch_norm_normalize_moments_mean-reduction.1335 (x.1336: f32[], y.1337: f32[]) -> f32[] {
  %x.1336 = f32[] parameter(0)
  %y.1337 = f32[] parameter(1)
  ROOT %add.1338 = f32[] add(f32[] %x.1336, f32[] %y.1337)
}

%RGB_inception_i3d_Mixed_4b_Branch_3_Conv3d_0b_1x1_batch_norm_normalize_moments_variance-reduction.1359 (x.1360: f32[], y.1361: f32[]) -> f32[] {
  %x.1360 = f32[] parameter(0)
  %y.1361 = f32[] parameter(1)
  ROOT %add.1362 = f32[] add(f32[] %x.1360, f32[] %y.1361)
}

%RGB_inception_i3d_Mixed_4c_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.1396 (x.1397: f32[], y.1398: f32[]) -> f32[] {
  %x.1397 = f32[] parameter(0)
  %y.1398 = f32[] parameter(1)
  ROOT %add.1399 = f32[] add(f32[] %x.1397, f32[] %y.1398)
}

%RGB_inception_i3d_Mixed_4c_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.1420 (x.1421: f32[], y.1422: f32[]) -> f32[] {
  %x.1421 = f32[] parameter(0)
  %y.1422 = f32[] parameter(1)
  ROOT %add.1423 = f32[] add(f32[] %x.1421, f32[] %y.1422)
}

%RGB_inception_i3d_Mixed_4c_Branch_1_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.1453 (x.1454: f32[], y.1455: f32[]) -> f32[] {
  %x.1454 = f32[] parameter(0)
  %y.1455 = f32[] parameter(1)
  ROOT %add.1456 = f32[] add(f32[] %x.1454, f32[] %y.1455)
}

%RGB_inception_i3d_Mixed_4c_Branch_1_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.1477 (x.1478: f32[], y.1479: f32[]) -> f32[] {
  %x.1478 = f32[] parameter(0)
  %y.1479 = f32[] parameter(1)
  ROOT %add.1480 = f32[] add(f32[] %x.1478, f32[] %y.1479)
}

%RGB_inception_i3d_Mixed_4c_Branch_1_Conv3d_0b_3x3_batch_norm_normalize_moments_mean-reduction.1513 (x.1514: f32[], y.1515: f32[]) -> f32[] {
  %x.1514 = f32[] parameter(0)
  %y.1515 = f32[] parameter(1)
  ROOT %add.1516 = f32[] add(f32[] %x.1514, f32[] %y.1515)
}

%RGB_inception_i3d_Mixed_4c_Branch_1_Conv3d_0b_3x3_batch_norm_normalize_moments_variance-reduction.1537 (x.1538: f32[], y.1539: f32[]) -> f32[] {
  %x.1538 = f32[] parameter(0)
  %y.1539 = f32[] parameter(1)
  ROOT %add.1540 = f32[] add(f32[] %x.1538, f32[] %y.1539)
}

%RGB_inception_i3d_Mixed_4c_Branch_2_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.1570 (x.1571: f32[], y.1572: f32[]) -> f32[] {
  %x.1571 = f32[] parameter(0)
  %y.1572 = f32[] parameter(1)
  ROOT %add.1573 = f32[] add(f32[] %x.1571, f32[] %y.1572)
}

%RGB_inception_i3d_Mixed_4c_Branch_2_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.1594 (x.1595: f32[], y.1596: f32[]) -> f32[] {
  %x.1595 = f32[] parameter(0)
  %y.1596 = f32[] parameter(1)
  ROOT %add.1597 = f32[] add(f32[] %x.1595, f32[] %y.1596)
}

%RGB_inception_i3d_Mixed_4c_Branch_2_Conv3d_0b_3x3_batch_norm_normalize_moments_mean-reduction.1630 (x.1631: f32[], y.1632: f32[]) -> f32[] {
  %x.1631 = f32[] parameter(0)
  %y.1632 = f32[] parameter(1)
  ROOT %add.1633 = f32[] add(f32[] %x.1631, f32[] %y.1632)
}

%RGB_inception_i3d_Mixed_4c_Branch_2_Conv3d_0b_3x3_batch_norm_normalize_moments_variance-reduction.1654 (x.1655: f32[], y.1656: f32[]) -> f32[] {
  %x.1655 = f32[] parameter(0)
  %y.1656 = f32[] parameter(1)
  ROOT %add.1657 = f32[] add(f32[] %x.1655, f32[] %y.1656)
}

%max_F32.1684 (lhs.1685: f32[], rhs.1686: f32[]) -> f32[] {
  %lhs.1685 = f32[] parameter(0)
  %rhs.1686 = f32[] parameter(1)
  ROOT %maximum.1687 = f32[] maximum(f32[] %lhs.1685, f32[] %rhs.1686)
}

%RGB_inception_i3d_Mixed_4c_Branch_3_Conv3d_0b_1x1_batch_norm_normalize_moments_mean-reduction.1693 (x.1694: f32[], y.1695: f32[]) -> f32[] {
  %x.1694 = f32[] parameter(0)
  %y.1695 = f32[] parameter(1)
  ROOT %add.1696 = f32[] add(f32[] %x.1694, f32[] %y.1695)
}

%RGB_inception_i3d_Mixed_4c_Branch_3_Conv3d_0b_1x1_batch_norm_normalize_moments_variance-reduction.1717 (x.1718: f32[], y.1719: f32[]) -> f32[] {
  %x.1718 = f32[] parameter(0)
  %y.1719 = f32[] parameter(1)
  ROOT %add.1720 = f32[] add(f32[] %x.1718, f32[] %y.1719)
}

%RGB_inception_i3d_Mixed_4d_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.1754 (x.1755: f32[], y.1756: f32[]) -> f32[] {
  %x.1755 = f32[] parameter(0)
  %y.1756 = f32[] parameter(1)
  ROOT %add.1757 = f32[] add(f32[] %x.1755, f32[] %y.1756)
}

%RGB_inception_i3d_Mixed_4d_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.1778 (x.1779: f32[], y.1780: f32[]) -> f32[] {
  %x.1779 = f32[] parameter(0)
  %y.1780 = f32[] parameter(1)
  ROOT %add.1781 = f32[] add(f32[] %x.1779, f32[] %y.1780)
}

%RGB_inception_i3d_Mixed_4d_Branch_1_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.1811 (x.1812: f32[], y.1813: f32[]) -> f32[] {
  %x.1812 = f32[] parameter(0)
  %y.1813 = f32[] parameter(1)
  ROOT %add.1814 = f32[] add(f32[] %x.1812, f32[] %y.1813)
}

%RGB_inception_i3d_Mixed_4d_Branch_1_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.1835 (x.1836: f32[], y.1837: f32[]) -> f32[] {
  %x.1836 = f32[] parameter(0)
  %y.1837 = f32[] parameter(1)
  ROOT %add.1838 = f32[] add(f32[] %x.1836, f32[] %y.1837)
}

%RGB_inception_i3d_Mixed_4d_Branch_1_Conv3d_0b_3x3_batch_norm_normalize_moments_mean-reduction.1871 (x.1872: f32[], y.1873: f32[]) -> f32[] {
  %x.1872 = f32[] parameter(0)
  %y.1873 = f32[] parameter(1)
  ROOT %add.1874 = f32[] add(f32[] %x.1872, f32[] %y.1873)
}

%RGB_inception_i3d_Mixed_4d_Branch_1_Conv3d_0b_3x3_batch_norm_normalize_moments_variance-reduction.1895 (x.1896: f32[], y.1897: f32[]) -> f32[] {
  %x.1896 = f32[] parameter(0)
  %y.1897 = f32[] parameter(1)
  ROOT %add.1898 = f32[] add(f32[] %x.1896, f32[] %y.1897)
}

%RGB_inception_i3d_Mixed_4d_Branch_2_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.1928 (x.1929: f32[], y.1930: f32[]) -> f32[] {
  %x.1929 = f32[] parameter(0)
  %y.1930 = f32[] parameter(1)
  ROOT %add.1931 = f32[] add(f32[] %x.1929, f32[] %y.1930)
}

%RGB_inception_i3d_Mixed_4d_Branch_2_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.1952 (x.1953: f32[], y.1954: f32[]) -> f32[] {
  %x.1953 = f32[] parameter(0)
  %y.1954 = f32[] parameter(1)
  ROOT %add.1955 = f32[] add(f32[] %x.1953, f32[] %y.1954)
}

%RGB_inception_i3d_Mixed_4d_Branch_2_Conv3d_0b_3x3_batch_norm_normalize_moments_mean-reduction.1988 (x.1989: f32[], y.1990: f32[]) -> f32[] {
  %x.1989 = f32[] parameter(0)
  %y.1990 = f32[] parameter(1)
  ROOT %add.1991 = f32[] add(f32[] %x.1989, f32[] %y.1990)
}

%RGB_inception_i3d_Mixed_4d_Branch_2_Conv3d_0b_3x3_batch_norm_normalize_moments_variance-reduction.2012 (x.2013: f32[], y.2014: f32[]) -> f32[] {
  %x.2013 = f32[] parameter(0)
  %y.2014 = f32[] parameter(1)
  ROOT %add.2015 = f32[] add(f32[] %x.2013, f32[] %y.2014)
}

%max_F32.2042 (lhs.2043: f32[], rhs.2044: f32[]) -> f32[] {
  %lhs.2043 = f32[] parameter(0)
  %rhs.2044 = f32[] parameter(1)
  ROOT %maximum.2045 = f32[] maximum(f32[] %lhs.2043, f32[] %rhs.2044)
}

%RGB_inception_i3d_Mixed_4d_Branch_3_Conv3d_0b_1x1_batch_norm_normalize_moments_mean-reduction.2051 (x.2052: f32[], y.2053: f32[]) -> f32[] {
  %x.2052 = f32[] parameter(0)
  %y.2053 = f32[] parameter(1)
  ROOT %add.2054 = f32[] add(f32[] %x.2052, f32[] %y.2053)
}

%RGB_inception_i3d_Mixed_4d_Branch_3_Conv3d_0b_1x1_batch_norm_normalize_moments_variance-reduction.2075 (x.2076: f32[], y.2077: f32[]) -> f32[] {
  %x.2076 = f32[] parameter(0)
  %y.2077 = f32[] parameter(1)
  ROOT %add.2078 = f32[] add(f32[] %x.2076, f32[] %y.2077)
}

%RGB_inception_i3d_Mixed_4e_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.2112 (x.2113: f32[], y.2114: f32[]) -> f32[] {
  %x.2113 = f32[] parameter(0)
  %y.2114 = f32[] parameter(1)
  ROOT %add.2115 = f32[] add(f32[] %x.2113, f32[] %y.2114)
}

%RGB_inception_i3d_Mixed_4e_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.2136 (x.2137: f32[], y.2138: f32[]) -> f32[] {
  %x.2137 = f32[] parameter(0)
  %y.2138 = f32[] parameter(1)
  ROOT %add.2139 = f32[] add(f32[] %x.2137, f32[] %y.2138)
}

%RGB_inception_i3d_Mixed_4e_Branch_1_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.2169 (x.2170: f32[], y.2171: f32[]) -> f32[] {
  %x.2170 = f32[] parameter(0)
  %y.2171 = f32[] parameter(1)
  ROOT %add.2172 = f32[] add(f32[] %x.2170, f32[] %y.2171)
}

%RGB_inception_i3d_Mixed_4e_Branch_1_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.2193 (x.2194: f32[], y.2195: f32[]) -> f32[] {
  %x.2194 = f32[] parameter(0)
  %y.2195 = f32[] parameter(1)
  ROOT %add.2196 = f32[] add(f32[] %x.2194, f32[] %y.2195)
}

%RGB_inception_i3d_Mixed_4e_Branch_1_Conv3d_0b_3x3_batch_norm_normalize_moments_mean-reduction.2229 (x.2230: f32[], y.2231: f32[]) -> f32[] {
  %x.2230 = f32[] parameter(0)
  %y.2231 = f32[] parameter(1)
  ROOT %add.2232 = f32[] add(f32[] %x.2230, f32[] %y.2231)
}

%RGB_inception_i3d_Mixed_4e_Branch_1_Conv3d_0b_3x3_batch_norm_normalize_moments_variance-reduction.2253 (x.2254: f32[], y.2255: f32[]) -> f32[] {
  %x.2254 = f32[] parameter(0)
  %y.2255 = f32[] parameter(1)
  ROOT %add.2256 = f32[] add(f32[] %x.2254, f32[] %y.2255)
}

%RGB_inception_i3d_Mixed_4e_Branch_2_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.2286 (x.2287: f32[], y.2288: f32[]) -> f32[] {
  %x.2287 = f32[] parameter(0)
  %y.2288 = f32[] parameter(1)
  ROOT %add.2289 = f32[] add(f32[] %x.2287, f32[] %y.2288)
}

%RGB_inception_i3d_Mixed_4e_Branch_2_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.2310 (x.2311: f32[], y.2312: f32[]) -> f32[] {
  %x.2311 = f32[] parameter(0)
  %y.2312 = f32[] parameter(1)
  ROOT %add.2313 = f32[] add(f32[] %x.2311, f32[] %y.2312)
}

%RGB_inception_i3d_Mixed_4e_Branch_2_Conv3d_0b_3x3_batch_norm_normalize_moments_mean-reduction.2346 (x.2347: f32[], y.2348: f32[]) -> f32[] {
  %x.2347 = f32[] parameter(0)
  %y.2348 = f32[] parameter(1)
  ROOT %add.2349 = f32[] add(f32[] %x.2347, f32[] %y.2348)
}

%RGB_inception_i3d_Mixed_4e_Branch_2_Conv3d_0b_3x3_batch_norm_normalize_moments_variance-reduction.2370 (x.2371: f32[], y.2372: f32[]) -> f32[] {
  %x.2371 = f32[] parameter(0)
  %y.2372 = f32[] parameter(1)
  ROOT %add.2373 = f32[] add(f32[] %x.2371, f32[] %y.2372)
}

%max_F32.2400 (lhs.2401: f32[], rhs.2402: f32[]) -> f32[] {
  %lhs.2401 = f32[] parameter(0)
  %rhs.2402 = f32[] parameter(1)
  ROOT %maximum.2403 = f32[] maximum(f32[] %lhs.2401, f32[] %rhs.2402)
}

%RGB_inception_i3d_Mixed_4e_Branch_3_Conv3d_0b_1x1_batch_norm_normalize_moments_mean-reduction.2409 (x.2410: f32[], y.2411: f32[]) -> f32[] {
  %x.2410 = f32[] parameter(0)
  %y.2411 = f32[] parameter(1)
  ROOT %add.2412 = f32[] add(f32[] %x.2410, f32[] %y.2411)
}

%RGB_inception_i3d_Mixed_4e_Branch_3_Conv3d_0b_1x1_batch_norm_normalize_moments_variance-reduction.2433 (x.2434: f32[], y.2435: f32[]) -> f32[] {
  %x.2434 = f32[] parameter(0)
  %y.2435 = f32[] parameter(1)
  ROOT %add.2436 = f32[] add(f32[] %x.2434, f32[] %y.2435)
}

%RGB_inception_i3d_Mixed_4f_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.2470 (x.2471: f32[], y.2472: f32[]) -> f32[] {
  %x.2471 = f32[] parameter(0)
  %y.2472 = f32[] parameter(1)
  ROOT %add.2473 = f32[] add(f32[] %x.2471, f32[] %y.2472)
}

%RGB_inception_i3d_Mixed_4f_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.2494 (x.2495: f32[], y.2496: f32[]) -> f32[] {
  %x.2495 = f32[] parameter(0)
  %y.2496 = f32[] parameter(1)
  ROOT %add.2497 = f32[] add(f32[] %x.2495, f32[] %y.2496)
}

%RGB_inception_i3d_Mixed_4f_Branch_1_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.2527 (x.2528: f32[], y.2529: f32[]) -> f32[] {
  %x.2528 = f32[] parameter(0)
  %y.2529 = f32[] parameter(1)
  ROOT %add.2530 = f32[] add(f32[] %x.2528, f32[] %y.2529)
}

%RGB_inception_i3d_Mixed_4f_Branch_1_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.2551 (x.2552: f32[], y.2553: f32[]) -> f32[] {
  %x.2552 = f32[] parameter(0)
  %y.2553 = f32[] parameter(1)
  ROOT %add.2554 = f32[] add(f32[] %x.2552, f32[] %y.2553)
}

%RGB_inception_i3d_Mixed_4f_Branch_1_Conv3d_0b_3x3_batch_norm_normalize_moments_mean-reduction.2587 (x.2588: f32[], y.2589: f32[]) -> f32[] {
  %x.2588 = f32[] parameter(0)
  %y.2589 = f32[] parameter(1)
  ROOT %add.2590 = f32[] add(f32[] %x.2588, f32[] %y.2589)
}

%RGB_inception_i3d_Mixed_4f_Branch_1_Conv3d_0b_3x3_batch_norm_normalize_moments_variance-reduction.2611 (x.2612: f32[], y.2613: f32[]) -> f32[] {
  %x.2612 = f32[] parameter(0)
  %y.2613 = f32[] parameter(1)
  ROOT %add.2614 = f32[] add(f32[] %x.2612, f32[] %y.2613)
}

%RGB_inception_i3d_Mixed_4f_Branch_2_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.2644 (x.2645: f32[], y.2646: f32[]) -> f32[] {
  %x.2645 = f32[] parameter(0)
  %y.2646 = f32[] parameter(1)
  ROOT %add.2647 = f32[] add(f32[] %x.2645, f32[] %y.2646)
}

%RGB_inception_i3d_Mixed_4f_Branch_2_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.2668 (x.2669: f32[], y.2670: f32[]) -> f32[] {
  %x.2669 = f32[] parameter(0)
  %y.2670 = f32[] parameter(1)
  ROOT %add.2671 = f32[] add(f32[] %x.2669, f32[] %y.2670)
}

%RGB_inception_i3d_Mixed_4f_Branch_2_Conv3d_0b_3x3_batch_norm_normalize_moments_mean-reduction.2704 (x.2705: f32[], y.2706: f32[]) -> f32[] {
  %x.2705 = f32[] parameter(0)
  %y.2706 = f32[] parameter(1)
  ROOT %add.2707 = f32[] add(f32[] %x.2705, f32[] %y.2706)
}

%RGB_inception_i3d_Mixed_4f_Branch_2_Conv3d_0b_3x3_batch_norm_normalize_moments_variance-reduction.2728 (x.2729: f32[], y.2730: f32[]) -> f32[] {
  %x.2729 = f32[] parameter(0)
  %y.2730 = f32[] parameter(1)
  ROOT %add.2731 = f32[] add(f32[] %x.2729, f32[] %y.2730)
}

%max_F32.2758 (lhs.2759: f32[], rhs.2760: f32[]) -> f32[] {
  %lhs.2759 = f32[] parameter(0)
  %rhs.2760 = f32[] parameter(1)
  ROOT %maximum.2761 = f32[] maximum(f32[] %lhs.2759, f32[] %rhs.2760)
}

%RGB_inception_i3d_Mixed_4f_Branch_3_Conv3d_0b_1x1_batch_norm_normalize_moments_mean-reduction.2767 (x.2768: f32[], y.2769: f32[]) -> f32[] {
  %x.2768 = f32[] parameter(0)
  %y.2769 = f32[] parameter(1)
  ROOT %add.2770 = f32[] add(f32[] %x.2768, f32[] %y.2769)
}

%RGB_inception_i3d_Mixed_4f_Branch_3_Conv3d_0b_1x1_batch_norm_normalize_moments_variance-reduction.2791 (x.2792: f32[], y.2793: f32[]) -> f32[] {
  %x.2792 = f32[] parameter(0)
  %y.2793 = f32[] parameter(1)
  ROOT %add.2794 = f32[] add(f32[] %x.2792, f32[] %y.2793)
}

%max_F32.2822 (lhs.2823: f32[], rhs.2824: f32[]) -> f32[] {
  %lhs.2823 = f32[] parameter(0)
  %rhs.2824 = f32[] parameter(1)
  ROOT %maximum.2825 = f32[] maximum(f32[] %lhs.2823, f32[] %rhs.2824)
}

%RGB_inception_i3d_Mixed_5b_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.2834 (x.2835: f32[], y.2836: f32[]) -> f32[] {
  %x.2835 = f32[] parameter(0)
  %y.2836 = f32[] parameter(1)
  ROOT %add.2837 = f32[] add(f32[] %x.2835, f32[] %y.2836)
}

%RGB_inception_i3d_Mixed_5b_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.2858 (x.2859: f32[], y.2860: f32[]) -> f32[] {
  %x.2859 = f32[] parameter(0)
  %y.2860 = f32[] parameter(1)
  ROOT %add.2861 = f32[] add(f32[] %x.2859, f32[] %y.2860)
}

%RGB_inception_i3d_Mixed_5b_Branch_1_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.2891 (x.2892: f32[], y.2893: f32[]) -> f32[] {
  %x.2892 = f32[] parameter(0)
  %y.2893 = f32[] parameter(1)
  ROOT %add.2894 = f32[] add(f32[] %x.2892, f32[] %y.2893)
}

%RGB_inception_i3d_Mixed_5b_Branch_1_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.2915 (x.2916: f32[], y.2917: f32[]) -> f32[] {
  %x.2916 = f32[] parameter(0)
  %y.2917 = f32[] parameter(1)
  ROOT %add.2918 = f32[] add(f32[] %x.2916, f32[] %y.2917)
}

%RGB_inception_i3d_Mixed_5b_Branch_1_Conv3d_0b_3x3_batch_norm_normalize_moments_mean-reduction.2951 (x.2952: f32[], y.2953: f32[]) -> f32[] {
  %x.2952 = f32[] parameter(0)
  %y.2953 = f32[] parameter(1)
  ROOT %add.2954 = f32[] add(f32[] %x.2952, f32[] %y.2953)
}

%RGB_inception_i3d_Mixed_5b_Branch_1_Conv3d_0b_3x3_batch_norm_normalize_moments_variance-reduction.2975 (x.2976: f32[], y.2977: f32[]) -> f32[] {
  %x.2976 = f32[] parameter(0)
  %y.2977 = f32[] parameter(1)
  ROOT %add.2978 = f32[] add(f32[] %x.2976, f32[] %y.2977)
}

%RGB_inception_i3d_Mixed_5b_Branch_2_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.3008 (x.3009: f32[], y.3010: f32[]) -> f32[] {
  %x.3009 = f32[] parameter(0)
  %y.3010 = f32[] parameter(1)
  ROOT %add.3011 = f32[] add(f32[] %x.3009, f32[] %y.3010)
}

%RGB_inception_i3d_Mixed_5b_Branch_2_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.3032 (x.3033: f32[], y.3034: f32[]) -> f32[] {
  %x.3033 = f32[] parameter(0)
  %y.3034 = f32[] parameter(1)
  ROOT %add.3035 = f32[] add(f32[] %x.3033, f32[] %y.3034)
}

%RGB_inception_i3d_Mixed_5b_Branch_2_Conv3d_0a_3x3_batch_norm_normalize_moments_mean-reduction.3068 (x.3069: f32[], y.3070: f32[]) -> f32[] {
  %x.3069 = f32[] parameter(0)
  %y.3070 = f32[] parameter(1)
  ROOT %add.3071 = f32[] add(f32[] %x.3069, f32[] %y.3070)
}

%RGB_inception_i3d_Mixed_5b_Branch_2_Conv3d_0a_3x3_batch_norm_normalize_moments_variance-reduction.3092 (x.3093: f32[], y.3094: f32[]) -> f32[] {
  %x.3093 = f32[] parameter(0)
  %y.3094 = f32[] parameter(1)
  ROOT %add.3095 = f32[] add(f32[] %x.3093, f32[] %y.3094)
}

%max_F32.3122 (lhs.3123: f32[], rhs.3124: f32[]) -> f32[] {
  %lhs.3123 = f32[] parameter(0)
  %rhs.3124 = f32[] parameter(1)
  ROOT %maximum.3125 = f32[] maximum(f32[] %lhs.3123, f32[] %rhs.3124)
}

%RGB_inception_i3d_Mixed_5b_Branch_3_Conv3d_0b_1x1_batch_norm_normalize_moments_mean-reduction.3131 (x.3132: f32[], y.3133: f32[]) -> f32[] {
  %x.3132 = f32[] parameter(0)
  %y.3133 = f32[] parameter(1)
  ROOT %add.3134 = f32[] add(f32[] %x.3132, f32[] %y.3133)
}

%RGB_inception_i3d_Mixed_5b_Branch_3_Conv3d_0b_1x1_batch_norm_normalize_moments_variance-reduction.3155 (x.3156: f32[], y.3157: f32[]) -> f32[] {
  %x.3156 = f32[] parameter(0)
  %y.3157 = f32[] parameter(1)
  ROOT %add.3158 = f32[] add(f32[] %x.3156, f32[] %y.3157)
}

%RGB_inception_i3d_Mixed_5c_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.3192 (x.3193: f32[], y.3194: f32[]) -> f32[] {
  %x.3193 = f32[] parameter(0)
  %y.3194 = f32[] parameter(1)
  ROOT %add.3195 = f32[] add(f32[] %x.3193, f32[] %y.3194)
}

%RGB_inception_i3d_Mixed_5c_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.3216 (x.3217: f32[], y.3218: f32[]) -> f32[] {
  %x.3217 = f32[] parameter(0)
  %y.3218 = f32[] parameter(1)
  ROOT %add.3219 = f32[] add(f32[] %x.3217, f32[] %y.3218)
}

%max_F32.3246 (lhs.3247: f32[], rhs.3248: f32[]) -> f32[] {
  %lhs.3247 = f32[] parameter(0)
  %rhs.3248 = f32[] parameter(1)
  ROOT %maximum.3249 = f32[] maximum(f32[] %lhs.3247, f32[] %rhs.3248)
}

%RGB_inception_i3d_Mixed_5c_Branch_1_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.3255 (x.3256: f32[], y.3257: f32[]) -> f32[] {
  %x.3256 = f32[] parameter(0)
  %y.3257 = f32[] parameter(1)
  ROOT %add.3258 = f32[] add(f32[] %x.3256, f32[] %y.3257)
}

%RGB_inception_i3d_Mixed_5c_Branch_1_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.3279 (x.3280: f32[], y.3281: f32[]) -> f32[] {
  %x.3280 = f32[] parameter(0)
  %y.3281 = f32[] parameter(1)
  ROOT %add.3282 = f32[] add(f32[] %x.3280, f32[] %y.3281)
}

%RGB_inception_i3d_Mixed_5c_Branch_1_Conv3d_0b_3x3_batch_norm_normalize_moments_mean-reduction.3315 (x.3316: f32[], y.3317: f32[]) -> f32[] {
  %x.3316 = f32[] parameter(0)
  %y.3317 = f32[] parameter(1)
  ROOT %add.3318 = f32[] add(f32[] %x.3316, f32[] %y.3317)
}

%RGB_inception_i3d_Mixed_5c_Branch_1_Conv3d_0b_3x3_batch_norm_normalize_moments_variance-reduction.3339 (x.3340: f32[], y.3341: f32[]) -> f32[] {
  %x.3340 = f32[] parameter(0)
  %y.3341 = f32[] parameter(1)
  ROOT %add.3342 = f32[] add(f32[] %x.3340, f32[] %y.3341)
}

%RGB_inception_i3d_Mixed_5c_Branch_2_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.3372 (x.3373: f32[], y.3374: f32[]) -> f32[] {
  %x.3373 = f32[] parameter(0)
  %y.3374 = f32[] parameter(1)
  ROOT %add.3375 = f32[] add(f32[] %x.3373, f32[] %y.3374)
}

%RGB_inception_i3d_Mixed_5c_Branch_2_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.3396 (x.3397: f32[], y.3398: f32[]) -> f32[] {
  %x.3397 = f32[] parameter(0)
  %y.3398 = f32[] parameter(1)
  ROOT %add.3399 = f32[] add(f32[] %x.3397, f32[] %y.3398)
}

%RGB_inception_i3d_Mixed_5c_Branch_2_Conv3d_0b_3x3_batch_norm_normalize_moments_mean-reduction.3432 (x.3433: f32[], y.3434: f32[]) -> f32[] {
  %x.3433 = f32[] parameter(0)
  %y.3434 = f32[] parameter(1)
  ROOT %add.3435 = f32[] add(f32[] %x.3433, f32[] %y.3434)
}

%RGB_inception_i3d_Mixed_5c_Branch_2_Conv3d_0b_3x3_batch_norm_normalize_moments_variance-reduction.3456 (x.3457: f32[], y.3458: f32[]) -> f32[] {
  %x.3457 = f32[] parameter(0)
  %y.3458 = f32[] parameter(1)
  ROOT %add.3459 = f32[] add(f32[] %x.3457, f32[] %y.3458)
}

%RGB_inception_i3d_Mixed_5c_Branch_3_Conv3d_0b_1x1_batch_norm_normalize_moments_mean-reduction.3489 (x.3490: f32[], y.3491: f32[]) -> f32[] {
  %x.3490 = f32[] parameter(0)
  %y.3491 = f32[] parameter(1)
  ROOT %add.3492 = f32[] add(f32[] %x.3490, f32[] %y.3491)
}

%RGB_inception_i3d_Mixed_5c_Branch_3_Conv3d_0b_1x1_batch_norm_normalize_moments_variance-reduction.3513 (x.3514: f32[], y.3515: f32[]) -> f32[] {
  %x.3514 = f32[] parameter(0)
  %y.3515 = f32[] parameter(1)
  ROOT %add.3516 = f32[] add(f32[] %x.3514, f32[] %y.3515)
}

%add_F32.3550 (lhs.3551: f32[], rhs.3552: f32[]) -> f32[] {
  %lhs.3551 = f32[] parameter(0)
  %rhs.3552 = f32[] parameter(1)
  ROOT %add.3553 = f32[] add(f32[] %lhs.3551, f32[] %rhs.3552)
}

%RGB_inception_i3d_Mean-reduction.3567 (x.3568: f32[], y.3569: f32[]) -> f32[] {
  %x.3568 = f32[] parameter(0)
  %y.3569 = f32[] parameter(1)
  ROOT %add.3570 = f32[] add(f32[] %x.3568, f32[] %y.3569)
}

ENTRY %cluster_1__XlaCompiledKernel_true__XlaNumConstantArgs_125__XlaNumResourceArgs_116_.3580 (arg0.1: f32[1,32,224,224,3], arg1.2: f32[7,7,7,3,64], arg2.3: f32[3,3,3,16,48], arg3.4: f32[1,1,1,832,160], arg4.5: f32[1,1,1,1,64], arg5.6: f32[1,1,1,1,48], arg6.7: f32[1,1,1,1,160], arg7.8: f32[1,1,1,64,64], arg8.9: f32[1,1,1,512,64], arg9.10: f32[1,1,1,1,64], arg10.11: f32[1,1,1,1,192], arg11.12: f32[1,1,1,480,64], arg12.13: f32[1,1,1,1,64], arg13.14: f32[3,3,3,64,192], arg14.15: f32[1,1,1,1,64], arg15.16: f32[1,1,1,832,128], arg16.17: f32[1,1,1,1,192], arg17.18: f32[1,1,1,1,160], arg18.19: f32[3,3,3,160,320], arg19.20: f32[1,1,1,192,64], arg20.21: f32[1,1,1,512,112], arg21.22: f32[1,1,1,512,160], arg22.23: f32[1,1,1,1,64], arg23.24: f32[1,1,1,1,160], arg24.25: f32[1,1,1,192,96], arg25.26: f32[1,1,1,1,112], arg26.27: f32[1,1,1,1,128], arg27.28: f32[1,1,1,1,96], arg28.29: f32[1,1,1,1,320], arg29.30: f32[1,1,1,512,112], arg30.31: f32[3,3,3,96,128], arg31.32: f32[1,1,1,1,112], arg32.33: f32[1,1,1,1,128], arg33.34: f32[1,1,1,1,128], arg34.35: f32[1,1,1,512,144], arg35.36: f32[3,3,3,160,320], arg36.37: f32[1,1,1,528,32], arg37.38: f32[1,1,1,192,16], arg38.39: f32[3,3,3,112,224], arg39.40: f32[1,1,1,1,144], arg40.41: f32[1,1,1,1,16], arg41.42: f32[1,1,1,1,48], arg42.43: f32[1,1,1,1,224], arg43.44: f32[3,3,3,16,32], arg44.45: f32[1,1,1,1,32], arg45.46: f32[1,1,1,1,32], arg46.47: f32[1,1,1,1,320], arg47.48: f32[3,3,3,144,288], arg48.49: f32[1,1,1,512,24], arg49.50: f32[3,3,3,192,384], arg50.51: f32[1,1,1,192,32], arg51.52: f32[1,1,1,1,24], arg52.53: f32[1,1,1,1,32], arg53.54: f32[1,1,1,1,288], arg54.55: f32[1,1,1,256,128], arg55.56: f32[3,3,3,24,64], arg56.57: f32[3,3,3,32,128], arg57.58: f32[1,1,1,832,384], arg58.59: f32[1,1,1,1,128], arg59.60: f32[1,1,1,1,64], arg60.61: f32[1,1,1,512,32], arg61.62: f32[1,1,1,256,128], arg62.63: f32[400], arg63.64: f32[1,1,1,832,32], arg64.65: f32[1,1,1,1,128], arg65.66: f32[1,1,1,512,64], arg66.67: f32[1,1,1,1,32], arg67.68: f32[1,1,1,1,128], arg68.69: f32[3,3,3,128,192], arg69.70: f32[1,1,1,1,64], arg70.71: f32[1,1,1,1,192], arg71.72: f32[1,1,1,1,384], arg72.73: f32[3,3,3,32,64], arg73.74: f32[1,1,1,256,32], arg74.75: f32[1,1,1,512,128], arg75.76: f32[1,1,1,1,32], arg76.77: f32[1,1,1,1,384], arg77.78: f32[1,1,1,1,32], arg78.79: f32[1,1,1,528,128], arg79.80: f32[1,1,1,1,128], arg80.81: f32[1,1,1,1,64], arg81.82: f32[3,3,3,32,96], arg82.83: f32[1,1,1,1,96], arg83.84: f32[1,1,1,512,128], arg84.85: f32[1,1,1,1,128], arg85.86: f32[1,1,1,256,64], arg86.87: f32[1,1,1,1,128], arg87.88: f32[1,1,1,512,64], arg88.89: f32[1,1,1,1,64], arg89.90: f32[1,1,1,1,64], arg90.91: f32[1,1,1,480,192], arg91.92: f32[3,3,3,128,256], arg92.93: f32[1,1,1,1,192], arg93.94: f32[1,1,1,1,128], arg94.95: f32[3,3,3,32,128], arg95.96: f32[1,1,1,1024,400], arg96.97: f32[1,1,1,1,256], arg97.98: f32[1,1,1,480,96], arg98.99: f32[3,3,3,48,128], arg99.100: f32[1,1,1,832,256], arg100.101: f32[1,1,1,832,192], arg101.102: f32[1,1,1,1,96], arg102.103: f32[1,1,1,528,256], arg103.104: f32[1,1,1,512,24], arg104.105: f32[3,3,3,96,208], arg105.106: f32[1,1,1,1,24], arg106.107: f32[1,1,1,1,208], arg107.108: f32[1,1,1,1,256], arg108.109: f32[1,1,1,1,128], arg109.110: f32[1,1,1,1,256], arg110.111: f32[3,3,3,24,64], arg111.112: f32[1,1,1,480,16], arg112.113: f32[1,1,1,832,128], arg113.114: f32[1,1,1,1,16], arg114.115: f32[1,1,1,1,64], arg115.116: f32[1,1,1,528,160], arg116.117: f32[1,1,1,832,48]) -> f32[1,400] {
  %arg62.63 = f32[400]{0} parameter(62), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.120 = f32[1,1,1,1,400]{4,3,2,1,0} reshape(f32[400]{0} %arg62.63), metadata={op_type="Reshape" op_name="RGB/inception_i3d/Logits/Conv3d_0c_1x1/conv_3d/Reshape"}
  %reshape.3560 = f32[1,1,1,400]{3,2,1,0} reshape(f32[1,1,1,1,400]{4,3,2,1,0} %reshape.120), metadata={op_type="Add" op_name="RGB/inception_i3d/Logits/Conv3d_0c_1x1/conv_3d/add"}
  %broadcast.3561 = f32[1,3,1,1,400]{4,3,2,1,0} broadcast(f32[1,1,1,400]{3,2,1,0} %reshape.3560), dimensions={0,2,3,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Logits/Conv3d_0c_1x1/conv_3d/add"}
  %constant.3543 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/Relu"}
  %broadcast.3544 = f32[1,4,7,7,1024]{4,3,2,1,0} broadcast(f32[] %constant.3543), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/Relu"}
  %constant.3233 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.3234 = f32[1,1,1,1,384]{4,3,2,1,0} broadcast(f32[] %constant.3233), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %constant.3185 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/Relu"}
  %broadcast.3186 = f32[1,4,7,7,832]{4,3,2,1,0} broadcast(f32[] %constant.3185), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/Relu"}
  %constant.2875 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.2876 = f32[1,1,1,1,256]{4,3,2,1,0} broadcast(f32[] %constant.2875), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %constant.2827 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/Relu"}
  %broadcast.2828 = f32[1,4,7,7,832]{4,3,2,1,0} broadcast(f32[] %constant.2827), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/Relu"}
  %constant.2511 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.2512 = f32[1,1,1,1,256]{4,3,2,1,0} broadcast(f32[] %constant.2511), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %constant.2463 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/Relu"}
  %broadcast.2464 = f32[1,8,14,14,528]{4,3,2,1,0} broadcast(f32[] %constant.2463), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/Relu"}
  %constant.2153 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.2154 = f32[1,1,1,1,112]{4,3,2,1,0} broadcast(f32[] %constant.2153), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %constant.2105 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/Relu"}
  %broadcast.2106 = f32[1,8,14,14,512]{4,3,2,1,0} broadcast(f32[] %constant.2105), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/Relu"}
  %constant.1795 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.1796 = f32[1,1,1,1,128]{4,3,2,1,0} broadcast(f32[] %constant.1795), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %constant.1747 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/Relu"}
  %broadcast.1748 = f32[1,8,14,14,512]{4,3,2,1,0} broadcast(f32[] %constant.1747), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/Relu"}
  %constant.1437 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.1438 = f32[1,1,1,1,160]{4,3,2,1,0} broadcast(f32[] %constant.1437), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %constant.1389 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/Relu"}
  %broadcast.1390 = f32[1,8,14,14,512]{4,3,2,1,0} broadcast(f32[] %constant.1389), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/Relu"}
  %constant.1079 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.1080 = f32[1,1,1,1,192]{4,3,2,1,0} broadcast(f32[] %constant.1079), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %constant.1031 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/Relu"}
  %broadcast.1032 = f32[1,8,14,14,480]{4,3,2,1,0} broadcast(f32[] %constant.1031), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/Relu"}
  %constant.715 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.716 = f32[1,1,1,1,128]{4,3,2,1,0} broadcast(f32[] %constant.715), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %constant.667 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/Relu"}
  %broadcast.668 = f32[1,16,28,28,256]{4,3,2,1,0} broadcast(f32[] %constant.667), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/Relu"}
  %constant.654 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.655 = f32[1,1,1,1,64]{4,3,2,1,0} broadcast(f32[] %constant.654), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %constant.309 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Conv3d_2c_3x3/Relu"}
  %broadcast.310 = f32[1,16,28,28,192]{4,3,2,1,0} broadcast(f32[] %constant.309), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Conv3d_2c_3x3/Relu"}
  %constant.291 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/batch_norm/add"}
  %broadcast.292 = f32[1,1,1,1,192]{4,3,2,1,0} broadcast(f32[] %constant.291), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/batch_norm/add"}
  %constant.243 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Conv3d_2b_1x1/Relu"}
  %broadcast.244 = f32[1,16,56,56,64]{4,3,2,1,0} broadcast(f32[] %constant.243), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Conv3d_2b_1x1/Relu"}
  %constant.231 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/batch_norm/add"}
  %broadcast.232 = f32[1,1,1,1,64]{4,3,2,1,0} broadcast(f32[] %constant.231), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/batch_norm/add"}
  %constant.183 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Conv3d_1a_7x7/Relu"}
  %broadcast.184 = f32[1,16,56,56,64]{4,3,2,1,0} broadcast(f32[] %constant.183), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Conv3d_1a_7x7/Relu"}
  %constant.165 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/add"}
  %broadcast.166 = f32[1,1,1,1,64]{4,3,2,1,0} broadcast(f32[] %constant.165), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/add"}
  %arg0.1 = f32[1,32,224,224,3]{4,3,2,1,0} parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.118 = f32[1,32,224,224,3]{4,3,2,1,0} reshape(f32[1,32,224,224,3]{4,3,2,1,0} %arg0.1)
  %arg1.2 = f32[7,7,7,3,64]{4,3,2,1,0} parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.119 = f32[1,16,112,112,64]{4,3,2,1,0} convolution(f32[1,32,224,224,3]{4,3,2,1,0} %reshape.118, f32[7,7,7,3,64]{4,3,2,1,0} %arg1.2), window={size=7x7x7 stride=2x2x2 pad=2_3x2_3x2_3}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Conv3d_1a_7x7/conv_3d/convolution"}
  %convert.121 = f32[1,16,112,112,64]{4,3,2,1,0} convert(f32[1,16,112,112,64]{4,3,2,1,0} %convolution.119), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/mean"}
  %constant.122 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/mean"}
  %convert.123 = f32[] convert(f32[] %constant.122), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/mean"}
  %reduce.128 = f32[64]{0} reduce(f32[1,16,112,112,64]{4,3,2,1,0} %convert.121, f32[] %convert.123), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Conv3d_1a_7x7_batch_norm_normalize_moments_mean-reduction.124, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/mean"}
  %get-dimension-size.129 = s32[] get-dimension-size(f32[1,16,112,112,64]{4,3,2,1,0} %convert.121), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/mean"}
  %get-dimension-size.130 = s32[] get-dimension-size(f32[1,16,112,112,64]{4,3,2,1,0} %convert.121), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/mean"}
  %multiply.131 = s32[] multiply(s32[] %get-dimension-size.129, s32[] %get-dimension-size.130), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/mean"}
  %get-dimension-size.132 = s32[] get-dimension-size(f32[1,16,112,112,64]{4,3,2,1,0} %convert.121), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/mean"}
  %multiply.133 = s32[] multiply(s32[] %multiply.131, s32[] %get-dimension-size.132), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/mean"}
  %get-dimension-size.134 = s32[] get-dimension-size(f32[1,16,112,112,64]{4,3,2,1,0} %convert.121), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/mean"}
  %multiply.135 = s32[] multiply(s32[] %multiply.133, s32[] %get-dimension-size.134), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/mean"}
  %convert.136 = f32[] convert(s32[] %multiply.135), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/mean"}
  %broadcast.137 = f32[64]{0} broadcast(f32[] %convert.136), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/mean"}
  %divide.138 = f32[64]{0} divide(f32[64]{0} %reduce.128, f32[64]{0} %broadcast.137), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/mean"}
  %convert.139 = f32[64]{0} convert(f32[64]{0} %divide.138), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/mean"}
  %reshape.140 = f32[1,1,1,1,64]{4,3,2,1,0} reshape(f32[64]{0} %convert.139), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/mean"}
  %reshape.141 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %reshape.140), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.142 = f32[1,16,112,112,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.141), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.143 = f32[1,16,112,112,64]{4,3,2,1,0} subtract(f32[1,16,112,112,64]{4,3,2,1,0} %broadcast.142, f32[1,16,112,112,64]{4,3,2,1,0} %convolution.119), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.144 = f32[1,16,112,112,64]{4,3,2,1,0} multiply(f32[1,16,112,112,64]{4,3,2,1,0} %subtract.143, f32[1,16,112,112,64]{4,3,2,1,0} %subtract.143), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/SquaredDifference"}
  %convert.145 = f32[1,16,112,112,64]{4,3,2,1,0} convert(f32[1,16,112,112,64]{4,3,2,1,0} %multiply.144), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/variance"}
  %constant.146 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/variance"}
  %convert.147 = f32[] convert(f32[] %constant.146), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/variance"}
  %reduce.152 = f32[64]{0} reduce(f32[1,16,112,112,64]{4,3,2,1,0} %convert.145, f32[] %convert.147), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Conv3d_1a_7x7_batch_norm_normalize_moments_variance-reduction.148, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/variance"}
  %get-dimension-size.153 = s32[] get-dimension-size(f32[1,16,112,112,64]{4,3,2,1,0} %convert.145), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/variance"}
  %get-dimension-size.154 = s32[] get-dimension-size(f32[1,16,112,112,64]{4,3,2,1,0} %convert.145), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/variance"}
  %multiply.155 = s32[] multiply(s32[] %get-dimension-size.153, s32[] %get-dimension-size.154), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/variance"}
  %get-dimension-size.156 = s32[] get-dimension-size(f32[1,16,112,112,64]{4,3,2,1,0} %convert.145), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/variance"}
  %multiply.157 = s32[] multiply(s32[] %multiply.155, s32[] %get-dimension-size.156), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/variance"}
  %get-dimension-size.158 = s32[] get-dimension-size(f32[1,16,112,112,64]{4,3,2,1,0} %convert.145), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/variance"}
  %multiply.159 = s32[] multiply(s32[] %multiply.157, s32[] %get-dimension-size.158), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/variance"}
  %convert.160 = f32[] convert(s32[] %multiply.159), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/variance"}
  %broadcast.161 = f32[64]{0} broadcast(f32[] %convert.160), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/variance"}
  %divide.162 = f32[64]{0} divide(f32[64]{0} %reduce.152, f32[64]{0} %broadcast.161), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/variance"}
  %convert.163 = f32[64]{0} convert(f32[64]{0} %divide.162), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/variance"}
  %reshape.164 = f32[1,1,1,1,64]{4,3,2,1,0} reshape(f32[64]{0} %convert.163), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/normalize_moments/variance"}
  %add.167 = f32[1,1,1,1,64]{4,3,2,1,0} add(f32[1,1,1,1,64]{4,3,2,1,0} %broadcast.166, f32[1,1,1,1,64]{4,3,2,1,0} %reshape.164), metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/add"}
  %rsqrt.168 = f32[1,1,1,1,64]{4,3,2,1,0} rsqrt(f32[1,1,1,1,64]{4,3,2,1,0} %add.167), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/Rsqrt"}
  %reshape.169 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.168), metadata={op_type="Mul" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/mul"}
  %broadcast.170 = f32[1,16,112,112,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.169), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/mul"}
  %multiply.171 = f32[1,16,112,112,64]{4,3,2,1,0} multiply(f32[1,16,112,112,64]{4,3,2,1,0} %broadcast.170, f32[1,16,112,112,64]{4,3,2,1,0} %convolution.119), metadata={op_type="Mul" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/mul"}
  %arg4.5 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(4), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.172 = f32[1,1,1,1,64]{4,3,2,1,0} multiply(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.168, f32[1,1,1,1,64]{4,3,2,1,0} %reshape.140), metadata={op_type="Mul" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/mul_1"}
  %subtract.173 = f32[1,1,1,1,64]{4,3,2,1,0} subtract(f32[1,1,1,1,64]{4,3,2,1,0} %arg4.5, f32[1,1,1,1,64]{4,3,2,1,0} %multiply.172), metadata={op_type="Sub" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/sub"}
  %reshape.174 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %subtract.173), metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/add_1"}
  %broadcast.175 = f32[1,16,112,112,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.174), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/add_1"}
  %add.176 = f32[1,16,112,112,64]{4,3,2,1,0} add(f32[1,16,112,112,64]{4,3,2,1,0} %multiply.171, f32[1,16,112,112,64]{4,3,2,1,0} %broadcast.175), metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/batch_norm/add_1"}
  %constant.177 = f32[] constant(-inf), metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/MaxPool3d_2a_3x3"}
  %reduce-window.182 = f32[1,16,56,56,64]{4,3,2,1,0} reduce-window(f32[1,16,112,112,64]{4,3,2,1,0} %add.176, f32[] %constant.177), window={size=1x1x3x3x1 stride=1x1x2x2x1 pad=0_0x0_0x0_1x0_1x0_0}, to_apply=%max_F32.178, metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/MaxPool3d_2a_3x3"}
  %maximum.185 = f32[1,16,56,56,64]{4,3,2,1,0} maximum(f32[1,16,56,56,64]{4,3,2,1,0} %broadcast.184, f32[1,16,56,56,64]{4,3,2,1,0} %reduce-window.182), metadata={op_type="Relu" op_name="RGB/inception_i3d/Conv3d_1a_7x7/Relu"}
  %arg7.8 = f32[1,1,1,64,64]{4,3,2,1,0} parameter(7), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.186 = f32[1,16,56,56,64]{4,3,2,1,0} convolution(f32[1,16,56,56,64]{4,3,2,1,0} %maximum.185, f32[1,1,1,64,64]{4,3,2,1,0} %arg7.8), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Conv3d_2b_1x1/conv_3d/convolution"}
  %convert.187 = f32[1,16,56,56,64]{4,3,2,1,0} convert(f32[1,16,56,56,64]{4,3,2,1,0} %convolution.186), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/mean"}
  %constant.188 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/mean"}
  %convert.189 = f32[] convert(f32[] %constant.188), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/mean"}
  %reduce.194 = f32[64]{0} reduce(f32[1,16,56,56,64]{4,3,2,1,0} %convert.187, f32[] %convert.189), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Conv3d_2b_1x1_batch_norm_normalize_moments_mean-reduction.190, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.195 = s32[] get-dimension-size(f32[1,16,56,56,64]{4,3,2,1,0} %convert.187), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.196 = s32[] get-dimension-size(f32[1,16,56,56,64]{4,3,2,1,0} %convert.187), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/mean"}
  %multiply.197 = s32[] multiply(s32[] %get-dimension-size.195, s32[] %get-dimension-size.196), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.198 = s32[] get-dimension-size(f32[1,16,56,56,64]{4,3,2,1,0} %convert.187), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/mean"}
  %multiply.199 = s32[] multiply(s32[] %multiply.197, s32[] %get-dimension-size.198), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.200 = s32[] get-dimension-size(f32[1,16,56,56,64]{4,3,2,1,0} %convert.187), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/mean"}
  %multiply.201 = s32[] multiply(s32[] %multiply.199, s32[] %get-dimension-size.200), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/mean"}
  %convert.202 = f32[] convert(s32[] %multiply.201), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.203 = f32[64]{0} broadcast(f32[] %convert.202), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/mean"}
  %divide.204 = f32[64]{0} divide(f32[64]{0} %reduce.194, f32[64]{0} %broadcast.203), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/mean"}
  %convert.205 = f32[64]{0} convert(f32[64]{0} %divide.204), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/mean"}
  %reshape.206 = f32[1,1,1,1,64]{4,3,2,1,0} reshape(f32[64]{0} %convert.205), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/mean"}
  %reshape.207 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %reshape.206), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.208 = f32[1,16,56,56,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.207), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.209 = f32[1,16,56,56,64]{4,3,2,1,0} subtract(f32[1,16,56,56,64]{4,3,2,1,0} %broadcast.208, f32[1,16,56,56,64]{4,3,2,1,0} %convolution.186), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.210 = f32[1,16,56,56,64]{4,3,2,1,0} multiply(f32[1,16,56,56,64]{4,3,2,1,0} %subtract.209, f32[1,16,56,56,64]{4,3,2,1,0} %subtract.209), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.211 = f32[1,16,56,56,64]{4,3,2,1,0} convert(f32[1,16,56,56,64]{4,3,2,1,0} %multiply.210), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/variance"}
  %constant.212 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/variance"}
  %convert.213 = f32[] convert(f32[] %constant.212), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/variance"}
  %reduce.218 = f32[64]{0} reduce(f32[1,16,56,56,64]{4,3,2,1,0} %convert.211, f32[] %convert.213), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Conv3d_2b_1x1_batch_norm_normalize_moments_variance-reduction.214, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.219 = s32[] get-dimension-size(f32[1,16,56,56,64]{4,3,2,1,0} %convert.211), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.220 = s32[] get-dimension-size(f32[1,16,56,56,64]{4,3,2,1,0} %convert.211), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/variance"}
  %multiply.221 = s32[] multiply(s32[] %get-dimension-size.219, s32[] %get-dimension-size.220), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.222 = s32[] get-dimension-size(f32[1,16,56,56,64]{4,3,2,1,0} %convert.211), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/variance"}
  %multiply.223 = s32[] multiply(s32[] %multiply.221, s32[] %get-dimension-size.222), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.224 = s32[] get-dimension-size(f32[1,16,56,56,64]{4,3,2,1,0} %convert.211), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/variance"}
  %multiply.225 = s32[] multiply(s32[] %multiply.223, s32[] %get-dimension-size.224), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/variance"}
  %convert.226 = f32[] convert(s32[] %multiply.225), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.227 = f32[64]{0} broadcast(f32[] %convert.226), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/variance"}
  %divide.228 = f32[64]{0} divide(f32[64]{0} %reduce.218, f32[64]{0} %broadcast.227), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/variance"}
  %convert.229 = f32[64]{0} convert(f32[64]{0} %divide.228), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/variance"}
  %reshape.230 = f32[1,1,1,1,64]{4,3,2,1,0} reshape(f32[64]{0} %convert.229), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/normalize_moments/variance"}
  %add.233 = f32[1,1,1,1,64]{4,3,2,1,0} add(f32[1,1,1,1,64]{4,3,2,1,0} %broadcast.232, f32[1,1,1,1,64]{4,3,2,1,0} %reshape.230), metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/batch_norm/add"}
  %rsqrt.234 = f32[1,1,1,1,64]{4,3,2,1,0} rsqrt(f32[1,1,1,1,64]{4,3,2,1,0} %add.233), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.235 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.234), metadata={op_type="Mul" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/batch_norm/mul"}
  %broadcast.236 = f32[1,16,56,56,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.235), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/batch_norm/mul"}
  %multiply.237 = f32[1,16,56,56,64]{4,3,2,1,0} multiply(f32[1,16,56,56,64]{4,3,2,1,0} %broadcast.236, f32[1,16,56,56,64]{4,3,2,1,0} %convolution.186), metadata={op_type="Mul" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/batch_norm/mul"}
  %arg9.10 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(9), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.238 = f32[1,1,1,1,64]{4,3,2,1,0} multiply(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.234, f32[1,1,1,1,64]{4,3,2,1,0} %reshape.206), metadata={op_type="Mul" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.239 = f32[1,1,1,1,64]{4,3,2,1,0} subtract(f32[1,1,1,1,64]{4,3,2,1,0} %arg9.10, f32[1,1,1,1,64]{4,3,2,1,0} %multiply.238), metadata={op_type="Sub" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/batch_norm/sub"}
  %reshape.240 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %subtract.239), metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.241 = f32[1,16,56,56,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.240), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/batch_norm/add_1"}
  %add.242 = f32[1,16,56,56,64]{4,3,2,1,0} add(f32[1,16,56,56,64]{4,3,2,1,0} %multiply.237, f32[1,16,56,56,64]{4,3,2,1,0} %broadcast.241), metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/batch_norm/add_1"}
  %maximum.245 = f32[1,16,56,56,64]{4,3,2,1,0} maximum(f32[1,16,56,56,64]{4,3,2,1,0} %broadcast.244, f32[1,16,56,56,64]{4,3,2,1,0} %add.242), metadata={op_type="Relu" op_name="RGB/inception_i3d/Conv3d_2b_1x1/Relu"}
  %arg13.14 = f32[3,3,3,64,192]{4,3,2,1,0} parameter(13), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.246 = f32[1,16,56,56,192]{4,3,2,1,0} convolution(f32[1,16,56,56,64]{4,3,2,1,0} %maximum.245, f32[3,3,3,64,192]{4,3,2,1,0} %arg13.14), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Conv3d_2c_3x3/conv_3d/convolution"}
  %convert.247 = f32[1,16,56,56,192]{4,3,2,1,0} convert(f32[1,16,56,56,192]{4,3,2,1,0} %convolution.246), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/mean"}
  %constant.248 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/mean"}
  %convert.249 = f32[] convert(f32[] %constant.248), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/mean"}
  %reduce.254 = f32[192]{0} reduce(f32[1,16,56,56,192]{4,3,2,1,0} %convert.247, f32[] %convert.249), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Conv3d_2c_3x3_batch_norm_normalize_moments_mean-reduction.250, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.255 = s32[] get-dimension-size(f32[1,16,56,56,192]{4,3,2,1,0} %convert.247), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.256 = s32[] get-dimension-size(f32[1,16,56,56,192]{4,3,2,1,0} %convert.247), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/mean"}
  %multiply.257 = s32[] multiply(s32[] %get-dimension-size.255, s32[] %get-dimension-size.256), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.258 = s32[] get-dimension-size(f32[1,16,56,56,192]{4,3,2,1,0} %convert.247), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/mean"}
  %multiply.259 = s32[] multiply(s32[] %multiply.257, s32[] %get-dimension-size.258), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.260 = s32[] get-dimension-size(f32[1,16,56,56,192]{4,3,2,1,0} %convert.247), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/mean"}
  %multiply.261 = s32[] multiply(s32[] %multiply.259, s32[] %get-dimension-size.260), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/mean"}
  %convert.262 = f32[] convert(s32[] %multiply.261), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/mean"}
  %broadcast.263 = f32[192]{0} broadcast(f32[] %convert.262), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/mean"}
  %divide.264 = f32[192]{0} divide(f32[192]{0} %reduce.254, f32[192]{0} %broadcast.263), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/mean"}
  %convert.265 = f32[192]{0} convert(f32[192]{0} %divide.264), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/mean"}
  %reshape.266 = f32[1,1,1,1,192]{4,3,2,1,0} reshape(f32[192]{0} %convert.265), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/mean"}
  %reshape.267 = f32[1,192]{1,0} reshape(f32[1,1,1,1,192]{4,3,2,1,0} %reshape.266), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.268 = f32[1,16,56,56,192]{4,3,2,1,0} broadcast(f32[1,192]{1,0} %reshape.267), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.269 = f32[1,16,56,56,192]{4,3,2,1,0} subtract(f32[1,16,56,56,192]{4,3,2,1,0} %broadcast.268, f32[1,16,56,56,192]{4,3,2,1,0} %convolution.246), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.270 = f32[1,16,56,56,192]{4,3,2,1,0} multiply(f32[1,16,56,56,192]{4,3,2,1,0} %subtract.269, f32[1,16,56,56,192]{4,3,2,1,0} %subtract.269), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %convert.271 = f32[1,16,56,56,192]{4,3,2,1,0} convert(f32[1,16,56,56,192]{4,3,2,1,0} %multiply.270), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/variance"}
  %constant.272 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/variance"}
  %convert.273 = f32[] convert(f32[] %constant.272), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/variance"}
  %reduce.278 = f32[192]{0} reduce(f32[1,16,56,56,192]{4,3,2,1,0} %convert.271, f32[] %convert.273), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Conv3d_2c_3x3_batch_norm_normalize_moments_variance-reduction.274, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.279 = s32[] get-dimension-size(f32[1,16,56,56,192]{4,3,2,1,0} %convert.271), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.280 = s32[] get-dimension-size(f32[1,16,56,56,192]{4,3,2,1,0} %convert.271), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/variance"}
  %multiply.281 = s32[] multiply(s32[] %get-dimension-size.279, s32[] %get-dimension-size.280), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.282 = s32[] get-dimension-size(f32[1,16,56,56,192]{4,3,2,1,0} %convert.271), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/variance"}
  %multiply.283 = s32[] multiply(s32[] %multiply.281, s32[] %get-dimension-size.282), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.284 = s32[] get-dimension-size(f32[1,16,56,56,192]{4,3,2,1,0} %convert.271), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/variance"}
  %multiply.285 = s32[] multiply(s32[] %multiply.283, s32[] %get-dimension-size.284), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/variance"}
  %convert.286 = f32[] convert(s32[] %multiply.285), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/variance"}
  %broadcast.287 = f32[192]{0} broadcast(f32[] %convert.286), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/variance"}
  %divide.288 = f32[192]{0} divide(f32[192]{0} %reduce.278, f32[192]{0} %broadcast.287), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/variance"}
  %convert.289 = f32[192]{0} convert(f32[192]{0} %divide.288), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/variance"}
  %reshape.290 = f32[1,1,1,1,192]{4,3,2,1,0} reshape(f32[192]{0} %convert.289), metadata={op_type="Mean" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/normalize_moments/variance"}
  %add.293 = f32[1,1,1,1,192]{4,3,2,1,0} add(f32[1,1,1,1,192]{4,3,2,1,0} %broadcast.292, f32[1,1,1,1,192]{4,3,2,1,0} %reshape.290), metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/batch_norm/add"}
  %rsqrt.294 = f32[1,1,1,1,192]{4,3,2,1,0} rsqrt(f32[1,1,1,1,192]{4,3,2,1,0} %add.293), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.295 = f32[1,192]{1,0} reshape(f32[1,1,1,1,192]{4,3,2,1,0} %rsqrt.294), metadata={op_type="Mul" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/batch_norm/mul"}
  %broadcast.296 = f32[1,16,56,56,192]{4,3,2,1,0} broadcast(f32[1,192]{1,0} %reshape.295), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/batch_norm/mul"}
  %multiply.297 = f32[1,16,56,56,192]{4,3,2,1,0} multiply(f32[1,16,56,56,192]{4,3,2,1,0} %broadcast.296, f32[1,16,56,56,192]{4,3,2,1,0} %convolution.246), metadata={op_type="Mul" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/batch_norm/mul"}
  %arg16.17 = f32[1,1,1,1,192]{4,3,2,1,0} parameter(16), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.298 = f32[1,1,1,1,192]{4,3,2,1,0} multiply(f32[1,1,1,1,192]{4,3,2,1,0} %rsqrt.294, f32[1,1,1,1,192]{4,3,2,1,0} %reshape.266), metadata={op_type="Mul" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.299 = f32[1,1,1,1,192]{4,3,2,1,0} subtract(f32[1,1,1,1,192]{4,3,2,1,0} %arg16.17, f32[1,1,1,1,192]{4,3,2,1,0} %multiply.298), metadata={op_type="Sub" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/batch_norm/sub"}
  %reshape.300 = f32[1,192]{1,0} reshape(f32[1,1,1,1,192]{4,3,2,1,0} %subtract.299), metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.301 = f32[1,16,56,56,192]{4,3,2,1,0} broadcast(f32[1,192]{1,0} %reshape.300), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/batch_norm/add_1"}
  %add.302 = f32[1,16,56,56,192]{4,3,2,1,0} add(f32[1,16,56,56,192]{4,3,2,1,0} %multiply.297, f32[1,16,56,56,192]{4,3,2,1,0} %broadcast.301), metadata={op_type="Add" op_name="RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/batch_norm/add_1"}
  %constant.303 = f32[] constant(-inf), metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/MaxPool3d_3a_3x3"}
  %reduce-window.308 = f32[1,16,28,28,192]{4,3,2,1,0} reduce-window(f32[1,16,56,56,192]{4,3,2,1,0} %add.302, f32[] %constant.303), window={size=1x1x3x3x1 stride=1x1x2x2x1 pad=0_0x0_0x0_1x0_1x0_0}, to_apply=%max_F32.304, metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/MaxPool3d_3a_3x3"}
  %maximum.311 = f32[1,16,28,28,192]{4,3,2,1,0} maximum(f32[1,16,28,28,192]{4,3,2,1,0} %broadcast.310, f32[1,16,28,28,192]{4,3,2,1,0} %reduce-window.308), metadata={op_type="Relu" op_name="RGB/inception_i3d/Conv3d_2c_3x3/Relu"}
  %arg19.20 = f32[1,1,1,192,64]{4,3,2,1,0} parameter(19), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.312 = f32[1,16,28,28,64]{4,3,2,1,0} convolution(f32[1,16,28,28,192]{4,3,2,1,0} %maximum.311, f32[1,1,1,192,64]{4,3,2,1,0} %arg19.20), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/conv_3d/convolution"}
  %convert.610 = f32[1,16,28,28,64]{4,3,2,1,0} convert(f32[1,16,28,28,64]{4,3,2,1,0} %convolution.312), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %constant.611 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.612 = f32[] convert(f32[] %constant.611), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reduce.617 = f32[64]{0} reduce(f32[1,16,28,28,64]{4,3,2,1,0} %convert.610, f32[] %convert.612), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_3b_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.613, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.618 = s32[] get-dimension-size(f32[1,16,28,28,64]{4,3,2,1,0} %convert.610), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.619 = s32[] get-dimension-size(f32[1,16,28,28,64]{4,3,2,1,0} %convert.610), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.620 = s32[] multiply(s32[] %get-dimension-size.618, s32[] %get-dimension-size.619), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.621 = s32[] get-dimension-size(f32[1,16,28,28,64]{4,3,2,1,0} %convert.610), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.622 = s32[] multiply(s32[] %multiply.620, s32[] %get-dimension-size.621), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.623 = s32[] get-dimension-size(f32[1,16,28,28,64]{4,3,2,1,0} %convert.610), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.624 = s32[] multiply(s32[] %multiply.622, s32[] %get-dimension-size.623), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.625 = f32[] convert(s32[] %multiply.624), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.626 = f32[64]{0} broadcast(f32[] %convert.625), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %divide.627 = f32[64]{0} divide(f32[64]{0} %reduce.617, f32[64]{0} %broadcast.626), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.628 = f32[64]{0} convert(f32[64]{0} %divide.627), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.629 = f32[1,1,1,1,64]{4,3,2,1,0} reshape(f32[64]{0} %convert.628), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.630 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %reshape.629), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.631 = f32[1,16,28,28,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.630), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.632 = f32[1,16,28,28,64]{4,3,2,1,0} subtract(f32[1,16,28,28,64]{4,3,2,1,0} %broadcast.631, f32[1,16,28,28,64]{4,3,2,1,0} %convolution.312), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.633 = f32[1,16,28,28,64]{4,3,2,1,0} multiply(f32[1,16,28,28,64]{4,3,2,1,0} %subtract.632, f32[1,16,28,28,64]{4,3,2,1,0} %subtract.632), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.634 = f32[1,16,28,28,64]{4,3,2,1,0} convert(f32[1,16,28,28,64]{4,3,2,1,0} %multiply.633), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %constant.635 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.636 = f32[] convert(f32[] %constant.635), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reduce.641 = f32[64]{0} reduce(f32[1,16,28,28,64]{4,3,2,1,0} %convert.634, f32[] %convert.636), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_3b_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.637, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.642 = s32[] get-dimension-size(f32[1,16,28,28,64]{4,3,2,1,0} %convert.634), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.643 = s32[] get-dimension-size(f32[1,16,28,28,64]{4,3,2,1,0} %convert.634), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.644 = s32[] multiply(s32[] %get-dimension-size.642, s32[] %get-dimension-size.643), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.645 = s32[] get-dimension-size(f32[1,16,28,28,64]{4,3,2,1,0} %convert.634), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.646 = s32[] multiply(s32[] %multiply.644, s32[] %get-dimension-size.645), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.647 = s32[] get-dimension-size(f32[1,16,28,28,64]{4,3,2,1,0} %convert.634), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.648 = s32[] multiply(s32[] %multiply.646, s32[] %get-dimension-size.647), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.649 = f32[] convert(s32[] %multiply.648), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.650 = f32[64]{0} broadcast(f32[] %convert.649), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %divide.651 = f32[64]{0} divide(f32[64]{0} %reduce.641, f32[64]{0} %broadcast.650), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.652 = f32[64]{0} convert(f32[64]{0} %divide.651), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reshape.653 = f32[1,1,1,1,64]{4,3,2,1,0} reshape(f32[64]{0} %convert.652), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %add.656 = f32[1,1,1,1,64]{4,3,2,1,0} add(f32[1,1,1,1,64]{4,3,2,1,0} %broadcast.655, f32[1,1,1,1,64]{4,3,2,1,0} %reshape.653), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.657 = f32[1,1,1,1,64]{4,3,2,1,0} rsqrt(f32[1,1,1,1,64]{4,3,2,1,0} %add.656), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.658 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.657), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.659 = f32[1,16,28,28,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.658), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %multiply.660 = f32[1,16,28,28,64]{4,3,2,1,0} multiply(f32[1,16,28,28,64]{4,3,2,1,0} %broadcast.659, f32[1,16,28,28,64]{4,3,2,1,0} %convolution.312), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg22.23 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(22), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.661 = f32[1,1,1,1,64]{4,3,2,1,0} multiply(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.657, f32[1,1,1,1,64]{4,3,2,1,0} %reshape.629), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.662 = f32[1,1,1,1,64]{4,3,2,1,0} subtract(f32[1,1,1,1,64]{4,3,2,1,0} %arg22.23, f32[1,1,1,1,64]{4,3,2,1,0} %multiply.661), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.663 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %subtract.662), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.664 = f32[1,16,28,28,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.663), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.665 = f32[1,16,28,28,64]{4,3,2,1,0} add(f32[1,16,28,28,64]{4,3,2,1,0} %multiply.660, f32[1,16,28,28,64]{4,3,2,1,0} %broadcast.664), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %constant.418 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %broadcast.419 = f32[1,1,1,1,128]{4,3,2,1,0} broadcast(f32[] %constant.418), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %constant.370 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/Relu"}
  %broadcast.371 = f32[1,16,28,28,96]{4,3,2,1,0} broadcast(f32[] %constant.370), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/Relu"}
  %constant.358 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.359 = f32[1,1,1,1,96]{4,3,2,1,0} broadcast(f32[] %constant.358), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %arg24.25 = f32[1,1,1,192,96]{4,3,2,1,0} parameter(24), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.313 = f32[1,16,28,28,96]{4,3,2,1,0} convolution(f32[1,16,28,28,192]{4,3,2,1,0} %maximum.311, f32[1,1,1,192,96]{4,3,2,1,0} %arg24.25), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/conv_3d/convolution"}
  %convert.314 = f32[1,16,28,28,96]{4,3,2,1,0} convert(f32[1,16,28,28,96]{4,3,2,1,0} %convolution.313), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %constant.315 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.316 = f32[] convert(f32[] %constant.315), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reduce.321 = f32[96]{0} reduce(f32[1,16,28,28,96]{4,3,2,1,0} %convert.314, f32[] %convert.316), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_3b_Branch_1_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.317, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.322 = s32[] get-dimension-size(f32[1,16,28,28,96]{4,3,2,1,0} %convert.314), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.323 = s32[] get-dimension-size(f32[1,16,28,28,96]{4,3,2,1,0} %convert.314), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.324 = s32[] multiply(s32[] %get-dimension-size.322, s32[] %get-dimension-size.323), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.325 = s32[] get-dimension-size(f32[1,16,28,28,96]{4,3,2,1,0} %convert.314), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.326 = s32[] multiply(s32[] %multiply.324, s32[] %get-dimension-size.325), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.327 = s32[] get-dimension-size(f32[1,16,28,28,96]{4,3,2,1,0} %convert.314), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.328 = s32[] multiply(s32[] %multiply.326, s32[] %get-dimension-size.327), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.329 = f32[] convert(s32[] %multiply.328), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.330 = f32[96]{0} broadcast(f32[] %convert.329), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %divide.331 = f32[96]{0} divide(f32[96]{0} %reduce.321, f32[96]{0} %broadcast.330), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.332 = f32[96]{0} convert(f32[96]{0} %divide.331), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.333 = f32[1,1,1,1,96]{4,3,2,1,0} reshape(f32[96]{0} %convert.332), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.334 = f32[1,96]{1,0} reshape(f32[1,1,1,1,96]{4,3,2,1,0} %reshape.333), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.335 = f32[1,16,28,28,96]{4,3,2,1,0} broadcast(f32[1,96]{1,0} %reshape.334), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.336 = f32[1,16,28,28,96]{4,3,2,1,0} subtract(f32[1,16,28,28,96]{4,3,2,1,0} %broadcast.335, f32[1,16,28,28,96]{4,3,2,1,0} %convolution.313), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.337 = f32[1,16,28,28,96]{4,3,2,1,0} multiply(f32[1,16,28,28,96]{4,3,2,1,0} %subtract.336, f32[1,16,28,28,96]{4,3,2,1,0} %subtract.336), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.338 = f32[1,16,28,28,96]{4,3,2,1,0} convert(f32[1,16,28,28,96]{4,3,2,1,0} %multiply.337), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %constant.339 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.340 = f32[] convert(f32[] %constant.339), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reduce.345 = f32[96]{0} reduce(f32[1,16,28,28,96]{4,3,2,1,0} %convert.338, f32[] %convert.340), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_3b_Branch_1_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.341, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.346 = s32[] get-dimension-size(f32[1,16,28,28,96]{4,3,2,1,0} %convert.338), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.347 = s32[] get-dimension-size(f32[1,16,28,28,96]{4,3,2,1,0} %convert.338), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.348 = s32[] multiply(s32[] %get-dimension-size.346, s32[] %get-dimension-size.347), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.349 = s32[] get-dimension-size(f32[1,16,28,28,96]{4,3,2,1,0} %convert.338), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.350 = s32[] multiply(s32[] %multiply.348, s32[] %get-dimension-size.349), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.351 = s32[] get-dimension-size(f32[1,16,28,28,96]{4,3,2,1,0} %convert.338), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.352 = s32[] multiply(s32[] %multiply.350, s32[] %get-dimension-size.351), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.353 = f32[] convert(s32[] %multiply.352), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.354 = f32[96]{0} broadcast(f32[] %convert.353), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %divide.355 = f32[96]{0} divide(f32[96]{0} %reduce.345, f32[96]{0} %broadcast.354), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.356 = f32[96]{0} convert(f32[96]{0} %divide.355), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reshape.357 = f32[1,1,1,1,96]{4,3,2,1,0} reshape(f32[96]{0} %convert.356), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %add.360 = f32[1,1,1,1,96]{4,3,2,1,0} add(f32[1,1,1,1,96]{4,3,2,1,0} %broadcast.359, f32[1,1,1,1,96]{4,3,2,1,0} %reshape.357), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.361 = f32[1,1,1,1,96]{4,3,2,1,0} rsqrt(f32[1,1,1,1,96]{4,3,2,1,0} %add.360), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.362 = f32[1,96]{1,0} reshape(f32[1,1,1,1,96]{4,3,2,1,0} %rsqrt.361), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.363 = f32[1,16,28,28,96]{4,3,2,1,0} broadcast(f32[1,96]{1,0} %reshape.362), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %multiply.364 = f32[1,16,28,28,96]{4,3,2,1,0} multiply(f32[1,16,28,28,96]{4,3,2,1,0} %broadcast.363, f32[1,16,28,28,96]{4,3,2,1,0} %convolution.313), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg27.28 = f32[1,1,1,1,96]{4,3,2,1,0} parameter(27), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.365 = f32[1,1,1,1,96]{4,3,2,1,0} multiply(f32[1,1,1,1,96]{4,3,2,1,0} %rsqrt.361, f32[1,1,1,1,96]{4,3,2,1,0} %reshape.333), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.366 = f32[1,1,1,1,96]{4,3,2,1,0} subtract(f32[1,1,1,1,96]{4,3,2,1,0} %arg27.28, f32[1,1,1,1,96]{4,3,2,1,0} %multiply.365), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.367 = f32[1,96]{1,0} reshape(f32[1,1,1,1,96]{4,3,2,1,0} %subtract.366), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.368 = f32[1,16,28,28,96]{4,3,2,1,0} broadcast(f32[1,96]{1,0} %reshape.367), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.369 = f32[1,16,28,28,96]{4,3,2,1,0} add(f32[1,16,28,28,96]{4,3,2,1,0} %multiply.364, f32[1,16,28,28,96]{4,3,2,1,0} %broadcast.368), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %maximum.372 = f32[1,16,28,28,96]{4,3,2,1,0} maximum(f32[1,16,28,28,96]{4,3,2,1,0} %broadcast.371, f32[1,16,28,28,96]{4,3,2,1,0} %add.369), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/Relu"}
  %arg30.31 = f32[3,3,3,96,128]{4,3,2,1,0} parameter(30), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.373 = f32[1,16,28,28,128]{4,3,2,1,0} convolution(f32[1,16,28,28,96]{4,3,2,1,0} %maximum.372, f32[3,3,3,96,128]{4,3,2,1,0} %arg30.31), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/conv_3d/convolution"}
  %convert.374 = f32[1,16,28,28,128]{4,3,2,1,0} convert(f32[1,16,28,28,128]{4,3,2,1,0} %convolution.373), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %constant.375 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.376 = f32[] convert(f32[] %constant.375), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reduce.381 = f32[128]{0} reduce(f32[1,16,28,28,128]{4,3,2,1,0} %convert.374, f32[] %convert.376), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_3b_Branch_1_Conv3d_0b_3x3_batch_norm_normalize_moments_mean-reduction.377, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.382 = s32[] get-dimension-size(f32[1,16,28,28,128]{4,3,2,1,0} %convert.374), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.383 = s32[] get-dimension-size(f32[1,16,28,28,128]{4,3,2,1,0} %convert.374), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.384 = s32[] multiply(s32[] %get-dimension-size.382, s32[] %get-dimension-size.383), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.385 = s32[] get-dimension-size(f32[1,16,28,28,128]{4,3,2,1,0} %convert.374), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.386 = s32[] multiply(s32[] %multiply.384, s32[] %get-dimension-size.385), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.387 = s32[] get-dimension-size(f32[1,16,28,28,128]{4,3,2,1,0} %convert.374), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.388 = s32[] multiply(s32[] %multiply.386, s32[] %get-dimension-size.387), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.389 = f32[] convert(s32[] %multiply.388), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %broadcast.390 = f32[128]{0} broadcast(f32[] %convert.389), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %divide.391 = f32[128]{0} divide(f32[128]{0} %reduce.381, f32[128]{0} %broadcast.390), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.392 = f32[128]{0} convert(f32[128]{0} %divide.391), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reshape.393 = f32[1,1,1,1,128]{4,3,2,1,0} reshape(f32[128]{0} %convert.392), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reshape.394 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %reshape.393), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.395 = f32[1,16,28,28,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.394), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.396 = f32[1,16,28,28,128]{4,3,2,1,0} subtract(f32[1,16,28,28,128]{4,3,2,1,0} %broadcast.395, f32[1,16,28,28,128]{4,3,2,1,0} %convolution.373), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.397 = f32[1,16,28,28,128]{4,3,2,1,0} multiply(f32[1,16,28,28,128]{4,3,2,1,0} %subtract.396, f32[1,16,28,28,128]{4,3,2,1,0} %subtract.396), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %convert.398 = f32[1,16,28,28,128]{4,3,2,1,0} convert(f32[1,16,28,28,128]{4,3,2,1,0} %multiply.397), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %constant.399 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.400 = f32[] convert(f32[] %constant.399), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %reduce.405 = f32[128]{0} reduce(f32[1,16,28,28,128]{4,3,2,1,0} %convert.398, f32[] %convert.400), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_3b_Branch_1_Conv3d_0b_3x3_batch_norm_normalize_moments_variance-reduction.401, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.406 = s32[] get-dimension-size(f32[1,16,28,28,128]{4,3,2,1,0} %convert.398), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.407 = s32[] get-dimension-size(f32[1,16,28,28,128]{4,3,2,1,0} %convert.398), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.408 = s32[] multiply(s32[] %get-dimension-size.406, s32[] %get-dimension-size.407), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.409 = s32[] get-dimension-size(f32[1,16,28,28,128]{4,3,2,1,0} %convert.398), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.410 = s32[] multiply(s32[] %multiply.408, s32[] %get-dimension-size.409), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.411 = s32[] get-dimension-size(f32[1,16,28,28,128]{4,3,2,1,0} %convert.398), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.412 = s32[] multiply(s32[] %multiply.410, s32[] %get-dimension-size.411), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.413 = f32[] convert(s32[] %multiply.412), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %broadcast.414 = f32[128]{0} broadcast(f32[] %convert.413), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %divide.415 = f32[128]{0} divide(f32[128]{0} %reduce.405, f32[128]{0} %broadcast.414), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.416 = f32[128]{0} convert(f32[128]{0} %divide.415), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %reshape.417 = f32[1,1,1,1,128]{4,3,2,1,0} reshape(f32[128]{0} %convert.416), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %add.420 = f32[1,1,1,1,128]{4,3,2,1,0} add(f32[1,1,1,1,128]{4,3,2,1,0} %broadcast.419, f32[1,1,1,1,128]{4,3,2,1,0} %reshape.417), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %rsqrt.421 = f32[1,1,1,1,128]{4,3,2,1,0} rsqrt(f32[1,1,1,1,128]{4,3,2,1,0} %add.420), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.422 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.421), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %broadcast.423 = f32[1,16,28,28,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.422), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %multiply.424 = f32[1,16,28,28,128]{4,3,2,1,0} multiply(f32[1,16,28,28,128]{4,3,2,1,0} %broadcast.423, f32[1,16,28,28,128]{4,3,2,1,0} %convolution.373), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %arg33.34 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(33), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.425 = f32[1,1,1,1,128]{4,3,2,1,0} multiply(f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.421, f32[1,1,1,1,128]{4,3,2,1,0} %reshape.393), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.426 = f32[1,1,1,1,128]{4,3,2,1,0} subtract(f32[1,1,1,1,128]{4,3,2,1,0} %arg33.34, f32[1,1,1,1,128]{4,3,2,1,0} %multiply.425), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/sub"}
  %reshape.427 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %subtract.426), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.428 = f32[1,16,28,28,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.427), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %add.429 = f32[1,16,28,28,128]{4,3,2,1,0} add(f32[1,16,28,28,128]{4,3,2,1,0} %multiply.424, f32[1,16,28,28,128]{4,3,2,1,0} %broadcast.428), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %constant.535 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %broadcast.536 = f32[1,1,1,1,32]{4,3,2,1,0} broadcast(f32[] %constant.535), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %constant.487 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/Relu"}
  %broadcast.488 = f32[1,16,28,28,16]{4,3,2,1,0} broadcast(f32[] %constant.487), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/Relu"}
  %constant.475 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.476 = f32[1,1,1,1,16]{4,3,2,1,0} broadcast(f32[] %constant.475), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %arg37.38 = f32[1,1,1,192,16]{4,3,2,1,0} parameter(37), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.430 = f32[1,16,28,28,16]{4,3,2,1,0} convolution(f32[1,16,28,28,192]{4,3,2,1,0} %maximum.311, f32[1,1,1,192,16]{4,3,2,1,0} %arg37.38), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/conv_3d/convolution"}
  %convert.431 = f32[1,16,28,28,16]{4,3,2,1,0} convert(f32[1,16,28,28,16]{4,3,2,1,0} %convolution.430), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %constant.432 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.433 = f32[] convert(f32[] %constant.432), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reduce.438 = f32[16]{0} reduce(f32[1,16,28,28,16]{4,3,2,1,0} %convert.431, f32[] %convert.433), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_3b_Branch_2_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.434, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.439 = s32[] get-dimension-size(f32[1,16,28,28,16]{4,3,2,1,0} %convert.431), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.440 = s32[] get-dimension-size(f32[1,16,28,28,16]{4,3,2,1,0} %convert.431), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.441 = s32[] multiply(s32[] %get-dimension-size.439, s32[] %get-dimension-size.440), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.442 = s32[] get-dimension-size(f32[1,16,28,28,16]{4,3,2,1,0} %convert.431), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.443 = s32[] multiply(s32[] %multiply.441, s32[] %get-dimension-size.442), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.444 = s32[] get-dimension-size(f32[1,16,28,28,16]{4,3,2,1,0} %convert.431), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.445 = s32[] multiply(s32[] %multiply.443, s32[] %get-dimension-size.444), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.446 = f32[] convert(s32[] %multiply.445), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.447 = f32[16]{0} broadcast(f32[] %convert.446), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %divide.448 = f32[16]{0} divide(f32[16]{0} %reduce.438, f32[16]{0} %broadcast.447), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.449 = f32[16]{0} convert(f32[16]{0} %divide.448), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.450 = f32[1,1,1,1,16]{4,3,2,1,0} reshape(f32[16]{0} %convert.449), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.451 = f32[1,16]{1,0} reshape(f32[1,1,1,1,16]{4,3,2,1,0} %reshape.450), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.452 = f32[1,16,28,28,16]{4,3,2,1,0} broadcast(f32[1,16]{1,0} %reshape.451), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.453 = f32[1,16,28,28,16]{4,3,2,1,0} subtract(f32[1,16,28,28,16]{4,3,2,1,0} %broadcast.452, f32[1,16,28,28,16]{4,3,2,1,0} %convolution.430), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.454 = f32[1,16,28,28,16]{4,3,2,1,0} multiply(f32[1,16,28,28,16]{4,3,2,1,0} %subtract.453, f32[1,16,28,28,16]{4,3,2,1,0} %subtract.453), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.455 = f32[1,16,28,28,16]{4,3,2,1,0} convert(f32[1,16,28,28,16]{4,3,2,1,0} %multiply.454), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %constant.456 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.457 = f32[] convert(f32[] %constant.456), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reduce.462 = f32[16]{0} reduce(f32[1,16,28,28,16]{4,3,2,1,0} %convert.455, f32[] %convert.457), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_3b_Branch_2_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.458, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.463 = s32[] get-dimension-size(f32[1,16,28,28,16]{4,3,2,1,0} %convert.455), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.464 = s32[] get-dimension-size(f32[1,16,28,28,16]{4,3,2,1,0} %convert.455), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.465 = s32[] multiply(s32[] %get-dimension-size.463, s32[] %get-dimension-size.464), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.466 = s32[] get-dimension-size(f32[1,16,28,28,16]{4,3,2,1,0} %convert.455), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.467 = s32[] multiply(s32[] %multiply.465, s32[] %get-dimension-size.466), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.468 = s32[] get-dimension-size(f32[1,16,28,28,16]{4,3,2,1,0} %convert.455), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.469 = s32[] multiply(s32[] %multiply.467, s32[] %get-dimension-size.468), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.470 = f32[] convert(s32[] %multiply.469), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.471 = f32[16]{0} broadcast(f32[] %convert.470), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %divide.472 = f32[16]{0} divide(f32[16]{0} %reduce.462, f32[16]{0} %broadcast.471), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.473 = f32[16]{0} convert(f32[16]{0} %divide.472), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reshape.474 = f32[1,1,1,1,16]{4,3,2,1,0} reshape(f32[16]{0} %convert.473), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %add.477 = f32[1,1,1,1,16]{4,3,2,1,0} add(f32[1,1,1,1,16]{4,3,2,1,0} %broadcast.476, f32[1,1,1,1,16]{4,3,2,1,0} %reshape.474), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.478 = f32[1,1,1,1,16]{4,3,2,1,0} rsqrt(f32[1,1,1,1,16]{4,3,2,1,0} %add.477), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.479 = f32[1,16]{1,0} reshape(f32[1,1,1,1,16]{4,3,2,1,0} %rsqrt.478), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.480 = f32[1,16,28,28,16]{4,3,2,1,0} broadcast(f32[1,16]{1,0} %reshape.479), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %multiply.481 = f32[1,16,28,28,16]{4,3,2,1,0} multiply(f32[1,16,28,28,16]{4,3,2,1,0} %broadcast.480, f32[1,16,28,28,16]{4,3,2,1,0} %convolution.430), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg40.41 = f32[1,1,1,1,16]{4,3,2,1,0} parameter(40), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.482 = f32[1,1,1,1,16]{4,3,2,1,0} multiply(f32[1,1,1,1,16]{4,3,2,1,0} %rsqrt.478, f32[1,1,1,1,16]{4,3,2,1,0} %reshape.450), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.483 = f32[1,1,1,1,16]{4,3,2,1,0} subtract(f32[1,1,1,1,16]{4,3,2,1,0} %arg40.41, f32[1,1,1,1,16]{4,3,2,1,0} %multiply.482), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.484 = f32[1,16]{1,0} reshape(f32[1,1,1,1,16]{4,3,2,1,0} %subtract.483), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.485 = f32[1,16,28,28,16]{4,3,2,1,0} broadcast(f32[1,16]{1,0} %reshape.484), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.486 = f32[1,16,28,28,16]{4,3,2,1,0} add(f32[1,16,28,28,16]{4,3,2,1,0} %multiply.481, f32[1,16,28,28,16]{4,3,2,1,0} %broadcast.485), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %maximum.489 = f32[1,16,28,28,16]{4,3,2,1,0} maximum(f32[1,16,28,28,16]{4,3,2,1,0} %broadcast.488, f32[1,16,28,28,16]{4,3,2,1,0} %add.486), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/Relu"}
  %arg43.44 = f32[3,3,3,16,32]{4,3,2,1,0} parameter(43), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.490 = f32[1,16,28,28,32]{4,3,2,1,0} convolution(f32[1,16,28,28,16]{4,3,2,1,0} %maximum.489, f32[3,3,3,16,32]{4,3,2,1,0} %arg43.44), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/conv_3d/convolution"}
  %convert.491 = f32[1,16,28,28,32]{4,3,2,1,0} convert(f32[1,16,28,28,32]{4,3,2,1,0} %convolution.490), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %constant.492 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.493 = f32[] convert(f32[] %constant.492), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reduce.498 = f32[32]{0} reduce(f32[1,16,28,28,32]{4,3,2,1,0} %convert.491, f32[] %convert.493), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_3b_Branch_2_Conv3d_0b_3x3_batch_norm_normalize_moments_mean-reduction.494, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.499 = s32[] get-dimension-size(f32[1,16,28,28,32]{4,3,2,1,0} %convert.491), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.500 = s32[] get-dimension-size(f32[1,16,28,28,32]{4,3,2,1,0} %convert.491), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.501 = s32[] multiply(s32[] %get-dimension-size.499, s32[] %get-dimension-size.500), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.502 = s32[] get-dimension-size(f32[1,16,28,28,32]{4,3,2,1,0} %convert.491), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.503 = s32[] multiply(s32[] %multiply.501, s32[] %get-dimension-size.502), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.504 = s32[] get-dimension-size(f32[1,16,28,28,32]{4,3,2,1,0} %convert.491), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.505 = s32[] multiply(s32[] %multiply.503, s32[] %get-dimension-size.504), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.506 = f32[] convert(s32[] %multiply.505), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %broadcast.507 = f32[32]{0} broadcast(f32[] %convert.506), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %divide.508 = f32[32]{0} divide(f32[32]{0} %reduce.498, f32[32]{0} %broadcast.507), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.509 = f32[32]{0} convert(f32[32]{0} %divide.508), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reshape.510 = f32[1,1,1,1,32]{4,3,2,1,0} reshape(f32[32]{0} %convert.509), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reshape.511 = f32[1,32]{1,0} reshape(f32[1,1,1,1,32]{4,3,2,1,0} %reshape.510), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.512 = f32[1,16,28,28,32]{4,3,2,1,0} broadcast(f32[1,32]{1,0} %reshape.511), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.513 = f32[1,16,28,28,32]{4,3,2,1,0} subtract(f32[1,16,28,28,32]{4,3,2,1,0} %broadcast.512, f32[1,16,28,28,32]{4,3,2,1,0} %convolution.490), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.514 = f32[1,16,28,28,32]{4,3,2,1,0} multiply(f32[1,16,28,28,32]{4,3,2,1,0} %subtract.513, f32[1,16,28,28,32]{4,3,2,1,0} %subtract.513), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %convert.515 = f32[1,16,28,28,32]{4,3,2,1,0} convert(f32[1,16,28,28,32]{4,3,2,1,0} %multiply.514), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %constant.516 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.517 = f32[] convert(f32[] %constant.516), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %reduce.522 = f32[32]{0} reduce(f32[1,16,28,28,32]{4,3,2,1,0} %convert.515, f32[] %convert.517), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_3b_Branch_2_Conv3d_0b_3x3_batch_norm_normalize_moments_variance-reduction.518, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.523 = s32[] get-dimension-size(f32[1,16,28,28,32]{4,3,2,1,0} %convert.515), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.524 = s32[] get-dimension-size(f32[1,16,28,28,32]{4,3,2,1,0} %convert.515), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.525 = s32[] multiply(s32[] %get-dimension-size.523, s32[] %get-dimension-size.524), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.526 = s32[] get-dimension-size(f32[1,16,28,28,32]{4,3,2,1,0} %convert.515), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.527 = s32[] multiply(s32[] %multiply.525, s32[] %get-dimension-size.526), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.528 = s32[] get-dimension-size(f32[1,16,28,28,32]{4,3,2,1,0} %convert.515), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.529 = s32[] multiply(s32[] %multiply.527, s32[] %get-dimension-size.528), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.530 = f32[] convert(s32[] %multiply.529), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %broadcast.531 = f32[32]{0} broadcast(f32[] %convert.530), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %divide.532 = f32[32]{0} divide(f32[32]{0} %reduce.522, f32[32]{0} %broadcast.531), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.533 = f32[32]{0} convert(f32[32]{0} %divide.532), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %reshape.534 = f32[1,1,1,1,32]{4,3,2,1,0} reshape(f32[32]{0} %convert.533), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %add.537 = f32[1,1,1,1,32]{4,3,2,1,0} add(f32[1,1,1,1,32]{4,3,2,1,0} %broadcast.536, f32[1,1,1,1,32]{4,3,2,1,0} %reshape.534), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %rsqrt.538 = f32[1,1,1,1,32]{4,3,2,1,0} rsqrt(f32[1,1,1,1,32]{4,3,2,1,0} %add.537), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.539 = f32[1,32]{1,0} reshape(f32[1,1,1,1,32]{4,3,2,1,0} %rsqrt.538), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %broadcast.540 = f32[1,16,28,28,32]{4,3,2,1,0} broadcast(f32[1,32]{1,0} %reshape.539), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %multiply.541 = f32[1,16,28,28,32]{4,3,2,1,0} multiply(f32[1,16,28,28,32]{4,3,2,1,0} %broadcast.540, f32[1,16,28,28,32]{4,3,2,1,0} %convolution.490), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %arg44.45 = f32[1,1,1,1,32]{4,3,2,1,0} parameter(44), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.542 = f32[1,1,1,1,32]{4,3,2,1,0} multiply(f32[1,1,1,1,32]{4,3,2,1,0} %rsqrt.538, f32[1,1,1,1,32]{4,3,2,1,0} %reshape.510), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.543 = f32[1,1,1,1,32]{4,3,2,1,0} subtract(f32[1,1,1,1,32]{4,3,2,1,0} %arg44.45, f32[1,1,1,1,32]{4,3,2,1,0} %multiply.542), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/sub"}
  %reshape.544 = f32[1,32]{1,0} reshape(f32[1,1,1,1,32]{4,3,2,1,0} %subtract.543), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.545 = f32[1,16,28,28,32]{4,3,2,1,0} broadcast(f32[1,32]{1,0} %reshape.544), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %add.546 = f32[1,16,28,28,32]{4,3,2,1,0} add(f32[1,16,28,28,32]{4,3,2,1,0} %multiply.541, f32[1,16,28,28,32]{4,3,2,1,0} %broadcast.545), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %constant.598 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %broadcast.599 = f32[1,1,1,1,32]{4,3,2,1,0} broadcast(f32[] %constant.598), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %constant.547 = f32[] constant(-inf), metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/MaxPool3d_0a_3x3"}
  %reduce-window.552 = f32[1,16,28,28,192]{4,3,2,1,0} reduce-window(f32[1,16,28,28,192]{4,3,2,1,0} %maximum.311, f32[] %constant.547), window={size=1x3x3x3x1 pad=0_0x1_1x1_1x1_1x0_0}, to_apply=%max_F32.548, metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/MaxPool3d_0a_3x3"}
  %arg50.51 = f32[1,1,1,192,32]{4,3,2,1,0} parameter(50), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.553 = f32[1,16,28,28,32]{4,3,2,1,0} convolution(f32[1,16,28,28,192]{4,3,2,1,0} %reduce-window.552, f32[1,1,1,192,32]{4,3,2,1,0} %arg50.51), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/conv_3d/convolution"}
  %convert.554 = f32[1,16,28,28,32]{4,3,2,1,0} convert(f32[1,16,28,28,32]{4,3,2,1,0} %convolution.553), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %constant.555 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %convert.556 = f32[] convert(f32[] %constant.555), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %reduce.561 = f32[32]{0} reduce(f32[1,16,28,28,32]{4,3,2,1,0} %convert.554, f32[] %convert.556), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_3b_Branch_3_Conv3d_0b_1x1_batch_norm_normalize_moments_mean-reduction.557, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.562 = s32[] get-dimension-size(f32[1,16,28,28,32]{4,3,2,1,0} %convert.554), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.563 = s32[] get-dimension-size(f32[1,16,28,28,32]{4,3,2,1,0} %convert.554), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %multiply.564 = s32[] multiply(s32[] %get-dimension-size.562, s32[] %get-dimension-size.563), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.565 = s32[] get-dimension-size(f32[1,16,28,28,32]{4,3,2,1,0} %convert.554), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %multiply.566 = s32[] multiply(s32[] %multiply.564, s32[] %get-dimension-size.565), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.567 = s32[] get-dimension-size(f32[1,16,28,28,32]{4,3,2,1,0} %convert.554), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %multiply.568 = s32[] multiply(s32[] %multiply.566, s32[] %get-dimension-size.567), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %convert.569 = f32[] convert(s32[] %multiply.568), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.570 = f32[32]{0} broadcast(f32[] %convert.569), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %divide.571 = f32[32]{0} divide(f32[32]{0} %reduce.561, f32[32]{0} %broadcast.570), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %convert.572 = f32[32]{0} convert(f32[32]{0} %divide.571), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %reshape.573 = f32[1,1,1,1,32]{4,3,2,1,0} reshape(f32[32]{0} %convert.572), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %reshape.574 = f32[1,32]{1,0} reshape(f32[1,1,1,1,32]{4,3,2,1,0} %reshape.573), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.575 = f32[1,16,28,28,32]{4,3,2,1,0} broadcast(f32[1,32]{1,0} %reshape.574), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.576 = f32[1,16,28,28,32]{4,3,2,1,0} subtract(f32[1,16,28,28,32]{4,3,2,1,0} %broadcast.575, f32[1,16,28,28,32]{4,3,2,1,0} %convolution.553), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.577 = f32[1,16,28,28,32]{4,3,2,1,0} multiply(f32[1,16,28,28,32]{4,3,2,1,0} %subtract.576, f32[1,16,28,28,32]{4,3,2,1,0} %subtract.576), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.578 = f32[1,16,28,28,32]{4,3,2,1,0} convert(f32[1,16,28,28,32]{4,3,2,1,0} %multiply.577), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %constant.579 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %convert.580 = f32[] convert(f32[] %constant.579), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %reduce.585 = f32[32]{0} reduce(f32[1,16,28,28,32]{4,3,2,1,0} %convert.578, f32[] %convert.580), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_3b_Branch_3_Conv3d_0b_1x1_batch_norm_normalize_moments_variance-reduction.581, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.586 = s32[] get-dimension-size(f32[1,16,28,28,32]{4,3,2,1,0} %convert.578), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.587 = s32[] get-dimension-size(f32[1,16,28,28,32]{4,3,2,1,0} %convert.578), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %multiply.588 = s32[] multiply(s32[] %get-dimension-size.586, s32[] %get-dimension-size.587), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.589 = s32[] get-dimension-size(f32[1,16,28,28,32]{4,3,2,1,0} %convert.578), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %multiply.590 = s32[] multiply(s32[] %multiply.588, s32[] %get-dimension-size.589), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.591 = s32[] get-dimension-size(f32[1,16,28,28,32]{4,3,2,1,0} %convert.578), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %multiply.592 = s32[] multiply(s32[] %multiply.590, s32[] %get-dimension-size.591), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %convert.593 = f32[] convert(s32[] %multiply.592), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.594 = f32[32]{0} broadcast(f32[] %convert.593), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %divide.595 = f32[32]{0} divide(f32[32]{0} %reduce.585, f32[32]{0} %broadcast.594), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %convert.596 = f32[32]{0} convert(f32[32]{0} %divide.595), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %reshape.597 = f32[1,1,1,1,32]{4,3,2,1,0} reshape(f32[32]{0} %convert.596), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %add.600 = f32[1,1,1,1,32]{4,3,2,1,0} add(f32[1,1,1,1,32]{4,3,2,1,0} %broadcast.599, f32[1,1,1,1,32]{4,3,2,1,0} %reshape.597), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %rsqrt.601 = f32[1,1,1,1,32]{4,3,2,1,0} rsqrt(f32[1,1,1,1,32]{4,3,2,1,0} %add.600), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.602 = f32[1,32]{1,0} reshape(f32[1,1,1,1,32]{4,3,2,1,0} %rsqrt.601), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %broadcast.603 = f32[1,16,28,28,32]{4,3,2,1,0} broadcast(f32[1,32]{1,0} %reshape.602), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %multiply.604 = f32[1,16,28,28,32]{4,3,2,1,0} multiply(f32[1,16,28,28,32]{4,3,2,1,0} %broadcast.603, f32[1,16,28,28,32]{4,3,2,1,0} %convolution.553), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %arg52.53 = f32[1,1,1,1,32]{4,3,2,1,0} parameter(52), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.605 = f32[1,1,1,1,32]{4,3,2,1,0} multiply(f32[1,1,1,1,32]{4,3,2,1,0} %rsqrt.601, f32[1,1,1,1,32]{4,3,2,1,0} %reshape.573), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.606 = f32[1,1,1,1,32]{4,3,2,1,0} subtract(f32[1,1,1,1,32]{4,3,2,1,0} %arg52.53, f32[1,1,1,1,32]{4,3,2,1,0} %multiply.605), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/sub"}
  %reshape.607 = f32[1,32]{1,0} reshape(f32[1,1,1,1,32]{4,3,2,1,0} %subtract.606), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.608 = f32[1,16,28,28,32]{4,3,2,1,0} broadcast(f32[1,32]{1,0} %reshape.607), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %add.609 = f32[1,16,28,28,32]{4,3,2,1,0} add(f32[1,16,28,28,32]{4,3,2,1,0} %multiply.604, f32[1,16,28,28,32]{4,3,2,1,0} %broadcast.608), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %concatenate.666 = f32[1,16,28,28,256]{4,3,2,1,0} concatenate(f32[1,16,28,28,64]{4,3,2,1,0} %add.665, f32[1,16,28,28,128]{4,3,2,1,0} %add.429, f32[1,16,28,28,32]{4,3,2,1,0} %add.546, f32[1,16,28,28,32]{4,3,2,1,0} %add.609), dimensions={4}, metadata={op_type="ConcatV2" op_name="RGB/inception_i3d/Mixed_3b/concat"}
  %maximum.669 = f32[1,16,28,28,256]{4,3,2,1,0} maximum(f32[1,16,28,28,256]{4,3,2,1,0} %broadcast.668, f32[1,16,28,28,256]{4,3,2,1,0} %concatenate.666), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/Relu"}
  %arg54.55 = f32[1,1,1,256,128]{4,3,2,1,0} parameter(54), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.670 = f32[1,16,28,28,128]{4,3,2,1,0} convolution(f32[1,16,28,28,256]{4,3,2,1,0} %maximum.669, f32[1,1,1,256,128]{4,3,2,1,0} %arg54.55), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/conv_3d/convolution"}
  %convert.671 = f32[1,16,28,28,128]{4,3,2,1,0} convert(f32[1,16,28,28,128]{4,3,2,1,0} %convolution.670), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %constant.672 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.673 = f32[] convert(f32[] %constant.672), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reduce.678 = f32[128]{0} reduce(f32[1,16,28,28,128]{4,3,2,1,0} %convert.671, f32[] %convert.673), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_3c_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.674, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.679 = s32[] get-dimension-size(f32[1,16,28,28,128]{4,3,2,1,0} %convert.671), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.680 = s32[] get-dimension-size(f32[1,16,28,28,128]{4,3,2,1,0} %convert.671), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.681 = s32[] multiply(s32[] %get-dimension-size.679, s32[] %get-dimension-size.680), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.682 = s32[] get-dimension-size(f32[1,16,28,28,128]{4,3,2,1,0} %convert.671), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.683 = s32[] multiply(s32[] %multiply.681, s32[] %get-dimension-size.682), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.684 = s32[] get-dimension-size(f32[1,16,28,28,128]{4,3,2,1,0} %convert.671), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.685 = s32[] multiply(s32[] %multiply.683, s32[] %get-dimension-size.684), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.686 = f32[] convert(s32[] %multiply.685), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.687 = f32[128]{0} broadcast(f32[] %convert.686), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %divide.688 = f32[128]{0} divide(f32[128]{0} %reduce.678, f32[128]{0} %broadcast.687), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.689 = f32[128]{0} convert(f32[128]{0} %divide.688), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.690 = f32[1,1,1,1,128]{4,3,2,1,0} reshape(f32[128]{0} %convert.689), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.691 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %reshape.690), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.692 = f32[1,16,28,28,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.691), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.693 = f32[1,16,28,28,128]{4,3,2,1,0} subtract(f32[1,16,28,28,128]{4,3,2,1,0} %broadcast.692, f32[1,16,28,28,128]{4,3,2,1,0} %convolution.670), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.694 = f32[1,16,28,28,128]{4,3,2,1,0} multiply(f32[1,16,28,28,128]{4,3,2,1,0} %subtract.693, f32[1,16,28,28,128]{4,3,2,1,0} %subtract.693), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.695 = f32[1,16,28,28,128]{4,3,2,1,0} convert(f32[1,16,28,28,128]{4,3,2,1,0} %multiply.694), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %constant.696 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.697 = f32[] convert(f32[] %constant.696), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reduce.702 = f32[128]{0} reduce(f32[1,16,28,28,128]{4,3,2,1,0} %convert.695, f32[] %convert.697), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_3c_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.698, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.703 = s32[] get-dimension-size(f32[1,16,28,28,128]{4,3,2,1,0} %convert.695), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.704 = s32[] get-dimension-size(f32[1,16,28,28,128]{4,3,2,1,0} %convert.695), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.705 = s32[] multiply(s32[] %get-dimension-size.703, s32[] %get-dimension-size.704), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.706 = s32[] get-dimension-size(f32[1,16,28,28,128]{4,3,2,1,0} %convert.695), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.707 = s32[] multiply(s32[] %multiply.705, s32[] %get-dimension-size.706), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.708 = s32[] get-dimension-size(f32[1,16,28,28,128]{4,3,2,1,0} %convert.695), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.709 = s32[] multiply(s32[] %multiply.707, s32[] %get-dimension-size.708), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.710 = f32[] convert(s32[] %multiply.709), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.711 = f32[128]{0} broadcast(f32[] %convert.710), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %divide.712 = f32[128]{0} divide(f32[128]{0} %reduce.702, f32[128]{0} %broadcast.711), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.713 = f32[128]{0} convert(f32[128]{0} %divide.712), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reshape.714 = f32[1,1,1,1,128]{4,3,2,1,0} reshape(f32[128]{0} %convert.713), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %add.717 = f32[1,1,1,1,128]{4,3,2,1,0} add(f32[1,1,1,1,128]{4,3,2,1,0} %broadcast.716, f32[1,1,1,1,128]{4,3,2,1,0} %reshape.714), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.718 = f32[1,1,1,1,128]{4,3,2,1,0} rsqrt(f32[1,1,1,1,128]{4,3,2,1,0} %add.717), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.719 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.718), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.720 = f32[1,16,28,28,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.719), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %multiply.721 = f32[1,16,28,28,128]{4,3,2,1,0} multiply(f32[1,16,28,28,128]{4,3,2,1,0} %broadcast.720, f32[1,16,28,28,128]{4,3,2,1,0} %convolution.670), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg58.59 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(58), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.722 = f32[1,1,1,1,128]{4,3,2,1,0} multiply(f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.718, f32[1,1,1,1,128]{4,3,2,1,0} %reshape.690), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.723 = f32[1,1,1,1,128]{4,3,2,1,0} subtract(f32[1,1,1,1,128]{4,3,2,1,0} %arg58.59, f32[1,1,1,1,128]{4,3,2,1,0} %multiply.722), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.724 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %subtract.723), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.725 = f32[1,16,28,28,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.724), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.726 = f32[1,16,28,28,128]{4,3,2,1,0} add(f32[1,16,28,28,128]{4,3,2,1,0} %multiply.721, f32[1,16,28,28,128]{4,3,2,1,0} %broadcast.725), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %constant.832 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %broadcast.833 = f32[1,1,1,1,192]{4,3,2,1,0} broadcast(f32[] %constant.832), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %constant.784 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/Relu"}
  %broadcast.785 = f32[1,16,28,28,128]{4,3,2,1,0} broadcast(f32[] %constant.784), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/Relu"}
  %constant.772 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.773 = f32[1,1,1,1,128]{4,3,2,1,0} broadcast(f32[] %constant.772), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %arg61.62 = f32[1,1,1,256,128]{4,3,2,1,0} parameter(61), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.727 = f32[1,16,28,28,128]{4,3,2,1,0} convolution(f32[1,16,28,28,256]{4,3,2,1,0} %maximum.669, f32[1,1,1,256,128]{4,3,2,1,0} %arg61.62), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/conv_3d/convolution"}
  %convert.728 = f32[1,16,28,28,128]{4,3,2,1,0} convert(f32[1,16,28,28,128]{4,3,2,1,0} %convolution.727), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %constant.729 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.730 = f32[] convert(f32[] %constant.729), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reduce.735 = f32[128]{0} reduce(f32[1,16,28,28,128]{4,3,2,1,0} %convert.728, f32[] %convert.730), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_3c_Branch_1_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.731, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.736 = s32[] get-dimension-size(f32[1,16,28,28,128]{4,3,2,1,0} %convert.728), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.737 = s32[] get-dimension-size(f32[1,16,28,28,128]{4,3,2,1,0} %convert.728), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.738 = s32[] multiply(s32[] %get-dimension-size.736, s32[] %get-dimension-size.737), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.739 = s32[] get-dimension-size(f32[1,16,28,28,128]{4,3,2,1,0} %convert.728), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.740 = s32[] multiply(s32[] %multiply.738, s32[] %get-dimension-size.739), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.741 = s32[] get-dimension-size(f32[1,16,28,28,128]{4,3,2,1,0} %convert.728), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.742 = s32[] multiply(s32[] %multiply.740, s32[] %get-dimension-size.741), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.743 = f32[] convert(s32[] %multiply.742), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.744 = f32[128]{0} broadcast(f32[] %convert.743), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %divide.745 = f32[128]{0} divide(f32[128]{0} %reduce.735, f32[128]{0} %broadcast.744), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.746 = f32[128]{0} convert(f32[128]{0} %divide.745), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.747 = f32[1,1,1,1,128]{4,3,2,1,0} reshape(f32[128]{0} %convert.746), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.748 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %reshape.747), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.749 = f32[1,16,28,28,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.748), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.750 = f32[1,16,28,28,128]{4,3,2,1,0} subtract(f32[1,16,28,28,128]{4,3,2,1,0} %broadcast.749, f32[1,16,28,28,128]{4,3,2,1,0} %convolution.727), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.751 = f32[1,16,28,28,128]{4,3,2,1,0} multiply(f32[1,16,28,28,128]{4,3,2,1,0} %subtract.750, f32[1,16,28,28,128]{4,3,2,1,0} %subtract.750), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.752 = f32[1,16,28,28,128]{4,3,2,1,0} convert(f32[1,16,28,28,128]{4,3,2,1,0} %multiply.751), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %constant.753 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.754 = f32[] convert(f32[] %constant.753), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reduce.759 = f32[128]{0} reduce(f32[1,16,28,28,128]{4,3,2,1,0} %convert.752, f32[] %convert.754), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_3c_Branch_1_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.755, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.760 = s32[] get-dimension-size(f32[1,16,28,28,128]{4,3,2,1,0} %convert.752), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.761 = s32[] get-dimension-size(f32[1,16,28,28,128]{4,3,2,1,0} %convert.752), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.762 = s32[] multiply(s32[] %get-dimension-size.760, s32[] %get-dimension-size.761), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.763 = s32[] get-dimension-size(f32[1,16,28,28,128]{4,3,2,1,0} %convert.752), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.764 = s32[] multiply(s32[] %multiply.762, s32[] %get-dimension-size.763), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.765 = s32[] get-dimension-size(f32[1,16,28,28,128]{4,3,2,1,0} %convert.752), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.766 = s32[] multiply(s32[] %multiply.764, s32[] %get-dimension-size.765), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.767 = f32[] convert(s32[] %multiply.766), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.768 = f32[128]{0} broadcast(f32[] %convert.767), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %divide.769 = f32[128]{0} divide(f32[128]{0} %reduce.759, f32[128]{0} %broadcast.768), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.770 = f32[128]{0} convert(f32[128]{0} %divide.769), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reshape.771 = f32[1,1,1,1,128]{4,3,2,1,0} reshape(f32[128]{0} %convert.770), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %add.774 = f32[1,1,1,1,128]{4,3,2,1,0} add(f32[1,1,1,1,128]{4,3,2,1,0} %broadcast.773, f32[1,1,1,1,128]{4,3,2,1,0} %reshape.771), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.775 = f32[1,1,1,1,128]{4,3,2,1,0} rsqrt(f32[1,1,1,1,128]{4,3,2,1,0} %add.774), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.776 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.775), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.777 = f32[1,16,28,28,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.776), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %multiply.778 = f32[1,16,28,28,128]{4,3,2,1,0} multiply(f32[1,16,28,28,128]{4,3,2,1,0} %broadcast.777, f32[1,16,28,28,128]{4,3,2,1,0} %convolution.727), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg64.65 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(64), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.779 = f32[1,1,1,1,128]{4,3,2,1,0} multiply(f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.775, f32[1,1,1,1,128]{4,3,2,1,0} %reshape.747), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.780 = f32[1,1,1,1,128]{4,3,2,1,0} subtract(f32[1,1,1,1,128]{4,3,2,1,0} %arg64.65, f32[1,1,1,1,128]{4,3,2,1,0} %multiply.779), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.781 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %subtract.780), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.782 = f32[1,16,28,28,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.781), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.783 = f32[1,16,28,28,128]{4,3,2,1,0} add(f32[1,16,28,28,128]{4,3,2,1,0} %multiply.778, f32[1,16,28,28,128]{4,3,2,1,0} %broadcast.782), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %maximum.786 = f32[1,16,28,28,128]{4,3,2,1,0} maximum(f32[1,16,28,28,128]{4,3,2,1,0} %broadcast.785, f32[1,16,28,28,128]{4,3,2,1,0} %add.783), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/Relu"}
  %arg68.69 = f32[3,3,3,128,192]{4,3,2,1,0} parameter(68), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.787 = f32[1,16,28,28,192]{4,3,2,1,0} convolution(f32[1,16,28,28,128]{4,3,2,1,0} %maximum.786, f32[3,3,3,128,192]{4,3,2,1,0} %arg68.69), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/conv_3d/convolution"}
  %convert.788 = f32[1,16,28,28,192]{4,3,2,1,0} convert(f32[1,16,28,28,192]{4,3,2,1,0} %convolution.787), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %constant.789 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.790 = f32[] convert(f32[] %constant.789), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reduce.795 = f32[192]{0} reduce(f32[1,16,28,28,192]{4,3,2,1,0} %convert.788, f32[] %convert.790), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_3c_Branch_1_Conv3d_0b_3x3_batch_norm_normalize_moments_mean-reduction.791, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.796 = s32[] get-dimension-size(f32[1,16,28,28,192]{4,3,2,1,0} %convert.788), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.797 = s32[] get-dimension-size(f32[1,16,28,28,192]{4,3,2,1,0} %convert.788), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.798 = s32[] multiply(s32[] %get-dimension-size.796, s32[] %get-dimension-size.797), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.799 = s32[] get-dimension-size(f32[1,16,28,28,192]{4,3,2,1,0} %convert.788), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.800 = s32[] multiply(s32[] %multiply.798, s32[] %get-dimension-size.799), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.801 = s32[] get-dimension-size(f32[1,16,28,28,192]{4,3,2,1,0} %convert.788), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.802 = s32[] multiply(s32[] %multiply.800, s32[] %get-dimension-size.801), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.803 = f32[] convert(s32[] %multiply.802), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %broadcast.804 = f32[192]{0} broadcast(f32[] %convert.803), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %divide.805 = f32[192]{0} divide(f32[192]{0} %reduce.795, f32[192]{0} %broadcast.804), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.806 = f32[192]{0} convert(f32[192]{0} %divide.805), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reshape.807 = f32[1,1,1,1,192]{4,3,2,1,0} reshape(f32[192]{0} %convert.806), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reshape.808 = f32[1,192]{1,0} reshape(f32[1,1,1,1,192]{4,3,2,1,0} %reshape.807), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.809 = f32[1,16,28,28,192]{4,3,2,1,0} broadcast(f32[1,192]{1,0} %reshape.808), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.810 = f32[1,16,28,28,192]{4,3,2,1,0} subtract(f32[1,16,28,28,192]{4,3,2,1,0} %broadcast.809, f32[1,16,28,28,192]{4,3,2,1,0} %convolution.787), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.811 = f32[1,16,28,28,192]{4,3,2,1,0} multiply(f32[1,16,28,28,192]{4,3,2,1,0} %subtract.810, f32[1,16,28,28,192]{4,3,2,1,0} %subtract.810), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %convert.812 = f32[1,16,28,28,192]{4,3,2,1,0} convert(f32[1,16,28,28,192]{4,3,2,1,0} %multiply.811), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %constant.813 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.814 = f32[] convert(f32[] %constant.813), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %reduce.819 = f32[192]{0} reduce(f32[1,16,28,28,192]{4,3,2,1,0} %convert.812, f32[] %convert.814), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_3c_Branch_1_Conv3d_0b_3x3_batch_norm_normalize_moments_variance-reduction.815, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.820 = s32[] get-dimension-size(f32[1,16,28,28,192]{4,3,2,1,0} %convert.812), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.821 = s32[] get-dimension-size(f32[1,16,28,28,192]{4,3,2,1,0} %convert.812), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.822 = s32[] multiply(s32[] %get-dimension-size.820, s32[] %get-dimension-size.821), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.823 = s32[] get-dimension-size(f32[1,16,28,28,192]{4,3,2,1,0} %convert.812), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.824 = s32[] multiply(s32[] %multiply.822, s32[] %get-dimension-size.823), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.825 = s32[] get-dimension-size(f32[1,16,28,28,192]{4,3,2,1,0} %convert.812), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.826 = s32[] multiply(s32[] %multiply.824, s32[] %get-dimension-size.825), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.827 = f32[] convert(s32[] %multiply.826), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %broadcast.828 = f32[192]{0} broadcast(f32[] %convert.827), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %divide.829 = f32[192]{0} divide(f32[192]{0} %reduce.819, f32[192]{0} %broadcast.828), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.830 = f32[192]{0} convert(f32[192]{0} %divide.829), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %reshape.831 = f32[1,1,1,1,192]{4,3,2,1,0} reshape(f32[192]{0} %convert.830), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %add.834 = f32[1,1,1,1,192]{4,3,2,1,0} add(f32[1,1,1,1,192]{4,3,2,1,0} %broadcast.833, f32[1,1,1,1,192]{4,3,2,1,0} %reshape.831), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %rsqrt.835 = f32[1,1,1,1,192]{4,3,2,1,0} rsqrt(f32[1,1,1,1,192]{4,3,2,1,0} %add.834), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.836 = f32[1,192]{1,0} reshape(f32[1,1,1,1,192]{4,3,2,1,0} %rsqrt.835), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %broadcast.837 = f32[1,16,28,28,192]{4,3,2,1,0} broadcast(f32[1,192]{1,0} %reshape.836), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %multiply.838 = f32[1,16,28,28,192]{4,3,2,1,0} multiply(f32[1,16,28,28,192]{4,3,2,1,0} %broadcast.837, f32[1,16,28,28,192]{4,3,2,1,0} %convolution.787), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %arg70.71 = f32[1,1,1,1,192]{4,3,2,1,0} parameter(70), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.839 = f32[1,1,1,1,192]{4,3,2,1,0} multiply(f32[1,1,1,1,192]{4,3,2,1,0} %rsqrt.835, f32[1,1,1,1,192]{4,3,2,1,0} %reshape.807), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.840 = f32[1,1,1,1,192]{4,3,2,1,0} subtract(f32[1,1,1,1,192]{4,3,2,1,0} %arg70.71, f32[1,1,1,1,192]{4,3,2,1,0} %multiply.839), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/sub"}
  %reshape.841 = f32[1,192]{1,0} reshape(f32[1,1,1,1,192]{4,3,2,1,0} %subtract.840), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.842 = f32[1,16,28,28,192]{4,3,2,1,0} broadcast(f32[1,192]{1,0} %reshape.841), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %add.843 = f32[1,16,28,28,192]{4,3,2,1,0} add(f32[1,16,28,28,192]{4,3,2,1,0} %multiply.838, f32[1,16,28,28,192]{4,3,2,1,0} %broadcast.842), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %constant.949 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %broadcast.950 = f32[1,1,1,1,96]{4,3,2,1,0} broadcast(f32[] %constant.949), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %constant.901 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/Relu"}
  %broadcast.902 = f32[1,16,28,28,32]{4,3,2,1,0} broadcast(f32[] %constant.901), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/Relu"}
  %constant.889 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.890 = f32[1,1,1,1,32]{4,3,2,1,0} broadcast(f32[] %constant.889), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %arg73.74 = f32[1,1,1,256,32]{4,3,2,1,0} parameter(73), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.844 = f32[1,16,28,28,32]{4,3,2,1,0} convolution(f32[1,16,28,28,256]{4,3,2,1,0} %maximum.669, f32[1,1,1,256,32]{4,3,2,1,0} %arg73.74), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/conv_3d/convolution"}
  %convert.845 = f32[1,16,28,28,32]{4,3,2,1,0} convert(f32[1,16,28,28,32]{4,3,2,1,0} %convolution.844), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %constant.846 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.847 = f32[] convert(f32[] %constant.846), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reduce.852 = f32[32]{0} reduce(f32[1,16,28,28,32]{4,3,2,1,0} %convert.845, f32[] %convert.847), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_3c_Branch_2_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.848, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.853 = s32[] get-dimension-size(f32[1,16,28,28,32]{4,3,2,1,0} %convert.845), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.854 = s32[] get-dimension-size(f32[1,16,28,28,32]{4,3,2,1,0} %convert.845), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.855 = s32[] multiply(s32[] %get-dimension-size.853, s32[] %get-dimension-size.854), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.856 = s32[] get-dimension-size(f32[1,16,28,28,32]{4,3,2,1,0} %convert.845), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.857 = s32[] multiply(s32[] %multiply.855, s32[] %get-dimension-size.856), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.858 = s32[] get-dimension-size(f32[1,16,28,28,32]{4,3,2,1,0} %convert.845), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.859 = s32[] multiply(s32[] %multiply.857, s32[] %get-dimension-size.858), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.860 = f32[] convert(s32[] %multiply.859), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.861 = f32[32]{0} broadcast(f32[] %convert.860), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %divide.862 = f32[32]{0} divide(f32[32]{0} %reduce.852, f32[32]{0} %broadcast.861), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.863 = f32[32]{0} convert(f32[32]{0} %divide.862), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.864 = f32[1,1,1,1,32]{4,3,2,1,0} reshape(f32[32]{0} %convert.863), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.865 = f32[1,32]{1,0} reshape(f32[1,1,1,1,32]{4,3,2,1,0} %reshape.864), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.866 = f32[1,16,28,28,32]{4,3,2,1,0} broadcast(f32[1,32]{1,0} %reshape.865), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.867 = f32[1,16,28,28,32]{4,3,2,1,0} subtract(f32[1,16,28,28,32]{4,3,2,1,0} %broadcast.866, f32[1,16,28,28,32]{4,3,2,1,0} %convolution.844), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.868 = f32[1,16,28,28,32]{4,3,2,1,0} multiply(f32[1,16,28,28,32]{4,3,2,1,0} %subtract.867, f32[1,16,28,28,32]{4,3,2,1,0} %subtract.867), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.869 = f32[1,16,28,28,32]{4,3,2,1,0} convert(f32[1,16,28,28,32]{4,3,2,1,0} %multiply.868), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %constant.870 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.871 = f32[] convert(f32[] %constant.870), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reduce.876 = f32[32]{0} reduce(f32[1,16,28,28,32]{4,3,2,1,0} %convert.869, f32[] %convert.871), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_3c_Branch_2_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.872, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.877 = s32[] get-dimension-size(f32[1,16,28,28,32]{4,3,2,1,0} %convert.869), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.878 = s32[] get-dimension-size(f32[1,16,28,28,32]{4,3,2,1,0} %convert.869), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.879 = s32[] multiply(s32[] %get-dimension-size.877, s32[] %get-dimension-size.878), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.880 = s32[] get-dimension-size(f32[1,16,28,28,32]{4,3,2,1,0} %convert.869), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.881 = s32[] multiply(s32[] %multiply.879, s32[] %get-dimension-size.880), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.882 = s32[] get-dimension-size(f32[1,16,28,28,32]{4,3,2,1,0} %convert.869), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.883 = s32[] multiply(s32[] %multiply.881, s32[] %get-dimension-size.882), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.884 = f32[] convert(s32[] %multiply.883), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.885 = f32[32]{0} broadcast(f32[] %convert.884), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %divide.886 = f32[32]{0} divide(f32[32]{0} %reduce.876, f32[32]{0} %broadcast.885), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.887 = f32[32]{0} convert(f32[32]{0} %divide.886), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reshape.888 = f32[1,1,1,1,32]{4,3,2,1,0} reshape(f32[32]{0} %convert.887), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %add.891 = f32[1,1,1,1,32]{4,3,2,1,0} add(f32[1,1,1,1,32]{4,3,2,1,0} %broadcast.890, f32[1,1,1,1,32]{4,3,2,1,0} %reshape.888), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.892 = f32[1,1,1,1,32]{4,3,2,1,0} rsqrt(f32[1,1,1,1,32]{4,3,2,1,0} %add.891), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.893 = f32[1,32]{1,0} reshape(f32[1,1,1,1,32]{4,3,2,1,0} %rsqrt.892), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.894 = f32[1,16,28,28,32]{4,3,2,1,0} broadcast(f32[1,32]{1,0} %reshape.893), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %multiply.895 = f32[1,16,28,28,32]{4,3,2,1,0} multiply(f32[1,16,28,28,32]{4,3,2,1,0} %broadcast.894, f32[1,16,28,28,32]{4,3,2,1,0} %convolution.844), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg77.78 = f32[1,1,1,1,32]{4,3,2,1,0} parameter(77), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.896 = f32[1,1,1,1,32]{4,3,2,1,0} multiply(f32[1,1,1,1,32]{4,3,2,1,0} %rsqrt.892, f32[1,1,1,1,32]{4,3,2,1,0} %reshape.864), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.897 = f32[1,1,1,1,32]{4,3,2,1,0} subtract(f32[1,1,1,1,32]{4,3,2,1,0} %arg77.78, f32[1,1,1,1,32]{4,3,2,1,0} %multiply.896), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.898 = f32[1,32]{1,0} reshape(f32[1,1,1,1,32]{4,3,2,1,0} %subtract.897), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.899 = f32[1,16,28,28,32]{4,3,2,1,0} broadcast(f32[1,32]{1,0} %reshape.898), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.900 = f32[1,16,28,28,32]{4,3,2,1,0} add(f32[1,16,28,28,32]{4,3,2,1,0} %multiply.895, f32[1,16,28,28,32]{4,3,2,1,0} %broadcast.899), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %maximum.903 = f32[1,16,28,28,32]{4,3,2,1,0} maximum(f32[1,16,28,28,32]{4,3,2,1,0} %broadcast.902, f32[1,16,28,28,32]{4,3,2,1,0} %add.900), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/Relu"}
  %arg81.82 = f32[3,3,3,32,96]{4,3,2,1,0} parameter(81), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.904 = f32[1,16,28,28,96]{4,3,2,1,0} convolution(f32[1,16,28,28,32]{4,3,2,1,0} %maximum.903, f32[3,3,3,32,96]{4,3,2,1,0} %arg81.82), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/conv_3d/convolution"}
  %convert.905 = f32[1,16,28,28,96]{4,3,2,1,0} convert(f32[1,16,28,28,96]{4,3,2,1,0} %convolution.904), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %constant.906 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.907 = f32[] convert(f32[] %constant.906), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reduce.912 = f32[96]{0} reduce(f32[1,16,28,28,96]{4,3,2,1,0} %convert.905, f32[] %convert.907), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_3c_Branch_2_Conv3d_0b_3x3_batch_norm_normalize_moments_mean-reduction.908, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.913 = s32[] get-dimension-size(f32[1,16,28,28,96]{4,3,2,1,0} %convert.905), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.914 = s32[] get-dimension-size(f32[1,16,28,28,96]{4,3,2,1,0} %convert.905), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.915 = s32[] multiply(s32[] %get-dimension-size.913, s32[] %get-dimension-size.914), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.916 = s32[] get-dimension-size(f32[1,16,28,28,96]{4,3,2,1,0} %convert.905), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.917 = s32[] multiply(s32[] %multiply.915, s32[] %get-dimension-size.916), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.918 = s32[] get-dimension-size(f32[1,16,28,28,96]{4,3,2,1,0} %convert.905), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.919 = s32[] multiply(s32[] %multiply.917, s32[] %get-dimension-size.918), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.920 = f32[] convert(s32[] %multiply.919), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %broadcast.921 = f32[96]{0} broadcast(f32[] %convert.920), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %divide.922 = f32[96]{0} divide(f32[96]{0} %reduce.912, f32[96]{0} %broadcast.921), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.923 = f32[96]{0} convert(f32[96]{0} %divide.922), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reshape.924 = f32[1,1,1,1,96]{4,3,2,1,0} reshape(f32[96]{0} %convert.923), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reshape.925 = f32[1,96]{1,0} reshape(f32[1,1,1,1,96]{4,3,2,1,0} %reshape.924), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.926 = f32[1,16,28,28,96]{4,3,2,1,0} broadcast(f32[1,96]{1,0} %reshape.925), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.927 = f32[1,16,28,28,96]{4,3,2,1,0} subtract(f32[1,16,28,28,96]{4,3,2,1,0} %broadcast.926, f32[1,16,28,28,96]{4,3,2,1,0} %convolution.904), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.928 = f32[1,16,28,28,96]{4,3,2,1,0} multiply(f32[1,16,28,28,96]{4,3,2,1,0} %subtract.927, f32[1,16,28,28,96]{4,3,2,1,0} %subtract.927), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %convert.929 = f32[1,16,28,28,96]{4,3,2,1,0} convert(f32[1,16,28,28,96]{4,3,2,1,0} %multiply.928), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %constant.930 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.931 = f32[] convert(f32[] %constant.930), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %reduce.936 = f32[96]{0} reduce(f32[1,16,28,28,96]{4,3,2,1,0} %convert.929, f32[] %convert.931), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_3c_Branch_2_Conv3d_0b_3x3_batch_norm_normalize_moments_variance-reduction.932, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.937 = s32[] get-dimension-size(f32[1,16,28,28,96]{4,3,2,1,0} %convert.929), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.938 = s32[] get-dimension-size(f32[1,16,28,28,96]{4,3,2,1,0} %convert.929), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.939 = s32[] multiply(s32[] %get-dimension-size.937, s32[] %get-dimension-size.938), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.940 = s32[] get-dimension-size(f32[1,16,28,28,96]{4,3,2,1,0} %convert.929), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.941 = s32[] multiply(s32[] %multiply.939, s32[] %get-dimension-size.940), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.942 = s32[] get-dimension-size(f32[1,16,28,28,96]{4,3,2,1,0} %convert.929), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.943 = s32[] multiply(s32[] %multiply.941, s32[] %get-dimension-size.942), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.944 = f32[] convert(s32[] %multiply.943), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %broadcast.945 = f32[96]{0} broadcast(f32[] %convert.944), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %divide.946 = f32[96]{0} divide(f32[96]{0} %reduce.936, f32[96]{0} %broadcast.945), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.947 = f32[96]{0} convert(f32[96]{0} %divide.946), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %reshape.948 = f32[1,1,1,1,96]{4,3,2,1,0} reshape(f32[96]{0} %convert.947), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %add.951 = f32[1,1,1,1,96]{4,3,2,1,0} add(f32[1,1,1,1,96]{4,3,2,1,0} %broadcast.950, f32[1,1,1,1,96]{4,3,2,1,0} %reshape.948), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %rsqrt.952 = f32[1,1,1,1,96]{4,3,2,1,0} rsqrt(f32[1,1,1,1,96]{4,3,2,1,0} %add.951), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.953 = f32[1,96]{1,0} reshape(f32[1,1,1,1,96]{4,3,2,1,0} %rsqrt.952), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %broadcast.954 = f32[1,16,28,28,96]{4,3,2,1,0} broadcast(f32[1,96]{1,0} %reshape.953), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %multiply.955 = f32[1,16,28,28,96]{4,3,2,1,0} multiply(f32[1,16,28,28,96]{4,3,2,1,0} %broadcast.954, f32[1,16,28,28,96]{4,3,2,1,0} %convolution.904), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %arg82.83 = f32[1,1,1,1,96]{4,3,2,1,0} parameter(82), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.956 = f32[1,1,1,1,96]{4,3,2,1,0} multiply(f32[1,1,1,1,96]{4,3,2,1,0} %rsqrt.952, f32[1,1,1,1,96]{4,3,2,1,0} %reshape.924), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.957 = f32[1,1,1,1,96]{4,3,2,1,0} subtract(f32[1,1,1,1,96]{4,3,2,1,0} %arg82.83, f32[1,1,1,1,96]{4,3,2,1,0} %multiply.956), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/sub"}
  %reshape.958 = f32[1,96]{1,0} reshape(f32[1,1,1,1,96]{4,3,2,1,0} %subtract.957), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.959 = f32[1,16,28,28,96]{4,3,2,1,0} broadcast(f32[1,96]{1,0} %reshape.958), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %add.960 = f32[1,16,28,28,96]{4,3,2,1,0} add(f32[1,16,28,28,96]{4,3,2,1,0} %multiply.955, f32[1,16,28,28,96]{4,3,2,1,0} %broadcast.959), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %constant.1012 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %broadcast.1013 = f32[1,1,1,1,64]{4,3,2,1,0} broadcast(f32[] %constant.1012), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %constant.961 = f32[] constant(-inf), metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/MaxPool3d_0a_3x3"}
  %reduce-window.966 = f32[1,16,28,28,256]{4,3,2,1,0} reduce-window(f32[1,16,28,28,256]{4,3,2,1,0} %maximum.669, f32[] %constant.961), window={size=1x3x3x3x1 pad=0_0x1_1x1_1x1_1x0_0}, to_apply=%max_F32.962, metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/MaxPool3d_0a_3x3"}
  %arg85.86 = f32[1,1,1,256,64]{4,3,2,1,0} parameter(85), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.967 = f32[1,16,28,28,64]{4,3,2,1,0} convolution(f32[1,16,28,28,256]{4,3,2,1,0} %reduce-window.966, f32[1,1,1,256,64]{4,3,2,1,0} %arg85.86), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/conv_3d/convolution"}
  %convert.968 = f32[1,16,28,28,64]{4,3,2,1,0} convert(f32[1,16,28,28,64]{4,3,2,1,0} %convolution.967), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %constant.969 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %convert.970 = f32[] convert(f32[] %constant.969), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %reduce.975 = f32[64]{0} reduce(f32[1,16,28,28,64]{4,3,2,1,0} %convert.968, f32[] %convert.970), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_3c_Branch_3_Conv3d_0b_1x1_batch_norm_normalize_moments_mean-reduction.971, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.976 = s32[] get-dimension-size(f32[1,16,28,28,64]{4,3,2,1,0} %convert.968), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.977 = s32[] get-dimension-size(f32[1,16,28,28,64]{4,3,2,1,0} %convert.968), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %multiply.978 = s32[] multiply(s32[] %get-dimension-size.976, s32[] %get-dimension-size.977), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.979 = s32[] get-dimension-size(f32[1,16,28,28,64]{4,3,2,1,0} %convert.968), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %multiply.980 = s32[] multiply(s32[] %multiply.978, s32[] %get-dimension-size.979), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.981 = s32[] get-dimension-size(f32[1,16,28,28,64]{4,3,2,1,0} %convert.968), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %multiply.982 = s32[] multiply(s32[] %multiply.980, s32[] %get-dimension-size.981), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %convert.983 = f32[] convert(s32[] %multiply.982), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.984 = f32[64]{0} broadcast(f32[] %convert.983), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %divide.985 = f32[64]{0} divide(f32[64]{0} %reduce.975, f32[64]{0} %broadcast.984), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %convert.986 = f32[64]{0} convert(f32[64]{0} %divide.985), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %reshape.987 = f32[1,1,1,1,64]{4,3,2,1,0} reshape(f32[64]{0} %convert.986), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %reshape.988 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %reshape.987), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.989 = f32[1,16,28,28,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.988), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.990 = f32[1,16,28,28,64]{4,3,2,1,0} subtract(f32[1,16,28,28,64]{4,3,2,1,0} %broadcast.989, f32[1,16,28,28,64]{4,3,2,1,0} %convolution.967), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.991 = f32[1,16,28,28,64]{4,3,2,1,0} multiply(f32[1,16,28,28,64]{4,3,2,1,0} %subtract.990, f32[1,16,28,28,64]{4,3,2,1,0} %subtract.990), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.992 = f32[1,16,28,28,64]{4,3,2,1,0} convert(f32[1,16,28,28,64]{4,3,2,1,0} %multiply.991), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %constant.993 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %convert.994 = f32[] convert(f32[] %constant.993), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %reduce.999 = f32[64]{0} reduce(f32[1,16,28,28,64]{4,3,2,1,0} %convert.992, f32[] %convert.994), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_3c_Branch_3_Conv3d_0b_1x1_batch_norm_normalize_moments_variance-reduction.995, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1000 = s32[] get-dimension-size(f32[1,16,28,28,64]{4,3,2,1,0} %convert.992), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1001 = s32[] get-dimension-size(f32[1,16,28,28,64]{4,3,2,1,0} %convert.992), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %multiply.1002 = s32[] multiply(s32[] %get-dimension-size.1000, s32[] %get-dimension-size.1001), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1003 = s32[] get-dimension-size(f32[1,16,28,28,64]{4,3,2,1,0} %convert.992), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %multiply.1004 = s32[] multiply(s32[] %multiply.1002, s32[] %get-dimension-size.1003), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1005 = s32[] get-dimension-size(f32[1,16,28,28,64]{4,3,2,1,0} %convert.992), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %multiply.1006 = s32[] multiply(s32[] %multiply.1004, s32[] %get-dimension-size.1005), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %convert.1007 = f32[] convert(s32[] %multiply.1006), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.1008 = f32[64]{0} broadcast(f32[] %convert.1007), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %divide.1009 = f32[64]{0} divide(f32[64]{0} %reduce.999, f32[64]{0} %broadcast.1008), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %convert.1010 = f32[64]{0} convert(f32[64]{0} %divide.1009), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %reshape.1011 = f32[1,1,1,1,64]{4,3,2,1,0} reshape(f32[64]{0} %convert.1010), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %add.1014 = f32[1,1,1,1,64]{4,3,2,1,0} add(f32[1,1,1,1,64]{4,3,2,1,0} %broadcast.1013, f32[1,1,1,1,64]{4,3,2,1,0} %reshape.1011), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %rsqrt.1015 = f32[1,1,1,1,64]{4,3,2,1,0} rsqrt(f32[1,1,1,1,64]{4,3,2,1,0} %add.1014), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.1016 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.1015), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %broadcast.1017 = f32[1,16,28,28,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.1016), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %multiply.1018 = f32[1,16,28,28,64]{4,3,2,1,0} multiply(f32[1,16,28,28,64]{4,3,2,1,0} %broadcast.1017, f32[1,16,28,28,64]{4,3,2,1,0} %convolution.967), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %arg88.89 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(88), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.1019 = f32[1,1,1,1,64]{4,3,2,1,0} multiply(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.1015, f32[1,1,1,1,64]{4,3,2,1,0} %reshape.987), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.1020 = f32[1,1,1,1,64]{4,3,2,1,0} subtract(f32[1,1,1,1,64]{4,3,2,1,0} %arg88.89, f32[1,1,1,1,64]{4,3,2,1,0} %multiply.1019), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/sub"}
  %reshape.1021 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %subtract.1020), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.1022 = f32[1,16,28,28,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.1021), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %add.1023 = f32[1,16,28,28,64]{4,3,2,1,0} add(f32[1,16,28,28,64]{4,3,2,1,0} %multiply.1018, f32[1,16,28,28,64]{4,3,2,1,0} %broadcast.1022), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %concatenate.1024 = f32[1,16,28,28,480]{4,3,2,1,0} concatenate(f32[1,16,28,28,128]{4,3,2,1,0} %add.726, f32[1,16,28,28,192]{4,3,2,1,0} %add.843, f32[1,16,28,28,96]{4,3,2,1,0} %add.960, f32[1,16,28,28,64]{4,3,2,1,0} %add.1023), dimensions={4}, metadata={op_type="ConcatV2" op_name="RGB/inception_i3d/Mixed_3c/concat"}
  %constant.1025 = f32[] constant(-inf), metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/MaxPool3d_4a_3x3"}
  %reduce-window.1030 = f32[1,8,14,14,480]{4,3,2,1,0} reduce-window(f32[1,16,28,28,480]{4,3,2,1,0} %concatenate.1024, f32[] %constant.1025), window={size=1x3x3x3x1 stride=1x2x2x2x1 pad=0_0x0_1x0_1x0_1x0_0}, to_apply=%max_F32.1026, metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/MaxPool3d_4a_3x3"}
  %maximum.1033 = f32[1,8,14,14,480]{4,3,2,1,0} maximum(f32[1,8,14,14,480]{4,3,2,1,0} %broadcast.1032, f32[1,8,14,14,480]{4,3,2,1,0} %reduce-window.1030), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/Relu"}
  %arg90.91 = f32[1,1,1,480,192]{4,3,2,1,0} parameter(90), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.1034 = f32[1,8,14,14,192]{4,3,2,1,0} convolution(f32[1,8,14,14,480]{4,3,2,1,0} %maximum.1033, f32[1,1,1,480,192]{4,3,2,1,0} %arg90.91), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/conv_3d/convolution"}
  %convert.1035 = f32[1,8,14,14,192]{4,3,2,1,0} convert(f32[1,8,14,14,192]{4,3,2,1,0} %convolution.1034), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %constant.1036 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.1037 = f32[] convert(f32[] %constant.1036), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reduce.1042 = f32[192]{0} reduce(f32[1,8,14,14,192]{4,3,2,1,0} %convert.1035, f32[] %convert.1037), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4b_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.1038, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1043 = s32[] get-dimension-size(f32[1,8,14,14,192]{4,3,2,1,0} %convert.1035), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1044 = s32[] get-dimension-size(f32[1,8,14,14,192]{4,3,2,1,0} %convert.1035), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.1045 = s32[] multiply(s32[] %get-dimension-size.1043, s32[] %get-dimension-size.1044), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1046 = s32[] get-dimension-size(f32[1,8,14,14,192]{4,3,2,1,0} %convert.1035), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.1047 = s32[] multiply(s32[] %multiply.1045, s32[] %get-dimension-size.1046), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1048 = s32[] get-dimension-size(f32[1,8,14,14,192]{4,3,2,1,0} %convert.1035), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.1049 = s32[] multiply(s32[] %multiply.1047, s32[] %get-dimension-size.1048), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.1050 = f32[] convert(s32[] %multiply.1049), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.1051 = f32[192]{0} broadcast(f32[] %convert.1050), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %divide.1052 = f32[192]{0} divide(f32[192]{0} %reduce.1042, f32[192]{0} %broadcast.1051), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.1053 = f32[192]{0} convert(f32[192]{0} %divide.1052), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.1054 = f32[1,1,1,1,192]{4,3,2,1,0} reshape(f32[192]{0} %convert.1053), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.1055 = f32[1,192]{1,0} reshape(f32[1,1,1,1,192]{4,3,2,1,0} %reshape.1054), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.1056 = f32[1,8,14,14,192]{4,3,2,1,0} broadcast(f32[1,192]{1,0} %reshape.1055), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.1057 = f32[1,8,14,14,192]{4,3,2,1,0} subtract(f32[1,8,14,14,192]{4,3,2,1,0} %broadcast.1056, f32[1,8,14,14,192]{4,3,2,1,0} %convolution.1034), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.1058 = f32[1,8,14,14,192]{4,3,2,1,0} multiply(f32[1,8,14,14,192]{4,3,2,1,0} %subtract.1057, f32[1,8,14,14,192]{4,3,2,1,0} %subtract.1057), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.1059 = f32[1,8,14,14,192]{4,3,2,1,0} convert(f32[1,8,14,14,192]{4,3,2,1,0} %multiply.1058), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %constant.1060 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.1061 = f32[] convert(f32[] %constant.1060), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reduce.1066 = f32[192]{0} reduce(f32[1,8,14,14,192]{4,3,2,1,0} %convert.1059, f32[] %convert.1061), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4b_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.1062, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1067 = s32[] get-dimension-size(f32[1,8,14,14,192]{4,3,2,1,0} %convert.1059), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1068 = s32[] get-dimension-size(f32[1,8,14,14,192]{4,3,2,1,0} %convert.1059), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.1069 = s32[] multiply(s32[] %get-dimension-size.1067, s32[] %get-dimension-size.1068), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1070 = s32[] get-dimension-size(f32[1,8,14,14,192]{4,3,2,1,0} %convert.1059), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.1071 = s32[] multiply(s32[] %multiply.1069, s32[] %get-dimension-size.1070), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1072 = s32[] get-dimension-size(f32[1,8,14,14,192]{4,3,2,1,0} %convert.1059), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.1073 = s32[] multiply(s32[] %multiply.1071, s32[] %get-dimension-size.1072), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.1074 = f32[] convert(s32[] %multiply.1073), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.1075 = f32[192]{0} broadcast(f32[] %convert.1074), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %divide.1076 = f32[192]{0} divide(f32[192]{0} %reduce.1066, f32[192]{0} %broadcast.1075), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.1077 = f32[192]{0} convert(f32[192]{0} %divide.1076), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reshape.1078 = f32[1,1,1,1,192]{4,3,2,1,0} reshape(f32[192]{0} %convert.1077), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %add.1081 = f32[1,1,1,1,192]{4,3,2,1,0} add(f32[1,1,1,1,192]{4,3,2,1,0} %broadcast.1080, f32[1,1,1,1,192]{4,3,2,1,0} %reshape.1078), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.1082 = f32[1,1,1,1,192]{4,3,2,1,0} rsqrt(f32[1,1,1,1,192]{4,3,2,1,0} %add.1081), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.1083 = f32[1,192]{1,0} reshape(f32[1,1,1,1,192]{4,3,2,1,0} %rsqrt.1082), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.1084 = f32[1,8,14,14,192]{4,3,2,1,0} broadcast(f32[1,192]{1,0} %reshape.1083), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %multiply.1085 = f32[1,8,14,14,192]{4,3,2,1,0} multiply(f32[1,8,14,14,192]{4,3,2,1,0} %broadcast.1084, f32[1,8,14,14,192]{4,3,2,1,0} %convolution.1034), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg92.93 = f32[1,1,1,1,192]{4,3,2,1,0} parameter(92), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.1086 = f32[1,1,1,1,192]{4,3,2,1,0} multiply(f32[1,1,1,1,192]{4,3,2,1,0} %rsqrt.1082, f32[1,1,1,1,192]{4,3,2,1,0} %reshape.1054), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.1087 = f32[1,1,1,1,192]{4,3,2,1,0} subtract(f32[1,1,1,1,192]{4,3,2,1,0} %arg92.93, f32[1,1,1,1,192]{4,3,2,1,0} %multiply.1086), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.1088 = f32[1,192]{1,0} reshape(f32[1,1,1,1,192]{4,3,2,1,0} %subtract.1087), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.1089 = f32[1,8,14,14,192]{4,3,2,1,0} broadcast(f32[1,192]{1,0} %reshape.1088), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.1090 = f32[1,8,14,14,192]{4,3,2,1,0} add(f32[1,8,14,14,192]{4,3,2,1,0} %multiply.1085, f32[1,8,14,14,192]{4,3,2,1,0} %broadcast.1089), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %constant.1196 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %broadcast.1197 = f32[1,1,1,1,208]{4,3,2,1,0} broadcast(f32[] %constant.1196), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %constant.1148 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/Relu"}
  %broadcast.1149 = f32[1,8,14,14,96]{4,3,2,1,0} broadcast(f32[] %constant.1148), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/Relu"}
  %constant.1136 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.1137 = f32[1,1,1,1,96]{4,3,2,1,0} broadcast(f32[] %constant.1136), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %arg97.98 = f32[1,1,1,480,96]{4,3,2,1,0} parameter(97), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.1091 = f32[1,8,14,14,96]{4,3,2,1,0} convolution(f32[1,8,14,14,480]{4,3,2,1,0} %maximum.1033, f32[1,1,1,480,96]{4,3,2,1,0} %arg97.98), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/conv_3d/convolution"}
  %convert.1092 = f32[1,8,14,14,96]{4,3,2,1,0} convert(f32[1,8,14,14,96]{4,3,2,1,0} %convolution.1091), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %constant.1093 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.1094 = f32[] convert(f32[] %constant.1093), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reduce.1099 = f32[96]{0} reduce(f32[1,8,14,14,96]{4,3,2,1,0} %convert.1092, f32[] %convert.1094), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4b_Branch_1_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.1095, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1100 = s32[] get-dimension-size(f32[1,8,14,14,96]{4,3,2,1,0} %convert.1092), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1101 = s32[] get-dimension-size(f32[1,8,14,14,96]{4,3,2,1,0} %convert.1092), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.1102 = s32[] multiply(s32[] %get-dimension-size.1100, s32[] %get-dimension-size.1101), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1103 = s32[] get-dimension-size(f32[1,8,14,14,96]{4,3,2,1,0} %convert.1092), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.1104 = s32[] multiply(s32[] %multiply.1102, s32[] %get-dimension-size.1103), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1105 = s32[] get-dimension-size(f32[1,8,14,14,96]{4,3,2,1,0} %convert.1092), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.1106 = s32[] multiply(s32[] %multiply.1104, s32[] %get-dimension-size.1105), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.1107 = f32[] convert(s32[] %multiply.1106), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.1108 = f32[96]{0} broadcast(f32[] %convert.1107), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %divide.1109 = f32[96]{0} divide(f32[96]{0} %reduce.1099, f32[96]{0} %broadcast.1108), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.1110 = f32[96]{0} convert(f32[96]{0} %divide.1109), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.1111 = f32[1,1,1,1,96]{4,3,2,1,0} reshape(f32[96]{0} %convert.1110), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.1112 = f32[1,96]{1,0} reshape(f32[1,1,1,1,96]{4,3,2,1,0} %reshape.1111), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.1113 = f32[1,8,14,14,96]{4,3,2,1,0} broadcast(f32[1,96]{1,0} %reshape.1112), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.1114 = f32[1,8,14,14,96]{4,3,2,1,0} subtract(f32[1,8,14,14,96]{4,3,2,1,0} %broadcast.1113, f32[1,8,14,14,96]{4,3,2,1,0} %convolution.1091), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.1115 = f32[1,8,14,14,96]{4,3,2,1,0} multiply(f32[1,8,14,14,96]{4,3,2,1,0} %subtract.1114, f32[1,8,14,14,96]{4,3,2,1,0} %subtract.1114), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.1116 = f32[1,8,14,14,96]{4,3,2,1,0} convert(f32[1,8,14,14,96]{4,3,2,1,0} %multiply.1115), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %constant.1117 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.1118 = f32[] convert(f32[] %constant.1117), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reduce.1123 = f32[96]{0} reduce(f32[1,8,14,14,96]{4,3,2,1,0} %convert.1116, f32[] %convert.1118), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4b_Branch_1_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.1119, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1124 = s32[] get-dimension-size(f32[1,8,14,14,96]{4,3,2,1,0} %convert.1116), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1125 = s32[] get-dimension-size(f32[1,8,14,14,96]{4,3,2,1,0} %convert.1116), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.1126 = s32[] multiply(s32[] %get-dimension-size.1124, s32[] %get-dimension-size.1125), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1127 = s32[] get-dimension-size(f32[1,8,14,14,96]{4,3,2,1,0} %convert.1116), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.1128 = s32[] multiply(s32[] %multiply.1126, s32[] %get-dimension-size.1127), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1129 = s32[] get-dimension-size(f32[1,8,14,14,96]{4,3,2,1,0} %convert.1116), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.1130 = s32[] multiply(s32[] %multiply.1128, s32[] %get-dimension-size.1129), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.1131 = f32[] convert(s32[] %multiply.1130), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.1132 = f32[96]{0} broadcast(f32[] %convert.1131), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %divide.1133 = f32[96]{0} divide(f32[96]{0} %reduce.1123, f32[96]{0} %broadcast.1132), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.1134 = f32[96]{0} convert(f32[96]{0} %divide.1133), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reshape.1135 = f32[1,1,1,1,96]{4,3,2,1,0} reshape(f32[96]{0} %convert.1134), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %add.1138 = f32[1,1,1,1,96]{4,3,2,1,0} add(f32[1,1,1,1,96]{4,3,2,1,0} %broadcast.1137, f32[1,1,1,1,96]{4,3,2,1,0} %reshape.1135), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.1139 = f32[1,1,1,1,96]{4,3,2,1,0} rsqrt(f32[1,1,1,1,96]{4,3,2,1,0} %add.1138), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.1140 = f32[1,96]{1,0} reshape(f32[1,1,1,1,96]{4,3,2,1,0} %rsqrt.1139), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.1141 = f32[1,8,14,14,96]{4,3,2,1,0} broadcast(f32[1,96]{1,0} %reshape.1140), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %multiply.1142 = f32[1,8,14,14,96]{4,3,2,1,0} multiply(f32[1,8,14,14,96]{4,3,2,1,0} %broadcast.1141, f32[1,8,14,14,96]{4,3,2,1,0} %convolution.1091), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg101.102 = f32[1,1,1,1,96]{4,3,2,1,0} parameter(101), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.1143 = f32[1,1,1,1,96]{4,3,2,1,0} multiply(f32[1,1,1,1,96]{4,3,2,1,0} %rsqrt.1139, f32[1,1,1,1,96]{4,3,2,1,0} %reshape.1111), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.1144 = f32[1,1,1,1,96]{4,3,2,1,0} subtract(f32[1,1,1,1,96]{4,3,2,1,0} %arg101.102, f32[1,1,1,1,96]{4,3,2,1,0} %multiply.1143), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.1145 = f32[1,96]{1,0} reshape(f32[1,1,1,1,96]{4,3,2,1,0} %subtract.1144), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.1146 = f32[1,8,14,14,96]{4,3,2,1,0} broadcast(f32[1,96]{1,0} %reshape.1145), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.1147 = f32[1,8,14,14,96]{4,3,2,1,0} add(f32[1,8,14,14,96]{4,3,2,1,0} %multiply.1142, f32[1,8,14,14,96]{4,3,2,1,0} %broadcast.1146), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %maximum.1150 = f32[1,8,14,14,96]{4,3,2,1,0} maximum(f32[1,8,14,14,96]{4,3,2,1,0} %broadcast.1149, f32[1,8,14,14,96]{4,3,2,1,0} %add.1147), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/Relu"}
  %arg104.105 = f32[3,3,3,96,208]{4,3,2,1,0} parameter(104), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.1151 = f32[1,8,14,14,208]{4,3,2,1,0} convolution(f32[1,8,14,14,96]{4,3,2,1,0} %maximum.1150, f32[3,3,3,96,208]{4,3,2,1,0} %arg104.105), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/conv_3d/convolution"}
  %convert.1152 = f32[1,8,14,14,208]{4,3,2,1,0} convert(f32[1,8,14,14,208]{4,3,2,1,0} %convolution.1151), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %constant.1153 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.1154 = f32[] convert(f32[] %constant.1153), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reduce.1159 = f32[208]{0} reduce(f32[1,8,14,14,208]{4,3,2,1,0} %convert.1152, f32[] %convert.1154), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4b_Branch_1_Conv3d_0b_3x3_batch_norm_normalize_moments_mean-reduction.1155, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1160 = s32[] get-dimension-size(f32[1,8,14,14,208]{4,3,2,1,0} %convert.1152), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1161 = s32[] get-dimension-size(f32[1,8,14,14,208]{4,3,2,1,0} %convert.1152), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.1162 = s32[] multiply(s32[] %get-dimension-size.1160, s32[] %get-dimension-size.1161), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1163 = s32[] get-dimension-size(f32[1,8,14,14,208]{4,3,2,1,0} %convert.1152), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.1164 = s32[] multiply(s32[] %multiply.1162, s32[] %get-dimension-size.1163), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1165 = s32[] get-dimension-size(f32[1,8,14,14,208]{4,3,2,1,0} %convert.1152), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.1166 = s32[] multiply(s32[] %multiply.1164, s32[] %get-dimension-size.1165), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.1167 = f32[] convert(s32[] %multiply.1166), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %broadcast.1168 = f32[208]{0} broadcast(f32[] %convert.1167), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %divide.1169 = f32[208]{0} divide(f32[208]{0} %reduce.1159, f32[208]{0} %broadcast.1168), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.1170 = f32[208]{0} convert(f32[208]{0} %divide.1169), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reshape.1171 = f32[1,1,1,1,208]{4,3,2,1,0} reshape(f32[208]{0} %convert.1170), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reshape.1172 = f32[1,208]{1,0} reshape(f32[1,1,1,1,208]{4,3,2,1,0} %reshape.1171), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.1173 = f32[1,8,14,14,208]{4,3,2,1,0} broadcast(f32[1,208]{1,0} %reshape.1172), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.1174 = f32[1,8,14,14,208]{4,3,2,1,0} subtract(f32[1,8,14,14,208]{4,3,2,1,0} %broadcast.1173, f32[1,8,14,14,208]{4,3,2,1,0} %convolution.1151), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.1175 = f32[1,8,14,14,208]{4,3,2,1,0} multiply(f32[1,8,14,14,208]{4,3,2,1,0} %subtract.1174, f32[1,8,14,14,208]{4,3,2,1,0} %subtract.1174), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %convert.1176 = f32[1,8,14,14,208]{4,3,2,1,0} convert(f32[1,8,14,14,208]{4,3,2,1,0} %multiply.1175), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %constant.1177 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.1178 = f32[] convert(f32[] %constant.1177), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %reduce.1183 = f32[208]{0} reduce(f32[1,8,14,14,208]{4,3,2,1,0} %convert.1176, f32[] %convert.1178), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4b_Branch_1_Conv3d_0b_3x3_batch_norm_normalize_moments_variance-reduction.1179, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1184 = s32[] get-dimension-size(f32[1,8,14,14,208]{4,3,2,1,0} %convert.1176), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1185 = s32[] get-dimension-size(f32[1,8,14,14,208]{4,3,2,1,0} %convert.1176), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.1186 = s32[] multiply(s32[] %get-dimension-size.1184, s32[] %get-dimension-size.1185), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1187 = s32[] get-dimension-size(f32[1,8,14,14,208]{4,3,2,1,0} %convert.1176), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.1188 = s32[] multiply(s32[] %multiply.1186, s32[] %get-dimension-size.1187), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1189 = s32[] get-dimension-size(f32[1,8,14,14,208]{4,3,2,1,0} %convert.1176), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.1190 = s32[] multiply(s32[] %multiply.1188, s32[] %get-dimension-size.1189), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.1191 = f32[] convert(s32[] %multiply.1190), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %broadcast.1192 = f32[208]{0} broadcast(f32[] %convert.1191), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %divide.1193 = f32[208]{0} divide(f32[208]{0} %reduce.1183, f32[208]{0} %broadcast.1192), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.1194 = f32[208]{0} convert(f32[208]{0} %divide.1193), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %reshape.1195 = f32[1,1,1,1,208]{4,3,2,1,0} reshape(f32[208]{0} %convert.1194), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %add.1198 = f32[1,1,1,1,208]{4,3,2,1,0} add(f32[1,1,1,1,208]{4,3,2,1,0} %broadcast.1197, f32[1,1,1,1,208]{4,3,2,1,0} %reshape.1195), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %rsqrt.1199 = f32[1,1,1,1,208]{4,3,2,1,0} rsqrt(f32[1,1,1,1,208]{4,3,2,1,0} %add.1198), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.1200 = f32[1,208]{1,0} reshape(f32[1,1,1,1,208]{4,3,2,1,0} %rsqrt.1199), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %broadcast.1201 = f32[1,8,14,14,208]{4,3,2,1,0} broadcast(f32[1,208]{1,0} %reshape.1200), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %multiply.1202 = f32[1,8,14,14,208]{4,3,2,1,0} multiply(f32[1,8,14,14,208]{4,3,2,1,0} %broadcast.1201, f32[1,8,14,14,208]{4,3,2,1,0} %convolution.1151), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %arg106.107 = f32[1,1,1,1,208]{4,3,2,1,0} parameter(106), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.1203 = f32[1,1,1,1,208]{4,3,2,1,0} multiply(f32[1,1,1,1,208]{4,3,2,1,0} %rsqrt.1199, f32[1,1,1,1,208]{4,3,2,1,0} %reshape.1171), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.1204 = f32[1,1,1,1,208]{4,3,2,1,0} subtract(f32[1,1,1,1,208]{4,3,2,1,0} %arg106.107, f32[1,1,1,1,208]{4,3,2,1,0} %multiply.1203), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/sub"}
  %reshape.1205 = f32[1,208]{1,0} reshape(f32[1,1,1,1,208]{4,3,2,1,0} %subtract.1204), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.1206 = f32[1,8,14,14,208]{4,3,2,1,0} broadcast(f32[1,208]{1,0} %reshape.1205), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %add.1207 = f32[1,8,14,14,208]{4,3,2,1,0} add(f32[1,8,14,14,208]{4,3,2,1,0} %multiply.1202, f32[1,8,14,14,208]{4,3,2,1,0} %broadcast.1206), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %constant.1313 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %broadcast.1314 = f32[1,1,1,1,48]{4,3,2,1,0} broadcast(f32[] %constant.1313), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %constant.1265 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/Relu"}
  %broadcast.1266 = f32[1,8,14,14,16]{4,3,2,1,0} broadcast(f32[] %constant.1265), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/Relu"}
  %constant.1253 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.1254 = f32[1,1,1,1,16]{4,3,2,1,0} broadcast(f32[] %constant.1253), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %arg111.112 = f32[1,1,1,480,16]{4,3,2,1,0} parameter(111), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.1208 = f32[1,8,14,14,16]{4,3,2,1,0} convolution(f32[1,8,14,14,480]{4,3,2,1,0} %maximum.1033, f32[1,1,1,480,16]{4,3,2,1,0} %arg111.112), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/conv_3d/convolution"}
  %convert.1209 = f32[1,8,14,14,16]{4,3,2,1,0} convert(f32[1,8,14,14,16]{4,3,2,1,0} %convolution.1208), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %constant.1210 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.1211 = f32[] convert(f32[] %constant.1210), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reduce.1216 = f32[16]{0} reduce(f32[1,8,14,14,16]{4,3,2,1,0} %convert.1209, f32[] %convert.1211), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4b_Branch_2_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.1212, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1217 = s32[] get-dimension-size(f32[1,8,14,14,16]{4,3,2,1,0} %convert.1209), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1218 = s32[] get-dimension-size(f32[1,8,14,14,16]{4,3,2,1,0} %convert.1209), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.1219 = s32[] multiply(s32[] %get-dimension-size.1217, s32[] %get-dimension-size.1218), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1220 = s32[] get-dimension-size(f32[1,8,14,14,16]{4,3,2,1,0} %convert.1209), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.1221 = s32[] multiply(s32[] %multiply.1219, s32[] %get-dimension-size.1220), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1222 = s32[] get-dimension-size(f32[1,8,14,14,16]{4,3,2,1,0} %convert.1209), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.1223 = s32[] multiply(s32[] %multiply.1221, s32[] %get-dimension-size.1222), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.1224 = f32[] convert(s32[] %multiply.1223), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.1225 = f32[16]{0} broadcast(f32[] %convert.1224), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %divide.1226 = f32[16]{0} divide(f32[16]{0} %reduce.1216, f32[16]{0} %broadcast.1225), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.1227 = f32[16]{0} convert(f32[16]{0} %divide.1226), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.1228 = f32[1,1,1,1,16]{4,3,2,1,0} reshape(f32[16]{0} %convert.1227), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.1229 = f32[1,16]{1,0} reshape(f32[1,1,1,1,16]{4,3,2,1,0} %reshape.1228), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.1230 = f32[1,8,14,14,16]{4,3,2,1,0} broadcast(f32[1,16]{1,0} %reshape.1229), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.1231 = f32[1,8,14,14,16]{4,3,2,1,0} subtract(f32[1,8,14,14,16]{4,3,2,1,0} %broadcast.1230, f32[1,8,14,14,16]{4,3,2,1,0} %convolution.1208), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.1232 = f32[1,8,14,14,16]{4,3,2,1,0} multiply(f32[1,8,14,14,16]{4,3,2,1,0} %subtract.1231, f32[1,8,14,14,16]{4,3,2,1,0} %subtract.1231), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.1233 = f32[1,8,14,14,16]{4,3,2,1,0} convert(f32[1,8,14,14,16]{4,3,2,1,0} %multiply.1232), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %constant.1234 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.1235 = f32[] convert(f32[] %constant.1234), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reduce.1240 = f32[16]{0} reduce(f32[1,8,14,14,16]{4,3,2,1,0} %convert.1233, f32[] %convert.1235), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4b_Branch_2_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.1236, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1241 = s32[] get-dimension-size(f32[1,8,14,14,16]{4,3,2,1,0} %convert.1233), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1242 = s32[] get-dimension-size(f32[1,8,14,14,16]{4,3,2,1,0} %convert.1233), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.1243 = s32[] multiply(s32[] %get-dimension-size.1241, s32[] %get-dimension-size.1242), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1244 = s32[] get-dimension-size(f32[1,8,14,14,16]{4,3,2,1,0} %convert.1233), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.1245 = s32[] multiply(s32[] %multiply.1243, s32[] %get-dimension-size.1244), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1246 = s32[] get-dimension-size(f32[1,8,14,14,16]{4,3,2,1,0} %convert.1233), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.1247 = s32[] multiply(s32[] %multiply.1245, s32[] %get-dimension-size.1246), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.1248 = f32[] convert(s32[] %multiply.1247), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.1249 = f32[16]{0} broadcast(f32[] %convert.1248), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %divide.1250 = f32[16]{0} divide(f32[16]{0} %reduce.1240, f32[16]{0} %broadcast.1249), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.1251 = f32[16]{0} convert(f32[16]{0} %divide.1250), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reshape.1252 = f32[1,1,1,1,16]{4,3,2,1,0} reshape(f32[16]{0} %convert.1251), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %add.1255 = f32[1,1,1,1,16]{4,3,2,1,0} add(f32[1,1,1,1,16]{4,3,2,1,0} %broadcast.1254, f32[1,1,1,1,16]{4,3,2,1,0} %reshape.1252), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.1256 = f32[1,1,1,1,16]{4,3,2,1,0} rsqrt(f32[1,1,1,1,16]{4,3,2,1,0} %add.1255), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.1257 = f32[1,16]{1,0} reshape(f32[1,1,1,1,16]{4,3,2,1,0} %rsqrt.1256), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.1258 = f32[1,8,14,14,16]{4,3,2,1,0} broadcast(f32[1,16]{1,0} %reshape.1257), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %multiply.1259 = f32[1,8,14,14,16]{4,3,2,1,0} multiply(f32[1,8,14,14,16]{4,3,2,1,0} %broadcast.1258, f32[1,8,14,14,16]{4,3,2,1,0} %convolution.1208), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg113.114 = f32[1,1,1,1,16]{4,3,2,1,0} parameter(113), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.1260 = f32[1,1,1,1,16]{4,3,2,1,0} multiply(f32[1,1,1,1,16]{4,3,2,1,0} %rsqrt.1256, f32[1,1,1,1,16]{4,3,2,1,0} %reshape.1228), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.1261 = f32[1,1,1,1,16]{4,3,2,1,0} subtract(f32[1,1,1,1,16]{4,3,2,1,0} %arg113.114, f32[1,1,1,1,16]{4,3,2,1,0} %multiply.1260), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.1262 = f32[1,16]{1,0} reshape(f32[1,1,1,1,16]{4,3,2,1,0} %subtract.1261), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.1263 = f32[1,8,14,14,16]{4,3,2,1,0} broadcast(f32[1,16]{1,0} %reshape.1262), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.1264 = f32[1,8,14,14,16]{4,3,2,1,0} add(f32[1,8,14,14,16]{4,3,2,1,0} %multiply.1259, f32[1,8,14,14,16]{4,3,2,1,0} %broadcast.1263), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %maximum.1267 = f32[1,8,14,14,16]{4,3,2,1,0} maximum(f32[1,8,14,14,16]{4,3,2,1,0} %broadcast.1266, f32[1,8,14,14,16]{4,3,2,1,0} %add.1264), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/Relu"}
  %arg2.3 = f32[3,3,3,16,48]{4,3,2,1,0} parameter(2), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.1268 = f32[1,8,14,14,48]{4,3,2,1,0} convolution(f32[1,8,14,14,16]{4,3,2,1,0} %maximum.1267, f32[3,3,3,16,48]{4,3,2,1,0} %arg2.3), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/conv_3d/convolution"}
  %convert.1269 = f32[1,8,14,14,48]{4,3,2,1,0} convert(f32[1,8,14,14,48]{4,3,2,1,0} %convolution.1268), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %constant.1270 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.1271 = f32[] convert(f32[] %constant.1270), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reduce.1276 = f32[48]{0} reduce(f32[1,8,14,14,48]{4,3,2,1,0} %convert.1269, f32[] %convert.1271), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4b_Branch_2_Conv3d_0b_3x3_batch_norm_normalize_moments_mean-reduction.1272, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1277 = s32[] get-dimension-size(f32[1,8,14,14,48]{4,3,2,1,0} %convert.1269), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1278 = s32[] get-dimension-size(f32[1,8,14,14,48]{4,3,2,1,0} %convert.1269), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.1279 = s32[] multiply(s32[] %get-dimension-size.1277, s32[] %get-dimension-size.1278), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1280 = s32[] get-dimension-size(f32[1,8,14,14,48]{4,3,2,1,0} %convert.1269), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.1281 = s32[] multiply(s32[] %multiply.1279, s32[] %get-dimension-size.1280), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1282 = s32[] get-dimension-size(f32[1,8,14,14,48]{4,3,2,1,0} %convert.1269), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.1283 = s32[] multiply(s32[] %multiply.1281, s32[] %get-dimension-size.1282), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.1284 = f32[] convert(s32[] %multiply.1283), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %broadcast.1285 = f32[48]{0} broadcast(f32[] %convert.1284), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %divide.1286 = f32[48]{0} divide(f32[48]{0} %reduce.1276, f32[48]{0} %broadcast.1285), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.1287 = f32[48]{0} convert(f32[48]{0} %divide.1286), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reshape.1288 = f32[1,1,1,1,48]{4,3,2,1,0} reshape(f32[48]{0} %convert.1287), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reshape.1289 = f32[1,48]{1,0} reshape(f32[1,1,1,1,48]{4,3,2,1,0} %reshape.1288), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.1290 = f32[1,8,14,14,48]{4,3,2,1,0} broadcast(f32[1,48]{1,0} %reshape.1289), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.1291 = f32[1,8,14,14,48]{4,3,2,1,0} subtract(f32[1,8,14,14,48]{4,3,2,1,0} %broadcast.1290, f32[1,8,14,14,48]{4,3,2,1,0} %convolution.1268), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.1292 = f32[1,8,14,14,48]{4,3,2,1,0} multiply(f32[1,8,14,14,48]{4,3,2,1,0} %subtract.1291, f32[1,8,14,14,48]{4,3,2,1,0} %subtract.1291), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %convert.1293 = f32[1,8,14,14,48]{4,3,2,1,0} convert(f32[1,8,14,14,48]{4,3,2,1,0} %multiply.1292), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %constant.1294 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.1295 = f32[] convert(f32[] %constant.1294), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %reduce.1300 = f32[48]{0} reduce(f32[1,8,14,14,48]{4,3,2,1,0} %convert.1293, f32[] %convert.1295), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4b_Branch_2_Conv3d_0b_3x3_batch_norm_normalize_moments_variance-reduction.1296, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1301 = s32[] get-dimension-size(f32[1,8,14,14,48]{4,3,2,1,0} %convert.1293), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1302 = s32[] get-dimension-size(f32[1,8,14,14,48]{4,3,2,1,0} %convert.1293), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.1303 = s32[] multiply(s32[] %get-dimension-size.1301, s32[] %get-dimension-size.1302), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1304 = s32[] get-dimension-size(f32[1,8,14,14,48]{4,3,2,1,0} %convert.1293), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.1305 = s32[] multiply(s32[] %multiply.1303, s32[] %get-dimension-size.1304), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1306 = s32[] get-dimension-size(f32[1,8,14,14,48]{4,3,2,1,0} %convert.1293), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.1307 = s32[] multiply(s32[] %multiply.1305, s32[] %get-dimension-size.1306), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.1308 = f32[] convert(s32[] %multiply.1307), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %broadcast.1309 = f32[48]{0} broadcast(f32[] %convert.1308), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %divide.1310 = f32[48]{0} divide(f32[48]{0} %reduce.1300, f32[48]{0} %broadcast.1309), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.1311 = f32[48]{0} convert(f32[48]{0} %divide.1310), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %reshape.1312 = f32[1,1,1,1,48]{4,3,2,1,0} reshape(f32[48]{0} %convert.1311), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %add.1315 = f32[1,1,1,1,48]{4,3,2,1,0} add(f32[1,1,1,1,48]{4,3,2,1,0} %broadcast.1314, f32[1,1,1,1,48]{4,3,2,1,0} %reshape.1312), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %rsqrt.1316 = f32[1,1,1,1,48]{4,3,2,1,0} rsqrt(f32[1,1,1,1,48]{4,3,2,1,0} %add.1315), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.1317 = f32[1,48]{1,0} reshape(f32[1,1,1,1,48]{4,3,2,1,0} %rsqrt.1316), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %broadcast.1318 = f32[1,8,14,14,48]{4,3,2,1,0} broadcast(f32[1,48]{1,0} %reshape.1317), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %multiply.1319 = f32[1,8,14,14,48]{4,3,2,1,0} multiply(f32[1,8,14,14,48]{4,3,2,1,0} %broadcast.1318, f32[1,8,14,14,48]{4,3,2,1,0} %convolution.1268), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %arg5.6 = f32[1,1,1,1,48]{4,3,2,1,0} parameter(5), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.1320 = f32[1,1,1,1,48]{4,3,2,1,0} multiply(f32[1,1,1,1,48]{4,3,2,1,0} %rsqrt.1316, f32[1,1,1,1,48]{4,3,2,1,0} %reshape.1288), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.1321 = f32[1,1,1,1,48]{4,3,2,1,0} subtract(f32[1,1,1,1,48]{4,3,2,1,0} %arg5.6, f32[1,1,1,1,48]{4,3,2,1,0} %multiply.1320), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/sub"}
  %reshape.1322 = f32[1,48]{1,0} reshape(f32[1,1,1,1,48]{4,3,2,1,0} %subtract.1321), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.1323 = f32[1,8,14,14,48]{4,3,2,1,0} broadcast(f32[1,48]{1,0} %reshape.1322), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %add.1324 = f32[1,8,14,14,48]{4,3,2,1,0} add(f32[1,8,14,14,48]{4,3,2,1,0} %multiply.1319, f32[1,8,14,14,48]{4,3,2,1,0} %broadcast.1323), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %constant.1376 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %broadcast.1377 = f32[1,1,1,1,64]{4,3,2,1,0} broadcast(f32[] %constant.1376), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %constant.1325 = f32[] constant(-inf), metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/MaxPool3d_0a_3x3"}
  %reduce-window.1330 = f32[1,8,14,14,480]{4,3,2,1,0} reduce-window(f32[1,8,14,14,480]{4,3,2,1,0} %maximum.1033, f32[] %constant.1325), window={size=1x3x3x3x1 pad=0_0x1_1x1_1x1_1x0_0}, to_apply=%max_F32.1326, metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/MaxPool3d_0a_3x3"}
  %arg11.12 = f32[1,1,1,480,64]{4,3,2,1,0} parameter(11), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.1331 = f32[1,8,14,14,64]{4,3,2,1,0} convolution(f32[1,8,14,14,480]{4,3,2,1,0} %reduce-window.1330, f32[1,1,1,480,64]{4,3,2,1,0} %arg11.12), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/conv_3d/convolution"}
  %convert.1332 = f32[1,8,14,14,64]{4,3,2,1,0} convert(f32[1,8,14,14,64]{4,3,2,1,0} %convolution.1331), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %constant.1333 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %convert.1334 = f32[] convert(f32[] %constant.1333), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %reduce.1339 = f32[64]{0} reduce(f32[1,8,14,14,64]{4,3,2,1,0} %convert.1332, f32[] %convert.1334), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4b_Branch_3_Conv3d_0b_1x1_batch_norm_normalize_moments_mean-reduction.1335, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1340 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.1332), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1341 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.1332), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %multiply.1342 = s32[] multiply(s32[] %get-dimension-size.1340, s32[] %get-dimension-size.1341), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1343 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.1332), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %multiply.1344 = s32[] multiply(s32[] %multiply.1342, s32[] %get-dimension-size.1343), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1345 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.1332), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %multiply.1346 = s32[] multiply(s32[] %multiply.1344, s32[] %get-dimension-size.1345), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %convert.1347 = f32[] convert(s32[] %multiply.1346), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.1348 = f32[64]{0} broadcast(f32[] %convert.1347), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %divide.1349 = f32[64]{0} divide(f32[64]{0} %reduce.1339, f32[64]{0} %broadcast.1348), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %convert.1350 = f32[64]{0} convert(f32[64]{0} %divide.1349), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %reshape.1351 = f32[1,1,1,1,64]{4,3,2,1,0} reshape(f32[64]{0} %convert.1350), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %reshape.1352 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %reshape.1351), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.1353 = f32[1,8,14,14,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.1352), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.1354 = f32[1,8,14,14,64]{4,3,2,1,0} subtract(f32[1,8,14,14,64]{4,3,2,1,0} %broadcast.1353, f32[1,8,14,14,64]{4,3,2,1,0} %convolution.1331), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.1355 = f32[1,8,14,14,64]{4,3,2,1,0} multiply(f32[1,8,14,14,64]{4,3,2,1,0} %subtract.1354, f32[1,8,14,14,64]{4,3,2,1,0} %subtract.1354), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.1356 = f32[1,8,14,14,64]{4,3,2,1,0} convert(f32[1,8,14,14,64]{4,3,2,1,0} %multiply.1355), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %constant.1357 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %convert.1358 = f32[] convert(f32[] %constant.1357), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %reduce.1363 = f32[64]{0} reduce(f32[1,8,14,14,64]{4,3,2,1,0} %convert.1356, f32[] %convert.1358), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4b_Branch_3_Conv3d_0b_1x1_batch_norm_normalize_moments_variance-reduction.1359, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1364 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.1356), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1365 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.1356), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %multiply.1366 = s32[] multiply(s32[] %get-dimension-size.1364, s32[] %get-dimension-size.1365), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1367 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.1356), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %multiply.1368 = s32[] multiply(s32[] %multiply.1366, s32[] %get-dimension-size.1367), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1369 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.1356), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %multiply.1370 = s32[] multiply(s32[] %multiply.1368, s32[] %get-dimension-size.1369), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %convert.1371 = f32[] convert(s32[] %multiply.1370), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.1372 = f32[64]{0} broadcast(f32[] %convert.1371), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %divide.1373 = f32[64]{0} divide(f32[64]{0} %reduce.1363, f32[64]{0} %broadcast.1372), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %convert.1374 = f32[64]{0} convert(f32[64]{0} %divide.1373), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %reshape.1375 = f32[1,1,1,1,64]{4,3,2,1,0} reshape(f32[64]{0} %convert.1374), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %add.1378 = f32[1,1,1,1,64]{4,3,2,1,0} add(f32[1,1,1,1,64]{4,3,2,1,0} %broadcast.1377, f32[1,1,1,1,64]{4,3,2,1,0} %reshape.1375), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %rsqrt.1379 = f32[1,1,1,1,64]{4,3,2,1,0} rsqrt(f32[1,1,1,1,64]{4,3,2,1,0} %add.1378), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.1380 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.1379), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %broadcast.1381 = f32[1,8,14,14,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.1380), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %multiply.1382 = f32[1,8,14,14,64]{4,3,2,1,0} multiply(f32[1,8,14,14,64]{4,3,2,1,0} %broadcast.1381, f32[1,8,14,14,64]{4,3,2,1,0} %convolution.1331), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %arg14.15 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(14), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.1383 = f32[1,1,1,1,64]{4,3,2,1,0} multiply(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.1379, f32[1,1,1,1,64]{4,3,2,1,0} %reshape.1351), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.1384 = f32[1,1,1,1,64]{4,3,2,1,0} subtract(f32[1,1,1,1,64]{4,3,2,1,0} %arg14.15, f32[1,1,1,1,64]{4,3,2,1,0} %multiply.1383), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/sub"}
  %reshape.1385 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %subtract.1384), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.1386 = f32[1,8,14,14,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.1385), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %add.1387 = f32[1,8,14,14,64]{4,3,2,1,0} add(f32[1,8,14,14,64]{4,3,2,1,0} %multiply.1382, f32[1,8,14,14,64]{4,3,2,1,0} %broadcast.1386), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %concatenate.1388 = f32[1,8,14,14,512]{4,3,2,1,0} concatenate(f32[1,8,14,14,192]{4,3,2,1,0} %add.1090, f32[1,8,14,14,208]{4,3,2,1,0} %add.1207, f32[1,8,14,14,48]{4,3,2,1,0} %add.1324, f32[1,8,14,14,64]{4,3,2,1,0} %add.1387), dimensions={4}, metadata={op_type="ConcatV2" op_name="RGB/inception_i3d/Mixed_4b/concat"}
  %maximum.1391 = f32[1,8,14,14,512]{4,3,2,1,0} maximum(f32[1,8,14,14,512]{4,3,2,1,0} %broadcast.1390, f32[1,8,14,14,512]{4,3,2,1,0} %concatenate.1388), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/Relu"}
  %arg21.22 = f32[1,1,1,512,160]{4,3,2,1,0} parameter(21), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.1392 = f32[1,8,14,14,160]{4,3,2,1,0} convolution(f32[1,8,14,14,512]{4,3,2,1,0} %maximum.1391, f32[1,1,1,512,160]{4,3,2,1,0} %arg21.22), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/conv_3d/convolution"}
  %convert.1393 = f32[1,8,14,14,160]{4,3,2,1,0} convert(f32[1,8,14,14,160]{4,3,2,1,0} %convolution.1392), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %constant.1394 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.1395 = f32[] convert(f32[] %constant.1394), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reduce.1400 = f32[160]{0} reduce(f32[1,8,14,14,160]{4,3,2,1,0} %convert.1393, f32[] %convert.1395), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4c_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.1396, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1401 = s32[] get-dimension-size(f32[1,8,14,14,160]{4,3,2,1,0} %convert.1393), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1402 = s32[] get-dimension-size(f32[1,8,14,14,160]{4,3,2,1,0} %convert.1393), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.1403 = s32[] multiply(s32[] %get-dimension-size.1401, s32[] %get-dimension-size.1402), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1404 = s32[] get-dimension-size(f32[1,8,14,14,160]{4,3,2,1,0} %convert.1393), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.1405 = s32[] multiply(s32[] %multiply.1403, s32[] %get-dimension-size.1404), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1406 = s32[] get-dimension-size(f32[1,8,14,14,160]{4,3,2,1,0} %convert.1393), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.1407 = s32[] multiply(s32[] %multiply.1405, s32[] %get-dimension-size.1406), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.1408 = f32[] convert(s32[] %multiply.1407), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.1409 = f32[160]{0} broadcast(f32[] %convert.1408), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %divide.1410 = f32[160]{0} divide(f32[160]{0} %reduce.1400, f32[160]{0} %broadcast.1409), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.1411 = f32[160]{0} convert(f32[160]{0} %divide.1410), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.1412 = f32[1,1,1,1,160]{4,3,2,1,0} reshape(f32[160]{0} %convert.1411), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.1413 = f32[1,160]{1,0} reshape(f32[1,1,1,1,160]{4,3,2,1,0} %reshape.1412), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.1414 = f32[1,8,14,14,160]{4,3,2,1,0} broadcast(f32[1,160]{1,0} %reshape.1413), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.1415 = f32[1,8,14,14,160]{4,3,2,1,0} subtract(f32[1,8,14,14,160]{4,3,2,1,0} %broadcast.1414, f32[1,8,14,14,160]{4,3,2,1,0} %convolution.1392), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.1416 = f32[1,8,14,14,160]{4,3,2,1,0} multiply(f32[1,8,14,14,160]{4,3,2,1,0} %subtract.1415, f32[1,8,14,14,160]{4,3,2,1,0} %subtract.1415), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.1417 = f32[1,8,14,14,160]{4,3,2,1,0} convert(f32[1,8,14,14,160]{4,3,2,1,0} %multiply.1416), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %constant.1418 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.1419 = f32[] convert(f32[] %constant.1418), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reduce.1424 = f32[160]{0} reduce(f32[1,8,14,14,160]{4,3,2,1,0} %convert.1417, f32[] %convert.1419), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4c_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.1420, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1425 = s32[] get-dimension-size(f32[1,8,14,14,160]{4,3,2,1,0} %convert.1417), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1426 = s32[] get-dimension-size(f32[1,8,14,14,160]{4,3,2,1,0} %convert.1417), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.1427 = s32[] multiply(s32[] %get-dimension-size.1425, s32[] %get-dimension-size.1426), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1428 = s32[] get-dimension-size(f32[1,8,14,14,160]{4,3,2,1,0} %convert.1417), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.1429 = s32[] multiply(s32[] %multiply.1427, s32[] %get-dimension-size.1428), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1430 = s32[] get-dimension-size(f32[1,8,14,14,160]{4,3,2,1,0} %convert.1417), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.1431 = s32[] multiply(s32[] %multiply.1429, s32[] %get-dimension-size.1430), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.1432 = f32[] convert(s32[] %multiply.1431), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.1433 = f32[160]{0} broadcast(f32[] %convert.1432), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %divide.1434 = f32[160]{0} divide(f32[160]{0} %reduce.1424, f32[160]{0} %broadcast.1433), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.1435 = f32[160]{0} convert(f32[160]{0} %divide.1434), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reshape.1436 = f32[1,1,1,1,160]{4,3,2,1,0} reshape(f32[160]{0} %convert.1435), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %add.1439 = f32[1,1,1,1,160]{4,3,2,1,0} add(f32[1,1,1,1,160]{4,3,2,1,0} %broadcast.1438, f32[1,1,1,1,160]{4,3,2,1,0} %reshape.1436), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.1440 = f32[1,1,1,1,160]{4,3,2,1,0} rsqrt(f32[1,1,1,1,160]{4,3,2,1,0} %add.1439), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.1441 = f32[1,160]{1,0} reshape(f32[1,1,1,1,160]{4,3,2,1,0} %rsqrt.1440), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.1442 = f32[1,8,14,14,160]{4,3,2,1,0} broadcast(f32[1,160]{1,0} %reshape.1441), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %multiply.1443 = f32[1,8,14,14,160]{4,3,2,1,0} multiply(f32[1,8,14,14,160]{4,3,2,1,0} %broadcast.1442, f32[1,8,14,14,160]{4,3,2,1,0} %convolution.1392), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg23.24 = f32[1,1,1,1,160]{4,3,2,1,0} parameter(23), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.1444 = f32[1,1,1,1,160]{4,3,2,1,0} multiply(f32[1,1,1,1,160]{4,3,2,1,0} %rsqrt.1440, f32[1,1,1,1,160]{4,3,2,1,0} %reshape.1412), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.1445 = f32[1,1,1,1,160]{4,3,2,1,0} subtract(f32[1,1,1,1,160]{4,3,2,1,0} %arg23.24, f32[1,1,1,1,160]{4,3,2,1,0} %multiply.1444), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.1446 = f32[1,160]{1,0} reshape(f32[1,1,1,1,160]{4,3,2,1,0} %subtract.1445), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.1447 = f32[1,8,14,14,160]{4,3,2,1,0} broadcast(f32[1,160]{1,0} %reshape.1446), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.1448 = f32[1,8,14,14,160]{4,3,2,1,0} add(f32[1,8,14,14,160]{4,3,2,1,0} %multiply.1443, f32[1,8,14,14,160]{4,3,2,1,0} %broadcast.1447), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %constant.1554 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %broadcast.1555 = f32[1,1,1,1,224]{4,3,2,1,0} broadcast(f32[] %constant.1554), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %constant.1506 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/Relu"}
  %broadcast.1507 = f32[1,8,14,14,112]{4,3,2,1,0} broadcast(f32[] %constant.1506), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/Relu"}
  %constant.1494 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.1495 = f32[1,1,1,1,112]{4,3,2,1,0} broadcast(f32[] %constant.1494), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %arg29.30 = f32[1,1,1,512,112]{4,3,2,1,0} parameter(29), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.1449 = f32[1,8,14,14,112]{4,3,2,1,0} convolution(f32[1,8,14,14,512]{4,3,2,1,0} %maximum.1391, f32[1,1,1,512,112]{4,3,2,1,0} %arg29.30), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/conv_3d/convolution"}
  %convert.1450 = f32[1,8,14,14,112]{4,3,2,1,0} convert(f32[1,8,14,14,112]{4,3,2,1,0} %convolution.1449), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %constant.1451 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.1452 = f32[] convert(f32[] %constant.1451), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reduce.1457 = f32[112]{0} reduce(f32[1,8,14,14,112]{4,3,2,1,0} %convert.1450, f32[] %convert.1452), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4c_Branch_1_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.1453, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1458 = s32[] get-dimension-size(f32[1,8,14,14,112]{4,3,2,1,0} %convert.1450), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1459 = s32[] get-dimension-size(f32[1,8,14,14,112]{4,3,2,1,0} %convert.1450), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.1460 = s32[] multiply(s32[] %get-dimension-size.1458, s32[] %get-dimension-size.1459), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1461 = s32[] get-dimension-size(f32[1,8,14,14,112]{4,3,2,1,0} %convert.1450), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.1462 = s32[] multiply(s32[] %multiply.1460, s32[] %get-dimension-size.1461), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1463 = s32[] get-dimension-size(f32[1,8,14,14,112]{4,3,2,1,0} %convert.1450), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.1464 = s32[] multiply(s32[] %multiply.1462, s32[] %get-dimension-size.1463), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.1465 = f32[] convert(s32[] %multiply.1464), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.1466 = f32[112]{0} broadcast(f32[] %convert.1465), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %divide.1467 = f32[112]{0} divide(f32[112]{0} %reduce.1457, f32[112]{0} %broadcast.1466), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.1468 = f32[112]{0} convert(f32[112]{0} %divide.1467), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.1469 = f32[1,1,1,1,112]{4,3,2,1,0} reshape(f32[112]{0} %convert.1468), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.1470 = f32[1,112]{1,0} reshape(f32[1,1,1,1,112]{4,3,2,1,0} %reshape.1469), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.1471 = f32[1,8,14,14,112]{4,3,2,1,0} broadcast(f32[1,112]{1,0} %reshape.1470), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.1472 = f32[1,8,14,14,112]{4,3,2,1,0} subtract(f32[1,8,14,14,112]{4,3,2,1,0} %broadcast.1471, f32[1,8,14,14,112]{4,3,2,1,0} %convolution.1449), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.1473 = f32[1,8,14,14,112]{4,3,2,1,0} multiply(f32[1,8,14,14,112]{4,3,2,1,0} %subtract.1472, f32[1,8,14,14,112]{4,3,2,1,0} %subtract.1472), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.1474 = f32[1,8,14,14,112]{4,3,2,1,0} convert(f32[1,8,14,14,112]{4,3,2,1,0} %multiply.1473), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %constant.1475 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.1476 = f32[] convert(f32[] %constant.1475), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reduce.1481 = f32[112]{0} reduce(f32[1,8,14,14,112]{4,3,2,1,0} %convert.1474, f32[] %convert.1476), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4c_Branch_1_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.1477, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1482 = s32[] get-dimension-size(f32[1,8,14,14,112]{4,3,2,1,0} %convert.1474), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1483 = s32[] get-dimension-size(f32[1,8,14,14,112]{4,3,2,1,0} %convert.1474), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.1484 = s32[] multiply(s32[] %get-dimension-size.1482, s32[] %get-dimension-size.1483), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1485 = s32[] get-dimension-size(f32[1,8,14,14,112]{4,3,2,1,0} %convert.1474), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.1486 = s32[] multiply(s32[] %multiply.1484, s32[] %get-dimension-size.1485), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1487 = s32[] get-dimension-size(f32[1,8,14,14,112]{4,3,2,1,0} %convert.1474), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.1488 = s32[] multiply(s32[] %multiply.1486, s32[] %get-dimension-size.1487), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.1489 = f32[] convert(s32[] %multiply.1488), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.1490 = f32[112]{0} broadcast(f32[] %convert.1489), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %divide.1491 = f32[112]{0} divide(f32[112]{0} %reduce.1481, f32[112]{0} %broadcast.1490), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.1492 = f32[112]{0} convert(f32[112]{0} %divide.1491), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reshape.1493 = f32[1,1,1,1,112]{4,3,2,1,0} reshape(f32[112]{0} %convert.1492), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %add.1496 = f32[1,1,1,1,112]{4,3,2,1,0} add(f32[1,1,1,1,112]{4,3,2,1,0} %broadcast.1495, f32[1,1,1,1,112]{4,3,2,1,0} %reshape.1493), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.1497 = f32[1,1,1,1,112]{4,3,2,1,0} rsqrt(f32[1,1,1,1,112]{4,3,2,1,0} %add.1496), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.1498 = f32[1,112]{1,0} reshape(f32[1,1,1,1,112]{4,3,2,1,0} %rsqrt.1497), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.1499 = f32[1,8,14,14,112]{4,3,2,1,0} broadcast(f32[1,112]{1,0} %reshape.1498), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %multiply.1500 = f32[1,8,14,14,112]{4,3,2,1,0} multiply(f32[1,8,14,14,112]{4,3,2,1,0} %broadcast.1499, f32[1,8,14,14,112]{4,3,2,1,0} %convolution.1449), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg31.32 = f32[1,1,1,1,112]{4,3,2,1,0} parameter(31), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.1501 = f32[1,1,1,1,112]{4,3,2,1,0} multiply(f32[1,1,1,1,112]{4,3,2,1,0} %rsqrt.1497, f32[1,1,1,1,112]{4,3,2,1,0} %reshape.1469), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.1502 = f32[1,1,1,1,112]{4,3,2,1,0} subtract(f32[1,1,1,1,112]{4,3,2,1,0} %arg31.32, f32[1,1,1,1,112]{4,3,2,1,0} %multiply.1501), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.1503 = f32[1,112]{1,0} reshape(f32[1,1,1,1,112]{4,3,2,1,0} %subtract.1502), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.1504 = f32[1,8,14,14,112]{4,3,2,1,0} broadcast(f32[1,112]{1,0} %reshape.1503), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.1505 = f32[1,8,14,14,112]{4,3,2,1,0} add(f32[1,8,14,14,112]{4,3,2,1,0} %multiply.1500, f32[1,8,14,14,112]{4,3,2,1,0} %broadcast.1504), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %maximum.1508 = f32[1,8,14,14,112]{4,3,2,1,0} maximum(f32[1,8,14,14,112]{4,3,2,1,0} %broadcast.1507, f32[1,8,14,14,112]{4,3,2,1,0} %add.1505), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/Relu"}
  %arg38.39 = f32[3,3,3,112,224]{4,3,2,1,0} parameter(38), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.1509 = f32[1,8,14,14,224]{4,3,2,1,0} convolution(f32[1,8,14,14,112]{4,3,2,1,0} %maximum.1508, f32[3,3,3,112,224]{4,3,2,1,0} %arg38.39), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/conv_3d/convolution"}
  %convert.1510 = f32[1,8,14,14,224]{4,3,2,1,0} convert(f32[1,8,14,14,224]{4,3,2,1,0} %convolution.1509), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %constant.1511 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.1512 = f32[] convert(f32[] %constant.1511), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reduce.1517 = f32[224]{0} reduce(f32[1,8,14,14,224]{4,3,2,1,0} %convert.1510, f32[] %convert.1512), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4c_Branch_1_Conv3d_0b_3x3_batch_norm_normalize_moments_mean-reduction.1513, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1518 = s32[] get-dimension-size(f32[1,8,14,14,224]{4,3,2,1,0} %convert.1510), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1519 = s32[] get-dimension-size(f32[1,8,14,14,224]{4,3,2,1,0} %convert.1510), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.1520 = s32[] multiply(s32[] %get-dimension-size.1518, s32[] %get-dimension-size.1519), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1521 = s32[] get-dimension-size(f32[1,8,14,14,224]{4,3,2,1,0} %convert.1510), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.1522 = s32[] multiply(s32[] %multiply.1520, s32[] %get-dimension-size.1521), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1523 = s32[] get-dimension-size(f32[1,8,14,14,224]{4,3,2,1,0} %convert.1510), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.1524 = s32[] multiply(s32[] %multiply.1522, s32[] %get-dimension-size.1523), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.1525 = f32[] convert(s32[] %multiply.1524), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %broadcast.1526 = f32[224]{0} broadcast(f32[] %convert.1525), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %divide.1527 = f32[224]{0} divide(f32[224]{0} %reduce.1517, f32[224]{0} %broadcast.1526), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.1528 = f32[224]{0} convert(f32[224]{0} %divide.1527), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reshape.1529 = f32[1,1,1,1,224]{4,3,2,1,0} reshape(f32[224]{0} %convert.1528), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reshape.1530 = f32[1,224]{1,0} reshape(f32[1,1,1,1,224]{4,3,2,1,0} %reshape.1529), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.1531 = f32[1,8,14,14,224]{4,3,2,1,0} broadcast(f32[1,224]{1,0} %reshape.1530), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.1532 = f32[1,8,14,14,224]{4,3,2,1,0} subtract(f32[1,8,14,14,224]{4,3,2,1,0} %broadcast.1531, f32[1,8,14,14,224]{4,3,2,1,0} %convolution.1509), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.1533 = f32[1,8,14,14,224]{4,3,2,1,0} multiply(f32[1,8,14,14,224]{4,3,2,1,0} %subtract.1532, f32[1,8,14,14,224]{4,3,2,1,0} %subtract.1532), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %convert.1534 = f32[1,8,14,14,224]{4,3,2,1,0} convert(f32[1,8,14,14,224]{4,3,2,1,0} %multiply.1533), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %constant.1535 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.1536 = f32[] convert(f32[] %constant.1535), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %reduce.1541 = f32[224]{0} reduce(f32[1,8,14,14,224]{4,3,2,1,0} %convert.1534, f32[] %convert.1536), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4c_Branch_1_Conv3d_0b_3x3_batch_norm_normalize_moments_variance-reduction.1537, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1542 = s32[] get-dimension-size(f32[1,8,14,14,224]{4,3,2,1,0} %convert.1534), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1543 = s32[] get-dimension-size(f32[1,8,14,14,224]{4,3,2,1,0} %convert.1534), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.1544 = s32[] multiply(s32[] %get-dimension-size.1542, s32[] %get-dimension-size.1543), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1545 = s32[] get-dimension-size(f32[1,8,14,14,224]{4,3,2,1,0} %convert.1534), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.1546 = s32[] multiply(s32[] %multiply.1544, s32[] %get-dimension-size.1545), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1547 = s32[] get-dimension-size(f32[1,8,14,14,224]{4,3,2,1,0} %convert.1534), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.1548 = s32[] multiply(s32[] %multiply.1546, s32[] %get-dimension-size.1547), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.1549 = f32[] convert(s32[] %multiply.1548), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %broadcast.1550 = f32[224]{0} broadcast(f32[] %convert.1549), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %divide.1551 = f32[224]{0} divide(f32[224]{0} %reduce.1541, f32[224]{0} %broadcast.1550), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.1552 = f32[224]{0} convert(f32[224]{0} %divide.1551), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %reshape.1553 = f32[1,1,1,1,224]{4,3,2,1,0} reshape(f32[224]{0} %convert.1552), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %add.1556 = f32[1,1,1,1,224]{4,3,2,1,0} add(f32[1,1,1,1,224]{4,3,2,1,0} %broadcast.1555, f32[1,1,1,1,224]{4,3,2,1,0} %reshape.1553), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %rsqrt.1557 = f32[1,1,1,1,224]{4,3,2,1,0} rsqrt(f32[1,1,1,1,224]{4,3,2,1,0} %add.1556), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.1558 = f32[1,224]{1,0} reshape(f32[1,1,1,1,224]{4,3,2,1,0} %rsqrt.1557), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %broadcast.1559 = f32[1,8,14,14,224]{4,3,2,1,0} broadcast(f32[1,224]{1,0} %reshape.1558), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %multiply.1560 = f32[1,8,14,14,224]{4,3,2,1,0} multiply(f32[1,8,14,14,224]{4,3,2,1,0} %broadcast.1559, f32[1,8,14,14,224]{4,3,2,1,0} %convolution.1509), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %arg42.43 = f32[1,1,1,1,224]{4,3,2,1,0} parameter(42), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.1561 = f32[1,1,1,1,224]{4,3,2,1,0} multiply(f32[1,1,1,1,224]{4,3,2,1,0} %rsqrt.1557, f32[1,1,1,1,224]{4,3,2,1,0} %reshape.1529), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.1562 = f32[1,1,1,1,224]{4,3,2,1,0} subtract(f32[1,1,1,1,224]{4,3,2,1,0} %arg42.43, f32[1,1,1,1,224]{4,3,2,1,0} %multiply.1561), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/sub"}
  %reshape.1563 = f32[1,224]{1,0} reshape(f32[1,1,1,1,224]{4,3,2,1,0} %subtract.1562), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.1564 = f32[1,8,14,14,224]{4,3,2,1,0} broadcast(f32[1,224]{1,0} %reshape.1563), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %add.1565 = f32[1,8,14,14,224]{4,3,2,1,0} add(f32[1,8,14,14,224]{4,3,2,1,0} %multiply.1560, f32[1,8,14,14,224]{4,3,2,1,0} %broadcast.1564), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %constant.1671 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %broadcast.1672 = f32[1,1,1,1,64]{4,3,2,1,0} broadcast(f32[] %constant.1671), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %constant.1623 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/Relu"}
  %broadcast.1624 = f32[1,8,14,14,24]{4,3,2,1,0} broadcast(f32[] %constant.1623), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/Relu"}
  %constant.1611 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.1612 = f32[1,1,1,1,24]{4,3,2,1,0} broadcast(f32[] %constant.1611), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %arg48.49 = f32[1,1,1,512,24]{4,3,2,1,0} parameter(48), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.1566 = f32[1,8,14,14,24]{4,3,2,1,0} convolution(f32[1,8,14,14,512]{4,3,2,1,0} %maximum.1391, f32[1,1,1,512,24]{4,3,2,1,0} %arg48.49), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/conv_3d/convolution"}
  %convert.1567 = f32[1,8,14,14,24]{4,3,2,1,0} convert(f32[1,8,14,14,24]{4,3,2,1,0} %convolution.1566), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %constant.1568 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.1569 = f32[] convert(f32[] %constant.1568), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reduce.1574 = f32[24]{0} reduce(f32[1,8,14,14,24]{4,3,2,1,0} %convert.1567, f32[] %convert.1569), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4c_Branch_2_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.1570, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1575 = s32[] get-dimension-size(f32[1,8,14,14,24]{4,3,2,1,0} %convert.1567), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1576 = s32[] get-dimension-size(f32[1,8,14,14,24]{4,3,2,1,0} %convert.1567), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.1577 = s32[] multiply(s32[] %get-dimension-size.1575, s32[] %get-dimension-size.1576), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1578 = s32[] get-dimension-size(f32[1,8,14,14,24]{4,3,2,1,0} %convert.1567), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.1579 = s32[] multiply(s32[] %multiply.1577, s32[] %get-dimension-size.1578), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1580 = s32[] get-dimension-size(f32[1,8,14,14,24]{4,3,2,1,0} %convert.1567), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.1581 = s32[] multiply(s32[] %multiply.1579, s32[] %get-dimension-size.1580), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.1582 = f32[] convert(s32[] %multiply.1581), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.1583 = f32[24]{0} broadcast(f32[] %convert.1582), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %divide.1584 = f32[24]{0} divide(f32[24]{0} %reduce.1574, f32[24]{0} %broadcast.1583), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.1585 = f32[24]{0} convert(f32[24]{0} %divide.1584), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.1586 = f32[1,1,1,1,24]{4,3,2,1,0} reshape(f32[24]{0} %convert.1585), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.1587 = f32[1,24]{1,0} reshape(f32[1,1,1,1,24]{4,3,2,1,0} %reshape.1586), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.1588 = f32[1,8,14,14,24]{4,3,2,1,0} broadcast(f32[1,24]{1,0} %reshape.1587), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.1589 = f32[1,8,14,14,24]{4,3,2,1,0} subtract(f32[1,8,14,14,24]{4,3,2,1,0} %broadcast.1588, f32[1,8,14,14,24]{4,3,2,1,0} %convolution.1566), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.1590 = f32[1,8,14,14,24]{4,3,2,1,0} multiply(f32[1,8,14,14,24]{4,3,2,1,0} %subtract.1589, f32[1,8,14,14,24]{4,3,2,1,0} %subtract.1589), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.1591 = f32[1,8,14,14,24]{4,3,2,1,0} convert(f32[1,8,14,14,24]{4,3,2,1,0} %multiply.1590), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %constant.1592 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.1593 = f32[] convert(f32[] %constant.1592), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reduce.1598 = f32[24]{0} reduce(f32[1,8,14,14,24]{4,3,2,1,0} %convert.1591, f32[] %convert.1593), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4c_Branch_2_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.1594, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1599 = s32[] get-dimension-size(f32[1,8,14,14,24]{4,3,2,1,0} %convert.1591), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1600 = s32[] get-dimension-size(f32[1,8,14,14,24]{4,3,2,1,0} %convert.1591), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.1601 = s32[] multiply(s32[] %get-dimension-size.1599, s32[] %get-dimension-size.1600), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1602 = s32[] get-dimension-size(f32[1,8,14,14,24]{4,3,2,1,0} %convert.1591), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.1603 = s32[] multiply(s32[] %multiply.1601, s32[] %get-dimension-size.1602), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1604 = s32[] get-dimension-size(f32[1,8,14,14,24]{4,3,2,1,0} %convert.1591), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.1605 = s32[] multiply(s32[] %multiply.1603, s32[] %get-dimension-size.1604), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.1606 = f32[] convert(s32[] %multiply.1605), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.1607 = f32[24]{0} broadcast(f32[] %convert.1606), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %divide.1608 = f32[24]{0} divide(f32[24]{0} %reduce.1598, f32[24]{0} %broadcast.1607), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.1609 = f32[24]{0} convert(f32[24]{0} %divide.1608), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reshape.1610 = f32[1,1,1,1,24]{4,3,2,1,0} reshape(f32[24]{0} %convert.1609), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %add.1613 = f32[1,1,1,1,24]{4,3,2,1,0} add(f32[1,1,1,1,24]{4,3,2,1,0} %broadcast.1612, f32[1,1,1,1,24]{4,3,2,1,0} %reshape.1610), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.1614 = f32[1,1,1,1,24]{4,3,2,1,0} rsqrt(f32[1,1,1,1,24]{4,3,2,1,0} %add.1613), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.1615 = f32[1,24]{1,0} reshape(f32[1,1,1,1,24]{4,3,2,1,0} %rsqrt.1614), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.1616 = f32[1,8,14,14,24]{4,3,2,1,0} broadcast(f32[1,24]{1,0} %reshape.1615), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %multiply.1617 = f32[1,8,14,14,24]{4,3,2,1,0} multiply(f32[1,8,14,14,24]{4,3,2,1,0} %broadcast.1616, f32[1,8,14,14,24]{4,3,2,1,0} %convolution.1566), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg51.52 = f32[1,1,1,1,24]{4,3,2,1,0} parameter(51), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.1618 = f32[1,1,1,1,24]{4,3,2,1,0} multiply(f32[1,1,1,1,24]{4,3,2,1,0} %rsqrt.1614, f32[1,1,1,1,24]{4,3,2,1,0} %reshape.1586), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.1619 = f32[1,1,1,1,24]{4,3,2,1,0} subtract(f32[1,1,1,1,24]{4,3,2,1,0} %arg51.52, f32[1,1,1,1,24]{4,3,2,1,0} %multiply.1618), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.1620 = f32[1,24]{1,0} reshape(f32[1,1,1,1,24]{4,3,2,1,0} %subtract.1619), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.1621 = f32[1,8,14,14,24]{4,3,2,1,0} broadcast(f32[1,24]{1,0} %reshape.1620), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.1622 = f32[1,8,14,14,24]{4,3,2,1,0} add(f32[1,8,14,14,24]{4,3,2,1,0} %multiply.1617, f32[1,8,14,14,24]{4,3,2,1,0} %broadcast.1621), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %maximum.1625 = f32[1,8,14,14,24]{4,3,2,1,0} maximum(f32[1,8,14,14,24]{4,3,2,1,0} %broadcast.1624, f32[1,8,14,14,24]{4,3,2,1,0} %add.1622), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/Relu"}
  %arg55.56 = f32[3,3,3,24,64]{4,3,2,1,0} parameter(55), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.1626 = f32[1,8,14,14,64]{4,3,2,1,0} convolution(f32[1,8,14,14,24]{4,3,2,1,0} %maximum.1625, f32[3,3,3,24,64]{4,3,2,1,0} %arg55.56), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/conv_3d/convolution"}
  %convert.1627 = f32[1,8,14,14,64]{4,3,2,1,0} convert(f32[1,8,14,14,64]{4,3,2,1,0} %convolution.1626), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %constant.1628 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.1629 = f32[] convert(f32[] %constant.1628), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reduce.1634 = f32[64]{0} reduce(f32[1,8,14,14,64]{4,3,2,1,0} %convert.1627, f32[] %convert.1629), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4c_Branch_2_Conv3d_0b_3x3_batch_norm_normalize_moments_mean-reduction.1630, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1635 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.1627), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1636 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.1627), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.1637 = s32[] multiply(s32[] %get-dimension-size.1635, s32[] %get-dimension-size.1636), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1638 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.1627), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.1639 = s32[] multiply(s32[] %multiply.1637, s32[] %get-dimension-size.1638), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1640 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.1627), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.1641 = s32[] multiply(s32[] %multiply.1639, s32[] %get-dimension-size.1640), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.1642 = f32[] convert(s32[] %multiply.1641), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %broadcast.1643 = f32[64]{0} broadcast(f32[] %convert.1642), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %divide.1644 = f32[64]{0} divide(f32[64]{0} %reduce.1634, f32[64]{0} %broadcast.1643), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.1645 = f32[64]{0} convert(f32[64]{0} %divide.1644), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reshape.1646 = f32[1,1,1,1,64]{4,3,2,1,0} reshape(f32[64]{0} %convert.1645), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reshape.1647 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %reshape.1646), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.1648 = f32[1,8,14,14,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.1647), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.1649 = f32[1,8,14,14,64]{4,3,2,1,0} subtract(f32[1,8,14,14,64]{4,3,2,1,0} %broadcast.1648, f32[1,8,14,14,64]{4,3,2,1,0} %convolution.1626), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.1650 = f32[1,8,14,14,64]{4,3,2,1,0} multiply(f32[1,8,14,14,64]{4,3,2,1,0} %subtract.1649, f32[1,8,14,14,64]{4,3,2,1,0} %subtract.1649), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %convert.1651 = f32[1,8,14,14,64]{4,3,2,1,0} convert(f32[1,8,14,14,64]{4,3,2,1,0} %multiply.1650), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %constant.1652 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.1653 = f32[] convert(f32[] %constant.1652), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %reduce.1658 = f32[64]{0} reduce(f32[1,8,14,14,64]{4,3,2,1,0} %convert.1651, f32[] %convert.1653), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4c_Branch_2_Conv3d_0b_3x3_batch_norm_normalize_moments_variance-reduction.1654, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1659 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.1651), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1660 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.1651), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.1661 = s32[] multiply(s32[] %get-dimension-size.1659, s32[] %get-dimension-size.1660), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1662 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.1651), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.1663 = s32[] multiply(s32[] %multiply.1661, s32[] %get-dimension-size.1662), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1664 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.1651), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.1665 = s32[] multiply(s32[] %multiply.1663, s32[] %get-dimension-size.1664), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.1666 = f32[] convert(s32[] %multiply.1665), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %broadcast.1667 = f32[64]{0} broadcast(f32[] %convert.1666), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %divide.1668 = f32[64]{0} divide(f32[64]{0} %reduce.1658, f32[64]{0} %broadcast.1667), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.1669 = f32[64]{0} convert(f32[64]{0} %divide.1668), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %reshape.1670 = f32[1,1,1,1,64]{4,3,2,1,0} reshape(f32[64]{0} %convert.1669), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %add.1673 = f32[1,1,1,1,64]{4,3,2,1,0} add(f32[1,1,1,1,64]{4,3,2,1,0} %broadcast.1672, f32[1,1,1,1,64]{4,3,2,1,0} %reshape.1670), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %rsqrt.1674 = f32[1,1,1,1,64]{4,3,2,1,0} rsqrt(f32[1,1,1,1,64]{4,3,2,1,0} %add.1673), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.1675 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.1674), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %broadcast.1676 = f32[1,8,14,14,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.1675), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %multiply.1677 = f32[1,8,14,14,64]{4,3,2,1,0} multiply(f32[1,8,14,14,64]{4,3,2,1,0} %broadcast.1676, f32[1,8,14,14,64]{4,3,2,1,0} %convolution.1626), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %arg59.60 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(59), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.1678 = f32[1,1,1,1,64]{4,3,2,1,0} multiply(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.1674, f32[1,1,1,1,64]{4,3,2,1,0} %reshape.1646), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.1679 = f32[1,1,1,1,64]{4,3,2,1,0} subtract(f32[1,1,1,1,64]{4,3,2,1,0} %arg59.60, f32[1,1,1,1,64]{4,3,2,1,0} %multiply.1678), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/sub"}
  %reshape.1680 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %subtract.1679), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.1681 = f32[1,8,14,14,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.1680), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %add.1682 = f32[1,8,14,14,64]{4,3,2,1,0} add(f32[1,8,14,14,64]{4,3,2,1,0} %multiply.1677, f32[1,8,14,14,64]{4,3,2,1,0} %broadcast.1681), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %constant.1734 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %broadcast.1735 = f32[1,1,1,1,64]{4,3,2,1,0} broadcast(f32[] %constant.1734), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %constant.1683 = f32[] constant(-inf), metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/MaxPool3d_0a_3x3"}
  %reduce-window.1688 = f32[1,8,14,14,512]{4,3,2,1,0} reduce-window(f32[1,8,14,14,512]{4,3,2,1,0} %maximum.1391, f32[] %constant.1683), window={size=1x3x3x3x1 pad=0_0x1_1x1_1x1_1x0_0}, to_apply=%max_F32.1684, metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/MaxPool3d_0a_3x3"}
  %arg65.66 = f32[1,1,1,512,64]{4,3,2,1,0} parameter(65), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.1689 = f32[1,8,14,14,64]{4,3,2,1,0} convolution(f32[1,8,14,14,512]{4,3,2,1,0} %reduce-window.1688, f32[1,1,1,512,64]{4,3,2,1,0} %arg65.66), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/conv_3d/convolution"}
  %convert.1690 = f32[1,8,14,14,64]{4,3,2,1,0} convert(f32[1,8,14,14,64]{4,3,2,1,0} %convolution.1689), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %constant.1691 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %convert.1692 = f32[] convert(f32[] %constant.1691), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %reduce.1697 = f32[64]{0} reduce(f32[1,8,14,14,64]{4,3,2,1,0} %convert.1690, f32[] %convert.1692), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4c_Branch_3_Conv3d_0b_1x1_batch_norm_normalize_moments_mean-reduction.1693, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1698 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.1690), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1699 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.1690), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %multiply.1700 = s32[] multiply(s32[] %get-dimension-size.1698, s32[] %get-dimension-size.1699), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1701 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.1690), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %multiply.1702 = s32[] multiply(s32[] %multiply.1700, s32[] %get-dimension-size.1701), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1703 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.1690), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %multiply.1704 = s32[] multiply(s32[] %multiply.1702, s32[] %get-dimension-size.1703), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %convert.1705 = f32[] convert(s32[] %multiply.1704), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.1706 = f32[64]{0} broadcast(f32[] %convert.1705), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %divide.1707 = f32[64]{0} divide(f32[64]{0} %reduce.1697, f32[64]{0} %broadcast.1706), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %convert.1708 = f32[64]{0} convert(f32[64]{0} %divide.1707), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %reshape.1709 = f32[1,1,1,1,64]{4,3,2,1,0} reshape(f32[64]{0} %convert.1708), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %reshape.1710 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %reshape.1709), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.1711 = f32[1,8,14,14,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.1710), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.1712 = f32[1,8,14,14,64]{4,3,2,1,0} subtract(f32[1,8,14,14,64]{4,3,2,1,0} %broadcast.1711, f32[1,8,14,14,64]{4,3,2,1,0} %convolution.1689), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.1713 = f32[1,8,14,14,64]{4,3,2,1,0} multiply(f32[1,8,14,14,64]{4,3,2,1,0} %subtract.1712, f32[1,8,14,14,64]{4,3,2,1,0} %subtract.1712), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.1714 = f32[1,8,14,14,64]{4,3,2,1,0} convert(f32[1,8,14,14,64]{4,3,2,1,0} %multiply.1713), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %constant.1715 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %convert.1716 = f32[] convert(f32[] %constant.1715), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %reduce.1721 = f32[64]{0} reduce(f32[1,8,14,14,64]{4,3,2,1,0} %convert.1714, f32[] %convert.1716), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4c_Branch_3_Conv3d_0b_1x1_batch_norm_normalize_moments_variance-reduction.1717, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1722 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.1714), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1723 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.1714), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %multiply.1724 = s32[] multiply(s32[] %get-dimension-size.1722, s32[] %get-dimension-size.1723), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1725 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.1714), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %multiply.1726 = s32[] multiply(s32[] %multiply.1724, s32[] %get-dimension-size.1725), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1727 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.1714), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %multiply.1728 = s32[] multiply(s32[] %multiply.1726, s32[] %get-dimension-size.1727), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %convert.1729 = f32[] convert(s32[] %multiply.1728), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.1730 = f32[64]{0} broadcast(f32[] %convert.1729), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %divide.1731 = f32[64]{0} divide(f32[64]{0} %reduce.1721, f32[64]{0} %broadcast.1730), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %convert.1732 = f32[64]{0} convert(f32[64]{0} %divide.1731), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %reshape.1733 = f32[1,1,1,1,64]{4,3,2,1,0} reshape(f32[64]{0} %convert.1732), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %add.1736 = f32[1,1,1,1,64]{4,3,2,1,0} add(f32[1,1,1,1,64]{4,3,2,1,0} %broadcast.1735, f32[1,1,1,1,64]{4,3,2,1,0} %reshape.1733), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %rsqrt.1737 = f32[1,1,1,1,64]{4,3,2,1,0} rsqrt(f32[1,1,1,1,64]{4,3,2,1,0} %add.1736), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.1738 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.1737), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %broadcast.1739 = f32[1,8,14,14,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.1738), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %multiply.1740 = f32[1,8,14,14,64]{4,3,2,1,0} multiply(f32[1,8,14,14,64]{4,3,2,1,0} %broadcast.1739, f32[1,8,14,14,64]{4,3,2,1,0} %convolution.1689), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %arg69.70 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(69), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.1741 = f32[1,1,1,1,64]{4,3,2,1,0} multiply(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.1737, f32[1,1,1,1,64]{4,3,2,1,0} %reshape.1709), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.1742 = f32[1,1,1,1,64]{4,3,2,1,0} subtract(f32[1,1,1,1,64]{4,3,2,1,0} %arg69.70, f32[1,1,1,1,64]{4,3,2,1,0} %multiply.1741), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/sub"}
  %reshape.1743 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %subtract.1742), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.1744 = f32[1,8,14,14,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.1743), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %add.1745 = f32[1,8,14,14,64]{4,3,2,1,0} add(f32[1,8,14,14,64]{4,3,2,1,0} %multiply.1740, f32[1,8,14,14,64]{4,3,2,1,0} %broadcast.1744), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %concatenate.1746 = f32[1,8,14,14,512]{4,3,2,1,0} concatenate(f32[1,8,14,14,160]{4,3,2,1,0} %add.1448, f32[1,8,14,14,224]{4,3,2,1,0} %add.1565, f32[1,8,14,14,64]{4,3,2,1,0} %add.1682, f32[1,8,14,14,64]{4,3,2,1,0} %add.1745), dimensions={4}, metadata={op_type="ConcatV2" op_name="RGB/inception_i3d/Mixed_4c/concat"}
  %maximum.1749 = f32[1,8,14,14,512]{4,3,2,1,0} maximum(f32[1,8,14,14,512]{4,3,2,1,0} %broadcast.1748, f32[1,8,14,14,512]{4,3,2,1,0} %concatenate.1746), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/Relu"}
  %arg74.75 = f32[1,1,1,512,128]{4,3,2,1,0} parameter(74), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.1750 = f32[1,8,14,14,128]{4,3,2,1,0} convolution(f32[1,8,14,14,512]{4,3,2,1,0} %maximum.1749, f32[1,1,1,512,128]{4,3,2,1,0} %arg74.75), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/conv_3d/convolution"}
  %convert.1751 = f32[1,8,14,14,128]{4,3,2,1,0} convert(f32[1,8,14,14,128]{4,3,2,1,0} %convolution.1750), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %constant.1752 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.1753 = f32[] convert(f32[] %constant.1752), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reduce.1758 = f32[128]{0} reduce(f32[1,8,14,14,128]{4,3,2,1,0} %convert.1751, f32[] %convert.1753), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4d_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.1754, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1759 = s32[] get-dimension-size(f32[1,8,14,14,128]{4,3,2,1,0} %convert.1751), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1760 = s32[] get-dimension-size(f32[1,8,14,14,128]{4,3,2,1,0} %convert.1751), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.1761 = s32[] multiply(s32[] %get-dimension-size.1759, s32[] %get-dimension-size.1760), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1762 = s32[] get-dimension-size(f32[1,8,14,14,128]{4,3,2,1,0} %convert.1751), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.1763 = s32[] multiply(s32[] %multiply.1761, s32[] %get-dimension-size.1762), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1764 = s32[] get-dimension-size(f32[1,8,14,14,128]{4,3,2,1,0} %convert.1751), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.1765 = s32[] multiply(s32[] %multiply.1763, s32[] %get-dimension-size.1764), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.1766 = f32[] convert(s32[] %multiply.1765), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.1767 = f32[128]{0} broadcast(f32[] %convert.1766), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %divide.1768 = f32[128]{0} divide(f32[128]{0} %reduce.1758, f32[128]{0} %broadcast.1767), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.1769 = f32[128]{0} convert(f32[128]{0} %divide.1768), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.1770 = f32[1,1,1,1,128]{4,3,2,1,0} reshape(f32[128]{0} %convert.1769), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.1771 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %reshape.1770), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.1772 = f32[1,8,14,14,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.1771), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.1773 = f32[1,8,14,14,128]{4,3,2,1,0} subtract(f32[1,8,14,14,128]{4,3,2,1,0} %broadcast.1772, f32[1,8,14,14,128]{4,3,2,1,0} %convolution.1750), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.1774 = f32[1,8,14,14,128]{4,3,2,1,0} multiply(f32[1,8,14,14,128]{4,3,2,1,0} %subtract.1773, f32[1,8,14,14,128]{4,3,2,1,0} %subtract.1773), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.1775 = f32[1,8,14,14,128]{4,3,2,1,0} convert(f32[1,8,14,14,128]{4,3,2,1,0} %multiply.1774), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %constant.1776 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.1777 = f32[] convert(f32[] %constant.1776), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reduce.1782 = f32[128]{0} reduce(f32[1,8,14,14,128]{4,3,2,1,0} %convert.1775, f32[] %convert.1777), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4d_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.1778, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1783 = s32[] get-dimension-size(f32[1,8,14,14,128]{4,3,2,1,0} %convert.1775), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1784 = s32[] get-dimension-size(f32[1,8,14,14,128]{4,3,2,1,0} %convert.1775), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.1785 = s32[] multiply(s32[] %get-dimension-size.1783, s32[] %get-dimension-size.1784), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1786 = s32[] get-dimension-size(f32[1,8,14,14,128]{4,3,2,1,0} %convert.1775), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.1787 = s32[] multiply(s32[] %multiply.1785, s32[] %get-dimension-size.1786), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1788 = s32[] get-dimension-size(f32[1,8,14,14,128]{4,3,2,1,0} %convert.1775), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.1789 = s32[] multiply(s32[] %multiply.1787, s32[] %get-dimension-size.1788), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.1790 = f32[] convert(s32[] %multiply.1789), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.1791 = f32[128]{0} broadcast(f32[] %convert.1790), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %divide.1792 = f32[128]{0} divide(f32[128]{0} %reduce.1782, f32[128]{0} %broadcast.1791), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.1793 = f32[128]{0} convert(f32[128]{0} %divide.1792), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reshape.1794 = f32[1,1,1,1,128]{4,3,2,1,0} reshape(f32[128]{0} %convert.1793), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %add.1797 = f32[1,1,1,1,128]{4,3,2,1,0} add(f32[1,1,1,1,128]{4,3,2,1,0} %broadcast.1796, f32[1,1,1,1,128]{4,3,2,1,0} %reshape.1794), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.1798 = f32[1,1,1,1,128]{4,3,2,1,0} rsqrt(f32[1,1,1,1,128]{4,3,2,1,0} %add.1797), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.1799 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.1798), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.1800 = f32[1,8,14,14,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.1799), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %multiply.1801 = f32[1,8,14,14,128]{4,3,2,1,0} multiply(f32[1,8,14,14,128]{4,3,2,1,0} %broadcast.1800, f32[1,8,14,14,128]{4,3,2,1,0} %convolution.1750), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg79.80 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(79), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.1802 = f32[1,1,1,1,128]{4,3,2,1,0} multiply(f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.1798, f32[1,1,1,1,128]{4,3,2,1,0} %reshape.1770), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.1803 = f32[1,1,1,1,128]{4,3,2,1,0} subtract(f32[1,1,1,1,128]{4,3,2,1,0} %arg79.80, f32[1,1,1,1,128]{4,3,2,1,0} %multiply.1802), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.1804 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %subtract.1803), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.1805 = f32[1,8,14,14,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.1804), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.1806 = f32[1,8,14,14,128]{4,3,2,1,0} add(f32[1,8,14,14,128]{4,3,2,1,0} %multiply.1801, f32[1,8,14,14,128]{4,3,2,1,0} %broadcast.1805), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %constant.1912 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %broadcast.1913 = f32[1,1,1,1,256]{4,3,2,1,0} broadcast(f32[] %constant.1912), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %constant.1864 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/Relu"}
  %broadcast.1865 = f32[1,8,14,14,128]{4,3,2,1,0} broadcast(f32[] %constant.1864), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/Relu"}
  %constant.1852 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.1853 = f32[1,1,1,1,128]{4,3,2,1,0} broadcast(f32[] %constant.1852), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %arg83.84 = f32[1,1,1,512,128]{4,3,2,1,0} parameter(83), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.1807 = f32[1,8,14,14,128]{4,3,2,1,0} convolution(f32[1,8,14,14,512]{4,3,2,1,0} %maximum.1749, f32[1,1,1,512,128]{4,3,2,1,0} %arg83.84), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/conv_3d/convolution"}
  %convert.1808 = f32[1,8,14,14,128]{4,3,2,1,0} convert(f32[1,8,14,14,128]{4,3,2,1,0} %convolution.1807), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %constant.1809 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.1810 = f32[] convert(f32[] %constant.1809), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reduce.1815 = f32[128]{0} reduce(f32[1,8,14,14,128]{4,3,2,1,0} %convert.1808, f32[] %convert.1810), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4d_Branch_1_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.1811, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1816 = s32[] get-dimension-size(f32[1,8,14,14,128]{4,3,2,1,0} %convert.1808), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1817 = s32[] get-dimension-size(f32[1,8,14,14,128]{4,3,2,1,0} %convert.1808), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.1818 = s32[] multiply(s32[] %get-dimension-size.1816, s32[] %get-dimension-size.1817), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1819 = s32[] get-dimension-size(f32[1,8,14,14,128]{4,3,2,1,0} %convert.1808), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.1820 = s32[] multiply(s32[] %multiply.1818, s32[] %get-dimension-size.1819), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1821 = s32[] get-dimension-size(f32[1,8,14,14,128]{4,3,2,1,0} %convert.1808), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.1822 = s32[] multiply(s32[] %multiply.1820, s32[] %get-dimension-size.1821), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.1823 = f32[] convert(s32[] %multiply.1822), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.1824 = f32[128]{0} broadcast(f32[] %convert.1823), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %divide.1825 = f32[128]{0} divide(f32[128]{0} %reduce.1815, f32[128]{0} %broadcast.1824), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.1826 = f32[128]{0} convert(f32[128]{0} %divide.1825), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.1827 = f32[1,1,1,1,128]{4,3,2,1,0} reshape(f32[128]{0} %convert.1826), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.1828 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %reshape.1827), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.1829 = f32[1,8,14,14,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.1828), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.1830 = f32[1,8,14,14,128]{4,3,2,1,0} subtract(f32[1,8,14,14,128]{4,3,2,1,0} %broadcast.1829, f32[1,8,14,14,128]{4,3,2,1,0} %convolution.1807), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.1831 = f32[1,8,14,14,128]{4,3,2,1,0} multiply(f32[1,8,14,14,128]{4,3,2,1,0} %subtract.1830, f32[1,8,14,14,128]{4,3,2,1,0} %subtract.1830), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.1832 = f32[1,8,14,14,128]{4,3,2,1,0} convert(f32[1,8,14,14,128]{4,3,2,1,0} %multiply.1831), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %constant.1833 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.1834 = f32[] convert(f32[] %constant.1833), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reduce.1839 = f32[128]{0} reduce(f32[1,8,14,14,128]{4,3,2,1,0} %convert.1832, f32[] %convert.1834), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4d_Branch_1_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.1835, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1840 = s32[] get-dimension-size(f32[1,8,14,14,128]{4,3,2,1,0} %convert.1832), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1841 = s32[] get-dimension-size(f32[1,8,14,14,128]{4,3,2,1,0} %convert.1832), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.1842 = s32[] multiply(s32[] %get-dimension-size.1840, s32[] %get-dimension-size.1841), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1843 = s32[] get-dimension-size(f32[1,8,14,14,128]{4,3,2,1,0} %convert.1832), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.1844 = s32[] multiply(s32[] %multiply.1842, s32[] %get-dimension-size.1843), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1845 = s32[] get-dimension-size(f32[1,8,14,14,128]{4,3,2,1,0} %convert.1832), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.1846 = s32[] multiply(s32[] %multiply.1844, s32[] %get-dimension-size.1845), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.1847 = f32[] convert(s32[] %multiply.1846), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.1848 = f32[128]{0} broadcast(f32[] %convert.1847), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %divide.1849 = f32[128]{0} divide(f32[128]{0} %reduce.1839, f32[128]{0} %broadcast.1848), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.1850 = f32[128]{0} convert(f32[128]{0} %divide.1849), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reshape.1851 = f32[1,1,1,1,128]{4,3,2,1,0} reshape(f32[128]{0} %convert.1850), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %add.1854 = f32[1,1,1,1,128]{4,3,2,1,0} add(f32[1,1,1,1,128]{4,3,2,1,0} %broadcast.1853, f32[1,1,1,1,128]{4,3,2,1,0} %reshape.1851), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.1855 = f32[1,1,1,1,128]{4,3,2,1,0} rsqrt(f32[1,1,1,1,128]{4,3,2,1,0} %add.1854), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.1856 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.1855), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.1857 = f32[1,8,14,14,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.1856), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %multiply.1858 = f32[1,8,14,14,128]{4,3,2,1,0} multiply(f32[1,8,14,14,128]{4,3,2,1,0} %broadcast.1857, f32[1,8,14,14,128]{4,3,2,1,0} %convolution.1807), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg86.87 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(86), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.1859 = f32[1,1,1,1,128]{4,3,2,1,0} multiply(f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.1855, f32[1,1,1,1,128]{4,3,2,1,0} %reshape.1827), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.1860 = f32[1,1,1,1,128]{4,3,2,1,0} subtract(f32[1,1,1,1,128]{4,3,2,1,0} %arg86.87, f32[1,1,1,1,128]{4,3,2,1,0} %multiply.1859), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.1861 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %subtract.1860), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.1862 = f32[1,8,14,14,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.1861), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.1863 = f32[1,8,14,14,128]{4,3,2,1,0} add(f32[1,8,14,14,128]{4,3,2,1,0} %multiply.1858, f32[1,8,14,14,128]{4,3,2,1,0} %broadcast.1862), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %maximum.1866 = f32[1,8,14,14,128]{4,3,2,1,0} maximum(f32[1,8,14,14,128]{4,3,2,1,0} %broadcast.1865, f32[1,8,14,14,128]{4,3,2,1,0} %add.1863), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/Relu"}
  %arg91.92 = f32[3,3,3,128,256]{4,3,2,1,0} parameter(91), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.1867 = f32[1,8,14,14,256]{4,3,2,1,0} convolution(f32[1,8,14,14,128]{4,3,2,1,0} %maximum.1866, f32[3,3,3,128,256]{4,3,2,1,0} %arg91.92), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/conv_3d/convolution"}
  %convert.1868 = f32[1,8,14,14,256]{4,3,2,1,0} convert(f32[1,8,14,14,256]{4,3,2,1,0} %convolution.1867), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %constant.1869 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.1870 = f32[] convert(f32[] %constant.1869), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reduce.1875 = f32[256]{0} reduce(f32[1,8,14,14,256]{4,3,2,1,0} %convert.1868, f32[] %convert.1870), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4d_Branch_1_Conv3d_0b_3x3_batch_norm_normalize_moments_mean-reduction.1871, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1876 = s32[] get-dimension-size(f32[1,8,14,14,256]{4,3,2,1,0} %convert.1868), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1877 = s32[] get-dimension-size(f32[1,8,14,14,256]{4,3,2,1,0} %convert.1868), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.1878 = s32[] multiply(s32[] %get-dimension-size.1876, s32[] %get-dimension-size.1877), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1879 = s32[] get-dimension-size(f32[1,8,14,14,256]{4,3,2,1,0} %convert.1868), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.1880 = s32[] multiply(s32[] %multiply.1878, s32[] %get-dimension-size.1879), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1881 = s32[] get-dimension-size(f32[1,8,14,14,256]{4,3,2,1,0} %convert.1868), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.1882 = s32[] multiply(s32[] %multiply.1880, s32[] %get-dimension-size.1881), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.1883 = f32[] convert(s32[] %multiply.1882), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %broadcast.1884 = f32[256]{0} broadcast(f32[] %convert.1883), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %divide.1885 = f32[256]{0} divide(f32[256]{0} %reduce.1875, f32[256]{0} %broadcast.1884), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.1886 = f32[256]{0} convert(f32[256]{0} %divide.1885), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reshape.1887 = f32[1,1,1,1,256]{4,3,2,1,0} reshape(f32[256]{0} %convert.1886), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reshape.1888 = f32[1,256]{1,0} reshape(f32[1,1,1,1,256]{4,3,2,1,0} %reshape.1887), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.1889 = f32[1,8,14,14,256]{4,3,2,1,0} broadcast(f32[1,256]{1,0} %reshape.1888), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.1890 = f32[1,8,14,14,256]{4,3,2,1,0} subtract(f32[1,8,14,14,256]{4,3,2,1,0} %broadcast.1889, f32[1,8,14,14,256]{4,3,2,1,0} %convolution.1867), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.1891 = f32[1,8,14,14,256]{4,3,2,1,0} multiply(f32[1,8,14,14,256]{4,3,2,1,0} %subtract.1890, f32[1,8,14,14,256]{4,3,2,1,0} %subtract.1890), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %convert.1892 = f32[1,8,14,14,256]{4,3,2,1,0} convert(f32[1,8,14,14,256]{4,3,2,1,0} %multiply.1891), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %constant.1893 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.1894 = f32[] convert(f32[] %constant.1893), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %reduce.1899 = f32[256]{0} reduce(f32[1,8,14,14,256]{4,3,2,1,0} %convert.1892, f32[] %convert.1894), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4d_Branch_1_Conv3d_0b_3x3_batch_norm_normalize_moments_variance-reduction.1895, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1900 = s32[] get-dimension-size(f32[1,8,14,14,256]{4,3,2,1,0} %convert.1892), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1901 = s32[] get-dimension-size(f32[1,8,14,14,256]{4,3,2,1,0} %convert.1892), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.1902 = s32[] multiply(s32[] %get-dimension-size.1900, s32[] %get-dimension-size.1901), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1903 = s32[] get-dimension-size(f32[1,8,14,14,256]{4,3,2,1,0} %convert.1892), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.1904 = s32[] multiply(s32[] %multiply.1902, s32[] %get-dimension-size.1903), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1905 = s32[] get-dimension-size(f32[1,8,14,14,256]{4,3,2,1,0} %convert.1892), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.1906 = s32[] multiply(s32[] %multiply.1904, s32[] %get-dimension-size.1905), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.1907 = f32[] convert(s32[] %multiply.1906), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %broadcast.1908 = f32[256]{0} broadcast(f32[] %convert.1907), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %divide.1909 = f32[256]{0} divide(f32[256]{0} %reduce.1899, f32[256]{0} %broadcast.1908), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.1910 = f32[256]{0} convert(f32[256]{0} %divide.1909), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %reshape.1911 = f32[1,1,1,1,256]{4,3,2,1,0} reshape(f32[256]{0} %convert.1910), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %add.1914 = f32[1,1,1,1,256]{4,3,2,1,0} add(f32[1,1,1,1,256]{4,3,2,1,0} %broadcast.1913, f32[1,1,1,1,256]{4,3,2,1,0} %reshape.1911), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %rsqrt.1915 = f32[1,1,1,1,256]{4,3,2,1,0} rsqrt(f32[1,1,1,1,256]{4,3,2,1,0} %add.1914), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.1916 = f32[1,256]{1,0} reshape(f32[1,1,1,1,256]{4,3,2,1,0} %rsqrt.1915), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %broadcast.1917 = f32[1,8,14,14,256]{4,3,2,1,0} broadcast(f32[1,256]{1,0} %reshape.1916), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %multiply.1918 = f32[1,8,14,14,256]{4,3,2,1,0} multiply(f32[1,8,14,14,256]{4,3,2,1,0} %broadcast.1917, f32[1,8,14,14,256]{4,3,2,1,0} %convolution.1867), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %arg96.97 = f32[1,1,1,1,256]{4,3,2,1,0} parameter(96), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.1919 = f32[1,1,1,1,256]{4,3,2,1,0} multiply(f32[1,1,1,1,256]{4,3,2,1,0} %rsqrt.1915, f32[1,1,1,1,256]{4,3,2,1,0} %reshape.1887), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.1920 = f32[1,1,1,1,256]{4,3,2,1,0} subtract(f32[1,1,1,1,256]{4,3,2,1,0} %arg96.97, f32[1,1,1,1,256]{4,3,2,1,0} %multiply.1919), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/sub"}
  %reshape.1921 = f32[1,256]{1,0} reshape(f32[1,1,1,1,256]{4,3,2,1,0} %subtract.1920), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.1922 = f32[1,8,14,14,256]{4,3,2,1,0} broadcast(f32[1,256]{1,0} %reshape.1921), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %add.1923 = f32[1,8,14,14,256]{4,3,2,1,0} add(f32[1,8,14,14,256]{4,3,2,1,0} %multiply.1918, f32[1,8,14,14,256]{4,3,2,1,0} %broadcast.1922), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %constant.2029 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %broadcast.2030 = f32[1,1,1,1,64]{4,3,2,1,0} broadcast(f32[] %constant.2029), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %constant.1981 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/Relu"}
  %broadcast.1982 = f32[1,8,14,14,24]{4,3,2,1,0} broadcast(f32[] %constant.1981), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/Relu"}
  %constant.1969 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.1970 = f32[1,1,1,1,24]{4,3,2,1,0} broadcast(f32[] %constant.1969), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %arg103.104 = f32[1,1,1,512,24]{4,3,2,1,0} parameter(103), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.1924 = f32[1,8,14,14,24]{4,3,2,1,0} convolution(f32[1,8,14,14,512]{4,3,2,1,0} %maximum.1749, f32[1,1,1,512,24]{4,3,2,1,0} %arg103.104), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/conv_3d/convolution"}
  %convert.1925 = f32[1,8,14,14,24]{4,3,2,1,0} convert(f32[1,8,14,14,24]{4,3,2,1,0} %convolution.1924), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %constant.1926 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.1927 = f32[] convert(f32[] %constant.1926), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reduce.1932 = f32[24]{0} reduce(f32[1,8,14,14,24]{4,3,2,1,0} %convert.1925, f32[] %convert.1927), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4d_Branch_2_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.1928, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1933 = s32[] get-dimension-size(f32[1,8,14,14,24]{4,3,2,1,0} %convert.1925), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1934 = s32[] get-dimension-size(f32[1,8,14,14,24]{4,3,2,1,0} %convert.1925), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.1935 = s32[] multiply(s32[] %get-dimension-size.1933, s32[] %get-dimension-size.1934), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1936 = s32[] get-dimension-size(f32[1,8,14,14,24]{4,3,2,1,0} %convert.1925), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.1937 = s32[] multiply(s32[] %multiply.1935, s32[] %get-dimension-size.1936), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1938 = s32[] get-dimension-size(f32[1,8,14,14,24]{4,3,2,1,0} %convert.1925), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.1939 = s32[] multiply(s32[] %multiply.1937, s32[] %get-dimension-size.1938), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.1940 = f32[] convert(s32[] %multiply.1939), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.1941 = f32[24]{0} broadcast(f32[] %convert.1940), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %divide.1942 = f32[24]{0} divide(f32[24]{0} %reduce.1932, f32[24]{0} %broadcast.1941), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.1943 = f32[24]{0} convert(f32[24]{0} %divide.1942), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.1944 = f32[1,1,1,1,24]{4,3,2,1,0} reshape(f32[24]{0} %convert.1943), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.1945 = f32[1,24]{1,0} reshape(f32[1,1,1,1,24]{4,3,2,1,0} %reshape.1944), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.1946 = f32[1,8,14,14,24]{4,3,2,1,0} broadcast(f32[1,24]{1,0} %reshape.1945), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.1947 = f32[1,8,14,14,24]{4,3,2,1,0} subtract(f32[1,8,14,14,24]{4,3,2,1,0} %broadcast.1946, f32[1,8,14,14,24]{4,3,2,1,0} %convolution.1924), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.1948 = f32[1,8,14,14,24]{4,3,2,1,0} multiply(f32[1,8,14,14,24]{4,3,2,1,0} %subtract.1947, f32[1,8,14,14,24]{4,3,2,1,0} %subtract.1947), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.1949 = f32[1,8,14,14,24]{4,3,2,1,0} convert(f32[1,8,14,14,24]{4,3,2,1,0} %multiply.1948), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %constant.1950 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.1951 = f32[] convert(f32[] %constant.1950), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reduce.1956 = f32[24]{0} reduce(f32[1,8,14,14,24]{4,3,2,1,0} %convert.1949, f32[] %convert.1951), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4d_Branch_2_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.1952, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1957 = s32[] get-dimension-size(f32[1,8,14,14,24]{4,3,2,1,0} %convert.1949), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1958 = s32[] get-dimension-size(f32[1,8,14,14,24]{4,3,2,1,0} %convert.1949), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.1959 = s32[] multiply(s32[] %get-dimension-size.1957, s32[] %get-dimension-size.1958), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1960 = s32[] get-dimension-size(f32[1,8,14,14,24]{4,3,2,1,0} %convert.1949), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.1961 = s32[] multiply(s32[] %multiply.1959, s32[] %get-dimension-size.1960), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.1962 = s32[] get-dimension-size(f32[1,8,14,14,24]{4,3,2,1,0} %convert.1949), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.1963 = s32[] multiply(s32[] %multiply.1961, s32[] %get-dimension-size.1962), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.1964 = f32[] convert(s32[] %multiply.1963), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.1965 = f32[24]{0} broadcast(f32[] %convert.1964), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %divide.1966 = f32[24]{0} divide(f32[24]{0} %reduce.1956, f32[24]{0} %broadcast.1965), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.1967 = f32[24]{0} convert(f32[24]{0} %divide.1966), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reshape.1968 = f32[1,1,1,1,24]{4,3,2,1,0} reshape(f32[24]{0} %convert.1967), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %add.1971 = f32[1,1,1,1,24]{4,3,2,1,0} add(f32[1,1,1,1,24]{4,3,2,1,0} %broadcast.1970, f32[1,1,1,1,24]{4,3,2,1,0} %reshape.1968), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.1972 = f32[1,1,1,1,24]{4,3,2,1,0} rsqrt(f32[1,1,1,1,24]{4,3,2,1,0} %add.1971), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.1973 = f32[1,24]{1,0} reshape(f32[1,1,1,1,24]{4,3,2,1,0} %rsqrt.1972), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.1974 = f32[1,8,14,14,24]{4,3,2,1,0} broadcast(f32[1,24]{1,0} %reshape.1973), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %multiply.1975 = f32[1,8,14,14,24]{4,3,2,1,0} multiply(f32[1,8,14,14,24]{4,3,2,1,0} %broadcast.1974, f32[1,8,14,14,24]{4,3,2,1,0} %convolution.1924), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg105.106 = f32[1,1,1,1,24]{4,3,2,1,0} parameter(105), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.1976 = f32[1,1,1,1,24]{4,3,2,1,0} multiply(f32[1,1,1,1,24]{4,3,2,1,0} %rsqrt.1972, f32[1,1,1,1,24]{4,3,2,1,0} %reshape.1944), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.1977 = f32[1,1,1,1,24]{4,3,2,1,0} subtract(f32[1,1,1,1,24]{4,3,2,1,0} %arg105.106, f32[1,1,1,1,24]{4,3,2,1,0} %multiply.1976), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.1978 = f32[1,24]{1,0} reshape(f32[1,1,1,1,24]{4,3,2,1,0} %subtract.1977), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.1979 = f32[1,8,14,14,24]{4,3,2,1,0} broadcast(f32[1,24]{1,0} %reshape.1978), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.1980 = f32[1,8,14,14,24]{4,3,2,1,0} add(f32[1,8,14,14,24]{4,3,2,1,0} %multiply.1975, f32[1,8,14,14,24]{4,3,2,1,0} %broadcast.1979), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %maximum.1983 = f32[1,8,14,14,24]{4,3,2,1,0} maximum(f32[1,8,14,14,24]{4,3,2,1,0} %broadcast.1982, f32[1,8,14,14,24]{4,3,2,1,0} %add.1980), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/Relu"}
  %arg110.111 = f32[3,3,3,24,64]{4,3,2,1,0} parameter(110), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.1984 = f32[1,8,14,14,64]{4,3,2,1,0} convolution(f32[1,8,14,14,24]{4,3,2,1,0} %maximum.1983, f32[3,3,3,24,64]{4,3,2,1,0} %arg110.111), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/conv_3d/convolution"}
  %convert.1985 = f32[1,8,14,14,64]{4,3,2,1,0} convert(f32[1,8,14,14,64]{4,3,2,1,0} %convolution.1984), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %constant.1986 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.1987 = f32[] convert(f32[] %constant.1986), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reduce.1992 = f32[64]{0} reduce(f32[1,8,14,14,64]{4,3,2,1,0} %convert.1985, f32[] %convert.1987), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4d_Branch_2_Conv3d_0b_3x3_batch_norm_normalize_moments_mean-reduction.1988, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1993 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.1985), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1994 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.1985), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.1995 = s32[] multiply(s32[] %get-dimension-size.1993, s32[] %get-dimension-size.1994), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1996 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.1985), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.1997 = s32[] multiply(s32[] %multiply.1995, s32[] %get-dimension-size.1996), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.1998 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.1985), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.1999 = s32[] multiply(s32[] %multiply.1997, s32[] %get-dimension-size.1998), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.2000 = f32[] convert(s32[] %multiply.1999), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %broadcast.2001 = f32[64]{0} broadcast(f32[] %convert.2000), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %divide.2002 = f32[64]{0} divide(f32[64]{0} %reduce.1992, f32[64]{0} %broadcast.2001), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.2003 = f32[64]{0} convert(f32[64]{0} %divide.2002), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reshape.2004 = f32[1,1,1,1,64]{4,3,2,1,0} reshape(f32[64]{0} %convert.2003), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reshape.2005 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %reshape.2004), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.2006 = f32[1,8,14,14,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.2005), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.2007 = f32[1,8,14,14,64]{4,3,2,1,0} subtract(f32[1,8,14,14,64]{4,3,2,1,0} %broadcast.2006, f32[1,8,14,14,64]{4,3,2,1,0} %convolution.1984), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.2008 = f32[1,8,14,14,64]{4,3,2,1,0} multiply(f32[1,8,14,14,64]{4,3,2,1,0} %subtract.2007, f32[1,8,14,14,64]{4,3,2,1,0} %subtract.2007), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %convert.2009 = f32[1,8,14,14,64]{4,3,2,1,0} convert(f32[1,8,14,14,64]{4,3,2,1,0} %multiply.2008), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %constant.2010 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.2011 = f32[] convert(f32[] %constant.2010), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %reduce.2016 = f32[64]{0} reduce(f32[1,8,14,14,64]{4,3,2,1,0} %convert.2009, f32[] %convert.2011), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4d_Branch_2_Conv3d_0b_3x3_batch_norm_normalize_moments_variance-reduction.2012, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2017 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.2009), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2018 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.2009), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.2019 = s32[] multiply(s32[] %get-dimension-size.2017, s32[] %get-dimension-size.2018), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2020 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.2009), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.2021 = s32[] multiply(s32[] %multiply.2019, s32[] %get-dimension-size.2020), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2022 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.2009), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.2023 = s32[] multiply(s32[] %multiply.2021, s32[] %get-dimension-size.2022), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.2024 = f32[] convert(s32[] %multiply.2023), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %broadcast.2025 = f32[64]{0} broadcast(f32[] %convert.2024), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %divide.2026 = f32[64]{0} divide(f32[64]{0} %reduce.2016, f32[64]{0} %broadcast.2025), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.2027 = f32[64]{0} convert(f32[64]{0} %divide.2026), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %reshape.2028 = f32[1,1,1,1,64]{4,3,2,1,0} reshape(f32[64]{0} %convert.2027), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %add.2031 = f32[1,1,1,1,64]{4,3,2,1,0} add(f32[1,1,1,1,64]{4,3,2,1,0} %broadcast.2030, f32[1,1,1,1,64]{4,3,2,1,0} %reshape.2028), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %rsqrt.2032 = f32[1,1,1,1,64]{4,3,2,1,0} rsqrt(f32[1,1,1,1,64]{4,3,2,1,0} %add.2031), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.2033 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.2032), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %broadcast.2034 = f32[1,8,14,14,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.2033), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %multiply.2035 = f32[1,8,14,14,64]{4,3,2,1,0} multiply(f32[1,8,14,14,64]{4,3,2,1,0} %broadcast.2034, f32[1,8,14,14,64]{4,3,2,1,0} %convolution.1984), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %arg114.115 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(114), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.2036 = f32[1,1,1,1,64]{4,3,2,1,0} multiply(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.2032, f32[1,1,1,1,64]{4,3,2,1,0} %reshape.2004), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.2037 = f32[1,1,1,1,64]{4,3,2,1,0} subtract(f32[1,1,1,1,64]{4,3,2,1,0} %arg114.115, f32[1,1,1,1,64]{4,3,2,1,0} %multiply.2036), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/sub"}
  %reshape.2038 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %subtract.2037), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.2039 = f32[1,8,14,14,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.2038), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %add.2040 = f32[1,8,14,14,64]{4,3,2,1,0} add(f32[1,8,14,14,64]{4,3,2,1,0} %multiply.2035, f32[1,8,14,14,64]{4,3,2,1,0} %broadcast.2039), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %constant.2092 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %broadcast.2093 = f32[1,1,1,1,64]{4,3,2,1,0} broadcast(f32[] %constant.2092), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %constant.2041 = f32[] constant(-inf), metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/MaxPool3d_0a_3x3"}
  %reduce-window.2046 = f32[1,8,14,14,512]{4,3,2,1,0} reduce-window(f32[1,8,14,14,512]{4,3,2,1,0} %maximum.1749, f32[] %constant.2041), window={size=1x3x3x3x1 pad=0_0x1_1x1_1x1_1x0_0}, to_apply=%max_F32.2042, metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/MaxPool3d_0a_3x3"}
  %arg8.9 = f32[1,1,1,512,64]{4,3,2,1,0} parameter(8), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.2047 = f32[1,8,14,14,64]{4,3,2,1,0} convolution(f32[1,8,14,14,512]{4,3,2,1,0} %reduce-window.2046, f32[1,1,1,512,64]{4,3,2,1,0} %arg8.9), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/conv_3d/convolution"}
  %convert.2048 = f32[1,8,14,14,64]{4,3,2,1,0} convert(f32[1,8,14,14,64]{4,3,2,1,0} %convolution.2047), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %constant.2049 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %convert.2050 = f32[] convert(f32[] %constant.2049), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %reduce.2055 = f32[64]{0} reduce(f32[1,8,14,14,64]{4,3,2,1,0} %convert.2048, f32[] %convert.2050), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4d_Branch_3_Conv3d_0b_1x1_batch_norm_normalize_moments_mean-reduction.2051, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2056 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.2048), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2057 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.2048), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %multiply.2058 = s32[] multiply(s32[] %get-dimension-size.2056, s32[] %get-dimension-size.2057), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2059 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.2048), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %multiply.2060 = s32[] multiply(s32[] %multiply.2058, s32[] %get-dimension-size.2059), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2061 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.2048), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %multiply.2062 = s32[] multiply(s32[] %multiply.2060, s32[] %get-dimension-size.2061), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %convert.2063 = f32[] convert(s32[] %multiply.2062), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.2064 = f32[64]{0} broadcast(f32[] %convert.2063), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %divide.2065 = f32[64]{0} divide(f32[64]{0} %reduce.2055, f32[64]{0} %broadcast.2064), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %convert.2066 = f32[64]{0} convert(f32[64]{0} %divide.2065), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %reshape.2067 = f32[1,1,1,1,64]{4,3,2,1,0} reshape(f32[64]{0} %convert.2066), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %reshape.2068 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %reshape.2067), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.2069 = f32[1,8,14,14,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.2068), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.2070 = f32[1,8,14,14,64]{4,3,2,1,0} subtract(f32[1,8,14,14,64]{4,3,2,1,0} %broadcast.2069, f32[1,8,14,14,64]{4,3,2,1,0} %convolution.2047), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.2071 = f32[1,8,14,14,64]{4,3,2,1,0} multiply(f32[1,8,14,14,64]{4,3,2,1,0} %subtract.2070, f32[1,8,14,14,64]{4,3,2,1,0} %subtract.2070), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.2072 = f32[1,8,14,14,64]{4,3,2,1,0} convert(f32[1,8,14,14,64]{4,3,2,1,0} %multiply.2071), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %constant.2073 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %convert.2074 = f32[] convert(f32[] %constant.2073), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %reduce.2079 = f32[64]{0} reduce(f32[1,8,14,14,64]{4,3,2,1,0} %convert.2072, f32[] %convert.2074), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4d_Branch_3_Conv3d_0b_1x1_batch_norm_normalize_moments_variance-reduction.2075, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2080 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.2072), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2081 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.2072), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %multiply.2082 = s32[] multiply(s32[] %get-dimension-size.2080, s32[] %get-dimension-size.2081), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2083 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.2072), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %multiply.2084 = s32[] multiply(s32[] %multiply.2082, s32[] %get-dimension-size.2083), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2085 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.2072), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %multiply.2086 = s32[] multiply(s32[] %multiply.2084, s32[] %get-dimension-size.2085), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %convert.2087 = f32[] convert(s32[] %multiply.2086), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.2088 = f32[64]{0} broadcast(f32[] %convert.2087), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %divide.2089 = f32[64]{0} divide(f32[64]{0} %reduce.2079, f32[64]{0} %broadcast.2088), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %convert.2090 = f32[64]{0} convert(f32[64]{0} %divide.2089), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %reshape.2091 = f32[1,1,1,1,64]{4,3,2,1,0} reshape(f32[64]{0} %convert.2090), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %add.2094 = f32[1,1,1,1,64]{4,3,2,1,0} add(f32[1,1,1,1,64]{4,3,2,1,0} %broadcast.2093, f32[1,1,1,1,64]{4,3,2,1,0} %reshape.2091), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %rsqrt.2095 = f32[1,1,1,1,64]{4,3,2,1,0} rsqrt(f32[1,1,1,1,64]{4,3,2,1,0} %add.2094), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.2096 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.2095), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %broadcast.2097 = f32[1,8,14,14,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.2096), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %multiply.2098 = f32[1,8,14,14,64]{4,3,2,1,0} multiply(f32[1,8,14,14,64]{4,3,2,1,0} %broadcast.2097, f32[1,8,14,14,64]{4,3,2,1,0} %convolution.2047), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %arg12.13 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(12), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.2099 = f32[1,1,1,1,64]{4,3,2,1,0} multiply(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.2095, f32[1,1,1,1,64]{4,3,2,1,0} %reshape.2067), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.2100 = f32[1,1,1,1,64]{4,3,2,1,0} subtract(f32[1,1,1,1,64]{4,3,2,1,0} %arg12.13, f32[1,1,1,1,64]{4,3,2,1,0} %multiply.2099), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/sub"}
  %reshape.2101 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %subtract.2100), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.2102 = f32[1,8,14,14,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.2101), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %add.2103 = f32[1,8,14,14,64]{4,3,2,1,0} add(f32[1,8,14,14,64]{4,3,2,1,0} %multiply.2098, f32[1,8,14,14,64]{4,3,2,1,0} %broadcast.2102), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %concatenate.2104 = f32[1,8,14,14,512]{4,3,2,1,0} concatenate(f32[1,8,14,14,128]{4,3,2,1,0} %add.1806, f32[1,8,14,14,256]{4,3,2,1,0} %add.1923, f32[1,8,14,14,64]{4,3,2,1,0} %add.2040, f32[1,8,14,14,64]{4,3,2,1,0} %add.2103), dimensions={4}, metadata={op_type="ConcatV2" op_name="RGB/inception_i3d/Mixed_4d/concat"}
  %maximum.2107 = f32[1,8,14,14,512]{4,3,2,1,0} maximum(f32[1,8,14,14,512]{4,3,2,1,0} %broadcast.2106, f32[1,8,14,14,512]{4,3,2,1,0} %concatenate.2104), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/Relu"}
  %arg20.21 = f32[1,1,1,512,112]{4,3,2,1,0} parameter(20), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.2108 = f32[1,8,14,14,112]{4,3,2,1,0} convolution(f32[1,8,14,14,512]{4,3,2,1,0} %maximum.2107, f32[1,1,1,512,112]{4,3,2,1,0} %arg20.21), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/conv_3d/convolution"}
  %convert.2109 = f32[1,8,14,14,112]{4,3,2,1,0} convert(f32[1,8,14,14,112]{4,3,2,1,0} %convolution.2108), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %constant.2110 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.2111 = f32[] convert(f32[] %constant.2110), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reduce.2116 = f32[112]{0} reduce(f32[1,8,14,14,112]{4,3,2,1,0} %convert.2109, f32[] %convert.2111), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4e_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.2112, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2117 = s32[] get-dimension-size(f32[1,8,14,14,112]{4,3,2,1,0} %convert.2109), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2118 = s32[] get-dimension-size(f32[1,8,14,14,112]{4,3,2,1,0} %convert.2109), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.2119 = s32[] multiply(s32[] %get-dimension-size.2117, s32[] %get-dimension-size.2118), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2120 = s32[] get-dimension-size(f32[1,8,14,14,112]{4,3,2,1,0} %convert.2109), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.2121 = s32[] multiply(s32[] %multiply.2119, s32[] %get-dimension-size.2120), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2122 = s32[] get-dimension-size(f32[1,8,14,14,112]{4,3,2,1,0} %convert.2109), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.2123 = s32[] multiply(s32[] %multiply.2121, s32[] %get-dimension-size.2122), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.2124 = f32[] convert(s32[] %multiply.2123), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.2125 = f32[112]{0} broadcast(f32[] %convert.2124), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %divide.2126 = f32[112]{0} divide(f32[112]{0} %reduce.2116, f32[112]{0} %broadcast.2125), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.2127 = f32[112]{0} convert(f32[112]{0} %divide.2126), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.2128 = f32[1,1,1,1,112]{4,3,2,1,0} reshape(f32[112]{0} %convert.2127), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.2129 = f32[1,112]{1,0} reshape(f32[1,1,1,1,112]{4,3,2,1,0} %reshape.2128), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.2130 = f32[1,8,14,14,112]{4,3,2,1,0} broadcast(f32[1,112]{1,0} %reshape.2129), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.2131 = f32[1,8,14,14,112]{4,3,2,1,0} subtract(f32[1,8,14,14,112]{4,3,2,1,0} %broadcast.2130, f32[1,8,14,14,112]{4,3,2,1,0} %convolution.2108), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.2132 = f32[1,8,14,14,112]{4,3,2,1,0} multiply(f32[1,8,14,14,112]{4,3,2,1,0} %subtract.2131, f32[1,8,14,14,112]{4,3,2,1,0} %subtract.2131), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.2133 = f32[1,8,14,14,112]{4,3,2,1,0} convert(f32[1,8,14,14,112]{4,3,2,1,0} %multiply.2132), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %constant.2134 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.2135 = f32[] convert(f32[] %constant.2134), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reduce.2140 = f32[112]{0} reduce(f32[1,8,14,14,112]{4,3,2,1,0} %convert.2133, f32[] %convert.2135), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4e_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.2136, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2141 = s32[] get-dimension-size(f32[1,8,14,14,112]{4,3,2,1,0} %convert.2133), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2142 = s32[] get-dimension-size(f32[1,8,14,14,112]{4,3,2,1,0} %convert.2133), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.2143 = s32[] multiply(s32[] %get-dimension-size.2141, s32[] %get-dimension-size.2142), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2144 = s32[] get-dimension-size(f32[1,8,14,14,112]{4,3,2,1,0} %convert.2133), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.2145 = s32[] multiply(s32[] %multiply.2143, s32[] %get-dimension-size.2144), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2146 = s32[] get-dimension-size(f32[1,8,14,14,112]{4,3,2,1,0} %convert.2133), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.2147 = s32[] multiply(s32[] %multiply.2145, s32[] %get-dimension-size.2146), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.2148 = f32[] convert(s32[] %multiply.2147), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.2149 = f32[112]{0} broadcast(f32[] %convert.2148), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %divide.2150 = f32[112]{0} divide(f32[112]{0} %reduce.2140, f32[112]{0} %broadcast.2149), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.2151 = f32[112]{0} convert(f32[112]{0} %divide.2150), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reshape.2152 = f32[1,1,1,1,112]{4,3,2,1,0} reshape(f32[112]{0} %convert.2151), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %add.2155 = f32[1,1,1,1,112]{4,3,2,1,0} add(f32[1,1,1,1,112]{4,3,2,1,0} %broadcast.2154, f32[1,1,1,1,112]{4,3,2,1,0} %reshape.2152), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.2156 = f32[1,1,1,1,112]{4,3,2,1,0} rsqrt(f32[1,1,1,1,112]{4,3,2,1,0} %add.2155), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.2157 = f32[1,112]{1,0} reshape(f32[1,1,1,1,112]{4,3,2,1,0} %rsqrt.2156), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.2158 = f32[1,8,14,14,112]{4,3,2,1,0} broadcast(f32[1,112]{1,0} %reshape.2157), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %multiply.2159 = f32[1,8,14,14,112]{4,3,2,1,0} multiply(f32[1,8,14,14,112]{4,3,2,1,0} %broadcast.2158, f32[1,8,14,14,112]{4,3,2,1,0} %convolution.2108), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg25.26 = f32[1,1,1,1,112]{4,3,2,1,0} parameter(25), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.2160 = f32[1,1,1,1,112]{4,3,2,1,0} multiply(f32[1,1,1,1,112]{4,3,2,1,0} %rsqrt.2156, f32[1,1,1,1,112]{4,3,2,1,0} %reshape.2128), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.2161 = f32[1,1,1,1,112]{4,3,2,1,0} subtract(f32[1,1,1,1,112]{4,3,2,1,0} %arg25.26, f32[1,1,1,1,112]{4,3,2,1,0} %multiply.2160), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.2162 = f32[1,112]{1,0} reshape(f32[1,1,1,1,112]{4,3,2,1,0} %subtract.2161), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.2163 = f32[1,8,14,14,112]{4,3,2,1,0} broadcast(f32[1,112]{1,0} %reshape.2162), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.2164 = f32[1,8,14,14,112]{4,3,2,1,0} add(f32[1,8,14,14,112]{4,3,2,1,0} %multiply.2159, f32[1,8,14,14,112]{4,3,2,1,0} %broadcast.2163), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %constant.2270 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %broadcast.2271 = f32[1,1,1,1,288]{4,3,2,1,0} broadcast(f32[] %constant.2270), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %constant.2222 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/Relu"}
  %broadcast.2223 = f32[1,8,14,14,144]{4,3,2,1,0} broadcast(f32[] %constant.2222), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/Relu"}
  %constant.2210 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.2211 = f32[1,1,1,1,144]{4,3,2,1,0} broadcast(f32[] %constant.2210), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %arg34.35 = f32[1,1,1,512,144]{4,3,2,1,0} parameter(34), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.2165 = f32[1,8,14,14,144]{4,3,2,1,0} convolution(f32[1,8,14,14,512]{4,3,2,1,0} %maximum.2107, f32[1,1,1,512,144]{4,3,2,1,0} %arg34.35), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/conv_3d/convolution"}
  %convert.2166 = f32[1,8,14,14,144]{4,3,2,1,0} convert(f32[1,8,14,14,144]{4,3,2,1,0} %convolution.2165), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %constant.2167 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.2168 = f32[] convert(f32[] %constant.2167), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reduce.2173 = f32[144]{0} reduce(f32[1,8,14,14,144]{4,3,2,1,0} %convert.2166, f32[] %convert.2168), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4e_Branch_1_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.2169, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2174 = s32[] get-dimension-size(f32[1,8,14,14,144]{4,3,2,1,0} %convert.2166), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2175 = s32[] get-dimension-size(f32[1,8,14,14,144]{4,3,2,1,0} %convert.2166), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.2176 = s32[] multiply(s32[] %get-dimension-size.2174, s32[] %get-dimension-size.2175), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2177 = s32[] get-dimension-size(f32[1,8,14,14,144]{4,3,2,1,0} %convert.2166), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.2178 = s32[] multiply(s32[] %multiply.2176, s32[] %get-dimension-size.2177), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2179 = s32[] get-dimension-size(f32[1,8,14,14,144]{4,3,2,1,0} %convert.2166), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.2180 = s32[] multiply(s32[] %multiply.2178, s32[] %get-dimension-size.2179), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.2181 = f32[] convert(s32[] %multiply.2180), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.2182 = f32[144]{0} broadcast(f32[] %convert.2181), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %divide.2183 = f32[144]{0} divide(f32[144]{0} %reduce.2173, f32[144]{0} %broadcast.2182), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.2184 = f32[144]{0} convert(f32[144]{0} %divide.2183), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.2185 = f32[1,1,1,1,144]{4,3,2,1,0} reshape(f32[144]{0} %convert.2184), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.2186 = f32[1,144]{1,0} reshape(f32[1,1,1,1,144]{4,3,2,1,0} %reshape.2185), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.2187 = f32[1,8,14,14,144]{4,3,2,1,0} broadcast(f32[1,144]{1,0} %reshape.2186), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.2188 = f32[1,8,14,14,144]{4,3,2,1,0} subtract(f32[1,8,14,14,144]{4,3,2,1,0} %broadcast.2187, f32[1,8,14,14,144]{4,3,2,1,0} %convolution.2165), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.2189 = f32[1,8,14,14,144]{4,3,2,1,0} multiply(f32[1,8,14,14,144]{4,3,2,1,0} %subtract.2188, f32[1,8,14,14,144]{4,3,2,1,0} %subtract.2188), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.2190 = f32[1,8,14,14,144]{4,3,2,1,0} convert(f32[1,8,14,14,144]{4,3,2,1,0} %multiply.2189), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %constant.2191 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.2192 = f32[] convert(f32[] %constant.2191), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reduce.2197 = f32[144]{0} reduce(f32[1,8,14,14,144]{4,3,2,1,0} %convert.2190, f32[] %convert.2192), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4e_Branch_1_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.2193, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2198 = s32[] get-dimension-size(f32[1,8,14,14,144]{4,3,2,1,0} %convert.2190), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2199 = s32[] get-dimension-size(f32[1,8,14,14,144]{4,3,2,1,0} %convert.2190), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.2200 = s32[] multiply(s32[] %get-dimension-size.2198, s32[] %get-dimension-size.2199), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2201 = s32[] get-dimension-size(f32[1,8,14,14,144]{4,3,2,1,0} %convert.2190), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.2202 = s32[] multiply(s32[] %multiply.2200, s32[] %get-dimension-size.2201), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2203 = s32[] get-dimension-size(f32[1,8,14,14,144]{4,3,2,1,0} %convert.2190), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.2204 = s32[] multiply(s32[] %multiply.2202, s32[] %get-dimension-size.2203), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.2205 = f32[] convert(s32[] %multiply.2204), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.2206 = f32[144]{0} broadcast(f32[] %convert.2205), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %divide.2207 = f32[144]{0} divide(f32[144]{0} %reduce.2197, f32[144]{0} %broadcast.2206), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.2208 = f32[144]{0} convert(f32[144]{0} %divide.2207), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reshape.2209 = f32[1,1,1,1,144]{4,3,2,1,0} reshape(f32[144]{0} %convert.2208), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %add.2212 = f32[1,1,1,1,144]{4,3,2,1,0} add(f32[1,1,1,1,144]{4,3,2,1,0} %broadcast.2211, f32[1,1,1,1,144]{4,3,2,1,0} %reshape.2209), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.2213 = f32[1,1,1,1,144]{4,3,2,1,0} rsqrt(f32[1,1,1,1,144]{4,3,2,1,0} %add.2212), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.2214 = f32[1,144]{1,0} reshape(f32[1,1,1,1,144]{4,3,2,1,0} %rsqrt.2213), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.2215 = f32[1,8,14,14,144]{4,3,2,1,0} broadcast(f32[1,144]{1,0} %reshape.2214), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %multiply.2216 = f32[1,8,14,14,144]{4,3,2,1,0} multiply(f32[1,8,14,14,144]{4,3,2,1,0} %broadcast.2215, f32[1,8,14,14,144]{4,3,2,1,0} %convolution.2165), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg39.40 = f32[1,1,1,1,144]{4,3,2,1,0} parameter(39), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.2217 = f32[1,1,1,1,144]{4,3,2,1,0} multiply(f32[1,1,1,1,144]{4,3,2,1,0} %rsqrt.2213, f32[1,1,1,1,144]{4,3,2,1,0} %reshape.2185), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.2218 = f32[1,1,1,1,144]{4,3,2,1,0} subtract(f32[1,1,1,1,144]{4,3,2,1,0} %arg39.40, f32[1,1,1,1,144]{4,3,2,1,0} %multiply.2217), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.2219 = f32[1,144]{1,0} reshape(f32[1,1,1,1,144]{4,3,2,1,0} %subtract.2218), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.2220 = f32[1,8,14,14,144]{4,3,2,1,0} broadcast(f32[1,144]{1,0} %reshape.2219), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.2221 = f32[1,8,14,14,144]{4,3,2,1,0} add(f32[1,8,14,14,144]{4,3,2,1,0} %multiply.2216, f32[1,8,14,14,144]{4,3,2,1,0} %broadcast.2220), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %maximum.2224 = f32[1,8,14,14,144]{4,3,2,1,0} maximum(f32[1,8,14,14,144]{4,3,2,1,0} %broadcast.2223, f32[1,8,14,14,144]{4,3,2,1,0} %add.2221), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/Relu"}
  %arg47.48 = f32[3,3,3,144,288]{4,3,2,1,0} parameter(47), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.2225 = f32[1,8,14,14,288]{4,3,2,1,0} convolution(f32[1,8,14,14,144]{4,3,2,1,0} %maximum.2224, f32[3,3,3,144,288]{4,3,2,1,0} %arg47.48), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/conv_3d/convolution"}
  %convert.2226 = f32[1,8,14,14,288]{4,3,2,1,0} convert(f32[1,8,14,14,288]{4,3,2,1,0} %convolution.2225), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %constant.2227 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.2228 = f32[] convert(f32[] %constant.2227), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reduce.2233 = f32[288]{0} reduce(f32[1,8,14,14,288]{4,3,2,1,0} %convert.2226, f32[] %convert.2228), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4e_Branch_1_Conv3d_0b_3x3_batch_norm_normalize_moments_mean-reduction.2229, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2234 = s32[] get-dimension-size(f32[1,8,14,14,288]{4,3,2,1,0} %convert.2226), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2235 = s32[] get-dimension-size(f32[1,8,14,14,288]{4,3,2,1,0} %convert.2226), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.2236 = s32[] multiply(s32[] %get-dimension-size.2234, s32[] %get-dimension-size.2235), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2237 = s32[] get-dimension-size(f32[1,8,14,14,288]{4,3,2,1,0} %convert.2226), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.2238 = s32[] multiply(s32[] %multiply.2236, s32[] %get-dimension-size.2237), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2239 = s32[] get-dimension-size(f32[1,8,14,14,288]{4,3,2,1,0} %convert.2226), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.2240 = s32[] multiply(s32[] %multiply.2238, s32[] %get-dimension-size.2239), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.2241 = f32[] convert(s32[] %multiply.2240), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %broadcast.2242 = f32[288]{0} broadcast(f32[] %convert.2241), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %divide.2243 = f32[288]{0} divide(f32[288]{0} %reduce.2233, f32[288]{0} %broadcast.2242), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.2244 = f32[288]{0} convert(f32[288]{0} %divide.2243), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reshape.2245 = f32[1,1,1,1,288]{4,3,2,1,0} reshape(f32[288]{0} %convert.2244), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reshape.2246 = f32[1,288]{1,0} reshape(f32[1,1,1,1,288]{4,3,2,1,0} %reshape.2245), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.2247 = f32[1,8,14,14,288]{4,3,2,1,0} broadcast(f32[1,288]{1,0} %reshape.2246), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.2248 = f32[1,8,14,14,288]{4,3,2,1,0} subtract(f32[1,8,14,14,288]{4,3,2,1,0} %broadcast.2247, f32[1,8,14,14,288]{4,3,2,1,0} %convolution.2225), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.2249 = f32[1,8,14,14,288]{4,3,2,1,0} multiply(f32[1,8,14,14,288]{4,3,2,1,0} %subtract.2248, f32[1,8,14,14,288]{4,3,2,1,0} %subtract.2248), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %convert.2250 = f32[1,8,14,14,288]{4,3,2,1,0} convert(f32[1,8,14,14,288]{4,3,2,1,0} %multiply.2249), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %constant.2251 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.2252 = f32[] convert(f32[] %constant.2251), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %reduce.2257 = f32[288]{0} reduce(f32[1,8,14,14,288]{4,3,2,1,0} %convert.2250, f32[] %convert.2252), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4e_Branch_1_Conv3d_0b_3x3_batch_norm_normalize_moments_variance-reduction.2253, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2258 = s32[] get-dimension-size(f32[1,8,14,14,288]{4,3,2,1,0} %convert.2250), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2259 = s32[] get-dimension-size(f32[1,8,14,14,288]{4,3,2,1,0} %convert.2250), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.2260 = s32[] multiply(s32[] %get-dimension-size.2258, s32[] %get-dimension-size.2259), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2261 = s32[] get-dimension-size(f32[1,8,14,14,288]{4,3,2,1,0} %convert.2250), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.2262 = s32[] multiply(s32[] %multiply.2260, s32[] %get-dimension-size.2261), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2263 = s32[] get-dimension-size(f32[1,8,14,14,288]{4,3,2,1,0} %convert.2250), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.2264 = s32[] multiply(s32[] %multiply.2262, s32[] %get-dimension-size.2263), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.2265 = f32[] convert(s32[] %multiply.2264), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %broadcast.2266 = f32[288]{0} broadcast(f32[] %convert.2265), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %divide.2267 = f32[288]{0} divide(f32[288]{0} %reduce.2257, f32[288]{0} %broadcast.2266), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.2268 = f32[288]{0} convert(f32[288]{0} %divide.2267), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %reshape.2269 = f32[1,1,1,1,288]{4,3,2,1,0} reshape(f32[288]{0} %convert.2268), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %add.2272 = f32[1,1,1,1,288]{4,3,2,1,0} add(f32[1,1,1,1,288]{4,3,2,1,0} %broadcast.2271, f32[1,1,1,1,288]{4,3,2,1,0} %reshape.2269), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %rsqrt.2273 = f32[1,1,1,1,288]{4,3,2,1,0} rsqrt(f32[1,1,1,1,288]{4,3,2,1,0} %add.2272), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.2274 = f32[1,288]{1,0} reshape(f32[1,1,1,1,288]{4,3,2,1,0} %rsqrt.2273), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %broadcast.2275 = f32[1,8,14,14,288]{4,3,2,1,0} broadcast(f32[1,288]{1,0} %reshape.2274), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %multiply.2276 = f32[1,8,14,14,288]{4,3,2,1,0} multiply(f32[1,8,14,14,288]{4,3,2,1,0} %broadcast.2275, f32[1,8,14,14,288]{4,3,2,1,0} %convolution.2225), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %arg53.54 = f32[1,1,1,1,288]{4,3,2,1,0} parameter(53), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.2277 = f32[1,1,1,1,288]{4,3,2,1,0} multiply(f32[1,1,1,1,288]{4,3,2,1,0} %rsqrt.2273, f32[1,1,1,1,288]{4,3,2,1,0} %reshape.2245), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.2278 = f32[1,1,1,1,288]{4,3,2,1,0} subtract(f32[1,1,1,1,288]{4,3,2,1,0} %arg53.54, f32[1,1,1,1,288]{4,3,2,1,0} %multiply.2277), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/sub"}
  %reshape.2279 = f32[1,288]{1,0} reshape(f32[1,1,1,1,288]{4,3,2,1,0} %subtract.2278), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.2280 = f32[1,8,14,14,288]{4,3,2,1,0} broadcast(f32[1,288]{1,0} %reshape.2279), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %add.2281 = f32[1,8,14,14,288]{4,3,2,1,0} add(f32[1,8,14,14,288]{4,3,2,1,0} %multiply.2276, f32[1,8,14,14,288]{4,3,2,1,0} %broadcast.2280), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %constant.2387 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %broadcast.2388 = f32[1,1,1,1,64]{4,3,2,1,0} broadcast(f32[] %constant.2387), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %constant.2339 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/Relu"}
  %broadcast.2340 = f32[1,8,14,14,32]{4,3,2,1,0} broadcast(f32[] %constant.2339), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/Relu"}
  %constant.2327 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.2328 = f32[1,1,1,1,32]{4,3,2,1,0} broadcast(f32[] %constant.2327), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %arg60.61 = f32[1,1,1,512,32]{4,3,2,1,0} parameter(60), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.2282 = f32[1,8,14,14,32]{4,3,2,1,0} convolution(f32[1,8,14,14,512]{4,3,2,1,0} %maximum.2107, f32[1,1,1,512,32]{4,3,2,1,0} %arg60.61), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/conv_3d/convolution"}
  %convert.2283 = f32[1,8,14,14,32]{4,3,2,1,0} convert(f32[1,8,14,14,32]{4,3,2,1,0} %convolution.2282), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %constant.2284 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.2285 = f32[] convert(f32[] %constant.2284), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reduce.2290 = f32[32]{0} reduce(f32[1,8,14,14,32]{4,3,2,1,0} %convert.2283, f32[] %convert.2285), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4e_Branch_2_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.2286, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2291 = s32[] get-dimension-size(f32[1,8,14,14,32]{4,3,2,1,0} %convert.2283), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2292 = s32[] get-dimension-size(f32[1,8,14,14,32]{4,3,2,1,0} %convert.2283), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.2293 = s32[] multiply(s32[] %get-dimension-size.2291, s32[] %get-dimension-size.2292), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2294 = s32[] get-dimension-size(f32[1,8,14,14,32]{4,3,2,1,0} %convert.2283), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.2295 = s32[] multiply(s32[] %multiply.2293, s32[] %get-dimension-size.2294), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2296 = s32[] get-dimension-size(f32[1,8,14,14,32]{4,3,2,1,0} %convert.2283), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.2297 = s32[] multiply(s32[] %multiply.2295, s32[] %get-dimension-size.2296), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.2298 = f32[] convert(s32[] %multiply.2297), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.2299 = f32[32]{0} broadcast(f32[] %convert.2298), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %divide.2300 = f32[32]{0} divide(f32[32]{0} %reduce.2290, f32[32]{0} %broadcast.2299), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.2301 = f32[32]{0} convert(f32[32]{0} %divide.2300), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.2302 = f32[1,1,1,1,32]{4,3,2,1,0} reshape(f32[32]{0} %convert.2301), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.2303 = f32[1,32]{1,0} reshape(f32[1,1,1,1,32]{4,3,2,1,0} %reshape.2302), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.2304 = f32[1,8,14,14,32]{4,3,2,1,0} broadcast(f32[1,32]{1,0} %reshape.2303), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.2305 = f32[1,8,14,14,32]{4,3,2,1,0} subtract(f32[1,8,14,14,32]{4,3,2,1,0} %broadcast.2304, f32[1,8,14,14,32]{4,3,2,1,0} %convolution.2282), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.2306 = f32[1,8,14,14,32]{4,3,2,1,0} multiply(f32[1,8,14,14,32]{4,3,2,1,0} %subtract.2305, f32[1,8,14,14,32]{4,3,2,1,0} %subtract.2305), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.2307 = f32[1,8,14,14,32]{4,3,2,1,0} convert(f32[1,8,14,14,32]{4,3,2,1,0} %multiply.2306), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %constant.2308 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.2309 = f32[] convert(f32[] %constant.2308), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reduce.2314 = f32[32]{0} reduce(f32[1,8,14,14,32]{4,3,2,1,0} %convert.2307, f32[] %convert.2309), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4e_Branch_2_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.2310, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2315 = s32[] get-dimension-size(f32[1,8,14,14,32]{4,3,2,1,0} %convert.2307), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2316 = s32[] get-dimension-size(f32[1,8,14,14,32]{4,3,2,1,0} %convert.2307), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.2317 = s32[] multiply(s32[] %get-dimension-size.2315, s32[] %get-dimension-size.2316), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2318 = s32[] get-dimension-size(f32[1,8,14,14,32]{4,3,2,1,0} %convert.2307), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.2319 = s32[] multiply(s32[] %multiply.2317, s32[] %get-dimension-size.2318), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2320 = s32[] get-dimension-size(f32[1,8,14,14,32]{4,3,2,1,0} %convert.2307), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.2321 = s32[] multiply(s32[] %multiply.2319, s32[] %get-dimension-size.2320), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.2322 = f32[] convert(s32[] %multiply.2321), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.2323 = f32[32]{0} broadcast(f32[] %convert.2322), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %divide.2324 = f32[32]{0} divide(f32[32]{0} %reduce.2314, f32[32]{0} %broadcast.2323), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.2325 = f32[32]{0} convert(f32[32]{0} %divide.2324), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reshape.2326 = f32[1,1,1,1,32]{4,3,2,1,0} reshape(f32[32]{0} %convert.2325), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %add.2329 = f32[1,1,1,1,32]{4,3,2,1,0} add(f32[1,1,1,1,32]{4,3,2,1,0} %broadcast.2328, f32[1,1,1,1,32]{4,3,2,1,0} %reshape.2326), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.2330 = f32[1,1,1,1,32]{4,3,2,1,0} rsqrt(f32[1,1,1,1,32]{4,3,2,1,0} %add.2329), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.2331 = f32[1,32]{1,0} reshape(f32[1,1,1,1,32]{4,3,2,1,0} %rsqrt.2330), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.2332 = f32[1,8,14,14,32]{4,3,2,1,0} broadcast(f32[1,32]{1,0} %reshape.2331), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %multiply.2333 = f32[1,8,14,14,32]{4,3,2,1,0} multiply(f32[1,8,14,14,32]{4,3,2,1,0} %broadcast.2332, f32[1,8,14,14,32]{4,3,2,1,0} %convolution.2282), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg66.67 = f32[1,1,1,1,32]{4,3,2,1,0} parameter(66), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.2334 = f32[1,1,1,1,32]{4,3,2,1,0} multiply(f32[1,1,1,1,32]{4,3,2,1,0} %rsqrt.2330, f32[1,1,1,1,32]{4,3,2,1,0} %reshape.2302), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.2335 = f32[1,1,1,1,32]{4,3,2,1,0} subtract(f32[1,1,1,1,32]{4,3,2,1,0} %arg66.67, f32[1,1,1,1,32]{4,3,2,1,0} %multiply.2334), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.2336 = f32[1,32]{1,0} reshape(f32[1,1,1,1,32]{4,3,2,1,0} %subtract.2335), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.2337 = f32[1,8,14,14,32]{4,3,2,1,0} broadcast(f32[1,32]{1,0} %reshape.2336), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.2338 = f32[1,8,14,14,32]{4,3,2,1,0} add(f32[1,8,14,14,32]{4,3,2,1,0} %multiply.2333, f32[1,8,14,14,32]{4,3,2,1,0} %broadcast.2337), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %maximum.2341 = f32[1,8,14,14,32]{4,3,2,1,0} maximum(f32[1,8,14,14,32]{4,3,2,1,0} %broadcast.2340, f32[1,8,14,14,32]{4,3,2,1,0} %add.2338), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/Relu"}
  %arg72.73 = f32[3,3,3,32,64]{4,3,2,1,0} parameter(72), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.2342 = f32[1,8,14,14,64]{4,3,2,1,0} convolution(f32[1,8,14,14,32]{4,3,2,1,0} %maximum.2341, f32[3,3,3,32,64]{4,3,2,1,0} %arg72.73), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/conv_3d/convolution"}
  %convert.2343 = f32[1,8,14,14,64]{4,3,2,1,0} convert(f32[1,8,14,14,64]{4,3,2,1,0} %convolution.2342), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %constant.2344 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.2345 = f32[] convert(f32[] %constant.2344), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reduce.2350 = f32[64]{0} reduce(f32[1,8,14,14,64]{4,3,2,1,0} %convert.2343, f32[] %convert.2345), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4e_Branch_2_Conv3d_0b_3x3_batch_norm_normalize_moments_mean-reduction.2346, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2351 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.2343), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2352 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.2343), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.2353 = s32[] multiply(s32[] %get-dimension-size.2351, s32[] %get-dimension-size.2352), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2354 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.2343), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.2355 = s32[] multiply(s32[] %multiply.2353, s32[] %get-dimension-size.2354), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2356 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.2343), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.2357 = s32[] multiply(s32[] %multiply.2355, s32[] %get-dimension-size.2356), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.2358 = f32[] convert(s32[] %multiply.2357), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %broadcast.2359 = f32[64]{0} broadcast(f32[] %convert.2358), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %divide.2360 = f32[64]{0} divide(f32[64]{0} %reduce.2350, f32[64]{0} %broadcast.2359), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.2361 = f32[64]{0} convert(f32[64]{0} %divide.2360), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reshape.2362 = f32[1,1,1,1,64]{4,3,2,1,0} reshape(f32[64]{0} %convert.2361), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reshape.2363 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %reshape.2362), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.2364 = f32[1,8,14,14,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.2363), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.2365 = f32[1,8,14,14,64]{4,3,2,1,0} subtract(f32[1,8,14,14,64]{4,3,2,1,0} %broadcast.2364, f32[1,8,14,14,64]{4,3,2,1,0} %convolution.2342), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.2366 = f32[1,8,14,14,64]{4,3,2,1,0} multiply(f32[1,8,14,14,64]{4,3,2,1,0} %subtract.2365, f32[1,8,14,14,64]{4,3,2,1,0} %subtract.2365), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %convert.2367 = f32[1,8,14,14,64]{4,3,2,1,0} convert(f32[1,8,14,14,64]{4,3,2,1,0} %multiply.2366), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %constant.2368 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.2369 = f32[] convert(f32[] %constant.2368), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %reduce.2374 = f32[64]{0} reduce(f32[1,8,14,14,64]{4,3,2,1,0} %convert.2367, f32[] %convert.2369), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4e_Branch_2_Conv3d_0b_3x3_batch_norm_normalize_moments_variance-reduction.2370, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2375 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.2367), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2376 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.2367), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.2377 = s32[] multiply(s32[] %get-dimension-size.2375, s32[] %get-dimension-size.2376), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2378 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.2367), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.2379 = s32[] multiply(s32[] %multiply.2377, s32[] %get-dimension-size.2378), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2380 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.2367), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.2381 = s32[] multiply(s32[] %multiply.2379, s32[] %get-dimension-size.2380), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.2382 = f32[] convert(s32[] %multiply.2381), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %broadcast.2383 = f32[64]{0} broadcast(f32[] %convert.2382), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %divide.2384 = f32[64]{0} divide(f32[64]{0} %reduce.2374, f32[64]{0} %broadcast.2383), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.2385 = f32[64]{0} convert(f32[64]{0} %divide.2384), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %reshape.2386 = f32[1,1,1,1,64]{4,3,2,1,0} reshape(f32[64]{0} %convert.2385), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %add.2389 = f32[1,1,1,1,64]{4,3,2,1,0} add(f32[1,1,1,1,64]{4,3,2,1,0} %broadcast.2388, f32[1,1,1,1,64]{4,3,2,1,0} %reshape.2386), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %rsqrt.2390 = f32[1,1,1,1,64]{4,3,2,1,0} rsqrt(f32[1,1,1,1,64]{4,3,2,1,0} %add.2389), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.2391 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.2390), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %broadcast.2392 = f32[1,8,14,14,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.2391), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %multiply.2393 = f32[1,8,14,14,64]{4,3,2,1,0} multiply(f32[1,8,14,14,64]{4,3,2,1,0} %broadcast.2392, f32[1,8,14,14,64]{4,3,2,1,0} %convolution.2342), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %arg80.81 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(80), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.2394 = f32[1,1,1,1,64]{4,3,2,1,0} multiply(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.2390, f32[1,1,1,1,64]{4,3,2,1,0} %reshape.2362), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.2395 = f32[1,1,1,1,64]{4,3,2,1,0} subtract(f32[1,1,1,1,64]{4,3,2,1,0} %arg80.81, f32[1,1,1,1,64]{4,3,2,1,0} %multiply.2394), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/sub"}
  %reshape.2396 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %subtract.2395), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.2397 = f32[1,8,14,14,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.2396), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %add.2398 = f32[1,8,14,14,64]{4,3,2,1,0} add(f32[1,8,14,14,64]{4,3,2,1,0} %multiply.2393, f32[1,8,14,14,64]{4,3,2,1,0} %broadcast.2397), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %constant.2450 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %broadcast.2451 = f32[1,1,1,1,64]{4,3,2,1,0} broadcast(f32[] %constant.2450), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %constant.2399 = f32[] constant(-inf), metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/MaxPool3d_0a_3x3"}
  %reduce-window.2404 = f32[1,8,14,14,512]{4,3,2,1,0} reduce-window(f32[1,8,14,14,512]{4,3,2,1,0} %maximum.2107, f32[] %constant.2399), window={size=1x3x3x3x1 pad=0_0x1_1x1_1x1_1x0_0}, to_apply=%max_F32.2400, metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/MaxPool3d_0a_3x3"}
  %arg87.88 = f32[1,1,1,512,64]{4,3,2,1,0} parameter(87), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.2405 = f32[1,8,14,14,64]{4,3,2,1,0} convolution(f32[1,8,14,14,512]{4,3,2,1,0} %reduce-window.2404, f32[1,1,1,512,64]{4,3,2,1,0} %arg87.88), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/conv_3d/convolution"}
  %convert.2406 = f32[1,8,14,14,64]{4,3,2,1,0} convert(f32[1,8,14,14,64]{4,3,2,1,0} %convolution.2405), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %constant.2407 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %convert.2408 = f32[] convert(f32[] %constant.2407), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %reduce.2413 = f32[64]{0} reduce(f32[1,8,14,14,64]{4,3,2,1,0} %convert.2406, f32[] %convert.2408), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4e_Branch_3_Conv3d_0b_1x1_batch_norm_normalize_moments_mean-reduction.2409, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2414 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.2406), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2415 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.2406), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %multiply.2416 = s32[] multiply(s32[] %get-dimension-size.2414, s32[] %get-dimension-size.2415), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2417 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.2406), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %multiply.2418 = s32[] multiply(s32[] %multiply.2416, s32[] %get-dimension-size.2417), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2419 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.2406), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %multiply.2420 = s32[] multiply(s32[] %multiply.2418, s32[] %get-dimension-size.2419), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %convert.2421 = f32[] convert(s32[] %multiply.2420), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.2422 = f32[64]{0} broadcast(f32[] %convert.2421), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %divide.2423 = f32[64]{0} divide(f32[64]{0} %reduce.2413, f32[64]{0} %broadcast.2422), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %convert.2424 = f32[64]{0} convert(f32[64]{0} %divide.2423), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %reshape.2425 = f32[1,1,1,1,64]{4,3,2,1,0} reshape(f32[64]{0} %convert.2424), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %reshape.2426 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %reshape.2425), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.2427 = f32[1,8,14,14,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.2426), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.2428 = f32[1,8,14,14,64]{4,3,2,1,0} subtract(f32[1,8,14,14,64]{4,3,2,1,0} %broadcast.2427, f32[1,8,14,14,64]{4,3,2,1,0} %convolution.2405), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.2429 = f32[1,8,14,14,64]{4,3,2,1,0} multiply(f32[1,8,14,14,64]{4,3,2,1,0} %subtract.2428, f32[1,8,14,14,64]{4,3,2,1,0} %subtract.2428), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.2430 = f32[1,8,14,14,64]{4,3,2,1,0} convert(f32[1,8,14,14,64]{4,3,2,1,0} %multiply.2429), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %constant.2431 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %convert.2432 = f32[] convert(f32[] %constant.2431), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %reduce.2437 = f32[64]{0} reduce(f32[1,8,14,14,64]{4,3,2,1,0} %convert.2430, f32[] %convert.2432), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4e_Branch_3_Conv3d_0b_1x1_batch_norm_normalize_moments_variance-reduction.2433, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2438 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.2430), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2439 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.2430), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %multiply.2440 = s32[] multiply(s32[] %get-dimension-size.2438, s32[] %get-dimension-size.2439), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2441 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.2430), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %multiply.2442 = s32[] multiply(s32[] %multiply.2440, s32[] %get-dimension-size.2441), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2443 = s32[] get-dimension-size(f32[1,8,14,14,64]{4,3,2,1,0} %convert.2430), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %multiply.2444 = s32[] multiply(s32[] %multiply.2442, s32[] %get-dimension-size.2443), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %convert.2445 = f32[] convert(s32[] %multiply.2444), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.2446 = f32[64]{0} broadcast(f32[] %convert.2445), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %divide.2447 = f32[64]{0} divide(f32[64]{0} %reduce.2437, f32[64]{0} %broadcast.2446), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %convert.2448 = f32[64]{0} convert(f32[64]{0} %divide.2447), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %reshape.2449 = f32[1,1,1,1,64]{4,3,2,1,0} reshape(f32[64]{0} %convert.2448), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %add.2452 = f32[1,1,1,1,64]{4,3,2,1,0} add(f32[1,1,1,1,64]{4,3,2,1,0} %broadcast.2451, f32[1,1,1,1,64]{4,3,2,1,0} %reshape.2449), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %rsqrt.2453 = f32[1,1,1,1,64]{4,3,2,1,0} rsqrt(f32[1,1,1,1,64]{4,3,2,1,0} %add.2452), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.2454 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.2453), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %broadcast.2455 = f32[1,8,14,14,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.2454), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %multiply.2456 = f32[1,8,14,14,64]{4,3,2,1,0} multiply(f32[1,8,14,14,64]{4,3,2,1,0} %broadcast.2455, f32[1,8,14,14,64]{4,3,2,1,0} %convolution.2405), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %arg89.90 = f32[1,1,1,1,64]{4,3,2,1,0} parameter(89), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.2457 = f32[1,1,1,1,64]{4,3,2,1,0} multiply(f32[1,1,1,1,64]{4,3,2,1,0} %rsqrt.2453, f32[1,1,1,1,64]{4,3,2,1,0} %reshape.2425), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.2458 = f32[1,1,1,1,64]{4,3,2,1,0} subtract(f32[1,1,1,1,64]{4,3,2,1,0} %arg89.90, f32[1,1,1,1,64]{4,3,2,1,0} %multiply.2457), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/sub"}
  %reshape.2459 = f32[1,64]{1,0} reshape(f32[1,1,1,1,64]{4,3,2,1,0} %subtract.2458), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.2460 = f32[1,8,14,14,64]{4,3,2,1,0} broadcast(f32[1,64]{1,0} %reshape.2459), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %add.2461 = f32[1,8,14,14,64]{4,3,2,1,0} add(f32[1,8,14,14,64]{4,3,2,1,0} %multiply.2456, f32[1,8,14,14,64]{4,3,2,1,0} %broadcast.2460), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %concatenate.2462 = f32[1,8,14,14,528]{4,3,2,1,0} concatenate(f32[1,8,14,14,112]{4,3,2,1,0} %add.2164, f32[1,8,14,14,288]{4,3,2,1,0} %add.2281, f32[1,8,14,14,64]{4,3,2,1,0} %add.2398, f32[1,8,14,14,64]{4,3,2,1,0} %add.2461), dimensions={4}, metadata={op_type="ConcatV2" op_name="RGB/inception_i3d/Mixed_4e/concat"}
  %maximum.2465 = f32[1,8,14,14,528]{4,3,2,1,0} maximum(f32[1,8,14,14,528]{4,3,2,1,0} %broadcast.2464, f32[1,8,14,14,528]{4,3,2,1,0} %concatenate.2462), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/Relu"}
  %arg102.103 = f32[1,1,1,528,256]{4,3,2,1,0} parameter(102), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.2466 = f32[1,8,14,14,256]{4,3,2,1,0} convolution(f32[1,8,14,14,528]{4,3,2,1,0} %maximum.2465, f32[1,1,1,528,256]{4,3,2,1,0} %arg102.103), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/conv_3d/convolution"}
  %convert.2467 = f32[1,8,14,14,256]{4,3,2,1,0} convert(f32[1,8,14,14,256]{4,3,2,1,0} %convolution.2466), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %constant.2468 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.2469 = f32[] convert(f32[] %constant.2468), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reduce.2474 = f32[256]{0} reduce(f32[1,8,14,14,256]{4,3,2,1,0} %convert.2467, f32[] %convert.2469), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4f_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.2470, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2475 = s32[] get-dimension-size(f32[1,8,14,14,256]{4,3,2,1,0} %convert.2467), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2476 = s32[] get-dimension-size(f32[1,8,14,14,256]{4,3,2,1,0} %convert.2467), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.2477 = s32[] multiply(s32[] %get-dimension-size.2475, s32[] %get-dimension-size.2476), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2478 = s32[] get-dimension-size(f32[1,8,14,14,256]{4,3,2,1,0} %convert.2467), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.2479 = s32[] multiply(s32[] %multiply.2477, s32[] %get-dimension-size.2478), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2480 = s32[] get-dimension-size(f32[1,8,14,14,256]{4,3,2,1,0} %convert.2467), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.2481 = s32[] multiply(s32[] %multiply.2479, s32[] %get-dimension-size.2480), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.2482 = f32[] convert(s32[] %multiply.2481), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.2483 = f32[256]{0} broadcast(f32[] %convert.2482), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %divide.2484 = f32[256]{0} divide(f32[256]{0} %reduce.2474, f32[256]{0} %broadcast.2483), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.2485 = f32[256]{0} convert(f32[256]{0} %divide.2484), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.2486 = f32[1,1,1,1,256]{4,3,2,1,0} reshape(f32[256]{0} %convert.2485), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.2487 = f32[1,256]{1,0} reshape(f32[1,1,1,1,256]{4,3,2,1,0} %reshape.2486), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.2488 = f32[1,8,14,14,256]{4,3,2,1,0} broadcast(f32[1,256]{1,0} %reshape.2487), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.2489 = f32[1,8,14,14,256]{4,3,2,1,0} subtract(f32[1,8,14,14,256]{4,3,2,1,0} %broadcast.2488, f32[1,8,14,14,256]{4,3,2,1,0} %convolution.2466), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.2490 = f32[1,8,14,14,256]{4,3,2,1,0} multiply(f32[1,8,14,14,256]{4,3,2,1,0} %subtract.2489, f32[1,8,14,14,256]{4,3,2,1,0} %subtract.2489), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.2491 = f32[1,8,14,14,256]{4,3,2,1,0} convert(f32[1,8,14,14,256]{4,3,2,1,0} %multiply.2490), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %constant.2492 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.2493 = f32[] convert(f32[] %constant.2492), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reduce.2498 = f32[256]{0} reduce(f32[1,8,14,14,256]{4,3,2,1,0} %convert.2491, f32[] %convert.2493), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4f_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.2494, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2499 = s32[] get-dimension-size(f32[1,8,14,14,256]{4,3,2,1,0} %convert.2491), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2500 = s32[] get-dimension-size(f32[1,8,14,14,256]{4,3,2,1,0} %convert.2491), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.2501 = s32[] multiply(s32[] %get-dimension-size.2499, s32[] %get-dimension-size.2500), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2502 = s32[] get-dimension-size(f32[1,8,14,14,256]{4,3,2,1,0} %convert.2491), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.2503 = s32[] multiply(s32[] %multiply.2501, s32[] %get-dimension-size.2502), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2504 = s32[] get-dimension-size(f32[1,8,14,14,256]{4,3,2,1,0} %convert.2491), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.2505 = s32[] multiply(s32[] %multiply.2503, s32[] %get-dimension-size.2504), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.2506 = f32[] convert(s32[] %multiply.2505), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.2507 = f32[256]{0} broadcast(f32[] %convert.2506), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %divide.2508 = f32[256]{0} divide(f32[256]{0} %reduce.2498, f32[256]{0} %broadcast.2507), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.2509 = f32[256]{0} convert(f32[256]{0} %divide.2508), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reshape.2510 = f32[1,1,1,1,256]{4,3,2,1,0} reshape(f32[256]{0} %convert.2509), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %add.2513 = f32[1,1,1,1,256]{4,3,2,1,0} add(f32[1,1,1,1,256]{4,3,2,1,0} %broadcast.2512, f32[1,1,1,1,256]{4,3,2,1,0} %reshape.2510), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.2514 = f32[1,1,1,1,256]{4,3,2,1,0} rsqrt(f32[1,1,1,1,256]{4,3,2,1,0} %add.2513), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.2515 = f32[1,256]{1,0} reshape(f32[1,1,1,1,256]{4,3,2,1,0} %rsqrt.2514), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.2516 = f32[1,8,14,14,256]{4,3,2,1,0} broadcast(f32[1,256]{1,0} %reshape.2515), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %multiply.2517 = f32[1,8,14,14,256]{4,3,2,1,0} multiply(f32[1,8,14,14,256]{4,3,2,1,0} %broadcast.2516, f32[1,8,14,14,256]{4,3,2,1,0} %convolution.2466), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg107.108 = f32[1,1,1,1,256]{4,3,2,1,0} parameter(107), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.2518 = f32[1,1,1,1,256]{4,3,2,1,0} multiply(f32[1,1,1,1,256]{4,3,2,1,0} %rsqrt.2514, f32[1,1,1,1,256]{4,3,2,1,0} %reshape.2486), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.2519 = f32[1,1,1,1,256]{4,3,2,1,0} subtract(f32[1,1,1,1,256]{4,3,2,1,0} %arg107.108, f32[1,1,1,1,256]{4,3,2,1,0} %multiply.2518), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.2520 = f32[1,256]{1,0} reshape(f32[1,1,1,1,256]{4,3,2,1,0} %subtract.2519), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.2521 = f32[1,8,14,14,256]{4,3,2,1,0} broadcast(f32[1,256]{1,0} %reshape.2520), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.2522 = f32[1,8,14,14,256]{4,3,2,1,0} add(f32[1,8,14,14,256]{4,3,2,1,0} %multiply.2517, f32[1,8,14,14,256]{4,3,2,1,0} %broadcast.2521), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %constant.2628 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %broadcast.2629 = f32[1,1,1,1,320]{4,3,2,1,0} broadcast(f32[] %constant.2628), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %constant.2580 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/Relu"}
  %broadcast.2581 = f32[1,8,14,14,160]{4,3,2,1,0} broadcast(f32[] %constant.2580), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/Relu"}
  %constant.2568 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.2569 = f32[1,1,1,1,160]{4,3,2,1,0} broadcast(f32[] %constant.2568), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %arg115.116 = f32[1,1,1,528,160]{4,3,2,1,0} parameter(115), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.2523 = f32[1,8,14,14,160]{4,3,2,1,0} convolution(f32[1,8,14,14,528]{4,3,2,1,0} %maximum.2465, f32[1,1,1,528,160]{4,3,2,1,0} %arg115.116), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/conv_3d/convolution"}
  %convert.2524 = f32[1,8,14,14,160]{4,3,2,1,0} convert(f32[1,8,14,14,160]{4,3,2,1,0} %convolution.2523), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %constant.2525 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.2526 = f32[] convert(f32[] %constant.2525), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reduce.2531 = f32[160]{0} reduce(f32[1,8,14,14,160]{4,3,2,1,0} %convert.2524, f32[] %convert.2526), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4f_Branch_1_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.2527, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2532 = s32[] get-dimension-size(f32[1,8,14,14,160]{4,3,2,1,0} %convert.2524), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2533 = s32[] get-dimension-size(f32[1,8,14,14,160]{4,3,2,1,0} %convert.2524), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.2534 = s32[] multiply(s32[] %get-dimension-size.2532, s32[] %get-dimension-size.2533), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2535 = s32[] get-dimension-size(f32[1,8,14,14,160]{4,3,2,1,0} %convert.2524), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.2536 = s32[] multiply(s32[] %multiply.2534, s32[] %get-dimension-size.2535), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2537 = s32[] get-dimension-size(f32[1,8,14,14,160]{4,3,2,1,0} %convert.2524), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.2538 = s32[] multiply(s32[] %multiply.2536, s32[] %get-dimension-size.2537), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.2539 = f32[] convert(s32[] %multiply.2538), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.2540 = f32[160]{0} broadcast(f32[] %convert.2539), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %divide.2541 = f32[160]{0} divide(f32[160]{0} %reduce.2531, f32[160]{0} %broadcast.2540), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.2542 = f32[160]{0} convert(f32[160]{0} %divide.2541), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.2543 = f32[1,1,1,1,160]{4,3,2,1,0} reshape(f32[160]{0} %convert.2542), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.2544 = f32[1,160]{1,0} reshape(f32[1,1,1,1,160]{4,3,2,1,0} %reshape.2543), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.2545 = f32[1,8,14,14,160]{4,3,2,1,0} broadcast(f32[1,160]{1,0} %reshape.2544), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.2546 = f32[1,8,14,14,160]{4,3,2,1,0} subtract(f32[1,8,14,14,160]{4,3,2,1,0} %broadcast.2545, f32[1,8,14,14,160]{4,3,2,1,0} %convolution.2523), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.2547 = f32[1,8,14,14,160]{4,3,2,1,0} multiply(f32[1,8,14,14,160]{4,3,2,1,0} %subtract.2546, f32[1,8,14,14,160]{4,3,2,1,0} %subtract.2546), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.2548 = f32[1,8,14,14,160]{4,3,2,1,0} convert(f32[1,8,14,14,160]{4,3,2,1,0} %multiply.2547), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %constant.2549 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.2550 = f32[] convert(f32[] %constant.2549), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reduce.2555 = f32[160]{0} reduce(f32[1,8,14,14,160]{4,3,2,1,0} %convert.2548, f32[] %convert.2550), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4f_Branch_1_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.2551, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2556 = s32[] get-dimension-size(f32[1,8,14,14,160]{4,3,2,1,0} %convert.2548), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2557 = s32[] get-dimension-size(f32[1,8,14,14,160]{4,3,2,1,0} %convert.2548), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.2558 = s32[] multiply(s32[] %get-dimension-size.2556, s32[] %get-dimension-size.2557), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2559 = s32[] get-dimension-size(f32[1,8,14,14,160]{4,3,2,1,0} %convert.2548), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.2560 = s32[] multiply(s32[] %multiply.2558, s32[] %get-dimension-size.2559), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2561 = s32[] get-dimension-size(f32[1,8,14,14,160]{4,3,2,1,0} %convert.2548), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.2562 = s32[] multiply(s32[] %multiply.2560, s32[] %get-dimension-size.2561), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.2563 = f32[] convert(s32[] %multiply.2562), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.2564 = f32[160]{0} broadcast(f32[] %convert.2563), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %divide.2565 = f32[160]{0} divide(f32[160]{0} %reduce.2555, f32[160]{0} %broadcast.2564), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.2566 = f32[160]{0} convert(f32[160]{0} %divide.2565), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reshape.2567 = f32[1,1,1,1,160]{4,3,2,1,0} reshape(f32[160]{0} %convert.2566), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %add.2570 = f32[1,1,1,1,160]{4,3,2,1,0} add(f32[1,1,1,1,160]{4,3,2,1,0} %broadcast.2569, f32[1,1,1,1,160]{4,3,2,1,0} %reshape.2567), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.2571 = f32[1,1,1,1,160]{4,3,2,1,0} rsqrt(f32[1,1,1,1,160]{4,3,2,1,0} %add.2570), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.2572 = f32[1,160]{1,0} reshape(f32[1,1,1,1,160]{4,3,2,1,0} %rsqrt.2571), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.2573 = f32[1,8,14,14,160]{4,3,2,1,0} broadcast(f32[1,160]{1,0} %reshape.2572), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %multiply.2574 = f32[1,8,14,14,160]{4,3,2,1,0} multiply(f32[1,8,14,14,160]{4,3,2,1,0} %broadcast.2573, f32[1,8,14,14,160]{4,3,2,1,0} %convolution.2523), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg6.7 = f32[1,1,1,1,160]{4,3,2,1,0} parameter(6), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.2575 = f32[1,1,1,1,160]{4,3,2,1,0} multiply(f32[1,1,1,1,160]{4,3,2,1,0} %rsqrt.2571, f32[1,1,1,1,160]{4,3,2,1,0} %reshape.2543), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.2576 = f32[1,1,1,1,160]{4,3,2,1,0} subtract(f32[1,1,1,1,160]{4,3,2,1,0} %arg6.7, f32[1,1,1,1,160]{4,3,2,1,0} %multiply.2575), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.2577 = f32[1,160]{1,0} reshape(f32[1,1,1,1,160]{4,3,2,1,0} %subtract.2576), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.2578 = f32[1,8,14,14,160]{4,3,2,1,0} broadcast(f32[1,160]{1,0} %reshape.2577), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.2579 = f32[1,8,14,14,160]{4,3,2,1,0} add(f32[1,8,14,14,160]{4,3,2,1,0} %multiply.2574, f32[1,8,14,14,160]{4,3,2,1,0} %broadcast.2578), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %maximum.2582 = f32[1,8,14,14,160]{4,3,2,1,0} maximum(f32[1,8,14,14,160]{4,3,2,1,0} %broadcast.2581, f32[1,8,14,14,160]{4,3,2,1,0} %add.2579), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/Relu"}
  %arg18.19 = f32[3,3,3,160,320]{4,3,2,1,0} parameter(18), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.2583 = f32[1,8,14,14,320]{4,3,2,1,0} convolution(f32[1,8,14,14,160]{4,3,2,1,0} %maximum.2582, f32[3,3,3,160,320]{4,3,2,1,0} %arg18.19), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/conv_3d/convolution"}
  %convert.2584 = f32[1,8,14,14,320]{4,3,2,1,0} convert(f32[1,8,14,14,320]{4,3,2,1,0} %convolution.2583), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %constant.2585 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.2586 = f32[] convert(f32[] %constant.2585), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reduce.2591 = f32[320]{0} reduce(f32[1,8,14,14,320]{4,3,2,1,0} %convert.2584, f32[] %convert.2586), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4f_Branch_1_Conv3d_0b_3x3_batch_norm_normalize_moments_mean-reduction.2587, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2592 = s32[] get-dimension-size(f32[1,8,14,14,320]{4,3,2,1,0} %convert.2584), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2593 = s32[] get-dimension-size(f32[1,8,14,14,320]{4,3,2,1,0} %convert.2584), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.2594 = s32[] multiply(s32[] %get-dimension-size.2592, s32[] %get-dimension-size.2593), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2595 = s32[] get-dimension-size(f32[1,8,14,14,320]{4,3,2,1,0} %convert.2584), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.2596 = s32[] multiply(s32[] %multiply.2594, s32[] %get-dimension-size.2595), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2597 = s32[] get-dimension-size(f32[1,8,14,14,320]{4,3,2,1,0} %convert.2584), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.2598 = s32[] multiply(s32[] %multiply.2596, s32[] %get-dimension-size.2597), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.2599 = f32[] convert(s32[] %multiply.2598), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %broadcast.2600 = f32[320]{0} broadcast(f32[] %convert.2599), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %divide.2601 = f32[320]{0} divide(f32[320]{0} %reduce.2591, f32[320]{0} %broadcast.2600), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.2602 = f32[320]{0} convert(f32[320]{0} %divide.2601), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reshape.2603 = f32[1,1,1,1,320]{4,3,2,1,0} reshape(f32[320]{0} %convert.2602), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reshape.2604 = f32[1,320]{1,0} reshape(f32[1,1,1,1,320]{4,3,2,1,0} %reshape.2603), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.2605 = f32[1,8,14,14,320]{4,3,2,1,0} broadcast(f32[1,320]{1,0} %reshape.2604), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.2606 = f32[1,8,14,14,320]{4,3,2,1,0} subtract(f32[1,8,14,14,320]{4,3,2,1,0} %broadcast.2605, f32[1,8,14,14,320]{4,3,2,1,0} %convolution.2583), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.2607 = f32[1,8,14,14,320]{4,3,2,1,0} multiply(f32[1,8,14,14,320]{4,3,2,1,0} %subtract.2606, f32[1,8,14,14,320]{4,3,2,1,0} %subtract.2606), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %convert.2608 = f32[1,8,14,14,320]{4,3,2,1,0} convert(f32[1,8,14,14,320]{4,3,2,1,0} %multiply.2607), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %constant.2609 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.2610 = f32[] convert(f32[] %constant.2609), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %reduce.2615 = f32[320]{0} reduce(f32[1,8,14,14,320]{4,3,2,1,0} %convert.2608, f32[] %convert.2610), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4f_Branch_1_Conv3d_0b_3x3_batch_norm_normalize_moments_variance-reduction.2611, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2616 = s32[] get-dimension-size(f32[1,8,14,14,320]{4,3,2,1,0} %convert.2608), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2617 = s32[] get-dimension-size(f32[1,8,14,14,320]{4,3,2,1,0} %convert.2608), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.2618 = s32[] multiply(s32[] %get-dimension-size.2616, s32[] %get-dimension-size.2617), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2619 = s32[] get-dimension-size(f32[1,8,14,14,320]{4,3,2,1,0} %convert.2608), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.2620 = s32[] multiply(s32[] %multiply.2618, s32[] %get-dimension-size.2619), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2621 = s32[] get-dimension-size(f32[1,8,14,14,320]{4,3,2,1,0} %convert.2608), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.2622 = s32[] multiply(s32[] %multiply.2620, s32[] %get-dimension-size.2621), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.2623 = f32[] convert(s32[] %multiply.2622), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %broadcast.2624 = f32[320]{0} broadcast(f32[] %convert.2623), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %divide.2625 = f32[320]{0} divide(f32[320]{0} %reduce.2615, f32[320]{0} %broadcast.2624), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.2626 = f32[320]{0} convert(f32[320]{0} %divide.2625), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %reshape.2627 = f32[1,1,1,1,320]{4,3,2,1,0} reshape(f32[320]{0} %convert.2626), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %add.2630 = f32[1,1,1,1,320]{4,3,2,1,0} add(f32[1,1,1,1,320]{4,3,2,1,0} %broadcast.2629, f32[1,1,1,1,320]{4,3,2,1,0} %reshape.2627), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %rsqrt.2631 = f32[1,1,1,1,320]{4,3,2,1,0} rsqrt(f32[1,1,1,1,320]{4,3,2,1,0} %add.2630), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.2632 = f32[1,320]{1,0} reshape(f32[1,1,1,1,320]{4,3,2,1,0} %rsqrt.2631), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %broadcast.2633 = f32[1,8,14,14,320]{4,3,2,1,0} broadcast(f32[1,320]{1,0} %reshape.2632), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %multiply.2634 = f32[1,8,14,14,320]{4,3,2,1,0} multiply(f32[1,8,14,14,320]{4,3,2,1,0} %broadcast.2633, f32[1,8,14,14,320]{4,3,2,1,0} %convolution.2583), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %arg28.29 = f32[1,1,1,1,320]{4,3,2,1,0} parameter(28), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.2635 = f32[1,1,1,1,320]{4,3,2,1,0} multiply(f32[1,1,1,1,320]{4,3,2,1,0} %rsqrt.2631, f32[1,1,1,1,320]{4,3,2,1,0} %reshape.2603), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.2636 = f32[1,1,1,1,320]{4,3,2,1,0} subtract(f32[1,1,1,1,320]{4,3,2,1,0} %arg28.29, f32[1,1,1,1,320]{4,3,2,1,0} %multiply.2635), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/sub"}
  %reshape.2637 = f32[1,320]{1,0} reshape(f32[1,1,1,1,320]{4,3,2,1,0} %subtract.2636), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.2638 = f32[1,8,14,14,320]{4,3,2,1,0} broadcast(f32[1,320]{1,0} %reshape.2637), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %add.2639 = f32[1,8,14,14,320]{4,3,2,1,0} add(f32[1,8,14,14,320]{4,3,2,1,0} %multiply.2634, f32[1,8,14,14,320]{4,3,2,1,0} %broadcast.2638), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %constant.2745 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %broadcast.2746 = f32[1,1,1,1,128]{4,3,2,1,0} broadcast(f32[] %constant.2745), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %constant.2697 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/Relu"}
  %broadcast.2698 = f32[1,8,14,14,32]{4,3,2,1,0} broadcast(f32[] %constant.2697), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/Relu"}
  %constant.2685 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.2686 = f32[1,1,1,1,32]{4,3,2,1,0} broadcast(f32[] %constant.2685), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %arg36.37 = f32[1,1,1,528,32]{4,3,2,1,0} parameter(36), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.2640 = f32[1,8,14,14,32]{4,3,2,1,0} convolution(f32[1,8,14,14,528]{4,3,2,1,0} %maximum.2465, f32[1,1,1,528,32]{4,3,2,1,0} %arg36.37), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/conv_3d/convolution"}
  %convert.2641 = f32[1,8,14,14,32]{4,3,2,1,0} convert(f32[1,8,14,14,32]{4,3,2,1,0} %convolution.2640), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %constant.2642 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.2643 = f32[] convert(f32[] %constant.2642), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reduce.2648 = f32[32]{0} reduce(f32[1,8,14,14,32]{4,3,2,1,0} %convert.2641, f32[] %convert.2643), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4f_Branch_2_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.2644, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2649 = s32[] get-dimension-size(f32[1,8,14,14,32]{4,3,2,1,0} %convert.2641), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2650 = s32[] get-dimension-size(f32[1,8,14,14,32]{4,3,2,1,0} %convert.2641), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.2651 = s32[] multiply(s32[] %get-dimension-size.2649, s32[] %get-dimension-size.2650), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2652 = s32[] get-dimension-size(f32[1,8,14,14,32]{4,3,2,1,0} %convert.2641), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.2653 = s32[] multiply(s32[] %multiply.2651, s32[] %get-dimension-size.2652), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2654 = s32[] get-dimension-size(f32[1,8,14,14,32]{4,3,2,1,0} %convert.2641), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.2655 = s32[] multiply(s32[] %multiply.2653, s32[] %get-dimension-size.2654), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.2656 = f32[] convert(s32[] %multiply.2655), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.2657 = f32[32]{0} broadcast(f32[] %convert.2656), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %divide.2658 = f32[32]{0} divide(f32[32]{0} %reduce.2648, f32[32]{0} %broadcast.2657), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.2659 = f32[32]{0} convert(f32[32]{0} %divide.2658), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.2660 = f32[1,1,1,1,32]{4,3,2,1,0} reshape(f32[32]{0} %convert.2659), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.2661 = f32[1,32]{1,0} reshape(f32[1,1,1,1,32]{4,3,2,1,0} %reshape.2660), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.2662 = f32[1,8,14,14,32]{4,3,2,1,0} broadcast(f32[1,32]{1,0} %reshape.2661), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.2663 = f32[1,8,14,14,32]{4,3,2,1,0} subtract(f32[1,8,14,14,32]{4,3,2,1,0} %broadcast.2662, f32[1,8,14,14,32]{4,3,2,1,0} %convolution.2640), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.2664 = f32[1,8,14,14,32]{4,3,2,1,0} multiply(f32[1,8,14,14,32]{4,3,2,1,0} %subtract.2663, f32[1,8,14,14,32]{4,3,2,1,0} %subtract.2663), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.2665 = f32[1,8,14,14,32]{4,3,2,1,0} convert(f32[1,8,14,14,32]{4,3,2,1,0} %multiply.2664), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %constant.2666 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.2667 = f32[] convert(f32[] %constant.2666), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reduce.2672 = f32[32]{0} reduce(f32[1,8,14,14,32]{4,3,2,1,0} %convert.2665, f32[] %convert.2667), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4f_Branch_2_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.2668, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2673 = s32[] get-dimension-size(f32[1,8,14,14,32]{4,3,2,1,0} %convert.2665), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2674 = s32[] get-dimension-size(f32[1,8,14,14,32]{4,3,2,1,0} %convert.2665), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.2675 = s32[] multiply(s32[] %get-dimension-size.2673, s32[] %get-dimension-size.2674), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2676 = s32[] get-dimension-size(f32[1,8,14,14,32]{4,3,2,1,0} %convert.2665), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.2677 = s32[] multiply(s32[] %multiply.2675, s32[] %get-dimension-size.2676), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2678 = s32[] get-dimension-size(f32[1,8,14,14,32]{4,3,2,1,0} %convert.2665), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.2679 = s32[] multiply(s32[] %multiply.2677, s32[] %get-dimension-size.2678), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.2680 = f32[] convert(s32[] %multiply.2679), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.2681 = f32[32]{0} broadcast(f32[] %convert.2680), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %divide.2682 = f32[32]{0} divide(f32[32]{0} %reduce.2672, f32[32]{0} %broadcast.2681), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.2683 = f32[32]{0} convert(f32[32]{0} %divide.2682), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reshape.2684 = f32[1,1,1,1,32]{4,3,2,1,0} reshape(f32[32]{0} %convert.2683), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %add.2687 = f32[1,1,1,1,32]{4,3,2,1,0} add(f32[1,1,1,1,32]{4,3,2,1,0} %broadcast.2686, f32[1,1,1,1,32]{4,3,2,1,0} %reshape.2684), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.2688 = f32[1,1,1,1,32]{4,3,2,1,0} rsqrt(f32[1,1,1,1,32]{4,3,2,1,0} %add.2687), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.2689 = f32[1,32]{1,0} reshape(f32[1,1,1,1,32]{4,3,2,1,0} %rsqrt.2688), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.2690 = f32[1,8,14,14,32]{4,3,2,1,0} broadcast(f32[1,32]{1,0} %reshape.2689), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %multiply.2691 = f32[1,8,14,14,32]{4,3,2,1,0} multiply(f32[1,8,14,14,32]{4,3,2,1,0} %broadcast.2690, f32[1,8,14,14,32]{4,3,2,1,0} %convolution.2640), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg45.46 = f32[1,1,1,1,32]{4,3,2,1,0} parameter(45), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.2692 = f32[1,1,1,1,32]{4,3,2,1,0} multiply(f32[1,1,1,1,32]{4,3,2,1,0} %rsqrt.2688, f32[1,1,1,1,32]{4,3,2,1,0} %reshape.2660), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.2693 = f32[1,1,1,1,32]{4,3,2,1,0} subtract(f32[1,1,1,1,32]{4,3,2,1,0} %arg45.46, f32[1,1,1,1,32]{4,3,2,1,0} %multiply.2692), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.2694 = f32[1,32]{1,0} reshape(f32[1,1,1,1,32]{4,3,2,1,0} %subtract.2693), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.2695 = f32[1,8,14,14,32]{4,3,2,1,0} broadcast(f32[1,32]{1,0} %reshape.2694), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.2696 = f32[1,8,14,14,32]{4,3,2,1,0} add(f32[1,8,14,14,32]{4,3,2,1,0} %multiply.2691, f32[1,8,14,14,32]{4,3,2,1,0} %broadcast.2695), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %maximum.2699 = f32[1,8,14,14,32]{4,3,2,1,0} maximum(f32[1,8,14,14,32]{4,3,2,1,0} %broadcast.2698, f32[1,8,14,14,32]{4,3,2,1,0} %add.2696), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/Relu"}
  %arg56.57 = f32[3,3,3,32,128]{4,3,2,1,0} parameter(56), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.2700 = f32[1,8,14,14,128]{4,3,2,1,0} convolution(f32[1,8,14,14,32]{4,3,2,1,0} %maximum.2699, f32[3,3,3,32,128]{4,3,2,1,0} %arg56.57), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/conv_3d/convolution"}
  %convert.2701 = f32[1,8,14,14,128]{4,3,2,1,0} convert(f32[1,8,14,14,128]{4,3,2,1,0} %convolution.2700), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %constant.2702 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.2703 = f32[] convert(f32[] %constant.2702), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reduce.2708 = f32[128]{0} reduce(f32[1,8,14,14,128]{4,3,2,1,0} %convert.2701, f32[] %convert.2703), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4f_Branch_2_Conv3d_0b_3x3_batch_norm_normalize_moments_mean-reduction.2704, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2709 = s32[] get-dimension-size(f32[1,8,14,14,128]{4,3,2,1,0} %convert.2701), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2710 = s32[] get-dimension-size(f32[1,8,14,14,128]{4,3,2,1,0} %convert.2701), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.2711 = s32[] multiply(s32[] %get-dimension-size.2709, s32[] %get-dimension-size.2710), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2712 = s32[] get-dimension-size(f32[1,8,14,14,128]{4,3,2,1,0} %convert.2701), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.2713 = s32[] multiply(s32[] %multiply.2711, s32[] %get-dimension-size.2712), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2714 = s32[] get-dimension-size(f32[1,8,14,14,128]{4,3,2,1,0} %convert.2701), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.2715 = s32[] multiply(s32[] %multiply.2713, s32[] %get-dimension-size.2714), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.2716 = f32[] convert(s32[] %multiply.2715), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %broadcast.2717 = f32[128]{0} broadcast(f32[] %convert.2716), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %divide.2718 = f32[128]{0} divide(f32[128]{0} %reduce.2708, f32[128]{0} %broadcast.2717), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.2719 = f32[128]{0} convert(f32[128]{0} %divide.2718), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reshape.2720 = f32[1,1,1,1,128]{4,3,2,1,0} reshape(f32[128]{0} %convert.2719), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reshape.2721 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %reshape.2720), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.2722 = f32[1,8,14,14,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.2721), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.2723 = f32[1,8,14,14,128]{4,3,2,1,0} subtract(f32[1,8,14,14,128]{4,3,2,1,0} %broadcast.2722, f32[1,8,14,14,128]{4,3,2,1,0} %convolution.2700), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.2724 = f32[1,8,14,14,128]{4,3,2,1,0} multiply(f32[1,8,14,14,128]{4,3,2,1,0} %subtract.2723, f32[1,8,14,14,128]{4,3,2,1,0} %subtract.2723), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %convert.2725 = f32[1,8,14,14,128]{4,3,2,1,0} convert(f32[1,8,14,14,128]{4,3,2,1,0} %multiply.2724), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %constant.2726 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.2727 = f32[] convert(f32[] %constant.2726), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %reduce.2732 = f32[128]{0} reduce(f32[1,8,14,14,128]{4,3,2,1,0} %convert.2725, f32[] %convert.2727), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4f_Branch_2_Conv3d_0b_3x3_batch_norm_normalize_moments_variance-reduction.2728, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2733 = s32[] get-dimension-size(f32[1,8,14,14,128]{4,3,2,1,0} %convert.2725), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2734 = s32[] get-dimension-size(f32[1,8,14,14,128]{4,3,2,1,0} %convert.2725), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.2735 = s32[] multiply(s32[] %get-dimension-size.2733, s32[] %get-dimension-size.2734), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2736 = s32[] get-dimension-size(f32[1,8,14,14,128]{4,3,2,1,0} %convert.2725), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.2737 = s32[] multiply(s32[] %multiply.2735, s32[] %get-dimension-size.2736), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2738 = s32[] get-dimension-size(f32[1,8,14,14,128]{4,3,2,1,0} %convert.2725), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.2739 = s32[] multiply(s32[] %multiply.2737, s32[] %get-dimension-size.2738), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.2740 = f32[] convert(s32[] %multiply.2739), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %broadcast.2741 = f32[128]{0} broadcast(f32[] %convert.2740), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %divide.2742 = f32[128]{0} divide(f32[128]{0} %reduce.2732, f32[128]{0} %broadcast.2741), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.2743 = f32[128]{0} convert(f32[128]{0} %divide.2742), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %reshape.2744 = f32[1,1,1,1,128]{4,3,2,1,0} reshape(f32[128]{0} %convert.2743), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %add.2747 = f32[1,1,1,1,128]{4,3,2,1,0} add(f32[1,1,1,1,128]{4,3,2,1,0} %broadcast.2746, f32[1,1,1,1,128]{4,3,2,1,0} %reshape.2744), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %rsqrt.2748 = f32[1,1,1,1,128]{4,3,2,1,0} rsqrt(f32[1,1,1,1,128]{4,3,2,1,0} %add.2747), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.2749 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.2748), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %broadcast.2750 = f32[1,8,14,14,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.2749), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %multiply.2751 = f32[1,8,14,14,128]{4,3,2,1,0} multiply(f32[1,8,14,14,128]{4,3,2,1,0} %broadcast.2750, f32[1,8,14,14,128]{4,3,2,1,0} %convolution.2700), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %arg67.68 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(67), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.2752 = f32[1,1,1,1,128]{4,3,2,1,0} multiply(f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.2748, f32[1,1,1,1,128]{4,3,2,1,0} %reshape.2720), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.2753 = f32[1,1,1,1,128]{4,3,2,1,0} subtract(f32[1,1,1,1,128]{4,3,2,1,0} %arg67.68, f32[1,1,1,1,128]{4,3,2,1,0} %multiply.2752), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/sub"}
  %reshape.2754 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %subtract.2753), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.2755 = f32[1,8,14,14,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.2754), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %add.2756 = f32[1,8,14,14,128]{4,3,2,1,0} add(f32[1,8,14,14,128]{4,3,2,1,0} %multiply.2751, f32[1,8,14,14,128]{4,3,2,1,0} %broadcast.2755), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %constant.2808 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %broadcast.2809 = f32[1,1,1,1,128]{4,3,2,1,0} broadcast(f32[] %constant.2808), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %constant.2757 = f32[] constant(-inf), metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/MaxPool3d_0a_3x3"}
  %reduce-window.2762 = f32[1,8,14,14,528]{4,3,2,1,0} reduce-window(f32[1,8,14,14,528]{4,3,2,1,0} %maximum.2465, f32[] %constant.2757), window={size=1x3x3x3x1 pad=0_0x1_1x1_1x1_1x0_0}, to_apply=%max_F32.2758, metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/MaxPool3d_0a_3x3"}
  %arg78.79 = f32[1,1,1,528,128]{4,3,2,1,0} parameter(78), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.2763 = f32[1,8,14,14,128]{4,3,2,1,0} convolution(f32[1,8,14,14,528]{4,3,2,1,0} %reduce-window.2762, f32[1,1,1,528,128]{4,3,2,1,0} %arg78.79), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/conv_3d/convolution"}
  %convert.2764 = f32[1,8,14,14,128]{4,3,2,1,0} convert(f32[1,8,14,14,128]{4,3,2,1,0} %convolution.2763), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %constant.2765 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %convert.2766 = f32[] convert(f32[] %constant.2765), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %reduce.2771 = f32[128]{0} reduce(f32[1,8,14,14,128]{4,3,2,1,0} %convert.2764, f32[] %convert.2766), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4f_Branch_3_Conv3d_0b_1x1_batch_norm_normalize_moments_mean-reduction.2767, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2772 = s32[] get-dimension-size(f32[1,8,14,14,128]{4,3,2,1,0} %convert.2764), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2773 = s32[] get-dimension-size(f32[1,8,14,14,128]{4,3,2,1,0} %convert.2764), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %multiply.2774 = s32[] multiply(s32[] %get-dimension-size.2772, s32[] %get-dimension-size.2773), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2775 = s32[] get-dimension-size(f32[1,8,14,14,128]{4,3,2,1,0} %convert.2764), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %multiply.2776 = s32[] multiply(s32[] %multiply.2774, s32[] %get-dimension-size.2775), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2777 = s32[] get-dimension-size(f32[1,8,14,14,128]{4,3,2,1,0} %convert.2764), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %multiply.2778 = s32[] multiply(s32[] %multiply.2776, s32[] %get-dimension-size.2777), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %convert.2779 = f32[] convert(s32[] %multiply.2778), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.2780 = f32[128]{0} broadcast(f32[] %convert.2779), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %divide.2781 = f32[128]{0} divide(f32[128]{0} %reduce.2771, f32[128]{0} %broadcast.2780), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %convert.2782 = f32[128]{0} convert(f32[128]{0} %divide.2781), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %reshape.2783 = f32[1,1,1,1,128]{4,3,2,1,0} reshape(f32[128]{0} %convert.2782), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %reshape.2784 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %reshape.2783), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.2785 = f32[1,8,14,14,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.2784), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.2786 = f32[1,8,14,14,128]{4,3,2,1,0} subtract(f32[1,8,14,14,128]{4,3,2,1,0} %broadcast.2785, f32[1,8,14,14,128]{4,3,2,1,0} %convolution.2763), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.2787 = f32[1,8,14,14,128]{4,3,2,1,0} multiply(f32[1,8,14,14,128]{4,3,2,1,0} %subtract.2786, f32[1,8,14,14,128]{4,3,2,1,0} %subtract.2786), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.2788 = f32[1,8,14,14,128]{4,3,2,1,0} convert(f32[1,8,14,14,128]{4,3,2,1,0} %multiply.2787), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %constant.2789 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %convert.2790 = f32[] convert(f32[] %constant.2789), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %reduce.2795 = f32[128]{0} reduce(f32[1,8,14,14,128]{4,3,2,1,0} %convert.2788, f32[] %convert.2790), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_4f_Branch_3_Conv3d_0b_1x1_batch_norm_normalize_moments_variance-reduction.2791, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2796 = s32[] get-dimension-size(f32[1,8,14,14,128]{4,3,2,1,0} %convert.2788), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2797 = s32[] get-dimension-size(f32[1,8,14,14,128]{4,3,2,1,0} %convert.2788), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %multiply.2798 = s32[] multiply(s32[] %get-dimension-size.2796, s32[] %get-dimension-size.2797), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2799 = s32[] get-dimension-size(f32[1,8,14,14,128]{4,3,2,1,0} %convert.2788), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %multiply.2800 = s32[] multiply(s32[] %multiply.2798, s32[] %get-dimension-size.2799), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2801 = s32[] get-dimension-size(f32[1,8,14,14,128]{4,3,2,1,0} %convert.2788), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %multiply.2802 = s32[] multiply(s32[] %multiply.2800, s32[] %get-dimension-size.2801), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %convert.2803 = f32[] convert(s32[] %multiply.2802), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.2804 = f32[128]{0} broadcast(f32[] %convert.2803), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %divide.2805 = f32[128]{0} divide(f32[128]{0} %reduce.2795, f32[128]{0} %broadcast.2804), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %convert.2806 = f32[128]{0} convert(f32[128]{0} %divide.2805), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %reshape.2807 = f32[1,1,1,1,128]{4,3,2,1,0} reshape(f32[128]{0} %convert.2806), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %add.2810 = f32[1,1,1,1,128]{4,3,2,1,0} add(f32[1,1,1,1,128]{4,3,2,1,0} %broadcast.2809, f32[1,1,1,1,128]{4,3,2,1,0} %reshape.2807), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %rsqrt.2811 = f32[1,1,1,1,128]{4,3,2,1,0} rsqrt(f32[1,1,1,1,128]{4,3,2,1,0} %add.2810), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.2812 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.2811), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %broadcast.2813 = f32[1,8,14,14,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.2812), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %multiply.2814 = f32[1,8,14,14,128]{4,3,2,1,0} multiply(f32[1,8,14,14,128]{4,3,2,1,0} %broadcast.2813, f32[1,8,14,14,128]{4,3,2,1,0} %convolution.2763), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %arg84.85 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(84), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.2815 = f32[1,1,1,1,128]{4,3,2,1,0} multiply(f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.2811, f32[1,1,1,1,128]{4,3,2,1,0} %reshape.2783), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.2816 = f32[1,1,1,1,128]{4,3,2,1,0} subtract(f32[1,1,1,1,128]{4,3,2,1,0} %arg84.85, f32[1,1,1,1,128]{4,3,2,1,0} %multiply.2815), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/sub"}
  %reshape.2817 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %subtract.2816), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.2818 = f32[1,8,14,14,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.2817), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %add.2819 = f32[1,8,14,14,128]{4,3,2,1,0} add(f32[1,8,14,14,128]{4,3,2,1,0} %multiply.2814, f32[1,8,14,14,128]{4,3,2,1,0} %broadcast.2818), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %concatenate.2820 = f32[1,8,14,14,832]{4,3,2,1,0} concatenate(f32[1,8,14,14,256]{4,3,2,1,0} %add.2522, f32[1,8,14,14,320]{4,3,2,1,0} %add.2639, f32[1,8,14,14,128]{4,3,2,1,0} %add.2756, f32[1,8,14,14,128]{4,3,2,1,0} %add.2819), dimensions={4}, metadata={op_type="ConcatV2" op_name="RGB/inception_i3d/Mixed_4f/concat"}
  %constant.2821 = f32[] constant(-inf), metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/MaxPool3d_5a_2x2"}
  %reduce-window.2826 = f32[1,4,7,7,832]{4,3,2,1,0} reduce-window(f32[1,8,14,14,832]{4,3,2,1,0} %concatenate.2820, f32[] %constant.2821), window={size=1x2x2x2x1 stride=1x2x2x2x1}, to_apply=%max_F32.2822, metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/MaxPool3d_5a_2x2"}
  %maximum.2829 = f32[1,4,7,7,832]{4,3,2,1,0} maximum(f32[1,4,7,7,832]{4,3,2,1,0} %broadcast.2828, f32[1,4,7,7,832]{4,3,2,1,0} %reduce-window.2826), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/Relu"}
  %arg99.100 = f32[1,1,1,832,256]{4,3,2,1,0} parameter(99), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.2830 = f32[1,4,7,7,256]{4,3,2,1,0} convolution(f32[1,4,7,7,832]{4,3,2,1,0} %maximum.2829, f32[1,1,1,832,256]{4,3,2,1,0} %arg99.100), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/conv_3d/convolution"}
  %convert.2831 = f32[1,4,7,7,256]{4,3,2,1,0} convert(f32[1,4,7,7,256]{4,3,2,1,0} %convolution.2830), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %constant.2832 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.2833 = f32[] convert(f32[] %constant.2832), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reduce.2838 = f32[256]{0} reduce(f32[1,4,7,7,256]{4,3,2,1,0} %convert.2831, f32[] %convert.2833), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_5b_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.2834, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2839 = s32[] get-dimension-size(f32[1,4,7,7,256]{4,3,2,1,0} %convert.2831), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2840 = s32[] get-dimension-size(f32[1,4,7,7,256]{4,3,2,1,0} %convert.2831), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.2841 = s32[] multiply(s32[] %get-dimension-size.2839, s32[] %get-dimension-size.2840), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2842 = s32[] get-dimension-size(f32[1,4,7,7,256]{4,3,2,1,0} %convert.2831), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.2843 = s32[] multiply(s32[] %multiply.2841, s32[] %get-dimension-size.2842), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2844 = s32[] get-dimension-size(f32[1,4,7,7,256]{4,3,2,1,0} %convert.2831), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.2845 = s32[] multiply(s32[] %multiply.2843, s32[] %get-dimension-size.2844), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.2846 = f32[] convert(s32[] %multiply.2845), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.2847 = f32[256]{0} broadcast(f32[] %convert.2846), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %divide.2848 = f32[256]{0} divide(f32[256]{0} %reduce.2838, f32[256]{0} %broadcast.2847), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.2849 = f32[256]{0} convert(f32[256]{0} %divide.2848), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.2850 = f32[1,1,1,1,256]{4,3,2,1,0} reshape(f32[256]{0} %convert.2849), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.2851 = f32[1,256]{1,0} reshape(f32[1,1,1,1,256]{4,3,2,1,0} %reshape.2850), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.2852 = f32[1,4,7,7,256]{4,3,2,1,0} broadcast(f32[1,256]{1,0} %reshape.2851), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.2853 = f32[1,4,7,7,256]{4,3,2,1,0} subtract(f32[1,4,7,7,256]{4,3,2,1,0} %broadcast.2852, f32[1,4,7,7,256]{4,3,2,1,0} %convolution.2830), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.2854 = f32[1,4,7,7,256]{4,3,2,1,0} multiply(f32[1,4,7,7,256]{4,3,2,1,0} %subtract.2853, f32[1,4,7,7,256]{4,3,2,1,0} %subtract.2853), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.2855 = f32[1,4,7,7,256]{4,3,2,1,0} convert(f32[1,4,7,7,256]{4,3,2,1,0} %multiply.2854), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %constant.2856 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.2857 = f32[] convert(f32[] %constant.2856), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reduce.2862 = f32[256]{0} reduce(f32[1,4,7,7,256]{4,3,2,1,0} %convert.2855, f32[] %convert.2857), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_5b_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.2858, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2863 = s32[] get-dimension-size(f32[1,4,7,7,256]{4,3,2,1,0} %convert.2855), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2864 = s32[] get-dimension-size(f32[1,4,7,7,256]{4,3,2,1,0} %convert.2855), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.2865 = s32[] multiply(s32[] %get-dimension-size.2863, s32[] %get-dimension-size.2864), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2866 = s32[] get-dimension-size(f32[1,4,7,7,256]{4,3,2,1,0} %convert.2855), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.2867 = s32[] multiply(s32[] %multiply.2865, s32[] %get-dimension-size.2866), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2868 = s32[] get-dimension-size(f32[1,4,7,7,256]{4,3,2,1,0} %convert.2855), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.2869 = s32[] multiply(s32[] %multiply.2867, s32[] %get-dimension-size.2868), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.2870 = f32[] convert(s32[] %multiply.2869), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.2871 = f32[256]{0} broadcast(f32[] %convert.2870), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %divide.2872 = f32[256]{0} divide(f32[256]{0} %reduce.2862, f32[256]{0} %broadcast.2871), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.2873 = f32[256]{0} convert(f32[256]{0} %divide.2872), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reshape.2874 = f32[1,1,1,1,256]{4,3,2,1,0} reshape(f32[256]{0} %convert.2873), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %add.2877 = f32[1,1,1,1,256]{4,3,2,1,0} add(f32[1,1,1,1,256]{4,3,2,1,0} %broadcast.2876, f32[1,1,1,1,256]{4,3,2,1,0} %reshape.2874), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.2878 = f32[1,1,1,1,256]{4,3,2,1,0} rsqrt(f32[1,1,1,1,256]{4,3,2,1,0} %add.2877), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.2879 = f32[1,256]{1,0} reshape(f32[1,1,1,1,256]{4,3,2,1,0} %rsqrt.2878), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.2880 = f32[1,4,7,7,256]{4,3,2,1,0} broadcast(f32[1,256]{1,0} %reshape.2879), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %multiply.2881 = f32[1,4,7,7,256]{4,3,2,1,0} multiply(f32[1,4,7,7,256]{4,3,2,1,0} %broadcast.2880, f32[1,4,7,7,256]{4,3,2,1,0} %convolution.2830), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg109.110 = f32[1,1,1,1,256]{4,3,2,1,0} parameter(109), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.2882 = f32[1,1,1,1,256]{4,3,2,1,0} multiply(f32[1,1,1,1,256]{4,3,2,1,0} %rsqrt.2878, f32[1,1,1,1,256]{4,3,2,1,0} %reshape.2850), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.2883 = f32[1,1,1,1,256]{4,3,2,1,0} subtract(f32[1,1,1,1,256]{4,3,2,1,0} %arg109.110, f32[1,1,1,1,256]{4,3,2,1,0} %multiply.2882), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.2884 = f32[1,256]{1,0} reshape(f32[1,1,1,1,256]{4,3,2,1,0} %subtract.2883), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.2885 = f32[1,4,7,7,256]{4,3,2,1,0} broadcast(f32[1,256]{1,0} %reshape.2884), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.2886 = f32[1,4,7,7,256]{4,3,2,1,0} add(f32[1,4,7,7,256]{4,3,2,1,0} %multiply.2881, f32[1,4,7,7,256]{4,3,2,1,0} %broadcast.2885), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %constant.2992 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %broadcast.2993 = f32[1,1,1,1,320]{4,3,2,1,0} broadcast(f32[] %constant.2992), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %constant.2944 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/Relu"}
  %broadcast.2945 = f32[1,4,7,7,160]{4,3,2,1,0} broadcast(f32[] %constant.2944), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/Relu"}
  %constant.2932 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.2933 = f32[1,1,1,1,160]{4,3,2,1,0} broadcast(f32[] %constant.2932), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %arg3.4 = f32[1,1,1,832,160]{4,3,2,1,0} parameter(3), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.2887 = f32[1,4,7,7,160]{4,3,2,1,0} convolution(f32[1,4,7,7,832]{4,3,2,1,0} %maximum.2829, f32[1,1,1,832,160]{4,3,2,1,0} %arg3.4), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/conv_3d/convolution"}
  %convert.2888 = f32[1,4,7,7,160]{4,3,2,1,0} convert(f32[1,4,7,7,160]{4,3,2,1,0} %convolution.2887), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %constant.2889 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.2890 = f32[] convert(f32[] %constant.2889), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reduce.2895 = f32[160]{0} reduce(f32[1,4,7,7,160]{4,3,2,1,0} %convert.2888, f32[] %convert.2890), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_5b_Branch_1_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.2891, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2896 = s32[] get-dimension-size(f32[1,4,7,7,160]{4,3,2,1,0} %convert.2888), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2897 = s32[] get-dimension-size(f32[1,4,7,7,160]{4,3,2,1,0} %convert.2888), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.2898 = s32[] multiply(s32[] %get-dimension-size.2896, s32[] %get-dimension-size.2897), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2899 = s32[] get-dimension-size(f32[1,4,7,7,160]{4,3,2,1,0} %convert.2888), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.2900 = s32[] multiply(s32[] %multiply.2898, s32[] %get-dimension-size.2899), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2901 = s32[] get-dimension-size(f32[1,4,7,7,160]{4,3,2,1,0} %convert.2888), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.2902 = s32[] multiply(s32[] %multiply.2900, s32[] %get-dimension-size.2901), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.2903 = f32[] convert(s32[] %multiply.2902), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.2904 = f32[160]{0} broadcast(f32[] %convert.2903), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %divide.2905 = f32[160]{0} divide(f32[160]{0} %reduce.2895, f32[160]{0} %broadcast.2904), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.2906 = f32[160]{0} convert(f32[160]{0} %divide.2905), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.2907 = f32[1,1,1,1,160]{4,3,2,1,0} reshape(f32[160]{0} %convert.2906), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.2908 = f32[1,160]{1,0} reshape(f32[1,1,1,1,160]{4,3,2,1,0} %reshape.2907), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.2909 = f32[1,4,7,7,160]{4,3,2,1,0} broadcast(f32[1,160]{1,0} %reshape.2908), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.2910 = f32[1,4,7,7,160]{4,3,2,1,0} subtract(f32[1,4,7,7,160]{4,3,2,1,0} %broadcast.2909, f32[1,4,7,7,160]{4,3,2,1,0} %convolution.2887), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.2911 = f32[1,4,7,7,160]{4,3,2,1,0} multiply(f32[1,4,7,7,160]{4,3,2,1,0} %subtract.2910, f32[1,4,7,7,160]{4,3,2,1,0} %subtract.2910), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.2912 = f32[1,4,7,7,160]{4,3,2,1,0} convert(f32[1,4,7,7,160]{4,3,2,1,0} %multiply.2911), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %constant.2913 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.2914 = f32[] convert(f32[] %constant.2913), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reduce.2919 = f32[160]{0} reduce(f32[1,4,7,7,160]{4,3,2,1,0} %convert.2912, f32[] %convert.2914), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_5b_Branch_1_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.2915, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2920 = s32[] get-dimension-size(f32[1,4,7,7,160]{4,3,2,1,0} %convert.2912), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2921 = s32[] get-dimension-size(f32[1,4,7,7,160]{4,3,2,1,0} %convert.2912), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.2922 = s32[] multiply(s32[] %get-dimension-size.2920, s32[] %get-dimension-size.2921), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2923 = s32[] get-dimension-size(f32[1,4,7,7,160]{4,3,2,1,0} %convert.2912), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.2924 = s32[] multiply(s32[] %multiply.2922, s32[] %get-dimension-size.2923), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2925 = s32[] get-dimension-size(f32[1,4,7,7,160]{4,3,2,1,0} %convert.2912), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.2926 = s32[] multiply(s32[] %multiply.2924, s32[] %get-dimension-size.2925), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.2927 = f32[] convert(s32[] %multiply.2926), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.2928 = f32[160]{0} broadcast(f32[] %convert.2927), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %divide.2929 = f32[160]{0} divide(f32[160]{0} %reduce.2919, f32[160]{0} %broadcast.2928), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.2930 = f32[160]{0} convert(f32[160]{0} %divide.2929), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reshape.2931 = f32[1,1,1,1,160]{4,3,2,1,0} reshape(f32[160]{0} %convert.2930), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %add.2934 = f32[1,1,1,1,160]{4,3,2,1,0} add(f32[1,1,1,1,160]{4,3,2,1,0} %broadcast.2933, f32[1,1,1,1,160]{4,3,2,1,0} %reshape.2931), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.2935 = f32[1,1,1,1,160]{4,3,2,1,0} rsqrt(f32[1,1,1,1,160]{4,3,2,1,0} %add.2934), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.2936 = f32[1,160]{1,0} reshape(f32[1,1,1,1,160]{4,3,2,1,0} %rsqrt.2935), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.2937 = f32[1,4,7,7,160]{4,3,2,1,0} broadcast(f32[1,160]{1,0} %reshape.2936), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %multiply.2938 = f32[1,4,7,7,160]{4,3,2,1,0} multiply(f32[1,4,7,7,160]{4,3,2,1,0} %broadcast.2937, f32[1,4,7,7,160]{4,3,2,1,0} %convolution.2887), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg17.18 = f32[1,1,1,1,160]{4,3,2,1,0} parameter(17), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.2939 = f32[1,1,1,1,160]{4,3,2,1,0} multiply(f32[1,1,1,1,160]{4,3,2,1,0} %rsqrt.2935, f32[1,1,1,1,160]{4,3,2,1,0} %reshape.2907), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.2940 = f32[1,1,1,1,160]{4,3,2,1,0} subtract(f32[1,1,1,1,160]{4,3,2,1,0} %arg17.18, f32[1,1,1,1,160]{4,3,2,1,0} %multiply.2939), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.2941 = f32[1,160]{1,0} reshape(f32[1,1,1,1,160]{4,3,2,1,0} %subtract.2940), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.2942 = f32[1,4,7,7,160]{4,3,2,1,0} broadcast(f32[1,160]{1,0} %reshape.2941), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.2943 = f32[1,4,7,7,160]{4,3,2,1,0} add(f32[1,4,7,7,160]{4,3,2,1,0} %multiply.2938, f32[1,4,7,7,160]{4,3,2,1,0} %broadcast.2942), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %maximum.2946 = f32[1,4,7,7,160]{4,3,2,1,0} maximum(f32[1,4,7,7,160]{4,3,2,1,0} %broadcast.2945, f32[1,4,7,7,160]{4,3,2,1,0} %add.2943), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/Relu"}
  %arg35.36 = f32[3,3,3,160,320]{4,3,2,1,0} parameter(35), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.2947 = f32[1,4,7,7,320]{4,3,2,1,0} convolution(f32[1,4,7,7,160]{4,3,2,1,0} %maximum.2946, f32[3,3,3,160,320]{4,3,2,1,0} %arg35.36), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/conv_3d/convolution"}
  %convert.2948 = f32[1,4,7,7,320]{4,3,2,1,0} convert(f32[1,4,7,7,320]{4,3,2,1,0} %convolution.2947), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %constant.2949 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.2950 = f32[] convert(f32[] %constant.2949), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reduce.2955 = f32[320]{0} reduce(f32[1,4,7,7,320]{4,3,2,1,0} %convert.2948, f32[] %convert.2950), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_5b_Branch_1_Conv3d_0b_3x3_batch_norm_normalize_moments_mean-reduction.2951, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2956 = s32[] get-dimension-size(f32[1,4,7,7,320]{4,3,2,1,0} %convert.2948), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2957 = s32[] get-dimension-size(f32[1,4,7,7,320]{4,3,2,1,0} %convert.2948), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.2958 = s32[] multiply(s32[] %get-dimension-size.2956, s32[] %get-dimension-size.2957), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2959 = s32[] get-dimension-size(f32[1,4,7,7,320]{4,3,2,1,0} %convert.2948), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.2960 = s32[] multiply(s32[] %multiply.2958, s32[] %get-dimension-size.2959), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.2961 = s32[] get-dimension-size(f32[1,4,7,7,320]{4,3,2,1,0} %convert.2948), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.2962 = s32[] multiply(s32[] %multiply.2960, s32[] %get-dimension-size.2961), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.2963 = f32[] convert(s32[] %multiply.2962), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %broadcast.2964 = f32[320]{0} broadcast(f32[] %convert.2963), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %divide.2965 = f32[320]{0} divide(f32[320]{0} %reduce.2955, f32[320]{0} %broadcast.2964), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.2966 = f32[320]{0} convert(f32[320]{0} %divide.2965), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reshape.2967 = f32[1,1,1,1,320]{4,3,2,1,0} reshape(f32[320]{0} %convert.2966), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reshape.2968 = f32[1,320]{1,0} reshape(f32[1,1,1,1,320]{4,3,2,1,0} %reshape.2967), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.2969 = f32[1,4,7,7,320]{4,3,2,1,0} broadcast(f32[1,320]{1,0} %reshape.2968), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.2970 = f32[1,4,7,7,320]{4,3,2,1,0} subtract(f32[1,4,7,7,320]{4,3,2,1,0} %broadcast.2969, f32[1,4,7,7,320]{4,3,2,1,0} %convolution.2947), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.2971 = f32[1,4,7,7,320]{4,3,2,1,0} multiply(f32[1,4,7,7,320]{4,3,2,1,0} %subtract.2970, f32[1,4,7,7,320]{4,3,2,1,0} %subtract.2970), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %convert.2972 = f32[1,4,7,7,320]{4,3,2,1,0} convert(f32[1,4,7,7,320]{4,3,2,1,0} %multiply.2971), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %constant.2973 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.2974 = f32[] convert(f32[] %constant.2973), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %reduce.2979 = f32[320]{0} reduce(f32[1,4,7,7,320]{4,3,2,1,0} %convert.2972, f32[] %convert.2974), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_5b_Branch_1_Conv3d_0b_3x3_batch_norm_normalize_moments_variance-reduction.2975, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2980 = s32[] get-dimension-size(f32[1,4,7,7,320]{4,3,2,1,0} %convert.2972), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2981 = s32[] get-dimension-size(f32[1,4,7,7,320]{4,3,2,1,0} %convert.2972), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.2982 = s32[] multiply(s32[] %get-dimension-size.2980, s32[] %get-dimension-size.2981), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2983 = s32[] get-dimension-size(f32[1,4,7,7,320]{4,3,2,1,0} %convert.2972), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.2984 = s32[] multiply(s32[] %multiply.2982, s32[] %get-dimension-size.2983), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.2985 = s32[] get-dimension-size(f32[1,4,7,7,320]{4,3,2,1,0} %convert.2972), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.2986 = s32[] multiply(s32[] %multiply.2984, s32[] %get-dimension-size.2985), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.2987 = f32[] convert(s32[] %multiply.2986), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %broadcast.2988 = f32[320]{0} broadcast(f32[] %convert.2987), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %divide.2989 = f32[320]{0} divide(f32[320]{0} %reduce.2979, f32[320]{0} %broadcast.2988), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.2990 = f32[320]{0} convert(f32[320]{0} %divide.2989), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %reshape.2991 = f32[1,1,1,1,320]{4,3,2,1,0} reshape(f32[320]{0} %convert.2990), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %add.2994 = f32[1,1,1,1,320]{4,3,2,1,0} add(f32[1,1,1,1,320]{4,3,2,1,0} %broadcast.2993, f32[1,1,1,1,320]{4,3,2,1,0} %reshape.2991), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %rsqrt.2995 = f32[1,1,1,1,320]{4,3,2,1,0} rsqrt(f32[1,1,1,1,320]{4,3,2,1,0} %add.2994), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.2996 = f32[1,320]{1,0} reshape(f32[1,1,1,1,320]{4,3,2,1,0} %rsqrt.2995), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %broadcast.2997 = f32[1,4,7,7,320]{4,3,2,1,0} broadcast(f32[1,320]{1,0} %reshape.2996), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %multiply.2998 = f32[1,4,7,7,320]{4,3,2,1,0} multiply(f32[1,4,7,7,320]{4,3,2,1,0} %broadcast.2997, f32[1,4,7,7,320]{4,3,2,1,0} %convolution.2947), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %arg46.47 = f32[1,1,1,1,320]{4,3,2,1,0} parameter(46), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.2999 = f32[1,1,1,1,320]{4,3,2,1,0} multiply(f32[1,1,1,1,320]{4,3,2,1,0} %rsqrt.2995, f32[1,1,1,1,320]{4,3,2,1,0} %reshape.2967), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.3000 = f32[1,1,1,1,320]{4,3,2,1,0} subtract(f32[1,1,1,1,320]{4,3,2,1,0} %arg46.47, f32[1,1,1,1,320]{4,3,2,1,0} %multiply.2999), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/sub"}
  %reshape.3001 = f32[1,320]{1,0} reshape(f32[1,1,1,1,320]{4,3,2,1,0} %subtract.3000), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.3002 = f32[1,4,7,7,320]{4,3,2,1,0} broadcast(f32[1,320]{1,0} %reshape.3001), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %add.3003 = f32[1,4,7,7,320]{4,3,2,1,0} add(f32[1,4,7,7,320]{4,3,2,1,0} %multiply.2998, f32[1,4,7,7,320]{4,3,2,1,0} %broadcast.3002), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %constant.3109 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/batch_norm/add"}
  %broadcast.3110 = f32[1,1,1,1,128]{4,3,2,1,0} broadcast(f32[] %constant.3109), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/batch_norm/add"}
  %constant.3061 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/Relu"}
  %broadcast.3062 = f32[1,4,7,7,32]{4,3,2,1,0} broadcast(f32[] %constant.3061), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/Relu"}
  %constant.3049 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.3050 = f32[1,1,1,1,32]{4,3,2,1,0} broadcast(f32[] %constant.3049), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %arg63.64 = f32[1,1,1,832,32]{4,3,2,1,0} parameter(63), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.3004 = f32[1,4,7,7,32]{4,3,2,1,0} convolution(f32[1,4,7,7,832]{4,3,2,1,0} %maximum.2829, f32[1,1,1,832,32]{4,3,2,1,0} %arg63.64), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/conv_3d/convolution"}
  %convert.3005 = f32[1,4,7,7,32]{4,3,2,1,0} convert(f32[1,4,7,7,32]{4,3,2,1,0} %convolution.3004), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %constant.3006 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.3007 = f32[] convert(f32[] %constant.3006), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reduce.3012 = f32[32]{0} reduce(f32[1,4,7,7,32]{4,3,2,1,0} %convert.3005, f32[] %convert.3007), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_5b_Branch_2_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.3008, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.3013 = s32[] get-dimension-size(f32[1,4,7,7,32]{4,3,2,1,0} %convert.3005), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.3014 = s32[] get-dimension-size(f32[1,4,7,7,32]{4,3,2,1,0} %convert.3005), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.3015 = s32[] multiply(s32[] %get-dimension-size.3013, s32[] %get-dimension-size.3014), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.3016 = s32[] get-dimension-size(f32[1,4,7,7,32]{4,3,2,1,0} %convert.3005), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.3017 = s32[] multiply(s32[] %multiply.3015, s32[] %get-dimension-size.3016), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.3018 = s32[] get-dimension-size(f32[1,4,7,7,32]{4,3,2,1,0} %convert.3005), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.3019 = s32[] multiply(s32[] %multiply.3017, s32[] %get-dimension-size.3018), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.3020 = f32[] convert(s32[] %multiply.3019), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.3021 = f32[32]{0} broadcast(f32[] %convert.3020), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %divide.3022 = f32[32]{0} divide(f32[32]{0} %reduce.3012, f32[32]{0} %broadcast.3021), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.3023 = f32[32]{0} convert(f32[32]{0} %divide.3022), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.3024 = f32[1,1,1,1,32]{4,3,2,1,0} reshape(f32[32]{0} %convert.3023), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.3025 = f32[1,32]{1,0} reshape(f32[1,1,1,1,32]{4,3,2,1,0} %reshape.3024), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.3026 = f32[1,4,7,7,32]{4,3,2,1,0} broadcast(f32[1,32]{1,0} %reshape.3025), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.3027 = f32[1,4,7,7,32]{4,3,2,1,0} subtract(f32[1,4,7,7,32]{4,3,2,1,0} %broadcast.3026, f32[1,4,7,7,32]{4,3,2,1,0} %convolution.3004), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.3028 = f32[1,4,7,7,32]{4,3,2,1,0} multiply(f32[1,4,7,7,32]{4,3,2,1,0} %subtract.3027, f32[1,4,7,7,32]{4,3,2,1,0} %subtract.3027), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.3029 = f32[1,4,7,7,32]{4,3,2,1,0} convert(f32[1,4,7,7,32]{4,3,2,1,0} %multiply.3028), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %constant.3030 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.3031 = f32[] convert(f32[] %constant.3030), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reduce.3036 = f32[32]{0} reduce(f32[1,4,7,7,32]{4,3,2,1,0} %convert.3029, f32[] %convert.3031), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_5b_Branch_2_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.3032, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.3037 = s32[] get-dimension-size(f32[1,4,7,7,32]{4,3,2,1,0} %convert.3029), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.3038 = s32[] get-dimension-size(f32[1,4,7,7,32]{4,3,2,1,0} %convert.3029), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.3039 = s32[] multiply(s32[] %get-dimension-size.3037, s32[] %get-dimension-size.3038), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.3040 = s32[] get-dimension-size(f32[1,4,7,7,32]{4,3,2,1,0} %convert.3029), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.3041 = s32[] multiply(s32[] %multiply.3039, s32[] %get-dimension-size.3040), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.3042 = s32[] get-dimension-size(f32[1,4,7,7,32]{4,3,2,1,0} %convert.3029), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.3043 = s32[] multiply(s32[] %multiply.3041, s32[] %get-dimension-size.3042), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.3044 = f32[] convert(s32[] %multiply.3043), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.3045 = f32[32]{0} broadcast(f32[] %convert.3044), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %divide.3046 = f32[32]{0} divide(f32[32]{0} %reduce.3036, f32[32]{0} %broadcast.3045), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.3047 = f32[32]{0} convert(f32[32]{0} %divide.3046), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reshape.3048 = f32[1,1,1,1,32]{4,3,2,1,0} reshape(f32[32]{0} %convert.3047), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %add.3051 = f32[1,1,1,1,32]{4,3,2,1,0} add(f32[1,1,1,1,32]{4,3,2,1,0} %broadcast.3050, f32[1,1,1,1,32]{4,3,2,1,0} %reshape.3048), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.3052 = f32[1,1,1,1,32]{4,3,2,1,0} rsqrt(f32[1,1,1,1,32]{4,3,2,1,0} %add.3051), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.3053 = f32[1,32]{1,0} reshape(f32[1,1,1,1,32]{4,3,2,1,0} %rsqrt.3052), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.3054 = f32[1,4,7,7,32]{4,3,2,1,0} broadcast(f32[1,32]{1,0} %reshape.3053), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %multiply.3055 = f32[1,4,7,7,32]{4,3,2,1,0} multiply(f32[1,4,7,7,32]{4,3,2,1,0} %broadcast.3054, f32[1,4,7,7,32]{4,3,2,1,0} %convolution.3004), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg75.76 = f32[1,1,1,1,32]{4,3,2,1,0} parameter(75), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.3056 = f32[1,1,1,1,32]{4,3,2,1,0} multiply(f32[1,1,1,1,32]{4,3,2,1,0} %rsqrt.3052, f32[1,1,1,1,32]{4,3,2,1,0} %reshape.3024), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.3057 = f32[1,1,1,1,32]{4,3,2,1,0} subtract(f32[1,1,1,1,32]{4,3,2,1,0} %arg75.76, f32[1,1,1,1,32]{4,3,2,1,0} %multiply.3056), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.3058 = f32[1,32]{1,0} reshape(f32[1,1,1,1,32]{4,3,2,1,0} %subtract.3057), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.3059 = f32[1,4,7,7,32]{4,3,2,1,0} broadcast(f32[1,32]{1,0} %reshape.3058), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.3060 = f32[1,4,7,7,32]{4,3,2,1,0} add(f32[1,4,7,7,32]{4,3,2,1,0} %multiply.3055, f32[1,4,7,7,32]{4,3,2,1,0} %broadcast.3059), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %maximum.3063 = f32[1,4,7,7,32]{4,3,2,1,0} maximum(f32[1,4,7,7,32]{4,3,2,1,0} %broadcast.3062, f32[1,4,7,7,32]{4,3,2,1,0} %add.3060), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/Relu"}
  %arg94.95 = f32[3,3,3,32,128]{4,3,2,1,0} parameter(94), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.3064 = f32[1,4,7,7,128]{4,3,2,1,0} convolution(f32[1,4,7,7,32]{4,3,2,1,0} %maximum.3063, f32[3,3,3,32,128]{4,3,2,1,0} %arg94.95), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/conv_3d/convolution"}
  %convert.3065 = f32[1,4,7,7,128]{4,3,2,1,0} convert(f32[1,4,7,7,128]{4,3,2,1,0} %convolution.3064), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/normalize_moments/mean"}
  %constant.3066 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/normalize_moments/mean"}
  %convert.3067 = f32[] convert(f32[] %constant.3066), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/normalize_moments/mean"}
  %reduce.3072 = f32[128]{0} reduce(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3065, f32[] %convert.3067), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_5b_Branch_2_Conv3d_0a_3x3_batch_norm_normalize_moments_mean-reduction.3068, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.3073 = s32[] get-dimension-size(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3065), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.3074 = s32[] get-dimension-size(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3065), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/normalize_moments/mean"}
  %multiply.3075 = s32[] multiply(s32[] %get-dimension-size.3073, s32[] %get-dimension-size.3074), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.3076 = s32[] get-dimension-size(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3065), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/normalize_moments/mean"}
  %multiply.3077 = s32[] multiply(s32[] %multiply.3075, s32[] %get-dimension-size.3076), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.3078 = s32[] get-dimension-size(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3065), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/normalize_moments/mean"}
  %multiply.3079 = s32[] multiply(s32[] %multiply.3077, s32[] %get-dimension-size.3078), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/normalize_moments/mean"}
  %convert.3080 = f32[] convert(s32[] %multiply.3079), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/normalize_moments/mean"}
  %broadcast.3081 = f32[128]{0} broadcast(f32[] %convert.3080), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/normalize_moments/mean"}
  %divide.3082 = f32[128]{0} divide(f32[128]{0} %reduce.3072, f32[128]{0} %broadcast.3081), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/normalize_moments/mean"}
  %convert.3083 = f32[128]{0} convert(f32[128]{0} %divide.3082), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/normalize_moments/mean"}
  %reshape.3084 = f32[1,1,1,1,128]{4,3,2,1,0} reshape(f32[128]{0} %convert.3083), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/normalize_moments/mean"}
  %reshape.3085 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %reshape.3084), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.3086 = f32[1,4,7,7,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.3085), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.3087 = f32[1,4,7,7,128]{4,3,2,1,0} subtract(f32[1,4,7,7,128]{4,3,2,1,0} %broadcast.3086, f32[1,4,7,7,128]{4,3,2,1,0} %convolution.3064), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.3088 = f32[1,4,7,7,128]{4,3,2,1,0} multiply(f32[1,4,7,7,128]{4,3,2,1,0} %subtract.3087, f32[1,4,7,7,128]{4,3,2,1,0} %subtract.3087), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %convert.3089 = f32[1,4,7,7,128]{4,3,2,1,0} convert(f32[1,4,7,7,128]{4,3,2,1,0} %multiply.3088), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/normalize_moments/variance"}
  %constant.3090 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/normalize_moments/variance"}
  %convert.3091 = f32[] convert(f32[] %constant.3090), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/normalize_moments/variance"}
  %reduce.3096 = f32[128]{0} reduce(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3089, f32[] %convert.3091), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_5b_Branch_2_Conv3d_0a_3x3_batch_norm_normalize_moments_variance-reduction.3092, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.3097 = s32[] get-dimension-size(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3089), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.3098 = s32[] get-dimension-size(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3089), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/normalize_moments/variance"}
  %multiply.3099 = s32[] multiply(s32[] %get-dimension-size.3097, s32[] %get-dimension-size.3098), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.3100 = s32[] get-dimension-size(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3089), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/normalize_moments/variance"}
  %multiply.3101 = s32[] multiply(s32[] %multiply.3099, s32[] %get-dimension-size.3100), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.3102 = s32[] get-dimension-size(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3089), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/normalize_moments/variance"}
  %multiply.3103 = s32[] multiply(s32[] %multiply.3101, s32[] %get-dimension-size.3102), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/normalize_moments/variance"}
  %convert.3104 = f32[] convert(s32[] %multiply.3103), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/normalize_moments/variance"}
  %broadcast.3105 = f32[128]{0} broadcast(f32[] %convert.3104), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/normalize_moments/variance"}
  %divide.3106 = f32[128]{0} divide(f32[128]{0} %reduce.3096, f32[128]{0} %broadcast.3105), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/normalize_moments/variance"}
  %convert.3107 = f32[128]{0} convert(f32[128]{0} %divide.3106), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/normalize_moments/variance"}
  %reshape.3108 = f32[1,1,1,1,128]{4,3,2,1,0} reshape(f32[128]{0} %convert.3107), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/normalize_moments/variance"}
  %add.3111 = f32[1,1,1,1,128]{4,3,2,1,0} add(f32[1,1,1,1,128]{4,3,2,1,0} %broadcast.3110, f32[1,1,1,1,128]{4,3,2,1,0} %reshape.3108), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/batch_norm/add"}
  %rsqrt.3112 = f32[1,1,1,1,128]{4,3,2,1,0} rsqrt(f32[1,1,1,1,128]{4,3,2,1,0} %add.3111), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.3113 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.3112), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/batch_norm/mul"}
  %broadcast.3114 = f32[1,4,7,7,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.3113), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/batch_norm/mul"}
  %multiply.3115 = f32[1,4,7,7,128]{4,3,2,1,0} multiply(f32[1,4,7,7,128]{4,3,2,1,0} %broadcast.3114, f32[1,4,7,7,128]{4,3,2,1,0} %convolution.3064), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/batch_norm/mul"}
  %arg108.109 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(108), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.3116 = f32[1,1,1,1,128]{4,3,2,1,0} multiply(f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.3112, f32[1,1,1,1,128]{4,3,2,1,0} %reshape.3084), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.3117 = f32[1,1,1,1,128]{4,3,2,1,0} subtract(f32[1,1,1,1,128]{4,3,2,1,0} %arg108.109, f32[1,1,1,1,128]{4,3,2,1,0} %multiply.3116), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/batch_norm/sub"}
  %reshape.3118 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %subtract.3117), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.3119 = f32[1,4,7,7,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.3118), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/batch_norm/add_1"}
  %add.3120 = f32[1,4,7,7,128]{4,3,2,1,0} add(f32[1,4,7,7,128]{4,3,2,1,0} %multiply.3115, f32[1,4,7,7,128]{4,3,2,1,0} %broadcast.3119), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/batch_norm/add_1"}
  %constant.3172 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %broadcast.3173 = f32[1,1,1,1,128]{4,3,2,1,0} broadcast(f32[] %constant.3172), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %constant.3121 = f32[] constant(-inf), metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/MaxPool3d_0a_3x3"}
  %reduce-window.3126 = f32[1,4,7,7,832]{4,3,2,1,0} reduce-window(f32[1,4,7,7,832]{4,3,2,1,0} %maximum.2829, f32[] %constant.3121), window={size=1x3x3x3x1 pad=0_0x1_1x1_1x1_1x0_0}, to_apply=%max_F32.3122, metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/MaxPool3d_0a_3x3"}
  %arg15.16 = f32[1,1,1,832,128]{4,3,2,1,0} parameter(15), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.3127 = f32[1,4,7,7,128]{4,3,2,1,0} convolution(f32[1,4,7,7,832]{4,3,2,1,0} %reduce-window.3126, f32[1,1,1,832,128]{4,3,2,1,0} %arg15.16), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/conv_3d/convolution"}
  %convert.3128 = f32[1,4,7,7,128]{4,3,2,1,0} convert(f32[1,4,7,7,128]{4,3,2,1,0} %convolution.3127), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %constant.3129 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %convert.3130 = f32[] convert(f32[] %constant.3129), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %reduce.3135 = f32[128]{0} reduce(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3128, f32[] %convert.3130), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_5b_Branch_3_Conv3d_0b_1x1_batch_norm_normalize_moments_mean-reduction.3131, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.3136 = s32[] get-dimension-size(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3128), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.3137 = s32[] get-dimension-size(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3128), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %multiply.3138 = s32[] multiply(s32[] %get-dimension-size.3136, s32[] %get-dimension-size.3137), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.3139 = s32[] get-dimension-size(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3128), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %multiply.3140 = s32[] multiply(s32[] %multiply.3138, s32[] %get-dimension-size.3139), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.3141 = s32[] get-dimension-size(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3128), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %multiply.3142 = s32[] multiply(s32[] %multiply.3140, s32[] %get-dimension-size.3141), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %convert.3143 = f32[] convert(s32[] %multiply.3142), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.3144 = f32[128]{0} broadcast(f32[] %convert.3143), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %divide.3145 = f32[128]{0} divide(f32[128]{0} %reduce.3135, f32[128]{0} %broadcast.3144), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %convert.3146 = f32[128]{0} convert(f32[128]{0} %divide.3145), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %reshape.3147 = f32[1,1,1,1,128]{4,3,2,1,0} reshape(f32[128]{0} %convert.3146), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %reshape.3148 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %reshape.3147), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.3149 = f32[1,4,7,7,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.3148), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.3150 = f32[1,4,7,7,128]{4,3,2,1,0} subtract(f32[1,4,7,7,128]{4,3,2,1,0} %broadcast.3149, f32[1,4,7,7,128]{4,3,2,1,0} %convolution.3127), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.3151 = f32[1,4,7,7,128]{4,3,2,1,0} multiply(f32[1,4,7,7,128]{4,3,2,1,0} %subtract.3150, f32[1,4,7,7,128]{4,3,2,1,0} %subtract.3150), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.3152 = f32[1,4,7,7,128]{4,3,2,1,0} convert(f32[1,4,7,7,128]{4,3,2,1,0} %multiply.3151), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %constant.3153 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %convert.3154 = f32[] convert(f32[] %constant.3153), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %reduce.3159 = f32[128]{0} reduce(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3152, f32[] %convert.3154), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_5b_Branch_3_Conv3d_0b_1x1_batch_norm_normalize_moments_variance-reduction.3155, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.3160 = s32[] get-dimension-size(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3152), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.3161 = s32[] get-dimension-size(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3152), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %multiply.3162 = s32[] multiply(s32[] %get-dimension-size.3160, s32[] %get-dimension-size.3161), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.3163 = s32[] get-dimension-size(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3152), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %multiply.3164 = s32[] multiply(s32[] %multiply.3162, s32[] %get-dimension-size.3163), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.3165 = s32[] get-dimension-size(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3152), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %multiply.3166 = s32[] multiply(s32[] %multiply.3164, s32[] %get-dimension-size.3165), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %convert.3167 = f32[] convert(s32[] %multiply.3166), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.3168 = f32[128]{0} broadcast(f32[] %convert.3167), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %divide.3169 = f32[128]{0} divide(f32[128]{0} %reduce.3159, f32[128]{0} %broadcast.3168), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %convert.3170 = f32[128]{0} convert(f32[128]{0} %divide.3169), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %reshape.3171 = f32[1,1,1,1,128]{4,3,2,1,0} reshape(f32[128]{0} %convert.3170), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %add.3174 = f32[1,1,1,1,128]{4,3,2,1,0} add(f32[1,1,1,1,128]{4,3,2,1,0} %broadcast.3173, f32[1,1,1,1,128]{4,3,2,1,0} %reshape.3171), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %rsqrt.3175 = f32[1,1,1,1,128]{4,3,2,1,0} rsqrt(f32[1,1,1,1,128]{4,3,2,1,0} %add.3174), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.3176 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.3175), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %broadcast.3177 = f32[1,4,7,7,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.3176), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %multiply.3178 = f32[1,4,7,7,128]{4,3,2,1,0} multiply(f32[1,4,7,7,128]{4,3,2,1,0} %broadcast.3177, f32[1,4,7,7,128]{4,3,2,1,0} %convolution.3127), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %arg32.33 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(32), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.3179 = f32[1,1,1,1,128]{4,3,2,1,0} multiply(f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.3175, f32[1,1,1,1,128]{4,3,2,1,0} %reshape.3147), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.3180 = f32[1,1,1,1,128]{4,3,2,1,0} subtract(f32[1,1,1,1,128]{4,3,2,1,0} %arg32.33, f32[1,1,1,1,128]{4,3,2,1,0} %multiply.3179), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/sub"}
  %reshape.3181 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %subtract.3180), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.3182 = f32[1,4,7,7,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.3181), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %add.3183 = f32[1,4,7,7,128]{4,3,2,1,0} add(f32[1,4,7,7,128]{4,3,2,1,0} %multiply.3178, f32[1,4,7,7,128]{4,3,2,1,0} %broadcast.3182), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %concatenate.3184 = f32[1,4,7,7,832]{4,3,2,1,0} concatenate(f32[1,4,7,7,256]{4,3,2,1,0} %add.2886, f32[1,4,7,7,320]{4,3,2,1,0} %add.3003, f32[1,4,7,7,128]{4,3,2,1,0} %add.3120, f32[1,4,7,7,128]{4,3,2,1,0} %add.3183), dimensions={4}, metadata={op_type="ConcatV2" op_name="RGB/inception_i3d/Mixed_5b/concat"}
  %maximum.3187 = f32[1,4,7,7,832]{4,3,2,1,0} maximum(f32[1,4,7,7,832]{4,3,2,1,0} %broadcast.3186, f32[1,4,7,7,832]{4,3,2,1,0} %concatenate.3184), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/Relu"}
  %arg57.58 = f32[1,1,1,832,384]{4,3,2,1,0} parameter(57), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.3188 = f32[1,4,7,7,384]{4,3,2,1,0} convolution(f32[1,4,7,7,832]{4,3,2,1,0} %maximum.3187, f32[1,1,1,832,384]{4,3,2,1,0} %arg57.58), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/conv_3d/convolution"}
  %convert.3189 = f32[1,4,7,7,384]{4,3,2,1,0} convert(f32[1,4,7,7,384]{4,3,2,1,0} %convolution.3188), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %constant.3190 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.3191 = f32[] convert(f32[] %constant.3190), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reduce.3196 = f32[384]{0} reduce(f32[1,4,7,7,384]{4,3,2,1,0} %convert.3189, f32[] %convert.3191), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_5c_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.3192, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.3197 = s32[] get-dimension-size(f32[1,4,7,7,384]{4,3,2,1,0} %convert.3189), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.3198 = s32[] get-dimension-size(f32[1,4,7,7,384]{4,3,2,1,0} %convert.3189), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.3199 = s32[] multiply(s32[] %get-dimension-size.3197, s32[] %get-dimension-size.3198), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.3200 = s32[] get-dimension-size(f32[1,4,7,7,384]{4,3,2,1,0} %convert.3189), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.3201 = s32[] multiply(s32[] %multiply.3199, s32[] %get-dimension-size.3200), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.3202 = s32[] get-dimension-size(f32[1,4,7,7,384]{4,3,2,1,0} %convert.3189), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.3203 = s32[] multiply(s32[] %multiply.3201, s32[] %get-dimension-size.3202), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.3204 = f32[] convert(s32[] %multiply.3203), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.3205 = f32[384]{0} broadcast(f32[] %convert.3204), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %divide.3206 = f32[384]{0} divide(f32[384]{0} %reduce.3196, f32[384]{0} %broadcast.3205), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.3207 = f32[384]{0} convert(f32[384]{0} %divide.3206), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.3208 = f32[1,1,1,1,384]{4,3,2,1,0} reshape(f32[384]{0} %convert.3207), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.3209 = f32[1,384]{1,0} reshape(f32[1,1,1,1,384]{4,3,2,1,0} %reshape.3208), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.3210 = f32[1,4,7,7,384]{4,3,2,1,0} broadcast(f32[1,384]{1,0} %reshape.3209), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.3211 = f32[1,4,7,7,384]{4,3,2,1,0} subtract(f32[1,4,7,7,384]{4,3,2,1,0} %broadcast.3210, f32[1,4,7,7,384]{4,3,2,1,0} %convolution.3188), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.3212 = f32[1,4,7,7,384]{4,3,2,1,0} multiply(f32[1,4,7,7,384]{4,3,2,1,0} %subtract.3211, f32[1,4,7,7,384]{4,3,2,1,0} %subtract.3211), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.3213 = f32[1,4,7,7,384]{4,3,2,1,0} convert(f32[1,4,7,7,384]{4,3,2,1,0} %multiply.3212), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %constant.3214 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.3215 = f32[] convert(f32[] %constant.3214), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reduce.3220 = f32[384]{0} reduce(f32[1,4,7,7,384]{4,3,2,1,0} %convert.3213, f32[] %convert.3215), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_5c_Branch_0_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.3216, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.3221 = s32[] get-dimension-size(f32[1,4,7,7,384]{4,3,2,1,0} %convert.3213), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.3222 = s32[] get-dimension-size(f32[1,4,7,7,384]{4,3,2,1,0} %convert.3213), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.3223 = s32[] multiply(s32[] %get-dimension-size.3221, s32[] %get-dimension-size.3222), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.3224 = s32[] get-dimension-size(f32[1,4,7,7,384]{4,3,2,1,0} %convert.3213), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.3225 = s32[] multiply(s32[] %multiply.3223, s32[] %get-dimension-size.3224), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.3226 = s32[] get-dimension-size(f32[1,4,7,7,384]{4,3,2,1,0} %convert.3213), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.3227 = s32[] multiply(s32[] %multiply.3225, s32[] %get-dimension-size.3226), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.3228 = f32[] convert(s32[] %multiply.3227), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.3229 = f32[384]{0} broadcast(f32[] %convert.3228), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %divide.3230 = f32[384]{0} divide(f32[384]{0} %reduce.3220, f32[384]{0} %broadcast.3229), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.3231 = f32[384]{0} convert(f32[384]{0} %divide.3230), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reshape.3232 = f32[1,1,1,1,384]{4,3,2,1,0} reshape(f32[384]{0} %convert.3231), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %add.3235 = f32[1,1,1,1,384]{4,3,2,1,0} add(f32[1,1,1,1,384]{4,3,2,1,0} %broadcast.3234, f32[1,1,1,1,384]{4,3,2,1,0} %reshape.3232), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.3236 = f32[1,1,1,1,384]{4,3,2,1,0} rsqrt(f32[1,1,1,1,384]{4,3,2,1,0} %add.3235), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.3237 = f32[1,384]{1,0} reshape(f32[1,1,1,1,384]{4,3,2,1,0} %rsqrt.3236), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.3238 = f32[1,4,7,7,384]{4,3,2,1,0} broadcast(f32[1,384]{1,0} %reshape.3237), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %multiply.3239 = f32[1,4,7,7,384]{4,3,2,1,0} multiply(f32[1,4,7,7,384]{4,3,2,1,0} %broadcast.3238, f32[1,4,7,7,384]{4,3,2,1,0} %convolution.3188), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg76.77 = f32[1,1,1,1,384]{4,3,2,1,0} parameter(76), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.3240 = f32[1,1,1,1,384]{4,3,2,1,0} multiply(f32[1,1,1,1,384]{4,3,2,1,0} %rsqrt.3236, f32[1,1,1,1,384]{4,3,2,1,0} %reshape.3208), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.3241 = f32[1,1,1,1,384]{4,3,2,1,0} subtract(f32[1,1,1,1,384]{4,3,2,1,0} %arg76.77, f32[1,1,1,1,384]{4,3,2,1,0} %multiply.3240), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.3242 = f32[1,384]{1,0} reshape(f32[1,1,1,1,384]{4,3,2,1,0} %subtract.3241), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.3243 = f32[1,4,7,7,384]{4,3,2,1,0} broadcast(f32[1,384]{1,0} %reshape.3242), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.3244 = f32[1,4,7,7,384]{4,3,2,1,0} add(f32[1,4,7,7,384]{4,3,2,1,0} %multiply.3239, f32[1,4,7,7,384]{4,3,2,1,0} %broadcast.3243), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %constant.3356 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %broadcast.3357 = f32[1,1,1,1,384]{4,3,2,1,0} broadcast(f32[] %constant.3356), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %constant.3308 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/Relu"}
  %broadcast.3309 = f32[1,4,7,7,192]{4,3,2,1,0} broadcast(f32[] %constant.3308), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/Relu"}
  %constant.3296 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.3297 = f32[1,1,1,1,192]{4,3,2,1,0} broadcast(f32[] %constant.3296), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %arg100.101 = f32[1,1,1,832,192]{4,3,2,1,0} parameter(100), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.3251 = f32[1,4,7,7,192]{4,3,2,1,0} convolution(f32[1,4,7,7,832]{4,3,2,1,0} %maximum.3187, f32[1,1,1,832,192]{4,3,2,1,0} %arg100.101), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/conv_3d/convolution"}
  %convert.3252 = f32[1,4,7,7,192]{4,3,2,1,0} convert(f32[1,4,7,7,192]{4,3,2,1,0} %convolution.3251), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %constant.3253 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.3254 = f32[] convert(f32[] %constant.3253), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reduce.3259 = f32[192]{0} reduce(f32[1,4,7,7,192]{4,3,2,1,0} %convert.3252, f32[] %convert.3254), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_5c_Branch_1_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.3255, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.3260 = s32[] get-dimension-size(f32[1,4,7,7,192]{4,3,2,1,0} %convert.3252), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.3261 = s32[] get-dimension-size(f32[1,4,7,7,192]{4,3,2,1,0} %convert.3252), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.3262 = s32[] multiply(s32[] %get-dimension-size.3260, s32[] %get-dimension-size.3261), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.3263 = s32[] get-dimension-size(f32[1,4,7,7,192]{4,3,2,1,0} %convert.3252), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.3264 = s32[] multiply(s32[] %multiply.3262, s32[] %get-dimension-size.3263), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.3265 = s32[] get-dimension-size(f32[1,4,7,7,192]{4,3,2,1,0} %convert.3252), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.3266 = s32[] multiply(s32[] %multiply.3264, s32[] %get-dimension-size.3265), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.3267 = f32[] convert(s32[] %multiply.3266), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.3268 = f32[192]{0} broadcast(f32[] %convert.3267), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %divide.3269 = f32[192]{0} divide(f32[192]{0} %reduce.3259, f32[192]{0} %broadcast.3268), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.3270 = f32[192]{0} convert(f32[192]{0} %divide.3269), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.3271 = f32[1,1,1,1,192]{4,3,2,1,0} reshape(f32[192]{0} %convert.3270), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.3272 = f32[1,192]{1,0} reshape(f32[1,1,1,1,192]{4,3,2,1,0} %reshape.3271), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.3273 = f32[1,4,7,7,192]{4,3,2,1,0} broadcast(f32[1,192]{1,0} %reshape.3272), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.3274 = f32[1,4,7,7,192]{4,3,2,1,0} subtract(f32[1,4,7,7,192]{4,3,2,1,0} %broadcast.3273, f32[1,4,7,7,192]{4,3,2,1,0} %convolution.3251), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.3275 = f32[1,4,7,7,192]{4,3,2,1,0} multiply(f32[1,4,7,7,192]{4,3,2,1,0} %subtract.3274, f32[1,4,7,7,192]{4,3,2,1,0} %subtract.3274), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.3276 = f32[1,4,7,7,192]{4,3,2,1,0} convert(f32[1,4,7,7,192]{4,3,2,1,0} %multiply.3275), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %constant.3277 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.3278 = f32[] convert(f32[] %constant.3277), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reduce.3283 = f32[192]{0} reduce(f32[1,4,7,7,192]{4,3,2,1,0} %convert.3276, f32[] %convert.3278), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_5c_Branch_1_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.3279, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.3284 = s32[] get-dimension-size(f32[1,4,7,7,192]{4,3,2,1,0} %convert.3276), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.3285 = s32[] get-dimension-size(f32[1,4,7,7,192]{4,3,2,1,0} %convert.3276), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.3286 = s32[] multiply(s32[] %get-dimension-size.3284, s32[] %get-dimension-size.3285), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.3287 = s32[] get-dimension-size(f32[1,4,7,7,192]{4,3,2,1,0} %convert.3276), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.3288 = s32[] multiply(s32[] %multiply.3286, s32[] %get-dimension-size.3287), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.3289 = s32[] get-dimension-size(f32[1,4,7,7,192]{4,3,2,1,0} %convert.3276), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.3290 = s32[] multiply(s32[] %multiply.3288, s32[] %get-dimension-size.3289), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.3291 = f32[] convert(s32[] %multiply.3290), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.3292 = f32[192]{0} broadcast(f32[] %convert.3291), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %divide.3293 = f32[192]{0} divide(f32[192]{0} %reduce.3283, f32[192]{0} %broadcast.3292), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.3294 = f32[192]{0} convert(f32[192]{0} %divide.3293), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reshape.3295 = f32[1,1,1,1,192]{4,3,2,1,0} reshape(f32[192]{0} %convert.3294), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %add.3298 = f32[1,1,1,1,192]{4,3,2,1,0} add(f32[1,1,1,1,192]{4,3,2,1,0} %broadcast.3297, f32[1,1,1,1,192]{4,3,2,1,0} %reshape.3295), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.3299 = f32[1,1,1,1,192]{4,3,2,1,0} rsqrt(f32[1,1,1,1,192]{4,3,2,1,0} %add.3298), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.3300 = f32[1,192]{1,0} reshape(f32[1,1,1,1,192]{4,3,2,1,0} %rsqrt.3299), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.3301 = f32[1,4,7,7,192]{4,3,2,1,0} broadcast(f32[1,192]{1,0} %reshape.3300), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %multiply.3302 = f32[1,4,7,7,192]{4,3,2,1,0} multiply(f32[1,4,7,7,192]{4,3,2,1,0} %broadcast.3301, f32[1,4,7,7,192]{4,3,2,1,0} %convolution.3251), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg10.11 = f32[1,1,1,1,192]{4,3,2,1,0} parameter(10), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.3303 = f32[1,1,1,1,192]{4,3,2,1,0} multiply(f32[1,1,1,1,192]{4,3,2,1,0} %rsqrt.3299, f32[1,1,1,1,192]{4,3,2,1,0} %reshape.3271), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.3304 = f32[1,1,1,1,192]{4,3,2,1,0} subtract(f32[1,1,1,1,192]{4,3,2,1,0} %arg10.11, f32[1,1,1,1,192]{4,3,2,1,0} %multiply.3303), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.3305 = f32[1,192]{1,0} reshape(f32[1,1,1,1,192]{4,3,2,1,0} %subtract.3304), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.3306 = f32[1,4,7,7,192]{4,3,2,1,0} broadcast(f32[1,192]{1,0} %reshape.3305), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.3307 = f32[1,4,7,7,192]{4,3,2,1,0} add(f32[1,4,7,7,192]{4,3,2,1,0} %multiply.3302, f32[1,4,7,7,192]{4,3,2,1,0} %broadcast.3306), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %maximum.3310 = f32[1,4,7,7,192]{4,3,2,1,0} maximum(f32[1,4,7,7,192]{4,3,2,1,0} %broadcast.3309, f32[1,4,7,7,192]{4,3,2,1,0} %add.3307), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/Relu"}
  %arg49.50 = f32[3,3,3,192,384]{4,3,2,1,0} parameter(49), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.3311 = f32[1,4,7,7,384]{4,3,2,1,0} convolution(f32[1,4,7,7,192]{4,3,2,1,0} %maximum.3310, f32[3,3,3,192,384]{4,3,2,1,0} %arg49.50), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/conv_3d/convolution"}
  %convert.3312 = f32[1,4,7,7,384]{4,3,2,1,0} convert(f32[1,4,7,7,384]{4,3,2,1,0} %convolution.3311), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %constant.3313 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.3314 = f32[] convert(f32[] %constant.3313), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reduce.3319 = f32[384]{0} reduce(f32[1,4,7,7,384]{4,3,2,1,0} %convert.3312, f32[] %convert.3314), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_5c_Branch_1_Conv3d_0b_3x3_batch_norm_normalize_moments_mean-reduction.3315, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.3320 = s32[] get-dimension-size(f32[1,4,7,7,384]{4,3,2,1,0} %convert.3312), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.3321 = s32[] get-dimension-size(f32[1,4,7,7,384]{4,3,2,1,0} %convert.3312), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.3322 = s32[] multiply(s32[] %get-dimension-size.3320, s32[] %get-dimension-size.3321), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.3323 = s32[] get-dimension-size(f32[1,4,7,7,384]{4,3,2,1,0} %convert.3312), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.3324 = s32[] multiply(s32[] %multiply.3322, s32[] %get-dimension-size.3323), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.3325 = s32[] get-dimension-size(f32[1,4,7,7,384]{4,3,2,1,0} %convert.3312), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.3326 = s32[] multiply(s32[] %multiply.3324, s32[] %get-dimension-size.3325), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.3327 = f32[] convert(s32[] %multiply.3326), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %broadcast.3328 = f32[384]{0} broadcast(f32[] %convert.3327), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %divide.3329 = f32[384]{0} divide(f32[384]{0} %reduce.3319, f32[384]{0} %broadcast.3328), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.3330 = f32[384]{0} convert(f32[384]{0} %divide.3329), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reshape.3331 = f32[1,1,1,1,384]{4,3,2,1,0} reshape(f32[384]{0} %convert.3330), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reshape.3332 = f32[1,384]{1,0} reshape(f32[1,1,1,1,384]{4,3,2,1,0} %reshape.3331), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.3333 = f32[1,4,7,7,384]{4,3,2,1,0} broadcast(f32[1,384]{1,0} %reshape.3332), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.3334 = f32[1,4,7,7,384]{4,3,2,1,0} subtract(f32[1,4,7,7,384]{4,3,2,1,0} %broadcast.3333, f32[1,4,7,7,384]{4,3,2,1,0} %convolution.3311), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.3335 = f32[1,4,7,7,384]{4,3,2,1,0} multiply(f32[1,4,7,7,384]{4,3,2,1,0} %subtract.3334, f32[1,4,7,7,384]{4,3,2,1,0} %subtract.3334), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %convert.3336 = f32[1,4,7,7,384]{4,3,2,1,0} convert(f32[1,4,7,7,384]{4,3,2,1,0} %multiply.3335), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %constant.3337 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.3338 = f32[] convert(f32[] %constant.3337), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %reduce.3343 = f32[384]{0} reduce(f32[1,4,7,7,384]{4,3,2,1,0} %convert.3336, f32[] %convert.3338), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_5c_Branch_1_Conv3d_0b_3x3_batch_norm_normalize_moments_variance-reduction.3339, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.3344 = s32[] get-dimension-size(f32[1,4,7,7,384]{4,3,2,1,0} %convert.3336), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.3345 = s32[] get-dimension-size(f32[1,4,7,7,384]{4,3,2,1,0} %convert.3336), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.3346 = s32[] multiply(s32[] %get-dimension-size.3344, s32[] %get-dimension-size.3345), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.3347 = s32[] get-dimension-size(f32[1,4,7,7,384]{4,3,2,1,0} %convert.3336), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.3348 = s32[] multiply(s32[] %multiply.3346, s32[] %get-dimension-size.3347), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.3349 = s32[] get-dimension-size(f32[1,4,7,7,384]{4,3,2,1,0} %convert.3336), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.3350 = s32[] multiply(s32[] %multiply.3348, s32[] %get-dimension-size.3349), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.3351 = f32[] convert(s32[] %multiply.3350), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %broadcast.3352 = f32[384]{0} broadcast(f32[] %convert.3351), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %divide.3353 = f32[384]{0} divide(f32[384]{0} %reduce.3343, f32[384]{0} %broadcast.3352), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.3354 = f32[384]{0} convert(f32[384]{0} %divide.3353), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %reshape.3355 = f32[1,1,1,1,384]{4,3,2,1,0} reshape(f32[384]{0} %convert.3354), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %add.3358 = f32[1,1,1,1,384]{4,3,2,1,0} add(f32[1,1,1,1,384]{4,3,2,1,0} %broadcast.3357, f32[1,1,1,1,384]{4,3,2,1,0} %reshape.3355), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %rsqrt.3359 = f32[1,1,1,1,384]{4,3,2,1,0} rsqrt(f32[1,1,1,1,384]{4,3,2,1,0} %add.3358), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.3360 = f32[1,384]{1,0} reshape(f32[1,1,1,1,384]{4,3,2,1,0} %rsqrt.3359), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %broadcast.3361 = f32[1,4,7,7,384]{4,3,2,1,0} broadcast(f32[1,384]{1,0} %reshape.3360), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %multiply.3362 = f32[1,4,7,7,384]{4,3,2,1,0} multiply(f32[1,4,7,7,384]{4,3,2,1,0} %broadcast.3361, f32[1,4,7,7,384]{4,3,2,1,0} %convolution.3311), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %arg71.72 = f32[1,1,1,1,384]{4,3,2,1,0} parameter(71), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.3363 = f32[1,1,1,1,384]{4,3,2,1,0} multiply(f32[1,1,1,1,384]{4,3,2,1,0} %rsqrt.3359, f32[1,1,1,1,384]{4,3,2,1,0} %reshape.3331), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.3364 = f32[1,1,1,1,384]{4,3,2,1,0} subtract(f32[1,1,1,1,384]{4,3,2,1,0} %arg71.72, f32[1,1,1,1,384]{4,3,2,1,0} %multiply.3363), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/sub"}
  %reshape.3365 = f32[1,384]{1,0} reshape(f32[1,1,1,1,384]{4,3,2,1,0} %subtract.3364), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.3366 = f32[1,4,7,7,384]{4,3,2,1,0} broadcast(f32[1,384]{1,0} %reshape.3365), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %add.3367 = f32[1,4,7,7,384]{4,3,2,1,0} add(f32[1,4,7,7,384]{4,3,2,1,0} %multiply.3362, f32[1,4,7,7,384]{4,3,2,1,0} %broadcast.3366), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %constant.3473 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %broadcast.3474 = f32[1,1,1,1,128]{4,3,2,1,0} broadcast(f32[] %constant.3473), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %constant.3425 = f32[] constant(0), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/Relu"}
  %broadcast.3426 = f32[1,4,7,7,48]{4,3,2,1,0} broadcast(f32[] %constant.3425), dimensions={}, metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/Relu"}
  %constant.3413 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %broadcast.3414 = f32[1,1,1,1,48]{4,3,2,1,0} broadcast(f32[] %constant.3413), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %arg116.117 = f32[1,1,1,832,48]{4,3,2,1,0} parameter(116), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.3368 = f32[1,4,7,7,48]{4,3,2,1,0} convolution(f32[1,4,7,7,832]{4,3,2,1,0} %maximum.3187, f32[1,1,1,832,48]{4,3,2,1,0} %arg116.117), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/conv_3d/convolution"}
  %convert.3369 = f32[1,4,7,7,48]{4,3,2,1,0} convert(f32[1,4,7,7,48]{4,3,2,1,0} %convolution.3368), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %constant.3370 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.3371 = f32[] convert(f32[] %constant.3370), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reduce.3376 = f32[48]{0} reduce(f32[1,4,7,7,48]{4,3,2,1,0} %convert.3369, f32[] %convert.3371), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_5c_Branch_2_Conv3d_0a_1x1_batch_norm_normalize_moments_mean-reduction.3372, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.3377 = s32[] get-dimension-size(f32[1,4,7,7,48]{4,3,2,1,0} %convert.3369), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.3378 = s32[] get-dimension-size(f32[1,4,7,7,48]{4,3,2,1,0} %convert.3369), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.3379 = s32[] multiply(s32[] %get-dimension-size.3377, s32[] %get-dimension-size.3378), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.3380 = s32[] get-dimension-size(f32[1,4,7,7,48]{4,3,2,1,0} %convert.3369), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.3381 = s32[] multiply(s32[] %multiply.3379, s32[] %get-dimension-size.3380), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.3382 = s32[] get-dimension-size(f32[1,4,7,7,48]{4,3,2,1,0} %convert.3369), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %multiply.3383 = s32[] multiply(s32[] %multiply.3381, s32[] %get-dimension-size.3382), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.3384 = f32[] convert(s32[] %multiply.3383), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.3385 = f32[48]{0} broadcast(f32[] %convert.3384), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %divide.3386 = f32[48]{0} divide(f32[48]{0} %reduce.3376, f32[48]{0} %broadcast.3385), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %convert.3387 = f32[48]{0} convert(f32[48]{0} %divide.3386), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.3388 = f32[1,1,1,1,48]{4,3,2,1,0} reshape(f32[48]{0} %convert.3387), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/mean"}
  %reshape.3389 = f32[1,48]{1,0} reshape(f32[1,1,1,1,48]{4,3,2,1,0} %reshape.3388), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.3390 = f32[1,4,7,7,48]{4,3,2,1,0} broadcast(f32[1,48]{1,0} %reshape.3389), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.3391 = f32[1,4,7,7,48]{4,3,2,1,0} subtract(f32[1,4,7,7,48]{4,3,2,1,0} %broadcast.3390, f32[1,4,7,7,48]{4,3,2,1,0} %convolution.3368), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.3392 = f32[1,4,7,7,48]{4,3,2,1,0} multiply(f32[1,4,7,7,48]{4,3,2,1,0} %subtract.3391, f32[1,4,7,7,48]{4,3,2,1,0} %subtract.3391), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.3393 = f32[1,4,7,7,48]{4,3,2,1,0} convert(f32[1,4,7,7,48]{4,3,2,1,0} %multiply.3392), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %constant.3394 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.3395 = f32[] convert(f32[] %constant.3394), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reduce.3400 = f32[48]{0} reduce(f32[1,4,7,7,48]{4,3,2,1,0} %convert.3393, f32[] %convert.3395), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_5c_Branch_2_Conv3d_0a_1x1_batch_norm_normalize_moments_variance-reduction.3396, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.3401 = s32[] get-dimension-size(f32[1,4,7,7,48]{4,3,2,1,0} %convert.3393), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.3402 = s32[] get-dimension-size(f32[1,4,7,7,48]{4,3,2,1,0} %convert.3393), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.3403 = s32[] multiply(s32[] %get-dimension-size.3401, s32[] %get-dimension-size.3402), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.3404 = s32[] get-dimension-size(f32[1,4,7,7,48]{4,3,2,1,0} %convert.3393), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.3405 = s32[] multiply(s32[] %multiply.3403, s32[] %get-dimension-size.3404), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.3406 = s32[] get-dimension-size(f32[1,4,7,7,48]{4,3,2,1,0} %convert.3393), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %multiply.3407 = s32[] multiply(s32[] %multiply.3405, s32[] %get-dimension-size.3406), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.3408 = f32[] convert(s32[] %multiply.3407), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.3409 = f32[48]{0} broadcast(f32[] %convert.3408), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %divide.3410 = f32[48]{0} divide(f32[48]{0} %reduce.3400, f32[48]{0} %broadcast.3409), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %convert.3411 = f32[48]{0} convert(f32[48]{0} %divide.3410), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %reshape.3412 = f32[1,1,1,1,48]{4,3,2,1,0} reshape(f32[48]{0} %convert.3411), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/normalize_moments/variance"}
  %add.3415 = f32[1,1,1,1,48]{4,3,2,1,0} add(f32[1,1,1,1,48]{4,3,2,1,0} %broadcast.3414, f32[1,1,1,1,48]{4,3,2,1,0} %reshape.3412), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add"}
  %rsqrt.3416 = f32[1,1,1,1,48]{4,3,2,1,0} rsqrt(f32[1,1,1,1,48]{4,3,2,1,0} %add.3415), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.3417 = f32[1,48]{1,0} reshape(f32[1,1,1,1,48]{4,3,2,1,0} %rsqrt.3416), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %broadcast.3418 = f32[1,4,7,7,48]{4,3,2,1,0} broadcast(f32[1,48]{1,0} %reshape.3417), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %multiply.3419 = f32[1,4,7,7,48]{4,3,2,1,0} multiply(f32[1,4,7,7,48]{4,3,2,1,0} %broadcast.3418, f32[1,4,7,7,48]{4,3,2,1,0} %convolution.3368), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul"}
  %arg41.42 = f32[1,1,1,1,48]{4,3,2,1,0} parameter(41), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.3420 = f32[1,1,1,1,48]{4,3,2,1,0} multiply(f32[1,1,1,1,48]{4,3,2,1,0} %rsqrt.3416, f32[1,1,1,1,48]{4,3,2,1,0} %reshape.3388), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.3421 = f32[1,1,1,1,48]{4,3,2,1,0} subtract(f32[1,1,1,1,48]{4,3,2,1,0} %arg41.42, f32[1,1,1,1,48]{4,3,2,1,0} %multiply.3420), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/sub"}
  %reshape.3422 = f32[1,48]{1,0} reshape(f32[1,1,1,1,48]{4,3,2,1,0} %subtract.3421), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.3423 = f32[1,4,7,7,48]{4,3,2,1,0} broadcast(f32[1,48]{1,0} %reshape.3422), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %add.3424 = f32[1,4,7,7,48]{4,3,2,1,0} add(f32[1,4,7,7,48]{4,3,2,1,0} %multiply.3419, f32[1,4,7,7,48]{4,3,2,1,0} %broadcast.3423), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/batch_norm/add_1"}
  %maximum.3427 = f32[1,4,7,7,48]{4,3,2,1,0} maximum(f32[1,4,7,7,48]{4,3,2,1,0} %broadcast.3426, f32[1,4,7,7,48]{4,3,2,1,0} %add.3424), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/Relu"}
  %arg98.99 = f32[3,3,3,48,128]{4,3,2,1,0} parameter(98), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.3428 = f32[1,4,7,7,128]{4,3,2,1,0} convolution(f32[1,4,7,7,48]{4,3,2,1,0} %maximum.3427, f32[3,3,3,48,128]{4,3,2,1,0} %arg98.99), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/conv_3d/convolution"}
  %convert.3429 = f32[1,4,7,7,128]{4,3,2,1,0} convert(f32[1,4,7,7,128]{4,3,2,1,0} %convolution.3428), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %constant.3430 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.3431 = f32[] convert(f32[] %constant.3430), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reduce.3436 = f32[128]{0} reduce(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3429, f32[] %convert.3431), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_5c_Branch_2_Conv3d_0b_3x3_batch_norm_normalize_moments_mean-reduction.3432, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.3437 = s32[] get-dimension-size(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3429), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.3438 = s32[] get-dimension-size(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3429), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.3439 = s32[] multiply(s32[] %get-dimension-size.3437, s32[] %get-dimension-size.3438), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.3440 = s32[] get-dimension-size(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3429), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.3441 = s32[] multiply(s32[] %multiply.3439, s32[] %get-dimension-size.3440), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %get-dimension-size.3442 = s32[] get-dimension-size(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3429), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %multiply.3443 = s32[] multiply(s32[] %multiply.3441, s32[] %get-dimension-size.3442), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.3444 = f32[] convert(s32[] %multiply.3443), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %broadcast.3445 = f32[128]{0} broadcast(f32[] %convert.3444), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %divide.3446 = f32[128]{0} divide(f32[128]{0} %reduce.3436, f32[128]{0} %broadcast.3445), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %convert.3447 = f32[128]{0} convert(f32[128]{0} %divide.3446), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reshape.3448 = f32[1,1,1,1,128]{4,3,2,1,0} reshape(f32[128]{0} %convert.3447), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/mean"}
  %reshape.3449 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %reshape.3448), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.3450 = f32[1,4,7,7,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.3449), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.3451 = f32[1,4,7,7,128]{4,3,2,1,0} subtract(f32[1,4,7,7,128]{4,3,2,1,0} %broadcast.3450, f32[1,4,7,7,128]{4,3,2,1,0} %convolution.3428), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.3452 = f32[1,4,7,7,128]{4,3,2,1,0} multiply(f32[1,4,7,7,128]{4,3,2,1,0} %subtract.3451, f32[1,4,7,7,128]{4,3,2,1,0} %subtract.3451), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/SquaredDifference"}
  %convert.3453 = f32[1,4,7,7,128]{4,3,2,1,0} convert(f32[1,4,7,7,128]{4,3,2,1,0} %multiply.3452), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %constant.3454 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.3455 = f32[] convert(f32[] %constant.3454), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %reduce.3460 = f32[128]{0} reduce(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3453, f32[] %convert.3455), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_5c_Branch_2_Conv3d_0b_3x3_batch_norm_normalize_moments_variance-reduction.3456, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.3461 = s32[] get-dimension-size(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3453), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.3462 = s32[] get-dimension-size(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3453), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.3463 = s32[] multiply(s32[] %get-dimension-size.3461, s32[] %get-dimension-size.3462), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.3464 = s32[] get-dimension-size(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3453), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.3465 = s32[] multiply(s32[] %multiply.3463, s32[] %get-dimension-size.3464), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %get-dimension-size.3466 = s32[] get-dimension-size(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3453), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %multiply.3467 = s32[] multiply(s32[] %multiply.3465, s32[] %get-dimension-size.3466), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.3468 = f32[] convert(s32[] %multiply.3467), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %broadcast.3469 = f32[128]{0} broadcast(f32[] %convert.3468), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %divide.3470 = f32[128]{0} divide(f32[128]{0} %reduce.3460, f32[128]{0} %broadcast.3469), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %convert.3471 = f32[128]{0} convert(f32[128]{0} %divide.3470), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %reshape.3472 = f32[1,1,1,1,128]{4,3,2,1,0} reshape(f32[128]{0} %convert.3471), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/normalize_moments/variance"}
  %add.3475 = f32[1,1,1,1,128]{4,3,2,1,0} add(f32[1,1,1,1,128]{4,3,2,1,0} %broadcast.3474, f32[1,1,1,1,128]{4,3,2,1,0} %reshape.3472), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add"}
  %rsqrt.3476 = f32[1,1,1,1,128]{4,3,2,1,0} rsqrt(f32[1,1,1,1,128]{4,3,2,1,0} %add.3475), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/Rsqrt"}
  %reshape.3477 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.3476), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %broadcast.3478 = f32[1,4,7,7,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.3477), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %multiply.3479 = f32[1,4,7,7,128]{4,3,2,1,0} multiply(f32[1,4,7,7,128]{4,3,2,1,0} %broadcast.3478, f32[1,4,7,7,128]{4,3,2,1,0} %convolution.3428), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul"}
  %arg26.27 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(26), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.3480 = f32[1,1,1,1,128]{4,3,2,1,0} multiply(f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.3476, f32[1,1,1,1,128]{4,3,2,1,0} %reshape.3448), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/mul_1"}
  %subtract.3481 = f32[1,1,1,1,128]{4,3,2,1,0} subtract(f32[1,1,1,1,128]{4,3,2,1,0} %arg26.27, f32[1,1,1,1,128]{4,3,2,1,0} %multiply.3480), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/sub"}
  %reshape.3482 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %subtract.3481), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %broadcast.3483 = f32[1,4,7,7,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.3482), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %add.3484 = f32[1,4,7,7,128]{4,3,2,1,0} add(f32[1,4,7,7,128]{4,3,2,1,0} %multiply.3479, f32[1,4,7,7,128]{4,3,2,1,0} %broadcast.3483), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/batch_norm/add_1"}
  %constant.3530 = f32[] constant(0.001), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %broadcast.3531 = f32[1,1,1,1,128]{4,3,2,1,0} broadcast(f32[] %constant.3530), dimensions={}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %constant.3245 = f32[] constant(-inf), metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/MaxPool3d_0a_3x3"}
  %reduce-window.3250 = f32[1,4,7,7,832]{4,3,2,1,0} reduce-window(f32[1,4,7,7,832]{4,3,2,1,0} %maximum.3187, f32[] %constant.3245), window={size=1x3x3x3x1 pad=0_0x1_1x1_1x1_1x0_0}, to_apply=%max_F32.3246, metadata={op_type="MaxPool3D" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/MaxPool3d_0a_3x3"}
  %arg112.113 = f32[1,1,1,832,128]{4,3,2,1,0} parameter(112), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.3485 = f32[1,4,7,7,128]{4,3,2,1,0} convolution(f32[1,4,7,7,832]{4,3,2,1,0} %reduce-window.3250, f32[1,1,1,832,128]{4,3,2,1,0} %arg112.113), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/conv_3d/convolution"}
  %convert.3486 = f32[1,4,7,7,128]{4,3,2,1,0} convert(f32[1,4,7,7,128]{4,3,2,1,0} %convolution.3485), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %constant.3487 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %convert.3488 = f32[] convert(f32[] %constant.3487), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %reduce.3493 = f32[128]{0} reduce(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3486, f32[] %convert.3488), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_5c_Branch_3_Conv3d_0b_1x1_batch_norm_normalize_moments_mean-reduction.3489, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.3494 = s32[] get-dimension-size(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3486), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.3495 = s32[] get-dimension-size(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3486), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %multiply.3496 = s32[] multiply(s32[] %get-dimension-size.3494, s32[] %get-dimension-size.3495), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.3497 = s32[] get-dimension-size(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3486), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %multiply.3498 = s32[] multiply(s32[] %multiply.3496, s32[] %get-dimension-size.3497), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %get-dimension-size.3499 = s32[] get-dimension-size(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3486), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %multiply.3500 = s32[] multiply(s32[] %multiply.3498, s32[] %get-dimension-size.3499), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %convert.3501 = f32[] convert(s32[] %multiply.3500), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %broadcast.3502 = f32[128]{0} broadcast(f32[] %convert.3501), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %divide.3503 = f32[128]{0} divide(f32[128]{0} %reduce.3493, f32[128]{0} %broadcast.3502), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %convert.3504 = f32[128]{0} convert(f32[128]{0} %divide.3503), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %reshape.3505 = f32[1,1,1,1,128]{4,3,2,1,0} reshape(f32[128]{0} %convert.3504), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/mean"}
  %reshape.3506 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %reshape.3505), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %broadcast.3507 = f32[1,4,7,7,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.3506), dimensions={0,4}, metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %subtract.3508 = f32[1,4,7,7,128]{4,3,2,1,0} subtract(f32[1,4,7,7,128]{4,3,2,1,0} %broadcast.3507, f32[1,4,7,7,128]{4,3,2,1,0} %convolution.3485), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %multiply.3509 = f32[1,4,7,7,128]{4,3,2,1,0} multiply(f32[1,4,7,7,128]{4,3,2,1,0} %subtract.3508, f32[1,4,7,7,128]{4,3,2,1,0} %subtract.3508), metadata={op_type="SquaredDifference" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/SquaredDifference"}
  %convert.3510 = f32[1,4,7,7,128]{4,3,2,1,0} convert(f32[1,4,7,7,128]{4,3,2,1,0} %multiply.3509), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %constant.3511 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %convert.3512 = f32[] convert(f32[] %constant.3511), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %reduce.3517 = f32[128]{0} reduce(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3510, f32[] %convert.3512), dimensions={0,1,2,3}, to_apply=%RGB_inception_i3d_Mixed_5c_Branch_3_Conv3d_0b_1x1_batch_norm_normalize_moments_variance-reduction.3513, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.3518 = s32[] get-dimension-size(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3510), dimensions={0}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.3519 = s32[] get-dimension-size(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3510), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %multiply.3520 = s32[] multiply(s32[] %get-dimension-size.3518, s32[] %get-dimension-size.3519), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.3521 = s32[] get-dimension-size(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3510), dimensions={2}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %multiply.3522 = s32[] multiply(s32[] %multiply.3520, s32[] %get-dimension-size.3521), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %get-dimension-size.3523 = s32[] get-dimension-size(f32[1,4,7,7,128]{4,3,2,1,0} %convert.3510), dimensions={3}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %multiply.3524 = s32[] multiply(s32[] %multiply.3522, s32[] %get-dimension-size.3523), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %convert.3525 = f32[] convert(s32[] %multiply.3524), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %broadcast.3526 = f32[128]{0} broadcast(f32[] %convert.3525), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %divide.3527 = f32[128]{0} divide(f32[128]{0} %reduce.3517, f32[128]{0} %broadcast.3526), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %convert.3528 = f32[128]{0} convert(f32[128]{0} %divide.3527), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %reshape.3529 = f32[1,1,1,1,128]{4,3,2,1,0} reshape(f32[128]{0} %convert.3528), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/normalize_moments/variance"}
  %add.3532 = f32[1,1,1,1,128]{4,3,2,1,0} add(f32[1,1,1,1,128]{4,3,2,1,0} %broadcast.3531, f32[1,1,1,1,128]{4,3,2,1,0} %reshape.3529), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add"}
  %rsqrt.3533 = f32[1,1,1,1,128]{4,3,2,1,0} rsqrt(f32[1,1,1,1,128]{4,3,2,1,0} %add.3532), metadata={op_type="Rsqrt" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/Rsqrt"}
  %reshape.3534 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.3533), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %broadcast.3535 = f32[1,4,7,7,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.3534), dimensions={0,4}, metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %multiply.3536 = f32[1,4,7,7,128]{4,3,2,1,0} multiply(f32[1,4,7,7,128]{4,3,2,1,0} %broadcast.3535, f32[1,4,7,7,128]{4,3,2,1,0} %convolution.3485), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul"}
  %arg93.94 = f32[1,1,1,1,128]{4,3,2,1,0} parameter(93), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %multiply.3537 = f32[1,1,1,1,128]{4,3,2,1,0} multiply(f32[1,1,1,1,128]{4,3,2,1,0} %rsqrt.3533, f32[1,1,1,1,128]{4,3,2,1,0} %reshape.3505), metadata={op_type="Mul" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/mul_1"}
  %subtract.3538 = f32[1,1,1,1,128]{4,3,2,1,0} subtract(f32[1,1,1,1,128]{4,3,2,1,0} %arg93.94, f32[1,1,1,1,128]{4,3,2,1,0} %multiply.3537), metadata={op_type="Sub" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/sub"}
  %reshape.3539 = f32[1,128]{1,0} reshape(f32[1,1,1,1,128]{4,3,2,1,0} %subtract.3538), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %broadcast.3540 = f32[1,4,7,7,128]{4,3,2,1,0} broadcast(f32[1,128]{1,0} %reshape.3539), dimensions={0,4}, metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %add.3541 = f32[1,4,7,7,128]{4,3,2,1,0} add(f32[1,4,7,7,128]{4,3,2,1,0} %multiply.3536, f32[1,4,7,7,128]{4,3,2,1,0} %broadcast.3540), metadata={op_type="Add" op_name="RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/batch_norm/add_1"}
  %concatenate.3542 = f32[1,4,7,7,1024]{4,3,2,1,0} concatenate(f32[1,4,7,7,384]{4,3,2,1,0} %add.3244, f32[1,4,7,7,384]{4,3,2,1,0} %add.3367, f32[1,4,7,7,128]{4,3,2,1,0} %add.3484, f32[1,4,7,7,128]{4,3,2,1,0} %add.3541), dimensions={4}, metadata={op_type="ConcatV2" op_name="RGB/inception_i3d/Mixed_5c/concat"}
  %maximum.3545 = f32[1,4,7,7,1024]{4,3,2,1,0} maximum(f32[1,4,7,7,1024]{4,3,2,1,0} %broadcast.3544, f32[1,4,7,7,1024]{4,3,2,1,0} %concatenate.3542), metadata={op_type="Relu" op_name="RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/Relu"}
  %convert.3546 = f32[1,4,7,7,1024]{4,3,2,1,0} convert(f32[1,4,7,7,1024]{4,3,2,1,0} %maximum.3545), metadata={op_type="AvgPool3D" op_name="RGB/inception_i3d/Logits/AvgPool3D"}
  %constant.3548 = f32[] constant(0), metadata={op_type="AvgPool3D" op_name="RGB/inception_i3d/Logits/AvgPool3D"}
  %pad.3549 = f32[1,4,7,7,1024]{4,3,2,1,0} pad(f32[1,4,7,7,1024]{4,3,2,1,0} %convert.3546, f32[] %constant.3548), padding=0_0x0_0x0_0x0_0x0_0, metadata={op_type="AvgPool3D" op_name="RGB/inception_i3d/Logits/AvgPool3D"}
  %constant.3547 = f32[] constant(0), metadata={op_type="AvgPool3D" op_name="RGB/inception_i3d/Logits/AvgPool3D"}
  %reduce-window.3554 = f32[1,3,1,1,1024]{4,3,2,1,0} reduce-window(f32[1,4,7,7,1024]{4,3,2,1,0} %pad.3549, f32[] %constant.3547), window={size=1x2x7x7x1}, to_apply=%add_F32.3550, metadata={op_type="AvgPool3D" op_name="RGB/inception_i3d/Logits/AvgPool3D"}
  %constant.3555 = f32[] constant(98), metadata={op_type="AvgPool3D" op_name="RGB/inception_i3d/Logits/AvgPool3D"}
  %broadcast.3556 = f32[1,3,1,1,1024]{4,3,2,1,0} broadcast(f32[] %constant.3555), dimensions={}, metadata={op_type="AvgPool3D" op_name="RGB/inception_i3d/Logits/AvgPool3D"}
  %divide.3557 = f32[1,3,1,1,1024]{4,3,2,1,0} divide(f32[1,3,1,1,1024]{4,3,2,1,0} %reduce-window.3554, f32[1,3,1,1,1024]{4,3,2,1,0} %broadcast.3556), metadata={op_type="AvgPool3D" op_name="RGB/inception_i3d/Logits/AvgPool3D"}
  %convert.3558 = f32[1,3,1,1,1024]{4,3,2,1,0} convert(f32[1,3,1,1,1024]{4,3,2,1,0} %divide.3557), metadata={op_type="AvgPool3D" op_name="RGB/inception_i3d/Logits/AvgPool3D"}
  %arg95.96 = f32[1,1,1,1024,400]{4,3,2,1,0} parameter(95), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %convolution.3559 = f32[1,3,1,1,400]{4,3,2,1,0} convolution(f32[1,3,1,1,1024]{4,3,2,1,0} %convert.3558, f32[1,1,1,1024,400]{4,3,2,1,0} %arg95.96), window={size=1x1x1}, dim_labels=b012f_012io->b012f, metadata={op_type="Conv3D" op_name="RGB/inception_i3d/Logits/Conv3d_0c_1x1/conv_3d/convolution"}
  %add.3562 = f32[1,3,1,1,400]{4,3,2,1,0} add(f32[1,3,1,1,400]{4,3,2,1,0} %broadcast.3561, f32[1,3,1,1,400]{4,3,2,1,0} %convolution.3559), metadata={op_type="Add" op_name="RGB/inception_i3d/Logits/Conv3d_0c_1x1/conv_3d/add"}
  %reshape.3563 = f32[1,3,400]{2,1,0} reshape(f32[1,3,1,1,400]{4,3,2,1,0} %add.3562), metadata={op_type="Squeeze" op_name="RGB/inception_i3d/Logits/SpatialSqueeze"}
  %convert.3564 = f32[1,3,400]{2,1,0} convert(f32[1,3,400]{2,1,0} %reshape.3563), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mean"}
  %constant.3565 = f32[] constant(0), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mean"}
  %convert.3566 = f32[] convert(f32[] %constant.3565), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mean"}
  %reduce.3571 = f32[1,400]{1,0} reduce(f32[1,3,400]{2,1,0} %convert.3564, f32[] %convert.3566), dimensions={1}, to_apply=%RGB_inception_i3d_Mean-reduction.3567, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mean"}
  %get-dimension-size.3572 = s32[] get-dimension-size(f32[1,3,400]{2,1,0} %convert.3564), dimensions={1}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mean"}
  %convert.3573 = f32[] convert(s32[] %get-dimension-size.3572), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mean"}
  %broadcast.3574 = f32[1,400]{1,0} broadcast(f32[] %convert.3573), dimensions={}, metadata={op_type="Mean" op_name="RGB/inception_i3d/Mean"}
  %divide.3575 = f32[1,400]{1,0} divide(f32[1,400]{1,0} %reduce.3571, f32[1,400]{1,0} %broadcast.3574), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mean"}
  %convert.3576 = f32[1,400]{1,0} convert(f32[1,400]{1,0} %divide.3575), metadata={op_type="Mean" op_name="RGB/inception_i3d/Mean"}
  %reshape.3577 = f32[1,400]{1,0} reshape(f32[1,400]{1,0} %convert.3576), metadata={op_name="XLA_Retvals"}
  %tuple.3578 = (f32[1,400]{1,0}) tuple(f32[1,400]{1,0} %reshape.3577), metadata={op_name="XLA_Retvals"}
  ROOT %get-tuple-element.3579 = f32[1,400]{1,0} get-tuple-element((f32[1,400]{1,0}) %tuple.3578), index=0, metadata={op_name="XLA_Retvals"}
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
