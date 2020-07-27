// Tests that show HLO Module conversion to PlaidML Program.

#include <algorithm>
#include <string>

#include <gtest/gtest.h>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/service/plaidml/compiler.h"
#include "tensorflow/compiler/xla/service/plaidml/tests/plaidml_codegen_test.h"
#include "tensorflow/compiler/xla/service/plaidml/tests/resnext50_pretrained_inputs_and_weights.h"
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

    }
    return Status::OK();
  }
};

TEST_P(PlaidMLResNeXTOperationTest, SimpleResNeXT) {

TestCaseVal ResNeXt50_WeightsInputs = {{0}, {0}, ::weights::stage4_unit3_bn3_mean, ::weights::stage4_unit3_bn3_scale, ::weights::stage4_unit3_bn3_var, {2e-05}, //
 ::weights::stage4_unit3_bn3_bias, ::weights::stage4_unit3_conv3_weight, {0}, ::weights::stage4_unit3_bn2_mean, ::weights::stage4_unit3_bn2_scale, {2e-05}, //
 ::weights::stage4_unit3_bn2_var, ::weights::stage4_unit3_bn2_bias, ::weights::stage4_unit3_conv2_weight, {0}, ::weights::stage4_unit3_bn1_mean, ::weights::stage4_unit3_bn1_scale, {2e-05}, //
 ::weights::stage4_unit3_bn1_var, ::weights::stage4_unit3_bn1_bias, ::weights::stage4_unit3_conv1_weight, {0}, ::weights::stage4_unit2_bn3_mean, ::weights::stage4_unit2_bn3_scale, {2e-05}, //
 ::weights::stage4_unit2_bn3_var, ::weights::stage4_unit2_bn3_bias, ::weights::stage4_unit2_conv3_weight, {0}, ::weights::stage4_unit2_bn2_mean, ::weights::stage4_unit2_bn2_scale, {2e-05}, //
 ::weights::stage4_unit2_bn2_var, ::weights::stage4_unit2_bn2_bias, ::weights::stage4_unit2_conv2_weight, {0}, ::weights::stage4_unit2_bn1_mean, ::weights::stage4_unit2_bn1_scale, {2e-05}, //
 ::weights::stage4_unit2_bn1_var, ::weights::stage4_unit2_bn1_bias, ::weights::stage4_unit2_conv1_weight, {0}, ::weights::stage4_unit1_sc_bn_mean, ::weights::stage4_unit1_sc_bn_scale, {2e-05}, //
 ::weights::stage4_unit1_sc_bn_var, ::weights::stage4_unit1_sc_bn_bias, ::weights::stage4_unit1_sc_weight, {0}, ::weights::stage3_unit6_bn3_mean, ::weights::stage3_unit6_bn3_scale, {2e-05}, //
 ::weights::stage3_unit6_bn3_var, ::weights::stage3_unit6_bn3_bias, ::weights::stage3_unit6_conv3_weight, {0}, ::weights::stage3_unit6_bn2_mean, ::weights::stage3_unit6_bn2_scale, {2e-05}, //
 ::weights::stage3_unit6_bn2_var, ::weights::stage3_unit6_bn2_bias, ::weights::stage3_unit6_conv2_weight, {0}, ::weights::stage3_unit6_bn1_mean, ::weights::stage3_unit6_bn1_scale, {2e-05}, //
 ::weights::stage3_unit6_bn1_var, ::weights::stage3_unit6_bn1_bias, ::weights::stage3_unit6_conv1_weight, {0}, ::weights::stage3_unit5_bn3_mean, ::weights::stage3_unit5_bn3_scale, {2e-05}, //
 ::weights::stage3_unit5_bn3_var, ::weights::stage3_unit5_bn3_bias, ::weights::stage3_unit5_conv3_weight, {0}, ::weights::stage3_unit5_bn2_mean, ::weights::stage3_unit5_bn2_scale, {2e-05}, //
 ::weights::stage3_unit5_bn2_var, ::weights::stage3_unit5_bn2_bias, ::weights::stage3_unit5_conv2_weight, {0}, ::weights::stage3_unit5_bn1_mean, ::weights::stage3_unit5_bn1_scale, {2e-05}, //
 ::weights::stage3_unit5_bn1_var, ::weights::stage3_unit5_bn1_bias, ::weights::stage3_unit5_conv1_weight, {0}, ::weights::stage3_unit4_bn3_mean, ::weights::stage3_unit4_bn3_scale, {2e-05}, //
 ::weights::stage3_unit4_bn3_var, ::weights::stage3_unit4_bn3_bias, ::weights::stage3_unit4_conv3_weight, {0}, ::weights::stage3_unit4_bn2_mean, ::weights::stage3_unit4_bn2_scale, {2e-05}, //
 ::weights::stage3_unit4_bn2_var, ::weights::stage3_unit4_bn2_bias, ::weights::stage3_unit4_conv2_weight, {0}, ::weights::stage3_unit4_bn1_mean, ::weights::stage3_unit4_bn1_scale, {2e-05}, //
 ::weights::stage3_unit4_bn1_var, ::weights::stage3_unit4_bn1_bias, ::weights::stage3_unit4_conv1_weight, {0}, ::weights::stage3_unit3_bn3_mean, ::weights::stage3_unit3_bn3_scale, {2e-05}, //
 ::weights::stage3_unit3_bn3_var, ::weights::stage3_unit3_bn3_bias, ::weights::stage3_unit3_conv3_weight, {0}, ::weights::stage3_unit3_bn2_mean, ::weights::stage3_unit3_bn2_scale, {2e-05}, //
 ::weights::stage3_unit3_bn2_var, ::weights::stage3_unit3_bn2_bias, ::weights::stage3_unit3_conv2_weight, {0}, ::weights::stage3_unit3_bn1_mean, ::weights::stage3_unit3_bn1_scale, {2e-05}, //
 ::weights::stage3_unit3_bn1_var, ::weights::stage3_unit3_bn1_bias, ::weights::stage3_unit3_conv1_weight, {0}, ::weights::stage3_unit2_bn3_mean, ::weights::stage3_unit2_bn3_scale, {2e-05}, //
 ::weights::stage3_unit2_bn3_var, ::weights::stage3_unit2_bn3_bias, ::weights::stage3_unit2_conv3_weight, {0}, ::weights::stage3_unit2_bn2_mean, ::weights::stage3_unit2_bn2_scale, {2e-05}, //
 ::weights::stage3_unit2_bn2_var, ::weights::stage3_unit2_bn2_bias, ::weights::stage3_unit2_conv2_weight, {0}, ::weights::stage3_unit2_bn1_mean, ::weights::stage3_unit2_bn1_scale, {2e-05}, //
 ::weights::stage3_unit2_bn1_var, ::weights::stage3_unit2_bn1_bias, ::weights::stage3_unit2_conv1_weight, {0}, ::weights::stage3_unit1_sc_bn_mean, ::weights::stage3_unit1_sc_bn_scale, {2e-05}, //
 ::weights::stage3_unit1_sc_bn_var, ::weights::stage3_unit1_sc_bn_bias, ::weights::stage3_unit1_sc_weight, {0}, ::weights::stage2_unit4_bn3_mean, ::weights::stage2_unit4_bn3_scale, {2e-05}, //
 ::weights::stage2_unit4_bn3_var, ::weights::stage2_unit4_bn3_bias, ::weights::stage2_unit4_conv3_weight, {0}, ::weights::stage2_unit4_bn2_mean, ::weights::stage2_unit4_bn2_scale, {2e-05}, //
 ::weights::stage2_unit4_bn2_var, ::weights::stage2_unit4_bn2_bias, ::weights::stage2_unit4_conv2_weight, {0}, ::weights::stage2_unit4_bn1_mean, ::weights::stage2_unit4_bn1_scale, {2e-05}, //
 ::weights::stage2_unit4_bn1_var, ::weights::stage2_unit4_bn1_bias, ::weights::stage2_unit4_conv1_weight, {0}, ::weights::stage2_unit3_bn3_mean, ::weights::stage2_unit3_bn3_scale, {2e-05}, //
 ::weights::stage2_unit3_bn3_var, ::weights::stage2_unit3_bn3_bias, ::weights::stage2_unit3_conv3_weight, {0}, ::weights::stage2_unit3_bn2_mean, ::weights::stage2_unit3_bn2_scale, {2e-05}, //
 ::weights::stage2_unit3_bn2_var, ::weights::stage2_unit3_bn2_bias, ::weights::stage2_unit3_conv2_weight, {0}, ::weights::stage2_unit3_bn1_mean, ::weights::stage2_unit3_bn1_scale, {2e-05}, //
 ::weights::stage2_unit3_bn1_var, ::weights::stage2_unit3_bn1_bias, ::weights::stage2_unit3_conv1_weight, {0}, ::weights::stage2_unit2_bn3_mean, ::weights::stage2_unit2_bn3_scale, {2e-05}, //
 ::weights::stage2_unit2_bn3_var, ::weights::stage2_unit2_bn3_bias, ::weights::stage2_unit2_conv3_weight, {0}, ::weights::stage2_unit2_bn2_mean, ::weights::stage2_unit2_bn2_scale, {2e-05}, //
 ::weights::stage2_unit2_bn2_var, ::weights::stage2_unit2_bn2_bias, ::weights::stage2_unit2_conv2_weight, {0}, ::weights::stage2_unit2_bn1_mean, ::weights::stage2_unit2_bn1_scale, {2e-05}, //
 ::weights::stage2_unit2_bn1_var, ::weights::stage2_unit2_bn1_bias, ::weights::stage2_unit2_conv1_weight, {0}, ::weights::stage2_unit1_sc_bn_mean, ::weights::stage2_unit1_sc_bn_scale, {2e-05}, //
 ::weights::stage2_unit1_sc_bn_var, ::weights::stage2_unit1_sc_bn_bias, ::weights::stage2_unit1_sc_weight, {0}, ::weights::stage1_unit3_bn3_mean, ::weights::stage1_unit3_bn3_scale, {2e-05}, //
 ::weights::stage1_unit3_bn3_var, ::weights::stage1_unit3_bn3_bias, ::weights::stage1_unit3_conv3_weight, {0}, ::weights::stage1_unit3_bn2_mean, ::weights::stage1_unit3_bn2_scale, {2e-05}, //
 ::weights::stage1_unit3_bn2_var, ::weights::stage1_unit3_bn2_bias, ::weights::stage1_unit3_conv2_weight, {0}, ::weights::stage1_unit3_bn1_mean, ::weights::stage1_unit3_bn1_scale, {2e-05}, //
 ::weights::stage1_unit3_bn1_var, ::weights::stage1_unit3_bn1_bias, ::weights::stage1_unit3_conv1_weight, {0}, ::weights::stage1_unit2_bn3_mean, ::weights::stage1_unit2_bn3_scale, {2e-05}, //
 ::weights::stage1_unit2_bn3_var, ::weights::stage1_unit2_bn3_bias, ::weights::stage1_unit2_conv3_weight, {0}, ::weights::stage1_unit2_bn2_mean, ::weights::stage1_unit2_bn2_scale, {2e-05}, //
 ::weights::stage1_unit2_bn2_var, ::weights::stage1_unit2_bn2_bias, ::weights::stage1_unit2_conv2_weight, {0}, ::weights::stage1_unit2_bn1_mean, ::weights::stage1_unit2_bn1_scale, {2e-05}, //
 ::weights::stage1_unit2_bn1_var, ::weights::stage1_unit2_bn1_bias, ::weights::stage1_unit2_conv1_weight, {0}, ::weights::stage1_unit1_sc_bn_mean, ::weights::stage1_unit1_sc_bn_scale, {2e-05}, //
 ::weights::stage1_unit1_sc_bn_var, ::weights::stage1_unit1_sc_bn_bias, ::weights::stage1_unit1_sc_weight, {0}, ::weights::bn0_mean, ::weights::bn0_scale, {2e-05}, //
 ::weights::bn0_var, ::weights::bn0_bias, ::weights::conv0_weight, ::weights::bn_data_mean, {2e-05}, //
 ::weights::bn_data_var, ::weights::bn_data_bias, ::weights::input_tensor, ::weights::stage1_unit1_bn3_mean, ::weights::stage1_unit1_bn3_scale, {2e-05}, //
 ::weights::stage1_unit1_bn3_var, ::weights::stage1_unit1_bn3_bias, ::weights::stage1_unit1_conv3_weight, {0}, ::weights::stage1_unit1_bn2_mean, ::weights::stage1_unit1_bn2_scale, {2e-05}, //
 ::weights::stage1_unit1_bn2_var, ::weights::stage1_unit1_bn2_bias, ::weights::stage1_unit1_conv2_weight, {0}, ::weights::stage1_unit1_bn1_mean, ::weights::stage1_unit1_bn1_scale, {2e-05}, //
 ::weights::stage1_unit1_bn1_var, ::weights::stage1_unit1_bn1_bias, ::weights::stage1_unit1_conv1_weight, ::weights::stage2_unit1_bn3_mean, ::weights::stage2_unit1_bn3_scale, {2e-05}, //
 ::weights::stage2_unit1_bn3_var, ::weights::stage2_unit1_bn3_bias, ::weights::stage2_unit1_conv3_weight, {0}, ::weights::stage2_unit1_bn2_mean, ::weights::stage2_unit1_bn2_scale, {2e-05}, //
 ::weights::stage2_unit1_bn2_var, ::weights::stage2_unit1_bn2_bias, ::weights::stage2_unit1_conv2_weight, {0}, ::weights::stage2_unit1_bn1_mean, ::weights::stage2_unit1_bn1_scale, {2e-05}, //
 ::weights::stage2_unit1_bn1_var, ::weights::stage2_unit1_bn1_bias, ::weights::stage2_unit1_conv1_weight, ::weights::stage3_unit1_bn3_mean, ::weights::stage3_unit1_bn3_scale, {2e-05}, //
 ::weights::stage3_unit1_bn3_var, ::weights::stage3_unit1_bn3_bias, ::weights::stage3_unit1_conv3_weight, {0}, ::weights::stage3_unit1_bn2_mean, ::weights::stage3_unit1_bn2_scale, {2e-05}, //
 ::weights::stage3_unit1_bn2_var, ::weights::stage3_unit1_bn2_bias, ::weights::stage3_unit1_conv2_weight, {0}, ::weights::stage3_unit1_bn1_mean, ::weights::stage3_unit1_bn1_scale, {2e-05}, //
 ::weights::stage3_unit1_bn1_var, ::weights::stage3_unit1_bn1_bias, ::weights::stage3_unit1_conv1_weight, ::weights::stage4_unit1_bn3_mean, ::weights::stage4_unit1_bn3_scale, {2e-05}, //
 ::weights::stage4_unit1_bn3_var, ::weights::stage4_unit1_bn3_bias, ::weights::stage4_unit1_conv3_weight, {0}, ::weights::stage4_unit1_bn2_mean, ::weights::stage4_unit1_bn2_scale, {2e-05}, //
 ::weights::stage4_unit1_bn2_var, ::weights::stage4_unit1_bn2_bias, ::weights::stage4_unit1_conv2_weight, {0}, ::weights::stage4_unit1_bn1_mean, ::weights::stage4_unit1_bn1_scale, {2e-05}, //
 ::weights::stage4_unit1_bn1_var, ::weights::stage4_unit1_bn1_bias, ::weights::stage4_unit1_conv1_weight};

 TestCaseVal ResNeXt50_Output = {                                //
{5.50651348e-05, 1.02617312e-04, 2.46670097e-04, 3.91743088e-05, //
 6.99452285e-05, 1.63910081e-04, 6.34302560e-05, 1.24250178e-03, //
 9.12103351e-05, 6.57027354e-04, 7.50391409e-05, 4.87675388e-05, //
 4.16809926e-05, 8.63198584e-05, 3.92582697e-05, 1.43127923e-04, //
 1.56804934e-04, 8.89936418e-05, 3.52463212e-05, 1.00429017e-04, //
 3.92404909e-05, 7.29392632e-05, 8.95423145e-05, 6.77058270e-05, //
 1.49653715e-04, 3.62389692e-05, 6.39260033e-05, 2.33376471e-04, //
 1.23167134e-04, 8.19247143e-05, 3.22844571e-05, 2.42171198e-04, //
 8.00494527e-05, 1.76369736e-04, 2.99566163e-04, 1.48946137e-05, //
 1.37829265e-04, 1.19199016e-04, 3.23461245e-05, 1.59191637e-04, //
 2.21417940e-04, 1.31364184e-04, 6.55783515e-05, 2.38502427e-04, //
 4.38263705e-05, 2.66033148e-05, 1.03360428e-04, 4.26316634e-04, //
 1.68444836e-04, 7.81720679e-04, 1.18548247e-04, 1.04540959e-04, //
 1.15592105e-04, 4.23703932e-05, 7.42222910e-05, 2.01620729e-04, //
 3.75579948e-05, 1.16308707e-04, 9.45794163e-05, 1.67487175e-04, //
 2.49268312e-04, 3.05931404e-04, 7.22366603e-05, 6.76214913e-05, //
 8.11922291e-05, 2.79286287e-05, 1.20124852e-04, 6.88342727e-04, //
 6.56307457e-05, 9.72160851e-05, 2.88040494e-04, 1.76985923e-04, //
 4.79123737e-05, 1.19690550e-04, 1.18747332e-04, 4.09332861e-05, //
 8.76334379e-05, 4.70570740e-05, 1.81484604e-04, 3.21576445e-05, //
 4.44524339e-05, 1.07538224e-04, 4.35279617e-05, 6.82876926e-05, //
 3.42438609e-04, 1.68470695e-04, 9.43551786e-05, 7.01807003e-05, //
 6.77266507e-05, 5.24716343e-05, 1.41347227e-05, 4.33502610e-05, //
 1.28405914e-03, 9.06537316e-05, 3.29739065e-04, 9.97233874e-05, //
 9.99594777e-05, 2.88563733e-05, 1.01147656e-04, 2.16952350e-04, //
 1.01970574e-04, 3.34532640e-04, 1.53291141e-04, 1.14909715e-04, //
 9.42115366e-05, 1.24888349e-04, 8.98118014e-05, 4.10383334e-04, //
 1.55923481e-04, 6.64273102e-05, 4.09763452e-05, 1.31662062e-04, //
 3.21878062e-04, 1.59708958e-04, 8.29689889e-05, 4.65095727e-05, //
 1.91711631e-04, 6.37966805e-05, 7.87222089e-05, 2.06328295e-05, //
 8.93222314e-05, 2.62730871e-04, 4.61446434e-05, 7.48246093e-05, //
 9.86962696e-05, 1.35667520e-04, 2.00343042e-04, 2.33085753e-04, //
 1.32532878e-04, 1.12119509e-04, 2.62032354e-05, 8.17872497e-05, //
 2.13730000e-05, 3.11197837e-05, 4.85567289e-05, 1.37340510e-04, //
 2.71682529e-05, 5.06595825e-05, 1.39254218e-04, 8.51916338e-05, //
 1.36571078e-04, 3.54321201e-05, 3.69027781e-04, 2.54284329e-04, //
 3.41456216e-05, 8.41956280e-05, 7.53403729e-05, 1.01250633e-04, //
 5.62501882e-05, 7.59732138e-05, 5.53068603e-05, 1.74174798e-04, //
 1.01869438e-04, 5.89568335e-05, 2.00481518e-04, 2.94833790e-05, //
 1.70780535e-04, 3.65029628e-05, 1.22360012e-04, 7.94315420e-05, //
 1.45051992e-04, 1.30967892e-05, 2.19650843e-04, 5.02051516e-05, //
 6.36913246e-05, 8.98543294e-05, 1.81516116e-05, 7.28391242e-05, //
 6.82645114e-05, 4.59113144e-05, 4.65038320e-05, 9.97621974e-05, //
 3.22279811e-04, 3.09213967e-04, 9.65814106e-05, 1.40965392e-04, //
 6.96798888e-05, 1.19168333e-04, 1.33797686e-04, 1.08341410e-04, //
 7.53334461e-05, 2.27722499e-04, 2.52729060e-05, 9.72651396e-05, //
 7.31147229e-05, 7.23605335e-05, 7.50116233e-05, 6.99029551e-05, //
 2.35148313e-04, 4.78852598e-05, 8.34442690e-05, 5.73854231e-05, //
 2.20458183e-04, 1.62025099e-04, 4.92939253e-05, 1.33813315e-04, //
 9.10512026e-05, 3.03952420e-05, 9.07273788e-05, 9.64296487e-05, //
 7.38863018e-05, 7.34672794e-05, 1.17346754e-04, 1.47409373e-04, //
 1.82924676e-04, 1.15614486e-04, 3.88978733e-05, 3.83411025e-05, //
 3.92934977e-04, 5.04204691e-05, 1.01534919e-04, 3.40186634e-05, //
 2.17473098e-05, 1.12378140e-04, 3.47267924e-05, 4.63854813e-05, //
 7.66294252e-04, 9.94596849e-05, 1.75847334e-03, 4.58103925e-04, //
 8.08934346e-05, 4.00575867e-04, 1.51941215e-03, 1.98546471e-03, //
 2.07055084e-04, 1.41256140e-03, 8.49361008e-04, 3.51254886e-04, //
 9.09373164e-04, 7.60942348e-05, 5.19888999e-05, 2.69552198e-04, //
 5.37973247e-04, 2.15650303e-03, 1.70945841e-05, 5.04216223e-05, //
 1.54493857e-04, 1.33685307e-05, 5.77397223e-05, 2.57650299e-05, //
 8.03465882e-05, 5.13939922e-05, 1.53278510e-04, 1.49028827e-04, //
 1.67119637e-04, 5.45622024e-05, 1.15603791e-04, 9.16508943e-05, //
 8.19005800e-06, 2.79107899e-05, 1.38869480e-04, 2.00008290e-05, //
 2.94471556e-05, 8.27159747e-05, 2.42930364e-05, 3.79639532e-05, //
 2.65428498e-05, 1.18604788e-04, 1.65083344e-04, 5.49912438e-05, //
 6.24617751e-05, 1.26309051e-05, 1.32558844e-05, 9.64647370e-06, //
 7.61347837e-05, 1.96937672e-05, 3.53599666e-04, 7.72335989e-05, //
 2.56922285e-05, 3.55304714e-04, 2.53106773e-05, 1.71899264e-05, //
 3.25414840e-05, 7.15067581e-05, 3.17686608e-05, 2.80359054e-05, //
 5.18567867e-05, 3.78256154e-05, 4.43574463e-05, 4.45734404e-05, //
 2.86199629e-05, 4.04950915e-05, 2.00760667e-04, 2.65494564e-05, //
 6.24432505e-05, 1.99369370e-05, 6.42208324e-05, 3.22917549e-05, //
 7.27730949e-05, 2.36593332e-05, 6.53546231e-05, 5.70644217e-04, //
 8.08442655e-06, 4.50222578e-05, 3.17059166e-05, 2.84890593e-05, //
 1.08413806e-04, 2.49261393e-05, 2.49472341e-05, 2.01168601e-04, //
 7.00183737e-05, 1.60827316e-04, 1.52782944e-04, 7.06171757e-03, //
 3.28888738e-04, 7.02612306e-05, 9.12767791e-05, 3.51598894e-04, //
 5.26121410e-04, 1.10219997e-04, 4.32022032e-04, 6.64730382e-04, //
 1.54186753e-04, 1.39889005e-03, 3.39535145e-05, 3.33688258e-05, //
 2.14250409e-04, 2.97154719e-03, 3.75806727e-03, 1.94884778e-03, //
 2.46832118e-04, 1.37656723e-04, 1.30461424e-03, 3.14504177e-05, //
 1.42637859e-04, 1.28144305e-03, 3.69892252e-04, 1.25091115e-04, //
 1.28285290e-04, 6.53934912e-05, 1.00054036e-04, 2.78346735e-04, //
 2.07128352e-04, 1.05401594e-03, 5.84872905e-04, 5.38997760e-04, //
 7.54083041e-04, 1.58093194e-03, 2.16555665e-04, 1.87245736e-04, //
 2.09846505e-04, 9.27958288e-04, 1.46167341e-03, 4.05575323e-04, //
 1.78994494e-03, 1.87182438e-03, 2.48552207e-03, 2.26513221e-04, //
 2.33262219e-03, 3.40378913e-03, 8.24162562e-04, 8.39091139e-04, //
 1.70037453e-03, 1.58856716e-03, 2.16830405e-03, 7.26378581e-04, //
 1.62251946e-03, 5.83136920e-04, 1.04289029e-04, 5.46432020e-05, //
 1.65146717e-04, 2.27209227e-03, 1.74194894e-04, 4.16665061e-05, //
 1.00549485e-04, 1.34166388e-04, 5.78325009e-04, 2.26678749e-04, //
 2.83952500e-03, 1.50784114e-02, 2.47245072e-03, 8.07863107e-05, //
 3.47876339e-04, 2.83706850e-05, 5.42661001e-04, 1.76058186e-03, //
 1.01403112e-03, 2.78839655e-03, 9.56183474e-04, 3.99633951e-04, //
 4.57454298e-05, 5.88979106e-04, 1.06490166e-04, 6.08642986e-05, //
 6.90478191e-05, 2.19152033e-04, 1.48267995e-04, 7.96065287e-05, //
 1.45985672e-04, 5.60633343e-05, 1.08226886e-04, 2.10906132e-04, //
 8.98581056e-05, 1.24069658e-04, 5.60897497e-05, 1.21006182e-04, //
 3.68356734e-04, 1.00525806e-03, 2.07061006e-04, 2.75948289e-04, //
 3.27963207e-04, 8.20670757e-05, 4.32355300e-05, 4.11306028e-05, //
 1.76734084e-05, 4.47577513e-05, 5.32666672e-05, 5.05499593e-05, //
 1.50118707e-04, 1.93006956e-04, 9.86038867e-05, 3.82765247e-05, //
 2.32784529e-04, 1.48555948e-04, 6.35119097e-04, 6.05335044e-05, //
 2.74616061e-04, 1.59379470e-05, 4.66534402e-05, 5.09824422e-05, //
 1.60490861e-04, 5.32801823e-05, 3.21617059e-04, 1.55422225e-04, //
 1.82472009e-04, 3.71814822e-04, 4.68673788e-05, 4.72050197e-05, //
 2.40707250e-05, 4.73954278e-05, 4.90685088e-05, 2.20751044e-05, //
 2.32684852e-05, 3.84951491e-05, 6.09538110e-05, 4.82611504e-05, //
 5.92963697e-05, 1.81555060e-05, 1.56208815e-04, 2.03650387e-04, //
 8.16868487e-05, 1.86768040e-04, 1.87426078e-04, 5.70774253e-04, //
 1.32845948e-03, 1.41835568e-04, 1.24300728e-04, 2.36096130e-05, //
 4.11070359e-05, 7.87690340e-04, 2.09520163e-04, 8.11905658e-04, //
 1.12025555e-04, 1.45197511e-04, 2.01492905e-04, 7.92027931e-05, //
 1.58670329e-04, 1.20059238e-04, 8.06942175e-04, 2.98377199e-05, //
 6.40970829e-05, 1.57061531e-05, 5.60405897e-04, 4.93233165e-05, //
 2.78816442e-04, 7.52787091e-05, 5.64849433e-05, 1.31082226e-04, //
 1.23113525e-04, 8.99840379e-05, 1.36112867e-04, 2.03118936e-04, //
 2.10770304e-05, 1.61439850e-04, 4.27498526e-05, 7.77094756e-05, //
 2.93797784e-04, 4.55639383e-04, 4.87608428e-04, 4.10539535e-04, //
 4.60162773e-05, 2.24805462e-05, 2.42742790e-05, 7.52446661e-03, //
 2.10448867e-03, 8.20602290e-05, 7.62794371e-05, 5.00283146e-04, //
 3.44632339e-04, 2.53988139e-04, 2.32561957e-04, 2.95337231e-05, //
 9.54450588e-05, 1.32254470e-04, 2.86223512e-04, 1.19820630e-04, //
 2.04534721e-04, 3.16248508e-04, 4.90473649e-05, 2.02537584e-03, //
 1.91606727e-04, 9.19290469e-04, 1.15205161e-03, 2.05204403e-03, //
 1.71233594e-04, 2.97986495e-04, 3.93450209e-05, 4.94629284e-03, //
 1.29229128e-02, 9.73040645e-04, 9.60797770e-04, 4.12984198e-04, //
 4.05546132e-04, 6.77712029e-04, 5.86606504e-04, 1.06241938e-03, //
 1.26756507e-03, 2.62556277e-04, 4.77376441e-03, 6.96427189e-04, //
 1.77508744e-04, 5.48640080e-03, 4.43028919e-02, 4.17495437e-04, //
 5.41685103e-03, 6.43935055e-05, 2.58475979e-04, 5.44616021e-03, //
 2.33245082e-03, 6.22467196e-04, 6.43143570e-03, 3.57492565e-04, //
 3.15670564e-04, 1.53740763e-03, 8.18114728e-04, 1.01060933e-03, //
 2.49605044e-04, 2.42744762e-04, 1.29550943e-04, 2.86779541e-04, //
 2.37691667e-04, 2.50628538e-04, 1.13072549e-03, 2.98989326e-04, //
 5.43168164e-04, 2.11205948e-02, 1.70355575e-04, 6.11648138e-04, //
 6.47807843e-04, 4.77366615e-04, 7.24872007e-05, 8.93447199e-04, //
 2.00939388e-03, 1.26048772e-05, 1.33022222e-05, 6.62002712e-05, //
 1.24316124e-04, 2.49749311e-04, 1.63337012e-04, 4.74887871e-04, //
 2.04025279e-03, 2.22008978e-03, 3.24379507e-04, 9.50702743e-05, //
 1.55890040e-04, 9.69421380e-05, 2.83406680e-05, 3.15151519e-05, //
 4.19822500e-05, 2.44904484e-04, 7.99890142e-04, 2.23370109e-04, //
 2.05508812e-04, 7.18913507e-04, 6.81551034e-03, 3.71976552e-04, //
 7.37025766e-05, 2.86007556e-03, 4.94066444e-05, 1.03671546e-03, //
 7.31813139e-04, 1.47625036e-03, 1.54796941e-03, 1.84437726e-03, //
 3.43662687e-03, 7.02592544e-03, 1.23484642e-03, 8.28685661e-05, //
 9.01092426e-04, 6.34198834e-04, 1.52501352e-02, 1.20926015e-02, //
 4.10964973e-02, 3.94913880e-03, 6.23491185e-04, 7.27712541e-05, //
 3.58212652e-04, 1.57161849e-04, 2.03033048e-03, 4.09483968e-04, //
 2.13245148e-04, 1.14261216e-04, 8.34409322e-04, 6.34408716e-05, //
 9.97561682e-03, 8.07190008e-05, 4.75242821e-04, 1.38821619e-04, //
 4.34300455e-04, 1.59171619e-03, 6.25098648e-04, 1.59187324e-03, //
 4.90568345e-05, 4.18156233e-05, 5.76968378e-05, 5.60885492e-05, //
 6.14894961e-05, 3.27414891e-05, 5.30396501e-05, 9.14774646e-05, //
 1.11003181e-04, 7.40332529e-04, 2.06959230e-04, 5.07793273e-04, //
 2.57424836e-04, 7.66463054e-05, 4.88075311e-04, 8.80996755e-04, //
 8.78752558e-04, 2.25926400e-04, 2.02957250e-04, 3.83907522e-04, //
 5.89826494e-04, 7.01884623e-04, 1.65878492e-03, 1.24878832e-04, //
 6.30789844e-04, 1.56037218e-03, 3.58333782e-04, 2.87904695e-04, //
 5.17134176e-05, 5.96352183e-05, 6.38906204e-05, 2.81296292e-04, //
 7.47291924e-05, 9.93925496e-04, 1.76499889e-03, 1.59316492e-04, //
 5.71295204e-05, 6.77147706e-04, 1.13314964e-01, 1.32879848e-03, //
 5.96576894e-04, 6.18986378e-04, 2.06136261e-04, 1.20169258e-04, //
 2.07031786e-04, 4.99625130e-05, 5.42980211e-04, 9.56030708e-05, //
 1.84259797e-03, 1.57670846e-04, 3.57496989e-04, 5.66852570e-04, //
 3.26671034e-05, 7.59384129e-05, 2.12945812e-03, 1.97209092e-03, //
 3.42535292e-04, 1.31485125e-04, 7.10419670e-04, 1.79181239e-04, //
 6.10288756e-04, 2.50733632e-04, 2.61914945e-04, 9.33641160e-04, //
 2.01532766e-05, 3.76174517e-04, 1.05448944e-05, 5.92079596e-05, //
 5.51982084e-04, 5.67563620e-05, 1.27923471e-04, 4.08192427e-05, //
 2.43202856e-04, 8.68358620e-05, 2.40196350e-05, 2.08529818e-05, //
 1.47632716e-04, 2.45956704e-04, 2.06049983e-04, 6.51782248e-05, //
 3.40541737e-04, 6.49441135e-05, 2.80586677e-03, 8.12003214e-04, //
 9.08019047e-06, 4.00714169e-04, 4.33226785e-04, 1.81725336e-04, //
 2.35561994e-04, 3.77048709e-04, 4.72070024e-05, 1.36533317e-05, //
 5.11362414e-05, 4.84773336e-05, 3.37208512e-05, 8.56403130e-06, //
 2.32690854e-05, 1.42548175e-04, 1.04763221e-05, 4.66438127e-04, //
 1.04869236e-04, 5.47190793e-05, 4.53753273e-05, 2.27020464e-05, //
 1.11239613e-04, 1.00066641e-03, 2.81959256e-05, 2.21273949e-05, //
 4.50442021e-05, 5.12982078e-04, 2.27454530e-05, 3.62212413e-05, //
 5.48365970e-05, 7.08978449e-04, 1.08527187e-04, 2.51991791e-04, //
 2.23578754e-04, 1.15499878e-03, 1.00881625e-04, 6.94529619e-04, //
 9.32984753e-04, 4.40575677e-05, 3.77983000e-04, 7.28300583e-05, //
 1.70398387e-04, 1.78163900e-04, 2.78442149e-05, 3.57371726e-04, //
 1.15301664e-04, 3.45061359e-04, 1.24860715e-04, 5.64006514e-05, //
 1.53906745e-04, 1.50790365e-04, 1.86125428e-04, 8.34881735e-04, //
 1.40911166e-03, 1.30345070e-04, 1.74031928e-04, 1.89815342e-04, //
 8.14285420e-04, 6.76126510e-05, 1.94237669e-04, 1.24635815e-04, //
 3.37034115e-04, 1.23645610e-03, 7.44269200e-05, 3.91700509e-04, //
 1.91001466e-03, 1.97574758e-04, 2.91971919e-05, 5.41322806e-04, //
 2.73686339e-04, 1.84961900e-05, 2.92357407e-03, 6.56473348e-05, //
 2.04388969e-04, 2.91154283e-04, 6.43363237e-05, 8.27608048e-04, //
 6.12238131e-04, 3.76588898e-04, 2.87955627e-05, 3.23486025e-03, //
 7.07281288e-04, 9.10406161e-05, 2.33193132e-04, 2.22620583e-04, //
 2.09894893e-03, 1.26557861e-04, 2.97272345e-04, 9.69925386e-05, //
 8.22860748e-05, 6.70238165e-04, 1.01737351e-04, 8.48724216e-04, //
 3.02929751e-04, 5.87718860e-05, 1.32703746e-04, 2.98623950e-03, //
 8.05277377e-05, 1.37390301e-03, 1.12459944e-04, 8.81121145e-04, //
 2.16899836e-03, 5.34153124e-03, 2.45119864e-03, 1.65222096e-04, //
 2.15228793e-04, 3.33108357e-04, 1.08212946e-04, 2.44365330e-03, //
 1.15675801e-04, 7.19235468e-05, 1.82422344e-03, 4.61491960e-04, //
 8.80191801e-04, 1.37190495e-04, 8.22364236e-04, 5.77352330e-05, //
 9.16919817e-05, 1.86401661e-02, 2.30479112e-04, 7.78569811e-05, //
 8.73598270e-04, 2.53834267e-04, 4.91340004e-04, 3.25640722e-04, //
 2.15775595e-04, 1.00230864e-04, 4.42220335e-04, 1.10167242e-03, //
 3.04726505e-04, 7.59828836e-05, 1.35231239e-03, 2.01572941e-04, //
 1.11120215e-04, 9.22142863e-05, 9.27345187e-04, 3.12938821e-03, //
 4.61115415e-04, 1.43291472e-04, 2.40639318e-03, 1.02862359e-04, //
 1.97239176e-04, 1.47885934e-04, 1.91495012e-04, 2.38463064e-04, //
 4.64516971e-03, 1.01100974e-04, 4.00936638e-04, 5.40192887e-05, //
 4.04003185e-05, 1.48531375e-03, 5.24300158e-05, 1.55200338e-04, //
 1.50174776e-03, 5.15846128e-04, 4.23058955e-04, 2.42272599e-05, //
 2.78038867e-02, 7.18823692e-04, 1.17162090e-04, 4.72995394e-04, //
 2.88907671e-04, 2.25821612e-04, 2.79603829e-03, 1.37790310e-04, //
 3.34549043e-03, 2.96251592e-03, 6.33845106e-04, 1.30221000e-04, //
 1.29934328e-04, 1.03611790e-04, 5.01796277e-03, 1.84821722e-04, //
 1.10045646e-03, 1.10013958e-03, 1.71047112e-04, 4.76097950e-04, //
 1.16152914e-04, 1.60165795e-03, 2.89000105e-04, 1.83430588e-04, //
 1.24591310e-03, 1.38745454e-04, 1.19520421e-03, 8.02618088e-05, //
 4.39188996e-04, 1.97306690e-05, 6.39256416e-03, 3.47314635e-04, //
 9.82591882e-05, 1.77403563e-04, 1.02508398e-04, 3.61695020e-05, //
 2.29998911e-03, 5.75470389e-04, 5.71390519e-05, 3.32290540e-04, //
 1.07335662e-04, 4.76319779e-04, 1.84238015e-03, 1.58976298e-03, //
 1.35935217e-04, 2.38239969e-04, 5.37582964e-04, 7.46710284e-05, //
 9.47919581e-03, 6.56067510e-04, 9.26583260e-02, 4.57655825e-03, //
 3.06270056e-04, 8.28241638e-04, 5.00521390e-03, 3.48211091e-04, //
 7.33081979e-05, 2.26662974e-04, 4.92043793e-04, 7.62404583e-04, //
 3.17437953e-05, 7.69545499e-04, 7.93307030e-04, 1.09652217e-04, //
 9.90325861e-05, 1.39600408e-04, 1.62005180e-03, 6.28282942e-05, //
 4.50995460e-04, 2.53953651e-04, 6.23940257e-04, 2.44366843e-03, //
 2.95464997e-04, 7.94046136e-05, 6.73338363e-05, 1.47384498e-03, //
 3.12933029e-04, 8.69485229e-05, 1.00936915e-03, 5.06406010e-04, //
 2.58427812e-03, 1.22012768e-03, 6.07763417e-04, 7.06228166e-05, //
 8.88726616e-04, 3.26876558e-04, 1.02180355e-04, 4.65980382e-04, //
 7.85218508e-05, 2.09805978e-04, 5.34053506e-05, 1.73884863e-03, //
 5.11434591e-05, 9.97981624e-05, 4.90399776e-04, 1.38578500e-04, //
 2.91935459e-04, 3.74878946e-05, 2.51602643e-04, 1.59118127e-03, //
 1.61901218e-04, 5.57693456e-05, 5.08740370e-04, 3.36556659e-05, //
 8.48493946e-05, 2.08716909e-03, 4.55337926e-04, 3.14865191e-03, //
 3.82046128e-04, 6.15931713e-05, 1.45739745e-04, 3.60452977e-04, //
 1.24838704e-03, 1.67804785e-04, 1.95581582e-03, 2.47975142e-04, //
 2.73212860e-03, 9.36546177e-03, 9.15271696e-04, 2.49455101e-04, //
 4.79117793e-04, 5.38238783e-05, 1.10092515e-04, 9.67487795e-05, //
 1.92187308e-05, 3.47932393e-04, 2.37666187e-04, 1.36006158e-03, //
 4.79851224e-05, 2.94038706e-04, 7.03297381e-04, 6.57579079e-02, //
 4.80025410e-05, 5.72817284e-04, 1.39874988e-03, 2.17847060e-03, //
 3.13172022e-05, 8.91306045e-05, 6.42098195e-04, 2.54075043e-03, //
 4.12981608e-05, 3.83167586e-04, 4.96551802e-04, 1.38449378e-03, //
 2.27733632e-03, 3.56824778e-04, 1.36296294e-04, 1.73579916e-04}};

  TestCasePairs testcase_pairs = {{ResNeXt50_WeightsInputs, ResNeXt50_Output}};

  ResNeXTTestSpec spec = GetParam();

  HloModuleConfig cfg;

  std::unique_ptr<VerifiedHloModule> hlo_module = absl::make_unique<VerifiedHloModule>(
      "module", cfg, false, false, nullptr);

std::string hlo_text = R"(HloModule cluster_0__XlaCompiledKernel_true__XlaNumConstantArgs_70__XlaNumResourceArgs_0_.2963

%max_F32.1049 (lhs.1050: f32[], rhs.1051: f32[]) -> f32[] {
  %lhs.1050 = f32[] parameter(0)
  %rhs.1051 = f32[] parameter(1)
  ROOT %maximum.1052 = f32[] maximum(f32[] %lhs.1050, f32[] %rhs.1051)
}

%add_F32.2930 (lhs.2931: f32[], rhs.2932: f32[]) -> f32[] {
  %lhs.2931 = f32[] parameter(0)
  %rhs.2932 = f32[] parameter(1)
  ROOT %add.2933 = f32[] add(f32[] %lhs.2931, f32[] %rhs.2932)
}

ENTRY %cluster_0__XlaCompiledKernel_true__XlaNumConstantArgs_70__XlaNumResourceArgs_0_.2963 (arg0.1: f32[3], arg1.2: f32[512], arg2.3: f32[64], arg3.4: f32[128], arg4.5: f32[3,3,16,512], arg5.6: f32[256], arg6.7: f32[3,3,4,128], arg7.8: f32[128], arg8.9: f32[256], arg9.10: f32[128], arg10.11: f32[3,3,4,128], arg11.12: f32[128], arg12.13: f32[512], arg13.14: f32[256], arg14.15: f32[128], arg15.16: f32[3,3,32,1024], arg16.17: f32[3,3,4,128], arg17.18: f32[1024], arg18.19: f32[128], arg19.20: f32[256], arg20.21: f32[256], arg21.22: f32[512], arg22.23: f32[1024], arg23.24: f32[3,3,8,256], arg24.25: f32[256], arg25.26: f32[512], arg26.27: f32[2048], arg27.28: f32[256], arg28.29: f32[3,3,8,256], arg29.30: f32[256], arg30.31: f32[3,3,32,1024], arg31.32: f32[512], arg32.33: f32[256], arg33.34: f32[3,3,8,256], arg34.35: f32[256], arg35.36: f32[512], arg36.37: f32[256], arg37.38: f32[1024], arg38.39: f32[3,3,8,256], arg39.40: f32[256], arg40.41: f32[512], arg41.42: f32[512], arg42.43: f32[2048], arg43.44: f32[1024], arg44.45: f32[1024], arg45.46: f32[3,3,16,512], arg46.47: f32[512], arg47.48: f32[1024], arg48.49: f32[1024], arg49.50: f32[512], arg50.51: f32[3,3,16,512], arg51.52: f32[512], arg52.53: f32[3,3,32,1024], arg53.54: f32[1024], arg54.55: f32[512], arg55.56: f32[3,3,16,512], arg56.57: f32[512], arg57.58: f32[1024], arg58.59: f32[1024], arg59.60: f32[512], arg60.61: f32[3,3,16,512], arg61.62: f32[512], arg62.63: f32[1024], arg63.64: f32[2048], arg64.65: f32[2048], arg65.66: f32[512], arg66.67: f32[3,3,16,512], arg67.68: f32[512], arg68.69: f32[1024], arg69.70: f32[1024], arg70.71: f32[3], arg71.72: f32[512], arg72.73: f32[64], arg73.74: f32[128], arg74.75: f32[256], arg75.76: f32[128], arg76.77: f32[256], arg77.78: f32[128], arg78.79: f32[128], arg79.80: f32[512], arg80.81: f32[256], arg81.82: f32[128], arg82.83: f32[1024], arg83.84: f32[128], arg84.85: f32[256], arg85.86: f32[256], arg86.87: f32[512], arg87.88: f32[1024], arg88.89: f32[256], arg89.90: f32[512], arg90.91: f32[2048], arg91.92: f32[256], arg92.93: f32[256], arg93.94: f32[512], arg94.95: f32[256], arg95.96: f32[256], arg96.97: f32[512], arg97.98: f32[256], arg98.99: f32[1024], arg99.100: f32[256], arg100.101: f32[512], arg101.102: f32[512], arg102.103: f32[2048], arg103.104: f32[1024], arg104.105: f32[1024], arg105.106: f32[512], arg106.107: f32[1024], arg107.108: f32[1024], arg108.109: f32[512], arg109.110: f32[512], arg110.111: f32[1024], arg111.112: f32[512], arg112.113: f32[512], arg113.114: f32[1024], arg114.115: f32[1024], arg115.116: f32[512], arg116.117: f32[512], arg117.118: f32[1024], arg118.119: f32[2048], arg119.120: f32[2048], arg120.121: f32[512], arg121.122: f32[512], arg122.123: f32[1024], arg123.124: f32[1024], arg124.125: f32[3], arg125.126: f32[512], arg126.127: f32[64], arg127.128: f32[128], arg128.129: f32[256], arg129.130: f32[128], arg130.131: f32[256], arg131.132: f32[128], arg132.133: f32[128], arg133.134: f32[512], arg134.135: f32[256], arg135.136: f32[128], arg136.137: f32[1024], arg137.138: f32[128], arg138.139: f32[256], arg139.140: f32[256], arg140.141: f32[512], arg141.142: f32[1024], arg142.143: f32[256], arg143.144: f32[512], arg144.145: f32[2048], arg145.146: f32[256], arg146.147: f32[256], arg147.148: f32[512], arg148.149: f32[256], arg149.150: f32[256], arg150.151: f32[512], arg151.152: f32[256], arg152.153: f32[1024], arg153.154: f32[256], arg154.155: f32[512], arg155.156: f32[512], arg156.157: f32[2048], arg157.158: f32[1024], arg158.159: f32[1024], arg159.160: f32[512], arg160.161: f32[1024], arg161.162: f32[1024], arg162.163: f32[512], arg163.164: f32[512], arg164.165: f32[1024], arg165.166: f32[512], arg166.167: f32[512], arg167.168: f32[1024], arg168.169: f32[1024], arg169.170: f32[512], arg170.171: f32[512], arg171.172: f32[1024], arg172.173: f32[2048], arg173.174: f32[2048], arg174.175: f32[512], arg175.176: f32[512], arg176.177: f32[1024], arg177.178: f32[1024], arg178.179: f32[512], arg179.180: f32[64], arg180.181: f32[128], arg181.182: f32[256], arg182.183: f32[128], arg183.184: f32[256], arg184.185: f32[128], arg185.186: f32[128], arg186.187: f32[512], arg187.188: f32[256], arg188.189: f32[128], arg189.190: f32[1024], arg190.191: f32[128], arg191.192: f32[256], arg192.193: f32[256], arg193.194: f32[512], arg194.195: f32[1024], arg195.196: f32[256], arg196.197: f32[512], arg197.198: f32[2048], arg198.199: f32[256], arg199.200: f32[256], arg200.201: f32[512], arg201.202: f32[256], arg202.203: f32[256], arg203.204: f32[512], arg204.205: f32[256], arg205.206: f32[1024], arg206.207: f32[256], arg207.208: f32[512], arg208.209: f32[512], arg209.210: f32[2048], arg210.211: f32[1024], arg211.212: f32[1024], arg212.213: f32[512], arg213.214: f32[1024], arg214.215: f32[1024], arg215.216: f32[512], arg216.217: f32[512], arg217.218: f32[1024], arg218.219: f32[512], arg219.220: f32[512], arg220.221: f32[1024], arg221.222: f32[1024], arg222.223: f32[512], arg223.224: f32[512], arg224.225: f32[1024], arg225.226: f32[2048], arg226.227: f32[2048], arg227.228: f32[512], arg228.229: f32[512], arg229.230: f32[1024], arg230.231: f32[1024], arg231.232: f32[7,7,3,64], arg232.233: f32[1,1,64,128], arg233.234: f32[1,1,64,256], arg234.235: f32[1,1,128,256], arg235.236: f32[1,1,256,128], arg236.237: f32[1,1,128,256], arg237.238: f32[1,1,256,128], arg238.239: f32[1,1,128,256], arg239.240: f32[1,1,256,256], arg240.241: f32[1,1,256,512], arg241.242: f32[1,1,256,512], arg242.243: f32[1,1,512,256], arg243.244: f32[1,1,256,512], arg244.245: f32[1,1,512,256], arg245.246: f32[1,1,256,512], arg246.247: f32[1,1,512,256], arg247.248: f32[1,1,256,512], arg248.249: f32[1,1,512,512], arg249.250: f32[1,1,512,1024], arg250.251: f32[1,1,512,1024], arg251.252: f32[1,1,1024,512], arg252.253: f32[1,1,512,1024], arg253.254: f32[1,1,1024,512], arg254.255: f32[1,1,512,1024], arg255.256: f32[1,1,1024,512], arg256.257: f32[1,1,512,1024], arg257.258: f32[1,1,1024,512], arg258.259: f32[1,1,512,1024], arg259.260: f32[1,1,1024,512], arg260.261: f32[1,1,512,1024], arg261.262: f32[1,1,1024,1024], arg262.263: f32[1,1,1024,2048], arg263.264: f32[1,1,1024,2048], arg264.265: f32[1,1,2048,1024], arg265.266: f32[1,1,1024,2048], arg266.267: f32[1,1,2048,1024], arg267.268: f32[1,1,1024,2048], arg268.269: f32[1,224,224,3]) -> f32[1,2048] {
  %constant.2923 = f32[] constant(0), metadata={op_type="Relu" op_name="stage4_unit3_relu"}
  %broadcast.2924 = f32[1,7,7,2048]{3,2,1,0} broadcast(f32[] %constant.2923), dimensions={}, metadata={op_type="Relu" op_name="stage4_unit3_relu"}
  %constant.2779 = f32[] constant(0), metadata={op_type="Relu" op_name="stage4_unit2_relu"}
  %broadcast.2780 = f32[1,7,7,2048]{3,2,1,0} broadcast(f32[] %constant.2779), dimensions={}, metadata={op_type="Relu" op_name="stage4_unit2_relu"}
  %constant.2667 = f32[] constant(0), metadata={op_type="Relu" op_name="stage4_unit1_relu"}
  %broadcast.2668 = f32[1,7,7,2048]{3,2,1,0} broadcast(f32[] %constant.2667), dimensions={}, metadata={op_type="Relu" op_name="stage4_unit1_relu"}
  %arg42.43 = f32[2048]{0} parameter(42), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.312 = f32[2048]{0} reshape(f32[2048]{0} %arg42.43)
  %constant.2644 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage4_unit1_bn3/add"}
  %broadcast.2645 = f32[2048]{0} broadcast(f32[] %constant.2644), dimensions={}, metadata={op_type="AddV2" op_name="stage4_unit1_bn3/add"}
  %add.2646 = f32[2048]{0} add(f32[2048]{0} %reshape.312, f32[2048]{0} %broadcast.2645), metadata={op_type="AddV2" op_name="stage4_unit1_bn3/add"}
  %rsqrt.2647 = f32[2048]{0} rsqrt(f32[2048]{0} %add.2646), metadata={op_type="Rsqrt" op_name="stage4_unit1_bn3/Rsqrt"}
  %arg102.103 = f32[2048]{0} parameter(102), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.372 = f32[2048]{0} reshape(f32[2048]{0} %arg102.103)
  %multiply.2648 = f32[2048]{0} multiply(f32[2048]{0} %rsqrt.2647, f32[2048]{0} %reshape.372), metadata={op_type="Mul" op_name="stage4_unit1_bn3/mul"}
  %broadcast.2649 = f32[1,7,7,2048]{3,2,1,0} broadcast(f32[2048]{0} %multiply.2648), dimensions={3}, metadata={op_type="Mul" op_name="stage4_unit1_bn3/mul_1"}
  %constant.2640 = f32[] constant(0), metadata={op_type="Relu" op_name="stage4_unit1_relu2"}
  %broadcast.2641 = f32[1,7,7,1024]{3,2,1,0} broadcast(f32[] %constant.2640), dimensions={}, metadata={op_type="Relu" op_name="stage4_unit1_relu2"}
  %constant.2559 = f32[] constant(0), metadata={op_type="Relu" op_name="stage4_unit1_relu1"}
  %broadcast.2560 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[] %constant.2559), dimensions={}, metadata={op_type="Relu" op_name="stage4_unit1_relu1"}
  %arg22.23 = f32[1024]{0} parameter(22), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.292 = f32[1024]{0} reshape(f32[1024]{0} %arg22.23)
  %constant.2548 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage4_unit1_bn1/add"}
  %broadcast.2549 = f32[1024]{0} broadcast(f32[] %constant.2548), dimensions={}, metadata={op_type="AddV2" op_name="stage4_unit1_bn1/add"}
  %add.2550 = f32[1024]{0} add(f32[1024]{0} %reshape.292, f32[1024]{0} %broadcast.2549), metadata={op_type="AddV2" op_name="stage4_unit1_bn1/add"}
  %rsqrt.2551 = f32[1024]{0} rsqrt(f32[1024]{0} %add.2550), metadata={op_type="Rsqrt" op_name="stage4_unit1_bn1/Rsqrt"}
  %arg87.88 = f32[1024]{0} parameter(87), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.357 = f32[1024]{0} reshape(f32[1024]{0} %arg87.88)
  %multiply.2552 = f32[1024]{0} multiply(f32[1024]{0} %rsqrt.2551, f32[1024]{0} %reshape.357), metadata={op_type="Mul" op_name="stage4_unit1_bn1/mul"}
  %broadcast.2553 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %multiply.2552), dimensions={3}, metadata={op_type="Mul" op_name="stage4_unit1_bn1/mul_1"}
  %constant.2543 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit6_relu"}
  %broadcast.2544 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[] %constant.2543), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit6_relu"}
  %constant.2431 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit5_relu"}
  %broadcast.2432 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[] %constant.2431), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit5_relu"}
  %constant.2319 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit4_relu"}
  %broadcast.2320 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[] %constant.2319), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit4_relu"}
  %constant.2207 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit3_relu"}
  %broadcast.2208 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[] %constant.2207), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit3_relu"}
  %constant.2095 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit2_relu"}
  %broadcast.2096 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[] %constant.2095), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit2_relu"}
  %constant.1983 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit1_relu"}
  %broadcast.1984 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[] %constant.1983), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit1_relu"}
  %arg47.48 = f32[1024]{0} parameter(47), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.317 = f32[1024]{0} reshape(f32[1024]{0} %arg47.48)
  %constant.1960 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit1_bn3/add"}
  %broadcast.1961 = f32[1024]{0} broadcast(f32[] %constant.1960), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit1_bn3/add"}
  %add.1962 = f32[1024]{0} add(f32[1024]{0} %reshape.317, f32[1024]{0} %broadcast.1961), metadata={op_type="AddV2" op_name="stage3_unit1_bn3/add"}
  %rsqrt.1963 = f32[1024]{0} rsqrt(f32[1024]{0} %add.1962), metadata={op_type="Rsqrt" op_name="stage3_unit1_bn3/Rsqrt"}
  %arg106.107 = f32[1024]{0} parameter(106), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.376 = f32[1024]{0} reshape(f32[1024]{0} %arg106.107)
  %multiply.1964 = f32[1024]{0} multiply(f32[1024]{0} %rsqrt.1963, f32[1024]{0} %reshape.376), metadata={op_type="Mul" op_name="stage3_unit1_bn3/mul"}
  %broadcast.1965 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %multiply.1964), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit1_bn3/mul_1"}
  %constant.1956 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit1_relu2"}
  %broadcast.1957 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[] %constant.1956), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit1_relu2"}
  %constant.1875 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit1_relu1"}
  %broadcast.1876 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[] %constant.1875), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit1_relu1"}
  %arg41.42 = f32[512]{0} parameter(41), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.311 = f32[512]{0} reshape(f32[512]{0} %arg41.42)
  %constant.1864 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit1_bn1/add"}
  %broadcast.1865 = f32[512]{0} broadcast(f32[] %constant.1864), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit1_bn1/add"}
  %add.1866 = f32[512]{0} add(f32[512]{0} %reshape.311, f32[512]{0} %broadcast.1865), metadata={op_type="AddV2" op_name="stage3_unit1_bn1/add"}
  %rsqrt.1867 = f32[512]{0} rsqrt(f32[512]{0} %add.1866), metadata={op_type="Rsqrt" op_name="stage3_unit1_bn1/Rsqrt"}
  %arg101.102 = f32[512]{0} parameter(101), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.371 = f32[512]{0} reshape(f32[512]{0} %arg101.102)
  %multiply.1868 = f32[512]{0} multiply(f32[512]{0} %rsqrt.1867, f32[512]{0} %reshape.371), metadata={op_type="Mul" op_name="stage3_unit1_bn1/mul"}
  %broadcast.1869 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.1868), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit1_bn1/mul_1"}
  %constant.1859 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit4_relu"}
  %broadcast.1860 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[] %constant.1859), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit4_relu"}
  %constant.1747 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit3_relu"}
  %broadcast.1748 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[] %constant.1747), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit3_relu"}
  %constant.1635 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit2_relu"}
  %broadcast.1636 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[] %constant.1635), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit2_relu"}
  %constant.1523 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit1_relu"}
  %broadcast.1524 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[] %constant.1523), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit1_relu"}
  %arg25.26 = f32[512]{0} parameter(25), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.295 = f32[512]{0} reshape(f32[512]{0} %arg25.26)
  %constant.1500 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit1_bn3/add"}
  %broadcast.1501 = f32[512]{0} broadcast(f32[] %constant.1500), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit1_bn3/add"}
  %add.1502 = f32[512]{0} add(f32[512]{0} %reshape.295, f32[512]{0} %broadcast.1501), metadata={op_type="AddV2" op_name="stage2_unit1_bn3/add"}
  %rsqrt.1503 = f32[512]{0} rsqrt(f32[512]{0} %add.1502), metadata={op_type="Rsqrt" op_name="stage2_unit1_bn3/Rsqrt"}
  %arg89.90 = f32[512]{0} parameter(89), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.359 = f32[512]{0} reshape(f32[512]{0} %arg89.90)
  %multiply.1504 = f32[512]{0} multiply(f32[512]{0} %rsqrt.1503, f32[512]{0} %reshape.359), metadata={op_type="Mul" op_name="stage2_unit1_bn3/mul"}
  %broadcast.1505 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.1504), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit1_bn3/mul_1"}
  %constant.1496 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit1_relu2"}
  %broadcast.1497 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[] %constant.1496), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit1_relu2"}
  %constant.1415 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit1_relu1"}
  %broadcast.1416 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[] %constant.1415), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit1_relu1"}
  %arg20.21 = f32[256]{0} parameter(20), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.290 = f32[256]{0} reshape(f32[256]{0} %arg20.21)
  %constant.1404 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit1_bn1/add"}
  %broadcast.1405 = f32[256]{0} broadcast(f32[] %constant.1404), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit1_bn1/add"}
  %add.1406 = f32[256]{0} add(f32[256]{0} %reshape.290, f32[256]{0} %broadcast.1405), metadata={op_type="AddV2" op_name="stage2_unit1_bn1/add"}
  %rsqrt.1407 = f32[256]{0} rsqrt(f32[256]{0} %add.1406), metadata={op_type="Rsqrt" op_name="stage2_unit1_bn1/Rsqrt"}
  %arg85.86 = f32[256]{0} parameter(85), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.355 = f32[256]{0} reshape(f32[256]{0} %arg85.86)
  %multiply.1408 = f32[256]{0} multiply(f32[256]{0} %rsqrt.1407, f32[256]{0} %reshape.355), metadata={op_type="Mul" op_name="stage2_unit1_bn1/mul"}
  %broadcast.1409 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.1408), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit1_bn1/mul_1"}
  %constant.1399 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit3_relu"}
  %broadcast.1400 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[] %constant.1399), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit3_relu"}
  %constant.1287 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit2_relu"}
  %broadcast.1288 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[] %constant.1287), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit2_relu"}
  %constant.1175 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit1_relu"}
  %broadcast.1176 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[] %constant.1175), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit1_relu"}
  %arg8.9 = f32[256]{0} parameter(8), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.278 = f32[256]{0} reshape(f32[256]{0} %arg8.9)
  %constant.1152 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit1_bn3/add"}
  %broadcast.1153 = f32[256]{0} broadcast(f32[] %constant.1152), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit1_bn3/add"}
  %add.1154 = f32[256]{0} add(f32[256]{0} %reshape.278, f32[256]{0} %broadcast.1153), metadata={op_type="AddV2" op_name="stage1_unit1_bn3/add"}
  %rsqrt.1155 = f32[256]{0} rsqrt(f32[256]{0} %add.1154), metadata={op_type="Rsqrt" op_name="stage1_unit1_bn3/Rsqrt"}
  %arg76.77 = f32[256]{0} parameter(76), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.346 = f32[256]{0} reshape(f32[256]{0} %arg76.77)
  %multiply.1156 = f32[256]{0} multiply(f32[256]{0} %rsqrt.1155, f32[256]{0} %reshape.346), metadata={op_type="Mul" op_name="stage1_unit1_bn3/mul"}
  %broadcast.1157 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.1156), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit1_bn3/mul_1"}
  %constant.1148 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit1_relu2"}
  %broadcast.1149 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[] %constant.1148), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit1_relu2"}
  %constant.1067 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit1_relu1"}
  %broadcast.1068 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[] %constant.1067), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit1_relu1"}
  %arg3.4 = f32[128]{0} parameter(3), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.273 = f32[128]{0} reshape(f32[128]{0} %arg3.4)
  %constant.1056 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit1_bn1/add"}
  %broadcast.1057 = f32[128]{0} broadcast(f32[] %constant.1056), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit1_bn1/add"}
  %add.1058 = f32[128]{0} add(f32[128]{0} %reshape.273, f32[128]{0} %broadcast.1057), metadata={op_type="AddV2" op_name="stage1_unit1_bn1/add"}
  %rsqrt.1059 = f32[128]{0} rsqrt(f32[128]{0} %add.1058), metadata={op_type="Rsqrt" op_name="stage1_unit1_bn1/Rsqrt"}
  %arg73.74 = f32[128]{0} parameter(73), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.343 = f32[128]{0} reshape(f32[128]{0} %arg73.74)
  %multiply.1060 = f32[128]{0} multiply(f32[128]{0} %rsqrt.1059, f32[128]{0} %reshape.343), metadata={op_type="Mul" op_name="stage1_unit1_bn1/mul"}
  %broadcast.1061 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %multiply.1060), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit1_bn1/mul_1"}
  %constant.1043 = f32[] constant(0), metadata={op_type="Relu" op_name="relu0"}
  %broadcast.1044 = f32[1,112,112,64]{3,2,1,0} broadcast(f32[] %constant.1043), dimensions={}, metadata={op_type="Relu" op_name="relu0"}
  %arg2.3 = f32[64]{0} parameter(2), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.272 = f32[64]{0} reshape(f32[64]{0} %arg2.3)
  %constant.1019 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="bn0/add"}
  %broadcast.1020 = f32[64]{0} broadcast(f32[] %constant.1019), dimensions={}, metadata={op_type="AddV2" op_name="bn0/add"}
  %add.1021 = f32[64]{0} add(f32[64]{0} %reshape.272, f32[64]{0} %broadcast.1020), metadata={op_type="AddV2" op_name="bn0/add"}
  %rsqrt.1022 = f32[64]{0} rsqrt(f32[64]{0} %add.1021), metadata={op_type="Rsqrt" op_name="bn0/Rsqrt"}
  %arg72.73 = f32[64]{0} parameter(72), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.342 = f32[64]{0} reshape(f32[64]{0} %arg72.73)
  %multiply.1023 = f32[64]{0} multiply(f32[64]{0} %rsqrt.1022, f32[64]{0} %reshape.342), metadata={op_type="Mul" op_name="bn0/mul"}
  %broadcast.1039 = f32[1,112,112,64]{3,2,1,0} broadcast(f32[64]{0} %multiply.1023), dimensions={3}, metadata={op_type="Mul" op_name="bn0/mul_1"}
  %arg0.1 = f32[3]{0} parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.270 = f32[3]{0} reshape(f32[3]{0} %arg0.1)
  %constant.1026 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="bn_data/add"}
  %broadcast.1027 = f32[3]{0} broadcast(f32[] %constant.1026), dimensions={}, metadata={op_type="AddV2" op_name="bn_data/add"}
  %add.1028 = f32[3]{0} add(f32[3]{0} %reshape.270, f32[3]{0} %broadcast.1027), metadata={op_type="AddV2" op_name="bn_data/add"}
  %rsqrt.1029 = f32[3]{0} rsqrt(f32[3]{0} %add.1028), metadata={op_type="Rsqrt" op_name="bn_data/Rsqrt"}
  %broadcast.1030 = f32[1,224,224,3]{3,2,1,0} broadcast(f32[3]{0} %rsqrt.1029), dimensions={3}, metadata={op_type="Mul" op_name="bn_data/mul"}
  %arg268.269 = f32[1,224,224,3]{3,2,1,0} parameter(268), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.538 = f32[1,224,224,3]{3,2,1,0} reshape(f32[1,224,224,3]{3,2,1,0} %arg268.269)
  %multiply.1031 = f32[1,224,224,3]{3,2,1,0} multiply(f32[1,224,224,3]{3,2,1,0} %broadcast.1030, f32[1,224,224,3]{3,2,1,0} %reshape.538), metadata={op_type="Mul" op_name="bn_data/mul"}
  %arg124.125 = f32[3]{0} parameter(124), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.394 = f32[3]{0} reshape(f32[3]{0} %arg124.125)
  %arg70.71 = f32[3]{0} parameter(70), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.340 = f32[3]{0} reshape(f32[3]{0} %arg70.71)
  %multiply.1032 = f32[3]{0} multiply(f32[3]{0} %rsqrt.1029, f32[3]{0} %reshape.340), metadata={op_type="Mul" op_name="bn_data/mul_1"}
  %subtract.1033 = f32[3]{0} subtract(f32[3]{0} %reshape.394, f32[3]{0} %multiply.1032), metadata={op_type="Sub" op_name="bn_data/sub"}
  %broadcast.1034 = f32[1,224,224,3]{3,2,1,0} broadcast(f32[3]{0} %subtract.1033), dimensions={3}, metadata={op_type="AddV2" op_name="bn_data/add_1"}
  %add.1035 = f32[1,224,224,3]{3,2,1,0} add(f32[1,224,224,3]{3,2,1,0} %multiply.1031, f32[1,224,224,3]{3,2,1,0} %broadcast.1034), metadata={op_type="AddV2" op_name="bn_data/add_1"}
  %constant.1036 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad"}
  %pad.1037 = f32[1,230,230,3]{3,2,1,0} pad(f32[1,224,224,3]{3,2,1,0} %add.1035, f32[] %constant.1036), padding=0_0x3_3x3_3x0_0, metadata={op_type="Pad" op_name="Pad"}
  %arg231.232 = f32[7,7,3,64]{3,2,1,0} parameter(231), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.501 = f32[7,7,3,64]{3,2,1,0} reshape(f32[7,7,3,64]{3,2,1,0} %arg231.232)
  %convolution.1038 = f32[1,112,112,64]{3,2,1,0} convolution(f32[1,230,230,3]{3,2,1,0} %pad.1037, f32[7,7,3,64]{3,2,1,0} %reshape.501), window={size=7x7 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="conv0"}
  %multiply.1040 = f32[1,112,112,64]{3,2,1,0} multiply(f32[1,112,112,64]{3,2,1,0} %broadcast.1039, f32[1,112,112,64]{3,2,1,0} %convolution.1038), metadata={op_type="Mul" op_name="bn0/mul_1"}
  %arg179.180 = f32[64]{0} parameter(179), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.449 = f32[64]{0} reshape(f32[64]{0} %arg179.180)
  %arg126.127 = f32[64]{0} parameter(126), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.396 = f32[64]{0} reshape(f32[64]{0} %arg126.127)
  %multiply.1024 = f32[64]{0} multiply(f32[64]{0} %multiply.1023, f32[64]{0} %reshape.396), metadata={op_type="Mul" op_name="bn0/mul_2"}
  %subtract.1025 = f32[64]{0} subtract(f32[64]{0} %reshape.449, f32[64]{0} %multiply.1024), metadata={op_type="Sub" op_name="bn0/sub"}
  %broadcast.1041 = f32[1,112,112,64]{3,2,1,0} broadcast(f32[64]{0} %subtract.1025), dimensions={3}, metadata={op_type="AddV2" op_name="bn0/add_1"}
  %add.1042 = f32[1,112,112,64]{3,2,1,0} add(f32[1,112,112,64]{3,2,1,0} %multiply.1040, f32[1,112,112,64]{3,2,1,0} %broadcast.1041), metadata={op_type="AddV2" op_name="bn0/add_1"}
  %maximum.1045 = f32[1,112,112,64]{3,2,1,0} maximum(f32[1,112,112,64]{3,2,1,0} %broadcast.1044, f32[1,112,112,64]{3,2,1,0} %add.1042), metadata={op_type="Relu" op_name="relu0"}
  %constant.1046 = f32[] constant(-inf), metadata={op_type="PadV2" op_name="PadV2"}
  %pad.1047 = f32[1,114,114,64]{3,2,1,0} pad(f32[1,112,112,64]{3,2,1,0} %maximum.1045, f32[] %constant.1046), padding=0_0x1_1x1_1x0_0, metadata={op_type="PadV2" op_name="PadV2"}
  %constant.1048 = f32[] constant(-inf), metadata={op_type="MaxPool" op_name="pooling0"}
  %reduce-window.1053 = f32[1,56,56,64]{3,2,1,0} reduce-window(f32[1,114,114,64]{3,2,1,0} %pad.1047, f32[] %constant.1048), window={size=1x3x3x1 stride=1x2x2x1}, to_apply=%max_F32.1049, metadata={op_type="MaxPool" op_name="pooling0"}
  %arg232.233 = f32[1,1,64,128]{3,2,1,0} parameter(232), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.502 = f32[1,1,64,128]{3,2,1,0} reshape(f32[1,1,64,128]{3,2,1,0} %arg232.233)
  %convolution.1054 = f32[1,56,56,128]{3,2,1,0} convolution(f32[1,56,56,64]{3,2,1,0} %reduce-window.1053, f32[1,1,64,128]{3,2,1,0} %reshape.502), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv1"}
  %multiply.1062 = f32[1,56,56,128]{3,2,1,0} multiply(f32[1,56,56,128]{3,2,1,0} %broadcast.1061, f32[1,56,56,128]{3,2,1,0} %convolution.1054), metadata={op_type="Mul" op_name="stage1_unit1_bn1/mul_1"}
  %arg180.181 = f32[128]{0} parameter(180), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.450 = f32[128]{0} reshape(f32[128]{0} %arg180.181)
  %arg127.128 = f32[128]{0} parameter(127), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.397 = f32[128]{0} reshape(f32[128]{0} %arg127.128)
  %multiply.1063 = f32[128]{0} multiply(f32[128]{0} %multiply.1060, f32[128]{0} %reshape.397), metadata={op_type="Mul" op_name="stage1_unit1_bn1/mul_2"}
  %subtract.1064 = f32[128]{0} subtract(f32[128]{0} %reshape.450, f32[128]{0} %multiply.1063), metadata={op_type="Sub" op_name="stage1_unit1_bn1/sub"}
  %broadcast.1065 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %subtract.1064), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit1_bn1/add_1"}
  %add.1066 = f32[1,56,56,128]{3,2,1,0} add(f32[1,56,56,128]{3,2,1,0} %multiply.1062, f32[1,56,56,128]{3,2,1,0} %broadcast.1065), metadata={op_type="AddV2" op_name="stage1_unit1_bn1/add_1"}
  %maximum.1069 = f32[1,56,56,128]{3,2,1,0} maximum(f32[1,56,56,128]{3,2,1,0} %broadcast.1068, f32[1,56,56,128]{3,2,1,0} %add.1066), metadata={op_type="Relu" op_name="stage1_unit1_relu1"}
  %constant.1070 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_1"}
  %pad.1071 = f32[1,58,58,128]{3,2,1,0} pad(f32[1,56,56,128]{3,2,1,0} %maximum.1069, f32[] %constant.1070), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_1"}
  %slice.1072 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1071), slice={[0:1], [0:58], [0:58], [0:4]}, metadata={op_type="Split" op_name="split_1"}
  %arg6.7 = f32[3,3,4,128]{3,2,1,0} parameter(6), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.276 = f32[3,3,4,128]{3,2,1,0} reshape(f32[3,3,4,128]{3,2,1,0} %arg6.7)
  %slice.539 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:4], [0:4]}, metadata={op_type="Split" op_name="split"}
  %convolution.1104 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1072, f32[3,3,4,4]{3,2,1,0} %slice.539), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2"}
  %slice.1073 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1071), slice={[0:1], [0:58], [0:58], [4:8]}, metadata={op_type="Split" op_name="split_1"}
  %slice.540 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:4], [4:8]}, metadata={op_type="Split" op_name="split"}
  %convolution.1105 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1073, f32[3,3,4,4]{3,2,1,0} %slice.540), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_1"}
  %slice.1074 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1071), slice={[0:1], [0:58], [0:58], [8:12]}, metadata={op_type="Split" op_name="split_1"}
  %slice.541 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:4], [8:12]}, metadata={op_type="Split" op_name="split"}
  %convolution.1116 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1074, f32[3,3,4,4]{3,2,1,0} %slice.541), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_2"}
  %slice.1075 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1071), slice={[0:1], [0:58], [0:58], [12:16]}, metadata={op_type="Split" op_name="split_1"}
  %slice.542 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:4], [12:16]}, metadata={op_type="Split" op_name="split"}
  %convolution.1127 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1075, f32[3,3,4,4]{3,2,1,0} %slice.542), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_3"}
  %slice.1076 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1071), slice={[0:1], [0:58], [0:58], [16:20]}, metadata={op_type="Split" op_name="split_1"}
  %slice.543 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:4], [16:20]}, metadata={op_type="Split" op_name="split"}
  %convolution.1130 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1076, f32[3,3,4,4]{3,2,1,0} %slice.543), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_4"}
  %slice.1077 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1071), slice={[0:1], [0:58], [0:58], [20:24]}, metadata={op_type="Split" op_name="split_1"}
  %slice.544 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:4], [20:24]}, metadata={op_type="Split" op_name="split"}
  %convolution.1131 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1077, f32[3,3,4,4]{3,2,1,0} %slice.544), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_5"}
  %slice.1078 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1071), slice={[0:1], [0:58], [0:58], [24:28]}, metadata={op_type="Split" op_name="split_1"}
  %slice.545 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:4], [24:28]}, metadata={op_type="Split" op_name="split"}
  %convolution.1132 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1078, f32[3,3,4,4]{3,2,1,0} %slice.545), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_6"}
  %slice.1079 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1071), slice={[0:1], [0:58], [0:58], [28:32]}, metadata={op_type="Split" op_name="split_1"}
  %slice.546 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:4], [28:32]}, metadata={op_type="Split" op_name="split"}
  %convolution.1133 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1079, f32[3,3,4,4]{3,2,1,0} %slice.546), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_7"}
  %slice.1080 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1071), slice={[0:1], [0:58], [0:58], [32:36]}, metadata={op_type="Split" op_name="split_1"}
  %slice.547 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:4], [32:36]}, metadata={op_type="Split" op_name="split"}
  %convolution.1134 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1080, f32[3,3,4,4]{3,2,1,0} %slice.547), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_8"}
  %slice.1081 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1071), slice={[0:1], [0:58], [0:58], [36:40]}, metadata={op_type="Split" op_name="split_1"}
  %slice.548 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:4], [36:40]}, metadata={op_type="Split" op_name="split"}
  %convolution.1135 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1081, f32[3,3,4,4]{3,2,1,0} %slice.548), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_9"}
  %slice.1082 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1071), slice={[0:1], [0:58], [0:58], [40:44]}, metadata={op_type="Split" op_name="split_1"}
  %slice.549 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:4], [40:44]}, metadata={op_type="Split" op_name="split"}
  %convolution.1106 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1082, f32[3,3,4,4]{3,2,1,0} %slice.549), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_10"}
  %slice.1083 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1071), slice={[0:1], [0:58], [0:58], [44:48]}, metadata={op_type="Split" op_name="split_1"}
  %slice.550 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:4], [44:48]}, metadata={op_type="Split" op_name="split"}
  %convolution.1107 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1083, f32[3,3,4,4]{3,2,1,0} %slice.550), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_11"}
  %slice.1084 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1071), slice={[0:1], [0:58], [0:58], [48:52]}, metadata={op_type="Split" op_name="split_1"}
  %slice.551 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:4], [48:52]}, metadata={op_type="Split" op_name="split"}
  %convolution.1108 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1084, f32[3,3,4,4]{3,2,1,0} %slice.551), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_12"}
  %slice.1085 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1071), slice={[0:1], [0:58], [0:58], [52:56]}, metadata={op_type="Split" op_name="split_1"}
  %slice.552 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:4], [52:56]}, metadata={op_type="Split" op_name="split"}
  %convolution.1109 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1085, f32[3,3,4,4]{3,2,1,0} %slice.552), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_13"}
  %slice.1086 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1071), slice={[0:1], [0:58], [0:58], [56:60]}, metadata={op_type="Split" op_name="split_1"}
  %slice.553 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:4], [56:60]}, metadata={op_type="Split" op_name="split"}
  %convolution.1110 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1086, f32[3,3,4,4]{3,2,1,0} %slice.553), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_14"}
  %slice.1087 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1071), slice={[0:1], [0:58], [0:58], [60:64]}, metadata={op_type="Split" op_name="split_1"}
  %slice.554 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:4], [60:64]}, metadata={op_type="Split" op_name="split"}
  %convolution.1111 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1087, f32[3,3,4,4]{3,2,1,0} %slice.554), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_15"}
  %slice.1088 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1071), slice={[0:1], [0:58], [0:58], [64:68]}, metadata={op_type="Split" op_name="split_1"}
  %slice.555 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:4], [64:68]}, metadata={op_type="Split" op_name="split"}
  %convolution.1112 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1088, f32[3,3,4,4]{3,2,1,0} %slice.555), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_16"}
  %slice.1089 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1071), slice={[0:1], [0:58], [0:58], [68:72]}, metadata={op_type="Split" op_name="split_1"}
  %slice.556 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:4], [68:72]}, metadata={op_type="Split" op_name="split"}
  %convolution.1113 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1089, f32[3,3,4,4]{3,2,1,0} %slice.556), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_17"}
  %slice.1090 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1071), slice={[0:1], [0:58], [0:58], [72:76]}, metadata={op_type="Split" op_name="split_1"}
  %slice.557 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:4], [72:76]}, metadata={op_type="Split" op_name="split"}
  %convolution.1114 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1090, f32[3,3,4,4]{3,2,1,0} %slice.557), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_18"}
  %slice.1091 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1071), slice={[0:1], [0:58], [0:58], [76:80]}, metadata={op_type="Split" op_name="split_1"}
  %slice.558 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:4], [76:80]}, metadata={op_type="Split" op_name="split"}
  %convolution.1115 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1091, f32[3,3,4,4]{3,2,1,0} %slice.558), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_19"}
  %slice.1092 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1071), slice={[0:1], [0:58], [0:58], [80:84]}, metadata={op_type="Split" op_name="split_1"}
  %slice.559 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:4], [80:84]}, metadata={op_type="Split" op_name="split"}
  %convolution.1117 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1092, f32[3,3,4,4]{3,2,1,0} %slice.559), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_20"}
  %slice.1093 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1071), slice={[0:1], [0:58], [0:58], [84:88]}, metadata={op_type="Split" op_name="split_1"}
  %slice.560 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:4], [84:88]}, metadata={op_type="Split" op_name="split"}
  %convolution.1118 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1093, f32[3,3,4,4]{3,2,1,0} %slice.560), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_21"}
  %slice.1094 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1071), slice={[0:1], [0:58], [0:58], [88:92]}, metadata={op_type="Split" op_name="split_1"}
  %slice.561 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:4], [88:92]}, metadata={op_type="Split" op_name="split"}
  %convolution.1119 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1094, f32[3,3,4,4]{3,2,1,0} %slice.561), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_22"}
  %slice.1095 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1071), slice={[0:1], [0:58], [0:58], [92:96]}, metadata={op_type="Split" op_name="split_1"}
  %slice.562 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:4], [92:96]}, metadata={op_type="Split" op_name="split"}
  %convolution.1120 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1095, f32[3,3,4,4]{3,2,1,0} %slice.562), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_23"}
  %slice.1096 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1071), slice={[0:1], [0:58], [0:58], [96:100]}, metadata={op_type="Split" op_name="split_1"}
  %slice.563 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:4], [96:100]}, metadata={op_type="Split" op_name="split"}
  %convolution.1121 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1096, f32[3,3,4,4]{3,2,1,0} %slice.563), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_24"}
  %slice.1097 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1071), slice={[0:1], [0:58], [0:58], [100:104]}, metadata={op_type="Split" op_name="split_1"}
  %slice.564 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:4], [100:104]}, metadata={op_type="Split" op_name="split"}
  %convolution.1122 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1097, f32[3,3,4,4]{3,2,1,0} %slice.564), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_25"}
  %slice.1098 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1071), slice={[0:1], [0:58], [0:58], [104:108]}, metadata={op_type="Split" op_name="split_1"}
  %slice.565 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:4], [104:108]}, metadata={op_type="Split" op_name="split"}
  %convolution.1123 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1098, f32[3,3,4,4]{3,2,1,0} %slice.565), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_26"}
  %slice.1099 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1071), slice={[0:1], [0:58], [0:58], [108:112]}, metadata={op_type="Split" op_name="split_1"}
  %slice.566 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:4], [108:112]}, metadata={op_type="Split" op_name="split"}
  %convolution.1124 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1099, f32[3,3,4,4]{3,2,1,0} %slice.566), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_27"}
  %slice.1100 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1071), slice={[0:1], [0:58], [0:58], [112:116]}, metadata={op_type="Split" op_name="split_1"}
  %slice.567 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:4], [112:116]}, metadata={op_type="Split" op_name="split"}
  %convolution.1125 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1100, f32[3,3,4,4]{3,2,1,0} %slice.567), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_28"}
  %slice.1101 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1071), slice={[0:1], [0:58], [0:58], [116:120]}, metadata={op_type="Split" op_name="split_1"}
  %slice.568 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:4], [116:120]}, metadata={op_type="Split" op_name="split"}
  %convolution.1126 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1101, f32[3,3,4,4]{3,2,1,0} %slice.568), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_29"}
  %slice.1102 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1071), slice={[0:1], [0:58], [0:58], [120:124]}, metadata={op_type="Split" op_name="split_1"}
  %slice.569 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:4], [120:124]}, metadata={op_type="Split" op_name="split"}
  %convolution.1128 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1102, f32[3,3,4,4]{3,2,1,0} %slice.569), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_30"}
  %slice.1103 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1071), slice={[0:1], [0:58], [0:58], [124:128]}, metadata={op_type="Split" op_name="split_1"}
  %slice.570 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.276), slice={[0:3], [0:3], [0:4], [124:128]}, metadata={op_type="Split" op_name="split"}
  %convolution.1129 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1103, f32[3,3,4,4]{3,2,1,0} %slice.570), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv2_31"}
  %concatenate.1136 = f32[1,56,56,128]{3,2,1,0} concatenate(f32[1,56,56,4]{3,2,1,0} %convolution.1104, f32[1,56,56,4]{3,2,1,0} %convolution.1105, f32[1,56,56,4]{3,2,1,0} %convolution.1116, f32[1,56,56,4]{3,2,1,0} %convolution.1127, f32[1,56,56,4]{3,2,1,0} %convolution.1130, f32[1,56,56,4]{3,2,1,0} %convolution.1131, f32[1,56,56,4]{3,2,1,0} %convolution.1132, f32[1,56,56,4]{3,2,1,0} %convolution.1133, f32[1,56,56,4]{3,2,1,0} %convolution.1134, f32[1,56,56,4]{3,2,1,0} %convolution.1135, f32[1,56,56,4]{3,2,1,0} %convolution.1106, f32[1,56,56,4]{3,2,1,0} %convolution.1107, f32[1,56,56,4]{3,2,1,0} %convolution.1108, f32[1,56,56,4]{3,2,1,0} %convolution.1109, f32[1,56,56,4]{3,2,1,0} %convolution.1110, f32[1,56,56,4]{3,2,1,0} %convolution.1111, f32[1,56,56,4]{3,2,1,0} %convolution.1112, f32[1,56,56,4]{3,2,1,0} %convolution.1113, f32[1,56,56,4]{3,2,1,0} %convolution.1114, f32[1,56,56,4]{3,2,1,0} %convolution.1115, f32[1,56,56,4]{3,2,1,0} %convolution.1117, f32[1,56,56,4]{3,2,1,0} %convolution.1118, f32[1,56,56,4]{3,2,1,0} %convolution.1119, f32[1,56,56,4]{3,2,1,0} %convolution.1120, f32[1,56,56,4]{3,2,1,0} %convolution.1121, f32[1,56,56,4]{3,2,1,0} %convolution.1122, f32[1,56,56,4]{3,2,1,0} %convolution.1123, f32[1,56,56,4]{3,2,1,0} %convolution.1124, f32[1,56,56,4]{3,2,1,0} %convolution.1125, f32[1,56,56,4]{3,2,1,0} %convolution.1126, f32[1,56,56,4]{3,2,1,0} %convolution.1128, f32[1,56,56,4]{3,2,1,0} %convolution.1129), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat"}
  %arg7.8 = f32[128]{0} parameter(7), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.277 = f32[128]{0} reshape(f32[128]{0} %arg7.8)
  %constant.1137 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit1_bn2/add"}
  %broadcast.1138 = f32[128]{0} broadcast(f32[] %constant.1137), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit1_bn2/add"}
  %add.1139 = f32[128]{0} add(f32[128]{0} %reshape.277, f32[128]{0} %broadcast.1138), metadata={op_type="AddV2" op_name="stage1_unit1_bn2/add"}
  %rsqrt.1140 = f32[128]{0} rsqrt(f32[128]{0} %add.1139), metadata={op_type="Rsqrt" op_name="stage1_unit1_bn2/Rsqrt"}
  %arg75.76 = f32[128]{0} parameter(75), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.345 = f32[128]{0} reshape(f32[128]{0} %arg75.76)
  %multiply.1141 = f32[128]{0} multiply(f32[128]{0} %rsqrt.1140, f32[128]{0} %reshape.345), metadata={op_type="Mul" op_name="stage1_unit1_bn2/mul"}
  %broadcast.1142 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %multiply.1141), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit1_bn2/mul_1"}
  %multiply.1143 = f32[1,56,56,128]{3,2,1,0} multiply(f32[1,56,56,128]{3,2,1,0} %concatenate.1136, f32[1,56,56,128]{3,2,1,0} %broadcast.1142), metadata={op_type="Mul" op_name="stage1_unit1_bn2/mul_1"}
  %arg182.183 = f32[128]{0} parameter(182), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.452 = f32[128]{0} reshape(f32[128]{0} %arg182.183)
  %arg129.130 = f32[128]{0} parameter(129), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.399 = f32[128]{0} reshape(f32[128]{0} %arg129.130)
  %multiply.1144 = f32[128]{0} multiply(f32[128]{0} %multiply.1141, f32[128]{0} %reshape.399), metadata={op_type="Mul" op_name="stage1_unit1_bn2/mul_2"}
  %subtract.1145 = f32[128]{0} subtract(f32[128]{0} %reshape.452, f32[128]{0} %multiply.1144), metadata={op_type="Sub" op_name="stage1_unit1_bn2/sub"}
  %broadcast.1146 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %subtract.1145), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit1_bn2/add_1"}
  %add.1147 = f32[1,56,56,128]{3,2,1,0} add(f32[1,56,56,128]{3,2,1,0} %multiply.1143, f32[1,56,56,128]{3,2,1,0} %broadcast.1146), metadata={op_type="AddV2" op_name="stage1_unit1_bn2/add_1"}
  %maximum.1150 = f32[1,56,56,128]{3,2,1,0} maximum(f32[1,56,56,128]{3,2,1,0} %broadcast.1149, f32[1,56,56,128]{3,2,1,0} %add.1147), metadata={op_type="Relu" op_name="stage1_unit1_relu2"}
  %arg234.235 = f32[1,1,128,256]{3,2,1,0} parameter(234), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.504 = f32[1,1,128,256]{3,2,1,0} reshape(f32[1,1,128,256]{3,2,1,0} %arg234.235)
  %convolution.1151 = f32[1,56,56,256]{3,2,1,0} convolution(f32[1,56,56,128]{3,2,1,0} %maximum.1150, f32[1,1,128,256]{3,2,1,0} %reshape.504), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_conv3"}
  %multiply.1158 = f32[1,56,56,256]{3,2,1,0} multiply(f32[1,56,56,256]{3,2,1,0} %broadcast.1157, f32[1,56,56,256]{3,2,1,0} %convolution.1151), metadata={op_type="Mul" op_name="stage1_unit1_bn3/mul_1"}
  %arg183.184 = f32[256]{0} parameter(183), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.453 = f32[256]{0} reshape(f32[256]{0} %arg183.184)
  %arg130.131 = f32[256]{0} parameter(130), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.400 = f32[256]{0} reshape(f32[256]{0} %arg130.131)
  %multiply.1159 = f32[256]{0} multiply(f32[256]{0} %multiply.1156, f32[256]{0} %reshape.400), metadata={op_type="Mul" op_name="stage1_unit1_bn3/mul_2"}
  %subtract.1160 = f32[256]{0} subtract(f32[256]{0} %reshape.453, f32[256]{0} %multiply.1159), metadata={op_type="Sub" op_name="stage1_unit1_bn3/sub"}
  %broadcast.1161 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.1160), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit1_bn3/add_1"}
  %add.1162 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %multiply.1158, f32[1,56,56,256]{3,2,1,0} %broadcast.1161), metadata={op_type="AddV2" op_name="stage1_unit1_bn3/add_1"}
  %arg233.234 = f32[1,1,64,256]{3,2,1,0} parameter(233), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.503 = f32[1,1,64,256]{3,2,1,0} reshape(f32[1,1,64,256]{3,2,1,0} %arg233.234)
  %convolution.1055 = f32[1,56,56,256]{3,2,1,0} convolution(f32[1,56,56,64]{3,2,1,0} %reduce-window.1053, f32[1,1,64,256]{3,2,1,0} %reshape.503), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit1_sc"}
  %arg5.6 = f32[256]{0} parameter(5), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.275 = f32[256]{0} reshape(f32[256]{0} %arg5.6)
  %constant.1163 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit1_sc_bn/add"}
  %broadcast.1164 = f32[256]{0} broadcast(f32[] %constant.1163), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit1_sc_bn/add"}
  %add.1165 = f32[256]{0} add(f32[256]{0} %reshape.275, f32[256]{0} %broadcast.1164), metadata={op_type="AddV2" op_name="stage1_unit1_sc_bn/add"}
  %rsqrt.1166 = f32[256]{0} rsqrt(f32[256]{0} %add.1165), metadata={op_type="Rsqrt" op_name="stage1_unit1_sc_bn/Rsqrt"}
  %arg74.75 = f32[256]{0} parameter(74), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.344 = f32[256]{0} reshape(f32[256]{0} %arg74.75)
  %multiply.1167 = f32[256]{0} multiply(f32[256]{0} %rsqrt.1166, f32[256]{0} %reshape.344), metadata={op_type="Mul" op_name="stage1_unit1_sc_bn/mul"}
  %broadcast.1168 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.1167), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit1_sc_bn/mul_1"}
  %multiply.1169 = f32[1,56,56,256]{3,2,1,0} multiply(f32[1,56,56,256]{3,2,1,0} %convolution.1055, f32[1,56,56,256]{3,2,1,0} %broadcast.1168), metadata={op_type="Mul" op_name="stage1_unit1_sc_bn/mul_1"}
  %arg181.182 = f32[256]{0} parameter(181), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.451 = f32[256]{0} reshape(f32[256]{0} %arg181.182)
  %arg128.129 = f32[256]{0} parameter(128), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.398 = f32[256]{0} reshape(f32[256]{0} %arg128.129)
  %multiply.1170 = f32[256]{0} multiply(f32[256]{0} %multiply.1167, f32[256]{0} %reshape.398), metadata={op_type="Mul" op_name="stage1_unit1_sc_bn/mul_2"}
  %subtract.1171 = f32[256]{0} subtract(f32[256]{0} %reshape.451, f32[256]{0} %multiply.1170), metadata={op_type="Sub" op_name="stage1_unit1_sc_bn/sub"}
  %broadcast.1172 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.1171), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit1_sc_bn/add_1"}
  %add.1173 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %multiply.1169, f32[1,56,56,256]{3,2,1,0} %broadcast.1172), metadata={op_type="AddV2" op_name="stage1_unit1_sc_bn/add_1"}
  %add.1174 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %add.1162, f32[1,56,56,256]{3,2,1,0} %add.1173), metadata={op_type="AddV2" op_name="add"}
  %maximum.1177 = f32[1,56,56,256]{3,2,1,0} maximum(f32[1,56,56,256]{3,2,1,0} %broadcast.1176, f32[1,56,56,256]{3,2,1,0} %add.1174), metadata={op_type="Relu" op_name="stage1_unit1_relu"}
  %arg13.14 = f32[256]{0} parameter(13), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.283 = f32[256]{0} reshape(f32[256]{0} %arg13.14)
  %constant.1275 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit2_bn3/add"}
  %broadcast.1276 = f32[256]{0} broadcast(f32[] %constant.1275), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit2_bn3/add"}
  %add.1277 = f32[256]{0} add(f32[256]{0} %reshape.283, f32[256]{0} %broadcast.1276), metadata={op_type="AddV2" op_name="stage1_unit2_bn3/add"}
  %rsqrt.1278 = f32[256]{0} rsqrt(f32[256]{0} %add.1277), metadata={op_type="Rsqrt" op_name="stage1_unit2_bn3/Rsqrt"}
  %arg80.81 = f32[256]{0} parameter(80), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.350 = f32[256]{0} reshape(f32[256]{0} %arg80.81)
  %multiply.1279 = f32[256]{0} multiply(f32[256]{0} %rsqrt.1278, f32[256]{0} %reshape.350), metadata={op_type="Mul" op_name="stage1_unit2_bn3/mul"}
  %broadcast.1280 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.1279), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit2_bn3/mul_1"}
  %constant.1271 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit2_relu2"}
  %broadcast.1272 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[] %constant.1271), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit2_relu2"}
  %constant.1190 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit2_relu1"}
  %broadcast.1191 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[] %constant.1190), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit2_relu1"}
  %arg9.10 = f32[128]{0} parameter(9), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.279 = f32[128]{0} reshape(f32[128]{0} %arg9.10)
  %constant.1179 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit2_bn1/add"}
  %broadcast.1180 = f32[128]{0} broadcast(f32[] %constant.1179), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit2_bn1/add"}
  %add.1181 = f32[128]{0} add(f32[128]{0} %reshape.279, f32[128]{0} %broadcast.1180), metadata={op_type="AddV2" op_name="stage1_unit2_bn1/add"}
  %rsqrt.1182 = f32[128]{0} rsqrt(f32[128]{0} %add.1181), metadata={op_type="Rsqrt" op_name="stage1_unit2_bn1/Rsqrt"}
  %arg77.78 = f32[128]{0} parameter(77), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.347 = f32[128]{0} reshape(f32[128]{0} %arg77.78)
  %multiply.1183 = f32[128]{0} multiply(f32[128]{0} %rsqrt.1182, f32[128]{0} %reshape.347), metadata={op_type="Mul" op_name="stage1_unit2_bn1/mul"}
  %broadcast.1184 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %multiply.1183), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit2_bn1/mul_1"}
  %arg235.236 = f32[1,1,256,128]{3,2,1,0} parameter(235), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.505 = f32[1,1,256,128]{3,2,1,0} reshape(f32[1,1,256,128]{3,2,1,0} %arg235.236)
  %convolution.1178 = f32[1,56,56,128]{3,2,1,0} convolution(f32[1,56,56,256]{3,2,1,0} %maximum.1177, f32[1,1,256,128]{3,2,1,0} %reshape.505), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv1"}
  %multiply.1185 = f32[1,56,56,128]{3,2,1,0} multiply(f32[1,56,56,128]{3,2,1,0} %broadcast.1184, f32[1,56,56,128]{3,2,1,0} %convolution.1178), metadata={op_type="Mul" op_name="stage1_unit2_bn1/mul_1"}
  %arg184.185 = f32[128]{0} parameter(184), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.454 = f32[128]{0} reshape(f32[128]{0} %arg184.185)
  %arg131.132 = f32[128]{0} parameter(131), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.401 = f32[128]{0} reshape(f32[128]{0} %arg131.132)
  %multiply.1186 = f32[128]{0} multiply(f32[128]{0} %multiply.1183, f32[128]{0} %reshape.401), metadata={op_type="Mul" op_name="stage1_unit2_bn1/mul_2"}
  %subtract.1187 = f32[128]{0} subtract(f32[128]{0} %reshape.454, f32[128]{0} %multiply.1186), metadata={op_type="Sub" op_name="stage1_unit2_bn1/sub"}
  %broadcast.1188 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %subtract.1187), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit2_bn1/add_1"}
  %add.1189 = f32[1,56,56,128]{3,2,1,0} add(f32[1,56,56,128]{3,2,1,0} %multiply.1185, f32[1,56,56,128]{3,2,1,0} %broadcast.1188), metadata={op_type="AddV2" op_name="stage1_unit2_bn1/add_1"}
  %maximum.1192 = f32[1,56,56,128]{3,2,1,0} maximum(f32[1,56,56,128]{3,2,1,0} %broadcast.1191, f32[1,56,56,128]{3,2,1,0} %add.1189), metadata={op_type="Relu" op_name="stage1_unit2_relu1"}
  %constant.1193 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_2"}
  %pad.1194 = f32[1,58,58,128]{3,2,1,0} pad(f32[1,56,56,128]{3,2,1,0} %maximum.1192, f32[] %constant.1193), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_2"}
  %slice.1195 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1194), slice={[0:1], [0:58], [0:58], [0:4]}, metadata={op_type="Split" op_name="split_3"}
  %arg10.11 = f32[3,3,4,128]{3,2,1,0} parameter(10), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.280 = f32[3,3,4,128]{3,2,1,0} reshape(f32[3,3,4,128]{3,2,1,0} %arg10.11)
  %slice.571 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.280), slice={[0:3], [0:3], [0:4], [0:4]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.1227 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1195, f32[3,3,4,4]{3,2,1,0} %slice.571), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2"}
  %slice.1196 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1194), slice={[0:1], [0:58], [0:58], [4:8]}, metadata={op_type="Split" op_name="split_3"}
  %slice.572 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.280), slice={[0:3], [0:3], [0:4], [4:8]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.1228 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1196, f32[3,3,4,4]{3,2,1,0} %slice.572), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_1"}
  %slice.1197 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1194), slice={[0:1], [0:58], [0:58], [8:12]}, metadata={op_type="Split" op_name="split_3"}
  %slice.573 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.280), slice={[0:3], [0:3], [0:4], [8:12]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.1239 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1197, f32[3,3,4,4]{3,2,1,0} %slice.573), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_2"}
  %slice.1198 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1194), slice={[0:1], [0:58], [0:58], [12:16]}, metadata={op_type="Split" op_name="split_3"}
  %slice.574 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.280), slice={[0:3], [0:3], [0:4], [12:16]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.1250 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1198, f32[3,3,4,4]{3,2,1,0} %slice.574), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_3"}
  %slice.1199 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1194), slice={[0:1], [0:58], [0:58], [16:20]}, metadata={op_type="Split" op_name="split_3"}
  %slice.575 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.280), slice={[0:3], [0:3], [0:4], [16:20]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.1253 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1199, f32[3,3,4,4]{3,2,1,0} %slice.575), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_4"}
  %slice.1200 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1194), slice={[0:1], [0:58], [0:58], [20:24]}, metadata={op_type="Split" op_name="split_3"}
  %slice.576 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.280), slice={[0:3], [0:3], [0:4], [20:24]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.1254 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1200, f32[3,3,4,4]{3,2,1,0} %slice.576), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_5"}
  %slice.1201 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1194), slice={[0:1], [0:58], [0:58], [24:28]}, metadata={op_type="Split" op_name="split_3"}
  %slice.577 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.280), slice={[0:3], [0:3], [0:4], [24:28]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.1255 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1201, f32[3,3,4,4]{3,2,1,0} %slice.577), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_6"}
  %slice.1202 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1194), slice={[0:1], [0:58], [0:58], [28:32]}, metadata={op_type="Split" op_name="split_3"}
  %slice.578 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.280), slice={[0:3], [0:3], [0:4], [28:32]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.1256 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1202, f32[3,3,4,4]{3,2,1,0} %slice.578), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_7"}
  %slice.1203 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1194), slice={[0:1], [0:58], [0:58], [32:36]}, metadata={op_type="Split" op_name="split_3"}
  %slice.579 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.280), slice={[0:3], [0:3], [0:4], [32:36]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.1257 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1203, f32[3,3,4,4]{3,2,1,0} %slice.579), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_8"}
  %slice.1204 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1194), slice={[0:1], [0:58], [0:58], [36:40]}, metadata={op_type="Split" op_name="split_3"}
  %slice.580 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.280), slice={[0:3], [0:3], [0:4], [36:40]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.1258 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1204, f32[3,3,4,4]{3,2,1,0} %slice.580), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_9"}
  %slice.1205 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1194), slice={[0:1], [0:58], [0:58], [40:44]}, metadata={op_type="Split" op_name="split_3"}
  %slice.581 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.280), slice={[0:3], [0:3], [0:4], [40:44]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.1229 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1205, f32[3,3,4,4]{3,2,1,0} %slice.581), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_10"}
  %slice.1206 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1194), slice={[0:1], [0:58], [0:58], [44:48]}, metadata={op_type="Split" op_name="split_3"}
  %slice.582 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.280), slice={[0:3], [0:3], [0:4], [44:48]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.1230 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1206, f32[3,3,4,4]{3,2,1,0} %slice.582), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_11"}
  %slice.1207 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1194), slice={[0:1], [0:58], [0:58], [48:52]}, metadata={op_type="Split" op_name="split_3"}
  %slice.583 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.280), slice={[0:3], [0:3], [0:4], [48:52]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.1231 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1207, f32[3,3,4,4]{3,2,1,0} %slice.583), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_12"}
  %slice.1208 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1194), slice={[0:1], [0:58], [0:58], [52:56]}, metadata={op_type="Split" op_name="split_3"}
  %slice.584 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.280), slice={[0:3], [0:3], [0:4], [52:56]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.1232 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1208, f32[3,3,4,4]{3,2,1,0} %slice.584), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_13"}
  %slice.1209 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1194), slice={[0:1], [0:58], [0:58], [56:60]}, metadata={op_type="Split" op_name="split_3"}
  %slice.585 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.280), slice={[0:3], [0:3], [0:4], [56:60]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.1233 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1209, f32[3,3,4,4]{3,2,1,0} %slice.585), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_14"}
  %slice.1210 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1194), slice={[0:1], [0:58], [0:58], [60:64]}, metadata={op_type="Split" op_name="split_3"}
  %slice.586 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.280), slice={[0:3], [0:3], [0:4], [60:64]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.1234 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1210, f32[3,3,4,4]{3,2,1,0} %slice.586), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_15"}
  %slice.1211 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1194), slice={[0:1], [0:58], [0:58], [64:68]}, metadata={op_type="Split" op_name="split_3"}
  %slice.587 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.280), slice={[0:3], [0:3], [0:4], [64:68]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.1235 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1211, f32[3,3,4,4]{3,2,1,0} %slice.587), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_16"}
  %slice.1212 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1194), slice={[0:1], [0:58], [0:58], [68:72]}, metadata={op_type="Split" op_name="split_3"}
  %slice.588 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.280), slice={[0:3], [0:3], [0:4], [68:72]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.1236 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1212, f32[3,3,4,4]{3,2,1,0} %slice.588), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_17"}
  %slice.1213 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1194), slice={[0:1], [0:58], [0:58], [72:76]}, metadata={op_type="Split" op_name="split_3"}
  %slice.589 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.280), slice={[0:3], [0:3], [0:4], [72:76]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.1237 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1213, f32[3,3,4,4]{3,2,1,0} %slice.589), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_18"}
  %slice.1214 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1194), slice={[0:1], [0:58], [0:58], [76:80]}, metadata={op_type="Split" op_name="split_3"}
  %slice.590 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.280), slice={[0:3], [0:3], [0:4], [76:80]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.1238 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1214, f32[3,3,4,4]{3,2,1,0} %slice.590), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_19"}
  %slice.1215 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1194), slice={[0:1], [0:58], [0:58], [80:84]}, metadata={op_type="Split" op_name="split_3"}
  %slice.591 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.280), slice={[0:3], [0:3], [0:4], [80:84]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.1240 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1215, f32[3,3,4,4]{3,2,1,0} %slice.591), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_20"}
  %slice.1216 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1194), slice={[0:1], [0:58], [0:58], [84:88]}, metadata={op_type="Split" op_name="split_3"}
  %slice.592 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.280), slice={[0:3], [0:3], [0:4], [84:88]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.1241 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1216, f32[3,3,4,4]{3,2,1,0} %slice.592), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_21"}
  %slice.1217 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1194), slice={[0:1], [0:58], [0:58], [88:92]}, metadata={op_type="Split" op_name="split_3"}
  %slice.593 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.280), slice={[0:3], [0:3], [0:4], [88:92]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.1242 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1217, f32[3,3,4,4]{3,2,1,0} %slice.593), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_22"}
  %slice.1218 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1194), slice={[0:1], [0:58], [0:58], [92:96]}, metadata={op_type="Split" op_name="split_3"}
  %slice.594 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.280), slice={[0:3], [0:3], [0:4], [92:96]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.1243 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1218, f32[3,3,4,4]{3,2,1,0} %slice.594), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_23"}
  %slice.1219 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1194), slice={[0:1], [0:58], [0:58], [96:100]}, metadata={op_type="Split" op_name="split_3"}
  %slice.595 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.280), slice={[0:3], [0:3], [0:4], [96:100]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.1244 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1219, f32[3,3,4,4]{3,2,1,0} %slice.595), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_24"}
  %slice.1220 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1194), slice={[0:1], [0:58], [0:58], [100:104]}, metadata={op_type="Split" op_name="split_3"}
  %slice.596 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.280), slice={[0:3], [0:3], [0:4], [100:104]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.1245 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1220, f32[3,3,4,4]{3,2,1,0} %slice.596), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_25"}
  %slice.1221 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1194), slice={[0:1], [0:58], [0:58], [104:108]}, metadata={op_type="Split" op_name="split_3"}
  %slice.597 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.280), slice={[0:3], [0:3], [0:4], [104:108]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.1246 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1221, f32[3,3,4,4]{3,2,1,0} %slice.597), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_26"}
  %slice.1222 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1194), slice={[0:1], [0:58], [0:58], [108:112]}, metadata={op_type="Split" op_name="split_3"}
  %slice.598 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.280), slice={[0:3], [0:3], [0:4], [108:112]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.1247 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1222, f32[3,3,4,4]{3,2,1,0} %slice.598), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_27"}
  %slice.1223 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1194), slice={[0:1], [0:58], [0:58], [112:116]}, metadata={op_type="Split" op_name="split_3"}
  %slice.599 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.280), slice={[0:3], [0:3], [0:4], [112:116]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.1248 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1223, f32[3,3,4,4]{3,2,1,0} %slice.599), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_28"}
  %slice.1224 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1194), slice={[0:1], [0:58], [0:58], [116:120]}, metadata={op_type="Split" op_name="split_3"}
  %slice.600 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.280), slice={[0:3], [0:3], [0:4], [116:120]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.1249 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1224, f32[3,3,4,4]{3,2,1,0} %slice.600), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_29"}
  %slice.1225 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1194), slice={[0:1], [0:58], [0:58], [120:124]}, metadata={op_type="Split" op_name="split_3"}
  %slice.601 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.280), slice={[0:3], [0:3], [0:4], [120:124]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.1251 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1225, f32[3,3,4,4]{3,2,1,0} %slice.601), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_30"}
  %slice.1226 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1194), slice={[0:1], [0:58], [0:58], [124:128]}, metadata={op_type="Split" op_name="split_3"}
  %slice.602 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.280), slice={[0:3], [0:3], [0:4], [124:128]}, metadata={op_type="Split" op_name="split_2"}
  %convolution.1252 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1226, f32[3,3,4,4]{3,2,1,0} %slice.602), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv2_31"}
  %concatenate.1259 = f32[1,56,56,128]{3,2,1,0} concatenate(f32[1,56,56,4]{3,2,1,0} %convolution.1227, f32[1,56,56,4]{3,2,1,0} %convolution.1228, f32[1,56,56,4]{3,2,1,0} %convolution.1239, f32[1,56,56,4]{3,2,1,0} %convolution.1250, f32[1,56,56,4]{3,2,1,0} %convolution.1253, f32[1,56,56,4]{3,2,1,0} %convolution.1254, f32[1,56,56,4]{3,2,1,0} %convolution.1255, f32[1,56,56,4]{3,2,1,0} %convolution.1256, f32[1,56,56,4]{3,2,1,0} %convolution.1257, f32[1,56,56,4]{3,2,1,0} %convolution.1258, f32[1,56,56,4]{3,2,1,0} %convolution.1229, f32[1,56,56,4]{3,2,1,0} %convolution.1230, f32[1,56,56,4]{3,2,1,0} %convolution.1231, f32[1,56,56,4]{3,2,1,0} %convolution.1232, f32[1,56,56,4]{3,2,1,0} %convolution.1233, f32[1,56,56,4]{3,2,1,0} %convolution.1234, f32[1,56,56,4]{3,2,1,0} %convolution.1235, f32[1,56,56,4]{3,2,1,0} %convolution.1236, f32[1,56,56,4]{3,2,1,0} %convolution.1237, f32[1,56,56,4]{3,2,1,0} %convolution.1238, f32[1,56,56,4]{3,2,1,0} %convolution.1240, f32[1,56,56,4]{3,2,1,0} %convolution.1241, f32[1,56,56,4]{3,2,1,0} %convolution.1242, f32[1,56,56,4]{3,2,1,0} %convolution.1243, f32[1,56,56,4]{3,2,1,0} %convolution.1244, f32[1,56,56,4]{3,2,1,0} %convolution.1245, f32[1,56,56,4]{3,2,1,0} %convolution.1246, f32[1,56,56,4]{3,2,1,0} %convolution.1247, f32[1,56,56,4]{3,2,1,0} %convolution.1248, f32[1,56,56,4]{3,2,1,0} %convolution.1249, f32[1,56,56,4]{3,2,1,0} %convolution.1251, f32[1,56,56,4]{3,2,1,0} %convolution.1252), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_1"}
  %arg11.12 = f32[128]{0} parameter(11), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.281 = f32[128]{0} reshape(f32[128]{0} %arg11.12)
  %constant.1260 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit2_bn2/add"}
  %broadcast.1261 = f32[128]{0} broadcast(f32[] %constant.1260), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit2_bn2/add"}
  %add.1262 = f32[128]{0} add(f32[128]{0} %reshape.281, f32[128]{0} %broadcast.1261), metadata={op_type="AddV2" op_name="stage1_unit2_bn2/add"}
  %rsqrt.1263 = f32[128]{0} rsqrt(f32[128]{0} %add.1262), metadata={op_type="Rsqrt" op_name="stage1_unit2_bn2/Rsqrt"}
  %arg78.79 = f32[128]{0} parameter(78), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.348 = f32[128]{0} reshape(f32[128]{0} %arg78.79)
  %multiply.1264 = f32[128]{0} multiply(f32[128]{0} %rsqrt.1263, f32[128]{0} %reshape.348), metadata={op_type="Mul" op_name="stage1_unit2_bn2/mul"}
  %broadcast.1265 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %multiply.1264), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit2_bn2/mul_1"}
  %multiply.1266 = f32[1,56,56,128]{3,2,1,0} multiply(f32[1,56,56,128]{3,2,1,0} %concatenate.1259, f32[1,56,56,128]{3,2,1,0} %broadcast.1265), metadata={op_type="Mul" op_name="stage1_unit2_bn2/mul_1"}
  %arg185.186 = f32[128]{0} parameter(185), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.455 = f32[128]{0} reshape(f32[128]{0} %arg185.186)
  %arg132.133 = f32[128]{0} parameter(132), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.402 = f32[128]{0} reshape(f32[128]{0} %arg132.133)
  %multiply.1267 = f32[128]{0} multiply(f32[128]{0} %multiply.1264, f32[128]{0} %reshape.402), metadata={op_type="Mul" op_name="stage1_unit2_bn2/mul_2"}
  %subtract.1268 = f32[128]{0} subtract(f32[128]{0} %reshape.455, f32[128]{0} %multiply.1267), metadata={op_type="Sub" op_name="stage1_unit2_bn2/sub"}
  %broadcast.1269 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %subtract.1268), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit2_bn2/add_1"}
  %add.1270 = f32[1,56,56,128]{3,2,1,0} add(f32[1,56,56,128]{3,2,1,0} %multiply.1266, f32[1,56,56,128]{3,2,1,0} %broadcast.1269), metadata={op_type="AddV2" op_name="stage1_unit2_bn2/add_1"}
  %maximum.1273 = f32[1,56,56,128]{3,2,1,0} maximum(f32[1,56,56,128]{3,2,1,0} %broadcast.1272, f32[1,56,56,128]{3,2,1,0} %add.1270), metadata={op_type="Relu" op_name="stage1_unit2_relu2"}
  %arg236.237 = f32[1,1,128,256]{3,2,1,0} parameter(236), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.506 = f32[1,1,128,256]{3,2,1,0} reshape(f32[1,1,128,256]{3,2,1,0} %arg236.237)
  %convolution.1274 = f32[1,56,56,256]{3,2,1,0} convolution(f32[1,56,56,128]{3,2,1,0} %maximum.1273, f32[1,1,128,256]{3,2,1,0} %reshape.506), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit2_conv3"}
  %multiply.1281 = f32[1,56,56,256]{3,2,1,0} multiply(f32[1,56,56,256]{3,2,1,0} %broadcast.1280, f32[1,56,56,256]{3,2,1,0} %convolution.1274), metadata={op_type="Mul" op_name="stage1_unit2_bn3/mul_1"}
  %arg187.188 = f32[256]{0} parameter(187), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.457 = f32[256]{0} reshape(f32[256]{0} %arg187.188)
  %arg134.135 = f32[256]{0} parameter(134), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.404 = f32[256]{0} reshape(f32[256]{0} %arg134.135)
  %multiply.1282 = f32[256]{0} multiply(f32[256]{0} %multiply.1279, f32[256]{0} %reshape.404), metadata={op_type="Mul" op_name="stage1_unit2_bn3/mul_2"}
  %subtract.1283 = f32[256]{0} subtract(f32[256]{0} %reshape.457, f32[256]{0} %multiply.1282), metadata={op_type="Sub" op_name="stage1_unit2_bn3/sub"}
  %broadcast.1284 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.1283), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit2_bn3/add_1"}
  %add.1285 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %multiply.1281, f32[1,56,56,256]{3,2,1,0} %broadcast.1284), metadata={op_type="AddV2" op_name="stage1_unit2_bn3/add_1"}
  %add.1286 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %maximum.1177, f32[1,56,56,256]{3,2,1,0} %add.1285), metadata={op_type="AddV2" op_name="add_1"}
  %maximum.1289 = f32[1,56,56,256]{3,2,1,0} maximum(f32[1,56,56,256]{3,2,1,0} %broadcast.1288, f32[1,56,56,256]{3,2,1,0} %add.1286), metadata={op_type="Relu" op_name="stage1_unit2_relu"}
  %arg19.20 = f32[256]{0} parameter(19), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.289 = f32[256]{0} reshape(f32[256]{0} %arg19.20)
  %constant.1387 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit3_bn3/add"}
  %broadcast.1388 = f32[256]{0} broadcast(f32[] %constant.1387), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit3_bn3/add"}
  %add.1389 = f32[256]{0} add(f32[256]{0} %reshape.289, f32[256]{0} %broadcast.1388), metadata={op_type="AddV2" op_name="stage1_unit3_bn3/add"}
  %rsqrt.1390 = f32[256]{0} rsqrt(f32[256]{0} %add.1389), metadata={op_type="Rsqrt" op_name="stage1_unit3_bn3/Rsqrt"}
  %arg84.85 = f32[256]{0} parameter(84), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.354 = f32[256]{0} reshape(f32[256]{0} %arg84.85)
  %multiply.1391 = f32[256]{0} multiply(f32[256]{0} %rsqrt.1390, f32[256]{0} %reshape.354), metadata={op_type="Mul" op_name="stage1_unit3_bn3/mul"}
  %broadcast.1392 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.1391), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit3_bn3/mul_1"}
  %constant.1383 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit3_relu2"}
  %broadcast.1384 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[] %constant.1383), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit3_relu2"}
  %constant.1302 = f32[] constant(0), metadata={op_type="Relu" op_name="stage1_unit3_relu1"}
  %broadcast.1303 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[] %constant.1302), dimensions={}, metadata={op_type="Relu" op_name="stage1_unit3_relu1"}
  %arg14.15 = f32[128]{0} parameter(14), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.284 = f32[128]{0} reshape(f32[128]{0} %arg14.15)
  %constant.1291 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit3_bn1/add"}
  %broadcast.1292 = f32[128]{0} broadcast(f32[] %constant.1291), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit3_bn1/add"}
  %add.1293 = f32[128]{0} add(f32[128]{0} %reshape.284, f32[128]{0} %broadcast.1292), metadata={op_type="AddV2" op_name="stage1_unit3_bn1/add"}
  %rsqrt.1294 = f32[128]{0} rsqrt(f32[128]{0} %add.1293), metadata={op_type="Rsqrt" op_name="stage1_unit3_bn1/Rsqrt"}
  %arg81.82 = f32[128]{0} parameter(81), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.351 = f32[128]{0} reshape(f32[128]{0} %arg81.82)
  %multiply.1295 = f32[128]{0} multiply(f32[128]{0} %rsqrt.1294, f32[128]{0} %reshape.351), metadata={op_type="Mul" op_name="stage1_unit3_bn1/mul"}
  %broadcast.1296 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %multiply.1295), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit3_bn1/mul_1"}
  %arg237.238 = f32[1,1,256,128]{3,2,1,0} parameter(237), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.507 = f32[1,1,256,128]{3,2,1,0} reshape(f32[1,1,256,128]{3,2,1,0} %arg237.238)
  %convolution.1290 = f32[1,56,56,128]{3,2,1,0} convolution(f32[1,56,56,256]{3,2,1,0} %maximum.1289, f32[1,1,256,128]{3,2,1,0} %reshape.507), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv1"}
  %multiply.1297 = f32[1,56,56,128]{3,2,1,0} multiply(f32[1,56,56,128]{3,2,1,0} %broadcast.1296, f32[1,56,56,128]{3,2,1,0} %convolution.1290), metadata={op_type="Mul" op_name="stage1_unit3_bn1/mul_1"}
  %arg188.189 = f32[128]{0} parameter(188), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.458 = f32[128]{0} reshape(f32[128]{0} %arg188.189)
  %arg135.136 = f32[128]{0} parameter(135), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.405 = f32[128]{0} reshape(f32[128]{0} %arg135.136)
  %multiply.1298 = f32[128]{0} multiply(f32[128]{0} %multiply.1295, f32[128]{0} %reshape.405), metadata={op_type="Mul" op_name="stage1_unit3_bn1/mul_2"}
  %subtract.1299 = f32[128]{0} subtract(f32[128]{0} %reshape.458, f32[128]{0} %multiply.1298), metadata={op_type="Sub" op_name="stage1_unit3_bn1/sub"}
  %broadcast.1300 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %subtract.1299), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit3_bn1/add_1"}
  %add.1301 = f32[1,56,56,128]{3,2,1,0} add(f32[1,56,56,128]{3,2,1,0} %multiply.1297, f32[1,56,56,128]{3,2,1,0} %broadcast.1300), metadata={op_type="AddV2" op_name="stage1_unit3_bn1/add_1"}
  %maximum.1304 = f32[1,56,56,128]{3,2,1,0} maximum(f32[1,56,56,128]{3,2,1,0} %broadcast.1303, f32[1,56,56,128]{3,2,1,0} %add.1301), metadata={op_type="Relu" op_name="stage1_unit3_relu1"}
  %constant.1305 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_3"}
  %pad.1306 = f32[1,58,58,128]{3,2,1,0} pad(f32[1,56,56,128]{3,2,1,0} %maximum.1304, f32[] %constant.1305), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_3"}
  %slice.1307 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1306), slice={[0:1], [0:58], [0:58], [0:4]}, metadata={op_type="Split" op_name="split_5"}
  %arg16.17 = f32[3,3,4,128]{3,2,1,0} parameter(16), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.286 = f32[3,3,4,128]{3,2,1,0} reshape(f32[3,3,4,128]{3,2,1,0} %arg16.17)
  %slice.603 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.286), slice={[0:3], [0:3], [0:4], [0:4]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.1339 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1307, f32[3,3,4,4]{3,2,1,0} %slice.603), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2"}
  %slice.1308 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1306), slice={[0:1], [0:58], [0:58], [4:8]}, metadata={op_type="Split" op_name="split_5"}
  %slice.604 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.286), slice={[0:3], [0:3], [0:4], [4:8]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.1340 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1308, f32[3,3,4,4]{3,2,1,0} %slice.604), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_1"}
  %slice.1309 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1306), slice={[0:1], [0:58], [0:58], [8:12]}, metadata={op_type="Split" op_name="split_5"}
  %slice.605 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.286), slice={[0:3], [0:3], [0:4], [8:12]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.1351 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1309, f32[3,3,4,4]{3,2,1,0} %slice.605), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_2"}
  %slice.1310 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1306), slice={[0:1], [0:58], [0:58], [12:16]}, metadata={op_type="Split" op_name="split_5"}
  %slice.606 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.286), slice={[0:3], [0:3], [0:4], [12:16]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.1362 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1310, f32[3,3,4,4]{3,2,1,0} %slice.606), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_3"}
  %slice.1311 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1306), slice={[0:1], [0:58], [0:58], [16:20]}, metadata={op_type="Split" op_name="split_5"}
  %slice.607 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.286), slice={[0:3], [0:3], [0:4], [16:20]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.1365 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1311, f32[3,3,4,4]{3,2,1,0} %slice.607), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_4"}
  %slice.1312 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1306), slice={[0:1], [0:58], [0:58], [20:24]}, metadata={op_type="Split" op_name="split_5"}
  %slice.608 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.286), slice={[0:3], [0:3], [0:4], [20:24]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.1366 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1312, f32[3,3,4,4]{3,2,1,0} %slice.608), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_5"}
  %slice.1313 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1306), slice={[0:1], [0:58], [0:58], [24:28]}, metadata={op_type="Split" op_name="split_5"}
  %slice.609 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.286), slice={[0:3], [0:3], [0:4], [24:28]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.1367 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1313, f32[3,3,4,4]{3,2,1,0} %slice.609), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_6"}
  %slice.1314 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1306), slice={[0:1], [0:58], [0:58], [28:32]}, metadata={op_type="Split" op_name="split_5"}
  %slice.610 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.286), slice={[0:3], [0:3], [0:4], [28:32]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.1368 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1314, f32[3,3,4,4]{3,2,1,0} %slice.610), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_7"}
  %slice.1315 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1306), slice={[0:1], [0:58], [0:58], [32:36]}, metadata={op_type="Split" op_name="split_5"}
  %slice.611 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.286), slice={[0:3], [0:3], [0:4], [32:36]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.1369 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1315, f32[3,3,4,4]{3,2,1,0} %slice.611), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_8"}
  %slice.1316 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1306), slice={[0:1], [0:58], [0:58], [36:40]}, metadata={op_type="Split" op_name="split_5"}
  %slice.612 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.286), slice={[0:3], [0:3], [0:4], [36:40]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.1370 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1316, f32[3,3,4,4]{3,2,1,0} %slice.612), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_9"}
  %slice.1317 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1306), slice={[0:1], [0:58], [0:58], [40:44]}, metadata={op_type="Split" op_name="split_5"}
  %slice.613 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.286), slice={[0:3], [0:3], [0:4], [40:44]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.1341 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1317, f32[3,3,4,4]{3,2,1,0} %slice.613), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_10"}
  %slice.1318 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1306), slice={[0:1], [0:58], [0:58], [44:48]}, metadata={op_type="Split" op_name="split_5"}
  %slice.614 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.286), slice={[0:3], [0:3], [0:4], [44:48]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.1342 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1318, f32[3,3,4,4]{3,2,1,0} %slice.614), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_11"}
  %slice.1319 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1306), slice={[0:1], [0:58], [0:58], [48:52]}, metadata={op_type="Split" op_name="split_5"}
  %slice.615 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.286), slice={[0:3], [0:3], [0:4], [48:52]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.1343 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1319, f32[3,3,4,4]{3,2,1,0} %slice.615), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_12"}
  %slice.1320 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1306), slice={[0:1], [0:58], [0:58], [52:56]}, metadata={op_type="Split" op_name="split_5"}
  %slice.616 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.286), slice={[0:3], [0:3], [0:4], [52:56]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.1344 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1320, f32[3,3,4,4]{3,2,1,0} %slice.616), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_13"}
  %slice.1321 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1306), slice={[0:1], [0:58], [0:58], [56:60]}, metadata={op_type="Split" op_name="split_5"}
  %slice.617 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.286), slice={[0:3], [0:3], [0:4], [56:60]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.1345 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1321, f32[3,3,4,4]{3,2,1,0} %slice.617), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_14"}
  %slice.1322 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1306), slice={[0:1], [0:58], [0:58], [60:64]}, metadata={op_type="Split" op_name="split_5"}
  %slice.618 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.286), slice={[0:3], [0:3], [0:4], [60:64]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.1346 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1322, f32[3,3,4,4]{3,2,1,0} %slice.618), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_15"}
  %slice.1323 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1306), slice={[0:1], [0:58], [0:58], [64:68]}, metadata={op_type="Split" op_name="split_5"}
  %slice.619 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.286), slice={[0:3], [0:3], [0:4], [64:68]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.1347 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1323, f32[3,3,4,4]{3,2,1,0} %slice.619), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_16"}
  %slice.1324 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1306), slice={[0:1], [0:58], [0:58], [68:72]}, metadata={op_type="Split" op_name="split_5"}
  %slice.620 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.286), slice={[0:3], [0:3], [0:4], [68:72]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.1348 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1324, f32[3,3,4,4]{3,2,1,0} %slice.620), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_17"}
  %slice.1325 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1306), slice={[0:1], [0:58], [0:58], [72:76]}, metadata={op_type="Split" op_name="split_5"}
  %slice.621 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.286), slice={[0:3], [0:3], [0:4], [72:76]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.1349 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1325, f32[3,3,4,4]{3,2,1,0} %slice.621), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_18"}
  %slice.1326 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1306), slice={[0:1], [0:58], [0:58], [76:80]}, metadata={op_type="Split" op_name="split_5"}
  %slice.622 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.286), slice={[0:3], [0:3], [0:4], [76:80]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.1350 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1326, f32[3,3,4,4]{3,2,1,0} %slice.622), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_19"}
  %slice.1327 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1306), slice={[0:1], [0:58], [0:58], [80:84]}, metadata={op_type="Split" op_name="split_5"}
  %slice.623 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.286), slice={[0:3], [0:3], [0:4], [80:84]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.1352 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1327, f32[3,3,4,4]{3,2,1,0} %slice.623), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_20"}
  %slice.1328 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1306), slice={[0:1], [0:58], [0:58], [84:88]}, metadata={op_type="Split" op_name="split_5"}
  %slice.624 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.286), slice={[0:3], [0:3], [0:4], [84:88]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.1353 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1328, f32[3,3,4,4]{3,2,1,0} %slice.624), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_21"}
  %slice.1329 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1306), slice={[0:1], [0:58], [0:58], [88:92]}, metadata={op_type="Split" op_name="split_5"}
  %slice.625 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.286), slice={[0:3], [0:3], [0:4], [88:92]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.1354 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1329, f32[3,3,4,4]{3,2,1,0} %slice.625), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_22"}
  %slice.1330 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1306), slice={[0:1], [0:58], [0:58], [92:96]}, metadata={op_type="Split" op_name="split_5"}
  %slice.626 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.286), slice={[0:3], [0:3], [0:4], [92:96]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.1355 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1330, f32[3,3,4,4]{3,2,1,0} %slice.626), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_23"}
  %slice.1331 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1306), slice={[0:1], [0:58], [0:58], [96:100]}, metadata={op_type="Split" op_name="split_5"}
  %slice.627 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.286), slice={[0:3], [0:3], [0:4], [96:100]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.1356 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1331, f32[3,3,4,4]{3,2,1,0} %slice.627), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_24"}
  %slice.1332 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1306), slice={[0:1], [0:58], [0:58], [100:104]}, metadata={op_type="Split" op_name="split_5"}
  %slice.628 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.286), slice={[0:3], [0:3], [0:4], [100:104]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.1357 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1332, f32[3,3,4,4]{3,2,1,0} %slice.628), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_25"}
  %slice.1333 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1306), slice={[0:1], [0:58], [0:58], [104:108]}, metadata={op_type="Split" op_name="split_5"}
  %slice.629 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.286), slice={[0:3], [0:3], [0:4], [104:108]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.1358 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1333, f32[3,3,4,4]{3,2,1,0} %slice.629), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_26"}
  %slice.1334 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1306), slice={[0:1], [0:58], [0:58], [108:112]}, metadata={op_type="Split" op_name="split_5"}
  %slice.630 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.286), slice={[0:3], [0:3], [0:4], [108:112]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.1359 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1334, f32[3,3,4,4]{3,2,1,0} %slice.630), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_27"}
  %slice.1335 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1306), slice={[0:1], [0:58], [0:58], [112:116]}, metadata={op_type="Split" op_name="split_5"}
  %slice.631 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.286), slice={[0:3], [0:3], [0:4], [112:116]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.1360 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1335, f32[3,3,4,4]{3,2,1,0} %slice.631), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_28"}
  %slice.1336 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1306), slice={[0:1], [0:58], [0:58], [116:120]}, metadata={op_type="Split" op_name="split_5"}
  %slice.632 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.286), slice={[0:3], [0:3], [0:4], [116:120]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.1361 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1336, f32[3,3,4,4]{3,2,1,0} %slice.632), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_29"}
  %slice.1337 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1306), slice={[0:1], [0:58], [0:58], [120:124]}, metadata={op_type="Split" op_name="split_5"}
  %slice.633 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.286), slice={[0:3], [0:3], [0:4], [120:124]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.1363 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1337, f32[3,3,4,4]{3,2,1,0} %slice.633), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_30"}
  %slice.1338 = f32[1,58,58,4]{3,2,1,0} slice(f32[1,58,58,128]{3,2,1,0} %pad.1306), slice={[0:1], [0:58], [0:58], [124:128]}, metadata={op_type="Split" op_name="split_5"}
  %slice.634 = f32[3,3,4,4]{3,2,1,0} slice(f32[3,3,4,128]{3,2,1,0} %reshape.286), slice={[0:3], [0:3], [0:4], [124:128]}, metadata={op_type="Split" op_name="split_4"}
  %convolution.1364 = f32[1,56,56,4]{3,2,1,0} convolution(f32[1,58,58,4]{3,2,1,0} %slice.1338, f32[3,3,4,4]{3,2,1,0} %slice.634), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv2_31"}
  %concatenate.1371 = f32[1,56,56,128]{3,2,1,0} concatenate(f32[1,56,56,4]{3,2,1,0} %convolution.1339, f32[1,56,56,4]{3,2,1,0} %convolution.1340, f32[1,56,56,4]{3,2,1,0} %convolution.1351, f32[1,56,56,4]{3,2,1,0} %convolution.1362, f32[1,56,56,4]{3,2,1,0} %convolution.1365, f32[1,56,56,4]{3,2,1,0} %convolution.1366, f32[1,56,56,4]{3,2,1,0} %convolution.1367, f32[1,56,56,4]{3,2,1,0} %convolution.1368, f32[1,56,56,4]{3,2,1,0} %convolution.1369, f32[1,56,56,4]{3,2,1,0} %convolution.1370, f32[1,56,56,4]{3,2,1,0} %convolution.1341, f32[1,56,56,4]{3,2,1,0} %convolution.1342, f32[1,56,56,4]{3,2,1,0} %convolution.1343, f32[1,56,56,4]{3,2,1,0} %convolution.1344, f32[1,56,56,4]{3,2,1,0} %convolution.1345, f32[1,56,56,4]{3,2,1,0} %convolution.1346, f32[1,56,56,4]{3,2,1,0} %convolution.1347, f32[1,56,56,4]{3,2,1,0} %convolution.1348, f32[1,56,56,4]{3,2,1,0} %convolution.1349, f32[1,56,56,4]{3,2,1,0} %convolution.1350, f32[1,56,56,4]{3,2,1,0} %convolution.1352, f32[1,56,56,4]{3,2,1,0} %convolution.1353, f32[1,56,56,4]{3,2,1,0} %convolution.1354, f32[1,56,56,4]{3,2,1,0} %convolution.1355, f32[1,56,56,4]{3,2,1,0} %convolution.1356, f32[1,56,56,4]{3,2,1,0} %convolution.1357, f32[1,56,56,4]{3,2,1,0} %convolution.1358, f32[1,56,56,4]{3,2,1,0} %convolution.1359, f32[1,56,56,4]{3,2,1,0} %convolution.1360, f32[1,56,56,4]{3,2,1,0} %convolution.1361, f32[1,56,56,4]{3,2,1,0} %convolution.1363, f32[1,56,56,4]{3,2,1,0} %convolution.1364), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_2"}
  %arg18.19 = f32[128]{0} parameter(18), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.288 = f32[128]{0} reshape(f32[128]{0} %arg18.19)
  %constant.1372 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage1_unit3_bn2/add"}
  %broadcast.1373 = f32[128]{0} broadcast(f32[] %constant.1372), dimensions={}, metadata={op_type="AddV2" op_name="stage1_unit3_bn2/add"}
  %add.1374 = f32[128]{0} add(f32[128]{0} %reshape.288, f32[128]{0} %broadcast.1373), metadata={op_type="AddV2" op_name="stage1_unit3_bn2/add"}
  %rsqrt.1375 = f32[128]{0} rsqrt(f32[128]{0} %add.1374), metadata={op_type="Rsqrt" op_name="stage1_unit3_bn2/Rsqrt"}
  %arg83.84 = f32[128]{0} parameter(83), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.353 = f32[128]{0} reshape(f32[128]{0} %arg83.84)
  %multiply.1376 = f32[128]{0} multiply(f32[128]{0} %rsqrt.1375, f32[128]{0} %reshape.353), metadata={op_type="Mul" op_name="stage1_unit3_bn2/mul"}
  %broadcast.1377 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %multiply.1376), dimensions={3}, metadata={op_type="Mul" op_name="stage1_unit3_bn2/mul_1"}
  %multiply.1378 = f32[1,56,56,128]{3,2,1,0} multiply(f32[1,56,56,128]{3,2,1,0} %concatenate.1371, f32[1,56,56,128]{3,2,1,0} %broadcast.1377), metadata={op_type="Mul" op_name="stage1_unit3_bn2/mul_1"}
  %arg190.191 = f32[128]{0} parameter(190), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.460 = f32[128]{0} reshape(f32[128]{0} %arg190.191)
  %arg137.138 = f32[128]{0} parameter(137), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.407 = f32[128]{0} reshape(f32[128]{0} %arg137.138)
  %multiply.1379 = f32[128]{0} multiply(f32[128]{0} %multiply.1376, f32[128]{0} %reshape.407), metadata={op_type="Mul" op_name="stage1_unit3_bn2/mul_2"}
  %subtract.1380 = f32[128]{0} subtract(f32[128]{0} %reshape.460, f32[128]{0} %multiply.1379), metadata={op_type="Sub" op_name="stage1_unit3_bn2/sub"}
  %broadcast.1381 = f32[1,56,56,128]{3,2,1,0} broadcast(f32[128]{0} %subtract.1380), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit3_bn2/add_1"}
  %add.1382 = f32[1,56,56,128]{3,2,1,0} add(f32[1,56,56,128]{3,2,1,0} %multiply.1378, f32[1,56,56,128]{3,2,1,0} %broadcast.1381), metadata={op_type="AddV2" op_name="stage1_unit3_bn2/add_1"}
  %maximum.1385 = f32[1,56,56,128]{3,2,1,0} maximum(f32[1,56,56,128]{3,2,1,0} %broadcast.1384, f32[1,56,56,128]{3,2,1,0} %add.1382), metadata={op_type="Relu" op_name="stage1_unit3_relu2"}
  %arg238.239 = f32[1,1,128,256]{3,2,1,0} parameter(238), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.508 = f32[1,1,128,256]{3,2,1,0} reshape(f32[1,1,128,256]{3,2,1,0} %arg238.239)
  %convolution.1386 = f32[1,56,56,256]{3,2,1,0} convolution(f32[1,56,56,128]{3,2,1,0} %maximum.1385, f32[1,1,128,256]{3,2,1,0} %reshape.508), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage1_unit3_conv3"}
  %multiply.1393 = f32[1,56,56,256]{3,2,1,0} multiply(f32[1,56,56,256]{3,2,1,0} %broadcast.1392, f32[1,56,56,256]{3,2,1,0} %convolution.1386), metadata={op_type="Mul" op_name="stage1_unit3_bn3/mul_1"}
  %arg191.192 = f32[256]{0} parameter(191), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.461 = f32[256]{0} reshape(f32[256]{0} %arg191.192)
  %arg138.139 = f32[256]{0} parameter(138), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.408 = f32[256]{0} reshape(f32[256]{0} %arg138.139)
  %multiply.1394 = f32[256]{0} multiply(f32[256]{0} %multiply.1391, f32[256]{0} %reshape.408), metadata={op_type="Mul" op_name="stage1_unit3_bn3/mul_2"}
  %subtract.1395 = f32[256]{0} subtract(f32[256]{0} %reshape.461, f32[256]{0} %multiply.1394), metadata={op_type="Sub" op_name="stage1_unit3_bn3/sub"}
  %broadcast.1396 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.1395), dimensions={3}, metadata={op_type="AddV2" op_name="stage1_unit3_bn3/add_1"}
  %add.1397 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %multiply.1393, f32[1,56,56,256]{3,2,1,0} %broadcast.1396), metadata={op_type="AddV2" op_name="stage1_unit3_bn3/add_1"}
  %add.1398 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %maximum.1289, f32[1,56,56,256]{3,2,1,0} %add.1397), metadata={op_type="AddV2" op_name="add_2"}
  %maximum.1401 = f32[1,56,56,256]{3,2,1,0} maximum(f32[1,56,56,256]{3,2,1,0} %broadcast.1400, f32[1,56,56,256]{3,2,1,0} %add.1398), metadata={op_type="Relu" op_name="stage1_unit3_relu"}
  %arg239.240 = f32[1,1,256,256]{3,2,1,0} parameter(239), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.509 = f32[1,1,256,256]{3,2,1,0} reshape(f32[1,1,256,256]{3,2,1,0} %arg239.240)
  %convolution.1402 = f32[1,56,56,256]{3,2,1,0} convolution(f32[1,56,56,256]{3,2,1,0} %maximum.1401, f32[1,1,256,256]{3,2,1,0} %reshape.509), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv1"}
  %multiply.1410 = f32[1,56,56,256]{3,2,1,0} multiply(f32[1,56,56,256]{3,2,1,0} %broadcast.1409, f32[1,56,56,256]{3,2,1,0} %convolution.1402), metadata={op_type="Mul" op_name="stage2_unit1_bn1/mul_1"}
  %arg192.193 = f32[256]{0} parameter(192), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.462 = f32[256]{0} reshape(f32[256]{0} %arg192.193)
  %arg139.140 = f32[256]{0} parameter(139), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.409 = f32[256]{0} reshape(f32[256]{0} %arg139.140)
  %multiply.1411 = f32[256]{0} multiply(f32[256]{0} %multiply.1408, f32[256]{0} %reshape.409), metadata={op_type="Mul" op_name="stage2_unit1_bn1/mul_2"}
  %subtract.1412 = f32[256]{0} subtract(f32[256]{0} %reshape.462, f32[256]{0} %multiply.1411), metadata={op_type="Sub" op_name="stage2_unit1_bn1/sub"}
  %broadcast.1413 = f32[1,56,56,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.1412), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit1_bn1/add_1"}
  %add.1414 = f32[1,56,56,256]{3,2,1,0} add(f32[1,56,56,256]{3,2,1,0} %multiply.1410, f32[1,56,56,256]{3,2,1,0} %broadcast.1413), metadata={op_type="AddV2" op_name="stage2_unit1_bn1/add_1"}
  %maximum.1417 = f32[1,56,56,256]{3,2,1,0} maximum(f32[1,56,56,256]{3,2,1,0} %broadcast.1416, f32[1,56,56,256]{3,2,1,0} %add.1414), metadata={op_type="Relu" op_name="stage2_unit1_relu1"}
  %constant.1418 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_4"}
  %pad.1419 = f32[1,58,58,256]{3,2,1,0} pad(f32[1,56,56,256]{3,2,1,0} %maximum.1417, f32[] %constant.1418), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_4"}
  %slice.1420 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1419), slice={[0:1], [0:58], [0:58], [0:8]}, metadata={op_type="Split" op_name="split_7"}
  %arg23.24 = f32[3,3,8,256]{3,2,1,0} parameter(23), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.293 = f32[3,3,8,256]{3,2,1,0} reshape(f32[3,3,8,256]{3,2,1,0} %arg23.24)
  %slice.635 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.293), slice={[0:3], [0:3], [0:8], [0:8]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1452 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1420, f32[3,3,8,8]{3,2,1,0} %slice.635), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2"}
  %slice.1421 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1419), slice={[0:1], [0:58], [0:58], [8:16]}, metadata={op_type="Split" op_name="split_7"}
  %slice.636 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.293), slice={[0:3], [0:3], [0:8], [8:16]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1453 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1421, f32[3,3,8,8]{3,2,1,0} %slice.636), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_1"}
  %slice.1422 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1419), slice={[0:1], [0:58], [0:58], [16:24]}, metadata={op_type="Split" op_name="split_7"}
  %slice.637 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.293), slice={[0:3], [0:3], [0:8], [16:24]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1464 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1422, f32[3,3,8,8]{3,2,1,0} %slice.637), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_2"}
  %slice.1423 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1419), slice={[0:1], [0:58], [0:58], [24:32]}, metadata={op_type="Split" op_name="split_7"}
  %slice.638 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.293), slice={[0:3], [0:3], [0:8], [24:32]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1475 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1423, f32[3,3,8,8]{3,2,1,0} %slice.638), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_3"}
  %slice.1424 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1419), slice={[0:1], [0:58], [0:58], [32:40]}, metadata={op_type="Split" op_name="split_7"}
  %slice.639 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.293), slice={[0:3], [0:3], [0:8], [32:40]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1478 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1424, f32[3,3,8,8]{3,2,1,0} %slice.639), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_4"}
  %slice.1425 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1419), slice={[0:1], [0:58], [0:58], [40:48]}, metadata={op_type="Split" op_name="split_7"}
  %slice.640 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.293), slice={[0:3], [0:3], [0:8], [40:48]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1479 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1425, f32[3,3,8,8]{3,2,1,0} %slice.640), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_5"}
  %slice.1426 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1419), slice={[0:1], [0:58], [0:58], [48:56]}, metadata={op_type="Split" op_name="split_7"}
  %slice.641 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.293), slice={[0:3], [0:3], [0:8], [48:56]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1480 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1426, f32[3,3,8,8]{3,2,1,0} %slice.641), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_6"}
  %slice.1427 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1419), slice={[0:1], [0:58], [0:58], [56:64]}, metadata={op_type="Split" op_name="split_7"}
  %slice.642 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.293), slice={[0:3], [0:3], [0:8], [56:64]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1481 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1427, f32[3,3,8,8]{3,2,1,0} %slice.642), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_7"}
  %slice.1428 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1419), slice={[0:1], [0:58], [0:58], [64:72]}, metadata={op_type="Split" op_name="split_7"}
  %slice.643 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.293), slice={[0:3], [0:3], [0:8], [64:72]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1482 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1428, f32[3,3,8,8]{3,2,1,0} %slice.643), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_8"}
  %slice.1429 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1419), slice={[0:1], [0:58], [0:58], [72:80]}, metadata={op_type="Split" op_name="split_7"}
  %slice.644 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.293), slice={[0:3], [0:3], [0:8], [72:80]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1483 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1429, f32[3,3,8,8]{3,2,1,0} %slice.644), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_9"}
  %slice.1430 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1419), slice={[0:1], [0:58], [0:58], [80:88]}, metadata={op_type="Split" op_name="split_7"}
  %slice.645 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.293), slice={[0:3], [0:3], [0:8], [80:88]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1454 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1430, f32[3,3,8,8]{3,2,1,0} %slice.645), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_10"}
  %slice.1431 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1419), slice={[0:1], [0:58], [0:58], [88:96]}, metadata={op_type="Split" op_name="split_7"}
  %slice.646 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.293), slice={[0:3], [0:3], [0:8], [88:96]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1455 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1431, f32[3,3,8,8]{3,2,1,0} %slice.646), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_11"}
  %slice.1432 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1419), slice={[0:1], [0:58], [0:58], [96:104]}, metadata={op_type="Split" op_name="split_7"}
  %slice.647 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.293), slice={[0:3], [0:3], [0:8], [96:104]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1456 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1432, f32[3,3,8,8]{3,2,1,0} %slice.647), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_12"}
  %slice.1433 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1419), slice={[0:1], [0:58], [0:58], [104:112]}, metadata={op_type="Split" op_name="split_7"}
  %slice.648 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.293), slice={[0:3], [0:3], [0:8], [104:112]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1457 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1433, f32[3,3,8,8]{3,2,1,0} %slice.648), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_13"}
  %slice.1434 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1419), slice={[0:1], [0:58], [0:58], [112:120]}, metadata={op_type="Split" op_name="split_7"}
  %slice.649 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.293), slice={[0:3], [0:3], [0:8], [112:120]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1458 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1434, f32[3,3,8,8]{3,2,1,0} %slice.649), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_14"}
  %slice.1435 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1419), slice={[0:1], [0:58], [0:58], [120:128]}, metadata={op_type="Split" op_name="split_7"}
  %slice.650 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.293), slice={[0:3], [0:3], [0:8], [120:128]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1459 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1435, f32[3,3,8,8]{3,2,1,0} %slice.650), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_15"}
  %slice.1436 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1419), slice={[0:1], [0:58], [0:58], [128:136]}, metadata={op_type="Split" op_name="split_7"}
  %slice.651 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.293), slice={[0:3], [0:3], [0:8], [128:136]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1460 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1436, f32[3,3,8,8]{3,2,1,0} %slice.651), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_16"}
  %slice.1437 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1419), slice={[0:1], [0:58], [0:58], [136:144]}, metadata={op_type="Split" op_name="split_7"}
  %slice.652 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.293), slice={[0:3], [0:3], [0:8], [136:144]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1461 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1437, f32[3,3,8,8]{3,2,1,0} %slice.652), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_17"}
  %slice.1438 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1419), slice={[0:1], [0:58], [0:58], [144:152]}, metadata={op_type="Split" op_name="split_7"}
  %slice.653 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.293), slice={[0:3], [0:3], [0:8], [144:152]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1462 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1438, f32[3,3,8,8]{3,2,1,0} %slice.653), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_18"}
  %slice.1439 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1419), slice={[0:1], [0:58], [0:58], [152:160]}, metadata={op_type="Split" op_name="split_7"}
  %slice.654 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.293), slice={[0:3], [0:3], [0:8], [152:160]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1463 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1439, f32[3,3,8,8]{3,2,1,0} %slice.654), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_19"}
  %slice.1440 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1419), slice={[0:1], [0:58], [0:58], [160:168]}, metadata={op_type="Split" op_name="split_7"}
  %slice.655 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.293), slice={[0:3], [0:3], [0:8], [160:168]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1465 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1440, f32[3,3,8,8]{3,2,1,0} %slice.655), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_20"}
  %slice.1441 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1419), slice={[0:1], [0:58], [0:58], [168:176]}, metadata={op_type="Split" op_name="split_7"}
  %slice.656 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.293), slice={[0:3], [0:3], [0:8], [168:176]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1466 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1441, f32[3,3,8,8]{3,2,1,0} %slice.656), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_21"}
  %slice.1442 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1419), slice={[0:1], [0:58], [0:58], [176:184]}, metadata={op_type="Split" op_name="split_7"}
  %slice.657 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.293), slice={[0:3], [0:3], [0:8], [176:184]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1467 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1442, f32[3,3,8,8]{3,2,1,0} %slice.657), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_22"}
  %slice.1443 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1419), slice={[0:1], [0:58], [0:58], [184:192]}, metadata={op_type="Split" op_name="split_7"}
  %slice.658 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.293), slice={[0:3], [0:3], [0:8], [184:192]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1468 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1443, f32[3,3,8,8]{3,2,1,0} %slice.658), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_23"}
  %slice.1444 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1419), slice={[0:1], [0:58], [0:58], [192:200]}, metadata={op_type="Split" op_name="split_7"}
  %slice.659 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.293), slice={[0:3], [0:3], [0:8], [192:200]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1469 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1444, f32[3,3,8,8]{3,2,1,0} %slice.659), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_24"}
  %slice.1445 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1419), slice={[0:1], [0:58], [0:58], [200:208]}, metadata={op_type="Split" op_name="split_7"}
  %slice.660 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.293), slice={[0:3], [0:3], [0:8], [200:208]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1470 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1445, f32[3,3,8,8]{3,2,1,0} %slice.660), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_25"}
  %slice.1446 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1419), slice={[0:1], [0:58], [0:58], [208:216]}, metadata={op_type="Split" op_name="split_7"}
  %slice.661 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.293), slice={[0:3], [0:3], [0:8], [208:216]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1471 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1446, f32[3,3,8,8]{3,2,1,0} %slice.661), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_26"}
  %slice.1447 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1419), slice={[0:1], [0:58], [0:58], [216:224]}, metadata={op_type="Split" op_name="split_7"}
  %slice.662 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.293), slice={[0:3], [0:3], [0:8], [216:224]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1472 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1447, f32[3,3,8,8]{3,2,1,0} %slice.662), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_27"}
  %slice.1448 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1419), slice={[0:1], [0:58], [0:58], [224:232]}, metadata={op_type="Split" op_name="split_7"}
  %slice.663 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.293), slice={[0:3], [0:3], [0:8], [224:232]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1473 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1448, f32[3,3,8,8]{3,2,1,0} %slice.663), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_28"}
  %slice.1449 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1419), slice={[0:1], [0:58], [0:58], [232:240]}, metadata={op_type="Split" op_name="split_7"}
  %slice.664 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.293), slice={[0:3], [0:3], [0:8], [232:240]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1474 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1449, f32[3,3,8,8]{3,2,1,0} %slice.664), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_29"}
  %slice.1450 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1419), slice={[0:1], [0:58], [0:58], [240:248]}, metadata={op_type="Split" op_name="split_7"}
  %slice.665 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.293), slice={[0:3], [0:3], [0:8], [240:248]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1476 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1450, f32[3,3,8,8]{3,2,1,0} %slice.665), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_30"}
  %slice.1451 = f32[1,58,58,8]{3,2,1,0} slice(f32[1,58,58,256]{3,2,1,0} %pad.1419), slice={[0:1], [0:58], [0:58], [248:256]}, metadata={op_type="Split" op_name="split_7"}
  %slice.666 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.293), slice={[0:3], [0:3], [0:8], [248:256]}, metadata={op_type="Split" op_name="split_6"}
  %convolution.1477 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,58,58,8]{3,2,1,0} %slice.1451, f32[3,3,8,8]{3,2,1,0} %slice.666), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv2_31"}
  %concatenate.1484 = f32[1,28,28,256]{3,2,1,0} concatenate(f32[1,28,28,8]{3,2,1,0} %convolution.1452, f32[1,28,28,8]{3,2,1,0} %convolution.1453, f32[1,28,28,8]{3,2,1,0} %convolution.1464, f32[1,28,28,8]{3,2,1,0} %convolution.1475, f32[1,28,28,8]{3,2,1,0} %convolution.1478, f32[1,28,28,8]{3,2,1,0} %convolution.1479, f32[1,28,28,8]{3,2,1,0} %convolution.1480, f32[1,28,28,8]{3,2,1,0} %convolution.1481, f32[1,28,28,8]{3,2,1,0} %convolution.1482, f32[1,28,28,8]{3,2,1,0} %convolution.1483, f32[1,28,28,8]{3,2,1,0} %convolution.1454, f32[1,28,28,8]{3,2,1,0} %convolution.1455, f32[1,28,28,8]{3,2,1,0} %convolution.1456, f32[1,28,28,8]{3,2,1,0} %convolution.1457, f32[1,28,28,8]{3,2,1,0} %convolution.1458, f32[1,28,28,8]{3,2,1,0} %convolution.1459, f32[1,28,28,8]{3,2,1,0} %convolution.1460, f32[1,28,28,8]{3,2,1,0} %convolution.1461, f32[1,28,28,8]{3,2,1,0} %convolution.1462, f32[1,28,28,8]{3,2,1,0} %convolution.1463, f32[1,28,28,8]{3,2,1,0} %convolution.1465, f32[1,28,28,8]{3,2,1,0} %convolution.1466, f32[1,28,28,8]{3,2,1,0} %convolution.1467, f32[1,28,28,8]{3,2,1,0} %convolution.1468, f32[1,28,28,8]{3,2,1,0} %convolution.1469, f32[1,28,28,8]{3,2,1,0} %convolution.1470, f32[1,28,28,8]{3,2,1,0} %convolution.1471, f32[1,28,28,8]{3,2,1,0} %convolution.1472, f32[1,28,28,8]{3,2,1,0} %convolution.1473, f32[1,28,28,8]{3,2,1,0} %convolution.1474, f32[1,28,28,8]{3,2,1,0} %convolution.1476, f32[1,28,28,8]{3,2,1,0} %convolution.1477), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_3"}
  %arg24.25 = f32[256]{0} parameter(24), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.294 = f32[256]{0} reshape(f32[256]{0} %arg24.25)
  %constant.1485 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit1_bn2/add"}
  %broadcast.1486 = f32[256]{0} broadcast(f32[] %constant.1485), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit1_bn2/add"}
  %add.1487 = f32[256]{0} add(f32[256]{0} %reshape.294, f32[256]{0} %broadcast.1486), metadata={op_type="AddV2" op_name="stage2_unit1_bn2/add"}
  %rsqrt.1488 = f32[256]{0} rsqrt(f32[256]{0} %add.1487), metadata={op_type="Rsqrt" op_name="stage2_unit1_bn2/Rsqrt"}
  %arg88.89 = f32[256]{0} parameter(88), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.358 = f32[256]{0} reshape(f32[256]{0} %arg88.89)
  %multiply.1489 = f32[256]{0} multiply(f32[256]{0} %rsqrt.1488, f32[256]{0} %reshape.358), metadata={op_type="Mul" op_name="stage2_unit1_bn2/mul"}
  %broadcast.1490 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.1489), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit1_bn2/mul_1"}
  %multiply.1491 = f32[1,28,28,256]{3,2,1,0} multiply(f32[1,28,28,256]{3,2,1,0} %concatenate.1484, f32[1,28,28,256]{3,2,1,0} %broadcast.1490), metadata={op_type="Mul" op_name="stage2_unit1_bn2/mul_1"}
  %arg195.196 = f32[256]{0} parameter(195), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.465 = f32[256]{0} reshape(f32[256]{0} %arg195.196)
  %arg142.143 = f32[256]{0} parameter(142), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.412 = f32[256]{0} reshape(f32[256]{0} %arg142.143)
  %multiply.1492 = f32[256]{0} multiply(f32[256]{0} %multiply.1489, f32[256]{0} %reshape.412), metadata={op_type="Mul" op_name="stage2_unit1_bn2/mul_2"}
  %subtract.1493 = f32[256]{0} subtract(f32[256]{0} %reshape.465, f32[256]{0} %multiply.1492), metadata={op_type="Sub" op_name="stage2_unit1_bn2/sub"}
  %broadcast.1494 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.1493), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit1_bn2/add_1"}
  %add.1495 = f32[1,28,28,256]{3,2,1,0} add(f32[1,28,28,256]{3,2,1,0} %multiply.1491, f32[1,28,28,256]{3,2,1,0} %broadcast.1494), metadata={op_type="AddV2" op_name="stage2_unit1_bn2/add_1"}
  %maximum.1498 = f32[1,28,28,256]{3,2,1,0} maximum(f32[1,28,28,256]{3,2,1,0} %broadcast.1497, f32[1,28,28,256]{3,2,1,0} %add.1495), metadata={op_type="Relu" op_name="stage2_unit1_relu2"}
  %arg241.242 = f32[1,1,256,512]{3,2,1,0} parameter(241), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.511 = f32[1,1,256,512]{3,2,1,0} reshape(f32[1,1,256,512]{3,2,1,0} %arg241.242)
  %convolution.1499 = f32[1,28,28,512]{3,2,1,0} convolution(f32[1,28,28,256]{3,2,1,0} %maximum.1498, f32[1,1,256,512]{3,2,1,0} %reshape.511), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_conv3"}
  %multiply.1506 = f32[1,28,28,512]{3,2,1,0} multiply(f32[1,28,28,512]{3,2,1,0} %broadcast.1505, f32[1,28,28,512]{3,2,1,0} %convolution.1499), metadata={op_type="Mul" op_name="stage2_unit1_bn3/mul_1"}
  %arg196.197 = f32[512]{0} parameter(196), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.466 = f32[512]{0} reshape(f32[512]{0} %arg196.197)
  %arg143.144 = f32[512]{0} parameter(143), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.413 = f32[512]{0} reshape(f32[512]{0} %arg143.144)
  %multiply.1507 = f32[512]{0} multiply(f32[512]{0} %multiply.1504, f32[512]{0} %reshape.413), metadata={op_type="Mul" op_name="stage2_unit1_bn3/mul_2"}
  %subtract.1508 = f32[512]{0} subtract(f32[512]{0} %reshape.466, f32[512]{0} %multiply.1507), metadata={op_type="Sub" op_name="stage2_unit1_bn3/sub"}
  %broadcast.1509 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.1508), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit1_bn3/add_1"}
  %add.1510 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %multiply.1506, f32[1,28,28,512]{3,2,1,0} %broadcast.1509), metadata={op_type="AddV2" op_name="stage2_unit1_bn3/add_1"}
  %arg240.241 = f32[1,1,256,512]{3,2,1,0} parameter(240), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.510 = f32[1,1,256,512]{3,2,1,0} reshape(f32[1,1,256,512]{3,2,1,0} %arg240.241)
  %convolution.1403 = f32[1,28,28,512]{3,2,1,0} convolution(f32[1,56,56,256]{3,2,1,0} %maximum.1401, f32[1,1,256,512]{3,2,1,0} %reshape.510), window={size=1x1 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit1_sc"}
  %arg21.22 = f32[512]{0} parameter(21), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.291 = f32[512]{0} reshape(f32[512]{0} %arg21.22)
  %constant.1511 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit1_sc_bn/add"}
  %broadcast.1512 = f32[512]{0} broadcast(f32[] %constant.1511), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit1_sc_bn/add"}
  %add.1513 = f32[512]{0} add(f32[512]{0} %reshape.291, f32[512]{0} %broadcast.1512), metadata={op_type="AddV2" op_name="stage2_unit1_sc_bn/add"}
  %rsqrt.1514 = f32[512]{0} rsqrt(f32[512]{0} %add.1513), metadata={op_type="Rsqrt" op_name="stage2_unit1_sc_bn/Rsqrt"}
  %arg86.87 = f32[512]{0} parameter(86), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.356 = f32[512]{0} reshape(f32[512]{0} %arg86.87)
  %multiply.1515 = f32[512]{0} multiply(f32[512]{0} %rsqrt.1514, f32[512]{0} %reshape.356), metadata={op_type="Mul" op_name="stage2_unit1_sc_bn/mul"}
  %broadcast.1516 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.1515), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit1_sc_bn/mul_1"}
  %multiply.1517 = f32[1,28,28,512]{3,2,1,0} multiply(f32[1,28,28,512]{3,2,1,0} %convolution.1403, f32[1,28,28,512]{3,2,1,0} %broadcast.1516), metadata={op_type="Mul" op_name="stage2_unit1_sc_bn/mul_1"}
  %arg193.194 = f32[512]{0} parameter(193), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.463 = f32[512]{0} reshape(f32[512]{0} %arg193.194)
  %arg140.141 = f32[512]{0} parameter(140), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.410 = f32[512]{0} reshape(f32[512]{0} %arg140.141)
  %multiply.1518 = f32[512]{0} multiply(f32[512]{0} %multiply.1515, f32[512]{0} %reshape.410), metadata={op_type="Mul" op_name="stage2_unit1_sc_bn/mul_2"}
  %subtract.1519 = f32[512]{0} subtract(f32[512]{0} %reshape.463, f32[512]{0} %multiply.1518), metadata={op_type="Sub" op_name="stage2_unit1_sc_bn/sub"}
  %broadcast.1520 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.1519), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit1_sc_bn/add_1"}
  %add.1521 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %multiply.1517, f32[1,28,28,512]{3,2,1,0} %broadcast.1520), metadata={op_type="AddV2" op_name="stage2_unit1_sc_bn/add_1"}
  %add.1522 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %add.1510, f32[1,28,28,512]{3,2,1,0} %add.1521), metadata={op_type="AddV2" op_name="add_3"}
  %maximum.1525 = f32[1,28,28,512]{3,2,1,0} maximum(f32[1,28,28,512]{3,2,1,0} %broadcast.1524, f32[1,28,28,512]{3,2,1,0} %add.1522), metadata={op_type="Relu" op_name="stage2_unit1_relu"}
  %arg31.32 = f32[512]{0} parameter(31), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.301 = f32[512]{0} reshape(f32[512]{0} %arg31.32)
  %constant.1623 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit2_bn3/add"}
  %broadcast.1624 = f32[512]{0} broadcast(f32[] %constant.1623), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit2_bn3/add"}
  %add.1625 = f32[512]{0} add(f32[512]{0} %reshape.301, f32[512]{0} %broadcast.1624), metadata={op_type="AddV2" op_name="stage2_unit2_bn3/add"}
  %rsqrt.1626 = f32[512]{0} rsqrt(f32[512]{0} %add.1625), metadata={op_type="Rsqrt" op_name="stage2_unit2_bn3/Rsqrt"}
  %arg93.94 = f32[512]{0} parameter(93), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.363 = f32[512]{0} reshape(f32[512]{0} %arg93.94)
  %multiply.1627 = f32[512]{0} multiply(f32[512]{0} %rsqrt.1626, f32[512]{0} %reshape.363), metadata={op_type="Mul" op_name="stage2_unit2_bn3/mul"}
  %broadcast.1628 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.1627), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit2_bn3/mul_1"}
  %constant.1619 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit2_relu2"}
  %broadcast.1620 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[] %constant.1619), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit2_relu2"}
  %constant.1538 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit2_relu1"}
  %broadcast.1539 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[] %constant.1538), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit2_relu1"}
  %arg27.28 = f32[256]{0} parameter(27), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.297 = f32[256]{0} reshape(f32[256]{0} %arg27.28)
  %constant.1527 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit2_bn1/add"}
  %broadcast.1528 = f32[256]{0} broadcast(f32[] %constant.1527), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit2_bn1/add"}
  %add.1529 = f32[256]{0} add(f32[256]{0} %reshape.297, f32[256]{0} %broadcast.1528), metadata={op_type="AddV2" op_name="stage2_unit2_bn1/add"}
  %rsqrt.1530 = f32[256]{0} rsqrt(f32[256]{0} %add.1529), metadata={op_type="Rsqrt" op_name="stage2_unit2_bn1/Rsqrt"}
  %arg91.92 = f32[256]{0} parameter(91), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.361 = f32[256]{0} reshape(f32[256]{0} %arg91.92)
  %multiply.1531 = f32[256]{0} multiply(f32[256]{0} %rsqrt.1530, f32[256]{0} %reshape.361), metadata={op_type="Mul" op_name="stage2_unit2_bn1/mul"}
  %broadcast.1532 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.1531), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit2_bn1/mul_1"}
  %arg242.243 = f32[1,1,512,256]{3,2,1,0} parameter(242), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.512 = f32[1,1,512,256]{3,2,1,0} reshape(f32[1,1,512,256]{3,2,1,0} %arg242.243)
  %convolution.1526 = f32[1,28,28,256]{3,2,1,0} convolution(f32[1,28,28,512]{3,2,1,0} %maximum.1525, f32[1,1,512,256]{3,2,1,0} %reshape.512), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv1"}
  %multiply.1533 = f32[1,28,28,256]{3,2,1,0} multiply(f32[1,28,28,256]{3,2,1,0} %broadcast.1532, f32[1,28,28,256]{3,2,1,0} %convolution.1526), metadata={op_type="Mul" op_name="stage2_unit2_bn1/mul_1"}
  %arg198.199 = f32[256]{0} parameter(198), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.468 = f32[256]{0} reshape(f32[256]{0} %arg198.199)
  %arg145.146 = f32[256]{0} parameter(145), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.415 = f32[256]{0} reshape(f32[256]{0} %arg145.146)
  %multiply.1534 = f32[256]{0} multiply(f32[256]{0} %multiply.1531, f32[256]{0} %reshape.415), metadata={op_type="Mul" op_name="stage2_unit2_bn1/mul_2"}
  %subtract.1535 = f32[256]{0} subtract(f32[256]{0} %reshape.468, f32[256]{0} %multiply.1534), metadata={op_type="Sub" op_name="stage2_unit2_bn1/sub"}
  %broadcast.1536 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.1535), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit2_bn1/add_1"}
  %add.1537 = f32[1,28,28,256]{3,2,1,0} add(f32[1,28,28,256]{3,2,1,0} %multiply.1533, f32[1,28,28,256]{3,2,1,0} %broadcast.1536), metadata={op_type="AddV2" op_name="stage2_unit2_bn1/add_1"}
  %maximum.1540 = f32[1,28,28,256]{3,2,1,0} maximum(f32[1,28,28,256]{3,2,1,0} %broadcast.1539, f32[1,28,28,256]{3,2,1,0} %add.1537), metadata={op_type="Relu" op_name="stage2_unit2_relu1"}
  %constant.1541 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_5"}
  %pad.1542 = f32[1,30,30,256]{3,2,1,0} pad(f32[1,28,28,256]{3,2,1,0} %maximum.1540, f32[] %constant.1541), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_5"}
  %slice.1543 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1542), slice={[0:1], [0:30], [0:30], [0:8]}, metadata={op_type="Split" op_name="split_9"}
  %arg28.29 = f32[3,3,8,256]{3,2,1,0} parameter(28), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.298 = f32[3,3,8,256]{3,2,1,0} reshape(f32[3,3,8,256]{3,2,1,0} %arg28.29)
  %slice.667 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.298), slice={[0:3], [0:3], [0:8], [0:8]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1575 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1543, f32[3,3,8,8]{3,2,1,0} %slice.667), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2"}
  %slice.1544 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1542), slice={[0:1], [0:30], [0:30], [8:16]}, metadata={op_type="Split" op_name="split_9"}
  %slice.668 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.298), slice={[0:3], [0:3], [0:8], [8:16]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1576 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1544, f32[3,3,8,8]{3,2,1,0} %slice.668), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_1"}
  %slice.1545 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1542), slice={[0:1], [0:30], [0:30], [16:24]}, metadata={op_type="Split" op_name="split_9"}
  %slice.669 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.298), slice={[0:3], [0:3], [0:8], [16:24]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1587 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1545, f32[3,3,8,8]{3,2,1,0} %slice.669), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_2"}
  %slice.1546 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1542), slice={[0:1], [0:30], [0:30], [24:32]}, metadata={op_type="Split" op_name="split_9"}
  %slice.670 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.298), slice={[0:3], [0:3], [0:8], [24:32]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1598 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1546, f32[3,3,8,8]{3,2,1,0} %slice.670), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_3"}
  %slice.1547 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1542), slice={[0:1], [0:30], [0:30], [32:40]}, metadata={op_type="Split" op_name="split_9"}
  %slice.671 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.298), slice={[0:3], [0:3], [0:8], [32:40]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1601 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1547, f32[3,3,8,8]{3,2,1,0} %slice.671), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_4"}
  %slice.1548 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1542), slice={[0:1], [0:30], [0:30], [40:48]}, metadata={op_type="Split" op_name="split_9"}
  %slice.672 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.298), slice={[0:3], [0:3], [0:8], [40:48]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1602 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1548, f32[3,3,8,8]{3,2,1,0} %slice.672), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_5"}
  %slice.1549 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1542), slice={[0:1], [0:30], [0:30], [48:56]}, metadata={op_type="Split" op_name="split_9"}
  %slice.673 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.298), slice={[0:3], [0:3], [0:8], [48:56]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1603 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1549, f32[3,3,8,8]{3,2,1,0} %slice.673), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_6"}
  %slice.1550 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1542), slice={[0:1], [0:30], [0:30], [56:64]}, metadata={op_type="Split" op_name="split_9"}
  %slice.674 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.298), slice={[0:3], [0:3], [0:8], [56:64]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1604 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1550, f32[3,3,8,8]{3,2,1,0} %slice.674), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_7"}
  %slice.1551 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1542), slice={[0:1], [0:30], [0:30], [64:72]}, metadata={op_type="Split" op_name="split_9"}
  %slice.675 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.298), slice={[0:3], [0:3], [0:8], [64:72]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1605 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1551, f32[3,3,8,8]{3,2,1,0} %slice.675), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_8"}
  %slice.1552 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1542), slice={[0:1], [0:30], [0:30], [72:80]}, metadata={op_type="Split" op_name="split_9"}
  %slice.676 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.298), slice={[0:3], [0:3], [0:8], [72:80]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1606 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1552, f32[3,3,8,8]{3,2,1,0} %slice.676), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_9"}
  %slice.1553 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1542), slice={[0:1], [0:30], [0:30], [80:88]}, metadata={op_type="Split" op_name="split_9"}
  %slice.677 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.298), slice={[0:3], [0:3], [0:8], [80:88]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1577 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1553, f32[3,3,8,8]{3,2,1,0} %slice.677), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_10"}
  %slice.1554 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1542), slice={[0:1], [0:30], [0:30], [88:96]}, metadata={op_type="Split" op_name="split_9"}
  %slice.678 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.298), slice={[0:3], [0:3], [0:8], [88:96]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1578 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1554, f32[3,3,8,8]{3,2,1,0} %slice.678), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_11"}
  %slice.1555 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1542), slice={[0:1], [0:30], [0:30], [96:104]}, metadata={op_type="Split" op_name="split_9"}
  %slice.679 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.298), slice={[0:3], [0:3], [0:8], [96:104]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1579 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1555, f32[3,3,8,8]{3,2,1,0} %slice.679), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_12"}
  %slice.1556 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1542), slice={[0:1], [0:30], [0:30], [104:112]}, metadata={op_type="Split" op_name="split_9"}
  %slice.680 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.298), slice={[0:3], [0:3], [0:8], [104:112]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1580 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1556, f32[3,3,8,8]{3,2,1,0} %slice.680), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_13"}
  %slice.1557 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1542), slice={[0:1], [0:30], [0:30], [112:120]}, metadata={op_type="Split" op_name="split_9"}
  %slice.681 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.298), slice={[0:3], [0:3], [0:8], [112:120]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1581 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1557, f32[3,3,8,8]{3,2,1,0} %slice.681), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_14"}
  %slice.1558 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1542), slice={[0:1], [0:30], [0:30], [120:128]}, metadata={op_type="Split" op_name="split_9"}
  %slice.682 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.298), slice={[0:3], [0:3], [0:8], [120:128]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1582 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1558, f32[3,3,8,8]{3,2,1,0} %slice.682), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_15"}
  %slice.1559 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1542), slice={[0:1], [0:30], [0:30], [128:136]}, metadata={op_type="Split" op_name="split_9"}
  %slice.683 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.298), slice={[0:3], [0:3], [0:8], [128:136]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1583 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1559, f32[3,3,8,8]{3,2,1,0} %slice.683), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_16"}
  %slice.1560 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1542), slice={[0:1], [0:30], [0:30], [136:144]}, metadata={op_type="Split" op_name="split_9"}
  %slice.684 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.298), slice={[0:3], [0:3], [0:8], [136:144]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1584 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1560, f32[3,3,8,8]{3,2,1,0} %slice.684), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_17"}
  %slice.1561 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1542), slice={[0:1], [0:30], [0:30], [144:152]}, metadata={op_type="Split" op_name="split_9"}
  %slice.685 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.298), slice={[0:3], [0:3], [0:8], [144:152]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1585 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1561, f32[3,3,8,8]{3,2,1,0} %slice.685), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_18"}
  %slice.1562 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1542), slice={[0:1], [0:30], [0:30], [152:160]}, metadata={op_type="Split" op_name="split_9"}
  %slice.686 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.298), slice={[0:3], [0:3], [0:8], [152:160]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1586 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1562, f32[3,3,8,8]{3,2,1,0} %slice.686), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_19"}
  %slice.1563 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1542), slice={[0:1], [0:30], [0:30], [160:168]}, metadata={op_type="Split" op_name="split_9"}
  %slice.687 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.298), slice={[0:3], [0:3], [0:8], [160:168]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1588 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1563, f32[3,3,8,8]{3,2,1,0} %slice.687), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_20"}
  %slice.1564 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1542), slice={[0:1], [0:30], [0:30], [168:176]}, metadata={op_type="Split" op_name="split_9"}
  %slice.688 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.298), slice={[0:3], [0:3], [0:8], [168:176]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1589 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1564, f32[3,3,8,8]{3,2,1,0} %slice.688), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_21"}
  %slice.1565 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1542), slice={[0:1], [0:30], [0:30], [176:184]}, metadata={op_type="Split" op_name="split_9"}
  %slice.689 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.298), slice={[0:3], [0:3], [0:8], [176:184]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1590 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1565, f32[3,3,8,8]{3,2,1,0} %slice.689), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_22"}
  %slice.1566 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1542), slice={[0:1], [0:30], [0:30], [184:192]}, metadata={op_type="Split" op_name="split_9"}
  %slice.690 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.298), slice={[0:3], [0:3], [0:8], [184:192]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1591 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1566, f32[3,3,8,8]{3,2,1,0} %slice.690), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_23"}
  %slice.1567 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1542), slice={[0:1], [0:30], [0:30], [192:200]}, metadata={op_type="Split" op_name="split_9"}
  %slice.691 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.298), slice={[0:3], [0:3], [0:8], [192:200]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1592 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1567, f32[3,3,8,8]{3,2,1,0} %slice.691), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_24"}
  %slice.1568 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1542), slice={[0:1], [0:30], [0:30], [200:208]}, metadata={op_type="Split" op_name="split_9"}
  %slice.692 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.298), slice={[0:3], [0:3], [0:8], [200:208]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1593 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1568, f32[3,3,8,8]{3,2,1,0} %slice.692), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_25"}
  %slice.1569 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1542), slice={[0:1], [0:30], [0:30], [208:216]}, metadata={op_type="Split" op_name="split_9"}
  %slice.693 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.298), slice={[0:3], [0:3], [0:8], [208:216]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1594 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1569, f32[3,3,8,8]{3,2,1,0} %slice.693), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_26"}
  %slice.1570 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1542), slice={[0:1], [0:30], [0:30], [216:224]}, metadata={op_type="Split" op_name="split_9"}
  %slice.694 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.298), slice={[0:3], [0:3], [0:8], [216:224]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1595 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1570, f32[3,3,8,8]{3,2,1,0} %slice.694), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_27"}
  %slice.1571 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1542), slice={[0:1], [0:30], [0:30], [224:232]}, metadata={op_type="Split" op_name="split_9"}
  %slice.695 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.298), slice={[0:3], [0:3], [0:8], [224:232]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1596 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1571, f32[3,3,8,8]{3,2,1,0} %slice.695), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_28"}
  %slice.1572 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1542), slice={[0:1], [0:30], [0:30], [232:240]}, metadata={op_type="Split" op_name="split_9"}
  %slice.696 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.298), slice={[0:3], [0:3], [0:8], [232:240]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1597 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1572, f32[3,3,8,8]{3,2,1,0} %slice.696), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_29"}
  %slice.1573 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1542), slice={[0:1], [0:30], [0:30], [240:248]}, metadata={op_type="Split" op_name="split_9"}
  %slice.697 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.298), slice={[0:3], [0:3], [0:8], [240:248]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1599 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1573, f32[3,3,8,8]{3,2,1,0} %slice.697), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_30"}
  %slice.1574 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1542), slice={[0:1], [0:30], [0:30], [248:256]}, metadata={op_type="Split" op_name="split_9"}
  %slice.698 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.298), slice={[0:3], [0:3], [0:8], [248:256]}, metadata={op_type="Split" op_name="split_8"}
  %convolution.1600 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1574, f32[3,3,8,8]{3,2,1,0} %slice.698), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv2_31"}
  %concatenate.1607 = f32[1,28,28,256]{3,2,1,0} concatenate(f32[1,28,28,8]{3,2,1,0} %convolution.1575, f32[1,28,28,8]{3,2,1,0} %convolution.1576, f32[1,28,28,8]{3,2,1,0} %convolution.1587, f32[1,28,28,8]{3,2,1,0} %convolution.1598, f32[1,28,28,8]{3,2,1,0} %convolution.1601, f32[1,28,28,8]{3,2,1,0} %convolution.1602, f32[1,28,28,8]{3,2,1,0} %convolution.1603, f32[1,28,28,8]{3,2,1,0} %convolution.1604, f32[1,28,28,8]{3,2,1,0} %convolution.1605, f32[1,28,28,8]{3,2,1,0} %convolution.1606, f32[1,28,28,8]{3,2,1,0} %convolution.1577, f32[1,28,28,8]{3,2,1,0} %convolution.1578, f32[1,28,28,8]{3,2,1,0} %convolution.1579, f32[1,28,28,8]{3,2,1,0} %convolution.1580, f32[1,28,28,8]{3,2,1,0} %convolution.1581, f32[1,28,28,8]{3,2,1,0} %convolution.1582, f32[1,28,28,8]{3,2,1,0} %convolution.1583, f32[1,28,28,8]{3,2,1,0} %convolution.1584, f32[1,28,28,8]{3,2,1,0} %convolution.1585, f32[1,28,28,8]{3,2,1,0} %convolution.1586, f32[1,28,28,8]{3,2,1,0} %convolution.1588, f32[1,28,28,8]{3,2,1,0} %convolution.1589, f32[1,28,28,8]{3,2,1,0} %convolution.1590, f32[1,28,28,8]{3,2,1,0} %convolution.1591, f32[1,28,28,8]{3,2,1,0} %convolution.1592, f32[1,28,28,8]{3,2,1,0} %convolution.1593, f32[1,28,28,8]{3,2,1,0} %convolution.1594, f32[1,28,28,8]{3,2,1,0} %convolution.1595, f32[1,28,28,8]{3,2,1,0} %convolution.1596, f32[1,28,28,8]{3,2,1,0} %convolution.1597, f32[1,28,28,8]{3,2,1,0} %convolution.1599, f32[1,28,28,8]{3,2,1,0} %convolution.1600), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_4"}
  %arg29.30 = f32[256]{0} parameter(29), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.299 = f32[256]{0} reshape(f32[256]{0} %arg29.30)
  %constant.1608 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit2_bn2/add"}
  %broadcast.1609 = f32[256]{0} broadcast(f32[] %constant.1608), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit2_bn2/add"}
  %add.1610 = f32[256]{0} add(f32[256]{0} %reshape.299, f32[256]{0} %broadcast.1609), metadata={op_type="AddV2" op_name="stage2_unit2_bn2/add"}
  %rsqrt.1611 = f32[256]{0} rsqrt(f32[256]{0} %add.1610), metadata={op_type="Rsqrt" op_name="stage2_unit2_bn2/Rsqrt"}
  %arg92.93 = f32[256]{0} parameter(92), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.362 = f32[256]{0} reshape(f32[256]{0} %arg92.93)
  %multiply.1612 = f32[256]{0} multiply(f32[256]{0} %rsqrt.1611, f32[256]{0} %reshape.362), metadata={op_type="Mul" op_name="stage2_unit2_bn2/mul"}
  %broadcast.1613 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.1612), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit2_bn2/mul_1"}
  %multiply.1614 = f32[1,28,28,256]{3,2,1,0} multiply(f32[1,28,28,256]{3,2,1,0} %concatenate.1607, f32[1,28,28,256]{3,2,1,0} %broadcast.1613), metadata={op_type="Mul" op_name="stage2_unit2_bn2/mul_1"}
  %arg199.200 = f32[256]{0} parameter(199), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.469 = f32[256]{0} reshape(f32[256]{0} %arg199.200)
  %arg146.147 = f32[256]{0} parameter(146), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.416 = f32[256]{0} reshape(f32[256]{0} %arg146.147)
  %multiply.1615 = f32[256]{0} multiply(f32[256]{0} %multiply.1612, f32[256]{0} %reshape.416), metadata={op_type="Mul" op_name="stage2_unit2_bn2/mul_2"}
  %subtract.1616 = f32[256]{0} subtract(f32[256]{0} %reshape.469, f32[256]{0} %multiply.1615), metadata={op_type="Sub" op_name="stage2_unit2_bn2/sub"}
  %broadcast.1617 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.1616), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit2_bn2/add_1"}
  %add.1618 = f32[1,28,28,256]{3,2,1,0} add(f32[1,28,28,256]{3,2,1,0} %multiply.1614, f32[1,28,28,256]{3,2,1,0} %broadcast.1617), metadata={op_type="AddV2" op_name="stage2_unit2_bn2/add_1"}
  %maximum.1621 = f32[1,28,28,256]{3,2,1,0} maximum(f32[1,28,28,256]{3,2,1,0} %broadcast.1620, f32[1,28,28,256]{3,2,1,0} %add.1618), metadata={op_type="Relu" op_name="stage2_unit2_relu2"}
  %arg243.244 = f32[1,1,256,512]{3,2,1,0} parameter(243), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.513 = f32[1,1,256,512]{3,2,1,0} reshape(f32[1,1,256,512]{3,2,1,0} %arg243.244)
  %convolution.1622 = f32[1,28,28,512]{3,2,1,0} convolution(f32[1,28,28,256]{3,2,1,0} %maximum.1621, f32[1,1,256,512]{3,2,1,0} %reshape.513), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit2_conv3"}
  %multiply.1629 = f32[1,28,28,512]{3,2,1,0} multiply(f32[1,28,28,512]{3,2,1,0} %broadcast.1628, f32[1,28,28,512]{3,2,1,0} %convolution.1622), metadata={op_type="Mul" op_name="stage2_unit2_bn3/mul_1"}
  %arg200.201 = f32[512]{0} parameter(200), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.470 = f32[512]{0} reshape(f32[512]{0} %arg200.201)
  %arg147.148 = f32[512]{0} parameter(147), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.417 = f32[512]{0} reshape(f32[512]{0} %arg147.148)
  %multiply.1630 = f32[512]{0} multiply(f32[512]{0} %multiply.1627, f32[512]{0} %reshape.417), metadata={op_type="Mul" op_name="stage2_unit2_bn3/mul_2"}
  %subtract.1631 = f32[512]{0} subtract(f32[512]{0} %reshape.470, f32[512]{0} %multiply.1630), metadata={op_type="Sub" op_name="stage2_unit2_bn3/sub"}
  %broadcast.1632 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.1631), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit2_bn3/add_1"}
  %add.1633 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %multiply.1629, f32[1,28,28,512]{3,2,1,0} %broadcast.1632), metadata={op_type="AddV2" op_name="stage2_unit2_bn3/add_1"}
  %add.1634 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %maximum.1525, f32[1,28,28,512]{3,2,1,0} %add.1633), metadata={op_type="AddV2" op_name="add_4"}
  %maximum.1637 = f32[1,28,28,512]{3,2,1,0} maximum(f32[1,28,28,512]{3,2,1,0} %broadcast.1636, f32[1,28,28,512]{3,2,1,0} %add.1634), metadata={op_type="Relu" op_name="stage2_unit2_relu"}
  %arg35.36 = f32[512]{0} parameter(35), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.305 = f32[512]{0} reshape(f32[512]{0} %arg35.36)
  %constant.1735 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit3_bn3/add"}
  %broadcast.1736 = f32[512]{0} broadcast(f32[] %constant.1735), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit3_bn3/add"}
  %add.1737 = f32[512]{0} add(f32[512]{0} %reshape.305, f32[512]{0} %broadcast.1736), metadata={op_type="AddV2" op_name="stage2_unit3_bn3/add"}
  %rsqrt.1738 = f32[512]{0} rsqrt(f32[512]{0} %add.1737), metadata={op_type="Rsqrt" op_name="stage2_unit3_bn3/Rsqrt"}
  %arg96.97 = f32[512]{0} parameter(96), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.366 = f32[512]{0} reshape(f32[512]{0} %arg96.97)
  %multiply.1739 = f32[512]{0} multiply(f32[512]{0} %rsqrt.1738, f32[512]{0} %reshape.366), metadata={op_type="Mul" op_name="stage2_unit3_bn3/mul"}
  %broadcast.1740 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.1739), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit3_bn3/mul_1"}
  %constant.1731 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit3_relu2"}
  %broadcast.1732 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[] %constant.1731), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit3_relu2"}
  %constant.1650 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit3_relu1"}
  %broadcast.1651 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[] %constant.1650), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit3_relu1"}
  %arg32.33 = f32[256]{0} parameter(32), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.302 = f32[256]{0} reshape(f32[256]{0} %arg32.33)
  %constant.1639 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit3_bn1/add"}
  %broadcast.1640 = f32[256]{0} broadcast(f32[] %constant.1639), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit3_bn1/add"}
  %add.1641 = f32[256]{0} add(f32[256]{0} %reshape.302, f32[256]{0} %broadcast.1640), metadata={op_type="AddV2" op_name="stage2_unit3_bn1/add"}
  %rsqrt.1642 = f32[256]{0} rsqrt(f32[256]{0} %add.1641), metadata={op_type="Rsqrt" op_name="stage2_unit3_bn1/Rsqrt"}
  %arg94.95 = f32[256]{0} parameter(94), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.364 = f32[256]{0} reshape(f32[256]{0} %arg94.95)
  %multiply.1643 = f32[256]{0} multiply(f32[256]{0} %rsqrt.1642, f32[256]{0} %reshape.364), metadata={op_type="Mul" op_name="stage2_unit3_bn1/mul"}
  %broadcast.1644 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.1643), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit3_bn1/mul_1"}
  %arg244.245 = f32[1,1,512,256]{3,2,1,0} parameter(244), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.514 = f32[1,1,512,256]{3,2,1,0} reshape(f32[1,1,512,256]{3,2,1,0} %arg244.245)
  %convolution.1638 = f32[1,28,28,256]{3,2,1,0} convolution(f32[1,28,28,512]{3,2,1,0} %maximum.1637, f32[1,1,512,256]{3,2,1,0} %reshape.514), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv1"}
  %multiply.1645 = f32[1,28,28,256]{3,2,1,0} multiply(f32[1,28,28,256]{3,2,1,0} %broadcast.1644, f32[1,28,28,256]{3,2,1,0} %convolution.1638), metadata={op_type="Mul" op_name="stage2_unit3_bn1/mul_1"}
  %arg201.202 = f32[256]{0} parameter(201), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.471 = f32[256]{0} reshape(f32[256]{0} %arg201.202)
  %arg148.149 = f32[256]{0} parameter(148), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.418 = f32[256]{0} reshape(f32[256]{0} %arg148.149)
  %multiply.1646 = f32[256]{0} multiply(f32[256]{0} %multiply.1643, f32[256]{0} %reshape.418), metadata={op_type="Mul" op_name="stage2_unit3_bn1/mul_2"}
  %subtract.1647 = f32[256]{0} subtract(f32[256]{0} %reshape.471, f32[256]{0} %multiply.1646), metadata={op_type="Sub" op_name="stage2_unit3_bn1/sub"}
  %broadcast.1648 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.1647), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit3_bn1/add_1"}
  %add.1649 = f32[1,28,28,256]{3,2,1,0} add(f32[1,28,28,256]{3,2,1,0} %multiply.1645, f32[1,28,28,256]{3,2,1,0} %broadcast.1648), metadata={op_type="AddV2" op_name="stage2_unit3_bn1/add_1"}
  %maximum.1652 = f32[1,28,28,256]{3,2,1,0} maximum(f32[1,28,28,256]{3,2,1,0} %broadcast.1651, f32[1,28,28,256]{3,2,1,0} %add.1649), metadata={op_type="Relu" op_name="stage2_unit3_relu1"}
  %constant.1653 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_6"}
  %pad.1654 = f32[1,30,30,256]{3,2,1,0} pad(f32[1,28,28,256]{3,2,1,0} %maximum.1652, f32[] %constant.1653), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_6"}
  %slice.1655 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1654), slice={[0:1], [0:30], [0:30], [0:8]}, metadata={op_type="Split" op_name="split_11"}
  %arg33.34 = f32[3,3,8,256]{3,2,1,0} parameter(33), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.303 = f32[3,3,8,256]{3,2,1,0} reshape(f32[3,3,8,256]{3,2,1,0} %arg33.34)
  %slice.699 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:8], [0:8]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1687 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1655, f32[3,3,8,8]{3,2,1,0} %slice.699), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2"}
  %slice.1656 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1654), slice={[0:1], [0:30], [0:30], [8:16]}, metadata={op_type="Split" op_name="split_11"}
  %slice.700 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:8], [8:16]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1688 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1656, f32[3,3,8,8]{3,2,1,0} %slice.700), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_1"}
  %slice.1657 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1654), slice={[0:1], [0:30], [0:30], [16:24]}, metadata={op_type="Split" op_name="split_11"}
  %slice.701 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:8], [16:24]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1699 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1657, f32[3,3,8,8]{3,2,1,0} %slice.701), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_2"}
  %slice.1658 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1654), slice={[0:1], [0:30], [0:30], [24:32]}, metadata={op_type="Split" op_name="split_11"}
  %slice.702 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:8], [24:32]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1710 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1658, f32[3,3,8,8]{3,2,1,0} %slice.702), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_3"}
  %slice.1659 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1654), slice={[0:1], [0:30], [0:30], [32:40]}, metadata={op_type="Split" op_name="split_11"}
  %slice.703 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:8], [32:40]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1713 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1659, f32[3,3,8,8]{3,2,1,0} %slice.703), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_4"}
  %slice.1660 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1654), slice={[0:1], [0:30], [0:30], [40:48]}, metadata={op_type="Split" op_name="split_11"}
  %slice.704 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:8], [40:48]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1714 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1660, f32[3,3,8,8]{3,2,1,0} %slice.704), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_5"}
  %slice.1661 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1654), slice={[0:1], [0:30], [0:30], [48:56]}, metadata={op_type="Split" op_name="split_11"}
  %slice.705 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:8], [48:56]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1715 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1661, f32[3,3,8,8]{3,2,1,0} %slice.705), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_6"}
  %slice.1662 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1654), slice={[0:1], [0:30], [0:30], [56:64]}, metadata={op_type="Split" op_name="split_11"}
  %slice.706 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:8], [56:64]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1716 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1662, f32[3,3,8,8]{3,2,1,0} %slice.706), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_7"}
  %slice.1663 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1654), slice={[0:1], [0:30], [0:30], [64:72]}, metadata={op_type="Split" op_name="split_11"}
  %slice.707 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:8], [64:72]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1717 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1663, f32[3,3,8,8]{3,2,1,0} %slice.707), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_8"}
  %slice.1664 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1654), slice={[0:1], [0:30], [0:30], [72:80]}, metadata={op_type="Split" op_name="split_11"}
  %slice.708 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:8], [72:80]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1718 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1664, f32[3,3,8,8]{3,2,1,0} %slice.708), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_9"}
  %slice.1665 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1654), slice={[0:1], [0:30], [0:30], [80:88]}, metadata={op_type="Split" op_name="split_11"}
  %slice.709 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:8], [80:88]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1689 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1665, f32[3,3,8,8]{3,2,1,0} %slice.709), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_10"}
  %slice.1666 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1654), slice={[0:1], [0:30], [0:30], [88:96]}, metadata={op_type="Split" op_name="split_11"}
  %slice.710 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:8], [88:96]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1690 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1666, f32[3,3,8,8]{3,2,1,0} %slice.710), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_11"}
  %slice.1667 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1654), slice={[0:1], [0:30], [0:30], [96:104]}, metadata={op_type="Split" op_name="split_11"}
  %slice.711 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:8], [96:104]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1691 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1667, f32[3,3,8,8]{3,2,1,0} %slice.711), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_12"}
  %slice.1668 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1654), slice={[0:1], [0:30], [0:30], [104:112]}, metadata={op_type="Split" op_name="split_11"}
  %slice.712 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:8], [104:112]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1692 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1668, f32[3,3,8,8]{3,2,1,0} %slice.712), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_13"}
  %slice.1669 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1654), slice={[0:1], [0:30], [0:30], [112:120]}, metadata={op_type="Split" op_name="split_11"}
  %slice.713 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:8], [112:120]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1693 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1669, f32[3,3,8,8]{3,2,1,0} %slice.713), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_14"}
  %slice.1670 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1654), slice={[0:1], [0:30], [0:30], [120:128]}, metadata={op_type="Split" op_name="split_11"}
  %slice.714 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:8], [120:128]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1694 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1670, f32[3,3,8,8]{3,2,1,0} %slice.714), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_15"}
  %slice.1671 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1654), slice={[0:1], [0:30], [0:30], [128:136]}, metadata={op_type="Split" op_name="split_11"}
  %slice.715 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:8], [128:136]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1695 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1671, f32[3,3,8,8]{3,2,1,0} %slice.715), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_16"}
  %slice.1672 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1654), slice={[0:1], [0:30], [0:30], [136:144]}, metadata={op_type="Split" op_name="split_11"}
  %slice.716 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:8], [136:144]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1696 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1672, f32[3,3,8,8]{3,2,1,0} %slice.716), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_17"}
  %slice.1673 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1654), slice={[0:1], [0:30], [0:30], [144:152]}, metadata={op_type="Split" op_name="split_11"}
  %slice.717 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:8], [144:152]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1697 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1673, f32[3,3,8,8]{3,2,1,0} %slice.717), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_18"}
  %slice.1674 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1654), slice={[0:1], [0:30], [0:30], [152:160]}, metadata={op_type="Split" op_name="split_11"}
  %slice.718 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:8], [152:160]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1698 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1674, f32[3,3,8,8]{3,2,1,0} %slice.718), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_19"}
  %slice.1675 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1654), slice={[0:1], [0:30], [0:30], [160:168]}, metadata={op_type="Split" op_name="split_11"}
  %slice.719 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:8], [160:168]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1700 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1675, f32[3,3,8,8]{3,2,1,0} %slice.719), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_20"}
  %slice.1676 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1654), slice={[0:1], [0:30], [0:30], [168:176]}, metadata={op_type="Split" op_name="split_11"}
  %slice.720 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:8], [168:176]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1701 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1676, f32[3,3,8,8]{3,2,1,0} %slice.720), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_21"}
  %slice.1677 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1654), slice={[0:1], [0:30], [0:30], [176:184]}, metadata={op_type="Split" op_name="split_11"}
  %slice.721 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:8], [176:184]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1702 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1677, f32[3,3,8,8]{3,2,1,0} %slice.721), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_22"}
  %slice.1678 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1654), slice={[0:1], [0:30], [0:30], [184:192]}, metadata={op_type="Split" op_name="split_11"}
  %slice.722 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:8], [184:192]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1703 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1678, f32[3,3,8,8]{3,2,1,0} %slice.722), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_23"}
  %slice.1679 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1654), slice={[0:1], [0:30], [0:30], [192:200]}, metadata={op_type="Split" op_name="split_11"}
  %slice.723 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:8], [192:200]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1704 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1679, f32[3,3,8,8]{3,2,1,0} %slice.723), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_24"}
  %slice.1680 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1654), slice={[0:1], [0:30], [0:30], [200:208]}, metadata={op_type="Split" op_name="split_11"}
  %slice.724 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:8], [200:208]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1705 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1680, f32[3,3,8,8]{3,2,1,0} %slice.724), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_25"}
  %slice.1681 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1654), slice={[0:1], [0:30], [0:30], [208:216]}, metadata={op_type="Split" op_name="split_11"}
  %slice.725 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:8], [208:216]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1706 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1681, f32[3,3,8,8]{3,2,1,0} %slice.725), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_26"}
  %slice.1682 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1654), slice={[0:1], [0:30], [0:30], [216:224]}, metadata={op_type="Split" op_name="split_11"}
  %slice.726 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:8], [216:224]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1707 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1682, f32[3,3,8,8]{3,2,1,0} %slice.726), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_27"}
  %slice.1683 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1654), slice={[0:1], [0:30], [0:30], [224:232]}, metadata={op_type="Split" op_name="split_11"}
  %slice.727 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:8], [224:232]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1708 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1683, f32[3,3,8,8]{3,2,1,0} %slice.727), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_28"}
  %slice.1684 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1654), slice={[0:1], [0:30], [0:30], [232:240]}, metadata={op_type="Split" op_name="split_11"}
  %slice.728 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:8], [232:240]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1709 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1684, f32[3,3,8,8]{3,2,1,0} %slice.728), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_29"}
  %slice.1685 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1654), slice={[0:1], [0:30], [0:30], [240:248]}, metadata={op_type="Split" op_name="split_11"}
  %slice.729 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:8], [240:248]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1711 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1685, f32[3,3,8,8]{3,2,1,0} %slice.729), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_30"}
  %slice.1686 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1654), slice={[0:1], [0:30], [0:30], [248:256]}, metadata={op_type="Split" op_name="split_11"}
  %slice.730 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.303), slice={[0:3], [0:3], [0:8], [248:256]}, metadata={op_type="Split" op_name="split_10"}
  %convolution.1712 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1686, f32[3,3,8,8]{3,2,1,0} %slice.730), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv2_31"}
  %concatenate.1719 = f32[1,28,28,256]{3,2,1,0} concatenate(f32[1,28,28,8]{3,2,1,0} %convolution.1687, f32[1,28,28,8]{3,2,1,0} %convolution.1688, f32[1,28,28,8]{3,2,1,0} %convolution.1699, f32[1,28,28,8]{3,2,1,0} %convolution.1710, f32[1,28,28,8]{3,2,1,0} %convolution.1713, f32[1,28,28,8]{3,2,1,0} %convolution.1714, f32[1,28,28,8]{3,2,1,0} %convolution.1715, f32[1,28,28,8]{3,2,1,0} %convolution.1716, f32[1,28,28,8]{3,2,1,0} %convolution.1717, f32[1,28,28,8]{3,2,1,0} %convolution.1718, f32[1,28,28,8]{3,2,1,0} %convolution.1689, f32[1,28,28,8]{3,2,1,0} %convolution.1690, f32[1,28,28,8]{3,2,1,0} %convolution.1691, f32[1,28,28,8]{3,2,1,0} %convolution.1692, f32[1,28,28,8]{3,2,1,0} %convolution.1693, f32[1,28,28,8]{3,2,1,0} %convolution.1694, f32[1,28,28,8]{3,2,1,0} %convolution.1695, f32[1,28,28,8]{3,2,1,0} %convolution.1696, f32[1,28,28,8]{3,2,1,0} %convolution.1697, f32[1,28,28,8]{3,2,1,0} %convolution.1698, f32[1,28,28,8]{3,2,1,0} %convolution.1700, f32[1,28,28,8]{3,2,1,0} %convolution.1701, f32[1,28,28,8]{3,2,1,0} %convolution.1702, f32[1,28,28,8]{3,2,1,0} %convolution.1703, f32[1,28,28,8]{3,2,1,0} %convolution.1704, f32[1,28,28,8]{3,2,1,0} %convolution.1705, f32[1,28,28,8]{3,2,1,0} %convolution.1706, f32[1,28,28,8]{3,2,1,0} %convolution.1707, f32[1,28,28,8]{3,2,1,0} %convolution.1708, f32[1,28,28,8]{3,2,1,0} %convolution.1709, f32[1,28,28,8]{3,2,1,0} %convolution.1711, f32[1,28,28,8]{3,2,1,0} %convolution.1712), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_5"}
  %arg34.35 = f32[256]{0} parameter(34), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.304 = f32[256]{0} reshape(f32[256]{0} %arg34.35)
  %constant.1720 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit3_bn2/add"}
  %broadcast.1721 = f32[256]{0} broadcast(f32[] %constant.1720), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit3_bn2/add"}
  %add.1722 = f32[256]{0} add(f32[256]{0} %reshape.304, f32[256]{0} %broadcast.1721), metadata={op_type="AddV2" op_name="stage2_unit3_bn2/add"}
  %rsqrt.1723 = f32[256]{0} rsqrt(f32[256]{0} %add.1722), metadata={op_type="Rsqrt" op_name="stage2_unit3_bn2/Rsqrt"}
  %arg95.96 = f32[256]{0} parameter(95), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.365 = f32[256]{0} reshape(f32[256]{0} %arg95.96)
  %multiply.1724 = f32[256]{0} multiply(f32[256]{0} %rsqrt.1723, f32[256]{0} %reshape.365), metadata={op_type="Mul" op_name="stage2_unit3_bn2/mul"}
  %broadcast.1725 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.1724), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit3_bn2/mul_1"}
  %multiply.1726 = f32[1,28,28,256]{3,2,1,0} multiply(f32[1,28,28,256]{3,2,1,0} %concatenate.1719, f32[1,28,28,256]{3,2,1,0} %broadcast.1725), metadata={op_type="Mul" op_name="stage2_unit3_bn2/mul_1"}
  %arg202.203 = f32[256]{0} parameter(202), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.472 = f32[256]{0} reshape(f32[256]{0} %arg202.203)
  %arg149.150 = f32[256]{0} parameter(149), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.419 = f32[256]{0} reshape(f32[256]{0} %arg149.150)
  %multiply.1727 = f32[256]{0} multiply(f32[256]{0} %multiply.1724, f32[256]{0} %reshape.419), metadata={op_type="Mul" op_name="stage2_unit3_bn2/mul_2"}
  %subtract.1728 = f32[256]{0} subtract(f32[256]{0} %reshape.472, f32[256]{0} %multiply.1727), metadata={op_type="Sub" op_name="stage2_unit3_bn2/sub"}
  %broadcast.1729 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.1728), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit3_bn2/add_1"}
  %add.1730 = f32[1,28,28,256]{3,2,1,0} add(f32[1,28,28,256]{3,2,1,0} %multiply.1726, f32[1,28,28,256]{3,2,1,0} %broadcast.1729), metadata={op_type="AddV2" op_name="stage2_unit3_bn2/add_1"}
  %maximum.1733 = f32[1,28,28,256]{3,2,1,0} maximum(f32[1,28,28,256]{3,2,1,0} %broadcast.1732, f32[1,28,28,256]{3,2,1,0} %add.1730), metadata={op_type="Relu" op_name="stage2_unit3_relu2"}
  %arg245.246 = f32[1,1,256,512]{3,2,1,0} parameter(245), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.515 = f32[1,1,256,512]{3,2,1,0} reshape(f32[1,1,256,512]{3,2,1,0} %arg245.246)
  %convolution.1734 = f32[1,28,28,512]{3,2,1,0} convolution(f32[1,28,28,256]{3,2,1,0} %maximum.1733, f32[1,1,256,512]{3,2,1,0} %reshape.515), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit3_conv3"}
  %multiply.1741 = f32[1,28,28,512]{3,2,1,0} multiply(f32[1,28,28,512]{3,2,1,0} %broadcast.1740, f32[1,28,28,512]{3,2,1,0} %convolution.1734), metadata={op_type="Mul" op_name="stage2_unit3_bn3/mul_1"}
  %arg203.204 = f32[512]{0} parameter(203), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.473 = f32[512]{0} reshape(f32[512]{0} %arg203.204)
  %arg150.151 = f32[512]{0} parameter(150), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.420 = f32[512]{0} reshape(f32[512]{0} %arg150.151)
  %multiply.1742 = f32[512]{0} multiply(f32[512]{0} %multiply.1739, f32[512]{0} %reshape.420), metadata={op_type="Mul" op_name="stage2_unit3_bn3/mul_2"}
  %subtract.1743 = f32[512]{0} subtract(f32[512]{0} %reshape.473, f32[512]{0} %multiply.1742), metadata={op_type="Sub" op_name="stage2_unit3_bn3/sub"}
  %broadcast.1744 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.1743), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit3_bn3/add_1"}
  %add.1745 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %multiply.1741, f32[1,28,28,512]{3,2,1,0} %broadcast.1744), metadata={op_type="AddV2" op_name="stage2_unit3_bn3/add_1"}
  %add.1746 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %maximum.1637, f32[1,28,28,512]{3,2,1,0} %add.1745), metadata={op_type="AddV2" op_name="add_5"}
  %maximum.1749 = f32[1,28,28,512]{3,2,1,0} maximum(f32[1,28,28,512]{3,2,1,0} %broadcast.1748, f32[1,28,28,512]{3,2,1,0} %add.1746), metadata={op_type="Relu" op_name="stage2_unit3_relu"}
  %arg40.41 = f32[512]{0} parameter(40), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.310 = f32[512]{0} reshape(f32[512]{0} %arg40.41)
  %constant.1847 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit4_bn3/add"}
  %broadcast.1848 = f32[512]{0} broadcast(f32[] %constant.1847), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit4_bn3/add"}
  %add.1849 = f32[512]{0} add(f32[512]{0} %reshape.310, f32[512]{0} %broadcast.1848), metadata={op_type="AddV2" op_name="stage2_unit4_bn3/add"}
  %rsqrt.1850 = f32[512]{0} rsqrt(f32[512]{0} %add.1849), metadata={op_type="Rsqrt" op_name="stage2_unit4_bn3/Rsqrt"}
  %arg100.101 = f32[512]{0} parameter(100), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.370 = f32[512]{0} reshape(f32[512]{0} %arg100.101)
  %multiply.1851 = f32[512]{0} multiply(f32[512]{0} %rsqrt.1850, f32[512]{0} %reshape.370), metadata={op_type="Mul" op_name="stage2_unit4_bn3/mul"}
  %broadcast.1852 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.1851), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit4_bn3/mul_1"}
  %constant.1843 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit4_relu2"}
  %broadcast.1844 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[] %constant.1843), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit4_relu2"}
  %constant.1762 = f32[] constant(0), metadata={op_type="Relu" op_name="stage2_unit4_relu1"}
  %broadcast.1763 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[] %constant.1762), dimensions={}, metadata={op_type="Relu" op_name="stage2_unit4_relu1"}
  %arg36.37 = f32[256]{0} parameter(36), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.306 = f32[256]{0} reshape(f32[256]{0} %arg36.37)
  %constant.1751 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit4_bn1/add"}
  %broadcast.1752 = f32[256]{0} broadcast(f32[] %constant.1751), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit4_bn1/add"}
  %add.1753 = f32[256]{0} add(f32[256]{0} %reshape.306, f32[256]{0} %broadcast.1752), metadata={op_type="AddV2" op_name="stage2_unit4_bn1/add"}
  %rsqrt.1754 = f32[256]{0} rsqrt(f32[256]{0} %add.1753), metadata={op_type="Rsqrt" op_name="stage2_unit4_bn1/Rsqrt"}
  %arg97.98 = f32[256]{0} parameter(97), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.367 = f32[256]{0} reshape(f32[256]{0} %arg97.98)
  %multiply.1755 = f32[256]{0} multiply(f32[256]{0} %rsqrt.1754, f32[256]{0} %reshape.367), metadata={op_type="Mul" op_name="stage2_unit4_bn1/mul"}
  %broadcast.1756 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.1755), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit4_bn1/mul_1"}
  %arg246.247 = f32[1,1,512,256]{3,2,1,0} parameter(246), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.516 = f32[1,1,512,256]{3,2,1,0} reshape(f32[1,1,512,256]{3,2,1,0} %arg246.247)
  %convolution.1750 = f32[1,28,28,256]{3,2,1,0} convolution(f32[1,28,28,512]{3,2,1,0} %maximum.1749, f32[1,1,512,256]{3,2,1,0} %reshape.516), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv1"}
  %multiply.1757 = f32[1,28,28,256]{3,2,1,0} multiply(f32[1,28,28,256]{3,2,1,0} %broadcast.1756, f32[1,28,28,256]{3,2,1,0} %convolution.1750), metadata={op_type="Mul" op_name="stage2_unit4_bn1/mul_1"}
  %arg204.205 = f32[256]{0} parameter(204), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.474 = f32[256]{0} reshape(f32[256]{0} %arg204.205)
  %arg151.152 = f32[256]{0} parameter(151), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.421 = f32[256]{0} reshape(f32[256]{0} %arg151.152)
  %multiply.1758 = f32[256]{0} multiply(f32[256]{0} %multiply.1755, f32[256]{0} %reshape.421), metadata={op_type="Mul" op_name="stage2_unit4_bn1/mul_2"}
  %subtract.1759 = f32[256]{0} subtract(f32[256]{0} %reshape.474, f32[256]{0} %multiply.1758), metadata={op_type="Sub" op_name="stage2_unit4_bn1/sub"}
  %broadcast.1760 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.1759), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit4_bn1/add_1"}
  %add.1761 = f32[1,28,28,256]{3,2,1,0} add(f32[1,28,28,256]{3,2,1,0} %multiply.1757, f32[1,28,28,256]{3,2,1,0} %broadcast.1760), metadata={op_type="AddV2" op_name="stage2_unit4_bn1/add_1"}
  %maximum.1764 = f32[1,28,28,256]{3,2,1,0} maximum(f32[1,28,28,256]{3,2,1,0} %broadcast.1763, f32[1,28,28,256]{3,2,1,0} %add.1761), metadata={op_type="Relu" op_name="stage2_unit4_relu1"}
  %constant.1765 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_7"}
  %pad.1766 = f32[1,30,30,256]{3,2,1,0} pad(f32[1,28,28,256]{3,2,1,0} %maximum.1764, f32[] %constant.1765), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_7"}
  %slice.1767 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1766), slice={[0:1], [0:30], [0:30], [0:8]}, metadata={op_type="Split" op_name="split_13"}
  %arg38.39 = f32[3,3,8,256]{3,2,1,0} parameter(38), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.308 = f32[3,3,8,256]{3,2,1,0} reshape(f32[3,3,8,256]{3,2,1,0} %arg38.39)
  %slice.731 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.308), slice={[0:3], [0:3], [0:8], [0:8]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1799 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1767, f32[3,3,8,8]{3,2,1,0} %slice.731), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2"}
  %slice.1768 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1766), slice={[0:1], [0:30], [0:30], [8:16]}, metadata={op_type="Split" op_name="split_13"}
  %slice.732 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.308), slice={[0:3], [0:3], [0:8], [8:16]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1800 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1768, f32[3,3,8,8]{3,2,1,0} %slice.732), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_1"}
  %slice.1769 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1766), slice={[0:1], [0:30], [0:30], [16:24]}, metadata={op_type="Split" op_name="split_13"}
  %slice.733 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.308), slice={[0:3], [0:3], [0:8], [16:24]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1811 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1769, f32[3,3,8,8]{3,2,1,0} %slice.733), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_2"}
  %slice.1770 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1766), slice={[0:1], [0:30], [0:30], [24:32]}, metadata={op_type="Split" op_name="split_13"}
  %slice.734 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.308), slice={[0:3], [0:3], [0:8], [24:32]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1822 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1770, f32[3,3,8,8]{3,2,1,0} %slice.734), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_3"}
  %slice.1771 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1766), slice={[0:1], [0:30], [0:30], [32:40]}, metadata={op_type="Split" op_name="split_13"}
  %slice.735 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.308), slice={[0:3], [0:3], [0:8], [32:40]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1825 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1771, f32[3,3,8,8]{3,2,1,0} %slice.735), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_4"}
  %slice.1772 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1766), slice={[0:1], [0:30], [0:30], [40:48]}, metadata={op_type="Split" op_name="split_13"}
  %slice.736 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.308), slice={[0:3], [0:3], [0:8], [40:48]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1826 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1772, f32[3,3,8,8]{3,2,1,0} %slice.736), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_5"}
  %slice.1773 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1766), slice={[0:1], [0:30], [0:30], [48:56]}, metadata={op_type="Split" op_name="split_13"}
  %slice.737 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.308), slice={[0:3], [0:3], [0:8], [48:56]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1827 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1773, f32[3,3,8,8]{3,2,1,0} %slice.737), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_6"}
  %slice.1774 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1766), slice={[0:1], [0:30], [0:30], [56:64]}, metadata={op_type="Split" op_name="split_13"}
  %slice.738 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.308), slice={[0:3], [0:3], [0:8], [56:64]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1828 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1774, f32[3,3,8,8]{3,2,1,0} %slice.738), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_7"}
  %slice.1775 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1766), slice={[0:1], [0:30], [0:30], [64:72]}, metadata={op_type="Split" op_name="split_13"}
  %slice.739 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.308), slice={[0:3], [0:3], [0:8], [64:72]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1829 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1775, f32[3,3,8,8]{3,2,1,0} %slice.739), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_8"}
  %slice.1776 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1766), slice={[0:1], [0:30], [0:30], [72:80]}, metadata={op_type="Split" op_name="split_13"}
  %slice.740 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.308), slice={[0:3], [0:3], [0:8], [72:80]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1830 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1776, f32[3,3,8,8]{3,2,1,0} %slice.740), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_9"}
  %slice.1777 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1766), slice={[0:1], [0:30], [0:30], [80:88]}, metadata={op_type="Split" op_name="split_13"}
  %slice.741 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.308), slice={[0:3], [0:3], [0:8], [80:88]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1801 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1777, f32[3,3,8,8]{3,2,1,0} %slice.741), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_10"}
  %slice.1778 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1766), slice={[0:1], [0:30], [0:30], [88:96]}, metadata={op_type="Split" op_name="split_13"}
  %slice.742 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.308), slice={[0:3], [0:3], [0:8], [88:96]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1802 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1778, f32[3,3,8,8]{3,2,1,0} %slice.742), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_11"}
  %slice.1779 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1766), slice={[0:1], [0:30], [0:30], [96:104]}, metadata={op_type="Split" op_name="split_13"}
  %slice.743 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.308), slice={[0:3], [0:3], [0:8], [96:104]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1803 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1779, f32[3,3,8,8]{3,2,1,0} %slice.743), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_12"}
  %slice.1780 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1766), slice={[0:1], [0:30], [0:30], [104:112]}, metadata={op_type="Split" op_name="split_13"}
  %slice.744 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.308), slice={[0:3], [0:3], [0:8], [104:112]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1804 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1780, f32[3,3,8,8]{3,2,1,0} %slice.744), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_13"}
  %slice.1781 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1766), slice={[0:1], [0:30], [0:30], [112:120]}, metadata={op_type="Split" op_name="split_13"}
  %slice.745 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.308), slice={[0:3], [0:3], [0:8], [112:120]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1805 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1781, f32[3,3,8,8]{3,2,1,0} %slice.745), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_14"}
  %slice.1782 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1766), slice={[0:1], [0:30], [0:30], [120:128]}, metadata={op_type="Split" op_name="split_13"}
  %slice.746 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.308), slice={[0:3], [0:3], [0:8], [120:128]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1806 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1782, f32[3,3,8,8]{3,2,1,0} %slice.746), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_15"}
  %slice.1783 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1766), slice={[0:1], [0:30], [0:30], [128:136]}, metadata={op_type="Split" op_name="split_13"}
  %slice.747 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.308), slice={[0:3], [0:3], [0:8], [128:136]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1807 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1783, f32[3,3,8,8]{3,2,1,0} %slice.747), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_16"}
  %slice.1784 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1766), slice={[0:1], [0:30], [0:30], [136:144]}, metadata={op_type="Split" op_name="split_13"}
  %slice.748 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.308), slice={[0:3], [0:3], [0:8], [136:144]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1808 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1784, f32[3,3,8,8]{3,2,1,0} %slice.748), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_17"}
  %slice.1785 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1766), slice={[0:1], [0:30], [0:30], [144:152]}, metadata={op_type="Split" op_name="split_13"}
  %slice.749 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.308), slice={[0:3], [0:3], [0:8], [144:152]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1809 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1785, f32[3,3,8,8]{3,2,1,0} %slice.749), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_18"}
  %slice.1786 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1766), slice={[0:1], [0:30], [0:30], [152:160]}, metadata={op_type="Split" op_name="split_13"}
  %slice.750 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.308), slice={[0:3], [0:3], [0:8], [152:160]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1810 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1786, f32[3,3,8,8]{3,2,1,0} %slice.750), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_19"}
  %slice.1787 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1766), slice={[0:1], [0:30], [0:30], [160:168]}, metadata={op_type="Split" op_name="split_13"}
  %slice.751 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.308), slice={[0:3], [0:3], [0:8], [160:168]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1812 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1787, f32[3,3,8,8]{3,2,1,0} %slice.751), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_20"}
  %slice.1788 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1766), slice={[0:1], [0:30], [0:30], [168:176]}, metadata={op_type="Split" op_name="split_13"}
  %slice.752 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.308), slice={[0:3], [0:3], [0:8], [168:176]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1813 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1788, f32[3,3,8,8]{3,2,1,0} %slice.752), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_21"}
  %slice.1789 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1766), slice={[0:1], [0:30], [0:30], [176:184]}, metadata={op_type="Split" op_name="split_13"}
  %slice.753 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.308), slice={[0:3], [0:3], [0:8], [176:184]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1814 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1789, f32[3,3,8,8]{3,2,1,0} %slice.753), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_22"}
  %slice.1790 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1766), slice={[0:1], [0:30], [0:30], [184:192]}, metadata={op_type="Split" op_name="split_13"}
  %slice.754 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.308), slice={[0:3], [0:3], [0:8], [184:192]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1815 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1790, f32[3,3,8,8]{3,2,1,0} %slice.754), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_23"}
  %slice.1791 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1766), slice={[0:1], [0:30], [0:30], [192:200]}, metadata={op_type="Split" op_name="split_13"}
  %slice.755 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.308), slice={[0:3], [0:3], [0:8], [192:200]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1816 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1791, f32[3,3,8,8]{3,2,1,0} %slice.755), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_24"}
  %slice.1792 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1766), slice={[0:1], [0:30], [0:30], [200:208]}, metadata={op_type="Split" op_name="split_13"}
  %slice.756 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.308), slice={[0:3], [0:3], [0:8], [200:208]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1817 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1792, f32[3,3,8,8]{3,2,1,0} %slice.756), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_25"}
  %slice.1793 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1766), slice={[0:1], [0:30], [0:30], [208:216]}, metadata={op_type="Split" op_name="split_13"}
  %slice.757 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.308), slice={[0:3], [0:3], [0:8], [208:216]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1818 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1793, f32[3,3,8,8]{3,2,1,0} %slice.757), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_26"}
  %slice.1794 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1766), slice={[0:1], [0:30], [0:30], [216:224]}, metadata={op_type="Split" op_name="split_13"}
  %slice.758 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.308), slice={[0:3], [0:3], [0:8], [216:224]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1819 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1794, f32[3,3,8,8]{3,2,1,0} %slice.758), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_27"}
  %slice.1795 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1766), slice={[0:1], [0:30], [0:30], [224:232]}, metadata={op_type="Split" op_name="split_13"}
  %slice.759 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.308), slice={[0:3], [0:3], [0:8], [224:232]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1820 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1795, f32[3,3,8,8]{3,2,1,0} %slice.759), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_28"}
  %slice.1796 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1766), slice={[0:1], [0:30], [0:30], [232:240]}, metadata={op_type="Split" op_name="split_13"}
  %slice.760 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.308), slice={[0:3], [0:3], [0:8], [232:240]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1821 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1796, f32[3,3,8,8]{3,2,1,0} %slice.760), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_29"}
  %slice.1797 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1766), slice={[0:1], [0:30], [0:30], [240:248]}, metadata={op_type="Split" op_name="split_13"}
  %slice.761 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.308), slice={[0:3], [0:3], [0:8], [240:248]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1823 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1797, f32[3,3,8,8]{3,2,1,0} %slice.761), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_30"}
  %slice.1798 = f32[1,30,30,8]{3,2,1,0} slice(f32[1,30,30,256]{3,2,1,0} %pad.1766), slice={[0:1], [0:30], [0:30], [248:256]}, metadata={op_type="Split" op_name="split_13"}
  %slice.762 = f32[3,3,8,8]{3,2,1,0} slice(f32[3,3,8,256]{3,2,1,0} %reshape.308), slice={[0:3], [0:3], [0:8], [248:256]}, metadata={op_type="Split" op_name="split_12"}
  %convolution.1824 = f32[1,28,28,8]{3,2,1,0} convolution(f32[1,30,30,8]{3,2,1,0} %slice.1798, f32[3,3,8,8]{3,2,1,0} %slice.762), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv2_31"}
  %concatenate.1831 = f32[1,28,28,256]{3,2,1,0} concatenate(f32[1,28,28,8]{3,2,1,0} %convolution.1799, f32[1,28,28,8]{3,2,1,0} %convolution.1800, f32[1,28,28,8]{3,2,1,0} %convolution.1811, f32[1,28,28,8]{3,2,1,0} %convolution.1822, f32[1,28,28,8]{3,2,1,0} %convolution.1825, f32[1,28,28,8]{3,2,1,0} %convolution.1826, f32[1,28,28,8]{3,2,1,0} %convolution.1827, f32[1,28,28,8]{3,2,1,0} %convolution.1828, f32[1,28,28,8]{3,2,1,0} %convolution.1829, f32[1,28,28,8]{3,2,1,0} %convolution.1830, f32[1,28,28,8]{3,2,1,0} %convolution.1801, f32[1,28,28,8]{3,2,1,0} %convolution.1802, f32[1,28,28,8]{3,2,1,0} %convolution.1803, f32[1,28,28,8]{3,2,1,0} %convolution.1804, f32[1,28,28,8]{3,2,1,0} %convolution.1805, f32[1,28,28,8]{3,2,1,0} %convolution.1806, f32[1,28,28,8]{3,2,1,0} %convolution.1807, f32[1,28,28,8]{3,2,1,0} %convolution.1808, f32[1,28,28,8]{3,2,1,0} %convolution.1809, f32[1,28,28,8]{3,2,1,0} %convolution.1810, f32[1,28,28,8]{3,2,1,0} %convolution.1812, f32[1,28,28,8]{3,2,1,0} %convolution.1813, f32[1,28,28,8]{3,2,1,0} %convolution.1814, f32[1,28,28,8]{3,2,1,0} %convolution.1815, f32[1,28,28,8]{3,2,1,0} %convolution.1816, f32[1,28,28,8]{3,2,1,0} %convolution.1817, f32[1,28,28,8]{3,2,1,0} %convolution.1818, f32[1,28,28,8]{3,2,1,0} %convolution.1819, f32[1,28,28,8]{3,2,1,0} %convolution.1820, f32[1,28,28,8]{3,2,1,0} %convolution.1821, f32[1,28,28,8]{3,2,1,0} %convolution.1823, f32[1,28,28,8]{3,2,1,0} %convolution.1824), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_6"}
  %arg39.40 = f32[256]{0} parameter(39), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.309 = f32[256]{0} reshape(f32[256]{0} %arg39.40)
  %constant.1832 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage2_unit4_bn2/add"}
  %broadcast.1833 = f32[256]{0} broadcast(f32[] %constant.1832), dimensions={}, metadata={op_type="AddV2" op_name="stage2_unit4_bn2/add"}
  %add.1834 = f32[256]{0} add(f32[256]{0} %reshape.309, f32[256]{0} %broadcast.1833), metadata={op_type="AddV2" op_name="stage2_unit4_bn2/add"}
  %rsqrt.1835 = f32[256]{0} rsqrt(f32[256]{0} %add.1834), metadata={op_type="Rsqrt" op_name="stage2_unit4_bn2/Rsqrt"}
  %arg99.100 = f32[256]{0} parameter(99), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.369 = f32[256]{0} reshape(f32[256]{0} %arg99.100)
  %multiply.1836 = f32[256]{0} multiply(f32[256]{0} %rsqrt.1835, f32[256]{0} %reshape.369), metadata={op_type="Mul" op_name="stage2_unit4_bn2/mul"}
  %broadcast.1837 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.1836), dimensions={3}, metadata={op_type="Mul" op_name="stage2_unit4_bn2/mul_1"}
  %multiply.1838 = f32[1,28,28,256]{3,2,1,0} multiply(f32[1,28,28,256]{3,2,1,0} %concatenate.1831, f32[1,28,28,256]{3,2,1,0} %broadcast.1837), metadata={op_type="Mul" op_name="stage2_unit4_bn2/mul_1"}
  %arg206.207 = f32[256]{0} parameter(206), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.476 = f32[256]{0} reshape(f32[256]{0} %arg206.207)
  %arg153.154 = f32[256]{0} parameter(153), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.423 = f32[256]{0} reshape(f32[256]{0} %arg153.154)
  %multiply.1839 = f32[256]{0} multiply(f32[256]{0} %multiply.1836, f32[256]{0} %reshape.423), metadata={op_type="Mul" op_name="stage2_unit4_bn2/mul_2"}
  %subtract.1840 = f32[256]{0} subtract(f32[256]{0} %reshape.476, f32[256]{0} %multiply.1839), metadata={op_type="Sub" op_name="stage2_unit4_bn2/sub"}
  %broadcast.1841 = f32[1,28,28,256]{3,2,1,0} broadcast(f32[256]{0} %subtract.1840), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit4_bn2/add_1"}
  %add.1842 = f32[1,28,28,256]{3,2,1,0} add(f32[1,28,28,256]{3,2,1,0} %multiply.1838, f32[1,28,28,256]{3,2,1,0} %broadcast.1841), metadata={op_type="AddV2" op_name="stage2_unit4_bn2/add_1"}
  %maximum.1845 = f32[1,28,28,256]{3,2,1,0} maximum(f32[1,28,28,256]{3,2,1,0} %broadcast.1844, f32[1,28,28,256]{3,2,1,0} %add.1842), metadata={op_type="Relu" op_name="stage2_unit4_relu2"}
  %arg247.248 = f32[1,1,256,512]{3,2,1,0} parameter(247), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.517 = f32[1,1,256,512]{3,2,1,0} reshape(f32[1,1,256,512]{3,2,1,0} %arg247.248)
  %convolution.1846 = f32[1,28,28,512]{3,2,1,0} convolution(f32[1,28,28,256]{3,2,1,0} %maximum.1845, f32[1,1,256,512]{3,2,1,0} %reshape.517), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage2_unit4_conv3"}
  %multiply.1853 = f32[1,28,28,512]{3,2,1,0} multiply(f32[1,28,28,512]{3,2,1,0} %broadcast.1852, f32[1,28,28,512]{3,2,1,0} %convolution.1846), metadata={op_type="Mul" op_name="stage2_unit4_bn3/mul_1"}
  %arg207.208 = f32[512]{0} parameter(207), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.477 = f32[512]{0} reshape(f32[512]{0} %arg207.208)
  %arg154.155 = f32[512]{0} parameter(154), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.424 = f32[512]{0} reshape(f32[512]{0} %arg154.155)
  %multiply.1854 = f32[512]{0} multiply(f32[512]{0} %multiply.1851, f32[512]{0} %reshape.424), metadata={op_type="Mul" op_name="stage2_unit4_bn3/mul_2"}
  %subtract.1855 = f32[512]{0} subtract(f32[512]{0} %reshape.477, f32[512]{0} %multiply.1854), metadata={op_type="Sub" op_name="stage2_unit4_bn3/sub"}
  %broadcast.1856 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.1855), dimensions={3}, metadata={op_type="AddV2" op_name="stage2_unit4_bn3/add_1"}
  %add.1857 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %multiply.1853, f32[1,28,28,512]{3,2,1,0} %broadcast.1856), metadata={op_type="AddV2" op_name="stage2_unit4_bn3/add_1"}
  %add.1858 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %maximum.1749, f32[1,28,28,512]{3,2,1,0} %add.1857), metadata={op_type="AddV2" op_name="add_6"}
  %maximum.1861 = f32[1,28,28,512]{3,2,1,0} maximum(f32[1,28,28,512]{3,2,1,0} %broadcast.1860, f32[1,28,28,512]{3,2,1,0} %add.1858), metadata={op_type="Relu" op_name="stage2_unit4_relu"}
  %arg248.249 = f32[1,1,512,512]{3,2,1,0} parameter(248), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.518 = f32[1,1,512,512]{3,2,1,0} reshape(f32[1,1,512,512]{3,2,1,0} %arg248.249)
  %convolution.1862 = f32[1,28,28,512]{3,2,1,0} convolution(f32[1,28,28,512]{3,2,1,0} %maximum.1861, f32[1,1,512,512]{3,2,1,0} %reshape.518), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv1"}
  %multiply.1870 = f32[1,28,28,512]{3,2,1,0} multiply(f32[1,28,28,512]{3,2,1,0} %broadcast.1869, f32[1,28,28,512]{3,2,1,0} %convolution.1862), metadata={op_type="Mul" op_name="stage3_unit1_bn1/mul_1"}
  %arg208.209 = f32[512]{0} parameter(208), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.478 = f32[512]{0} reshape(f32[512]{0} %arg208.209)
  %arg155.156 = f32[512]{0} parameter(155), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.425 = f32[512]{0} reshape(f32[512]{0} %arg155.156)
  %multiply.1871 = f32[512]{0} multiply(f32[512]{0} %multiply.1868, f32[512]{0} %reshape.425), metadata={op_type="Mul" op_name="stage3_unit1_bn1/mul_2"}
  %subtract.1872 = f32[512]{0} subtract(f32[512]{0} %reshape.478, f32[512]{0} %multiply.1871), metadata={op_type="Sub" op_name="stage3_unit1_bn1/sub"}
  %broadcast.1873 = f32[1,28,28,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.1872), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit1_bn1/add_1"}
  %add.1874 = f32[1,28,28,512]{3,2,1,0} add(f32[1,28,28,512]{3,2,1,0} %multiply.1870, f32[1,28,28,512]{3,2,1,0} %broadcast.1873), metadata={op_type="AddV2" op_name="stage3_unit1_bn1/add_1"}
  %maximum.1877 = f32[1,28,28,512]{3,2,1,0} maximum(f32[1,28,28,512]{3,2,1,0} %broadcast.1876, f32[1,28,28,512]{3,2,1,0} %add.1874), metadata={op_type="Relu" op_name="stage3_unit1_relu1"}
  %constant.1878 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_8"}
  %pad.1879 = f32[1,30,30,512]{3,2,1,0} pad(f32[1,28,28,512]{3,2,1,0} %maximum.1877, f32[] %constant.1878), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_8"}
  %slice.1880 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1879), slice={[0:1], [0:30], [0:30], [0:16]}, metadata={op_type="Split" op_name="split_15"}
  %arg45.46 = f32[3,3,16,512]{3,2,1,0} parameter(45), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.315 = f32[3,3,16,512]{3,2,1,0} reshape(f32[3,3,16,512]{3,2,1,0} %arg45.46)
  %slice.763 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.315), slice={[0:3], [0:3], [0:16], [0:16]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1912 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1880, f32[3,3,16,16]{3,2,1,0} %slice.763), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2"}
  %slice.1881 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1879), slice={[0:1], [0:30], [0:30], [16:32]}, metadata={op_type="Split" op_name="split_15"}
  %slice.764 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.315), slice={[0:3], [0:3], [0:16], [16:32]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1913 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1881, f32[3,3,16,16]{3,2,1,0} %slice.764), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_1"}
  %slice.1882 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1879), slice={[0:1], [0:30], [0:30], [32:48]}, metadata={op_type="Split" op_name="split_15"}
  %slice.765 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.315), slice={[0:3], [0:3], [0:16], [32:48]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1924 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1882, f32[3,3,16,16]{3,2,1,0} %slice.765), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_2"}
  %slice.1883 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1879), slice={[0:1], [0:30], [0:30], [48:64]}, metadata={op_type="Split" op_name="split_15"}
  %slice.766 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.315), slice={[0:3], [0:3], [0:16], [48:64]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1935 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1883, f32[3,3,16,16]{3,2,1,0} %slice.766), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_3"}
  %slice.1884 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1879), slice={[0:1], [0:30], [0:30], [64:80]}, metadata={op_type="Split" op_name="split_15"}
  %slice.767 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.315), slice={[0:3], [0:3], [0:16], [64:80]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1938 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1884, f32[3,3,16,16]{3,2,1,0} %slice.767), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_4"}
  %slice.1885 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1879), slice={[0:1], [0:30], [0:30], [80:96]}, metadata={op_type="Split" op_name="split_15"}
  %slice.768 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.315), slice={[0:3], [0:3], [0:16], [80:96]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1939 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1885, f32[3,3,16,16]{3,2,1,0} %slice.768), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_5"}
  %slice.1886 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1879), slice={[0:1], [0:30], [0:30], [96:112]}, metadata={op_type="Split" op_name="split_15"}
  %slice.769 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.315), slice={[0:3], [0:3], [0:16], [96:112]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1940 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1886, f32[3,3,16,16]{3,2,1,0} %slice.769), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_6"}
  %slice.1887 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1879), slice={[0:1], [0:30], [0:30], [112:128]}, metadata={op_type="Split" op_name="split_15"}
  %slice.770 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.315), slice={[0:3], [0:3], [0:16], [112:128]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1941 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1887, f32[3,3,16,16]{3,2,1,0} %slice.770), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_7"}
  %slice.1888 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1879), slice={[0:1], [0:30], [0:30], [128:144]}, metadata={op_type="Split" op_name="split_15"}
  %slice.771 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.315), slice={[0:3], [0:3], [0:16], [128:144]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1942 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1888, f32[3,3,16,16]{3,2,1,0} %slice.771), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_8"}
  %slice.1889 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1879), slice={[0:1], [0:30], [0:30], [144:160]}, metadata={op_type="Split" op_name="split_15"}
  %slice.772 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.315), slice={[0:3], [0:3], [0:16], [144:160]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1943 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1889, f32[3,3,16,16]{3,2,1,0} %slice.772), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_9"}
  %slice.1890 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1879), slice={[0:1], [0:30], [0:30], [160:176]}, metadata={op_type="Split" op_name="split_15"}
  %slice.773 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.315), slice={[0:3], [0:3], [0:16], [160:176]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1914 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1890, f32[3,3,16,16]{3,2,1,0} %slice.773), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_10"}
  %slice.1891 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1879), slice={[0:1], [0:30], [0:30], [176:192]}, metadata={op_type="Split" op_name="split_15"}
  %slice.774 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.315), slice={[0:3], [0:3], [0:16], [176:192]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1915 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1891, f32[3,3,16,16]{3,2,1,0} %slice.774), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_11"}
  %slice.1892 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1879), slice={[0:1], [0:30], [0:30], [192:208]}, metadata={op_type="Split" op_name="split_15"}
  %slice.775 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.315), slice={[0:3], [0:3], [0:16], [192:208]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1916 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1892, f32[3,3,16,16]{3,2,1,0} %slice.775), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_12"}
  %slice.1893 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1879), slice={[0:1], [0:30], [0:30], [208:224]}, metadata={op_type="Split" op_name="split_15"}
  %slice.776 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.315), slice={[0:3], [0:3], [0:16], [208:224]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1917 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1893, f32[3,3,16,16]{3,2,1,0} %slice.776), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_13"}
  %slice.1894 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1879), slice={[0:1], [0:30], [0:30], [224:240]}, metadata={op_type="Split" op_name="split_15"}
  %slice.777 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.315), slice={[0:3], [0:3], [0:16], [224:240]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1918 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1894, f32[3,3,16,16]{3,2,1,0} %slice.777), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_14"}
  %slice.1895 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1879), slice={[0:1], [0:30], [0:30], [240:256]}, metadata={op_type="Split" op_name="split_15"}
  %slice.778 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.315), slice={[0:3], [0:3], [0:16], [240:256]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1919 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1895, f32[3,3,16,16]{3,2,1,0} %slice.778), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_15"}
  %slice.1896 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1879), slice={[0:1], [0:30], [0:30], [256:272]}, metadata={op_type="Split" op_name="split_15"}
  %slice.779 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.315), slice={[0:3], [0:3], [0:16], [256:272]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1920 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1896, f32[3,3,16,16]{3,2,1,0} %slice.779), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_16"}
  %slice.1897 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1879), slice={[0:1], [0:30], [0:30], [272:288]}, metadata={op_type="Split" op_name="split_15"}
  %slice.780 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.315), slice={[0:3], [0:3], [0:16], [272:288]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1921 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1897, f32[3,3,16,16]{3,2,1,0} %slice.780), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_17"}
  %slice.1898 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1879), slice={[0:1], [0:30], [0:30], [288:304]}, metadata={op_type="Split" op_name="split_15"}
  %slice.781 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.315), slice={[0:3], [0:3], [0:16], [288:304]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1922 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1898, f32[3,3,16,16]{3,2,1,0} %slice.781), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_18"}
  %slice.1899 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1879), slice={[0:1], [0:30], [0:30], [304:320]}, metadata={op_type="Split" op_name="split_15"}
  %slice.782 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.315), slice={[0:3], [0:3], [0:16], [304:320]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1923 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1899, f32[3,3,16,16]{3,2,1,0} %slice.782), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_19"}
  %slice.1900 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1879), slice={[0:1], [0:30], [0:30], [320:336]}, metadata={op_type="Split" op_name="split_15"}
  %slice.783 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.315), slice={[0:3], [0:3], [0:16], [320:336]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1925 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1900, f32[3,3,16,16]{3,2,1,0} %slice.783), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_20"}
  %slice.1901 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1879), slice={[0:1], [0:30], [0:30], [336:352]}, metadata={op_type="Split" op_name="split_15"}
  %slice.784 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.315), slice={[0:3], [0:3], [0:16], [336:352]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1926 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1901, f32[3,3,16,16]{3,2,1,0} %slice.784), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_21"}
  %slice.1902 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1879), slice={[0:1], [0:30], [0:30], [352:368]}, metadata={op_type="Split" op_name="split_15"}
  %slice.785 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.315), slice={[0:3], [0:3], [0:16], [352:368]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1927 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1902, f32[3,3,16,16]{3,2,1,0} %slice.785), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_22"}
  %slice.1903 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1879), slice={[0:1], [0:30], [0:30], [368:384]}, metadata={op_type="Split" op_name="split_15"}
  %slice.786 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.315), slice={[0:3], [0:3], [0:16], [368:384]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1928 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1903, f32[3,3,16,16]{3,2,1,0} %slice.786), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_23"}
  %slice.1904 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1879), slice={[0:1], [0:30], [0:30], [384:400]}, metadata={op_type="Split" op_name="split_15"}
  %slice.787 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.315), slice={[0:3], [0:3], [0:16], [384:400]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1929 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1904, f32[3,3,16,16]{3,2,1,0} %slice.787), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_24"}
  %slice.1905 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1879), slice={[0:1], [0:30], [0:30], [400:416]}, metadata={op_type="Split" op_name="split_15"}
  %slice.788 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.315), slice={[0:3], [0:3], [0:16], [400:416]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1930 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1905, f32[3,3,16,16]{3,2,1,0} %slice.788), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_25"}
  %slice.1906 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1879), slice={[0:1], [0:30], [0:30], [416:432]}, metadata={op_type="Split" op_name="split_15"}
  %slice.789 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.315), slice={[0:3], [0:3], [0:16], [416:432]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1931 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1906, f32[3,3,16,16]{3,2,1,0} %slice.789), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_26"}
  %slice.1907 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1879), slice={[0:1], [0:30], [0:30], [432:448]}, metadata={op_type="Split" op_name="split_15"}
  %slice.790 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.315), slice={[0:3], [0:3], [0:16], [432:448]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1932 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1907, f32[3,3,16,16]{3,2,1,0} %slice.790), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_27"}
  %slice.1908 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1879), slice={[0:1], [0:30], [0:30], [448:464]}, metadata={op_type="Split" op_name="split_15"}
  %slice.791 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.315), slice={[0:3], [0:3], [0:16], [448:464]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1933 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1908, f32[3,3,16,16]{3,2,1,0} %slice.791), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_28"}
  %slice.1909 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1879), slice={[0:1], [0:30], [0:30], [464:480]}, metadata={op_type="Split" op_name="split_15"}
  %slice.792 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.315), slice={[0:3], [0:3], [0:16], [464:480]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1934 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1909, f32[3,3,16,16]{3,2,1,0} %slice.792), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_29"}
  %slice.1910 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1879), slice={[0:1], [0:30], [0:30], [480:496]}, metadata={op_type="Split" op_name="split_15"}
  %slice.793 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.315), slice={[0:3], [0:3], [0:16], [480:496]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1936 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1910, f32[3,3,16,16]{3,2,1,0} %slice.793), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_30"}
  %slice.1911 = f32[1,30,30,16]{3,2,1,0} slice(f32[1,30,30,512]{3,2,1,0} %pad.1879), slice={[0:1], [0:30], [0:30], [496:512]}, metadata={op_type="Split" op_name="split_15"}
  %slice.794 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.315), slice={[0:3], [0:3], [0:16], [496:512]}, metadata={op_type="Split" op_name="split_14"}
  %convolution.1937 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,30,30,16]{3,2,1,0} %slice.1911, f32[3,3,16,16]{3,2,1,0} %slice.794), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv2_31"}
  %concatenate.1944 = f32[1,14,14,512]{3,2,1,0} concatenate(f32[1,14,14,16]{3,2,1,0} %convolution.1912, f32[1,14,14,16]{3,2,1,0} %convolution.1913, f32[1,14,14,16]{3,2,1,0} %convolution.1924, f32[1,14,14,16]{3,2,1,0} %convolution.1935, f32[1,14,14,16]{3,2,1,0} %convolution.1938, f32[1,14,14,16]{3,2,1,0} %convolution.1939, f32[1,14,14,16]{3,2,1,0} %convolution.1940, f32[1,14,14,16]{3,2,1,0} %convolution.1941, f32[1,14,14,16]{3,2,1,0} %convolution.1942, f32[1,14,14,16]{3,2,1,0} %convolution.1943, f32[1,14,14,16]{3,2,1,0} %convolution.1914, f32[1,14,14,16]{3,2,1,0} %convolution.1915, f32[1,14,14,16]{3,2,1,0} %convolution.1916, f32[1,14,14,16]{3,2,1,0} %convolution.1917, f32[1,14,14,16]{3,2,1,0} %convolution.1918, f32[1,14,14,16]{3,2,1,0} %convolution.1919, f32[1,14,14,16]{3,2,1,0} %convolution.1920, f32[1,14,14,16]{3,2,1,0} %convolution.1921, f32[1,14,14,16]{3,2,1,0} %convolution.1922, f32[1,14,14,16]{3,2,1,0} %convolution.1923, f32[1,14,14,16]{3,2,1,0} %convolution.1925, f32[1,14,14,16]{3,2,1,0} %convolution.1926, f32[1,14,14,16]{3,2,1,0} %convolution.1927, f32[1,14,14,16]{3,2,1,0} %convolution.1928, f32[1,14,14,16]{3,2,1,0} %convolution.1929, f32[1,14,14,16]{3,2,1,0} %convolution.1930, f32[1,14,14,16]{3,2,1,0} %convolution.1931, f32[1,14,14,16]{3,2,1,0} %convolution.1932, f32[1,14,14,16]{3,2,1,0} %convolution.1933, f32[1,14,14,16]{3,2,1,0} %convolution.1934, f32[1,14,14,16]{3,2,1,0} %convolution.1936, f32[1,14,14,16]{3,2,1,0} %convolution.1937), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_7"}
  %arg46.47 = f32[512]{0} parameter(46), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.316 = f32[512]{0} reshape(f32[512]{0} %arg46.47)
  %constant.1945 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit1_bn2/add"}
  %broadcast.1946 = f32[512]{0} broadcast(f32[] %constant.1945), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit1_bn2/add"}
  %add.1947 = f32[512]{0} add(f32[512]{0} %reshape.316, f32[512]{0} %broadcast.1946), metadata={op_type="AddV2" op_name="stage3_unit1_bn2/add"}
  %rsqrt.1948 = f32[512]{0} rsqrt(f32[512]{0} %add.1947), metadata={op_type="Rsqrt" op_name="stage3_unit1_bn2/Rsqrt"}
  %arg105.106 = f32[512]{0} parameter(105), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.375 = f32[512]{0} reshape(f32[512]{0} %arg105.106)
  %multiply.1949 = f32[512]{0} multiply(f32[512]{0} %rsqrt.1948, f32[512]{0} %reshape.375), metadata={op_type="Mul" op_name="stage3_unit1_bn2/mul"}
  %broadcast.1950 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.1949), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit1_bn2/mul_1"}
  %multiply.1951 = f32[1,14,14,512]{3,2,1,0} multiply(f32[1,14,14,512]{3,2,1,0} %concatenate.1944, f32[1,14,14,512]{3,2,1,0} %broadcast.1950), metadata={op_type="Mul" op_name="stage3_unit1_bn2/mul_1"}
  %arg212.213 = f32[512]{0} parameter(212), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.482 = f32[512]{0} reshape(f32[512]{0} %arg212.213)
  %arg159.160 = f32[512]{0} parameter(159), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.429 = f32[512]{0} reshape(f32[512]{0} %arg159.160)
  %multiply.1952 = f32[512]{0} multiply(f32[512]{0} %multiply.1949, f32[512]{0} %reshape.429), metadata={op_type="Mul" op_name="stage3_unit1_bn2/mul_2"}
  %subtract.1953 = f32[512]{0} subtract(f32[512]{0} %reshape.482, f32[512]{0} %multiply.1952), metadata={op_type="Sub" op_name="stage3_unit1_bn2/sub"}
  %broadcast.1954 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.1953), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit1_bn2/add_1"}
  %add.1955 = f32[1,14,14,512]{3,2,1,0} add(f32[1,14,14,512]{3,2,1,0} %multiply.1951, f32[1,14,14,512]{3,2,1,0} %broadcast.1954), metadata={op_type="AddV2" op_name="stage3_unit1_bn2/add_1"}
  %maximum.1958 = f32[1,14,14,512]{3,2,1,0} maximum(f32[1,14,14,512]{3,2,1,0} %broadcast.1957, f32[1,14,14,512]{3,2,1,0} %add.1955), metadata={op_type="Relu" op_name="stage3_unit1_relu2"}
  %arg250.251 = f32[1,1,512,1024]{3,2,1,0} parameter(250), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.520 = f32[1,1,512,1024]{3,2,1,0} reshape(f32[1,1,512,1024]{3,2,1,0} %arg250.251)
  %convolution.1959 = f32[1,14,14,1024]{3,2,1,0} convolution(f32[1,14,14,512]{3,2,1,0} %maximum.1958, f32[1,1,512,1024]{3,2,1,0} %reshape.520), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_conv3"}
  %multiply.1966 = f32[1,14,14,1024]{3,2,1,0} multiply(f32[1,14,14,1024]{3,2,1,0} %broadcast.1965, f32[1,14,14,1024]{3,2,1,0} %convolution.1959), metadata={op_type="Mul" op_name="stage3_unit1_bn3/mul_1"}
  %arg213.214 = f32[1024]{0} parameter(213), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.483 = f32[1024]{0} reshape(f32[1024]{0} %arg213.214)
  %arg160.161 = f32[1024]{0} parameter(160), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.430 = f32[1024]{0} reshape(f32[1024]{0} %arg160.161)
  %multiply.1967 = f32[1024]{0} multiply(f32[1024]{0} %multiply.1964, f32[1024]{0} %reshape.430), metadata={op_type="Mul" op_name="stage3_unit1_bn3/mul_2"}
  %subtract.1968 = f32[1024]{0} subtract(f32[1024]{0} %reshape.483, f32[1024]{0} %multiply.1967), metadata={op_type="Sub" op_name="stage3_unit1_bn3/sub"}
  %broadcast.1969 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %subtract.1968), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit1_bn3/add_1"}
  %add.1970 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %multiply.1966, f32[1,14,14,1024]{3,2,1,0} %broadcast.1969), metadata={op_type="AddV2" op_name="stage3_unit1_bn3/add_1"}
  %arg249.250 = f32[1,1,512,1024]{3,2,1,0} parameter(249), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.519 = f32[1,1,512,1024]{3,2,1,0} reshape(f32[1,1,512,1024]{3,2,1,0} %arg249.250)
  %convolution.1863 = f32[1,14,14,1024]{3,2,1,0} convolution(f32[1,28,28,512]{3,2,1,0} %maximum.1861, f32[1,1,512,1024]{3,2,1,0} %reshape.519), window={size=1x1 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit1_sc"}
  %arg44.45 = f32[1024]{0} parameter(44), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.314 = f32[1024]{0} reshape(f32[1024]{0} %arg44.45)
  %constant.1971 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit1_sc_bn/add"}
  %broadcast.1972 = f32[1024]{0} broadcast(f32[] %constant.1971), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit1_sc_bn/add"}
  %add.1973 = f32[1024]{0} add(f32[1024]{0} %reshape.314, f32[1024]{0} %broadcast.1972), metadata={op_type="AddV2" op_name="stage3_unit1_sc_bn/add"}
  %rsqrt.1974 = f32[1024]{0} rsqrt(f32[1024]{0} %add.1973), metadata={op_type="Rsqrt" op_name="stage3_unit1_sc_bn/Rsqrt"}
  %arg104.105 = f32[1024]{0} parameter(104), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.374 = f32[1024]{0} reshape(f32[1024]{0} %arg104.105)
  %multiply.1975 = f32[1024]{0} multiply(f32[1024]{0} %rsqrt.1974, f32[1024]{0} %reshape.374), metadata={op_type="Mul" op_name="stage3_unit1_sc_bn/mul"}
  %broadcast.1976 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %multiply.1975), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit1_sc_bn/mul_1"}
  %multiply.1977 = f32[1,14,14,1024]{3,2,1,0} multiply(f32[1,14,14,1024]{3,2,1,0} %convolution.1863, f32[1,14,14,1024]{3,2,1,0} %broadcast.1976), metadata={op_type="Mul" op_name="stage3_unit1_sc_bn/mul_1"}
  %arg211.212 = f32[1024]{0} parameter(211), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.481 = f32[1024]{0} reshape(f32[1024]{0} %arg211.212)
  %arg158.159 = f32[1024]{0} parameter(158), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.428 = f32[1024]{0} reshape(f32[1024]{0} %arg158.159)
  %multiply.1978 = f32[1024]{0} multiply(f32[1024]{0} %multiply.1975, f32[1024]{0} %reshape.428), metadata={op_type="Mul" op_name="stage3_unit1_sc_bn/mul_2"}
  %subtract.1979 = f32[1024]{0} subtract(f32[1024]{0} %reshape.481, f32[1024]{0} %multiply.1978), metadata={op_type="Sub" op_name="stage3_unit1_sc_bn/sub"}
  %broadcast.1980 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %subtract.1979), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit1_sc_bn/add_1"}
  %add.1981 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %multiply.1977, f32[1,14,14,1024]{3,2,1,0} %broadcast.1980), metadata={op_type="AddV2" op_name="stage3_unit1_sc_bn/add_1"}
  %add.1982 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %add.1970, f32[1,14,14,1024]{3,2,1,0} %add.1981), metadata={op_type="AddV2" op_name="add_7"}
  %maximum.1985 = f32[1,14,14,1024]{3,2,1,0} maximum(f32[1,14,14,1024]{3,2,1,0} %broadcast.1984, f32[1,14,14,1024]{3,2,1,0} %add.1982), metadata={op_type="Relu" op_name="stage3_unit1_relu"}
  %arg53.54 = f32[1024]{0} parameter(53), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.323 = f32[1024]{0} reshape(f32[1024]{0} %arg53.54)
  %constant.2083 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit2_bn3/add"}
  %broadcast.2084 = f32[1024]{0} broadcast(f32[] %constant.2083), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit2_bn3/add"}
  %add.2085 = f32[1024]{0} add(f32[1024]{0} %reshape.323, f32[1024]{0} %broadcast.2084), metadata={op_type="AddV2" op_name="stage3_unit2_bn3/add"}
  %rsqrt.2086 = f32[1024]{0} rsqrt(f32[1024]{0} %add.2085), metadata={op_type="Rsqrt" op_name="stage3_unit2_bn3/Rsqrt"}
  %arg110.111 = f32[1024]{0} parameter(110), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.380 = f32[1024]{0} reshape(f32[1024]{0} %arg110.111)
  %multiply.2087 = f32[1024]{0} multiply(f32[1024]{0} %rsqrt.2086, f32[1024]{0} %reshape.380), metadata={op_type="Mul" op_name="stage3_unit2_bn3/mul"}
  %broadcast.2088 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %multiply.2087), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit2_bn3/mul_1"}
  %constant.2079 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit2_relu2"}
  %broadcast.2080 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[] %constant.2079), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit2_relu2"}
  %constant.1998 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit2_relu1"}
  %broadcast.1999 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[] %constant.1998), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit2_relu1"}
  %arg49.50 = f32[512]{0} parameter(49), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.319 = f32[512]{0} reshape(f32[512]{0} %arg49.50)
  %constant.1987 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit2_bn1/add"}
  %broadcast.1988 = f32[512]{0} broadcast(f32[] %constant.1987), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit2_bn1/add"}
  %add.1989 = f32[512]{0} add(f32[512]{0} %reshape.319, f32[512]{0} %broadcast.1988), metadata={op_type="AddV2" op_name="stage3_unit2_bn1/add"}
  %rsqrt.1990 = f32[512]{0} rsqrt(f32[512]{0} %add.1989), metadata={op_type="Rsqrt" op_name="stage3_unit2_bn1/Rsqrt"}
  %arg108.109 = f32[512]{0} parameter(108), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.378 = f32[512]{0} reshape(f32[512]{0} %arg108.109)
  %multiply.1991 = f32[512]{0} multiply(f32[512]{0} %rsqrt.1990, f32[512]{0} %reshape.378), metadata={op_type="Mul" op_name="stage3_unit2_bn1/mul"}
  %broadcast.1992 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.1991), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit2_bn1/mul_1"}
  %arg251.252 = f32[1,1,1024,512]{3,2,1,0} parameter(251), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.521 = f32[1,1,1024,512]{3,2,1,0} reshape(f32[1,1,1024,512]{3,2,1,0} %arg251.252)
  %convolution.1986 = f32[1,14,14,512]{3,2,1,0} convolution(f32[1,14,14,1024]{3,2,1,0} %maximum.1985, f32[1,1,1024,512]{3,2,1,0} %reshape.521), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv1"}
  %multiply.1993 = f32[1,14,14,512]{3,2,1,0} multiply(f32[1,14,14,512]{3,2,1,0} %broadcast.1992, f32[1,14,14,512]{3,2,1,0} %convolution.1986), metadata={op_type="Mul" op_name="stage3_unit2_bn1/mul_1"}
  %arg215.216 = f32[512]{0} parameter(215), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.485 = f32[512]{0} reshape(f32[512]{0} %arg215.216)
  %arg162.163 = f32[512]{0} parameter(162), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.432 = f32[512]{0} reshape(f32[512]{0} %arg162.163)
  %multiply.1994 = f32[512]{0} multiply(f32[512]{0} %multiply.1991, f32[512]{0} %reshape.432), metadata={op_type="Mul" op_name="stage3_unit2_bn1/mul_2"}
  %subtract.1995 = f32[512]{0} subtract(f32[512]{0} %reshape.485, f32[512]{0} %multiply.1994), metadata={op_type="Sub" op_name="stage3_unit2_bn1/sub"}
  %broadcast.1996 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.1995), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit2_bn1/add_1"}
  %add.1997 = f32[1,14,14,512]{3,2,1,0} add(f32[1,14,14,512]{3,2,1,0} %multiply.1993, f32[1,14,14,512]{3,2,1,0} %broadcast.1996), metadata={op_type="AddV2" op_name="stage3_unit2_bn1/add_1"}
  %maximum.2000 = f32[1,14,14,512]{3,2,1,0} maximum(f32[1,14,14,512]{3,2,1,0} %broadcast.1999, f32[1,14,14,512]{3,2,1,0} %add.1997), metadata={op_type="Relu" op_name="stage3_unit2_relu1"}
  %constant.2001 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_9"}
  %pad.2002 = f32[1,16,16,512]{3,2,1,0} pad(f32[1,14,14,512]{3,2,1,0} %maximum.2000, f32[] %constant.2001), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_9"}
  %slice.2003 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2002), slice={[0:1], [0:16], [0:16], [0:16]}, metadata={op_type="Split" op_name="split_17"}
  %arg50.51 = f32[3,3,16,512]{3,2,1,0} parameter(50), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.320 = f32[3,3,16,512]{3,2,1,0} reshape(f32[3,3,16,512]{3,2,1,0} %arg50.51)
  %slice.795 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.320), slice={[0:3], [0:3], [0:16], [0:16]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.2035 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2003, f32[3,3,16,16]{3,2,1,0} %slice.795), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2"}
  %slice.2004 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2002), slice={[0:1], [0:16], [0:16], [16:32]}, metadata={op_type="Split" op_name="split_17"}
  %slice.796 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.320), slice={[0:3], [0:3], [0:16], [16:32]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.2036 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2004, f32[3,3,16,16]{3,2,1,0} %slice.796), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_1"}
  %slice.2005 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2002), slice={[0:1], [0:16], [0:16], [32:48]}, metadata={op_type="Split" op_name="split_17"}
  %slice.797 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.320), slice={[0:3], [0:3], [0:16], [32:48]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.2047 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2005, f32[3,3,16,16]{3,2,1,0} %slice.797), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_2"}
  %slice.2006 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2002), slice={[0:1], [0:16], [0:16], [48:64]}, metadata={op_type="Split" op_name="split_17"}
  %slice.798 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.320), slice={[0:3], [0:3], [0:16], [48:64]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.2058 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2006, f32[3,3,16,16]{3,2,1,0} %slice.798), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_3"}
  %slice.2007 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2002), slice={[0:1], [0:16], [0:16], [64:80]}, metadata={op_type="Split" op_name="split_17"}
  %slice.799 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.320), slice={[0:3], [0:3], [0:16], [64:80]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.2061 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2007, f32[3,3,16,16]{3,2,1,0} %slice.799), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_4"}
  %slice.2008 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2002), slice={[0:1], [0:16], [0:16], [80:96]}, metadata={op_type="Split" op_name="split_17"}
  %slice.800 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.320), slice={[0:3], [0:3], [0:16], [80:96]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.2062 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2008, f32[3,3,16,16]{3,2,1,0} %slice.800), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_5"}
  %slice.2009 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2002), slice={[0:1], [0:16], [0:16], [96:112]}, metadata={op_type="Split" op_name="split_17"}
  %slice.801 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.320), slice={[0:3], [0:3], [0:16], [96:112]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.2063 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2009, f32[3,3,16,16]{3,2,1,0} %slice.801), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_6"}
  %slice.2010 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2002), slice={[0:1], [0:16], [0:16], [112:128]}, metadata={op_type="Split" op_name="split_17"}
  %slice.802 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.320), slice={[0:3], [0:3], [0:16], [112:128]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.2064 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2010, f32[3,3,16,16]{3,2,1,0} %slice.802), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_7"}
  %slice.2011 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2002), slice={[0:1], [0:16], [0:16], [128:144]}, metadata={op_type="Split" op_name="split_17"}
  %slice.803 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.320), slice={[0:3], [0:3], [0:16], [128:144]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.2065 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2011, f32[3,3,16,16]{3,2,1,0} %slice.803), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_8"}
  %slice.2012 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2002), slice={[0:1], [0:16], [0:16], [144:160]}, metadata={op_type="Split" op_name="split_17"}
  %slice.804 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.320), slice={[0:3], [0:3], [0:16], [144:160]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.2066 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2012, f32[3,3,16,16]{3,2,1,0} %slice.804), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_9"}
  %slice.2013 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2002), slice={[0:1], [0:16], [0:16], [160:176]}, metadata={op_type="Split" op_name="split_17"}
  %slice.805 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.320), slice={[0:3], [0:3], [0:16], [160:176]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.2037 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2013, f32[3,3,16,16]{3,2,1,0} %slice.805), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_10"}
  %slice.2014 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2002), slice={[0:1], [0:16], [0:16], [176:192]}, metadata={op_type="Split" op_name="split_17"}
  %slice.806 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.320), slice={[0:3], [0:3], [0:16], [176:192]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.2038 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2014, f32[3,3,16,16]{3,2,1,0} %slice.806), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_11"}
  %slice.2015 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2002), slice={[0:1], [0:16], [0:16], [192:208]}, metadata={op_type="Split" op_name="split_17"}
  %slice.807 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.320), slice={[0:3], [0:3], [0:16], [192:208]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.2039 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2015, f32[3,3,16,16]{3,2,1,0} %slice.807), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_12"}
  %slice.2016 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2002), slice={[0:1], [0:16], [0:16], [208:224]}, metadata={op_type="Split" op_name="split_17"}
  %slice.808 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.320), slice={[0:3], [0:3], [0:16], [208:224]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.2040 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2016, f32[3,3,16,16]{3,2,1,0} %slice.808), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_13"}
  %slice.2017 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2002), slice={[0:1], [0:16], [0:16], [224:240]}, metadata={op_type="Split" op_name="split_17"}
  %slice.809 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.320), slice={[0:3], [0:3], [0:16], [224:240]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.2041 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2017, f32[3,3,16,16]{3,2,1,0} %slice.809), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_14"}
  %slice.2018 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2002), slice={[0:1], [0:16], [0:16], [240:256]}, metadata={op_type="Split" op_name="split_17"}
  %slice.810 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.320), slice={[0:3], [0:3], [0:16], [240:256]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.2042 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2018, f32[3,3,16,16]{3,2,1,0} %slice.810), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_15"}
  %slice.2019 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2002), slice={[0:1], [0:16], [0:16], [256:272]}, metadata={op_type="Split" op_name="split_17"}
  %slice.811 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.320), slice={[0:3], [0:3], [0:16], [256:272]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.2043 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2019, f32[3,3,16,16]{3,2,1,0} %slice.811), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_16"}
  %slice.2020 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2002), slice={[0:1], [0:16], [0:16], [272:288]}, metadata={op_type="Split" op_name="split_17"}
  %slice.812 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.320), slice={[0:3], [0:3], [0:16], [272:288]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.2044 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2020, f32[3,3,16,16]{3,2,1,0} %slice.812), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_17"}
  %slice.2021 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2002), slice={[0:1], [0:16], [0:16], [288:304]}, metadata={op_type="Split" op_name="split_17"}
  %slice.813 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.320), slice={[0:3], [0:3], [0:16], [288:304]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.2045 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2021, f32[3,3,16,16]{3,2,1,0} %slice.813), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_18"}
  %slice.2022 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2002), slice={[0:1], [0:16], [0:16], [304:320]}, metadata={op_type="Split" op_name="split_17"}
  %slice.814 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.320), slice={[0:3], [0:3], [0:16], [304:320]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.2046 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2022, f32[3,3,16,16]{3,2,1,0} %slice.814), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_19"}
  %slice.2023 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2002), slice={[0:1], [0:16], [0:16], [320:336]}, metadata={op_type="Split" op_name="split_17"}
  %slice.815 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.320), slice={[0:3], [0:3], [0:16], [320:336]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.2048 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2023, f32[3,3,16,16]{3,2,1,0} %slice.815), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_20"}
  %slice.2024 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2002), slice={[0:1], [0:16], [0:16], [336:352]}, metadata={op_type="Split" op_name="split_17"}
  %slice.816 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.320), slice={[0:3], [0:3], [0:16], [336:352]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.2049 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2024, f32[3,3,16,16]{3,2,1,0} %slice.816), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_21"}
  %slice.2025 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2002), slice={[0:1], [0:16], [0:16], [352:368]}, metadata={op_type="Split" op_name="split_17"}
  %slice.817 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.320), slice={[0:3], [0:3], [0:16], [352:368]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.2050 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2025, f32[3,3,16,16]{3,2,1,0} %slice.817), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_22"}
  %slice.2026 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2002), slice={[0:1], [0:16], [0:16], [368:384]}, metadata={op_type="Split" op_name="split_17"}
  %slice.818 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.320), slice={[0:3], [0:3], [0:16], [368:384]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.2051 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2026, f32[3,3,16,16]{3,2,1,0} %slice.818), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_23"}
  %slice.2027 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2002), slice={[0:1], [0:16], [0:16], [384:400]}, metadata={op_type="Split" op_name="split_17"}
  %slice.819 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.320), slice={[0:3], [0:3], [0:16], [384:400]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.2052 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2027, f32[3,3,16,16]{3,2,1,0} %slice.819), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_24"}
  %slice.2028 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2002), slice={[0:1], [0:16], [0:16], [400:416]}, metadata={op_type="Split" op_name="split_17"}
  %slice.820 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.320), slice={[0:3], [0:3], [0:16], [400:416]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.2053 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2028, f32[3,3,16,16]{3,2,1,0} %slice.820), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_25"}
  %slice.2029 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2002), slice={[0:1], [0:16], [0:16], [416:432]}, metadata={op_type="Split" op_name="split_17"}
  %slice.821 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.320), slice={[0:3], [0:3], [0:16], [416:432]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.2054 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2029, f32[3,3,16,16]{3,2,1,0} %slice.821), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_26"}
  %slice.2030 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2002), slice={[0:1], [0:16], [0:16], [432:448]}, metadata={op_type="Split" op_name="split_17"}
  %slice.822 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.320), slice={[0:3], [0:3], [0:16], [432:448]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.2055 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2030, f32[3,3,16,16]{3,2,1,0} %slice.822), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_27"}
  %slice.2031 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2002), slice={[0:1], [0:16], [0:16], [448:464]}, metadata={op_type="Split" op_name="split_17"}
  %slice.823 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.320), slice={[0:3], [0:3], [0:16], [448:464]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.2056 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2031, f32[3,3,16,16]{3,2,1,0} %slice.823), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_28"}
  %slice.2032 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2002), slice={[0:1], [0:16], [0:16], [464:480]}, metadata={op_type="Split" op_name="split_17"}
  %slice.824 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.320), slice={[0:3], [0:3], [0:16], [464:480]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.2057 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2032, f32[3,3,16,16]{3,2,1,0} %slice.824), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_29"}
  %slice.2033 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2002), slice={[0:1], [0:16], [0:16], [480:496]}, metadata={op_type="Split" op_name="split_17"}
  %slice.825 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.320), slice={[0:3], [0:3], [0:16], [480:496]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.2059 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2033, f32[3,3,16,16]{3,2,1,0} %slice.825), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_30"}
  %slice.2034 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2002), slice={[0:1], [0:16], [0:16], [496:512]}, metadata={op_type="Split" op_name="split_17"}
  %slice.826 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.320), slice={[0:3], [0:3], [0:16], [496:512]}, metadata={op_type="Split" op_name="split_16"}
  %convolution.2060 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2034, f32[3,3,16,16]{3,2,1,0} %slice.826), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv2_31"}
  %concatenate.2067 = f32[1,14,14,512]{3,2,1,0} concatenate(f32[1,14,14,16]{3,2,1,0} %convolution.2035, f32[1,14,14,16]{3,2,1,0} %convolution.2036, f32[1,14,14,16]{3,2,1,0} %convolution.2047, f32[1,14,14,16]{3,2,1,0} %convolution.2058, f32[1,14,14,16]{3,2,1,0} %convolution.2061, f32[1,14,14,16]{3,2,1,0} %convolution.2062, f32[1,14,14,16]{3,2,1,0} %convolution.2063, f32[1,14,14,16]{3,2,1,0} %convolution.2064, f32[1,14,14,16]{3,2,1,0} %convolution.2065, f32[1,14,14,16]{3,2,1,0} %convolution.2066, f32[1,14,14,16]{3,2,1,0} %convolution.2037, f32[1,14,14,16]{3,2,1,0} %convolution.2038, f32[1,14,14,16]{3,2,1,0} %convolution.2039, f32[1,14,14,16]{3,2,1,0} %convolution.2040, f32[1,14,14,16]{3,2,1,0} %convolution.2041, f32[1,14,14,16]{3,2,1,0} %convolution.2042, f32[1,14,14,16]{3,2,1,0} %convolution.2043, f32[1,14,14,16]{3,2,1,0} %convolution.2044, f32[1,14,14,16]{3,2,1,0} %convolution.2045, f32[1,14,14,16]{3,2,1,0} %convolution.2046, f32[1,14,14,16]{3,2,1,0} %convolution.2048, f32[1,14,14,16]{3,2,1,0} %convolution.2049, f32[1,14,14,16]{3,2,1,0} %convolution.2050, f32[1,14,14,16]{3,2,1,0} %convolution.2051, f32[1,14,14,16]{3,2,1,0} %convolution.2052, f32[1,14,14,16]{3,2,1,0} %convolution.2053, f32[1,14,14,16]{3,2,1,0} %convolution.2054, f32[1,14,14,16]{3,2,1,0} %convolution.2055, f32[1,14,14,16]{3,2,1,0} %convolution.2056, f32[1,14,14,16]{3,2,1,0} %convolution.2057, f32[1,14,14,16]{3,2,1,0} %convolution.2059, f32[1,14,14,16]{3,2,1,0} %convolution.2060), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_8"}
  %arg51.52 = f32[512]{0} parameter(51), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.321 = f32[512]{0} reshape(f32[512]{0} %arg51.52)
  %constant.2068 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit2_bn2/add"}
  %broadcast.2069 = f32[512]{0} broadcast(f32[] %constant.2068), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit2_bn2/add"}
  %add.2070 = f32[512]{0} add(f32[512]{0} %reshape.321, f32[512]{0} %broadcast.2069), metadata={op_type="AddV2" op_name="stage3_unit2_bn2/add"}
  %rsqrt.2071 = f32[512]{0} rsqrt(f32[512]{0} %add.2070), metadata={op_type="Rsqrt" op_name="stage3_unit2_bn2/Rsqrt"}
  %arg109.110 = f32[512]{0} parameter(109), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.379 = f32[512]{0} reshape(f32[512]{0} %arg109.110)
  %multiply.2072 = f32[512]{0} multiply(f32[512]{0} %rsqrt.2071, f32[512]{0} %reshape.379), metadata={op_type="Mul" op_name="stage3_unit2_bn2/mul"}
  %broadcast.2073 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.2072), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit2_bn2/mul_1"}
  %multiply.2074 = f32[1,14,14,512]{3,2,1,0} multiply(f32[1,14,14,512]{3,2,1,0} %concatenate.2067, f32[1,14,14,512]{3,2,1,0} %broadcast.2073), metadata={op_type="Mul" op_name="stage3_unit2_bn2/mul_1"}
  %arg216.217 = f32[512]{0} parameter(216), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.486 = f32[512]{0} reshape(f32[512]{0} %arg216.217)
  %arg163.164 = f32[512]{0} parameter(163), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.433 = f32[512]{0} reshape(f32[512]{0} %arg163.164)
  %multiply.2075 = f32[512]{0} multiply(f32[512]{0} %multiply.2072, f32[512]{0} %reshape.433), metadata={op_type="Mul" op_name="stage3_unit2_bn2/mul_2"}
  %subtract.2076 = f32[512]{0} subtract(f32[512]{0} %reshape.486, f32[512]{0} %multiply.2075), metadata={op_type="Sub" op_name="stage3_unit2_bn2/sub"}
  %broadcast.2077 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.2076), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit2_bn2/add_1"}
  %add.2078 = f32[1,14,14,512]{3,2,1,0} add(f32[1,14,14,512]{3,2,1,0} %multiply.2074, f32[1,14,14,512]{3,2,1,0} %broadcast.2077), metadata={op_type="AddV2" op_name="stage3_unit2_bn2/add_1"}
  %maximum.2081 = f32[1,14,14,512]{3,2,1,0} maximum(f32[1,14,14,512]{3,2,1,0} %broadcast.2080, f32[1,14,14,512]{3,2,1,0} %add.2078), metadata={op_type="Relu" op_name="stage3_unit2_relu2"}
  %arg252.253 = f32[1,1,512,1024]{3,2,1,0} parameter(252), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.522 = f32[1,1,512,1024]{3,2,1,0} reshape(f32[1,1,512,1024]{3,2,1,0} %arg252.253)
  %convolution.2082 = f32[1,14,14,1024]{3,2,1,0} convolution(f32[1,14,14,512]{3,2,1,0} %maximum.2081, f32[1,1,512,1024]{3,2,1,0} %reshape.522), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit2_conv3"}
  %multiply.2089 = f32[1,14,14,1024]{3,2,1,0} multiply(f32[1,14,14,1024]{3,2,1,0} %broadcast.2088, f32[1,14,14,1024]{3,2,1,0} %convolution.2082), metadata={op_type="Mul" op_name="stage3_unit2_bn3/mul_1"}
  %arg217.218 = f32[1024]{0} parameter(217), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.487 = f32[1024]{0} reshape(f32[1024]{0} %arg217.218)
  %arg164.165 = f32[1024]{0} parameter(164), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.434 = f32[1024]{0} reshape(f32[1024]{0} %arg164.165)
  %multiply.2090 = f32[1024]{0} multiply(f32[1024]{0} %multiply.2087, f32[1024]{0} %reshape.434), metadata={op_type="Mul" op_name="stage3_unit2_bn3/mul_2"}
  %subtract.2091 = f32[1024]{0} subtract(f32[1024]{0} %reshape.487, f32[1024]{0} %multiply.2090), metadata={op_type="Sub" op_name="stage3_unit2_bn3/sub"}
  %broadcast.2092 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %subtract.2091), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit2_bn3/add_1"}
  %add.2093 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %multiply.2089, f32[1,14,14,1024]{3,2,1,0} %broadcast.2092), metadata={op_type="AddV2" op_name="stage3_unit2_bn3/add_1"}
  %add.2094 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %maximum.1985, f32[1,14,14,1024]{3,2,1,0} %add.2093), metadata={op_type="AddV2" op_name="add_8"}
  %maximum.2097 = f32[1,14,14,1024]{3,2,1,0} maximum(f32[1,14,14,1024]{3,2,1,0} %broadcast.2096, f32[1,14,14,1024]{3,2,1,0} %add.2094), metadata={op_type="Relu" op_name="stage3_unit2_relu"}
  %arg57.58 = f32[1024]{0} parameter(57), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.327 = f32[1024]{0} reshape(f32[1024]{0} %arg57.58)
  %constant.2195 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit3_bn3/add"}
  %broadcast.2196 = f32[1024]{0} broadcast(f32[] %constant.2195), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit3_bn3/add"}
  %add.2197 = f32[1024]{0} add(f32[1024]{0} %reshape.327, f32[1024]{0} %broadcast.2196), metadata={op_type="AddV2" op_name="stage3_unit3_bn3/add"}
  %rsqrt.2198 = f32[1024]{0} rsqrt(f32[1024]{0} %add.2197), metadata={op_type="Rsqrt" op_name="stage3_unit3_bn3/Rsqrt"}
  %arg113.114 = f32[1024]{0} parameter(113), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.383 = f32[1024]{0} reshape(f32[1024]{0} %arg113.114)
  %multiply.2199 = f32[1024]{0} multiply(f32[1024]{0} %rsqrt.2198, f32[1024]{0} %reshape.383), metadata={op_type="Mul" op_name="stage3_unit3_bn3/mul"}
  %broadcast.2200 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %multiply.2199), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit3_bn3/mul_1"}
  %constant.2191 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit3_relu2"}
  %broadcast.2192 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[] %constant.2191), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit3_relu2"}
  %constant.2110 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit3_relu1"}
  %broadcast.2111 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[] %constant.2110), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit3_relu1"}
  %arg54.55 = f32[512]{0} parameter(54), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.324 = f32[512]{0} reshape(f32[512]{0} %arg54.55)
  %constant.2099 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit3_bn1/add"}
  %broadcast.2100 = f32[512]{0} broadcast(f32[] %constant.2099), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit3_bn1/add"}
  %add.2101 = f32[512]{0} add(f32[512]{0} %reshape.324, f32[512]{0} %broadcast.2100), metadata={op_type="AddV2" op_name="stage3_unit3_bn1/add"}
  %rsqrt.2102 = f32[512]{0} rsqrt(f32[512]{0} %add.2101), metadata={op_type="Rsqrt" op_name="stage3_unit3_bn1/Rsqrt"}
  %arg111.112 = f32[512]{0} parameter(111), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.381 = f32[512]{0} reshape(f32[512]{0} %arg111.112)
  %multiply.2103 = f32[512]{0} multiply(f32[512]{0} %rsqrt.2102, f32[512]{0} %reshape.381), metadata={op_type="Mul" op_name="stage3_unit3_bn1/mul"}
  %broadcast.2104 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.2103), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit3_bn1/mul_1"}
  %arg253.254 = f32[1,1,1024,512]{3,2,1,0} parameter(253), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.523 = f32[1,1,1024,512]{3,2,1,0} reshape(f32[1,1,1024,512]{3,2,1,0} %arg253.254)
  %convolution.2098 = f32[1,14,14,512]{3,2,1,0} convolution(f32[1,14,14,1024]{3,2,1,0} %maximum.2097, f32[1,1,1024,512]{3,2,1,0} %reshape.523), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv1"}
  %multiply.2105 = f32[1,14,14,512]{3,2,1,0} multiply(f32[1,14,14,512]{3,2,1,0} %broadcast.2104, f32[1,14,14,512]{3,2,1,0} %convolution.2098), metadata={op_type="Mul" op_name="stage3_unit3_bn1/mul_1"}
  %arg218.219 = f32[512]{0} parameter(218), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.488 = f32[512]{0} reshape(f32[512]{0} %arg218.219)
  %arg165.166 = f32[512]{0} parameter(165), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.435 = f32[512]{0} reshape(f32[512]{0} %arg165.166)
  %multiply.2106 = f32[512]{0} multiply(f32[512]{0} %multiply.2103, f32[512]{0} %reshape.435), metadata={op_type="Mul" op_name="stage3_unit3_bn1/mul_2"}
  %subtract.2107 = f32[512]{0} subtract(f32[512]{0} %reshape.488, f32[512]{0} %multiply.2106), metadata={op_type="Sub" op_name="stage3_unit3_bn1/sub"}
  %broadcast.2108 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.2107), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit3_bn1/add_1"}
  %add.2109 = f32[1,14,14,512]{3,2,1,0} add(f32[1,14,14,512]{3,2,1,0} %multiply.2105, f32[1,14,14,512]{3,2,1,0} %broadcast.2108), metadata={op_type="AddV2" op_name="stage3_unit3_bn1/add_1"}
  %maximum.2112 = f32[1,14,14,512]{3,2,1,0} maximum(f32[1,14,14,512]{3,2,1,0} %broadcast.2111, f32[1,14,14,512]{3,2,1,0} %add.2109), metadata={op_type="Relu" op_name="stage3_unit3_relu1"}
  %constant.2113 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_10"}
  %pad.2114 = f32[1,16,16,512]{3,2,1,0} pad(f32[1,14,14,512]{3,2,1,0} %maximum.2112, f32[] %constant.2113), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_10"}
  %slice.2115 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2114), slice={[0:1], [0:16], [0:16], [0:16]}, metadata={op_type="Split" op_name="split_19"}
  %arg55.56 = f32[3,3,16,512]{3,2,1,0} parameter(55), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.325 = f32[3,3,16,512]{3,2,1,0} reshape(f32[3,3,16,512]{3,2,1,0} %arg55.56)
  %slice.827 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.325), slice={[0:3], [0:3], [0:16], [0:16]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2147 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2115, f32[3,3,16,16]{3,2,1,0} %slice.827), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2"}
  %slice.2116 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2114), slice={[0:1], [0:16], [0:16], [16:32]}, metadata={op_type="Split" op_name="split_19"}
  %slice.828 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.325), slice={[0:3], [0:3], [0:16], [16:32]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2148 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2116, f32[3,3,16,16]{3,2,1,0} %slice.828), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_1"}
  %slice.2117 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2114), slice={[0:1], [0:16], [0:16], [32:48]}, metadata={op_type="Split" op_name="split_19"}
  %slice.829 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.325), slice={[0:3], [0:3], [0:16], [32:48]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2159 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2117, f32[3,3,16,16]{3,2,1,0} %slice.829), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_2"}
  %slice.2118 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2114), slice={[0:1], [0:16], [0:16], [48:64]}, metadata={op_type="Split" op_name="split_19"}
  %slice.830 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.325), slice={[0:3], [0:3], [0:16], [48:64]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2170 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2118, f32[3,3,16,16]{3,2,1,0} %slice.830), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_3"}
  %slice.2119 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2114), slice={[0:1], [0:16], [0:16], [64:80]}, metadata={op_type="Split" op_name="split_19"}
  %slice.831 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.325), slice={[0:3], [0:3], [0:16], [64:80]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2173 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2119, f32[3,3,16,16]{3,2,1,0} %slice.831), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_4"}
  %slice.2120 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2114), slice={[0:1], [0:16], [0:16], [80:96]}, metadata={op_type="Split" op_name="split_19"}
  %slice.832 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.325), slice={[0:3], [0:3], [0:16], [80:96]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2174 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2120, f32[3,3,16,16]{3,2,1,0} %slice.832), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_5"}
  %slice.2121 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2114), slice={[0:1], [0:16], [0:16], [96:112]}, metadata={op_type="Split" op_name="split_19"}
  %slice.833 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.325), slice={[0:3], [0:3], [0:16], [96:112]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2175 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2121, f32[3,3,16,16]{3,2,1,0} %slice.833), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_6"}
  %slice.2122 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2114), slice={[0:1], [0:16], [0:16], [112:128]}, metadata={op_type="Split" op_name="split_19"}
  %slice.834 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.325), slice={[0:3], [0:3], [0:16], [112:128]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2176 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2122, f32[3,3,16,16]{3,2,1,0} %slice.834), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_7"}
  %slice.2123 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2114), slice={[0:1], [0:16], [0:16], [128:144]}, metadata={op_type="Split" op_name="split_19"}
  %slice.835 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.325), slice={[0:3], [0:3], [0:16], [128:144]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2177 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2123, f32[3,3,16,16]{3,2,1,0} %slice.835), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_8"}
  %slice.2124 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2114), slice={[0:1], [0:16], [0:16], [144:160]}, metadata={op_type="Split" op_name="split_19"}
  %slice.836 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.325), slice={[0:3], [0:3], [0:16], [144:160]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2178 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2124, f32[3,3,16,16]{3,2,1,0} %slice.836), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_9"}
  %slice.2125 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2114), slice={[0:1], [0:16], [0:16], [160:176]}, metadata={op_type="Split" op_name="split_19"}
  %slice.837 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.325), slice={[0:3], [0:3], [0:16], [160:176]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2149 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2125, f32[3,3,16,16]{3,2,1,0} %slice.837), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_10"}
  %slice.2126 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2114), slice={[0:1], [0:16], [0:16], [176:192]}, metadata={op_type="Split" op_name="split_19"}
  %slice.838 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.325), slice={[0:3], [0:3], [0:16], [176:192]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2150 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2126, f32[3,3,16,16]{3,2,1,0} %slice.838), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_11"}
  %slice.2127 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2114), slice={[0:1], [0:16], [0:16], [192:208]}, metadata={op_type="Split" op_name="split_19"}
  %slice.839 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.325), slice={[0:3], [0:3], [0:16], [192:208]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2151 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2127, f32[3,3,16,16]{3,2,1,0} %slice.839), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_12"}
  %slice.2128 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2114), slice={[0:1], [0:16], [0:16], [208:224]}, metadata={op_type="Split" op_name="split_19"}
  %slice.840 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.325), slice={[0:3], [0:3], [0:16], [208:224]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2152 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2128, f32[3,3,16,16]{3,2,1,0} %slice.840), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_13"}
  %slice.2129 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2114), slice={[0:1], [0:16], [0:16], [224:240]}, metadata={op_type="Split" op_name="split_19"}
  %slice.841 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.325), slice={[0:3], [0:3], [0:16], [224:240]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2153 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2129, f32[3,3,16,16]{3,2,1,0} %slice.841), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_14"}
  %slice.2130 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2114), slice={[0:1], [0:16], [0:16], [240:256]}, metadata={op_type="Split" op_name="split_19"}
  %slice.842 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.325), slice={[0:3], [0:3], [0:16], [240:256]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2154 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2130, f32[3,3,16,16]{3,2,1,0} %slice.842), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_15"}
  %slice.2131 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2114), slice={[0:1], [0:16], [0:16], [256:272]}, metadata={op_type="Split" op_name="split_19"}
  %slice.843 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.325), slice={[0:3], [0:3], [0:16], [256:272]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2155 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2131, f32[3,3,16,16]{3,2,1,0} %slice.843), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_16"}
  %slice.2132 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2114), slice={[0:1], [0:16], [0:16], [272:288]}, metadata={op_type="Split" op_name="split_19"}
  %slice.844 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.325), slice={[0:3], [0:3], [0:16], [272:288]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2156 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2132, f32[3,3,16,16]{3,2,1,0} %slice.844), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_17"}
  %slice.2133 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2114), slice={[0:1], [0:16], [0:16], [288:304]}, metadata={op_type="Split" op_name="split_19"}
  %slice.845 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.325), slice={[0:3], [0:3], [0:16], [288:304]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2157 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2133, f32[3,3,16,16]{3,2,1,0} %slice.845), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_18"}
  %slice.2134 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2114), slice={[0:1], [0:16], [0:16], [304:320]}, metadata={op_type="Split" op_name="split_19"}
  %slice.846 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.325), slice={[0:3], [0:3], [0:16], [304:320]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2158 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2134, f32[3,3,16,16]{3,2,1,0} %slice.846), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_19"}
  %slice.2135 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2114), slice={[0:1], [0:16], [0:16], [320:336]}, metadata={op_type="Split" op_name="split_19"}
  %slice.847 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.325), slice={[0:3], [0:3], [0:16], [320:336]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2160 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2135, f32[3,3,16,16]{3,2,1,0} %slice.847), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_20"}
  %slice.2136 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2114), slice={[0:1], [0:16], [0:16], [336:352]}, metadata={op_type="Split" op_name="split_19"}
  %slice.848 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.325), slice={[0:3], [0:3], [0:16], [336:352]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2161 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2136, f32[3,3,16,16]{3,2,1,0} %slice.848), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_21"}
  %slice.2137 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2114), slice={[0:1], [0:16], [0:16], [352:368]}, metadata={op_type="Split" op_name="split_19"}
  %slice.849 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.325), slice={[0:3], [0:3], [0:16], [352:368]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2162 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2137, f32[3,3,16,16]{3,2,1,0} %slice.849), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_22"}
  %slice.2138 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2114), slice={[0:1], [0:16], [0:16], [368:384]}, metadata={op_type="Split" op_name="split_19"}
  %slice.850 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.325), slice={[0:3], [0:3], [0:16], [368:384]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2163 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2138, f32[3,3,16,16]{3,2,1,0} %slice.850), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_23"}
  %slice.2139 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2114), slice={[0:1], [0:16], [0:16], [384:400]}, metadata={op_type="Split" op_name="split_19"}
  %slice.851 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.325), slice={[0:3], [0:3], [0:16], [384:400]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2164 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2139, f32[3,3,16,16]{3,2,1,0} %slice.851), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_24"}
  %slice.2140 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2114), slice={[0:1], [0:16], [0:16], [400:416]}, metadata={op_type="Split" op_name="split_19"}
  %slice.852 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.325), slice={[0:3], [0:3], [0:16], [400:416]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2165 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2140, f32[3,3,16,16]{3,2,1,0} %slice.852), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_25"}
  %slice.2141 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2114), slice={[0:1], [0:16], [0:16], [416:432]}, metadata={op_type="Split" op_name="split_19"}
  %slice.853 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.325), slice={[0:3], [0:3], [0:16], [416:432]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2166 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2141, f32[3,3,16,16]{3,2,1,0} %slice.853), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_26"}
  %slice.2142 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2114), slice={[0:1], [0:16], [0:16], [432:448]}, metadata={op_type="Split" op_name="split_19"}
  %slice.854 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.325), slice={[0:3], [0:3], [0:16], [432:448]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2167 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2142, f32[3,3,16,16]{3,2,1,0} %slice.854), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_27"}
  %slice.2143 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2114), slice={[0:1], [0:16], [0:16], [448:464]}, metadata={op_type="Split" op_name="split_19"}
  %slice.855 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.325), slice={[0:3], [0:3], [0:16], [448:464]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2168 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2143, f32[3,3,16,16]{3,2,1,0} %slice.855), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_28"}
  %slice.2144 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2114), slice={[0:1], [0:16], [0:16], [464:480]}, metadata={op_type="Split" op_name="split_19"}
  %slice.856 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.325), slice={[0:3], [0:3], [0:16], [464:480]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2169 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2144, f32[3,3,16,16]{3,2,1,0} %slice.856), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_29"}
  %slice.2145 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2114), slice={[0:1], [0:16], [0:16], [480:496]}, metadata={op_type="Split" op_name="split_19"}
  %slice.857 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.325), slice={[0:3], [0:3], [0:16], [480:496]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2171 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2145, f32[3,3,16,16]{3,2,1,0} %slice.857), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_30"}
  %slice.2146 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2114), slice={[0:1], [0:16], [0:16], [496:512]}, metadata={op_type="Split" op_name="split_19"}
  %slice.858 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.325), slice={[0:3], [0:3], [0:16], [496:512]}, metadata={op_type="Split" op_name="split_18"}
  %convolution.2172 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2146, f32[3,3,16,16]{3,2,1,0} %slice.858), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv2_31"}
  %concatenate.2179 = f32[1,14,14,512]{3,2,1,0} concatenate(f32[1,14,14,16]{3,2,1,0} %convolution.2147, f32[1,14,14,16]{3,2,1,0} %convolution.2148, f32[1,14,14,16]{3,2,1,0} %convolution.2159, f32[1,14,14,16]{3,2,1,0} %convolution.2170, f32[1,14,14,16]{3,2,1,0} %convolution.2173, f32[1,14,14,16]{3,2,1,0} %convolution.2174, f32[1,14,14,16]{3,2,1,0} %convolution.2175, f32[1,14,14,16]{3,2,1,0} %convolution.2176, f32[1,14,14,16]{3,2,1,0} %convolution.2177, f32[1,14,14,16]{3,2,1,0} %convolution.2178, f32[1,14,14,16]{3,2,1,0} %convolution.2149, f32[1,14,14,16]{3,2,1,0} %convolution.2150, f32[1,14,14,16]{3,2,1,0} %convolution.2151, f32[1,14,14,16]{3,2,1,0} %convolution.2152, f32[1,14,14,16]{3,2,1,0} %convolution.2153, f32[1,14,14,16]{3,2,1,0} %convolution.2154, f32[1,14,14,16]{3,2,1,0} %convolution.2155, f32[1,14,14,16]{3,2,1,0} %convolution.2156, f32[1,14,14,16]{3,2,1,0} %convolution.2157, f32[1,14,14,16]{3,2,1,0} %convolution.2158, f32[1,14,14,16]{3,2,1,0} %convolution.2160, f32[1,14,14,16]{3,2,1,0} %convolution.2161, f32[1,14,14,16]{3,2,1,0} %convolution.2162, f32[1,14,14,16]{3,2,1,0} %convolution.2163, f32[1,14,14,16]{3,2,1,0} %convolution.2164, f32[1,14,14,16]{3,2,1,0} %convolution.2165, f32[1,14,14,16]{3,2,1,0} %convolution.2166, f32[1,14,14,16]{3,2,1,0} %convolution.2167, f32[1,14,14,16]{3,2,1,0} %convolution.2168, f32[1,14,14,16]{3,2,1,0} %convolution.2169, f32[1,14,14,16]{3,2,1,0} %convolution.2171, f32[1,14,14,16]{3,2,1,0} %convolution.2172), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_9"}
  %arg56.57 = f32[512]{0} parameter(56), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.326 = f32[512]{0} reshape(f32[512]{0} %arg56.57)
  %constant.2180 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit3_bn2/add"}
  %broadcast.2181 = f32[512]{0} broadcast(f32[] %constant.2180), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit3_bn2/add"}
  %add.2182 = f32[512]{0} add(f32[512]{0} %reshape.326, f32[512]{0} %broadcast.2181), metadata={op_type="AddV2" op_name="stage3_unit3_bn2/add"}
  %rsqrt.2183 = f32[512]{0} rsqrt(f32[512]{0} %add.2182), metadata={op_type="Rsqrt" op_name="stage3_unit3_bn2/Rsqrt"}
  %arg112.113 = f32[512]{0} parameter(112), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.382 = f32[512]{0} reshape(f32[512]{0} %arg112.113)
  %multiply.2184 = f32[512]{0} multiply(f32[512]{0} %rsqrt.2183, f32[512]{0} %reshape.382), metadata={op_type="Mul" op_name="stage3_unit3_bn2/mul"}
  %broadcast.2185 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.2184), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit3_bn2/mul_1"}
  %multiply.2186 = f32[1,14,14,512]{3,2,1,0} multiply(f32[1,14,14,512]{3,2,1,0} %concatenate.2179, f32[1,14,14,512]{3,2,1,0} %broadcast.2185), metadata={op_type="Mul" op_name="stage3_unit3_bn2/mul_1"}
  %arg219.220 = f32[512]{0} parameter(219), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.489 = f32[512]{0} reshape(f32[512]{0} %arg219.220)
  %arg166.167 = f32[512]{0} parameter(166), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.436 = f32[512]{0} reshape(f32[512]{0} %arg166.167)
  %multiply.2187 = f32[512]{0} multiply(f32[512]{0} %multiply.2184, f32[512]{0} %reshape.436), metadata={op_type="Mul" op_name="stage3_unit3_bn2/mul_2"}
  %subtract.2188 = f32[512]{0} subtract(f32[512]{0} %reshape.489, f32[512]{0} %multiply.2187), metadata={op_type="Sub" op_name="stage3_unit3_bn2/sub"}
  %broadcast.2189 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.2188), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit3_bn2/add_1"}
  %add.2190 = f32[1,14,14,512]{3,2,1,0} add(f32[1,14,14,512]{3,2,1,0} %multiply.2186, f32[1,14,14,512]{3,2,1,0} %broadcast.2189), metadata={op_type="AddV2" op_name="stage3_unit3_bn2/add_1"}
  %maximum.2193 = f32[1,14,14,512]{3,2,1,0} maximum(f32[1,14,14,512]{3,2,1,0} %broadcast.2192, f32[1,14,14,512]{3,2,1,0} %add.2190), metadata={op_type="Relu" op_name="stage3_unit3_relu2"}
  %arg254.255 = f32[1,1,512,1024]{3,2,1,0} parameter(254), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.524 = f32[1,1,512,1024]{3,2,1,0} reshape(f32[1,1,512,1024]{3,2,1,0} %arg254.255)
  %convolution.2194 = f32[1,14,14,1024]{3,2,1,0} convolution(f32[1,14,14,512]{3,2,1,0} %maximum.2193, f32[1,1,512,1024]{3,2,1,0} %reshape.524), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit3_conv3"}
  %multiply.2201 = f32[1,14,14,1024]{3,2,1,0} multiply(f32[1,14,14,1024]{3,2,1,0} %broadcast.2200, f32[1,14,14,1024]{3,2,1,0} %convolution.2194), metadata={op_type="Mul" op_name="stage3_unit3_bn3/mul_1"}
  %arg220.221 = f32[1024]{0} parameter(220), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.490 = f32[1024]{0} reshape(f32[1024]{0} %arg220.221)
  %arg167.168 = f32[1024]{0} parameter(167), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.437 = f32[1024]{0} reshape(f32[1024]{0} %arg167.168)
  %multiply.2202 = f32[1024]{0} multiply(f32[1024]{0} %multiply.2199, f32[1024]{0} %reshape.437), metadata={op_type="Mul" op_name="stage3_unit3_bn3/mul_2"}
  %subtract.2203 = f32[1024]{0} subtract(f32[1024]{0} %reshape.490, f32[1024]{0} %multiply.2202), metadata={op_type="Sub" op_name="stage3_unit3_bn3/sub"}
  %broadcast.2204 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %subtract.2203), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit3_bn3/add_1"}
  %add.2205 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %multiply.2201, f32[1,14,14,1024]{3,2,1,0} %broadcast.2204), metadata={op_type="AddV2" op_name="stage3_unit3_bn3/add_1"}
  %add.2206 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %maximum.2097, f32[1,14,14,1024]{3,2,1,0} %add.2205), metadata={op_type="AddV2" op_name="add_9"}
  %maximum.2209 = f32[1,14,14,1024]{3,2,1,0} maximum(f32[1,14,14,1024]{3,2,1,0} %broadcast.2208, f32[1,14,14,1024]{3,2,1,0} %add.2206), metadata={op_type="Relu" op_name="stage3_unit3_relu"}
  %arg62.63 = f32[1024]{0} parameter(62), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.332 = f32[1024]{0} reshape(f32[1024]{0} %arg62.63)
  %constant.2307 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit4_bn3/add"}
  %broadcast.2308 = f32[1024]{0} broadcast(f32[] %constant.2307), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit4_bn3/add"}
  %add.2309 = f32[1024]{0} add(f32[1024]{0} %reshape.332, f32[1024]{0} %broadcast.2308), metadata={op_type="AddV2" op_name="stage3_unit4_bn3/add"}
  %rsqrt.2310 = f32[1024]{0} rsqrt(f32[1024]{0} %add.2309), metadata={op_type="Rsqrt" op_name="stage3_unit4_bn3/Rsqrt"}
  %arg117.118 = f32[1024]{0} parameter(117), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.387 = f32[1024]{0} reshape(f32[1024]{0} %arg117.118)
  %multiply.2311 = f32[1024]{0} multiply(f32[1024]{0} %rsqrt.2310, f32[1024]{0} %reshape.387), metadata={op_type="Mul" op_name="stage3_unit4_bn3/mul"}
  %broadcast.2312 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %multiply.2311), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit4_bn3/mul_1"}
  %constant.2303 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit4_relu2"}
  %broadcast.2304 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[] %constant.2303), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit4_relu2"}
  %constant.2222 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit4_relu1"}
  %broadcast.2223 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[] %constant.2222), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit4_relu1"}
  %arg59.60 = f32[512]{0} parameter(59), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.329 = f32[512]{0} reshape(f32[512]{0} %arg59.60)
  %constant.2211 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit4_bn1/add"}
  %broadcast.2212 = f32[512]{0} broadcast(f32[] %constant.2211), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit4_bn1/add"}
  %add.2213 = f32[512]{0} add(f32[512]{0} %reshape.329, f32[512]{0} %broadcast.2212), metadata={op_type="AddV2" op_name="stage3_unit4_bn1/add"}
  %rsqrt.2214 = f32[512]{0} rsqrt(f32[512]{0} %add.2213), metadata={op_type="Rsqrt" op_name="stage3_unit4_bn1/Rsqrt"}
  %arg115.116 = f32[512]{0} parameter(115), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.385 = f32[512]{0} reshape(f32[512]{0} %arg115.116)
  %multiply.2215 = f32[512]{0} multiply(f32[512]{0} %rsqrt.2214, f32[512]{0} %reshape.385), metadata={op_type="Mul" op_name="stage3_unit4_bn1/mul"}
  %broadcast.2216 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.2215), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit4_bn1/mul_1"}
  %arg255.256 = f32[1,1,1024,512]{3,2,1,0} parameter(255), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.525 = f32[1,1,1024,512]{3,2,1,0} reshape(f32[1,1,1024,512]{3,2,1,0} %arg255.256)
  %convolution.2210 = f32[1,14,14,512]{3,2,1,0} convolution(f32[1,14,14,1024]{3,2,1,0} %maximum.2209, f32[1,1,1024,512]{3,2,1,0} %reshape.525), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv1"}
  %multiply.2217 = f32[1,14,14,512]{3,2,1,0} multiply(f32[1,14,14,512]{3,2,1,0} %broadcast.2216, f32[1,14,14,512]{3,2,1,0} %convolution.2210), metadata={op_type="Mul" op_name="stage3_unit4_bn1/mul_1"}
  %arg222.223 = f32[512]{0} parameter(222), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.492 = f32[512]{0} reshape(f32[512]{0} %arg222.223)
  %arg169.170 = f32[512]{0} parameter(169), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.439 = f32[512]{0} reshape(f32[512]{0} %arg169.170)
  %multiply.2218 = f32[512]{0} multiply(f32[512]{0} %multiply.2215, f32[512]{0} %reshape.439), metadata={op_type="Mul" op_name="stage3_unit4_bn1/mul_2"}
  %subtract.2219 = f32[512]{0} subtract(f32[512]{0} %reshape.492, f32[512]{0} %multiply.2218), metadata={op_type="Sub" op_name="stage3_unit4_bn1/sub"}
  %broadcast.2220 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.2219), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit4_bn1/add_1"}
  %add.2221 = f32[1,14,14,512]{3,2,1,0} add(f32[1,14,14,512]{3,2,1,0} %multiply.2217, f32[1,14,14,512]{3,2,1,0} %broadcast.2220), metadata={op_type="AddV2" op_name="stage3_unit4_bn1/add_1"}
  %maximum.2224 = f32[1,14,14,512]{3,2,1,0} maximum(f32[1,14,14,512]{3,2,1,0} %broadcast.2223, f32[1,14,14,512]{3,2,1,0} %add.2221), metadata={op_type="Relu" op_name="stage3_unit4_relu1"}
  %constant.2225 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_11"}
  %pad.2226 = f32[1,16,16,512]{3,2,1,0} pad(f32[1,14,14,512]{3,2,1,0} %maximum.2224, f32[] %constant.2225), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_11"}
  %slice.2227 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2226), slice={[0:1], [0:16], [0:16], [0:16]}, metadata={op_type="Split" op_name="split_21"}
  %arg60.61 = f32[3,3,16,512]{3,2,1,0} parameter(60), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.330 = f32[3,3,16,512]{3,2,1,0} reshape(f32[3,3,16,512]{3,2,1,0} %arg60.61)
  %slice.859 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.330), slice={[0:3], [0:3], [0:16], [0:16]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2259 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2227, f32[3,3,16,16]{3,2,1,0} %slice.859), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2"}
  %slice.2228 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2226), slice={[0:1], [0:16], [0:16], [16:32]}, metadata={op_type="Split" op_name="split_21"}
  %slice.860 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.330), slice={[0:3], [0:3], [0:16], [16:32]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2260 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2228, f32[3,3,16,16]{3,2,1,0} %slice.860), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_1"}
  %slice.2229 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2226), slice={[0:1], [0:16], [0:16], [32:48]}, metadata={op_type="Split" op_name="split_21"}
  %slice.861 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.330), slice={[0:3], [0:3], [0:16], [32:48]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2271 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2229, f32[3,3,16,16]{3,2,1,0} %slice.861), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_2"}
  %slice.2230 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2226), slice={[0:1], [0:16], [0:16], [48:64]}, metadata={op_type="Split" op_name="split_21"}
  %slice.862 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.330), slice={[0:3], [0:3], [0:16], [48:64]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2282 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2230, f32[3,3,16,16]{3,2,1,0} %slice.862), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_3"}
  %slice.2231 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2226), slice={[0:1], [0:16], [0:16], [64:80]}, metadata={op_type="Split" op_name="split_21"}
  %slice.863 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.330), slice={[0:3], [0:3], [0:16], [64:80]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2285 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2231, f32[3,3,16,16]{3,2,1,0} %slice.863), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_4"}
  %slice.2232 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2226), slice={[0:1], [0:16], [0:16], [80:96]}, metadata={op_type="Split" op_name="split_21"}
  %slice.864 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.330), slice={[0:3], [0:3], [0:16], [80:96]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2286 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2232, f32[3,3,16,16]{3,2,1,0} %slice.864), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_5"}
  %slice.2233 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2226), slice={[0:1], [0:16], [0:16], [96:112]}, metadata={op_type="Split" op_name="split_21"}
  %slice.865 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.330), slice={[0:3], [0:3], [0:16], [96:112]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2287 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2233, f32[3,3,16,16]{3,2,1,0} %slice.865), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_6"}
  %slice.2234 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2226), slice={[0:1], [0:16], [0:16], [112:128]}, metadata={op_type="Split" op_name="split_21"}
  %slice.866 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.330), slice={[0:3], [0:3], [0:16], [112:128]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2288 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2234, f32[3,3,16,16]{3,2,1,0} %slice.866), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_7"}
  %slice.2235 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2226), slice={[0:1], [0:16], [0:16], [128:144]}, metadata={op_type="Split" op_name="split_21"}
  %slice.867 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.330), slice={[0:3], [0:3], [0:16], [128:144]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2289 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2235, f32[3,3,16,16]{3,2,1,0} %slice.867), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_8"}
  %slice.2236 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2226), slice={[0:1], [0:16], [0:16], [144:160]}, metadata={op_type="Split" op_name="split_21"}
  %slice.868 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.330), slice={[0:3], [0:3], [0:16], [144:160]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2290 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2236, f32[3,3,16,16]{3,2,1,0} %slice.868), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_9"}
  %slice.2237 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2226), slice={[0:1], [0:16], [0:16], [160:176]}, metadata={op_type="Split" op_name="split_21"}
  %slice.869 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.330), slice={[0:3], [0:3], [0:16], [160:176]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2261 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2237, f32[3,3,16,16]{3,2,1,0} %slice.869), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_10"}
  %slice.2238 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2226), slice={[0:1], [0:16], [0:16], [176:192]}, metadata={op_type="Split" op_name="split_21"}
  %slice.870 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.330), slice={[0:3], [0:3], [0:16], [176:192]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2262 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2238, f32[3,3,16,16]{3,2,1,0} %slice.870), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_11"}
  %slice.2239 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2226), slice={[0:1], [0:16], [0:16], [192:208]}, metadata={op_type="Split" op_name="split_21"}
  %slice.871 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.330), slice={[0:3], [0:3], [0:16], [192:208]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2263 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2239, f32[3,3,16,16]{3,2,1,0} %slice.871), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_12"}
  %slice.2240 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2226), slice={[0:1], [0:16], [0:16], [208:224]}, metadata={op_type="Split" op_name="split_21"}
  %slice.872 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.330), slice={[0:3], [0:3], [0:16], [208:224]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2264 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2240, f32[3,3,16,16]{3,2,1,0} %slice.872), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_13"}
  %slice.2241 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2226), slice={[0:1], [0:16], [0:16], [224:240]}, metadata={op_type="Split" op_name="split_21"}
  %slice.873 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.330), slice={[0:3], [0:3], [0:16], [224:240]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2265 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2241, f32[3,3,16,16]{3,2,1,0} %slice.873), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_14"}
  %slice.2242 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2226), slice={[0:1], [0:16], [0:16], [240:256]}, metadata={op_type="Split" op_name="split_21"}
  %slice.874 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.330), slice={[0:3], [0:3], [0:16], [240:256]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2266 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2242, f32[3,3,16,16]{3,2,1,0} %slice.874), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_15"}
  %slice.2243 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2226), slice={[0:1], [0:16], [0:16], [256:272]}, metadata={op_type="Split" op_name="split_21"}
  %slice.875 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.330), slice={[0:3], [0:3], [0:16], [256:272]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2267 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2243, f32[3,3,16,16]{3,2,1,0} %slice.875), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_16"}
  %slice.2244 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2226), slice={[0:1], [0:16], [0:16], [272:288]}, metadata={op_type="Split" op_name="split_21"}
  %slice.876 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.330), slice={[0:3], [0:3], [0:16], [272:288]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2268 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2244, f32[3,3,16,16]{3,2,1,0} %slice.876), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_17"}
  %slice.2245 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2226), slice={[0:1], [0:16], [0:16], [288:304]}, metadata={op_type="Split" op_name="split_21"}
  %slice.877 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.330), slice={[0:3], [0:3], [0:16], [288:304]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2269 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2245, f32[3,3,16,16]{3,2,1,0} %slice.877), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_18"}
  %slice.2246 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2226), slice={[0:1], [0:16], [0:16], [304:320]}, metadata={op_type="Split" op_name="split_21"}
  %slice.878 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.330), slice={[0:3], [0:3], [0:16], [304:320]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2270 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2246, f32[3,3,16,16]{3,2,1,0} %slice.878), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_19"}
  %slice.2247 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2226), slice={[0:1], [0:16], [0:16], [320:336]}, metadata={op_type="Split" op_name="split_21"}
  %slice.879 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.330), slice={[0:3], [0:3], [0:16], [320:336]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2272 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2247, f32[3,3,16,16]{3,2,1,0} %slice.879), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_20"}
  %slice.2248 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2226), slice={[0:1], [0:16], [0:16], [336:352]}, metadata={op_type="Split" op_name="split_21"}
  %slice.880 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.330), slice={[0:3], [0:3], [0:16], [336:352]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2273 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2248, f32[3,3,16,16]{3,2,1,0} %slice.880), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_21"}
  %slice.2249 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2226), slice={[0:1], [0:16], [0:16], [352:368]}, metadata={op_type="Split" op_name="split_21"}
  %slice.881 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.330), slice={[0:3], [0:3], [0:16], [352:368]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2274 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2249, f32[3,3,16,16]{3,2,1,0} %slice.881), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_22"}
  %slice.2250 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2226), slice={[0:1], [0:16], [0:16], [368:384]}, metadata={op_type="Split" op_name="split_21"}
  %slice.882 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.330), slice={[0:3], [0:3], [0:16], [368:384]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2275 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2250, f32[3,3,16,16]{3,2,1,0} %slice.882), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_23"}
  %slice.2251 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2226), slice={[0:1], [0:16], [0:16], [384:400]}, metadata={op_type="Split" op_name="split_21"}
  %slice.883 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.330), slice={[0:3], [0:3], [0:16], [384:400]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2276 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2251, f32[3,3,16,16]{3,2,1,0} %slice.883), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_24"}
  %slice.2252 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2226), slice={[0:1], [0:16], [0:16], [400:416]}, metadata={op_type="Split" op_name="split_21"}
  %slice.884 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.330), slice={[0:3], [0:3], [0:16], [400:416]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2277 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2252, f32[3,3,16,16]{3,2,1,0} %slice.884), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_25"}
  %slice.2253 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2226), slice={[0:1], [0:16], [0:16], [416:432]}, metadata={op_type="Split" op_name="split_21"}
  %slice.885 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.330), slice={[0:3], [0:3], [0:16], [416:432]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2278 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2253, f32[3,3,16,16]{3,2,1,0} %slice.885), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_26"}
  %slice.2254 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2226), slice={[0:1], [0:16], [0:16], [432:448]}, metadata={op_type="Split" op_name="split_21"}
  %slice.886 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.330), slice={[0:3], [0:3], [0:16], [432:448]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2279 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2254, f32[3,3,16,16]{3,2,1,0} %slice.886), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_27"}
  %slice.2255 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2226), slice={[0:1], [0:16], [0:16], [448:464]}, metadata={op_type="Split" op_name="split_21"}
  %slice.887 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.330), slice={[0:3], [0:3], [0:16], [448:464]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2280 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2255, f32[3,3,16,16]{3,2,1,0} %slice.887), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_28"}
  %slice.2256 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2226), slice={[0:1], [0:16], [0:16], [464:480]}, metadata={op_type="Split" op_name="split_21"}
  %slice.888 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.330), slice={[0:3], [0:3], [0:16], [464:480]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2281 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2256, f32[3,3,16,16]{3,2,1,0} %slice.888), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_29"}
  %slice.2257 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2226), slice={[0:1], [0:16], [0:16], [480:496]}, metadata={op_type="Split" op_name="split_21"}
  %slice.889 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.330), slice={[0:3], [0:3], [0:16], [480:496]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2283 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2257, f32[3,3,16,16]{3,2,1,0} %slice.889), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_30"}
  %slice.2258 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2226), slice={[0:1], [0:16], [0:16], [496:512]}, metadata={op_type="Split" op_name="split_21"}
  %slice.890 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.330), slice={[0:3], [0:3], [0:16], [496:512]}, metadata={op_type="Split" op_name="split_20"}
  %convolution.2284 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2258, f32[3,3,16,16]{3,2,1,0} %slice.890), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv2_31"}
  %concatenate.2291 = f32[1,14,14,512]{3,2,1,0} concatenate(f32[1,14,14,16]{3,2,1,0} %convolution.2259, f32[1,14,14,16]{3,2,1,0} %convolution.2260, f32[1,14,14,16]{3,2,1,0} %convolution.2271, f32[1,14,14,16]{3,2,1,0} %convolution.2282, f32[1,14,14,16]{3,2,1,0} %convolution.2285, f32[1,14,14,16]{3,2,1,0} %convolution.2286, f32[1,14,14,16]{3,2,1,0} %convolution.2287, f32[1,14,14,16]{3,2,1,0} %convolution.2288, f32[1,14,14,16]{3,2,1,0} %convolution.2289, f32[1,14,14,16]{3,2,1,0} %convolution.2290, f32[1,14,14,16]{3,2,1,0} %convolution.2261, f32[1,14,14,16]{3,2,1,0} %convolution.2262, f32[1,14,14,16]{3,2,1,0} %convolution.2263, f32[1,14,14,16]{3,2,1,0} %convolution.2264, f32[1,14,14,16]{3,2,1,0} %convolution.2265, f32[1,14,14,16]{3,2,1,0} %convolution.2266, f32[1,14,14,16]{3,2,1,0} %convolution.2267, f32[1,14,14,16]{3,2,1,0} %convolution.2268, f32[1,14,14,16]{3,2,1,0} %convolution.2269, f32[1,14,14,16]{3,2,1,0} %convolution.2270, f32[1,14,14,16]{3,2,1,0} %convolution.2272, f32[1,14,14,16]{3,2,1,0} %convolution.2273, f32[1,14,14,16]{3,2,1,0} %convolution.2274, f32[1,14,14,16]{3,2,1,0} %convolution.2275, f32[1,14,14,16]{3,2,1,0} %convolution.2276, f32[1,14,14,16]{3,2,1,0} %convolution.2277, f32[1,14,14,16]{3,2,1,0} %convolution.2278, f32[1,14,14,16]{3,2,1,0} %convolution.2279, f32[1,14,14,16]{3,2,1,0} %convolution.2280, f32[1,14,14,16]{3,2,1,0} %convolution.2281, f32[1,14,14,16]{3,2,1,0} %convolution.2283, f32[1,14,14,16]{3,2,1,0} %convolution.2284), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_10"}
  %arg61.62 = f32[512]{0} parameter(61), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.331 = f32[512]{0} reshape(f32[512]{0} %arg61.62)
  %constant.2292 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit4_bn2/add"}
  %broadcast.2293 = f32[512]{0} broadcast(f32[] %constant.2292), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit4_bn2/add"}
  %add.2294 = f32[512]{0} add(f32[512]{0} %reshape.331, f32[512]{0} %broadcast.2293), metadata={op_type="AddV2" op_name="stage3_unit4_bn2/add"}
  %rsqrt.2295 = f32[512]{0} rsqrt(f32[512]{0} %add.2294), metadata={op_type="Rsqrt" op_name="stage3_unit4_bn2/Rsqrt"}
  %arg116.117 = f32[512]{0} parameter(116), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.386 = f32[512]{0} reshape(f32[512]{0} %arg116.117)
  %multiply.2296 = f32[512]{0} multiply(f32[512]{0} %rsqrt.2295, f32[512]{0} %reshape.386), metadata={op_type="Mul" op_name="stage3_unit4_bn2/mul"}
  %broadcast.2297 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.2296), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit4_bn2/mul_1"}
  %multiply.2298 = f32[1,14,14,512]{3,2,1,0} multiply(f32[1,14,14,512]{3,2,1,0} %concatenate.2291, f32[1,14,14,512]{3,2,1,0} %broadcast.2297), metadata={op_type="Mul" op_name="stage3_unit4_bn2/mul_1"}
  %arg223.224 = f32[512]{0} parameter(223), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.493 = f32[512]{0} reshape(f32[512]{0} %arg223.224)
  %arg170.171 = f32[512]{0} parameter(170), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.440 = f32[512]{0} reshape(f32[512]{0} %arg170.171)
  %multiply.2299 = f32[512]{0} multiply(f32[512]{0} %multiply.2296, f32[512]{0} %reshape.440), metadata={op_type="Mul" op_name="stage3_unit4_bn2/mul_2"}
  %subtract.2300 = f32[512]{0} subtract(f32[512]{0} %reshape.493, f32[512]{0} %multiply.2299), metadata={op_type="Sub" op_name="stage3_unit4_bn2/sub"}
  %broadcast.2301 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.2300), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit4_bn2/add_1"}
  %add.2302 = f32[1,14,14,512]{3,2,1,0} add(f32[1,14,14,512]{3,2,1,0} %multiply.2298, f32[1,14,14,512]{3,2,1,0} %broadcast.2301), metadata={op_type="AddV2" op_name="stage3_unit4_bn2/add_1"}
  %maximum.2305 = f32[1,14,14,512]{3,2,1,0} maximum(f32[1,14,14,512]{3,2,1,0} %broadcast.2304, f32[1,14,14,512]{3,2,1,0} %add.2302), metadata={op_type="Relu" op_name="stage3_unit4_relu2"}
  %arg256.257 = f32[1,1,512,1024]{3,2,1,0} parameter(256), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.526 = f32[1,1,512,1024]{3,2,1,0} reshape(f32[1,1,512,1024]{3,2,1,0} %arg256.257)
  %convolution.2306 = f32[1,14,14,1024]{3,2,1,0} convolution(f32[1,14,14,512]{3,2,1,0} %maximum.2305, f32[1,1,512,1024]{3,2,1,0} %reshape.526), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit4_conv3"}
  %multiply.2313 = f32[1,14,14,1024]{3,2,1,0} multiply(f32[1,14,14,1024]{3,2,1,0} %broadcast.2312, f32[1,14,14,1024]{3,2,1,0} %convolution.2306), metadata={op_type="Mul" op_name="stage3_unit4_bn3/mul_1"}
  %arg224.225 = f32[1024]{0} parameter(224), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.494 = f32[1024]{0} reshape(f32[1024]{0} %arg224.225)
  %arg171.172 = f32[1024]{0} parameter(171), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.441 = f32[1024]{0} reshape(f32[1024]{0} %arg171.172)
  %multiply.2314 = f32[1024]{0} multiply(f32[1024]{0} %multiply.2311, f32[1024]{0} %reshape.441), metadata={op_type="Mul" op_name="stage3_unit4_bn3/mul_2"}
  %subtract.2315 = f32[1024]{0} subtract(f32[1024]{0} %reshape.494, f32[1024]{0} %multiply.2314), metadata={op_type="Sub" op_name="stage3_unit4_bn3/sub"}
  %broadcast.2316 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %subtract.2315), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit4_bn3/add_1"}
  %add.2317 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %multiply.2313, f32[1,14,14,1024]{3,2,1,0} %broadcast.2316), metadata={op_type="AddV2" op_name="stage3_unit4_bn3/add_1"}
  %add.2318 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %maximum.2209, f32[1,14,14,1024]{3,2,1,0} %add.2317), metadata={op_type="AddV2" op_name="add_10"}
  %maximum.2321 = f32[1,14,14,1024]{3,2,1,0} maximum(f32[1,14,14,1024]{3,2,1,0} %broadcast.2320, f32[1,14,14,1024]{3,2,1,0} %add.2318), metadata={op_type="Relu" op_name="stage3_unit4_relu"}
  %arg68.69 = f32[1024]{0} parameter(68), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.338 = f32[1024]{0} reshape(f32[1024]{0} %arg68.69)
  %constant.2419 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit5_bn3/add"}
  %broadcast.2420 = f32[1024]{0} broadcast(f32[] %constant.2419), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit5_bn3/add"}
  %add.2421 = f32[1024]{0} add(f32[1024]{0} %reshape.338, f32[1024]{0} %broadcast.2420), metadata={op_type="AddV2" op_name="stage3_unit5_bn3/add"}
  %rsqrt.2422 = f32[1024]{0} rsqrt(f32[1024]{0} %add.2421), metadata={op_type="Rsqrt" op_name="stage3_unit5_bn3/Rsqrt"}
  %arg122.123 = f32[1024]{0} parameter(122), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.392 = f32[1024]{0} reshape(f32[1024]{0} %arg122.123)
  %multiply.2423 = f32[1024]{0} multiply(f32[1024]{0} %rsqrt.2422, f32[1024]{0} %reshape.392), metadata={op_type="Mul" op_name="stage3_unit5_bn3/mul"}
  %broadcast.2424 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %multiply.2423), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit5_bn3/mul_1"}
  %constant.2415 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit5_relu2"}
  %broadcast.2416 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[] %constant.2415), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit5_relu2"}
  %constant.2334 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit5_relu1"}
  %broadcast.2335 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[] %constant.2334), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit5_relu1"}
  %arg65.66 = f32[512]{0} parameter(65), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.335 = f32[512]{0} reshape(f32[512]{0} %arg65.66)
  %constant.2323 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit5_bn1/add"}
  %broadcast.2324 = f32[512]{0} broadcast(f32[] %constant.2323), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit5_bn1/add"}
  %add.2325 = f32[512]{0} add(f32[512]{0} %reshape.335, f32[512]{0} %broadcast.2324), metadata={op_type="AddV2" op_name="stage3_unit5_bn1/add"}
  %rsqrt.2326 = f32[512]{0} rsqrt(f32[512]{0} %add.2325), metadata={op_type="Rsqrt" op_name="stage3_unit5_bn1/Rsqrt"}
  %arg120.121 = f32[512]{0} parameter(120), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.390 = f32[512]{0} reshape(f32[512]{0} %arg120.121)
  %multiply.2327 = f32[512]{0} multiply(f32[512]{0} %rsqrt.2326, f32[512]{0} %reshape.390), metadata={op_type="Mul" op_name="stage3_unit5_bn1/mul"}
  %broadcast.2328 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.2327), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit5_bn1/mul_1"}
  %arg257.258 = f32[1,1,1024,512]{3,2,1,0} parameter(257), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.527 = f32[1,1,1024,512]{3,2,1,0} reshape(f32[1,1,1024,512]{3,2,1,0} %arg257.258)
  %convolution.2322 = f32[1,14,14,512]{3,2,1,0} convolution(f32[1,14,14,1024]{3,2,1,0} %maximum.2321, f32[1,1,1024,512]{3,2,1,0} %reshape.527), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv1"}
  %multiply.2329 = f32[1,14,14,512]{3,2,1,0} multiply(f32[1,14,14,512]{3,2,1,0} %broadcast.2328, f32[1,14,14,512]{3,2,1,0} %convolution.2322), metadata={op_type="Mul" op_name="stage3_unit5_bn1/mul_1"}
  %arg227.228 = f32[512]{0} parameter(227), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.497 = f32[512]{0} reshape(f32[512]{0} %arg227.228)
  %arg174.175 = f32[512]{0} parameter(174), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.444 = f32[512]{0} reshape(f32[512]{0} %arg174.175)
  %multiply.2330 = f32[512]{0} multiply(f32[512]{0} %multiply.2327, f32[512]{0} %reshape.444), metadata={op_type="Mul" op_name="stage3_unit5_bn1/mul_2"}
  %subtract.2331 = f32[512]{0} subtract(f32[512]{0} %reshape.497, f32[512]{0} %multiply.2330), metadata={op_type="Sub" op_name="stage3_unit5_bn1/sub"}
  %broadcast.2332 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.2331), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit5_bn1/add_1"}
  %add.2333 = f32[1,14,14,512]{3,2,1,0} add(f32[1,14,14,512]{3,2,1,0} %multiply.2329, f32[1,14,14,512]{3,2,1,0} %broadcast.2332), metadata={op_type="AddV2" op_name="stage3_unit5_bn1/add_1"}
  %maximum.2336 = f32[1,14,14,512]{3,2,1,0} maximum(f32[1,14,14,512]{3,2,1,0} %broadcast.2335, f32[1,14,14,512]{3,2,1,0} %add.2333), metadata={op_type="Relu" op_name="stage3_unit5_relu1"}
  %constant.2337 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_12"}
  %pad.2338 = f32[1,16,16,512]{3,2,1,0} pad(f32[1,14,14,512]{3,2,1,0} %maximum.2336, f32[] %constant.2337), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_12"}
  %slice.2339 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2338), slice={[0:1], [0:16], [0:16], [0:16]}, metadata={op_type="Split" op_name="split_23"}
  %arg66.67 = f32[3,3,16,512]{3,2,1,0} parameter(66), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.336 = f32[3,3,16,512]{3,2,1,0} reshape(f32[3,3,16,512]{3,2,1,0} %arg66.67)
  %slice.891 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.336), slice={[0:3], [0:3], [0:16], [0:16]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2371 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2339, f32[3,3,16,16]{3,2,1,0} %slice.891), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2"}
  %slice.2340 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2338), slice={[0:1], [0:16], [0:16], [16:32]}, metadata={op_type="Split" op_name="split_23"}
  %slice.892 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.336), slice={[0:3], [0:3], [0:16], [16:32]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2372 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2340, f32[3,3,16,16]{3,2,1,0} %slice.892), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_1"}
  %slice.2341 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2338), slice={[0:1], [0:16], [0:16], [32:48]}, metadata={op_type="Split" op_name="split_23"}
  %slice.893 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.336), slice={[0:3], [0:3], [0:16], [32:48]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2383 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2341, f32[3,3,16,16]{3,2,1,0} %slice.893), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_2"}
  %slice.2342 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2338), slice={[0:1], [0:16], [0:16], [48:64]}, metadata={op_type="Split" op_name="split_23"}
  %slice.894 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.336), slice={[0:3], [0:3], [0:16], [48:64]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2394 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2342, f32[3,3,16,16]{3,2,1,0} %slice.894), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_3"}
  %slice.2343 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2338), slice={[0:1], [0:16], [0:16], [64:80]}, metadata={op_type="Split" op_name="split_23"}
  %slice.895 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.336), slice={[0:3], [0:3], [0:16], [64:80]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2397 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2343, f32[3,3,16,16]{3,2,1,0} %slice.895), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_4"}
  %slice.2344 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2338), slice={[0:1], [0:16], [0:16], [80:96]}, metadata={op_type="Split" op_name="split_23"}
  %slice.896 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.336), slice={[0:3], [0:3], [0:16], [80:96]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2398 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2344, f32[3,3,16,16]{3,2,1,0} %slice.896), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_5"}
  %slice.2345 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2338), slice={[0:1], [0:16], [0:16], [96:112]}, metadata={op_type="Split" op_name="split_23"}
  %slice.897 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.336), slice={[0:3], [0:3], [0:16], [96:112]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2399 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2345, f32[3,3,16,16]{3,2,1,0} %slice.897), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_6"}
  %slice.2346 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2338), slice={[0:1], [0:16], [0:16], [112:128]}, metadata={op_type="Split" op_name="split_23"}
  %slice.898 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.336), slice={[0:3], [0:3], [0:16], [112:128]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2400 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2346, f32[3,3,16,16]{3,2,1,0} %slice.898), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_7"}
  %slice.2347 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2338), slice={[0:1], [0:16], [0:16], [128:144]}, metadata={op_type="Split" op_name="split_23"}
  %slice.899 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.336), slice={[0:3], [0:3], [0:16], [128:144]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2401 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2347, f32[3,3,16,16]{3,2,1,0} %slice.899), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_8"}
  %slice.2348 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2338), slice={[0:1], [0:16], [0:16], [144:160]}, metadata={op_type="Split" op_name="split_23"}
  %slice.900 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.336), slice={[0:3], [0:3], [0:16], [144:160]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2402 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2348, f32[3,3,16,16]{3,2,1,0} %slice.900), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_9"}
  %slice.2349 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2338), slice={[0:1], [0:16], [0:16], [160:176]}, metadata={op_type="Split" op_name="split_23"}
  %slice.901 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.336), slice={[0:3], [0:3], [0:16], [160:176]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2373 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2349, f32[3,3,16,16]{3,2,1,0} %slice.901), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_10"}
  %slice.2350 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2338), slice={[0:1], [0:16], [0:16], [176:192]}, metadata={op_type="Split" op_name="split_23"}
  %slice.902 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.336), slice={[0:3], [0:3], [0:16], [176:192]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2374 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2350, f32[3,3,16,16]{3,2,1,0} %slice.902), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_11"}
  %slice.2351 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2338), slice={[0:1], [0:16], [0:16], [192:208]}, metadata={op_type="Split" op_name="split_23"}
  %slice.903 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.336), slice={[0:3], [0:3], [0:16], [192:208]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2375 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2351, f32[3,3,16,16]{3,2,1,0} %slice.903), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_12"}
  %slice.2352 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2338), slice={[0:1], [0:16], [0:16], [208:224]}, metadata={op_type="Split" op_name="split_23"}
  %slice.904 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.336), slice={[0:3], [0:3], [0:16], [208:224]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2376 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2352, f32[3,3,16,16]{3,2,1,0} %slice.904), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_13"}
  %slice.2353 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2338), slice={[0:1], [0:16], [0:16], [224:240]}, metadata={op_type="Split" op_name="split_23"}
  %slice.905 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.336), slice={[0:3], [0:3], [0:16], [224:240]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2377 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2353, f32[3,3,16,16]{3,2,1,0} %slice.905), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_14"}
  %slice.2354 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2338), slice={[0:1], [0:16], [0:16], [240:256]}, metadata={op_type="Split" op_name="split_23"}
  %slice.906 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.336), slice={[0:3], [0:3], [0:16], [240:256]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2378 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2354, f32[3,3,16,16]{3,2,1,0} %slice.906), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_15"}
  %slice.2355 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2338), slice={[0:1], [0:16], [0:16], [256:272]}, metadata={op_type="Split" op_name="split_23"}
  %slice.907 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.336), slice={[0:3], [0:3], [0:16], [256:272]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2379 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2355, f32[3,3,16,16]{3,2,1,0} %slice.907), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_16"}
  %slice.2356 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2338), slice={[0:1], [0:16], [0:16], [272:288]}, metadata={op_type="Split" op_name="split_23"}
  %slice.908 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.336), slice={[0:3], [0:3], [0:16], [272:288]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2380 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2356, f32[3,3,16,16]{3,2,1,0} %slice.908), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_17"}
  %slice.2357 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2338), slice={[0:1], [0:16], [0:16], [288:304]}, metadata={op_type="Split" op_name="split_23"}
  %slice.909 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.336), slice={[0:3], [0:3], [0:16], [288:304]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2381 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2357, f32[3,3,16,16]{3,2,1,0} %slice.909), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_18"}
  %slice.2358 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2338), slice={[0:1], [0:16], [0:16], [304:320]}, metadata={op_type="Split" op_name="split_23"}
  %slice.910 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.336), slice={[0:3], [0:3], [0:16], [304:320]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2382 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2358, f32[3,3,16,16]{3,2,1,0} %slice.910), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_19"}
  %slice.2359 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2338), slice={[0:1], [0:16], [0:16], [320:336]}, metadata={op_type="Split" op_name="split_23"}
  %slice.911 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.336), slice={[0:3], [0:3], [0:16], [320:336]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2384 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2359, f32[3,3,16,16]{3,2,1,0} %slice.911), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_20"}
  %slice.2360 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2338), slice={[0:1], [0:16], [0:16], [336:352]}, metadata={op_type="Split" op_name="split_23"}
  %slice.912 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.336), slice={[0:3], [0:3], [0:16], [336:352]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2385 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2360, f32[3,3,16,16]{3,2,1,0} %slice.912), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_21"}
  %slice.2361 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2338), slice={[0:1], [0:16], [0:16], [352:368]}, metadata={op_type="Split" op_name="split_23"}
  %slice.913 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.336), slice={[0:3], [0:3], [0:16], [352:368]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2386 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2361, f32[3,3,16,16]{3,2,1,0} %slice.913), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_22"}
  %slice.2362 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2338), slice={[0:1], [0:16], [0:16], [368:384]}, metadata={op_type="Split" op_name="split_23"}
  %slice.914 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.336), slice={[0:3], [0:3], [0:16], [368:384]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2387 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2362, f32[3,3,16,16]{3,2,1,0} %slice.914), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_23"}
  %slice.2363 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2338), slice={[0:1], [0:16], [0:16], [384:400]}, metadata={op_type="Split" op_name="split_23"}
  %slice.915 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.336), slice={[0:3], [0:3], [0:16], [384:400]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2388 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2363, f32[3,3,16,16]{3,2,1,0} %slice.915), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_24"}
  %slice.2364 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2338), slice={[0:1], [0:16], [0:16], [400:416]}, metadata={op_type="Split" op_name="split_23"}
  %slice.916 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.336), slice={[0:3], [0:3], [0:16], [400:416]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2389 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2364, f32[3,3,16,16]{3,2,1,0} %slice.916), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_25"}
  %slice.2365 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2338), slice={[0:1], [0:16], [0:16], [416:432]}, metadata={op_type="Split" op_name="split_23"}
  %slice.917 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.336), slice={[0:3], [0:3], [0:16], [416:432]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2390 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2365, f32[3,3,16,16]{3,2,1,0} %slice.917), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_26"}
  %slice.2366 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2338), slice={[0:1], [0:16], [0:16], [432:448]}, metadata={op_type="Split" op_name="split_23"}
  %slice.918 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.336), slice={[0:3], [0:3], [0:16], [432:448]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2391 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2366, f32[3,3,16,16]{3,2,1,0} %slice.918), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_27"}
  %slice.2367 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2338), slice={[0:1], [0:16], [0:16], [448:464]}, metadata={op_type="Split" op_name="split_23"}
  %slice.919 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.336), slice={[0:3], [0:3], [0:16], [448:464]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2392 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2367, f32[3,3,16,16]{3,2,1,0} %slice.919), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_28"}
  %slice.2368 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2338), slice={[0:1], [0:16], [0:16], [464:480]}, metadata={op_type="Split" op_name="split_23"}
  %slice.920 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.336), slice={[0:3], [0:3], [0:16], [464:480]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2393 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2368, f32[3,3,16,16]{3,2,1,0} %slice.920), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_29"}
  %slice.2369 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2338), slice={[0:1], [0:16], [0:16], [480:496]}, metadata={op_type="Split" op_name="split_23"}
  %slice.921 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.336), slice={[0:3], [0:3], [0:16], [480:496]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2395 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2369, f32[3,3,16,16]{3,2,1,0} %slice.921), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_30"}
  %slice.2370 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2338), slice={[0:1], [0:16], [0:16], [496:512]}, metadata={op_type="Split" op_name="split_23"}
  %slice.922 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.336), slice={[0:3], [0:3], [0:16], [496:512]}, metadata={op_type="Split" op_name="split_22"}
  %convolution.2396 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2370, f32[3,3,16,16]{3,2,1,0} %slice.922), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv2_31"}
  %concatenate.2403 = f32[1,14,14,512]{3,2,1,0} concatenate(f32[1,14,14,16]{3,2,1,0} %convolution.2371, f32[1,14,14,16]{3,2,1,0} %convolution.2372, f32[1,14,14,16]{3,2,1,0} %convolution.2383, f32[1,14,14,16]{3,2,1,0} %convolution.2394, f32[1,14,14,16]{3,2,1,0} %convolution.2397, f32[1,14,14,16]{3,2,1,0} %convolution.2398, f32[1,14,14,16]{3,2,1,0} %convolution.2399, f32[1,14,14,16]{3,2,1,0} %convolution.2400, f32[1,14,14,16]{3,2,1,0} %convolution.2401, f32[1,14,14,16]{3,2,1,0} %convolution.2402, f32[1,14,14,16]{3,2,1,0} %convolution.2373, f32[1,14,14,16]{3,2,1,0} %convolution.2374, f32[1,14,14,16]{3,2,1,0} %convolution.2375, f32[1,14,14,16]{3,2,1,0} %convolution.2376, f32[1,14,14,16]{3,2,1,0} %convolution.2377, f32[1,14,14,16]{3,2,1,0} %convolution.2378, f32[1,14,14,16]{3,2,1,0} %convolution.2379, f32[1,14,14,16]{3,2,1,0} %convolution.2380, f32[1,14,14,16]{3,2,1,0} %convolution.2381, f32[1,14,14,16]{3,2,1,0} %convolution.2382, f32[1,14,14,16]{3,2,1,0} %convolution.2384, f32[1,14,14,16]{3,2,1,0} %convolution.2385, f32[1,14,14,16]{3,2,1,0} %convolution.2386, f32[1,14,14,16]{3,2,1,0} %convolution.2387, f32[1,14,14,16]{3,2,1,0} %convolution.2388, f32[1,14,14,16]{3,2,1,0} %convolution.2389, f32[1,14,14,16]{3,2,1,0} %convolution.2390, f32[1,14,14,16]{3,2,1,0} %convolution.2391, f32[1,14,14,16]{3,2,1,0} %convolution.2392, f32[1,14,14,16]{3,2,1,0} %convolution.2393, f32[1,14,14,16]{3,2,1,0} %convolution.2395, f32[1,14,14,16]{3,2,1,0} %convolution.2396), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_11"}
  %arg67.68 = f32[512]{0} parameter(67), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.337 = f32[512]{0} reshape(f32[512]{0} %arg67.68)
  %constant.2404 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit5_bn2/add"}
  %broadcast.2405 = f32[512]{0} broadcast(f32[] %constant.2404), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit5_bn2/add"}
  %add.2406 = f32[512]{0} add(f32[512]{0} %reshape.337, f32[512]{0} %broadcast.2405), metadata={op_type="AddV2" op_name="stage3_unit5_bn2/add"}
  %rsqrt.2407 = f32[512]{0} rsqrt(f32[512]{0} %add.2406), metadata={op_type="Rsqrt" op_name="stage3_unit5_bn2/Rsqrt"}
  %arg121.122 = f32[512]{0} parameter(121), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.391 = f32[512]{0} reshape(f32[512]{0} %arg121.122)
  %multiply.2408 = f32[512]{0} multiply(f32[512]{0} %rsqrt.2407, f32[512]{0} %reshape.391), metadata={op_type="Mul" op_name="stage3_unit5_bn2/mul"}
  %broadcast.2409 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.2408), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit5_bn2/mul_1"}
  %multiply.2410 = f32[1,14,14,512]{3,2,1,0} multiply(f32[1,14,14,512]{3,2,1,0} %concatenate.2403, f32[1,14,14,512]{3,2,1,0} %broadcast.2409), metadata={op_type="Mul" op_name="stage3_unit5_bn2/mul_1"}
  %arg228.229 = f32[512]{0} parameter(228), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.498 = f32[512]{0} reshape(f32[512]{0} %arg228.229)
  %arg175.176 = f32[512]{0} parameter(175), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.445 = f32[512]{0} reshape(f32[512]{0} %arg175.176)
  %multiply.2411 = f32[512]{0} multiply(f32[512]{0} %multiply.2408, f32[512]{0} %reshape.445), metadata={op_type="Mul" op_name="stage3_unit5_bn2/mul_2"}
  %subtract.2412 = f32[512]{0} subtract(f32[512]{0} %reshape.498, f32[512]{0} %multiply.2411), metadata={op_type="Sub" op_name="stage3_unit5_bn2/sub"}
  %broadcast.2413 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.2412), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit5_bn2/add_1"}
  %add.2414 = f32[1,14,14,512]{3,2,1,0} add(f32[1,14,14,512]{3,2,1,0} %multiply.2410, f32[1,14,14,512]{3,2,1,0} %broadcast.2413), metadata={op_type="AddV2" op_name="stage3_unit5_bn2/add_1"}
  %maximum.2417 = f32[1,14,14,512]{3,2,1,0} maximum(f32[1,14,14,512]{3,2,1,0} %broadcast.2416, f32[1,14,14,512]{3,2,1,0} %add.2414), metadata={op_type="Relu" op_name="stage3_unit5_relu2"}
  %arg258.259 = f32[1,1,512,1024]{3,2,1,0} parameter(258), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.528 = f32[1,1,512,1024]{3,2,1,0} reshape(f32[1,1,512,1024]{3,2,1,0} %arg258.259)
  %convolution.2418 = f32[1,14,14,1024]{3,2,1,0} convolution(f32[1,14,14,512]{3,2,1,0} %maximum.2417, f32[1,1,512,1024]{3,2,1,0} %reshape.528), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit5_conv3"}
  %multiply.2425 = f32[1,14,14,1024]{3,2,1,0} multiply(f32[1,14,14,1024]{3,2,1,0} %broadcast.2424, f32[1,14,14,1024]{3,2,1,0} %convolution.2418), metadata={op_type="Mul" op_name="stage3_unit5_bn3/mul_1"}
  %arg229.230 = f32[1024]{0} parameter(229), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.499 = f32[1024]{0} reshape(f32[1024]{0} %arg229.230)
  %arg176.177 = f32[1024]{0} parameter(176), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.446 = f32[1024]{0} reshape(f32[1024]{0} %arg176.177)
  %multiply.2426 = f32[1024]{0} multiply(f32[1024]{0} %multiply.2423, f32[1024]{0} %reshape.446), metadata={op_type="Mul" op_name="stage3_unit5_bn3/mul_2"}
  %subtract.2427 = f32[1024]{0} subtract(f32[1024]{0} %reshape.499, f32[1024]{0} %multiply.2426), metadata={op_type="Sub" op_name="stage3_unit5_bn3/sub"}
  %broadcast.2428 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %subtract.2427), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit5_bn3/add_1"}
  %add.2429 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %multiply.2425, f32[1,14,14,1024]{3,2,1,0} %broadcast.2428), metadata={op_type="AddV2" op_name="stage3_unit5_bn3/add_1"}
  %add.2430 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %maximum.2321, f32[1,14,14,1024]{3,2,1,0} %add.2429), metadata={op_type="AddV2" op_name="add_11"}
  %maximum.2433 = f32[1,14,14,1024]{3,2,1,0} maximum(f32[1,14,14,1024]{3,2,1,0} %broadcast.2432, f32[1,14,14,1024]{3,2,1,0} %add.2430), metadata={op_type="Relu" op_name="stage3_unit5_relu"}
  %arg17.18 = f32[1024]{0} parameter(17), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.287 = f32[1024]{0} reshape(f32[1024]{0} %arg17.18)
  %constant.2531 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit6_bn3/add"}
  %broadcast.2532 = f32[1024]{0} broadcast(f32[] %constant.2531), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit6_bn3/add"}
  %add.2533 = f32[1024]{0} add(f32[1024]{0} %reshape.287, f32[1024]{0} %broadcast.2532), metadata={op_type="AddV2" op_name="stage3_unit6_bn3/add"}
  %rsqrt.2534 = f32[1024]{0} rsqrt(f32[1024]{0} %add.2533), metadata={op_type="Rsqrt" op_name="stage3_unit6_bn3/Rsqrt"}
  %arg82.83 = f32[1024]{0} parameter(82), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.352 = f32[1024]{0} reshape(f32[1024]{0} %arg82.83)
  %multiply.2535 = f32[1024]{0} multiply(f32[1024]{0} %rsqrt.2534, f32[1024]{0} %reshape.352), metadata={op_type="Mul" op_name="stage3_unit6_bn3/mul"}
  %broadcast.2536 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %multiply.2535), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit6_bn3/mul_1"}
  %constant.2527 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit6_relu2"}
  %broadcast.2528 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[] %constant.2527), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit6_relu2"}
  %constant.2446 = f32[] constant(0), metadata={op_type="Relu" op_name="stage3_unit6_relu1"}
  %broadcast.2447 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[] %constant.2446), dimensions={}, metadata={op_type="Relu" op_name="stage3_unit6_relu1"}
  %arg1.2 = f32[512]{0} parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.271 = f32[512]{0} reshape(f32[512]{0} %arg1.2)
  %constant.2435 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit6_bn1/add"}
  %broadcast.2436 = f32[512]{0} broadcast(f32[] %constant.2435), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit6_bn1/add"}
  %add.2437 = f32[512]{0} add(f32[512]{0} %reshape.271, f32[512]{0} %broadcast.2436), metadata={op_type="AddV2" op_name="stage3_unit6_bn1/add"}
  %rsqrt.2438 = f32[512]{0} rsqrt(f32[512]{0} %add.2437), metadata={op_type="Rsqrt" op_name="stage3_unit6_bn1/Rsqrt"}
  %arg71.72 = f32[512]{0} parameter(71), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.341 = f32[512]{0} reshape(f32[512]{0} %arg71.72)
  %multiply.2439 = f32[512]{0} multiply(f32[512]{0} %rsqrt.2438, f32[512]{0} %reshape.341), metadata={op_type="Mul" op_name="stage3_unit6_bn1/mul"}
  %broadcast.2440 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.2439), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit6_bn1/mul_1"}
  %arg259.260 = f32[1,1,1024,512]{3,2,1,0} parameter(259), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.529 = f32[1,1,1024,512]{3,2,1,0} reshape(f32[1,1,1024,512]{3,2,1,0} %arg259.260)
  %convolution.2434 = f32[1,14,14,512]{3,2,1,0} convolution(f32[1,14,14,1024]{3,2,1,0} %maximum.2433, f32[1,1,1024,512]{3,2,1,0} %reshape.529), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv1"}
  %multiply.2441 = f32[1,14,14,512]{3,2,1,0} multiply(f32[1,14,14,512]{3,2,1,0} %broadcast.2440, f32[1,14,14,512]{3,2,1,0} %convolution.2434), metadata={op_type="Mul" op_name="stage3_unit6_bn1/mul_1"}
  %arg178.179 = f32[512]{0} parameter(178), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.448 = f32[512]{0} reshape(f32[512]{0} %arg178.179)
  %arg125.126 = f32[512]{0} parameter(125), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.395 = f32[512]{0} reshape(f32[512]{0} %arg125.126)
  %multiply.2442 = f32[512]{0} multiply(f32[512]{0} %multiply.2439, f32[512]{0} %reshape.395), metadata={op_type="Mul" op_name="stage3_unit6_bn1/mul_2"}
  %subtract.2443 = f32[512]{0} subtract(f32[512]{0} %reshape.448, f32[512]{0} %multiply.2442), metadata={op_type="Sub" op_name="stage3_unit6_bn1/sub"}
  %broadcast.2444 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.2443), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit6_bn1/add_1"}
  %add.2445 = f32[1,14,14,512]{3,2,1,0} add(f32[1,14,14,512]{3,2,1,0} %multiply.2441, f32[1,14,14,512]{3,2,1,0} %broadcast.2444), metadata={op_type="AddV2" op_name="stage3_unit6_bn1/add_1"}
  %maximum.2448 = f32[1,14,14,512]{3,2,1,0} maximum(f32[1,14,14,512]{3,2,1,0} %broadcast.2447, f32[1,14,14,512]{3,2,1,0} %add.2445), metadata={op_type="Relu" op_name="stage3_unit6_relu1"}
  %constant.2449 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_13"}
  %pad.2450 = f32[1,16,16,512]{3,2,1,0} pad(f32[1,14,14,512]{3,2,1,0} %maximum.2448, f32[] %constant.2449), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_13"}
  %slice.2451 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2450), slice={[0:1], [0:16], [0:16], [0:16]}, metadata={op_type="Split" op_name="split_25"}
  %arg4.5 = f32[3,3,16,512]{3,2,1,0} parameter(4), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.274 = f32[3,3,16,512]{3,2,1,0} reshape(f32[3,3,16,512]{3,2,1,0} %arg4.5)
  %slice.923 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.274), slice={[0:3], [0:3], [0:16], [0:16]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2483 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2451, f32[3,3,16,16]{3,2,1,0} %slice.923), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2"}
  %slice.2452 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2450), slice={[0:1], [0:16], [0:16], [16:32]}, metadata={op_type="Split" op_name="split_25"}
  %slice.924 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.274), slice={[0:3], [0:3], [0:16], [16:32]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2484 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2452, f32[3,3,16,16]{3,2,1,0} %slice.924), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_1"}
  %slice.2453 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2450), slice={[0:1], [0:16], [0:16], [32:48]}, metadata={op_type="Split" op_name="split_25"}
  %slice.925 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.274), slice={[0:3], [0:3], [0:16], [32:48]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2495 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2453, f32[3,3,16,16]{3,2,1,0} %slice.925), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_2"}
  %slice.2454 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2450), slice={[0:1], [0:16], [0:16], [48:64]}, metadata={op_type="Split" op_name="split_25"}
  %slice.926 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.274), slice={[0:3], [0:3], [0:16], [48:64]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2506 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2454, f32[3,3,16,16]{3,2,1,0} %slice.926), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_3"}
  %slice.2455 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2450), slice={[0:1], [0:16], [0:16], [64:80]}, metadata={op_type="Split" op_name="split_25"}
  %slice.927 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.274), slice={[0:3], [0:3], [0:16], [64:80]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2509 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2455, f32[3,3,16,16]{3,2,1,0} %slice.927), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_4"}
  %slice.2456 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2450), slice={[0:1], [0:16], [0:16], [80:96]}, metadata={op_type="Split" op_name="split_25"}
  %slice.928 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.274), slice={[0:3], [0:3], [0:16], [80:96]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2510 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2456, f32[3,3,16,16]{3,2,1,0} %slice.928), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_5"}
  %slice.2457 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2450), slice={[0:1], [0:16], [0:16], [96:112]}, metadata={op_type="Split" op_name="split_25"}
  %slice.929 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.274), slice={[0:3], [0:3], [0:16], [96:112]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2511 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2457, f32[3,3,16,16]{3,2,1,0} %slice.929), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_6"}
  %slice.2458 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2450), slice={[0:1], [0:16], [0:16], [112:128]}, metadata={op_type="Split" op_name="split_25"}
  %slice.930 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.274), slice={[0:3], [0:3], [0:16], [112:128]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2512 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2458, f32[3,3,16,16]{3,2,1,0} %slice.930), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_7"}
  %slice.2459 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2450), slice={[0:1], [0:16], [0:16], [128:144]}, metadata={op_type="Split" op_name="split_25"}
  %slice.931 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.274), slice={[0:3], [0:3], [0:16], [128:144]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2513 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2459, f32[3,3,16,16]{3,2,1,0} %slice.931), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_8"}
  %slice.2460 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2450), slice={[0:1], [0:16], [0:16], [144:160]}, metadata={op_type="Split" op_name="split_25"}
  %slice.932 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.274), slice={[0:3], [0:3], [0:16], [144:160]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2514 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2460, f32[3,3,16,16]{3,2,1,0} %slice.932), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_9"}
  %slice.2461 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2450), slice={[0:1], [0:16], [0:16], [160:176]}, metadata={op_type="Split" op_name="split_25"}
  %slice.933 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.274), slice={[0:3], [0:3], [0:16], [160:176]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2485 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2461, f32[3,3,16,16]{3,2,1,0} %slice.933), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_10"}
  %slice.2462 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2450), slice={[0:1], [0:16], [0:16], [176:192]}, metadata={op_type="Split" op_name="split_25"}
  %slice.934 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.274), slice={[0:3], [0:3], [0:16], [176:192]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2486 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2462, f32[3,3,16,16]{3,2,1,0} %slice.934), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_11"}
  %slice.2463 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2450), slice={[0:1], [0:16], [0:16], [192:208]}, metadata={op_type="Split" op_name="split_25"}
  %slice.935 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.274), slice={[0:3], [0:3], [0:16], [192:208]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2487 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2463, f32[3,3,16,16]{3,2,1,0} %slice.935), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_12"}
  %slice.2464 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2450), slice={[0:1], [0:16], [0:16], [208:224]}, metadata={op_type="Split" op_name="split_25"}
  %slice.936 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.274), slice={[0:3], [0:3], [0:16], [208:224]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2488 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2464, f32[3,3,16,16]{3,2,1,0} %slice.936), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_13"}
  %slice.2465 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2450), slice={[0:1], [0:16], [0:16], [224:240]}, metadata={op_type="Split" op_name="split_25"}
  %slice.937 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.274), slice={[0:3], [0:3], [0:16], [224:240]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2489 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2465, f32[3,3,16,16]{3,2,1,0} %slice.937), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_14"}
  %slice.2466 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2450), slice={[0:1], [0:16], [0:16], [240:256]}, metadata={op_type="Split" op_name="split_25"}
  %slice.938 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.274), slice={[0:3], [0:3], [0:16], [240:256]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2490 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2466, f32[3,3,16,16]{3,2,1,0} %slice.938), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_15"}
  %slice.2467 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2450), slice={[0:1], [0:16], [0:16], [256:272]}, metadata={op_type="Split" op_name="split_25"}
  %slice.939 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.274), slice={[0:3], [0:3], [0:16], [256:272]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2491 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2467, f32[3,3,16,16]{3,2,1,0} %slice.939), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_16"}
  %slice.2468 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2450), slice={[0:1], [0:16], [0:16], [272:288]}, metadata={op_type="Split" op_name="split_25"}
  %slice.940 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.274), slice={[0:3], [0:3], [0:16], [272:288]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2492 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2468, f32[3,3,16,16]{3,2,1,0} %slice.940), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_17"}
  %slice.2469 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2450), slice={[0:1], [0:16], [0:16], [288:304]}, metadata={op_type="Split" op_name="split_25"}
  %slice.941 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.274), slice={[0:3], [0:3], [0:16], [288:304]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2493 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2469, f32[3,3,16,16]{3,2,1,0} %slice.941), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_18"}
  %slice.2470 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2450), slice={[0:1], [0:16], [0:16], [304:320]}, metadata={op_type="Split" op_name="split_25"}
  %slice.942 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.274), slice={[0:3], [0:3], [0:16], [304:320]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2494 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2470, f32[3,3,16,16]{3,2,1,0} %slice.942), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_19"}
  %slice.2471 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2450), slice={[0:1], [0:16], [0:16], [320:336]}, metadata={op_type="Split" op_name="split_25"}
  %slice.943 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.274), slice={[0:3], [0:3], [0:16], [320:336]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2496 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2471, f32[3,3,16,16]{3,2,1,0} %slice.943), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_20"}
  %slice.2472 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2450), slice={[0:1], [0:16], [0:16], [336:352]}, metadata={op_type="Split" op_name="split_25"}
  %slice.944 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.274), slice={[0:3], [0:3], [0:16], [336:352]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2497 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2472, f32[3,3,16,16]{3,2,1,0} %slice.944), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_21"}
  %slice.2473 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2450), slice={[0:1], [0:16], [0:16], [352:368]}, metadata={op_type="Split" op_name="split_25"}
  %slice.945 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.274), slice={[0:3], [0:3], [0:16], [352:368]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2498 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2473, f32[3,3,16,16]{3,2,1,0} %slice.945), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_22"}
  %slice.2474 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2450), slice={[0:1], [0:16], [0:16], [368:384]}, metadata={op_type="Split" op_name="split_25"}
  %slice.946 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.274), slice={[0:3], [0:3], [0:16], [368:384]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2499 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2474, f32[3,3,16,16]{3,2,1,0} %slice.946), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_23"}
  %slice.2475 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2450), slice={[0:1], [0:16], [0:16], [384:400]}, metadata={op_type="Split" op_name="split_25"}
  %slice.947 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.274), slice={[0:3], [0:3], [0:16], [384:400]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2500 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2475, f32[3,3,16,16]{3,2,1,0} %slice.947), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_24"}
  %slice.2476 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2450), slice={[0:1], [0:16], [0:16], [400:416]}, metadata={op_type="Split" op_name="split_25"}
  %slice.948 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.274), slice={[0:3], [0:3], [0:16], [400:416]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2501 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2476, f32[3,3,16,16]{3,2,1,0} %slice.948), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_25"}
  %slice.2477 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2450), slice={[0:1], [0:16], [0:16], [416:432]}, metadata={op_type="Split" op_name="split_25"}
  %slice.949 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.274), slice={[0:3], [0:3], [0:16], [416:432]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2502 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2477, f32[3,3,16,16]{3,2,1,0} %slice.949), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_26"}
  %slice.2478 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2450), slice={[0:1], [0:16], [0:16], [432:448]}, metadata={op_type="Split" op_name="split_25"}
  %slice.950 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.274), slice={[0:3], [0:3], [0:16], [432:448]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2503 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2478, f32[3,3,16,16]{3,2,1,0} %slice.950), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_27"}
  %slice.2479 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2450), slice={[0:1], [0:16], [0:16], [448:464]}, metadata={op_type="Split" op_name="split_25"}
  %slice.951 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.274), slice={[0:3], [0:3], [0:16], [448:464]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2504 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2479, f32[3,3,16,16]{3,2,1,0} %slice.951), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_28"}
  %slice.2480 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2450), slice={[0:1], [0:16], [0:16], [464:480]}, metadata={op_type="Split" op_name="split_25"}
  %slice.952 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.274), slice={[0:3], [0:3], [0:16], [464:480]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2505 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2480, f32[3,3,16,16]{3,2,1,0} %slice.952), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_29"}
  %slice.2481 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2450), slice={[0:1], [0:16], [0:16], [480:496]}, metadata={op_type="Split" op_name="split_25"}
  %slice.953 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.274), slice={[0:3], [0:3], [0:16], [480:496]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2507 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2481, f32[3,3,16,16]{3,2,1,0} %slice.953), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_30"}
  %slice.2482 = f32[1,16,16,16]{3,2,1,0} slice(f32[1,16,16,512]{3,2,1,0} %pad.2450), slice={[0:1], [0:16], [0:16], [496:512]}, metadata={op_type="Split" op_name="split_25"}
  %slice.954 = f32[3,3,16,16]{3,2,1,0} slice(f32[3,3,16,512]{3,2,1,0} %reshape.274), slice={[0:3], [0:3], [0:16], [496:512]}, metadata={op_type="Split" op_name="split_24"}
  %convolution.2508 = f32[1,14,14,16]{3,2,1,0} convolution(f32[1,16,16,16]{3,2,1,0} %slice.2482, f32[3,3,16,16]{3,2,1,0} %slice.954), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv2_31"}
  %concatenate.2515 = f32[1,14,14,512]{3,2,1,0} concatenate(f32[1,14,14,16]{3,2,1,0} %convolution.2483, f32[1,14,14,16]{3,2,1,0} %convolution.2484, f32[1,14,14,16]{3,2,1,0} %convolution.2495, f32[1,14,14,16]{3,2,1,0} %convolution.2506, f32[1,14,14,16]{3,2,1,0} %convolution.2509, f32[1,14,14,16]{3,2,1,0} %convolution.2510, f32[1,14,14,16]{3,2,1,0} %convolution.2511, f32[1,14,14,16]{3,2,1,0} %convolution.2512, f32[1,14,14,16]{3,2,1,0} %convolution.2513, f32[1,14,14,16]{3,2,1,0} %convolution.2514, f32[1,14,14,16]{3,2,1,0} %convolution.2485, f32[1,14,14,16]{3,2,1,0} %convolution.2486, f32[1,14,14,16]{3,2,1,0} %convolution.2487, f32[1,14,14,16]{3,2,1,0} %convolution.2488, f32[1,14,14,16]{3,2,1,0} %convolution.2489, f32[1,14,14,16]{3,2,1,0} %convolution.2490, f32[1,14,14,16]{3,2,1,0} %convolution.2491, f32[1,14,14,16]{3,2,1,0} %convolution.2492, f32[1,14,14,16]{3,2,1,0} %convolution.2493, f32[1,14,14,16]{3,2,1,0} %convolution.2494, f32[1,14,14,16]{3,2,1,0} %convolution.2496, f32[1,14,14,16]{3,2,1,0} %convolution.2497, f32[1,14,14,16]{3,2,1,0} %convolution.2498, f32[1,14,14,16]{3,2,1,0} %convolution.2499, f32[1,14,14,16]{3,2,1,0} %convolution.2500, f32[1,14,14,16]{3,2,1,0} %convolution.2501, f32[1,14,14,16]{3,2,1,0} %convolution.2502, f32[1,14,14,16]{3,2,1,0} %convolution.2503, f32[1,14,14,16]{3,2,1,0} %convolution.2504, f32[1,14,14,16]{3,2,1,0} %convolution.2505, f32[1,14,14,16]{3,2,1,0} %convolution.2507, f32[1,14,14,16]{3,2,1,0} %convolution.2508), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_12"}
  %arg12.13 = f32[512]{0} parameter(12), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.282 = f32[512]{0} reshape(f32[512]{0} %arg12.13)
  %constant.2516 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage3_unit6_bn2/add"}
  %broadcast.2517 = f32[512]{0} broadcast(f32[] %constant.2516), dimensions={}, metadata={op_type="AddV2" op_name="stage3_unit6_bn2/add"}
  %add.2518 = f32[512]{0} add(f32[512]{0} %reshape.282, f32[512]{0} %broadcast.2517), metadata={op_type="AddV2" op_name="stage3_unit6_bn2/add"}
  %rsqrt.2519 = f32[512]{0} rsqrt(f32[512]{0} %add.2518), metadata={op_type="Rsqrt" op_name="stage3_unit6_bn2/Rsqrt"}
  %arg79.80 = f32[512]{0} parameter(79), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.349 = f32[512]{0} reshape(f32[512]{0} %arg79.80)
  %multiply.2520 = f32[512]{0} multiply(f32[512]{0} %rsqrt.2519, f32[512]{0} %reshape.349), metadata={op_type="Mul" op_name="stage3_unit6_bn2/mul"}
  %broadcast.2521 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %multiply.2520), dimensions={3}, metadata={op_type="Mul" op_name="stage3_unit6_bn2/mul_1"}
  %multiply.2522 = f32[1,14,14,512]{3,2,1,0} multiply(f32[1,14,14,512]{3,2,1,0} %concatenate.2515, f32[1,14,14,512]{3,2,1,0} %broadcast.2521), metadata={op_type="Mul" op_name="stage3_unit6_bn2/mul_1"}
  %arg186.187 = f32[512]{0} parameter(186), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.456 = f32[512]{0} reshape(f32[512]{0} %arg186.187)
  %arg133.134 = f32[512]{0} parameter(133), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.403 = f32[512]{0} reshape(f32[512]{0} %arg133.134)
  %multiply.2523 = f32[512]{0} multiply(f32[512]{0} %multiply.2520, f32[512]{0} %reshape.403), metadata={op_type="Mul" op_name="stage3_unit6_bn2/mul_2"}
  %subtract.2524 = f32[512]{0} subtract(f32[512]{0} %reshape.456, f32[512]{0} %multiply.2523), metadata={op_type="Sub" op_name="stage3_unit6_bn2/sub"}
  %broadcast.2525 = f32[1,14,14,512]{3,2,1,0} broadcast(f32[512]{0} %subtract.2524), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit6_bn2/add_1"}
  %add.2526 = f32[1,14,14,512]{3,2,1,0} add(f32[1,14,14,512]{3,2,1,0} %multiply.2522, f32[1,14,14,512]{3,2,1,0} %broadcast.2525), metadata={op_type="AddV2" op_name="stage3_unit6_bn2/add_1"}
  %maximum.2529 = f32[1,14,14,512]{3,2,1,0} maximum(f32[1,14,14,512]{3,2,1,0} %broadcast.2528, f32[1,14,14,512]{3,2,1,0} %add.2526), metadata={op_type="Relu" op_name="stage3_unit6_relu2"}
  %arg260.261 = f32[1,1,512,1024]{3,2,1,0} parameter(260), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.530 = f32[1,1,512,1024]{3,2,1,0} reshape(f32[1,1,512,1024]{3,2,1,0} %arg260.261)
  %convolution.2530 = f32[1,14,14,1024]{3,2,1,0} convolution(f32[1,14,14,512]{3,2,1,0} %maximum.2529, f32[1,1,512,1024]{3,2,1,0} %reshape.530), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage3_unit6_conv3"}
  %multiply.2537 = f32[1,14,14,1024]{3,2,1,0} multiply(f32[1,14,14,1024]{3,2,1,0} %broadcast.2536, f32[1,14,14,1024]{3,2,1,0} %convolution.2530), metadata={op_type="Mul" op_name="stage3_unit6_bn3/mul_1"}
  %arg189.190 = f32[1024]{0} parameter(189), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.459 = f32[1024]{0} reshape(f32[1024]{0} %arg189.190)
  %arg136.137 = f32[1024]{0} parameter(136), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.406 = f32[1024]{0} reshape(f32[1024]{0} %arg136.137)
  %multiply.2538 = f32[1024]{0} multiply(f32[1024]{0} %multiply.2535, f32[1024]{0} %reshape.406), metadata={op_type="Mul" op_name="stage3_unit6_bn3/mul_2"}
  %subtract.2539 = f32[1024]{0} subtract(f32[1024]{0} %reshape.459, f32[1024]{0} %multiply.2538), metadata={op_type="Sub" op_name="stage3_unit6_bn3/sub"}
  %broadcast.2540 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %subtract.2539), dimensions={3}, metadata={op_type="AddV2" op_name="stage3_unit6_bn3/add_1"}
  %add.2541 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %multiply.2537, f32[1,14,14,1024]{3,2,1,0} %broadcast.2540), metadata={op_type="AddV2" op_name="stage3_unit6_bn3/add_1"}
  %add.2542 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %maximum.2433, f32[1,14,14,1024]{3,2,1,0} %add.2541), metadata={op_type="AddV2" op_name="add_12"}
  %maximum.2545 = f32[1,14,14,1024]{3,2,1,0} maximum(f32[1,14,14,1024]{3,2,1,0} %broadcast.2544, f32[1,14,14,1024]{3,2,1,0} %add.2542), metadata={op_type="Relu" op_name="stage3_unit6_relu"}
  %arg261.262 = f32[1,1,1024,1024]{3,2,1,0} parameter(261), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.531 = f32[1,1,1024,1024]{3,2,1,0} reshape(f32[1,1,1024,1024]{3,2,1,0} %arg261.262)
  %convolution.2546 = f32[1,14,14,1024]{3,2,1,0} convolution(f32[1,14,14,1024]{3,2,1,0} %maximum.2545, f32[1,1,1024,1024]{3,2,1,0} %reshape.531), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv1"}
  %multiply.2554 = f32[1,14,14,1024]{3,2,1,0} multiply(f32[1,14,14,1024]{3,2,1,0} %broadcast.2553, f32[1,14,14,1024]{3,2,1,0} %convolution.2546), metadata={op_type="Mul" op_name="stage4_unit1_bn1/mul_1"}
  %arg194.195 = f32[1024]{0} parameter(194), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.464 = f32[1024]{0} reshape(f32[1024]{0} %arg194.195)
  %arg141.142 = f32[1024]{0} parameter(141), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.411 = f32[1024]{0} reshape(f32[1024]{0} %arg141.142)
  %multiply.2555 = f32[1024]{0} multiply(f32[1024]{0} %multiply.2552, f32[1024]{0} %reshape.411), metadata={op_type="Mul" op_name="stage4_unit1_bn1/mul_2"}
  %subtract.2556 = f32[1024]{0} subtract(f32[1024]{0} %reshape.464, f32[1024]{0} %multiply.2555), metadata={op_type="Sub" op_name="stage4_unit1_bn1/sub"}
  %broadcast.2557 = f32[1,14,14,1024]{3,2,1,0} broadcast(f32[1024]{0} %subtract.2556), dimensions={3}, metadata={op_type="AddV2" op_name="stage4_unit1_bn1/add_1"}
  %add.2558 = f32[1,14,14,1024]{3,2,1,0} add(f32[1,14,14,1024]{3,2,1,0} %multiply.2554, f32[1,14,14,1024]{3,2,1,0} %broadcast.2557), metadata={op_type="AddV2" op_name="stage4_unit1_bn1/add_1"}
  %maximum.2561 = f32[1,14,14,1024]{3,2,1,0} maximum(f32[1,14,14,1024]{3,2,1,0} %broadcast.2560, f32[1,14,14,1024]{3,2,1,0} %add.2558), metadata={op_type="Relu" op_name="stage4_unit1_relu1"}
  %constant.2562 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_14"}
  %pad.2563 = f32[1,16,16,1024]{3,2,1,0} pad(f32[1,14,14,1024]{3,2,1,0} %maximum.2561, f32[] %constant.2562), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_14"}
  %slice.2564 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2563), slice={[0:1], [0:16], [0:16], [0:32]}, metadata={op_type="Split" op_name="split_27"}
  %arg30.31 = f32[3,3,32,1024]{3,2,1,0} parameter(30), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.300 = f32[3,3,32,1024]{3,2,1,0} reshape(f32[3,3,32,1024]{3,2,1,0} %arg30.31)
  %slice.955 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:32], [0:32]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2596 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2564, f32[3,3,32,32]{3,2,1,0} %slice.955), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2"}
  %slice.2565 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2563), slice={[0:1], [0:16], [0:16], [32:64]}, metadata={op_type="Split" op_name="split_27"}
  %slice.956 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:32], [32:64]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2597 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2565, f32[3,3,32,32]{3,2,1,0} %slice.956), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_1"}
  %slice.2566 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2563), slice={[0:1], [0:16], [0:16], [64:96]}, metadata={op_type="Split" op_name="split_27"}
  %slice.957 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:32], [64:96]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2608 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2566, f32[3,3,32,32]{3,2,1,0} %slice.957), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_2"}
  %slice.2567 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2563), slice={[0:1], [0:16], [0:16], [96:128]}, metadata={op_type="Split" op_name="split_27"}
  %slice.958 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:32], [96:128]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2619 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2567, f32[3,3,32,32]{3,2,1,0} %slice.958), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_3"}
  %slice.2568 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2563), slice={[0:1], [0:16], [0:16], [128:160]}, metadata={op_type="Split" op_name="split_27"}
  %slice.959 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:32], [128:160]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2622 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2568, f32[3,3,32,32]{3,2,1,0} %slice.959), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_4"}
  %slice.2569 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2563), slice={[0:1], [0:16], [0:16], [160:192]}, metadata={op_type="Split" op_name="split_27"}
  %slice.960 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:32], [160:192]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2623 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2569, f32[3,3,32,32]{3,2,1,0} %slice.960), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_5"}
  %slice.2570 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2563), slice={[0:1], [0:16], [0:16], [192:224]}, metadata={op_type="Split" op_name="split_27"}
  %slice.961 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:32], [192:224]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2624 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2570, f32[3,3,32,32]{3,2,1,0} %slice.961), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_6"}
  %slice.2571 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2563), slice={[0:1], [0:16], [0:16], [224:256]}, metadata={op_type="Split" op_name="split_27"}
  %slice.962 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:32], [224:256]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2625 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2571, f32[3,3,32,32]{3,2,1,0} %slice.962), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_7"}
  %slice.2572 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2563), slice={[0:1], [0:16], [0:16], [256:288]}, metadata={op_type="Split" op_name="split_27"}
  %slice.963 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:32], [256:288]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2626 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2572, f32[3,3,32,32]{3,2,1,0} %slice.963), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_8"}
  %slice.2573 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2563), slice={[0:1], [0:16], [0:16], [288:320]}, metadata={op_type="Split" op_name="split_27"}
  %slice.964 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:32], [288:320]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2627 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2573, f32[3,3,32,32]{3,2,1,0} %slice.964), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_9"}
  %slice.2574 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2563), slice={[0:1], [0:16], [0:16], [320:352]}, metadata={op_type="Split" op_name="split_27"}
  %slice.965 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:32], [320:352]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2598 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2574, f32[3,3,32,32]{3,2,1,0} %slice.965), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_10"}
  %slice.2575 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2563), slice={[0:1], [0:16], [0:16], [352:384]}, metadata={op_type="Split" op_name="split_27"}
  %slice.966 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:32], [352:384]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2599 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2575, f32[3,3,32,32]{3,2,1,0} %slice.966), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_11"}
  %slice.2576 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2563), slice={[0:1], [0:16], [0:16], [384:416]}, metadata={op_type="Split" op_name="split_27"}
  %slice.967 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:32], [384:416]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2600 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2576, f32[3,3,32,32]{3,2,1,0} %slice.967), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_12"}
  %slice.2577 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2563), slice={[0:1], [0:16], [0:16], [416:448]}, metadata={op_type="Split" op_name="split_27"}
  %slice.968 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:32], [416:448]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2601 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2577, f32[3,3,32,32]{3,2,1,0} %slice.968), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_13"}
  %slice.2578 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2563), slice={[0:1], [0:16], [0:16], [448:480]}, metadata={op_type="Split" op_name="split_27"}
  %slice.969 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:32], [448:480]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2602 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2578, f32[3,3,32,32]{3,2,1,0} %slice.969), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_14"}
  %slice.2579 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2563), slice={[0:1], [0:16], [0:16], [480:512]}, metadata={op_type="Split" op_name="split_27"}
  %slice.970 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:32], [480:512]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2603 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2579, f32[3,3,32,32]{3,2,1,0} %slice.970), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_15"}
  %slice.2580 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2563), slice={[0:1], [0:16], [0:16], [512:544]}, metadata={op_type="Split" op_name="split_27"}
  %slice.971 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:32], [512:544]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2604 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2580, f32[3,3,32,32]{3,2,1,0} %slice.971), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_16"}
  %slice.2581 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2563), slice={[0:1], [0:16], [0:16], [544:576]}, metadata={op_type="Split" op_name="split_27"}
  %slice.972 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:32], [544:576]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2605 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2581, f32[3,3,32,32]{3,2,1,0} %slice.972), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_17"}
  %slice.2582 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2563), slice={[0:1], [0:16], [0:16], [576:608]}, metadata={op_type="Split" op_name="split_27"}
  %slice.973 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:32], [576:608]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2606 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2582, f32[3,3,32,32]{3,2,1,0} %slice.973), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_18"}
  %slice.2583 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2563), slice={[0:1], [0:16], [0:16], [608:640]}, metadata={op_type="Split" op_name="split_27"}
  %slice.974 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:32], [608:640]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2607 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2583, f32[3,3,32,32]{3,2,1,0} %slice.974), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_19"}
  %slice.2584 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2563), slice={[0:1], [0:16], [0:16], [640:672]}, metadata={op_type="Split" op_name="split_27"}
  %slice.975 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:32], [640:672]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2609 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2584, f32[3,3,32,32]{3,2,1,0} %slice.975), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_20"}
  %slice.2585 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2563), slice={[0:1], [0:16], [0:16], [672:704]}, metadata={op_type="Split" op_name="split_27"}
  %slice.976 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:32], [672:704]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2610 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2585, f32[3,3,32,32]{3,2,1,0} %slice.976), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_21"}
  %slice.2586 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2563), slice={[0:1], [0:16], [0:16], [704:736]}, metadata={op_type="Split" op_name="split_27"}
  %slice.977 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:32], [704:736]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2611 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2586, f32[3,3,32,32]{3,2,1,0} %slice.977), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_22"}
  %slice.2587 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2563), slice={[0:1], [0:16], [0:16], [736:768]}, metadata={op_type="Split" op_name="split_27"}
  %slice.978 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:32], [736:768]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2612 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2587, f32[3,3,32,32]{3,2,1,0} %slice.978), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_23"}
  %slice.2588 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2563), slice={[0:1], [0:16], [0:16], [768:800]}, metadata={op_type="Split" op_name="split_27"}
  %slice.979 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:32], [768:800]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2613 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2588, f32[3,3,32,32]{3,2,1,0} %slice.979), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_24"}
  %slice.2589 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2563), slice={[0:1], [0:16], [0:16], [800:832]}, metadata={op_type="Split" op_name="split_27"}
  %slice.980 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:32], [800:832]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2614 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2589, f32[3,3,32,32]{3,2,1,0} %slice.980), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_25"}
  %slice.2590 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2563), slice={[0:1], [0:16], [0:16], [832:864]}, metadata={op_type="Split" op_name="split_27"}
  %slice.981 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:32], [832:864]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2615 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2590, f32[3,3,32,32]{3,2,1,0} %slice.981), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_26"}
  %slice.2591 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2563), slice={[0:1], [0:16], [0:16], [864:896]}, metadata={op_type="Split" op_name="split_27"}
  %slice.982 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:32], [864:896]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2616 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2591, f32[3,3,32,32]{3,2,1,0} %slice.982), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_27"}
  %slice.2592 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2563), slice={[0:1], [0:16], [0:16], [896:928]}, metadata={op_type="Split" op_name="split_27"}
  %slice.983 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:32], [896:928]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2617 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2592, f32[3,3,32,32]{3,2,1,0} %slice.983), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_28"}
  %slice.2593 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2563), slice={[0:1], [0:16], [0:16], [928:960]}, metadata={op_type="Split" op_name="split_27"}
  %slice.984 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:32], [928:960]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2618 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2593, f32[3,3,32,32]{3,2,1,0} %slice.984), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_29"}
  %slice.2594 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2563), slice={[0:1], [0:16], [0:16], [960:992]}, metadata={op_type="Split" op_name="split_27"}
  %slice.985 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:32], [960:992]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2620 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2594, f32[3,3,32,32]{3,2,1,0} %slice.985), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_30"}
  %slice.2595 = f32[1,16,16,32]{3,2,1,0} slice(f32[1,16,16,1024]{3,2,1,0} %pad.2563), slice={[0:1], [0:16], [0:16], [992:1024]}, metadata={op_type="Split" op_name="split_27"}
  %slice.986 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.300), slice={[0:3], [0:3], [0:32], [992:1024]}, metadata={op_type="Split" op_name="split_26"}
  %convolution.2621 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,16,16,32]{3,2,1,0} %slice.2595, f32[3,3,32,32]{3,2,1,0} %slice.986), window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv2_31"}
  %concatenate.2628 = f32[1,7,7,1024]{3,2,1,0} concatenate(f32[1,7,7,32]{3,2,1,0} %convolution.2596, f32[1,7,7,32]{3,2,1,0} %convolution.2597, f32[1,7,7,32]{3,2,1,0} %convolution.2608, f32[1,7,7,32]{3,2,1,0} %convolution.2619, f32[1,7,7,32]{3,2,1,0} %convolution.2622, f32[1,7,7,32]{3,2,1,0} %convolution.2623, f32[1,7,7,32]{3,2,1,0} %convolution.2624, f32[1,7,7,32]{3,2,1,0} %convolution.2625, f32[1,7,7,32]{3,2,1,0} %convolution.2626, f32[1,7,7,32]{3,2,1,0} %convolution.2627, f32[1,7,7,32]{3,2,1,0} %convolution.2598, f32[1,7,7,32]{3,2,1,0} %convolution.2599, f32[1,7,7,32]{3,2,1,0} %convolution.2600, f32[1,7,7,32]{3,2,1,0} %convolution.2601, f32[1,7,7,32]{3,2,1,0} %convolution.2602, f32[1,7,7,32]{3,2,1,0} %convolution.2603, f32[1,7,7,32]{3,2,1,0} %convolution.2604, f32[1,7,7,32]{3,2,1,0} %convolution.2605, f32[1,7,7,32]{3,2,1,0} %convolution.2606, f32[1,7,7,32]{3,2,1,0} %convolution.2607, f32[1,7,7,32]{3,2,1,0} %convolution.2609, f32[1,7,7,32]{3,2,1,0} %convolution.2610, f32[1,7,7,32]{3,2,1,0} %convolution.2611, f32[1,7,7,32]{3,2,1,0} %convolution.2612, f32[1,7,7,32]{3,2,1,0} %convolution.2613, f32[1,7,7,32]{3,2,1,0} %convolution.2614, f32[1,7,7,32]{3,2,1,0} %convolution.2615, f32[1,7,7,32]{3,2,1,0} %convolution.2616, f32[1,7,7,32]{3,2,1,0} %convolution.2617, f32[1,7,7,32]{3,2,1,0} %convolution.2618, f32[1,7,7,32]{3,2,1,0} %convolution.2620, f32[1,7,7,32]{3,2,1,0} %convolution.2621), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_13"}
  %arg37.38 = f32[1024]{0} parameter(37), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.307 = f32[1024]{0} reshape(f32[1024]{0} %arg37.38)
  %constant.2629 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage4_unit1_bn2/add"}
  %broadcast.2630 = f32[1024]{0} broadcast(f32[] %constant.2629), dimensions={}, metadata={op_type="AddV2" op_name="stage4_unit1_bn2/add"}
  %add.2631 = f32[1024]{0} add(f32[1024]{0} %reshape.307, f32[1024]{0} %broadcast.2630), metadata={op_type="AddV2" op_name="stage4_unit1_bn2/add"}
  %rsqrt.2632 = f32[1024]{0} rsqrt(f32[1024]{0} %add.2631), metadata={op_type="Rsqrt" op_name="stage4_unit1_bn2/Rsqrt"}
  %arg98.99 = f32[1024]{0} parameter(98), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.368 = f32[1024]{0} reshape(f32[1024]{0} %arg98.99)
  %multiply.2633 = f32[1024]{0} multiply(f32[1024]{0} %rsqrt.2632, f32[1024]{0} %reshape.368), metadata={op_type="Mul" op_name="stage4_unit1_bn2/mul"}
  %broadcast.2634 = f32[1,7,7,1024]{3,2,1,0} broadcast(f32[1024]{0} %multiply.2633), dimensions={3}, metadata={op_type="Mul" op_name="stage4_unit1_bn2/mul_1"}
  %multiply.2635 = f32[1,7,7,1024]{3,2,1,0} multiply(f32[1,7,7,1024]{3,2,1,0} %concatenate.2628, f32[1,7,7,1024]{3,2,1,0} %broadcast.2634), metadata={op_type="Mul" op_name="stage4_unit1_bn2/mul_1"}
  %arg205.206 = f32[1024]{0} parameter(205), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.475 = f32[1024]{0} reshape(f32[1024]{0} %arg205.206)
  %arg152.153 = f32[1024]{0} parameter(152), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.422 = f32[1024]{0} reshape(f32[1024]{0} %arg152.153)
  %multiply.2636 = f32[1024]{0} multiply(f32[1024]{0} %multiply.2633, f32[1024]{0} %reshape.422), metadata={op_type="Mul" op_name="stage4_unit1_bn2/mul_2"}
  %subtract.2637 = f32[1024]{0} subtract(f32[1024]{0} %reshape.475, f32[1024]{0} %multiply.2636), metadata={op_type="Sub" op_name="stage4_unit1_bn2/sub"}
  %broadcast.2638 = f32[1,7,7,1024]{3,2,1,0} broadcast(f32[1024]{0} %subtract.2637), dimensions={3}, metadata={op_type="AddV2" op_name="stage4_unit1_bn2/add_1"}
  %add.2639 = f32[1,7,7,1024]{3,2,1,0} add(f32[1,7,7,1024]{3,2,1,0} %multiply.2635, f32[1,7,7,1024]{3,2,1,0} %broadcast.2638), metadata={op_type="AddV2" op_name="stage4_unit1_bn2/add_1"}
  %maximum.2642 = f32[1,7,7,1024]{3,2,1,0} maximum(f32[1,7,7,1024]{3,2,1,0} %broadcast.2641, f32[1,7,7,1024]{3,2,1,0} %add.2639), metadata={op_type="Relu" op_name="stage4_unit1_relu2"}
  %arg263.264 = f32[1,1,1024,2048]{3,2,1,0} parameter(263), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.533 = f32[1,1,1024,2048]{3,2,1,0} reshape(f32[1,1,1024,2048]{3,2,1,0} %arg263.264)
  %convolution.2643 = f32[1,7,7,2048]{3,2,1,0} convolution(f32[1,7,7,1024]{3,2,1,0} %maximum.2642, f32[1,1,1024,2048]{3,2,1,0} %reshape.533), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_conv3"}
  %multiply.2650 = f32[1,7,7,2048]{3,2,1,0} multiply(f32[1,7,7,2048]{3,2,1,0} %broadcast.2649, f32[1,7,7,2048]{3,2,1,0} %convolution.2643), metadata={op_type="Mul" op_name="stage4_unit1_bn3/mul_1"}
  %arg209.210 = f32[2048]{0} parameter(209), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.479 = f32[2048]{0} reshape(f32[2048]{0} %arg209.210)
  %arg156.157 = f32[2048]{0} parameter(156), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.426 = f32[2048]{0} reshape(f32[2048]{0} %arg156.157)
  %multiply.2651 = f32[2048]{0} multiply(f32[2048]{0} %multiply.2648, f32[2048]{0} %reshape.426), metadata={op_type="Mul" op_name="stage4_unit1_bn3/mul_2"}
  %subtract.2652 = f32[2048]{0} subtract(f32[2048]{0} %reshape.479, f32[2048]{0} %multiply.2651), metadata={op_type="Sub" op_name="stage4_unit1_bn3/sub"}
  %broadcast.2653 = f32[1,7,7,2048]{3,2,1,0} broadcast(f32[2048]{0} %subtract.2652), dimensions={3}, metadata={op_type="AddV2" op_name="stage4_unit1_bn3/add_1"}
  %add.2654 = f32[1,7,7,2048]{3,2,1,0} add(f32[1,7,7,2048]{3,2,1,0} %multiply.2650, f32[1,7,7,2048]{3,2,1,0} %broadcast.2653), metadata={op_type="AddV2" op_name="stage4_unit1_bn3/add_1"}
  %arg262.263 = f32[1,1,1024,2048]{3,2,1,0} parameter(262), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.532 = f32[1,1,1024,2048]{3,2,1,0} reshape(f32[1,1,1024,2048]{3,2,1,0} %arg262.263)
  %convolution.2547 = f32[1,7,7,2048]{3,2,1,0} convolution(f32[1,14,14,1024]{3,2,1,0} %maximum.2545, f32[1,1,1024,2048]{3,2,1,0} %reshape.532), window={size=1x1 stride=2x2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit1_sc"}
  %arg26.27 = f32[2048]{0} parameter(26), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.296 = f32[2048]{0} reshape(f32[2048]{0} %arg26.27)
  %constant.2655 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage4_unit1_sc_bn/add"}
  %broadcast.2656 = f32[2048]{0} broadcast(f32[] %constant.2655), dimensions={}, metadata={op_type="AddV2" op_name="stage4_unit1_sc_bn/add"}
  %add.2657 = f32[2048]{0} add(f32[2048]{0} %reshape.296, f32[2048]{0} %broadcast.2656), metadata={op_type="AddV2" op_name="stage4_unit1_sc_bn/add"}
  %rsqrt.2658 = f32[2048]{0} rsqrt(f32[2048]{0} %add.2657), metadata={op_type="Rsqrt" op_name="stage4_unit1_sc_bn/Rsqrt"}
  %arg90.91 = f32[2048]{0} parameter(90), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.360 = f32[2048]{0} reshape(f32[2048]{0} %arg90.91)
  %multiply.2659 = f32[2048]{0} multiply(f32[2048]{0} %rsqrt.2658, f32[2048]{0} %reshape.360), metadata={op_type="Mul" op_name="stage4_unit1_sc_bn/mul"}
  %broadcast.2660 = f32[1,7,7,2048]{3,2,1,0} broadcast(f32[2048]{0} %multiply.2659), dimensions={3}, metadata={op_type="Mul" op_name="stage4_unit1_sc_bn/mul_1"}
  %multiply.2661 = f32[1,7,7,2048]{3,2,1,0} multiply(f32[1,7,7,2048]{3,2,1,0} %convolution.2547, f32[1,7,7,2048]{3,2,1,0} %broadcast.2660), metadata={op_type="Mul" op_name="stage4_unit1_sc_bn/mul_1"}
  %arg197.198 = f32[2048]{0} parameter(197), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.467 = f32[2048]{0} reshape(f32[2048]{0} %arg197.198)
  %arg144.145 = f32[2048]{0} parameter(144), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.414 = f32[2048]{0} reshape(f32[2048]{0} %arg144.145)
  %multiply.2662 = f32[2048]{0} multiply(f32[2048]{0} %multiply.2659, f32[2048]{0} %reshape.414), metadata={op_type="Mul" op_name="stage4_unit1_sc_bn/mul_2"}
  %subtract.2663 = f32[2048]{0} subtract(f32[2048]{0} %reshape.467, f32[2048]{0} %multiply.2662), metadata={op_type="Sub" op_name="stage4_unit1_sc_bn/sub"}
  %broadcast.2664 = f32[1,7,7,2048]{3,2,1,0} broadcast(f32[2048]{0} %subtract.2663), dimensions={3}, metadata={op_type="AddV2" op_name="stage4_unit1_sc_bn/add_1"}
  %add.2665 = f32[1,7,7,2048]{3,2,1,0} add(f32[1,7,7,2048]{3,2,1,0} %multiply.2661, f32[1,7,7,2048]{3,2,1,0} %broadcast.2664), metadata={op_type="AddV2" op_name="stage4_unit1_sc_bn/add_1"}
  %add.2666 = f32[1,7,7,2048]{3,2,1,0} add(f32[1,7,7,2048]{3,2,1,0} %add.2654, f32[1,7,7,2048]{3,2,1,0} %add.2665), metadata={op_type="AddV2" op_name="add_13"}
  %maximum.2669 = f32[1,7,7,2048]{3,2,1,0} maximum(f32[1,7,7,2048]{3,2,1,0} %broadcast.2668, f32[1,7,7,2048]{3,2,1,0} %add.2666), metadata={op_type="Relu" op_name="stage4_unit1_relu"}
  %arg63.64 = f32[2048]{0} parameter(63), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.333 = f32[2048]{0} reshape(f32[2048]{0} %arg63.64)
  %constant.2767 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage4_unit2_bn3/add"}
  %broadcast.2768 = f32[2048]{0} broadcast(f32[] %constant.2767), dimensions={}, metadata={op_type="AddV2" op_name="stage4_unit2_bn3/add"}
  %add.2769 = f32[2048]{0} add(f32[2048]{0} %reshape.333, f32[2048]{0} %broadcast.2768), metadata={op_type="AddV2" op_name="stage4_unit2_bn3/add"}
  %rsqrt.2770 = f32[2048]{0} rsqrt(f32[2048]{0} %add.2769), metadata={op_type="Rsqrt" op_name="stage4_unit2_bn3/Rsqrt"}
  %arg118.119 = f32[2048]{0} parameter(118), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.388 = f32[2048]{0} reshape(f32[2048]{0} %arg118.119)
  %multiply.2771 = f32[2048]{0} multiply(f32[2048]{0} %rsqrt.2770, f32[2048]{0} %reshape.388), metadata={op_type="Mul" op_name="stage4_unit2_bn3/mul"}
  %broadcast.2772 = f32[1,7,7,2048]{3,2,1,0} broadcast(f32[2048]{0} %multiply.2771), dimensions={3}, metadata={op_type="Mul" op_name="stage4_unit2_bn3/mul_1"}
  %constant.2763 = f32[] constant(0), metadata={op_type="Relu" op_name="stage4_unit2_relu2"}
  %broadcast.2764 = f32[1,7,7,1024]{3,2,1,0} broadcast(f32[] %constant.2763), dimensions={}, metadata={op_type="Relu" op_name="stage4_unit2_relu2"}
  %constant.2682 = f32[] constant(0), metadata={op_type="Relu" op_name="stage4_unit2_relu1"}
  %broadcast.2683 = f32[1,7,7,1024]{3,2,1,0} broadcast(f32[] %constant.2682), dimensions={}, metadata={op_type="Relu" op_name="stage4_unit2_relu1"}
  %arg48.49 = f32[1024]{0} parameter(48), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.318 = f32[1024]{0} reshape(f32[1024]{0} %arg48.49)
  %constant.2671 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage4_unit2_bn1/add"}
  %broadcast.2672 = f32[1024]{0} broadcast(f32[] %constant.2671), dimensions={}, metadata={op_type="AddV2" op_name="stage4_unit2_bn1/add"}
  %add.2673 = f32[1024]{0} add(f32[1024]{0} %reshape.318, f32[1024]{0} %broadcast.2672), metadata={op_type="AddV2" op_name="stage4_unit2_bn1/add"}
  %rsqrt.2674 = f32[1024]{0} rsqrt(f32[1024]{0} %add.2673), metadata={op_type="Rsqrt" op_name="stage4_unit2_bn1/Rsqrt"}
  %arg107.108 = f32[1024]{0} parameter(107), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.377 = f32[1024]{0} reshape(f32[1024]{0} %arg107.108)
  %multiply.2675 = f32[1024]{0} multiply(f32[1024]{0} %rsqrt.2674, f32[1024]{0} %reshape.377), metadata={op_type="Mul" op_name="stage4_unit2_bn1/mul"}
  %broadcast.2676 = f32[1,7,7,1024]{3,2,1,0} broadcast(f32[1024]{0} %multiply.2675), dimensions={3}, metadata={op_type="Mul" op_name="stage4_unit2_bn1/mul_1"}
  %arg264.265 = f32[1,1,2048,1024]{3,2,1,0} parameter(264), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.534 = f32[1,1,2048,1024]{3,2,1,0} reshape(f32[1,1,2048,1024]{3,2,1,0} %arg264.265)
  %convolution.2670 = f32[1,7,7,1024]{3,2,1,0} convolution(f32[1,7,7,2048]{3,2,1,0} %maximum.2669, f32[1,1,2048,1024]{3,2,1,0} %reshape.534), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv1"}
  %multiply.2677 = f32[1,7,7,1024]{3,2,1,0} multiply(f32[1,7,7,1024]{3,2,1,0} %broadcast.2676, f32[1,7,7,1024]{3,2,1,0} %convolution.2670), metadata={op_type="Mul" op_name="stage4_unit2_bn1/mul_1"}
  %arg214.215 = f32[1024]{0} parameter(214), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.484 = f32[1024]{0} reshape(f32[1024]{0} %arg214.215)
  %arg161.162 = f32[1024]{0} parameter(161), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.431 = f32[1024]{0} reshape(f32[1024]{0} %arg161.162)
  %multiply.2678 = f32[1024]{0} multiply(f32[1024]{0} %multiply.2675, f32[1024]{0} %reshape.431), metadata={op_type="Mul" op_name="stage4_unit2_bn1/mul_2"}
  %subtract.2679 = f32[1024]{0} subtract(f32[1024]{0} %reshape.484, f32[1024]{0} %multiply.2678), metadata={op_type="Sub" op_name="stage4_unit2_bn1/sub"}
  %broadcast.2680 = f32[1,7,7,1024]{3,2,1,0} broadcast(f32[1024]{0} %subtract.2679), dimensions={3}, metadata={op_type="AddV2" op_name="stage4_unit2_bn1/add_1"}
  %add.2681 = f32[1,7,7,1024]{3,2,1,0} add(f32[1,7,7,1024]{3,2,1,0} %multiply.2677, f32[1,7,7,1024]{3,2,1,0} %broadcast.2680), metadata={op_type="AddV2" op_name="stage4_unit2_bn1/add_1"}
  %maximum.2684 = f32[1,7,7,1024]{3,2,1,0} maximum(f32[1,7,7,1024]{3,2,1,0} %broadcast.2683, f32[1,7,7,1024]{3,2,1,0} %add.2681), metadata={op_type="Relu" op_name="stage4_unit2_relu1"}
  %constant.2685 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_15"}
  %pad.2686 = f32[1,9,9,1024]{3,2,1,0} pad(f32[1,7,7,1024]{3,2,1,0} %maximum.2684, f32[] %constant.2685), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_15"}
  %slice.2687 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2686), slice={[0:1], [0:9], [0:9], [0:32]}, metadata={op_type="Split" op_name="split_29"}
  %arg52.53 = f32[3,3,32,1024]{3,2,1,0} parameter(52), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.322 = f32[3,3,32,1024]{3,2,1,0} reshape(f32[3,3,32,1024]{3,2,1,0} %arg52.53)
  %slice.987 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:32], [0:32]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2719 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2687, f32[3,3,32,32]{3,2,1,0} %slice.987), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2"}
  %slice.2688 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2686), slice={[0:1], [0:9], [0:9], [32:64]}, metadata={op_type="Split" op_name="split_29"}
  %slice.988 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:32], [32:64]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2720 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2688, f32[3,3,32,32]{3,2,1,0} %slice.988), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_1"}
  %slice.2689 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2686), slice={[0:1], [0:9], [0:9], [64:96]}, metadata={op_type="Split" op_name="split_29"}
  %slice.989 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:32], [64:96]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2731 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2689, f32[3,3,32,32]{3,2,1,0} %slice.989), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_2"}
  %slice.2690 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2686), slice={[0:1], [0:9], [0:9], [96:128]}, metadata={op_type="Split" op_name="split_29"}
  %slice.990 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:32], [96:128]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2742 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2690, f32[3,3,32,32]{3,2,1,0} %slice.990), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_3"}
  %slice.2691 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2686), slice={[0:1], [0:9], [0:9], [128:160]}, metadata={op_type="Split" op_name="split_29"}
  %slice.991 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:32], [128:160]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2745 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2691, f32[3,3,32,32]{3,2,1,0} %slice.991), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_4"}
  %slice.2692 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2686), slice={[0:1], [0:9], [0:9], [160:192]}, metadata={op_type="Split" op_name="split_29"}
  %slice.992 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:32], [160:192]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2746 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2692, f32[3,3,32,32]{3,2,1,0} %slice.992), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_5"}
  %slice.2693 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2686), slice={[0:1], [0:9], [0:9], [192:224]}, metadata={op_type="Split" op_name="split_29"}
  %slice.993 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:32], [192:224]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2747 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2693, f32[3,3,32,32]{3,2,1,0} %slice.993), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_6"}
  %slice.2694 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2686), slice={[0:1], [0:9], [0:9], [224:256]}, metadata={op_type="Split" op_name="split_29"}
  %slice.994 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:32], [224:256]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2748 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2694, f32[3,3,32,32]{3,2,1,0} %slice.994), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_7"}
  %slice.2695 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2686), slice={[0:1], [0:9], [0:9], [256:288]}, metadata={op_type="Split" op_name="split_29"}
  %slice.995 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:32], [256:288]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2749 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2695, f32[3,3,32,32]{3,2,1,0} %slice.995), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_8"}
  %slice.2696 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2686), slice={[0:1], [0:9], [0:9], [288:320]}, metadata={op_type="Split" op_name="split_29"}
  %slice.996 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:32], [288:320]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2750 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2696, f32[3,3,32,32]{3,2,1,0} %slice.996), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_9"}
  %slice.2697 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2686), slice={[0:1], [0:9], [0:9], [320:352]}, metadata={op_type="Split" op_name="split_29"}
  %slice.997 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:32], [320:352]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2721 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2697, f32[3,3,32,32]{3,2,1,0} %slice.997), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_10"}
  %slice.2698 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2686), slice={[0:1], [0:9], [0:9], [352:384]}, metadata={op_type="Split" op_name="split_29"}
  %slice.998 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:32], [352:384]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2722 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2698, f32[3,3,32,32]{3,2,1,0} %slice.998), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_11"}
  %slice.2699 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2686), slice={[0:1], [0:9], [0:9], [384:416]}, metadata={op_type="Split" op_name="split_29"}
  %slice.999 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:32], [384:416]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2723 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2699, f32[3,3,32,32]{3,2,1,0} %slice.999), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_12"}
  %slice.2700 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2686), slice={[0:1], [0:9], [0:9], [416:448]}, metadata={op_type="Split" op_name="split_29"}
  %slice.1000 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:32], [416:448]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2724 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2700, f32[3,3,32,32]{3,2,1,0} %slice.1000), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_13"}
  %slice.2701 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2686), slice={[0:1], [0:9], [0:9], [448:480]}, metadata={op_type="Split" op_name="split_29"}
  %slice.1001 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:32], [448:480]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2725 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2701, f32[3,3,32,32]{3,2,1,0} %slice.1001), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_14"}
  %slice.2702 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2686), slice={[0:1], [0:9], [0:9], [480:512]}, metadata={op_type="Split" op_name="split_29"}
  %slice.1002 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:32], [480:512]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2726 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2702, f32[3,3,32,32]{3,2,1,0} %slice.1002), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_15"}
  %slice.2703 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2686), slice={[0:1], [0:9], [0:9], [512:544]}, metadata={op_type="Split" op_name="split_29"}
  %slice.1003 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:32], [512:544]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2727 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2703, f32[3,3,32,32]{3,2,1,0} %slice.1003), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_16"}
  %slice.2704 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2686), slice={[0:1], [0:9], [0:9], [544:576]}, metadata={op_type="Split" op_name="split_29"}
  %slice.1004 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:32], [544:576]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2728 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2704, f32[3,3,32,32]{3,2,1,0} %slice.1004), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_17"}
  %slice.2705 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2686), slice={[0:1], [0:9], [0:9], [576:608]}, metadata={op_type="Split" op_name="split_29"}
  %slice.1005 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:32], [576:608]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2729 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2705, f32[3,3,32,32]{3,2,1,0} %slice.1005), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_18"}
  %slice.2706 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2686), slice={[0:1], [0:9], [0:9], [608:640]}, metadata={op_type="Split" op_name="split_29"}
  %slice.1006 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:32], [608:640]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2730 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2706, f32[3,3,32,32]{3,2,1,0} %slice.1006), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_19"}
  %slice.2707 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2686), slice={[0:1], [0:9], [0:9], [640:672]}, metadata={op_type="Split" op_name="split_29"}
  %slice.1007 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:32], [640:672]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2732 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2707, f32[3,3,32,32]{3,2,1,0} %slice.1007), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_20"}
  %slice.2708 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2686), slice={[0:1], [0:9], [0:9], [672:704]}, metadata={op_type="Split" op_name="split_29"}
  %slice.1008 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:32], [672:704]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2733 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2708, f32[3,3,32,32]{3,2,1,0} %slice.1008), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_21"}
  %slice.2709 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2686), slice={[0:1], [0:9], [0:9], [704:736]}, metadata={op_type="Split" op_name="split_29"}
  %slice.1009 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:32], [704:736]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2734 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2709, f32[3,3,32,32]{3,2,1,0} %slice.1009), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_22"}
  %slice.2710 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2686), slice={[0:1], [0:9], [0:9], [736:768]}, metadata={op_type="Split" op_name="split_29"}
  %slice.1010 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:32], [736:768]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2735 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2710, f32[3,3,32,32]{3,2,1,0} %slice.1010), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_23"}
  %slice.2711 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2686), slice={[0:1], [0:9], [0:9], [768:800]}, metadata={op_type="Split" op_name="split_29"}
  %slice.1011 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:32], [768:800]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2736 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2711, f32[3,3,32,32]{3,2,1,0} %slice.1011), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_24"}
  %slice.2712 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2686), slice={[0:1], [0:9], [0:9], [800:832]}, metadata={op_type="Split" op_name="split_29"}
  %slice.1012 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:32], [800:832]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2737 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2712, f32[3,3,32,32]{3,2,1,0} %slice.1012), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_25"}
  %slice.2713 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2686), slice={[0:1], [0:9], [0:9], [832:864]}, metadata={op_type="Split" op_name="split_29"}
  %slice.1013 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:32], [832:864]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2738 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2713, f32[3,3,32,32]{3,2,1,0} %slice.1013), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_26"}
  %slice.2714 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2686), slice={[0:1], [0:9], [0:9], [864:896]}, metadata={op_type="Split" op_name="split_29"}
  %slice.1014 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:32], [864:896]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2739 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2714, f32[3,3,32,32]{3,2,1,0} %slice.1014), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_27"}
  %slice.2715 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2686), slice={[0:1], [0:9], [0:9], [896:928]}, metadata={op_type="Split" op_name="split_29"}
  %slice.1015 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:32], [896:928]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2740 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2715, f32[3,3,32,32]{3,2,1,0} %slice.1015), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_28"}
  %slice.2716 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2686), slice={[0:1], [0:9], [0:9], [928:960]}, metadata={op_type="Split" op_name="split_29"}
  %slice.1016 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:32], [928:960]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2741 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2716, f32[3,3,32,32]{3,2,1,0} %slice.1016), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_29"}
  %slice.2717 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2686), slice={[0:1], [0:9], [0:9], [960:992]}, metadata={op_type="Split" op_name="split_29"}
  %slice.1017 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:32], [960:992]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2743 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2717, f32[3,3,32,32]{3,2,1,0} %slice.1017), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_30"}
  %slice.2718 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2686), slice={[0:1], [0:9], [0:9], [992:1024]}, metadata={op_type="Split" op_name="split_29"}
  %slice.1018 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.322), slice={[0:3], [0:3], [0:32], [992:1024]}, metadata={op_type="Split" op_name="split_28"}
  %convolution.2744 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2718, f32[3,3,32,32]{3,2,1,0} %slice.1018), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv2_31"}
  %concatenate.2751 = f32[1,7,7,1024]{3,2,1,0} concatenate(f32[1,7,7,32]{3,2,1,0} %convolution.2719, f32[1,7,7,32]{3,2,1,0} %convolution.2720, f32[1,7,7,32]{3,2,1,0} %convolution.2731, f32[1,7,7,32]{3,2,1,0} %convolution.2742, f32[1,7,7,32]{3,2,1,0} %convolution.2745, f32[1,7,7,32]{3,2,1,0} %convolution.2746, f32[1,7,7,32]{3,2,1,0} %convolution.2747, f32[1,7,7,32]{3,2,1,0} %convolution.2748, f32[1,7,7,32]{3,2,1,0} %convolution.2749, f32[1,7,7,32]{3,2,1,0} %convolution.2750, f32[1,7,7,32]{3,2,1,0} %convolution.2721, f32[1,7,7,32]{3,2,1,0} %convolution.2722, f32[1,7,7,32]{3,2,1,0} %convolution.2723, f32[1,7,7,32]{3,2,1,0} %convolution.2724, f32[1,7,7,32]{3,2,1,0} %convolution.2725, f32[1,7,7,32]{3,2,1,0} %convolution.2726, f32[1,7,7,32]{3,2,1,0} %convolution.2727, f32[1,7,7,32]{3,2,1,0} %convolution.2728, f32[1,7,7,32]{3,2,1,0} %convolution.2729, f32[1,7,7,32]{3,2,1,0} %convolution.2730, f32[1,7,7,32]{3,2,1,0} %convolution.2732, f32[1,7,7,32]{3,2,1,0} %convolution.2733, f32[1,7,7,32]{3,2,1,0} %convolution.2734, f32[1,7,7,32]{3,2,1,0} %convolution.2735, f32[1,7,7,32]{3,2,1,0} %convolution.2736, f32[1,7,7,32]{3,2,1,0} %convolution.2737, f32[1,7,7,32]{3,2,1,0} %convolution.2738, f32[1,7,7,32]{3,2,1,0} %convolution.2739, f32[1,7,7,32]{3,2,1,0} %convolution.2740, f32[1,7,7,32]{3,2,1,0} %convolution.2741, f32[1,7,7,32]{3,2,1,0} %convolution.2743, f32[1,7,7,32]{3,2,1,0} %convolution.2744), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_14"}
  %arg58.59 = f32[1024]{0} parameter(58), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.328 = f32[1024]{0} reshape(f32[1024]{0} %arg58.59)
  %constant.2752 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage4_unit2_bn2/add"}
  %broadcast.2753 = f32[1024]{0} broadcast(f32[] %constant.2752), dimensions={}, metadata={op_type="AddV2" op_name="stage4_unit2_bn2/add"}
  %add.2754 = f32[1024]{0} add(f32[1024]{0} %reshape.328, f32[1024]{0} %broadcast.2753), metadata={op_type="AddV2" op_name="stage4_unit2_bn2/add"}
  %rsqrt.2755 = f32[1024]{0} rsqrt(f32[1024]{0} %add.2754), metadata={op_type="Rsqrt" op_name="stage4_unit2_bn2/Rsqrt"}
  %arg114.115 = f32[1024]{0} parameter(114), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.384 = f32[1024]{0} reshape(f32[1024]{0} %arg114.115)
  %multiply.2756 = f32[1024]{0} multiply(f32[1024]{0} %rsqrt.2755, f32[1024]{0} %reshape.384), metadata={op_type="Mul" op_name="stage4_unit2_bn2/mul"}
  %broadcast.2757 = f32[1,7,7,1024]{3,2,1,0} broadcast(f32[1024]{0} %multiply.2756), dimensions={3}, metadata={op_type="Mul" op_name="stage4_unit2_bn2/mul_1"}
  %multiply.2758 = f32[1,7,7,1024]{3,2,1,0} multiply(f32[1,7,7,1024]{3,2,1,0} %concatenate.2751, f32[1,7,7,1024]{3,2,1,0} %broadcast.2757), metadata={op_type="Mul" op_name="stage4_unit2_bn2/mul_1"}
  %arg221.222 = f32[1024]{0} parameter(221), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.491 = f32[1024]{0} reshape(f32[1024]{0} %arg221.222)
  %arg168.169 = f32[1024]{0} parameter(168), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.438 = f32[1024]{0} reshape(f32[1024]{0} %arg168.169)
  %multiply.2759 = f32[1024]{0} multiply(f32[1024]{0} %multiply.2756, f32[1024]{0} %reshape.438), metadata={op_type="Mul" op_name="stage4_unit2_bn2/mul_2"}
  %subtract.2760 = f32[1024]{0} subtract(f32[1024]{0} %reshape.491, f32[1024]{0} %multiply.2759), metadata={op_type="Sub" op_name="stage4_unit2_bn2/sub"}
  %broadcast.2761 = f32[1,7,7,1024]{3,2,1,0} broadcast(f32[1024]{0} %subtract.2760), dimensions={3}, metadata={op_type="AddV2" op_name="stage4_unit2_bn2/add_1"}
  %add.2762 = f32[1,7,7,1024]{3,2,1,0} add(f32[1,7,7,1024]{3,2,1,0} %multiply.2758, f32[1,7,7,1024]{3,2,1,0} %broadcast.2761), metadata={op_type="AddV2" op_name="stage4_unit2_bn2/add_1"}
  %maximum.2765 = f32[1,7,7,1024]{3,2,1,0} maximum(f32[1,7,7,1024]{3,2,1,0} %broadcast.2764, f32[1,7,7,1024]{3,2,1,0} %add.2762), metadata={op_type="Relu" op_name="stage4_unit2_relu2"}
  %arg265.266 = f32[1,1,1024,2048]{3,2,1,0} parameter(265), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.535 = f32[1,1,1024,2048]{3,2,1,0} reshape(f32[1,1,1024,2048]{3,2,1,0} %arg265.266)
  %convolution.2766 = f32[1,7,7,2048]{3,2,1,0} convolution(f32[1,7,7,1024]{3,2,1,0} %maximum.2765, f32[1,1,1024,2048]{3,2,1,0} %reshape.535), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit2_conv3"}
  %multiply.2773 = f32[1,7,7,2048]{3,2,1,0} multiply(f32[1,7,7,2048]{3,2,1,0} %broadcast.2772, f32[1,7,7,2048]{3,2,1,0} %convolution.2766), metadata={op_type="Mul" op_name="stage4_unit2_bn3/mul_1"}
  %arg225.226 = f32[2048]{0} parameter(225), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.495 = f32[2048]{0} reshape(f32[2048]{0} %arg225.226)
  %arg172.173 = f32[2048]{0} parameter(172), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.442 = f32[2048]{0} reshape(f32[2048]{0} %arg172.173)
  %multiply.2774 = f32[2048]{0} multiply(f32[2048]{0} %multiply.2771, f32[2048]{0} %reshape.442), metadata={op_type="Mul" op_name="stage4_unit2_bn3/mul_2"}
  %subtract.2775 = f32[2048]{0} subtract(f32[2048]{0} %reshape.495, f32[2048]{0} %multiply.2774), metadata={op_type="Sub" op_name="stage4_unit2_bn3/sub"}
  %broadcast.2776 = f32[1,7,7,2048]{3,2,1,0} broadcast(f32[2048]{0} %subtract.2775), dimensions={3}, metadata={op_type="AddV2" op_name="stage4_unit2_bn3/add_1"}
  %add.2777 = f32[1,7,7,2048]{3,2,1,0} add(f32[1,7,7,2048]{3,2,1,0} %multiply.2773, f32[1,7,7,2048]{3,2,1,0} %broadcast.2776), metadata={op_type="AddV2" op_name="stage4_unit2_bn3/add_1"}
  %add.2778 = f32[1,7,7,2048]{3,2,1,0} add(f32[1,7,7,2048]{3,2,1,0} %maximum.2669, f32[1,7,7,2048]{3,2,1,0} %add.2777), metadata={op_type="AddV2" op_name="add_14"}
  %maximum.2781 = f32[1,7,7,2048]{3,2,1,0} maximum(f32[1,7,7,2048]{3,2,1,0} %broadcast.2780, f32[1,7,7,2048]{3,2,1,0} %add.2778), metadata={op_type="Relu" op_name="stage4_unit2_relu"}
  %constant.2796 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage4_unit3_bn3/add"}
  %broadcast.2797 = f32[2048]{0} broadcast(f32[] %constant.2796), dimensions={}, metadata={op_type="AddV2" op_name="stage4_unit3_bn3/add"}
  %arg64.65 = f32[2048]{0} parameter(64), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.334 = f32[2048]{0} reshape(f32[2048]{0} %arg64.65)
  %add.2798 = f32[2048]{0} add(f32[2048]{0} %broadcast.2797, f32[2048]{0} %reshape.334), metadata={op_type="AddV2" op_name="stage4_unit3_bn3/add"}
  %rsqrt.2799 = f32[2048]{0} rsqrt(f32[2048]{0} %add.2798), metadata={op_type="Rsqrt" op_name="stage4_unit3_bn3/Rsqrt"}
  %arg119.120 = f32[2048]{0} parameter(119), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.389 = f32[2048]{0} reshape(f32[2048]{0} %arg119.120)
  %multiply.2800 = f32[2048]{0} multiply(f32[2048]{0} %rsqrt.2799, f32[2048]{0} %reshape.389), metadata={op_type="Mul" op_name="stage4_unit3_bn3/mul"}
  %broadcast.2918 = f32[1,7,7,2048]{3,2,1,0} broadcast(f32[2048]{0} %multiply.2800), dimensions={3}, metadata={op_type="Mul" op_name="stage4_unit3_bn3/mul_1"}
  %constant.2914 = f32[] constant(0), metadata={op_type="Relu" op_name="stage4_unit3_relu2"}
  %broadcast.2915 = f32[1,7,7,1024]{3,2,1,0} broadcast(f32[] %constant.2914), dimensions={}, metadata={op_type="Relu" op_name="stage4_unit3_relu2"}
  %constant.2808 = f32[] constant(0), metadata={op_type="Relu" op_name="stage4_unit3_relu1"}
  %broadcast.2809 = f32[1,7,7,1024]{3,2,1,0} broadcast(f32[] %constant.2808), dimensions={}, metadata={op_type="Relu" op_name="stage4_unit3_relu1"}
  %arg69.70 = f32[1024]{0} parameter(69), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.339 = f32[1024]{0} reshape(f32[1024]{0} %arg69.70)
  %constant.2782 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage4_unit3_bn1/add"}
  %broadcast.2783 = f32[1024]{0} broadcast(f32[] %constant.2782), dimensions={}, metadata={op_type="AddV2" op_name="stage4_unit3_bn1/add"}
  %add.2784 = f32[1024]{0} add(f32[1024]{0} %reshape.339, f32[1024]{0} %broadcast.2783), metadata={op_type="AddV2" op_name="stage4_unit3_bn1/add"}
  %rsqrt.2785 = f32[1024]{0} rsqrt(f32[1024]{0} %add.2784), metadata={op_type="Rsqrt" op_name="stage4_unit3_bn1/Rsqrt"}
  %arg123.124 = f32[1024]{0} parameter(123), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.393 = f32[1024]{0} reshape(f32[1024]{0} %arg123.124)
  %multiply.2786 = f32[1024]{0} multiply(f32[1024]{0} %rsqrt.2785, f32[1024]{0} %reshape.393), metadata={op_type="Mul" op_name="stage4_unit3_bn1/mul"}
  %broadcast.2804 = f32[1,7,7,1024]{3,2,1,0} broadcast(f32[1024]{0} %multiply.2786), dimensions={3}, metadata={op_type="Mul" op_name="stage4_unit3_bn1/mul_1"}
  %arg266.267 = f32[1,1,2048,1024]{3,2,1,0} parameter(266), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.536 = f32[1,1,2048,1024]{3,2,1,0} reshape(f32[1,1,2048,1024]{3,2,1,0} %arg266.267)
  %convolution.2803 = f32[1,7,7,1024]{3,2,1,0} convolution(f32[1,7,7,2048]{3,2,1,0} %maximum.2781, f32[1,1,2048,1024]{3,2,1,0} %reshape.536), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv1"}
  %multiply.2805 = f32[1,7,7,1024]{3,2,1,0} multiply(f32[1,7,7,1024]{3,2,1,0} %broadcast.2804, f32[1,7,7,1024]{3,2,1,0} %convolution.2803), metadata={op_type="Mul" op_name="stage4_unit3_bn1/mul_1"}
  %arg230.231 = f32[1024]{0} parameter(230), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.500 = f32[1024]{0} reshape(f32[1024]{0} %arg230.231)
  %arg177.178 = f32[1024]{0} parameter(177), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.447 = f32[1024]{0} reshape(f32[1024]{0} %arg177.178)
  %multiply.2787 = f32[1024]{0} multiply(f32[1024]{0} %multiply.2786, f32[1024]{0} %reshape.447), metadata={op_type="Mul" op_name="stage4_unit3_bn1/mul_2"}
  %subtract.2788 = f32[1024]{0} subtract(f32[1024]{0} %reshape.500, f32[1024]{0} %multiply.2787), metadata={op_type="Sub" op_name="stage4_unit3_bn1/sub"}
  %broadcast.2806 = f32[1,7,7,1024]{3,2,1,0} broadcast(f32[1024]{0} %subtract.2788), dimensions={3}, metadata={op_type="AddV2" op_name="stage4_unit3_bn1/add_1"}
  %add.2807 = f32[1,7,7,1024]{3,2,1,0} add(f32[1,7,7,1024]{3,2,1,0} %multiply.2805, f32[1,7,7,1024]{3,2,1,0} %broadcast.2806), metadata={op_type="AddV2" op_name="stage4_unit3_bn1/add_1"}
  %maximum.2810 = f32[1,7,7,1024]{3,2,1,0} maximum(f32[1,7,7,1024]{3,2,1,0} %broadcast.2809, f32[1,7,7,1024]{3,2,1,0} %add.2807), metadata={op_type="Relu" op_name="stage4_unit3_relu1"}
  %constant.2811 = f32[] constant(0), metadata={op_type="Pad" op_name="Pad_16"}
  %pad.2812 = f32[1,9,9,1024]{3,2,1,0} pad(f32[1,7,7,1024]{3,2,1,0} %maximum.2810, f32[] %constant.2811), padding=0_0x1_1x1_1x0_0, metadata={op_type="Pad" op_name="Pad_16"}
  %slice.2813 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [0:32]}, metadata={op_type="Split" op_name="split_31"}
  %arg15.16 = f32[3,3,32,1024]{3,2,1,0} parameter(15), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.285 = f32[3,3,32,1024]{3,2,1,0} reshape(f32[3,3,32,1024]{3,2,1,0} %arg15.16)
  %slice.2845 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.285), slice={[0:3], [0:3], [0:32], [0:32]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2877 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2813, f32[3,3,32,32]{3,2,1,0} %slice.2845), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2"}
  %slice.2814 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [32:64]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2846 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.285), slice={[0:3], [0:3], [0:32], [32:64]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2878 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2814, f32[3,3,32,32]{3,2,1,0} %slice.2846), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_1"}
  %slice.2815 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [64:96]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2847 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.285), slice={[0:3], [0:3], [0:32], [64:96]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2889 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2815, f32[3,3,32,32]{3,2,1,0} %slice.2847), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_2"}
  %slice.2816 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [96:128]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2848 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.285), slice={[0:3], [0:3], [0:32], [96:128]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2900 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2816, f32[3,3,32,32]{3,2,1,0} %slice.2848), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_3"}
  %slice.2817 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [128:160]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2849 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.285), slice={[0:3], [0:3], [0:32], [128:160]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2903 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2817, f32[3,3,32,32]{3,2,1,0} %slice.2849), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_4"}
  %slice.2818 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [160:192]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2850 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.285), slice={[0:3], [0:3], [0:32], [160:192]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2904 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2818, f32[3,3,32,32]{3,2,1,0} %slice.2850), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_5"}
  %slice.2819 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [192:224]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2851 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.285), slice={[0:3], [0:3], [0:32], [192:224]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2905 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2819, f32[3,3,32,32]{3,2,1,0} %slice.2851), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_6"}
  %slice.2820 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [224:256]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2852 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.285), slice={[0:3], [0:3], [0:32], [224:256]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2906 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2820, f32[3,3,32,32]{3,2,1,0} %slice.2852), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_7"}
  %slice.2821 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [256:288]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2853 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.285), slice={[0:3], [0:3], [0:32], [256:288]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2907 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2821, f32[3,3,32,32]{3,2,1,0} %slice.2853), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_8"}
  %slice.2822 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [288:320]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2854 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.285), slice={[0:3], [0:3], [0:32], [288:320]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2908 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2822, f32[3,3,32,32]{3,2,1,0} %slice.2854), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_9"}
  %slice.2823 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [320:352]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2855 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.285), slice={[0:3], [0:3], [0:32], [320:352]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2879 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2823, f32[3,3,32,32]{3,2,1,0} %slice.2855), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_10"}
  %slice.2824 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [352:384]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2856 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.285), slice={[0:3], [0:3], [0:32], [352:384]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2880 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2824, f32[3,3,32,32]{3,2,1,0} %slice.2856), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_11"}
  %slice.2825 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [384:416]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2857 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.285), slice={[0:3], [0:3], [0:32], [384:416]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2881 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2825, f32[3,3,32,32]{3,2,1,0} %slice.2857), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_12"}
  %slice.2826 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [416:448]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2858 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.285), slice={[0:3], [0:3], [0:32], [416:448]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2882 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2826, f32[3,3,32,32]{3,2,1,0} %slice.2858), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_13"}
  %slice.2827 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [448:480]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2859 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.285), slice={[0:3], [0:3], [0:32], [448:480]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2883 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2827, f32[3,3,32,32]{3,2,1,0} %slice.2859), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_14"}
  %slice.2828 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [480:512]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2860 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.285), slice={[0:3], [0:3], [0:32], [480:512]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2884 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2828, f32[3,3,32,32]{3,2,1,0} %slice.2860), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_15"}
  %slice.2829 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [512:544]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2861 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.285), slice={[0:3], [0:3], [0:32], [512:544]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2885 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2829, f32[3,3,32,32]{3,2,1,0} %slice.2861), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_16"}
  %slice.2830 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [544:576]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2862 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.285), slice={[0:3], [0:3], [0:32], [544:576]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2886 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2830, f32[3,3,32,32]{3,2,1,0} %slice.2862), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_17"}
  %slice.2831 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [576:608]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2863 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.285), slice={[0:3], [0:3], [0:32], [576:608]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2887 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2831, f32[3,3,32,32]{3,2,1,0} %slice.2863), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_18"}
  %slice.2832 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [608:640]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2864 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.285), slice={[0:3], [0:3], [0:32], [608:640]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2888 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2832, f32[3,3,32,32]{3,2,1,0} %slice.2864), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_19"}
  %slice.2833 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [640:672]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2865 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.285), slice={[0:3], [0:3], [0:32], [640:672]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2890 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2833, f32[3,3,32,32]{3,2,1,0} %slice.2865), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_20"}
  %slice.2834 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [672:704]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2866 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.285), slice={[0:3], [0:3], [0:32], [672:704]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2891 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2834, f32[3,3,32,32]{3,2,1,0} %slice.2866), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_21"}
  %slice.2835 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [704:736]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2867 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.285), slice={[0:3], [0:3], [0:32], [704:736]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2892 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2835, f32[3,3,32,32]{3,2,1,0} %slice.2867), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_22"}
  %slice.2836 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [736:768]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2868 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.285), slice={[0:3], [0:3], [0:32], [736:768]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2893 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2836, f32[3,3,32,32]{3,2,1,0} %slice.2868), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_23"}
  %slice.2837 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [768:800]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2869 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.285), slice={[0:3], [0:3], [0:32], [768:800]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2894 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2837, f32[3,3,32,32]{3,2,1,0} %slice.2869), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_24"}
  %slice.2838 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [800:832]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2870 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.285), slice={[0:3], [0:3], [0:32], [800:832]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2895 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2838, f32[3,3,32,32]{3,2,1,0} %slice.2870), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_25"}
  %slice.2839 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [832:864]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2871 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.285), slice={[0:3], [0:3], [0:32], [832:864]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2896 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2839, f32[3,3,32,32]{3,2,1,0} %slice.2871), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_26"}
  %slice.2840 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [864:896]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2872 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.285), slice={[0:3], [0:3], [0:32], [864:896]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2897 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2840, f32[3,3,32,32]{3,2,1,0} %slice.2872), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_27"}
  %slice.2841 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [896:928]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2873 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.285), slice={[0:3], [0:3], [0:32], [896:928]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2898 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2841, f32[3,3,32,32]{3,2,1,0} %slice.2873), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_28"}
  %slice.2842 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [928:960]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2874 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.285), slice={[0:3], [0:3], [0:32], [928:960]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2899 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2842, f32[3,3,32,32]{3,2,1,0} %slice.2874), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_29"}
  %slice.2843 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [960:992]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2875 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.285), slice={[0:3], [0:3], [0:32], [960:992]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2901 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2843, f32[3,3,32,32]{3,2,1,0} %slice.2875), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_30"}
  %slice.2844 = f32[1,9,9,32]{3,2,1,0} slice(f32[1,9,9,1024]{3,2,1,0} %pad.2812), slice={[0:1], [0:9], [0:9], [992:1024]}, metadata={op_type="Split" op_name="split_31"}
  %slice.2876 = f32[3,3,32,32]{3,2,1,0} slice(f32[3,3,32,1024]{3,2,1,0} %reshape.285), slice={[0:3], [0:3], [0:32], [992:1024]}, metadata={op_type="Split" op_name="split_30"}
  %convolution.2902 = f32[1,7,7,32]{3,2,1,0} convolution(f32[1,9,9,32]{3,2,1,0} %slice.2844, f32[3,3,32,32]{3,2,1,0} %slice.2876), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv2_31"}
  %concatenate.2909 = f32[1,7,7,1024]{3,2,1,0} concatenate(f32[1,7,7,32]{3,2,1,0} %convolution.2877, f32[1,7,7,32]{3,2,1,0} %convolution.2878, f32[1,7,7,32]{3,2,1,0} %convolution.2889, f32[1,7,7,32]{3,2,1,0} %convolution.2900, f32[1,7,7,32]{3,2,1,0} %convolution.2903, f32[1,7,7,32]{3,2,1,0} %convolution.2904, f32[1,7,7,32]{3,2,1,0} %convolution.2905, f32[1,7,7,32]{3,2,1,0} %convolution.2906, f32[1,7,7,32]{3,2,1,0} %convolution.2907, f32[1,7,7,32]{3,2,1,0} %convolution.2908, f32[1,7,7,32]{3,2,1,0} %convolution.2879, f32[1,7,7,32]{3,2,1,0} %convolution.2880, f32[1,7,7,32]{3,2,1,0} %convolution.2881, f32[1,7,7,32]{3,2,1,0} %convolution.2882, f32[1,7,7,32]{3,2,1,0} %convolution.2883, f32[1,7,7,32]{3,2,1,0} %convolution.2884, f32[1,7,7,32]{3,2,1,0} %convolution.2885, f32[1,7,7,32]{3,2,1,0} %convolution.2886, f32[1,7,7,32]{3,2,1,0} %convolution.2887, f32[1,7,7,32]{3,2,1,0} %convolution.2888, f32[1,7,7,32]{3,2,1,0} %convolution.2890, f32[1,7,7,32]{3,2,1,0} %convolution.2891, f32[1,7,7,32]{3,2,1,0} %convolution.2892, f32[1,7,7,32]{3,2,1,0} %convolution.2893, f32[1,7,7,32]{3,2,1,0} %convolution.2894, f32[1,7,7,32]{3,2,1,0} %convolution.2895, f32[1,7,7,32]{3,2,1,0} %convolution.2896, f32[1,7,7,32]{3,2,1,0} %convolution.2897, f32[1,7,7,32]{3,2,1,0} %convolution.2898, f32[1,7,7,32]{3,2,1,0} %convolution.2899, f32[1,7,7,32]{3,2,1,0} %convolution.2901, f32[1,7,7,32]{3,2,1,0} %convolution.2902), dimensions={3}, metadata={op_type="ConcatV2" op_name="concat_15"}
  %arg43.44 = f32[1024]{0} parameter(43), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.313 = f32[1024]{0} reshape(f32[1024]{0} %arg43.44)
  %constant.2789 = f32[] constant(2e-05), metadata={op_type="AddV2" op_name="stage4_unit3_bn2/add"}
  %broadcast.2790 = f32[1024]{0} broadcast(f32[] %constant.2789), dimensions={}, metadata={op_type="AddV2" op_name="stage4_unit3_bn2/add"}
  %add.2791 = f32[1024]{0} add(f32[1024]{0} %reshape.313, f32[1024]{0} %broadcast.2790), metadata={op_type="AddV2" op_name="stage4_unit3_bn2/add"}
  %rsqrt.2792 = f32[1024]{0} rsqrt(f32[1024]{0} %add.2791), metadata={op_type="Rsqrt" op_name="stage4_unit3_bn2/Rsqrt"}
  %arg103.104 = f32[1024]{0} parameter(103), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.373 = f32[1024]{0} reshape(f32[1024]{0} %arg103.104)
  %multiply.2793 = f32[1024]{0} multiply(f32[1024]{0} %rsqrt.2792, f32[1024]{0} %reshape.373), metadata={op_type="Mul" op_name="stage4_unit3_bn2/mul"}
  %broadcast.2910 = f32[1,7,7,1024]{3,2,1,0} broadcast(f32[1024]{0} %multiply.2793), dimensions={3}, metadata={op_type="Mul" op_name="stage4_unit3_bn2/mul_1"}
  %multiply.2911 = f32[1,7,7,1024]{3,2,1,0} multiply(f32[1,7,7,1024]{3,2,1,0} %concatenate.2909, f32[1,7,7,1024]{3,2,1,0} %broadcast.2910), metadata={op_type="Mul" op_name="stage4_unit3_bn2/mul_1"}
  %arg210.211 = f32[1024]{0} parameter(210), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.480 = f32[1024]{0} reshape(f32[1024]{0} %arg210.211)
  %arg157.158 = f32[1024]{0} parameter(157), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.427 = f32[1024]{0} reshape(f32[1024]{0} %arg157.158)
  %multiply.2794 = f32[1024]{0} multiply(f32[1024]{0} %multiply.2793, f32[1024]{0} %reshape.427), metadata={op_type="Mul" op_name="stage4_unit3_bn2/mul_2"}
  %subtract.2795 = f32[1024]{0} subtract(f32[1024]{0} %reshape.480, f32[1024]{0} %multiply.2794), metadata={op_type="Sub" op_name="stage4_unit3_bn2/sub"}
  %broadcast.2912 = f32[1,7,7,1024]{3,2,1,0} broadcast(f32[1024]{0} %subtract.2795), dimensions={3}, metadata={op_type="AddV2" op_name="stage4_unit3_bn2/add_1"}
  %add.2913 = f32[1,7,7,1024]{3,2,1,0} add(f32[1,7,7,1024]{3,2,1,0} %multiply.2911, f32[1,7,7,1024]{3,2,1,0} %broadcast.2912), metadata={op_type="AddV2" op_name="stage4_unit3_bn2/add_1"}
  %maximum.2916 = f32[1,7,7,1024]{3,2,1,0} maximum(f32[1,7,7,1024]{3,2,1,0} %broadcast.2915, f32[1,7,7,1024]{3,2,1,0} %add.2913), metadata={op_type="Relu" op_name="stage4_unit3_relu2"}
  %arg267.268 = f32[1,1,1024,2048]{3,2,1,0} parameter(267), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.537 = f32[1,1,1024,2048]{3,2,1,0} reshape(f32[1,1,1024,2048]{3,2,1,0} %arg267.268)
  %convolution.2917 = f32[1,7,7,2048]{3,2,1,0} convolution(f32[1,7,7,1024]{3,2,1,0} %maximum.2916, f32[1,1,1024,2048]{3,2,1,0} %reshape.537), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="stage4_unit3_conv3"}
  %multiply.2919 = f32[1,7,7,2048]{3,2,1,0} multiply(f32[1,7,7,2048]{3,2,1,0} %broadcast.2918, f32[1,7,7,2048]{3,2,1,0} %convolution.2917), metadata={op_type="Mul" op_name="stage4_unit3_bn3/mul_1"}
  %arg226.227 = f32[2048]{0} parameter(226), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.496 = f32[2048]{0} reshape(f32[2048]{0} %arg226.227)
  %arg173.174 = f32[2048]{0} parameter(173), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.443 = f32[2048]{0} reshape(f32[2048]{0} %arg173.174)
  %multiply.2801 = f32[2048]{0} multiply(f32[2048]{0} %multiply.2800, f32[2048]{0} %reshape.443), metadata={op_type="Mul" op_name="stage4_unit3_bn3/mul_2"}
  %subtract.2802 = f32[2048]{0} subtract(f32[2048]{0} %reshape.496, f32[2048]{0} %multiply.2801), metadata={op_type="Sub" op_name="stage4_unit3_bn3/sub"}
  %broadcast.2920 = f32[1,7,7,2048]{3,2,1,0} broadcast(f32[2048]{0} %subtract.2802), dimensions={3}, metadata={op_type="AddV2" op_name="stage4_unit3_bn3/add_1"}
  %add.2921 = f32[1,7,7,2048]{3,2,1,0} add(f32[1,7,7,2048]{3,2,1,0} %multiply.2919, f32[1,7,7,2048]{3,2,1,0} %broadcast.2920), metadata={op_type="AddV2" op_name="stage4_unit3_bn3/add_1"}
  %add.2922 = f32[1,7,7,2048]{3,2,1,0} add(f32[1,7,7,2048]{3,2,1,0} %maximum.2781, f32[1,7,7,2048]{3,2,1,0} %add.2921), metadata={op_type="AddV2" op_name="add_15"}
  %maximum.2925 = f32[1,7,7,2048]{3,2,1,0} maximum(f32[1,7,7,2048]{3,2,1,0} %broadcast.2924, f32[1,7,7,2048]{3,2,1,0} %add.2922), metadata={op_type="Relu" op_name="stage4_unit3_relu"}
  %convert.2926 = f32[1,7,7,2048]{3,2,1,0} convert(f32[1,7,7,2048]{3,2,1,0} %maximum.2925), metadata={op_type="AvgPool" op_name="pool1"}
  %constant.2928 = f32[] constant(0), metadata={op_type="AvgPool" op_name="pool1"}
  %pad.2929 = f32[1,7,7,2048]{3,2,1,0} pad(f32[1,7,7,2048]{3,2,1,0} %convert.2926, f32[] %constant.2928), padding=0_0x0_0x0_0x0_0, metadata={op_type="AvgPool" op_name="pool1"}
  %constant.2927 = f32[] constant(0), metadata={op_type="AvgPool" op_name="pool1"}
  %reduce-window.2934 = f32[1,1,1,2048]{3,2,1,0} reduce-window(f32[1,7,7,2048]{3,2,1,0} %pad.2929, f32[] %constant.2927), window={size=1x7x7x1}, to_apply=%add_F32.2930, metadata={op_type="AvgPool" op_name="pool1"}
  %constant.2935 = f32[] constant(49), metadata={op_type="AvgPool" op_name="pool1"}
  %broadcast.2936 = f32[1,1,1,2048]{3,2,1,0} broadcast(f32[] %constant.2935), dimensions={}, metadata={op_type="AvgPool" op_name="pool1"}
  %divide.2937 = f32[1,1,1,2048]{3,2,1,0} divide(f32[1,1,1,2048]{3,2,1,0} %reduce-window.2934, f32[1,1,1,2048]{3,2,1,0} %broadcast.2936), metadata={op_type="AvgPool" op_name="pool1"}
  %convert.2938 = f32[1,1,1,2048]{3,2,1,0} convert(f32[1,1,1,2048]{3,2,1,0} %divide.2937), metadata={op_type="AvgPool" op_name="pool1"}
  %get-dimension-size.2939 = s32[] get-dimension-size(f32[1,1,1,2048]{3,2,1,0} %convert.2938), dimensions={0}, metadata={op_type="Shape" op_name="Flatten/flatten/Shape"}
  %convert.2940 = s32[] convert(s32[] %get-dimension-size.2939), metadata={op_type="Shape" op_name="Flatten/flatten/Shape"}
  %broadcast.2941 = s32[1]{0} broadcast(s32[] %convert.2940), dimensions={}, metadata={op_type="Shape" op_name="Flatten/flatten/Shape"}
  %get-dimension-size.2942 = s32[] get-dimension-size(f32[1,1,1,2048]{3,2,1,0} %convert.2938), dimensions={1}, metadata={op_type="Shape" op_name="Flatten/flatten/Shape"}
  %convert.2943 = s32[] convert(s32[] %get-dimension-size.2942), metadata={op_type="Shape" op_name="Flatten/flatten/Shape"}
  %broadcast.2944 = s32[1]{0} broadcast(s32[] %convert.2943), dimensions={}, metadata={op_type="Shape" op_name="Flatten/flatten/Shape"}
  %get-dimension-size.2945 = s32[] get-dimension-size(f32[1,1,1,2048]{3,2,1,0} %convert.2938), dimensions={2}, metadata={op_type="Shape" op_name="Flatten/flatten/Shape"}
  %convert.2946 = s32[] convert(s32[] %get-dimension-size.2945), metadata={op_type="Shape" op_name="Flatten/flatten/Shape"}
  %broadcast.2947 = s32[1]{0} broadcast(s32[] %convert.2946), dimensions={}, metadata={op_type="Shape" op_name="Flatten/flatten/Shape"}
  %get-dimension-size.2948 = s32[] get-dimension-size(f32[1,1,1,2048]{3,2,1,0} %convert.2938), dimensions={3}, metadata={op_type="Shape" op_name="Flatten/flatten/Shape"}
  %convert.2949 = s32[] convert(s32[] %get-dimension-size.2948), metadata={op_type="Shape" op_name="Flatten/flatten/Shape"}
  %broadcast.2950 = s32[1]{0} broadcast(s32[] %convert.2949), dimensions={}, metadata={op_type="Shape" op_name="Flatten/flatten/Shape"}
  %concatenate.2951 = s32[4]{0} concatenate(s32[1]{0} %broadcast.2941, s32[1]{0} %broadcast.2944, s32[1]{0} %broadcast.2947, s32[1]{0} %broadcast.2950), dimensions={0}, metadata={op_type="Shape" op_name="Flatten/flatten/Shape"}
  %slice.2952 = s32[1]{0} slice(s32[4]{0} %concatenate.2951), slice={[0:1]}, metadata={op_type="StridedSlice" op_name="Flatten/flatten/strided_slice"}
  %reshape.2953 = s32[] reshape(s32[1]{0} %slice.2952), metadata={op_type="StridedSlice" op_name="Flatten/flatten/strided_slice"}
  %reshape.2955 = s32[1]{0} reshape(s32[] %reshape.2953), metadata={op_type="Pack" op_name="Flatten/flatten/Reshape/shape"}
  %constant.2954 = s32[] constant(-1), metadata={op_type="Pack" op_name="Flatten/flatten/Reshape/shape"}
  %reshape.2956 = s32[1]{0} reshape(s32[] %constant.2954), metadata={op_type="Pack" op_name="Flatten/flatten/Reshape/shape"}
  %concatenate.2957 = s32[2]{0} concatenate(s32[1]{0} %reshape.2955, s32[1]{0} %reshape.2956), dimensions={0}, metadata={op_type="Pack" op_name="Flatten/flatten/Reshape/shape"}
  %reshape.2959 = f32[1,2048]{1,0} reshape(f32[1,1,1,2048]{3,2,1,0} %convert.2938), inferred_dimension=1, metadata={op_type="Reshape" op_name="Flatten/flatten/Reshape"}
  %reshape.2960 = f32[1,2048]{1,0} reshape(f32[1,2048]{1,0} %reshape.2959), metadata={op_name="XLA_Retvals"}
  %tuple.2961 = (f32[1,2048]{1,0}) tuple(f32[1,2048]{1,0} %reshape.2960), metadata={op_name="XLA_Retvals"}
  ROOT %get-tuple-element.2962 = f32[1,2048]{1,0} get-tuple-element((f32[1,2048]{1,0}) %tuple.2961), index=0, metadata={op_name="XLA_Retvals"}
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
