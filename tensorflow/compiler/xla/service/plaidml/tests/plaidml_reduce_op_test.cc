// Tests that show HLO Module conversion to PlaidML Program.

#include <algorithm>
#include <string>

#include <gtest/gtest.h>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/service/plaidml/compiler.h"
#include "tensorflow/compiler/xla/service/plaidml/tests/plaidml_codegen_test.h"
#include "tensorflow/compiler/xla/service/plaidml/tests/plaidml_reduce_op_test.h.inc"
#include "tensorflow/compiler/xla/service/plaidml/tests/plaidml_redwin_op_test.h.inc"
#include "tensorflow/compiler/xla/tests/verified_hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
using ::plaidml::edsl::TensorBuffers;

namespace xla {
namespace plaidml {
namespace {

using TestCaseVal = std::vector<std::vector<float>>;
using TestCasePairs = std::map<TestCaseVal, TestCaseVal>;

struct ReduceTestSpec {
  PrimitiveType primitive_type;
  string filecheck_lines;
};

string ReduceTestSpecToString(const ::testing::TestParamInfo<ReduceTestSpec>& info) {
  return PrimitiveType_Name(info.param.primitive_type);
}

class PlaidMLReduceOperationTest
    : public PlaidMLCodegenTest,
      public ::testing::WithParamInterface<ReduceTestSpec> {
 protected:

  Status CompileAndCheck(std::unique_ptr<HloModule> hlo_module,
                         const string& filecheck_lines,
                         const TestCasePairs& testcase_pairs) {

    auto program = CompileToProgram(std::move(hlo_module));

    StatusOr<bool> fc_result = RunFileCheck(program->str(), filecheck_lines);

    //TF_ASSERT_OK(fc_result.status());
    EXPECT_TRUE(fc_result.ValueOrDie());

    VLOG(2) << "Evaluating results";

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

      VLOG(2) << "Calling checkProgram";

      checkProgram(*program, inp, exp);

    }
    return Status::OK();
  }
};

TEST_P(PlaidMLReduceOperationTest, ReduceTest){
  VLOG(0) << "Testing generated examples";

  for (std::size_t i = 0; i < reduce_modules.size(); ++i) {
    std::string set_des = reduce_descriptions[i];
    VLOG(0) << "Testing set " << i << ": " << set_des;
    std::vector<float> input_val = reduce_is[i];
    std::vector<float> expected_val = reduce_os[i];
    std::string module_text = reduce_modules[i];

    TestCaseVal inputs;
    if (set_des.find('AVG') != std::string::npos) {
      std::vector<float> avgpool = {0};
      inputs = {avgpool, input_val};
    }else{
      inputs = {input_val};
    }
    TestCaseVal results = {expected_val};
    
    TestCasePairs testcase_pairs ={{inputs, results}};

    ReduceTestSpec spec = GetParam();

    HloModuleConfig cfg;

    std::unique_ptr<VerifiedHloModule> hlo_module = absl::make_unique<VerifiedHloModule>(
      "module", cfg, false, false, nullptr);

    hlo_module->ParseHloStringAndVerifyModule(module_text); 

    CompileAndCheck(std::move(hlo_module), spec.filecheck_lines, testcase_pairs);
  }
}

TEST_P(PlaidMLReduceOperationTest, RedWinTest){
  VLOG(0) << "Testing generated examples";

  for (std::size_t i = 0; i < redwin_modules.size(); ++i) {
    std::string set_des = redwin_descriptions[i];
    VLOG(0) << "Testing set " << i << ": " << set_des;
    std::vector<float> input_val = redwin_is[i];
    std::vector<float> expected_val = redwin_os[i];
    std::string module_text = redwin_modules[i];

    TestCaseVal inputs;
    if (set_des.find("AVG") != std::string::npos) {
      std::vector<float> zer = {2};
      inputs = {zer, input_val};
    }else{
      inputs = {input_val};
    }
    TestCaseVal results = {expected_val};
    
    TestCasePairs testcase_pairs ={{inputs, results}};

    ReduceTestSpec spec = GetParam();

    HloModuleConfig cfg;

    std::unique_ptr<VerifiedHloModule> hlo_module = absl::make_unique<VerifiedHloModule>(
      "module", cfg, false, false, nullptr);

    hlo_module->ParseHloStringAndVerifyModule(module_text); 

    CompileAndCheck(std::move(hlo_module), spec.filecheck_lines, testcase_pairs);
  }
}

std::vector<ReduceTestSpec> GetReduceTestCases() {
  std::vector<ReduceTestSpec> result;
  auto check_str = R"#(
                        CHECK: func @hlo_module{{.*}}%[[I:.*]]: tensor<[[is:.*]]x[[prec:.*]]>) -> tensor<[[os:.*]]x[[prec]]> {
                        CHECK: return %{{.*}} : tensor<[[os.*]]x[[prec]]>
                    )#";
  result.push_back(
      {F32, check_str});
  result.push_back(
      {F64, check_str});
  return result;
}

/**/
// TODO: INSTANTIATE_TEST_CASE_P was deprecated in favor for INSTANTIATE_TEST_SUITE_P, but the version of gtest that bazel links in is looking for INSTANTIATE_TEST_CASE_P right now.
INSTANTIATE_TEST_CASE_P(All,
                         PlaidMLReduceOperationTest,
                         ::testing::ValuesIn(GetReduceTestCases()),
                         ReduceTestSpecToString);
/**/
}  // namespace
}  // namespace plaidml
}  // namespace xla
