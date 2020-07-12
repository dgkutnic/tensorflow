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
using ::plaidml::edsl::TensorBuffers;

namespace xla {
namespace plaidml {
namespace {

using TestCaseVal = std::vector<std::vector<float>>;
using TestCasePairs = std::map<TestCaseVal, TestCaseVal>;

struct ReduceWindowTestSpec {
  PrimitiveType primitive_type;
  string filecheck_lines;
};

string ReduceWindowTestSpecToString(const ::testing::TestParamInfo<ReduceWindowTestSpec>& info) {
  return PrimitiveType_Name(info.param.primitive_type);
}

class PlaidMLReduceWindowOperationTest
    : public PlaidMLCodegenTest,
      public ::testing::WithParamInterface<ReduceWindowTestSpec> {
 protected:
  Status CompileAndCheck(std::unique_ptr<HloComputation> entry_computation,
                         const string& filecheck_lines,
                         const TestCasePairs& testcase_pairs) {
    HloModuleConfig cfg;

    std::unique_ptr<HloModule> hlo_module = absl::make_unique<HloModule>("module", cfg);
    hlo_module->AddEntryComputation(std::move(entry_computation));

    auto program = CompileToProgram(std::move(hlo_module));

    VLOG(2) << "Program:\n" << program->str();

    StatusOr<bool> fc_result = RunFileCheck(program->str(), filecheck_lines);

    //TF_ASSERT_OK(fc_result.status());
    EXPECT_TRUE(fc_result.ValueOrDie());

    VLOG(2) << "Evaluating results";

    for (auto pair : testcase_pairs) {

      TensorBuffers inp;
      TensorBuffers exp;

      auto program_inputs = program->inputs();

      for (auto i = 0; i < program_inputs.size(); i++) {
        inp.insert(std::make_pair(program_inputs[i].tensor, pair.first[i]));
      }

      auto program_outputs = program->outputs();

      for (auto i = 0; i < program_outputs.size(); i++) {
        exp.insert(std::make_pair(program_outputs[i].tensor, pair.second[i]));
      }

      checkProgram(*program, inp, exp);

    }

    return Status::OK();

  }
};

TEST_P(PlaidMLReduceWindowOperationTest, SimpleReduceWindowOp) {
  std::vector<float> input_val = {0.05,  0.05,  0.05,  0.05,  0.05,  //
                                  0.025, 0.025, 0.025, 0.025, 0.025, //
                                  0.01,  0.01,  0.01,  0.01,  0.01,  //
                                  0.025, 0.025, 0.025, 0.025, 0.025, //
                                  0.05,  0.05,  0.05,  0.05,  0.05};
  std::vector<float> kernel_val = {1, 1, 1, //
                                   1, 0, 1, //
                                   1, 1, 1};
  std::vector<float> expected_val = {0.23, 0.23, 0.23, //
                                     0.17, 0.17, 0.17, //
                                     0.23, 0.23, 0.23};

  TestCaseVal inputs = {kernel_val, input_val};
  TestCaseVal results = {expected_val};
  TestCasePairs testcase_pairs = {{inputs, results}};

  HloComputation::Builder builder("SimpleReduceWindowOp");
  ReduceWindowTestSpec spec = GetParam();

  auto input_shape = ShapeUtil::MakeShape(spec.primitive_type, {1, 5, 5, 1});
  auto kernel_shape = ShapeUtil::MakeShape(spec.primitive_type, {3, 3, 1, 1});

  HloInstruction* input = builder.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "input"));
  HloInstruction* kernel = builder.AddInstruction(
      HloInstruction::CreateParameter(1, kernel_shape, "input"));

  auto result_shape = Shape();
  result_shape.set_element_type(spec.primitive_type);
  result_shape.add_dimensions(1);
  result_shape.add_dimensions(219);
  result_shape.add_dimensions(219);
  result_shape.add_dimensions(64);
  Window window;
  WindowDimension* dim_1 = window.add_dimensions();
  dim_1->set_size(1);
  dim_1->set_padding_low(0);
  dim_1->set_padding_high(0);
  dim_1->set_stride(1);
  dim_1->set_window_dilation(1);
  dim_1->set_base_dilation(1);
  dim_1->set_window_reversal(false);
  WindowDimension* dim_2 = window.add_dimensions();
  dim_2->set_size(2);
  dim_2->set_padding_low(0);
  dim_2->set_padding_high(0);
  dim_2->set_stride(1);
  dim_2->set_window_dilation(1);
  dim_2->set_base_dilation(1);
  dim_2->set_window_reversal(false);
  WindowDimension* dim_3 = window.add_dimensions();
  dim_3->set_size(2);
  dim_3->set_padding_low(0);
  dim_3->set_padding_high(0);
  dim_3->set_stride(1);
  dim_3->set_window_dilation(1);
  dim_3->set_base_dilation(1);
  dim_3->set_window_reversal(false);
  WindowDimension* dim_4 = window.add_dimensions();
  dim_4->set_size(1);
  dim_4->set_padding_low(0);
  dim_4->set_padding_high(0);
  dim_4->set_stride(1);
  dim_4->set_window_dilation(1);
  dim_4->set_base_dilation(1);
  dim_4->set_window_reversal(false);
  auto reduce_window = builder.AddInstruction(HloInstruction::CreateReduceWindow(result_shape, input, constant_28, window, max_computation));

  CompileAndCheck(builder.Build(), spec.filecheck_lines, testcase_pairs);
}

std::vector<ReduceWindowTestSpec> GetReduceWindowTestCases() {
  std::vector<ReduceWindowTestSpec> result;
  result.push_back(
      {F32, R"(CHECK: func @hlo_module(%arg0: tensor<3x3x1x1xf32>, %arg1: tensor<1x5x5x1xf32>) -> tensor<1x3x3x1xf32>)"});
  result.push_back(
      {F64, R"(CHECK: func @hlo_module(%arg0: tensor<3x3x1x1xf32>, %arg1: tensor<1x5x5x1xf32>) -> tensor<1x3x3x1xf32>)"});
  return result;
}

/**/
// TODO: INSTANTIATE_TEST_CASE_P was deprecated in favor for INSTANTIATE_TEST_SUITE_P, but the version of gtest that bazel links in is looking for INSTANTIATE_TEST_CASE_P right now.
INSTANTIATE_TEST_CASE_P(All,
                         PlaidMLReduceWindowOperationTest,
                         ::testing::ValuesIn(GetReduceWindowTestCases()),
                         ReduceWindowTestSpecToString);
/**/
}  // namespace
}  // namespace plaidml
}  // namespace xla
