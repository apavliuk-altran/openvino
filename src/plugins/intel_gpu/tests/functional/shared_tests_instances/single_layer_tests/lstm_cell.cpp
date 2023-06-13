// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/lstm_cell.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
std::vector<bool> should_decompose{false, true};
std::vector<size_t> batch{5};
std::vector<size_t> hidden_size{1, 10};
std::vector<size_t> hidden_size_smoke{1};
std::vector<size_t> input_size{1, 30};
std::vector<std::vector<std::string>> activations_smoke = {{"relu", "sigmoid", "tanh"}};
std::vector<std::vector<std::string>> activations = {{"relu", "sigmoid", "tanh"}, {"sigmoid", "tanh", "tanh"},
                                                     {"tanh", "relu", "sigmoid"}, {"sigmoid", "sigmoid", "sigmoid"},
                                                     {"tanh", "tanh", "tanh"}, {"relu", "relu", "relu"}};
std::vector<float> clip{0.f, 0.7f};
std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                        InferenceEngine::Precision::FP16};

INSTANTIATE_TEST_SUITE_P(LSTMCellCommon, LSTMCellTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(should_decompose),
                                ::testing::ValuesIn(batch),
                                ::testing::ValuesIn(hidden_size),
                                ::testing::ValuesIn(input_size),
                                ::testing::ValuesIn(activations),
                                ::testing::ValuesIn(clip),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        LSTMCellTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_LSTMCellCommon, LSTMCellTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(should_decompose),
                                ::testing::ValuesIn(batch),
                                ::testing::ValuesIn(hidden_size_smoke),
                                ::testing::ValuesIn(input_size),
                                ::testing::ValuesIn(activations_smoke),
                                ::testing::ValuesIn(clip),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        LSTMCellTest::getTestCaseName);


class CUDNNLSTMCellTest : public LSTMCellTest {
public:
    void SetUp() override {
        LSTMCellTest::SetUp();
        constexpr float up_to = 5.0f;
        constexpr float start_from = -5.0f;

        // const auto& netPrecision = std::get<InferenceEngine::Precision>(this->GetParam());
        const auto& netPrecision = std::get<6>(this->GetParam());
        if (netPrecision == InferenceEngine::Precision::FP16) {
            this->threshold = 0.12f;
        }

        const auto& ops = function->get_ordered_ops();
        int seed = 1;
        for (const auto& op : ops) {
            if (std::dynamic_pointer_cast<ngraph::opset1::Constant>(op)) {
                const auto constant = ngraph::builder::makeConstant(
                    op->get_element_type(), op->get_shape(), std::vector<float>{}, true, up_to, start_from, seed);
                function->replace_node(op, constant);
            }
        }
    }

    void ConvertRefsParams() override {}
};


TEST_P(CUDNNLSTMCellTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run();
};

const bool should_decompose1 = false;
const std::vector<std::string> activations1{"sigmoid", "tanh", "tanh"};
const std::vector<InferenceEngine::Precision> netPrecisions1 = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

struct LSTMCellTestParams {
    size_t batch;
    size_t input_size;
    size_t hidden_size;
    float clip;
};

LSTMCellTestParams tacotron2_dec_02{1, 1536, 1024, 0.0f};

INSTANTIATE_TEST_CASE_P(LSTMCell_Tacotron2_dec_02,
                        CUDNNLSTMCellTest,
                        ::testing::Combine(::testing::Values(should_decompose1),
                                           ::testing::Values(tacotron2_dec_02.batch),
                                           ::testing::Values(tacotron2_dec_02.hidden_size),
                                           ::testing::Values(tacotron2_dec_02.input_size),
                                           ::testing::Values(activations1),
                                           ::testing::Values(tacotron2_dec_02.clip),
                                           ::testing::ValuesIn(netPrecisions1),
                                           ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        CUDNNLSTMCellTest::getTestCaseName);

} // namespace