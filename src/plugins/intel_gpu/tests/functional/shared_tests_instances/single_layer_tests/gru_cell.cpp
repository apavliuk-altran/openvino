// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/gru_cell.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
    std::vector<bool> should_decompose{false, true};
    std::vector<size_t> batch{5};
    std::vector<size_t> hidden_size{1, 10};
    std::vector<size_t> input_size{1, 30};
    std::vector<std::vector<std::string>> activations = {{"relu", "tanh"}, {"tanh", "sigmoid"}, {"sigmoid", "tanh"},
                                                         {"tanh", "relu"}};
    std::vector<float> clip = {0.0f, 0.7f};
    std::vector<bool> linear_before_reset = {true, false};
    std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                             InferenceEngine::Precision::FP16};

    INSTANTIATE_TEST_SUITE_P(GRUCellCommon, GRUCellTest,
            ::testing::Combine(
            ::testing::ValuesIn(should_decompose),
            ::testing::ValuesIn(batch),
            ::testing::ValuesIn(hidden_size),
            ::testing::ValuesIn(input_size),
            ::testing::ValuesIn(activations),
            ::testing::ValuesIn(clip),
            ::testing::ValuesIn(linear_before_reset),
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_GPU)),
            GRUCellTest::getTestCaseName);


class CUDNNGRUCellTest : public GRUCellTest {
protected:
    void SetUp() override {
        GRUCellTest::SetUp();

        const auto& netPrecision = std::get<7>(this->GetParam());
        if (netPrecision == InferenceEngine::Precision::FP16) {
            this->threshold = 0.18f;
        }

        constexpr float up_to = 1.5f;
        constexpr float start_from = -1.5f;
        int seed = 1;

        const auto& ops = function->get_ordered_ops();
        for (const auto& op : ops) {
            if (std::dynamic_pointer_cast<ngraph::opset1::Constant>(op)) {
                const auto constant = ngraph::builder::makeConstant(
                    op->get_element_type(), op->get_shape(), std::vector<float>{}, true, up_to, start_from, seed);
                function->replace_node(op, constant);
            }
        }
    }
};

TEST_P(CUDNNGRUCellTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run();
}

const bool should_decompose1 = false;
const std::vector<std::string> activations1{"sigmoid", "tanh"};

const std::vector<float> clips1{0.0f};

const std::vector<bool> linear_before_reset1{true};

const std::vector<InferenceEngine::Precision> net_precisions1 = {InferenceEngine::Precision::FP32,
                                                                InferenceEngine::Precision::FP16};
const size_t lpc_batch = 64;
const std::vector<size_t> lpc_hidden_sizes = {16, 384};
const size_t lpc_input_size = 512;

INSTANTIATE_TEST_CASE_P(GRUCell_LPCNet,
                        CUDNNGRUCellTest,
                        ::testing::Combine(::testing::Values(should_decompose1),
                                           ::testing::Values(lpc_batch),
                                           ::testing::ValuesIn(lpc_hidden_sizes),
                                           ::testing::Values(lpc_input_size),
                                           ::testing::Values(activations1),
                                           ::testing::ValuesIn(clips1),
                                           ::testing::ValuesIn(linear_before_reset1),
                                           ::testing::ValuesIn(net_precisions1),
                                           ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        GRUCellTest::getTestCaseName);

}  // namespace
