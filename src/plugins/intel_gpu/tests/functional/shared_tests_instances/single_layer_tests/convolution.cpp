// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/convolution.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

class MyConvolutionLayerTest : public ConvolutionLayerTest {
public:
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 20, -10, 1, 1);
    }

protected:
    void SetUp() override {
        ConvolutionLayerTest::SetUp();

        auto params = this->GetParam();
        auto netPrecision = std::get<1>(params);
        if (netPrecision.getPrecVal() == InferenceEngine::Precision::FP16) {
            this->threshold = 0.5;
        }
    }
};

TEST_P(MyConvolutionLayerTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run();
}

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

/* ============= 2D Convolution ============= */
const std::vector<std::vector<size_t >> kernels = {{3, 3},
                                                   {3, 5}};
const std::vector<std::vector<size_t >> strides = {{1, 1},
                                                   {1, 3}};
const std::vector<std::vector<ptrdiff_t>> padBegins = {{0, 0},
                                                       {0, 3}};
const std::vector<std::vector<ptrdiff_t>> padEnds = {{0, 0},
                                                     {0, 3}};
const std::vector<std::vector<size_t >> dilations = {{1, 1},
                                                     {3, 1}};
const std::vector<size_t> numOutChannels = {1, 5};
const std::vector<ngraph::op::PadType> padTypes = {
        ngraph::op::PadType::EXPLICIT,
        ngraph::op::PadType::VALID
};
const auto conv2DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels),
        ::testing::ValuesIn(strides),
        ::testing::ValuesIn(padBegins),
        ::testing::ValuesIn(padEnds),
        ::testing::ValuesIn(dilations),
        ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);
const auto conv2DParams_AutoPadValid = ::testing::Combine(
        ::testing::ValuesIn(kernels),
        ::testing::ValuesIn(strides),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::ValuesIn(dilations),
        ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ngraph::op::PadType::VALID)
);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_ExplicitPadding, ConvolutionLayerTest,
                         ::testing::Combine(
                                 conv2DParams_ExplicitPadding,
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                 ::testing::Values(InferenceEngine::Layout::ANY),
                                 ::testing::Values(InferenceEngine::Layout::ANY),
                                 ::testing::Values(std::vector<size_t >({1, 3, 30, 30})),
                                 ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                         ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_AutoPadValid, ConvolutionLayerTest,
                         ::testing::Combine(
                                 conv2DParams_AutoPadValid,
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                 ::testing::Values(InferenceEngine::Layout::ANY),
                                 ::testing::Values(InferenceEngine::Layout::ANY),
                                 ::testing::Values(std::vector<size_t >({1, 3, 30, 30})),
                                 ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                         ConvolutionLayerTest::getTestCaseName);
/* ============= 3D Convolution ============= */
const std::vector<std::vector<size_t >> kernels3d = {{3, 3, 3},
                                                     {3, 5, 3}};

const std::vector<std::vector<ptrdiff_t>> paddings3d = {{0, 0, 0},
                                                        {0, 2, 0}};

const std::vector<std::vector<size_t >> strides3d = {{1, 1, 1},
                                                     {1, 2, 1}};

const std::vector<std::vector<size_t >> dilations3d = { {1, 1, 1} };

const std::vector<size_t > numOutChannels3d = {1, 5, 16};

const auto conv3DParams = ::testing::Combine(
        ::testing::ValuesIn(kernels3d),
        ::testing::ValuesIn(strides3d),
        ::testing::ValuesIn(paddings3d),
        ::testing::ValuesIn(paddings3d),
        ::testing::ValuesIn(dilations3d),
        ::testing::ValuesIn(numOutChannels3d),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution3D_Basic1, ConvolutionLayerTest,
                         ::testing::Combine(
                                 conv3DParams,
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                 ::testing::Values(InferenceEngine::Layout::ANY),
                                 ::testing::Values(InferenceEngine::Layout::ANY),
                                 ::testing::Values(std::vector<size_t >({1, 3, 10, 10, 10})),
                                 ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                         ConvolutionLayerTest::getTestCaseName);


/* ======================================================================== */

// Attrs:  {'auto_pad': 'explicit', 'dilations': '1', 'pads_begin': '2', 'pads_end': '2', 'strides': '1'}
// In:     (1, 512, 1000), (80, 512, 5)
// Out:    (1, 80, 1000)
// Operators: 'Tacotron2-graph-transform-cuda-postnet:opid22' [FP32], 'Tacotron2-postnet:opid22' [FP32]
INSTANTIATE_TEST_CASE_P(
    Convolution_Tacotron2_graph_transform_cuda_postnet_opid22,
    MyConvolutionLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({5})), // kernel
            ::testing::Values(std::vector<size_t>({1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({2})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({2})), // pads_end
            ::testing::Values(std::vector<size_t>({1})), // dilations
            ::testing::Values(80), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 1000})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    MyConvolutionLayerTest::getTestCaseName);

// Attrs:  {'auto_pad': 'explicit', 'dilations': '1', 'pads_begin': '0', 'pads_end': '0', 'strides': '1'}
// In:     (1, 16, 32), (8, 16, 5)
// Out:    (1, 8, 32)
INSTANTIATE_TEST_CASE_P(
    tst001,
    MyConvolutionLayerTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({5})), // kernel
            ::testing::Values(std::vector<size_t>({1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0})), // pads_end
            ::testing::Values(std::vector<size_t>({1})), // dilations
            ::testing::Values(8), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 16, 32})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    MyConvolutionLayerTest::getTestCaseName);

}  // namespace
