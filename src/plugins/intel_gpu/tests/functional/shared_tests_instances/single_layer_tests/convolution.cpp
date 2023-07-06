// Copyright (C) 2019-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/convolution.hpp"

// #include <cuda_test_constants.hpp>
#include <vector>

#include "common_test_utils/test_constants.hpp"
// #include "finite_comparer.hpp"

using namespace LayerTestsDefinitions;

namespace {

class ConvolutionCudaLayerThresholdTest : public ConvolutionLayerTest {
public:
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 20, -10, 1, 1);
    }

protected:
    void SetUp() override {
        ConvolutionLayerTest::SetUp();
    }
};

TEST_P(ConvolutionCudaLayerThresholdTest, CudaCompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run();
}

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::FP32,
};

/* ============= 1D Convolution ============= */
const std::vector<std::vector<size_t>> kernels1D = {{3}, {5}};
const std::vector<std::vector<size_t>> strides1D = {{1}, {3}};
const std::vector<std::vector<size_t>> dilations1D = {{1}, {3}};
const std::vector<size_t> numOutChannels1D = {1, 5};

const auto conv1DParams_ExplicitPaddingSymmetric1 =
    ::testing::Combine(::testing::ValuesIn(kernels1D),
                       ::testing::ValuesIn(strides1D),
                       ::testing::Values(std::vector<ptrdiff_t>({0})),  // pads_begin
                       ::testing::Values(std::vector<ptrdiff_t>({0})),  // pads_end
                       ::testing::ValuesIn(dilations1D),
                       ::testing::ValuesIn(numOutChannels1D),
                       ::testing::Values(ov::op::PadType::EXPLICIT));
const auto conv1DParams_ExplicitPaddingSymmetric2 =
    ::testing::Combine(::testing::ValuesIn(kernels1D),
                       ::testing::ValuesIn(strides1D),
                       ::testing::Values(std::vector<ptrdiff_t>({3})),  // pads_begin
                       ::testing::Values(std::vector<ptrdiff_t>({3})),  // pads_end
                       ::testing::ValuesIn(dilations1D),
                       ::testing::ValuesIn(numOutChannels1D),
                       ::testing::Values(ov::op::PadType::EXPLICIT));
const auto conv1DParams_ExplicitPaddingAsymmetric1 =
    ::testing::Combine(::testing::ValuesIn(kernels1D),
                       ::testing::ValuesIn(strides1D),
                       ::testing::Values(std::vector<ptrdiff_t>({0})),  // pads_begin
                       ::testing::Values(std::vector<ptrdiff_t>({3})),  // pads_end
                       ::testing::ValuesIn(dilations1D),
                       ::testing::ValuesIn(numOutChannels1D),
                       ::testing::Values(ov::op::PadType::EXPLICIT));
const auto conv1DParams_ExplicitPaddingAsymmetric2 =
    ::testing::Combine(::testing::ValuesIn(kernels1D),
                       ::testing::ValuesIn(strides1D),
                       ::testing::Values(std::vector<ptrdiff_t>({3})),  // pads_begin
                       ::testing::Values(std::vector<ptrdiff_t>({0})),  // pads_end
                       ::testing::ValuesIn(dilations1D),
                       ::testing::ValuesIn(numOutChannels1D),
                       ::testing::Values(ov::op::PadType::EXPLICIT));
const auto conv1DParams_AutoPadValid = ::testing::Combine(::testing::ValuesIn(kernels1D),
                                                          ::testing::ValuesIn(strides1D),
                                                          ::testing::Values(std::vector<ptrdiff_t>({0})),
                                                          ::testing::Values(std::vector<ptrdiff_t>({0})),
                                                          ::testing::ValuesIn(dilations1D),
                                                          ::testing::ValuesIn(numOutChannels1D),
                                                          ::testing::Values(ov::op::PadType::VALID));

INSTANTIATE_TEST_CASE_P(smoke_Convolution1D_ExplicitPaddingSymmetric1,
                        ConvolutionCudaLayerThresholdTest,
                        ::testing::Combine(conv1DParams_ExplicitPaddingSymmetric1,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 3, 30})),
                                           ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        ConvolutionCudaLayerThresholdTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(smoke_Convolution1D_ExplicitPaddingSymmetric2,
                        ConvolutionCudaLayerThresholdTest,
                        ::testing::Combine(conv1DParams_ExplicitPaddingSymmetric2,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 3, 30})),
                                           ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        ConvolutionCudaLayerThresholdTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(smoke_Convolution1D_ExplicitPaddingAsymmetric1,
                        ConvolutionCudaLayerThresholdTest,
                        ::testing::Combine(conv1DParams_ExplicitPaddingAsymmetric1,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 3, 30})),
                                           ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        ConvolutionCudaLayerThresholdTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(smoke_Convolution1D_ExplicitPaddingAsymmetric2,
                        ConvolutionCudaLayerThresholdTest,
                        ::testing::Combine(conv1DParams_ExplicitPaddingAsymmetric2,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 3, 30})),
                                           ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        ConvolutionCudaLayerThresholdTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Convolution1D_AutoPadValid,
                        ConvolutionCudaLayerThresholdTest,
                        ::testing::Combine(conv1DParams_AutoPadValid,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 3, 30})),
                                           ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        ConvolutionCudaLayerThresholdTest::getTestCaseName);

/* ============= 2D Convolution ============= */
const std::vector<std::vector<size_t>> kernels = {{3, 3}, {3, 5}};
const std::vector<std::vector<size_t>> strides = {{1, 1}, {1, 3}};
const std::vector<std::vector<size_t>> dilations = {{1, 1}, {3, 1}};
const std::vector<size_t> numOutChannels = {1, 5};

const auto conv2DParams_ExplicitPaddingSymmetric1 =
    ::testing::Combine(::testing::ValuesIn(kernels),
                       ::testing::ValuesIn(strides),
                       ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  // pads_begin
                       ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  // pads_end
                       ::testing::ValuesIn(dilations),
                       ::testing::ValuesIn(numOutChannels),
                       ::testing::Values(ov::op::PadType::EXPLICIT));
const auto conv2DParams_ExplicitPaddingSymmetric2 =
    ::testing::Combine(::testing::ValuesIn(kernels),
                       ::testing::ValuesIn(strides),
                       ::testing::Values(std::vector<ptrdiff_t>({0, 3})),  // pads_begin
                       ::testing::Values(std::vector<ptrdiff_t>({0, 3})),  // pads_end
                       ::testing::ValuesIn(dilations),
                       ::testing::ValuesIn(numOutChannels),
                       ::testing::Values(ov::op::PadType::EXPLICIT));
const auto conv2DParams_ExplicitPaddingAsymmetric1 =
    ::testing::Combine(::testing::ValuesIn(kernels),
                       ::testing::ValuesIn(strides),
                       ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  // pads_begin
                       ::testing::Values(std::vector<ptrdiff_t>({0, 3})),  // pads_end
                       ::testing::ValuesIn(dilations),
                       ::testing::ValuesIn(numOutChannels),
                       ::testing::Values(ov::op::PadType::EXPLICIT));
const auto conv2DParams_ExplicitPaddingAsymmetric2 =
    ::testing::Combine(::testing::ValuesIn(kernels),
                       ::testing::ValuesIn(strides),
                       ::testing::Values(std::vector<ptrdiff_t>({0, 3})),  // pads_begin
                       ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  // pads_end
                       ::testing::ValuesIn(dilations),
                       ::testing::ValuesIn(numOutChannels),
                       ::testing::Values(ov::op::PadType::EXPLICIT));

const auto conv2DParams_AutoPadValid = ::testing::Combine(::testing::ValuesIn(kernels),
                                                          ::testing::ValuesIn(strides),
                                                          ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                                                          ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                                                          ::testing::ValuesIn(dilations),
                                                          ::testing::ValuesIn(numOutChannels),
                                                          ::testing::Values(ov::op::PadType::VALID));

INSTANTIATE_TEST_CASE_P(smoke_Convolution2D_ExplicitPaddingSymmetric1,
                        ConvolutionCudaLayerThresholdTest,
                        ::testing::Combine(conv2DParams_ExplicitPaddingSymmetric1,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
                                           ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        ConvolutionCudaLayerThresholdTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Convolution2D_ExplicitPaddingSymmetric2_FP32,
                        ConvolutionCudaLayerThresholdTest,
                        ::testing::Combine(conv2DParams_ExplicitPaddingSymmetric2,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
                                           ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        ConvolutionCudaLayerThresholdTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Convolution2D_ExplicitPaddingSymmetric2,
                        ConvolutionCudaLayerThresholdTest,
                        ::testing::Combine(conv2DParams_ExplicitPaddingSymmetric2,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
                                           ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        ConvolutionCudaLayerThresholdTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Convolution2D_ExplicitPaddingAsymmetric1,
                        ConvolutionCudaLayerThresholdTest,
                        ::testing::Combine(conv2DParams_ExplicitPaddingAsymmetric1,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
                                           ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        ConvolutionCudaLayerThresholdTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Convolution2D_ExplicitPaddingAsymmetric2,
                        ConvolutionCudaLayerThresholdTest,
                        ::testing::Combine(conv2DParams_ExplicitPaddingAsymmetric2,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
                                           ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        ConvolutionCudaLayerThresholdTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Convolution2D_AutoPadValid,
                        ConvolutionCudaLayerThresholdTest,
                        ::testing::Combine(conv2DParams_AutoPadValid,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
                                           ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        ConvolutionCudaLayerThresholdTest::getTestCaseName);

/* ============= 3D Convolution ============= */
const std::vector<std::vector<size_t>> kernels3d = {{3, 3, 3}, {3, 5, 3}};
const std::vector<std::vector<size_t>> strides3d = {{1, 1, 1}, {1, 2, 1}};
const std::vector<std::vector<size_t>> dilations3d = {{1, 1, 1}, {1, 2, 1}};
const std::vector<size_t> numOutChannels3D = {1, 5};

const auto conv3DParams_ExplicitPaddingSymmetric1 =
    ::testing::Combine(::testing::ValuesIn(kernels3d),
                       ::testing::ValuesIn(strides3d),
                       ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),  // pads_begin
                       ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),  // pads_end
                       ::testing::ValuesIn(dilations3d),
                       ::testing::ValuesIn(numOutChannels3D),
                       ::testing::Values(ov::op::PadType::EXPLICIT));
const auto conv3DParams_ExplicitPaddingSymmetric2 =
    ::testing::Combine(::testing::ValuesIn(kernels3d),
                       ::testing::ValuesIn(strides3d),
                       ::testing::Values(std::vector<ptrdiff_t>({0, 2, 0})),  // pads_begin
                       ::testing::Values(std::vector<ptrdiff_t>({0, 2, 0})),  // pads_end
                       ::testing::ValuesIn(dilations3d),
                       ::testing::ValuesIn(numOutChannels3D),
                       ::testing::Values(ov::op::PadType::EXPLICIT));
const auto conv3DParams_ExplicitPaddingAsymmetric1 =
    ::testing::Combine(::testing::ValuesIn(kernels3d),
                       ::testing::ValuesIn(strides3d),
                       ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),  // pads_begin
                       ::testing::Values(std::vector<ptrdiff_t>({0, 2, 0})),  // pads_end
                       ::testing::ValuesIn(dilations3d),
                       ::testing::ValuesIn(numOutChannels3D),
                       ::testing::Values(ov::op::PadType::EXPLICIT));
const auto conv3DParams_ExplicitPaddingAsymmetric2 =
    ::testing::Combine(::testing::ValuesIn(kernels3d),
                       ::testing::ValuesIn(strides3d),
                       ::testing::Values(std::vector<ptrdiff_t>({0, 2, 0})),  // pads_begin
                       ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),  // pads_end
                       ::testing::ValuesIn(dilations3d),
                       ::testing::ValuesIn(numOutChannels3D),
                       ::testing::Values(ov::op::PadType::EXPLICIT));
const auto conv3DParams_AutoPadValid = ::testing::Combine(::testing::ValuesIn(kernels3d),
                                                          ::testing::ValuesIn(strides3d),
                                                          ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
                                                          ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
                                                          ::testing::ValuesIn(dilations3d),
                                                          ::testing::ValuesIn(numOutChannels3D),
                                                          ::testing::Values(ov::op::PadType::VALID));

INSTANTIATE_TEST_CASE_P(smoke_Convolution3D_ExplicitPaddingSymmetric1,
                        ConvolutionCudaLayerThresholdTest,
                        ::testing::Combine(conv3DParams_ExplicitPaddingSymmetric1,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 3, 10, 10, 10})),
                                           ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        ConvolutionCudaLayerThresholdTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Convolution3D_ExplicitPaddingSymmetric2,
                        ConvolutionCudaLayerThresholdTest,
                        ::testing::Combine(conv3DParams_ExplicitPaddingSymmetric2,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 3, 10, 10, 10})),
                                           ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        ConvolutionCudaLayerThresholdTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Convolution3D_ExplicitPaddingAsymmetric1,
                        ConvolutionCudaLayerThresholdTest,
                        ::testing::Combine(conv3DParams_ExplicitPaddingAsymmetric1,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 3, 10, 10, 10})),
                                           ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        ConvolutionCudaLayerThresholdTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Convolution3D_ExplicitPaddingAsymmetric2,
                        ConvolutionCudaLayerThresholdTest,
                        ::testing::Combine(conv3DParams_ExplicitPaddingAsymmetric2,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 3, 10, 10, 10})),
                                           ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        ConvolutionCudaLayerThresholdTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Convolution3D_AutoPadValid,
                        ConvolutionCudaLayerThresholdTest,
                        ::testing::Combine(conv3DParams_AutoPadValid,
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 3, 10, 10, 10})),
                                           ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        ConvolutionCudaLayerThresholdTest::getTestCaseName);

// =============================================================================
// clang-format off
// {AUTOGENERATED_TESTS_BEGIN_TAG}

// Attrs:  {'auto_pad': 'explicit', 'dilations': '1', 'pads_begin': '15', 'pads_end': '15', 'strides': '1'}
// In:     (1, 2, 1000), (32, 2, 31)
// Out:    (1, 32, 1000)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_Tacotron2_decoder_iter_opid107,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({31})), // kernel
            ::testing::Values(std::vector<size_t>({1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({15})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({15})), // pads_end
            ::testing::Values(std::vector<size_t>({1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 2, 1000})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1', 'pads_begin': '2', 'pads_end': '2', 'strides': '1'}
// In:     (1, 512, 1000), (512, 512, 5)
// Out:    (1, 512, 1000)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_Tacotron2_encoder_opid12,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({5})), // kernel
            ::testing::Values(std::vector<size_t>({1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({2})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({2})), // pads_end
            ::testing::Values(std::vector<size_t>({1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 1000})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1', 'pads_begin': '2', 'pads_end': '2', 'strides': '1'}
// In:     (1, 512, 1000), (80, 512, 5)
// Out:    (1, 80, 1000)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_Tacotron2_graph_transform_cuda_postnet_opid22,
    ConvolutionCudaLayerThresholdTest,
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
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1', 'pads_begin': '2', 'pads_end': '2', 'strides': '1'}
// In:     (1, 80, 1000), (512, 80, 5)
// Out:    (1, 512, 1000)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_Tacotron2_graph_transform_cuda_postnet_opid2,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({5})), // kernel
            ::testing::Values(std::vector<size_t>({1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({2})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({2})), // pads_end
            ::testing::Values(std::vector<size_t>({1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 80, 1000})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1024, 14, 14), (256, 1024, 1, 1)
// Out:    (1, 256, 14, 14)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid152,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1024, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1024, 14, 14), (512, 1024, 1, 1)
// Out:    (1, 512, 14, 14)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid232,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1024, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1024, 20, 20), (512, 1024, 1, 1)
// Out:    (1, 512, 20, 20)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid194,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1024, 20, 20})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 114, 114), (64, 128, 3, 3)
// Out:    (1, 64, 112, 112)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_photo_style_transfer_opid184,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 114, 114})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 28, 28), (32, 128, 1, 1)
// Out:    (1, 32, 28, 28)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_squeezenet1_1_opid43,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 28, 28})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 28, 28), (512, 128, 1, 1)
// Out:    (1, 512, 28, 28)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid110,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 28, 28})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 40, 40), (128, 128, 1, 1)
// Out:    (1, 128, 40, 40)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid136,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 40, 40})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 56, 56), (16, 128, 1, 1)
// Out:    (1, 16, 56, 56)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_squeezenet1_1_opid26,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(16), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 56, 56})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 58, 58), (128, 128, 3, 3)
// Out:    (1, 128, 56, 56)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_photo_style_transfer_opid109,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 58, 58})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 80, 80), (128, 128, 1, 1)
// Out:    (1, 128, 80, 80)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid121,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 80, 80})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 80, 80), (255, 128, 1, 1)
// Out:    (1, 255, 80, 80)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid319,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(255), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 80, 80})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 80, 80), (64, 128, 1, 1)
// Out:    (1, 64, 80, 80)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid115,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 80, 80})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 16, 56, 56), (64, 16, 1, 1)
// Out:    (1, 64, 56, 56)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_squeezenet1_1_opid15,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 16, 56, 56})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 2048, 7, 7), (512, 2048, 1, 1)
// Out:    (1, 512, 7, 7)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid252,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 2048, 7, 7})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 14, 14), (1024, 256, 1, 1)
// Out:    (1, 1024, 14, 14)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid142,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(1024), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 14, 14), (48, 256, 1, 1)
// Out:    (1, 48, 14, 14)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_squeezenet1_1_opid76,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(48), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 20, 20), (256, 256, 1, 1)
// Out:    (1, 256, 20, 20)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid204,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 20, 20})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 28, 28), (32, 256, 1, 1)
// Out:    (1, 32, 28, 28)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_squeezenet1_1_opid59,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 28, 28})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 40, 40), (128, 256, 1, 1)
// Out:    (1, 128, 40, 40)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid131,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 40, 40})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 40, 40), (255, 256, 1, 1)
// Out:    (1, 255, 40, 40)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid404,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(255), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 40, 40})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 40, 40), (256, 256, 1, 1)
// Out:    (1, 256, 40, 40)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid175,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 40, 40})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 56, 56), (128, 256, 1, 1)
// Out:    (1, 128, 56, 56)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid64,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 56, 56})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 56, 56), (64, 256, 1, 1)
// Out:    (1, 64, 56, 56)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid32,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 56, 56})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 80, 80), (64, 256, 1, 1)
// Out:    (1, 64, 80, 80)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid293,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 80, 80})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 3, 232, 232), (32, 3, 9, 9)
// Out:    (1, 32, 224, 224)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_photo_style_transfer_opid5,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({9, 9})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 3, 232, 232})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 32, 160, 160), (32, 32, 1, 1)
// Out:    (1, 32, 160, 160)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid50,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 160, 160})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 32, 232, 232), (3, 32, 9, 9)
// Out:    (1, 3, 224, 224)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_photo_style_transfer_opid220,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({9, 9})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(3), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 232, 232})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 32, 28, 28), (128, 32, 1, 1)
// Out:    (1, 128, 28, 28)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_squeezenet1_1_opid48,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 28, 28})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 384, 14, 14), (48, 384, 1, 1)
// Out:    (1, 48, 14, 14)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_squeezenet1_1_opid92,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(48), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 384, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 384, 14, 14), (64, 384, 1, 1)
// Out:    (1, 64, 14, 14)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_squeezenet1_1_opid108,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 384, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 48, 14, 14), (192, 48, 1, 1)
// Out:    (1, 192, 14, 14)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_squeezenet1_1_opid81,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(192), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 48, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 14, 14), (1000, 512, 1, 1)
// Out:    (1, 1000, 14, 14)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_squeezenet1_1_opid140,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(1000), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 14, 14), (64, 512, 1, 1)
// Out:    (1, 64, 14, 14)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_squeezenet1_1_opid124,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 20, 20), (255, 512, 1, 1)
// Out:    (1, 255, 20, 20)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid489,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(255), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 20, 20})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 20, 20), (256, 512, 1, 1)
// Out:    (1, 256, 20, 20)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid185,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 20, 20})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 20, 20), (512, 512, 1, 1)
// Out:    (1, 512, 20, 20)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid220,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 20, 20})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 28, 28), (128, 512, 1, 1)
// Out:    (1, 128, 28, 28)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid100,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 28, 28})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 28, 28), (256, 512, 1, 1)
// Out:    (1, 256, 28, 28)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid132,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 28, 28})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 40, 40), (128, 512, 1, 1)
// Out:    (1, 128, 40, 40)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid246,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 40, 40})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 7, 7), (2048, 512, 1, 1)
// Out:    (1, 2048, 7, 7)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid242,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(2048), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 7, 7})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 14, 14), (256, 64, 1, 1)
// Out:    (1, 256, 14, 14)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_squeezenet1_1_opid113,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 160, 160), (32, 64, 1, 1)
// Out:    (1, 32, 160, 160)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid45,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 160, 160})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 160, 160), (64, 64, 1, 1)
// Out:    (1, 64, 160, 160)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid67,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 160, 160})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 226, 226), (32, 64, 3, 3)
// Out:    (1, 32, 224, 224)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_photo_style_transfer_opid207,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 226, 226})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 56, 56), (16, 64, 1, 1)
// Out:    (1, 16, 56, 56)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_squeezenet1_1_opid10,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(16), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 56, 56})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 56, 56), (256, 64, 1, 1)
// Out:    (1, 256, 56, 56)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid22,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 56, 56})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 56, 56), (64, 64, 1, 1)
// Out:    (1, 64, 56, 56)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid12,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 56, 56})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 80, 80), (64, 64, 1, 1)
// Out:    (1, 64, 80, 80)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid104,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 80, 80})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (100, 5, 1, 1), (5, 5, 1, 1)
// Out:    (100, 5, 1, 1)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid306,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(5), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({100, 5, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 1024, 14, 14), (2048, 1024, 1, 1)
// Out:    (1, 2048, 7, 7)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid246,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(2048), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1024, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 256, 56, 56), (512, 256, 1, 1)
// Out:    (1, 512, 28, 28)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid78,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 56, 56})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 3, 227, 227), (64, 3, 3, 3)
// Out:    (1, 64, 113, 113)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_squeezenet1_1_opid4,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 3, 227, 227})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 32, 226, 226), (64, 32, 3, 3)
// Out:    (1, 64, 112, 112)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_photo_style_transfer_opid18,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 226, 226})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 512, 28, 28), (1024, 512, 1, 1)
// Out:    (1, 1024, 14, 14)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid146,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(1024), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 28, 28})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 64, 114, 114), (128, 64, 3, 3)
// Out:    (1, 128, 56, 56)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_photo_style_transfer_opid31,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 114, 114})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 128, 104, 104), (256, 128, 3, 3)
// Out:    (1, 256, 52, 52)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid59,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 104, 104})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 128, 152, 152), (256, 128, 3, 3)
// Out:    (1, 256, 76, 76)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid92,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 152, 152})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 128, 76, 76), (256, 128, 3, 3)
// Out:    (1, 256, 38, 38)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid540,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 76, 76})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 256, 38, 38), (512, 256, 3, 3)
// Out:    (1, 512, 19, 19)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid588,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 38, 38})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 256, 52, 52), (512, 256, 3, 3)
// Out:    (1, 512, 26, 26)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid169,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 52, 52})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 256, 76, 76), (512, 256, 3, 3)
// Out:    (1, 512, 38, 38)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid212,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 76, 76})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 32, 416, 416), (64, 32, 3, 3)
// Out:    (1, 64, 208, 208)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid8,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 416, 416})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 32, 608, 608), (64, 32, 3, 3)
// Out:    (1, 64, 304, 304)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid7,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 608, 608})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 512, 26, 26), (1024, 512, 3, 3)
// Out:    (1, 1024, 13, 13)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid279,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(1024), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 26, 26})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 512, 38, 38), (1024, 512, 3, 3)
// Out:    (1, 1024, 19, 19)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid332,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(1024), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 38, 38})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 64, 208, 208), (128, 64, 3, 3)
// Out:    (1, 128, 104, 104)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid27,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 208, 208})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 64, 304, 304), (128, 64, 3, 3)
// Out:    (1, 128, 152, 152)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid44,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 304, 304})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 12, 320, 320), (32, 12, 3, 3)
// Out:    (1, 32, 320, 320)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid35,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 12, 320, 320})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 128, 112, 112), (128, 128, 3, 3)
// Out:    (1, 128, 112, 112)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_vgg16_IR_opid20,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 112, 112})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 128, 28, 28), (128, 128, 3, 3)
// Out:    (1, 128, 28, 28)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid105,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 28, 28})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 128, 40, 40), (128, 128, 3, 3)
// Out:    (1, 128, 40, 40)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid141,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 40, 40})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 128, 56, 56), (256, 128, 3, 3)
// Out:    (1, 256, 56, 56)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_vgg16_IR_opid26,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 56, 56})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 16, 56, 56), (64, 16, 3, 3)
// Out:    (1, 64, 56, 56)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_squeezenet1_1_opid20,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 16, 56, 56})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 256, 14, 14), (256, 256, 3, 3)
// Out:    (1, 256, 14, 14)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid157,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 256, 20, 20), (256, 256, 3, 3)
// Out:    (1, 256, 20, 20)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid209,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 20, 20})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 256, 28, 28), (512, 256, 3, 3)
// Out:    (1, 512, 28, 28)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_vgg16_IR_opid42,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::Values(InferenceEngine::Precision::FP32), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 28, 28})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 256, 56, 56), (256, 256, 3, 3)
// Out:    (1, 256, 56, 56)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_vgg16_IR_opid31,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::Values(InferenceEngine::Precision::FP32), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 56, 56})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 3, 224, 224), (64, 3, 3, 3)
// Out:    (1, 64, 224, 224)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_vgg16_IR_opid4,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 3, 224, 224})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 32, 160, 160), (32, 32, 3, 3)
// Out:    (1, 32, 160, 160)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid55,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 160, 160})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 32, 28, 28), (128, 32, 3, 3)
// Out:    (1, 128, 28, 28)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_squeezenet1_1_opid53,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 28, 28})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 48, 14, 14), (192, 48, 3, 3)
// Out:    (1, 192, 14, 14)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_squeezenet1_1_opid102,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(192), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 48, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 512, 14, 14), (512, 512, 3, 3)
// Out:    (1, 512, 14, 14)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_vgg16_IR_opid58,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 512, 28, 28), (512, 512, 3, 3)
// Out:    (1, 512, 28, 28)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_vgg16_IR_opid47,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 28, 28})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 512, 7, 7), (512, 512, 3, 3)
// Out:    (1, 512, 7, 7)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid257,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 7, 7})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 64, 112, 112), (128, 64, 3, 3)
// Out:    (1, 128, 112, 112)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_vgg16_IR_opid15,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 112, 112})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 64, 14, 14), (256, 64, 3, 3)
// Out:    (1, 256, 14, 14)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_squeezenet1_1_opid118,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 64, 224, 224), (64, 64, 3, 3)
// Out:    (1, 64, 224, 224)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_vgg16_IR_opid9,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 224, 224})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 64, 56, 56), (64, 64, 3, 3)
// Out:    (1, 64, 56, 56)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid17,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 56, 56})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '1,1'}
// In:     (1, 64, 80, 80), (64, 64, 3, 3)
// Out:    (1, 64, 80, 80)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid109,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 80, 80})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '2,2'}
// In:     (1, 128, 56, 56), (128, 128, 3, 3)
// Out:    (1, 128, 28, 28)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid69,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 56, 56})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '2,2'}
// In:     (1, 128, 80, 80), (128, 128, 3, 3)
// Out:    (1, 128, 40, 40)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid372,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 80, 80})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '2,2'}
// In:     (1, 128, 80, 80), (256, 128, 3, 3)
// Out:    (1, 256, 40, 40)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid126,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 80, 80})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '2,2'}
// In:     (1, 256, 28, 28), (256, 256, 3, 3)
// Out:    (1, 256, 14, 14)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid137,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 28, 28})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '2,2'}
// In:     (1, 256, 40, 40), (256, 256, 3, 3)
// Out:    (1, 256, 20, 20)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid457,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 40, 40})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '2,2'}
// In:     (1, 256, 40, 40), (512, 256, 3, 3)
// Out:    (1, 512, 20, 20)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid180,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 40, 40})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '2,2'}
// In:     (1, 32, 320, 320), (64, 32, 3, 3)
// Out:    (1, 64, 160, 160)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid40,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 320, 320})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '2,2'}
// In:     (1, 512, 14, 14), (512, 512, 3, 3)
// Out:    (1, 512, 7, 7)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid237,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '1,1', 'pads_end': '1,1', 'strides': '2,2'}
// In:     (1, 64, 160, 160), (128, 64, 3, 3)
// Out:    (1, 128, 80, 80)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolov5_640x640_IR_opid72,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({1, 1})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 160, 160})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'explicit', 'dilations': '1,1', 'pads_begin': '3,3', 'pads_end': '3,3', 'strides': '2,2'}
// In:     (1, 3, 224, 224), (64, 3, 7, 7)
// Out:    (1, 64, 112, 112)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_caffe2_opid6,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({7, 7})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({3, 3})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({3, 3})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::EXPLICIT)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 3, 224, 224})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1', 'pads_begin': '0', 'pads_end': '0', 'strides': '1'}
// In:     (64, 106, 64), (128, 106, 3)
// Out:    (64, 128, 64)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_LPCnet_lpcnet_enc_opid12,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3})), // kernel
            ::testing::Values(std::vector<size_t>({1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0})), // pads_end
            ::testing::Values(std::vector<size_t>({1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({64, 106, 64})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1', 'pads_begin': '0', 'pads_end': '0', 'strides': '1'}
// In:     (64, 128, 64), (128, 128, 3)
// Out:    (64, 128, 64)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_LPCnet_lpcnet_enc_opid17,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3})), // kernel
            ::testing::Values(std::vector<size_t>({1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0})), // pads_end
            ::testing::Values(std::vector<size_t>({1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({64, 128, 64})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1, 128, 128), (16, 1, 3, 3)
// Out:    (1, 16, 128, 128)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_2d_unet_graph_transform_opid2,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(16), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1, 128, 128})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1, 224, 224), (32, 1, 5, 5)
// Out:    (1, 32, 224, 224)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_super_resolution_opid2,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({5, 5})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1, 224, 224})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 10, 1, 1), (240, 10, 1, 1)
// Out:    (1, 240, 1, 1)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid183,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(240), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 10, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1024, 13, 13), (255, 1024, 1, 1)
// Out:    (1, 255, 13, 13)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid373,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(255), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1024, 13, 13})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1024, 13, 13), (512, 1024, 1, 1)
// Out:    (1, 512, 13, 13)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid285,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1024, 13, 13})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1024, 14, 14), (256, 1024, 1, 1)
// Out:    (1, 256, 14, 14)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_tf_opid150,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1024, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1024, 14, 14), (512, 1024, 1, 1)
// Out:    (1, 512, 14, 14)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_tf_opid230,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1024, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1024, 17, 17), (128, 1024, 1, 1)
// Out:    (1, 128, 17, 17)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid282,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1024, 17, 17})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1024, 17, 17), (192, 1024, 1, 1)
// Out:    (1, 192, 17, 17)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid241,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(192), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1024, 17, 17})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1024, 17, 17), (256, 1024, 1, 1)
// Out:    (1, 256, 17, 17)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid610,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1024, 17, 17})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1024, 17, 17), (384, 1024, 1, 1)
// Out:    (1, 384, 17, 17)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid236,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(384), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1024, 17, 17})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1024, 19, 19), (1024, 1024, 1, 1)
// Out:    (1, 1024, 19, 19)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid397,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(1024), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1024, 19, 19})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1024, 19, 19), (255, 1024, 1, 1)
// Out:    (1, 255, 19, 19)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid631,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(255), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1024, 19, 19})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1024, 19, 19), (512, 1024, 1, 1)
// Out:    (1, 512, 19, 19)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid337,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1024, 19, 19})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 112, 40, 40), (672, 112, 1, 1)
// Out:    (1, 672, 40, 40)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid364,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(672), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 112, 40, 40})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 112, 40, 40), (88, 112, 1, 1)
// Out:    (1, 88, 40, 40)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid448,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(88), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 112, 40, 40})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1152, 1, 1), (48, 1152, 1, 1)
// Out:    (1, 48, 1, 1)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid491,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(48), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1152, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1152, 20, 20), (192, 1152, 1, 1)
// Out:    (1, 192, 20, 20)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid502,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(192), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1152, 20, 20})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1152, 20, 20), (320, 1152, 1, 1)
// Out:    (1, 320, 20, 20)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid614,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(320), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1152, 20, 20})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 1, 1), (24, 128, 3, 3)
// Out:    (1, 24, 1, 1)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid359,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(24), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 1, 1), (546, 128, 3, 3)
// Out:    (1, 546, 1, 1)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid410,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(546), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 104, 104), (64, 128, 1, 1)
// Out:    (1, 64, 104, 104)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid33,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 104, 104})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 152, 152), (128, 128, 1, 1)
// Out:    (1, 128, 152, 152)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid87,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 152, 152})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 152, 152), (64, 128, 1, 1)
// Out:    (1, 64, 152, 152)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid49,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 152, 152})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 16, 16), (128, 128, 3, 3)
// Out:    (1, 128, 16, 16)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_2d_unet_graph_transform_opid40,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 16, 16})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 2, 2), (256, 128, 1, 1)
// Out:    (1, 256, 2, 2)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid331,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 2, 2})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 28, 28), (128, 128, 3, 3)
// Out:    (1, 128, 28, 28)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_tf_opid103,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 28, 28})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 28, 28), (512, 128, 1, 1)
// Out:    (1, 512, 28, 28)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_tf_opid108,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 28, 28})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 3, 3), (256, 128, 1, 1)
// Out:    (1, 256, 3, 3)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid308,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 3, 3})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 304, 304), (64, 128, 1, 1)
// Out:    (1, 64, 304, 304)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid39,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 304, 304})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 32, 32), (64, 128, 3, 3)
// Out:    (1, 64, 32, 32)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_2d_unet_graph_transform_opid78,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 32, 32})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 50, 86), (128, 128, 3, 3)
// Out:    (1, 128, 50, 86)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid151,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 50, 86})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 50, 86), (160, 128, 3, 3)
// Out:    (1, 160, 50, 86)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid210,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(160), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 50, 86})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 50, 86), (192, 128, 3, 3)
// Out:    (1, 192, 50, 86)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid247,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(192), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 50, 86})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 52, 52), (256, 128, 3, 3)
// Out:    (1, 256, 52, 52)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid110,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 52, 52})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 76, 76), (128, 128, 1, 1)
// Out:    (1, 128, 76, 76)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid102,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 76, 76})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 76, 76), (128, 128, 3, 3)
// Out:    (1, 128, 76, 76)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid107,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 76, 76})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 76, 76), (256, 128, 3, 3)
// Out:    (1, 256, 76, 76)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid516,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 76, 76})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 128, 8, 8), (256, 128, 3, 3)
// Out:    (1, 256, 8, 8)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_2d_unet_graph_transform_opid46,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 8, 8})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1280, 10, 10), (24, 1280, 3, 3)
// Out:    (1, 24, 10, 10)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid267,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(24), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1280, 10, 10})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1280, 10, 10), (256, 1280, 1, 1)
// Out:    (1, 256, 10, 10)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid275,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1280, 10, 10})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1280, 10, 10), (546, 1280, 3, 3)
// Out:    (1, 546, 10, 10)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid378,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(546), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1280, 10, 10})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 144, 1, 1), (6, 144, 1, 1)
// Out:    (1, 6, 1, 1)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid123,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(6), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 144, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 144, 160, 160), (24, 144, 1, 1)
// Out:    (1, 24, 160, 160)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid106,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(24), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 144, 160, 160})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 144, 38, 38), (32, 144, 1, 1)
// Out:    (1, 32, 38, 38)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid59,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 144, 38, 38})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 144, 75, 75), (24, 144, 1, 1)
// Out:    (1, 24, 75, 75)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid44,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(24), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 144, 75, 75})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 144, 80, 80), (40, 144, 1, 1)
// Out:    (1, 40, 80, 80)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid162,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(40), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 144, 80, 80})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1536, 8, 8), (256, 1536, 1, 1)
// Out:    (1, 256, 8, 8)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid632,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1536, 8, 8})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1536, 8, 8), (384, 1536, 1, 1)
// Out:    (1, 384, 8, 8)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid637,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(384), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1536, 8, 8})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 16, 1, 1), (4, 16, 1, 1)
// Out:    (1, 4, 1, 1)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid40,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(4), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 16, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 16, 128, 128), (16, 16, 3, 3)
// Out:    (1, 16, 128, 128)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_2d_unet_graph_transform_opid115,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(16), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 16, 128, 128})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 16, 150, 150), (96, 16, 1, 1)
// Out:    (1, 96, 150, 150)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid20,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(96), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 16, 150, 150})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 16, 320, 320), (16, 16, 1, 1)
// Out:    (1, 16, 320, 320)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid51,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(16), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 16, 320, 320})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 16, 320, 320), (96, 16, 1, 1)
// Out:    (1, 96, 320, 320)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid56,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(96), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 16, 320, 320})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 16, 64, 64), (32, 16, 3, 3)
// Out:    (1, 32, 64, 64)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_2d_unet_graph_transform_opid13,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 16, 64, 64})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 160, 10, 10), (960, 160, 1, 1)
// Out:    (1, 960, 10, 10)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid218,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(960), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 160, 10, 10})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 160, 50, 86), (160, 160, 3, 3)
// Out:    (1, 160, 50, 86)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid225,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(160), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 160, 50, 86})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 160, 50, 86), (192, 160, 3, 3)
// Out:    (1, 192, 50, 86)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid257,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(192), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 160, 50, 86})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 160, 73, 73), (64, 160, 1, 1)
// Out:    (1, 64, 73, 73)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid28,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 160, 73, 73})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 192, 100, 171), (32, 192, 1, 1)
// Out:    (1, 32, 100, 171)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid56,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 192, 100, 171})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 192, 100, 171), (64, 192, 1, 1)
// Out:    (1, 64, 100, 171)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid25,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 192, 100, 171})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 192, 17, 17), (192, 192, 7, 1)
// Out:    (1, 192, 17, 17)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid261,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({7, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(192), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 192, 17, 17})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 192, 17, 17), (224, 192, 1, 7)
// Out:    (1, 224, 17, 17)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid246,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 7})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(224), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 192, 17, 17})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 192, 19, 19), (64, 192, 1, 1)
// Out:    (1, 64, 19, 19)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid103,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 192, 19, 19})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 192, 20, 20), (1152, 192, 1, 1)
// Out:    (1, 1152, 20, 20)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid479,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(1152), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 192, 20, 20})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 192, 35, 35), (224, 192, 3, 3)
// Out:    (1, 224, 35, 35)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid224,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(224), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 192, 35, 35})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 192, 38, 38), (32, 192, 1, 1)
// Out:    (1, 32, 38, 38)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid73,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 192, 38, 38})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 192, 50, 86), (192, 192, 3, 3)
// Out:    (1, 192, 50, 86)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid262,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(192), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 192, 50, 86})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1920, 1, 1), (80, 1920, 1, 1)
// Out:    (1, 80, 1, 1)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid630,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(80), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1920, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 1920, 20, 20), (320, 1920, 1, 1)
// Out:    (1, 320, 20, 20)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid641,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(320), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1920, 20, 20})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 20, 1, 1), (480, 20, 1, 1)
// Out:    (1, 480, 1, 1)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid270,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(480), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 20, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 2048, 19, 19), (512, 2048, 1, 1)
// Out:    (1, 512, 19, 19)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid424,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 2048, 19, 19})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 2048, 7, 7), (512, 2048, 1, 1)
// Out:    (1, 512, 7, 7)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_tf_opid250,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 2048, 7, 7})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 224, 17, 17), (224, 224, 7, 1)
// Out:    (1, 224, 17, 17)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid271,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({7, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(224), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 224, 17, 17})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 224, 17, 17), (256, 224, 1, 7)
// Out:    (1, 256, 17, 17)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid276,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 7})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 224, 17, 17})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 224, 17, 17), (256, 224, 7, 1)
// Out:    (1, 256, 17, 17)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid251,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({7, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 224, 17, 17})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 24, 160, 160), (144, 24, 1, 1)
// Out:    (1, 144, 160, 160)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid111,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(144), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 24, 160, 160})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 24, 75, 75), (144, 24, 1, 1)
// Out:    (1, 144, 75, 75)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid34,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(144), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 24, 75, 75})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 240, 1, 1), (10, 240, 1, 1)
// Out:    (1, 10, 1, 1)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid178,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(10), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 240, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 240, 40, 40), (80, 240, 1, 1)
// Out:    (1, 80, 40, 40)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid249,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(80), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 240, 40, 40})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 240, 80, 80), (40, 240, 1, 1)
// Out:    (1, 40, 80, 80)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid189,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(40), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 240, 80, 80})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 100, 171), (64, 256, 1, 1)
// Out:    (1, 64, 100, 171)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid62,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 100, 171})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 14, 14), (1024, 256, 1, 1)
// Out:    (1, 1024, 14, 14)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_tf_opid140,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(1024), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 14, 14), (256, 256, 3, 3)
// Out:    (1, 256, 14, 14)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_tf_opid155,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 16, 16), (128, 256, 3, 3)
// Out:    (1, 128, 16, 16)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_2d_unet_graph_transform_opid62,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 16, 16})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 17, 17), (256, 256, 1, 7)
// Out:    (1, 256, 17, 17)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid615,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 7})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 17, 17})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 17, 17), (320, 256, 7, 1)
// Out:    (1, 320, 17, 17)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid620,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({7, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(320), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 17, 17})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 2, 2), (24, 256, 3, 3)
// Out:    (1, 24, 2, 2)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid336,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(24), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 2, 2})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 2, 2), (546, 256, 3, 3)
// Out:    (1, 546, 2, 2)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid402,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(546), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 2, 2})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 2, 2), (64, 256, 1, 1)
// Out:    (1, 64, 2, 2)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid344,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 2, 2})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 26, 26), (128, 256, 1, 1)
// Out:    (1, 128, 26, 26)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid436,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 26, 26})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 26, 26), (512, 256, 3, 3)
// Out:    (1, 512, 26, 26)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid181,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 26, 26})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 3, 3), (128, 256, 1, 1)
// Out:    (1, 128, 3, 3)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid321,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 3, 3})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 3, 3), (24, 256, 3, 3)
// Out:    (1, 24, 3, 3)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid313,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(24), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 3, 3})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 3, 3), (546, 256, 3, 3)
// Out:    (1, 546, 3, 3)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid394,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(546), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 3, 3})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 38, 38), (128, 256, 1, 1)
// Out:    (1, 128, 38, 38)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid491,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 38, 38})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 38, 38), (256, 256, 1, 1)
// Out:    (1, 256, 38, 38)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid222,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 38, 38})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 38, 38), (256, 256, 3, 3)
// Out:    (1, 256, 38, 38)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid227,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::Values(InferenceEngine::Precision::FP32), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 38, 38})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 38, 38), (512, 256, 3, 3)
// Out:    (1, 512, 38, 38)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid467,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 38, 38})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 5, 5), (512, 256, 1, 1)
// Out:    (1, 512, 5, 5)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid285,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 5, 5})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 52, 52), (128, 256, 1, 1)
// Out:    (1, 128, 52, 52)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid104,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 52, 52})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 52, 52), (255, 256, 1, 1)
// Out:    (1, 255, 52, 52)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid491,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(255), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 52, 52})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 56, 56), (128, 256, 1, 1)
// Out:    (1, 128, 56, 56)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_tf_opid62,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 56, 56})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 56, 56), (64, 256, 1, 1)
// Out:    (1, 64, 56, 56)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_tf_opid30,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 56, 56})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 76, 76), (128, 256, 1, 1)
// Out:    (1, 128, 76, 76)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid195,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 76, 76})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 76, 76), (255, 256, 1, 1)
// Out:    (1, 255, 76, 76)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid642,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(255), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 76, 76})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 76, 76), (256, 256, 1, 1)
// Out:    (1, 256, 76, 76)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid201,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 76, 76})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 256, 8, 8), (256, 256, 3, 3)
// Out:    (1, 256, 8, 8)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_2d_unet_graph_transform_opid51,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 8, 8})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 28, 1, 1), (672, 28, 1, 1)
// Out:    (1, 672, 1, 1)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid381,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(672), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 28, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 3, 416, 416), (32, 3, 3, 3)
// Out:    (1, 32, 416, 416)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid2,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 3, 416, 416})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 3, 608, 608), (32, 3, 3, 3)
// Out:    (1, 32, 608, 608)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid2,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 3, 608, 608})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 32, 1, 1), (8, 32, 1, 1)
// Out:    (1, 8, 1, 1)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid18,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(8), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 32, 128, 128), (16, 32, 3, 3)
// Out:    (1, 16, 128, 128)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_2d_unet_graph_transform_opid110,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(16), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 128, 128})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 32, 147, 147), (64, 32, 3, 3)
// Out:    (1, 64, 147, 147)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid16,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 147, 147})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 32, 150, 150), (16, 32, 1, 1)
// Out:    (1, 16, 150, 150)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid16,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(16), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 150, 150})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 32, 208, 208), (64, 32, 3, 3)
// Out:    (1, 64, 208, 208)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid20,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 208, 208})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 32, 224, 224), (5, 32, 1, 1)
// Out:    (1, 5, 224, 224)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_super_resolution_opid8,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(5), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 224, 224})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 32, 224, 224), (9, 32, 1, 1)
// Out:    (1, 9, 224, 224)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_super_resolution_opid26,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(9), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 224, 224})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 32, 304, 304), (64, 32, 3, 3)
// Out:    (1, 64, 304, 304)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid22,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 304, 304})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 32, 32, 32), (64, 32, 3, 3)
// Out:    (1, 64, 32, 32)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_2d_unet_graph_transform_opid24,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 32, 32})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 32, 320, 320), (16, 32, 1, 1)
// Out:    (1, 16, 320, 320)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid29,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(16), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 320, 320})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 32, 38, 38), (192, 32, 1, 1)
// Out:    (1, 192, 38, 38)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid63,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(192), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 38, 38})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 32, 64, 64), (32, 32, 3, 3)
// Out:    (1, 32, 64, 64)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_2d_unet_graph_transform_opid18,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 64, 64})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 320, 10, 10), (1280, 320, 1, 1)
// Out:    (1, 1280, 10, 10)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid262,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(1280), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 320, 10, 10})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 320, 100, 171), (128, 320, 1, 1)
// Out:    (1, 128, 100, 171)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid99,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 320, 100, 171})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 320, 100, 171), (64, 320, 1, 1)
// Out:    (1, 64, 100, 171)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid109,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 320, 100, 171})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 320, 20, 20), (1920, 320, 1, 1)
// Out:    (1, 1920, 20, 20)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid618,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(1920), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 320, 20, 20})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 320, 20, 20), (88, 320, 1, 1)
// Out:    (1, 88, 20, 20)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid646,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(88), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 320, 20, 20})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 384, 19, 19), (64, 384, 1, 1)
// Out:    (1, 64, 19, 19)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid117,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 384, 19, 19})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 384, 19, 19), (96, 384, 1, 1)
// Out:    (1, 96, 19, 19)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid162,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(96), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 384, 19, 19})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 384, 35, 35), (192, 384, 1, 1)
// Out:    (1, 192, 35, 35)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid219,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(192), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 384, 35, 35})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 384, 35, 35), (64, 384, 1, 1)
// Out:    (1, 64, 35, 35)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid108,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 384, 35, 35})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 384, 35, 35), (96, 384, 1, 1)
// Out:    (1, 96, 35, 35)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid103,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(96), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 384, 35, 35})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 384, 52, 52), (128, 384, 1, 1)
// Out:    (1, 128, 52, 52)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid455,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 384, 52, 52})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 384, 8, 8), (256, 384, 1, 3)
// Out:    (1, 256, 8, 8)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid642,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 384, 8, 8})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 384, 8, 8), (256, 384, 3, 1)
// Out:    (1, 256, 8, 8)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid647,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 384, 8, 8})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 384, 8, 8), (448, 384, 3, 1)
// Out:    (1, 448, 8, 8)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid658,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(448), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 384, 8, 8})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 4, 1, 1), (16, 4, 1, 1)
// Out:    (1, 16, 1, 1)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid45,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(16), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 4, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 4, 1, 1), (96, 4, 1, 1)
// Out:    (1, 96, 1, 1)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid73,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(96), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 4, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 40, 80, 80), (240, 40, 1, 1)
// Out:    (1, 240, 80, 80)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid166,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(240), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 40, 80, 80})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 40, 80, 80), (88, 40, 1, 1)
// Out:    (1, 88, 80, 80)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid222,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(88), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 40, 80, 80})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 448, 8, 8), (512, 448, 1, 3)
// Out:    (1, 512, 8, 8)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid663,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 448, 8, 8})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 48, 1, 1), (1152, 48, 1, 1)
// Out:    (1, 1152, 1, 1)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid496,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(1152), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 48, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 480, 1, 1), (20, 480, 1, 1)
// Out:    (1, 20, 1, 1)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid265,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(20), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 480, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 480, 40, 40), (112, 480, 1, 1)
// Out:    (1, 112, 40, 40)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid360,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(112), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 480, 40, 40})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 480, 40, 40), (80, 480, 1, 1)
// Out:    (1, 80, 40, 40)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid276,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(80), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 480, 40, 40})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 5, 224, 224), (32, 5, 1, 1)
// Out:    (1, 32, 224, 224)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_super_resolution_opid20,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 5, 224, 224})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 5, 224, 224), (5, 5, 3, 3)
// Out:    (1, 5, 224, 224)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_super_resolution_opid14,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(5), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 5, 224, 224})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 13, 13), (1024, 512, 3, 3)
// Out:    (1, 1024, 13, 13)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid291,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(1024), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 13, 13})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 13, 13), (256, 512, 1, 1)
// Out:    (1, 256, 13, 13)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid377,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 13, 13})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 19, 19), (1024, 512, 3, 3)
// Out:    (1, 1024, 19, 19)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid408,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(1024), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 19, 19})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 19, 19), (256, 512, 1, 1)
// Out:    (1, 256, 19, 19)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid442,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 19, 19})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 19, 19), (512, 512, 1, 1)
// Out:    (1, 512, 19, 19)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid342,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 19, 19})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 19, 19), (512, 512, 3, 3)
// Out:    (1, 512, 19, 19)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid347,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 19, 19})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 26, 26), (255, 512, 1, 1)
// Out:    (1, 255, 26, 26)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid432,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(255), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 26, 26})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 26, 26), (256, 512, 1, 1)
// Out:    (1, 256, 26, 26)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid175,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 26, 26})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 28, 28), (128, 512, 1, 1)
// Out:    (1, 128, 28, 28)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_tf_opid114,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 28, 28})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 28, 28), (256, 512, 1, 1)
// Out:    (1, 256, 28, 28)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_tf_opid130,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 28, 28})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 38, 38), (255, 512, 1, 1)
// Out:    (1, 255, 38, 38)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid583,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(255), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 38, 38})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 38, 38), (256, 512, 1, 1)
// Out:    (1, 256, 38, 38)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid217,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 38, 38})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 38, 38), (512, 512, 1, 1)
// Out:    (1, 512, 38, 38)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid321,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 38, 38})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 5, 5), (128, 512, 1, 1)
// Out:    (1, 128, 5, 5)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid298,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 5, 5})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 5, 5), (24, 512, 3, 3)
// Out:    (1, 24, 5, 5)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid290,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(24), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 5, 5})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 5, 5), (546, 512, 3, 3)
// Out:    (1, 546, 5, 5)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid386,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(546), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 5, 5})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 50, 86), (24, 512, 1, 1)
// Out:    (1, 24, 50, 86)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid279,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(24), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 50, 86})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 50, 86), (48, 512, 1, 1)
// Out:    (1, 48, 50, 86)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid296,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(48), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 50, 86})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 7, 7), (2048, 512, 1, 1)
// Out:    (1, 2048, 7, 7)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_tf_opid240,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(2048), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 7, 7})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 7, 7), (512, 512, 3, 3)
// Out:    (1, 512, 7, 7)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_tf_opid255,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 7, 7})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 8, 8), (256, 512, 1, 3)
// Out:    (1, 256, 8, 8)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid668,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 8, 8})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 512, 8, 8), (256, 512, 3, 1)
// Out:    (1, 256, 8, 8)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid673,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 8, 8})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 576, 10, 10), (160, 576, 1, 1)
// Out:    (1, 160, 10, 10)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid214,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(160), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 576, 10, 10})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 576, 19, 19), (12, 576, 3, 3)
// Out:    (1, 12, 19, 19)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid201,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(12), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 576, 19, 19})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 576, 19, 19), (273, 576, 3, 3)
// Out:    (1, 273, 19, 19)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid370,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(273), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 576, 19, 19})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 576, 19, 19), (96, 576, 1, 1)
// Out:    (1, 96, 19, 19)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid176,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(96), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 576, 19, 19})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 576, 50, 86), (128, 576, 1, 1)
// Out:    (1, 128, 50, 86)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid157,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 576, 50, 86})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 576, 50, 86), (160, 576, 1, 1)
// Out:    (1, 160, 50, 86)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid200,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(160), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 576, 50, 86})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 576, 50, 86), (192, 576, 1, 1)
// Out:    (1, 192, 50, 86)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid163,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(192), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 576, 50, 86})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 576, 50, 86), (224, 576, 1, 1)
// Out:    (1, 224, 50, 86)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid126,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(224), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 576, 50, 86})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 576, 50, 86), (512, 576, 3, 3)
// Out:    (1, 512, 50, 86)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid274,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 576, 50, 86})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 576, 50, 86), (64, 576, 1, 1)
// Out:    (1, 64, 50, 86)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid131,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 576, 50, 86})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 576, 50, 86), (96, 576, 1, 1)
// Out:    (1, 96, 50, 86)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid141,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(96), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 576, 50, 86})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 6, 1, 1), (144, 6, 1, 1)
// Out:    (1, 144, 1, 1)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid100,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(144), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 6, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 1, 1), (128, 64, 1, 1)
// Out:    (1, 128, 1, 1)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid354,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 100, 171), (64, 64, 3, 3)
// Out:    (1, 64, 100, 171)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid35,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 100, 171})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 100, 171), (96, 64, 3, 3)
// Out:    (1, 96, 100, 171)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid114,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(96), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 100, 171})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 104, 104), (128, 64, 3, 3)
// Out:    (1, 128, 104, 104)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid39,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 104, 104})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 152, 152), (64, 64, 1, 1)
// Out:    (1, 64, 152, 152)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid54,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 152, 152})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 152, 152), (64, 64, 3, 3)
// Out:    (1, 64, 152, 152)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid59,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 152, 152})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 16, 16), (128, 64, 3, 3)
// Out:    (1, 128, 16, 16)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_2d_unet_graph_transform_opid35,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 16, 16})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 19, 19), (384, 64, 1, 1)
// Out:    (1, 384, 19, 19)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid107,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(384), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 19, 19})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 200, 342), (192, 64, 3, 3)
// Out:    (1, 192, 200, 342)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid19,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(192), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 200, 342})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 200, 342), (64, 64, 1, 1)
// Out:    (1, 64, 200, 342)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid14,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 200, 342})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 208, 208), (32, 64, 1, 1)
// Out:    (1, 32, 208, 208)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid14,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 208, 208})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 304, 304), (32, 64, 1, 1)
// Out:    (1, 32, 304, 304)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid17,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 304, 304})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 304, 304), (64, 64, 1, 1)
// Out:    (1, 64, 304, 304)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v4_tf_opid12,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 304, 304})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 32, 32), (64, 64, 3, 3)
// Out:    (1, 64, 32, 32)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_2d_unet_graph_transform_opid29,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 32, 32})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 35, 35), (96, 64, 3, 3)
// Out:    (1, 96, 35, 35)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid113,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(96), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 35, 35})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 50, 86), (96, 64, 3, 3)
// Out:    (1, 96, 50, 86)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid136,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(96), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 50, 86})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 56, 56), (256, 64, 1, 1)
// Out:    (1, 256, 56, 56)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_tf_opid20,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 56, 56})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 56, 56), (64, 64, 1, 1)
// Out:    (1, 64, 56, 56)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_tf_opid10,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 56, 56})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 56, 56), (64, 64, 3, 3)
// Out:    (1, 64, 56, 56)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_tf_opid15,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 56, 56})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 64, 64), (32, 64, 3, 3)
// Out:    (1, 32, 64, 64)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_2d_unet_graph_transform_opid94,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 64, 64})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 73, 73), (64, 64, 1, 7)
// Out:    (1, 64, 73, 73)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid43,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 7})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 73, 73})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 73, 73), (64, 64, 7, 1)
// Out:    (1, 64, 73, 73)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid48,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({7, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 73, 73})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 672, 1, 1), (28, 672, 1, 1)
// Out:    (1, 28, 1, 1)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid376,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(28), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 672, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 672, 20, 20), (192, 672, 1, 1)
// Out:    (1, 192, 20, 20)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid475,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(192), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 672, 20, 20})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 672, 40, 40), (112, 672, 1, 1)
// Out:    (1, 112, 40, 40)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid387,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(112), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 672, 40, 40})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 768, 26, 26), (256, 768, 1, 1)
// Out:    (1, 256, 26, 26)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_yolo_v3_tf_opid396,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 768, 26, 26})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 8, 1, 1), (32, 8, 1, 1)
// Out:    (1, 32, 1, 1)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid23,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 8, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 80, 1, 1), (1920, 80, 1, 1)
// Out:    (1, 1920, 1, 1)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid635,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(1920), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 80, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 80, 40, 40), (480, 80, 1, 1)
// Out:    (1, 480, 40, 40)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid253,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(480), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 80, 40, 40})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 96, 1, 1), (4, 96, 1, 1)
// Out:    (1, 4, 1, 1)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid68,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(4), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 96, 1, 1})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 96, 100, 171), (96, 96, 3, 3)
// Out:    (1, 96, 100, 171)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid50,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(96), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 96, 100, 171})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 96, 160, 160), (24, 96, 1, 1)
// Out:    (1, 24, 160, 160)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid79,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(24), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 96, 160, 160})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 96, 19, 19), (576, 96, 1, 1)
// Out:    (1, 576, 19, 19)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid166,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(576), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 96, 19, 19})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 96, 35, 35), (96, 96, 3, 3)
// Out:    (1, 96, 35, 35)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid128,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(96), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 96, 35, 35})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 96, 50, 86), (128, 96, 3, 3)
// Out:    (1, 128, 50, 86)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid146,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 96, 50, 86})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 96, 75, 75), (24, 96, 1, 1)
// Out:    (1, 24, 75, 75)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid30,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(24), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 96, 75, 75})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 960, 10, 10), (160, 960, 1, 1)
// Out:    (1, 160, 10, 10)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid228,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(160), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 960, 10, 10})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 960, 10, 10), (320, 960, 1, 1)
// Out:    (1, 320, 10, 10)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid258,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(320), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 960, 10, 10})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (100, 1024, 15, 15), (256, 1024, 3, 3)
// Out:    (100, 256, 15, 15)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid573,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({100, 1024, 15, 15})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (100, 1024, 4, 4), (128, 1024, 1, 1)
// Out:    (100, 128, 4, 4)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid370,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({100, 1024, 4, 4})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (100, 1024, 4, 4), (160, 1024, 1, 1)
// Out:    (100, 160, 4, 4)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid354,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(160), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({100, 1024, 4, 4})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (100, 1024, 4, 4), (192, 1024, 1, 1)
// Out:    (100, 192, 4, 4)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid344,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(192), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({100, 1024, 4, 4})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (100, 1024, 4, 4), (352, 1024, 1, 1)
// Out:    (100, 352, 4, 4)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid339,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(352), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({100, 1024, 4, 4})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (100, 160, 4, 4), (224, 160, 3, 3)
// Out:    (100, 224, 4, 4)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid359,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(224), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({100, 160, 4, 4})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (100, 192, 4, 4), (224, 192, 3, 3)
// Out:    (100, 224, 4, 4)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid396,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(224), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({100, 192, 4, 4})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (100, 192, 4, 4), (320, 192, 3, 3)
// Out:    (100, 320, 4, 4)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid349,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(320), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({100, 192, 4, 4})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (100, 192, 7, 7), (256, 192, 3, 3)
// Out:    (100, 256, 7, 7)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid327,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({100, 192, 7, 7})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (100, 224, 4, 4), (224, 224, 3, 3)
// Out:    (100, 224, 4, 4)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid364,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(224), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({100, 224, 4, 4})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (100, 256, 15, 15), (90, 256, 3, 3)
// Out:    (100, 90, 15, 15)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid578,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(90), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({100, 256, 15, 15})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (100, 576, 7, 7), (128, 576, 1, 1)
// Out:    (100, 128, 7, 7)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid312,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({100, 576, 7, 7})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (100, 576, 7, 7), (192, 576, 1, 1)
// Out:    (100, 192, 7, 7)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid322,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(192), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({100, 576, 7, 7})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 128, 100, 171), (160, 128, 3, 3)
// Out:    (1, 160, 50, 86)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid104,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(160), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 100, 171})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 3, 300, 300), (32, 3, 3, 3)
// Out:    (1, 32, 150, 150)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_ssd_mobilenet_v2_coco_opid6,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 3, 300, 300})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 3, 640, 640), (32, 3, 3, 3)
// Out:    (1, 32, 320, 320)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid6,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 3, 640, 640})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 96, 100, 171), (96, 96, 3, 3)
// Out:    (1, 96, 50, 86)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid119,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(96), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 96, 100, 171})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (100, 128, 7, 7), (192, 128, 3, 3)
// Out:    (100, 192, 4, 4)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid317,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(192), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({100, 128, 7, 7})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (100, 256, 7, 7), (256, 256, 3, 3)
// Out:    (100, 256, 4, 4)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid332,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({100, 256, 7, 7})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '0,0,0', 'pads_end': '0,0,0', 'strides': '1,1,1'}
// In:     (1, 1, 144, 144, 144), (16, 1, 3, 3, 3)
// Out:    (1, 16, 144, 144, 144)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_3d_unet_graph_transform_opid2,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
            ::testing::Values(16), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1, 144, 144, 144})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '0,0,0', 'pads_end': '0,0,0', 'strides': '1,1,1'}
// In:     (1, 128, 18, 18, 18), (128, 128, 3, 3, 3)
// Out:    (1, 128, 18, 18, 18)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_3d_unet_graph_transform_opid40,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 18, 18, 18})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '0,0,0', 'pads_end': '0,0,0', 'strides': '1,1,1'}
// In:     (1, 128, 36, 36, 36), (64, 128, 3, 3, 3)
// Out:    (1, 64, 36, 36, 36)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_3d_unet_graph_transform_opid78,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 36, 36, 36})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '0,0,0', 'pads_end': '0,0,0', 'strides': '1,1,1'}
// In:     (1, 128, 9, 9, 9), (256, 128, 3, 3, 3)
// Out:    (1, 256, 9, 9, 9)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_3d_unet_graph_transform_opid46,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 128, 9, 9, 9})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '0,0,0', 'pads_end': '0,0,0', 'strides': '1,1,1'}
// In:     (1, 16, 144, 144, 144), (16, 16, 3, 3, 3)
// Out:    (1, 16, 144, 144, 144)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_3d_unet_graph_transform_opid115,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
            ::testing::Values(16), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 16, 144, 144, 144})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '0,0,0', 'pads_end': '0,0,0', 'strides': '1,1,1'}
// In:     (1, 16, 72, 72, 72), (32, 16, 3, 3, 3)
// Out:    (1, 32, 72, 72, 72)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_3d_unet_graph_transform_opid13,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 16, 72, 72, 72})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '0,0,0', 'pads_end': '0,0,0', 'strides': '1,1,1'}
// In:     (1, 256, 18, 18, 18), (128, 256, 3, 3, 3)
// Out:    (1, 128, 18, 18, 18)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_3d_unet_graph_transform_opid62,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 18, 18, 18})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '0,0,0', 'pads_end': '0,0,0', 'strides': '1,1,1'}
// In:     (1, 256, 9, 9, 9), (256, 256, 3, 3, 3)
// Out:    (1, 256, 9, 9, 9)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_3d_unet_graph_transform_opid51,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 9, 9, 9})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '0,0,0', 'pads_end': '0,0,0', 'strides': '1,1,1'}
// In:     (1, 32, 144, 144, 144), (16, 32, 3, 3, 3)
// Out:    (1, 16, 144, 144, 144)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_3d_unet_graph_transform_opid110,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
            ::testing::Values(16), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 144, 144, 144})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '0,0,0', 'pads_end': '0,0,0', 'strides': '1,1,1'}
// In:     (1, 32, 36, 36, 36), (64, 32, 3, 3, 3)
// Out:    (1, 64, 36, 36, 36)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_3d_unet_graph_transform_opid24,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 36, 36, 36})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '0,0,0', 'pads_end': '0,0,0', 'strides': '1,1,1'}
// In:     (1, 32, 72, 72, 72), (32, 32, 3, 3, 3)
// Out:    (1, 32, 72, 72, 72)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_3d_unet_graph_transform_opid18,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 72, 72, 72})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '0,0,0', 'pads_end': '0,0,0', 'strides': '1,1,1'}
// In:     (1, 64, 18, 18, 18), (128, 64, 3, 3, 3)
// Out:    (1, 128, 18, 18, 18)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_3d_unet_graph_transform_opid35,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
            ::testing::Values(128), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 18, 18, 18})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '0,0,0', 'pads_end': '0,0,0', 'strides': '1,1,1'}
// In:     (1, 64, 36, 36, 36), (64, 64, 3, 3, 3)
// Out:    (1, 64, 36, 36, 36)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_3d_unet_graph_transform_opid29,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 36, 36, 36})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'same_upper', 'dilations': '1,1,1', 'pads_begin': '0,0,0', 'pads_end': '0,0,0', 'strides': '1,1,1'}
// In:     (1, 64, 72, 72, 72), (32, 64, 3, 3, 3)
// Out:    (1, 32, 72, 72, 72)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_3d_unet_graph_transform_opid94,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::SAME_UPPER)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 72, 72, 72})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 16, 128, 128), (1, 16, 1, 1)
// Out:    (1, 1, 128, 128)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_2d_unet_graph_transform_opid120,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(1), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 16, 128, 128})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 24, 400, 683), (64, 24, 1, 1)
// Out:    (1, 64, 400, 683)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_mask_rcnn_inception_v2_coco_opid8,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(64), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 24, 400, 683})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 32, 149, 149), (32, 32, 3, 3)
// Out:    (1, 32, 147, 147)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid11,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 32, 149, 149})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 64, 73, 73), (96, 64, 3, 3)
// Out:    (1, 96, 71, 71)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid33,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(96), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 73, 73})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 88, 10, 10), (36, 88, 1, 1)
// Out:    (1, 36, 10, 10)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid1225,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(36), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 88, 10, 10})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 88, 10, 10), (810, 88, 1, 1)
// Out:    (1, 810, 10, 10)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid1396,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(810), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 88, 10, 10})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 88, 10, 10), (88, 88, 1, 1)
// Out:    (1, 88, 10, 10)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid1021,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(88), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 88, 10, 10})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 88, 20, 20), (36, 88, 1, 1)
// Out:    (1, 36, 20, 20)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid1178,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(36), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 88, 20, 20})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 88, 20, 20), (810, 88, 1, 1)
// Out:    (1, 810, 20, 20)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid1365,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(810), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 88, 20, 20})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 88, 20, 20), (88, 88, 1, 1)
// Out:    (1, 88, 20, 20)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid1033,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(88), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 88, 20, 20})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 88, 40, 40), (36, 88, 1, 1)
// Out:    (1, 36, 40, 40)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid1131,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(36), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 88, 40, 40})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 88, 40, 40), (810, 88, 1, 1)
// Out:    (1, 810, 40, 40)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid1334,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(810), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 88, 40, 40})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 88, 40, 40), (88, 88, 1, 1)
// Out:    (1, 88, 40, 40)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid1045,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(88), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 88, 40, 40})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 88, 5, 5), (36, 88, 1, 1)
// Out:    (1, 36, 5, 5)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid1269,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(36), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 88, 5, 5})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 88, 5, 5), (810, 88, 1, 1)
// Out:    (1, 810, 5, 5)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid1427,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(810), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 88, 5, 5})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 88, 5, 5), (88, 88, 1, 1)
// Out:    (1, 88, 5, 5)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid1009,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(88), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 88, 5, 5})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 88, 80, 80), (36, 88, 1, 1)
// Out:    (1, 36, 80, 80)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid1084,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(36), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 88, 80, 80})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 88, 80, 80), (810, 88, 1, 1)
// Out:    (1, 810, 80, 80)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid1303,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(810), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 88, 80, 80})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '1,1'}
// In:     (1, 88, 80, 80), (88, 88, 1, 1)
// Out:    (1, 88, 80, 80)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_efficientdet_d1_tf_opid1057,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(88), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 88, 80, 80})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 1024, 14, 14), (2048, 1024, 1, 1)
// Out:    (1, 2048, 7, 7)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_tf_opid244,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(2048), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 1024, 14, 14})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 192, 17, 17), (192, 192, 3, 3)
// Out:    (1, 192, 8, 8)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid605,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(192), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 192, 17, 17})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 192, 71, 71), (192, 192, 3, 3)
// Out:    (1, 192, 35, 35)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid59,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(192), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 192, 71, 71})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 224, 35, 35), (256, 224, 3, 3)
// Out:    (1, 256, 17, 17)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid229,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(256), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 224, 35, 35})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 256, 56, 56), (512, 256, 1, 1)
// Out:    (1, 512, 28, 28)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_tf_opid76,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(512), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 256, 56, 56})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 3, 299, 299), (32, 3, 3, 3)
// Out:    (1, 32, 149, 149)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid6,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(32), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 3, 299, 299})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 320, 17, 17), (320, 320, 3, 3)
// Out:    (1, 320, 8, 8)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid625,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(320), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 320, 17, 17})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 384, 35, 35), (384, 384, 3, 3)
// Out:    (1, 384, 17, 17)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid214,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(384), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 384, 35, 35})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 512, 28, 28), (1024, 512, 1, 1)
// Out:    (1, 1024, 14, 14)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_resnet_50_tf_opid144,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(1024), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 512, 28, 28})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1', 'pads_begin': '0,0', 'pads_end': '0,0', 'strides': '2,2'}
// In:     (1, 64, 147, 147), (96, 64, 3, 3)
// Out:    (1, 96, 73, 73)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_googlenet_v4_tf_opid22,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({3, 3})), // kernel
            ::testing::Values(std::vector<size_t>({2, 2})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1})), // dilations
            ::testing::Values(96), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 64, 147, 147})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);


// Attrs:  {'auto_pad': 'valid', 'dilations': '1,1,1', 'pads_begin': '0,0,0', 'pads_end': '0,0,0', 'strides': '1,1,1'}
// In:     (1, 16, 144, 144, 144), (1, 16, 1, 1, 1)
// Out:    (1, 1, 144, 144, 144)
// Operators
INSTANTIATE_TEST_CASE_P(
    autogen_Convolution_3d_unet_graph_transform_opid120,
    ConvolutionCudaLayerThresholdTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // kernel
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // strides
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_begin
            ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), // pads_end
            ::testing::Values(std::vector<size_t>({1, 1, 1})), // dilations
            ::testing::Values(1), // Num out channels
            ::testing::Values(ov::op::PadType::VALID)), // Padding type
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})), // Net precisions
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY), // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY), // Output layout
        ::testing::Values(std::vector<size_t>({1, 16, 144, 144, 144})), // Input shape
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    ConvolutionCudaLayerThresholdTest::getTestCaseName);

// {AUTOGENERATED_TESTS_END_TAG}
// clang-format on
// =============================================================================

}  // namespace
