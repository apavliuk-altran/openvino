// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/interpolate.hpp"
#include "common_test_utils/test_constants.hpp"

#include "benchmark.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::FP32,
};

const std::vector<std::vector<size_t>> inShapes = {
        {1, 4, 6, 6},
};

const  std::vector<ngraph::op::v4::Interpolate::InterpolateMode> modesWithoutNearest = {
        ngraph::op::v4::Interpolate::InterpolateMode::LINEAR,
        ngraph::op::v4::Interpolate::InterpolateMode::LINEAR_ONNX,
        ngraph::op::v4::Interpolate::InterpolateMode::CUBIC,
};

const  std::vector<ngraph::op::v4::Interpolate::InterpolateMode> nearestMode = {
        ngraph::op::v4::Interpolate::InterpolateMode::NEAREST,
};

const std::vector<ngraph::op::v4::Interpolate::CoordinateTransformMode> coordinateTransformModes = {
        ngraph::op::v4::Interpolate::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::HALF_PIXEL,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::ASYMMETRIC,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::ALIGN_CORNERS,
};

const std::vector<ngraph::op::v4::Interpolate::ShapeCalcMode> shapeCalculationMode = {
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,
        ngraph::op::v4::Interpolate::ShapeCalcMode::SCALES,
};

const std::vector<ngraph::op::v4::Interpolate::NearestMode> nearestModes = {
        ngraph::op::v4::Interpolate::NearestMode::SIMPLE,
        ngraph::op::v4::Interpolate::NearestMode::ROUND_PREFER_FLOOR,
        ngraph::op::v4::Interpolate::NearestMode::FLOOR,
        ngraph::op::v4::Interpolate::NearestMode::CEIL,
        ngraph::op::v4::Interpolate::NearestMode::ROUND_PREFER_CEIL,
};

const std::vector<ngraph::op::v4::Interpolate::NearestMode> defaultNearestMode = {
        ngraph::op::v4::Interpolate::NearestMode::ROUND_PREFER_FLOOR,
};

const std::vector<std::vector<size_t>> pads = {
        {0, 0, 1, 1},
        {0, 0, 0, 0},
};

const std::vector<bool> antialias = {
// Not enabled in Inference Engine
//        true,
        false,
};

const std::vector<double> cubeCoefs = {
        -0.75f,
};

const std::vector<std::vector<int64_t>> defaultAxes = {
    {0, 1, 2, 3}
};

const std::vector<std::vector<size_t>> targetShapes = {
    {1, 4, 8, 8},
};

const std::vector<std::vector<float>> defaultScales = {
    {1.f, 1.f, 1.333333f, 1.333333f}
};

std::map<std::string, std::string> additional_config = {};

const auto interpolateCasesWithoutNearest = ::testing::Combine(
        ::testing::ValuesIn(modesWithoutNearest),
        ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModes),
        ::testing::ValuesIn(defaultNearestMode),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs),
        ::testing::ValuesIn(defaultAxes),
        ::testing::ValuesIn(defaultScales));

const auto interpolateCases = ::testing::Combine(
        ::testing::ValuesIn(nearestMode),
        ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModes),
        ::testing::ValuesIn(nearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs),
        ::testing::ValuesIn(defaultAxes),
        ::testing::ValuesIn(defaultScales));

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Basic, InterpolateLayerTest, ::testing::Combine(
        interpolateCasesWithoutNearest,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(targetShapes),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(additional_config)),
    InterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Nearest, InterpolateLayerTest, ::testing::Combine(
        interpolateCases,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(targetShapes),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(additional_config)),
    InterpolateLayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> targetShapesTailTest = {
        {1, 4, 2, 11},  // cover down sample and tails process code path
};

const std::vector<std::vector<float>> defaultScalesTailTest = {
    {1.f, 1.f, 0.333333f, 1.833333f}
};

const auto interpolateCasesWithoutNearestTail = ::testing::Combine(
        ::testing::ValuesIn(modesWithoutNearest),
        ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModes),
        ::testing::ValuesIn(defaultNearestMode),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs),
        ::testing::ValuesIn(defaultAxes),
        ::testing::ValuesIn(defaultScalesTailTest));

const auto interpolateCasesTail = ::testing::Combine(
        ::testing::ValuesIn(nearestMode),
        ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModes),
        ::testing::ValuesIn(nearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs),
        ::testing::ValuesIn(defaultAxes),
        ::testing::ValuesIn(defaultScalesTailTest));

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Basic_Down_Sample_Tail, InterpolateLayerTest, ::testing::Combine(
        interpolateCasesWithoutNearestTail,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(targetShapesTailTest),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(additional_config)),
    InterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Nearest_Down_Sample_Tail, InterpolateLayerTest, ::testing::Combine(
        interpolateCasesTail,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(targetShapesTailTest),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(additional_config)),
    InterpolateLayerTest::getTestCaseName);


////////////////////////Benchmark/////////////////////////////
namespace benchmark {

struct InterpolateLayerBenchmarkTest : BenchmarkLayerTest<InterpolateLayerTest> {};

TEST_P(InterpolateLayerBenchmarkTest, benchmark) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run("Interpolate", std::chrono::milliseconds(2000), 100);
}

std::vector<ngraph::op::v4::Interpolate::InterpolateMode> interpolateModesBenchmark{
    ngraph::op::v4::Interpolate::InterpolateMode::NEAREST,
//     ngraph::op::v4::Interpolate::InterpolateMode::LINEAR,
//     ngraph::op::v4::Interpolate::InterpolateMode::LINEAR_ONNX,
//     ngraph::op::v4::Interpolate::InterpolateMode::CUBIC,
};

const std::vector<ngraph::op::v4::Interpolate::CoordinateTransformMode> coordinateTransformModesBenchmark = {
        ngraph::op::v4::Interpolate::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::HALF_PIXEL,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::ASYMMETRIC,
        // ngraph::op::v4::Interpolate::CoordinateTransformMode::ALIGN_CORNERS,
};

const auto interpolateCasesBenchmark = ::testing::Combine(
        ::testing::ValuesIn(interpolateModesBenchmark),       // InterpolateMode
        ::testing::Values(ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES),   // ShapeCalculationMode
        ::testing::ValuesIn(coordinateTransformModesBenchmark),    // CoordinateTransformMode
        ::testing::ValuesIn(nearestModes),    // NearestMode
        ::testing::Values(false),       // AntiAlias
        ::testing::ValuesIn(pads),        // PadBegin
        ::testing::ValuesIn(pads),        // PadEnd
        ::testing::ValuesIn(cubeCoefs),        // Cube coef
        ::testing::ValuesIn(defaultAxes),        // Axes
        ::testing::Values(std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f}));      // Scales

INSTANTIATE_TEST_SUITE_P(Interpolate_Benchmark_Test_00, InterpolateLayerBenchmarkTest, ::testing::Combine(
        interpolateCasesBenchmark,
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::I8}), // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Output layout

        ::testing::Values(std::vector<size_t>{1, 1, 80, 9150}),                            // Input shapes
        ::testing::Values(std::vector<size_t>{1, 1, 80, 100650}),                          // Target shapes

        // ::testing::Values(std::vector<size_t>{1, 1, 128, 99550}),                            // Input shapes
        // ::testing::Values(std::vector<size_t>{1, 1, 128, 497750}),                          // Target shapes

        // ::testing::Values(std::vector<size_t>{1, 1, 128, 196}),                            // Input shapes
        // ::testing::Values(std::vector<size_t>{1, 1, 128, 53900}),                          // Target shapes

        // ::testing::Values(std::vector<size_t>{1, 1, 80, 200}),                            // Input shapes
        // ::testing::Values(std::vector<size_t>{1, 1, 80, 1000}),                          // Target shapes

        // ::testing::Values(std::vector<size_t>{1, 1, 80, 1000}),                            // Input shapes
        // ::testing::Values(std::vector<size_t>{1, 1, 80, 5000}),                          // Target shapes

        // ::testing::Values(std::vector<size_t>{1, 1, 80, 5000}),                            // Input shapes
        // ::testing::Values(std::vector<size_t>{1, 1, 80, 55000}),                          // Target shapes

        // ::testing::Values(std::vector<size_t>{1, 128, 6, 10}),                            // Input shapes
        // ::testing::Values(std::vector<size_t>{1, 128, 24, 40}),                          // Target shapes

        // ::testing::Values(std::vector<size_t>{1, 128, 23, 40}),                            // Input shapes
        // ::testing::Values(std::vector<size_t>{1, 128, 46, 80}),                          // Target shapes

        // ::testing::Values(std::vector<size_t>{1, 64, 45, 80}),                            // Input shapes
        // ::testing::Values(std::vector<size_t>{1, 64, 90, 160}),                          // Target shapes

        // ::testing::Values(std::vector<size_t>{1, 32, 90, 160}),                            // Input shapes
        // ::testing::Values(std::vector<size_t>{1, 32, 180, 320}),                          // Target shapes

        // ::testing::Values(std::vector<size_t>{1, 16, 180, 320}),                            // Input shapes
        // ::testing::Values(std::vector<size_t>{1, 16, 360, 640}),                          // Target shapes

        ::testing::Values(CommonTestUtils::DEVICE_CPU),                   // Device name
        ::testing::Values(additional_config)),                          // Additional network configuration
    InterpolateLayerBenchmarkTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Interpolate_Benchmark_Test_01, InterpolateLayerBenchmarkTest, ::testing::Combine(
        interpolateCasesBenchmark,
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::I8}), // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Output layout
        ::testing::Values(std::vector<size_t>{1, 1, 128, 99550}),                            // Input shapes
        ::testing::Values(std::vector<size_t>{1, 1, 128, 497750}),                          // Target shapes
        ::testing::Values(CommonTestUtils::DEVICE_CPU),                   // Device name
        ::testing::Values(additional_config)),                          // Additional network configuration
    InterpolateLayerBenchmarkTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Interpolate_Benchmark_Test_02, InterpolateLayerBenchmarkTest, ::testing::Combine(
        interpolateCasesBenchmark,
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::I8}), // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Output layout
        ::testing::Values(std::vector<size_t>{1, 1, 128, 196}),                            // Input shapes
        ::testing::Values(std::vector<size_t>{1, 1, 128, 53900}),                          // Target shapes
        ::testing::Values(CommonTestUtils::DEVICE_CPU),                   // Device name
        ::testing::Values(additional_config)),                          // Additional network configuration
    InterpolateLayerBenchmarkTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Interpolate_Benchmark_Test_03, InterpolateLayerBenchmarkTest, ::testing::Combine(
        interpolateCasesBenchmark,
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::I8}), // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Output layout
        ::testing::Values(std::vector<size_t>{1, 1, 80, 200}),                            // Input shapes
        ::testing::Values(std::vector<size_t>{1, 1, 80, 1000}),                          // Target shapes
        ::testing::Values(CommonTestUtils::DEVICE_CPU),                   // Device name
        ::testing::Values(additional_config)),                          // Additional network configuration
    InterpolateLayerBenchmarkTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Interpolate_Benchmark_Test_04, InterpolateLayerBenchmarkTest, ::testing::Combine(
        interpolateCasesBenchmark,
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::I8}), // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Output layout
        ::testing::Values(std::vector<size_t>{1, 1, 80, 1000}),                            // Input shapes
        ::testing::Values(std::vector<size_t>{1, 1, 80, 5000}),                          // Target shapes
        ::testing::Values(CommonTestUtils::DEVICE_CPU),                   // Device name
        ::testing::Values(additional_config)),                          // Additional network configuration
    InterpolateLayerBenchmarkTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Interpolate_Benchmark_Test_05, InterpolateLayerBenchmarkTest, ::testing::Combine(
        interpolateCasesBenchmark,
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::I8}), // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Output layout
        ::testing::Values(std::vector<size_t>{1, 1, 80, 5000}),                            // Input shapes
        ::testing::Values(std::vector<size_t>{1, 1, 80, 55000}),                          // Target shapes
        ::testing::Values(CommonTestUtils::DEVICE_CPU),                   // Device name
        ::testing::Values(additional_config)),                          // Additional network configuration
    InterpolateLayerBenchmarkTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Interpolate_Benchmark_Test_06, InterpolateLayerBenchmarkTest, ::testing::Combine(
        interpolateCasesBenchmark,
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::I8}), // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Output layout
        ::testing::Values(std::vector<size_t>{1, 128, 6, 10}),                            // Input shapes
        ::testing::Values(std::vector<size_t>{1, 128, 24, 40}),                          // Target shapes
        ::testing::Values(CommonTestUtils::DEVICE_CPU),                   // Device name
        ::testing::Values(additional_config)),                          // Additional network configuration
    InterpolateLayerBenchmarkTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Interpolate_Benchmark_Test_07, InterpolateLayerBenchmarkTest, ::testing::Combine(
        interpolateCasesBenchmark,
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::I8}), // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Output layout
        ::testing::Values(std::vector<size_t>{1, 128, 23, 40}),                            // Input shapes
        ::testing::Values(std::vector<size_t>{1, 128, 46, 80}),                          // Target shapes
        ::testing::Values(CommonTestUtils::DEVICE_CPU),                   // Device name
        ::testing::Values(additional_config)),                          // Additional network configuration
    InterpolateLayerBenchmarkTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Interpolate_Benchmark_Test_08, InterpolateLayerBenchmarkTest, ::testing::Combine(
        interpolateCasesBenchmark,
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::I8}), // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Output layout
        ::testing::Values(std::vector<size_t>{1, 64, 45, 80}),                            // Input shapes
        ::testing::Values(std::vector<size_t>{1, 64, 90, 160}),                          // Target shapes
        ::testing::Values(CommonTestUtils::DEVICE_CPU),                   // Device name
        ::testing::Values(additional_config)),                          // Additional network configuration
    InterpolateLayerBenchmarkTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Interpolate_Benchmark_Test_09, InterpolateLayerBenchmarkTest, ::testing::Combine(
        interpolateCasesBenchmark,
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::I8}), // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Output layout
        ::testing::Values(std::vector<size_t>{1, 32, 90, 160}),                            // Input shapes
        ::testing::Values(std::vector<size_t>{1, 32, 180, 320}),                          // Target shapes
        ::testing::Values(CommonTestUtils::DEVICE_CPU),                   // Device name
        ::testing::Values(additional_config)),                          // Additional network configuration
    InterpolateLayerBenchmarkTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Interpolate_Benchmark_Test_10, InterpolateLayerBenchmarkTest, ::testing::Combine(
        interpolateCasesBenchmark,
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::I8}), // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Output layout
        ::testing::Values(std::vector<size_t>{1, 16, 180, 320}),                            // Input shapes
        ::testing::Values(std::vector<size_t>{1, 16, 360, 640}),                          // Target shapes
        ::testing::Values(CommonTestUtils::DEVICE_CPU),                   // Device name
        ::testing::Values(additional_config)),                          // Additional network configuration
    InterpolateLayerBenchmarkTest::getTestCaseName);


////////////////////////I8 vs F32 precision/////////////////////////////

INSTANTIATE_TEST_SUITE_P(Interpolate_I8_Test_00, InterpolateLayerTest, ::testing::Combine(
        interpolateCasesBenchmark,
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::I8}), // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Output layout
        ::testing::Values(std::vector<size_t>{1, 1, 80, 9150}),                            // Input shapes
        ::testing::Values(std::vector<size_t>{1, 1, 80, 100650}),                          // Target shapes
        ::testing::Values(CommonTestUtils::DEVICE_CPU),                   // Device name
        ::testing::Values(additional_config)),                          // Additional network configuration
    InterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Interpolate_I8_Test_01, InterpolateLayerTest, ::testing::Combine(
        interpolateCasesBenchmark,
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::I8}), // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Output layout
        ::testing::Values(std::vector<size_t>{1, 1, 128, 99550}),                            // Input shapes
        ::testing::Values(std::vector<size_t>{1, 1, 128, 497750}),                          // Target shapes
        ::testing::Values(CommonTestUtils::DEVICE_CPU),                   // Device name
        ::testing::Values(additional_config)),                          // Additional network configuration
    InterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Interpolate_I8_Test_02, InterpolateLayerTest, ::testing::Combine(
        interpolateCasesBenchmark,
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::I8}), // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Output layout
        ::testing::Values(std::vector<size_t>{1, 1, 128, 196}),                            // Input shapes
        ::testing::Values(std::vector<size_t>{1, 1, 128, 53900}),                          // Target shapes
        ::testing::Values(CommonTestUtils::DEVICE_CPU),                   // Device name
        ::testing::Values(additional_config)),                          // Additional network configuration
    InterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Interpolate_I8_Test_03, InterpolateLayerTest, ::testing::Combine(
        interpolateCasesBenchmark,
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::I8}), // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Output layout
        ::testing::Values(std::vector<size_t>{1, 1, 80, 200}),                            // Input shapes
        ::testing::Values(std::vector<size_t>{1, 1, 80, 1000}),                          // Target shapes
        ::testing::Values(CommonTestUtils::DEVICE_CPU),                   // Device name
        ::testing::Values(additional_config)),                          // Additional network configuration
    InterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Interpolate_I8_Test_04, InterpolateLayerTest, ::testing::Combine(
        interpolateCasesBenchmark,
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::I8}), // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Output layout
        ::testing::Values(std::vector<size_t>{1, 1, 80, 1000}),                            // Input shapes
        ::testing::Values(std::vector<size_t>{1, 1, 80, 5000}),                          // Target shapes
        ::testing::Values(CommonTestUtils::DEVICE_CPU),                   // Device name
        ::testing::Values(additional_config)),                          // Additional network configuration
    InterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Interpolate_I8_Test_05, InterpolateLayerTest, ::testing::Combine(
        interpolateCasesBenchmark,
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::I8}), // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Output layout
        ::testing::Values(std::vector<size_t>{1, 1, 80, 5000}),                            // Input shapes
        ::testing::Values(std::vector<size_t>{1, 1, 80, 55000}),                          // Target shapes
        ::testing::Values(CommonTestUtils::DEVICE_CPU),                   // Device name
        ::testing::Values(additional_config)),                          // Additional network configuration
    InterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Interpolate_I8_Test_06, InterpolateLayerTest, ::testing::Combine(
        interpolateCasesBenchmark,
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::I8}), // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Output layout
        ::testing::Values(std::vector<size_t>{1, 128, 6, 10}),                            // Input shapes
        ::testing::Values(std::vector<size_t>{1, 128, 24, 40}),                          // Target shapes
        ::testing::Values(CommonTestUtils::DEVICE_CPU),                   // Device name
        ::testing::Values(additional_config)),                          // Additional network configuration
    InterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Interpolate_I8_Test_07, InterpolateLayerTest, ::testing::Combine(
        interpolateCasesBenchmark,
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::I8}), // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Output layout
        ::testing::Values(std::vector<size_t>{1, 128, 23, 40}),                            // Input shapes
        ::testing::Values(std::vector<size_t>{1, 128, 46, 80}),                          // Target shapes
        ::testing::Values(CommonTestUtils::DEVICE_CPU),                   // Device name
        ::testing::Values(additional_config)),                          // Additional network configuration
    InterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Interpolate_I8_Test_08, InterpolateLayerTest, ::testing::Combine(
        interpolateCasesBenchmark,
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::I8}), // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Output layout
        ::testing::Values(std::vector<size_t>{1, 64, 45, 80}),                            // Input shapes
        ::testing::Values(std::vector<size_t>{1, 64, 90, 160}),                          // Target shapes
        ::testing::Values(CommonTestUtils::DEVICE_CPU),                   // Device name
        ::testing::Values(additional_config)),                          // Additional network configuration
    InterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Interpolate_I8_Test_09, InterpolateLayerTest, ::testing::Combine(
        interpolateCasesBenchmark,
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::I8}), // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Output layout
        ::testing::Values(std::vector<size_t>{1, 32, 90, 160}),                            // Input shapes
        ::testing::Values(std::vector<size_t>{1, 32, 180, 320}),                          // Target shapes
        ::testing::Values(CommonTestUtils::DEVICE_CPU),                   // Device name
        ::testing::Values(additional_config)),                          // Additional network configuration
    InterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Interpolate_I8_Test_10, InterpolateLayerTest, ::testing::Combine(
        interpolateCasesBenchmark,
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>{InferenceEngine::Precision::FP32, InferenceEngine::Precision::I8}), // Net precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Input precision
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),       // Output precision
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Input layout
        ::testing::Values(InferenceEngine::Layout::ANY),                  // Output layout
        ::testing::Values(std::vector<size_t>{1, 16, 180, 320}),                            // Input shapes
        ::testing::Values(std::vector<size_t>{1, 16, 360, 640}),                          // Target shapes
        ::testing::Values(CommonTestUtils::DEVICE_CPU),                   // Device name
        ::testing::Values(additional_config)),                          // Additional network configuration
    InterpolateLayerTest::getTestCaseName);




// wavernn-upsampler.xml
// <data antialias="false" coordinate_transformation_mode="asymmetric" cube_coeff="-0.75" mode="nearest"
// nearest_mode="floor" pads_begin="0, 0, 0, 0" pads_end="0, 0, 0, 0" shape_calculation_mode="sizes"/>

// vcam_model/network.xml
// <data align_corners="0" axes="2,3" axis="2,3" coordinate_transformation_mode="asymmetric" mode="linear"
// pad_beg="0" pad_end="0" shape_calculation_mode="sizes" zoom_factor="2"  />

// <data align_corners="0" axes="2,3" axis="2,3" coordinate_transformation_mode="asymmetric" mode="linear"
// pad_beg="0" pad_end="0" shape_calculation_mode="sizes" zoom_factor="4"  />

}  // namespace benchmark
} // namespace
