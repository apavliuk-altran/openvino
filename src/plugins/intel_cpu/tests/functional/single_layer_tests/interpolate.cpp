// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/fusing_test_utils.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "benchmark.hpp"

using namespace ov::test;
using namespace CPUTestUtils;
using ngraph::helpers::operator<<;

namespace CPULayerTestsDefinitions {

using InterpolateSpecificParams = std::tuple<ngraph::op::v4::Interpolate::InterpolateMode,          // InterpolateMode
                                             ngraph::op::v4::Interpolate::CoordinateTransformMode,  // CoordinateTransformMode
                                             ngraph::op::v4::Interpolate::NearestMode,              // NearestMode
                                             bool,                                                  // AntiAlias
                                             std::vector<size_t>,                                   // PadBegin
                                             std::vector<size_t>,                                   // PadEnd
                                             double>;                                               // Cube coef

using ShapeParams = std::tuple<ngraph::op::v4::Interpolate::ShapeCalcMode, // ShapeCalculationMode
                               InputShape,                                 // Input shapes
                               // params describing input, choice of which depends on ShapeCalcMode
                               ngraph::helpers::InputLayerType,            // input type
                               std::vector<std::vector<float>>,            // scales or sizes values
                               std::vector<int64_t>>;                      // axes

using InterpolateLayerCPUTestParamsSet = std::tuple<InterpolateSpecificParams,
                                                    ShapeParams,
                                                    ElementType,
                                                    CPUSpecificParams,
                                                    fusingSpecificParams,
                                                    std::map<std::string, std::string>>;

class InterpolateLayerCPUTest : public testing::WithParamInterface<InterpolateLayerCPUTestParamsSet>,
                                virtual public SubgraphBaseTest, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InterpolateLayerCPUTestParamsSet> obj) {
        InterpolateSpecificParams specificParams;
        ShapeParams shapeParams;
        ElementType prec;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::map<std::string, std::string> additionalConfig;
        std::tie(specificParams, shapeParams, prec, cpuParams, fusingParams, additionalConfig) = obj.param;

        ngraph::op::v4::Interpolate::InterpolateMode mode;
        ngraph::op::v4::Interpolate::CoordinateTransformMode transfMode;
        ngraph::op::v4::Interpolate::NearestMode nearMode;
        bool antiAlias;
        std::vector<size_t> padBegin;
        std::vector<size_t> padEnd;
        double cubeCoef;
        std::tie(mode, transfMode, nearMode, antiAlias, padBegin, padEnd, cubeCoef) = specificParams;

        ngraph::op::v4::Interpolate::ShapeCalcMode shapeCalcMode;
        InputShape inputShapes;
        ngraph::helpers::InputLayerType shapeInputType;
        std::vector<std::vector<float>> shapeDataForInput;
        std::vector<int64_t> axes;
        std::tie(shapeCalcMode, inputShapes, shapeInputType, shapeDataForInput, axes) = shapeParams;

        std::ostringstream result;
        result << "ShapeCalcMode=" << shapeCalcMode << "_";
        result << "IS=";
        result << CommonTestUtils::partialShape2str({inputShapes.first}) << "_";
        result << "TS=";
        for (const auto& shape : inputShapes.second) {
            result << CommonTestUtils::vec2str(shape) << "_";
        }
        if (shapeCalcMode == ngraph::op::v4::Interpolate::ShapeCalcMode::SCALES) {
            result << "Scales=";
        } else {
            result << "Sizes=";
        }
        for (const auto &data : shapeDataForInput) {
            result << CommonTestUtils::vec2str(data) << "_";
        }
        result << shapeInputType << "_";
        result << "InterpolateMode=" << mode << "_";
        result << "CoordinateTransformMode=" << transfMode << "_";
        result << "NearestMode=" << nearMode << "_";
        result << "CubeCoef=" << cubeCoef << "_";
        result << "Antialias=" << antiAlias << "_";
        result << "PB=" << CommonTestUtils::vec2str(padBegin) << "_";
        result << "PE=" << CommonTestUtils::vec2str(padEnd) << "_";
        result << "Axes=" << CommonTestUtils::vec2str(axes) << "_";
        result << "PRC=" << prec << "_";

        result << CPUTestsBase::getTestCaseName(cpuParams);
        result << CpuTestWithFusing::getTestCaseName(fusingParams);

        if (!additionalConfig.empty()) {
            result << "_PluginConf";
            for (auto& item : additionalConfig) {
                result << "_" << item.first << "=" << item.second;
            }
        }

        return result.str();
    }

    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (int i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;

            if (i == 1) {
                if (shapeCalcMode == ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES) {
                    tensor = ov::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i], sizes[inferRequestNum].data());
                } else {
                    tensor = ov::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i], scales[inferRequestNum].data());
                }
            } else {
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], 2560, 0, 256);
            }

            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
        inferRequestNum++;
    }

    void configure_model() override {
        ov::preprocess::PrePostProcessor p(function);
        {
            auto& params = function->get_parameters();
            for (size_t i = 0; i < params.size(); i++) {
                if (i > 0) {
                    continue;
                }
                if (inType != ov::element::Type_t::undefined) {
                    p.input(i).tensor().set_element_type(inType);
                }
            }
        }
        {
            auto results = function->get_results();
            for (size_t i = 0; i < results.size(); i++) {
                if (outType != ov::element::Type_t::undefined) {
                    p.output(i).tensor().set_element_type(outType);
                }
            }
        }
        function = p.build();
    }

protected:
    std::vector<std::vector<float>> scales;
    std::vector<std::vector<int32_t>> sizes;
    ngraph::op::v4::Interpolate::ShapeCalcMode shapeCalcMode;
    size_t inferRequestNum = 0;

    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;

        InterpolateSpecificParams specificParams;
        ShapeParams shapeParams;
        ElementType ngPrc;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::map<std::string, std::string> additionalConfig;
        std::tie(specificParams, shapeParams, ngPrc, cpuParams, fusingParams, additionalConfig) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        std::tie(postOpMgrPtr, fusedOps) = fusingParams;
        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        ngraph::op::v4::Interpolate::InterpolateMode mode;
        ngraph::op::v4::Interpolate::CoordinateTransformMode transfMode;
        ngraph::op::v4::Interpolate::NearestMode nearMode;
        bool antiAlias;
        std::vector<size_t> padBegin;
        std::vector<size_t> padEnd;
        double cubeCoef;
        std::tie(mode, transfMode, nearMode, antiAlias, padBegin, padEnd, cubeCoef) = specificParams;

        InputShape dataShape;
        ngraph::helpers::InputLayerType shapeInputType;
        std::vector<std::vector<float>> shapeDataForInput;
        std::vector<int64_t> axes;
        std::tie(shapeCalcMode, dataShape, shapeInputType, shapeDataForInput, axes) = shapeParams;

        if (shapeCalcMode == ngraph::op::v4::Interpolate::ShapeCalcMode::SCALES) {
            scales = shapeDataForInput;
            sizes.resize(scales.size(), std::vector<int32_t>(scales.front().size(), 0));
        } else {
            sizes.resize(shapeDataForInput.size());
            for (size_t i = 0; i < shapeDataForInput.size(); i++) {
                for (size_t j = 0; j < shapeDataForInput[i].size(); j++) {
                    sizes[i].push_back(shapeDataForInput[i][j]);
                }
            }
            scales.resize(sizes.size(), std::vector<float>(sizes.front().size(), 0));
        }

        std::vector<InputShape> inputShapes;
        inputShapes.push_back(dataShape);
        if (shapeInputType == ngraph::helpers::InputLayerType::PARAMETER) {
            inputShapes.push_back(InputShape({static_cast<int64_t>(axes.size())}, std::vector<ov::Shape>(dataShape.second.size(), {axes.size()})));
        }

        if (additionalConfig[InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16] == InferenceEngine::PluginConfigParams::YES) {
            inType = outType = ngPrc = ElementType::bf16;
            rel_threshold = 1e-2f;
        } else {
            inType = outType = ngPrc;
        }

        init_input_shapes(inputShapes);

        auto params = ngraph::builder::makeDynamicParams(ngPrc, {inputDynamicShapes.front()});

        std::shared_ptr<ov::Node> sizesInput, scalesInput;
        if (shapeCalcMode == ngraph::op::v4::Interpolate::ShapeCalcMode::SCALES) {
            if (shapeInputType == ngraph::helpers::InputLayerType::PARAMETER) {
                auto paramNode = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::Type_t::f32, ov::Shape{scales.front().size()});
                params.push_back(paramNode);
                scalesInput = paramNode;
            } else {
                scalesInput = std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::f32, ov::Shape{scales.front().size()}, scales.front());
            }
            sizesInput = std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::i32, ov::Shape{sizes.front().size()}, sizes.front());
        } else {
            if (shapeInputType == ngraph::helpers::InputLayerType::PARAMETER) {
                auto paramNode = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::Type_t::i32, ov::Shape{sizes.front().size()});
                params.push_back(paramNode);
                sizesInput = paramNode;
            } else {
                sizesInput = std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::i32, ov::Shape{sizes.front().size()}, sizes.front());
            }
            scalesInput = std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::f32, ov::Shape{scales.front().size()}, scales.front());
        }
        auto axesInput = std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::i64, ov::Shape{axes.size()}, axes);

        for (size_t i = 0; i < params.size(); i++) {
            params[i]->set_friendly_name(std::string("param_") + std::to_string(i));
        }

        ngraph::op::v4::Interpolate::InterpolateAttrs interpAttr{mode, shapeCalcMode, padBegin, padEnd, transfMode, nearMode,
                                                                            antiAlias, cubeCoef};

        auto interpolate = std::make_shared<ngraph::op::v4::Interpolate>(params[0],
                                                                         sizesInput,
                                                                         scalesInput,
                                                                         axesInput,
                                                                         interpAttr);

        function = makeNgraphFunction(ngPrc, params, interpolate, "InterpolateCPU");

        // To be able to run CPU tests on all shapes real thresholds should be ignored
        interpolate->get_rt_info().insert({"enforceAllSupportedLayouts", true});

        if (selectedType.empty()) {
            selectedType = getPrimitiveType();
        }
        selectedType = makeSelectedTypeStr(selectedType, ngPrc);

        const auto selLength = selectedType.length();
        if (selLength > 1 && selectedType[selLength - 2] == 'U' && selectedType[selLength - 1] == '8') {
            selectedType[selLength - 2] = 'I';
        }
    }
};

TEST_P(InterpolateLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
    CheckPluginRelatedResults(compiledModel, "Interpolate");
}

namespace {

/* CPU PARAMS */
std::vector<CPUSpecificParams> filterCPUInfoForDevice(bool forcePlanarForAvx512 = false, bool forcePlanarForSse = false) {
    std::vector<CPUSpecificParams> resCPUParams;
    if (InferenceEngine::with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{nChw16c, x, x, x}, {nChw16c}, {"jit_avx512"}, "jit_avx512"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc, x, x, x}, {nhwc}, {"jit_avx512"}, "jit_avx512"});
        if (forcePlanarForAvx512) {
            resCPUParams.push_back(CPUSpecificParams{{nchw, x, x, x}, {nchw}, {"jit_avx512"}, "jit_avx512"});
        }
    } else if (InferenceEngine::with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{nChw8c, x, x, x}, {nChw8c}, {"jit_avx2"}, "jit_avx2"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc, x, x, x}, {nhwc}, {"jit_avx2"}, "jit_avx2"});
        resCPUParams.push_back(CPUSpecificParams{{nchw, x, x, x}, {nchw}, {"jit_avx2"}, "jit_avx2"});
    } else if (InferenceEngine::with_cpu_x86_sse42()) {
        resCPUParams.push_back(CPUSpecificParams{{nChw8c, x, x, x}, {nChw8c}, {"jit_sse42"}, "jit_sse42"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc, x, x, x}, {nhwc}, {"jit_sse42"}, "jit_sse42"});
        if (forcePlanarForSse) {
            resCPUParams.push_back(CPUSpecificParams{{nchw, x, x, x}, {nchw}, {"jit_sse42"}, "jit_sse42"});
        }
    } else {
        resCPUParams.push_back(CPUSpecificParams{{nchw, x, x, x}, {nchw}, {"ref"}, "ref"});
    }
    return resCPUParams;
}
/* ========== */
const std::vector<ngraph::op::v4::Interpolate::CoordinateTransformMode> coordinateTransformModes_Smoke = {
        ngraph::op::v4::Interpolate::CoordinateTransformMode::HALF_PIXEL,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::ASYMMETRIC,
};

const std::vector<ngraph::op::v4::Interpolate::CoordinateTransformMode> coordinateTransformModes_Full = {
        ngraph::op::v4::Interpolate::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::HALF_PIXEL,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::ASYMMETRIC,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::ALIGN_CORNERS,
};

const std::vector<ngraph::op::v4::Interpolate::NearestMode> nearestModes_Smoke = {
        ngraph::op::v4::Interpolate::NearestMode::SIMPLE,
        ngraph::op::v4::Interpolate::NearestMode::ROUND_PREFER_FLOOR,
        ngraph::op::v4::Interpolate::NearestMode::FLOOR,
};

const std::vector<ngraph::op::v4::Interpolate::NearestMode> nearestModes_Full = {
        ngraph::op::v4::Interpolate::NearestMode::SIMPLE,
        ngraph::op::v4::Interpolate::NearestMode::ROUND_PREFER_FLOOR,
        ngraph::op::v4::Interpolate::NearestMode::FLOOR,
        ngraph::op::v4::Interpolate::NearestMode::CEIL,
        ngraph::op::v4::Interpolate::NearestMode::ROUND_PREFER_CEIL,
};

const std::vector<ngraph::op::v4::Interpolate::NearestMode> defNearestModes = {
        ngraph::op::v4::Interpolate::NearestMode::ROUND_PREFER_FLOOR,
};

const std::vector<bool> antialias = {
        false,
};

const std::vector<double> cubeCoefs = {
        -0.75f,
};

const std::vector<fusingSpecificParams> interpolateFusingParamsSet{
        emptyFusingSpec,
        fusingSwish,
        fusingFakeQuantizePerTensorRelu,
};

std::vector<std::map<std::string, std::string>> filterAdditionalConfig(bool enforceBF16 = true) {
    if (InferenceEngine::with_cpu_x86_avx512f()) {
        if (enforceBF16) {
            return {
                {{InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16, InferenceEngine::PluginConfigParams::NO}},
                {{InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16, InferenceEngine::PluginConfigParams::YES}}
            };
        }
        return {
            {{InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16, InferenceEngine::PluginConfigParams::NO}}
        };

    } else {
        return {
            // default config as an stub for target without avx512, otherwise all tests with BF16 in its name are skipped
            {{InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::NO}}
        };
    }
}

const std::vector<std::vector<size_t>> pads4D = {
        {0, 0, 0, 0},
        {0, 0, 1, 1},
};

const std::vector<std::vector<int64_t>> defaultAxes4D = {
    {0, 1, 2, 3}
};

const std::vector<ShapeParams> shapeParams4D_Smoke = {
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{}, {{1, 11, 4, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1.f, 1.f, 1.25f, 1.5f}},
        defaultAxes4D.front()
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{}, {{1, 11, 4, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1, 11, 5, 6}},
        defaultAxes4D.front()
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 11, 4, 4}, {2, 7, 6, 5}, {1, 11, 4, 4}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        {{1.f, 1.f, 1.25f, 1.5f}, {1.f, 1.f, 1.25f, 1.25f}, {1.f, 1.f, 1.25f, 1.5f}},
        defaultAxes4D.front()
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 11, 4, 4}, {2, 7, 6, 5}, {1, 11, 4, 4}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        {{1, 11, 6, 7}, {2, 7, 8, 7}, {1, 11, 6, 7}},
        defaultAxes4D.front()
    }
};

const std::vector<ShapeParams> shapeParams4D_Full = {
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 11, 4, 4}, {2, 7, 6, 5}, {1, 11, 4, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1.f, 1.f, 1.25f, 1.5f}},
        defaultAxes4D.front()
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 11, 4, 4}, {1, 11, 5, 5}, {1, 11, 4, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1, 11, 5, 6}},
        defaultAxes4D.front()
    }
};

const auto interpolateCasesNN_Smoke = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::nearest),
        ::testing::ValuesIn(coordinateTransformModes_Smoke),
        ::testing::ValuesIn(nearestModes_Smoke),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs));

const auto interpolateCasesNN_Full = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::nearest),
        ::testing::ValuesIn(coordinateTransformModes_Full),
        ::testing::ValuesIn(nearestModes_Full),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateNN_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
             interpolateCasesNN_Smoke,
            ::testing::ValuesIn(shapeParams4D_Smoke),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateNN_Layout_Test, InterpolateLayerCPUTest,
         ::testing::Combine(
             interpolateCasesNN_Full,
             ::testing::ValuesIn(shapeParams4D_Full),
             ::testing::Values(ElementType::f32),
             ::testing::ValuesIn(filterCPUInfoForDevice()),
             ::testing::ValuesIn(interpolateFusingParamsSet),
             ::testing::ValuesIn(filterAdditionalConfig())),
     InterpolateLayerCPUTest::getTestCaseName);

const std::vector<ShapeParams> shapeParams4D_fixed_C = {
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{}, {{1, 11, 4, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1.f, 1.f, 1.25f, 1.5f}},
        defaultAxes4D.front()
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, 16, -1, -1}, {{1, 16, 4, 4}, {1, 16, 6, 5}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1, 16, 6, 7}},
        defaultAxes4D.front()
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateNN_Layout_PerChannelFuse_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesNN_Smoke,
            ::testing::ValuesIn(shapeParams4D_fixed_C),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::Values(fusingFakeQuantizePerChannelRelu),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateNN_Layout_PerChannelFuse_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesNN_Full,
            ::testing::ValuesIn(shapeParams4D_fixed_C),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::Values(fusingFakeQuantizePerChannelRelu),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

const auto interpolateCasesLinearOnnx_Smoke = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::linear_onnx),
        ::testing::ValuesIn(coordinateTransformModes_Smoke),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs));

const auto interpolateCasesLinearOnnx_Full = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::linear_onnx),
        ::testing::ValuesIn(coordinateTransformModes_Full),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateLinearOnnx_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesLinearOnnx_Smoke,
            ::testing::ValuesIn(shapeParams4D_Smoke),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateLinearOnnx_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesLinearOnnx_Full,
            ::testing::ValuesIn(shapeParams4D_Full),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

const auto interpolateCasesLinear_Smoke = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::linear),
        ::testing::ValuesIn(coordinateTransformModes_Smoke),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs));

const auto interpolateCasesLinear_Full = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::linear),
        ::testing::ValuesIn(coordinateTransformModes_Full),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateLinear_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesLinear_Smoke,
            ::testing::ValuesIn(shapeParams4D_Smoke),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateLinear_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesLinear_Full,
            ::testing::ValuesIn(shapeParams4D_Full),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

const auto interpolateCasesCubic_Smoke = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::cubic),
        ::testing::ValuesIn(coordinateTransformModes_Smoke),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs));

const auto interpolateCasesCubic_Full = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::cubic),
        ::testing::ValuesIn(coordinateTransformModes_Full),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateCubic_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesCubic_Smoke,
            ::testing::ValuesIn(shapeParams4D_Smoke),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateCubic_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesCubic_Full,
            ::testing::ValuesIn(shapeParams4D_Full),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);


const std::vector<ShapeParams> shapeParams4D_NearestExt_Smoke = {
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{1, 1, 6, 10}, {{1, 1, 6, 10}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1, 1, 24, 40}},
        defaultAxes4D.front()
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{1, 10, 5, 5}, {{1, 10, 5, 5}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        {{1, 10, 20, 20}},
        defaultAxes4D.front()
    }
};

const std::vector<ShapeParams> shapeParams4D_NearestExt_Full = {
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{1, 1, 20, 50}, {{1, 1, 20, 50}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1, 1, 20, 250}},
        defaultAxes4D.front()
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{1, 21, 23, 40}, {{1, 21, 23, 40}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        {{1, 21, 46, 80}},
        defaultAxes4D.front()
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{1, 16, 36, 64}, {{1, 16, 36, 64}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1, 16, 18, 32}},
        defaultAxes4D.front()
    }
};

const auto interpolateCasesNearestExt_Smoke = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::nearest),
        ::testing::ValuesIn(coordinateTransformModes_Smoke),
        ::testing::ValuesIn(nearestModes_Smoke),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs));

const auto interpolateCasesNearestExt_Full = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::nearest),
        ::testing::ValuesIn(coordinateTransformModes_Full),
        ::testing::ValuesIn(nearestModes_Full),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateNearestExt_FP32_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesNearestExt_Smoke,
            ::testing::ValuesIn(shapeParams4D_NearestExt_Smoke),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice(true, false)),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig(false))),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateNearestExt_FP32_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesNearestExt_Full,
            ::testing::ValuesIn(shapeParams4D_NearestExt_Full),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice(true, false)),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig(false))),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateNearestExt_I8_U8_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesNearestExt_Smoke,
            ::testing::ValuesIn(shapeParams4D_NearestExt_Smoke),
            ::testing::Values(ElementType::i8, ElementType::u8),
            ::testing::ValuesIn(filterCPUInfoForDevice(true, true)),
            ::testing::Values(emptyFusingSpec),
            ::testing::ValuesIn(filterAdditionalConfig(false))),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateNearestExt_I8_U8_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesNearestExt_Full,
            ::testing::ValuesIn(shapeParams4D_NearestExt_Full),
            ::testing::Values(ElementType::i8, ElementType::u8),
            ::testing::ValuesIn(filterCPUInfoForDevice(true, true)),
            ::testing::Values(emptyFusingSpec),
            ::testing::ValuesIn(filterAdditionalConfig(false))),
    InterpolateLayerCPUTest::getTestCaseName);

////////////////////////5D/////////////////////////////
std::vector<CPUSpecificParams> filterCPUInfoForDevice5D(bool forcePlanarForAvx512 = false, bool forcePlanarForSse = false) {
    std::vector<CPUSpecificParams> resCPUParams;
    if (InferenceEngine::with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{nCdhw16c, x, x, x}, {nCdhw16c}, {"jit_avx512"}, "jit_avx512"});
        resCPUParams.push_back(CPUSpecificParams{{ndhwc, x, x, x}, {ndhwc}, {"jit_avx512"}, "jit_avx512"});
        if (forcePlanarForAvx512) {
            resCPUParams.push_back(CPUSpecificParams{{ncdhw, x, x, x}, {ncdhw}, {"jit_avx512"}, "jit_avx512"});
        }
    } else if (InferenceEngine::with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{nCdhw8c, x, x, x}, {nCdhw8c}, {"jit_avx2"}, "jit_avx2"});
        resCPUParams.push_back(CPUSpecificParams{{ndhwc, x, x, x}, {ndhwc}, {"jit_avx2"}, "jit_avx2"});
        resCPUParams.push_back(CPUSpecificParams{{ncdhw, x, x, x}, {ncdhw}, {"jit_avx2"}, "jit_avx2"});
    } else if (InferenceEngine::with_cpu_x86_sse42()) {
        resCPUParams.push_back(CPUSpecificParams{{nCdhw8c, x, x, x}, {nCdhw8c}, {"jit_sse42"}, "jit_sse42"});
        resCPUParams.push_back(CPUSpecificParams{{ndhwc, x, x, x}, {ndhwc}, {"jit_sse42"}, "jit_sse42"});
        if (forcePlanarForSse) {
            resCPUParams.push_back(CPUSpecificParams{{ncdhw, x, x, x}, {ncdhw}, {"jit_sse42"}, "jit_sse42"});
        }
    } else {
        resCPUParams.push_back(CPUSpecificParams{{ncdhw, x, x, x}, {ncdhw}, {"ref"}, "ref"});
    }
    return resCPUParams;
}

const std::vector<std::vector<size_t>> pads5D = {
        {0, 0, 0, 0, 0}
};

const std::vector<std::vector<int64_t>> defaultAxes5D = {
    {0, 1, 2, 3, 4}
};

const std::vector<ShapeParams> shapeParams5D_Smoke = {
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{}, {{1, 11, 4, 4, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1.f, 1.f, 1.25f, 1.5f, 0.5f}},
        defaultAxes5D.front()
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{}, {{1, 11, 4, 4, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1, 11, 5, 6, 2}},
        defaultAxes5D.front()
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, {2, 20}, -1, -1, -1}, {{1, 11, 4, 4, 4}, {2, 7, 6, 5, 8}, {1, 11, 4, 4, 4}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        {{1.f, 1.f, 1.25f, 1.5f, 0.5f}, {1.f, 1.f, 1.25f, 1.25f, 1.25f}, {1.f, 1.f, 1.25f, 1.5f, 0.5f}},
        defaultAxes5D.front()
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 20}, -1, -1, -1}, {{1, 11, 4, 4, 4}, {2, 7, 6, 5, 8}, {1, 11, 4, 4, 4}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        {{1, 11, 6, 7, 2}, {2, 7, 8, 7, 4}, {1, 11, 6, 7, 2}},
        defaultAxes5D.front()
    },
};

const std::vector<ShapeParams> shapeParams5D_Full = {
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, {2, 20}, -1, -1, -1}, {{1, 11, 4, 4, 4}, {2, 7, 6, 5, 8}, {1, 11, 4, 4, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1.f, 1.f, 1.25f, 1.5f, 0.5f}},
        defaultAxes5D.front()
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 20}, -1, -1, -1}, {{1, 11, 4, 4, 4}, {1, 11, 5, 5, 8}, {1, 11, 4, 4, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1, 11, 5, 6, 4}},
        defaultAxes5D.front()
    }
};

const auto interpolateCasesLinearOnnx5D_Smoke = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::linear_onnx),
        ::testing::ValuesIn(coordinateTransformModes_Smoke),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(cubeCoefs));
const auto interpolateCasesLinearOnnx5D_Full = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::linear_onnx),
        ::testing::ValuesIn(coordinateTransformModes_Full),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(cubeCoefs));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateLinearOnnx5D_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesLinearOnnx5D_Smoke,
            ::testing::ValuesIn(shapeParams5D_Smoke),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice5D()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateLinearOnnx5D_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesLinearOnnx5D_Full,
            ::testing::ValuesIn(shapeParams5D_Full),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice5D()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

const auto interpolateCasesNN5D_Smoke = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::nearest),
        ::testing::ValuesIn(coordinateTransformModes_Smoke),
        ::testing::ValuesIn(nearestModes_Smoke),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(cubeCoefs));

const auto interpolateCasesNN5D_Full = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::nearest),
        ::testing::ValuesIn(coordinateTransformModes_Full),
        ::testing::ValuesIn(nearestModes_Full),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(cubeCoefs));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateNN5D_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesNN5D_Smoke,
            ::testing::ValuesIn(shapeParams5D_Smoke),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice5D()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateNN5D_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesNN5D_Full,
            ::testing::ValuesIn(shapeParams5D_Full),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice5D()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);


const std::vector<std::vector<size_t>> pads5D_NearestExt = {
        {0, 0, 0, 0, 0},
        {0, 0, 1, 1, 1},
};

const std::vector<ShapeParams> shapeParams5D_NearestExt_Smoke = {
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{1, 1, 6, 6, 10}, {{1, 1, 6, 6, 10}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1, 1, 24, 24, 40}},
        defaultAxes5D.front()
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{1, 10, 3, 5, 5}, {{1, 10, 3, 5, 5}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        {{1, 10, 12, 20, 20}},
        defaultAxes5D.front()
    }
};

const std::vector<ShapeParams> shapeParams5D_NearestExt_Full = {
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{1, 7, 2, 23, 40}, {{1, 7, 2, 23, 40}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        {{1, 7, 4, 46, 80}},
        defaultAxes5D.front()
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{1, 16, 9, 9, 16}, {{1, 16, 9, 9, 16}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1, 16, 3, 3, 4}},
        defaultAxes5D.front()
    }
};

const auto interpolateCasesNearestExt5D_Smoke = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::nearest),
        ::testing::ValuesIn(coordinateTransformModes_Smoke),
        ::testing::ValuesIn(nearestModes_Smoke),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads5D_NearestExt),
        ::testing::ValuesIn(pads5D_NearestExt),
        ::testing::ValuesIn(cubeCoefs));

const auto interpolateCasesNearestExt5D_Full = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::nearest),
        ::testing::ValuesIn(coordinateTransformModes_Full),
        ::testing::ValuesIn(nearestModes_Full),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads5D_NearestExt),
        ::testing::ValuesIn(pads5D_NearestExt),
        ::testing::ValuesIn(cubeCoefs));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateNearestExt_5D_FP32_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesNearestExt5D_Smoke,
            ::testing::ValuesIn(shapeParams5D_NearestExt_Smoke),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice5D(true, false)),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig(false))),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateNearestExt_5D_FP32_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesNearestExt5D_Full,
            ::testing::ValuesIn(shapeParams5D_NearestExt_Full),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice5D(true, false)),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig(false))),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateNearestExt_5D_I8_U8_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesNearestExt5D_Smoke,
            ::testing::ValuesIn(shapeParams5D_NearestExt_Smoke),
            ::testing::Values(ElementType::i8, ElementType::u8),
            ::testing::ValuesIn(filterCPUInfoForDevice5D(true, true)),
            ::testing::Values(emptyFusingSpec),
            ::testing::ValuesIn(filterAdditionalConfig(false))),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateNearestExt_5D_I8_U8_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesNearestExt5D_Full,
            ::testing::ValuesIn(shapeParams5D_NearestExt_Full),
            ::testing::Values(ElementType::i8, ElementType::u8),
            ::testing::ValuesIn(filterCPUInfoForDevice5D(true, true)),
            ::testing::Values(emptyFusingSpec),
            ::testing::ValuesIn(filterAdditionalConfig(false))),
    InterpolateLayerCPUTest::getTestCaseName);

// corner cases
const std::vector<ShapeParams> shapeParams4D_corner = {
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{1, 11, 4, 4}, {{1, 11, 4, 4}, {1, 11, 4, 4}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        {{1.f, 1.f, 1.25f, 1.5f}, {1.f, 1.f, 1.25f, 1.25f}},
        defaultAxes4D.front()
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{1, 11, 4, 4}, {{1, 11, 4, 4}, {1, 11, 4, 4}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        {{1, 11, 6, 7}, {1, 11, 8, 7}},
        defaultAxes4D.front()
    }
};

const auto interpolateCornerCases = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::nearest),
        ::testing::Values(ngraph::op::v4::Interpolate::CoordinateTransformMode::ASYMMETRIC),
        ::testing::Values(ngraph::op::v4::Interpolate::NearestMode::SIMPLE),
        ::testing::ValuesIn(antialias),
        ::testing::Values(std::vector<size_t>{0, 0, 0, 0}),
        ::testing::Values(std::vector<size_t>{0, 0, 0, 0}),
        ::testing::ValuesIn(cubeCoefs));

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_corner_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCornerCases,
            ::testing::ValuesIn(shapeParams4D_corner),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

} // namespace

////////////////////////Benchmark/////////////////////////////
namespace benchmark {

struct InterpolateLayerCPUBenchmarkTest : BenchmarkLayerTest<InterpolateLayerCPUTest> {};

TEST_P(InterpolateLayerCPUBenchmarkTest, benchmark) {
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
        // ngraph::op::v4::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL,
        // ngraph::op::v4::Interpolate::CoordinateTransformMode::HALF_PIXEL,
        // ngraph::op::v4::Interpolate::CoordinateTransformMode::ASYMMETRIC,

        // ngraph::op::v4::Interpolate::CoordinateTransformMode::ALIGN_CORNERS,
};

const std::vector<ngraph::op::v4::Interpolate::NearestMode> nearestModesBenchmark = {
        ngraph::op::v4::Interpolate::NearestMode::SIMPLE,
        // ngraph::op::v4::Interpolate::NearestMode::ROUND_PREFER_FLOOR,
        // ngraph::op::v4::Interpolate::NearestMode::FLOOR,
        // ngraph::op::v4::Interpolate::NearestMode::CEIL,
        // ngraph::op::v4::Interpolate::NearestMode::ROUND_PREFER_CEIL,
};

const std::vector<bool> antialiasBenchmark = {
        false,
};

const std::vector<std::vector<size_t>> padsBeginBenchmark = {
        {0, 0, 0, 0},
        {0, 0, 1, 1},
};

const std::vector<std::vector<size_t>> padsEndBenchmark = {
        {0, 0, 0, 0}
};
const std::vector<double> cubeCoefsBenchmark = {
        -0.75f,
};

const std::vector<std::vector<int64_t>> defaultAxesBenchmark = {
    {0, 1, 2, 3}
};

const std::vector<fusingSpecificParams> interpolateFusingParamsBenchmark{
        emptyFusingSpec,
        // fusingSwish,
        // fusingFakeQuantizePerTensorRelu,
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-declarations"

std::vector<std::map<std::string, std::string>> filterAdditionalConfigBenchmark() {
    // if (InferenceEngine::with_cpu_x86_avx512f()) {
    //     return {
    //         {{InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16, InferenceEngine::PluginConfigParams::NO}},
    //         {{InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16, InferenceEngine::PluginConfigParams::YES}}
    //     };
    // } else {
        return {
            // default config as an stub for target without avx512, otherwise all tests with BF16 in its name are skipped
            {{InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES}}
        };
    // }
}

std::vector<CPUSpecificParams> filterCPUInfoForDeviceTestI8() {
    std::vector<CPUSpecificParams> resCPUParams;

    resCPUParams.push_back(CPUSpecificParams{{nChw16c, x, x, x}, {nChw16c}, {"jit_avx512"}, "jit_avx512"});
    resCPUParams.push_back(CPUSpecificParams{{nhwc, x, x, x}, {nhwc}, {"jit_avx512"}, "jit_avx512"});
    resCPUParams.push_back(CPUSpecificParams{{nchw, x, x, x}, {nchw}, {"jit_avx512"}, "jit_avx512"});

    // resCPUParams.push_back(CPUSpecificParams{{nChw8c, x, x, x}, {nChw8c}, {"jit_avx2"}, "jit_avx2"});
    // resCPUParams.push_back(CPUSpecificParams{{nhwc, x, x, x}, {nhwc}, {"jit_avx2"}, "jit_avx2"});
    // resCPUParams.push_back(CPUSpecificParams{{nchw, x, x, x}, {nchw}, {"jit_avx2"}, "jit_avx2"});

    // resCPUParams.push_back(CPUSpecificParams{{nChw8c, x, x, x}, {nChw8c}, {"jit_sse42"}, "jit_sse42"});
    // resCPUParams.push_back(CPUSpecificParams{{nhwc, x, x, x}, {nhwc}, {"jit_sse42"}, "jit_sse42"});
    // resCPUParams.push_back(CPUSpecificParams{{nchw, x, x, x}, {nchw}, {"jit_sse42"}, "jit_sse42"});
    return resCPUParams;
}

#pragma GCC diagnostic pop

const std::vector<ElementType> prcBenchmark = {ElementType::f32, ElementType::i8};
// const std::vector<ElementType> prcBenchmark = {ElementType::f32, ElementType::i8, ElementType::u8};
// const std::vector<ElementType> prcBenchmark = {ElementType::f32};
// const std::vector<ElementType> prcBenchmark = {ElementType::i8, ElementType::u8};
// const std::vector<ElementType> prcBenchmark = {ElementType::i8};
// const std::vector<ElementType> prcBenchmark = {ElementType::u8};

const auto interpolateCasesBenchmark = ::testing::Combine(
        ::testing::ValuesIn(interpolateModesBenchmark),       // InterpolateMode
        ::testing::ValuesIn(coordinateTransformModesBenchmark),    // CoordinateTransformMode
        ::testing::ValuesIn(nearestModesBenchmark),    // NearestMode
        ::testing::ValuesIn(antialiasBenchmark),       // AntiAlias
        ::testing::ValuesIn(padsBeginBenchmark),        // PadBegin
        ::testing::ValuesIn(padsEndBenchmark),        // PadEnd
        ::testing::ValuesIn(cubeCoefsBenchmark));        // Cube coef


const std::vector<ShapeParams> shapeParamsBenchmark_00 = {
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 1, 80, 9150}, {{1, 1, 80, 9150}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 1, 80, 100650}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 2, 80, 9150}, {{1, 2, 80, 9150}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 2, 80, 100650}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 3, 80, 9150}, {{1, 3, 80, 9150}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 3, 80, 100650}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 4, 80, 9150}, {{1, 4, 80, 9150}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 4, 80, 100650}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 5, 80, 9150}, {{1, 5, 80, 9150}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 5, 80, 100650}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 8, 80, 9150}, {{1, 8, 80, 9150}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 8, 80, 100650}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 10, 80, 9150}, {{1, 10, 80, 9150}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 10, 80, 100650}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 16, 80, 9150}, {{1, 16, 80, 9150}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 16, 80, 100650}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 21, 80, 9150}, {{1, 21, 80, 9150}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 21, 80, 100650}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 32, 80, 9150}, {{1, 32, 80, 9150}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 32, 80, 100650}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 40, 80, 9150}, {{1, 40, 80, 9150}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 40, 80, 100650}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    // ShapeParams{
    //     ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
    //     InputShape{{1, 64, 80, 9150}, {{1, 64, 80, 9150}}},              // Input shapes
    //     ngraph::helpers::InputLayerType::PARAMETER,                    // input type
    //     {{1, 64, 80, 100650}},                                            // scales or sizes values
    //     defaultAxesBenchmark.front()                                   // axes
    // },
    // ShapeParams{
    //     ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
    //     InputShape{{1, 111, 80, 9150}, {{1, 111, 80, 9150}}},              // Input shapes
    //     ngraph::helpers::InputLayerType::PARAMETER,                    // input type
    //     {{1, 111, 80, 100650}},                                            // scales or sizes values
    //     defaultAxesBenchmark.front()                                   // axes
    // },
    // ShapeParams{
    //     ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
    //     InputShape{{1, 128, 80, 9150}, {{1, 128, 80, 9150}}},              // Input shapes
    //     ngraph::helpers::InputLayerType::PARAMETER,                    // input type
    //     {{1, 128, 80, 100650}},                                            // scales or sizes values
    //     defaultAxesBenchmark.front()                                   // axes
    // },
    // ShapeParams{
    //     ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
    //     InputShape{{1, 205, 80, 9150}, {{1, 205, 80, 9150}}},              // Input shapes
    //     ngraph::helpers::InputLayerType::PARAMETER,                    // input type
    //     {{1, 205, 80, 100650}},                                            // scales or sizes values
    //     defaultAxesBenchmark.front()                                   // axes
    // },
    // ShapeParams{
    //     ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
    //     InputShape{{1, 256, 80, 9150}, {{1, 256, 80, 9150}}},              // Input shapes
    //     ngraph::helpers::InputLayerType::PARAMETER,                    // input type
    //     {{1, 256, 80, 100650}},                                            // scales or sizes values
    //     defaultAxesBenchmark.front()                                   // axes
    // },
    // ShapeParams{
    //     ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
    //     InputShape{{1, 512, 80, 9150}, {{1, 512, 80, 9150}}},              // Input shapes
    //     ngraph::helpers::InputLayerType::PARAMETER,                    // input type
    //     {{1, 512, 80, 100650}},                                            // scales or sizes values
    //     defaultAxesBenchmark.front()                                   // axes
    // },
};

INSTANTIATE_TEST_SUITE_P(Interpolate_I8_CPU_Test_00, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesBenchmark,                                     // InterpolateSpecificParams
            ::testing::ValuesIn(shapeParamsBenchmark_00),               // ShapeParams
            ::testing::ValuesIn(prcBenchmark),     // ElementType
            ::testing::ValuesIn(filterCPUInfoForDeviceTestI8()),              // CPUSpecificParams
            ::testing::ValuesIn(interpolateFusingParamsBenchmark),      // fusingSpecificParams
            ::testing::ValuesIn(filterAdditionalConfigBenchmark())),             // AdditionalConfig
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Interpolate_Benchmark_CPU_Test_00, InterpolateLayerCPUBenchmarkTest,
        ::testing::Combine(
            interpolateCasesBenchmark,                                     // InterpolateSpecificParams
            ::testing::ValuesIn(shapeParamsBenchmark_00),               // ShapeParams
            ::testing::ValuesIn(prcBenchmark),     // ElementType
            ::testing::ValuesIn(filterCPUInfoForDeviceTestI8()),              // CPUSpecificParams
            ::testing::ValuesIn(interpolateFusingParamsBenchmark),      // fusingSpecificParams
            ::testing::ValuesIn(filterAdditionalConfigBenchmark())),             // AdditionalConfig
    InterpolateLayerCPUBenchmarkTest::getTestCaseName);


// const std::vector<ShapeParams> shapeParamsBenchmark_01 = {
//     ShapeParams{
//         ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
//         InputShape{{1, 1, 128, 99550}, {{1, 1, 128, 99550}}},              // Input shapes
//         ngraph::helpers::InputLayerType::PARAMETER,                    // input type
//         {{1, 1, 128, 497750}},                                            // scales or sizes values
//         defaultAxesBenchmark.front()                                   // axes
//     }
// };

// INSTANTIATE_TEST_SUITE_P(Interpolate_I8_CPU_Test_01, InterpolateLayerCPUTest,
//         ::testing::Combine(
//             interpolateCasesBenchmark,                                     // InterpolateSpecificParams
//             ::testing::ValuesIn(shapeParamsBenchmark_01),               // ShapeParams
//             ::testing::ValuesIn(prcBenchmark),     // ElementType
//             ::testing::ValuesIn(filterCPUInfoForDeviceTestI8()),              // CPUSpecificParams
//             ::testing::ValuesIn(interpolateFusingParamsBenchmark),      // fusingSpecificParams
//             ::testing::ValuesIn(filterAdditionalConfigBenchmark())),             // AdditionalConfig
//     InterpolateLayerCPUTest::getTestCaseName);

// INSTANTIATE_TEST_SUITE_P(Interpolate_Benchmark_CPU_Test_01, InterpolateLayerCPUBenchmarkTest,
//         ::testing::Combine(
//             interpolateCasesBenchmark,                                     // InterpolateSpecificParams
//             ::testing::ValuesIn(shapeParamsBenchmark_01),               // ShapeParams
//             ::testing::ValuesIn(prcBenchmark),     // ElementType
//             ::testing::ValuesIn(filterCPUInfoForDeviceTestI8()),              // CPUSpecificParams
//             ::testing::ValuesIn(interpolateFusingParamsBenchmark),      // fusingSpecificParams
//             ::testing::ValuesIn(filterAdditionalConfigBenchmark())),             // AdditionalConfig
//     InterpolateLayerCPUBenchmarkTest::getTestCaseName);


const std::vector<ShapeParams> shapeParamsBenchmark_02 = {
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 1, 128, 196}, {{1, 1, 128, 196}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 1, 128, 53900}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 2, 128, 196}, {{1, 2, 128, 196}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 2, 128, 53900}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 3, 128, 196}, {{1, 3, 128, 196}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 3, 128, 53900}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 4, 128, 196}, {{1, 4, 128, 196}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 4, 128, 53900}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 5, 128, 196}, {{1, 5, 128, 196}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 5, 128, 53900}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 8, 128, 196}, {{1, 8, 128, 196}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 8, 128, 53900}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 10, 128, 196}, {{1, 10, 128, 196}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 10, 128, 53900}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 16, 128, 196}, {{1, 16, 128, 196}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 16, 128, 53900}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 21, 128, 196}, {{1, 21, 128, 196}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 21, 128, 53900}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 32, 128, 196}, {{1, 32, 128, 196}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 32, 128, 53900}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 40, 128, 196}, {{1, 40, 128, 196}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 40, 128, 53900}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 64, 128, 196}, {{1, 64, 128, 196}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 64, 128, 53900}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    // ShapeParams{
    //     ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
    //     InputShape{{1, 128, 128, 196}, {{1, 128, 128, 196}}},              // Input shapes
    //     ngraph::helpers::InputLayerType::PARAMETER,                    // input type
    //     {{1, 128, 128, 53900}},                                            // scales or sizes values
    //     defaultAxesBenchmark.front()                                   // axes
    // },
    // ShapeParams{
    //     ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
    //     InputShape{{1, 205, 128, 196}, {{1, 205, 128, 196}}},              // Input shapes
    //     ngraph::helpers::InputLayerType::PARAMETER,                    // input type
    //     {{1, 205, 128, 53900}},                                            // scales or sizes values
    //     defaultAxesBenchmark.front()                                   // axes
    // },
    // ShapeParams{
    //     ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
    //     InputShape{{1, 256, 128, 196}, {{1, 256, 128, 196}}},              // Input shapes
    //     ngraph::helpers::InputLayerType::PARAMETER,                    // input type
    //     {{1, 256, 128, 53900}},                                            // scales or sizes values
    //     defaultAxesBenchmark.front()                                   // axes
    // },
    // ShapeParams{
    //     ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
    //     InputShape{{1, 512, 128, 196}, {{1, 512, 128, 196}}},              // Input shapes
    //     ngraph::helpers::InputLayerType::PARAMETER,                    // input type
    //     {{1, 512, 128, 53900}},                                            // scales or sizes values
    //     defaultAxesBenchmark.front()                                   // axes
    // },
};

INSTANTIATE_TEST_SUITE_P(Interpolate_I8_CPU_Test_02, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesBenchmark,                                     // InterpolateSpecificParams
            ::testing::ValuesIn(shapeParamsBenchmark_02),               // ShapeParams
            ::testing::ValuesIn(prcBenchmark),     // ElementType
            ::testing::ValuesIn(filterCPUInfoForDeviceTestI8()),              // CPUSpecificParams
            ::testing::ValuesIn(interpolateFusingParamsBenchmark),      // fusingSpecificParams
            ::testing::ValuesIn(filterAdditionalConfigBenchmark())),             // AdditionalConfig
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Interpolate_Benchmark_CPU_Test_02, InterpolateLayerCPUBenchmarkTest,
        ::testing::Combine(
            interpolateCasesBenchmark,                                     // InterpolateSpecificParams
            ::testing::ValuesIn(shapeParamsBenchmark_02),               // ShapeParams
            ::testing::ValuesIn(prcBenchmark),     // ElementType
            ::testing::ValuesIn(filterCPUInfoForDeviceTestI8()),              // CPUSpecificParams
            ::testing::ValuesIn(interpolateFusingParamsBenchmark),      // fusingSpecificParams
            ::testing::ValuesIn(filterAdditionalConfigBenchmark())),             // AdditionalConfig
    InterpolateLayerCPUBenchmarkTest::getTestCaseName);


const std::vector<ShapeParams> shapeParamsBenchmark_03 = {
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 1, 80, 200}, {{1, 1, 80, 200}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 1, 80, 1000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 2, 80, 200}, {{1, 2, 80, 200}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 2, 80, 1000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 3, 80, 200}, {{1, 3, 80, 200}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 3, 80, 1000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 4, 80, 200}, {{1, 4, 80, 200}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 4, 80, 1000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 5, 80, 200}, {{1, 5, 80, 200}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 5, 80, 1000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 8, 80, 200}, {{1, 8, 80, 200}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 8, 80, 1000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 10, 80, 200}, {{1, 10, 80, 200}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 10, 80, 1000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 16, 80, 200}, {{1, 16, 80, 200}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 16, 80, 1000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 21, 80, 200}, {{1, 21, 80, 200}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 21, 80, 1000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 32, 80, 200}, {{1, 32, 80, 200}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 32, 80, 1000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 40, 80, 200}, {{1, 40, 80, 200}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 40, 80, 1000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 64, 80, 200}, {{1, 64, 80, 200}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 64, 80, 1000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 111, 80, 200}, {{1, 111, 80, 200}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 111, 80, 1000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 128, 80, 200}, {{1, 128, 80, 200}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 128, 80, 1000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 205, 80, 200}, {{1, 205, 80, 200}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 205, 80, 1000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 256, 80, 200}, {{1, 256, 80, 200}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 256, 80, 1000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 512, 80, 200}, {{1, 512, 80, 200}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 512, 80, 1000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
};

INSTANTIATE_TEST_SUITE_P(Interpolate_I8_CPU_Test_03, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesBenchmark,                                     // InterpolateSpecificParams
            ::testing::ValuesIn(shapeParamsBenchmark_03),               // ShapeParams
            ::testing::ValuesIn(prcBenchmark),     // ElementType
            ::testing::ValuesIn(filterCPUInfoForDeviceTestI8()),              // CPUSpecificParams
            ::testing::ValuesIn(interpolateFusingParamsBenchmark),      // fusingSpecificParams
            ::testing::ValuesIn(filterAdditionalConfigBenchmark())),             // AdditionalConfig
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Interpolate_Benchmark_CPU_Test_03, InterpolateLayerCPUBenchmarkTest,
        ::testing::Combine(
            interpolateCasesBenchmark,                                     // InterpolateSpecificParams
            ::testing::ValuesIn(shapeParamsBenchmark_03),               // ShapeParams
            ::testing::ValuesIn(prcBenchmark),     // ElementType
            ::testing::ValuesIn(filterCPUInfoForDeviceTestI8()),              // CPUSpecificParams
            ::testing::ValuesIn(interpolateFusingParamsBenchmark),      // fusingSpecificParams
            ::testing::ValuesIn(filterAdditionalConfigBenchmark())),             // AdditionalConfig
    InterpolateLayerCPUBenchmarkTest::getTestCaseName);


const std::vector<ShapeParams> shapeParamsBenchmark_04 = {
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 1, 80, 1000}, {{1, 1, 80, 1000}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 1, 80, 5000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 2, 80, 1000}, {{1, 2, 80, 1000}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 2, 80, 5000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 3, 80, 1000}, {{1, 3, 80, 1000}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 3, 80, 5000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 4, 80, 1000}, {{1, 4, 80, 1000}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 4, 80, 5000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 5, 80, 1000}, {{1, 5, 80, 1000}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 5, 80, 5000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 8, 80, 1000}, {{1, 8, 80, 1000}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 8, 80, 5000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 10, 80, 1000}, {{1, 10, 80, 1000}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 10, 80, 5000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 16, 80, 1000}, {{1, 16, 80, 1000}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 16, 80, 5000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 21, 80, 1000}, {{1, 21, 80, 1000}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 21, 80, 5000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 32, 80, 1000}, {{1, 32, 80, 1000}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 32, 80, 5000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 40, 80, 1000}, {{1, 40, 80, 1000}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 40, 80, 5000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 64, 80, 1000}, {{1, 64, 80, 1000}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 64, 80, 5000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 111, 80, 1000}, {{1, 111, 80, 1000}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 111, 80, 5000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 128, 80, 1000}, {{1, 128, 80, 1000}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 128, 80, 5000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 205, 80, 1000}, {{1, 205, 80, 1000}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 205, 80, 5000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 256, 80, 1000}, {{1, 256, 80, 1000}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 256, 80, 5000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 512, 80, 1000}, {{1, 512, 80, 1000}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 512, 80, 5000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
};

INSTANTIATE_TEST_SUITE_P(Interpolate_I8_CPU_Test_04, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesBenchmark,                                     // InterpolateSpecificParams
            ::testing::ValuesIn(shapeParamsBenchmark_04),               // ShapeParams
            ::testing::ValuesIn(prcBenchmark),     // ElementType
            ::testing::ValuesIn(filterCPUInfoForDeviceTestI8()),              // CPUSpecificParams
            ::testing::ValuesIn(interpolateFusingParamsBenchmark),      // fusingSpecificParams
            ::testing::ValuesIn(filterAdditionalConfigBenchmark())),             // AdditionalConfig
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Interpolate_Benchmark_CPU_Test_04, InterpolateLayerCPUBenchmarkTest,
        ::testing::Combine(
            interpolateCasesBenchmark,                                     // InterpolateSpecificParams
            ::testing::ValuesIn(shapeParamsBenchmark_04),               // ShapeParams
            ::testing::ValuesIn(prcBenchmark),     // ElementType
            ::testing::ValuesIn(filterCPUInfoForDeviceTestI8()),              // CPUSpecificParams
            ::testing::ValuesIn(interpolateFusingParamsBenchmark),      // fusingSpecificParams
            ::testing::ValuesIn(filterAdditionalConfigBenchmark())),             // AdditionalConfig
    InterpolateLayerCPUBenchmarkTest::getTestCaseName);


const std::vector<ShapeParams> shapeParamsBenchmark_05 = {
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 1, 80, 5000}, {{1, 1, 80, 5000}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 1, 80, 55000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 2, 80, 5000}, {{1, 2, 80, 5000}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 2, 80, 55000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 3, 80, 5000}, {{1, 3, 80, 5000}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 3, 80, 55000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 4, 80, 5000}, {{1, 4, 80, 5000}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 4, 80, 55000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 5, 80, 5000}, {{1, 5, 80, 5000}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 5, 80, 55000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 8, 80, 5000}, {{1, 8, 80, 5000}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 8, 80, 55000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 10, 80, 5000}, {{1, 10, 80, 5000}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 10, 80, 55000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 16, 80, 5000}, {{1, 16, 80, 5000}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 16, 80, 55000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 21, 80, 5000}, {{1, 21, 80, 5000}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 21, 80, 55000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 32, 80, 5000}, {{1, 32, 80, 5000}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 32, 80, 55000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 40, 80, 5000}, {{1, 40, 80, 5000}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 40, 80, 55000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 64, 80, 5000}, {{1, 64, 80, 5000}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 64, 80, 55000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 111, 80, 5000}, {{1, 111, 80, 5000}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 111, 80, 55000}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    // ShapeParams{
    //     ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
    //     InputShape{{1, 128, 80, 5000}, {{1, 128, 80, 5000}}},              // Input shapes
    //     ngraph::helpers::InputLayerType::PARAMETER,                    // input type
    //     {{1, 128, 80, 55000}},                                            // scales or sizes values
    //     defaultAxesBenchmark.front()                                   // axes
    // },
    // ShapeParams{
    //     ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
    //     InputShape{{1, 205, 80, 5000}, {{1, 205, 80, 5000}}},              // Input shapes
    //     ngraph::helpers::InputLayerType::PARAMETER,                    // input type
    //     {{1, 205, 80, 55000}},                                            // scales or sizes values
    //     defaultAxesBenchmark.front()                                   // axes
    // },
    // ShapeParams{
    //     ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
    //     InputShape{{1, 256, 80, 5000}, {{1, 256, 80, 5000}}},              // Input shapes
    //     ngraph::helpers::InputLayerType::PARAMETER,                    // input type
    //     {{1, 256, 80, 55000}},                                            // scales or sizes values
    //     defaultAxesBenchmark.front()                                   // axes
    // },
    // ShapeParams{
    //     ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
    //     InputShape{{1, 512, 80, 5000}, {{1, 512, 80, 5000}}},              // Input shapes
    //     ngraph::helpers::InputLayerType::PARAMETER,                    // input type
    //     {{1, 512, 80, 55000}},                                            // scales or sizes values
    //     defaultAxesBenchmark.front()                                   // axes
    // },
};

INSTANTIATE_TEST_SUITE_P(Interpolate_I8_CPU_Test_05, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesBenchmark,                                     // InterpolateSpecificParams
            ::testing::ValuesIn(shapeParamsBenchmark_05),               // ShapeParams
            ::testing::ValuesIn(prcBenchmark),     // ElementType
            ::testing::ValuesIn(filterCPUInfoForDeviceTestI8()),              // CPUSpecificParams
            ::testing::ValuesIn(interpolateFusingParamsBenchmark),      // fusingSpecificParams
            ::testing::ValuesIn(filterAdditionalConfigBenchmark())),             // AdditionalConfig
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Interpolate_Benchmark_CPU_Test_05, InterpolateLayerCPUBenchmarkTest,
        ::testing::Combine(
            interpolateCasesBenchmark,                                     // InterpolateSpecificParams
            ::testing::ValuesIn(shapeParamsBenchmark_05),               // ShapeParams
            ::testing::ValuesIn(prcBenchmark),     // ElementType
            ::testing::ValuesIn(filterCPUInfoForDeviceTestI8()),              // CPUSpecificParams
            ::testing::ValuesIn(interpolateFusingParamsBenchmark),      // fusingSpecificParams
            ::testing::ValuesIn(filterAdditionalConfigBenchmark())),             // AdditionalConfig
    InterpolateLayerCPUBenchmarkTest::getTestCaseName);


const std::vector<ShapeParams> shapeParamsBenchmark_06 = {
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 1, 6, 10}, {{1, 1, 6, 10}}},                // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 1, 24, 40}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 2, 6, 10}, {{1, 2, 6, 10}}},                // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 2, 24, 40}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 3, 6, 10}, {{1, 3, 6, 10}}},                // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 3, 24, 40}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 4, 6, 10}, {{1, 4, 6, 10}}},                // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 4, 24, 40}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 5, 6, 10}, {{1, 5, 6, 10}}},                // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 5, 24, 40}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 8, 6, 10}, {{1, 8, 6, 10}}},                // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 8, 24, 40}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 10, 6, 10}, {{1, 10, 6, 10}}},                // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 10, 24, 40}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 16, 6, 10}, {{1, 16, 6, 10}}},                // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 16, 24, 40}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 21, 6, 10}, {{1, 21, 6, 10}}},                // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 21, 24, 40}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 32, 6, 10}, {{1, 32, 6, 10}}},                // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 32, 24, 40}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 40, 6, 10}, {{1, 40, 6, 10}}},                // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 40, 24, 40}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 64, 6, 10}, {{1, 64, 6, 10}}},                // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 64, 24, 40}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 111, 6, 10}, {{1, 111, 6, 10}}},                // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 111, 24, 40}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 128, 6, 10}, {{1, 128, 6, 10}}},                // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 128, 24, 40}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 205, 6, 10}, {{1, 205, 6, 10}}},                // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 205, 24, 40}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 256, 6, 10}, {{1, 256, 6, 10}}},                // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 256, 24, 40}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 512, 6, 10}, {{1, 512, 6, 10}}},                // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 512, 24, 40}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
};

INSTANTIATE_TEST_SUITE_P(Interpolate_I8_CPU_Test_06, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesBenchmark,                                     // InterpolateSpecificParams
            ::testing::ValuesIn(shapeParamsBenchmark_06),               // ShapeParams
            ::testing::ValuesIn(prcBenchmark),     // ElementType
            ::testing::ValuesIn(filterCPUInfoForDeviceTestI8()),              // CPUSpecificParams
            ::testing::ValuesIn(interpolateFusingParamsBenchmark),      // fusingSpecificParams
            ::testing::ValuesIn(filterAdditionalConfigBenchmark())),             // AdditionalConfig
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Interpolate_Benchmark_CPU_Test_06, InterpolateLayerCPUBenchmarkTest,
        ::testing::Combine(
            interpolateCasesBenchmark,                                     // InterpolateSpecificParams
            ::testing::ValuesIn(shapeParamsBenchmark_06),               // ShapeParams
            ::testing::ValuesIn(prcBenchmark),     // ElementType
            ::testing::ValuesIn(filterCPUInfoForDeviceTestI8()),              // CPUSpecificParams
            ::testing::ValuesIn(interpolateFusingParamsBenchmark),      // fusingSpecificParams
            ::testing::ValuesIn(filterAdditionalConfigBenchmark())),             // AdditionalConfig
    InterpolateLayerCPUBenchmarkTest::getTestCaseName);


const std::vector<ShapeParams> shapeParamsBenchmark_07 = {
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 1, 23, 40}, {{1, 1, 23, 40}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 1, 46, 80}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 2, 23, 40}, {{1, 2, 23, 40}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 2, 46, 80}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 3, 23, 40}, {{1, 3, 23, 40}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 3, 46, 80}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 4, 23, 40}, {{1, 4, 23, 40}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 4, 46, 80}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 5, 23, 40}, {{1, 5, 23, 40}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 5, 46, 80}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 8, 23, 40}, {{1, 8, 23, 40}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 8, 46, 80}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 10, 23, 40}, {{1, 10, 23, 40}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 10, 46, 80}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 16, 23, 40}, {{1, 16, 23, 40}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 16, 46, 80}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 21, 23, 40}, {{1, 21, 23, 40}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 21, 46, 80}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 32, 23, 40}, {{1, 32, 23, 40}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 32, 46, 80}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 40, 23, 40}, {{1, 40, 23, 40}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 40, 46, 80}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 64, 23, 40}, {{1, 64, 23, 40}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 64, 46, 80}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 111, 23, 40}, {{1, 111, 23, 40}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 111, 46, 80}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 128, 23, 40}, {{1, 128, 23, 40}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 128, 46, 80}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 205, 23, 40}, {{1, 205, 23, 40}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 205, 46, 80}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 256, 23, 40}, {{1, 256, 23, 40}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 256, 46, 80}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 512, 23, 40}, {{1, 512, 23, 40}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 512, 46, 80}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
};

INSTANTIATE_TEST_SUITE_P(Interpolate_I8_CPU_Test_07, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesBenchmark,                                     // InterpolateSpecificParams
            ::testing::ValuesIn(shapeParamsBenchmark_07),               // ShapeParams
            ::testing::ValuesIn(prcBenchmark),     // ElementType
            ::testing::ValuesIn(filterCPUInfoForDeviceTestI8()),              // CPUSpecificParams
            ::testing::ValuesIn(interpolateFusingParamsBenchmark),      // fusingSpecificParams
            ::testing::ValuesIn(filterAdditionalConfigBenchmark())),             // AdditionalConfig
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Interpolate_Benchmark_CPU_Test_07, InterpolateLayerCPUBenchmarkTest,
        ::testing::Combine(
            interpolateCasesBenchmark,                                     // InterpolateSpecificParams
            ::testing::ValuesIn(shapeParamsBenchmark_07),               // ShapeParams
            ::testing::ValuesIn(prcBenchmark),     // ElementType
            ::testing::ValuesIn(filterCPUInfoForDeviceTestI8()),              // CPUSpecificParams
            ::testing::ValuesIn(interpolateFusingParamsBenchmark),      // fusingSpecificParams
            ::testing::ValuesIn(filterAdditionalConfigBenchmark())),             // AdditionalConfig
    InterpolateLayerCPUBenchmarkTest::getTestCaseName);


const std::vector<ShapeParams> shapeParamsBenchmark_08 = {
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 1, 45, 80}, {{1, 1, 45, 80}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 1, 90, 160}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 2, 45, 80}, {{1, 2, 45, 80}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 2, 90, 160}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 3, 45, 80}, {{1, 3, 45, 80}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 3, 90, 160}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 4, 45, 80}, {{1, 4, 45, 80}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 4, 90, 160}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 5, 45, 80}, {{1, 5, 45, 80}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 5, 90, 160}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 8, 45, 80}, {{1, 8, 45, 80}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 8, 90, 160}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 10, 45, 80}, {{1, 10, 45, 80}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 10, 90, 160}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 16, 45, 80}, {{1, 16, 45, 80}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 16, 90, 160}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 21, 45, 80}, {{1, 21, 45, 80}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 21, 90, 160}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 32, 45, 80}, {{1, 32, 45, 80}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 32, 90, 160}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 40, 45, 80}, {{1, 40, 45, 80}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 40, 90, 160}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 64, 45, 80}, {{1, 64, 45, 80}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 64, 90, 160}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 111, 45, 80}, {{1, 111, 45, 80}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 111, 90, 160}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 128, 45, 80}, {{1, 128, 45, 80}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 128, 90, 160}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 205, 45, 80}, {{1, 205, 45, 80}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 205, 90, 160}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 256, 45, 80}, {{1, 256, 45, 80}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 256, 90, 160}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 512, 45, 80}, {{1, 512, 45, 80}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 512, 90, 160}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
};

INSTANTIATE_TEST_SUITE_P(Interpolate_I8_CPU_Test_08, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesBenchmark,                                     // InterpolateSpecificParams
            ::testing::ValuesIn(shapeParamsBenchmark_08),               // ShapeParams
            ::testing::ValuesIn(prcBenchmark),     // ElementType
            ::testing::ValuesIn(filterCPUInfoForDeviceTestI8()),              // CPUSpecificParams
            ::testing::ValuesIn(interpolateFusingParamsBenchmark),      // fusingSpecificParams
            ::testing::ValuesIn(filterAdditionalConfigBenchmark())),             // AdditionalConfig
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Interpolate_Benchmark_CPU_Test_08, InterpolateLayerCPUBenchmarkTest,
        ::testing::Combine(
            interpolateCasesBenchmark,                                     // InterpolateSpecificParams
            ::testing::ValuesIn(shapeParamsBenchmark_08),               // ShapeParams
            ::testing::ValuesIn(prcBenchmark),     // ElementType
            ::testing::ValuesIn(filterCPUInfoForDeviceTestI8()),              // CPUSpecificParams
            ::testing::ValuesIn(interpolateFusingParamsBenchmark),      // fusingSpecificParams
            ::testing::ValuesIn(filterAdditionalConfigBenchmark())),             // AdditionalConfig
    InterpolateLayerCPUBenchmarkTest::getTestCaseName);


const std::vector<ShapeParams> shapeParamsBenchmark_09 = {
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 1, 90, 160}, {{1, 1, 90, 160}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 1, 180, 320}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 2, 90, 160}, {{1, 2, 90, 160}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 2, 180, 320}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 3, 90, 160}, {{1, 3, 90, 160}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 3, 180, 320}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 4, 90, 160}, {{1, 4, 90, 160}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 4, 180, 320}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 5, 90, 160}, {{1, 5, 90, 160}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 5, 180, 320}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 8, 90, 160}, {{1, 8, 90, 160}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 8, 180, 320}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 10, 90, 160}, {{1, 10, 90, 160}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 10, 180, 320}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 16, 90, 160}, {{1, 16, 90, 160}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 16, 180, 320}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 21, 90, 160}, {{1, 21, 90, 160}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 21, 180, 320}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 32, 90, 160}, {{1, 32, 90, 160}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 32, 180, 320}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 40, 90, 160}, {{1, 40, 90, 160}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 40, 180, 320}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 64, 90, 160}, {{1, 64, 90, 160}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 64, 180, 320}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 111, 90, 160}, {{1, 111, 90, 160}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 111, 180, 320}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 128, 90, 160}, {{1, 128, 90, 160}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 128, 180, 320}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 205, 90, 160}, {{1, 205, 90, 160}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 205, 180, 320}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 256, 90, 160}, {{1, 256, 90, 160}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 256, 180, 320}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 512, 90, 160}, {{1, 512, 90, 160}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 512, 180, 320}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
};

INSTANTIATE_TEST_SUITE_P(Interpolate_I8_CPU_Test_09, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesBenchmark,                                     // InterpolateSpecificParams
            ::testing::ValuesIn(shapeParamsBenchmark_09),               // ShapeParams
            ::testing::ValuesIn(prcBenchmark),     // ElementType
            ::testing::ValuesIn(filterCPUInfoForDeviceTestI8()),              // CPUSpecificParams
            ::testing::ValuesIn(interpolateFusingParamsBenchmark),      // fusingSpecificParams
            ::testing::ValuesIn(filterAdditionalConfigBenchmark())),             // AdditionalConfig
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Interpolate_Benchmark_CPU_Test_09, InterpolateLayerCPUBenchmarkTest,
        ::testing::Combine(
            interpolateCasesBenchmark,                                     // InterpolateSpecificParams
            ::testing::ValuesIn(shapeParamsBenchmark_09),               // ShapeParams
            ::testing::ValuesIn(prcBenchmark),     // ElementType
            ::testing::ValuesIn(filterCPUInfoForDeviceTestI8()),              // CPUSpecificParams
            ::testing::ValuesIn(interpolateFusingParamsBenchmark),      // fusingSpecificParams
            ::testing::ValuesIn(filterAdditionalConfigBenchmark())),             // AdditionalConfig
    InterpolateLayerCPUBenchmarkTest::getTestCaseName);


const std::vector<ShapeParams> shapeParamsBenchmark_10 = {
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 1, 180, 320}, {{1, 1, 180, 320}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 1, 360, 640}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 2, 180, 320}, {{1, 2, 180, 320}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 2, 360, 640}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 3, 180, 320}, {{1, 3, 180, 320}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 3, 360, 640}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 4, 180, 320}, {{1, 4, 180, 320}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 4, 360, 640}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 5, 180, 320}, {{1, 5, 180, 320}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 5, 360, 640}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 8, 180, 320}, {{1, 8, 180, 320}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 8, 360, 640}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 10, 180, 320}, {{1, 10, 180, 320}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 10, 360, 640}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 16, 180, 320}, {{1, 16, 180, 320}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 16, 360, 640}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 21, 180, 320}, {{1, 21, 180, 320}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 21, 360, 640}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 32, 180, 320}, {{1, 32, 180, 320}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 32, 360, 640}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 40, 180, 320}, {{1, 40, 180, 320}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 40, 360, 640}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 64, 180, 320}, {{1, 64, 180, 320}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 64, 360, 640}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 111, 180, 320}, {{1, 111, 180, 320}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 111, 360, 640}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 128, 180, 320}, {{1, 128, 180, 320}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 128, 360, 640}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 205, 180, 320}, {{1, 205, 180, 320}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 205, 360, 640}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 256, 180, 320}, {{1, 256, 180, 320}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 256, 360, 640}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 512, 180, 320}, {{1, 512, 180, 320}}},              // Input shapes
        ngraph::helpers::InputLayerType::PARAMETER,                    // input type
        {{1, 512, 360, 640}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
};

INSTANTIATE_TEST_SUITE_P(Interpolate_I8_CPU_Test_10, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesBenchmark,                                     // InterpolateSpecificParams
            ::testing::ValuesIn(shapeParamsBenchmark_10),               // ShapeParams
            ::testing::ValuesIn(prcBenchmark),     // ElementType
            ::testing::ValuesIn(filterCPUInfoForDeviceTestI8()),              // CPUSpecificParams
            ::testing::ValuesIn(interpolateFusingParamsBenchmark),      // fusingSpecificParams
            ::testing::ValuesIn(filterAdditionalConfigBenchmark())),             // AdditionalConfig
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Interpolate_Benchmark_CPU_Test_10, InterpolateLayerCPUBenchmarkTest,
        ::testing::Combine(
            interpolateCasesBenchmark,                                     // InterpolateSpecificParams
            ::testing::ValuesIn(shapeParamsBenchmark_10),               // ShapeParams
            ::testing::ValuesIn(prcBenchmark),     // ElementType
            ::testing::ValuesIn(filterCPUInfoForDeviceTestI8()),              // CPUSpecificParams
            ::testing::ValuesIn(interpolateFusingParamsBenchmark),      // fusingSpecificParams
            ::testing::ValuesIn(filterAdditionalConfigBenchmark())),             // AdditionalConfig
    InterpolateLayerCPUBenchmarkTest::getTestCaseName);


const std::vector<ShapeParams> shapeParamsBenchmark_11 = {
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 2, 16, 1}, {{1, 2, 16, 1}}},              // Input shapes
        ngraph::helpers::InputLayerType::CONSTANT,                    // input type
        {{1, 2, 16, 1}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 2, 16, 2}, {{1, 2, 16, 2}}},              // Input shapes
        ngraph::helpers::InputLayerType::CONSTANT,                    // input type
        {{1, 2, 16, 2}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 2, 16, 3}, {{1, 2, 16, 3}}},              // Input shapes
        ngraph::helpers::InputLayerType::CONSTANT,                    // input type
        {{1, 2, 16, 3}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 2, 16, 4}, {{1, 2, 16, 4}}},              // Input shapes
        ngraph::helpers::InputLayerType::CONSTANT,                    // input type
        {{1, 2, 16, 4}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 2, 16, 5}, {{1, 2, 16, 5}}},              // Input shapes
        ngraph::helpers::InputLayerType::CONSTANT,                    // input type
        {{1, 2, 16, 5}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 2, 16, 8}, {{1, 2, 16, 8}}},              // Input shapes
        ngraph::helpers::InputLayerType::CONSTANT,                    // input type
        {{1, 2, 16, 8}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 2, 16, 10}, {{1, 2, 16, 10}}},              // Input shapes
        ngraph::helpers::InputLayerType::CONSTANT,                    // input type
        {{1, 2, 16, 10}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 2, 16, 16}, {{1, 2, 16, 16}}},              // Input shapes
        ngraph::helpers::InputLayerType::CONSTANT,                    // input type
        {{1, 2, 16, 16}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 2, 16, 21}, {{1, 2, 16, 21}}},              // Input shapes
        ngraph::helpers::InputLayerType::CONSTANT,                    // input type
        {{1, 2, 16, 21}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 2, 16, 32}, {{1, 2, 16, 32}}},              // Input shapes
        ngraph::helpers::InputLayerType::CONSTANT,                    // input type
        {{1, 2, 16, 32}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 2, 16, 40}, {{1, 2, 16, 40}}},              // Input shapes
        ngraph::helpers::InputLayerType::CONSTANT,                    // input type
        {{1, 2, 16, 40}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 2, 16, 64}, {{1, 2, 16, 64}}},              // Input shapes
        ngraph::helpers::InputLayerType::CONSTANT,                    // input type
        {{1, 2, 16, 64}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 2, 16, 111}, {{1, 2, 16, 111}}},              // Input shapes
        ngraph::helpers::InputLayerType::CONSTANT,                    // input type
        {{1, 2, 16, 111}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 2, 16, 128}, {{1, 2, 16, 128}}},              // Input shapes
        ngraph::helpers::InputLayerType::CONSTANT,                    // input type
        {{1, 2, 16, 128}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 2, 16, 192}, {{1, 2, 16, 192}}},              // Input shapes
        ngraph::helpers::InputLayerType::CONSTANT,                    // input type
        {{1, 2, 16, 192}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 2, 16, 199}, {{1, 2, 16, 199}}},              // Input shapes
        ngraph::helpers::InputLayerType::CONSTANT,                    // input type
        {{1, 2, 16, 199}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 3, 16, 1}, {{1, 3, 16, 1}}},              // Input shapes
        ngraph::helpers::InputLayerType::CONSTANT,                    // input type
        {{1, 3, 16, 1}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 3, 16, 2}, {{1, 3, 16, 2}}},              // Input shapes
        ngraph::helpers::InputLayerType::CONSTANT,                    // input type
        {{1, 3, 16, 2}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 3, 16, 3}, {{1, 3, 16, 3}}},              // Input shapes
        ngraph::helpers::InputLayerType::CONSTANT,                    // input type
        {{1, 3, 16, 3}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 3, 16, 4}, {{1, 3, 16, 4}}},              // Input shapes
        ngraph::helpers::InputLayerType::CONSTANT,                    // input type
        {{1, 3, 16, 4}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 3, 16, 5}, {{1, 3, 16, 5}}},              // Input shapes
        ngraph::helpers::InputLayerType::CONSTANT,                    // input type
        {{1, 3, 16, 5}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 3, 16, 8}, {{1, 3, 16, 8}}},              // Input shapes
        ngraph::helpers::InputLayerType::CONSTANT,                    // input type
        {{1, 3, 16, 8}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 3, 16, 10}, {{1, 3, 16, 10}}},              // Input shapes
        ngraph::helpers::InputLayerType::CONSTANT,                    // input type
        {{1, 3, 16, 10}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 3, 16, 16}, {{1, 3, 16, 16}}},              // Input shapes
        ngraph::helpers::InputLayerType::CONSTANT,                    // input type
        {{1, 3, 16, 16}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 3, 16, 21}, {{1, 3, 16, 21}}},              // Input shapes
        ngraph::helpers::InputLayerType::CONSTANT,                    // input type
        {{1, 3, 16, 21}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 3, 16, 32}, {{1, 3, 16, 32}}},              // Input shapes
        ngraph::helpers::InputLayerType::CONSTANT,                    // input type
        {{1, 3, 16, 32}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 3, 16, 40}, {{1, 3, 16, 40}}},              // Input shapes
        ngraph::helpers::InputLayerType::CONSTANT,                    // input type
        {{1, 3, 16, 40}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 3, 16, 64}, {{1, 3, 16, 64}}},              // Input shapes
        ngraph::helpers::InputLayerType::CONSTANT,                    // input type
        {{1, 3, 16, 64}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 3, 16, 111}, {{1, 3, 16, 111}}},              // Input shapes
        ngraph::helpers::InputLayerType::CONSTANT,                    // input type
        {{1, 3, 16, 111}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 3, 16, 128}, {{1, 3, 16, 128}}},              // Input shapes
        ngraph::helpers::InputLayerType::CONSTANT,                    // input type
        {{1, 3, 16, 128}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 3, 16, 192}, {{1, 3, 16, 192}}},              // Input shapes
        ngraph::helpers::InputLayerType::CONSTANT,                    // input type
        {{1, 3, 16, 192}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,             // ShapeCalculationMode
        InputShape{{1, 3, 16, 199}, {{1, 3, 16, 199}}},              // Input shapes
        ngraph::helpers::InputLayerType::CONSTANT,                    // input type
        {{1, 3, 16, 199}},                                            // scales or sizes values
        defaultAxesBenchmark.front()                                   // axes
    },
};

INSTANTIATE_TEST_SUITE_P(Interpolate_I8_CPU_Test_11, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesBenchmark,                                     // InterpolateSpecificParams
            ::testing::ValuesIn(shapeParamsBenchmark_11),               // ShapeParams
            ::testing::ValuesIn(prcBenchmark),     // ElementType
            ::testing::ValuesIn(filterCPUInfoForDeviceTestI8()),              // CPUSpecificParams
            ::testing::ValuesIn(interpolateFusingParamsBenchmark),      // fusingSpecificParams
            ::testing::ValuesIn(filterAdditionalConfigBenchmark())),             // AdditionalConfig
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Interpolate_Benchmark_CPU_Test_11, InterpolateLayerCPUBenchmarkTest,
        ::testing::Combine(
            interpolateCasesBenchmark,                                     // InterpolateSpecificParams
            ::testing::ValuesIn(shapeParamsBenchmark_11),               // ShapeParams
            ::testing::ValuesIn(prcBenchmark),     // ElementType
            ::testing::ValuesIn(filterCPUInfoForDeviceTestI8()),              // CPUSpecificParams
            ::testing::ValuesIn(interpolateFusingParamsBenchmark),      // fusingSpecificParams
            ::testing::ValuesIn(filterAdditionalConfigBenchmark())),             // AdditionalConfig
    InterpolateLayerCPUBenchmarkTest::getTestCaseName);

}  // namespace benchmark

} // namespace CPULayerTestsDefinitions
