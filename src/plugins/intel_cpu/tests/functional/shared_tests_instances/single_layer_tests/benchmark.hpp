// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cctype>
#include <iostream>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace LayerTestsDefinitions {

template <typename BaseLayerTest>
class BenchmarkLayerTest : public BaseLayerTest, virtual public LayerTestsUtils::LayerTestsCommon {
    static_assert(std::is_base_of<LayerTestsUtils::LayerTestsCommon, BaseLayerTest>::value,
                  "BaseLayerTest should inherit from SubgraphBaseTest");

public:
    void Run(const std::initializer_list<std::string>& names,
             const std::chrono::milliseconds warmupTime = std::chrono::milliseconds(2000),
             const int numAttempts = 100) {
        bench_names_ = names;
        warmup_time_ = warmupTime;
        num_attempts_ = numAttempts;
        configuration = {{"PERF_COUNT", "YES"}};
        BaseLayerTest::Run();
    }

    void Run(const std::string& name,
             const std::chrono::milliseconds warmupTime = std::chrono::milliseconds(2000),
             const int numAttempts = 100) {
        if (!name.empty()) {
            Run({name}, warmupTime, numAttempts);
        } else {
            Run({}, warmupTime, numAttempts);
        }
    }

    void Validate() override {
        // NOTE: Validation is ignored because we are interested in benchmarks results
    }

protected:
    void Infer() override {
        // Operation names search
        std::map<std::string, long long> results_us{};
        BaseLayerTest::Infer();
        const auto& perf_results = inferRequest.GetPerformanceCounts();
        // const auto& perf_results = inferRequest.get_profiling_info();
        for (const auto& name : bench_names_) {
            bool found = false;
            for (const auto& result : perf_results) {
                const auto& resName = result.first;
                // const auto& resName = result.node_name;
                const bool shouldAdd =
                    !name.empty() && resName.find(name) != std::string::npos && resName.find('_') != std::string::npos;
                // Adding operations with numbers for the case there are several operations of the same type
                if (shouldAdd) {
                    found = true;
                    results_us.emplace(std::make_pair(resName, 0));
                }
            }
            if (!found) {
                std::cout << "WARNING! Performance count for \"" << name << "\" wasn't found!\n";
            }
        }
        // If no operations were found adding the time of all operations except Parameter and Result
        if (results_us.empty()) {
            for (const auto& result : perf_results) {
                const auto& resName = result.first;
                // const auto& resName = result.node_name;
                const bool shouldAdd = (resName.find("Parameter") == std::string::npos) &&
                                       (resName.find("Result") == std::string::npos) &&
                                       (resName.find('_') != std::string::npos);
                if (shouldAdd) {
                    results_us.emplace(std::make_pair(resName, 0));
                }
            }
        }
        // Warmup
        auto warmCur = std::chrono::steady_clock::now();
        const auto warmEnd = warmCur + warmup_time_;
        while (warmCur < warmEnd) {
            // BaseLayerTest::Infer();
            inferRequest.Infer();
            warmCur = std::chrono::steady_clock::now();
        }
        // Benchmark
        for (int i = 0; i < num_attempts_; ++i) {
            // BaseLayerTest::Infer();
            inferRequest.Infer();
            const auto& perf_results = inferRequest.GetPerformanceCounts();
            // const auto& perf_results = inferRequest.get_profiling_info();
            for (auto& el : results_us) {
                const auto& name = el.first;
                auto& time = el.second;
                time += perf_results.at(name).realTime_uSec;
            }
            // for (const auto& result : perf_results) {
            //     results_us.at(result.node_name) += result.real_time.count();
            // }
        }

        long long total_us = 0;
        for (const auto& el : results_us) {
            const auto& name = el.first;
            const auto time = el.second / num_attempts_;
            total_us += time;
            std::cout << std::fixed << std::setfill('0') << name << ": " << time << " us\n";
        }
        std::cout << std::fixed << std::setfill('0') << "Total time: " << total_us << " us\n";
    }

private:
    std::vector<std::string> bench_names_;
    std::chrono::milliseconds warmup_time_;
    int num_attempts_;
};
} // namespace LayerTestsDefinitions

// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// #pragma once

// #include <cctype>
// #include <iostream>

// // #include "layer_test_utils.hpp"
// #include "shared_test_classes/base/ov_subgraph.hpp"

// namespace LayerTestsDefinitions {

// template <typename BaseLayerTest>
// class BenchmarkLayerTest : public BaseLayerTest {
//     static_assert(std::is_base_of<LayerTestsUtils::LayerTestsCommon, BaseLayerTest>::value,
//                   "BaseLayerTest should inherit from LayerTestsUtils::LayerTestsCommon");

// public:
//     static constexpr int kDefaultNumberOfAttempts = 100;

//     void RunBenchmark(const std::initializer_list<std::string>& node_type_names,
//                       const std::chrono::milliseconds warmupTime = std::chrono::milliseconds(2000),
//                       const int numAttempts = kDefaultNumberOfAttempts) {
//         bench_node_type_names_ = node_type_names;
//         warmup_time_ = warmupTime;
//         num_attempts_ = numAttempts;
//         LayerTestsUtils::LayerTestsCommon::configuration = {{"PERF_COUNT", "YES"}};
//         LayerTestsUtils::LayerTestsCommon::Run();
//     }

//     void RunBenchmark(const std::string& name,
//                       const std::chrono::milliseconds warmupTime = std::chrono::milliseconds(2000),
//                       const int numAttempts = kDefaultNumberOfAttempts) {
//         if (!name.empty()) {
//             RunBenchmark({name}, warmupTime, numAttempts);
//         } else {
//             RunBenchmark({}, warmupTime, numAttempts);
//         }
//     }

//     // NOTE: Validation is ignored because we are interested in benchmarks results.
//     //       In future the validate method could check if new benchmark results are not worse than previous one
//     //       (regression test), and in case of any performance issue report it in PR.
//     void Validate() override {
//     }

// protected:
//     void Infer() override {
//         std::map<std::string, long long> results_us{};
//         for (const auto& node_type_name : bench_node_type_names_) {
//             results_us[node_type_name] = {};
//         }

//         // Warmup
//         auto warm_current = std::chrono::steady_clock::now();
//         const auto warm_end = warm_current + warmup_time_;
//         while (warm_current < warm_end) {
//             LayerTestsUtils::LayerTestsCommon::Infer();
//             warm_current = std::chrono::steady_clock::now();
//         }

//         // Benchmark
//         for (int i = 0; i < num_attempts_; ++i) {
//             LayerTestsUtils::LayerTestsCommon::Infer();
//             const auto& perf_results = LayerTestsUtils::LayerTestsCommon::inferRequest.GetPerformanceCounts();
//             for (auto& res : results_us) {
//                 const std::string node_type_name = res.first;
//                 long long& time = res.second;
//                 auto found_profile = std::find_if(perf_results.begin(), perf_results.end(),
//                     // [&node_type_name](const InferenceEngine::InferenceEngineProfileInfo& profile) {
//                     [&node_type_name](const std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>::value_type& profile) {
//                         return profile.second.layer_type == node_type_name;
//                     });
//                 assert(found_profile != perf_results.end());
//                 time += found_profile->second.realTime_uSec;
//             }
//         }

//         long long total_us = 0;
//         for (auto& res : results_us) {
//             const std::string node_type_name = res.first;
//             long long& time = res.second;
//             time /= num_attempts_;
//             total_us += time;
//             std::cout << std::fixed << std::setfill('0') << node_type_name << ": " << time << " us\n";
//         }
//         std::cout << std::fixed << std::setfill('0') << "Total time: " << total_us << " us\n";
//     }

// private:
//     std::vector<std::string> bench_node_type_names_;
//     std::chrono::milliseconds warmup_time_;
//     int num_attempts_;
// };

// }  // namespace LayerTestsDefinitions

// namespace ov {
// namespace test {

// template <typename BaseLayerTest>
// class BenchmarkLayerTest : public BaseLayerTest {
//     static_assert(std::is_base_of<SubgraphBaseTest, BaseLayerTest>::value,
//                   "BaseLayerTest should inherit from LayerTestsUtils::LayerTestsCommon");

//  public:
//     static constexpr int kDefaultNumberOfAttempts = 100;

//     void run(const std::initializer_list<std::string>& node_type_names,
//              const std::chrono::milliseconds warmupTime = std::chrono::milliseconds(2000),
//              const int numAttempts = kDefaultNumberOfAttempts) {
//         bench_node_type_names_ = node_type_names;
//         warmup_time_ = warmupTime;
//         num_attempts_ = numAttempts;
//         SubgraphBaseTest::configuration = {{"PERF_COUNT", "YES"}};
//         SubgraphBaseTest::run();
//     }

//     void run(const std::string& node_type_name,
//              const std::chrono::milliseconds warmupTime = std::chrono::milliseconds(2000),
//              const int numAttempts = kDefaultNumberOfAttempts) {
//         if (!node_type_name.empty()) {
//             run({node_type_name}, warmupTime, numAttempts);
//         } else {
//             run({}, warmupTime, numAttempts);
//         }
//     }

//     // NOTE: Validation is ignored because we are interested in benchmarks results.
//     //       In future the validate method could check if new benchmark results are not worse than previous one
//     //       (regression test), and in case of any performance issue report it in PR.
//     void validate() override {
//     }

//  protected:
//     void infer() override {
//         std::map<std::string, long long> results_us{};
//         for (const auto& node_type_name : bench_node_type_names_) {
//             results_us[node_type_name] = {};
//         }

//         // Warmup
//         auto warm_current = std::chrono::steady_clock::now();
//         const auto warm_end = warm_current + warmup_time_;
//         while (warm_current < warm_end) {
//             SubgraphBaseTest::infer();
//             warm_current = std::chrono::steady_clock::now();
//         }

//         // Benchmark
//         for (int i = 0; i < num_attempts_; ++i) {
//             SubgraphBaseTest::infer();
//             const auto& profiling_info = SubgraphBaseTest::inferRequest.get_profiling_info();
//             for (auto& res : results_us) {
//                 const std::string node_type_name = res.first;
//                 long long& time = res.second;
//                 auto found_profile = std::find_if(profiling_info.begin(), profiling_info.end(),
//                     [&node_type_name](const ProfilingInfo& profile) {
//                         return profile.node_type == node_type_name;
//                     });
//                 assert(found_profile != profiling_info.end());
//                 time += found_profile->real_time.count();
//             }
//         }

//         long long total_us = 0;
//         for (auto& res : results_us) {
//             const std::string node_type_name = res.first;
//             long long& time = res.second;
//             time /= num_attempts_;
//             total_us += time;
//             std::cout << std::fixed << std::setfill('0') << node_type_name << ": " << time << " us\n";
//         }
//         std::cout << std::fixed << std::setfill('0') << "Total time: " << total_us << " us\n";
//     }

//  private:
//     std::vector<std::string> bench_node_type_names_;
//     std::chrono::milliseconds warmup_time_;
//     int num_attempts_;
// };

// } // namespace test
// } // namespace ov
