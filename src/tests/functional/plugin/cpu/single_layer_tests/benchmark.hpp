// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cctype>
#include <iostream>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace CPULayerTestsDefinitions {

using namespace ov::test;

template <typename BaseTest>
class BenchmarkLayerTest : public BaseTest, virtual public SubgraphBaseTest {
    static_assert(std::is_base_of<SubgraphBaseTest, BaseTest>::value,
                  "BaseTest should inherit from SubgraphBaseTest");

public:
    void Run(const std::initializer_list<std::string>& names,
             const std::chrono::milliseconds warmupTime = std::chrono::milliseconds(2000),
             const int numAttempts = 100) {
        bench_names_ = names;
        warmup_time_ = warmupTime;
        num_attempts_ = numAttempts;
        // configuration = {{"PERF_COUNT", "YES"}};
        // BaseTest::Run();
        BaseTest::run();
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

    // void Validate() override {
    void validate() override {
        // NOTE: Validation is ignored because we are interested in benchmarks results
    }

protected:
    // void Infer() override {
    void infer() override {
        // Operation names search
        std::map<std::string, long long> results_us{};
        // BaseTest::Infer();
        BaseTest::infer();
        // const auto& perf_results = inferRequest.GetPerformanceCounts();
        const auto& perf_results = inferRequest.get_profiling_info();
        for (const auto& name : bench_names_) {
            bool found = false;
            for (const auto& result : perf_results) {
                // const auto& resName = result.first;
                const auto& resName = result.node_name;
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
                // const auto& resName = result.first;
                const auto& resName = result.node_name;
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
            // BaseTest::Infer();
            BaseTest::infer();
            warmCur = std::chrono::steady_clock::now();
        }
        // Benchmark
        for (int i = 0; i < num_attempts_; ++i) {
            // BaseTest::Infer();
            BaseTest::infer();
            // const auto& perf_results = inferRequest.GetPerformanceCounts();
            const auto& perf_results = inferRequest.get_profiling_info();
            // for (auto& el : results_us) {
            //     const auto& name = el.first;
            //     auto& time = el.second;
            //     time += perf_results.at(name).realTime_uSec;
            // }
            for (const auto& result : perf_results) {
                // results_us.at(result.node_name) += result.real_time.count();
                auto result_us = results_us.find(result.node_name);
                if (result_us != results_us.end()) {
                    result_us->second += result.real_time.count();
                }
            }
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
} // namespace CPULayerTestsDefinitions
