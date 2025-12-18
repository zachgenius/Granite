#include <catch2/catch_test_macros.hpp>
#include <granite/granite.h>

using namespace granite;

TEST_CASE("Config presets", "[Config]") {
    SECTION("Performance preset") {
        auto config = Config::Performance();
        REQUIRE(config.attention_backend == AttentionBackend::MetalFlash);
        REQUIRE(config.power_mode == PowerMode::Performance);
        REQUIRE(config.thermal_mode == ThermalMode::Unrestricted);
        REQUIRE(config.kv_cache_max_seq == 4096);
    }

    SECTION("Balanced preset") {
        auto config = Config::Balanced();
        REQUIRE(config.attention_backend == AttentionBackend::Auto);
        REQUIRE(config.power_mode == PowerMode::Balanced);
        REQUIRE(config.thermal_mode == ThermalMode::Adaptive);
        REQUIRE(config.max_memory_mb == 1024);
    }

    SECTION("Battery preset") {
        auto config = Config::Battery();
        REQUIRE(config.attention_backend == AttentionBackend::CoreML);
        REQUIRE(config.power_mode == PowerMode::LowPower);
        REQUIRE(config.use_neural_engine == true);
        REQUIRE(config.prefer_efficiency_cores == true);
    }

    SECTION("LowMemory preset") {
        auto config = Config::LowMemory();
        REQUIRE(config.max_memory_mb == 256);
        REQUIRE(config.kv_cache_max_seq == 512);
        REQUIRE(config.weight_cache_policy == CachePolicy::Streaming);
    }

    SECTION("Background preset") {
        auto config = Config::Background();
        REQUIRE(config.yield_interval_ms == 50);
        REQUIRE(config.prefer_efficiency_cores == true);
    }
}

TEST_CASE("ConfigBuilder fluent API", "[Config]") {
    SECTION("Build custom config") {
        auto config = ConfigBuilder()
            .attention_backend(AttentionBackend::MetalLegacy)
            .max_memory_mb(512)
            .kv_cache_max_seq(1024)
            .power_mode(PowerMode::LowPower)
            .enable_profiling(true)
            .build();

        REQUIRE(config.attention_backend == AttentionBackend::MetalLegacy);
        REQUIRE(config.max_memory_mb == 512);
        REQUIRE(config.kv_cache_max_seq == 1024);
        REQUIRE(config.power_mode == PowerMode::LowPower);
        REQUIRE(config.enable_profiling == true);
    }

    SECTION("Build from preset base") {
        auto config = ConfigBuilder(Config::Performance())
            .max_memory_mb(2048)
            .enable_profiling(true)
            .build();

        // Should have Performance preset values except what we changed
        REQUIRE(config.attention_backend == AttentionBackend::MetalFlash);
        REQUIRE(config.power_mode == PowerMode::Performance);
        // But with our overrides
        REQUIRE(config.max_memory_mb == 2048);
        REQUIRE(config.enable_profiling == true);
    }
}

TEST_CASE("String conversions", "[Config]") {
    SECTION("AttentionBackend to_string") {
        REQUIRE(std::string(to_string(AttentionBackend::Auto)) == "Auto");
        REQUIRE(std::string(to_string(AttentionBackend::MetalFlash)) == "MetalFlash");
        REQUIRE(std::string(to_string(AttentionBackend::CoreML)) == "CoreML");
    }

    SECTION("PowerMode to_string") {
        REQUIRE(std::string(to_string(PowerMode::Performance)) == "Performance");
        REQUIRE(std::string(to_string(PowerMode::Balanced)) == "Balanced");
        REQUIRE(std::string(to_string(PowerMode::LowPower)) == "LowPower");
    }

    SECTION("ThermalMode to_string") {
        REQUIRE(std::string(to_string(ThermalMode::Unrestricted)) == "Unrestricted");
        REQUIRE(std::string(to_string(ThermalMode::Adaptive)) == "Adaptive");
        REQUIRE(std::string(to_string(ThermalMode::Conservative)) == "Conservative");
    }
}

TEST_CASE("Platform detection", "[Config]") {
    SECTION("get_device_info returns valid info") {
        auto info = platform::get_device_info();

        // On macOS, should detect the platform
        #if defined(__APPLE__) && !TARGET_OS_SIMULATOR
            REQUIRE(info.platform != platform::Platform::Unknown);
            REQUIRE(info.total_memory_bytes > 0);
        #endif
    }

    SECTION("select_attention_backend respects explicit selection") {
        platform::DeviceInfo device;
        device.platform = platform::Platform::macOS;
        device.chip_family = platform::ChipFamily::M3;
        device.has_neural_engine = true;

        Config config;
        config.attention_backend = AttentionBackend::CPU;

        auto selected = platform::select_attention_backend(config, device);
        REQUIRE(selected == AttentionBackend::CPU);
    }

    SECTION("select_attention_backend auto selects MetalFlash on M-series") {
        platform::DeviceInfo device;
        device.platform = platform::Platform::macOS;
        device.chip_family = platform::ChipFamily::M3;
        device.has_neural_engine = true;

        Config config;
        config.attention_backend = AttentionBackend::Auto;

        auto selected = platform::select_attention_backend(config, device);
        REQUIRE(selected == AttentionBackend::MetalFlash);
    }

    SECTION("select_attention_backend returns CPU for Simulator") {
        platform::DeviceInfo device;
        device.platform = platform::Platform::Simulator;

        Config config;
        config.attention_backend = AttentionBackend::Auto;

        auto selected = platform::select_attention_backend(config, device);
        REQUIRE(selected == AttentionBackend::CPU);
    }
}
