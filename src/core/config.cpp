#include <granite/config.h>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#endif

namespace granite {
namespace platform {

DeviceInfo get_device_info() {
    DeviceInfo info;

#if defined(__APPLE__)
    // Detect platform
    #if TARGET_OS_SIMULATOR
        info.platform = Platform::Simulator;
    #elif TARGET_OS_IOS
        info.platform = Platform::iOS;
    #elif TARGET_OS_TV
        info.platform = Platform::tvOS;
    #elif TARGET_OS_OSX
        info.platform = Platform::macOS;
    #else
        info.platform = Platform::Unknown;
    #endif

    // Get total memory
    int64_t mem_size = 0;
    size_t len = sizeof(mem_size);
    if (sysctlbyname("hw.memsize", &mem_size, &len, nullptr, 0) == 0) {
        info.total_memory_bytes = static_cast<uint64_t>(mem_size);
    }

    // Get available memory
    vm_statistics64_data_t vm_stats;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                          reinterpret_cast<host_info64_t>(&vm_stats), &count) == KERN_SUCCESS) {
        uint64_t page_size = 0;
        size_t page_size_len = sizeof(page_size);
        sysctlbyname("hw.pagesize", &page_size, &page_size_len, nullptr, 0);
        info.available_memory_bytes = vm_stats.free_count * page_size;
    }

    // Detect chip family from machine model
    char machine[256] = {0};
    size_t machine_len = sizeof(machine);
    if (sysctlbyname("hw.machine", machine, &machine_len, nullptr, 0) == 0) {
        std::string model(machine);

        // M-series detection (macOS)
        if (model.find("arm64") != std::string::npos) {
            // Check for specific chip
            char chip[256] = {0};
            size_t chip_len = sizeof(chip);
            if (sysctlbyname("machdep.cpu.brand_string", chip, &chip_len, nullptr, 0) == 0) {
                std::string chip_str(chip);
                if (chip_str.find("M4") != std::string::npos) {
                    info.chip_family = ChipFamily::M4;
                } else if (chip_str.find("M3") != std::string::npos) {
                    info.chip_family = ChipFamily::M3;
                } else if (chip_str.find("M2") != std::string::npos) {
                    info.chip_family = ChipFamily::M2;
                } else if (chip_str.find("M1") != std::string::npos) {
                    info.chip_family = ChipFamily::M1;
                }
            }
        }

        // A-series detection (iOS)
        if (model.find("iPhone") != std::string::npos || model.find("iPad") != std::string::npos) {
            // Parse version numbers for A-series chips
            // iPhone14,x = A15, iPhone15,x = A16, iPhone16,x = A17, etc.
            size_t pos = model.find_first_of("0123456789");
            if (pos != std::string::npos) {
                int version = std::stoi(model.substr(pos));
                if (version >= 17) info.chip_family = ChipFamily::A18;
                else if (version >= 16) info.chip_family = ChipFamily::A17;
                else if (version >= 15) info.chip_family = ChipFamily::A16;
                else if (version >= 14) info.chip_family = ChipFamily::A15;
                else if (version >= 13) info.chip_family = ChipFamily::A14;
            }
        }
    }

    // Neural Engine available on A11+ and M1+
    info.has_neural_engine = (info.chip_family >= ChipFamily::A14) ||
                             (info.chip_family >= ChipFamily::M1);

    // Metal 3 available on A14+ and M1+
    info.has_metal3 = (info.chip_family >= ChipFamily::A14) ||
                      (info.chip_family >= ChipFamily::M1);

    // Thermal state (would need IOKit for real implementation)
    info.thermal_state = 0.0f;  // Default to cool

    // Power state
    #if TARGET_OS_OSX
        // macOS: assume plugged in (would need IOKit for real check)
        info.is_plugged_in = true;
        info.is_low_power_mode = false;
    #else
        // iOS: default assumptions (would need UIDevice for real check)
        info.is_plugged_in = false;
        info.is_low_power_mode = false;
    #endif

#else
    // Non-Apple platform
    info.platform = Platform::Unknown;
#endif

    return info;
}

AttentionBackend select_attention_backend(const granite::Config& config, const DeviceInfo& device) {
    // Explicit selection overrides auto
    if (config.attention_backend != AttentionBackend::Auto) {
        return config.attention_backend;
    }

    // Simulator: CPU only
    if (device.platform == Platform::Simulator) {
        return AttentionBackend::CPU;
    }

    // iOS with ANE + battery/thermal concerns -> CoreML
    if (device.platform == Platform::iOS) {
        bool prefer_ane = config.power_mode != PowerMode::Performance
                       || config.thermal_mode == ThermalMode::Conservative
                       || device.thermal_state > config.thermal_throttle_threshold
                       || device.is_low_power_mode;

        if (prefer_ane && device.has_neural_engine && config.use_neural_engine) {
            return AttentionBackend::CoreML;
        }
    }

    // macOS or iOS Performance mode -> Metal Flash
    bool has_modern_chip = (device.chip_family >= ChipFamily::M1) ||
                           (device.chip_family >= ChipFamily::A14);
    if (has_modern_chip) {
        return AttentionBackend::MetalFlash;
    }

    // Older devices -> Legacy Metal
    return AttentionBackend::MetalLegacy;
}

} // namespace platform
} // namespace granite
