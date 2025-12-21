#include <granite/granite.h>

#ifdef __ANDROID__

#include <jni.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr const char* kLogTag = "GraniteJNI";

void log_error(const char* msg) {
    __android_log_print(ANDROID_LOG_ERROR, kLogTag, "%s", msg);
}

void log_info(const char* msg) {
    __android_log_print(ANDROID_LOG_INFO, kLogTag, "%s", msg);
}

bool copy_asset(AAssetManager* manager, const std::string& asset_path,
                const std::filesystem::path& out_path) {
    AAsset* asset = AAssetManager_open(manager, asset_path.c_str(), AASSET_MODE_STREAMING);
    if (!asset) {
        return false;
    }

    std::ofstream out(out_path, std::ios::binary);
    if (!out.is_open()) {
        AAsset_close(asset);
        return false;
    }

    std::vector<char> buffer(16 * 1024);
    int read_bytes = 0;
    while ((read_bytes = AAsset_read(asset, buffer.data(),
                                     static_cast<int>(buffer.size()))) > 0) {
        out.write(buffer.data(), read_bytes);
    }

    AAsset_close(asset);
    return true;
}

bool extract_asset_dir(AAssetManager* manager,
                       const std::string& asset_dir,
                       const std::filesystem::path& out_dir) {
    if (!manager) {
        return false;
    }

    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);

    AAssetDir* dir = AAssetManager_openDir(manager, asset_dir.c_str());
    if (!dir) {
        return false;
    }

    bool copied_any = false;
    const char* filename = nullptr;
    while ((filename = AAssetDir_getNextFileName(dir)) != nullptr) {
        std::string asset_path = asset_dir + "/" + filename;
        std::filesystem::path out_path = out_dir / filename;
        if (copy_asset(manager, asset_path, out_path)) {
            copied_any = true;
        }
    }

    AAssetDir_close(dir);
    return copied_any;
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_granite_GraniteNative_getVersion(JNIEnv* env, jclass) {
    return env->NewStringUTF(granite::version_string());
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_granite_GraniteNative_extractSpirvAssets(
    JNIEnv* env,
    jclass,
    jobject asset_manager,
    jstring asset_dir,
    jstring output_dir) {
    if (!asset_manager || !asset_dir || !output_dir) {
        log_error("Invalid asset manager or paths");
        return JNI_FALSE;
    }

    const char* asset_dir_c = env->GetStringUTFChars(asset_dir, nullptr);
    const char* output_dir_c = env->GetStringUTFChars(output_dir, nullptr);

    AAssetManager* manager = AAssetManager_fromJava(env, asset_manager);
    bool ok = false;

    if (manager) {
        std::filesystem::path out_dir(output_dir_c);
        ok = extract_asset_dir(manager, asset_dir_c, out_dir);
        if (ok) {
            setenv("GRANITE_VULKAN_PRECOMPILED_DIR", output_dir_c, 1);
            log_info("Extracted SPIR-V assets and set GRANITE_VULKAN_PRECOMPILED_DIR");
        } else {
            log_error("Failed to extract SPIR-V assets");
        }
    } else {
        log_error("Failed to get AAssetManager");
    }

    env->ReleaseStringUTFChars(asset_dir, asset_dir_c);
    env->ReleaseStringUTFChars(output_dir, output_dir_c);

    return ok ? JNI_TRUE : JNI_FALSE;
}

#endif  // __ANDROID__
