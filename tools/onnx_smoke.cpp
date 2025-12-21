#include <granite/granite.h>

#include <iostream>

int main(int argc, char* argv[]) {
    granite::init_logging(spdlog::level::info);

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx>\n";
        return 1;
    }

#if !defined(GRANITE_HAS_ONNX)
    std::cerr << "ONNX loader is disabled. Rebuild with GRANITE_ENABLE_ONNX=ON.\n";
    return 2;
#else
    const std::string path = argv[1];
    auto model_result = granite::load_onnx_model(path);
    if (!model_result.ok()) {
        std::cerr << "Failed to load ONNX: " << model_result.error().message() << "\n";
        return 3;
    }

    const auto& model = model_result.value();
    std::cout << model.summary() << "\n";
    std::cout << "Inputs: " << model.graph_inputs.size()
              << " Outputs: " << model.graph_outputs.size() << "\n";
    return 0;
#endif
}
