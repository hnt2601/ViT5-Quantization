from pathlib import Path
from progress.bar import Bar
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType


def quantize(models_config, approach="dynamic"):
    """
    Quantize the weights of the model from float32 to in8 to allow very efficient inference on modern CPU

    Uses unsigned ints for activation values, signed ints for weights, per
    https://onnxruntime.ai/docs/performance/quantization.html#data-type-selection
    it is faster on most CPU architectures
    Args:
        onnx_model_path: Path to location the exported ONNX model is stored
    Returns: The Path generated for the quantized
    """

    bar = Bar("Quantizing...", max=3)

    quant_model_paths = []
    for config_name, config_data in models_config.items():
        model_path = config_data["model_path"]
        model_name = model_path.as_posix()
        nodes_to_quantize = config_data["nodes_to_quantize"]
        output_model_name = Path(f"{model_name[:-5]}-quantized.onnx")

        if approach == "dynamic":
            quantize_dynamic(
                model_input=model_path,
                model_output=output_model_name,
                weight_type=QuantType.QInt8,
                per_channel=True,
                reduce_range=True,
                nodes_to_quantize=nodes_to_quantize,
            )

            quant_model_paths.append(output_model_name)

        elif approach == "static":
            pass

        bar.next()

    bar.finish()

    return tuple(quant_model_paths)
