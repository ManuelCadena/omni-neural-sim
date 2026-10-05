#!/usr/bin/env python3
"""
OmniNeuralSim - Python Exporter Bridge
Converts Keras (TensorFlow) or PyTorch models and weights into OmniNeuralSim JSON specification.
"""

import json
import numpy as np

def export_keras_model(model, filepath="model_spec.json", course_context="CSCI E-89b"):
    """
    Exports a Keras Sequential or Functional model architecture and layer weights.
    """
    layers_spec = []
    weights_dict = {}

    for i, layer in enumerate(model.layers):
        l_type = layer.__class__.__name__.lower()
        cfg = layer.get_config()
        in_shape = list(layer.input_shape)[1:] if hasattr(layer, "input_shape") and layer.input_shape else []
        out_shape = list(layer.output_shape)[1:] if hasattr(layer, "output_shape") and layer.output_shape else []
        
        # Clean shapes from None
        in_shape = [s for s in in_shape if s is not None]
        out_shape = [s for s in out_shape if s is not None]

        weights = layer.get_weights()
        if len(weights) > 0:
            for w_idx, w in enumerate(weights):
                key = f"{layer.name}_w{w_idx}"
                weights_dict[key] = w.flatten().tolist()

        layers_spec.append({
            "id": f"layer_{i}_{layer.name}",
            "name": f"{layer.__class__.__name__} ({layer.name})",
            "type": l_type,
            "inShape": in_shape,
            "outShape": out_shape,
            "activation": cfg.get("activation", "linear"),
            "paramsCount": int(layer.count_params()),
            "config": {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool))}
        })

    spec = {
        "version": "1.0.0",
        "name": getattr(model, "name", "exported_model"),
        "courseContext": course_context,
        "modelType": "cnn_2d" if "conv2d" in [l["type"] for l in layers_spec] else "mlp",
        "layers": layers_spec,
        "totalParameters": int(model.count_params()),
        "weights": weights_dict,
        "metadata": {
            "description": f"Exported from Keras for live browser simulation",
            "recommendedVisualization": "cnn_explainer" if any("conv" in l["type"] for l in layers_spec) else "graph_2d"
        }
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)

    print(f"✅ Successfully exported model to {filepath} ({len(layers_spec)} layers, {spec['totalParameters']:,} parameters)")
    return spec

if __name__ == "__main__":
    print("OmniNeuralSim Python Bridge Ready. Call export_keras_model(model, filepath)")
