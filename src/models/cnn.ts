/**
 * OmniNeuralSim - Convolutional Neural Network (CNN) Engine
 * 2D & 1D convolutions, interactive kernel sliding, feature map generation,
 * receptive field calculations, and live MNIST digit classification.
 */

import { Tensor } from "../core/tensor.ts";
import { ExecutionTracer } from "../core/tracer.ts";

export interface ConvLayerDef {
  filters: number;
  kernelSize: [number, number];
  stride: number;
  padding: number;
  activation: "relu" | "linear";
}

export class CNNModel {
  tracer = new ExecutionTracer();

  // Assignment 3 exact architecture:
  // Conv2D(32, 3x3) -> MaxPool(2x2) -> Conv2D(64, 3x3) -> MaxPool(2x2) -> Conv2D(64, 3x3) -> Flatten -> Dense(64) -> Dense(10, softmax)
  kernel1: Tensor;
  bias1: Tensor;
  kernel2: Tensor;
  bias2: Tensor;
  kernel3: Tensor;
  bias3: Tensor;
  dense1_W: Tensor;
  dense1_B: Tensor;
  dense2_W: Tensor;
  dense2_B: Tensor;

  constructor() {
    // Initialize with scaled random weights or pre-trained checkpoints
    this.kernel1 = Tensor.random([3, 3, 1, 32], -0.2, 0.2);
    this.bias1 = Tensor.zeros([32]);

    this.kernel2 = Tensor.random([3, 3, 32, 64], -0.15, 0.15);
    this.bias2 = Tensor.zeros([64]);

    this.kernel3 = Tensor.random([3, 3, 64, 64], -0.15, 0.15);
    this.bias3 = Tensor.zeros([64]);

    this.dense1_W = Tensor.random([576, 64], -0.1, 0.1);
    this.dense1_B = Tensor.zeros([1, 64]);

    this.dense2_W = Tensor.random([64, 10], -0.1, 0.1);
    this.dense2_B = Tensor.zeros([1, 10]);
  }

  forward(image28x28x1: Tensor, trace = true): {
    probabilities: number[];
    predictedDigit: number;
    conv1Out: Tensor;
    pool1Out: Tensor;
    conv2Out: Tensor;
    pool2Out: Tensor;
    conv3Out: Tensor;
  } {
    if (trace) this.tracer.clear();

    // 1. Conv2D (32, 3x3, valid) -> 26x26x32
    const c1 = image28x28x1.conv2d(this.kernel1, this.bias1, 1, 0).relu();
    if (trace) {
      this.tracer.record({
        layerId: "conv1",
        layerName: "Conv2D (32 filters, 3x3)",
        operation: "2D Convolution + ReLU",
        formula: "H_out = (28 - 3 + 0)/1 + 1 = 26; Params = (3*3*1 + 1)*32 = 320",
        inputShapes: [image28x28x1.shape, this.kernel1.shape],
        outputShape: c1.shape,
        tensorPreview: Array.from(c1.data.slice(0, 5)),
        fullOutput: c1,
        pedagogicalInsight: "Extracts 32 low-level edge and texture feature maps from the raw 28x28 pixel grid.",
        metadata: { filters: 32, kernel: "3x3", params: 320 }
      });
    }

    // 2. MaxPool (2x2) -> 13x13x32
    const p1 = c1.maxPool2d(2, 2);
    if (trace) {
      this.tracer.record({
        layerId: "pool1",
        layerName: "MaxPooling2D (2x2)",
        operation: "Spatial Downsampling",
        formula: "H_out = floor(26/2) = 13; W_out = 13",
        inputShapes: [c1.shape],
        outputShape: p1.shape,
        tensorPreview: Array.from(p1.data.slice(0, 5)),
        fullOutput: p1,
        pedagogicalInsight: "Preserves the most salient local features while reducing spatial dimensions by 75%."
      });
    }

    // 3. Conv2D (64, 3x3, valid) -> 11x11x64
    const c2 = p1.conv2d(this.kernel2, this.bias2, 1, 0).relu();
    if (trace) {
      this.tracer.record({
        layerId: "conv2",
        layerName: "Conv2D (64 filters, 3x3)",
        operation: "2D Convolution + ReLU",
        formula: "H_out = (13 - 3 + 0)/1 + 1 = 11; Params = (3*3*32 + 1)*64 = 18,496",
        inputShapes: [p1.shape, this.kernel2.shape],
        outputShape: c2.shape,
        tensorPreview: Array.from(c2.data.slice(0, 5)),
        fullOutput: c2,
        pedagogicalInsight: "Combines local features into mid-level parts (corners, loops, strokes)."
      });
    }

    // 4. MaxPool (2x2) -> 5x5x64
    const p2 = c2.maxPool2d(2, 2);
    if (trace) {
      this.tracer.record({
        layerId: "pool2",
        layerName: "MaxPooling2D (2x2)",
        operation: "Spatial Downsampling",
        formula: "H_out = floor(11/2) = 5; W_out = 5",
        inputShapes: [c2.shape],
        outputShape: p2.shape,
        tensorPreview: Array.from(p2.data.slice(0, 5)),
        fullOutput: p2,
        pedagogicalInsight: "Further downsamples to 5x5 feature grids."
      });
    }

    // 5. Conv2D (64, 3x3, valid) -> 3x3x64
    const c3 = p2.conv2d(this.kernel3, this.bias3, 1, 0).relu();
    if (trace) {
      this.tracer.record({
        layerId: "conv3",
        layerName: "Conv2D (64 filters, 3x3)",
        operation: "2D Convolution + ReLU",
        formula: "H_out = (5 - 3 + 0)/1 + 1 = 3; Params = (3*3*64 + 1)*64 = 36,928",
        inputShapes: [p2.shape, this.kernel3.shape],
        outputShape: c3.shape,
        tensorPreview: Array.from(c3.data.slice(0, 5)),
        fullOutput: c3,
        pedagogicalInsight: "High-level digit shape representations."
      });
    }

    // 6. Flatten -> 3*3*64 = 576
    const flat = c3.flatten();
    const flat2D = new Tensor([1, 576], flat.data);

    // 7. Dense (64, relu)
    const d1 = flat2D.matmul(this.dense1_W).add(this.dense1_B).relu();

    // 8. Dense (10, softmax)
    const logits = d1.matmul(this.dense2_W).add(this.dense2_B);
    const probsTensor = logits.softmax(-1);
    const probs = Array.from(probsTensor.data);

    let bestDigit = 0;
    let maxP = -1;
    for (let i = 0; i < 10; i++) {
      if (probs[i] > maxP) {
        maxP = probs[i];
        bestDigit = i;
      }
    }

    if (trace) {
      this.tracer.record({
        layerId: "digit_probs",
        layerName: "Softmax Classification Head",
        operation: "Softmax(logits) -> Argmax",
        formula: "P(digit=k) = exp(z_k) / sum(exp(z)); Predicted = argmax_k(P)",
        inputShapes: [logits.shape],
        outputShape: [10],
        tensorPreview: probs,
        pedagogicalInsight: `Final digit classification with top prediction ${bestDigit} (${(maxP * 100).toFixed(1)}%).`
      });
    }

    return {
      probabilities: probs,
      predictedDigit: bestDigit,
      conv1Out: c1,
      pool1Out: p1,
      conv2Out: c2,
      pool2Out: p2,
      conv3Out: c3
    };
  }

  loadWeightsFromJSON(weightsJSON: any): void {
    if (weightsJSON.kernel1) this.kernel1 = new Tensor([3, 3, 1, 32], weightsJSON.kernel1);
    if (weightsJSON.bias1) this.bias1 = new Tensor([32], weightsJSON.bias1);
    if (weightsJSON.dense2_W) this.dense2_W = new Tensor([64, 10], weightsJSON.dense2_W);
    if (weightsJSON.dense2_B) this.dense2_B = new Tensor([1, 10], weightsJSON.dense2_B);
  }
}
