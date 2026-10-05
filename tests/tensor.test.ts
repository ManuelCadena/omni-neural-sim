import { describe, it, expect } from "vitest";
import { Tensor } from "../src/core/tensor.ts";
import { MLP } from "../src/models/mlp.ts";
import { CNNModel } from "../src/models/cnn.ts";
import { LSTMModel } from "../src/models/rnn.ts";
import { TransformerAttention } from "../src/models/transformer.ts";

describe("Tensor Operations", () => {
  it("computes 2D matrix multiplication correctly", () => {
    // [[1, 2], [3, 4]] x [[2, 0], [1, 2]]
    // row 0: 1*2 + 2*1 = 4, 1*0 + 2*2 = 4
    // row 1: 3*2 + 4*1 = 10, 3*0 + 4*2 = 8
    const A = new Tensor([2, 2], [1, 2, 3, 4]);
    const B = new Tensor([2, 2], [2, 0, 1, 2]);
    const C = A.matmul(B);

    expect(C.shape).toEqual([2, 2]);
    expect(C.get(0, 0)).toBe(4);
    expect(C.get(0, 1)).toBe(4);
    expect(C.get(1, 0)).toBe(10);
    expect(C.get(1, 1)).toBe(8);
  });

  it("computes 2D convolution output shape correctly", () => {
    // 28x28x1 image with 3x3 kernel, 32 filters -> 26x26x32
    const img = Tensor.zeros([28, 28, 1]);
    const kernel = Tensor.zeros([3, 3, 1, 32]);
    const out = img.conv2d(kernel, undefined, 1, 0);

    expect(out.shape).toEqual([26, 26, 32]);
  });

  it("computes max pooling 2x2 shape correctly", () => {
    // 26x26x32 -> 13x13x32
    const feat = Tensor.zeros([26, 26, 32]);
    const pooled = feat.maxPool2d(2, 2);

    expect(pooled.shape).toEqual([13, 13, 32]);
  });
});

describe("Model Architectures", () => {
  it("MLP executes forward pass and traces steps", () => {
    const mlp = new MLP({
      layerSizes: [2, 4, 1],
      activations: ["tanh", "sigmoid"],
      learningRate: 0.05
    });
    const x = new Tensor([1, 2], [0.5, -0.5]);
    const out = mlp.forward(x, true);

    expect(out.shape).toEqual([1, 1]);
    expect(mlp.tracer.steps.length).toBe(2);
  });

  it("CNN executes full MNIST forward pipeline", () => {
    const cnn = new CNNModel();
    const img = Tensor.zeros([28, 28, 1]);
    const res = cnn.forward(img, true);

    expect(res.probabilities.length).toBe(10);
    expect(res.conv1Out.shape).toEqual([26, 26, 32]);
    expect(res.pool1Out.shape).toEqual([13, 13, 32]);
    expect(res.conv2Out.shape).toEqual([11, 11, 64]);
    expect(res.conv3Out.shape).toEqual([3, 3, 64]);
  });

  it("LSTM executes sequence unrolling with 4 gates", () => {
    const lstm = new LSTMModel(3, 4);
    const seq = [
      { token: "Hello", vector: [0.1, 0.2, 0.3] },
      { token: "World", vector: [0.4, 0.5, 0.6] }
    ];
    const history = lstm.unroll(seq);

    expect(history.length).toBe(2);
    expect(history[0].f_gate.length).toBe(4);
    expect(history[0].i_gate.length).toBe(4);
    expect(history[0].o_gate.length).toBe(4);
    expect(history[0].c_t.length).toBe(4);
  });

  it("Transformer computes scaled dot-product attention matrix", () => {
    const tf = new TransformerAttention(8, 2);
    const X = Tensor.random([4, 8]);
    const res = tf.forward(X, true, true);

    expect(res.attentionWeights.length).toBe(4);
    expect(res.attentionWeights[0].length).toBe(4);
    // Causal mask check: row 0 token can only attend to itself, so weight at [0][1] must be ~0
    expect(res.attentionWeights[0][1]).toBeLessThan(1e-5);
  });
});
