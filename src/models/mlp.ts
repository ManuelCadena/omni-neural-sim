/**
 * OmniNeuralSim - Multi-Layer Perceptron (MLP) Engine
 * Full forward and backward propagation with analytical gradients,
 * activation comparisons, and 2D decision boundary simulation.
 */

import { Tensor } from "../core/tensor.ts";
import { ExecutionTracer } from "../core/tracer.ts";

export interface MLPConfig {
  layerSizes: number[]; // e.g. [2, 4, 4, 1]
  activations: ("relu" | "sigmoid" | "tanh" | "linear")[];
  learningRate: number;
}

export class MLP {
  weights: Tensor[] = [];
  biases: Tensor[] = [];
  activations: ("relu" | "sigmoid" | "tanh" | "linear")[] = [];
  tracer = new ExecutionTracer();

  constructor(public config: MLPConfig) {
    this.activations = config.activations;
    for (let i = 0; i < config.layerSizes.length - 1; i++) {
      const inDim = config.layerSizes[i];
      const outDim = config.layerSizes[i + 1];
      // Xavier / He initialization
      const scale = Math.sqrt(2.0 / inDim);
      this.weights.push(Tensor.random([inDim, outDim], -scale, scale));
      this.biases.push(Tensor.zeros([1, outDim]));
    }
  }

  forward(input: Tensor, trace = false): Tensor {
    if (trace) this.tracer.clear();
    let current = input;

    for (let l = 0; l < this.weights.length; l++) {
      const W = this.weights[l];
      const B = this.biases[l];
      const act = this.activations[l];

      const z = current.matmul(W).add(B);
      let a = z;
      if (act === "relu") a = z.relu();
      else if (act === "sigmoid") a = z.sigmoid();
      else if (act === "tanh") a = z.tanh();

      if (trace) {
        this.tracer.record({
          layerId: `layer_${l + 1}`,
          layerName: `Hidden Layer ${l + 1} (${act.toUpperCase()})`,
          operation: "Dense Matmul + Bias + Activation",
          formula: `a^[${l + 1}] = ${act}(W^[${l + 1}] * a^[${l}] + b^[${l + 1}])`,
          inputShapes: [current.shape, W.shape],
          outputShape: a.shape,
          tensorPreview: Array.from(a.data.slice(0, 4)),
          pedagogicalInsight: `Layer ${l + 1} transforms ${W.shape[0]} inputs into ${W.shape[1]} linear combinations, activated by ${act}.`
        });
      }
      current = a;
    }
    return current;
  }

  // Train a single epoch on a dataset X, Y
  trainStep(X: number[][], Y: number[][]): { loss: number } {
    let totalLoss = 0;
    const lr = this.config.learningRate;

    for (let i = 0; i < X.length; i++) {
      const xTensor = Tensor.fromArray([X[i]]);
      const target = Y[i];

      // Forward pass keeping activations
      const layerInputs: Tensor[] = [xTensor];
      const layerZs: Tensor[] = [];
      let a = xTensor;

      for (let l = 0; l < this.weights.length; l++) {
        const z = a.matmul(this.weights[l]).add(this.biases[l]);
        layerZs.push(z);
        const act = this.activations[l];
        if (act === "relu") a = z.relu();
        else if (act === "sigmoid") a = z.sigmoid();
        else if (act === "tanh") a = z.tanh();
        layerInputs.push(a);
      }

      // Compute loss (MSE)
      const pred = a.data[0];
      const diff = pred - target[0];
      totalLoss += 0.5 * diff * diff;

      // Backward pass
      let delta = new Tensor([1, 1], [diff]); // dL/dOutput
      for (let l = this.weights.length - 1; l >= 0; l--) {
        const z = layerZs[l];
        const act = this.activations[l];
        const prevA = layerInputs[l];

        // Activation derivative
        const dAct = new Tensor(z.shape);
        for (let j = 0; j < z.size; j++) {
          if (act === "relu") dAct.data[j] = z.data[j] > 0 ? 1 : 0;
          else if (act === "sigmoid") {
            const s = 1 / (1 + Math.exp(-z.data[j]));
            dAct.data[j] = s * (1 - s);
          } else if (act === "tanh") {
            const t = Math.tanh(z.data[j]);
            dAct.data[j] = 1 - t * t;
          } else {
            dAct.data[j] = 1;
          }
        }

        const dZ = delta.mul(dAct);
        const dW = prevA.transpose().matmul(dZ);

        // Gradient descent update
        for (let k = 0; k < this.weights[l].size; k++) {
          this.weights[l].data[k] -= lr * dW.data[k];
        }
        for (let k = 0; k < this.biases[l].size; k++) {
          this.biases[l].data[k] -= lr * dZ.data[k];
        }

        if (l > 0) {
          delta = dZ.matmul(this.weights[l].transpose());
        }
      }
    }

    return { loss: totalLoss / X.length };
  }

  /**
   * Evaluate a 2D grid of points for boundary visualization (e.g., XOR, Circles, Moons)
   */
  evaluateGrid(resolution = 30, range = 4): { grid: number[][]; xRange: number[]; yRange: number[] } {
    const grid: number[][] = [];
    const step = (range * 2) / resolution;
    const xRange = [];
    const yRange = [];

    for (let r = 0; r < resolution; r++) {
      const row: number[] = [];
      const y = range - r * step;
      yRange.push(y);
      for (let c = 0; c < resolution; c++) {
        const x = -range + c * step;
        if (r === 0) xRange.push(x);
        const out = this.forward(new Tensor([1, 2], [x, y]));
        row.push(out.data[0]);
      }
      grid.push(row);
    }
    return { grid, xRange, yRange };
  }
}
