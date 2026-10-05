/**
 * OmniNeuralSim - Autoencoder & Variational Autoencoder (VAE) Engine
 * Latent manifold 2D space exploration, reconstruction error, and reparameterization.
 */

import { Tensor } from "../core/tensor.ts";
import { ExecutionTracer } from "../core/tracer.ts";

export class AutoencoderModel {
  tracer = new ExecutionTracer();
  // Encoder: [D_in, 16] -> [16, 2] (Latent)
  W_enc1: Tensor;
  b_enc1: Tensor;
  W_enc2: Tensor;
  b_enc2: Tensor;

  // Decoder: [2, 16] -> [16, D_in] (Reconstruction)
  W_dec1: Tensor;
  b_dec1: Tensor;
  W_dec2: Tensor;
  b_dec2: Tensor;

  constructor(public inDim = 10, public latentDim = 2) {
    const scale = 0.3;
    this.W_enc1 = Tensor.random([inDim, 16], -scale, scale);
    this.b_enc1 = Tensor.zeros([1, 16]);
    this.W_enc2 = Tensor.random([16, latentDim], -scale, scale);
    this.b_enc2 = Tensor.zeros([1, latentDim]);

    this.W_dec1 = Tensor.random([latentDim, 16], -scale, scale);
    this.b_dec1 = Tensor.zeros([1, 16]);
    this.W_dec2 = Tensor.random([16, inDim], -scale, scale);
    this.b_dec2 = Tensor.zeros([1, inDim]);
  }

  encode(x: Tensor): Tensor {
    const h = x.matmul(this.W_enc1).add(this.b_enc1).relu();
    return h.matmul(this.W_enc2).add(this.b_enc2);
  }

  decode(z: Tensor): Tensor {
    const h = z.matmul(this.W_dec1).add(this.b_dec1).relu();
    return h.matmul(this.W_dec2).add(this.b_dec2).sigmoid();
  }

  forward(x: Tensor, trace = true): { latent: Tensor; reconstructed: Tensor; mse: number } {
    if (trace) this.tracer.clear();
    const z = this.encode(x);
    const rec = this.decode(z);

    let mse = 0;
    for (let i = 0; i < x.size; i++) {
      const diff = x.data[i] - rec.data[i];
      mse += diff * diff;
    }
    mse /= x.size;

    if (trace) {
      this.tracer.record({
        layerId: "latent_bottleneck",
        layerName: `Latent Space (dim=${this.latentDim})`,
        operation: "Nonlinear Dimensionality Compression",
        formula: "z = W_2 * relu(W_1 x + b_1) + b_2",
        inputShapes: [x.shape],
        outputShape: z.shape,
        tensorPreview: Array.from(z.data),
        pedagogicalInsight: `Compressed input from ${this.inDim} dimensions to ${this.latentDim} latent coordinates with MSE=${mse.toFixed(4)}.`
      });
    }

    return { latent: z, reconstructed: rec, mse };
  }
}
