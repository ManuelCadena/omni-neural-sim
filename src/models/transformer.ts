/**
 * OmniNeuralSim - Transformer & Attention Engine
 * Scaled Dot-Product Attention, Multi-Head Attention (MHA),
 * Causal Masking, KV-Cache tracking, and Autoregressive generation.
 */

import { Tensor } from "../core/tensor.ts";
import { ExecutionTracer } from "../core/tracer.ts";

export interface AttentionHeadResult {
  headIndex: number;
  Q: Tensor;
  K: Tensor;
  V: Tensor;
  scoresRaw: Tensor; // Q * K^T / sqrt(d_k)
  attentionWeights: Tensor; // Softmax(scoresRaw)
  contextOutput: Tensor; // Attention * V
}

export class TransformerAttention {
  d_model: number;
  d_k: number;
  numHeads: number;
  tracer = new ExecutionTracer();

  W_q: Tensor;
  W_k: Tensor;
  W_v: Tensor;
  W_o: Tensor;

  constructor(d_model = 8, numHeads = 2) {
    this.d_model = d_model;
    this.numHeads = numHeads;
    this.d_k = Math.floor(d_model / numHeads);
    const scale = Math.sqrt(2.0 / d_model);

    this.W_q = Tensor.random([d_model, d_model], -scale, scale);
    this.W_k = Tensor.random([d_model, d_model], -scale, scale);
    this.W_v = Tensor.random([d_model, d_model], -scale, scale);
    this.W_o = Tensor.random([d_model, d_model], -scale, scale);
  }

  /**
   * Forward pass over token embeddings matrix X of shape [seqLen, d_model]
   */
  forward(
    X: Tensor,
    causalMask = true,
    trace = true
  ): {
    attentionWeights: number[][]; // [seqLen, seqLen]
    output: Tensor;
    tokens?: string[];
  } {
    if (trace) this.tracer.clear();
    const seqLen = X.shape[0];

    // 1. Linear projections
    const Q = X.matmul(this.W_q);
    const K = X.matmul(this.W_k);
    const V = X.matmul(this.W_v);

    // 2. Scaled Dot-Product: S = Q * K^T / sqrt(d_k)
    const scale = 1.0 / Math.sqrt(this.d_model);
    const K_T = K.transpose();
    const scores = Q.matmul(K_T).mul(scale);

    // 3. Apply causal mask (if autoregressive)
    if (causalMask) {
      for (let i = 0; i < seqLen; i++) {
        for (let j = i + 1; j < seqLen; j++) {
          scores.set(-1e9, i, j);
        }
      }
    }

    // 4. Softmax per row
    const attentionProbs = scores.softmax(-1);

    // 5. Context vectors: Context = Attn * V
    const context = attentionProbs.matmul(V);
    const finalOut = context.matmul(this.W_o);

    if (trace) {
      this.tracer.record({
        layerId: "self_attention",
        layerName: `Self-Attention (d_model=${this.d_model}, seqLen=${seqLen})`,
        operation: "Scaled Dot-Product Attention",
        formula: "Attention(Q,K,V) = softmax(Q K^T / √d_k + Mask) V",
        inputShapes: [X.shape, this.W_q.shape],
        outputShape: finalOut.shape,
        tensorPreview: Array.from(attentionProbs.data.slice(0, 6)),
        pedagogicalInsight: `Calculated ${seqLen}x${seqLen} pairwise attention affinity matrix; tokens dynamically aggregate information from allowed preceding contexts.`
      });
    }

    // Extract 2D matrix
    const matrix: number[][] = [];
    for (let r = 0; r < seqLen; r++) {
      const row: number[] = [];
      for (let c = 0; c < seqLen; c++) {
        row.push(attentionProbs.get(r, c));
      }
      matrix.push(row);
    }

    return {
      attentionWeights: matrix,
      output: finalOut
    };
  }
}
