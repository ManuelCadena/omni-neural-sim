/**
 * OmniNeuralSim - Recurrent Neural Network (RNN / LSTM / GRU) Engine
 * Detailed gate state inspection (f_t, i_t, o_t, c_t for LSTM; r_t, z_t for GRU)
 * and sequence unrolling across time steps.
 */

import { Tensor } from "../core/tensor.ts";
import { ExecutionTracer } from "../core/tracer.ts";

export interface LSTMStepState {
  t: number;
  inputToken: string;
  x_t: number[];
  f_gate: number[]; // Forget gate (0 = forget, 1 = retain)
  i_gate: number[]; // Input gate (0 = ignore, 1 = write)
  c_tilde: number[]; // Candidate state
  c_t: number[]; // Cell memory state
  o_gate: number[]; // Output gate (exposure)
  h_t: number[]; // Hidden state
}

export class LSTMModel {
  hiddenSize: number;
  inputSize: number;
  tracer = new ExecutionTracer();

  // Gates: f, i, c, o (combined or separate)
  W_f: Tensor; U_f: Tensor; b_f: Tensor;
  W_i: Tensor; U_i: Tensor; b_i: Tensor;
  W_c: Tensor; U_c: Tensor; b_c: Tensor;
  W_o: Tensor; U_o: Tensor; b_o: Tensor;

  constructor(inputSize = 4, hiddenSize = 4) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    const scale = 0.5;

    this.W_f = Tensor.random([inputSize, hiddenSize], -scale, scale);
    this.U_f = Tensor.random([hiddenSize, hiddenSize], -scale, scale);
    this.b_f = Tensor.ones([1, hiddenSize]); // Bias initialized to 1 to prevent early vanishing

    this.W_i = Tensor.random([inputSize, hiddenSize], -scale, scale);
    this.U_i = Tensor.random([hiddenSize, hiddenSize], -scale, scale);
    this.b_i = Tensor.zeros([1, hiddenSize]);

    this.W_c = Tensor.random([inputSize, hiddenSize], -scale, scale);
    this.U_c = Tensor.random([hiddenSize, hiddenSize], -scale, scale);
    this.b_c = Tensor.zeros([1, hiddenSize]);

    this.W_o = Tensor.random([inputSize, hiddenSize], -scale, scale);
    this.U_o = Tensor.random([hiddenSize, hiddenSize], -scale, scale);
    this.b_o = Tensor.zeros([1, hiddenSize]);
  }

  step(
    x_t: Tensor,
    prev_h: Tensor,
    prev_c: Tensor
  ): { h: Tensor; c: Tensor; gates: { f: Tensor; i: Tensor; c_tilde: Tensor; o: Tensor } } {
    // 1. Forget gate: f_t = sigmoid(x_t * W_f + h_{t-1} * U_f + b_f)
    const f = x_t.matmul(this.W_f).add(prev_h.matmul(this.U_f)).add(this.b_f).sigmoid();

    // 2. Input gate: i_t = sigmoid(x_t * W_i + h_{t-1} * U_i + b_i)
    const i = x_t.matmul(this.W_i).add(prev_h.matmul(this.U_i)).add(this.b_i).sigmoid();

    // 3. Candidate: c_tilde = tanh(x_t * W_c + h_{t-1} * U_c + b_c)
    const c_tilde = x_t.matmul(this.W_c).add(prev_h.matmul(this.U_c)).add(this.b_c).tanh();

    // 4. Cell state: c_t = f_t * c_{t-1} + i_t * c_tilde
    const c = f.mul(prev_c).add(i.mul(c_tilde));

    // 5. Output gate: o_t = sigmoid(x_t * W_o + h_{t-1} * U_o + b_o)
    const o = x_t.matmul(this.W_o).add(prev_h.matmul(this.U_o)).add(this.b_o).sigmoid();

    // 6. Hidden state: h_t = o_t * tanh(c_t)
    const h = o.mul(c.tanh());

    return { h, c, gates: { f, i, c_tilde, o } };
  }

  unroll(sequence: { token: string; vector: number[] }[]): LSTMStepState[] {
    this.tracer.clear();
    const history: LSTMStepState[] = [];

    let h = Tensor.zeros([1, this.hiddenSize]);
    let c = Tensor.zeros([1, this.hiddenSize]);

    for (let t = 0; t < sequence.length; t++) {
      const item = sequence[t];
      const x = new Tensor([1, this.inputSize], item.vector);
      const res = this.step(x, h, c);

      h = res.h;
      c = res.c;

      const stepState: LSTMStepState = {
        t,
        inputToken: item.token,
        x_t: Array.from(x.data),
        f_gate: Array.from(res.gates.f.data),
        i_gate: Array.from(res.gates.i.data),
        c_tilde: Array.from(res.gates.c_tilde.data),
        c_t: Array.from(c.data),
        o_gate: Array.from(res.gates.o.data),
        h_t: Array.from(h.data)
      };

      history.push(stepState);

      this.tracer.record({
        layerId: `lstm_step_${t}`,
        layerName: `LSTM Step t=${t} ("${item.token}")`,
        operation: "Recurrent Cell State Transition",
        formula: "c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t; h_t = o_t ⊙ tanh(c_t)",
        inputShapes: [x.shape, h.shape, c.shape],
        outputShape: h.shape,
        tensorPreview: Array.from(h.data),
        pedagogicalInsight: `Step t=${t}: Forget gate retained ${(stepState.f_gate[0] * 100).toFixed(0)}% of prior cell memory while input gate injected ${(stepState.i_gate[0] * 100).toFixed(0)}% of new token information.`
      });
    }

    return history;
  }
}
