var z = Object.defineProperty;
var E = (b, t, s) => t in b ? z(b, t, { enumerable: !0, configurable: !0, writable: !0, value: s }) : b[t] = s;
var u = (b, t, s) => E(b, typeof t != "symbol" ? t + "" : t, s);
class c {
  constructor(t, s) {
    u(this, "data");
    u(this, "shape");
    u(this, "strides");
    u(this, "size");
    this.shape = [...t], this.size = t.reduce((e, o) => e * o, 1), this.strides = c.computeStrides(this.shape), s ? this.data = s instanceof Float32Array ? s : new Float32Array(s) : this.data = new Float32Array(this.size);
  }
  static computeStrides(t) {
    const s = new Array(t.length);
    let e = 1;
    for (let o = t.length - 1; o >= 0; o--)
      s[o] = e, e *= t[o];
    return s;
  }
  static zeros(t) {
    return new c(t);
  }
  static ones(t) {
    const s = new c(t);
    return s.data.fill(1), s;
  }
  static random(t, s = -1, e = 1) {
    const o = new c(t), a = e - s;
    for (let d = 0; d < o.size; d++)
      o.data[d] = s + Math.random() * a;
    return o;
  }
  static fromArray(t) {
    const s = [];
    let e = t;
    for (; Array.isArray(e); )
      s.push(e.length), e = e[0];
    const o = [];
    function a(d) {
      if (Array.isArray(d))
        for (const i of d) a(i);
      else
        o.push(Number(d));
    }
    return a(t), new c(s, o);
  }
  clone() {
    return new c(this.shape, new Float32Array(this.data));
  }
  getIndex(...t) {
    let s = 0;
    for (let e = 0; e < t.length; e++)
      s += t[e] * this.strides[e];
    return s;
  }
  get(...t) {
    return this.data[this.getIndex(...t)];
  }
  set(t, ...s) {
    this.data[this.getIndex(...s)] = t;
  }
  // --- Element-wise arithmetic ---
  add(t) {
    const s = new c(this.shape);
    if (typeof t == "number")
      for (let e = 0; e < this.size; e++) s.data[e] = this.data[e] + t;
    else
      for (let e = 0; e < this.size; e++) s.data[e] = this.data[e] + t.data[e];
    return s;
  }
  sub(t) {
    const s = new c(this.shape);
    if (typeof t == "number")
      for (let e = 0; e < this.size; e++) s.data[e] = this.data[e] - t;
    else
      for (let e = 0; e < this.size; e++) s.data[e] = this.data[e] - t.data[e];
    return s;
  }
  mul(t) {
    const s = new c(this.shape);
    if (typeof t == "number")
      for (let e = 0; e < this.size; e++) s.data[e] = this.data[e] * t;
    else
      for (let e = 0; e < this.size; e++) s.data[e] = this.data[e] * t.data[e];
    return s;
  }
  // --- Matrix Multiplication (2D) ---
  matmul(t) {
    if (this.shape.length !== 2 || t.shape.length !== 2)
      throw new Error(`matmul requires 2D tensors, got ${this.shape} and ${t.shape}`);
    const [s, e] = this.shape, [o, a] = t.shape;
    if (e !== o)
      throw new Error(`Incompatible matrix dims: [${s}, ${e}] x [${o}, ${a}]`);
    const d = new c([s, a]);
    for (let i = 0; i < s; i++)
      for (let n = 0; n < a; n++) {
        let r = 0;
        for (let l = 0; l < e; l++)
          r += this.get(i, l) * t.get(l, n);
        d.set(r, i, n);
      }
    return d;
  }
  transpose() {
    if (this.shape.length !== 2)
      throw new Error("transpose currently supports 2D tensors");
    const [t, s] = this.shape, e = new c([s, t]);
    for (let o = 0; o < t; o++)
      for (let a = 0; a < s; a++)
        e.set(this.get(o, a), a, o);
    return e;
  }
  // --- Activations ---
  relu() {
    const t = new c(this.shape);
    for (let s = 0; s < this.size; s++)
      t.data[s] = Math.max(0, this.data[s]);
    return t;
  }
  sigmoid() {
    const t = new c(this.shape);
    for (let s = 0; s < this.size; s++)
      t.data[s] = 1 / (1 + Math.exp(-this.data[s]));
    return t;
  }
  tanh() {
    const t = new c(this.shape);
    for (let s = 0; s < this.size; s++)
      t.data[s] = Math.tanh(this.data[s]);
    return t;
  }
  gelu() {
    const t = new c(this.shape), s = Math.sqrt(2 / Math.PI);
    for (let e = 0; e < this.size; e++) {
      const o = this.data[e];
      t.data[e] = 0.5 * o * (1 + Math.tanh(s * (o + 0.044715 * Math.pow(o, 3))));
    }
    return t;
  }
  softmax(t = -1) {
    const s = new c(this.shape);
    if (this.shape.length === 1) {
      let e = -1 / 0;
      for (let a = 0; a < this.size; a++) this.data[a] > e && (e = this.data[a]);
      let o = 0;
      for (let a = 0; a < this.size; a++)
        s.data[a] = Math.exp(this.data[a] - e), o += s.data[a];
      for (let a = 0; a < this.size; a++) s.data[a] /= o;
      return s;
    }
    if (this.shape.length === 2) {
      const [e, o] = this.shape;
      for (let a = 0; a < e; a++) {
        let d = -1 / 0;
        for (let n = 0; n < o; n++) {
          const r = this.get(a, n);
          r > d && (d = r);
        }
        let i = 0;
        for (let n = 0; n < o; n++) {
          const r = Math.exp(this.get(a, n) - d);
          s.set(r, a, n), i += r;
        }
        for (let n = 0; n < o; n++)
          s.set(s.get(a, n) / i, a, n);
      }
      return s;
    }
    throw new Error("softmax supports 1D or 2D tensors");
  }
  // --- 2D Convolution & Pooling ---
  /**
   * 2D Convolution on [H, W, InChannels] with Kernel [Kh, Kw, InChannels, OutChannels]
   */
  conv2d(t, s, e = 1, o = 0) {
    const [a, d, i] = this.shape, [n, r, l, h] = t.shape;
    if (i !== l)
      throw new Error(`Channel mismatch: input has ${i}, kernel expects ${l}`);
    const p = Math.floor((a - n + 2 * o) / e) + 1, f = Math.floor((d - r + 2 * o) / e) + 1, g = new c([p, f, h]);
    for (let m = 0; m < h; m++) {
      const x = s ? s.data[m] : 0;
      for (let w = 0; w < p; w++)
        for (let _ = 0; _ < f; _++) {
          let v = x;
          const C = w * e - o, y = _ * e - o;
          for (let S = 0; S < n; S++) {
            const W = C + S;
            if (!(W < 0 || W >= a))
              for (let k = 0; k < r; k++) {
                const M = y + k;
                if (!(M < 0 || M >= d))
                  for (let A = 0; A < i; A++)
                    v += this.get(W, M, A) * t.get(S, k, A, m);
              }
          }
          g.set(v, w, _, m);
        }
    }
    return g;
  }
  maxPool2d(t = 2, s = 2) {
    const [e, o, a] = this.shape, d = Math.floor((e - t) / s) + 1, i = Math.floor((o - t) / s) + 1, n = new c([d, i, a]);
    for (let r = 0; r < a; r++)
      for (let l = 0; l < d; l++)
        for (let h = 0; h < i; h++) {
          let p = -1 / 0;
          const f = l * s, g = h * s;
          for (let m = 0; m < t; m++)
            for (let x = 0; x < t; x++) {
              const w = this.get(f + m, g + x, r);
              w > p && (p = w);
            }
          n.set(p, l, h, r);
        }
    return n;
  }
  flatten() {
    return new c([this.size], this.data);
  }
  toArray() {
    if (this.shape.length === 1)
      return Array.from(this.data);
    if (this.shape.length === 2) {
      const [t, s] = this.shape, e = [];
      for (let o = 0; o < t; o++) {
        const a = [];
        for (let d = 0; d < s; d++) a.push(this.get(o, d));
        e.push(a);
      }
      return e;
    }
    return Array.from(this.data);
  }
}
class $ {
  constructor() {
    u(this, "steps", []);
  }
  record(t) {
    this.steps.push({
      stepIndex: this.steps.length,
      ...t
    });
  }
  clear() {
    this.steps = [];
  }
  getTraceSummary() {
    return this.steps.map(
      (t) => `[Step ${t.stepIndex}] ${t.layerName} (${t.operation}): ${t.formula} -> Shape [${t.outputShape.join(", ")}]`
    ).join(`
`);
  }
}
class I {
  constructor(t) {
    u(this, "weights", []);
    u(this, "biases", []);
    u(this, "activations", []);
    u(this, "tracer", new $());
    this.config = t, this.activations = t.activations;
    for (let s = 0; s < t.layerSizes.length - 1; s++) {
      const e = t.layerSizes[s], o = t.layerSizes[s + 1], a = Math.sqrt(2 / e);
      this.weights.push(c.random([e, o], -a, a)), this.biases.push(c.zeros([1, o]));
    }
  }
  forward(t, s = !1) {
    s && this.tracer.clear();
    let e = t;
    for (let o = 0; o < this.weights.length; o++) {
      const a = this.weights[o], d = this.biases[o], i = this.activations[o], n = e.matmul(a).add(d);
      let r = n;
      i === "relu" ? r = n.relu() : i === "sigmoid" ? r = n.sigmoid() : i === "tanh" && (r = n.tanh()), s && this.tracer.record({
        layerId: `layer_${o + 1}`,
        layerName: `Hidden Layer ${o + 1} (${i.toUpperCase()})`,
        operation: "Dense Matmul + Bias + Activation",
        formula: `a^[${o + 1}] = ${i}(W^[${o + 1}] * a^[${o}] + b^[${o + 1}])`,
        inputShapes: [e.shape, a.shape],
        outputShape: r.shape,
        tensorPreview: Array.from(r.data.slice(0, 4)),
        pedagogicalInsight: `Layer ${o + 1} transforms ${a.shape[0]} inputs into ${a.shape[1]} linear combinations, activated by ${i}.`
      }), e = r;
    }
    return e;
  }
  // Train a single epoch on a dataset X, Y
  trainStep(t, s) {
    let e = 0;
    const o = this.config.learningRate;
    for (let a = 0; a < t.length; a++) {
      const d = c.fromArray([t[a]]), i = s[a], n = [d], r = [];
      let l = d;
      for (let g = 0; g < this.weights.length; g++) {
        const m = l.matmul(this.weights[g]).add(this.biases[g]);
        r.push(m);
        const x = this.activations[g];
        x === "relu" ? l = m.relu() : x === "sigmoid" ? l = m.sigmoid() : x === "tanh" && (l = m.tanh()), n.push(l);
      }
      const p = l.data[0] - i[0];
      e += 0.5 * p * p;
      let f = new c([1, 1], [p]);
      for (let g = this.weights.length - 1; g >= 0; g--) {
        const m = r[g], x = this.activations[g], w = n[g], _ = new c(m.shape);
        for (let y = 0; y < m.size; y++)
          if (x === "relu") _.data[y] = m.data[y] > 0 ? 1 : 0;
          else if (x === "sigmoid") {
            const S = 1 / (1 + Math.exp(-m.data[y]));
            _.data[y] = S * (1 - S);
          } else if (x === "tanh") {
            const S = Math.tanh(m.data[y]);
            _.data[y] = 1 - S * S;
          } else
            _.data[y] = 1;
        const v = f.mul(_), C = w.transpose().matmul(v);
        for (let y = 0; y < this.weights[g].size; y++)
          this.weights[g].data[y] -= o * C.data[y];
        for (let y = 0; y < this.biases[g].size; y++)
          this.biases[g].data[y] -= o * v.data[y];
        g > 0 && (f = v.matmul(this.weights[g].transpose()));
      }
    }
    return { loss: e / t.length };
  }
  /**
   * Evaluate a 2D grid of points for boundary visualization (e.g., XOR, Circles, Moons)
   */
  evaluateGrid(t = 30, s = 4) {
    const e = [], o = s * 2 / t, a = [], d = [];
    for (let i = 0; i < t; i++) {
      const n = [], r = s - i * o;
      d.push(r);
      for (let l = 0; l < t; l++) {
        const h = -s + l * o;
        i === 0 && a.push(h);
        const p = this.forward(new c([1, 2], [h, r]));
        n.push(p.data[0]);
      }
      e.push(n);
    }
    return { grid: e, xRange: a, yRange: d };
  }
}
class B {
  constructor() {
    u(this, "tracer", new $());
    // Assignment 3 exact architecture:
    // Conv2D(32, 3x3) -> MaxPool(2x2) -> Conv2D(64, 3x3) -> MaxPool(2x2) -> Conv2D(64, 3x3) -> Flatten -> Dense(64) -> Dense(10, softmax)
    u(this, "kernel1");
    u(this, "bias1");
    u(this, "kernel2");
    u(this, "bias2");
    u(this, "kernel3");
    u(this, "bias3");
    u(this, "dense1_W");
    u(this, "dense1_B");
    u(this, "dense2_W");
    u(this, "dense2_B");
    this.kernel1 = c.random([3, 3, 1, 32], -0.2, 0.2), this.bias1 = c.zeros([32]), this.kernel2 = c.random([3, 3, 32, 64], -0.15, 0.15), this.bias2 = c.zeros([64]), this.kernel3 = c.random([3, 3, 64, 64], -0.15, 0.15), this.bias3 = c.zeros([64]), this.dense1_W = c.random([576, 64], -0.1, 0.1), this.dense1_B = c.zeros([1, 64]), this.dense2_W = c.random([64, 10], -0.1, 0.1), this.dense2_B = c.zeros([1, 10]);
  }
  forward(t, s = !0) {
    s && this.tracer.clear();
    const e = t.conv2d(this.kernel1, this.bias1, 1, 0).relu();
    s && this.tracer.record({
      layerId: "conv1",
      layerName: "Conv2D (32 filters, 3x3)",
      operation: "2D Convolution + ReLU",
      formula: "H_out = (28 - 3 + 0)/1 + 1 = 26; Params = (3*3*1 + 1)*32 = 320",
      inputShapes: [t.shape, this.kernel1.shape],
      outputShape: e.shape,
      tensorPreview: Array.from(e.data.slice(0, 5)),
      fullOutput: e,
      pedagogicalInsight: "Extracts 32 low-level edge and texture feature maps from the raw 28x28 pixel grid.",
      metadata: { filters: 32, kernel: "3x3", params: 320 }
    });
    const o = e.maxPool2d(2, 2);
    s && this.tracer.record({
      layerId: "pool1",
      layerName: "MaxPooling2D (2x2)",
      operation: "Spatial Downsampling",
      formula: "H_out = floor(26/2) = 13; W_out = 13",
      inputShapes: [e.shape],
      outputShape: o.shape,
      tensorPreview: Array.from(o.data.slice(0, 5)),
      fullOutput: o,
      pedagogicalInsight: "Preserves the most salient local features while reducing spatial dimensions by 75%."
    });
    const a = o.conv2d(this.kernel2, this.bias2, 1, 0).relu();
    s && this.tracer.record({
      layerId: "conv2",
      layerName: "Conv2D (64 filters, 3x3)",
      operation: "2D Convolution + ReLU",
      formula: "H_out = (13 - 3 + 0)/1 + 1 = 11; Params = (3*3*32 + 1)*64 = 18,496",
      inputShapes: [o.shape, this.kernel2.shape],
      outputShape: a.shape,
      tensorPreview: Array.from(a.data.slice(0, 5)),
      fullOutput: a,
      pedagogicalInsight: "Combines local features into mid-level parts (corners, loops, strokes)."
    });
    const d = a.maxPool2d(2, 2);
    s && this.tracer.record({
      layerId: "pool2",
      layerName: "MaxPooling2D (2x2)",
      operation: "Spatial Downsampling",
      formula: "H_out = floor(11/2) = 5; W_out = 5",
      inputShapes: [a.shape],
      outputShape: d.shape,
      tensorPreview: Array.from(d.data.slice(0, 5)),
      fullOutput: d,
      pedagogicalInsight: "Further downsamples to 5x5 feature grids."
    });
    const i = d.conv2d(this.kernel3, this.bias3, 1, 0).relu();
    s && this.tracer.record({
      layerId: "conv3",
      layerName: "Conv2D (64 filters, 3x3)",
      operation: "2D Convolution + ReLU",
      formula: "H_out = (5 - 3 + 0)/1 + 1 = 3; Params = (3*3*64 + 1)*64 = 36,928",
      inputShapes: [d.shape, this.kernel3.shape],
      outputShape: i.shape,
      tensorPreview: Array.from(i.data.slice(0, 5)),
      fullOutput: i,
      pedagogicalInsight: "High-level digit shape representations."
    });
    const n = i.flatten(), h = new c([1, 576], n.data).matmul(this.dense1_W).add(this.dense1_B).relu().matmul(this.dense2_W).add(this.dense2_B), p = h.softmax(-1), f = Array.from(p.data);
    let g = 0, m = -1;
    for (let x = 0; x < 10; x++)
      f[x] > m && (m = f[x], g = x);
    return s && this.tracer.record({
      layerId: "digit_probs",
      layerName: "Softmax Classification Head",
      operation: "Softmax(logits) -> Argmax",
      formula: "P(digit=k) = exp(z_k) / sum(exp(z)); Predicted = argmax_k(P)",
      inputShapes: [h.shape],
      outputShape: [10],
      tensorPreview: f,
      pedagogicalInsight: `Final digit classification with top prediction ${g} (${(m * 100).toFixed(1)}%).`
    }), {
      probabilities: f,
      predictedDigit: g,
      conv1Out: e,
      pool1Out: o,
      conv2Out: a,
      pool2Out: d,
      conv3Out: i
    };
  }
  loadWeightsFromJSON(t) {
    t.kernel1 && (this.kernel1 = new c([3, 3, 1, 32], t.kernel1)), t.bias1 && (this.bias1 = new c([32], t.bias1)), t.dense2_W && (this.dense2_W = new c([64, 10], t.dense2_W)), t.dense2_B && (this.dense2_B = new c([1, 10], t.dense2_B));
  }
}
class P {
  constructor(t = 4, s = 4) {
    u(this, "hiddenSize");
    u(this, "inputSize");
    u(this, "tracer", new $());
    // Gates: f, i, c, o (combined or separate)
    u(this, "W_f");
    u(this, "U_f");
    u(this, "b_f");
    u(this, "W_i");
    u(this, "U_i");
    u(this, "b_i");
    u(this, "W_c");
    u(this, "U_c");
    u(this, "b_c");
    u(this, "W_o");
    u(this, "U_o");
    u(this, "b_o");
    this.inputSize = t, this.hiddenSize = s;
    const e = 0.5;
    this.W_f = c.random([t, s], -e, e), this.U_f = c.random([s, s], -e, e), this.b_f = c.ones([1, s]), this.W_i = c.random([t, s], -e, e), this.U_i = c.random([s, s], -e, e), this.b_i = c.zeros([1, s]), this.W_c = c.random([t, s], -e, e), this.U_c = c.random([s, s], -e, e), this.b_c = c.zeros([1, s]), this.W_o = c.random([t, s], -e, e), this.U_o = c.random([s, s], -e, e), this.b_o = c.zeros([1, s]);
  }
  step(t, s, e) {
    const o = t.matmul(this.W_f).add(s.matmul(this.U_f)).add(this.b_f).sigmoid(), a = t.matmul(this.W_i).add(s.matmul(this.U_i)).add(this.b_i).sigmoid(), d = t.matmul(this.W_c).add(s.matmul(this.U_c)).add(this.b_c).tanh(), i = o.mul(e).add(a.mul(d)), n = t.matmul(this.W_o).add(s.matmul(this.U_o)).add(this.b_o).sigmoid();
    return { h: n.mul(i.tanh()), c: i, gates: { f: o, i: a, c_tilde: d, o: n } };
  }
  unroll(t) {
    this.tracer.clear();
    const s = [];
    let e = c.zeros([1, this.hiddenSize]), o = c.zeros([1, this.hiddenSize]);
    for (let a = 0; a < t.length; a++) {
      const d = t[a], i = new c([1, this.inputSize], d.vector), n = this.step(i, e, o);
      e = n.h, o = n.c;
      const r = {
        t: a,
        inputToken: d.token,
        x_t: Array.from(i.data),
        f_gate: Array.from(n.gates.f.data),
        i_gate: Array.from(n.gates.i.data),
        c_tilde: Array.from(n.gates.c_tilde.data),
        c_t: Array.from(o.data),
        o_gate: Array.from(n.gates.o.data),
        h_t: Array.from(e.data)
      };
      s.push(r), this.tracer.record({
        layerId: `lstm_step_${a}`,
        layerName: `LSTM Step t=${a} ("${d.token}")`,
        operation: "Recurrent Cell State Transition",
        formula: "c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t; h_t = o_t ⊙ tanh(c_t)",
        inputShapes: [i.shape, e.shape, o.shape],
        outputShape: e.shape,
        tensorPreview: Array.from(e.data),
        pedagogicalInsight: `Step t=${a}: Forget gate retained ${(r.f_gate[0] * 100).toFixed(0)}% of prior cell memory while input gate injected ${(r.i_gate[0] * 100).toFixed(0)}% of new token information.`
      });
    }
    return s;
  }
}
class L {
  constructor(t = 8, s = 2) {
    u(this, "d_model");
    u(this, "d_k");
    u(this, "numHeads");
    u(this, "tracer", new $());
    u(this, "W_q");
    u(this, "W_k");
    u(this, "W_v");
    u(this, "W_o");
    this.d_model = t, this.numHeads = s, this.d_k = Math.floor(t / s);
    const e = Math.sqrt(2 / t);
    this.W_q = c.random([t, t], -e, e), this.W_k = c.random([t, t], -e, e), this.W_v = c.random([t, t], -e, e), this.W_o = c.random([t, t], -e, e);
  }
  /**
   * Forward pass over token embeddings matrix X of shape [seqLen, d_model]
   */
  forward(t, s = !0, e = !0) {
    e && this.tracer.clear();
    const o = t.shape[0], a = t.matmul(this.W_q), d = t.matmul(this.W_k), i = t.matmul(this.W_v), n = 1 / Math.sqrt(this.d_model), r = d.transpose(), l = a.matmul(r).mul(n);
    if (s)
      for (let m = 0; m < o; m++)
        for (let x = m + 1; x < o; x++)
          l.set(-1e9, m, x);
    const h = l.softmax(-1), f = h.matmul(i).matmul(this.W_o);
    e && this.tracer.record({
      layerId: "self_attention",
      layerName: `Self-Attention (d_model=${this.d_model}, seqLen=${o})`,
      operation: "Scaled Dot-Product Attention",
      formula: "Attention(Q,K,V) = softmax(Q K^T / √d_k + Mask) V",
      inputShapes: [t.shape, this.W_q.shape],
      outputShape: f.shape,
      tensorPreview: Array.from(h.data.slice(0, 6)),
      pedagogicalInsight: `Calculated ${o}x${o} pairwise attention affinity matrix; tokens dynamically aggregate information from allowed preceding contexts.`
    });
    const g = [];
    for (let m = 0; m < o; m++) {
      const x = [];
      for (let w = 0; w < o; w++)
        x.push(h.get(m, w));
      g.push(x);
    }
    return {
      attentionWeights: g,
      output: f
    };
  }
}
class N {
  constructor(t = 10, s = 2) {
    u(this, "tracer", new $());
    // Encoder: [D_in, 16] -> [16, 2] (Latent)
    u(this, "W_enc1");
    u(this, "b_enc1");
    u(this, "W_enc2");
    u(this, "b_enc2");
    // Decoder: [2, 16] -> [16, D_in] (Reconstruction)
    u(this, "W_dec1");
    u(this, "b_dec1");
    u(this, "W_dec2");
    u(this, "b_dec2");
    this.inDim = t, this.latentDim = s;
    const e = 0.3;
    this.W_enc1 = c.random([t, 16], -e, e), this.b_enc1 = c.zeros([1, 16]), this.W_enc2 = c.random([16, s], -e, e), this.b_enc2 = c.zeros([1, s]), this.W_dec1 = c.random([s, 16], -e, e), this.b_dec1 = c.zeros([1, 16]), this.W_dec2 = c.random([16, t], -e, e), this.b_dec2 = c.zeros([1, t]);
  }
  encode(t) {
    return t.matmul(this.W_enc1).add(this.b_enc1).relu().matmul(this.W_enc2).add(this.b_enc2);
  }
  decode(t) {
    return t.matmul(this.W_dec1).add(this.b_dec1).relu().matmul(this.W_dec2).add(this.b_dec2).sigmoid();
  }
  forward(t, s = !0) {
    s && this.tracer.clear();
    const e = this.encode(t), o = this.decode(e);
    let a = 0;
    for (let d = 0; d < t.size; d++) {
      const i = t.data[d] - o.data[d];
      a += i * i;
    }
    return a /= t.size, s && this.tracer.record({
      layerId: "latent_bottleneck",
      layerName: `Latent Space (dim=${this.latentDim})`,
      operation: "Nonlinear Dimensionality Compression",
      formula: "z = W_2 * relu(W_1 x + b_1) + b_2",
      inputShapes: [t.shape],
      outputShape: e.shape,
      tensorPreview: Array.from(e.data),
      pedagogicalInsight: `Compressed input from ${this.inDim} dimensions to ${this.latentDim} latent coordinates with MSE=${a.toFixed(4)}.`
    }), { latent: e, reconstructed: o, mse: a };
  }
}
class F {
  static renderNetwork(t, s = 800, e = 360, o = -1) {
    const a = t.length, d = s / (a + 1), i = [];
    t.forEach((r, l) => {
      const h = (l + 1) * d, p = Math.min(r.outShape[r.outShape.length - 1] || 4, 8), f = e / (p + 1), g = [];
      for (let m = 0; m < p; m++)
        g.push({
          layerIndex: l,
          nodeIndex: m,
          x: h,
          y: (m + 1) * f,
          label: `${r.name} [${m}]`
        });
      i.push(g);
    });
    let n = `<svg viewBox="0 0 ${s} ${e}" width="100%" height="${e}" style="background:#0b1329; border-radius:12px; font-family:system-ui, -apple-system, sans-serif;">`;
    n += `<defs>
      <linearGradient id="edgeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="#0F8B8D" stop-opacity="0.4"/>
        <stop offset="100%" stop-color="#6B2D7B" stop-opacity="0.6"/>
      </linearGradient>
      <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
        <feGaussianBlur stdDeviation="3" result="blur" />
        <feComposite in="SourceGraphic" in2="blur" operator="over" />
      </filter>
    </defs>`;
    for (let r = 0; r < i.length - 1; r++) {
      const l = i[r], h = i[r + 1];
      for (const p of l)
        for (const f of h)
          n += `<line x1="${p.x}" y1="${p.y}" x2="${f.x}" y2="${f.y}" stroke="url(#edgeGrad)" stroke-width="1.2"/>`;
    }
    return t.forEach((r, l) => {
      const h = (l + 1) * d;
      n += `<text x="${h}" y="24" fill="${l === o ? "#FFD700" : "#E2E8F0"}" font-size="12" font-weight="600" text-anchor="middle">${r.name}</text>`, n += `<text x="${h}" y="40" fill="#94A3B8" font-size="10" font-family="monospace" text-anchor="middle">[${r.outShape.join("×")}]</text>`, r.paramsCount > 0 && (n += `<text x="${h}" y="${e - 12}" fill="#38BDF8" font-size="10" font-family="monospace" text-anchor="middle">${r.paramsCount.toLocaleString()} params</text>`);
    }), i.forEach((r, l) => {
      const h = l === o, p = h ? "#FFD700" : l === 0 ? "#0F8B8D" : l === i.length - 1 ? "#C8102E" : "#6B2D7B";
      for (const f of r)
        n += `<circle cx="${f.x}" cy="${f.y}" r="9" fill="${p}" stroke="#ffffff" stroke-width="1.5" ${h ? 'filter="url(#glow)"' : ""}/>`;
    }), n += "</svg>", n;
  }
}
class D {
  /**
   * Render a 2D scalar grid as a color heatmap (Decision Boundary)
   */
  static renderDecisionBoundary(t, s, e) {
    const o = t.getContext("2d");
    if (!o) return;
    const a = s.length, d = s[0].length, i = t.width / d, n = t.height / a;
    for (let r = 0; r < a; r++)
      for (let l = 0; l < d; l++) {
        const h = s[r][l], p = Math.floor(h * 200 + (1 - h) * 20), f = Math.floor((1 - Math.abs(h - 0.5) * 2) * 150), g = Math.floor((1 - h) * 200 + h * 30);
        o.fillStyle = `rgb(${p}, ${f}, ${g})`, o.fillRect(l * i, r * n, i + 1, n + 1);
      }
    if (e)
      for (const r of e) {
        const l = (r.x + 4) / 8 * t.width, h = (4 - r.y) / 8 * t.height;
        o.beginPath(), o.arc(l, h, 5, 0, Math.PI * 2), o.fillStyle = r.label === 1 ? "#FFD700" : "#FFFFFF", o.strokeStyle = "#000000", o.lineWidth = 1.5, o.fill(), o.stroke();
      }
  }
  /**
   * Render Attention Weight Matrix [seqLen x seqLen] with token labels
   */
  static renderAttentionMatrix(t, s, e) {
    const o = t.getContext("2d");
    if (!o) return;
    const a = s.length, d = 70, i = 70, n = t.width - d - 10, r = t.height - i - 10, l = n / a, h = r / a;
    o.clearRect(0, 0, t.width, t.height), o.fillStyle = "#0F172A", o.fillRect(0, 0, t.width, t.height);
    for (let p = 0; p < a; p++)
      for (let f = 0; f < a; f++) {
        const g = s[p][f];
        o.fillStyle = `rgba(15, 139, 141, ${Math.max(0.08, g)})`, o.fillRect(d + f * l, i + p * h, l - 1, h - 1), l > 30 && (o.fillStyle = g > 0.4 ? "#FFFFFF" : "#94A3B8", o.font = "10px monospace", o.textAlign = "center", o.textBaseline = "middle", o.fillText(
          g.toFixed(2),
          d + f * l + l / 2,
          i + p * h + h / 2
        ));
      }
    o.fillStyle = "#E2E8F0", o.font = "12px sans-serif", o.textAlign = "right", o.textBaseline = "middle";
    for (let p = 0; p < a; p++) {
      const f = e[p] || `t_${p}`;
      o.fillText(f, d - 8, i + p * h + h / 2);
    }
    o.textAlign = "center", o.textBaseline = "bottom";
    for (let p = 0; p < a; p++) {
      const f = e[p] || `t_${p}`;
      o.fillText(f, d + p * l + l / 2, i - 8);
    }
  }
}
class H extends HTMLElement {
  constructor() {
    super(...arguments);
    u(this, "container", null);
  }
  static get observedAttributes() {
    return ["model", "dataset", "weights", "tokens", "sequence"];
  }
  connectedCallback() {
    this.render();
  }
  attributeChangedCallback() {
    this.render();
  }
  render() {
    const s = this.getAttribute("model") || "mlp";
    this.innerHTML = `
      <div class="omni-sim-card" style="font-family:system-ui, -apple-system, sans-serif; background:#0F172A; color:#F8FAFC; border:1px solid #334155; border-radius:12px; padding:20px; margin:16px 0; box-shadow:0 10px 25px -5px rgba(0,0,0,0.3);">
        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #1E293B; padding-bottom:12px; margin-bottom:16px;">
          <div>
            <span style="background:#0F8B8D; color:#fff; font-size:11px; font-weight:700; text-transform:uppercase; padding:3px 8px; border-radius:4px; letter-spacing:0.5px;">OmniNeuralSim SOTA</span>
            <h3 style="margin:6px 0 0 0; font-size:1.25rem; color:#FFFFFF;">Model Simulator: <span style="color:#38BDF8;">${s.toUpperCase()}</span></h3>
          </div>
          <div style="display:flex; gap:8px;">
            <button class="btn-step" style="background:#1E293B; color:#F8FAFC; border:1px solid #475569; border-radius:6px; padding:6px 12px; font-size:12px; cursor:pointer; transition:all 0.2s;">Step Forward ⏭</button>
            <button class="btn-reset" style="background:#C8102E; color:#FFFFFF; border:none; border-radius:6px; padding:6px 12px; font-size:12px; cursor:pointer;">Reset</button>
          </div>
        </div>

        <div class="sim-viewport" style="display:grid; grid-template-columns: 1fr 1fr; gap:16px;">
          <div class="sim-diagram-panel" style="background:#020617; border-radius:8px; padding:12px; border:1px solid #1E293B;">
            <div style="font-size:12px; font-weight:600; color:#94A3B8; margin-bottom:8px;">MODEL ARCHITECTURE & DATA FLOW</div>
            <div class="diagram-host"></div>
          </div>
          <div class="sim-vis-panel" style="background:#020617; border-radius:8px; padding:12px; border:1px solid #1E293B; display:flex; flex-direction:column; align-items:center;">
            <div style="font-size:12px; font-weight:600; color:#94A3B8; margin-bottom:8px; width:100%;">LIVE ACTIVATIONS & INFERENCE</div>
            <canvas class="sim-canvas" width="320" height="260" style="border-radius:6px; background:#000;"></canvas>
            <div class="sim-metrics" style="width:100%; margin-top:12px; font-size:12px; font-family:monospace; color:#E2E8F0;"></div>
          </div>
        </div>

        <div class="sim-trace-panel" style="margin-top:16px; background:#020617; border:1px solid #1E293B; border-radius:8px; padding:12px;">
          <div style="font-size:11px; font-weight:700; color:#38BDF8; margin-bottom:4px;">PEDAGOGICAL EXECUTION TRACE:</div>
          <div class="trace-output" style="font-size:12px; font-family:monospace; color:#A5B4FC; line-height:1.5;"></div>
        </div>
      </div>
    `, this.initModel(s);
  }
  initModel(s) {
    const e = this.querySelector(".sim-canvas"), o = this.querySelector(".diagram-host"), a = this.querySelector(".trace-output");
    this.querySelector(".sim-metrics");
    const d = this.querySelector(".btn-step");
    if (s === "mlp") {
      const i = new I({
        layerSizes: [2, 6, 4, 1],
        activations: ["tanh", "tanh", "sigmoid"],
        learningRate: 0.1
      }), n = [[-2, -2], [-2, 2], [2, -2], [2, 2]], r = [[0], [1], [1], [0]], l = () => {
        const { grid: h } = i.evaluateGrid(30, 4);
        D.renderDecisionBoundary(
          e,
          h,
          n.map((p, f) => ({ x: p[0], y: p[1], label: r[f][0] }))
        ), o.innerHTML = F.renderNetwork([
          { id: "in", name: "Input [x1, x2]", type: "dense", inShape: [2], outShape: [2], paramsCount: 0 },
          { id: "h1", name: "Hidden 1 (Tanh)", type: "dense", inShape: [2], outShape: [6], paramsCount: 18 },
          { id: "h2", name: "Hidden 2 (Tanh)", type: "dense", inShape: [6], outShape: [4], paramsCount: 28 },
          { id: "out", name: "Output (Sigmoid)", type: "dense", inShape: [4], outShape: [1], paramsCount: 5 }
        ], 380, 220), i.forward(new c([1, 2], [1.5, -1.5]), !0), a.innerHTML = i.tracer.steps.map((p) => `• <b>${p.layerName}:</b> ${p.formula} → [${p.outputShape}]`).join("<br>");
      };
      d.onclick = () => {
        for (let h = 0; h < 25; h++) i.trainStep(n, r);
        l();
      }, l();
    } else if (s === "cnn") {
      const i = new B(), n = c.zeros([28, 28, 1]);
      for (let h = 4; h < 24; h++) n.set(1, h, 14, 0);
      const r = i.forward(n, !0);
      o.innerHTML = F.renderNetwork([
        { id: "conv1", name: "Conv2D (3x3)", type: "conv2d", inShape: [28, 28, 1], outShape: [26, 26, 32], paramsCount: 320 },
        { id: "pool1", name: "MaxPool", type: "maxpool2d", inShape: [26, 26, 32], outShape: [13, 13, 32], paramsCount: 0 },
        { id: "conv2", name: "Conv2D (3x3)", type: "conv2d", inShape: [13, 13, 32], outShape: [11, 11, 64], paramsCount: 18496 },
        { id: "dense", name: "Dense Head", type: "dense", inShape: [576], outShape: [10], paramsCount: 37578 }
      ], 380, 220);
      const l = e.getContext("2d");
      l && (l.fillStyle = "#000", l.fillRect(0, 0, e.width, e.height), l.fillStyle = "#38BDF8", l.font = "14px monospace", l.fillText(`Top Predicted Digit: ${r.predictedDigit}`, 20, 40), l.fillText(`Confidence: ${(r.probabilities[r.predictedDigit] * 100).toFixed(1)}%`, 20, 65), l.fillText("Conv1 Output Shape: 26×26×32", 20, 100), l.fillText("Layer 1 Parameters: 320", 20, 125)), a.innerHTML = i.tracer.steps.map((h) => `• <b>${h.layerName}:</b> ${h.formula} → ${h.pedagogicalInsight}`).join("<br>");
    } else if (s === "transformer" || s === "attention") {
      const i = new L(8, 2), n = ["The", "neural", "network", "attends", "to", "tokens"], r = c.random([n.length, 8], -0.5, 0.5), l = i.forward(r, !0, !0);
      D.renderAttentionMatrix(e, l.attentionWeights, n), o.innerHTML = F.renderNetwork([
        { id: "emb", name: "Token Embeddings", type: "embedding", inShape: [6], outShape: [6, 8], paramsCount: 48 },
        { id: "qkv", name: "Q, K, V Projections", type: "dense", inShape: [6, 8], outShape: [6, 8], paramsCount: 192 },
        { id: "attn", name: "Attention Matrix", type: "self_attention", inShape: [6, 8], outShape: [6, 6], paramsCount: 0 },
        { id: "out", name: "Linear Projection", type: "dense", inShape: [6, 8], outShape: [6, 8], paramsCount: 64 }
      ], 380, 220), a.innerHTML = i.tracer.steps.map((h) => `• <b>${h.layerName}:</b> ${h.formula} → ${h.pedagogicalInsight}`).join("<br>");
    } else if (s === "lstm" || s === "rnn") {
      const i = new P(4, 4), r = ["Natural", "Language", "Processing", "Recurrent"].map((p) => ({ token: p, vector: [0.5, -0.2, 0.8, -0.1] })), l = i.unroll(r);
      o.innerHTML = F.renderNetwork([
        { id: "in", name: "x_t Token Vector", type: "dense", inShape: [4], outShape: [4], paramsCount: 0 },
        { id: "gates", name: "Gates [f, i, c, o]", type: "lstm_cell", inShape: [4], outShape: [16], paramsCount: 144 },
        { id: "cell", name: "c_t Memory Cell", type: "lstm_cell", inShape: [4], outShape: [4], paramsCount: 0 },
        { id: "h", name: "h_t Hidden State", type: "lstm_cell", inShape: [4], outShape: [4], paramsCount: 0 }
      ], 380, 220);
      const h = e.getContext("2d");
      h && (h.fillStyle = "#000", h.fillRect(0, 0, e.width, e.height), h.fillStyle = "#38BDF8", h.font = "12px monospace", h.fillText("LSTM GATE DYNAMICS:", 15, 25), l.forEach((p, f) => {
        h.fillText(`t=${p.t} ("${p.inputToken}"): f=${p.f_gate[0].toFixed(2)}, i=${p.i_gate[0].toFixed(2)}, o=${p.o_gate[0].toFixed(2)}`, 15, 55 + f * 30);
      })), a.innerHTML = i.tracer.steps.map((p) => `• <b>${p.layerName}:</b> ${p.formula} → ${p.pedagogicalInsight}`).join("<br>");
    }
  }
}
typeof window < "u" && !customElements.get("neural-sim") && customElements.define("neural-sim", H);
export {
  N as AutoencoderModel,
  B as CNNModel,
  D as CanvasVisualizer,
  $ as ExecutionTracer,
  P as LSTMModel,
  I as MLP,
  H as NeuralSimElement,
  F as SVGDiagramRenderer,
  c as Tensor,
  L as TransformerAttention
};
//# sourceMappingURL=omni-neural-sim.es.js.map
