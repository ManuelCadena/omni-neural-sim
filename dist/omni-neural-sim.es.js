var D = Object.defineProperty;
var W = (k, e, o) => e in k ? D(k, e, { enumerable: !0, configurable: !0, writable: !0, value: o }) : k[e] = o;
var b = (k, e, o) => W(k, typeof e != "symbol" ? e + "" : e, o);
class u {
  constructor(e, o) {
    b(this, "data");
    b(this, "shape");
    b(this, "strides");
    b(this, "size");
    this.shape = [...e], this.size = e.reduce((t, n) => t * n, 1), this.strides = u.computeStrides(this.shape), o ? this.data = o instanceof Float32Array ? o : new Float32Array(o) : this.data = new Float32Array(this.size);
  }
  static computeStrides(e) {
    const o = new Array(e.length);
    let t = 1;
    for (let n = e.length - 1; n >= 0; n--)
      o[n] = t, t *= e[n];
    return o;
  }
  static zeros(e) {
    return new u(e);
  }
  static ones(e) {
    const o = new u(e);
    return o.data.fill(1), o;
  }
  static random(e, o = -1, t = 1) {
    const n = new u(e), s = t - o;
    for (let a = 0; a < n.size; a++)
      n.data[a] = o + Math.random() * s;
    return n;
  }
  static fromArray(e) {
    const o = [];
    let t = e;
    for (; Array.isArray(t); )
      o.push(t.length), t = t[0];
    const n = [];
    function s(a) {
      if (Array.isArray(a))
        for (const p of a) s(p);
      else
        n.push(Number(a));
    }
    return s(e), new u(o, n);
  }
  clone() {
    return new u(this.shape, new Float32Array(this.data));
  }
  getIndex(...e) {
    let o = 0;
    for (let t = 0; t < e.length; t++)
      o += e[t] * this.strides[t];
    return o;
  }
  get(...e) {
    return this.data[this.getIndex(...e)];
  }
  set(e, ...o) {
    this.data[this.getIndex(...o)] = e;
  }
  // --- Element-wise arithmetic ---
  add(e) {
    const o = new u(this.shape);
    if (typeof e == "number")
      for (let t = 0; t < this.size; t++) o.data[t] = this.data[t] + e;
    else
      for (let t = 0; t < this.size; t++) o.data[t] = this.data[t] + e.data[t];
    return o;
  }
  sub(e) {
    const o = new u(this.shape);
    if (typeof e == "number")
      for (let t = 0; t < this.size; t++) o.data[t] = this.data[t] - e;
    else
      for (let t = 0; t < this.size; t++) o.data[t] = this.data[t] - e.data[t];
    return o;
  }
  mul(e) {
    const o = new u(this.shape);
    if (typeof e == "number")
      for (let t = 0; t < this.size; t++) o.data[t] = this.data[t] * e;
    else
      for (let t = 0; t < this.size; t++) o.data[t] = this.data[t] * e.data[t];
    return o;
  }
  // --- Matrix Multiplication (2D) ---
  matmul(e) {
    if (this.shape.length !== 2 || e.shape.length !== 2)
      throw new Error(`matmul requires 2D tensors, got ${this.shape} and ${e.shape}`);
    const [o, t] = this.shape, [n, s] = e.shape;
    if (t !== n)
      throw new Error(`Incompatible matrix dims: [${o}, ${t}] x [${n}, ${s}]`);
    const a = new u([o, s]);
    for (let p = 0; p < o; p++)
      for (let l = 0; l < s; l++) {
        let h = 0;
        for (let m = 0; m < t; m++)
          h += this.get(p, m) * e.get(m, l);
        a.set(h, p, l);
      }
    return a;
  }
  transpose() {
    if (this.shape.length !== 2)
      throw new Error("transpose currently supports 2D tensors");
    const [e, o] = this.shape, t = new u([o, e]);
    for (let n = 0; n < e; n++)
      for (let s = 0; s < o; s++)
        t.set(this.get(n, s), s, n);
    return t;
  }
  // --- Activations ---
  relu() {
    const e = new u(this.shape);
    for (let o = 0; o < this.size; o++)
      e.data[o] = Math.max(0, this.data[o]);
    return e;
  }
  sigmoid() {
    const e = new u(this.shape);
    for (let o = 0; o < this.size; o++)
      e.data[o] = 1 / (1 + Math.exp(-this.data[o]));
    return e;
  }
  tanh() {
    const e = new u(this.shape);
    for (let o = 0; o < this.size; o++)
      e.data[o] = Math.tanh(this.data[o]);
    return e;
  }
  gelu() {
    const e = new u(this.shape), o = Math.sqrt(2 / Math.PI);
    for (let t = 0; t < this.size; t++) {
      const n = this.data[t];
      e.data[t] = 0.5 * n * (1 + Math.tanh(o * (n + 0.044715 * Math.pow(n, 3))));
    }
    return e;
  }
  softmax(e = -1) {
    const o = new u(this.shape);
    if (this.shape.length === 1) {
      let t = -1 / 0;
      for (let s = 0; s < this.size; s++) this.data[s] > t && (t = this.data[s]);
      let n = 0;
      for (let s = 0; s < this.size; s++)
        o.data[s] = Math.exp(this.data[s] - t), n += o.data[s];
      for (let s = 0; s < this.size; s++) o.data[s] /= n;
      return o;
    }
    if (this.shape.length === 2) {
      const [t, n] = this.shape;
      for (let s = 0; s < t; s++) {
        let a = -1 / 0;
        for (let l = 0; l < n; l++) {
          const h = this.get(s, l);
          h > a && (a = h);
        }
        let p = 0;
        for (let l = 0; l < n; l++) {
          const h = Math.exp(this.get(s, l) - a);
          o.set(h, s, l), p += h;
        }
        for (let l = 0; l < n; l++)
          o.set(o.get(s, l) / p, s, l);
      }
      return o;
    }
    throw new Error("softmax supports 1D or 2D tensors");
  }
  // --- 2D Convolution & Pooling ---
  /**
   * 2D Convolution on [H, W, InChannels] with Kernel [Kh, Kw, InChannels, OutChannels]
   */
  conv2d(e, o, t = 1, n = 0) {
    const [s, a, p] = this.shape, [l, h, m, g] = e.shape;
    if (p !== m)
      throw new Error(`Channel mismatch: input has ${p}, kernel expects ${m}`);
    const f = Math.floor((s - l + 2 * n) / t) + 1, i = Math.floor((a - h + 2 * n) / t) + 1, r = new u([f, i, g]);
    for (let c = 0; c < g; c++) {
      const d = o ? o.data[c] : 0;
      for (let x = 0; x < f; x++)
        for (let w = 0; w < i; w++) {
          let S = d;
          const _ = x * t - n, y = w * t - n;
          for (let $ = 0; $ < l; $++) {
            const T = _ + $;
            if (!(T < 0 || T >= s))
              for (let F = 0; F < h; F++) {
                const A = y + F;
                if (!(A < 0 || A >= a))
                  for (let v = 0; v < p; v++)
                    S += this.get(T, A, v) * e.get($, F, v, c);
              }
          }
          r.set(S, x, w, c);
        }
    }
    return r;
  }
  maxPool2d(e = 2, o = 2) {
    const [t, n, s] = this.shape, a = Math.floor((t - e) / o) + 1, p = Math.floor((n - e) / o) + 1, l = new u([a, p, s]);
    for (let h = 0; h < s; h++)
      for (let m = 0; m < a; m++)
        for (let g = 0; g < p; g++) {
          let f = -1 / 0;
          const i = m * o, r = g * o;
          for (let c = 0; c < e; c++)
            for (let d = 0; d < e; d++) {
              const x = this.get(i + c, r + d, h);
              x > f && (f = x);
            }
          l.set(f, m, g, h);
        }
    return l;
  }
  flatten() {
    return new u([this.size], this.data);
  }
  toArray() {
    if (this.shape.length === 1)
      return Array.from(this.data);
    if (this.shape.length === 2) {
      const [e, o] = this.shape, t = [];
      for (let n = 0; n < e; n++) {
        const s = [];
        for (let a = 0; a < o; a++) s.push(this.get(n, a));
        t.push(s);
      }
      return t;
    }
    return Array.from(this.data);
  }
}
class E {
  constructor() {
    b(this, "steps", []);
  }
  record(e) {
    this.steps.push({
      stepIndex: this.steps.length,
      ...e
    });
  }
  clear() {
    this.steps = [];
  }
  getTraceSummary() {
    return this.steps.map(
      (e) => `[Step ${e.stepIndex}] ${e.layerName} (${e.operation}): ${e.formula} -> Shape [${e.outputShape.join(", ")}]`
    ).join(`
`);
  }
}
class L {
  constructor(e) {
    b(this, "weights", []);
    b(this, "biases", []);
    b(this, "activations", []);
    b(this, "tracer", new E());
    this.config = e, this.activations = e.activations;
    for (let o = 0; o < e.layerSizes.length - 1; o++) {
      const t = e.layerSizes[o], n = e.layerSizes[o + 1], s = Math.sqrt(2 / t);
      this.weights.push(u.random([t, n], -s, s)), this.biases.push(u.zeros([1, n]));
    }
  }
  forward(e, o = !1) {
    o && this.tracer.clear();
    let t = e;
    for (let n = 0; n < this.weights.length; n++) {
      const s = this.weights[n], a = this.biases[n], p = this.activations[n], l = t.matmul(s).add(a);
      let h = l;
      p === "relu" ? h = l.relu() : p === "sigmoid" ? h = l.sigmoid() : p === "tanh" && (h = l.tanh()), o && this.tracer.record({
        layerId: `layer_${n + 1}`,
        layerName: `Hidden Layer ${n + 1} (${p.toUpperCase()})`,
        operation: "Dense Matmul + Bias + Activation",
        formula: `a^[${n + 1}] = ${p}(W^[${n + 1}] * a^[${n}] + b^[${n + 1}])`,
        inputShapes: [t.shape, s.shape],
        outputShape: h.shape,
        tensorPreview: Array.from(h.data.slice(0, 4)),
        pedagogicalInsight: `Layer ${n + 1} transforms ${s.shape[0]} inputs into ${s.shape[1]} linear combinations, activated by ${p}.`
      }), t = h;
    }
    return t;
  }
  // Train a single epoch on a dataset X, Y
  trainStep(e, o) {
    let t = 0;
    const n = this.config.learningRate;
    for (let s = 0; s < e.length; s++) {
      const a = u.fromArray([e[s]]), p = o[s], l = [a], h = [];
      let m = a;
      for (let r = 0; r < this.weights.length; r++) {
        const c = m.matmul(this.weights[r]).add(this.biases[r]);
        h.push(c);
        const d = this.activations[r];
        d === "relu" ? m = c.relu() : d === "sigmoid" ? m = c.sigmoid() : d === "tanh" && (m = c.tanh()), l.push(m);
      }
      const f = m.data[0] - p[0];
      t += 0.5 * f * f;
      let i = new u([1, 1], [f]);
      for (let r = this.weights.length - 1; r >= 0; r--) {
        const c = h[r], d = this.activations[r], x = l[r], w = new u(c.shape);
        for (let y = 0; y < c.size; y++)
          if (d === "relu") w.data[y] = c.data[y] > 0 ? 1 : 0;
          else if (d === "sigmoid") {
            const $ = 1 / (1 + Math.exp(-c.data[y]));
            w.data[y] = $ * (1 - $);
          } else if (d === "tanh") {
            const $ = Math.tanh(c.data[y]);
            w.data[y] = 1 - $ * $;
          } else
            w.data[y] = 1;
        const S = i.mul(w), _ = x.transpose().matmul(S);
        for (let y = 0; y < this.weights[r].size; y++)
          this.weights[r].data[y] -= n * _.data[y];
        for (let y = 0; y < this.biases[r].size; y++)
          this.biases[r].data[y] -= n * S.data[y];
        r > 0 && (i = S.matmul(this.weights[r].transpose()));
      }
    }
    return { loss: t / e.length };
  }
  /**
   * Evaluate a 2D grid of points for boundary visualization (e.g., XOR, Circles, Moons)
   */
  evaluateGrid(e = 30, o = 4) {
    const t = [], n = o * 2 / e, s = [], a = [];
    for (let p = 0; p < e; p++) {
      const l = [], h = o - p * n;
      a.push(h);
      for (let m = 0; m < e; m++) {
        const g = -o + m * n;
        p === 0 && s.push(g);
        const f = this.forward(new u([1, 2], [g, h]));
        l.push(f.data[0]);
      }
      t.push(l);
    }
    return { grid: t, xRange: s, yRange: a };
  }
}
class P {
  constructor() {
    b(this, "tracer", new E());
    // Assignment 3 exact architecture:
    // Conv2D(32, 3x3) -> MaxPool(2x2) -> Conv2D(64, 3x3) -> MaxPool(2x2) -> Conv2D(64, 3x3) -> Flatten -> Dense(64) -> Dense(10, softmax)
    b(this, "kernel1");
    b(this, "bias1");
    b(this, "kernel2");
    b(this, "bias2");
    b(this, "kernel3");
    b(this, "bias3");
    b(this, "dense1_W");
    b(this, "dense1_B");
    b(this, "dense2_W");
    b(this, "dense2_B");
    this.kernel1 = u.random([3, 3, 1, 32], -0.2, 0.2), this.bias1 = u.zeros([32]), this.kernel2 = u.random([3, 3, 32, 64], -0.15, 0.15), this.bias2 = u.zeros([64]), this.kernel3 = u.random([3, 3, 64, 64], -0.15, 0.15), this.bias3 = u.zeros([64]), this.dense1_W = u.random([576, 64], -0.1, 0.1), this.dense1_B = u.zeros([1, 64]), this.dense2_W = u.random([64, 10], -0.1, 0.1), this.dense2_B = u.zeros([1, 10]);
  }
  forward(e, o = !0) {
    o && this.tracer.clear();
    const t = e.conv2d(this.kernel1, this.bias1, 1, 0).relu();
    o && this.tracer.record({
      layerId: "conv1",
      layerName: "Conv2D (32 filters, 3x3)",
      operation: "2D Convolution + ReLU",
      formula: "H_out = (28 - 3 + 0)/1 + 1 = 26; Params = (3*3*1 + 1)*32 = 320",
      inputShapes: [e.shape, this.kernel1.shape],
      outputShape: t.shape,
      tensorPreview: Array.from(t.data.slice(0, 5)),
      fullOutput: t,
      pedagogicalInsight: "Extracts 32 low-level edge and texture feature maps from the raw 28x28 pixel grid.",
      metadata: { filters: 32, kernel: "3x3", params: 320 }
    });
    const n = t.maxPool2d(2, 2);
    o && this.tracer.record({
      layerId: "pool1",
      layerName: "MaxPooling2D (2x2)",
      operation: "Spatial Downsampling",
      formula: "H_out = floor(26/2) = 13; W_out = 13",
      inputShapes: [t.shape],
      outputShape: n.shape,
      tensorPreview: Array.from(n.data.slice(0, 5)),
      fullOutput: n,
      pedagogicalInsight: "Preserves the most salient local features while reducing spatial dimensions by 75%."
    });
    const s = n.conv2d(this.kernel2, this.bias2, 1, 0).relu();
    o && this.tracer.record({
      layerId: "conv2",
      layerName: "Conv2D (64 filters, 3x3)",
      operation: "2D Convolution + ReLU",
      formula: "H_out = (13 - 3 + 0)/1 + 1 = 11; Params = (3*3*32 + 1)*64 = 18,496",
      inputShapes: [n.shape, this.kernel2.shape],
      outputShape: s.shape,
      tensorPreview: Array.from(s.data.slice(0, 5)),
      fullOutput: s,
      pedagogicalInsight: "Combines local features into mid-level parts (corners, loops, strokes)."
    });
    const a = s.maxPool2d(2, 2);
    o && this.tracer.record({
      layerId: "pool2",
      layerName: "MaxPooling2D (2x2)",
      operation: "Spatial Downsampling",
      formula: "H_out = floor(11/2) = 5; W_out = 5",
      inputShapes: [s.shape],
      outputShape: a.shape,
      tensorPreview: Array.from(a.data.slice(0, 5)),
      fullOutput: a,
      pedagogicalInsight: "Further downsamples to 5x5 feature grids."
    });
    const p = a.conv2d(this.kernel3, this.bias3, 1, 0).relu();
    o && this.tracer.record({
      layerId: "conv3",
      layerName: "Conv2D (64 filters, 3x3)",
      operation: "2D Convolution + ReLU",
      formula: "H_out = (5 - 3 + 0)/1 + 1 = 3; Params = (3*3*64 + 1)*64 = 36,928",
      inputShapes: [a.shape, this.kernel3.shape],
      outputShape: p.shape,
      tensorPreview: Array.from(p.data.slice(0, 5)),
      fullOutput: p,
      pedagogicalInsight: "High-level digit shape representations."
    });
    const l = p.flatten(), g = new u([1, 576], l.data).matmul(this.dense1_W).add(this.dense1_B).relu().matmul(this.dense2_W).add(this.dense2_B), f = g.softmax(-1), i = Array.from(f.data);
    let r = 0, c = -1;
    for (let d = 0; d < 10; d++)
      i[d] > c && (c = i[d], r = d);
    return o && this.tracer.record({
      layerId: "digit_probs",
      layerName: "Softmax Classification Head",
      operation: "Softmax(logits) -> Argmax",
      formula: "P(digit=k) = exp(z_k) / sum(exp(z)); Predicted = argmax_k(P)",
      inputShapes: [g.shape],
      outputShape: [10],
      tensorPreview: i,
      pedagogicalInsight: `Final digit classification with top prediction ${r} (${(c * 100).toFixed(1)}%).`
    }), {
      probabilities: i,
      predictedDigit: r,
      conv1Out: t,
      pool1Out: n,
      conv2Out: s,
      pool2Out: a,
      conv3Out: p
    };
  }
  loadWeightsFromJSON(e) {
    e.kernel1 && (this.kernel1 = new u([3, 3, 1, 32], e.kernel1)), e.bias1 && (this.bias1 = new u([32], e.bias1)), e.dense2_W && (this.dense2_W = new u([64, 10], e.dense2_W)), e.dense2_B && (this.dense2_B = new u([1, 10], e.dense2_B));
  }
}
class I {
  constructor(e = 4, o = 4) {
    b(this, "hiddenSize");
    b(this, "inputSize");
    b(this, "tracer", new E());
    // Gates: f, i, c, o (combined or separate)
    b(this, "W_f");
    b(this, "U_f");
    b(this, "b_f");
    b(this, "W_i");
    b(this, "U_i");
    b(this, "b_i");
    b(this, "W_c");
    b(this, "U_c");
    b(this, "b_c");
    b(this, "W_o");
    b(this, "U_o");
    b(this, "b_o");
    this.inputSize = e, this.hiddenSize = o;
    const t = 0.5;
    this.W_f = u.random([e, o], -t, t), this.U_f = u.random([o, o], -t, t), this.b_f = u.ones([1, o]), this.W_i = u.random([e, o], -t, t), this.U_i = u.random([o, o], -t, t), this.b_i = u.zeros([1, o]), this.W_c = u.random([e, o], -t, t), this.U_c = u.random([o, o], -t, t), this.b_c = u.zeros([1, o]), this.W_o = u.random([e, o], -t, t), this.U_o = u.random([o, o], -t, t), this.b_o = u.zeros([1, o]);
  }
  step(e, o, t) {
    const n = e.matmul(this.W_f).add(o.matmul(this.U_f)).add(this.b_f).sigmoid(), s = e.matmul(this.W_i).add(o.matmul(this.U_i)).add(this.b_i).sigmoid(), a = e.matmul(this.W_c).add(o.matmul(this.U_c)).add(this.b_c).tanh(), p = n.mul(t).add(s.mul(a)), l = e.matmul(this.W_o).add(o.matmul(this.U_o)).add(this.b_o).sigmoid();
    return { h: l.mul(p.tanh()), c: p, gates: { f: n, i: s, c_tilde: a, o: l } };
  }
  unroll(e) {
    this.tracer.clear();
    const o = [];
    let t = u.zeros([1, this.hiddenSize]), n = u.zeros([1, this.hiddenSize]);
    for (let s = 0; s < e.length; s++) {
      const a = e[s], p = new u([1, this.inputSize], a.vector), l = this.step(p, t, n);
      t = l.h, n = l.c;
      const h = {
        t: s,
        inputToken: a.token,
        x_t: Array.from(p.data),
        f_gate: Array.from(l.gates.f.data),
        i_gate: Array.from(l.gates.i.data),
        c_tilde: Array.from(l.gates.c_tilde.data),
        c_t: Array.from(n.data),
        o_gate: Array.from(l.gates.o.data),
        h_t: Array.from(t.data)
      };
      o.push(h), this.tracer.record({
        layerId: `lstm_step_${s}`,
        layerName: `LSTM Step t=${s} ("${a.token}")`,
        operation: "Recurrent Cell State Transition",
        formula: "c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t; h_t = o_t ⊙ tanh(c_t)",
        inputShapes: [p.shape, t.shape, n.shape],
        outputShape: t.shape,
        tensorPreview: Array.from(t.data),
        pedagogicalInsight: `Step t=${s}: Forget gate retained ${(h.f_gate[0] * 100).toFixed(0)}% of prior cell memory while input gate injected ${(h.i_gate[0] * 100).toFixed(0)}% of new token information.`
      });
    }
    return o;
  }
}
class z {
  constructor(e = 8, o = 2) {
    b(this, "d_model");
    b(this, "d_k");
    b(this, "numHeads");
    b(this, "tracer", new E());
    b(this, "W_q");
    b(this, "W_k");
    b(this, "W_v");
    b(this, "W_o");
    this.d_model = e, this.numHeads = o, this.d_k = Math.floor(e / o);
    const t = Math.sqrt(2 / e);
    this.W_q = u.random([e, e], -t, t), this.W_k = u.random([e, e], -t, t), this.W_v = u.random([e, e], -t, t), this.W_o = u.random([e, e], -t, t);
  }
  /**
   * Forward pass over token embeddings matrix X of shape [seqLen, d_model]
   */
  forward(e, o = !0, t = !0) {
    t && this.tracer.clear();
    const n = e.shape[0], s = e.matmul(this.W_q), a = e.matmul(this.W_k), p = e.matmul(this.W_v), l = 1 / Math.sqrt(this.d_model), h = a.transpose(), m = s.matmul(h).mul(l);
    if (o)
      for (let c = 0; c < n; c++)
        for (let d = c + 1; d < n; d++)
          m.set(-1e9, c, d);
    const g = m.softmax(-1), i = g.matmul(p).matmul(this.W_o);
    t && this.tracer.record({
      layerId: "self_attention",
      layerName: `Self-Attention (d_model=${this.d_model}, seqLen=${n})`,
      operation: "Scaled Dot-Product Attention",
      formula: "Attention(Q,K,V) = softmax(Q K^T / √d_k + Mask) V",
      inputShapes: [e.shape, this.W_q.shape],
      outputShape: i.shape,
      tensorPreview: Array.from(g.data.slice(0, 6)),
      pedagogicalInsight: `Calculated ${n}x${n} pairwise attention affinity matrix; tokens dynamically aggregate information from allowed preceding contexts.`
    });
    const r = [];
    for (let c = 0; c < n; c++) {
      const d = [];
      for (let x = 0; x < n; x++)
        d.push(g.get(c, x));
      r.push(d);
    }
    return {
      attentionWeights: r,
      output: i
    };
  }
}
class R {
  constructor(e = 10, o = 2) {
    b(this, "tracer", new E());
    // Encoder: [D_in, 16] -> [16, 2] (Latent)
    b(this, "W_enc1");
    b(this, "b_enc1");
    b(this, "W_enc2");
    b(this, "b_enc2");
    // Decoder: [2, 16] -> [16, D_in] (Reconstruction)
    b(this, "W_dec1");
    b(this, "b_dec1");
    b(this, "W_dec2");
    b(this, "b_dec2");
    this.inDim = e, this.latentDim = o;
    const t = 0.3;
    this.W_enc1 = u.random([e, 16], -t, t), this.b_enc1 = u.zeros([1, 16]), this.W_enc2 = u.random([16, o], -t, t), this.b_enc2 = u.zeros([1, o]), this.W_dec1 = u.random([o, 16], -t, t), this.b_dec1 = u.zeros([1, 16]), this.W_dec2 = u.random([16, e], -t, t), this.b_dec2 = u.zeros([1, e]);
  }
  encode(e) {
    return e.matmul(this.W_enc1).add(this.b_enc1).relu().matmul(this.W_enc2).add(this.b_enc2);
  }
  decode(e) {
    return e.matmul(this.W_dec1).add(this.b_dec1).relu().matmul(this.W_dec2).add(this.b_dec2).sigmoid();
  }
  forward(e, o = !0) {
    o && this.tracer.clear();
    const t = this.encode(e), n = this.decode(t);
    let s = 0;
    for (let a = 0; a < e.size; a++) {
      const p = e.data[a] - n.data[a];
      s += p * p;
    }
    return s /= e.size, o && this.tracer.record({
      layerId: "latent_bottleneck",
      layerName: `Latent Space (dim=${this.latentDim})`,
      operation: "Nonlinear Dimensionality Compression",
      formula: "z = W_2 * relu(W_1 x + b_1) + b_2",
      inputShapes: [e.shape],
      outputShape: t.shape,
      tensorPreview: Array.from(t.data),
      pedagogicalInsight: `Compressed input from ${this.inDim} dimensions to ${this.latentDim} latent coordinates with MSE=${s.toFixed(4)}.`
    }), { latent: t, reconstructed: n, mse: s };
  }
}
class C {
  static renderNetwork(e, o = 800, t = 360, n = -1, s = "forward") {
    const a = e.length, p = o / (a + 1), l = [];
    e.forEach((i, r) => {
      const c = (r + 1) * p, d = Math.min(i.outShape[i.outShape.length - 1] || 4, 8), x = t / (d + 1), w = [];
      for (let S = 0; S < d; S++)
        w.push({
          layerIndex: r,
          nodeIndex: S,
          x: c,
          y: (S + 1) * x,
          label: `${i.name} [${S}]`
        });
      l.push(w);
    });
    const h = s === "backward", m = h ? "#C8102E" : "#0F8B8D", g = h ? "reverseFlow" : "flowPulse";
    let f = `<svg viewBox="0 0 ${o} ${t}" width="100%" height="${t}" style="background: radial-gradient(circle at 50% 50%, #0F172A 0%, #020617 100%); border-radius:12px; font-family:system-ui, -apple-system, sans-serif;">`;
    f += `<defs>
      <linearGradient id="edgeGradFwd" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="#0F8B8D" stop-opacity="0.6"/>
        <stop offset="100%" stop-color="#6B2D7B" stop-opacity="0.8"/>
      </linearGradient>
      <linearGradient id="edgeGradBwd" x1="100%" y1="0%" x2="0%" y2="0%">
        <stop offset="0%" stop-color="#C8102E" stop-opacity="0.8"/>
        <stop offset="100%" stop-color="#FFD700" stop-opacity="0.6"/>
      </linearGradient>
      <filter id="glowGold" x="-30%" y="-30%" width="160%" height="160%">
        <feGaussianBlur stdDeviation="4" result="blur" />
        <feComposite in="SourceGraphic" in2="blur" operator="over" />
      </filter>
      <style>
        @keyframes flowPulse { from { stroke-dashoffset: 24; } to { stroke-dashoffset: 0; } }
        @keyframes reverseFlow { from { stroke-dashoffset: 0; } to { stroke-dashoffset: 24; } }
        .flow-line { stroke-dasharray: 6 3; animation: ${g} 0.9s linear infinite; }
        .active-pulse { animation: nodePulse 1.2s ease-in-out infinite alternate; }
        @keyframes nodePulse { from { transform: scale(1); filter: drop-shadow(0 0 2px #FFD700); } to { transform: scale(1.2); filter: drop-shadow(0 0 8px #FFD700); } }
      </style>
    </defs>`;
    for (let i = 0; i < l.length - 1; i++) {
      const r = l[i], c = l[i + 1], d = i === n || i + 1 === n, x = h ? "url(#edgeGradBwd)" : "url(#edgeGradFwd)";
      for (const w of r)
        for (const S of c) {
          const _ = d ? 'class="flow-line"' : "", y = d ? "2.2" : "1.0", $ = d ? "0.9" : "0.3";
          f += `<line x1="${w.x}" y1="${w.y}" x2="${S.x}" y2="${S.y}" stroke="${x}" stroke-width="${y}" stroke-opacity="${$}" ${_}/>`;
        }
    }
    return e.forEach((i, r) => {
      const c = (r + 1) * p, d = r === n;
      f += `<text x="${c}" y="22" fill="${d ? "#FFD700" : "#E2E8F0"}" font-size="12" font-weight="700" text-anchor="middle">${i.name}</text>`, f += `<rect x="${c - 45}" y="30" width="90" height="18" rx="4" fill="#1E293B" stroke="${d ? "#FFD700" : "#334155"}" stroke-width="1"/>`, f += `<text x="${c}" y="43" fill="#38BDF8" font-size="10" font-family="monospace" font-weight="600" text-anchor="middle">[${i.outShape.join("×")}]</text>`, i.paramsCount > 0 && (f += `<text x="${c}" y="${t - 12}" fill="#94A3B8" font-size="10" font-family="monospace" text-anchor="middle">${i.paramsCount.toLocaleString()} params</text>`);
    }), l.forEach((i, r) => {
      const c = r === n, d = c ? "#FFD700" : r === 0 ? "#0F8B8D" : r === l.length - 1 ? h ? "#C8102E" : "#10B981" : "#6B2D7B";
      for (const x of i) {
        const w = c ? 'filter="url(#glowGold)" class="active-pulse"' : "";
        f += `<circle cx="${x.x}" cy="${x.y}" r="${c ? 11 : 9}" fill="${d}" stroke="#ffffff" stroke-width="1.8" ${w}/>`;
      }
    }), f += `<g transform="translate(${o / 2 - 80}, ${t - 35})">
      <rect width="160" height="22" rx="11" fill="#1E293B" stroke="${m}" stroke-width="1.2"/>
      <text x="80" y="15" fill="${m}" font-size="10" font-weight="700" text-anchor="middle" letter-spacing="0.5">
        ${h ? "◀ BACKPROPAGATION GRADIENTS" : "FORWARD INFERENCE FLOW ▶"}
      </text>
    </g>`, f += "</svg>", f;
  }
}
class M {
  /**
   * Helper to set up HiDPI canvas for ultra-sharp Retina rendering
   */
  static setupHiDPI(e) {
    const o = window.devicePixelRatio || 1, t = e.getBoundingClientRect(), n = t.width || e.width, s = t.height || e.height;
    e.width = n * o, e.height = s * o;
    const a = e.getContext("2d");
    return a && a.scale(o, o), a;
  }
  /**
   * 1. Decision Boundary & Active Gradient Heatmap
   */
  static renderDecisionBoundary(e, o, t) {
    const n = e.getContext("2d");
    if (!n) return;
    const s = o.length, a = o[0].length, p = e.width / a, l = e.height / s;
    for (let h = 0; h < s; h++)
      for (let m = 0; m < a; m++) {
        const g = o[h][m], f = Math.floor(g * 220 + (1 - g) * 15), i = Math.floor((1 - Math.abs(g - 0.5) * 2) * 160), r = Math.floor((1 - g) * 220 + g * 25);
        n.fillStyle = `rgb(${f}, ${i}, ${r})`, n.fillRect(m * p, h * l, p + 1, l + 1);
      }
    if (t)
      for (const h of t) {
        const m = (h.x + 4) / 8 * e.width, g = (4 - h.y) / 8 * e.height;
        n.beginPath(), n.arc(m, g, 6, 0, Math.PI * 2), n.fillStyle = h.label === 1 ? "#FFD700" : "#FFFFFF", n.strokeStyle = "#0B1329", n.lineWidth = 2, n.fill(), n.stroke();
      }
  }
  /**
   * 2. CNN Interactive 3x3 Sliding Kernel Visualizer (Stanford CS231n / CNN Explainer style)
   */
  static renderCNNKernelSlide(e, o, t, n, s = 0.05) {
    var _;
    const a = e.getContext("2d");
    if (!a) return { activation: 0, formulaText: "" };
    a.fillStyle = "#020617", a.fillRect(0, 0, e.width, e.height);
    const p = 16, l = 32, h = 140, m = o.length, g = o[0].length, f = h / m;
    a.fillStyle = "#38BDF8", a.font = "bold 11px monospace", a.fillText("INPUT IMAGE (28×28)", p, 20);
    for (let y = 0; y < m; y++)
      for (let $ = 0; $ < g; $++) {
        const T = o[y][$], F = Math.floor(T * 255);
        a.fillStyle = `rgb(${F}, ${F}, ${F})`, a.fillRect(p + $ * f, l + y * f, f - 0.5, f - 0.5);
      }
    const i = n.row, r = n.col;
    a.strokeStyle = "#FFD700", a.lineWidth = 2, a.strokeRect(p + r * f, l + i * f, f * 3, f * 3);
    const c = 180, d = 32;
    a.fillStyle = "#0F172A", a.strokeStyle = "#334155", a.lineWidth = 1, a.fillRect(c, d, 140, 140), a.strokeRect(c, d, 140, 140), a.fillStyle = "#FFD700", a.font = "bold 10px monospace", a.fillText("3×3 KERNEL MULTIPLY", c + 6, d + 14);
    let x = s;
    const w = 36;
    for (let y = 0; y < 3; y++)
      for (let $ = 0; $ < 3; $++) {
        const T = ((_ = o[i + y]) == null ? void 0 : _[r + $]) || 0, F = t[y][$], A = T * F;
        x += A;
        const v = c + 8 + $ * (w + 4), B = d + 22 + y * (w + 4);
        a.fillStyle = "#1E293B", a.fillRect(v, B, w, w), a.strokeStyle = "#475569", a.strokeRect(v, B, w, w), a.fillStyle = "#E2E8F0", a.font = "9px monospace", a.textAlign = "center", a.fillText(`x:${T.toFixed(1)}`, v + w / 2, B + 14), a.fillStyle = "#38BDF8", a.fillText(`w:${F.toFixed(1)}`, v + w / 2, B + 28);
      }
    a.textAlign = "left";
    const S = Math.max(0, x);
    return a.fillStyle = "#F8FAFC", a.font = "11px monospace", a.fillText(`Linear Sum z = (Σ x_i·w_i) + b = ${x.toFixed(3)}`, 16, 195), a.fillStyle = "#10B981", a.fillText(`Feature Map Pixel = ReLU(z) = ${S.toFixed(3)}`, 16, 215), a.fillStyle = "#94A3B8", a.font = "10px monospace", a.fillText(`Active Output Pixel: (${i}, ${r}) in 26×26 feature map`, 16, 235), {
      activation: S,
      formulaText: `z = Σ x_i·w_i + ${s.toFixed(2)} = ${x.toFixed(3)} → ReLU = ${S.toFixed(3)}`
    };
  }
  /**
   * 3. LSTM Memory Conveyor Belt & 4-Gate Anatomy
   */
  static renderLSTMConveyorBelt(e, o, t, n) {
    const s = e.getContext("2d");
    if (!s) return;
    s.fillStyle = "#020617", s.fillRect(0, 0, e.width, e.height), s.fillStyle = "#38BDF8", s.font = "bold 12px monospace", s.fillText(`LSTM CELL ANATOMY (Step t=${o}: "${t}")`, 16, 24), s.strokeStyle = "#6B2D7B", s.lineWidth = 4, s.beginPath(), s.moveTo(20, 60), s.lineTo(e.width - 20, 60), s.stroke(), s.fillStyle = "#FFD700", s.font = "10px monospace", s.fillText(`Cell State c_t = ${n.c.toFixed(3)} (Memory Belt)`, 30, 50);
    const a = [
      { name: "Forget (f_t)", val: n.f, color: "#C8102E", desc: "0=Drop, 1=Keep" },
      { name: "Input (i_t)", val: n.i, color: "#0F8B8D", desc: "Write weight" },
      { name: "Candidate (c̃_t)", val: n.c_tilde, color: "#38BDF8", desc: "New candidate" },
      { name: "Output (o_t)", val: n.o, color: "#10B981", desc: "Hidden filter" }
    ], p = 68, l = 20;
    a.forEach((h, m) => {
      const g = l + m * (p + 12), f = 85;
      s.fillStyle = "#1E293B", s.fillRect(g, f, p, 95), s.strokeStyle = "#334155", s.strokeRect(g, f, p, 95);
      const i = Math.min(80, Math.max(5, Math.abs(h.val) * 80));
      s.fillStyle = h.color, s.fillRect(g + 4, f + 90 - i, p - 8, i), s.fillStyle = "#FFF", s.font = "bold 9px sans-serif", s.fillText(h.name, g + 5, f + 15), s.fillStyle = "#FFD700", s.font = "bold 11px monospace", s.fillText(h.val.toFixed(2), g + 16, f + 55);
    }), s.fillStyle = "#F8FAFC", s.font = "11px monospace", s.fillText(`Hidden Emission h_t = o_t ⊙ tanh(c_t) = ${n.h.toFixed(4)}`, 16, 210), s.fillStyle = "#94A3B8", s.font = "10px monospace", s.fillText(`Forget Gate: ${(n.f * 100).toFixed(0)}% retention | Input: ${(n.i * 100).toFixed(0)}% written`, 16, 230);
  }
  /**
   * 4. Scaled Dot-Product Attention Matrix with Causal Mask
   */
  static renderAttentionMatrix(e, o, t, n = -1) {
    const s = e.getContext("2d");
    if (!s) return;
    const a = o.length, p = 65, l = 50, h = e.width - p - 15, m = e.height - l - 15, g = h / a, f = m / a;
    s.clearRect(0, 0, e.width, e.height), s.fillStyle = "#020617", s.fillRect(0, 0, e.width, e.height), s.fillStyle = "#38BDF8", s.font = "bold 11px monospace", s.fillText("ATTENTION WEIGHTS: Softmax(Q K^T / √d_k)", 15, 20);
    for (let i = 0; i < a; i++) {
      const r = i === n;
      for (let c = 0; c < a; c++) {
        const d = o[i][c], x = c > i;
        x ? s.fillStyle = "#0F172A" : s.fillStyle = r ? `rgba(255, 215, 0, ${Math.max(0.15, d)})` : `rgba(15, 139, 141, ${Math.max(0.1, d)})`, s.fillRect(p + c * g, l + i * f, g - 1.5, f - 1.5), g > 28 && (s.fillStyle = x ? "#334155" : d > 0.4 ? "#FFFFFF" : "#94A3B8", s.font = "9px monospace", s.textAlign = "center", s.textBaseline = "middle", s.fillText(
          x ? "—" : d.toFixed(2),
          p + c * g + g / 2,
          l + i * f + f / 2
        ));
      }
    }
    s.fillStyle = "#E2E8F0", s.font = "11px sans-serif", s.textAlign = "right", s.textBaseline = "middle";
    for (let i = 0; i < a; i++) {
      const r = t[i] || `t_${i}`;
      s.fillStyle = i === n ? "#FFD700" : "#CBD5E1", s.fillText(r, p - 6, l + i * f + f / 2);
    }
    s.textAlign = "center", s.textBaseline = "bottom";
    for (let i = 0; i < a; i++) {
      const r = t[i] || `t_${i}`;
      s.fillText(r, p + i * g + g / 2, l - 6);
    }
  }
  /**
   * 5. 3D Optimizer Loss Landscape (SGD vs Momentum vs Adam)
   */
  static renderOptimizerContour(e, o) {
    const t = e.getContext("2d");
    if (!t) return;
    t.fillStyle = "#020617", t.fillRect(0, 0, e.width, e.height), t.fillStyle = "#38BDF8", t.font = "bold 11px monospace", t.fillText("LOSS LANDSCAPE TRAJECTORY (Ravine Contour)", 15, 22);
    const n = e.width / 2, s = e.height / 2 + 10;
    for (let l = 1; l <= 6; l++)
      t.strokeStyle = `rgba(15, 139, 141, ${0.12 * l})`, t.lineWidth = 1.2, t.beginPath(), t.ellipse(n, s, l * 25, l * 12, -Math.PI / 6, 0, Math.PI * 2), t.stroke();
    const a = Math.min(1, o % 20 / 19);
    t.strokeStyle = "#C8102E", t.lineWidth = 2, t.beginPath(), t.moveTo(n - 100, s - 60);
    const p = 8;
    for (let l = 1; l <= p * a; l++) {
      const h = (l % 2 === 0 ? 30 : -30) * (1 - l / p);
      t.lineTo(n - 100 + l * 14, s - 60 + l * 8 + h);
    }
    t.stroke(), t.strokeStyle = "#D98E04", t.lineWidth = 2, t.beginPath(), t.moveTo(n - 100, s - 60);
    for (let l = 1; l <= p * a; l++) {
      const h = (l % 2 === 0 ? 12 : -12) * (1 - l / p);
      t.lineTo(n - 100 + l * 15, s - 60 + l * 8 + h);
    }
    t.stroke(), t.strokeStyle = "#10B981", t.lineWidth = 2.5, t.beginPath(), t.moveTo(n - 100, s - 60);
    for (let l = 1; l <= p * a; l++)
      t.lineTo(n - 100 + l * 16, s - 60 + l * 8.5);
    t.stroke(), t.font = "10px monospace", t.fillStyle = "#C8102E", t.fillText("● SGD (High oscillation)", 15, 195), t.fillStyle = "#D98E04", t.fillText("● Momentum (Damped velocity)", 15, 212), t.fillStyle = "#10B981", t.fillText("● Adam (Fast adaptive convergence)", 15, 230);
  }
}
class N extends HTMLElement {
  constructor() {
    super(...arguments);
    b(this, "timer", null);
    b(this, "isPlaying", !1);
    b(this, "currentStep", 0);
  }
  static get observedAttributes() {
    return ["model", "dataset", "weights", "tokens", "sequence", "title", "course"];
  }
  connectedCallback() {
    this.render();
  }
  disconnectedCallback() {
    this.timer && clearInterval(this.timer);
  }
  attributeChangedCallback() {
    this.render();
  }
  normalizeModelType(o) {
    const t = o.toLowerCase().trim();
    return ["mlp", "neural-networks", "dl-w01", "nlp-w01", "backprop"].includes(t) ? "mlp" : ["lstm", "rnn", "gru", "dl-w10", "dl-w11", "nlp-w02", "recurrent"].includes(t) ? "lstm" : ["cnn", "tokenization-cnn", "dl-w05", "nlp-w03", "convnet", "mnist"].includes(t) ? "cnn" : ["ngrams", "bow", "bag-of-words", "nlp-w04"].includes(t) ? "ngrams" : ["embeddings", "word2vec", "glove", "nlp-w05"].includes(t) ? "embeddings" : ["autoencoder", "vae", "tsne", "dl-w06", "nlp-w06", "latent"].includes(t) ? "autoencoder" : ["lda", "topic-modeling", "topics", "nlp-w07"].includes(t) ? "lda" : ["stm", "structural-topics", "nlp-w08"].includes(t) ? "stm" : ["classification", "regularization", "dropout", "dl-w02", "nlp-w09"].includes(t) ? "classification" : ["ner", "sequence-tagging", "bilstm-ner", "nlp-w10"].includes(t) ? "ner" : ["gan", "gans", "dcgan", "dl-w08", "nlp-w11", "crf-bilstm"].includes(t) ? "gan" : ["transformer", "bert", "attention", "dl-w12", "nlp-w12", "gpt"].includes(t) ? "transformer" : ["regression", "optimizers", "adam", "dl-w03", "loss-landscape"].includes(t) ? "optimizers" : ["transfer-learning", "fine-tuning", "dl-w07"].includes(t) ? "transfer-learning" : ["text-cnn", "conv1d", "dl-w09"].includes(t) ? "text-cnn" : "mlp";
  }
  render() {
    this.timer && (clearInterval(this.timer), this.timer = null, this.isPlaying = !1), this.currentStep = 0;
    const o = this.getAttribute("model") || "mlp", t = this.normalizeModelType(o), n = this.getAttribute("title") || `Model Simulator: ${t.toUpperCase()}`, s = this.getAttribute("course") || "Harvard CSCI E-89 / E-89b SOTA";
    this.innerHTML = `
      <div class="omni-sim-card" style="font-family:system-ui, -apple-system, sans-serif; background:#0B1329; color:#F8FAFC; border:1px solid #1E293B; border-radius:12px; padding:20px; margin:20px 0; box-shadow:0 12px 30px -5px rgba(0,0,0,0.5);">
        <!-- Top Toolbar -->
        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #1E293B; padding-bottom:12px; margin-bottom:14px; flex-wrap:wrap; gap:10px;">
          <div>
            <span style="background:#0F8B8D; color:#fff; font-size:11px; font-weight:700; text-transform:uppercase; padding:3px 8px; border-radius:4px; letter-spacing:0.5px;">${s}</span>
            <h3 style="margin:6px 0 0 0; font-size:1.25rem; color:#FFFFFF;">${n}</h3>
          </div>
          <div style="display:flex; gap:8px; align-items:center;">
            <select class="sim-model-selector" style="background:#1E293B; color:#38BDF8; border:1px solid #475569; border-radius:6px; padding:6px 10px; font-size:12px; font-weight:600; cursor:pointer;">
              <option value="mlp" ${t === "mlp" ? "selected" : ""}>1. MLP & Backprop (W1)</option>
              <option value="lstm" ${t === "lstm" ? "selected" : ""}>2. Recurrent LSTM & Gates (W2)</option>
              <option value="cnn" ${t === "cnn" ? "selected" : ""}>3. Conv2D & MNIST (W3 / DL W5)</option>
              <option value="ngrams" ${t === "ngrams" ? "selected" : ""}>4. N-grams & BoW (W4)</option>
              <option value="embeddings" ${t === "embeddings" ? "selected" : ""}>5. Word2Vec & GloVe (W5)</option>
              <option value="autoencoder" ${t === "autoencoder" ? "selected" : ""}>6. Autoencoder & VAE (W6)</option>
              <option value="lda" ${t === "lda" ? "selected" : ""}>7. Topic Modeling (LDA) (W7)</option>
              <option value="stm" ${t === "stm" ? "selected" : ""}>8. Structural Topic Models (W8)</option>
              <option value="classification" ${t === "classification" ? "selected" : ""}>9. Text Classification & Regularization (W9)</option>
              <option value="ner" ${t === "ner" ? "selected" : ""}>10. NER Sequence Tagging (W10)</option>
              <option value="gan" ${t === "gan" ? "selected" : ""}>11. GANs & CRF (W11 / DL W8)</option>
              <option value="transformer" ${t === "transformer" ? "selected" : ""}>12. Transformers & Attention (W12)</option>
              <option value="optimizers" ${t === "optimizers" ? "selected" : ""}>13. Optimizers & Loss Landscapes (DL W3)</option>
            </select>
            <button class="btn-play" style="background:#1E293B; color:#38BDF8; border:1px solid #38BDF8; border-radius:6px; padding:6px 12px; font-size:12px; font-weight:600; cursor:pointer;">▶ Auto</button>
            <button class="btn-step" style="background:#0F8B8D; color:#FFFFFF; border:none; border-radius:6px; padding:6px 14px; font-size:12px; font-weight:700; cursor:pointer; box-shadow:0 0 10px rgba(15,139,141,0.4);">Step Forward ⏭</button>
            <button class="btn-reset" style="background:#C8102E; color:#FFFFFF; border:none; border-radius:6px; padding:6px 12px; font-size:12px; cursor:pointer;">Reset</button>
          </div>
        </div>

        <!-- State Breadcrumb Bar -->
        <div class="state-breadcrumb" style="display:flex; gap:6px; align-items:center; margin-bottom:14px; background:#020617; padding:8px 12px; border-radius:6px; border:1px solid #1E293B; overflow-x:auto; font-size:11px; font-family:monospace;">
          <span style="color:#94A3B8; font-weight:700;">PHASE:</span>
          <span class="crumb-step" style="background:#0F8B8D; color:#fff; padding:2px 8px; border-radius:4px;">1. INFERENCE</span>
          <span style="color:#475569;">→</span>
          <span class="crumb-step" style="color:#64748B;">2. LOSS</span>
          <span style="color:#475569;">→</span>
          <span class="crumb-step" style="color:#64748B;">3. BACKPROP</span>
          <span style="color:#475569;">→</span>
          <span class="crumb-step" style="color:#64748B;">4. UPDATE</span>
        </div>

        <!-- Viewport (Diagram + Canvas) -->
        <div class="sim-viewport" style="display:grid; grid-template-columns: 1fr 1fr; gap:16px;">
          <div class="sim-diagram-panel" style="background:#020617; border-radius:8px; padding:12px; border:1px solid #1E293B;">
            <div style="font-size:12px; font-weight:700; color:#94A3B8; margin-bottom:8px; letter-spacing:0.5px;">ACTIVE NETWORK ARCHITECTURE & DATA FLOW</div>
            <div class="diagram-host"></div>
          </div>
          <div class="sim-vis-panel" style="background:#020617; border-radius:8px; padding:12px; border:1px solid #1E293B; display:flex; flex-direction:column; align-items:center;">
            <div style="font-size:12px; font-weight:700; color:#94A3B8; margin-bottom:8px; width:100%; letter-spacing:0.5px;">LIVE ACTIVATIONS & COMPUTATIONAL MECHANICS</div>
            <canvas class="sim-canvas" width="340" height="250" style="border-radius:6px; background:#000; width:100%; max-width:340px; height:250px;"></canvas>
            <div class="sim-metrics" style="width:100%; margin-top:10px; font-size:12px; font-family:monospace; color:#E2E8F0;"></div>
          </div>
        </div>

        <!-- Pedagogical Execution Trace -->
        <div class="sim-trace-panel" style="margin-top:16px; background:#020617; border:1px solid #1E293B; border-radius:8px; padding:14px;">
          <div style="font-size:11px; font-weight:700; color:#FFD700; margin-bottom:6px; letter-spacing:0.5px;">PEDAGOGICAL MATHEMATICAL STEP TRACE:</div>
          <div class="trace-output" style="font-size:12px; font-family:monospace; color:#A5B4FC; line-height:1.6;"></div>
        </div>
      </div>
    `, this.initModel(t);
    const a = this.querySelector(".sim-model-selector");
    a && (a.onchange = (p) => {
      const l = p.target.value;
      this.setAttribute("model", l);
    });
  }
  initModel(o) {
    const t = this.querySelector(".sim-canvas"), n = this.querySelector(".diagram-host"), s = this.querySelector(".trace-output"), a = this.querySelector(".btn-step"), p = this.querySelector(".btn-play"), l = this.querySelector(".btn-reset"), h = this.querySelector(".state-breadcrumb");
    let m = () => {
    };
    if (o === "mlp") {
      const g = new L({
        layerSizes: [2, 6, 4, 1],
        activations: ["tanh", "tanh", "sigmoid"],
        learningRate: 0.15
      }), f = [[-2, -2], [-2, 2], [2, -2], [2, 2]], i = [[0], [1], [1], [0]], r = [
        { id: "in", name: "Input [x1, x2]", type: "dense", inShape: [2], outShape: [2], paramsCount: 0 },
        { id: "h1", name: "Hidden 1 (Tanh)", type: "dense", inShape: [2], outShape: [6], paramsCount: 18 },
        { id: "h2", name: "Hidden 2 (Tanh)", type: "dense", inShape: [6], outShape: [4], paramsCount: 28 },
        { id: "out", name: "Output (Sigmoid)", type: "dense", inShape: [4], outShape: [1], paramsCount: 5 }
      ], c = ["forward_1", "forward_2", "loss", "backward_out", "backward_hidden", "update"];
      m = () => {
        const d = this.currentStep % c.length, x = c[d];
        if (x === "update")
          for (let y = 0; y < 15; y++) g.trainStep(f, i);
        const w = x === "forward_1" ? 1 : x === "forward_2" ? 2 : x === "loss" || x === "backward_out" ? 3 : x === "backward_hidden" ? 1 : -1, S = x.startsWith("backward") ? "backward" : "forward";
        n.innerHTML = C.renderNetwork(r, 380, 210, w, S);
        const { grid: _ } = g.evaluateGrid(30, 4);
        M.renderDecisionBoundary(
          t,
          _,
          f.map((y, $) => ({ x: y[0], y: y[1], label: i[$][0] }))
        ), h.innerHTML = `
          <span style="color:#94A3B8; font-weight:700;">STEP ${this.currentStep + 1} (${x.toUpperCase()}):</span>
          <span style="background:${S === "backward" ? "#C8102E" : "#0F8B8D"}; color:#fff; padding:2px 8px; border-radius:4px;">
            ${x === "forward_1" ? "1. LAYER 1 LINEAR + TANH" : x === "forward_2" ? "2. LAYER 2 LINEAR + TANH" : x === "loss" ? "3. SIGMOID + MSE LOSS" : x === "backward_out" ? "4. BACKPROP δ_out" : x === "backward_hidden" ? "5. BACKPROP δ_hidden" : "6. WEIGHT GRADIENT UPDATE"}
          </span>
        `, x === "forward_1" ? s.innerHTML = "• <b>Forward Layer 1:</b> $z^{[1]} = W^{[1]} x + b^{[1]}$; $a^{[1]} = \\tanh(z^{[1]})$. Produces 6 hidden activations." : x === "forward_2" ? s.innerHTML = "• <b>Forward Layer 2:</b> $z^{[2]} = W^{[2]} a^{[1]} + b^{[2]}$; $a^{[2]} = \\tanh(z^{[2]})$. Intermediate nonlinear mapping." : x === "loss" ? s.innerHTML = "• <b>Output Activation & Loss:</b> $\\hat{y} = \\sigma(z^{[3]})$; $L = \\frac{1}{2}(\\hat{y} - y)^2$. Error computed against target." : x === "backward_out" ? s.innerHTML = "• <b>Output Delta:</b> $\\delta^{[3]} = (\\hat{y} - y) \\odot \\sigma'(z^{[3]})$. Sensitivities flow backwards in red." : x === "backward_hidden" ? s.innerHTML = "• <b>Hidden Delta:</b> $\\delta^{[l]} = ((W^{[l+1]})^T \\delta^{[l+1]}) \\odot \\tanh'(z^{[l]})$. Chain rule propagates credit." : s.innerHTML = "• <b>Weight Update:</b> $W^{[l]} \\leftarrow W^{[l]} - \\eta (a^{[l-1]})^T \\delta^{[l]}$. Decision boundary visibly adapts!", this.currentStep++;
      }, m();
    } else if (o === "lstm") {
      const g = new I(4, 4), i = ["Natural", "Language", "Processing", "Recurrent", "Memory"].map((c) => ({ token: c, vector: [0.6, -0.3, 0.7, -0.2] })), r = g.unroll(i);
      m = () => {
        const c = this.currentStep % r.length, d = r[c];
        n.innerHTML = C.renderNetwork([
          { id: "x", name: `Input x_${c}`, type: "dense", inShape: [4], outShape: [4], paramsCount: 0 },
          { id: "gates", name: "Gates [f, i, c̃, o]", type: "lstm_cell", inShape: [4], outShape: [16], paramsCount: 144 },
          { id: "c", name: "Cell State c_t", type: "lstm_cell", inShape: [4], outShape: [4], paramsCount: 0 },
          { id: "h", name: "Hidden State h_t", type: "lstm_cell", inShape: [4], outShape: [4], paramsCount: 0 }
        ], 380, 210, 1, "forward"), M.renderLSTMConveyorBelt(t, d.t, d.inputToken, {
          f: d.f_gate[0],
          i: d.i_gate[0],
          c_tilde: d.c_tilde[0],
          o: d.o_gate[0],
          c: d.c_t[0],
          h: d.h_t[0]
        }), h.innerHTML = `
          <span style="color:#94A3B8; font-weight:700;">SEQUENCE STEP:</span>
          <span style="background:#0F8B8D; color:#fff; padding:2px 8px; border-radius:4px;">Token ${c + 1}/${r.length}: "${d.inputToken}"</span>
        `, s.innerHTML = `• <b>Step t=${d.t} ("${d.inputToken}"):</b><br>&nbsp;&nbsp;1. Forget gate: $f_t = \\sigma(W_f x_t + U_f h_{t-1} + b_f) = ${d.f_gate[0].toFixed(3)}$ (Retains ${(d.f_gate[0] * 100).toFixed(0)}% memory)<br>&nbsp;&nbsp;2. Input gate: $i_t = \\sigma(W_i x_t + U_i h_{t-1} + b_i) = ${d.i_gate[0].toFixed(3)}$<br>&nbsp;&nbsp;3. Cell update: $c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde{c}_t = ${d.c_t[0].toFixed(3)}$ (No vanishing gradient!)<br>&nbsp;&nbsp;4. Hidden emission: $h_t = o_t \\odot \\tanh(c_t) = ${d.h_t[0].toFixed(3)}$`, this.currentStep++;
      }, m();
    } else if (o === "cnn") {
      new P();
      const g = Array(28).fill(0).map(() => Array(28).fill(0));
      for (let i = 5; i < 23; i++)
        g[i][13] = 0.9, g[i][14] = 1, g[i][15] = 0.8;
      g[5][12] = 0.6;
      const f = [
        [0.2, 0.8, -0.4],
        [-0.5, 1.2, -0.5],
        [-0.4, 0.8, 0.2]
      ];
      m = () => {
        const i = 4 + this.currentStep % 18, r = 8 + Math.floor(this.currentStep / 2) % 12;
        n.innerHTML = C.renderNetwork([
          { id: "in", name: "Input Image", type: "conv2d", inShape: [28, 28, 1], outShape: [28, 28, 1], paramsCount: 0 },
          { id: "conv1", name: "Conv2D (3x3)", type: "conv2d", inShape: [28, 28, 1], outShape: [26, 26, 32], paramsCount: 320 },
          { id: "pool", name: "MaxPooling2D", type: "maxpool2d", inShape: [26, 26, 32], outShape: [13, 13, 32], paramsCount: 0 },
          { id: "out", name: "Dense Head", type: "dense", inShape: [576], outShape: [10], paramsCount: 37578 }
        ], 380, 210, 1, "forward");
        const c = M.renderCNNKernelSlide(t, g, f, { row: i, col: r });
        h.innerHTML = `
          <span style="color:#94A3B8; font-weight:700;">KERNEL SLIDE:</span>
          <span style="background:#FFD700; color:#0B1329; font-weight:700; padding:2px 8px; border-radius:4px;">Receptive Window [${i}:${i + 3}, ${r}:${r + 3}]</span>
        `, s.innerHTML = `• <b>Convolving at position (${i}, ${r}):</b><br>&nbsp;&nbsp;1. 9 Pixel Multiplications: $z = \\sum_{i=1}^3 \\sum_{j=1}^3 x_{i,j} \\cdot w_{i,j} + b = ${c.formulaText}$<br>&nbsp;&nbsp;2. Parameter Count Law: $(K_h \\cdot K_w \\cdot C_{in} + 1) \\cdot C_{out} = (3 \\cdot 3 \\cdot 1 + 1) \\cdot 32 = \\mathbf{320\\text{ params}}$<br>&nbsp;&nbsp;3. Spatial Shrinkage: $H_{out} = \\lfloor(28 - 3 + 0)/1 + 1\\rfloor = \\mathbf{26}$. Output map shape: $\\mathbf{26 \\times 26 \\times 32}$.`, this.currentStep++;
      }, m();
    } else if (o === "ngrams") {
      const g = ["deep", "learning", "models", "process", "language", "tokens", "with", "neural", "attention", "unknown_word_1", "unknown_word_2"], f = { deep: 0, learning: 1, models: 2, process: 3, language: 4, tokens: 5, with: 6, neural: 7, attention: 8, "<UNK>": 9 }, i = { deep: 1, learning: 1, models: 1, language: 1, "<UNK>": 0 };
      m = () => {
        const r = g[this.currentStep % g.length], c = !(r in f) || r.startsWith("unknown");
        c ? i["<UNK>"] = (i["<UNK>"] || 0) + 1 : i[r] = (i[r] || 0) + 1, n.innerHTML = C.renderNetwork([
          { id: "text", name: `Token: "${r}"`, type: "dense", inShape: [1], outShape: [1], paramsCount: 0 },
          { id: "hash", name: "Vocabulary Hash Table", type: "dense", inShape: [1], outShape: [10], paramsCount: 0 },
          { id: "bow", name: "BoW Frequency Vector", type: "dense", inShape: [10], outShape: [10], paramsCount: 0 }
        ], 380, 210, c ? 2 : 1, c ? "backward" : "forward");
        const d = t.getContext("2d");
        d && (d.fillStyle = "#020617", d.fillRect(0, 0, t.width, t.height), d.fillStyle = "#38BDF8", d.font = "bold 12px monospace", d.fillText(`STREAMING BoW COUNTER (Active: "${r}")`, 15, 24), Object.keys(i).forEach((w, S) => {
          const _ = w === "<UNK>";
          d.fillStyle = _ ? "#C8102E" : "#0F8B8D", d.fillRect(15, 45 + S * 26, i[w] * 24, 18), d.fillStyle = "#FFF", d.font = "11px monospace", d.fillText(`${w}: ${i[w]}`, 22, 59 + S * 26);
        })), h.innerHTML = `
          <span style="color:#94A3B8; font-weight:700;">TOKEN DISPATCH:</span>
          <span style="background:${c ? "#C8102E" : "#0F8B8D"}; color:#fff; padding:2px 8px; border-radius:4px;">"${r}" → ${c ? "<UNK> ROUTE" : "VOCAB HIT"}</span>
        `, s.innerHTML = `• <b>Token Processing:</b> "${r}"<br>` + (c ? `&nbsp;&nbsp;⚠️ <b>OOV Firewall Triggered:</b> Word not in training vocabulary. Routed to <code>&lt;UNK&gt;</code> (total count: ${i["<UNK>"]}). Vector length remains strictly fixed.` : `&nbsp;&nbsp;✅ <b>In-Vocabulary Match:</b> Incremented frequency count for index <code>${f[r]}</code>.`), this.currentStep++;
      }, m();
    } else if (o === "embeddings") {
      const g = [
        { u: "king", v: "man", w: "woman", target: "queen", desc: "vec(king) - vec(man) + vec(woman) ≈ vec(queen)" },
        { u: "paris", v: "france", w: "italy", target: "rome", desc: "vec(paris) - vec(france) + vec(italy) ≈ vec(rome)" },
        { u: "walking", v: "walk", w: "swim", target: "swimming", desc: "vec(walking) - vec(walk) + vec(swim) ≈ vec(swimming)" }
      ];
      m = () => {
        const f = g[this.currentStep % g.length];
        n.innerHTML = C.renderNetwork([
          { id: "onehot", name: "One-Hot Words", type: "dense", inShape: [1e4], outShape: [300], paramsCount: 3e6 },
          { id: "embed", name: "Dense Embedding Space", type: "dense", inShape: [300], outShape: [300], paramsCount: 0 },
          { id: "proj", name: "2D t-SNE Projection", type: "dense", inShape: [300], outShape: [2], paramsCount: 600 }
        ], 380, 210, 1, "forward");
        const i = t.getContext("2d");
        if (i) {
          i.fillStyle = "#020617", i.fillRect(0, 0, t.width, t.height), i.fillStyle = "#38BDF8", i.font = "bold 12px monospace", i.fillText("WORD2VEC VECTOR SPACE PROJECTION", 15, 24);
          const r = t.width / 2, c = t.height / 2;
          i.strokeStyle = "#334155", i.lineWidth = 1, i.beginPath(), i.moveTo(r, 0), i.lineTo(r, t.height), i.moveTo(0, c), i.lineTo(t.width, c), i.stroke();
          const d = [
            { label: f.u, x: r - 60, y: c - 40, col: "#38BDF8" },
            { label: f.v, x: r - 80, y: c + 30, col: "#94A3B8" },
            { label: f.w, x: r + 40, y: c + 40, col: "#0F8B8D" },
            { label: f.target, x: r + 60, y: c - 30, col: "#FFD700" }
          ];
          d.forEach((x) => {
            i.beginPath(), i.arc(x.x, x.y, 6, 0, Math.PI * 2), i.fillStyle = x.col, i.fill(), i.fillStyle = "#FFF", i.font = "bold 11px sans-serif", i.fillText(x.label, x.x + 8, x.y + 4);
          }), i.strokeStyle = "#FFD700", i.lineWidth = 2, i.setLineDash([4, 2]), i.beginPath(), i.moveTo(d[0].x, d[0].y), i.lineTo(d[3].x, d[3].y), i.stroke(), i.setLineDash([]);
        }
        h.innerHTML = `
          <span style="color:#94A3B8; font-weight:700;">VECTOR ARITHMETIC:</span>
          <span style="background:#0F8B8D; color:#fff; padding:2px 8px; border-radius:4px;">${f.desc}</span>
        `, s.innerHTML = `• <b>Semantic Geometry:</b> <code>${f.desc}</code><br>&nbsp;&nbsp;Cosine similarity $\\cos(\\theta) = \\frac{u \\cdot v}{\\|u\\| \\|v\\|} = 0.884$. Word2Vec captures relational linear translations across concepts.`, this.currentStep++;
      }, m();
    } else if (o === "autoencoder") {
      const g = new R(10, 2);
      m = () => {
        const f = u.random([1, 10], 0, 1), i = g.forward(f, !0);
        n.innerHTML = C.renderNetwork([
          { id: "in", name: "Input x [10]", type: "dense", inShape: [10], outShape: [16], paramsCount: 160 },
          { id: "z", name: "Latent Bottleneck z [2]", type: "latent_space", inShape: [16], outShape: [2], paramsCount: 32 },
          { id: "out", name: "Reconstruction x̂ [10]", type: "dense", inShape: [2], outShape: [10], paramsCount: 160 }
        ], 380, 210, 1, "forward");
        const r = t.getContext("2d");
        if (r) {
          r.fillStyle = "#020617", r.fillRect(0, 0, t.width, t.height), r.fillStyle = "#38BDF8", r.font = "bold 12px monospace", r.fillText("2D LATENT MANIFOLD BOTTLENECK", 15, 24);
          const c = t.width / 2, d = t.height / 2;
          r.strokeStyle = "#1E293B";
          for (let y = -100; y <= 100; y += 25)
            r.beginPath(), r.moveTo(c + y, 0), r.lineTo(c + y, t.height), r.stroke(), r.beginPath(), r.moveTo(0, d + y), r.lineTo(t.width, d + y), r.stroke();
          const x = i.latent.data[0], w = i.latent.data[1], S = c + x * 35, _ = d - w * 35;
          r.beginPath(), r.arc(S, _, 7, 0, Math.PI * 2), r.fillStyle = "#FFD700", r.fill(), r.fillStyle = "#FFF", r.font = "11px monospace", r.fillText(`z = (${x.toFixed(2)}, ${w.toFixed(2)})`, S + 10, _ - 6), r.fillStyle = "#10B981", r.fillText(`Reconstruction Loss MSE: ${i.mse.toFixed(4)}`, 15, 215);
        }
        h.innerHTML = `
          <span style="color:#94A3B8; font-weight:700;">BOTTLENECK COMPRESSION:</span>
          <span style="background:#6B2D7B; color:#fff; padding:2px 8px; border-radius:4px;">10D → 2D Latent → 10D Reconstructed</span>
        `, s.innerHTML = `• <b>Compression Step:</b> Latent coordinates $z = [${i.latent.data[0].toFixed(3)}, ${i.latent.data[1].toFixed(3)}]$.<br>&nbsp;&nbsp;Information bottleneck forces the network to learn the lowest-dimensional intrinsic data manifold without identity memorization.`, this.currentStep++;
      }, m();
    } else if (o === "optimizers")
      m = () => {
        n.innerHTML = C.renderNetwork([
          { id: "loss", name: "Loss L(w)", type: "dense", inShape: [2], outShape: [1], paramsCount: 0 },
          { id: "grad", name: "Gradient ∇L", type: "dense", inShape: [2], outShape: [2], paramsCount: 0 },
          { id: "mom", name: "Momentum v_t", type: "dense", inShape: [2], outShape: [2], paramsCount: 0 },
          { id: "adam", name: "Adam m_t / √v_t", type: "dense", inShape: [2], outShape: [2], paramsCount: 0 }
        ], 380, 210, 2, "forward"), M.renderOptimizerContour(t, this.currentStep), h.innerHTML = `
          <span style="color:#94A3B8; font-weight:700;">OPTIMIZER RACE:</span>
          <span style="background:#0F8B8D; color:#fff; padding:2px 8px; border-radius:4px;">Step ${this.currentStep + 1}: Adam vs Momentum vs SGD</span>
        `, s.innerHTML = `• <b>Adam Update Step ${this.currentStep + 1}:</b><br>&nbsp;&nbsp;1. 1st Moment: $m_t = \\beta_1 m_{t-1} + (1 - \\beta_1) g_t$<br>&nbsp;&nbsp;2. 2nd Moment: $v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2$<br>&nbsp;&nbsp;3. Adaptive Descent: $w_{t+1} = w_t - \\frac{\\alpha}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t$. Adam d製品es rapidly along flat ravines.`, this.currentStep++;
      }, m();
    else if (o === "transformer") {
      const g = new z(8, 2), f = ["The", "neural", "network", "attends", "to", "tokens"], i = u.random([f.length, 8], -0.5, 0.5);
      m = () => {
        const r = this.currentStep % f.length, c = g.forward(i, !0, !0);
        n.innerHTML = C.renderNetwork([
          { id: "emb", name: "Tokens + PosEnc", type: "embedding", inShape: [6], outShape: [6, 8], paramsCount: 48 },
          { id: "qkv", name: "Q, K, V Linear", type: "dense", inShape: [6, 8], outShape: [6, 8], paramsCount: 192 },
          { id: "attn", name: "Attention Matrix", type: "self_attention", inShape: [6, 8], outShape: [6, 6], paramsCount: 0 },
          { id: "out", name: "Output Projection", type: "dense", inShape: [6, 8], outShape: [6, 8], paramsCount: 64 }
        ], 380, 210, 2, "forward"), M.renderAttentionMatrix(t, c.attentionWeights, f, r), h.innerHTML = `
          <span style="color:#94A3B8; font-weight:700;">QUERY FOCUS:</span>
          <span style="background:#FFD700; color:#0B1329; font-weight:700; padding:2px 8px; border-radius:4px;">Token ${r + 1}: "${f[r]}"</span>
        `, s.innerHTML = `• <b>Active Query:</b> Token "${f[r]}" attends to preceding context tokens:<br>&nbsp;&nbsp;1. Scaled Affinity: $S_{${r}, j} = \\frac{q_{${r}} \\cdot k_j^T}{\\sqrt{d_k}}$<br>&nbsp;&nbsp;2. Causal Masking: Future positions $(j > ${r})$ set to $-\\infty$ (shown as —).<br>&nbsp;&nbsp;3. Normalization: $\\alpha_{${r}, j} = \\operatorname{softmax}(S_{${r}, j})$. Context output aggregates $V$.`, this.currentStep++;
      }, m();
    } else
      m = () => {
        this.currentStep++, s.innerHTML = `• <b>Model ${o.toUpperCase()}:</b> Stepping through active forward propagation...`;
      }, m();
    a.onclick = () => {
      m();
    }, l.onclick = () => {
      this.timer && (clearInterval(this.timer), this.timer = null, this.isPlaying = !1, p.innerText = "▶ Auto"), this.currentStep = 0, this.initModel(o);
    }, p.onclick = () => {
      this.isPlaying ? (clearInterval(this.timer), this.timer = null, this.isPlaying = !1, p.innerText = "▶ Auto") : (this.isPlaying = !0, p.innerText = "⏸ Pause", this.timer = setInterval(() => {
        m();
      }, 850));
    };
  }
}
typeof window < "u" && !customElements.get("neural-sim") && customElements.define("neural-sim", N);
export {
  R as AutoencoderModel,
  P as CNNModel,
  M as CanvasVisualizer,
  E as ExecutionTracer,
  I as LSTMModel,
  L as MLP,
  N as NeuralSimElement,
  C as SVGDiagramRenderer,
  u as Tensor,
  z as TransformerAttention
};
//# sourceMappingURL=omni-neural-sim.es.js.map
