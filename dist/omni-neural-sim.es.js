var I = Object.defineProperty;
var P = (C, e, o) => e in C ? I(C, e, { enumerable: !0, configurable: !0, writable: !0, value: o }) : C[e] = o;
var w = (C, e, o) => P(C, typeof e != "symbol" ? e + "" : e, o);
class g {
  constructor(e, o) {
    w(this, "data");
    w(this, "shape");
    w(this, "strides");
    w(this, "size");
    this.shape = [...e], this.size = e.reduce((t, a) => t * a, 1), this.strides = g.computeStrides(this.shape), o ? this.data = o instanceof Float32Array ? o : new Float32Array(o) : this.data = new Float32Array(this.size);
  }
  static computeStrides(e) {
    const o = new Array(e.length);
    let t = 1;
    for (let a = e.length - 1; a >= 0; a--)
      o[a] = t, t *= e[a];
    return o;
  }
  static zeros(e) {
    return new g(e);
  }
  static ones(e) {
    const o = new g(e);
    return o.data.fill(1), o;
  }
  static random(e, o = -1, t = 1) {
    const a = new g(e), n = t - o;
    for (let s = 0; s < a.size; s++)
      a.data[s] = o + Math.random() * n;
    return a;
  }
  static fromArray(e) {
    const o = [];
    let t = e;
    for (; Array.isArray(t); )
      o.push(t.length), t = t[0];
    const a = [];
    function n(s) {
      if (Array.isArray(s))
        for (const f of s) n(f);
      else
        a.push(Number(s));
    }
    return n(e), new g(o, a);
  }
  clone() {
    return new g(this.shape, new Float32Array(this.data));
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
    const o = new g(this.shape);
    if (typeof e == "number")
      for (let t = 0; t < this.size; t++) o.data[t] = this.data[t] + e;
    else
      for (let t = 0; t < this.size; t++) o.data[t] = this.data[t] + e.data[t];
    return o;
  }
  sub(e) {
    const o = new g(this.shape);
    if (typeof e == "number")
      for (let t = 0; t < this.size; t++) o.data[t] = this.data[t] - e;
    else
      for (let t = 0; t < this.size; t++) o.data[t] = this.data[t] - e.data[t];
    return o;
  }
  mul(e) {
    const o = new g(this.shape);
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
    const [o, t] = this.shape, [a, n] = e.shape;
    if (t !== a)
      throw new Error(`Incompatible matrix dims: [${o}, ${t}] x [${a}, ${n}]`);
    const s = new g([o, n]);
    for (let f = 0; f < o; f++)
      for (let c = 0; c < n; c++) {
        let u = 0;
        for (let d = 0; d < t; d++)
          u += this.get(f, d) * e.get(d, c);
        s.set(u, f, c);
      }
    return s;
  }
  transpose() {
    if (this.shape.length !== 2)
      throw new Error("transpose currently supports 2D tensors");
    const [e, o] = this.shape, t = new g([o, e]);
    for (let a = 0; a < e; a++)
      for (let n = 0; n < o; n++)
        t.set(this.get(a, n), n, a);
    return t;
  }
  // --- Activations ---
  relu() {
    const e = new g(this.shape);
    for (let o = 0; o < this.size; o++)
      e.data[o] = Math.max(0, this.data[o]);
    return e;
  }
  sigmoid() {
    const e = new g(this.shape);
    for (let o = 0; o < this.size; o++)
      e.data[o] = 1 / (1 + Math.exp(-this.data[o]));
    return e;
  }
  tanh() {
    const e = new g(this.shape);
    for (let o = 0; o < this.size; o++)
      e.data[o] = Math.tanh(this.data[o]);
    return e;
  }
  gelu() {
    const e = new g(this.shape), o = Math.sqrt(2 / Math.PI);
    for (let t = 0; t < this.size; t++) {
      const a = this.data[t];
      e.data[t] = 0.5 * a * (1 + Math.tanh(o * (a + 0.044715 * Math.pow(a, 3))));
    }
    return e;
  }
  softmax(e = -1) {
    const o = new g(this.shape);
    if (this.shape.length === 1) {
      let t = -1 / 0;
      for (let n = 0; n < this.size; n++) this.data[n] > t && (t = this.data[n]);
      let a = 0;
      for (let n = 0; n < this.size; n++)
        o.data[n] = Math.exp(this.data[n] - t), a += o.data[n];
      for (let n = 0; n < this.size; n++) o.data[n] /= a;
      return o;
    }
    if (this.shape.length === 2) {
      const [t, a] = this.shape;
      for (let n = 0; n < t; n++) {
        let s = -1 / 0;
        for (let c = 0; c < a; c++) {
          const u = this.get(n, c);
          u > s && (s = u);
        }
        let f = 0;
        for (let c = 0; c < a; c++) {
          const u = Math.exp(this.get(n, c) - s);
          o.set(u, n, c), f += u;
        }
        for (let c = 0; c < a; c++)
          o.set(o.get(n, c) / f, n, c);
      }
      return o;
    }
    throw new Error("softmax supports 1D or 2D tensors");
  }
  // --- 2D Convolution & Pooling ---
  /**
   * 2D Convolution on [H, W, InChannels] with Kernel [Kh, Kw, InChannels, OutChannels]
   */
  conv2d(e, o, t = 1, a = 0) {
    const [n, s, f] = this.shape, [c, u, d, y] = e.shape;
    if (f !== d)
      throw new Error(`Channel mismatch: input has ${f}, kernel expects ${d}`);
    const x = Math.floor((n - c + 2 * a) / t) + 1, p = Math.floor((s - u + 2 * a) / t) + 1, l = new g([x, p, y]);
    for (let i = 0; i < y; i++) {
      const r = o ? o.data[i] : 0;
      for (let m = 0; m < x; m++)
        for (let h = 0; h < p; h++) {
          let b = r;
          const F = m * t - a, $ = h * t - a;
          for (let _ = 0; _ < c; _++) {
            const S = F + _;
            if (!(S < 0 || S >= n))
              for (let k = 0; k < u; k++) {
                const v = $ + k;
                if (!(v < 0 || v >= s))
                  for (let T = 0; T < f; T++)
                    b += this.get(S, v, T) * e.get(_, k, T, i);
              }
          }
          l.set(b, m, h, i);
        }
    }
    return l;
  }
  maxPool2d(e = 2, o = 2) {
    const [t, a, n] = this.shape, s = Math.floor((t - e) / o) + 1, f = Math.floor((a - e) / o) + 1, c = new g([s, f, n]);
    for (let u = 0; u < n; u++)
      for (let d = 0; d < s; d++)
        for (let y = 0; y < f; y++) {
          let x = -1 / 0;
          const p = d * o, l = y * o;
          for (let i = 0; i < e; i++)
            for (let r = 0; r < e; r++) {
              const m = this.get(p + i, l + r, u);
              m > x && (x = m);
            }
          c.set(x, d, y, u);
        }
    return c;
  }
  flatten() {
    return new g([this.size], this.data);
  }
  toArray() {
    if (this.shape.length === 1)
      return Array.from(this.data);
    if (this.shape.length === 2) {
      const [e, o] = this.shape, t = [];
      for (let a = 0; a < e; a++) {
        const n = [];
        for (let s = 0; s < o; s++) n.push(this.get(a, s));
        t.push(n);
      }
      return t;
    }
    return Array.from(this.data);
  }
}
class D {
  constructor() {
    w(this, "steps", []);
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
class H {
  constructor(e) {
    w(this, "weights", []);
    w(this, "biases", []);
    w(this, "activations", []);
    w(this, "tracer", new D());
    this.config = e, this.activations = e.activations;
    for (let o = 0; o < e.layerSizes.length - 1; o++) {
      const t = e.layerSizes[o], a = e.layerSizes[o + 1], n = Math.sqrt(2 / t);
      this.weights.push(g.random([t, a], -n, n)), this.biases.push(g.zeros([1, a]));
    }
  }
  forward(e, o = !1) {
    o && this.tracer.clear();
    let t = e;
    for (let a = 0; a < this.weights.length; a++) {
      const n = this.weights[a], s = this.biases[a], f = this.activations[a], c = t.matmul(n).add(s);
      let u = c;
      f === "relu" ? u = c.relu() : f === "sigmoid" ? u = c.sigmoid() : f === "tanh" && (u = c.tanh()), o && this.tracer.record({
        layerId: `layer_${a + 1}`,
        layerName: `Hidden Layer ${a + 1} (${f.toUpperCase()})`,
        operation: "Dense Matmul + Bias + Activation",
        formula: `a^[${a + 1}] = ${f}(W^[${a + 1}] * a^[${a}] + b^[${a + 1}])`,
        inputShapes: [t.shape, n.shape],
        outputShape: u.shape,
        tensorPreview: Array.from(u.data.slice(0, 4)),
        pedagogicalInsight: `Layer ${a + 1} transforms ${n.shape[0]} inputs into ${n.shape[1]} linear combinations, activated by ${f}.`
      }), t = u;
    }
    return t;
  }
  // Train a single epoch on a dataset X, Y
  trainStep(e, o) {
    let t = 0;
    const a = this.config.learningRate;
    for (let n = 0; n < e.length; n++) {
      const s = g.fromArray([e[n]]), f = o[n], c = [s], u = [];
      let d = s;
      for (let l = 0; l < this.weights.length; l++) {
        const i = d.matmul(this.weights[l]).add(this.biases[l]);
        u.push(i);
        const r = this.activations[l];
        r === "relu" ? d = i.relu() : r === "sigmoid" ? d = i.sigmoid() : r === "tanh" && (d = i.tanh()), c.push(d);
      }
      const x = d.data[0] - f[0];
      t += 0.5 * x * x;
      let p = new g([1, 1], [x]);
      for (let l = this.weights.length - 1; l >= 0; l--) {
        const i = u[l], r = this.activations[l], m = c[l], h = new g(i.shape);
        for (let $ = 0; $ < i.size; $++)
          if (r === "relu") h.data[$] = i.data[$] > 0 ? 1 : 0;
          else if (r === "sigmoid") {
            const _ = 1 / (1 + Math.exp(-i.data[$]));
            h.data[$] = _ * (1 - _);
          } else if (r === "tanh") {
            const _ = Math.tanh(i.data[$]);
            h.data[$] = 1 - _ * _;
          } else
            h.data[$] = 1;
        const b = p.mul(h), F = m.transpose().matmul(b);
        for (let $ = 0; $ < this.weights[l].size; $++)
          this.weights[l].data[$] -= a * F.data[$];
        for (let $ = 0; $ < this.biases[l].size; $++)
          this.biases[l].data[$] -= a * b.data[$];
        l > 0 && (p = b.matmul(this.weights[l].transpose()));
      }
    }
    return { loss: t / e.length };
  }
  /**
   * Evaluate a 2D grid of points for boundary visualization (e.g., XOR, Circles, Moons)
   */
  evaluateGrid(e = 30, o = 4) {
    const t = [], a = o * 2 / e, n = [], s = [];
    for (let f = 0; f < e; f++) {
      const c = [], u = o - f * a;
      s.push(u);
      for (let d = 0; d < e; d++) {
        const y = -o + d * a;
        f === 0 && n.push(y);
        const x = this.forward(new g([1, 2], [y, u]));
        c.push(x.data[0]);
      }
      t.push(c);
    }
    return { grid: t, xRange: n, yRange: s };
  }
}
class z {
  constructor() {
    w(this, "tracer", new D());
    // Assignment 3 exact architecture:
    // Conv2D(32, 3x3) -> MaxPool(2x2) -> Conv2D(64, 3x3) -> MaxPool(2x2) -> Conv2D(64, 3x3) -> Flatten -> Dense(64) -> Dense(10, softmax)
    w(this, "kernel1");
    w(this, "bias1");
    w(this, "kernel2");
    w(this, "bias2");
    w(this, "kernel3");
    w(this, "bias3");
    w(this, "dense1_W");
    w(this, "dense1_B");
    w(this, "dense2_W");
    w(this, "dense2_B");
    this.kernel1 = g.random([3, 3, 1, 32], -0.2, 0.2), this.bias1 = g.zeros([32]), this.kernel2 = g.random([3, 3, 32, 64], -0.15, 0.15), this.bias2 = g.zeros([64]), this.kernel3 = g.random([3, 3, 64, 64], -0.15, 0.15), this.bias3 = g.zeros([64]), this.dense1_W = g.random([576, 64], -0.1, 0.1), this.dense1_B = g.zeros([1, 64]), this.dense2_W = g.random([64, 10], -0.1, 0.1), this.dense2_B = g.zeros([1, 10]);
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
    const a = t.maxPool2d(2, 2);
    o && this.tracer.record({
      layerId: "pool1",
      layerName: "MaxPooling2D (2x2)",
      operation: "Spatial Downsampling",
      formula: "H_out = floor(26/2) = 13; W_out = 13",
      inputShapes: [t.shape],
      outputShape: a.shape,
      tensorPreview: Array.from(a.data.slice(0, 5)),
      fullOutput: a,
      pedagogicalInsight: "Preserves the most salient local features while reducing spatial dimensions by 75%."
    });
    const n = a.conv2d(this.kernel2, this.bias2, 1, 0).relu();
    o && this.tracer.record({
      layerId: "conv2",
      layerName: "Conv2D (64 filters, 3x3)",
      operation: "2D Convolution + ReLU",
      formula: "H_out = (13 - 3 + 0)/1 + 1 = 11; Params = (3*3*32 + 1)*64 = 18,496",
      inputShapes: [a.shape, this.kernel2.shape],
      outputShape: n.shape,
      tensorPreview: Array.from(n.data.slice(0, 5)),
      fullOutput: n,
      pedagogicalInsight: "Combines local features into mid-level parts (corners, loops, strokes)."
    });
    const s = n.maxPool2d(2, 2);
    o && this.tracer.record({
      layerId: "pool2",
      layerName: "MaxPooling2D (2x2)",
      operation: "Spatial Downsampling",
      formula: "H_out = floor(11/2) = 5; W_out = 5",
      inputShapes: [n.shape],
      outputShape: s.shape,
      tensorPreview: Array.from(s.data.slice(0, 5)),
      fullOutput: s,
      pedagogicalInsight: "Further downsamples to 5x5 feature grids."
    });
    const f = s.conv2d(this.kernel3, this.bias3, 1, 0).relu();
    o && this.tracer.record({
      layerId: "conv3",
      layerName: "Conv2D (64 filters, 3x3)",
      operation: "2D Convolution + ReLU",
      formula: "H_out = (5 - 3 + 0)/1 + 1 = 3; Params = (3*3*64 + 1)*64 = 36,928",
      inputShapes: [s.shape, this.kernel3.shape],
      outputShape: f.shape,
      tensorPreview: Array.from(f.data.slice(0, 5)),
      fullOutput: f,
      pedagogicalInsight: "High-level digit shape representations."
    });
    const c = f.flatten(), y = new g([1, 576], c.data).matmul(this.dense1_W).add(this.dense1_B).relu().matmul(this.dense2_W).add(this.dense2_B), x = y.softmax(-1), p = Array.from(x.data);
    let l = 0, i = -1;
    for (let r = 0; r < 10; r++)
      p[r] > i && (i = p[r], l = r);
    return o && this.tracer.record({
      layerId: "digit_probs",
      layerName: "Softmax Classification Head",
      operation: "Softmax(logits) -> Argmax",
      formula: "P(digit=k) = exp(z_k) / sum(exp(z)); Predicted = argmax_k(P)",
      inputShapes: [y.shape],
      outputShape: [10],
      tensorPreview: p,
      pedagogicalInsight: `Final digit classification with top prediction ${l} (${(i * 100).toFixed(1)}%).`
    }), {
      probabilities: p,
      predictedDigit: l,
      conv1Out: t,
      pool1Out: a,
      conv2Out: n,
      pool2Out: s,
      conv3Out: f
    };
  }
  loadWeightsFromJSON(e) {
    e.kernel1 && (this.kernel1 = new g([3, 3, 1, 32], e.kernel1)), e.bias1 && (this.bias1 = new g([32], e.bias1)), e.dense2_W && (this.dense2_W = new g([64, 10], e.dense2_W)), e.dense2_B && (this.dense2_B = new g([1, 10], e.dense2_B));
  }
}
class N {
  constructor(e = 4, o = 4) {
    w(this, "hiddenSize");
    w(this, "inputSize");
    w(this, "tracer", new D());
    // Gates: f, i, c, o (combined or separate)
    w(this, "W_f");
    w(this, "U_f");
    w(this, "b_f");
    w(this, "W_i");
    w(this, "U_i");
    w(this, "b_i");
    w(this, "W_c");
    w(this, "U_c");
    w(this, "b_c");
    w(this, "W_o");
    w(this, "U_o");
    w(this, "b_o");
    this.inputSize = e, this.hiddenSize = o;
    const t = 0.5;
    this.W_f = g.random([e, o], -t, t), this.U_f = g.random([o, o], -t, t), this.b_f = g.ones([1, o]), this.W_i = g.random([e, o], -t, t), this.U_i = g.random([o, o], -t, t), this.b_i = g.zeros([1, o]), this.W_c = g.random([e, o], -t, t), this.U_c = g.random([o, o], -t, t), this.b_c = g.zeros([1, o]), this.W_o = g.random([e, o], -t, t), this.U_o = g.random([o, o], -t, t), this.b_o = g.zeros([1, o]);
  }
  step(e, o, t) {
    const a = e.matmul(this.W_f).add(o.matmul(this.U_f)).add(this.b_f).sigmoid(), n = e.matmul(this.W_i).add(o.matmul(this.U_i)).add(this.b_i).sigmoid(), s = e.matmul(this.W_c).add(o.matmul(this.U_c)).add(this.b_c).tanh(), f = a.mul(t).add(n.mul(s)), c = e.matmul(this.W_o).add(o.matmul(this.U_o)).add(this.b_o).sigmoid();
    return { h: c.mul(f.tanh()), c: f, gates: { f: a, i: n, c_tilde: s, o: c } };
  }
  unroll(e) {
    this.tracer.clear();
    const o = [];
    let t = g.zeros([1, this.hiddenSize]), a = g.zeros([1, this.hiddenSize]);
    for (let n = 0; n < e.length; n++) {
      const s = e[n], f = new g([1, this.inputSize], s.vector), c = this.step(f, t, a);
      t = c.h, a = c.c;
      const u = {
        t: n,
        inputToken: s.token,
        x_t: Array.from(f.data),
        f_gate: Array.from(c.gates.f.data),
        i_gate: Array.from(c.gates.i.data),
        c_tilde: Array.from(c.gates.c_tilde.data),
        c_t: Array.from(a.data),
        o_gate: Array.from(c.gates.o.data),
        h_t: Array.from(t.data)
      };
      o.push(u), this.tracer.record({
        layerId: `lstm_step_${n}`,
        layerName: `LSTM Step t=${n} ("${s.token}")`,
        operation: "Recurrent Cell State Transition",
        formula: "c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t; h_t = o_t ⊙ tanh(c_t)",
        inputShapes: [f.shape, t.shape, a.shape],
        outputShape: t.shape,
        tensorPreview: Array.from(t.data),
        pedagogicalInsight: `Step t=${n}: Forget gate retained ${(u.f_gate[0] * 100).toFixed(0)}% of prior cell memory while input gate injected ${(u.i_gate[0] * 100).toFixed(0)}% of new token information.`
      });
    }
    return o;
  }
}
class O {
  constructor(e = 8, o = 2) {
    w(this, "d_model");
    w(this, "d_k");
    w(this, "numHeads");
    w(this, "tracer", new D());
    w(this, "W_q");
    w(this, "W_k");
    w(this, "W_v");
    w(this, "W_o");
    this.d_model = e, this.numHeads = o, this.d_k = Math.floor(e / o);
    const t = Math.sqrt(2 / e);
    this.W_q = g.random([e, e], -t, t), this.W_k = g.random([e, e], -t, t), this.W_v = g.random([e, e], -t, t), this.W_o = g.random([e, e], -t, t);
  }
  /**
   * Forward pass over token embeddings matrix X of shape [seqLen, d_model]
   */
  forward(e, o = !0, t = !0) {
    t && this.tracer.clear();
    const a = e.shape[0], n = e.matmul(this.W_q), s = e.matmul(this.W_k), f = e.matmul(this.W_v), c = 1 / Math.sqrt(this.d_model), u = s.transpose(), d = n.matmul(u).mul(c);
    if (o)
      for (let i = 0; i < a; i++)
        for (let r = i + 1; r < a; r++)
          d.set(-1e9, i, r);
    const y = d.softmax(-1), p = y.matmul(f).matmul(this.W_o);
    t && this.tracer.record({
      layerId: "self_attention",
      layerName: `Self-Attention (d_model=${this.d_model}, seqLen=${a})`,
      operation: "Scaled Dot-Product Attention",
      formula: "Attention(Q,K,V) = softmax(Q K^T / √d_k + Mask) V",
      inputShapes: [e.shape, this.W_q.shape],
      outputShape: p.shape,
      tensorPreview: Array.from(y.data.slice(0, 6)),
      pedagogicalInsight: `Calculated ${a}x${a} pairwise attention affinity matrix; tokens dynamically aggregate information from allowed preceding contexts.`
    });
    const l = [];
    for (let i = 0; i < a; i++) {
      const r = [];
      for (let m = 0; m < a; m++)
        r.push(y.get(i, m));
      l.push(r);
    }
    return {
      attentionWeights: l,
      output: p
    };
  }
}
class U {
  constructor(e = 10, o = 2) {
    w(this, "tracer", new D());
    // Encoder: [D_in, 16] -> [16, 2] (Latent)
    w(this, "W_enc1");
    w(this, "b_enc1");
    w(this, "W_enc2");
    w(this, "b_enc2");
    // Decoder: [2, 16] -> [16, D_in] (Reconstruction)
    w(this, "W_dec1");
    w(this, "b_dec1");
    w(this, "W_dec2");
    w(this, "b_dec2");
    this.inDim = e, this.latentDim = o;
    const t = 0.3;
    this.W_enc1 = g.random([e, 16], -t, t), this.b_enc1 = g.zeros([1, 16]), this.W_enc2 = g.random([16, o], -t, t), this.b_enc2 = g.zeros([1, o]), this.W_dec1 = g.random([o, 16], -t, t), this.b_dec1 = g.zeros([1, 16]), this.W_dec2 = g.random([16, e], -t, t), this.b_dec2 = g.zeros([1, e]);
  }
  encode(e) {
    return e.matmul(this.W_enc1).add(this.b_enc1).relu().matmul(this.W_enc2).add(this.b_enc2);
  }
  decode(e) {
    return e.matmul(this.W_dec1).add(this.b_dec1).relu().matmul(this.W_dec2).add(this.b_dec2).sigmoid();
  }
  forward(e, o = !0) {
    o && this.tracer.clear();
    const t = this.encode(e), a = this.decode(t);
    let n = 0;
    for (let s = 0; s < e.size; s++) {
      const f = e.data[s] - a.data[s];
      n += f * f;
    }
    return n /= e.size, o && this.tracer.record({
      layerId: "latent_bottleneck",
      layerName: `Latent Space (dim=${this.latentDim})`,
      operation: "Nonlinear Dimensionality Compression",
      formula: "z = W_2 * relu(W_1 x + b_1) + b_2",
      inputShapes: [e.shape],
      outputShape: t.shape,
      tensorPreview: Array.from(t.data),
      pedagogicalInsight: `Compressed input from ${this.inDim} dimensions to ${this.latentDim} latent coordinates with MSE=${n.toFixed(4)}.`
    }), { latent: t, reconstructed: a, mse: n };
  }
}
class A {
  static renderNetwork(e, o = 640, t = 360, a = -1, n = "forward") {
    const s = e.length, f = o / (s + 1), c = [], u = 68, y = t - u - 55;
    e.forEach((m, h) => {
      const b = (h + 1) * f, F = Math.min(m.outShape[m.outShape.length - 1] || 4, 8), $ = y / (F + 1), _ = [];
      for (let S = 0; S < F; S++)
        _.push({
          layerIndex: h,
          nodeIndex: S,
          x: b,
          y: u + (S + 1) * $,
          label: `${m.name} [${S}]`
        });
      c.push(_);
    });
    const x = n === "backward", p = x ? "#C8102E" : "#0F8B8D", l = x ? "reverseFlow" : "flowPulse";
    let i = `<svg viewBox="0 0 ${o} ${t}" width="100%" height="${t}" preserveAspectRatio="xMidYMid meet" style="background: radial-gradient(circle at 50% 50%, #0F172A 0%, #020617 100%); border-radius:12px; font-family:system-ui, -apple-system, sans-serif; display:block;">`;
    i += `<defs>
      <linearGradient id="edgeGradFwd" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="#0F8B8D" stop-opacity="0.5"/>
        <stop offset="100%" stop-color="#6B2D7B" stop-opacity="0.7"/>
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
        .flow-line { stroke-dasharray: 6 3; animation: ${l} 0.9s linear infinite; }
        .active-pulse { animation: nodePulse 1.2s ease-in-out infinite alternate; }
        @keyframes nodePulse { from { transform: scale(1); filter: drop-shadow(0 0 2px #FFD700); } to { transform: scale(1.2); filter: drop-shadow(0 0 8px #FFD700); } }
      </style>
    </defs>`;
    for (let m = 0; m < c.length - 1; m++) {
      const h = c[m], b = c[m + 1], F = m === a || m + 1 === a, $ = x ? "url(#edgeGradBwd)" : "url(#edgeGradFwd)";
      for (const _ of h)
        for (const S of b) {
          const k = F ? 'class="flow-line"' : "", v = F ? "2.2" : "1.0", T = F ? "0.9" : "0.3";
          i += `<line x1="${_.x}" y1="${_.y}" x2="${S.x}" y2="${S.y}" stroke="${$}" stroke-width="${v}" stroke-opacity="${T}" ${k}/>`;
        }
    }
    e.forEach((m, h) => {
      const b = (h + 1) * f, F = h === a, $ = F ? "#FFD700" : "#E2E8F0";
      let _ = m.name;
      _.length > 20 && (_ = _.slice(0, 18) + "…"), i += `<text x="${b}" y="22" fill="${$}" font-size="11.5" font-weight="700" text-anchor="middle">${_}</text>`;
      const S = 76;
      i += `<rect x="${b - S / 2}" y="30" width="${S}" height="18" rx="4" fill="#1E293B" stroke="${F ? "#FFD700" : "#334155"}" stroke-width="1"/>`, i += `<text x="${b}" y="43" fill="#38BDF8" font-size="10" font-family="monospace" font-weight="600" text-anchor="middle">[${m.outShape.join("×")}]</text>`, m.paramsCount > 0 && (i += `<text x="${b}" y="${t - 36}" fill="#94A3B8" font-size="10" font-family="monospace" text-anchor="middle">${m.paramsCount.toLocaleString()} params</text>`);
    }), c.forEach((m, h) => {
      const b = h === a, F = b ? "#FFD700" : h === 0 ? "#0F8B8D" : h === c.length - 1 ? x ? "#C8102E" : "#10B981" : "#6B2D7B";
      for (const $ of m) {
        const _ = b ? 'filter="url(#glowGold)" class="active-pulse"' : "";
        i += `<circle cx="${$.x}" cy="${$.y}" r="${b ? 11 : 8.5}" fill="${F}" stroke="#ffffff" stroke-width="1.8" ${_}/>`;
      }
    });
    const r = 190;
    return i += `<g transform="translate(${o / 2 - r / 2}, ${t - 24})">
      <rect width="${r}" height="20" rx="10" fill="#1E293B" stroke="${p}" stroke-width="1.2"/>
      <text x="${r / 2}" y="14" fill="${p}" font-size="9.5" font-weight="700" text-anchor="middle" letter-spacing="0.5">
        ${x ? "◀ BACKPROPAGATION GRADIENTS" : "FORWARD INFERENCE FLOW ▶"}
      </text>
    </g>`, i += "</svg>", i;
  }
}
class M {
  /**
   * Helper to set up HiDPI canvas for ultra-sharp Retina rendering
   */
  static setupHiDPI(e) {
    const o = window.devicePixelRatio || 1, t = e.getBoundingClientRect(), a = t.width || e.width, n = t.height || e.height;
    (e.width !== a * o || e.height !== n * o) && (e.width = a * o, e.height = n * o);
    const s = e.getContext("2d");
    return s && s.setTransform(o, 0, 0, o, 0, 0), s;
  }
  /**
   * 1. Decision Boundary & Active Gradient Heatmap
   */
  static renderDecisionBoundary(e, o, t) {
    const a = e.getContext("2d");
    if (!a) return;
    const n = e.width, s = e.height, f = o.length, c = o[0].length, u = n / c, d = s / f;
    for (let y = 0; y < f; y++)
      for (let x = 0; x < c; x++) {
        const p = o[y][x], l = Math.floor(p * 220 + (1 - p) * 15), i = Math.floor((1 - Math.abs(p - 0.5) * 2) * 160), r = Math.floor((1 - p) * 220 + p * 25);
        a.fillStyle = `rgb(${l}, ${i}, ${r})`, a.fillRect(x * u, y * d, u + 1, d + 1);
      }
    if (t)
      for (const y of t) {
        const x = (y.x + 4) / 8 * n, p = (4 - y.y) / 8 * s;
        a.beginPath(), a.arc(x, p, 7, 0, Math.PI * 2), a.fillStyle = y.label === 1 ? "#FFD700" : "#FFFFFF", a.strokeStyle = "#0B1329", a.lineWidth = 2.5, a.fill(), a.stroke();
      }
  }
  /**
   * 2. CNN Interactive 3x3 Sliding Kernel Visualizer (Stanford CS231n / CNN Explainer style)
   */
  static renderCNNKernelSlide(e, o, t, a, n = 0.05) {
    var k;
    const s = e.getContext("2d");
    if (!s) return { activation: 0, formulaText: "" };
    const f = e.width, c = e.height;
    s.fillStyle = "#020617", s.fillRect(0, 0, f, c);
    const u = 20, d = 38, y = 180, x = o.length, p = o[0].length, l = y / x;
    s.fillStyle = "#38BDF8", s.font = "bold 12px monospace", s.fillText("INPUT DIGIT PIXELS (28×28)", u, 22);
    for (let v = 0; v < x; v++)
      for (let T = 0; T < p; T++) {
        const B = o[v][T], E = Math.floor(B * 255);
        s.fillStyle = `rgb(${E}, ${E}, ${E})`, s.fillRect(u + T * l, d + v * l, l - 0.5, l - 0.5);
      }
    const i = a.row, r = a.col;
    s.strokeStyle = "#FFD700", s.lineWidth = 2.5, s.strokeRect(u + r * l, d + i * l, l * 3, l * 3);
    const m = 225, h = 38, b = 240, F = 180;
    s.fillStyle = "#0F172A", s.strokeStyle = "#334155", s.lineWidth = 1, s.fillRect(m, h, b, F), s.strokeRect(m, h, b, F), s.fillStyle = "#FFD700", s.font = "bold 11px monospace", s.fillText("3×3 RECEPTIVE FIELD CALCULATION", m + 10, h + 18);
    let $ = n;
    const _ = 48;
    for (let v = 0; v < 3; v++)
      for (let T = 0; T < 3; T++) {
        const B = ((k = o[i + v]) == null ? void 0 : k[r + T]) || 0, E = t[v][T], R = B * E;
        $ += R;
        const W = m + 12 + T * (_ + 6), L = h + 28 + v * (_ + 4);
        s.fillStyle = "#1E293B", s.fillRect(W, L, _, _), s.strokeStyle = "#475569", s.strokeRect(W, L, _, _), s.fillStyle = "#E2E8F0", s.font = "10px monospace", s.textAlign = "center", s.fillText(`x:${B.toFixed(1)}`, W + _ / 2, L + 16), s.fillStyle = "#38BDF8", s.fillText(`w:${E.toFixed(1)}`, W + _ / 2, L + 34);
      }
    s.textAlign = "left";
    const S = Math.max(0, $);
    return s.fillStyle = "#F8FAFC", s.font = "12px monospace", s.fillText(`Linear Sum z = Σ(x_i·w_i) + b = ${$.toFixed(3)}`, 20, 245), s.fillStyle = "#10B981", s.fillText(`Feature Activation = ReLU(z) = ${S.toFixed(3)}`, 20, 270), s.fillStyle = "#94A3B8", s.font = "11px monospace", s.fillText(`Output location: (${i}, ${r}) in 26×26 feature map`, 20, 295), {
      activation: S,
      formulaText: `z = Σ x_i·w_i + ${n.toFixed(2)} = ${$.toFixed(3)} → ReLU = ${S.toFixed(3)}`
    };
  }
  /**
   * 3. LSTM Memory Conveyor Belt & 4-Gate Anatomy
   */
  static renderLSTMConveyorBelt(e, o, t, a) {
    const n = e.getContext("2d");
    if (!n) return;
    const s = e.width, f = e.height;
    n.fillStyle = "#020617", n.fillRect(0, 0, s, f), n.fillStyle = "#38BDF8", n.font = "bold 13px monospace", n.fillText(`LSTM RECURRENT CELL ANATOMY (Step t=${o}: "${t}")`, 20, 28), n.strokeStyle = "#6B2D7B", n.lineWidth = 5, n.beginPath(), n.moveTo(20, 70), n.lineTo(s - 20, 70), n.stroke(), n.fillStyle = "#FFD700", n.font = "12px monospace", n.fillText(`Cell State c_t = ${a.c.toFixed(3)} (Constant Error Carousel)`, 30, 58);
    const c = [
      { name: "Forget (f_t)", val: a.f, color: "#C8102E", desc: "Retention" },
      { name: "Input (i_t)", val: a.i, color: "#0F8B8D", desc: "Write weight" },
      { name: "Candidate (c̃_t)", val: a.c_tilde, color: "#38BDF8", desc: "New signal" },
      { name: "Output (o_t)", val: a.o, color: "#10B981", desc: "Exposure" }
    ], u = Math.min(100, (s - 60) / 4), d = 20;
    c.forEach((y, x) => {
      const p = d + x * (u + 12), l = 100, i = 130;
      n.fillStyle = "#0F172A", n.fillRect(p, l, u, i), n.strokeStyle = "#334155", n.strokeRect(p, l, u, i);
      const r = Math.min(i - 25, Math.max(6, Math.abs(y.val) * (i - 25)));
      n.fillStyle = y.color, n.fillRect(p + 5, l + i - r - 5, u - 10, r), n.fillStyle = "#FFF", n.font = "bold 10.5px sans-serif", n.fillText(y.name, p + 6, l + 18), n.fillStyle = "#FFD700", n.font = "bold 13px monospace", n.fillText(y.val.toFixed(2), p + 15, l + 75);
    }), n.fillStyle = "#F8FAFC", n.font = "12px monospace", n.fillText(`Hidden State: h_t = o_t ⊙ tanh(c_t) = ${a.h.toFixed(4)}`, 20, 265), n.fillStyle = "#94A3B8", n.font = "11px monospace", n.fillText(`Retention: ${(a.f * 100).toFixed(0)}% prior memory retained | ${(a.i * 100).toFixed(0)}% new candidate injected`, 20, 290);
  }
  /**
   * 4. Scaled Dot-Product Attention Matrix with Causal Mask
   */
  static renderAttentionMatrix(e, o, t, a = -1) {
    const n = e.getContext("2d");
    if (!n) return;
    const s = e.width, f = e.height, c = o.length, u = 90, d = 60, y = s - u - 20, x = f - d - 20, p = y / c, l = x / c;
    n.clearRect(0, 0, s, f), n.fillStyle = "#020617", n.fillRect(0, 0, s, f), n.fillStyle = "#38BDF8", n.font = "bold 13px monospace", n.fillText("ATTENTION WEIGHTS: Softmax(Q K^T / √d_k)", 20, 26);
    for (let i = 0; i < c; i++) {
      const r = i === a;
      for (let m = 0; m < c; m++) {
        const h = o[i][m], b = m > i;
        b ? n.fillStyle = "#0F172A" : n.fillStyle = r ? `rgba(255, 215, 0, ${Math.max(0.2, h)})` : `rgba(15, 139, 141, ${Math.max(0.12, h)})`, n.fillRect(u + m * p, d + i * l, p - 2, l - 2), p > 28 && (n.fillStyle = b ? "#334155" : h > 0.4 ? "#FFFFFF" : "#94A3B8", n.font = "10px monospace", n.textAlign = "center", n.textBaseline = "middle", n.fillText(
          b ? "—" : h.toFixed(2),
          u + m * p + p / 2,
          d + i * l + l / 2
        ));
      }
    }
    n.font = "12px sans-serif", n.textAlign = "right", n.textBaseline = "middle";
    for (let i = 0; i < c; i++) {
      const r = t[i] || `t_${i}`;
      n.fillStyle = i === a ? "#FFD700" : "#CBD5E1", n.fillText(r, u - 10, d + i * l + l / 2);
    }
    n.textAlign = "center", n.textBaseline = "bottom";
    for (let i = 0; i < c; i++) {
      const r = t[i] || `t_${i}`;
      n.fillText(r, u + i * p + p / 2, d - 8);
    }
  }
  /**
   * 5. 2D Latent Space Manifold & Feature Reconstruction (Autoencoders / VAE)
   */
  static renderLatentManifold(e, o, t, a, n) {
    const s = e.getContext("2d");
    if (!s) return;
    const f = e.width, c = e.height;
    s.fillStyle = "#020617", s.fillRect(0, 0, f, c), s.fillStyle = "#38BDF8", s.font = "bold 13px monospace", s.fillText("2D LATENT SPACE & FEATURE RECONSTRUCTION", 20, 24);
    const u = 200, d = 200, y = 20, x = 45, p = y + u / 2, l = x + d / 2;
    s.fillStyle = "#0F172A", s.fillRect(y, x, u, d), s.strokeStyle = "#334155", s.strokeRect(y, x, u, d), s.strokeStyle = "#1E293B";
    for (let S = -80; S <= 80; S += 25)
      s.beginPath(), s.moveTo(p + S, x), s.lineTo(p + S, x + d), s.stroke(), s.beginPath(), s.moveTo(y, l + S), s.lineTo(y + u, l + S), s.stroke();
    [
      { x: p - 40, y: l - 35, col: "#0F8B8D" },
      { x: p + 45, y: l + 40, col: "#6B2D7B" },
      { x: p + 35, y: l - 45, col: "#38BDF8" }
    ].forEach((S) => {
      s.fillStyle = S.col;
      for (let k = 0; k < 6; k++) {
        const v = Math.sin(k * 1.5) * 20, T = Math.cos(k * 1.5) * 18;
        s.beginPath(), s.arc(S.x + v, S.y + T, 3, 0, Math.PI * 2), s.fill();
      }
    });
    const r = p + o[0] * 40, m = l - o[1] * 40;
    s.beginPath(), s.arc(r, m, 7, 0, Math.PI * 2), s.fillStyle = "#FFD700", s.fill(), s.strokeStyle = "#FFFFFF", s.lineWidth = 2, s.stroke(), s.fillStyle = "#FFD700", s.font = "bold 11px monospace", s.fillText(`z = (${o[0].toFixed(2)}, ${o[1].toFixed(2)})`, y + 10, x + d + 18);
    const h = 245, b = 45, F = f - h - 25, $ = Math.min(8, t.length), _ = 22;
    s.fillStyle = "#E2E8F0", s.font = "bold 11px sans-serif", s.fillText("Feature Reconstruction Comparison:", h, b - 8);
    for (let S = 0; S < $; S++) {
      const k = b + S * _, v = t[S], T = a[S];
      s.fillStyle = "#94A3B8", s.font = "10px monospace", s.fillText(`dim ${S}:`, h, k + 10), s.fillStyle = "#0F8B8D", s.fillRect(h + 42, k, Math.max(3, v * (F - 45)), 7), s.fillStyle = "#FFD700", s.fillRect(h + 42, k + 9, Math.max(3, T * (F - 45)), 7);
    }
    s.font = "10px sans-serif", s.fillStyle = "#0F8B8D", s.fillRect(h, c - 42, 10, 10), s.fillStyle = "#E2E8F0", s.fillText("Original x", h + 16, c - 33), s.fillStyle = "#FFD700", s.fillRect(h + 85, c - 42, 10, 10), s.fillStyle = "#E2E8F0", s.fillText("Reconstructed x̂", h + 101, c - 33), s.fillStyle = "#10B981", s.font = "bold 12px monospace", s.fillText(`Reconstruction MSE: ${n.toFixed(4)}`, h, c - 14);
  }
  /**
   * 6. 3D Optimizer Loss Landscape (SGD vs Momentum vs Adam)
   */
  static renderOptimizerContour(e, o) {
    const t = e.getContext("2d");
    if (!t) return;
    const a = e.width, n = e.height;
    t.fillStyle = "#020617", t.fillRect(0, 0, a, n), t.fillStyle = "#38BDF8", t.font = "bold 13px monospace", t.fillText("LOSS LANDSCAPE TRAJECTORY (3D Ravine Contour)", 20, 26);
    const s = a / 2, f = n / 2 + 10;
    for (let d = 1; d <= 6; d++)
      t.strokeStyle = `rgba(15, 139, 141, ${0.12 * d})`, t.lineWidth = 1.3, t.beginPath(), t.ellipse(s, f, d * 32, d * 16, -Math.PI / 6, 0, Math.PI * 2), t.stroke();
    const c = Math.min(1, o % 20 / 19), u = 9;
    t.strokeStyle = "#C8102E", t.lineWidth = 2.2, t.beginPath(), t.moveTo(s - 130, f - 80);
    for (let d = 1; d <= u * c; d++) {
      const y = (d % 2 === 0 ? 38 : -38) * (1 - d / u);
      t.lineTo(s - 130 + d * 18, f - 80 + d * 10 + y);
    }
    t.stroke(), t.strokeStyle = "#D98E04", t.lineWidth = 2.2, t.beginPath(), t.moveTo(s - 130, f - 80);
    for (let d = 1; d <= u * c; d++) {
      const y = (d % 2 === 0 ? 14 : -14) * (1 - d / u);
      t.lineTo(s - 130 + d * 19, f - 80 + d * 10 + y);
    }
    t.stroke(), t.strokeStyle = "#10B981", t.lineWidth = 2.8, t.beginPath(), t.moveTo(s - 130, f - 80);
    for (let d = 1; d <= u * c; d++)
      t.lineTo(s - 130 + d * 20, f - 80 + d * 11);
    t.stroke(), t.font = "11px monospace", t.fillStyle = "#C8102E", t.fillText("● SGD: Oscillates on steep walls", 20, n - 55), t.fillStyle = "#D98E04", t.fillText("● Momentum: Dampens ravine oscillations", 20, n - 35), t.fillStyle = "#10B981", t.fillText("● Adam: Direct adaptive descent to global minimum", 20, n - 15);
  }
}
class G extends HTMLElement {
  constructor() {
    super(...arguments);
    w(this, "timer", null);
    w(this, "isPlaying", !1);
    w(this, "currentStep", 0);
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
    const o = this.getAttribute("model") || "mlp", t = this.normalizeModelType(o), a = this.getAttribute("title") || `Model Simulator: ${t.toUpperCase()}`, n = this.getAttribute("course") || "Harvard CSCI E-89 / E-89b SOTA";
    this.innerHTML = `
      <div class="omni-sim-card" style="font-family:system-ui, -apple-system, sans-serif; background:#0B1329; color:#F8FAFC; border:1px solid #1E293B; border-radius:12px; padding:22px; margin:20px 0; box-shadow:0 12px 30px -5px rgba(0,0,0,0.5);">
        <!-- Top Toolbar -->
        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #1E293B; padding-bottom:14px; margin-bottom:16px; flex-wrap:wrap; gap:12px;">
          <div>
            <span style="background:#0F8B8D; color:#fff; font-size:11px; font-weight:700; text-transform:uppercase; padding:3px 8px; border-radius:4px; letter-spacing:0.5px;">${n}</span>
            <h3 style="margin:6px 0 0 0; font-size:1.35rem; color:#FFFFFF;">${a}</h3>
          </div>
          <div style="display:flex; gap:10px; align-items:center;">
            <select class="sim-model-selector" style="background:#1E293B; color:#38BDF8; border:1px solid #475569; border-radius:6px; padding:7px 12px; font-size:12.5px; font-weight:600; cursor:pointer;">
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
            <button class="btn-play" style="background:#1E293B; color:#38BDF8; border:1px solid #38BDF8; border-radius:6px; padding:7px 14px; font-size:12.5px; font-weight:600; cursor:pointer;">▶ Auto</button>
            <button class="btn-step" style="background:#0F8B8D; color:#FFFFFF; border:none; border-radius:6px; padding:7px 16px; font-size:12.5px; font-weight:700; cursor:pointer; box-shadow:0 0 12px rgba(15,139,141,0.4);">Step Forward ⏭</button>
            <button class="btn-reset" style="background:#C8102E; color:#FFFFFF; border:none; border-radius:6px; padding:7px 12px; font-size:12.5px; cursor:pointer;">Reset</button>
          </div>
        </div>

        <!-- State Breadcrumb Bar -->
        <div class="state-breadcrumb" style="display:flex; gap:8px; align-items:center; margin-bottom:16px; background:#020617; padding:10px 14px; border-radius:8px; border:1px solid #1E293B; overflow-x:auto; font-size:11.5px; font-family:monospace;">
          <span style="color:#94A3B8; font-weight:700;">PHASE:</span>
          <span class="crumb-step" style="background:#0F8B8D; color:#fff; padding:3px 10px; border-radius:4px;">1. INFERENCE</span>
          <span style="color:#475569;">→</span>
          <span class="crumb-step" style="color:#64748B;">2. LOSS</span>
          <span style="color:#475569;">→</span>
          <span class="crumb-step" style="color:#64748B;">3. BACKPROP</span>
          <span style="color:#475569;">→</span>
          <span class="crumb-step" style="color:#64748B;">4. UPDATE</span>
        </div>

        <!-- Viewport (Spacious Layout: Diagram + Canvas) -->
        <div class="sim-viewport" style="display:grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr); gap:20px; align-items:stretch;">
          <div class="sim-diagram-panel" style="background:#020617; border-radius:10px; padding:14px; border:1px solid #1E293B; min-height:360px; display:flex; flex-direction:column;">
            <div style="font-size:12px; font-weight:700; color:#94A3B8; margin-bottom:10px; letter-spacing:0.5px;">ACTIVE NETWORK ARCHITECTURE & DATA FLOW</div>
            <div class="diagram-host" style="flex:1; display:flex; align-items:center; justify-content:center;"></div>
          </div>
          <div class="sim-vis-panel" style="background:#020617; border-radius:10px; padding:14px; border:1px solid #1E293B; display:flex; flex-direction:column; align-items:center; min-height:360px;">
            <div style="font-size:12px; font-weight:700; color:#94A3B8; margin-bottom:10px; width:100%; letter-spacing:0.5px;">LIVE ACTIVATIONS & COMPUTATIONAL MECHANICS</div>
            <canvas class="sim-canvas" width="460" height="320" style="border-radius:8px; background:#000; width:100%; height:320px; display:block;"></canvas>
            <div class="sim-metrics" style="width:100%; margin-top:10px; font-size:12px; font-family:monospace; color:#E2E8F0;"></div>
          </div>
        </div>

        <!-- Pedagogical Execution Trace -->
        <div class="sim-trace-panel" style="margin-top:18px; background:#020617; border:1px solid #1E293B; border-radius:8px; padding:14px;">
          <div style="font-size:11px; font-weight:700; color:#FFD700; margin-bottom:6px; letter-spacing:0.5px;">PEDAGOGICAL MATHEMATICAL STEP TRACE:</div>
          <div class="trace-output" style="font-size:12px; font-family:monospace; color:#A5B4FC; line-height:1.6;"></div>
        </div>
      </div>
    `, this.initModel(t);
    const s = this.querySelector(".sim-model-selector");
    s && (s.onchange = (f) => {
      const c = f.target.value;
      this.setAttribute("model", c);
    });
  }
  initModel(o) {
    const t = this.querySelector(".sim-canvas"), a = this.querySelector(".diagram-host"), n = this.querySelector(".trace-output"), s = this.querySelector(".btn-step"), f = this.querySelector(".btn-play"), c = this.querySelector(".btn-reset"), u = this.querySelector(".state-breadcrumb"), d = 620, y = 340;
    let x = () => {
    };
    if (o === "mlp") {
      const p = new H({
        layerSizes: [2, 6, 4, 1],
        activations: ["tanh", "tanh", "sigmoid"],
        learningRate: 0.15
      }), l = [[-2, -2], [-2, 2], [2, -2], [2, 2]], i = [[0], [1], [1], [0]], r = [
        { id: "in", name: "Input [x1, x2]", type: "dense", inShape: [2], outShape: [2], paramsCount: 0 },
        { id: "h1", name: "Hidden 1 (Tanh)", type: "dense", inShape: [2], outShape: [6], paramsCount: 18 },
        { id: "h2", name: "Hidden 2 (Tanh)", type: "dense", inShape: [6], outShape: [4], paramsCount: 28 },
        { id: "out", name: "Output (Sigmoid)", type: "dense", inShape: [4], outShape: [1], paramsCount: 5 }
      ], m = ["forward_1", "forward_2", "loss", "backward_out", "backward_hidden", "update"];
      x = () => {
        const h = this.currentStep % m.length, b = m[h];
        if (b === "update")
          for (let S = 0; S < 15; S++) p.trainStep(l, i);
        const F = b === "forward_1" ? 1 : b === "forward_2" ? 2 : b === "loss" || b === "backward_out" ? 3 : b === "backward_hidden" ? 1 : -1, $ = b.startsWith("backward") ? "backward" : "forward";
        a.innerHTML = A.renderNetwork(r, d, y, F, $);
        const { grid: _ } = p.evaluateGrid(30, 4);
        M.renderDecisionBoundary(
          t,
          _,
          l.map((S, k) => ({ x: S[0], y: S[1], label: i[k][0] }))
        ), u.innerHTML = `
          <span style="color:#94A3B8; font-weight:700;">STEP ${this.currentStep + 1}:</span>
          <span style="background:${$ === "backward" ? "#C8102E" : "#0F8B8D"}; color:#fff; padding:3px 10px; border-radius:4px;">
            ${b === "forward_1" ? "1. LAYER 1 LINEAR + TANH" : b === "forward_2" ? "2. LAYER 2 LINEAR + TANH" : b === "loss" ? "3. SIGMOID + MSE LOSS" : b === "backward_out" ? "4. BACKPROP δ_out" : b === "backward_hidden" ? "5. BACKPROP δ_hidden" : "6. WEIGHT GRADIENT UPDATE"}
          </span>
        `, b === "forward_1" ? n.innerHTML = "• <b>Forward Layer 1:</b> $z^{[1]} = W^{[1]} x + b^{[1]}$; $a^{[1]} = \\tanh(z^{[1]})$. Produces 6 hidden activations." : b === "forward_2" ? n.innerHTML = "• <b>Forward Layer 2:</b> $z^{[2]} = W^{[2]} a^{[1]} + b^{[2]}$; $a^{[2]} = \\tanh(z^{[2]})$. Intermediate nonlinear mapping." : b === "loss" ? n.innerHTML = "• <b>Output Activation & Loss:</b> $\\hat{y} = \\sigma(z^{[3]})$; $L = \\frac{1}{2}(\\hat{y} - y)^2$. Error computed against target." : b === "backward_out" ? n.innerHTML = "• <b>Output Delta:</b> $\\delta^{[3]} = (\\hat{y} - y) \\odot \\sigma'(z^{[3]})$. Sensitivities flow backwards in red." : b === "backward_hidden" ? n.innerHTML = "• <b>Hidden Delta:</b> $\\delta^{[l]} = ((W^{[l+1]})^T \\delta^{[l+1]}) \\odot \\tanh'(z^{[l]})$. Chain rule propagates credit." : n.innerHTML = "• <b>Weight Update:</b> $W^{[l]} \\leftarrow W^{[l]} - \\eta (a^{[l-1]})^T \\delta^{[l]}$. Decision boundary visibly adapts!", this.currentStep++;
      }, x();
    } else if (o === "lstm") {
      const p = new N(4, 4), i = ["Natural", "Language", "Processing", "Recurrent", "Memory"].map((m) => ({ token: m, vector: [0.6, -0.3, 0.7, -0.2] })), r = p.unroll(i);
      x = () => {
        const m = this.currentStep % r.length, h = r[m];
        a.innerHTML = A.renderNetwork([
          { id: "x", name: `Token Input x_${m}`, type: "dense", inShape: [4], outShape: [4], paramsCount: 0 },
          { id: "gates", name: "Gates [f, i, c̃, o]", type: "lstm_cell", inShape: [4], outShape: [16], paramsCount: 144 },
          { id: "c", name: "Cell State c_t", type: "lstm_cell", inShape: [4], outShape: [4], paramsCount: 0 },
          { id: "h", name: "Hidden State h_t", type: "lstm_cell", inShape: [4], outShape: [4], paramsCount: 0 }
        ], d, y, 1, "forward"), M.renderLSTMConveyorBelt(t, h.t, h.inputToken, {
          f: h.f_gate[0],
          i: h.i_gate[0],
          c_tilde: h.c_tilde[0],
          o: h.o_gate[0],
          c: h.c_t[0],
          h: h.h_t[0]
        }), u.innerHTML = `
          <span style="color:#94A3B8; font-weight:700;">SEQUENCE STEP:</span>
          <span style="background:#0F8B8D; color:#fff; padding:3px 10px; border-radius:4px;">Token ${m + 1}/${r.length}: "${h.inputToken}"</span>
        `, n.innerHTML = `• <b>Step t=${h.t} ("${h.inputToken}"):</b><br>&nbsp;&nbsp;1. Forget gate: $f_t = \\sigma(W_f x_t + U_f h_{t-1} + b_f) = ${h.f_gate[0].toFixed(3)}$ (Retains ${(h.f_gate[0] * 100).toFixed(0)}% memory)<br>&nbsp;&nbsp;2. Input gate: $i_t = \\sigma(W_i x_t + U_i h_{t-1} + b_i) = ${h.i_gate[0].toFixed(3)}$<br>&nbsp;&nbsp;3. Cell update: $c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde{c}_t = ${h.c_t[0].toFixed(3)}$ (No vanishing gradient!)<br>&nbsp;&nbsp;4. Hidden emission: $h_t = o_t \\odot \\tanh(c_t) = ${h.h_t[0].toFixed(3)}$`, this.currentStep++;
      }, x();
    } else if (o === "cnn") {
      new z();
      const p = Array(28).fill(0).map(() => Array(28).fill(0));
      for (let i = 5; i < 23; i++)
        p[i][13] = 0.9, p[i][14] = 1, p[i][15] = 0.8;
      p[5][12] = 0.6;
      const l = [
        [0.2, 0.8, -0.4],
        [-0.5, 1.2, -0.5],
        [-0.4, 0.8, 0.2]
      ];
      x = () => {
        const i = 4 + this.currentStep % 18, r = 8 + Math.floor(this.currentStep / 2) % 12;
        a.innerHTML = A.renderNetwork([
          { id: "in", name: "Input Image", type: "conv2d", inShape: [28, 28, 1], outShape: [28, 28, 1], paramsCount: 0 },
          { id: "conv1", name: "Conv2D (3x3)", type: "conv2d", inShape: [28, 28, 1], outShape: [26, 26, 32], paramsCount: 320 },
          { id: "pool", name: "MaxPooling2D", type: "maxpool2d", inShape: [26, 26, 32], outShape: [13, 13, 32], paramsCount: 0 },
          { id: "out", name: "Dense Head", type: "dense", inShape: [576], outShape: [10], paramsCount: 37578 }
        ], d, y, 1, "forward");
        const m = M.renderCNNKernelSlide(t, p, l, { row: i, col: r });
        u.innerHTML = `
          <span style="color:#94A3B8; font-weight:700;">KERNEL SLIDE:</span>
          <span style="background:#FFD700; color:#0B1329; font-weight:700; padding:3px 10px; border-radius:4px;">Receptive Window [${i}:${i + 3}, ${r}:${r + 3}]</span>
        `, n.innerHTML = `• <b>Convolving at position (${i}, ${r}):</b><br>&nbsp;&nbsp;1. 9 Pixel Multiplications: $z = \\sum_{i=1}^3 \\sum_{j=1}^3 x_{i,j} \\cdot w_{i,j} + b = ${m.formulaText}$<br>&nbsp;&nbsp;2. Parameter Count Law: $(K_h \\cdot K_w \\cdot C_{in} + 1) \\cdot C_{out} = (3 \\cdot 3 \\cdot 1 + 1) \\cdot 32 = \\mathbf{320\\text{ params}}$<br>&nbsp;&nbsp;3. Spatial Shrinkage: $H_{out} = \\lfloor(28 - 3 + 0)/1 + 1\\rfloor = \\mathbf{26}$. Output map shape: $\\mathbf{26 \\times 26 \\times 32}$.`, this.currentStep++;
      }, x();
    } else if (o === "ngrams") {
      const p = ["deep", "learning", "models", "process", "language", "tokens", "with", "neural", "attention", "unknown_1", "unknown_2"], l = { deep: 0, learning: 1, models: 2, process: 3, language: 4, tokens: 5, with: 6, neural: 7, attention: 8, "<UNK>": 9 }, i = { deep: 1, learning: 1, models: 1, language: 1, "<UNK>": 0 };
      x = () => {
        const r = p[this.currentStep % p.length], m = !(r in l) || r.startsWith("unknown");
        m ? i["<UNK>"] = (i["<UNK>"] || 0) + 1 : i[r] = (i[r] || 0) + 1, a.innerHTML = A.renderNetwork([
          { id: "text", name: `Token: "${r}"`, type: "dense", inShape: [1], outShape: [1], paramsCount: 0 },
          { id: "hash", name: "Vocabulary Hash Table", type: "dense", inShape: [1], outShape: [10], paramsCount: 0 },
          { id: "bow", name: "BoW Frequency Vector", type: "dense", inShape: [10], outShape: [10], paramsCount: 0 }
        ], d, y, m ? 2 : 1, m ? "backward" : "forward");
        const h = t.getContext("2d");
        h && (h.fillStyle = "#020617", h.fillRect(0, 0, t.width, t.height), h.fillStyle = "#38BDF8", h.font = "bold 13px monospace", h.fillText(`STREAMING BoW COUNTER (Active: "${r}")`, 20, 26), Object.keys(i).forEach((F, $) => {
          const _ = F === "<UNK>";
          h.fillStyle = _ ? "#C8102E" : "#0F8B8D", h.fillRect(20, 48 + $ * 26, i[F] * 32, 19), h.fillStyle = "#FFF", h.font = "11px monospace", h.fillText(`${F}: ${i[F]}`, 28, 62 + $ * 26);
        })), u.innerHTML = `
          <span style="color:#94A3B8; font-weight:700;">TOKEN DISPATCH:</span>
          <span style="background:${m ? "#C8102E" : "#0F8B8D"}; color:#fff; padding:3px 10px; border-radius:4px;">"${r}" → ${m ? "<UNK> ROUTE" : "VOCAB HIT"}</span>
        `, n.innerHTML = `• <b>Token Processing:</b> "${r}"<br>` + (m ? `&nbsp;&nbsp;⚠️ <b>OOV Firewall Triggered:</b> Word not in training vocabulary. Routed to <code>&lt;UNK&gt;</code> (total count: ${i["<UNK>"]}). Vector length remains strictly fixed.` : `&nbsp;&nbsp;✅ <b>In-Vocabulary Match:</b> Incremented frequency count for index <code>${l[r]}</code>.`), this.currentStep++;
      }, x();
    } else if (o === "embeddings") {
      const p = [
        { u: "king", v: "man", w: "woman", target: "queen", desc: "vec(king) - vec(man) + vec(woman) ≈ vec(queen)" },
        { u: "paris", v: "france", w: "italy", target: "rome", desc: "vec(paris) - vec(france) + vec(italy) ≈ vec(rome)" },
        { u: "walking", v: "walk", w: "swim", target: "swimming", desc: "vec(walking) - vec(walk) + vec(swim) ≈ vec(swimming)" }
      ];
      x = () => {
        const l = p[this.currentStep % p.length];
        a.innerHTML = A.renderNetwork([
          { id: "onehot", name: "One-Hot Words", type: "dense", inShape: [1e4], outShape: [300], paramsCount: 3e6 },
          { id: "embed", name: "Embedding Space", type: "dense", inShape: [300], outShape: [300], paramsCount: 0 },
          { id: "proj", name: "2D t-SNE Projection", type: "dense", inShape: [300], outShape: [2], paramsCount: 600 }
        ], d, y, 1, "forward");
        const i = t.getContext("2d");
        if (i) {
          i.fillStyle = "#020617", i.fillRect(0, 0, t.width, t.height), i.fillStyle = "#38BDF8", i.font = "bold 13px monospace", i.fillText("WORD2VEC VECTOR SPACE PROJECTION", 20, 26);
          const r = t.width / 2, m = t.height / 2;
          i.strokeStyle = "#334155", i.lineWidth = 1, i.beginPath(), i.moveTo(r, 0), i.lineTo(r, t.height), i.moveTo(0, m), i.lineTo(t.width, m), i.stroke();
          const h = [
            { label: l.u, x: r - 90, y: m - 60, col: "#38BDF8" },
            { label: l.v, x: r - 110, y: m + 40, col: "#94A3B8" },
            { label: l.w, x: r + 60, y: m + 50, col: "#0F8B8D" },
            { label: l.target, x: r + 80, y: m - 50, col: "#FFD700" }
          ];
          h.forEach((b) => {
            i.beginPath(), i.arc(b.x, b.y, 8, 0, Math.PI * 2), i.fillStyle = b.col, i.fill(), i.fillStyle = "#FFF", i.font = "bold 12px sans-serif", i.fillText(b.label, b.x + 10, b.y + 4);
          }), i.strokeStyle = "#FFD700", i.lineWidth = 2.5, i.setLineDash([5, 3]), i.beginPath(), i.moveTo(h[0].x, h[0].y), i.lineTo(h[3].x, h[3].y), i.stroke(), i.setLineDash([]);
        }
        u.innerHTML = `
          <span style="color:#94A3B8; font-weight:700;">VECTOR ARITHMETIC:</span>
          <span style="background:#0F8B8D; color:#fff; padding:3px 10px; border-radius:4px;">${l.desc}</span>
        `, n.innerHTML = `• <b>Semantic Geometry:</b> <code>${l.desc}</code><br>&nbsp;&nbsp;Cosine similarity $\\cos(\\theta) = \\frac{u \\cdot v}{\\|u\\| \\|v\\|} = 0.884$. Word2Vec captures relational linear translations across concepts.`, this.currentStep++;
      }, x();
    } else if (o === "autoencoder") {
      const p = new U(10, 2);
      x = () => {
        const l = g.random([1, 10], 0.1, 0.9), i = p.forward(l, !0);
        a.innerHTML = A.renderNetwork([
          { id: "in", name: "Input x", type: "dense", inShape: [10], outShape: [10], paramsCount: 0 },
          { id: "enc", name: "Encoder Dense", type: "dense", inShape: [10], outShape: [16], paramsCount: 160 },
          { id: "z", name: "Latent Bottleneck", type: "latent_space", inShape: [16], outShape: [2], paramsCount: 32 },
          { id: "dec", name: "Decoder Dense", type: "dense", inShape: [2], outShape: [16], paramsCount: 32 },
          { id: "out", name: "Reconstruction x̂", type: "dense", inShape: [16], outShape: [10], paramsCount: 160 }
        ], d, y, 2, "forward"), M.renderLatentManifold(
          t,
          [i.latent.data[0], i.latent.data[1]],
          Array.from(l.data),
          Array.from(i.reconstructed.data),
          i.mse
        ), u.innerHTML = `
          <span style="color:#94A3B8; font-weight:700;">BOTTLENECK COMPRESSION:</span>
          <span style="background:#6B2D7B; color:#fff; padding:3px 10px; border-radius:4px;">10D Input → 2D Latent Manifold → 10D Reconstruction</span>
        `, n.innerHTML = `• <b>Latent Coordinates:</b> $z = [${i.latent.data[0].toFixed(3)}, ${i.latent.data[1].toFixed(3)}]$<br>&nbsp;&nbsp;Reconstruction error (MSE) = $\\mathbf{${i.mse.toFixed(4)}}$. The 2D bottleneck compresses high-dimensional variance into an interpretable manifold.`, this.currentStep++;
      }, x();
    } else if (o === "optimizers")
      x = () => {
        a.innerHTML = A.renderNetwork([
          { id: "loss", name: "Loss L(w)", type: "dense", inShape: [2], outShape: [1], paramsCount: 0 },
          { id: "grad", name: "Gradient ∇L", type: "dense", inShape: [2], outShape: [2], paramsCount: 0 },
          { id: "mom", name: "Momentum v_t", type: "dense", inShape: [2], outShape: [2], paramsCount: 0 },
          { id: "adam", name: "Adam m_t / √v_t", type: "dense", inShape: [2], outShape: [2], paramsCount: 0 }
        ], d, y, 2, "forward"), M.renderOptimizerContour(t, this.currentStep), u.innerHTML = `
          <span style="color:#94A3B8; font-weight:700;">OPTIMIZER RACE:</span>
          <span style="background:#0F8B8D; color:#fff; padding:3px 10px; border-radius:4px;">Step ${this.currentStep + 1}: Adam vs Momentum vs SGD</span>
        `, n.innerHTML = `• <b>Adam Update Step ${this.currentStep + 1}:</b><br>&nbsp;&nbsp;1. 1st Moment: $m_t = \\beta_1 m_{t-1} + (1 - \\beta_1) g_t$<br>&nbsp;&nbsp;2. 2nd Moment: $v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2$<br>&nbsp;&nbsp;3. Adaptive Descent: $w_{t+1} = w_t - \\frac{\\alpha}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t$. Adam descends rapidly along flat ravines.`, this.currentStep++;
      }, x();
    else if (o === "transformer") {
      const p = new O(8, 2), l = ["The", "neural", "network", "attends", "to", "tokens"], i = g.random([l.length, 8], -0.5, 0.5);
      x = () => {
        const r = this.currentStep % l.length, m = p.forward(i, !0, !0);
        a.innerHTML = A.renderNetwork([
          { id: "emb", name: "Token Embedding", type: "embedding", inShape: [6], outShape: [6, 8], paramsCount: 48 },
          { id: "qkv", name: "Q, K, V Linear", type: "dense", inShape: [6, 8], outShape: [6, 8], paramsCount: 192 },
          { id: "attn", name: "Attention Matrix", type: "self_attention", inShape: [6, 8], outShape: [6, 6], paramsCount: 0 },
          { id: "out", name: "Output Projection", type: "dense", inShape: [6, 8], outShape: [6, 8], paramsCount: 64 }
        ], d, y, 2, "forward"), M.renderAttentionMatrix(t, m.attentionWeights, l, r), u.innerHTML = `
          <span style="color:#94A3B8; font-weight:700;">QUERY FOCUS:</span>
          <span style="background:#FFD700; color:#0B1329; font-weight:700; padding:3px 10px; border-radius:4px;">Token ${r + 1}: "${l[r]}"</span>
        `, n.innerHTML = `• <b>Active Query:</b> Token "${l[r]}" attends to preceding context tokens:<br>&nbsp;&nbsp;1. Scaled Affinity: $S_{${r}, j} = \\frac{q_{${r}} \\cdot k_j^T}{\\sqrt{d_k}}$<br>&nbsp;&nbsp;2. Causal Masking: Future positions $(j > ${r})$ set to $-\\infty$ (shown as —).<br>&nbsp;&nbsp;3. Normalization: $\\alpha_{${r}, j} = \\operatorname{softmax}(S_{${r}, j})$. Context output aggregates $V$.`, this.currentStep++;
      }, x();
    } else if (o === "lda") {
      const p = [
        { name: "Topic 1 (NLP / Deep Learning)", words: ["neural", "transformer", "attention", "gradient"], color: "#0F8B8D" },
        { name: "Topic 2 (Finance & Risk)", words: ["return", "portfolio", "volatility", "arbitrage"], color: "#6B2D7B" },
        { name: "Topic 3 (Clinical / Healthcare)", words: ["patient", "treatment", "diagnosis", "trial"], color: "#D98E04" }
      ];
      x = () => {
        const l = this.currentStep % p.length, i = p[l];
        a.innerHTML = A.renderNetwork([
          { id: "alpha", name: "Dirichlet Prior α", type: "dense", inShape: [1], outShape: [3], paramsCount: 0 },
          { id: "theta", name: "Topic Mixture θ", type: "dense", inShape: [3], outShape: [3], paramsCount: 0 },
          { id: "beta", name: "Word Distribution β", type: "dense", inShape: [3], outShape: [100], paramsCount: 0 }
        ], d, y, 1, "forward");
        const r = t.getContext("2d");
        r && (r.fillStyle = "#020617", r.fillRect(0, 0, t.width, t.height), r.fillStyle = "#38BDF8", r.font = "bold 13px monospace", r.fillText("LATENT DIRICHLET ALLOCATION (LDA)", 20, 26), p.forEach((m, h) => {
          const b = h === l, F = 55 + h * 80;
          r.fillStyle = b ? "#FFD700" : m.color, r.font = "bold 12px sans-serif", r.fillText(`${m.name} ${b ? "◀ ACTIVE" : ""}`, 20, F), m.words.forEach(($, _) => {
            const S = 20 + _ * 105;
            r.fillStyle = b ? "#0F8B8D" : "#1E293B", r.fillRect(S, F + 10, 95, 24), r.strokeStyle = "#334155", r.strokeRect(S, F + 10, 95, 24), r.fillStyle = "#FFF", r.font = "11px monospace", r.fillText($, S + 8, F + 26);
          });
        })), u.innerHTML = `
          <span style="color:#94A3B8; font-weight:700;">GIBBS SAMPLING:</span>
          <span style="background:${i.color}; color:#fff; padding:3px 10px; border-radius:4px;">${i.name}</span>
        `, n.innerHTML = `• <b>Generative Story Step:</b> For document $d$, sampled topic proportions $\\theta_d \\sim \\text{Dir}(\\alpha)$.<br>&nbsp;&nbsp;Active word emissions from $\\beta$: <code>${i.words.join(", ")}</code>.`, this.currentStep++;
      }, x();
    } else
      x = () => {
        a.innerHTML = A.renderNetwork([
          { id: "in", name: "Input Layer", type: "dense", inShape: [4], outShape: [8], paramsCount: 32 },
          { id: "h", name: "Hidden Features", type: "dense", inShape: [8], outShape: [8], paramsCount: 64 },
          { id: "out", name: "Output Head", type: "dense", inShape: [8], outShape: [2], paramsCount: 16 }
        ], d, y, 1, "forward");
        const p = t.getContext("2d");
        p && (p.fillStyle = "#020617", p.fillRect(0, 0, t.width, t.height), p.fillStyle = "#38BDF8", p.font = "bold 13px monospace", p.fillText(`MODEL ARCHITECTURE: ${o.toUpperCase()}`, 20, 30), p.fillStyle = "#E2E8F0", p.font = "12px sans-serif", p.fillText(`Step ${this.currentStep + 1}: Computing forward tensor flow...`, 20, 70)), n.innerHTML = `• <b>Model ${o.toUpperCase()}:</b> Advancing step ${this.currentStep + 1} through network activations.`, this.currentStep++;
      }, x();
    s.onclick = () => x(), c.onclick = () => {
      this.timer && (clearInterval(this.timer), this.timer = null, this.isPlaying = !1, f.innerText = "▶ Auto"), this.currentStep = 0, this.initModel(o);
    }, f.onclick = () => {
      this.isPlaying ? (clearInterval(this.timer), this.timer = null, this.isPlaying = !1, f.innerText = "▶ Auto") : (this.isPlaying = !0, f.innerText = "⏸ Pause", this.timer = setInterval(() => x(), 900));
    };
  }
}
typeof window < "u" && !customElements.get("neural-sim") && customElements.define("neural-sim", G);
export {
  U as AutoencoderModel,
  z as CNNModel,
  M as CanvasVisualizer,
  D as ExecutionTracer,
  N as LSTMModel,
  H as MLP,
  G as NeuralSimElement,
  A as SVGDiagramRenderer,
  g as Tensor,
  O as TransformerAttention
};
//# sourceMappingURL=omni-neural-sim.es.js.map
