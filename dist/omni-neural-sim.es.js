var W = Object.defineProperty;
var I = (w, e, t) => e in w ? W(w, e, { enumerable: !0, configurable: !0, writable: !0, value: t }) : w[e] = t;
var u = (w, e, t) => I(w, typeof e != "symbol" ? e + "" : e, t);
class d {
  constructor(e, t) {
    u(this, "data");
    u(this, "shape");
    u(this, "strides");
    u(this, "size");
    this.shape = [...e], this.size = e.reduce((o, a) => o * a, 1), this.strides = d.computeStrides(this.shape), t ? this.data = t instanceof Float32Array ? t : new Float32Array(t) : this.data = new Float32Array(this.size);
  }
  static computeStrides(e) {
    const t = new Array(e.length);
    let o = 1;
    for (let a = e.length - 1; a >= 0; a--)
      t[a] = o, o *= e[a];
    return t;
  }
  static zeros(e) {
    return new d(e);
  }
  static ones(e) {
    const t = new d(e);
    return t.data.fill(1), t;
  }
  static random(e, t = -1, o = 1) {
    const a = new d(e), s = o - t;
    for (let p = 0; p < a.size; p++)
      a.data[p] = t + Math.random() * s;
    return a;
  }
  static fromArray(e) {
    const t = [];
    let o = e;
    for (; Array.isArray(o); )
      t.push(o.length), o = o[0];
    const a = [];
    function s(p) {
      if (Array.isArray(p))
        for (const i of p) s(i);
      else
        a.push(Number(p));
    }
    return s(e), new d(t, a);
  }
  clone() {
    return new d(this.shape, new Float32Array(this.data));
  }
  getIndex(...e) {
    let t = 0;
    for (let o = 0; o < e.length; o++)
      t += e[o] * this.strides[o];
    return t;
  }
  get(...e) {
    return this.data[this.getIndex(...e)];
  }
  set(e, ...t) {
    this.data[this.getIndex(...t)] = e;
  }
  // --- Element-wise arithmetic ---
  add(e) {
    const t = new d(this.shape);
    if (typeof e == "number")
      for (let o = 0; o < this.size; o++) t.data[o] = this.data[o] + e;
    else
      for (let o = 0; o < this.size; o++) t.data[o] = this.data[o] + e.data[o];
    return t;
  }
  sub(e) {
    const t = new d(this.shape);
    if (typeof e == "number")
      for (let o = 0; o < this.size; o++) t.data[o] = this.data[o] - e;
    else
      for (let o = 0; o < this.size; o++) t.data[o] = this.data[o] - e.data[o];
    return t;
  }
  mul(e) {
    const t = new d(this.shape);
    if (typeof e == "number")
      for (let o = 0; o < this.size; o++) t.data[o] = this.data[o] * e;
    else
      for (let o = 0; o < this.size; o++) t.data[o] = this.data[o] * e.data[o];
    return t;
  }
  // --- Matrix Multiplication (2D) ---
  matmul(e) {
    if (this.shape.length !== 2 || e.shape.length !== 2)
      throw new Error(`matmul requires 2D tensors, got ${this.shape} and ${e.shape}`);
    const [t, o] = this.shape, [a, s] = e.shape;
    if (o !== a)
      throw new Error(`Incompatible matrix dims: [${t}, ${o}] x [${a}, ${s}]`);
    const p = new d([t, s]);
    for (let i = 0; i < t; i++)
      for (let r = 0; r < s; r++) {
        let l = 0;
        for (let n = 0; n < o; n++)
          l += this.get(i, n) * e.get(n, r);
        p.set(l, i, r);
      }
    return p;
  }
  transpose() {
    if (this.shape.length !== 2)
      throw new Error("transpose currently supports 2D tensors");
    const [e, t] = this.shape, o = new d([t, e]);
    for (let a = 0; a < e; a++)
      for (let s = 0; s < t; s++)
        o.set(this.get(a, s), s, a);
    return o;
  }
  // --- Activations ---
  relu() {
    const e = new d(this.shape);
    for (let t = 0; t < this.size; t++)
      e.data[t] = Math.max(0, this.data[t]);
    return e;
  }
  sigmoid() {
    const e = new d(this.shape);
    for (let t = 0; t < this.size; t++)
      e.data[t] = 1 / (1 + Math.exp(-this.data[t]));
    return e;
  }
  tanh() {
    const e = new d(this.shape);
    for (let t = 0; t < this.size; t++)
      e.data[t] = Math.tanh(this.data[t]);
    return e;
  }
  gelu() {
    const e = new d(this.shape), t = Math.sqrt(2 / Math.PI);
    for (let o = 0; o < this.size; o++) {
      const a = this.data[o];
      e.data[o] = 0.5 * a * (1 + Math.tanh(t * (a + 0.044715 * Math.pow(a, 3))));
    }
    return e;
  }
  softmax(e = -1) {
    const t = new d(this.shape);
    if (this.shape.length === 1) {
      let o = -1 / 0;
      for (let s = 0; s < this.size; s++) this.data[s] > o && (o = this.data[s]);
      let a = 0;
      for (let s = 0; s < this.size; s++)
        t.data[s] = Math.exp(this.data[s] - o), a += t.data[s];
      for (let s = 0; s < this.size; s++) t.data[s] /= a;
      return t;
    }
    if (this.shape.length === 2) {
      const [o, a] = this.shape;
      for (let s = 0; s < o; s++) {
        let p = -1 / 0;
        for (let r = 0; r < a; r++) {
          const l = this.get(s, r);
          l > p && (p = l);
        }
        let i = 0;
        for (let r = 0; r < a; r++) {
          const l = Math.exp(this.get(s, r) - p);
          t.set(l, s, r), i += l;
        }
        for (let r = 0; r < a; r++)
          t.set(t.get(s, r) / i, s, r);
      }
      return t;
    }
    throw new Error("softmax supports 1D or 2D tensors");
  }
  // --- 2D Convolution & Pooling ---
  /**
   * 2D Convolution on [H, W, InChannels] with Kernel [Kh, Kw, InChannels, OutChannels]
   */
  conv2d(e, t, o = 1, a = 0) {
    const [s, p, i] = this.shape, [r, l, n, c] = e.shape;
    if (i !== n)
      throw new Error(`Channel mismatch: input has ${i}, kernel expects ${n}`);
    const h = Math.floor((s - r + 2 * a) / o) + 1, f = Math.floor((p - l + 2 * a) / o) + 1, m = new d([h, f, c]);
    for (let g = 0; g < c; g++) {
      const x = t ? t.data[g] : 0;
      for (let S = 0; S < h; S++)
        for (let v = 0; v < f; v++) {
          let T = x;
          const k = S * o - a, y = v * o - a;
          for (let _ = 0; _ < r; _++) {
            const M = k + _;
            if (!(M < 0 || M >= s))
              for (let $ = 0; $ < l; $++) {
                const A = y + $;
                if (!(A < 0 || A >= p))
                  for (let F = 0; F < i; F++)
                    T += this.get(M, A, F) * e.get(_, $, F, g);
              }
          }
          m.set(T, S, v, g);
        }
    }
    return m;
  }
  maxPool2d(e = 2, t = 2) {
    const [o, a, s] = this.shape, p = Math.floor((o - e) / t) + 1, i = Math.floor((a - e) / t) + 1, r = new d([p, i, s]);
    for (let l = 0; l < s; l++)
      for (let n = 0; n < p; n++)
        for (let c = 0; c < i; c++) {
          let h = -1 / 0;
          const f = n * t, m = c * t;
          for (let g = 0; g < e; g++)
            for (let x = 0; x < e; x++) {
              const S = this.get(f + g, m + x, l);
              S > h && (h = S);
            }
          r.set(h, n, c, l);
        }
    return r;
  }
  flatten() {
    return new d([this.size], this.data);
  }
  toArray() {
    if (this.shape.length === 1)
      return Array.from(this.data);
    if (this.shape.length === 2) {
      const [e, t] = this.shape, o = [];
      for (let a = 0; a < e; a++) {
        const s = [];
        for (let p = 0; p < t; p++) s.push(this.get(a, p));
        o.push(s);
      }
      return o;
    }
    return Array.from(this.data);
  }
}
class C {
  constructor() {
    u(this, "steps", []);
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
class B {
  constructor(e) {
    u(this, "weights", []);
    u(this, "biases", []);
    u(this, "activations", []);
    u(this, "tracer", new C());
    this.config = e, this.activations = e.activations;
    for (let t = 0; t < e.layerSizes.length - 1; t++) {
      const o = e.layerSizes[t], a = e.layerSizes[t + 1], s = Math.sqrt(2 / o);
      this.weights.push(d.random([o, a], -s, s)), this.biases.push(d.zeros([1, a]));
    }
  }
  forward(e, t = !1) {
    t && this.tracer.clear();
    let o = e;
    for (let a = 0; a < this.weights.length; a++) {
      const s = this.weights[a], p = this.biases[a], i = this.activations[a], r = o.matmul(s).add(p);
      let l = r;
      i === "relu" ? l = r.relu() : i === "sigmoid" ? l = r.sigmoid() : i === "tanh" && (l = r.tanh()), t && this.tracer.record({
        layerId: `layer_${a + 1}`,
        layerName: `Hidden Layer ${a + 1} (${i.toUpperCase()})`,
        operation: "Dense Matmul + Bias + Activation",
        formula: `a^[${a + 1}] = ${i}(W^[${a + 1}] * a^[${a}] + b^[${a + 1}])`,
        inputShapes: [o.shape, s.shape],
        outputShape: l.shape,
        tensorPreview: Array.from(l.data.slice(0, 4)),
        pedagogicalInsight: `Layer ${a + 1} transforms ${s.shape[0]} inputs into ${s.shape[1]} linear combinations, activated by ${i}.`
      }), o = l;
    }
    return o;
  }
  // Train a single epoch on a dataset X, Y
  trainStep(e, t) {
    let o = 0;
    const a = this.config.learningRate;
    for (let s = 0; s < e.length; s++) {
      const p = d.fromArray([e[s]]), i = t[s], r = [p], l = [];
      let n = p;
      for (let m = 0; m < this.weights.length; m++) {
        const g = n.matmul(this.weights[m]).add(this.biases[m]);
        l.push(g);
        const x = this.activations[m];
        x === "relu" ? n = g.relu() : x === "sigmoid" ? n = g.sigmoid() : x === "tanh" && (n = g.tanh()), r.push(n);
      }
      const h = n.data[0] - i[0];
      o += 0.5 * h * h;
      let f = new d([1, 1], [h]);
      for (let m = this.weights.length - 1; m >= 0; m--) {
        const g = l[m], x = this.activations[m], S = r[m], v = new d(g.shape);
        for (let y = 0; y < g.size; y++)
          if (x === "relu") v.data[y] = g.data[y] > 0 ? 1 : 0;
          else if (x === "sigmoid") {
            const _ = 1 / (1 + Math.exp(-g.data[y]));
            v.data[y] = _ * (1 - _);
          } else if (x === "tanh") {
            const _ = Math.tanh(g.data[y]);
            v.data[y] = 1 - _ * _;
          } else
            v.data[y] = 1;
        const T = f.mul(v), k = S.transpose().matmul(T);
        for (let y = 0; y < this.weights[m].size; y++)
          this.weights[m].data[y] -= a * k.data[y];
        for (let y = 0; y < this.biases[m].size; y++)
          this.biases[m].data[y] -= a * T.data[y];
        m > 0 && (f = T.matmul(this.weights[m].transpose()));
      }
    }
    return { loss: o / e.length };
  }
  /**
   * Evaluate a 2D grid of points for boundary visualization (e.g., XOR, Circles, Moons)
   */
  evaluateGrid(e = 30, t = 4) {
    const o = [], a = t * 2 / e, s = [], p = [];
    for (let i = 0; i < e; i++) {
      const r = [], l = t - i * a;
      p.push(l);
      for (let n = 0; n < e; n++) {
        const c = -t + n * a;
        i === 0 && s.push(c);
        const h = this.forward(new d([1, 2], [c, l]));
        r.push(h.data[0]);
      }
      o.push(r);
    }
    return { grid: o, xRange: s, yRange: p };
  }
}
class E {
  constructor() {
    u(this, "tracer", new C());
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
    this.kernel1 = d.random([3, 3, 1, 32], -0.2, 0.2), this.bias1 = d.zeros([32]), this.kernel2 = d.random([3, 3, 32, 64], -0.15, 0.15), this.bias2 = d.zeros([64]), this.kernel3 = d.random([3, 3, 64, 64], -0.15, 0.15), this.bias3 = d.zeros([64]), this.dense1_W = d.random([576, 64], -0.1, 0.1), this.dense1_B = d.zeros([1, 64]), this.dense2_W = d.random([64, 10], -0.1, 0.1), this.dense2_B = d.zeros([1, 10]);
  }
  forward(e, t = !0) {
    t && this.tracer.clear();
    const o = e.conv2d(this.kernel1, this.bias1, 1, 0).relu();
    t && this.tracer.record({
      layerId: "conv1",
      layerName: "Conv2D (32 filters, 3x3)",
      operation: "2D Convolution + ReLU",
      formula: "H_out = (28 - 3 + 0)/1 + 1 = 26; Params = (3*3*1 + 1)*32 = 320",
      inputShapes: [e.shape, this.kernel1.shape],
      outputShape: o.shape,
      tensorPreview: Array.from(o.data.slice(0, 5)),
      fullOutput: o,
      pedagogicalInsight: "Extracts 32 low-level edge and texture feature maps from the raw 28x28 pixel grid.",
      metadata: { filters: 32, kernel: "3x3", params: 320 }
    });
    const a = o.maxPool2d(2, 2);
    t && this.tracer.record({
      layerId: "pool1",
      layerName: "MaxPooling2D (2x2)",
      operation: "Spatial Downsampling",
      formula: "H_out = floor(26/2) = 13; W_out = 13",
      inputShapes: [o.shape],
      outputShape: a.shape,
      tensorPreview: Array.from(a.data.slice(0, 5)),
      fullOutput: a,
      pedagogicalInsight: "Preserves the most salient local features while reducing spatial dimensions by 75%."
    });
    const s = a.conv2d(this.kernel2, this.bias2, 1, 0).relu();
    t && this.tracer.record({
      layerId: "conv2",
      layerName: "Conv2D (64 filters, 3x3)",
      operation: "2D Convolution + ReLU",
      formula: "H_out = (13 - 3 + 0)/1 + 1 = 11; Params = (3*3*32 + 1)*64 = 18,496",
      inputShapes: [a.shape, this.kernel2.shape],
      outputShape: s.shape,
      tensorPreview: Array.from(s.data.slice(0, 5)),
      fullOutput: s,
      pedagogicalInsight: "Combines local features into mid-level parts (corners, loops, strokes)."
    });
    const p = s.maxPool2d(2, 2);
    t && this.tracer.record({
      layerId: "pool2",
      layerName: "MaxPooling2D (2x2)",
      operation: "Spatial Downsampling",
      formula: "H_out = floor(11/2) = 5; W_out = 5",
      inputShapes: [s.shape],
      outputShape: p.shape,
      tensorPreview: Array.from(p.data.slice(0, 5)),
      fullOutput: p,
      pedagogicalInsight: "Further downsamples to 5x5 feature grids."
    });
    const i = p.conv2d(this.kernel3, this.bias3, 1, 0).relu();
    t && this.tracer.record({
      layerId: "conv3",
      layerName: "Conv2D (64 filters, 3x3)",
      operation: "2D Convolution + ReLU",
      formula: "H_out = (5 - 3 + 0)/1 + 1 = 3; Params = (3*3*64 + 1)*64 = 36,928",
      inputShapes: [p.shape, this.kernel3.shape],
      outputShape: i.shape,
      tensorPreview: Array.from(i.data.slice(0, 5)),
      fullOutput: i,
      pedagogicalInsight: "High-level digit shape representations."
    });
    const r = i.flatten(), c = new d([1, 576], r.data).matmul(this.dense1_W).add(this.dense1_B).relu().matmul(this.dense2_W).add(this.dense2_B), h = c.softmax(-1), f = Array.from(h.data);
    let m = 0, g = -1;
    for (let x = 0; x < 10; x++)
      f[x] > g && (g = f[x], m = x);
    return t && this.tracer.record({
      layerId: "digit_probs",
      layerName: "Softmax Classification Head",
      operation: "Softmax(logits) -> Argmax",
      formula: "P(digit=k) = exp(z_k) / sum(exp(z)); Predicted = argmax_k(P)",
      inputShapes: [c.shape],
      outputShape: [10],
      tensorPreview: f,
      pedagogicalInsight: `Final digit classification with top prediction ${m} (${(g * 100).toFixed(1)}%).`
    }), {
      probabilities: f,
      predictedDigit: m,
      conv1Out: o,
      pool1Out: a,
      conv2Out: s,
      pool2Out: p,
      conv3Out: i
    };
  }
  loadWeightsFromJSON(e) {
    e.kernel1 && (this.kernel1 = new d([3, 3, 1, 32], e.kernel1)), e.bias1 && (this.bias1 = new d([32], e.bias1)), e.dense2_W && (this.dense2_W = new d([64, 10], e.dense2_W)), e.dense2_B && (this.dense2_B = new d([1, 10], e.dense2_B));
  }
}
class L {
  constructor(e = 4, t = 4) {
    u(this, "hiddenSize");
    u(this, "inputSize");
    u(this, "tracer", new C());
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
    this.inputSize = e, this.hiddenSize = t;
    const o = 0.5;
    this.W_f = d.random([e, t], -o, o), this.U_f = d.random([t, t], -o, o), this.b_f = d.ones([1, t]), this.W_i = d.random([e, t], -o, o), this.U_i = d.random([t, t], -o, o), this.b_i = d.zeros([1, t]), this.W_c = d.random([e, t], -o, o), this.U_c = d.random([t, t], -o, o), this.b_c = d.zeros([1, t]), this.W_o = d.random([e, t], -o, o), this.U_o = d.random([t, t], -o, o), this.b_o = d.zeros([1, t]);
  }
  step(e, t, o) {
    const a = e.matmul(this.W_f).add(t.matmul(this.U_f)).add(this.b_f).sigmoid(), s = e.matmul(this.W_i).add(t.matmul(this.U_i)).add(this.b_i).sigmoid(), p = e.matmul(this.W_c).add(t.matmul(this.U_c)).add(this.b_c).tanh(), i = a.mul(o).add(s.mul(p)), r = e.matmul(this.W_o).add(t.matmul(this.U_o)).add(this.b_o).sigmoid();
    return { h: r.mul(i.tanh()), c: i, gates: { f: a, i: s, c_tilde: p, o: r } };
  }
  unroll(e) {
    this.tracer.clear();
    const t = [];
    let o = d.zeros([1, this.hiddenSize]), a = d.zeros([1, this.hiddenSize]);
    for (let s = 0; s < e.length; s++) {
      const p = e[s], i = new d([1, this.inputSize], p.vector), r = this.step(i, o, a);
      o = r.h, a = r.c;
      const l = {
        t: s,
        inputToken: p.token,
        x_t: Array.from(i.data),
        f_gate: Array.from(r.gates.f.data),
        i_gate: Array.from(r.gates.i.data),
        c_tilde: Array.from(r.gates.c_tilde.data),
        c_t: Array.from(a.data),
        o_gate: Array.from(r.gates.o.data),
        h_t: Array.from(o.data)
      };
      t.push(l), this.tracer.record({
        layerId: `lstm_step_${s}`,
        layerName: `LSTM Step t=${s} ("${p.token}")`,
        operation: "Recurrent Cell State Transition",
        formula: "c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t; h_t = o_t ⊙ tanh(c_t)",
        inputShapes: [i.shape, o.shape, a.shape],
        outputShape: o.shape,
        tensorPreview: Array.from(o.data),
        pedagogicalInsight: `Step t=${s}: Forget gate retained ${(l.f_gate[0] * 100).toFixed(0)}% of prior cell memory while input gate injected ${(l.i_gate[0] * 100).toFixed(0)}% of new token information.`
      });
    }
    return t;
  }
}
class z {
  constructor(e = 8, t = 2) {
    u(this, "d_model");
    u(this, "d_k");
    u(this, "numHeads");
    u(this, "tracer", new C());
    u(this, "W_q");
    u(this, "W_k");
    u(this, "W_v");
    u(this, "W_o");
    this.d_model = e, this.numHeads = t, this.d_k = Math.floor(e / t);
    const o = Math.sqrt(2 / e);
    this.W_q = d.random([e, e], -o, o), this.W_k = d.random([e, e], -o, o), this.W_v = d.random([e, e], -o, o), this.W_o = d.random([e, e], -o, o);
  }
  /**
   * Forward pass over token embeddings matrix X of shape [seqLen, d_model]
   */
  forward(e, t = !0, o = !0) {
    o && this.tracer.clear();
    const a = e.shape[0], s = e.matmul(this.W_q), p = e.matmul(this.W_k), i = e.matmul(this.W_v), r = 1 / Math.sqrt(this.d_model), l = p.transpose(), n = s.matmul(l).mul(r);
    if (t)
      for (let g = 0; g < a; g++)
        for (let x = g + 1; x < a; x++)
          n.set(-1e9, g, x);
    const c = n.softmax(-1), f = c.matmul(i).matmul(this.W_o);
    o && this.tracer.record({
      layerId: "self_attention",
      layerName: `Self-Attention (d_model=${this.d_model}, seqLen=${a})`,
      operation: "Scaled Dot-Product Attention",
      formula: "Attention(Q,K,V) = softmax(Q K^T / √d_k + Mask) V",
      inputShapes: [e.shape, this.W_q.shape],
      outputShape: f.shape,
      tensorPreview: Array.from(c.data.slice(0, 6)),
      pedagogicalInsight: `Calculated ${a}x${a} pairwise attention affinity matrix; tokens dynamically aggregate information from allowed preceding contexts.`
    });
    const m = [];
    for (let g = 0; g < a; g++) {
      const x = [];
      for (let S = 0; S < a; S++)
        x.push(c.get(g, S));
      m.push(x);
    }
    return {
      attentionWeights: m,
      output: f
    };
  }
}
class N {
  constructor(e = 10, t = 2) {
    u(this, "tracer", new C());
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
    this.inDim = e, this.latentDim = t;
    const o = 0.3;
    this.W_enc1 = d.random([e, 16], -o, o), this.b_enc1 = d.zeros([1, 16]), this.W_enc2 = d.random([16, t], -o, o), this.b_enc2 = d.zeros([1, t]), this.W_dec1 = d.random([t, 16], -o, o), this.b_dec1 = d.zeros([1, 16]), this.W_dec2 = d.random([16, e], -o, o), this.b_dec2 = d.zeros([1, e]);
  }
  encode(e) {
    return e.matmul(this.W_enc1).add(this.b_enc1).relu().matmul(this.W_enc2).add(this.b_enc2);
  }
  decode(e) {
    return e.matmul(this.W_dec1).add(this.b_dec1).relu().matmul(this.W_dec2).add(this.b_dec2).sigmoid();
  }
  forward(e, t = !0) {
    t && this.tracer.clear();
    const o = this.encode(e), a = this.decode(o);
    let s = 0;
    for (let p = 0; p < e.size; p++) {
      const i = e.data[p] - a.data[p];
      s += i * i;
    }
    return s /= e.size, t && this.tracer.record({
      layerId: "latent_bottleneck",
      layerName: `Latent Space (dim=${this.latentDim})`,
      operation: "Nonlinear Dimensionality Compression",
      formula: "z = W_2 * relu(W_1 x + b_1) + b_2",
      inputShapes: [e.shape],
      outputShape: o.shape,
      tensorPreview: Array.from(o.data),
      pedagogicalInsight: `Compressed input from ${this.inDim} dimensions to ${this.latentDim} latent coordinates with MSE=${s.toFixed(4)}.`
    }), { latent: o, reconstructed: a, mse: s };
  }
}
class b {
  static renderNetwork(e, t = 800, o = 360, a = -1) {
    const s = e.length, p = t / (s + 1), i = [];
    e.forEach((l, n) => {
      const c = (n + 1) * p, h = Math.min(l.outShape[l.outShape.length - 1] || 4, 8), f = o / (h + 1), m = [];
      for (let g = 0; g < h; g++)
        m.push({
          layerIndex: n,
          nodeIndex: g,
          x: c,
          y: (g + 1) * f,
          label: `${l.name} [${g}]`
        });
      i.push(m);
    });
    let r = `<svg viewBox="0 0 ${t} ${o}" width="100%" height="${o}" style="background:#0b1329; border-radius:12px; font-family:system-ui, -apple-system, sans-serif;">`;
    r += `<defs>
      <linearGradient id="edgeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="#0F8B8D" stop-opacity="0.4"/>
        <stop offset="100%" stop-color="#6B2D7B" stop-opacity="0.6"/>
      </linearGradient>
      <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
        <feGaussianBlur stdDeviation="3" result="blur" />
        <feComposite in="SourceGraphic" in2="blur" operator="over" />
      </filter>
    </defs>`;
    for (let l = 0; l < i.length - 1; l++) {
      const n = i[l], c = i[l + 1];
      for (const h of n)
        for (const f of c)
          r += `<line x1="${h.x}" y1="${h.y}" x2="${f.x}" y2="${f.y}" stroke="url(#edgeGrad)" stroke-width="1.2"/>`;
    }
    return e.forEach((l, n) => {
      const c = (n + 1) * p;
      r += `<text x="${c}" y="24" fill="${n === a ? "#FFD700" : "#E2E8F0"}" font-size="12" font-weight="600" text-anchor="middle">${l.name}</text>`, r += `<text x="${c}" y="40" fill="#94A3B8" font-size="10" font-family="monospace" text-anchor="middle">[${l.outShape.join("×")}]</text>`, l.paramsCount > 0 && (r += `<text x="${c}" y="${o - 12}" fill="#38BDF8" font-size="10" font-family="monospace" text-anchor="middle">${l.paramsCount.toLocaleString()} params</text>`);
    }), i.forEach((l, n) => {
      const c = n === a, h = c ? "#FFD700" : n === 0 ? "#0F8B8D" : n === i.length - 1 ? "#C8102E" : "#6B2D7B";
      for (const f of l)
        r += `<circle cx="${f.x}" cy="${f.y}" r="9" fill="${h}" stroke="#ffffff" stroke-width="1.5" ${c ? 'filter="url(#glow)"' : ""}/>`;
    }), r += "</svg>", r;
  }
}
class D {
  /**
   * Render a 2D scalar grid as a color heatmap (Decision Boundary)
   */
  static renderDecisionBoundary(e, t, o) {
    const a = e.getContext("2d");
    if (!a) return;
    const s = t.length, p = t[0].length, i = e.width / p, r = e.height / s;
    for (let l = 0; l < s; l++)
      for (let n = 0; n < p; n++) {
        const c = t[l][n], h = Math.floor(c * 200 + (1 - c) * 20), f = Math.floor((1 - Math.abs(c - 0.5) * 2) * 150), m = Math.floor((1 - c) * 200 + c * 30);
        a.fillStyle = `rgb(${h}, ${f}, ${m})`, a.fillRect(n * i, l * r, i + 1, r + 1);
      }
    if (o)
      for (const l of o) {
        const n = (l.x + 4) / 8 * e.width, c = (4 - l.y) / 8 * e.height;
        a.beginPath(), a.arc(n, c, 5, 0, Math.PI * 2), a.fillStyle = l.label === 1 ? "#FFD700" : "#FFFFFF", a.strokeStyle = "#000000", a.lineWidth = 1.5, a.fill(), a.stroke();
      }
  }
  /**
   * Render Attention Weight Matrix [seqLen x seqLen] with token labels
   */
  static renderAttentionMatrix(e, t, o) {
    const a = e.getContext("2d");
    if (!a) return;
    const s = t.length, p = 70, i = 70, r = e.width - p - 10, l = e.height - i - 10, n = r / s, c = l / s;
    a.clearRect(0, 0, e.width, e.height), a.fillStyle = "#0F172A", a.fillRect(0, 0, e.width, e.height);
    for (let h = 0; h < s; h++)
      for (let f = 0; f < s; f++) {
        const m = t[h][f];
        a.fillStyle = `rgba(15, 139, 141, ${Math.max(0.08, m)})`, a.fillRect(p + f * n, i + h * c, n - 1, c - 1), n > 30 && (a.fillStyle = m > 0.4 ? "#FFFFFF" : "#94A3B8", a.font = "10px monospace", a.textAlign = "center", a.textBaseline = "middle", a.fillText(
          m.toFixed(2),
          p + f * n + n / 2,
          i + h * c + c / 2
        ));
      }
    a.fillStyle = "#E2E8F0", a.font = "12px sans-serif", a.textAlign = "right", a.textBaseline = "middle";
    for (let h = 0; h < s; h++) {
      const f = o[h] || `t_${h}`;
      a.fillText(f, p - 8, i + h * c + c / 2);
    }
    a.textAlign = "center", a.textBaseline = "bottom";
    for (let h = 0; h < s; h++) {
      const f = o[h] || `t_${h}`;
      a.fillText(f, p + h * n + n / 2, i - 8);
    }
  }
}
class R extends HTMLElement {
  static get observedAttributes() {
    return ["model", "dataset", "weights", "tokens", "sequence", "title", "course"];
  }
  connectedCallback() {
    this.render();
  }
  attributeChangedCallback() {
    this.render();
  }
  normalizeModelType(e) {
    const t = e.toLowerCase().trim();
    return ["mlp", "neural-networks", "dl-w01", "nlp-w01", "backprop"].includes(t) ? "mlp" : ["lstm", "rnn", "gru", "dl-w10", "dl-w11", "nlp-w02", "recurrent"].includes(t) ? "lstm" : ["cnn", "tokenization-cnn", "dl-w05", "nlp-w03", "convnet", "mnist"].includes(t) ? "cnn" : ["ngrams", "bow", "bag-of-words", "nlp-w04"].includes(t) ? "ngrams" : ["embeddings", "word2vec", "glove", "nlp-w05"].includes(t) ? "embeddings" : ["autoencoder", "vae", "tsne", "dl-w06", "nlp-w06", "latent"].includes(t) ? "autoencoder" : ["lda", "topic-modeling", "topics", "nlp-w07"].includes(t) ? "lda" : ["stm", "structural-topics", "nlp-w08"].includes(t) ? "stm" : ["classification", "regularization", "dropout", "dl-w02", "nlp-w09"].includes(t) ? "classification" : ["ner", "sequence-tagging", "bilstm-ner", "nlp-w10"].includes(t) ? "ner" : ["gan", "gans", "dcgan", "dl-w08", "nlp-w11", "crf-bilstm"].includes(t) ? "gan" : ["transformer", "bert", "attention", "dl-w12", "nlp-w12", "gpt"].includes(t) ? "transformer" : ["regression", "optimizers", "adam", "dl-w03", "loss-landscape"].includes(t) ? "optimizers" : ["transfer-learning", "fine-tuning", "dl-w07"].includes(t) ? "transfer-learning" : ["text-cnn", "conv1d", "dl-w09"].includes(t) ? "text-cnn" : "mlp";
  }
  render() {
    const e = this.getAttribute("model") || "mlp", t = this.normalizeModelType(e), o = this.getAttribute("title") || `Model Simulator: ${t.toUpperCase()}`, a = this.getAttribute("course") || "Harvard CSCI E-89 / E-89b SOTA";
    this.innerHTML = `
      <div class="omni-sim-card" style="font-family:system-ui, -apple-system, sans-serif; background:#0F172A; color:#F8FAFC; border:1px solid #334155; border-radius:12px; padding:20px; margin:20px 0; box-shadow:0 10px 25px -5px rgba(0,0,0,0.3);">
        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #1E293B; padding-bottom:12px; margin-bottom:16px; flex-wrap:wrap; gap:10px;">
          <div>
            <span style="background:#0F8B8D; color:#fff; font-size:11px; font-weight:700; text-transform:uppercase; padding:3px 8px; border-radius:4px; letter-spacing:0.5px;">${a}</span>
            <h3 style="margin:6px 0 0 0; font-size:1.25rem; color:#FFFFFF;">${o}</h3>
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
            <button class="btn-step" style="background:#0F8B8D; color:#F8FAFC; border:none; border-radius:6px; padding:6px 14px; font-size:12px; font-weight:600; cursor:pointer; transition:all 0.2s;">Step Forward ⏭</button>
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
            <canvas class="sim-canvas" width="340" height="250" style="border-radius:6px; background:#000;"></canvas>
            <div class="sim-metrics" style="width:100%; margin-top:10px; font-size:12px; font-family:monospace; color:#E2E8F0;"></div>
          </div>
        </div>

        <div class="sim-trace-panel" style="margin-top:16px; background:#020617; border:1px solid #1E293B; border-radius:8px; padding:12px;">
          <div style="font-size:11px; font-weight:700; color:#38BDF8; margin-bottom:4px;">PEDAGOGICAL MATHEMATICAL TRACE & INSIGHTS:</div>
          <div class="trace-output" style="font-size:12px; font-family:monospace; color:#A5B4FC; line-height:1.5;"></div>
        </div>
      </div>
    `, this.initModel(t);
    const s = this.querySelector(".sim-model-selector");
    s && (s.onchange = (p) => {
      const i = p.target.value;
      this.setAttribute("model", i);
    });
  }
  initModel(e) {
    const t = this.querySelector(".sim-canvas"), o = this.querySelector(".diagram-host"), a = this.querySelector(".trace-output");
    this.querySelector(".sim-metrics");
    const s = this.querySelector(".btn-step"), p = this.querySelector(".btn-reset");
    if (e === "mlp") {
      const i = new B({
        layerSizes: [2, 6, 4, 1],
        activations: ["tanh", "tanh", "sigmoid"],
        learningRate: 0.1
      }), r = [[-2, -2], [-2, 2], [2, -2], [2, 2]], l = [[0], [1], [1], [0]], n = () => {
        const { grid: c } = i.evaluateGrid(30, 4);
        D.renderDecisionBoundary(
          t,
          c,
          r.map((h, f) => ({ x: h[0], y: h[1], label: l[f][0] }))
        ), o.innerHTML = b.renderNetwork([
          { id: "in", name: "Input [x1, x2]", type: "dense", inShape: [2], outShape: [2], paramsCount: 0 },
          { id: "h1", name: "Hidden 1 (Tanh)", type: "dense", inShape: [2], outShape: [6], paramsCount: 18 },
          { id: "h2", name: "Hidden 2 (Tanh)", type: "dense", inShape: [6], outShape: [4], paramsCount: 28 },
          { id: "out", name: "Output (Sigmoid)", type: "dense", inShape: [4], outShape: [1], paramsCount: 5 }
        ], 380, 210), i.forward(new d([1, 2], [1.5, -1.5]), !0), a.innerHTML = i.tracer.steps.map((h) => `• <b>${h.layerName}:</b> ${h.formula} → [${h.outputShape}]`).join("<br>");
      };
      s.onclick = () => {
        for (let c = 0; c < 20; c++) i.trainStep(r, l);
        n();
      }, p.onclick = () => this.initModel("mlp"), n();
    } else if (e === "lstm") {
      const i = new L(4, 4), l = ["Natural", "Language", "Processing", "Recurrent"].map((h) => ({ token: h, vector: [0.5, -0.2, 0.8, -0.1] }));
      let n = 0;
      const c = () => {
        const h = i.unroll(l);
        o.innerHTML = b.renderNetwork([
          { id: "in", name: "x_t Token Vector", type: "dense", inShape: [4], outShape: [4], paramsCount: 0 },
          { id: "gates", name: "Gates [f, i, c, o]", type: "lstm_cell", inShape: [4], outShape: [16], paramsCount: 144 },
          { id: "cell", name: "c_t Memory Cell", type: "lstm_cell", inShape: [4], outShape: [4], paramsCount: 0 },
          { id: "h", name: "h_t Hidden State", type: "lstm_cell", inShape: [4], outShape: [4], paramsCount: 0 }
        ], 380, 210);
        const f = t.getContext("2d");
        f && (f.fillStyle = "#020617", f.fillRect(0, 0, t.width, t.height), f.fillStyle = "#38BDF8", f.font = "12px monospace", f.fillText("LSTM GATE TRANSITION DYNAMICS:", 15, 25), h.forEach((m, g) => {
          const x = g === n % h.length;
          f.fillStyle = x ? "#FFD700" : "#94A3B8", f.fillText(
            `t=${m.t} ("${m.inputToken}"): f=${m.f_gate[0].toFixed(2)}, i=${m.i_gate[0].toFixed(2)}, o=${m.o_gate[0].toFixed(2)} ${x ? "◀" : ""}`,
            15,
            55 + g * 32
          );
        })), a.innerHTML = i.tracer.steps.map((m) => `• <b>${m.layerName}:</b> ${m.formula} → ${m.pedagogicalInsight}`).join("<br>");
      };
      s.onclick = () => {
        n++, c();
      }, p.onclick = () => {
        n = 0, c();
      }, c();
    } else if (e === "cnn") {
      const i = new E(), r = d.zeros([28, 28, 1]);
      for (let c = 4; c < 24; c++) r.set(1, c, 14, 0);
      const l = i.forward(r, !0);
      o.innerHTML = b.renderNetwork([
        { id: "conv1", name: "Conv2D (3x3)", type: "conv2d", inShape: [28, 28, 1], outShape: [26, 26, 32], paramsCount: 320 },
        { id: "pool1", name: "MaxPool", type: "maxpool2d", inShape: [26, 26, 32], outShape: [13, 13, 32], paramsCount: 0 },
        { id: "conv2", name: "Conv2D (3x3)", type: "conv2d", inShape: [13, 13, 32], outShape: [11, 11, 64], paramsCount: 18496 },
        { id: "dense", name: "Dense Head", type: "dense", inShape: [576], outShape: [10], paramsCount: 37578 }
      ], 380, 210);
      const n = t.getContext("2d");
      n && (n.fillStyle = "#020617", n.fillRect(0, 0, t.width, t.height), n.fillStyle = "#38BDF8", n.font = "14px monospace", n.fillText(`Top Predicted Digit: ${l.predictedDigit}`, 20, 40), n.fillText(`Softmax Confidence: ${(l.probabilities[l.predictedDigit] * 100).toFixed(1)}%`, 20, 68), n.fillText("Conv1 Output Tensor: 26×26×32", 20, 105), n.fillText("Layer 1 Parameter Count: 320", 20, 135), n.fillStyle = "#94A3B8", n.fillText("(3×3×1 + 1) × 32 = 320 params", 20, 160)), a.innerHTML = i.tracer.steps.map((c) => `• <b>${c.layerName}:</b> ${c.formula} → ${c.pedagogicalInsight}`).join("<br>"), s.onclick = () => {
        const c = Math.floor(Math.random() * 6) - 3, h = d.zeros([28, 28, 1]);
        for (let m = 4; m < 24; m++) h.set(1, m, Math.min(27, Math.max(0, 14 + c)), 0);
        const f = i.forward(h, !0);
        n && (n.fillStyle = "#020617", n.fillRect(0, 0, t.width, t.height), n.fillStyle = "#38BDF8", n.fillText(`Top Predicted Digit: ${f.predictedDigit}`, 20, 40), n.fillText(`Confidence: ${(f.probabilities[f.predictedDigit] * 100).toFixed(1)}%`, 20, 68), n.fillText("Conv1 Output Tensor: 26×26×32", 20, 105), n.fillText("Params = (3·3·1 + 1)·32 = 320", 20, 135));
      }, p.onclick = () => this.initModel("cnn");
    } else if (e === "ngrams") {
      const i = ["the", "neural", "network", "processes", "language", "<UNK>"], r = { the: 4, neural: 3, network: 3, processes: 2, language: 2, "<UNK>": 1 };
      o.innerHTML = b.renderNetwork([
        { id: "tokens", name: "Text Tokens", type: "embedding", inShape: [1], outShape: [6], paramsCount: 0 },
        { id: "count", name: "1-Gram Counter", type: "dense", inShape: [6], outShape: [6], paramsCount: 0 },
        { id: "oov", name: "<UNK> Firewall", type: "dense", inShape: [6], outShape: [6], paramsCount: 0 }
      ], 380, 210);
      const l = t.getContext("2d");
      l && (l.fillStyle = "#020617", l.fillRect(0, 0, t.width, t.height), l.fillStyle = "#38BDF8", l.font = "12px monospace", l.fillText("VOCABULARY & BoW DISTRIBUTION:", 15, 25), i.forEach((n, c) => {
        const h = n === "<UNK>";
        l.fillStyle = h ? "#C8102E" : "#0F8B8D", l.fillRect(15, 45 + c * 30, r[n] * 35, 18), l.fillStyle = "#FFF", l.fillText(`${n}: ${r[n]}`, 20, 58 + c * 30);
      })), a.innerHTML = `• <b>Vocabulary Size:</b> ${i.length} unique terms.<br>• <b>OOV Firewall:</b> Unseen test words route to <code>&lt;UNK&gt;</code> without corrupting vector length.<br>• <b>Formula:</b> BoW_vector[i] = count(word_i ∈ doc).`, s.onclick = () => {
        r["<UNK>"]++, this.initModel("ngrams");
      }, p.onclick = () => {
        r["<UNK>"] = 1, this.initModel("ngrams");
      };
    } else if (e === "embeddings") {
      const i = [
        { w: "king", x: 1.8, y: 1.5 },
        { w: "queen", x: 1.7, y: -0.5 },
        { w: "man", x: -1.2, y: 1.4 },
        { w: "woman", x: -1.3, y: -0.6 },
        { w: "apple", x: 0.1, y: -2.2 }
      ];
      o.innerHTML = b.renderNetwork([
        { id: "onehot", name: "One-Hot Index", type: "dense", inShape: [1e4], outShape: [300], paramsCount: 3e6 },
        { id: "proj", name: "2D t-SNE / PCA", type: "dense", inShape: [300], outShape: [2], paramsCount: 600 }
      ], 380, 210);
      const r = t.getContext("2d");
      r && (r.fillStyle = "#020617", r.fillRect(0, 0, t.width, t.height), r.strokeStyle = "#334155", r.lineWidth = 1, r.beginPath(), r.moveTo(t.width / 2, 0), r.lineTo(t.width / 2, t.height), r.moveTo(0, t.height / 2), r.lineTo(t.width, t.height / 2), r.stroke(), i.forEach((l) => {
        const n = t.width / 2 + l.x * 45, c = t.height / 2 - l.y * 45;
        r.beginPath(), r.arc(n, c, 5, 0, Math.PI * 2), r.fillStyle = "#0F8B8D", r.fill(), r.fillStyle = "#FFF", r.font = "11px sans-serif", r.fillText(l.w, n + 8, c + 4);
      }), r.strokeStyle = "#FFD700", r.lineWidth = 2, r.beginPath(), r.moveTo(t.width / 2 + 1.8 * 45, t.height / 2 - 1.5 * 45), r.lineTo(t.width / 2 + 1.7 * 45, t.height / 2 - -0.5 * 45), r.stroke()), a.innerHTML = '• <b>Vector Analogy:</b> <code>vec("king") - vec("man") + vec("woman") ≈ vec("queen")</code><br>• <b>Cosine Similarity:</b> cos(θ) = (u · v) / (||u|| ||v||) captures geometric semantic alignment.';
    } else if (e === "autoencoder") {
      const i = new N(10, 2), r = d.random([1, 10], 0, 1), l = i.forward(r, !0);
      o.innerHTML = b.renderNetwork([
        { id: "in", name: "Input Features [10]", type: "dense", inShape: [10], outShape: [16], paramsCount: 160 },
        { id: "z", name: "Latent Bottleneck [2]", type: "latent_space", inShape: [16], outShape: [2], paramsCount: 32 },
        { id: "out", name: "Reconstruction [10]", type: "dense", inShape: [2], outShape: [10], paramsCount: 160 }
      ], 380, 210);
      const n = t.getContext("2d");
      n && (n.fillStyle = "#020617", n.fillRect(0, 0, t.width, t.height), n.fillStyle = "#38BDF8", n.font = "13px monospace", n.fillText("2D LATENT MANIFOLD PROJECTION:", 20, 35), n.fillText(`z = [${l.latent.data[0].toFixed(3)}, ${l.latent.data[1].toFixed(3)}]`, 20, 65), n.fillText(`Reconstruction Loss (MSE): ${l.mse.toFixed(4)}`, 20, 95), n.fillStyle = "#FFD700", n.fillRect(t.width / 2 + l.latent.data[0] * 30, t.height / 2 + l.latent.data[1] * 30, 8, 8)), a.innerHTML = i.tracer.steps.map((c) => `• <b>${c.layerName}:</b> ${c.formula} → ${c.pedagogicalInsight}`).join("<br>");
    } else if (e === "lda") {
      o.innerHTML = b.renderNetwork([
        { id: "alpha", name: "Dirichlet Prior α", type: "dense", inShape: [1], outShape: [3], paramsCount: 0 },
        { id: "topics", name: "Topic Mixtures θ", type: "dense", inShape: [3], outShape: [5], paramsCount: 0 },
        { id: "words", name: "Word Distribution β", type: "dense", inShape: [5], outShape: [100], paramsCount: 0 }
      ], 380, 210);
      const i = t.getContext("2d");
      i && (i.fillStyle = "#020617", i.fillRect(0, 0, t.width, t.height), i.fillStyle = "#38BDF8", i.font = "12px monospace", i.fillText("LATENT DIRICHLET ALLOCATION:", 15, 25), i.fillStyle = "#0F8B8D", i.fillText("Topic 1 (Tech): neural (0.42), data (0.31), compute (0.22)", 15, 60), i.fillStyle = "#6B2D7B", i.fillText("Topic 2 (Finance): return (0.38), risk (0.29), market (0.24)", 15, 95), i.fillStyle = "#D98E04", i.fillText("Topic 3 (Health): patient (0.45), trial (0.28), clinic (0.19)", 15, 130)), a.innerHTML = "• <b>Generative Story:</b> For each document $d$, sample $\\theta_d \\sim \\text{Dir}(\\alpha)$. For each word token $w_n$, sample topic $z_n \\sim \\text{Multinomial}(\\theta_d)$ and emit word from $\\beta_{z_n}$.";
    } else if (e === "stm") {
      o.innerHTML = b.renderNetwork([
        { id: "cov", name: "Document Covariates X", type: "dense", inShape: [3], outShape: [3], paramsCount: 0 },
        { id: "prev", name: "Topic Prevalence Γ", type: "dense", inShape: [3], outShape: [4], paramsCount: 12 },
        { id: "words", name: "Content Covariates Y", type: "dense", inShape: [4], outShape: [100], paramsCount: 400 }
      ], 380, 210);
      const i = t.getContext("2d");
      i && (i.fillStyle = "#020617", i.fillRect(0, 0, t.width, t.height), i.fillStyle = "#38BDF8", i.font = "12px monospace", i.fillText("STRUCTURAL TOPIC MODEL (STM):", 15, 25), i.fillStyle = "#E2E8F0", i.fillText("Topic Prevalence ~ X * Gamma (Author Party, Date)", 15, 60), i.fillText("Topic Content ~ Y * Kappa (Vocabulary Shifts)", 15, 90)), a.innerHTML = "• <b>Key Advantage:</b> Explicitly integrates document-level metadata (covariates) into topic prevalence and content.";
    } else if (e === "classification") {
      o.innerHTML = b.renderNetwork([
        { id: "in", name: "Text Features", type: "dense", inShape: [500], outShape: [64], paramsCount: 32e3 },
        { id: "drop", name: "Dropout (p=0.5)", type: "dense", inShape: [64], outShape: [64], paramsCount: 0 },
        { id: "out", name: "Softmax Classes", type: "dense", inShape: [64], outShape: [4], paramsCount: 256 }
      ], 380, 210);
      const i = t.getContext("2d");
      i && (i.fillStyle = "#020617", i.fillRect(0, 0, t.width, t.height), i.fillStyle = "#38BDF8", i.font = "12px monospace", i.fillText("CLASSIFICATION PROBABILITIES:", 15, 30), [
        { name: "Politics", p: 0.72 },
        { name: "Sports", p: 0.14 },
        { name: "Sci/Tech", p: 0.09 },
        { name: "Business", p: 0.05 }
      ].forEach((l, n) => {
        i.fillStyle = "#0F8B8D", i.fillRect(15, 55 + n * 35, l.p * 220, 20), i.fillStyle = "#FFF", i.fillText(`${l.name}: ${(l.p * 100).toFixed(0)}%`, 25, 70 + n * 35);
      })), a.innerHTML = "• <b>Dropout Layer:</b> Randomly zeroes 50% of activations during training to prevent co-adaptation.<br>• <b>Softmax Head:</b> Turns raw logits into well-calibrated class posterior probabilities.";
    } else if (e === "ner") {
      o.innerHTML = b.renderNetwork([
        { id: "tok", name: "Tokens", type: "dense", inShape: [5], outShape: [64], paramsCount: 0 },
        { id: "bilstm", name: "BiLSTM (Fwd+Bwd)", type: "lstm_cell", inShape: [64], outShape: [128], paramsCount: 48e3 },
        { id: "bio", name: "BIO Tag Logits", type: "dense", inShape: [128], outShape: [7], paramsCount: 896 }
      ], 380, 210);
      const i = t.getContext("2d");
      i && (i.fillStyle = "#020617", i.fillRect(0, 0, t.width, t.height), i.fillStyle = "#38BDF8", i.font = "12px monospace", i.fillText("NAMED ENTITY RECOGNITION (BIO):", 15, 25), [
        { t: "Barack", tag: "B-PER" },
        { t: "Obama", tag: "I-PER" },
        { t: "visited", tag: "O" },
        { t: "Harvard", tag: "B-ORG" },
        { t: "University", tag: "I-ORG" }
      ].forEach((l, n) => {
        i.fillStyle = l.tag.startsWith("B") ? "#C8102E" : l.tag.startsWith("I") ? "#D98E04" : "#475569", i.fillRect(15, 50 + n * 35, 75, 22), i.fillStyle = "#FFF", i.fillText(l.tag, 22, 66 + n * 35), i.fillStyle = "#E2E8F0", i.fillText(`"${l.t}"`, 105, 66 + n * 35);
      })), a.innerHTML = "• <b>BIO Schema:</b> <code>B-</code> (Begin entity), <code>I-</code> (Inside entity), <code>O</code> (Outside entity).";
    } else if (e === "gan") {
      o.innerHTML = b.renderNetwork([
        { id: "noise", name: "Noise Vector z", type: "dense", inShape: [100], outShape: [128], paramsCount: 12800 },
        { id: "gen", name: "Generator G(z)", type: "dense", inShape: [128], outShape: [784], paramsCount: 100352 },
        { id: "disc", name: "Discriminator D(x)", type: "dense", inShape: [784], outShape: [1], paramsCount: 785 }
      ], 380, 210);
      const i = t.getContext("2d");
      i && (i.fillStyle = "#020617", i.fillRect(0, 0, t.width, t.height), i.fillStyle = "#38BDF8", i.font = "12px monospace", i.fillText("MINIMAX GAME: min_G max_D V(D, G)", 15, 25), i.fillStyle = "#C8102E", i.fillText("Discriminator D(x) -> 1 (Real) vs 0 (Fake)", 15, 65), i.fillStyle = "#0F8B8D", i.fillText("Generator G(z) -> Fooled D(G(z)) -> 1", 15, 100)), a.innerHTML = "• <b>Zero-Sum Game Objective:</b> $\\min_G \\max_D \\mathbb{E}_{x}[\\log D(x)] + \\mathbb{E}_{z}[\\log(1 - D(G(z)))]$.";
    } else if (e === "transformer") {
      const i = new z(8, 2), r = ["The", "neural", "network", "attends", "to", "tokens"], l = d.random([r.length, 8], -0.5, 0.5), n = i.forward(l, !0, !0);
      D.renderAttentionMatrix(t, n.attentionWeights, r), o.innerHTML = b.renderNetwork([
        { id: "emb", name: "Token Embeddings", type: "embedding", inShape: [6], outShape: [6, 8], paramsCount: 48 },
        { id: "qkv", name: "Q, K, V Projections", type: "dense", inShape: [6, 8], outShape: [6, 8], paramsCount: 192 },
        { id: "attn", name: "Attention Matrix", type: "self_attention", inShape: [6, 8], outShape: [6, 6], paramsCount: 0 },
        { id: "out", name: "Linear Projection", type: "dense", inShape: [6, 8], outShape: [6, 8], paramsCount: 64 }
      ], 380, 210), a.innerHTML = i.tracer.steps.map((c) => `• <b>${c.layerName}:</b> ${c.formula} → ${c.pedagogicalInsight}`).join("<br>"), s.onclick = () => this.initModel("transformer");
    } else if (e === "optimizers") {
      o.innerHTML = b.renderNetwork([
        { id: "loss", name: "Loss L(w)", type: "dense", inShape: [2], outShape: [1], paramsCount: 0 },
        { id: "grad", name: "Gradient ∇L", type: "dense", inShape: [2], outShape: [2], paramsCount: 0 },
        { id: "mom", name: "Momentum v_t", type: "dense", inShape: [2], outShape: [2], paramsCount: 0 },
        { id: "adam", name: "Adam m_t / √v_t", type: "dense", inShape: [2], outShape: [2], paramsCount: 0 }
      ], 380, 210);
      const i = t.getContext("2d");
      i && (i.fillStyle = "#020617", i.fillRect(0, 0, t.width, t.height), i.fillStyle = "#38BDF8", i.font = "12px monospace", i.fillText("OPTIMIZER TRAJECTORY RACE:", 15, 25), i.fillStyle = "#C8102E", i.fillText("● SGD: High oscillation in ravines", 15, 60), i.fillStyle = "#D98E04", i.fillText("● Momentum: Damps oscillations along ridges", 15, 95), i.fillStyle = "#0F8B8D", i.fillText("● Adam: Adaptive learning rates per parameter", 15, 130)), a.innerHTML = "• <b>Adam Update Rule:</b> $w_{t+1} = w_t - \\frac{\\alpha}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t$ combining first and second raw moments.";
    }
  }
}
typeof window < "u" && !customElements.get("neural-sim") && customElements.define("neural-sim", R);
export {
  N as AutoencoderModel,
  E as CNNModel,
  D as CanvasVisualizer,
  C as ExecutionTracer,
  L as LSTMModel,
  B as MLP,
  R as NeuralSimElement,
  b as SVGDiagramRenderer,
  d as Tensor,
  z as TransformerAttention
};
//# sourceMappingURL=omni-neural-sim.es.js.map
