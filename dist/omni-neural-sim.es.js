var P = Object.defineProperty;
var R = (D, t, a) => t in D ? P(D, t, { enumerable: !0, configurable: !0, writable: !0, value: a }) : D[t] = a;
var w = (D, t, a) => R(D, typeof t != "symbol" ? t + "" : t, a);
class b {
  constructor(t, a) {
    w(this, "data");
    w(this, "shape");
    w(this, "strides");
    w(this, "size");
    this.shape = [...t], this.size = t.reduce((e, o) => e * o, 1), this.strides = b.computeStrides(this.shape), a ? this.data = a instanceof Float32Array ? a : new Float32Array(a) : this.data = new Float32Array(this.size);
  }
  static computeStrides(t) {
    const a = new Array(t.length);
    let e = 1;
    for (let o = t.length - 1; o >= 0; o--)
      a[o] = e, e *= t[o];
    return a;
  }
  static zeros(t) {
    return new b(t);
  }
  static ones(t) {
    const a = new b(t);
    return a.data.fill(1), a;
  }
  static random(t, a = -1, e = 1) {
    const o = new b(t), i = e - a;
    for (let n = 0; n < o.size; n++)
      o.data[n] = a + Math.random() * i;
    return o;
  }
  static fromArray(t) {
    const a = [];
    let e = t;
    for (; Array.isArray(e); )
      a.push(e.length), e = e[0];
    const o = [];
    function i(n) {
      if (Array.isArray(n))
        for (const p of n) i(p);
      else
        o.push(Number(n));
    }
    return i(t), new b(a, o);
  }
  clone() {
    return new b(this.shape, new Float32Array(this.data));
  }
  getIndex(...t) {
    let a = 0;
    for (let e = 0; e < t.length; e++)
      a += t[e] * this.strides[e];
    return a;
  }
  get(...t) {
    return this.data[this.getIndex(...t)];
  }
  set(t, ...a) {
    this.data[this.getIndex(...a)] = t;
  }
  // --- Element-wise arithmetic ---
  add(t) {
    const a = new b(this.shape);
    if (typeof t == "number")
      for (let e = 0; e < this.size; e++) a.data[e] = this.data[e] + t;
    else
      for (let e = 0; e < this.size; e++) a.data[e] = this.data[e] + t.data[e];
    return a;
  }
  sub(t) {
    const a = new b(this.shape);
    if (typeof t == "number")
      for (let e = 0; e < this.size; e++) a.data[e] = this.data[e] - t;
    else
      for (let e = 0; e < this.size; e++) a.data[e] = this.data[e] - t.data[e];
    return a;
  }
  mul(t) {
    const a = new b(this.shape);
    if (typeof t == "number")
      for (let e = 0; e < this.size; e++) a.data[e] = this.data[e] * t;
    else
      for (let e = 0; e < this.size; e++) a.data[e] = this.data[e] * t.data[e];
    return a;
  }
  // --- Matrix Multiplication (2D) ---
  matmul(t) {
    if (this.shape.length !== 2 || t.shape.length !== 2)
      throw new Error(`matmul requires 2D tensors, got ${this.shape} and ${t.shape}`);
    const [a, e] = this.shape, [o, i] = t.shape;
    if (e !== o)
      throw new Error(`Incompatible matrix dims: [${a}, ${e}] x [${o}, ${i}]`);
    const n = new b([a, i]);
    for (let p = 0; p < a; p++)
      for (let l = 0; l < i; l++) {
        let h = 0;
        for (let m = 0; m < e; m++)
          h += this.get(p, m) * t.get(m, l);
        n.set(h, p, l);
      }
    return n;
  }
  transpose() {
    if (this.shape.length !== 2)
      throw new Error("transpose currently supports 2D tensors");
    const [t, a] = this.shape, e = new b([a, t]);
    for (let o = 0; o < t; o++)
      for (let i = 0; i < a; i++)
        e.set(this.get(o, i), i, o);
    return e;
  }
  // --- Activations ---
  relu() {
    const t = new b(this.shape);
    for (let a = 0; a < this.size; a++)
      t.data[a] = Math.max(0, this.data[a]);
    return t;
  }
  sigmoid() {
    const t = new b(this.shape);
    for (let a = 0; a < this.size; a++)
      t.data[a] = 1 / (1 + Math.exp(-this.data[a]));
    return t;
  }
  tanh() {
    const t = new b(this.shape);
    for (let a = 0; a < this.size; a++)
      t.data[a] = Math.tanh(this.data[a]);
    return t;
  }
  gelu() {
    const t = new b(this.shape), a = Math.sqrt(2 / Math.PI);
    for (let e = 0; e < this.size; e++) {
      const o = this.data[e];
      t.data[e] = 0.5 * o * (1 + Math.tanh(a * (o + 0.044715 * Math.pow(o, 3))));
    }
    return t;
  }
  softmax(t = -1) {
    const a = new b(this.shape);
    if (this.shape.length === 1) {
      let e = -1 / 0;
      for (let i = 0; i < this.size; i++) this.data[i] > e && (e = this.data[i]);
      let o = 0;
      for (let i = 0; i < this.size; i++)
        a.data[i] = Math.exp(this.data[i] - e), o += a.data[i];
      for (let i = 0; i < this.size; i++) a.data[i] /= o;
      return a;
    }
    if (this.shape.length === 2) {
      const [e, o] = this.shape;
      for (let i = 0; i < e; i++) {
        let n = -1 / 0;
        for (let l = 0; l < o; l++) {
          const h = this.get(i, l);
          h > n && (n = h);
        }
        let p = 0;
        for (let l = 0; l < o; l++) {
          const h = Math.exp(this.get(i, l) - n);
          a.set(h, i, l), p += h;
        }
        for (let l = 0; l < o; l++)
          a.set(a.get(i, l) / p, i, l);
      }
      return a;
    }
    throw new Error("softmax supports 1D or 2D tensors");
  }
  // --- 2D Convolution & Pooling ---
  /**
   * 2D Convolution on [H, W, InChannels] with Kernel [Kh, Kw, InChannels, OutChannels]
   */
  conv2d(t, a, e = 1, o = 0) {
    const [i, n, p] = this.shape, [l, h, m, v] = t.shape;
    if (p !== m)
      throw new Error(`Channel mismatch: input has ${p}, kernel expects ${m}`);
    const _ = Math.floor((i - l + 2 * o) / e) + 1, x = Math.floor((n - h + 2 * o) / e) + 1, g = new b([_, x, v]);
    for (let r = 0; r < v; r++) {
      const c = a ? a.data[r] : 0;
      for (let f = 0; f < _; f++)
        for (let s = 0; s < x; s++) {
          let d = c;
          const S = f * e - o, u = s * e - o;
          for (let y = 0; y < l; y++) {
            const k = S + y;
            if (!(k < 0 || k >= i))
              for (let C = 0; C < h; C++) {
                const F = u + C;
                if (!(F < 0 || F >= n))
                  for (let T = 0; T < p; T++)
                    d += this.get(k, F, T) * t.get(y, C, T, r);
              }
          }
          g.set(d, f, s, r);
        }
    }
    return g;
  }
  maxPool2d(t = 2, a = 2) {
    const [e, o, i] = this.shape, n = Math.floor((e - t) / a) + 1, p = Math.floor((o - t) / a) + 1, l = new b([n, p, i]);
    for (let h = 0; h < i; h++)
      for (let m = 0; m < n; m++)
        for (let v = 0; v < p; v++) {
          let _ = -1 / 0;
          const x = m * a, g = v * a;
          for (let r = 0; r < t; r++)
            for (let c = 0; c < t; c++) {
              const f = this.get(x + r, g + c, h);
              f > _ && (_ = f);
            }
          l.set(_, m, v, h);
        }
    return l;
  }
  flatten() {
    return new b([this.size], this.data);
  }
  toArray() {
    if (this.shape.length === 1)
      return Array.from(this.data);
    if (this.shape.length === 2) {
      const [t, a] = this.shape, e = [];
      for (let o = 0; o < t; o++) {
        const i = [];
        for (let n = 0; n < a; n++) i.push(this.get(o, n));
        e.push(i);
      }
      return e;
    }
    return Array.from(this.data);
  }
}
class W {
  constructor() {
    w(this, "steps", []);
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
class N {
  constructor(t) {
    w(this, "weights", []);
    w(this, "biases", []);
    w(this, "activations", []);
    w(this, "tracer", new W());
    this.config = t, this.activations = t.activations;
    for (let a = 0; a < t.layerSizes.length - 1; a++) {
      const e = t.layerSizes[a], o = t.layerSizes[a + 1], i = Math.sqrt(2 / e);
      this.weights.push(b.random([e, o], -i, i)), this.biases.push(b.zeros([1, o]));
    }
  }
  forward(t, a = !1) {
    a && this.tracer.clear();
    let e = t;
    for (let o = 0; o < this.weights.length; o++) {
      const i = this.weights[o], n = this.biases[o], p = this.activations[o], l = e.matmul(i).add(n);
      let h = l;
      p === "relu" ? h = l.relu() : p === "sigmoid" ? h = l.sigmoid() : p === "tanh" && (h = l.tanh()), a && this.tracer.record({
        layerId: `layer_${o + 1}`,
        layerName: `Hidden Layer ${o + 1} (${p.toUpperCase()})`,
        operation: "Dense Matmul + Bias + Activation",
        formula: `a^[${o + 1}] = ${p}(W^[${o + 1}] * a^[${o}] + b^[${o + 1}])`,
        inputShapes: [e.shape, i.shape],
        outputShape: h.shape,
        tensorPreview: Array.from(h.data.slice(0, 4)),
        pedagogicalInsight: `Layer ${o + 1} transforms ${i.shape[0]} inputs into ${i.shape[1]} linear combinations, activated by ${p}.`
      }), e = h;
    }
    return e;
  }
  // Train a single epoch on a dataset X, Y
  trainStep(t, a) {
    let e = 0;
    const o = this.config.learningRate;
    for (let i = 0; i < t.length; i++) {
      const n = b.fromArray([t[i]]), p = a[i], l = [n], h = [];
      let m = n;
      for (let g = 0; g < this.weights.length; g++) {
        const r = m.matmul(this.weights[g]).add(this.biases[g]);
        h.push(r);
        const c = this.activations[g];
        c === "relu" ? m = r.relu() : c === "sigmoid" ? m = r.sigmoid() : c === "tanh" && (m = r.tanh()), l.push(m);
      }
      const _ = m.data[0] - p[0];
      e += 0.5 * _ * _;
      let x = new b([1, 1], [_]);
      for (let g = this.weights.length - 1; g >= 0; g--) {
        const r = h[g], c = this.activations[g], f = l[g], s = new b(r.shape);
        for (let u = 0; u < r.size; u++)
          if (c === "relu") s.data[u] = r.data[u] > 0 ? 1 : 0;
          else if (c === "sigmoid") {
            const y = 1 / (1 + Math.exp(-r.data[u]));
            s.data[u] = y * (1 - y);
          } else if (c === "tanh") {
            const y = Math.tanh(r.data[u]);
            s.data[u] = 1 - y * y;
          } else
            s.data[u] = 1;
        const d = x.mul(s), S = f.transpose().matmul(d);
        for (let u = 0; u < this.weights[g].size; u++)
          this.weights[g].data[u] -= o * S.data[u];
        for (let u = 0; u < this.biases[g].size; u++)
          this.biases[g].data[u] -= o * d.data[u];
        g > 0 && (x = d.matmul(this.weights[g].transpose()));
      }
    }
    return { loss: e / t.length };
  }
  /**
   * Evaluate a 2D grid of points for boundary visualization (e.g., XOR, Circles, Moons)
   */
  evaluateGrid(t = 30, a = 4) {
    const e = [], o = a * 2 / t, i = [], n = [];
    for (let p = 0; p < t; p++) {
      const l = [], h = a - p * o;
      n.push(h);
      for (let m = 0; m < t; m++) {
        const v = -a + m * o;
        p === 0 && i.push(v);
        const _ = this.forward(new b([1, 2], [v, h]));
        l.push(_.data[0]);
      }
      e.push(l);
    }
    return { grid: e, xRange: i, yRange: n };
  }
}
class G {
  constructor() {
    w(this, "tracer", new W());
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
    this.kernel1 = b.random([3, 3, 1, 32], -0.2, 0.2), this.bias1 = b.zeros([32]), this.kernel2 = b.random([3, 3, 32, 64], -0.15, 0.15), this.bias2 = b.zeros([64]), this.kernel3 = b.random([3, 3, 64, 64], -0.15, 0.15), this.bias3 = b.zeros([64]), this.dense1_W = b.random([576, 64], -0.1, 0.1), this.dense1_B = b.zeros([1, 64]), this.dense2_W = b.random([64, 10], -0.1, 0.1), this.dense2_B = b.zeros([1, 10]);
  }
  forward(t, a = !0) {
    a && this.tracer.clear();
    const e = t.conv2d(this.kernel1, this.bias1, 1, 0).relu();
    a && this.tracer.record({
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
    a && this.tracer.record({
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
    const i = o.conv2d(this.kernel2, this.bias2, 1, 0).relu();
    a && this.tracer.record({
      layerId: "conv2",
      layerName: "Conv2D (64 filters, 3x3)",
      operation: "2D Convolution + ReLU",
      formula: "H_out = (13 - 3 + 0)/1 + 1 = 11; Params = (3*3*32 + 1)*64 = 18,496",
      inputShapes: [o.shape, this.kernel2.shape],
      outputShape: i.shape,
      tensorPreview: Array.from(i.data.slice(0, 5)),
      fullOutput: i,
      pedagogicalInsight: "Combines local features into mid-level parts (corners, loops, strokes)."
    });
    const n = i.maxPool2d(2, 2);
    a && this.tracer.record({
      layerId: "pool2",
      layerName: "MaxPooling2D (2x2)",
      operation: "Spatial Downsampling",
      formula: "H_out = floor(11/2) = 5; W_out = 5",
      inputShapes: [i.shape],
      outputShape: n.shape,
      tensorPreview: Array.from(n.data.slice(0, 5)),
      fullOutput: n,
      pedagogicalInsight: "Further downsamples to 5x5 feature grids."
    });
    const p = n.conv2d(this.kernel3, this.bias3, 1, 0).relu();
    a && this.tracer.record({
      layerId: "conv3",
      layerName: "Conv2D (64 filters, 3x3)",
      operation: "2D Convolution + ReLU",
      formula: "H_out = (5 - 3 + 0)/1 + 1 = 3; Params = (3*3*64 + 1)*64 = 36,928",
      inputShapes: [n.shape, this.kernel3.shape],
      outputShape: p.shape,
      tensorPreview: Array.from(p.data.slice(0, 5)),
      fullOutput: p,
      pedagogicalInsight: "High-level digit shape representations."
    });
    const l = p.flatten(), v = new b([1, 576], l.data).matmul(this.dense1_W).add(this.dense1_B).relu().matmul(this.dense2_W).add(this.dense2_B), _ = v.softmax(-1), x = Array.from(_.data);
    let g = 0, r = -1;
    for (let c = 0; c < 10; c++)
      x[c] > r && (r = x[c], g = c);
    return a && this.tracer.record({
      layerId: "digit_probs",
      layerName: "Softmax Classification Head",
      operation: "Softmax(logits) -> Argmax",
      formula: "P(digit=k) = exp(z_k) / sum(exp(z)); Predicted = argmax_k(P)",
      inputShapes: [v.shape],
      outputShape: [10],
      tensorPreview: x,
      pedagogicalInsight: `Final digit classification with top prediction ${g} (${(r * 100).toFixed(1)}%).`
    }), {
      probabilities: x,
      predictedDigit: g,
      conv1Out: e,
      pool1Out: o,
      conv2Out: i,
      pool2Out: n,
      conv3Out: p
    };
  }
  loadWeightsFromJSON(t) {
    t.kernel1 && (this.kernel1 = new b([3, 3, 1, 32], t.kernel1)), t.bias1 && (this.bias1 = new b([32], t.bias1)), t.dense2_W && (this.dense2_W = new b([64, 10], t.dense2_W)), t.dense2_B && (this.dense2_B = new b([1, 10], t.dense2_B));
  }
}
class H {
  constructor(t = 4, a = 4) {
    w(this, "hiddenSize");
    w(this, "inputSize");
    w(this, "tracer", new W());
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
    this.inputSize = t, this.hiddenSize = a;
    const e = 0.5;
    this.W_f = b.random([t, a], -e, e), this.U_f = b.random([a, a], -e, e), this.b_f = b.ones([1, a]), this.W_i = b.random([t, a], -e, e), this.U_i = b.random([a, a], -e, e), this.b_i = b.zeros([1, a]), this.W_c = b.random([t, a], -e, e), this.U_c = b.random([a, a], -e, e), this.b_c = b.zeros([1, a]), this.W_o = b.random([t, a], -e, e), this.U_o = b.random([a, a], -e, e), this.b_o = b.zeros([1, a]);
  }
  step(t, a, e) {
    const o = t.matmul(this.W_f).add(a.matmul(this.U_f)).add(this.b_f).sigmoid(), i = t.matmul(this.W_i).add(a.matmul(this.U_i)).add(this.b_i).sigmoid(), n = t.matmul(this.W_c).add(a.matmul(this.U_c)).add(this.b_c).tanh(), p = o.mul(e).add(i.mul(n)), l = t.matmul(this.W_o).add(a.matmul(this.U_o)).add(this.b_o).sigmoid();
    return { h: l.mul(p.tanh()), c: p, gates: { f: o, i, c_tilde: n, o: l } };
  }
  unroll(t) {
    this.tracer.clear();
    const a = [];
    let e = b.zeros([1, this.hiddenSize]), o = b.zeros([1, this.hiddenSize]);
    for (let i = 0; i < t.length; i++) {
      const n = t[i], p = new b([1, this.inputSize], n.vector), l = this.step(p, e, o);
      e = l.h, o = l.c;
      const h = {
        t: i,
        inputToken: n.token,
        x_t: Array.from(p.data),
        f_gate: Array.from(l.gates.f.data),
        i_gate: Array.from(l.gates.i.data),
        c_tilde: Array.from(l.gates.c_tilde.data),
        c_t: Array.from(o.data),
        o_gate: Array.from(l.gates.o.data),
        h_t: Array.from(e.data)
      };
      a.push(h), this.tracer.record({
        layerId: `lstm_step_${i}`,
        layerName: `LSTM Step t=${i} ("${n.token}")`,
        operation: "Recurrent Cell State Transition",
        formula: "c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t; h_t = o_t ⊙ tanh(c_t)",
        inputShapes: [p.shape, e.shape, o.shape],
        outputShape: e.shape,
        tensorPreview: Array.from(e.data),
        pedagogicalInsight: `Step t=${i}: Forget gate retained ${(h.f_gate[0] * 100).toFixed(0)}% of prior cell memory while input gate injected ${(h.i_gate[0] * 100).toFixed(0)}% of new token information.`
      });
    }
    return a;
  }
}
class O {
  constructor(t = 8, a = 2) {
    w(this, "d_model");
    w(this, "d_k");
    w(this, "numHeads");
    w(this, "tracer", new W());
    w(this, "W_q");
    w(this, "W_k");
    w(this, "W_v");
    w(this, "W_o");
    this.d_model = t, this.numHeads = a, this.d_k = Math.floor(t / a);
    const e = Math.sqrt(2 / t);
    this.W_q = b.random([t, t], -e, e), this.W_k = b.random([t, t], -e, e), this.W_v = b.random([t, t], -e, e), this.W_o = b.random([t, t], -e, e);
  }
  /**
   * Forward pass over token embeddings matrix X of shape [seqLen, d_model]
   */
  forward(t, a = !0, e = !0) {
    e && this.tracer.clear();
    const o = t.shape[0], i = t.matmul(this.W_q), n = t.matmul(this.W_k), p = t.matmul(this.W_v), l = 1 / Math.sqrt(this.d_model), h = n.transpose(), m = i.matmul(h).mul(l);
    if (a)
      for (let r = 0; r < o; r++)
        for (let c = r + 1; c < o; c++)
          m.set(-1e9, r, c);
    const v = m.softmax(-1), x = v.matmul(p).matmul(this.W_o);
    e && this.tracer.record({
      layerId: "self_attention",
      layerName: `Self-Attention (d_model=${this.d_model}, seqLen=${o})`,
      operation: "Scaled Dot-Product Attention",
      formula: "Attention(Q,K,V) = softmax(Q K^T / √d_k + Mask) V",
      inputShapes: [t.shape, this.W_q.shape],
      outputShape: x.shape,
      tensorPreview: Array.from(v.data.slice(0, 6)),
      pedagogicalInsight: `Calculated ${o}x${o} pairwise attention affinity matrix; tokens dynamically aggregate information from allowed preceding contexts.`
    });
    const g = [];
    for (let r = 0; r < o; r++) {
      const c = [];
      for (let f = 0; f < o; f++)
        c.push(v.get(r, f));
      g.push(c);
    }
    return {
      attentionWeights: g,
      output: x
    };
  }
}
class V {
  constructor(t = 10, a = 2) {
    w(this, "tracer", new W());
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
    this.inDim = t, this.latentDim = a;
    const e = 0.3;
    this.W_enc1 = b.random([t, 16], -e, e), this.b_enc1 = b.zeros([1, 16]), this.W_enc2 = b.random([16, a], -e, e), this.b_enc2 = b.zeros([1, a]), this.W_dec1 = b.random([a, 16], -e, e), this.b_dec1 = b.zeros([1, 16]), this.W_dec2 = b.random([16, t], -e, e), this.b_dec2 = b.zeros([1, t]);
  }
  encode(t) {
    return t.matmul(this.W_enc1).add(this.b_enc1).relu().matmul(this.W_enc2).add(this.b_enc2);
  }
  decode(t) {
    return t.matmul(this.W_dec1).add(this.b_dec1).relu().matmul(this.W_dec2).add(this.b_dec2).sigmoid();
  }
  forward(t, a = !0) {
    a && this.tracer.clear();
    const e = this.encode(t), o = this.decode(e);
    let i = 0;
    for (let n = 0; n < t.size; n++) {
      const p = t.data[n] - o.data[n];
      i += p * p;
    }
    return i /= t.size, a && this.tracer.record({
      layerId: "latent_bottleneck",
      layerName: `Latent Space (dim=${this.latentDim})`,
      operation: "Nonlinear Dimensionality Compression",
      formula: "z = W_2 * relu(W_1 x + b_1) + b_2",
      inputShapes: [t.shape],
      outputShape: e.shape,
      tensorPreview: Array.from(e.data),
      pedagogicalInsight: `Compressed input from ${this.inDim} dimensions to ${this.latentDim} latent coordinates with MSE=${i.toFixed(4)}.`
    }), { latent: e, reconstructed: o, mse: i };
  }
}
class A {
  static renderNetwork(t, a = 640, e = 360, o = -1, i = "forward") {
    const n = t.length, p = a / (n + 1), l = [], h = 68, v = e - h - 55;
    t.forEach((f, s) => {
      const d = (s + 1) * p, S = Math.min(f.outShape[f.outShape.length - 1] || 4, 8), u = v / (S + 1), y = [];
      for (let k = 0; k < S; k++)
        y.push({
          layerIndex: s,
          nodeIndex: k,
          x: d,
          y: h + (k + 1) * u,
          label: `${f.name} [${k}]`
        });
      l.push(y);
    });
    const _ = i === "backward", x = _ ? "#C8102E" : "#0F8B8D", g = _ ? "reverseFlow" : "flowPulse";
    let r = `<svg viewBox="0 0 ${a} ${e}" width="100%" height="${e}" preserveAspectRatio="xMidYMid meet" style="background: radial-gradient(circle at 50% 50%, #0F172A 0%, #020617 100%); border-radius:12px; font-family:system-ui, -apple-system, sans-serif; display:block;">`;
    r += `<defs>
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
        .flow-line { stroke-dasharray: 6 3; animation: ${g} 0.9s linear infinite; }
        .active-pulse { animation: nodePulse 1.2s ease-in-out infinite alternate; }
        @keyframes nodePulse { from { transform: scale(1); filter: drop-shadow(0 0 2px #FFD700); } to { transform: scale(1.2); filter: drop-shadow(0 0 8px #FFD700); } }
      </style>
    </defs>`;
    for (let f = 0; f < l.length - 1; f++) {
      const s = l[f], d = l[f + 1], S = f === o || f + 1 === o, u = _ ? "url(#edgeGradBwd)" : "url(#edgeGradFwd)";
      for (const y of s)
        for (const k of d) {
          const C = S ? 'class="flow-line"' : "", F = S ? "2.2" : "1.0", T = S ? "0.9" : "0.3";
          r += `<line x1="${y.x}" y1="${y.y}" x2="${k.x}" y2="${k.y}" stroke="${u}" stroke-width="${F}" stroke-opacity="${T}" ${C}/>`;
        }
    }
    t.forEach((f, s) => {
      const d = (s + 1) * p, S = s === o, u = S ? "#FFD700" : "#E2E8F0";
      let y = f.name;
      y.length > 20 && (y = y.slice(0, 18) + "…"), r += `<text x="${d}" y="22" fill="${u}" font-size="11.5" font-weight="700" text-anchor="middle">${y}</text>`;
      const k = 76;
      r += `<rect x="${d - k / 2}" y="30" width="${k}" height="18" rx="4" fill="#1E293B" stroke="${S ? "#FFD700" : "#334155"}" stroke-width="1"/>`, r += `<text x="${d}" y="43" fill="#38BDF8" font-size="10" font-family="monospace" font-weight="600" text-anchor="middle">[${f.outShape.join("×")}]</text>`, f.paramsCount > 0 && (r += `<text x="${d}" y="${e - 36}" fill="#94A3B8" font-size="10" font-family="monospace" text-anchor="middle">${f.paramsCount.toLocaleString()} params</text>`);
    }), l.forEach((f, s) => {
      const d = s === o, S = d ? "#FFD700" : s === 0 ? "#0F8B8D" : s === l.length - 1 ? _ ? "#C8102E" : "#10B981" : "#6B2D7B";
      for (const u of f) {
        const y = d ? 'filter="url(#glowGold)" class="active-pulse"' : "";
        r += `<circle cx="${u.x}" cy="${u.y}" r="${d ? 11 : 8.5}" fill="${S}" stroke="#ffffff" stroke-width="1.8" ${y}/>`;
      }
    });
    const c = 190;
    return r += `<g transform="translate(${a / 2 - c / 2}, ${e - 24})">
      <rect width="${c}" height="20" rx="10" fill="#1E293B" stroke="${x}" stroke-width="1.2"/>
      <text x="${c / 2}" y="14" fill="${x}" font-size="9.5" font-weight="700" text-anchor="middle" letter-spacing="0.5">
        ${_ ? "◀ BACKPROPAGATION GRADIENTS" : "FORWARD INFERENCE FLOW ▶"}
      </text>
    </g>`, r += "</svg>", r;
  }
}
class E {
  /**
   * Helper to set up HiDPI canvas for ultra-sharp Retina rendering
   */
  static setupHiDPI(t) {
    const a = window.devicePixelRatio || 1, e = t.getBoundingClientRect(), o = e.width || t.width, i = e.height || t.height;
    (t.width !== o * a || t.height !== i * a) && (t.width = o * a, t.height = i * a);
    const n = t.getContext("2d");
    return n && n.setTransform(a, 0, 0, a, 0, 0), n;
  }
  /**
   * 1. Decision Boundary & Active Gradient Heatmap
   */
  static renderDecisionBoundary(t, a, e) {
    const o = t.getContext("2d");
    if (!o) return;
    const i = t.width, n = t.height, p = a.length, l = a[0].length, h = i / l, m = n / p;
    for (let v = 0; v < p; v++)
      for (let _ = 0; _ < l; _++) {
        const x = a[v][_], g = Math.floor(x * 220 + (1 - x) * 15), r = Math.floor((1 - Math.abs(x - 0.5) * 2) * 160), c = Math.floor((1 - x) * 220 + x * 25);
        o.fillStyle = `rgb(${g}, ${r}, ${c})`, o.fillRect(_ * h, v * m, h + 1, m + 1);
      }
    if (e)
      for (const v of e) {
        const _ = (v.x + 4) / 8 * i, x = (4 - v.y) / 8 * n;
        o.beginPath(), o.arc(_, x, 7, 0, Math.PI * 2), o.fillStyle = v.label === 1 ? "#FFD700" : "#FFFFFF", o.strokeStyle = "#0B1329", o.lineWidth = 2.5, o.fill(), o.stroke();
      }
  }
  /**
   * 2. CNN Interactive 3x3 Sliding Kernel Visualizer (Stanford CS231n / CNN Explainer style)
   */
  static renderCNNKernelSlide(t, a, e, o, i = 0.05) {
    var C;
    const n = t.getContext("2d");
    if (!n) return { activation: 0, formulaText: "" };
    const p = t.width, l = t.height;
    n.fillStyle = "#020617", n.fillRect(0, 0, p, l);
    const h = 20, m = 38, v = 180, _ = a.length, x = a[0].length, g = v / _;
    n.fillStyle = "#38BDF8", n.font = "bold 12px monospace", n.fillText("INPUT DIGIT PIXELS (28×28)", h, 22);
    for (let F = 0; F < _; F++)
      for (let T = 0; T < x; T++) {
        const $ = a[F][T], M = Math.floor($ * 255);
        n.fillStyle = `rgb(${M}, ${M}, ${M})`, n.fillRect(h + T * g, m + F * g, g - 0.5, g - 0.5);
      }
    const r = o.row, c = o.col;
    n.strokeStyle = "#FFD700", n.lineWidth = 2.5, n.strokeRect(h + c * g, m + r * g, g * 3, g * 3);
    const f = 225, s = 38, d = 240, S = 180;
    n.fillStyle = "#0F172A", n.strokeStyle = "#334155", n.lineWidth = 1, n.fillRect(f, s, d, S), n.strokeRect(f, s, d, S), n.fillStyle = "#FFD700", n.font = "bold 11px monospace", n.fillText("3×3 RECEPTIVE FIELD CALCULATION", f + 10, s + 18);
    let u = i;
    const y = 48;
    for (let F = 0; F < 3; F++)
      for (let T = 0; T < 3; T++) {
        const $ = ((C = a[r + F]) == null ? void 0 : C[c + T]) || 0, M = e[F][T], I = $ * M;
        u += I;
        const L = f + 12 + T * (y + 6), B = s + 28 + F * (y + 4);
        n.fillStyle = "#1E293B", n.fillRect(L, B, y, y), n.strokeStyle = "#475569", n.strokeRect(L, B, y, y), n.fillStyle = "#E2E8F0", n.font = "10px monospace", n.textAlign = "center", n.fillText(`x:${$.toFixed(1)}`, L + y / 2, B + 16), n.fillStyle = "#38BDF8", n.fillText(`w:${M.toFixed(1)}`, L + y / 2, B + 34);
      }
    n.textAlign = "left";
    const k = Math.max(0, u);
    return n.fillStyle = "#F8FAFC", n.font = "12px monospace", n.fillText(`Linear Sum z = Σ(x_i·w_i) + b = ${u.toFixed(3)}`, 20, 245), n.fillStyle = "#10B981", n.fillText(`Feature Activation = ReLU(z) = ${k.toFixed(3)}`, 20, 270), n.fillStyle = "#94A3B8", n.font = "11px monospace", n.fillText(`Output location: (${r}, ${c}) in 26×26 feature map`, 20, 295), {
      activation: k,
      formulaText: `z = Σ x_i·w_i + ${i.toFixed(2)} = ${u.toFixed(3)} → ReLU = ${k.toFixed(3)}`
    };
  }
  /**
   * 3. LSTM Memory Conveyor Belt & 4-Gate Anatomy
   */
  static renderLSTMConveyorBelt(t, a, e, o) {
    const i = t.getContext("2d");
    if (!i) return;
    const n = t.width, p = t.height;
    i.fillStyle = "#020617", i.fillRect(0, 0, n, p), i.fillStyle = "#38BDF8", i.font = "bold 13px monospace", i.fillText(`LSTM RECURRENT CELL ANATOMY (Step t=${a}: "${e}")`, 20, 28), i.strokeStyle = "#6B2D7B", i.lineWidth = 5, i.beginPath(), i.moveTo(20, 70), i.lineTo(n - 20, 70), i.stroke(), i.fillStyle = "#FFD700", i.font = "12px monospace", i.fillText(`Cell State c_t = ${o.c.toFixed(3)} (Constant Error Carousel)`, 30, 58);
    const l = [
      { name: "Forget (f_t)", val: o.f, color: "#C8102E", desc: "Retention" },
      { name: "Input (i_t)", val: o.i, color: "#0F8B8D", desc: "Write weight" },
      { name: "Candidate (c̃_t)", val: o.c_tilde, color: "#38BDF8", desc: "New signal" },
      { name: "Output (o_t)", val: o.o, color: "#10B981", desc: "Exposure" }
    ], h = Math.min(100, (n - 60) / 4), m = 20;
    l.forEach((v, _) => {
      const x = m + _ * (h + 12), g = 100, r = 130;
      i.fillStyle = "#0F172A", i.fillRect(x, g, h, r), i.strokeStyle = "#334155", i.strokeRect(x, g, h, r);
      const c = Math.min(r - 25, Math.max(6, Math.abs(v.val) * (r - 25)));
      i.fillStyle = v.color, i.fillRect(x + 5, g + r - c - 5, h - 10, c), i.fillStyle = "#FFF", i.font = "bold 10.5px sans-serif", i.fillText(v.name, x + 6, g + 18), i.fillStyle = "#FFD700", i.font = "bold 13px monospace", i.fillText(v.val.toFixed(2), x + 15, g + 75);
    }), i.fillStyle = "#F8FAFC", i.font = "12px monospace", i.fillText(`Hidden State: h_t = o_t ⊙ tanh(c_t) = ${o.h.toFixed(4)}`, 20, 265), i.fillStyle = "#94A3B8", i.font = "11px monospace", i.fillText(`Retention: ${(o.f * 100).toFixed(0)}% prior memory retained | ${(o.i * 100).toFixed(0)}% new candidate injected`, 20, 290);
  }
  /**
   * 4. Scaled Dot-Product Attention Matrix with Causal Mask
   */
  static renderAttentionMatrix(t, a, e, o = -1) {
    const i = t.getContext("2d");
    if (!i) return;
    const n = t.width, p = t.height, l = a.length, h = 90, m = 60, v = n - h - 20, _ = p - m - 20, x = v / l, g = _ / l;
    i.clearRect(0, 0, n, p), i.fillStyle = "#020617", i.fillRect(0, 0, n, p), i.fillStyle = "#38BDF8", i.font = "bold 13px monospace", i.fillText("ATTENTION WEIGHTS: Softmax(Q K^T / √d_k)", 20, 26);
    for (let r = 0; r < l; r++) {
      const c = r === o;
      for (let f = 0; f < l; f++) {
        const s = a[r][f], d = f > r;
        d ? i.fillStyle = "#0F172A" : i.fillStyle = c ? `rgba(255, 215, 0, ${Math.max(0.2, s)})` : `rgba(15, 139, 141, ${Math.max(0.12, s)})`, i.fillRect(h + f * x, m + r * g, x - 2, g - 2), x > 28 && (i.fillStyle = d ? "#334155" : s > 0.4 ? "#FFFFFF" : "#94A3B8", i.font = "10px monospace", i.textAlign = "center", i.textBaseline = "middle", i.fillText(
          d ? "—" : s.toFixed(2),
          h + f * x + x / 2,
          m + r * g + g / 2
        ));
      }
    }
    i.font = "12px sans-serif", i.textAlign = "right", i.textBaseline = "middle";
    for (let r = 0; r < l; r++) {
      const c = e[r] || `t_${r}`;
      i.fillStyle = r === o ? "#FFD700" : "#CBD5E1", i.fillText(c, h - 10, m + r * g + g / 2);
    }
    i.textAlign = "center", i.textBaseline = "bottom";
    for (let r = 0; r < l; r++) {
      const c = e[r] || `t_${r}`;
      i.fillText(c, h + r * x + x / 2, m - 8);
    }
  }
  /**
   * 5. 2D Latent Space Manifold & Feature Reconstruction (Autoencoders / VAE)
   */
  static renderLatentManifold(t, a, e, o, i) {
    const n = t.getContext("2d");
    if (!n) return;
    const p = t.width, l = t.height;
    n.fillStyle = "#020617", n.fillRect(0, 0, p, l), n.fillStyle = "#38BDF8", n.font = "bold 13px monospace", n.fillText("2D LATENT SPACE & FEATURE RECONSTRUCTION", 20, 24);
    const h = 200, m = 200, v = 20, _ = 45, x = v + h / 2, g = _ + m / 2;
    n.fillStyle = "#0F172A", n.fillRect(v, _, h, m), n.strokeStyle = "#334155", n.strokeRect(v, _, h, m), n.strokeStyle = "#1E293B";
    for (let k = -80; k <= 80; k += 25)
      n.beginPath(), n.moveTo(x + k, _), n.lineTo(x + k, _ + m), n.stroke(), n.beginPath(), n.moveTo(v, g + k), n.lineTo(v + h, g + k), n.stroke();
    [
      { x: x - 40, y: g - 35, col: "#0F8B8D" },
      { x: x + 45, y: g + 40, col: "#6B2D7B" },
      { x: x + 35, y: g - 45, col: "#38BDF8" }
    ].forEach((k) => {
      n.fillStyle = k.col;
      for (let C = 0; C < 6; C++) {
        const F = Math.sin(C * 1.5) * 20, T = Math.cos(C * 1.5) * 18;
        n.beginPath(), n.arc(k.x + F, k.y + T, 3, 0, Math.PI * 2), n.fill();
      }
    });
    const c = x + a[0] * 40, f = g - a[1] * 40;
    n.beginPath(), n.arc(c, f, 7, 0, Math.PI * 2), n.fillStyle = "#FFD700", n.fill(), n.strokeStyle = "#FFFFFF", n.lineWidth = 2, n.stroke(), n.fillStyle = "#FFD700", n.font = "bold 11px monospace", n.fillText(`z = (${a[0].toFixed(2)}, ${a[1].toFixed(2)})`, v + 10, _ + m + 18);
    const s = 245, d = 45, S = p - s - 25, u = Math.min(8, e.length), y = 22;
    n.fillStyle = "#E2E8F0", n.font = "bold 11px sans-serif", n.fillText("Feature Reconstruction Comparison:", s, d - 8);
    for (let k = 0; k < u; k++) {
      const C = d + k * y, F = e[k], T = o[k];
      n.fillStyle = "#94A3B8", n.font = "10px monospace", n.fillText(`dim ${k}:`, s, C + 10), n.fillStyle = "#0F8B8D", n.fillRect(s + 42, C, Math.max(3, F * (S - 45)), 7), n.fillStyle = "#FFD700", n.fillRect(s + 42, C + 9, Math.max(3, T * (S - 45)), 7);
    }
    n.font = "10px sans-serif", n.fillStyle = "#0F8B8D", n.fillRect(s, l - 42, 10, 10), n.fillStyle = "#E2E8F0", n.fillText("Original x", s + 16, l - 33), n.fillStyle = "#FFD700", n.fillRect(s + 85, l - 42, 10, 10), n.fillStyle = "#E2E8F0", n.fillText("Reconstructed x̂", s + 101, l - 33), n.fillStyle = "#10B981", n.font = "bold 12px monospace", n.fillText(`Reconstruction MSE: ${i.toFixed(4)}`, s, l - 14);
  }
  /**
   * 6. 3D Optimizer Loss Landscape (SGD vs Momentum vs Adam)
   */
  static renderOptimizerContour(t, a) {
    const e = t.getContext("2d");
    if (!e) return;
    const o = t.width, i = t.height;
    e.fillStyle = "#020617", e.fillRect(0, 0, o, i), e.fillStyle = "#38BDF8", e.font = "bold 13px monospace", e.fillText("LOSS LANDSCAPE TRAJECTORY (3D Ravine Contour)", 20, 26);
    const n = o / 2, p = i / 2 + 10;
    for (let m = 1; m <= 6; m++)
      e.strokeStyle = `rgba(15, 139, 141, ${0.12 * m})`, e.lineWidth = 1.3, e.beginPath(), e.ellipse(n, p, m * 32, m * 16, -Math.PI / 6, 0, Math.PI * 2), e.stroke();
    const l = Math.min(1, a % 20 / 19), h = 9;
    e.strokeStyle = "#C8102E", e.lineWidth = 2.2, e.beginPath(), e.moveTo(n - 130, p - 80);
    for (let m = 1; m <= h * l; m++) {
      const v = (m % 2 === 0 ? 38 : -38) * (1 - m / h);
      e.lineTo(n - 130 + m * 18, p - 80 + m * 10 + v);
    }
    e.stroke(), e.strokeStyle = "#D98E04", e.lineWidth = 2.2, e.beginPath(), e.moveTo(n - 130, p - 80);
    for (let m = 1; m <= h * l; m++) {
      const v = (m % 2 === 0 ? 14 : -14) * (1 - m / h);
      e.lineTo(n - 130 + m * 19, p - 80 + m * 10 + v);
    }
    e.stroke(), e.strokeStyle = "#10B981", e.lineWidth = 2.8, e.beginPath(), e.moveTo(n - 130, p - 80);
    for (let m = 1; m <= h * l; m++)
      e.lineTo(n - 130 + m * 20, p - 80 + m * 11);
    e.stroke(), e.font = "11px monospace", e.fillStyle = "#C8102E", e.fillText("● SGD: Oscillates on steep walls", 20, i - 55), e.fillStyle = "#D98E04", e.fillText("● Momentum: Dampens ravine oscillations", 20, i - 35), e.fillStyle = "#10B981", e.fillText("● Adam: Direct adaptive descent to global minimum", 20, i - 15);
  }
}
const z = {
  mlp: {
    title: "Multi-Layer Perceptron (MLP) & Analytical Backpropagation",
    syllabusContext: "Harvard CSCI E-89b Week 1 & CSCI E-89 Week 1",
    problemStatement: "Linear models (like a single perceptron) fail on non-linearly separable problems like XOR because they can only draw a single straight hyper-plane decision boundary. An MLP solves this by stacking layers of affine transformations interleaved with non-linear activation functions (Tanh, ReLU, Sigmoid), warping the input space so classes become linearly separable in the final hidden representation.",
    mathematicalMechanics: {
      forwardPass: `1. Layer 1: z^[1] = W^[1] · x + b^[1] → a^[1] = tanh(z^[1])
2. Layer 2: z^[2] = W^[2] · a^[1] + b^[2] → a^[2] = tanh(z^[2])
3. Output Layer: z^[3] = W^[3] · a^[2] + b^[3] → ŷ = σ(z^[3])`,
      lossAndUpdate: `Loss: L = 1/2 · (ŷ - y)^2
Output error: δ^[3] = (ŷ - y) ⊙ σ'(z^[3])
Hidden error: δ^[l] = ((W^[l+1])^T · δ^[l+1]) ⊙ tanh'(z^[l])
Weight gradient: ∂L/∂W^[l] = (a^[l-1])^T · δ^[l]
Update rule: W^[l] ← W^[l] - η · ∂L/∂W^[l]`,
      tensorShapes: "Input: [1, 2] → Hidden 1: [1, 6] (18 params) → Hidden 2: [1, 4] (28 params) → Output: [1, 1] (5 params). Total: 51 parameters."
    },
    visualGuide: {
      diagramElements: "Circles represent individual neurons. Connecting lines represent synaptic weight parameters. Line thickness indicates weight magnitude. During forward pass, pulses travel left-to-right (teal). During backpropagation, error gradients travel in reverse (crimson).",
      canvasMechanics: "The 2D canvas displays the continuous decision boundary surface over [-4, 4]^2. Dark navy represents class 0, teal represents neutral transition, and yellow points represent XOR training coordinates ((-2,-2)→0, (-2,2)→1, (2,-2)→1, (2,2)→0). Notice how the boundary bends non-linearly as you step through backpropagation updates.",
      colorSignaling: `• Teal (#0F8B8D): Forward activations & positive weights
• Crimson (#C8102E): Backward gradient flow & negative weights
• Gold (#FFD700): Active updating layer & class 1 labels`
    },
    pedagogicalTakeaways: [
      "Non-linear activations are mandatory; without them, stacking dense layers collapses into a single trivial linear transformation W_total = W_n · ... · W_1.",
      "Backpropagation is not a different machine learning model; it is simply an efficient computational application of the multivariate chain rule that computes exact partial derivatives in O(parameters) time rather than O(parameters^2)."
    ]
  },
  cnn: {
    title: "Convolutional Neural Networks (CNN) & Receptive Field Mechanics",
    syllabusContext: "Harvard CSCI E-89b Week 3 & CSCI E-89 Week 5",
    problemStatement: "Fully connected networks flatten images into 1D vectors, completely discarding spatial 2D locality and requiring an explosion of parameters. CNNs introduce two fundamental inductive biases: (1) Local connectivity (neurons only connect to a small patch called the receptive field), and (2) Weight sharing (the same learnable filter slides across the entire image), ensuring translation invariance and drastic parameter reduction.",
    mathematicalMechanics: {
      forwardPass: `1. 2D Convolution: z_{r,c,k} = (Σ_i Σ_j x_{r+i, c+j} · w_{i,j,k}) + b_k
2. Activation: a_{r,c,k} = ReLU(z_{r,c,k})
3. MaxPooling: p_{r,c,k} = max_{i,j ∈ {0,1}} a_{2r+i, 2c+j, k}
4. Flatten: [3, 3, 64] → [576]
5. Dense + Softmax: P(digit = k) = exp(z_k) / Σ_j exp(z_j)`,
      lossAndUpdate: `Output spatial dimensions: H_out = ⌊(H_in - K_h + 2P)/S⌋ + 1 = ⌊(28 - 3 + 0)/1⌋ + 1 = 26.
Parameter Count Law: (K_h · K_w · C_in + 1) · C_out = (3 · 3 · 1 + 1) · 32 = 320 parameters.`,
      tensorShapes: "Input: [28, 28, 1] → Conv1: [26, 26, 32] (320 params) → MaxPool1: [13, 13, 32] → Conv2: [11, 11, 64] (18,496 params) → MaxPool2: [5, 5, 64] → Conv3: [3, 3, 64] (36,928 params) → Flatten: [576] → Dense: [64] → Softmax: [10]."
    },
    visualGuide: {
      diagramElements: "The SVG illustrates the dimensional shrinkage of feature maps through convolution and 2x2 downsampling pooling layers, terminating in the flattened dense classification head.",
      canvasMechanics: "The left pane shows the raw 28x28 grayscale MNIST digit with a sliding yellow 3x3 receptive field bounding box. The right pane magnifies this exact 3x3 window, displaying the 9 numerical input pixels multiplied by the 9 filter weights, adding bias, and producing the single active feature map activation.",
      colorSignaling: `• Yellow box: Active 3x3 receptive field sliding across spatial coordinates
• Green text: Post-ReLU activation value lighting up the output feature map
• Blue numbers: Learnable filter weights W_{i,j}`
    },
    pedagogicalTakeaways: [
      "Layer 1 has 320 parameters regardless of whether the image is 28x28 or 4K resolution, because kernel weights depend exclusively on kernel size (3x3), input channels (1), and filter count (32).",
      "Softmax turns raw unbounded real numbers (logits) into a valid probability distribution that sums to 1.0; argmax selects the index with the maximum posterior probability."
    ]
  },
  lstm: {
    title: "Long Short-Term Memory (LSTM) & Constant Error Carousel",
    syllabusContext: "Harvard CSCI E-89b Week 2 & CSCI E-89 Week 11",
    problemStatement: "Vanilla RNNs suffer from vanishing and exploding gradients over long sequences because the repeated matrix multiplication (W_hh)^T causes gradients to decay exponentially to zero (if eigenvalues < 1) or explode to infinity. LSTMs solve this by establishing an additive memory highway (the Constant Error Carousel, c_t) regulated by three multiplicative sigmoid gates that decide what to forget, what to write, and what to expose.",
    mathematicalMechanics: {
      forwardPass: `1. Forget gate: f_t = σ(W_f · x_t + U_f · h_{t-1} + b_f)
2. Input gate: i_t = σ(W_i · x_t + U_i · h_{t-1} + b_i)
3. Candidate memory: c̃_t = tanh(W_c · x_t + U_c · h_{t-1} + b_c)
4. Cell state update: c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
5. Output gate: o_t = σ(W_o · x_t + U_o · h_{t-1} + b_o)
6. Hidden state: h_t = o_t ⊙ tanh(c_t)`,
      lossAndUpdate: "Because c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t is an ADDITIVE update, the gradient ∂c_t / ∂c_{t-1} = f_t. If f_t ≈ 1, error signals flow backwards across hundreds of time steps without multiplying by small weights, eliminating vanishing gradients.",
      tensorShapes: "Token input x_t: [1, 4] → Gates [f, i, c̃, o]: [1, 16] (144 params) → Cell state c_t: [1, 4] → Hidden state h_t: [1, 4]."
    },
    visualGuide: {
      diagramElements: "The SVG unrolls the recurrent transition across sequential time steps, highlighting the internal gate computations and hidden state emissions.",
      canvasMechanics: "The top purple rail represents the horizontal memory conveyor belt (Cell State c_t). Four vertical gauge meters monitor the live value of each gate: Forget gate f_t (crimson), Input gate i_t (teal), Candidate c̃_t (blue), and Output gate o_t (green). Notice how changing inputs modifies the retention percentage.",
      colorSignaling: `• Crimson gauge (Forget): 0 = purge memory, 1 = retain past indefinitely
• Teal gauge (Input): Controls write volume of current token into memory
• Gold highlight: Current active time step t and token input string`
    },
    pedagogicalTakeaways: [
      "The cell state c_t is internal long-term storage; the hidden state h_t is the filtered short-term output exposed to downstream layers.",
      "Initializing the forget gate bias b_f to +1.0 or +2.0 at the start of training is standard deep learning practice to prevent the model from inadvertently forgetting early context before learning what to remember."
    ]
  },
  transformer: {
    title: "Transformers, Scaled Dot-Product Attention & Causal Masking",
    syllabusContext: "Harvard CSCI E-89b Week 12 & CSCI E-89 Week 12",
    problemStatement: "Recurrent networks process text sequentially (O(T) time complexity), creating an inescapable computational bottleneck that prevents parallel GPU execution and compresses all historical context into a fixed-size vector. Transformers eliminate recurrence completely, enabling all tokens to attend directly to each other in parallel via Query-Key-Value dot-product affinities.",
    mathematicalMechanics: {
      forwardPass: `1. Projections: Q = X · W_Q,  K = X · W_K,  V = X · W_V
2. Attention Matrix: S = (Q · K^T) / √d_k
3. Causal Masking: S_{i,j} = -∞ for j > i (autoregressive decoder)
4. Attention Weights: A = softmax(S)
5. Context Representation: Output = A · V · W_O`,
      lossAndUpdate: "Division by √d_k is critical: for large embedding dimensions d_k, the dot products grow large in magnitude, pushing the softmax function into regions with extremely small gradients. Dividing by √d_k stabilizes the variance to 1.0.",
      tensorShapes: "Sequence X: [N, d_model] → Q, K, V: [N, d_k] → Attention Matrix: [N, N] → Output: [N, d_model]."
    },
    visualGuide: {
      diagramElements: "The SVG shows token embedding projections into Q, K, V subspace spaces, through the NxN attention affinity tensor, and recombining through the multi-head projection layer.",
      canvasMechanics: "The canvas renders the live NxN attention heatmap. Each row corresponds to a Query token; each column corresponds to a Key token. Dark tiles with '—' represent causally masked future positions that autoregressive decoders are strictly prohibited from attending to. The highlighted row indicates the token currently computing context.",
      colorSignaling: `• Gold row & cells: Current Query token and its strongest attention weights
• Teal cells: Permitted historical attention affinities
• Dark grey ('—'): Causal mask blocking future information leakage`
    },
    pedagogicalTakeaways: [
      "Attention is permutation-equivariant; without sinusoidal or learned positional encodings added to input embeddings, the model cannot distinguish between 'dog bites man' and 'man bites dog'.",
      "In autoregressive generation (GPT), the KV-Cache caches previously computed K and V vectors so generating the next token only requires computing Q for the single new token (O(N) instead of O(N^2) per step)."
    ]
  },
  autoencoder: {
    title: "Autoencoders & Variational Autoencoders (VAE) Latent Manifolds",
    syllabusContext: "Harvard CSCI E-89b Week 6 & CSCI E-89 Week 6",
    problemStatement: "High-dimensional real-world data (like text embeddings or images) is noisy and sparse, but typically lies on a lower-dimensional smooth manifold. Autoencoders discover this intrinsic coordinate system by compressing inputs through an informational bottleneck (encoder) and learning to reconstruct the original features (decoder), performing non-linear dimensionality reduction superior to linear PCA.",
    mathematicalMechanics: {
      forwardPass: `1. Encoder: h = ReLU(W_enc1 · x + b_1) → z = W_enc2 · h + b_2  (Latent Bottleneck)
2. Decoder: h_dec = ReLU(W_dec1 · z + b_3) → x̂ = σ(W_dec2 · h_dec + b_4)
3. Reconstruction Loss: L_rec = 1/D · Σ_{d=1}^D (x_d - x̂_d)^2
4. VAE KL Divergence: D_KL(q(z|x) || p(z)) = -1/2 · Σ (1 + log σ^2 - μ^2 - σ^2)`,
      lossAndUpdate: "Reparameterization trick: to backpropagate through stochastic latent sampling in VAEs, draw ε ~ N(0, I) and compute z = μ + ε ⊙ σ, moving randomness outside the differentiable computational graph.",
      tensorShapes: "Input x: [1, 10] → Encoder: [1, 16] → Latent z: [1, 2] → Decoder: [1, 16] → Reconstructed x̂: [1, 10]."
    },
    visualGuide: {
      diagramElements: "The hourglass SVG clearly visualizes the bottleneck architecture: wide input layer tapering down to the narrow 2-dimensional latent bottleneck node, then expanding back to the output reconstruction.",
      canvasMechanics: "The left pane depicts the 2D latent space coordinate plane with data cluster clouds and the active latent point z = (z1, z2). The right pane plots a direct feature-by-feature bar comparison between original input features x (teal) and decoded features x̂ (gold), alongside the live Mean Squared Error (MSE).",
      colorSignaling: `• Purple: Bottleneck latent representation z
• Teal bars: Original uncompressed ground-truth features x
• Gold bars: Reconstructed features x̂ synthesized by the decoder`
    },
    pedagogicalTakeaways: [
      "The capacity of the latent space must be strictly restricted (bottleneck); otherwise a network with sufficient capacity will simply learn the identity function without extracting meaningful semantic abstractions.",
      "Standard autoencoders have gaps in their latent space where decoded points produce unrecognizable noise; VAEs solve this by enforcing Gaussian distribution priors through KL-divergence regularization."
    ]
  },
  optimizers: {
    title: "Loss Landscapes & Deep Optimization: SGD vs Momentum vs Adam",
    syllabusContext: "Harvard CSCI E-89 Week 3 & Week 13",
    problemStatement: "Deep neural network loss functions are non-convex surfaces filled with ravines, saddle points, and plateaus. Standard Stochastic Gradient Descent (SGD) struggles severely in ravines where surface curvature is much steeper in one direction than another, oscillating violently across walls with negligible progress along the bottom toward the minimum.",
    mathematicalMechanics: {
      forwardPass: `1. Gradient: g_t = ∇_w L(w_t)
2. SGD: w_{t+1} = w_t - η · g_t
3. Momentum: v_t = γ · v_{t-1} + η · g_t → w_{t+1} = w_t - v_t
4. Adam (Adaptive Moment Estimation):
   m_t = β_1 · m_{t-1} + (1 - β_1) · g_t  (1st moment: mean)
   v_t = β_2 · v_{t-1} + (1 - β_2) · g_t^2  (2nd moment: uncentered variance)
   m̂_t = m_t / (1 - β_1^t),  v̂_t = v_t / (1 - β_2^t)  (bias correction)
   w_{t+1} = w_t - (η / (√v̂_t + ε)) · m̂_t`,
      lossAndUpdate: "Adam automatically scales the step size inversely with the square root of recent gradient magnitudes: parameters with consistently large gradients take smaller cautious steps; parameters with sparse or flat gradients take larger progressive steps.",
      tensorShapes: "Parameter vector w: [D] → Gradient ∇L: [D] → Moments m, v: [D] → Update step: [D]."
    },
    visualGuide: {
      diagramElements: "The SVG illustrates the optimizer update pipeline: converting scalar loss into directional gradient vectors, accumulating historical momentum velocity, and computing Adam's adaptive step vector.",
      canvasMechanics: "The canvas plots an elliptical ravine loss contour landscape. Three distinct optimization trajectories race down the ravine: SGD in crimson (oscillating violently between steep walls), Momentum in amber (dampening oscillations by accumulating momentum down the valley floor), and Adam in green (smooth, direct adaptive descent straight to the global minimum).",
      colorSignaling: `• Crimson line: Vanilla SGD (high oscillation, slow convergence in ravines)
• Amber line: Classical Momentum (damped oscillations, fast valley traversal)
• Green line: Adam (adaptive per-coordinate learning rate, state-of-the-art default)`
    },
    pedagogicalTakeaways: [
      "Momentum accelerates SGD in directions where gradients consistently point the same way while cancelling out oscillations in directions where gradients alternate signs.",
      "Adam's bias corrections (1 - β^t) are essential during early iterations (t=1,2,...) to prevent the moving averages from being severely biased toward their initial zero values."
    ]
  },
  ngrams: {
    title: "N-grams, Bag-of-Words (BoW) & Vocabulary OOV Firewalls",
    syllabusContext: "Harvard CSCI E-89b Week 4",
    problemStatement: "Raw natural language consists of variable-length unstructured strings that cannot be directly multiplied by neural weight matrices. N-grams and Bag-of-Words models convert text into fixed-length numerical frequency vectors, but introduce a severe data leakage risk if test-set words leak into the vocabulary.",
    mathematicalMechanics: {
      forwardPass: `1. Tokenization: Text → List of discrete tokens [w_1, w_2, ..., w_T]
2. Lemmatization: w_i → canonical dictionary lemma l_i
3. Vocabulary Mapping: Index(l) if l ∈ V else Index(<UNK>)
4. BoW Frequency Count: x[k] = Σ_{t=1}^T 𝟙(token_t = word_k)
5. N-gram Markov Chain: P(w_n | w_1, ..., w_{n-1}) ≈ P(w_n | w_{n-1})`,
      lossAndUpdate: "Maximum Likelihood Estimation for Bigrams: P(w_i | w_{i-1}) = Count(w_{i-1}, w_i) / Count(w_{i-1}). Add-1 (Laplace) smoothing handles unseen pairs: P_Laplace = (Count + 1) / (Total + |V|).",
      tensorShapes: "Vocabulary V: [|V|] terms → Document vector: [1, |V|] frequency counts."
    },
    visualGuide: {
      diagramElements: "The SVG displays the text pipeline: token stream entering a vocabulary hash table, passing through the OOV firewall, and forming the fixed-dimensional frequency vector.",
      canvasMechanics: "The canvas provides a streaming frequency bar chart for each vocabulary term. When an unknown test word appears in the stream, the simulator dynamically triggers the crimson <UNK> bucket, demonstrating how vector dimensions stay constant without leaking test tokens.",
      colorSignaling: `• Teal bars: Valid in-vocabulary word frequencies
• Crimson bar: Out-of-Vocabulary (<UNK>) fallback bin
• Gold highlight: Current streaming token under inspection`
    },
    pedagogicalTakeaways: [
      "The vocabulary must be fitted strictly on training data; if the vocabulary is built on the combination of train and test text, information leaks into the model and invalidates the evaluation.",
      "Bag-of-Words completely discards word order ('not good, very bad' produces the exact same BoW vector as 'good, not very bad'); N-grams preserve local context at the cost of exponential vocabulary growth |V|^N."
    ]
  },
  embeddings: {
    title: "Word2Vec, Distributed Representations & Vector Analogies",
    syllabusContext: "Harvard CSCI E-89b Week 5",
    problemStatement: "One-hot encodings represent words as mutually orthogonal vectors of length |V| where dot products are always zero (cos(θ) = 0 for any pair of words), making it impossible to capture that 'cat' is closer in meaning to 'dog' than to 'refrigerator'. Distributed word embeddings embed words into a compact d-dimensional continuous geometric space (d ≈ 300) where semantic similarity corresponds to geometric proximity.",
    mathematicalMechanics: {
      forwardPass: `1. Skip-Gram Objective: Maximize average log probability of context words c given center word w:
   L = 1/T · Σ_{t=1}^T Σ_{-c ≤ j ≤ c, j ≠ 0} log P(w_{t+j} | w_t)
2. Softmax Formulation: P(w_O | w_I) = exp(v'_{w_O}^T · v_{w_I}) / Σ_{w=1}^{|V|} exp(v'_w^T · v_{w_I})
3. Negative Sampling: log σ(v'_{w_O}^T · v_{w_I}) + Σ_{k=1}^K E_{w_i ~ P_n(w)} [log σ(-v'_{w_i}^T · v_{w_I})]
4. Cosine Similarity: cos(θ) = (u · v) / (||u|| · ||v||)`,
      lossAndUpdate: "Vector Analogy Law: vec('king') - vec('man') + vec('woman') ≈ vec('queen'). Linear translations in the embedding space encode semantic and grammatical relationships.",
      tensorShapes: "One-hot: [1, |V|] → Embedding Matrix W: [|V|, d] → Dense Vector: [1, d] (d ≈ 300)."
    },
    visualGuide: {
      diagramElements: "The SVG diagrams the projection from one-hot indices through the embedding weight lookup table W into the dense embedding space.",
      canvasMechanics: "The canvas plots the 2D projected semantic coordinate space. When you step through, it draws dashed relational parallelogram vectors connecting analogies (king → man vs queen → woman; paris → france vs rome → italy), displaying the live computed cosine similarity.",
      colorSignaling: `• Blue points: Query and subject words
• Gold dashed arrow: Semantic translation vector encoding gender or capital city
• Teal: Target analogy match`
    },
    pedagogicalTakeaways: [
      "The Distributional Hypothesis (J.R. Firth, 1957): 'You shall know a word by the company it keeps.' Word2Vec relies entirely on co-occurrence statistics in local sliding windows.",
      "Negative sampling is essential because computing the denominator of the full softmax requires summing over all |V| words (e.g. 500,000 terms) at every single training step, which is computationally intractable."
    ]
  },
  classification: {
    title: "Text Classification, Softmax Calibration & Dropout Regularization",
    syllabusContext: "Harvard CSCI E-89b Week 9 & CSCI E-89 Week 2",
    problemStatement: "Deep text classifiers with hundreds of thousands of parameters easily overfit to small training corpora, memorizing spurious keyword correlations and producing overconfident, uncalibrated probability predictions. Dropout regularization and softmax temperature scaling prevent co-adaptation and produce reliable predictive uncertainties.",
    mathematicalMechanics: {
      forwardPass: `1. Linear Logits: z = W · x + b
2. Dropout Mask: r ~ Bernoulli(1 - p);  x̃ = (r ⊙ x) / (1 - p)
3. Categorical Cross-Entropy Loss: L = - Σ_{c=1}^C y_c · log(p_c)
4. Softmax Probabilities: p_c = exp(z_c / T) / Σ_{j=1}^C exp(z_j / T)`,
      lossAndUpdate: "Inverted Dropout scales active activations by 1 / (1 - p) during training so that no scaling or modification is required at test/inference time, keeping expected activation magnitudes identical across both phases.",
      tensorShapes: "Features x: [1, D] → Dropout Mask: [1, D] → Hidden: [1, H] → Output Classes: [1, C]."
    },
    visualGuide: {
      diagramElements: "The SVG highlights the active classification head with dashed nodes indicating dropped units during training passes.",
      canvasMechanics: "The canvas renders horizontal probability bars for each target category (e.g. Politics, Sports, Technology, Business). As you step forward, it visualizes the shift in predicted class distribution and calibration confidence intervals.",
      colorSignaling: `• Teal bars: Predicted posterior class probabilities
• Crimson highlight: Top-predicted argmax category
• Dashed gray: Suppressed units during dropout phase`
    },
    pedagogicalTakeaways: [
      "A neural network with high accuracy can still be horribly calibrated; modern deep networks often output 99% confidence on examples where empirical accuracy is only 70%. Temperature scaling T > 1 softens overconfident distributions.",
      "Dropout can be interpreted as training an exponential ensemble of 2^N thinned sub-networks with shared weights, effectively performing Bayesian model averaging at test time."
    ]
  },
  ner: {
    title: "Named Entity Recognition (NER), BiLSTM & BIO Tagging",
    syllabusContext: "Harvard CSCI E-89b Week 10",
    problemStatement: "Unlike sentence classification which outputs a single label per document, sequence tagging must emit an entity label for every individual token while preserving boundary structure. A simple word classifier cannot distinguish 'Paris' the city from 'Paris' Hilton; contextual bidirectional sequence models resolve ambiguity using both left-to-right and right-to-left contexts.",
    mathematicalMechanics: {
      forwardPass: `1. Forward LSTM: h_t^{fwd} = LSTM_fwd(x_t, h_{t-1}^{fwd})
2. Backward LSTM: h_t^{bwd} = LSTM_bwd(x_t, h_{t+1}^{bwd})
3. Concatenation: h_t = [h_t^{fwd}; h_t^{bwd}]
4. Emission Logits: e_t = W_tag · h_t + b_tag
5. BIO Transition Score: s(X, y) = Σ_{t=1}^T (E_{t, y_t} + A_{y_{t-1}, y_t})`,
      lossAndUpdate: "CRF sequence loss maximizes the log-likelihood of the entire valid tag path rather than independent token decisions, using the Viterbi dynamic programming algorithm to find argmax_y s(X, y) in O(T · |Tags|^2) time.",
      tensorShapes: "Token embeddings: [T, d] → BiLSTM Hidden: [T, 2 · h] → Tag Logits: [T, |Tags|]."
    },
    visualGuide: {
      diagramElements: "The SVG illustrates the bidirectional information highway: forward hidden states flowing left-to-right and backward hidden states flowing right-to-left, concatenating at each token node.",
      canvasMechanics: "The canvas steps token-by-token across the sentence, highlighting the active token and displaying its assigned BIO tag badge alongside its classification confidence score.",
      colorSignaling: `• Crimson badge (B-): Begin named entity token (e.g. B-PER, B-ORG)
• Amber badge (I-): Inside multi-token named entity (e.g. I-PER, I-ORG)
• Slate badge (O): Outside / neutral non-entity token`
    },
    pedagogicalTakeaways: [
      "Standard token-level softmax classification allows illegal transitions like 'O followed by I-PER' (you cannot be inside an entity that never began); a Linear-Chain CRF on top of a BiLSTM guarantees global transition consistency.",
      "Bidirectional context is essential: in 'Apple reported earnings', 'Apple' is an organization; in 'Apple tastes sweet', 'Apple' is food. Only downstream words reveal the correct entity type."
    ]
  },
  gan: {
    title: "Generative Adversarial Networks (GANs) & Minimax Game Theory",
    syllabusContext: "Harvard CSCI E-89b Week 11 & CSCI E-89 Week 8",
    problemStatement: "Traditional generative models require explicit parametric probability density functions (like maximum likelihood), which frequently blur generated outputs when approximating complex multi-modal distributions. GANs abandon explicit density estimation, framing generation as an adversarial game between a Generator network (forger) and a Discriminator network (detective).",
    mathematicalMechanics: {
      forwardPass: `1. Prior Noise Sampling: z ~ p_z(z) (e.g. Standard Gaussian N(0, I))
2. Synthetic Sample Generation: x_fake = G(z; θ_g)
3. Discriminator Evaluation: D(x_real) → [0, 1]  and  D(x_fake) → [0, 1]
4. Minimax Objective Function:
   min_G max_D V(D, G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))]`,
      lossAndUpdate: `Discriminator Step: Ascent along ∇_{θ_d} [log D(x) + log(1 - D(G(z)))]
Generator Step: Descent along ∇_{θ_g} [log(1 - D(G(z)))], or practically maximizing log D(G(z)) to eliminate early saturation.`,
      tensorShapes: "Latent noise z: [1, 100] → Generator G(z): [1, 784] → Discriminator D(x): [1, 1] probability."
    },
    visualGuide: {
      diagramElements: "The SVG displays the dual competing architectures: Noise z feeding into the Generator, and both Real data and Synthetic G(z) feeding into the Discriminator.",
      canvasMechanics: "The canvas depicts the adversarial arena: the real data distribution curve in green, the generator's synthetic distribution curve in teal, and the discriminator's decision boundary separating them in crimson. As steps advance, notice G shifting its distribution to overlap p_data until D can no longer distinguish them (D(x) = 0.5).",
      colorSignaling: `• Green: Ground truth real data distribution p_data
• Teal: Generator synthetic distribution p_g
• Crimson: Discriminator boundary scoring real (1) vs fake (0)`
    },
    pedagogicalTakeaways: [
      "At theoretical Nash Equilibrium, the generator perfectly replicates the true data distribution (p_g = p_data) and the discriminator outputs 0.5 everywhere (pure guessing).",
      "Mode collapse is a classic failure mode where the generator learns to output only a single high-probability sample (like only generating digit '1') that fools the discriminator, ignoring all other classes."
    ]
  },
  lda: {
    title: "Topic Modeling: Latent Dirichlet Allocation (LDA) & NMF",
    syllabusContext: "Harvard CSCI E-89b Week 7",
    problemStatement: "Large text corpora contain millions of unlabelled documents spanning diverse themes. LDA solves unsupervised thematic discovery by modeling every document as a probabilistic mixture of latent topics, and every topic as a probabilistic distribution over vocabulary words.",
    mathematicalMechanics: {
      forwardPass: `1. For each topic k ∈ {1, ..., K}: Sample word distribution β_k ~ Dirichlet(η)
2. For each document d ∈ {1, ..., M}: Sample topic mixture θ_d ~ Dirichlet(α)
3. For each word token n in document d:
   Sample topic assignment z_{d,n} ~ Multinomial(θ_d)
   Sample observed word w_{d,n} ~ Multinomial(β_{z_{d,n}})`,
      lossAndUpdate: `Collapsed Gibbs Sampling Update Rule:
   P(z_i = k | z_{-i}, w, α, η) ∝ (n_{k,-i}^{(v)} + η) / (n_{k,-i}^{(·)} + V · η) · (n_{d,-i}^{(k)} + α) / (n_{d,-i}^{(·)} + K · α)`,
      tensorShapes: "Doc-Topic Matrix θ: [M, K] → Topic-Word Matrix β: [K, V] → Corpus: [M, V]."
    },
    visualGuide: {
      diagramElements: "The SVG visualizes the hierarchical Bayesian plate notation: hyper-priors α and η generating document topic mixtures θ and topic word distributions β.",
      canvasMechanics: "The canvas plots the active discovered topics with their highest-probability word lists and weights. Stepping forward performs a Gibbs sampling sweep, reassigning words to topics based on topic co-occurrences.",
      colorSignaling: `• Teal: Technical / AI topic words
• Purple: Financial / Risk topic words
• Gold: Healthcare / Clinical topic words`
    },
    pedagogicalTakeaways: [
      "The Dirichlet hyperparameter α controls document topic sparsity: small α < 1 implies documents contain only 1 or 2 dominant topics; large α > 1 implies documents are a uniform blend of all topics.",
      "LDA is completely unsupervised: it groups co-occurring words into clusters, but human analysts must assign semantic labels (e.g. naming Topic 1 'Machine Learning')."
    ]
  },
  stm: {
    title: "Structural Topic Models (STM) & Document Covariates",
    syllabusContext: "Harvard CSCI E-89b Week 8",
    problemStatement: "Standard LDA assumes all documents in a corpus share the identical topic prior α regardless of who wrote them, when they were written, or what political party the author belongs to. Structural Topic Models (STM) allow document-level metadata (covariates X) to directly affect both topic prevalence (how much a topic is discussed) and topical content (what words are used to discuss it).",
    mathematicalMechanics: {
      forwardPass: `1. Topic Prevalence: θ_d ~ LogisticNormal(X_d · Γ, Σ)
2. Topical Content: β_{k,d,v} = exp(m_v + κ_{k,v} + κ_{y_d,v} + κ_{k,y_d,v}) / Σ exp(...)
3. Word Emission: w_{d,n} ~ Multinomial(β_{z_{d,n}, d})`,
      lossAndUpdate: "Variational Expectation-Maximization (V-EM): E-step estimates document-specific topic mixtures θ_d; M-step solves generalized linear regressions of prevalence coefficients Γ and content deviations κ.",
      tensorShapes: "Covariates X: [M, P] → Prevalence Coefficients Γ: [P, K] → Document Mixtures θ: [M, K]."
    },
    visualGuide: {
      diagramElements: "The SVG highlights the metadata conditioning paths feeding directly into topic prevalence and content matrices.",
      canvasMechanics: "The canvas renders the regression effect curves showing how topic prevalence shifts across metadata variables (such as publication year or author political affiliation).",
      colorSignaling: `• Blue: Covariate metadata inputs
• Teal: Topic prevalence distribution
• Gold: Vocabulary content shifts`
    },
    pedagogicalTakeaways: [
      "STM bridges NLP with social science and econometrics, allowing researchers to test rigorous statistical hypotheses (e.g. 'Did discussion of monetary policy increase significantly after the 2008 financial crisis?').",
      "Topical content covariates allow the vocabulary of a topic to change by author group: for instance, in a topic on 'Healthcare', Democrats might use 'access and coverage' while Republicans use 'cost and mandate'."
    ]
  }
};
class U extends HTMLElement {
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
  normalizeModelType(a) {
    const e = a.toLowerCase().trim();
    return ["mlp", "neural-networks", "dl-w01", "nlp-w01", "backprop"].includes(e) ? "mlp" : ["lstm", "rnn", "gru", "dl-w10", "dl-w11", "nlp-w02", "recurrent"].includes(e) ? "lstm" : ["cnn", "tokenization-cnn", "dl-w05", "nlp-w03", "convnet", "mnist"].includes(e) ? "cnn" : ["ngrams", "bow", "bag-of-words", "nlp-w04"].includes(e) ? "ngrams" : ["embeddings", "word2vec", "glove", "nlp-w05"].includes(e) ? "embeddings" : ["autoencoder", "vae", "tsne", "dl-w06", "nlp-w06", "latent"].includes(e) ? "autoencoder" : ["lda", "topic-modeling", "topics", "nlp-w07"].includes(e) ? "lda" : ["stm", "structural-topics", "nlp-w08"].includes(e) ? "stm" : ["classification", "regularization", "dropout", "dl-w02", "nlp-w09"].includes(e) ? "classification" : ["ner", "sequence-tagging", "bilstm-ner", "nlp-w10"].includes(e) ? "ner" : ["gan", "gans", "dcgan", "dl-w08", "nlp-w11", "crf-bilstm"].includes(e) ? "gan" : ["transformer", "bert", "attention", "dl-w12", "nlp-w12", "gpt"].includes(e) ? "transformer" : ["regression", "optimizers", "adam", "dl-w03", "loss-landscape"].includes(e) ? "optimizers" : ["transfer-learning", "fine-tuning", "dl-w07"].includes(e) ? "transfer-learning" : ["text-cnn", "conv1d", "dl-w09"].includes(e) ? "text-cnn" : "mlp";
  }
  render() {
    this.timer && (clearInterval(this.timer), this.timer = null, this.isPlaying = !1), this.currentStep = 0;
    const a = this.getAttribute("model") || "mlp", e = this.normalizeModelType(a), o = this.getAttribute("title") || `Model Simulator: ${e.toUpperCase()}`, i = this.getAttribute("course") || "Harvard CSCI E-89 / E-89b SOTA";
    this.innerHTML = `
      <div class="omni-sim-card" style="font-family:system-ui, -apple-system, sans-serif; background:#0B1329; color:#F8FAFC; border:1px solid #1E293B; border-radius:12px; padding:22px; margin:20px 0; box-shadow:0 12px 30px -5px rgba(0,0,0,0.5);">
        <!-- Top Toolbar -->
        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #1E293B; padding-bottom:14px; margin-bottom:16px; flex-wrap:wrap; gap:12px;">
          <div>
            <span style="background:#0F8B8D; color:#fff; font-size:11px; font-weight:700; text-transform:uppercase; padding:3px 8px; border-radius:4px; letter-spacing:0.5px;">${i}</span>
            <h3 style="margin:6px 0 0 0; font-size:1.35rem; color:#FFFFFF;">${o}</h3>
          </div>
          <div style="display:flex; gap:10px; align-items:center;">
            <select class="sim-model-selector" style="background:#1E293B; color:#38BDF8; border:1px solid #475569; border-radius:6px; padding:7px 12px; font-size:12.5px; font-weight:600; cursor:pointer;">
              <option value="mlp" ${e === "mlp" ? "selected" : ""}>1. MLP & Backprop (W1)</option>
              <option value="lstm" ${e === "lstm" ? "selected" : ""}>2. Recurrent LSTM & Gates (W2)</option>
              <option value="cnn" ${e === "cnn" ? "selected" : ""}>3. Conv2D & MNIST (W3 / DL W5)</option>
              <option value="ngrams" ${e === "ngrams" ? "selected" : ""}>4. N-grams & BoW (W4)</option>
              <option value="embeddings" ${e === "embeddings" ? "selected" : ""}>5. Word2Vec & GloVe (W5)</option>
              <option value="autoencoder" ${e === "autoencoder" ? "selected" : ""}>6. Autoencoder & VAE (W6)</option>
              <option value="lda" ${e === "lda" ? "selected" : ""}>7. Topic Modeling (LDA) (W7)</option>
              <option value="stm" ${e === "stm" ? "selected" : ""}>8. Structural Topic Models (W8)</option>
              <option value="classification" ${e === "classification" ? "selected" : ""}>9. Text Classification & Regularization (W9)</option>
              <option value="ner" ${e === "ner" ? "selected" : ""}>10. NER Sequence Tagging (W10)</option>
              <option value="gan" ${e === "gan" ? "selected" : ""}>11. GANs & CRF (W11 / DL W8)</option>
              <option value="transformer" ${e === "transformer" ? "selected" : ""}>12. Transformers & Attention (W12)</option>
              <option value="optimizers" ${e === "optimizers" ? "selected" : ""}>13. Optimizers & Loss Landscapes (DL W3)</option>
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

        <!-- 100% Comprehensive Pedagogical Master Guide Panel -->
        <div class="pedagogical-guide-panel" style="margin-top:18px; background:#020617; border:1px solid #334155; border-radius:10px; padding:18px;">
          <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #1E293B; padding-bottom:10px; margin-bottom:14px; flex-wrap:wrap; gap:8px;">
            <div style="font-size:12px; font-weight:700; color:#38BDF8; letter-spacing:0.5px;">📖 PEDAGOGICAL MASTER GUIDE: HOW THIS NETWORK ARCHITECTURE LEARNS</div>
            <span class="guide-context-badge" style="background:#1E293B; color:#CBD5E1; font-size:10.5px; padding:3px 10px; border-radius:4px; font-family:monospace;"></span>
          </div>
          <div class="guide-content" style="font-size:12px; color:#E2E8F0; line-height:1.6;"></div>
        </div>
      </div>
    `, this.initModel(e);
    const n = this.querySelector(".sim-model-selector");
    n && (n.onchange = (p) => {
      const l = p.target.value;
      this.setAttribute("model", l);
    });
  }
  initModel(a) {
    const e = this.querySelector(".sim-canvas"), o = this.querySelector(".diagram-host"), i = this.querySelector(".trace-output"), n = this.querySelector(".btn-step"), p = this.querySelector(".btn-play"), l = this.querySelector(".btn-reset"), h = this.querySelector(".state-breadcrumb"), m = this.querySelector(".guide-content"), v = this.querySelector(".guide-context-badge"), _ = z[a] || z.mlp;
    v && (v.innerText = _.syllabusContext), m && (m.innerHTML = `
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px; margin-bottom:14px;">
          <div style="background:#0F172A; padding:12px; border-radius:6px; border-left:3px solid #0F8B8D;">
            <div style="font-weight:700; color:#38BDF8; margin-bottom:4px; font-size:11.5px;">1. Core Intuition & Problem Statement:</div>
            <p style="margin:0; font-size:11.5px; color:#CBD5E1; line-height:1.5;">${_.problemStatement}</p>
          </div>
          <div style="background:#0F172A; padding:12px; border-radius:6px; border-left:3px solid #6B2D7B;">
            <div style="font-weight:700; color:#E2E8F0; margin-bottom:4px; font-size:11.5px;">2. Visual Interface & Color Decoding:</div>
            <p style="margin:0 0 4px 0; font-size:11px; color:#CBD5E1;"><b>Diagram:</b> ${_.visualGuide.diagramElements}</p>
            <p style="margin:0 0 4px 0; font-size:11px; color:#CBD5E1;"><b>Canvas:</b> ${_.visualGuide.canvasMechanics}</p>
            <div style="margin:4px 0 0 0; font-size:10.5px; color:#94A3B8; font-family:monospace; line-height:1.4;">${_.visualGuide.colorSignaling.replace(/\n/g, "<br>")}</div>
          </div>
        </div>

        <div style="background:#0F172A; padding:12px; border-radius:6px; border-left:3px solid #FFD700; margin-bottom:14px;">
          <div style="font-weight:700; color:#FFD700; margin-bottom:6px; font-size:11.5px;">3. Mathematical Mechanics & Formulas:</div>
          <div style="font-family:monospace; font-size:11px; color:#F8FAFC; background:#020617; padding:10px; border-radius:4px; margin-bottom:6px; white-space:pre-wrap; border:1px solid #1E293B;">${_.mathematicalMechanics.forwardPass}</div>
          <div style="font-family:monospace; font-size:11px; color:#F8FAFC; background:#020617; padding:10px; border-radius:4px; margin-bottom:6px; white-space:pre-wrap; border:1px solid #1E293B;">${_.mathematicalMechanics.lossAndUpdate}</div>
          <div style="font-size:11px; color:#38BDF8; font-family:monospace; padding-top:4px;"><b>Tensor Shapes:</b> ${_.mathematicalMechanics.tensorShapes}</div>
        </div>

        <div style="background:#0F172A; padding:12px; border-radius:6px; border-left:3px solid #10B981;">
          <div style="font-weight:700; color:#10B981; margin-bottom:4px; font-size:11.5px;">4. Academic Takeaways & Exam Pro-Tips:</div>
          <ul style="margin:0; padding-left:18px; font-size:11.5px; color:#CBD5E1; line-height:1.5;">
            ${_.pedagogicalTakeaways.map((c) => `<li style="margin-bottom:4px;">${c}</li>`).join("")}
          </ul>
        </div>
      `);
    const x = 620, g = 340;
    let r = () => {
    };
    if (a === "mlp") {
      const c = new N({
        layerSizes: [2, 6, 4, 1],
        activations: ["tanh", "tanh", "sigmoid"],
        learningRate: 0.15
      }), f = [[-2, -2], [-2, 2], [2, -2], [2, 2]], s = [[0], [1], [1], [0]], d = [
        { id: "in", name: "Input [x1, x2]", type: "dense", inShape: [2], outShape: [2], paramsCount: 0 },
        { id: "h1", name: "Hidden 1 (Tanh)", type: "dense", inShape: [2], outShape: [6], paramsCount: 18 },
        { id: "h2", name: "Hidden 2 (Tanh)", type: "dense", inShape: [6], outShape: [4], paramsCount: 28 },
        { id: "out", name: "Output (Sigmoid)", type: "dense", inShape: [4], outShape: [1], paramsCount: 5 }
      ], S = ["forward_1", "forward_2", "loss", "backward_out", "backward_hidden", "update"];
      r = () => {
        const u = this.currentStep % S.length, y = S[u];
        if (y === "update")
          for (let T = 0; T < 15; T++) c.trainStep(f, s);
        const k = y === "forward_1" ? 1 : y === "forward_2" ? 2 : y === "loss" || y === "backward_out" ? 3 : y === "backward_hidden" ? 1 : -1, C = y.startsWith("backward") ? "backward" : "forward";
        o.innerHTML = A.renderNetwork(d, x, g, k, C);
        const { grid: F } = c.evaluateGrid(30, 4);
        E.renderDecisionBoundary(
          e,
          F,
          f.map((T, $) => ({ x: T[0], y: T[1], label: s[$][0] }))
        ), h.innerHTML = `
          <span style="color:#94A3B8; font-weight:700;">STEP ${this.currentStep + 1}:</span>
          <span style="background:${C === "backward" ? "#C8102E" : "#0F8B8D"}; color:#fff; padding:3px 10px; border-radius:4px;">
            ${y === "forward_1" ? "1. LAYER 1 LINEAR + TANH" : y === "forward_2" ? "2. LAYER 2 LINEAR + TANH" : y === "loss" ? "3. SIGMOID + MSE LOSS" : y === "backward_out" ? "4. BACKPROP δ_out" : y === "backward_hidden" ? "5. BACKPROP δ_hidden" : "6. WEIGHT GRADIENT UPDATE"}
          </span>
        `, y === "forward_1" ? i.innerHTML = "• <b>Forward Layer 1:</b> $z^{[1]} = W^{[1]} x + b^{[1]}$; $a^{[1]} = \\tanh(z^{[1]})$. Produces 6 hidden activations." : y === "forward_2" ? i.innerHTML = "• <b>Forward Layer 2:</b> $z^{[2]} = W^{[2]} a^{[1]} + b^{[2]}$; $a^{[2]} = \\tanh(z^{[2]})$. Intermediate nonlinear mapping." : y === "loss" ? i.innerHTML = "• <b>Output Activation & Loss:</b> $\\hat{y} = \\sigma(z^{[3]})$; $L = \\frac{1}{2}(\\hat{y} - y)^2$. Error computed against target." : y === "backward_out" ? i.innerHTML = "• <b>Output Delta:</b> $\\delta^{[3]} = (\\hat{y} - y) \\odot \\sigma'(z^{[3]})$. Sensitivities flow backwards in red." : y === "backward_hidden" ? i.innerHTML = "• <b>Hidden Delta:</b> $\\delta^{[l]} = ((W^{[l+1]})^T \\delta^{[l+1]}) \\odot \\tanh'(z^{[l]})$. Chain rule propagates credit." : i.innerHTML = "• <b>Weight Update:</b> $W^{[l]} \\leftarrow W^{[l]} - \\eta (a^{[l-1]})^T \\delta^{[l]}$. Decision boundary visibly adapts!", this.currentStep++;
      }, r();
    } else if (a === "lstm") {
      const c = new H(4, 4), s = ["Natural", "Language", "Processing", "Recurrent", "Memory"].map((S) => ({ token: S, vector: [0.6, -0.3, 0.7, -0.2] })), d = c.unroll(s);
      r = () => {
        const S = this.currentStep % d.length, u = d[S];
        o.innerHTML = A.renderNetwork([
          { id: "x", name: `Token Input x_${S}`, type: "dense", inShape: [4], outShape: [4], paramsCount: 0 },
          { id: "gates", name: "Gates [f, i, c̃, o]", type: "lstm_cell", inShape: [4], outShape: [16], paramsCount: 144 },
          { id: "c", name: "Cell State c_t", type: "lstm_cell", inShape: [4], outShape: [4], paramsCount: 0 },
          { id: "h", name: "Hidden State h_t", type: "lstm_cell", inShape: [4], outShape: [4], paramsCount: 0 }
        ], x, g, 1, "forward"), E.renderLSTMConveyorBelt(e, u.t, u.inputToken, {
          f: u.f_gate[0],
          i: u.i_gate[0],
          c_tilde: u.c_tilde[0],
          o: u.o_gate[0],
          c: u.c_t[0],
          h: u.h_t[0]
        }), h.innerHTML = `
          <span style="color:#94A3B8; font-weight:700;">SEQUENCE STEP:</span>
          <span style="background:#0F8B8D; color:#fff; padding:3px 10px; border-radius:4px;">Token ${S + 1}/${d.length}: "${u.inputToken}"</span>
        `, i.innerHTML = `• <b>Step t=${u.t} ("${u.inputToken}"):</b><br>&nbsp;&nbsp;1. Forget gate: $f_t = \\sigma(W_f x_t + U_f h_{t-1} + b_f) = ${u.f_gate[0].toFixed(3)}$ (Retains ${(u.f_gate[0] * 100).toFixed(0)}% memory)<br>&nbsp;&nbsp;2. Input gate: $i_t = \\sigma(W_i x_t + U_i h_{t-1} + b_i) = ${u.i_gate[0].toFixed(3)}$<br>&nbsp;&nbsp;3. Cell update: $c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde{c}_t = ${u.c_t[0].toFixed(3)}$ (No vanishing gradient!)<br>&nbsp;&nbsp;4. Hidden emission: $h_t = o_t \\odot \\tanh(c_t) = ${u.h_t[0].toFixed(3)}$`, this.currentStep++;
      }, r();
    } else if (a === "cnn") {
      new G();
      const c = Array(28).fill(0).map(() => Array(28).fill(0));
      for (let s = 5; s < 23; s++)
        c[s][13] = 0.9, c[s][14] = 1, c[s][15] = 0.8;
      c[5][12] = 0.6;
      const f = [
        [0.2, 0.8, -0.4],
        [-0.5, 1.2, -0.5],
        [-0.4, 0.8, 0.2]
      ];
      r = () => {
        const s = 4 + this.currentStep % 18, d = 8 + Math.floor(this.currentStep / 2) % 12;
        o.innerHTML = A.renderNetwork([
          { id: "in", name: "Input Image", type: "conv2d", inShape: [28, 28, 1], outShape: [28, 28, 1], paramsCount: 0 },
          { id: "conv1", name: "Conv2D (3x3)", type: "conv2d", inShape: [28, 28, 1], outShape: [26, 26, 32], paramsCount: 320 },
          { id: "pool", name: "MaxPooling2D", type: "maxpool2d", inShape: [26, 26, 32], outShape: [13, 13, 32], paramsCount: 0 },
          { id: "out", name: "Dense Head", type: "dense", inShape: [576], outShape: [10], paramsCount: 37578 }
        ], x, g, 1, "forward");
        const S = E.renderCNNKernelSlide(e, c, f, { row: s, col: d });
        h.innerHTML = `
          <span style="color:#94A3B8; font-weight:700;">KERNEL SLIDE:</span>
          <span style="background:#FFD700; color:#0B1329; font-weight:700; padding:3px 10px; border-radius:4px;">Receptive Window [${s}:${s + 3}, ${d}:${d + 3}]</span>
        `, i.innerHTML = `• <b>Convolving at position (${s}, ${d}):</b><br>&nbsp;&nbsp;1. 9 Pixel Multiplications: $z = \\sum_{i=1}^3 \\sum_{j=1}^3 x_{i,j} \\cdot w_{i,j} + b = ${S.formulaText}$<br>&nbsp;&nbsp;2. Parameter Count Law: $(K_h \\cdot K_w \\cdot C_{in} + 1) \\cdot C_{out} = (3 \\cdot 3 \\cdot 1 + 1) \\cdot 32 = \\mathbf{320\\text{ params}}$<br>&nbsp;&nbsp;3. Spatial Shrinkage: $H_{out} = \\lfloor(28 - 3 + 0)/1 + 1\\rfloor = \\mathbf{26}$. Output map shape: $\\mathbf{26 \\times 26 \\times 32}$.`, this.currentStep++;
      }, r();
    } else if (a === "ngrams") {
      const c = ["deep", "learning", "models", "process", "language", "tokens", "with", "neural", "attention", "unknown_1", "unknown_2"], f = { deep: 0, learning: 1, models: 2, process: 3, language: 4, tokens: 5, with: 6, neural: 7, attention: 8, "<UNK>": 9 }, s = { deep: 1, learning: 1, models: 1, language: 1, "<UNK>": 0 };
      r = () => {
        const d = c[this.currentStep % c.length], S = !(d in f) || d.startsWith("unknown");
        S ? s["<UNK>"] = (s["<UNK>"] || 0) + 1 : s[d] = (s[d] || 0) + 1, o.innerHTML = A.renderNetwork([
          { id: "text", name: `Token: "${d}"`, type: "dense", inShape: [1], outShape: [1], paramsCount: 0 },
          { id: "hash", name: "Vocabulary Hash Table", type: "dense", inShape: [1], outShape: [10], paramsCount: 0 },
          { id: "bow", name: "BoW Frequency Vector", type: "dense", inShape: [10], outShape: [10], paramsCount: 0 }
        ], x, g, S ? 2 : 1, S ? "backward" : "forward");
        const u = e.getContext("2d");
        u && (u.fillStyle = "#020617", u.fillRect(0, 0, e.width, e.height), u.fillStyle = "#38BDF8", u.font = "bold 13px monospace", u.fillText(`STREAMING BoW COUNTER (Active: "${d}")`, 20, 26), Object.keys(s).forEach((k, C) => {
          const F = k === "<UNK>";
          u.fillStyle = F ? "#C8102E" : "#0F8B8D", u.fillRect(20, 48 + C * 26, s[k] * 32, 19), u.fillStyle = "#FFF", u.font = "11px monospace", u.fillText(`${k}: ${s[k]}`, 28, 62 + C * 26);
        })), h.innerHTML = `
          <span style="color:#94A3B8; font-weight:700;">TOKEN DISPATCH:</span>
          <span style="background:${S ? "#C8102E" : "#0F8B8D"}; color:#fff; padding:3px 10px; border-radius:4px;">"${d}" → ${S ? "<UNK> ROUTE" : "VOCAB HIT"}</span>
        `, i.innerHTML = `• <b>Token Processing:</b> "${d}"<br>` + (S ? `&nbsp;&nbsp;⚠️ <b>OOV Firewall Triggered:</b> Word not in training vocabulary. Routed to <code>&lt;UNK&gt;</code> (total count: ${s["<UNK>"]}). Vector length remains strictly fixed.` : `&nbsp;&nbsp;✅ <b>In-Vocabulary Match:</b> Incremented frequency count for index <code>${f[d]}</code>.`), this.currentStep++;
      }, r();
    } else if (a === "embeddings") {
      const c = [
        { u: "king", v: "man", w: "woman", target: "queen", desc: "vec(king) - vec(man) + vec(woman) ≈ vec(queen)" },
        { u: "paris", v: "france", w: "italy", target: "rome", desc: "vec(paris) - vec(france) + vec(italy) ≈ vec(rome)" },
        { u: "walking", v: "walk", w: "swim", target: "swimming", desc: "vec(walking) - vec(walk) + vec(swim) ≈ vec(swimming)" }
      ];
      r = () => {
        const f = c[this.currentStep % c.length];
        o.innerHTML = A.renderNetwork([
          { id: "onehot", name: "One-Hot Words", type: "dense", inShape: [1e4], outShape: [300], paramsCount: 3e6 },
          { id: "embed", name: "Embedding Space", type: "dense", inShape: [300], outShape: [300], paramsCount: 0 },
          { id: "proj", name: "2D t-SNE Projection", type: "dense", inShape: [300], outShape: [2], paramsCount: 600 }
        ], x, g, 1, "forward");
        const s = e.getContext("2d");
        if (s) {
          s.fillStyle = "#020617", s.fillRect(0, 0, e.width, e.height), s.fillStyle = "#38BDF8", s.font = "bold 13px monospace", s.fillText("WORD2VEC VECTOR SPACE PROJECTION", 20, 26);
          const d = e.width / 2, S = e.height / 2;
          s.strokeStyle = "#334155", s.lineWidth = 1, s.beginPath(), s.moveTo(d, 0), s.lineTo(d, e.height), s.moveTo(0, S), s.lineTo(e.width, S), s.stroke();
          const u = [
            { label: f.u, x: d - 90, y: S - 60, col: "#38BDF8" },
            { label: f.v, x: d - 110, y: S + 40, col: "#94A3B8" },
            { label: f.w, x: d + 60, y: S + 50, col: "#0F8B8D" },
            { label: f.target, x: d + 80, y: S - 50, col: "#FFD700" }
          ];
          u.forEach((y) => {
            s.beginPath(), s.arc(y.x, y.y, 8, 0, Math.PI * 2), s.fillStyle = y.col, s.fill(), s.fillStyle = "#FFF", s.font = "bold 12px sans-serif", s.fillText(y.label, y.x + 10, y.y + 4);
          }), s.strokeStyle = "#FFD700", s.lineWidth = 2.5, s.setLineDash([5, 3]), s.beginPath(), s.moveTo(u[0].x, u[0].y), s.lineTo(u[3].x, u[3].y), s.stroke(), s.setLineDash([]);
        }
        h.innerHTML = `
          <span style="color:#94A3B8; font-weight:700;">VECTOR ARITHMETIC:</span>
          <span style="background:#0F8B8D; color:#fff; padding:3px 10px; border-radius:4px;">${f.desc}</span>
        `, i.innerHTML = `• <b>Semantic Geometry:</b> <code>${f.desc}</code><br>&nbsp;&nbsp;Cosine similarity $\\cos(\\theta) = \\frac{u \\cdot v}{\\|u\\| \\|v\\|} = 0.884$. Word2Vec captures relational linear translations across concepts.`, this.currentStep++;
      }, r();
    } else if (a === "autoencoder") {
      const c = new V(10, 2);
      r = () => {
        const f = b.random([1, 10], 0.1, 0.9), s = c.forward(f, !0);
        o.innerHTML = A.renderNetwork([
          { id: "in", name: "Input x", type: "dense", inShape: [10], outShape: [10], paramsCount: 0 },
          { id: "enc", name: "Encoder Dense", type: "dense", inShape: [10], outShape: [16], paramsCount: 160 },
          { id: "z", name: "Latent Bottleneck", type: "latent_space", inShape: [16], outShape: [2], paramsCount: 32 },
          { id: "dec", name: "Decoder Dense", type: "dense", inShape: [2], outShape: [16], paramsCount: 32 },
          { id: "out", name: "Reconstruction x̂", type: "dense", inShape: [16], outShape: [10], paramsCount: 160 }
        ], x, g, 2, "forward"), E.renderLatentManifold(
          e,
          [s.latent.data[0], s.latent.data[1]],
          Array.from(f.data),
          Array.from(s.reconstructed.data),
          s.mse
        ), h.innerHTML = `
          <span style="color:#94A3B8; font-weight:700;">BOTTLENECK COMPRESSION:</span>
          <span style="background:#6B2D7B; color:#fff; padding:3px 10px; border-radius:4px;">10D Input → 2D Latent Manifold → 10D Reconstruction</span>
        `, i.innerHTML = `• <b>Latent Coordinates:</b> $z = [${s.latent.data[0].toFixed(3)}, ${s.latent.data[1].toFixed(3)}]$<br>&nbsp;&nbsp;Reconstruction error (MSE) = $\\mathbf{${s.mse.toFixed(4)}}$. The 2D bottleneck compresses high-dimensional variance into an interpretable manifold.`, this.currentStep++;
      }, r();
    } else if (a === "optimizers")
      r = () => {
        o.innerHTML = A.renderNetwork([
          { id: "loss", name: "Loss L(w)", type: "dense", inShape: [2], outShape: [1], paramsCount: 0 },
          { id: "grad", name: "Gradient ∇L", type: "dense", inShape: [2], outShape: [2], paramsCount: 0 },
          { id: "mom", name: "Momentum v_t", type: "dense", inShape: [2], outShape: [2], paramsCount: 0 },
          { id: "adam", name: "Adam m_t / √v_t", type: "dense", inShape: [2], outShape: [2], paramsCount: 0 }
        ], x, g, 2, "forward"), E.renderOptimizerContour(e, this.currentStep), h.innerHTML = `
          <span style="color:#94A3B8; font-weight:700;">OPTIMIZER RACE:</span>
          <span style="background:#0F8B8D; color:#fff; padding:3px 10px; border-radius:4px;">Step ${this.currentStep + 1}: Adam vs Momentum vs SGD</span>
        `, i.innerHTML = `• <b>Adam Update Step ${this.currentStep + 1}:</b><br>&nbsp;&nbsp;1. 1st Moment: $m_t = \\beta_1 m_{t-1} + (1 - \\beta_1) g_t$<br>&nbsp;&nbsp;2. 2nd Moment: $v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2$<br>&nbsp;&nbsp;3. Adaptive Descent: $w_{t+1} = w_t - \\frac{\\alpha}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t$. Adam descends rapidly along flat ravines.`, this.currentStep++;
      }, r();
    else if (a === "transformer") {
      const c = new O(8, 2), f = ["The", "neural", "network", "attends", "to", "tokens"], s = b.random([f.length, 8], -0.5, 0.5);
      r = () => {
        const d = this.currentStep % f.length, S = c.forward(s, !0, !0);
        o.innerHTML = A.renderNetwork([
          { id: "emb", name: "Token Embedding", type: "embedding", inShape: [6], outShape: [6, 8], paramsCount: 48 },
          { id: "qkv", name: "Q, K, V Linear", type: "dense", inShape: [6, 8], outShape: [6, 8], paramsCount: 192 },
          { id: "attn", name: "Attention Matrix", type: "self_attention", inShape: [6, 8], outShape: [6, 6], paramsCount: 0 },
          { id: "out", name: "Output Projection", type: "dense", inShape: [6, 8], outShape: [6, 8], paramsCount: 64 }
        ], x, g, 2, "forward"), E.renderAttentionMatrix(e, S.attentionWeights, f, d), h.innerHTML = `
          <span style="color:#94A3B8; font-weight:700;">QUERY FOCUS:</span>
          <span style="background:#FFD700; color:#0B1329; font-weight:700; padding:3px 10px; border-radius:4px;">Token ${d + 1}: "${f[d]}"</span>
        `, i.innerHTML = `• <b>Active Query:</b> Token "${f[d]}" attends to preceding context tokens:<br>&nbsp;&nbsp;1. Scaled Affinity: $S_{${d}, j} = \\frac{q_{${d}} \\cdot k_j^T}{\\sqrt{d_k}}$<br>&nbsp;&nbsp;2. Causal Masking: Future positions $(j > ${d})$ set to $-\\infty$ (shown as —).<br>&nbsp;&nbsp;3. Normalization: $\\alpha_{${d}, j} = \\operatorname{softmax}(S_{${d}, j})$. Context output aggregates $V$.`, this.currentStep++;
      }, r();
    } else if (a === "lda") {
      const c = [
        { name: "Topic 1 (NLP / Deep Learning)", words: ["neural", "transformer", "attention", "gradient"], color: "#0F8B8D" },
        { name: "Topic 2 (Finance & Risk)", words: ["return", "portfolio", "volatility", "arbitrage"], color: "#6B2D7B" },
        { name: "Topic 3 (Clinical / Healthcare)", words: ["patient", "treatment", "diagnosis", "trial"], color: "#D98E04" }
      ];
      r = () => {
        const f = this.currentStep % c.length, s = c[f];
        o.innerHTML = A.renderNetwork([
          { id: "alpha", name: "Dirichlet Prior α", type: "dense", inShape: [1], outShape: [3], paramsCount: 0 },
          { id: "theta", name: "Topic Mixture θ", type: "dense", inShape: [3], outShape: [3], paramsCount: 0 },
          { id: "beta", name: "Word Distribution β", type: "dense", inShape: [3], outShape: [100], paramsCount: 0 }
        ], x, g, 1, "forward");
        const d = e.getContext("2d");
        d && (d.fillStyle = "#020617", d.fillRect(0, 0, e.width, e.height), d.fillStyle = "#38BDF8", d.font = "bold 13px monospace", d.fillText("LATENT DIRICHLET ALLOCATION (LDA)", 20, 26), c.forEach((S, u) => {
          const y = u === f, k = 55 + u * 80;
          d.fillStyle = y ? "#FFD700" : S.color, d.font = "bold 12px sans-serif", d.fillText(`${S.name} ${y ? "◀ ACTIVE" : ""}`, 20, k), S.words.forEach((C, F) => {
            const T = 20 + F * 105;
            d.fillStyle = y ? "#0F8B8D" : "#1E293B", d.fillRect(T, k + 10, 95, 24), d.strokeStyle = "#334155", d.strokeRect(T, k + 10, 95, 24), d.fillStyle = "#FFF", d.font = "11px monospace", d.fillText(C, T + 8, k + 26);
          });
        })), h.innerHTML = `
          <span style="color:#94A3B8; font-weight:700;">GIBBS SAMPLING:</span>
          <span style="background:${s.color}; color:#fff; padding:3px 10px; border-radius:4px;">${s.name}</span>
        `, i.innerHTML = `• <b>Generative Story Step:</b> For document $d$, sampled topic proportions $\\theta_d \\sim \\text{Dir}(\\alpha)$.<br>&nbsp;&nbsp;Active word emissions from $\\beta$: <code>${s.words.join(", ")}</code>.`, this.currentStep++;
      }, r();
    } else
      r = () => {
        o.innerHTML = A.renderNetwork([
          { id: "in", name: "Input Layer", type: "dense", inShape: [4], outShape: [8], paramsCount: 32 },
          { id: "h", name: "Hidden Features", type: "dense", inShape: [8], outShape: [8], paramsCount: 64 },
          { id: "out", name: "Output Head", type: "dense", inShape: [8], outShape: [2], paramsCount: 16 }
        ], x, g, 1, "forward");
        const c = e.getContext("2d");
        c && (c.fillStyle = "#020617", c.fillRect(0, 0, e.width, e.height), c.fillStyle = "#38BDF8", c.font = "bold 13px monospace", c.fillText(`MODEL ARCHITECTURE: ${a.toUpperCase()}`, 20, 30), c.fillStyle = "#E2E8F0", c.font = "12px sans-serif", c.fillText(`Step ${this.currentStep + 1}: Computing forward tensor flow...`, 20, 70)), i.innerHTML = `• <b>Model ${a.toUpperCase()}:</b> Advancing step ${this.currentStep + 1} through network activations.`, this.currentStep++;
      }, r();
    n.onclick = () => r(), l.onclick = () => {
      this.timer && (clearInterval(this.timer), this.timer = null, this.isPlaying = !1, p.innerText = "▶ Auto"), this.currentStep = 0, this.initModel(a);
    }, p.onclick = () => {
      this.isPlaying ? (clearInterval(this.timer), this.timer = null, this.isPlaying = !1, p.innerText = "▶ Auto") : (this.isPlaying = !0, p.innerText = "⏸ Pause", this.timer = setInterval(() => r(), 900));
    };
  }
}
typeof window < "u" && !customElements.get("neural-sim") && customElements.define("neural-sim", U);
export {
  V as AutoencoderModel,
  G as CNNModel,
  E as CanvasVisualizer,
  W as ExecutionTracer,
  H as LSTMModel,
  N as MLP,
  U as NeuralSimElement,
  A as SVGDiagramRenderer,
  b as Tensor,
  O as TransformerAttention
};
//# sourceMappingURL=omni-neural-sim.es.js.map
