/**
 * OmniNeuralSim Core Tensor Engine
 * Lightweight, zero-dependency multidimensional tensor engine in TypeScript.
 */

export type Shape = number[];

export class Tensor {
  data: Float32Array;
  shape: Shape;
  strides: number[];
  size: number;

  constructor(shape: Shape, data?: Float32Array | number[]) {
    this.shape = [...shape];
    this.size = shape.reduce((a, b) => a * b, 1);
    this.strides = Tensor.computeStrides(this.shape);
    if (data) {
      this.data = data instanceof Float32Array ? data : new Float32Array(data);
    } else {
      this.data = new Float32Array(this.size);
    }
  }

  static computeStrides(shape: Shape): number[] {
    const strides = new Array(shape.length);
    let stride = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
      strides[i] = stride;
      stride *= shape[i];
    }
    return strides;
  }

  static zeros(shape: Shape): Tensor {
    return new Tensor(shape);
  }

  static ones(shape: Shape): Tensor {
    const t = new Tensor(shape);
    t.data.fill(1.0);
    return t;
  }

  static random(shape: Shape, min = -1.0, max = 1.0): Tensor {
    const t = new Tensor(shape);
    const range = max - min;
    for (let i = 0; i < t.size; i++) {
      t.data[i] = min + Math.random() * range;
    }
    return t;
  }

  static fromArray(arr: any): Tensor {
    const shape: number[] = [];
    let curr = arr;
    while (Array.isArray(curr)) {
      shape.push(curr.length);
      curr = curr[0];
    }
    const flat: number[] = [];
    function flatten(a: any) {
      if (Array.isArray(a)) {
        for (const item of a) flatten(item);
      } else {
        flat.push(Number(a));
      }
    }
    flatten(arr);
    return new Tensor(shape, flat);
  }

  clone(): Tensor {
    return new Tensor(this.shape, new Float32Array(this.data));
  }

  getIndex(...indices: number[]): number {
    let idx = 0;
    for (let i = 0; i < indices.length; i++) {
      idx += indices[i] * this.strides[i];
    }
    return idx;
  }

  get(...indices: number[]): number {
    return this.data[this.getIndex(...indices)];
  }

  set(val: number, ...indices: number[]): void {
    this.data[this.getIndex(...indices)] = val;
  }

  // --- Element-wise arithmetic ---

  add(other: Tensor | number): Tensor {
    const out = new Tensor(this.shape);
    if (typeof other === "number") {
      for (let i = 0; i < this.size; i++) out.data[i] = this.data[i] + other;
    } else {
      for (let i = 0; i < this.size; i++) out.data[i] = this.data[i] + other.data[i];
    }
    return out;
  }

  sub(other: Tensor | number): Tensor {
    const out = new Tensor(this.shape);
    if (typeof other === "number") {
      for (let i = 0; i < this.size; i++) out.data[i] = this.data[i] - other;
    } else {
      for (let i = 0; i < this.size; i++) out.data[i] = this.data[i] - other.data[i];
    }
    return out;
  }

  mul(other: Tensor | number): Tensor {
    const out = new Tensor(this.shape);
    if (typeof other === "number") {
      for (let i = 0; i < this.size; i++) out.data[i] = this.data[i] * other;
    } else {
      for (let i = 0; i < this.size; i++) out.data[i] = this.data[i] * other.data[i];
    }
    return out;
  }

  // --- Matrix Multiplication (2D) ---
  matmul(other: Tensor): Tensor {
    if (this.shape.length !== 2 || other.shape.length !== 2) {
      throw new Error(`matmul requires 2D tensors, got ${this.shape} and ${other.shape}`);
    }
    const [M, K1] = this.shape;
    const [K2, N] = other.shape;
    if (K1 !== K2) {
      throw new Error(`Incompatible matrix dims: [${M}, ${K1}] x [${K2}, ${N}]`);
    }
    const out = new Tensor([M, N]);
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let sum = 0;
        for (let k = 0; k < K1; k++) {
          sum += this.get(i, k) * other.get(k, j);
        }
        out.set(sum, i, j);
      }
    }
    return out;
  }

  transpose(): Tensor {
    if (this.shape.length !== 2) {
      throw new Error("transpose currently supports 2D tensors");
    }
    const [rows, cols] = this.shape;
    const out = new Tensor([cols, rows]);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        out.set(this.get(r, c), c, r);
      }
    }
    return out;
  }

  // --- Activations ---
  relu(): Tensor {
    const out = new Tensor(this.shape);
    for (let i = 0; i < this.size; i++) {
      out.data[i] = Math.max(0, this.data[i]);
    }
    return out;
  }

  sigmoid(): Tensor {
    const out = new Tensor(this.shape);
    for (let i = 0; i < this.size; i++) {
      out.data[i] = 1 / (1 + Math.exp(-this.data[i]));
    }
    return out;
  }

  tanh(): Tensor {
    const out = new Tensor(this.shape);
    for (let i = 0; i < this.size; i++) {
      out.data[i] = Math.tanh(this.data[i]);
    }
    return out;
  }

  gelu(): Tensor {
    const out = new Tensor(this.shape);
    const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);
    for (let i = 0; i < this.size; i++) {
      const x = this.data[i];
      out.data[i] = 0.5 * x * (1 + Math.tanh(SQRT_2_OVER_PI * (x + 0.044715 * Math.pow(x, 3))));
    }
    return out;
  }

  softmax(axis = -1): Tensor {
    const out = new Tensor(this.shape);
    if (this.shape.length === 1) {
      let maxVal = -Infinity;
      for (let i = 0; i < this.size; i++) if (this.data[i] > maxVal) maxVal = this.data[i];
      let sumExp = 0;
      for (let i = 0; i < this.size; i++) {
        out.data[i] = Math.exp(this.data[i] - maxVal);
        sumExp += out.data[i];
      }
      for (let i = 0; i < this.size; i++) out.data[i] /= sumExp;
      return out;
    }
    if (this.shape.length === 2) {
      const [rows, cols] = this.shape;
      for (let r = 0; r < rows; r++) {
        let maxVal = -Infinity;
        for (let c = 0; c < cols; c++) {
          const val = this.get(r, c);
          if (val > maxVal) maxVal = val;
        }
        let sumExp = 0;
        for (let c = 0; c < cols; c++) {
          const expVal = Math.exp(this.get(r, c) - maxVal);
          out.set(expVal, r, c);
          sumExp += expVal;
        }
        for (let c = 0; c < cols; c++) {
          out.set(out.get(r, c) / sumExp, r, c);
        }
      }
      return out;
    }
    throw new Error("softmax supports 1D or 2D tensors");
  }

  // --- 2D Convolution & Pooling ---
  /**
   * 2D Convolution on [H, W, InChannels] with Kernel [Kh, Kw, InChannels, OutChannels]
   */
  conv2d(kernel: Tensor, bias?: Tensor, stride = 1, padding = 0): Tensor {
    const [H, W, inC] = this.shape;
    const [Kh, Kw, kInC, outC] = kernel.shape;
    if (inC !== kInC) {
      throw new Error(`Channel mismatch: input has ${inC}, kernel expects ${kInC}`);
    }
    const outH = Math.floor((H - Kh + 2 * padding) / stride) + 1;
    const outW = Math.floor((W - Kw + 2 * padding) / stride) + 1;
    const out = new Tensor([outH, outW, outC]);

    for (let oc = 0; oc < outC; oc++) {
      const b = bias ? bias.data[oc] : 0;
      for (let oh = 0; oh < outH; oh++) {
        for (let ow = 0; ow < outW; ow++) {
          let sum = b;
          const ihStart = oh * stride - padding;
          const iwStart = ow * stride - padding;
          for (let kh = 0; kh < Kh; kh++) {
            const ih = ihStart + kh;
            if (ih < 0 || ih >= H) continue;
            for (let kw = 0; kw < Kw; kw++) {
              const iw = iwStart + kw;
              if (iw < 0 || iw >= W) continue;
              for (let ic = 0; ic < inC; ic++) {
                sum += this.get(ih, iw, ic) * kernel.get(kh, kw, ic, oc);
              }
            }
          }
          out.set(sum, oh, ow, oc);
        }
      }
    }
    return out;
  }

  maxPool2d(poolSize = 2, stride = 2): Tensor {
    const [H, W, C] = this.shape;
    const outH = Math.floor((H - poolSize) / stride) + 1;
    const outW = Math.floor((W - poolSize) / stride) + 1;
    const out = new Tensor([outH, outW, C]);

    for (let c = 0; c < C; c++) {
      for (let oh = 0; oh < outH; oh++) {
        for (let ow = 0; ow < outW; ow++) {
          let maxVal = -Infinity;
          const ihStart = oh * stride;
          const iwStart = ow * stride;
          for (let ph = 0; ph < poolSize; ph++) {
            for (let pw = 0; pw < poolSize; pw++) {
              const val = this.get(ihStart + ph, iwStart + pw, c);
              if (val > maxVal) maxVal = val;
            }
          }
          out.set(maxVal, oh, ow, c);
        }
      }
    }
    return out;
  }

  flatten(): Tensor {
    return new Tensor([this.size], this.data);
  }

  toArray(): any {
    if (this.shape.length === 1) {
      return Array.from(this.data);
    }
    if (this.shape.length === 2) {
      const [rows, cols] = this.shape;
      const res = [];
      for (let r = 0; r < rows; r++) {
        const row = [];
        for (let c = 0; c < cols; c++) row.push(this.get(r, c));
        res.push(row);
      }
      return res;
    }
    return Array.from(this.data);
  }
}
