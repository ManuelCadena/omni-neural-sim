/**
 * OmniNeuralSim - <neural-sim> Web Component
 * Standalone, zero-framework Custom Element embeddable into any HTML document/assignment.
 */

import { MLP } from "../models/mlp.ts";
import { CNNModel } from "../models/cnn.ts";
import { LSTMModel } from "../models/rnn.ts";
import { TransformerAttention } from "../models/transformer.ts";
import { SVGDiagramRenderer } from "../renderers/svg-diagram.ts";
import { CanvasVisualizer } from "../renderers/interactive-canvas.ts";
import { Tensor } from "../core/tensor.ts";

export class NeuralSimElement extends HTMLElement {
  private container: HTMLDivElement | null = null;

  static get observedAttributes() {
    return ["model", "dataset", "weights", "tokens", "sequence"];
  }

  connectedCallback() {
    this.render();
  }

  attributeChangedCallback() {
    this.render();
  }

  private render() {
    const modelType = this.getAttribute("model") || "mlp";

    this.innerHTML = `
      <div class="omni-sim-card" style="font-family:system-ui, -apple-system, sans-serif; background:#0F172A; color:#F8FAFC; border:1px solid #334155; border-radius:12px; padding:20px; margin:16px 0; box-shadow:0 10px 25px -5px rgba(0,0,0,0.3);">
        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #1E293B; padding-bottom:12px; margin-bottom:16px;">
          <div>
            <span style="background:#0F8B8D; color:#fff; font-size:11px; font-weight:700; text-transform:uppercase; padding:3px 8px; border-radius:4px; letter-spacing:0.5px;">OmniNeuralSim SOTA</span>
            <h3 style="margin:6px 0 0 0; font-size:1.25rem; color:#FFFFFF;">Model Simulator: <span style="color:#38BDF8;">${modelType.toUpperCase()}</span></h3>
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
    `;

    this.initModel(modelType);
  }

  private initModel(type: string) {
    const canvas = this.querySelector(".sim-canvas") as HTMLCanvasElement;
    const diagramHost = this.querySelector(".diagram-host") as HTMLDivElement;
    const traceOutput = this.querySelector(".trace-output") as HTMLDivElement;
    const metricsHost = this.querySelector(".sim-metrics") as HTMLDivElement;
    const stepBtn = this.querySelector(".btn-step") as HTMLButtonElement;

    if (type === "mlp") {
      const mlp = new MLP({
        layerSizes: [2, 6, 4, 1],
        activations: ["tanh", "tanh", "sigmoid"],
        learningRate: 0.1
      });

      // XOR dataset
      const X = [[-2, -2], [-2, 2], [2, -2], [2, 2]];
      const Y = [[0], [1], [1], [0]];

      const updateUI = () => {
        const { grid } = mlp.evaluateGrid(30, 4);
        CanvasVisualizer.renderDecisionBoundary(
          canvas,
          grid,
          X.map((pt, i) => ({ x: pt[0], y: pt[1], label: Y[i][0] }))
        );
        diagramHost.innerHTML = SVGDiagramRenderer.renderNetwork([
          { id: "in", name: "Input [x1, x2]", type: "dense", inShape: [2], outShape: [2], paramsCount: 0 },
          { id: "h1", name: "Hidden 1 (Tanh)", type: "dense", inShape: [2], outShape: [6], paramsCount: 18 },
          { id: "h2", name: "Hidden 2 (Tanh)", type: "dense", inShape: [6], outShape: [4], paramsCount: 28 },
          { id: "out", name: "Output (Sigmoid)", type: "dense", inShape: [4], outShape: [1], paramsCount: 5 }
        ], 380, 220);

        mlp.forward(new Tensor([1, 2], [1.5, -1.5]), true);
        traceOutput.innerHTML = mlp.tracer.steps.map(s => `• <b>${s.layerName}:</b> ${s.formula} → [${s.outputShape}]`).join("<br>");
      };

      stepBtn.onclick = () => {
        for (let e = 0; e < 25; e++) mlp.trainStep(X, Y);
        updateUI();
      };
      updateUI();
    } else if (type === "cnn") {
      const cnn = new CNNModel();
      const dummyImage = Tensor.zeros([28, 28, 1]);
      // Draw a line simulating digit 1
      for (let r = 4; r < 24; r++) dummyImage.set(1.0, r, 14, 0);

      const res = cnn.forward(dummyImage, true);

      diagramHost.innerHTML = SVGDiagramRenderer.renderNetwork([
        { id: "conv1", name: "Conv2D (3x3)", type: "conv2d", inShape: [28, 28, 1], outShape: [26, 26, 32], paramsCount: 320 },
        { id: "pool1", name: "MaxPool", type: "maxpool2d", inShape: [26, 26, 32], outShape: [13, 13, 32], paramsCount: 0 },
        { id: "conv2", name: "Conv2D (3x3)", type: "conv2d", inShape: [13, 13, 32], outShape: [11, 11, 64], paramsCount: 18496 },
        { id: "dense", name: "Dense Head", type: "dense", inShape: [576], outShape: [10], paramsCount: 37578 }
      ], 380, 220);

      // Render Conv1 feature map sample on canvas
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.fillStyle = "#000";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#38BDF8";
        ctx.font = "14px monospace";
        ctx.fillText(`Top Predicted Digit: ${res.predictedDigit}`, 20, 40);
        ctx.fillText(`Confidence: ${(res.probabilities[res.predictedDigit] * 100).toFixed(1)}%`, 20, 65);
        ctx.fillText(`Conv1 Output Shape: 26×26×32`, 20, 100);
        ctx.fillText(`Layer 1 Parameters: 320`, 20, 125);
      }

      traceOutput.innerHTML = cnn.tracer.steps.map(s => `• <b>${s.layerName}:</b> ${s.formula} → ${s.pedagogicalInsight}`).join("<br>");
    } else if (type === "transformer" || type === "attention") {
      const transformer = new TransformerAttention(8, 2);
      const tokens = ["The", "neural", "network", "attends", "to", "tokens"];
      const X = Tensor.random([tokens.length, 8], -0.5, 0.5);
      const res = transformer.forward(X, true, true);

      CanvasVisualizer.renderAttentionMatrix(canvas, res.attentionWeights, tokens);

      diagramHost.innerHTML = SVGDiagramRenderer.renderNetwork([
        { id: "emb", name: "Token Embeddings", type: "embedding", inShape: [6], outShape: [6, 8], paramsCount: 48 },
        { id: "qkv", name: "Q, K, V Projections", type: "dense", inShape: [6, 8], outShape: [6, 8], paramsCount: 192 },
        { id: "attn", name: "Attention Matrix", type: "self_attention", inShape: [6, 8], outShape: [6, 6], paramsCount: 0 },
        { id: "out", name: "Linear Projection", type: "dense", inShape: [6, 8], outShape: [6, 8], paramsCount: 64 }
      ], 380, 220);

      traceOutput.innerHTML = transformer.tracer.steps.map(s => `• <b>${s.layerName}:</b> ${s.formula} → ${s.pedagogicalInsight}`).join("<br>");
    } else if (type === "lstm" || type === "rnn") {
      const lstm = new LSTMModel(4, 4);
      const tokens = ["Natural", "Language", "Processing", "Recurrent"];
      const seq = tokens.map(t => ({ token: t, vector: [0.5, -0.2, 0.8, -0.1] }));
      const history = lstm.unroll(seq);

      diagramHost.innerHTML = SVGDiagramRenderer.renderNetwork([
        { id: "in", name: "x_t Token Vector", type: "dense", inShape: [4], outShape: [4], paramsCount: 0 },
        { id: "gates", name: "Gates [f, i, c, o]", type: "lstm_cell", inShape: [4], outShape: [16], paramsCount: 144 },
        { id: "cell", name: "c_t Memory Cell", type: "lstm_cell", inShape: [4], outShape: [4], paramsCount: 0 },
        { id: "h", name: "h_t Hidden State", type: "lstm_cell", inShape: [4], outShape: [4], paramsCount: 0 }
      ], 380, 220);

      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.fillStyle = "#000";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#38BDF8";
        ctx.font = "12px monospace";
        ctx.fillText(`LSTM GATE DYNAMICS:`, 15, 25);
        history.forEach((st, idx) => {
          ctx.fillText(`t=${st.t} ("${st.inputToken}"): f=${st.f_gate[0].toFixed(2)}, i=${st.i_gate[0].toFixed(2)}, o=${st.o_gate[0].toFixed(2)}`, 15, 55 + idx * 30);
        });
      }

      traceOutput.innerHTML = lstm.tracer.steps.map(s => `• <b>${s.layerName}:</b> ${s.formula} → ${s.pedagogicalInsight}`).join("<br>");
    }
  }
}

if (typeof window !== "undefined" && !customElements.get("neural-sim")) {
  customElements.define("neural-sim", NeuralSimElement);
}
