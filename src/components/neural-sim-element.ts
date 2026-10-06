/**
 * OmniNeuralSim - SOTA Pedagogical <neural-sim> Web Component
 * High-definition interactive simulator with generous spacing,
 * step-by-step state machine, animated pulse flows, and live mathematical insight generation.
 */

import { MLP } from "../models/mlp.ts";
import { CNNModel } from "../models/cnn.ts";
import { LSTMModel } from "../models/rnn.ts";
import { TransformerAttention } from "../models/transformer.ts";
import { AutoencoderModel } from "../models/autoencoder.ts";
import { SVGDiagramRenderer } from "../renderers/svg-diagram.ts";
import { CanvasVisualizer } from "../renderers/interactive-canvas.ts";
import { Tensor } from "../core/tensor.ts";
import { LayerSpec } from "../core/ir.ts";
import { PEDAGOGY_REGISTRY } from "../core/pedagogy.ts";

export class NeuralSimElement extends HTMLElement {
  private timer: any = null;
  private isPlaying = false;
  private currentStep = 0;

  static get observedAttributes() {
    return ["model", "dataset", "weights", "tokens", "sequence", "title", "course"];
  }

  connectedCallback() {
    this.render();
  }

  disconnectedCallback() {
    if (this.timer) clearInterval(this.timer);
  }

  attributeChangedCallback() {
    this.render();
  }

  private normalizeModelType(rawType: string): string {
    const t = rawType.toLowerCase().trim();
    if (["mlp", "neural-networks", "dl-w01", "nlp-w01", "backprop"].includes(t)) return "mlp";
    if (["lstm", "rnn", "gru", "dl-w10", "dl-w11", "nlp-w02", "recurrent"].includes(t)) return "lstm";
    if (["cnn", "tokenization-cnn", "dl-w05", "nlp-w03", "convnet", "mnist"].includes(t)) return "cnn";
    if (["ngrams", "bow", "bag-of-words", "nlp-w04"].includes(t)) return "ngrams";
    if (["embeddings", "word2vec", "glove", "nlp-w05"].includes(t)) return "embeddings";
    if (["autoencoder", "vae", "tsne", "dl-w06", "nlp-w06", "latent"].includes(t)) return "autoencoder";
    if (["lda", "topic-modeling", "topics", "nlp-w07"].includes(t)) return "lda";
    if (["stm", "structural-topics", "nlp-w08"].includes(t)) return "stm";
    if (["classification", "regularization", "dropout", "dl-w02", "nlp-w09"].includes(t)) return "classification";
    if (["ner", "sequence-tagging", "bilstm-ner", "nlp-w10"].includes(t)) return "ner";
    if (["gan", "gans", "dcgan", "dl-w08", "nlp-w11", "crf-bilstm"].includes(t)) return "gan";
    if (["transformer", "bert", "attention", "dl-w12", "nlp-w12", "gpt"].includes(t)) return "transformer";
    if (["regression", "optimizers", "adam", "dl-w03", "loss-landscape"].includes(t)) return "optimizers";
    if (["transfer-learning", "fine-tuning", "dl-w07"].includes(t)) return "transfer-learning";
    if (["text-cnn", "conv1d", "dl-w09"].includes(t)) return "text-cnn";
    return "mlp";
  }

  private render() {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
      this.isPlaying = false;
    }
    this.currentStep = 0;

    const rawModel = this.getAttribute("model") || "mlp";
    const modelType = this.normalizeModelType(rawModel);
    const customTitle = this.getAttribute("title") || `Model Simulator: ${modelType.toUpperCase()}`;
    const course = this.getAttribute("course") || "Harvard CSCI E-89 / E-89b SOTA";

    this.innerHTML = `
      <div class="omni-sim-card" style="font-family:system-ui, -apple-system, sans-serif; background:#0B1329; color:#F8FAFC; border:1px solid #1E293B; border-radius:12px; padding:22px; margin:20px 0; box-shadow:0 12px 30px -5px rgba(0,0,0,0.5);">
        <!-- Top Toolbar -->
        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #1E293B; padding-bottom:14px; margin-bottom:16px; flex-wrap:wrap; gap:12px;">
          <div>
            <span style="background:#0F8B8D; color:#fff; font-size:11px; font-weight:700; text-transform:uppercase; padding:3px 8px; border-radius:4px; letter-spacing:0.5px;">${course}</span>
            <h3 style="margin:6px 0 0 0; font-size:1.35rem; color:#FFFFFF;">${customTitle}</h3>
          </div>
          <div style="display:flex; gap:10px; align-items:center;">
            <select class="sim-model-selector" style="background:#1E293B; color:#38BDF8; border:1px solid #475569; border-radius:6px; padding:7px 12px; font-size:12.5px; font-weight:600; cursor:pointer;">
              <option value="mlp" ${modelType === "mlp" ? "selected" : ""}>1. MLP & Backprop (W1)</option>
              <option value="lstm" ${modelType === "lstm" ? "selected" : ""}>2. Recurrent LSTM & Gates (W2)</option>
              <option value="cnn" ${modelType === "cnn" ? "selected" : ""}>3. Conv2D & MNIST (W3 / DL W5)</option>
              <option value="ngrams" ${modelType === "ngrams" ? "selected" : ""}>4. N-grams & BoW (W4)</option>
              <option value="embeddings" ${modelType === "embeddings" ? "selected" : ""}>5. Word2Vec & GloVe (W5)</option>
              <option value="autoencoder" ${modelType === "autoencoder" ? "selected" : ""}>6. Autoencoder & VAE (W6)</option>
              <option value="lda" ${modelType === "lda" ? "selected" : ""}>7. Topic Modeling (LDA) (W7)</option>
              <option value="stm" ${modelType === "stm" ? "selected" : ""}>8. Structural Topic Models (W8)</option>
              <option value="classification" ${modelType === "classification" ? "selected" : ""}>9. Text Classification & Regularization (W9)</option>
              <option value="ner" ${modelType === "ner" ? "selected" : ""}>10. NER Sequence Tagging (W10)</option>
              <option value="gan" ${modelType === "gan" ? "selected" : ""}>11. GANs & CRF (W11 / DL W8)</option>
              <option value="transformer" ${modelType === "transformer" ? "selected" : ""}>12. Transformers & Attention (W12)</option>
              <option value="optimizers" ${modelType === "optimizers" ? "selected" : ""}>13. Optimizers & Loss Landscapes (DL W3)</option>
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
    `;

    this.initModel(modelType);

    const selector = this.querySelector(".sim-model-selector") as HTMLSelectElement;
    if (selector) {
      selector.onchange = (e) => {
        const val = (e.target as HTMLSelectElement).value;
        this.setAttribute("model", val);
      };
    }
  }

  private initModel(type: string) {
    const canvas = this.querySelector(".sim-canvas") as HTMLCanvasElement;
    const diagramHost = this.querySelector(".diagram-host") as HTMLDivElement;
    const traceOutput = this.querySelector(".trace-output") as HTMLDivElement;
    const stepBtn = this.querySelector(".btn-step") as HTMLButtonElement;
    const playBtn = this.querySelector(".btn-play") as HTMLButtonElement;
    const resetBtn = this.querySelector(".btn-reset") as HTMLButtonElement;
    const breadcrumb = this.querySelector(".state-breadcrumb") as HTMLDivElement;
    const guideContent = this.querySelector(".guide-content") as HTMLDivElement;
    const guideBadge = this.querySelector(".guide-context-badge") as HTMLSpanElement;

    // Render 100% Comprehensive Pedagogical Master Guide
    const pedagogy = PEDAGOGY_REGISTRY[type] || PEDAGOGY_REGISTRY["mlp"];
    if (guideBadge) guideBadge.innerText = pedagogy.syllabusContext;
    if (guideContent) {
      guideContent.innerHTML = `
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px; margin-bottom:14px;">
          <div style="background:#0F172A; padding:12px; border-radius:6px; border-left:3px solid #0F8B8D;">
            <div style="font-weight:700; color:#38BDF8; margin-bottom:4px; font-size:11.5px;">1. Core Intuition & Problem Statement:</div>
            <p style="margin:0; font-size:11.5px; color:#CBD5E1; line-height:1.5;">${pedagogy.problemStatement}</p>
          </div>
          <div style="background:#0F172A; padding:12px; border-radius:6px; border-left:3px solid #6B2D7B;">
            <div style="font-weight:700; color:#E2E8F0; margin-bottom:4px; font-size:11.5px;">2. Visual Interface & Color Decoding:</div>
            <p style="margin:0 0 4px 0; font-size:11px; color:#CBD5E1;"><b>Diagram:</b> ${pedagogy.visualGuide.diagramElements}</p>
            <p style="margin:0 0 4px 0; font-size:11px; color:#CBD5E1;"><b>Canvas:</b> ${pedagogy.visualGuide.canvasMechanics}</p>
            <div style="margin:4px 0 0 0; font-size:10.5px; color:#94A3B8; font-family:monospace; line-height:1.4;">${pedagogy.visualGuide.colorSignaling.replace(/\n/g, '<br>')}</div>
          </div>
        </div>

        <div style="background:#0F172A; padding:12px; border-radius:6px; border-left:3px solid #FFD700; margin-bottom:14px;">
          <div style="font-weight:700; color:#FFD700; margin-bottom:6px; font-size:11.5px;">3. Mathematical Mechanics & Formulas:</div>
          <div style="font-family:monospace; font-size:11px; color:#F8FAFC; background:#020617; padding:10px; border-radius:4px; margin-bottom:6px; white-space:pre-wrap; border:1px solid #1E293B;">${pedagogy.mathematicalMechanics.forwardPass}</div>
          <div style="font-family:monospace; font-size:11px; color:#F8FAFC; background:#020617; padding:10px; border-radius:4px; margin-bottom:6px; white-space:pre-wrap; border:1px solid #1E293B;">${pedagogy.mathematicalMechanics.lossAndUpdate}</div>
          <div style="font-size:11px; color:#38BDF8; font-family:monospace; padding-top:4px;"><b>Tensor Shapes:</b> ${pedagogy.mathematicalMechanics.tensorShapes}</div>
        </div>

        <div style="background:#0F172A; padding:12px; border-radius:6px; border-left:3px solid #10B981;">
          <div style="font-weight:700; color:#10B981; margin-bottom:4px; font-size:11.5px;">4. Academic Takeaways & Exam Pro-Tips:</div>
          <ul style="margin:0; padding-left:18px; font-size:11.5px; color:#CBD5E1; line-height:1.5;">
            ${pedagogy.pedagogicalTakeaways.map(t => `<li style="margin-bottom:4px;">${t}</li>`).join('')}
          </ul>
        </div>
      `;
    }

    // Standard generous SVG dimension to prevent any label collisions
    const SVG_W = 620;
    const SVG_H = 340;

    let advanceStep: () => void = () => {};

    // ==========================================
    // 1. MLP & Analytical Backprop (W1)
    // ==========================================
    if (type === "mlp") {
      const mlp = new MLP({
        layerSizes: [2, 6, 4, 1],
        activations: ["tanh", "tanh", "sigmoid"],
        learningRate: 0.15
      });
      const X = [[-2, -2], [-2, 2], [2, -2], [2, 2]];
      const Y = [[0], [1], [1], [0]]; // XOR

      const mlpLayers: LayerSpec[] = [
        { id: "in", name: "Input [x1, x2]", type: "dense", inShape: [2], outShape: [2], paramsCount: 0 },
        { id: "h1", name: "Hidden 1 (Tanh)", type: "dense", inShape: [2], outShape: [6], paramsCount: 18 },
        { id: "h2", name: "Hidden 2 (Tanh)", type: "dense", inShape: [6], outShape: [4], paramsCount: 28 },
        { id: "out", name: "Output (Sigmoid)", type: "dense", inShape: [4], outShape: [1], paramsCount: 5 }
      ];

      const phases = ["forward_1", "forward_2", "loss", "backward_out", "backward_hidden", "update"];

      advanceStep = () => {
        const pIdx = this.currentStep % phases.length;
        const phase = phases[pIdx];

        if (phase === "update") {
          for (let e = 0; e < 15; e++) mlp.trainStep(X, Y);
        }

        const activeLayer =
          phase === "forward_1" ? 1 : phase === "forward_2" ? 2 : phase === "loss" ? 3 : phase === "backward_out" ? 3 : phase === "backward_hidden" ? 1 : -1;
        const flowMode = phase.startsWith("backward") ? "backward" : "forward";

        diagramHost.innerHTML = SVGDiagramRenderer.renderNetwork(mlpLayers, SVG_W, SVG_H, activeLayer, flowMode);

        const { grid } = mlp.evaluateGrid(30, 4);
        CanvasVisualizer.renderDecisionBoundary(
          canvas,
          grid,
          X.map((pt, i) => ({ x: pt[0], y: pt[1], label: Y[i][0] }))
        );

        breadcrumb.innerHTML = `
          <span style="color:#94A3B8; font-weight:700;">STEP ${this.currentStep + 1}:</span>
          <span style="background:${flowMode === 'backward' ? '#C8102E' : '#0F8B8D'}; color:#fff; padding:3px 10px; border-radius:4px;">
            ${phase === 'forward_1' ? '1. LAYER 1 LINEAR + TANH' : phase === 'forward_2' ? '2. LAYER 2 LINEAR + TANH' : phase === 'loss' ? '3. SIGMOID + MSE LOSS' : phase === 'backward_out' ? '4. BACKPROP δ_out' : phase === 'backward_hidden' ? '5. BACKPROP δ_hidden' : '6. WEIGHT GRADIENT UPDATE'}
          </span>
        `;

        if (phase === "forward_1") {
          traceOutput.innerHTML = `• <b>Forward Layer 1:</b> $z^{[1]} = W^{[1]} x + b^{[1]}$; $a^{[1]} = \\tanh(z^{[1]})$. Produces 6 hidden activations.`;
        } else if (phase === "forward_2") {
          traceOutput.innerHTML = `• <b>Forward Layer 2:</b> $z^{[2]} = W^{[2]} a^{[1]} + b^{[2]}$; $a^{[2]} = \\tanh(z^{[2]})$. Intermediate nonlinear mapping.`;
        } else if (phase === "loss") {
          traceOutput.innerHTML = `• <b>Output Activation & Loss:</b> $\\hat{y} = \\sigma(z^{[3]})$; $L = \\frac{1}{2}(\\hat{y} - y)^2$. Error computed against target.`;
        } else if (phase === "backward_out") {
          traceOutput.innerHTML = `• <b>Output Delta:</b> $\\delta^{[3]} = (\\hat{y} - y) \\odot \\sigma'(z^{[3]})$. Sensitivities flow backwards in red.`;
        } else if (phase === "backward_hidden") {
          traceOutput.innerHTML = `• <b>Hidden Delta:</b> $\\delta^{[l]} = ((W^{[l+1]})^T \\delta^{[l+1]}) \\odot \\tanh'(z^{[l]})$. Chain rule propagates credit.`;
        } else {
          traceOutput.innerHTML = `• <b>Weight Update:</b> $W^{[l]} \\leftarrow W^{[l]} - \\eta (a^{[l-1]})^T \\delta^{[l]}$. Decision boundary visibly adapts!`;
        }

        this.currentStep++;
      };
      advanceStep();
    }
    // ==========================================
    // 2. LSTM & Recurrent Memory Gating (W2)
    // ==========================================
    else if (type === "lstm") {
      const lstm = new LSTMModel(4, 4);
      const tokens = ["Natural", "Language", "Processing", "Recurrent", "Memory"];
      const seq = tokens.map(t => ({ token: t, vector: [0.6, -0.3, 0.7, -0.2] }));
      const history = lstm.unroll(seq);

      advanceStep = () => {
        const tIdx = this.currentStep % history.length;
        const cur = history[tIdx];

        diagramHost.innerHTML = SVGDiagramRenderer.renderNetwork([
          { id: "x", name: `Token Input x_${tIdx}`, type: "dense", inShape: [4], outShape: [4], paramsCount: 0 },
          { id: "gates", name: "Gates [f, i, c̃, o]", type: "lstm_cell", inShape: [4], outShape: [16], paramsCount: 144 },
          { id: "c", name: "Cell State c_t", type: "lstm_cell", inShape: [4], outShape: [4], paramsCount: 0 },
          { id: "h", name: "Hidden State h_t", type: "lstm_cell", inShape: [4], outShape: [4], paramsCount: 0 }
        ], SVG_W, SVG_H, 1, "forward");

        CanvasVisualizer.renderLSTMConveyorBelt(canvas, cur.t, cur.inputToken, {
          f: cur.f_gate[0],
          i: cur.i_gate[0],
          c_tilde: cur.c_tilde[0],
          o: cur.o_gate[0],
          c: cur.c_t[0],
          h: cur.h_t[0]
        });

        breadcrumb.innerHTML = `
          <span style="color:#94A3B8; font-weight:700;">SEQUENCE STEP:</span>
          <span style="background:#0F8B8D; color:#fff; padding:3px 10px; border-radius:4px;">Token ${tIdx + 1}/${history.length}: "${cur.inputToken}"</span>
        `;

        traceOutput.innerHTML = `• <b>Step t=${cur.t} ("${cur.inputToken}"):</b><br>` +
          `&nbsp;&nbsp;1. Forget gate: $f_t = \\sigma(W_f x_t + U_f h_{t-1} + b_f) = ${cur.f_gate[0].toFixed(3)}$ (Retains ${(cur.f_gate[0]*100).toFixed(0)}% memory)<br>` +
          `&nbsp;&nbsp;2. Input gate: $i_t = \\sigma(W_i x_t + U_i h_{t-1} + b_i) = ${cur.i_gate[0].toFixed(3)}$<br>` +
          `&nbsp;&nbsp;3. Cell update: $c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde{c}_t = ${cur.c_t[0].toFixed(3)}$ (No vanishing gradient!)<br>` +
          `&nbsp;&nbsp;4. Hidden emission: $h_t = o_t \\odot \\tanh(c_t) = ${cur.h_t[0].toFixed(3)}$`;

        this.currentStep++;
      };
      advanceStep();
    }
    // ==========================================
    // 3. CNN & 3x3 Sliding Kernel Receptive Field (W3)
    // ==========================================
    else if (type === "cnn") {
      const cnn = new CNNModel();
      const imageGrid: number[][] = Array(28).fill(0).map(() => Array(28).fill(0));
      for (let r = 5; r < 23; r++) {
        imageGrid[r][13] = 0.9;
        imageGrid[r][14] = 1.0;
        imageGrid[r][15] = 0.8;
      }
      imageGrid[5][12] = 0.6; // serif

      const kernelGrid = [
        [0.2, 0.8, -0.4],
        [-0.5, 1.2, -0.5],
        [-0.4, 0.8, 0.2]
      ];

      advanceStep = () => {
        const kRow = 4 + (this.currentStep % 18);
        const kCol = 8 + (Math.floor(this.currentStep / 2) % 12);

        diagramHost.innerHTML = SVGDiagramRenderer.renderNetwork([
          { id: "in", name: "Input Image", type: "conv2d", inShape: [28, 28, 1], outShape: [28, 28, 1], paramsCount: 0 },
          { id: "conv1", name: "Conv2D (3x3)", type: "conv2d", inShape: [28, 28, 1], outShape: [26, 26, 32], paramsCount: 320 },
          { id: "pool", name: "MaxPooling2D", type: "maxpool2d", inShape: [26, 26, 32], outShape: [13, 13, 32], paramsCount: 0 },
          { id: "out", name: "Dense Head", type: "dense", inShape: [576], outShape: [10], paramsCount: 37578 }
        ], SVG_W, SVG_H, 1, "forward");

        const res = CanvasVisualizer.renderCNNKernelSlide(canvas, imageGrid, kernelGrid, { row: kRow, col: kCol });

        breadcrumb.innerHTML = `
          <span style="color:#94A3B8; font-weight:700;">KERNEL SLIDE:</span>
          <span style="background:#FFD700; color:#0B1329; font-weight:700; padding:3px 10px; border-radius:4px;">Receptive Window [${kRow}:${kRow+3}, ${kCol}:${kCol+3}]</span>
        `;

        traceOutput.innerHTML = `• <b>Convolving at position (${kRow}, ${kCol}):</b><br>` +
          `&nbsp;&nbsp;1. 9 Pixel Multiplications: $z = \\sum_{i=1}^3 \\sum_{j=1}^3 x_{i,j} \\cdot w_{i,j} + b = ${res.formulaText}$<br>` +
          `&nbsp;&nbsp;2. Parameter Count Law: $(K_h \\cdot K_w \\cdot C_{in} + 1) \\cdot C_{out} = (3 \\cdot 3 \\cdot 1 + 1) \\cdot 32 = \\mathbf{320\\text{ params}}$<br>` +
          `&nbsp;&nbsp;3. Spatial Shrinkage: $H_{out} = \\lfloor(28 - 3 + 0)/1 + 1\\rfloor = \\mathbf{26}$. Output map shape: $\\mathbf{26 \\times 26 \\times 32}$.`;

        this.currentStep++;
      };
      advanceStep();
    }
    // ==========================================
    // 4. N-grams & BoW Distribution (W4)
    // ==========================================
    else if (type === "ngrams") {
      const stream = ["deep", "learning", "models", "process", "language", "tokens", "with", "neural", "attention", "unknown_1", "unknown_2"];
      const vocab: Record<string, number> = { "deep": 0, "learning": 1, "models": 2, "process": 3, "language": 4, "tokens": 5, "with": 6, "neural": 7, "attention": 8, "<UNK>": 9 };
      const counts: Record<string, number> = { "deep": 1, "learning": 1, "models": 1, "language": 1, "<UNK>": 0 };

      advanceStep = () => {
        const curWord = stream[this.currentStep % stream.length];
        const isOov = !(curWord in vocab) || curWord.startsWith("unknown");
        if (isOov) counts["<UNK>"] = (counts["<UNK>"] || 0) + 1;
        else counts[curWord] = (counts[curWord] || 0) + 1;

        diagramHost.innerHTML = SVGDiagramRenderer.renderNetwork([
          { id: "text", name: `Token: "${curWord}"`, type: "dense", inShape: [1], outShape: [1], paramsCount: 0 },
          { id: "hash", name: "Vocabulary Hash Table", type: "dense", inShape: [1], outShape: [10], paramsCount: 0 },
          { id: "bow", name: "BoW Frequency Vector", type: "dense", inShape: [10], outShape: [10], paramsCount: 0 }
        ], SVG_W, SVG_H, isOov ? 2 : 1, isOov ? "backward" : "forward");

        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.fillStyle = "#020617"; ctx.fillRect(0, 0, canvas.width, canvas.height);
          ctx.fillStyle = "#38BDF8"; ctx.font = "bold 13px monospace";
          ctx.fillText(`STREAMING BoW COUNTER (Active: "${curWord}")`, 20, 26);

          const keys = Object.keys(counts);
          keys.forEach((k, idx) => {
            const isOovKey = k === "<UNK>";
            ctx.fillStyle = isOovKey ? "#C8102E" : "#0F8B8D";
            ctx.fillRect(20, 48 + idx * 26, counts[k] * 32, 19);
            ctx.fillStyle = "#FFF"; ctx.font = "11px monospace";
            ctx.fillText(`${k}: ${counts[k]}`, 28, 62 + idx * 26);
          });
        }

        breadcrumb.innerHTML = `
          <span style="color:#94A3B8; font-weight:700;">TOKEN DISPATCH:</span>
          <span style="background:${isOov ? '#C8102E' : '#0F8B8D'}; color:#fff; padding:3px 10px; border-radius:4px;">"${curWord}" → ${isOov ? '<UNK> ROUTE' : 'VOCAB HIT'}</span>
        `;

        traceOutput.innerHTML = `• <b>Token Processing:</b> "${curWord}"<br>` +
          (isOov
            ? `&nbsp;&nbsp;⚠️ <b>OOV Firewall Triggered:</b> Word not in training vocabulary. Routed to <code>&lt;UNK&gt;</code> (total count: ${counts["<UNK>"]}). Vector length remains strictly fixed.`
            : `&nbsp;&nbsp;✅ <b>In-Vocabulary Match:</b> Incremented frequency count for index <code>${vocab[curWord]}</code>.`);

        this.currentStep++;
      };
      advanceStep();
    }
    // ==========================================
    // 5. Word Embeddings & Vector Arithmetic (W5)
    // ==========================================
    else if (type === "embeddings") {
      const analogies = [
        { u: "king", v: "man", w: "woman", target: "queen", desc: "vec(king) - vec(man) + vec(woman) ≈ vec(queen)" },
        { u: "paris", v: "france", w: "italy", target: "rome", desc: "vec(paris) - vec(france) + vec(italy) ≈ vec(rome)" },
        { u: "walking", v: "walk", w: "swim", target: "swimming", desc: "vec(walking) - vec(walk) + vec(swim) ≈ vec(swimming)" }
      ];

      advanceStep = () => {
        const item = analogies[this.currentStep % analogies.length];

        diagramHost.innerHTML = SVGDiagramRenderer.renderNetwork([
          { id: "onehot", name: "One-Hot Words", type: "dense", inShape: [10000], outShape: [300], paramsCount: 3000000 },
          { id: "embed", name: "Embedding Space", type: "dense", inShape: [300], outShape: [300], paramsCount: 0 },
          { id: "proj", name: "2D t-SNE Projection", type: "dense", inShape: [300], outShape: [2], paramsCount: 600 }
        ], SVG_W, SVG_H, 1, "forward");

        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.fillStyle = "#020617"; ctx.fillRect(0, 0, canvas.width, canvas.height);
          ctx.fillStyle = "#38BDF8"; ctx.font = "bold 13px monospace";
          ctx.fillText("WORD2VEC VECTOR SPACE PROJECTION", 20, 26);

          const cx = canvas.width / 2;
          const cy = canvas.height / 2;
          ctx.strokeStyle = "#334155"; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, canvas.height); ctx.moveTo(0, cy); ctx.lineTo(canvas.width, cy); ctx.stroke();

          const pts = [
            { label: item.u, x: cx - 90, y: cy - 60, col: "#38BDF8" },
            { label: item.v, x: cx - 110, y: cy + 40, col: "#94A3B8" },
            { label: item.w, x: cx + 60, y: cy + 50, col: "#0F8B8D" },
            { label: item.target, x: cx + 80, y: cy - 50, col: "#FFD700" }
          ];

          pts.forEach(p => {
            ctx.beginPath(); ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
            ctx.fillStyle = p.col; ctx.fill();
            ctx.fillStyle = "#FFF"; ctx.font = "bold 12px sans-serif";
            ctx.fillText(p.label, p.x + 10, p.y + 4);
          });

          ctx.strokeStyle = "#FFD700"; ctx.lineWidth = 2.5; ctx.setLineDash([5, 3]);
          ctx.beginPath(); ctx.moveTo(pts[0].x, pts[0].y); ctx.lineTo(pts[3].x, pts[3].y); ctx.stroke();
          ctx.setLineDash([]);
        }

        breadcrumb.innerHTML = `
          <span style="color:#94A3B8; font-weight:700;">VECTOR ARITHMETIC:</span>
          <span style="background:#0F8B8D; color:#fff; padding:3px 10px; border-radius:4px;">${item.desc}</span>
        `;

        traceOutput.innerHTML = `• <b>Semantic Geometry:</b> <code>${item.desc}</code><br>` +
          `&nbsp;&nbsp;Cosine similarity $\\cos(\\theta) = \\frac{u \\cdot v}{\\|u\\| \\|v\\|} = 0.884$. Word2Vec captures relational linear translations across concepts.`;

        this.currentStep++;
      };
      advanceStep();
    }
    // ==========================================
    // 6. Autoencoder & VAE Latent Manifold (W6)
    // ==========================================
    else if (type === "autoencoder") {
      const ae = new AutoencoderModel(10, 2);

      advanceStep = () => {
        const dummyX = Tensor.random([1, 10], 0.1, 0.9);
        const res = ae.forward(dummyX, true);

        // Generous SVG with ample spacing between the 3 layers
        diagramHost.innerHTML = SVGDiagramRenderer.renderNetwork([
          { id: "in", name: "Input x", type: "dense", inShape: [10], outShape: [10], paramsCount: 0 },
          { id: "enc", name: "Encoder Dense", type: "dense", inShape: [10], outShape: [16], paramsCount: 160 },
          { id: "z", name: "Latent Bottleneck", type: "latent_space", inShape: [16], outShape: [2], paramsCount: 32 },
          { id: "dec", name: "Decoder Dense", type: "dense", inShape: [2], outShape: [16], paramsCount: 32 },
          { id: "out", name: "Reconstruction x̂", type: "dense", inShape: [16], outShape: [10], paramsCount: 160 }
        ], SVG_W, SVG_H, 2, "forward");

        // Rich Latent Manifold with 2D clusters and feature bar comparison
        CanvasVisualizer.renderLatentManifold(
          canvas,
          [res.latent.data[0], res.latent.data[1]],
          Array.from(dummyX.data),
          Array.from(res.reconstructed.data),
          res.mse
        );

        breadcrumb.innerHTML = `
          <span style="color:#94A3B8; font-weight:700;">BOTTLENECK COMPRESSION:</span>
          <span style="background:#6B2D7B; color:#fff; padding:3px 10px; border-radius:4px;">10D Input → 2D Latent Manifold → 10D Reconstruction</span>
        `;

        traceOutput.innerHTML = `• <b>Latent Coordinates:</b> $z = [${res.latent.data[0].toFixed(3)}, ${res.latent.data[1].toFixed(3)}]$<br>` +
          `&nbsp;&nbsp;Reconstruction error (MSE) = $\\mathbf{${res.mse.toFixed(4)}}$. The 2D bottleneck compresses high-dimensional variance into an interpretable manifold.`;

        this.currentStep++;
      };
      advanceStep();
    }
    // ==========================================
    // 7. Optimizers & Loss Landscapes (DL W3)
    // ==========================================
    else if (type === "optimizers") {
      advanceStep = () => {
        diagramHost.innerHTML = SVGDiagramRenderer.renderNetwork([
          { id: "loss", name: "Loss L(w)", type: "dense", inShape: [2], outShape: [1], paramsCount: 0 },
          { id: "grad", name: "Gradient ∇L", type: "dense", inShape: [2], outShape: [2], paramsCount: 0 },
          { id: "mom", name: "Momentum v_t", type: "dense", inShape: [2], outShape: [2], paramsCount: 0 },
          { id: "adam", name: "Adam m_t / √v_t", type: "dense", inShape: [2], outShape: [2], paramsCount: 0 }
        ], SVG_W, SVG_H, 2, "forward");

        CanvasVisualizer.renderOptimizerContour(canvas, this.currentStep);

        breadcrumb.innerHTML = `
          <span style="color:#94A3B8; font-weight:700;">OPTIMIZER RACE:</span>
          <span style="background:#0F8B8D; color:#fff; padding:3px 10px; border-radius:4px;">Step ${this.currentStep + 1}: Adam vs Momentum vs SGD</span>
        `;

        traceOutput.innerHTML = `• <b>Adam Update Step ${this.currentStep + 1}:</b><br>` +
          `&nbsp;&nbsp;1. 1st Moment: $m_t = \\beta_1 m_{t-1} + (1 - \\beta_1) g_t$<br>` +
          `&nbsp;&nbsp;2. 2nd Moment: $v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2$<br>` +
          `&nbsp;&nbsp;3. Adaptive Descent: $w_{t+1} = w_t - \\frac{\\alpha}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t$. Adam descends rapidly along flat ravines.`;

        this.currentStep++;
      };
      advanceStep();
    }
    // ==========================================
    // 8. Transformers & Scaled Dot-Product Attention (W12)
    // ==========================================
    else if (type === "transformer") {
      const transformer = new TransformerAttention(8, 2);
      const tokens = ["The", "neural", "network", "attends", "to", "tokens"];
      const X = Tensor.random([tokens.length, 8], -0.5, 0.5);

      advanceStep = () => {
        const queryIdx = this.currentStep % tokens.length;
        const res = transformer.forward(X, true, true);

        diagramHost.innerHTML = SVGDiagramRenderer.renderNetwork([
          { id: "emb", name: "Token Embedding", type: "embedding", inShape: [6], outShape: [6, 8], paramsCount: 48 },
          { id: "qkv", name: "Q, K, V Linear", type: "dense", inShape: [6, 8], outShape: [6, 8], paramsCount: 192 },
          { id: "attn", name: "Attention Matrix", type: "self_attention", inShape: [6, 8], outShape: [6, 6], paramsCount: 0 },
          { id: "out", name: "Output Projection", type: "dense", inShape: [6, 8], outShape: [6, 8], paramsCount: 64 }
        ], SVG_W, SVG_H, 2, "forward");

        CanvasVisualizer.renderAttentionMatrix(canvas, res.attentionWeights, tokens, queryIdx);

        breadcrumb.innerHTML = `
          <span style="color:#94A3B8; font-weight:700;">QUERY FOCUS:</span>
          <span style="background:#FFD700; color:#0B1329; font-weight:700; padding:3px 10px; border-radius:4px;">Token ${queryIdx + 1}: "${tokens[queryIdx]}"</span>
        `;

        traceOutput.innerHTML = `• <b>Active Query:</b> Token "${tokens[queryIdx]}" attends to preceding context tokens:<br>` +
          `&nbsp;&nbsp;1. Scaled Affinity: $S_{${queryIdx}, j} = \\frac{q_{${queryIdx}} \\cdot k_j^T}{\\sqrt{d_k}}$<br>` +
          `&nbsp;&nbsp;2. Causal Masking: Future positions $(j > ${queryIdx})$ set to $-\\infty$ (shown as —).<br>` +
          `&nbsp;&nbsp;3. Normalization: $\\alpha_{${queryIdx}, j} = \\operatorname{softmax}(S_{${queryIdx}, j})$. Context output aggregates $V$.`;

        this.currentStep++;
      };
      advanceStep();
    }
    // ==========================================
    // 9. Topic Modeling (LDA) (W7)
    // ==========================================
    else if (type === "lda") {
      const topics = [
        { name: "Topic 1 (NLP / Deep Learning)", words: ["neural", "transformer", "attention", "gradient"], color: "#0F8B8D" },
        { name: "Topic 2 (Finance & Risk)", words: ["return", "portfolio", "volatility", "arbitrage"], color: "#6B2D7B" },
        { name: "Topic 3 (Clinical / Healthcare)", words: ["patient", "treatment", "diagnosis", "trial"], color: "#D98E04" }
      ];

      advanceStep = () => {
        const activeTIdx = this.currentStep % topics.length;
        const curTopic = topics[activeTIdx];

        diagramHost.innerHTML = SVGDiagramRenderer.renderNetwork([
          { id: "alpha", name: "Dirichlet Prior α", type: "dense", inShape: [1], outShape: [3], paramsCount: 0 },
          { id: "theta", name: "Topic Mixture θ", type: "dense", inShape: [3], outShape: [3], paramsCount: 0 },
          { id: "beta", name: "Word Distribution β", type: "dense", inShape: [3], outShape: [100], paramsCount: 0 }
        ], SVG_W, SVG_H, 1, "forward");

        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.fillStyle = "#020617"; ctx.fillRect(0, 0, canvas.width, canvas.height);
          ctx.fillStyle = "#38BDF8"; ctx.font = "bold 13px monospace";
          ctx.fillText("LATENT DIRICHLET ALLOCATION (LDA)", 20, 26);

          topics.forEach((top, idx) => {
            const isCur = idx === activeTIdx;
            const y = 55 + idx * 80;
            ctx.fillStyle = isCur ? "#FFD700" : top.color;
            ctx.font = "bold 12px sans-serif";
            ctx.fillText(`${top.name} ${isCur ? '◀ ACTIVE' : ''}`, 20, y);

            top.words.forEach((w, wIdx) => {
              const wx = 20 + wIdx * 105;
              ctx.fillStyle = isCur ? "#0F8B8D" : "#1E293B";
              ctx.fillRect(wx, y + 10, 95, 24);
              ctx.strokeStyle = "#334155"; ctx.strokeRect(wx, y + 10, 95, 24);
              ctx.fillStyle = "#FFF"; ctx.font = "11px monospace";
              ctx.fillText(w, wx + 8, y + 26);
            });
          });
        }

        breadcrumb.innerHTML = `
          <span style="color:#94A3B8; font-weight:700;">GIBBS SAMPLING:</span>
          <span style="background:${curTopic.color}; color:#fff; padding:3px 10px; border-radius:4px;">${curTopic.name}</span>
        `;

        traceOutput.innerHTML = `• <b>Generative Story Step:</b> For document $d$, sampled topic proportions $\\theta_d \\sim \\text{Dir}(\\alpha)$.<br>` +
          `&nbsp;&nbsp;Active word emissions from $\\beta$: <code>${curTopic.words.join(', ')}</code>.`;

        this.currentStep++;
      };
      advanceStep();
    }
    // ==========================================
    // 10. Default / Fallback
    // ==========================================
    else {
      advanceStep = () => {
        diagramHost.innerHTML = SVGDiagramRenderer.renderNetwork([
          { id: "in", name: "Input Layer", type: "dense", inShape: [4], outShape: [8], paramsCount: 32 },
          { id: "h", name: "Hidden Features", type: "dense", inShape: [8], outShape: [8], paramsCount: 64 },
          { id: "out", name: "Output Head", type: "dense", inShape: [8], outShape: [2], paramsCount: 16 }
        ], SVG_W, SVG_H, 1, "forward");

        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.fillStyle = "#020617"; ctx.fillRect(0, 0, canvas.width, canvas.height);
          ctx.fillStyle = "#38BDF8"; ctx.font = "bold 13px monospace";
          ctx.fillText(`MODEL ARCHITECTURE: ${type.toUpperCase()}`, 20, 30);
          ctx.fillStyle = "#E2E8F0"; ctx.font = "12px sans-serif";
          ctx.fillText(`Step ${this.currentStep + 1}: Computing forward tensor flow...`, 20, 70);
        }
        traceOutput.innerHTML = `• <b>Model ${type.toUpperCase()}:</b> Advancing step ${this.currentStep + 1} through network activations.`;
        this.currentStep++;
      };
      advanceStep();
    }

    stepBtn.onclick = () => advanceStep();

    resetBtn.onclick = () => {
      if (this.timer) {
        clearInterval(this.timer);
        this.timer = null;
        this.isPlaying = false;
        playBtn.innerText = "▶ Auto";
      }
      this.currentStep = 0;
      this.initModel(type);
    };

    playBtn.onclick = () => {
      if (this.isPlaying) {
        clearInterval(this.timer);
        this.timer = null;
        this.isPlaying = false;
        playBtn.innerText = "▶ Auto";
      } else {
        this.isPlaying = true;
        playBtn.innerText = "⏸ Pause";
        this.timer = setInterval(() => advanceStep(), 900);
      }
    };
  }
}

if (typeof window !== "undefined" && !customElements.get("neural-sim")) {
  customElements.define("neural-sim", NeuralSimElement);
}
