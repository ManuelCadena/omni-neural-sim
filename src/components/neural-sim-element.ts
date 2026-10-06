/**
 * OmniNeuralSim - <neural-sim> Web Component
 * Standalone, zero-framework Custom Element embeddable into any HTML document/assignment.
 * Supports all Harvard CSCI E-89 (Deep Learning) & CSCI E-89b (NLP) weekly models.
 */

import { MLP } from "../models/mlp.ts";
import { CNNModel } from "../models/cnn.ts";
import { LSTMModel } from "../models/rnn.ts";
import { TransformerAttention } from "../models/transformer.ts";
import { AutoencoderModel } from "../models/autoencoder.ts";
import { SVGDiagramRenderer } from "../renderers/svg-diagram.ts";
import { CanvasVisualizer } from "../renderers/interactive-canvas.ts";
import { Tensor } from "../core/tensor.ts";

export class NeuralSimElement extends HTMLElement {
  static get observedAttributes() {
    return ["model", "dataset", "weights", "tokens", "sequence", "title", "course"];
  }

  connectedCallback() {
    this.render();
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
    const rawModel = this.getAttribute("model") || "mlp";
    const modelType = this.normalizeModelType(rawModel);
    const customTitle = this.getAttribute("title") || `Model Simulator: ${modelType.toUpperCase()}`;
    const course = this.getAttribute("course") || "Harvard CSCI E-89 / E-89b SOTA";

    this.innerHTML = `
      <div class="omni-sim-card" style="font-family:system-ui, -apple-system, sans-serif; background:#0F172A; color:#F8FAFC; border:1px solid #334155; border-radius:12px; padding:20px; margin:20px 0; box-shadow:0 10px 25px -5px rgba(0,0,0,0.3);">
        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #1E293B; padding-bottom:12px; margin-bottom:16px; flex-wrap:wrap; gap:10px;">
          <div>
            <span style="background:#0F8B8D; color:#fff; font-size:11px; font-weight:700; text-transform:uppercase; padding:3px 8px; border-radius:4px; letter-spacing:0.5px;">${course}</span>
            <h3 style="margin:6px 0 0 0; font-size:1.25rem; color:#FFFFFF;">${customTitle}</h3>
          </div>
          <div style="display:flex; gap:8px; align-items:center;">
            <select class="sim-model-selector" style="background:#1E293B; color:#38BDF8; border:1px solid #475569; border-radius:6px; padding:6px 10px; font-size:12px; font-weight:600; cursor:pointer;">
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
    const metricsHost = this.querySelector(".sim-metrics") as HTMLDivElement;
    const stepBtn = this.querySelector(".btn-step") as HTMLButtonElement;
    const resetBtn = this.querySelector(".btn-reset") as HTMLButtonElement;

    // --- 1. MLP & Backprop (W1) ---
    if (type === "mlp") {
      const mlp = new MLP({
        layerSizes: [2, 6, 4, 1],
        activations: ["tanh", "tanh", "sigmoid"],
        learningRate: 0.1
      });
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
        ], 380, 210);

        mlp.forward(new Tensor([1, 2], [1.5, -1.5]), true);
        traceOutput.innerHTML = mlp.tracer.steps.map(s => `• <b>${s.layerName}:</b> ${s.formula} → [${s.outputShape}]`).join("<br>");
      };

      stepBtn.onclick = () => {
        for (let e = 0; e < 20; e++) mlp.trainStep(X, Y);
        updateUI();
      };
      resetBtn.onclick = () => this.initModel("mlp");
      updateUI();
    }
    // --- 2. Recurrent LSTM & Gates (W2) ---
    else if (type === "lstm") {
      const lstm = new LSTMModel(4, 4);
      const tokens = ["Natural", "Language", "Processing", "Recurrent"];
      const seq = tokens.map(t => ({ token: t, vector: [0.5, -0.2, 0.8, -0.1] }));
      let stepIdx = 0;

      const updateUI = () => {
        const history = lstm.unroll(seq);
        diagramHost.innerHTML = SVGDiagramRenderer.renderNetwork([
          { id: "in", name: "x_t Token Vector", type: "dense", inShape: [4], outShape: [4], paramsCount: 0 },
          { id: "gates", name: "Gates [f, i, c, o]", type: "lstm_cell", inShape: [4], outShape: [16], paramsCount: 144 },
          { id: "cell", name: "c_t Memory Cell", type: "lstm_cell", inShape: [4], outShape: [4], paramsCount: 0 },
          { id: "h", name: "h_t Hidden State", type: "lstm_cell", inShape: [4], outShape: [4], paramsCount: 0 }
        ], 380, 210);

        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.fillStyle = "#020617";
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          ctx.fillStyle = "#38BDF8";
          ctx.font = "12px monospace";
          ctx.fillText("LSTM GATE TRANSITION DYNAMICS:", 15, 25);

          history.forEach((st, idx) => {
            const isCur = idx === stepIdx % history.length;
            ctx.fillStyle = isCur ? "#FFD700" : "#94A3B8";
            ctx.fillText(
              `t=${st.t} ("${st.inputToken}"): f=${st.f_gate[0].toFixed(2)}, i=${st.i_gate[0].toFixed(2)}, o=${st.o_gate[0].toFixed(2)} ${isCur ? "◀" : ""}`,
              15, 55 + idx * 32
            );
          });
        }
        traceOutput.innerHTML = lstm.tracer.steps.map(s => `• <b>${s.layerName}:</b> ${s.formula} → ${s.pedagogicalInsight}`).join("<br>");
      };

      stepBtn.onclick = () => { stepIdx++; updateUI(); };
      resetBtn.onclick = () => { stepIdx = 0; updateUI(); };
      updateUI();
    }
    // --- 3. Conv2D & MNIST (W3 / DL W5) ---
    else if (type === "cnn") {
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
      ], 380, 210);

      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.fillStyle = "#020617";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#38BDF8";
        ctx.font = "14px monospace";
        ctx.fillText(`Top Predicted Digit: ${res.predictedDigit}`, 20, 40);
        ctx.fillText(`Softmax Confidence: ${(res.probabilities[res.predictedDigit] * 100).toFixed(1)}%`, 20, 68);
        ctx.fillText(`Conv1 Output Tensor: 26×26×32`, 20, 105);
        ctx.fillText(`Layer 1 Parameter Count: 320`, 20, 135);
        ctx.fillStyle = "#94A3B8";
        ctx.fillText(`(3×3×1 + 1) × 32 = 320 params`, 20, 160);
      }
      traceOutput.innerHTML = cnn.tracer.steps.map(s => `• <b>${s.layerName}:</b> ${s.formula} → ${s.pedagogicalInsight}`).join("<br>");
      stepBtn.onclick = () => {
        // Shift line to simulate digit variation
        const offset = Math.floor(Math.random() * 6) - 3;
        const img = Tensor.zeros([28, 28, 1]);
        for (let r = 4; r < 24; r++) img.set(1.0, r, Math.min(27, Math.max(0, 14 + offset)), 0);
        const r = cnn.forward(img, true);
        if (ctx) {
          ctx.fillStyle = "#020617";
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          ctx.fillStyle = "#38BDF8";
          ctx.fillText(`Top Predicted Digit: ${r.predictedDigit}`, 20, 40);
          ctx.fillText(`Confidence: ${(r.probabilities[r.predictedDigit] * 100).toFixed(1)}%`, 20, 68);
          ctx.fillText(`Conv1 Output Tensor: 26×26×32`, 20, 105);
          ctx.fillText(`Params = (3·3·1 + 1)·32 = 320`, 20, 135);
        }
      };
      resetBtn.onclick = () => this.initModel("cnn");
    }
    // --- 4. N-grams & Bag of Words (W4) ---
    else if (type === "ngrams") {
      const vocab = ["the", "neural", "network", "processes", "language", "<UNK>"];
      const counts: Record<string, number> = { "the": 4, "neural": 3, "network": 3, "processes": 2, "language": 2, "<UNK>": 1 };

      diagramHost.innerHTML = SVGDiagramRenderer.renderNetwork([
        { id: "tokens", name: "Text Tokens", type: "embedding", inShape: [1], outShape: [6], paramsCount: 0 },
        { id: "count", name: "1-Gram Counter", type: "dense", inShape: [6], outShape: [6], paramsCount: 0 },
        { id: "oov", name: "<UNK> Firewall", type: "dense", inShape: [6], outShape: [6], paramsCount: 0 }
      ], 380, 210);

      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.fillStyle = "#020617";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#38BDF8";
        ctx.font = "12px monospace";
        ctx.fillText("VOCABULARY & BoW DISTRIBUTION:", 15, 25);
        vocab.forEach((v, idx) => {
          const isOov = v === "<UNK>";
          ctx.fillStyle = isOov ? "#C8102E" : "#0F8B8D";
          ctx.fillRect(15, 45 + idx * 30, counts[v] * 35, 18);
          ctx.fillStyle = "#FFF";
          ctx.fillText(`${v}: ${counts[v]}`, 20, 58 + idx * 30);
        });
      }
      traceOutput.innerHTML = `• <b>Vocabulary Size:</b> ${vocab.length} unique terms.<br>• <b>OOV Firewall:</b> Unseen test words route to <code>&lt;UNK&gt;</code> without corrupting vector length.<br>• <b>Formula:</b> BoW_vector[i] = count(word_i ∈ doc).`;
      stepBtn.onclick = () => { counts["<UNK>"]++; this.initModel("ngrams"); };
      resetBtn.onclick = () => { counts["<UNK>"] = 1; this.initModel("ngrams"); };
    }
    // --- 5. Word Embeddings / Word2Vec (W5) ---
    else if (type === "embeddings") {
      const words = [
        { w: "king", x: 1.8, y: 1.5 },
        { w: "queen", x: 1.7, y: -0.5 },
        { w: "man", x: -1.2, y: 1.4 },
        { w: "woman", x: -1.3, y: -0.6 },
        { w: "apple", x: 0.1, y: -2.2 }
      ];

      diagramHost.innerHTML = SVGDiagramRenderer.renderNetwork([
        { id: "onehot", name: "One-Hot Index", type: "dense", inShape: [10000], outShape: [300], paramsCount: 3000000 },
        { id: "proj", name: "2D t-SNE / PCA", type: "dense", inShape: [300], outShape: [2], paramsCount: 600 }
      ], 380, 210);

      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.fillStyle = "#020617";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        // Draw coordinate axes
        ctx.strokeStyle = "#334155";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(canvas.width / 2, 0); ctx.lineTo(canvas.width / 2, canvas.height);
        ctx.moveTo(0, canvas.height / 2); ctx.lineTo(canvas.width, canvas.height / 2);
        ctx.stroke();

        words.forEach(p => {
          const px = canvas.width / 2 + p.x * 45;
          const py = canvas.height / 2 - p.y * 45;
          ctx.beginPath();
          ctx.arc(px, py, 5, 0, Math.PI * 2);
          ctx.fillStyle = "#0F8B8D";
          ctx.fill();
          ctx.fillStyle = "#FFF";
          ctx.font = "11px sans-serif";
          ctx.fillText(p.w, px + 8, py + 4);
        });

        // Analogy arrow king -> queen
        ctx.strokeStyle = "#FFD700";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(canvas.width / 2 + 1.8 * 45, canvas.height / 2 - 1.5 * 45);
        ctx.lineTo(canvas.width / 2 + 1.7 * 45, canvas.height / 2 - (-0.5) * 45);
        ctx.stroke();
      }
      traceOutput.innerHTML = `• <b>Vector Analogy:</b> <code>vec("king") - vec("man") + vec("woman") ≈ vec("queen")</code><br>• <b>Cosine Similarity:</b> cos(θ) = (u · v) / (||u|| ||v||) captures geometric semantic alignment.`;
    }
    // --- 6. Autoencoders & VAE (W6 / DL W6) ---
    else if (type === "autoencoder") {
      const ae = new AutoencoderModel(10, 2);
      const dummyX = Tensor.random([1, 10], 0, 1);
      const res = ae.forward(dummyX, true);

      diagramHost.innerHTML = SVGDiagramRenderer.renderNetwork([
        { id: "in", name: "Input Features [10]", type: "dense", inShape: [10], outShape: [16], paramsCount: 160 },
        { id: "z", name: "Latent Bottleneck [2]", type: "latent_space", inShape: [16], outShape: [2], paramsCount: 32 },
        { id: "out", name: "Reconstruction [10]", type: "dense", inShape: [2], outShape: [10], paramsCount: 160 }
      ], 380, 210);

      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.fillStyle = "#020617";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#38BDF8";
        ctx.font = "13px monospace";
        ctx.fillText("2D LATENT MANIFOLD PROJECTION:", 20, 35);
        ctx.fillText(`z = [${res.latent.data[0].toFixed(3)}, ${res.latent.data[1].toFixed(3)}]`, 20, 65);
        ctx.fillText(`Reconstruction Loss (MSE): ${res.mse.toFixed(4)}`, 20, 95);
        ctx.fillStyle = "#FFD700";
        ctx.fillRect(canvas.width / 2 + res.latent.data[0] * 30, canvas.height / 2 + res.latent.data[1] * 30, 8, 8);
      }
      traceOutput.innerHTML = ae.tracer.steps.map(s => `• <b>${s.layerName}:</b> ${s.formula} → ${s.pedagogicalInsight}`).join("<br>");
    }
    // --- 7. Topic Modeling (LDA) (W7) ---
    else if (type === "lda") {
      diagramHost.innerHTML = SVGDiagramRenderer.renderNetwork([
        { id: "alpha", name: "Dirichlet Prior α", type: "dense", inShape: [1], outShape: [3], paramsCount: 0 },
        { id: "topics", name: "Topic Mixtures θ", type: "dense", inShape: [3], outShape: [5], paramsCount: 0 },
        { id: "words", name: "Word Distribution β", type: "dense", inShape: [5], outShape: [100], paramsCount: 0 }
      ], 380, 210);

      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.fillStyle = "#020617";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#38BDF8";
        ctx.font = "12px monospace";
        ctx.fillText("LATENT DIRICHLET ALLOCATION:", 15, 25);
        ctx.fillStyle = "#0F8B8D";
        ctx.fillText("Topic 1 (Tech): neural (0.42), data (0.31), compute (0.22)", 15, 60);
        ctx.fillStyle = "#6B2D7B";
        ctx.fillText("Topic 2 (Finance): return (0.38), risk (0.29), market (0.24)", 15, 95);
        ctx.fillStyle = "#D98E04";
        ctx.fillText("Topic 3 (Health): patient (0.45), trial (0.28), clinic (0.19)", 15, 130);
      }
      traceOutput.innerHTML = `• <b>Generative Story:</b> For each document $d$, sample $\\theta_d \\sim \\text{Dir}(\\alpha)$. For each word token $w_n$, sample topic $z_n \\sim \\text{Multinomial}(\\theta_d)$ and emit word from $\\beta_{z_n}$.`;
    }
    // --- 8. Structural Topic Models (W8) ---
    else if (type === "stm") {
      diagramHost.innerHTML = SVGDiagramRenderer.renderNetwork([
        { id: "cov", name: "Document Covariates X", type: "dense", inShape: [3], outShape: [3], paramsCount: 0 },
        { id: "prev", name: "Topic Prevalence Γ", type: "dense", inShape: [3], outShape: [4], paramsCount: 12 },
        { id: "words", name: "Content Covariates Y", type: "dense", inShape: [4], outShape: [100], paramsCount: 400 }
      ], 380, 210);

      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.fillStyle = "#020617";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#38BDF8";
        ctx.font = "12px monospace";
        ctx.fillText("STRUCTURAL TOPIC MODEL (STM):", 15, 25);
        ctx.fillStyle = "#E2E8F0";
        ctx.fillText("Topic Prevalence ~ X * Gamma (Author Party, Date)", 15, 60);
        ctx.fillText("Topic Content ~ Y * Kappa (Vocabulary Shifts)", 15, 90);
      }
      traceOutput.innerHTML = `• <b>Key Advantage:</b> Explicitly integrates document-level metadata (covariates) into topic prevalence and content.`;
    }
    // --- 9. Classification & Regularization (W9 / DL W2) ---
    else if (type === "classification") {
      diagramHost.innerHTML = SVGDiagramRenderer.renderNetwork([
        { id: "in", name: "Text Features", type: "dense", inShape: [500], outShape: [64], paramsCount: 32000 },
        { id: "drop", name: "Dropout (p=0.5)", type: "dense", inShape: [64], outShape: [64], paramsCount: 0 },
        { id: "out", name: "Softmax Classes", type: "dense", inShape: [64], outShape: [4], paramsCount: 256 }
      ], 380, 210);

      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.fillStyle = "#020617";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#38BDF8";
        ctx.font = "12px monospace";
        ctx.fillText("CLASSIFICATION PROBABILITIES:", 15, 30);
        const classes = [
          { name: "Politics", p: 0.72 },
          { name: "Sports", p: 0.14 },
          { name: "Sci/Tech", p: 0.09 },
          { name: "Business", p: 0.05 }
        ];
        classes.forEach((c, idx) => {
          ctx.fillStyle = "#0F8B8D";
          ctx.fillRect(15, 55 + idx * 35, c.p * 220, 20);
          ctx.fillStyle = "#FFF";
          ctx.fillText(`${c.name}: ${(c.p * 100).toFixed(0)}%`, 25, 70 + idx * 35);
        });
      }
      traceOutput.innerHTML = `• <b>Dropout Layer:</b> Randomly zeroes 50% of activations during training to prevent co-adaptation.<br>• <b>Softmax Head:</b> Turns raw logits into well-calibrated class posterior probabilities.`;
    }
    // --- 10. NER Sequence Tagging (W10) ---
    else if (type === "ner") {
      diagramHost.innerHTML = SVGDiagramRenderer.renderNetwork([
        { id: "tok", name: "Tokens", type: "dense", inShape: [5], outShape: [64], paramsCount: 0 },
        { id: "bilstm", name: "BiLSTM (Fwd+Bwd)", type: "lstm_cell", inShape: [64], outShape: [128], paramsCount: 48000 },
        { id: "bio", name: "BIO Tag Logits", type: "dense", inShape: [128], outShape: [7], paramsCount: 896 }
      ], 380, 210);

      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.fillStyle = "#020617";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#38BDF8";
        ctx.font = "12px monospace";
        ctx.fillText("NAMED ENTITY RECOGNITION (BIO):", 15, 25);
        const tokens = [
          { t: "Barack", tag: "B-PER" },
          { t: "Obama", tag: "I-PER" },
          { t: "visited", tag: "O" },
          { t: "Harvard", tag: "B-ORG" },
          { t: "University", tag: "I-ORG" }
        ];
        tokens.forEach((tk, idx) => {
          ctx.fillStyle = tk.tag.startsWith("B") ? "#C8102E" : tk.tag.startsWith("I") ? "#D98E04" : "#475569";
          ctx.fillRect(15, 50 + idx * 35, 75, 22);
          ctx.fillStyle = "#FFF";
          ctx.fillText(tk.tag, 22, 66 + idx * 35);
          ctx.fillStyle = "#E2E8F0";
          ctx.fillText(`"${tk.t}"`, 105, 66 + idx * 35);
        });
      }
      traceOutput.innerHTML = `• <b>BIO Schema:</b> <code>B-</code> (Begin entity), <code>I-</code> (Inside entity), <code>O</code> (Outside entity).`;
    }
    // --- 11. GANs (W11 / DL W8) ---
    else if (type === "gan") {
      diagramHost.innerHTML = SVGDiagramRenderer.renderNetwork([
        { id: "noise", name: "Noise Vector z", type: "dense", inShape: [100], outShape: [128], paramsCount: 12800 },
        { id: "gen", name: "Generator G(z)", type: "dense", inShape: [128], outShape: [784], paramsCount: 100352 },
        { id: "disc", name: "Discriminator D(x)", type: "dense", inShape: [784], outShape: [1], paramsCount: 785 }
      ], 380, 210);

      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.fillStyle = "#020617";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#38BDF8";
        ctx.font = "12px monospace";
        ctx.fillText("MINIMAX GAME: min_G max_D V(D, G)", 15, 25);
        ctx.fillStyle = "#C8102E";
        ctx.fillText("Discriminator D(x) -> 1 (Real) vs 0 (Fake)", 15, 65);
        ctx.fillStyle = "#0F8B8D";
        ctx.fillText("Generator G(z) -> Fooled D(G(z)) -> 1", 15, 100);
      }
      traceOutput.innerHTML = `• <b>Zero-Sum Game Objective:</b> $\\min_G \\max_D \\mathbb{E}_{x}[\\log D(x)] + \\mathbb{E}_{z}[\\log(1 - D(G(z)))]$.`;
    }
    // --- 12. Transformers & Attention (W12 / DL W12) ---
    else if (type === "transformer") {
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
      ], 380, 210);

      traceOutput.innerHTML = transformer.tracer.steps.map(s => `• <b>${s.layerName}:</b> ${s.formula} → ${s.pedagogicalInsight}`).join("<br>");
      stepBtn.onclick = () => this.initModel("transformer");
    }
    // --- 13. Optimizers & Loss Landscapes (DL W3) ---
    else if (type === "optimizers") {
      diagramHost.innerHTML = SVGDiagramRenderer.renderNetwork([
        { id: "loss", name: "Loss L(w)", type: "dense", inShape: [2], outShape: [1], paramsCount: 0 },
        { id: "grad", name: "Gradient ∇L", type: "dense", inShape: [2], outShape: [2], paramsCount: 0 },
        { id: "mom", name: "Momentum v_t", type: "dense", inShape: [2], outShape: [2], paramsCount: 0 },
        { id: "adam", name: "Adam m_t / √v_t", type: "dense", inShape: [2], outShape: [2], paramsCount: 0 }
      ], 380, 210);

      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.fillStyle = "#020617";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#38BDF8";
        ctx.font = "12px monospace";
        ctx.fillText("OPTIMIZER TRAJECTORY RACE:", 15, 25);
        ctx.fillStyle = "#C8102E";
        ctx.fillText("● SGD: High oscillation in ravines", 15, 60);
        ctx.fillStyle = "#D98E04";
        ctx.fillText("● Momentum: Damps oscillations along ridges", 15, 95);
        ctx.fillStyle = "#0F8B8D";
        ctx.fillText("● Adam: Adaptive learning rates per parameter", 15, 130);
      }
      traceOutput.innerHTML = `• <b>Adam Update Rule:</b> $w_{t+1} = w_t - \\frac{\\alpha}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t$ combining first and second raw moments.`;
    }
  }
}

if (typeof window !== "undefined" && !customElements.get("neural-sim")) {
  customElements.define("neural-sim", NeuralSimElement);
}
