/**
 * OmniNeuralSim - High-Definition Interactive Canvas 2D Engine
 * SOTA Visualizations: CNN Kernel Sliding, LSTM Conveyor Belt,
 * Attention Flow Arcs, and 3D Optimization Landscapes.
 */

export class CanvasVisualizer {
  /**
   * Helper to set up HiDPI canvas for ultra-sharp Retina rendering
   */
  static setupHiDPI(canvas: HTMLCanvasElement): CanvasRenderingContext2D | null {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = rect.width || canvas.width;
    const height = rect.height || canvas.height;

    canvas.width = width * dpr;
    canvas.height = height * dpr;

    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.scale(dpr, dpr);
    }
    return ctx;
  }

  /**
   * 1. Decision Boundary & Active Gradient Heatmap
   */
  static renderDecisionBoundary(
    canvas: HTMLCanvasElement,
    grid: number[][],
    dataPoints?: { x: number; y: number; label: number }[]
  ): void {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rows = grid.length;
    const cols = grid[0].length;
    const cellW = canvas.width / cols;
    const cellH = canvas.height / rows;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const val = grid[r][c]; // 0.0 to 1.0
        const red = Math.floor(val * 220 + (1 - val) * 15);
        const green = Math.floor((1 - Math.abs(val - 0.5) * 2) * 160);
        const blue = Math.floor((1 - val) * 220 + val * 25);

        ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`;
        ctx.fillRect(c * cellW, r * cellH, cellW + 1, cellH + 1);
      }
    }

    if (dataPoints) {
      for (const p of dataPoints) {
        const px = ((p.x + 4) / 8) * canvas.width;
        const py = ((4 - p.y) / 8) * canvas.height;

        ctx.beginPath();
        ctx.arc(px, py, 6, 0, Math.PI * 2);
        ctx.fillStyle = p.label === 1 ? "#FFD700" : "#FFFFFF";
        ctx.strokeStyle = "#0B1329";
        ctx.lineWidth = 2;
        ctx.fill();
        ctx.stroke();
      }
    }
  }

  /**
   * 2. CNN Interactive 3x3 Sliding Kernel Visualizer (Stanford CS231n / CNN Explainer style)
   */
  static renderCNNKernelSlide(
    canvas: HTMLCanvasElement,
    imageGrid: number[][],
    kernelGrid: number[][],
    kernelPos: { row: number; col: number },
    bias = 0.05
  ): { activation: number; formulaText: string } {
    const ctx = canvas.getContext("2d");
    if (!ctx) return { activation: 0, formulaText: "" };

    ctx.fillStyle = "#020617";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const padLeft = 16;
    const padTop = 32;
    const gridPx = 140;
    const H = imageGrid.length;
    const W = imageGrid[0].length;
    const cellSz = gridPx / H;

    // Header label
    ctx.fillStyle = "#38BDF8";
    ctx.font = "bold 11px monospace";
    ctx.fillText("INPUT IMAGE (28×28)", padLeft, 20);

    // Draw Input Image pixels
    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        const val = imageGrid[r][c];
        const gray = Math.floor(val * 255);
        ctx.fillStyle = `rgb(${gray}, ${gray}, ${gray})`;
        ctx.fillRect(padLeft + c * cellSz, padTop + r * cellSz, cellSz - 0.5, cellSz - 0.5);
      }
    }

    // Highlight active 3x3 Receptive Field
    const kR = kernelPos.row;
    const kC = kernelPos.col;
    ctx.strokeStyle = "#FFD700";
    ctx.lineWidth = 2;
    ctx.strokeRect(padLeft + kC * cellSz, padTop + kR * cellSz, cellSz * 3, cellSz * 3);

    // Draw Magnified 3x3 Receptive Field calculation box
    const zoomX = 180;
    const zoomY = 32;
    ctx.fillStyle = "#0F172A";
    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 1;
    ctx.fillRect(zoomX, zoomY, 140, 140);
    ctx.strokeRect(zoomX, zoomY, 140, 140);

    ctx.fillStyle = "#FFD700";
    ctx.font = "bold 10px monospace";
    ctx.fillText("3×3 KERNEL MULTIPLY", zoomX + 6, zoomY + 14);

    let sum = bias;
    const boxSz = 36;
    for (let kr = 0; kr < 3; kr++) {
      for (let kc = 0; kc < 3; kc++) {
        const pixVal = imageGrid[kR + kr]?.[kC + kc] || 0;
        const wVal = kernelGrid[kr][kc];
        const prod = pixVal * wVal;
        sum += prod;

        const bx = zoomX + 8 + kc * (boxSz + 4);
        const by = zoomY + 22 + kr * (boxSz + 4);

        ctx.fillStyle = "#1E293B";
        ctx.fillRect(bx, by, boxSz, boxSz);
        ctx.strokeStyle = "#475569";
        ctx.strokeRect(bx, by, boxSz, boxSz);

        ctx.fillStyle = "#E2E8F0";
        ctx.font = "9px monospace";
        ctx.textAlign = "center";
        ctx.fillText(`x:${pixVal.toFixed(1)}`, bx + boxSz / 2, by + 14);
        ctx.fillStyle = "#38BDF8";
        ctx.fillText(`w:${wVal.toFixed(1)}`, bx + boxSz / 2, by + 28);
      }
    }

    ctx.textAlign = "left";
    const act = Math.max(0, sum); // ReLU

    // Summary calculation text
    ctx.fillStyle = "#F8FAFC";
    ctx.font = "11px monospace";
    ctx.fillText(`Linear Sum z = (Σ x_i·w_i) + b = ${sum.toFixed(3)}`, 16, 195);
    ctx.fillStyle = "#10B981";
    ctx.fillText(`Feature Map Pixel = ReLU(z) = ${act.toFixed(3)}`, 16, 215);

    // Arrow to output feature map
    ctx.fillStyle = "#94A3B8";
    ctx.font = "10px monospace";
    ctx.fillText(`Active Output Pixel: (${kR}, ${kC}) in 26×26 feature map`, 16, 235);

    return {
      activation: act,
      formulaText: `z = Σ x_i·w_i + ${bias.toFixed(2)} = ${sum.toFixed(3)} → ReLU = ${act.toFixed(3)}`
    };
  }

  /**
   * 3. LSTM Memory Conveyor Belt & 4-Gate Anatomy
   */
  static renderLSTMConveyorBelt(
    canvas: HTMLCanvasElement,
    t: number,
    token: string,
    gates: { f: number; i: number; c_tilde: number; o: number; c: number; h: number }
  ): void {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = "#020617";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = "#38BDF8";
    ctx.font = "bold 12px monospace";
    ctx.fillText(`LSTM CELL ANATOMY (Step t=${t}: "${token}")`, 16, 24);

    // 1. Draw Cell State Conveyor Belt (Horizontal top rail)
    ctx.strokeStyle = "#6B2D7B";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(20, 60); ctx.lineTo(canvas.width - 20, 60);
    ctx.stroke();

    ctx.fillStyle = "#FFD700";
    ctx.font = "10px monospace";
    ctx.fillText(`Cell State c_t = ${gates.c.toFixed(3)} (Memory Belt)`, 30, 50);

    // 2. Draw 4 Gate Bars with Gauges
    const gateDefs = [
      { name: "Forget (f_t)", val: gates.f, color: "#C8102E", desc: "0=Drop, 1=Keep" },
      { name: "Input (i_t)", val: gates.i, color: "#0F8B8D", desc: "Write weight" },
      { name: "Candidate (c̃_t)", val: gates.c_tilde, color: "#38BDF8", desc: "New candidate" },
      { name: "Output (o_t)", val: gates.o, color: "#10B981", desc: "Hidden filter" }
    ];

    const barW = 68;
    const startX = 20;

    gateDefs.forEach((g, idx) => {
      const gx = startX + idx * (barW + 12);
      const gy = 85;

      ctx.fillStyle = "#1E293B";
      ctx.fillRect(gx, gy, barW, 95);
      ctx.strokeStyle = "#334155";
      ctx.strokeRect(gx, gy, barW, 95);

      // Gauge level
      const fillH = Math.min(80, Math.max(5, Math.abs(g.val) * 80));
      ctx.fillStyle = g.color;
      ctx.fillRect(gx + 4, gy + 90 - fillH, barW - 8, fillH);

      ctx.fillStyle = "#FFF";
      ctx.font = "bold 9px sans-serif";
      ctx.fillText(g.name, gx + 5, gy + 15);

      ctx.fillStyle = "#FFD700";
      ctx.font = "bold 11px monospace";
      ctx.fillText(g.val.toFixed(2), gx + 16, gy + 55);
    });

    // 3. Hidden State Output
    ctx.fillStyle = "#F8FAFC";
    ctx.font = "11px monospace";
    ctx.fillText(`Hidden Emission h_t = o_t ⊙ tanh(c_t) = ${gates.h.toFixed(4)}`, 16, 210);
    ctx.fillStyle = "#94A3B8";
    ctx.font = "10px monospace";
    ctx.fillText(`Forget Gate: ${(gates.f * 100).toFixed(0)}% retention | Input: ${(gates.i * 100).toFixed(0)}% written`, 16, 230);
  }

  /**
   * 4. Scaled Dot-Product Attention Matrix with Causal Mask
   */
  static renderAttentionMatrix(
    canvas: HTMLCanvasElement,
    matrix: number[][],
    tokens: string[],
    activeQueryIdx = -1
  ): void {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const N = matrix.length;
    const padLeft = 65;
    const padTop = 50;
    const matrixW = canvas.width - padLeft - 15;
    const matrixH = canvas.height - padTop - 15;
    const cellW = matrixW / N;
    const cellH = matrixH / N;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#020617";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = "#38BDF8";
    ctx.font = "bold 11px monospace";
    ctx.fillText("ATTENTION WEIGHTS: Softmax(Q K^T / √d_k)", 15, 20);

    for (let i = 0; i < N; i++) {
      const isCurQuery = i === activeQueryIdx;
      for (let j = 0; j < N; j++) {
        const weight = matrix[i][j];
        const isCausalMasked = j > i;

        if (isCausalMasked) {
          ctx.fillStyle = "#0F172A";
        } else {
          ctx.fillStyle = isCurQuery
            ? `rgba(255, 215, 0, ${Math.max(0.15, weight)})`
            : `rgba(15, 139, 141, ${Math.max(0.1, weight)})`;
        }
        ctx.fillRect(padLeft + j * cellW, padTop + i * cellH, cellW - 1.5, cellH - 1.5);

        if (cellW > 28) {
          ctx.fillStyle = isCausalMasked ? "#334155" : weight > 0.4 ? "#FFFFFF" : "#94A3B8";
          ctx.font = "9px monospace";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(
            isCausalMasked ? "—" : weight.toFixed(2),
            padLeft + j * cellW + cellW / 2,
            padTop + i * cellH + cellH / 2
          );
        }
      }
    }

    // Token Labels
    ctx.fillStyle = "#E2E8F0";
    ctx.font = "11px sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";

    for (let i = 0; i < N; i++) {
      const token = tokens[i] || `t_${i}`;
      ctx.fillStyle = i === activeQueryIdx ? "#FFD700" : "#CBD5E1";
      ctx.fillText(token, padLeft - 6, padTop + i * cellH + cellH / 2);
    }

    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    for (let j = 0; j < N; j++) {
      const token = tokens[j] || `t_${j}`;
      ctx.fillText(token, padLeft + j * cellW + cellW / 2, padTop - 6);
    }
  }

  /**
   * 5. 3D Optimizer Loss Landscape (SGD vs Momentum vs Adam)
   */
  static renderOptimizerContour(
    canvas: HTMLCanvasElement,
    step: number
  ): void {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = "#020617";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = "#38BDF8";
    ctx.font = "bold 11px monospace";
    ctx.fillText("LOSS LANDSCAPE TRAJECTORY (Ravine Contour)", 15, 22);

    const cx = canvas.width / 2;
    const cy = canvas.height / 2 + 10;

    // Draw elliptical ravine contours
    for (let r = 1; r <= 6; r++) {
      ctx.strokeStyle = `rgba(15, 139, 141, ${0.12 * r})`;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.ellipse(cx, cy, r * 25, r * 12, -Math.PI / 6, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Trajectory curves
    const progress = Math.min(1.0, (step % 20) / 19);

    // 1. SGD (Oscillating aggressively across walls)
    ctx.strokeStyle = "#C8102E";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx - 100, cy - 60);
    const sgdPoints = 8;
    for (let i = 1; i <= sgdPoints * progress; i++) {
      const osc = (i % 2 === 0 ? 30 : -30) * (1 - i / sgdPoints);
      ctx.lineTo(cx - 100 + i * 14, cy - 60 + i * 8 + osc);
    }
    ctx.stroke();

    // 2. Momentum (Damped oscillations, moving along ridge)
    ctx.strokeStyle = "#D98E04";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx - 100, cy - 60);
    for (let i = 1; i <= sgdPoints * progress; i++) {
      const osc = (i % 2 === 0 ? 12 : -12) * (1 - i / sgdPoints);
      ctx.lineTo(cx - 100 + i * 15, cy - 60 + i * 8 + osc);
    }
    ctx.stroke();

    // 3. Adam (Adaptive per-parameter learning rate, fast descent to optimum)
    ctx.strokeStyle = "#10B981";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(cx - 100, cy - 60);
    for (let i = 1; i <= sgdPoints * progress; i++) {
      ctx.lineTo(cx - 100 + i * 16, cy - 60 + i * 8.5);
    }
    ctx.stroke();

    // Legend
    ctx.font = "10px monospace";
    ctx.fillStyle = "#C8102E"; ctx.fillText("● SGD (High oscillation)", 15, 195);
    ctx.fillStyle = "#D98E04"; ctx.fillText("● Momentum (Damped velocity)", 15, 212);
    ctx.fillStyle = "#10B981"; ctx.fillText("● Adam (Fast adaptive convergence)", 15, 230);
  }
}
