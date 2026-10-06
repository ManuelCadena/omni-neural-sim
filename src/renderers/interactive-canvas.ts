/**
 * OmniNeuralSim - High-Definition Interactive Canvas 2D Engine
 * SOTA Visualizations: CNN Kernel Sliding, LSTM Conveyor Belt,
 * Attention Flow Arcs, Latent Space Clusters & 3D Loss Landscapes.
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

    if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
      canvas.width = width * dpr;
      canvas.height = height * dpr;
    }

    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
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

    const width = canvas.width;
    const height = canvas.height;
    const rows = grid.length;
    const cols = grid[0].length;
    const cellW = width / cols;
    const cellH = height / rows;

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
        const px = ((p.x + 4) / 8) * width;
        const py = ((4 - p.y) / 8) * height;

        ctx.beginPath();
        ctx.arc(px, py, 7, 0, Math.PI * 2);
        ctx.fillStyle = p.label === 1 ? "#FFD700" : "#FFFFFF";
        ctx.strokeStyle = "#0B1329";
        ctx.lineWidth = 2.5;
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

    const width = canvas.width;
    const height = canvas.height;

    ctx.fillStyle = "#020617";
    ctx.fillRect(0, 0, width, height);

    const padLeft = 20;
    const padTop = 38;
    const gridPx = 180;
    const H = imageGrid.length;
    const W = imageGrid[0].length;
    const cellSz = gridPx / H;

    // Header label
    ctx.fillStyle = "#38BDF8";
    ctx.font = "bold 12px monospace";
    ctx.fillText("INPUT DIGIT PIXELS (28×28)", padLeft, 22);

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
    ctx.lineWidth = 2.5;
    ctx.strokeRect(padLeft + kC * cellSz, padTop + kR * cellSz, cellSz * 3, cellSz * 3);

    // Draw Magnified 3x3 Receptive Field calculation box
    const zoomX = 225;
    const zoomY = 38;
    const zoomW = 240;
    const zoomH = 180;
    ctx.fillStyle = "#0F172A";
    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 1;
    ctx.fillRect(zoomX, zoomY, zoomW, zoomH);
    ctx.strokeRect(zoomX, zoomY, zoomW, zoomH);

    ctx.fillStyle = "#FFD700";
    ctx.font = "bold 11px monospace";
    ctx.fillText("3×3 RECEPTIVE FIELD CALCULATION", zoomX + 10, zoomY + 18);

    let sum = bias;
    const boxSz = 48;
    for (let kr = 0; kr < 3; kr++) {
      for (let kc = 0; kc < 3; kc++) {
        const pixVal = imageGrid[kR + kr]?.[kC + kc] || 0;
        const wVal = kernelGrid[kr][kc];
        const prod = pixVal * wVal;
        sum += prod;

        const bx = zoomX + 12 + kc * (boxSz + 6);
        const by = zoomY + 28 + kr * (boxSz + 4);

        ctx.fillStyle = "#1E293B";
        ctx.fillRect(bx, by, boxSz, boxSz);
        ctx.strokeStyle = "#475569";
        ctx.strokeRect(bx, by, boxSz, boxSz);

        ctx.fillStyle = "#E2E8F0";
        ctx.font = "10px monospace";
        ctx.textAlign = "center";
        ctx.fillText(`x:${pixVal.toFixed(1)}`, bx + boxSz / 2, by + 16);
        ctx.fillStyle = "#38BDF8";
        ctx.fillText(`w:${wVal.toFixed(1)}`, bx + boxSz / 2, by + 34);
      }
    }

    ctx.textAlign = "left";
    const act = Math.max(0, sum); // ReLU

    // Summary calculation text
    ctx.fillStyle = "#F8FAFC";
    ctx.font = "12px monospace";
    ctx.fillText(`Linear Sum z = Σ(x_i·w_i) + b = ${sum.toFixed(3)}`, 20, 245);
    ctx.fillStyle = "#10B981";
    ctx.fillText(`Feature Activation = ReLU(z) = ${act.toFixed(3)}`, 20, 270);

    ctx.fillStyle = "#94A3B8";
    ctx.font = "11px monospace";
    ctx.fillText(`Output location: (${kR}, ${kC}) in 26×26 feature map`, 20, 295);

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

    const width = canvas.width;
    const height = canvas.height;

    ctx.fillStyle = "#020617";
    ctx.fillRect(0, 0, width, height);

    ctx.fillStyle = "#38BDF8";
    ctx.font = "bold 13px monospace";
    ctx.fillText(`LSTM RECURRENT CELL ANATOMY (Step t=${t}: "${token}")`, 20, 28);

    // 1. Draw Cell State Conveyor Belt
    ctx.strokeStyle = "#6B2D7B";
    ctx.lineWidth = 5;
    ctx.beginPath();
    ctx.moveTo(20, 70); ctx.lineTo(width - 20, 70);
    ctx.stroke();

    ctx.fillStyle = "#FFD700";
    ctx.font = "12px monospace";
    ctx.fillText(`Cell State c_t = ${gates.c.toFixed(3)} (Constant Error Carousel)`, 30, 58);

    // 2. Draw 4 Gate Bars with Gauges
    const gateDefs = [
      { name: "Forget (f_t)", val: gates.f, color: "#C8102E", desc: "Retention" },
      { name: "Input (i_t)", val: gates.i, color: "#0F8B8D", desc: "Write weight" },
      { name: "Candidate (c̃_t)", val: gates.c_tilde, color: "#38BDF8", desc: "New signal" },
      { name: "Output (o_t)", val: gates.o, color: "#10B981", desc: "Exposure" }
    ];

    const barW = Math.min(100, (width - 60) / 4);
    const startX = 20;

    gateDefs.forEach((g, idx) => {
      const gx = startX + idx * (barW + 12);
      const gy = 100;
      const gh = 130;

      ctx.fillStyle = "#0F172A";
      ctx.fillRect(gx, gy, barW, gh);
      ctx.strokeStyle = "#334155";
      ctx.strokeRect(gx, gy, barW, gh);

      // Gauge level
      const fillH = Math.min(gh - 25, Math.max(6, Math.abs(g.val) * (gh - 25)));
      ctx.fillStyle = g.color;
      ctx.fillRect(gx + 5, gy + gh - fillH - 5, barW - 10, fillH);

      ctx.fillStyle = "#FFF";
      ctx.font = "bold 10.5px sans-serif";
      ctx.fillText(g.name, gx + 6, gy + 18);

      ctx.fillStyle = "#FFD700";
      ctx.font = "bold 13px monospace";
      ctx.fillText(g.val.toFixed(2), gx + 15, gy + 75);
    });

    // 3. Hidden State Output
    ctx.fillStyle = "#F8FAFC";
    ctx.font = "12px monospace";
    ctx.fillText(`Hidden State: h_t = o_t ⊙ tanh(c_t) = ${gates.h.toFixed(4)}`, 20, 265);
    ctx.fillStyle = "#94A3B8";
    ctx.font = "11px monospace";
    ctx.fillText(`Retention: ${(gates.f * 100).toFixed(0)}% prior memory retained | ${(gates.i * 100).toFixed(0)}% new candidate injected`, 20, 290);
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

    const width = canvas.width;
    const height = canvas.height;
    const N = matrix.length;
    const padLeft = 90;
    const padTop = 60;
    const matrixW = width - padLeft - 20;
    const matrixH = height - padTop - 20;
    const cellW = matrixW / N;
    const cellH = matrixH / N;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#020617";
    ctx.fillRect(0, 0, width, height);

    ctx.fillStyle = "#38BDF8";
    ctx.font = "bold 13px monospace";
    ctx.fillText("ATTENTION WEIGHTS: Softmax(Q K^T / √d_k)", 20, 26);

    for (let i = 0; i < N; i++) {
      const isCurQuery = i === activeQueryIdx;
      for (let j = 0; j < N; j++) {
        const weight = matrix[i][j];
        const isCausalMasked = j > i;

        if (isCausalMasked) {
          ctx.fillStyle = "#0F172A";
        } else {
          ctx.fillStyle = isCurQuery
            ? `rgba(255, 215, 0, ${Math.max(0.2, weight)})`
            : `rgba(15, 139, 141, ${Math.max(0.12, weight)})`;
        }
        ctx.fillRect(padLeft + j * cellW, padTop + i * cellH, cellW - 2, cellH - 2);

        if (cellW > 28) {
          ctx.fillStyle = isCausalMasked ? "#334155" : weight > 0.4 ? "#FFFFFF" : "#94A3B8";
          ctx.font = "10px monospace";
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
    ctx.font = "12px sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";

    for (let i = 0; i < N; i++) {
      const token = tokens[i] || `t_${i}`;
      ctx.fillStyle = i === activeQueryIdx ? "#FFD700" : "#CBD5E1";
      ctx.fillText(token, padLeft - 10, padTop + i * cellH + cellH / 2);
    }

    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    for (let j = 0; j < N; j++) {
      const token = tokens[j] || `t_${j}`;
      ctx.fillText(token, padLeft + j * cellW + cellW / 2, padTop - 8);
    }
  }

  /**
   * 5. 2D Latent Space Manifold & Feature Reconstruction (Autoencoders / VAE)
   */
  static renderLatentManifold(
    canvas: HTMLCanvasElement,
    latentZ: [number, number],
    originalX: number[],
    reconstructedX: number[],
    mse: number
  ): void {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    ctx.fillStyle = "#020617";
    ctx.fillRect(0, 0, width, height);

    ctx.fillStyle = "#38BDF8";
    ctx.font = "bold 13px monospace";
    ctx.fillText("2D LATENT SPACE & FEATURE RECONSTRUCTION", 20, 24);

    // Left half: 2D Latent Manifold Grid
    const gridW = 200;
    const gridH = 200;
    const gLeft = 20;
    const gTop = 45;
    const cx = gLeft + gridW / 2;
    const cy = gTop + gridH / 2;

    ctx.fillStyle = "#0F172A";
    ctx.fillRect(gLeft, gTop, gridW, gridH);
    ctx.strokeStyle = "#334155";
    ctx.strokeRect(gLeft, gTop, gridW, gridH);

    // Grid lines
    ctx.strokeStyle = "#1E293B";
    for (let d = -80; d <= 80; d += 25) {
      ctx.beginPath(); ctx.moveTo(cx + d, gTop); ctx.lineTo(cx + d, gTop + gridH); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(gLeft, cy + d); ctx.lineTo(gLeft + gridW, cy + d); ctx.stroke();
    }

    // Synthetic cluster clouds in latent space
    const clusters = [
      { x: cx - 40, y: cy - 35, col: "#0F8B8D" },
      { x: cx + 45, y: cy + 40, col: "#6B2D7B" },
      { x: cx + 35, y: cy - 45, col: "#38BDF8" }
    ];
    clusters.forEach(c => {
      ctx.fillStyle = c.col;
      for (let k = 0; k < 6; k++) {
        const ox = (Math.sin(k * 1.5) * 20);
        const oy = (Math.cos(k * 1.5) * 18);
        ctx.beginPath();
        ctx.arc(c.x + ox, c.y + oy, 3, 0, Math.PI * 2);
        ctx.fill();
      }
    });

    // Active Latent Vector z point
    const pzX = cx + latentZ[0] * 40;
    const pzY = cy - latentZ[1] * 40;
    ctx.beginPath();
    ctx.arc(pzX, pzY, 7, 0, Math.PI * 2);
    ctx.fillStyle = "#FFD700";
    ctx.fill();
    ctx.strokeStyle = "#FFFFFF";
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = "#FFD700";
    ctx.font = "bold 11px monospace";
    ctx.fillText(`z = (${latentZ[0].toFixed(2)}, ${latentZ[1].toFixed(2)})`, gLeft + 10, gTop + gridH + 18);

    // Right half: Feature Comparison Bars (Input vs Reconstructed)
    const barStartX = 245;
    const barStartY = 45;
    const barW = width - barStartX - 25;
    const dimCount = Math.min(8, originalX.length);
    const rowH = 22;

    ctx.fillStyle = "#E2E8F0";
    ctx.font = "bold 11px sans-serif";
    ctx.fillText("Feature Reconstruction Comparison:", barStartX, barStartY - 8);

    for (let i = 0; i < dimCount; i++) {
      const yPos = barStartY + i * rowH;
      const inVal = originalX[i];
      const recVal = reconstructedX[i];

      ctx.fillStyle = "#94A3B8";
      ctx.font = "10px monospace";
      ctx.fillText(`dim ${i}:`, barStartX, yPos + 10);

      // Input bar (Teal)
      ctx.fillStyle = "#0F8B8D";
      ctx.fillRect(barStartX + 42, yPos, Math.max(3, inVal * (barW - 45)), 7);

      // Reconstructed bar (Gold)
      ctx.fillStyle = "#FFD700";
      ctx.fillRect(barStartX + 42, yPos + 9, Math.max(3, recVal * (barW - 45)), 7);
    }

    // Legend & MSE metric
    ctx.font = "10px sans-serif";
    ctx.fillStyle = "#0F8B8D"; ctx.fillRect(barStartX, height - 42, 10, 10);
    ctx.fillStyle = "#E2E8F0"; ctx.fillText("Original x", barStartX + 16, height - 33);

    ctx.fillStyle = "#FFD700"; ctx.fillRect(barStartX + 85, height - 42, 10, 10);
    ctx.fillStyle = "#E2E8F0"; ctx.fillText("Reconstructed x̂", barStartX + 101, height - 33);

    ctx.fillStyle = "#10B981";
    ctx.font = "bold 12px monospace";
    ctx.fillText(`Reconstruction MSE: ${mse.toFixed(4)}`, barStartX, height - 14);
  }

  /**
   * 6. 3D Optimizer Loss Landscape (SGD vs Momentum vs Adam)
   */
  static renderOptimizerContour(
    canvas: HTMLCanvasElement,
    step: number
  ): void {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    ctx.fillStyle = "#020617";
    ctx.fillRect(0, 0, width, height);

    ctx.fillStyle = "#38BDF8";
    ctx.font = "bold 13px monospace";
    ctx.fillText("LOSS LANDSCAPE TRAJECTORY (3D Ravine Contour)", 20, 26);

    const cx = width / 2;
    const cy = height / 2 + 10;

    // Draw elliptical ravine contours
    for (let r = 1; r <= 6; r++) {
      ctx.strokeStyle = `rgba(15, 139, 141, ${0.12 * r})`;
      ctx.lineWidth = 1.3;
      ctx.beginPath();
      ctx.ellipse(cx, cy, r * 32, r * 16, -Math.PI / 6, 0, Math.PI * 2);
      ctx.stroke();
    }

    const progress = Math.min(1.0, (step % 20) / 19);
    const sgdPoints = 9;

    // 1. SGD (Oscillating across steep ravine walls)
    ctx.strokeStyle = "#C8102E";
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.moveTo(cx - 130, cy - 80);
    for (let i = 1; i <= sgdPoints * progress; i++) {
      const osc = (i % 2 === 0 ? 38 : -38) * (1 - i / sgdPoints);
      ctx.lineTo(cx - 130 + i * 18, cy - 80 + i * 10 + osc);
    }
    ctx.stroke();

    // 2. Momentum (Damped oscillations along bottom)
    ctx.strokeStyle = "#D98E04";
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.moveTo(cx - 130, cy - 80);
    for (let i = 1; i <= sgdPoints * progress; i++) {
      const osc = (i % 2 === 0 ? 14 : -14) * (1 - i / sgdPoints);
      ctx.lineTo(cx - 130 + i * 19, cy - 80 + i * 10 + osc);
    }
    ctx.stroke();

    // 3. Adam (Adaptive per-parameter learning rates)
    ctx.strokeStyle = "#10B981";
    ctx.lineWidth = 2.8;
    ctx.beginPath();
    ctx.moveTo(cx - 130, cy - 80);
    for (let i = 1; i <= sgdPoints * progress; i++) {
      ctx.lineTo(cx - 130 + i * 20, cy - 80 + i * 11);
    }
    ctx.stroke();

    // Legend
    ctx.font = "11px monospace";
    ctx.fillStyle = "#C8102E"; ctx.fillText("● SGD: Oscillates on steep walls", 20, height - 55);
    ctx.fillStyle = "#D98E04"; ctx.fillText("● Momentum: Dampens ravine oscillations", 20, height - 35);
    ctx.fillStyle = "#10B981"; ctx.fillText("● Adam: Direct adaptive descent to global minimum", 20, height - 15);
  }
}
