/**
 * OmniNeuralSim - Interactive Canvas 2D Visualizer
 * High-performance 2D canvas routines for decision boundaries,
 * MNIST interactive drawing, and Attention heatmaps.
 */

export class CanvasVisualizer {
  /**
   * Render a 2D scalar grid as a color heatmap (Decision Boundary)
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
        // Interpolate Navy (#14213D) -> Teal (#0F8B8D) -> Crimson (#C8102E)
        const red = Math.floor(val * 200 + (1 - val) * 20);
        const green = Math.floor((1 - Math.abs(val - 0.5) * 2) * 150);
        const blue = Math.floor((1 - val) * 200 + val * 30);

        ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`;
        ctx.fillRect(c * cellW, r * cellH, cellW + 1, cellH + 1);
      }
    }

    // Overlay dataset points
    if (dataPoints) {
      for (const p of dataPoints) {
        // Map [-4, 4] to canvas coords
        const px = ((p.x + 4) / 8) * canvas.width;
        const py = ((4 - p.y) / 8) * canvas.height;

        ctx.beginPath();
        ctx.arc(px, py, 5, 0, Math.PI * 2);
        ctx.fillStyle = p.label === 1 ? "#FFD700" : "#FFFFFF";
        ctx.strokeStyle = "#000000";
        ctx.lineWidth = 1.5;
        ctx.fill();
        ctx.stroke();
      }
    }
  }

  /**
   * Render Attention Weight Matrix [seqLen x seqLen] with token labels
   */
  static renderAttentionMatrix(
    canvas: HTMLCanvasElement,
    matrix: number[][],
    tokens: string[]
  ): void {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const N = matrix.length;
    const padLeft = 70;
    const padTop = 70;
    const matrixW = canvas.width - padLeft - 10;
    const matrixH = canvas.height - padTop - 10;
    const cellW = matrixW / N;
    const cellH = matrixH / N;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#0F172A";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw Heatmap Cells
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const weight = matrix[i][j]; // 0 to 1
        // Teal intensity
        ctx.fillStyle = `rgba(15, 139, 141, ${Math.max(0.08, weight)})`;
        ctx.fillRect(padLeft + j * cellW, padTop + i * cellH, cellW - 1, cellH - 1);

        if (cellW > 30) {
          ctx.fillStyle = weight > 0.4 ? "#FFFFFF" : "#94A3B8";
          ctx.font = "10px monospace";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(
            weight.toFixed(2),
            padLeft + j * cellW + cellW / 2,
            padTop + i * cellH + cellH / 2
          );
        }
      }
    }

    // Draw Labels
    ctx.fillStyle = "#E2E8F0";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";

    for (let i = 0; i < N; i++) {
      const token = tokens[i] || `t_${i}`;
      ctx.fillText(token, padLeft - 8, padTop + i * cellH + cellH / 2);
    }

    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    for (let j = 0; j < N; j++) {
      const token = tokens[j] || `t_${j}`;
      ctx.fillText(token, padLeft + j * cellW + cellW / 2, padTop - 8);
    }
  }
}
