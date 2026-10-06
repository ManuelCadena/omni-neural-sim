/**
 * OmniNeuralSim - SOTA SVG Architecture & Flow Renderer
 * High-definition vector diagrams with ample spacing, responsive viewBox,
 * non-overlapping badges, animated pulse flows, and clean typography.
 */

import { LayerSpec } from "../core/ir.ts";

export interface NodePosition {
  layerIndex: number;
  nodeIndex: number;
  x: number;
  y: number;
  label: string;
}

export class SVGDiagramRenderer {
  static renderNetwork(
    layers: LayerSpec[],
    width = 640,
    height = 360,
    activeLayerIndex = -1,
    flowPhase: "idle" | "forward" | "backward" | "update" = "forward"
  ): string {
    const numLayers = layers.length;
    // Generous horizontal spacing
    const xStep = width / (numLayers + 1);

    const nodesByLayer: NodePosition[][] = [];
    const topMargin = 68;
    const bottomMargin = 55;
    const usableHeight = height - topMargin - bottomMargin;

    layers.forEach((layer, lIdx) => {
      const x = (lIdx + 1) * xStep;
      const count = Math.min(layer.outShape[layer.outShape.length - 1] || 4, 8);
      const yStep = usableHeight / (count + 1);
      const layerNodes: NodePosition[] = [];

      for (let n = 0; n < count; n++) {
        layerNodes.push({
          layerIndex: lIdx,
          nodeIndex: n,
          x,
          y: topMargin + (n + 1) * yStep,
          label: `${layer.name} [${n}]`
        });
      }
      nodesByLayer.push(layerNodes);
    });

    const isBackprop = flowPhase === "backward";
    const flowColor = isBackprop ? "#C8102E" : "#0F8B8D";
    const flowDashDir = isBackprop ? "reverseFlow" : "flowPulse";

    let svg = `<svg viewBox="0 0 ${width} ${height}" width="100%" height="${height}" preserveAspectRatio="xMidYMid meet" style="background: radial-gradient(circle at 50% 50%, #0F172A 0%, #020617 100%); border-radius:12px; font-family:system-ui, -apple-system, sans-serif; display:block;">`;
    svg += `<defs>
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
        .flow-line { stroke-dasharray: 6 3; animation: ${flowDashDir} 0.9s linear infinite; }
        .active-pulse { animation: nodePulse 1.2s ease-in-out infinite alternate; }
        @keyframes nodePulse { from { transform: scale(1); filter: drop-shadow(0 0 2px #FFD700); } to { transform: scale(1.2); filter: drop-shadow(0 0 8px #FFD700); } }
      </style>
    </defs>`;

    // 1. Connection lines
    for (let l = 0; l < nodesByLayer.length - 1; l++) {
      const fromNodes = nodesByLayer[l];
      const toNodes = nodesByLayer[l + 1];
      const isLayerActive = l === activeLayerIndex || l + 1 === activeLayerIndex;
      const strokeUrl = isBackprop ? "url(#edgeGradBwd)" : "url(#edgeGradFwd)";

      for (const from of fromNodes) {
        for (const to of toNodes) {
          const cls = isLayerActive ? "class=\"flow-line\"" : "";
          const strokeWidth = isLayerActive ? "2.2" : "1.0";
          const opacity = isLayerActive ? "0.9" : "0.3";
          svg += `<line x1="${from.x}" y1="${from.y}" x2="${to.x}" y2="${to.y}" stroke="${strokeUrl}" stroke-width="${strokeWidth}" stroke-opacity="${opacity}" ${cls}/>`;
        }
      }
    }

    // 2. Layer Titles, Badges, and Dimensions
    layers.forEach((layer, lIdx) => {
      const x = (lIdx + 1) * xStep;
      const isActive = lIdx === activeLayerIndex;
      const titleColor = isActive ? "#FFD700" : "#E2E8F0";

      // Truncate title if very long
      let displayName = layer.name;
      if (displayName.length > 20) {
        displayName = displayName.slice(0, 18) + "…";
      }

      // Title
      svg += `<text x="${x}" y="22" fill="${titleColor}" font-size="11.5" font-weight="700" text-anchor="middle">${displayName}</text>`;

      // Dimension Badge
      const badgeW = 76;
      svg += `<rect x="${x - badgeW / 2}" y="30" width="${badgeW}" height="18" rx="4" fill="#1E293B" stroke="${isActive ? '#FFD700' : '#334155'}" stroke-width="1"/>`;
      svg += `<text x="${x}" y="43" fill="#38BDF8" font-size="10" font-family="monospace" font-weight="600" text-anchor="middle">[${layer.outShape.join("×")}]</text>`;

      // Parameter count badge at bottom (above flow badge)
      if (layer.paramsCount > 0) {
        svg += `<text x="${x}" y="${height - 36}" fill="#94A3B8" font-size="10" font-family="monospace" text-anchor="middle">${layer.paramsCount.toLocaleString()} params</text>`;
      }
    });

    // 3. Draw Nodes with glow effects
    nodesByLayer.forEach((layerNodes, lIdx) => {
      const isActive = lIdx === activeLayerIndex;
      const fillColor = isActive
        ? "#FFD700"
        : lIdx === 0
        ? "#0F8B8D"
        : lIdx === nodesByLayer.length - 1
        ? (isBackprop ? "#C8102E" : "#10B981")
        : "#6B2D7B";

      for (const node of layerNodes) {
        const glowAttr = isActive ? 'filter="url(#glowGold)" class="active-pulse"' : "";
        svg += `<circle cx="${node.x}" cy="${node.y}" r="${isActive ? 11 : 8.5}" fill="${fillColor}" stroke="#ffffff" stroke-width="1.8" ${glowAttr}/>`;
      }
    });

    // Flow direction indicator badge at the very bottom
    const badgeWidth = 190;
    svg += `<g transform="translate(${width / 2 - badgeWidth / 2}, ${height - 24})">
      <rect width="${badgeWidth}" height="20" rx="10" fill="#1E293B" stroke="${flowColor}" stroke-width="1.2"/>
      <text x="${badgeWidth / 2}" y="14" fill="${flowColor}" font-size="9.5" font-weight="700" text-anchor="middle" letter-spacing="0.5">
        ${isBackprop ? "◀ BACKPROPAGATION GRADIENTS" : "FORWARD INFERENCE FLOW ▶"}
      </text>
    </g>`;

    svg += `</svg>`;
    return svg;
  }
}
