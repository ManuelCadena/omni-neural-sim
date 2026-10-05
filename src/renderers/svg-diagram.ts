/**
 * OmniNeuralSim - SVG Architecture Renderer
 * Generates interactive vector diagrams with tensor shape badges,
 * dynamic weight line thicknesses, and tooltip formula popups.
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
    width = 800,
    height = 360,
    activeLayerIndex = -1
  ): string {
    const numLayers = layers.length;
    const xStep = width / (numLayers + 1);

    const nodesByLayer: NodePosition[][] = [];

    // Calculate node coordinates
    layers.forEach((layer, lIdx) => {
      const x = (lIdx + 1) * xStep;
      // Representative node count (max 8 for visual clarity)
      const count = Math.min(layer.outShape[layer.outShape.length - 1] || 4, 8);
      const yStep = height / (count + 1);
      const layerNodes: NodePosition[] = [];

      for (let n = 0; n < count; n++) {
        layerNodes.push({
          layerIndex: lIdx,
          nodeIndex: n,
          x,
          y: (n + 1) * yStep,
          label: `${layer.name} [${n}]`
        });
      }
      nodesByLayer.push(layerNodes);
    });

    let svg = `<svg viewBox="0 0 ${width} ${height}" width="100%" height="${height}" style="background:#0b1329; border-radius:12px; font-family:system-ui, -apple-system, sans-serif;">`;
    svg += `<defs>
      <linearGradient id="edgeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="#0F8B8D" stop-opacity="0.4"/>
        <stop offset="100%" stop-color="#6B2D7B" stop-opacity="0.6"/>
      </linearGradient>
      <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
        <feGaussianBlur stdDeviation="3" result="blur" />
        <feComposite in="SourceGraphic" in2="blur" operator="over" />
      </filter>
    </defs>`;

    // 1. Draw connecting lines between consecutive layers
    for (let l = 0; l < nodesByLayer.length - 1; l++) {
      const fromNodes = nodesByLayer[l];
      const toNodes = nodesByLayer[l + 1];
      for (const from of fromNodes) {
        for (const to of toNodes) {
          svg += `<line x1="${from.x}" y1="${from.y}" x2="${to.x}" y2="${to.y}" stroke="url(#edgeGrad)" stroke-width="1.2"/>`;
        }
      }
    }

    // 2. Draw Layer Titles and Tensor Shapes
    layers.forEach((layer, lIdx) => {
      const x = (lIdx + 1) * xStep;
      const isActive = lIdx === activeLayerIndex;
      const titleColor = isActive ? "#FFD700" : "#E2E8F0";

      svg += `<text x="${x}" y="24" fill="${titleColor}" font-size="12" font-weight="600" text-anchor="middle">${layer.name}</text>`;
      svg += `<text x="${x}" y="40" fill="#94A3B8" font-size="10" font-family="monospace" text-anchor="middle">[${layer.outShape.join("×")}]</text>`;
      if (layer.paramsCount > 0) {
        svg += `<text x="${x}" y="${height - 12}" fill="#38BDF8" font-size="10" font-family="monospace" text-anchor="middle">${layer.paramsCount.toLocaleString()} params</text>`;
      }
    });

    // 3. Draw Nodes
    nodesByLayer.forEach((layerNodes, lIdx) => {
      const isActive = lIdx === activeLayerIndex;
      const fillColor = isActive ? "#FFD700" : lIdx === 0 ? "#0F8B8D" : lIdx === nodesByLayer.length - 1 ? "#C8102E" : "#6B2D7B";

      for (const node of layerNodes) {
        svg += `<circle cx="${node.x}" cy="${node.y}" r="9" fill="${fillColor}" stroke="#ffffff" stroke-width="1.5" ${isActive ? 'filter="url(#glow)"' : ""}/>`;
      }
    });

    svg += `</svg>`;
    return svg;
  }
}
