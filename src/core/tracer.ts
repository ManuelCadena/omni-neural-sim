/**
 * OmniNeuralSim Execution Tracer
 * Captures live tensor evaluations, gate states, attention matrices,
 * and mathematical step-by-step explanations.
 */

import { Tensor } from "./tensor.ts";

export interface TraceStep {
  stepIndex: number;
  layerId: string;
  layerName: string;
  operation: string;
  formula: string;
  inputShapes: number[][];
  outputShape: number[];
  tensorPreview: number[]; // First few elements or summary
  fullOutput?: Tensor;
  pedagogicalInsight: string;
  metadata?: Record<string, any>;
}

export class ExecutionTracer {
  steps: TraceStep[] = [];

  record(step: Omit<TraceStep, "stepIndex">): void {
    this.steps.push({
      stepIndex: this.steps.length,
      ...step
    });
  }

  clear(): void {
    this.steps = [];
  }

  getTraceSummary(): string {
    return this.steps
      .map(
        (s) =>
          `[Step ${s.stepIndex}] ${s.layerName} (${s.operation}): ${s.formula} -> Shape [${s.outputShape.join(", ")}]`
      )
      .join("\n");
  }
}
