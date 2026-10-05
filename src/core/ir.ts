/**
 * OmniNeuralSim IR (Intermediate Representation)
 * Universal Declarative Specification for Deep Learning Models.
 */

export type ModelType =
  | "mlp"
  | "cnn_2d"
  | "cnn_1d"
  | "rnn_vanilla"
  | "lstm"
  | "gru"
  | "transformer_attention"
  | "autoencoder"
  | "vae"
  | "gan";

export type ActivationType = "relu" | "sigmoid" | "tanh" | "gelu" | "softmax" | "linear";

export interface LayerSpec {
  id: string;
  name: string;
  type:
    | "dense"
    | "conv2d"
    | "maxpool2d"
    | "flatten"
    | "lstm_cell"
    | "gru_cell"
    | "self_attention"
    | "embedding"
    | "latent_space";
  inShape: number[];
  outShape: number[];
  activation?: ActivationType;
  paramsCount: number;
  config?: Record<string, any>;
  pedagogicalNote?: string;
}

export interface ModelSpec {
  version: "1.0.0";
  name: string;
  courseContext?: string; // e.g. "Harvard CSCI E-89b Week 3"
  modelType: ModelType;
  layers: LayerSpec[];
  totalParameters: number;
  weights?: Record<string, number[]>;
  metadata: {
    author?: string;
    description: string;
    targetAccuracy?: number;
    recommendedVisualization: "graph_2d" | "cnn_explainer" | "sequence_unroll" | "attention_matrix" | "latent_manifold";
  };
}
