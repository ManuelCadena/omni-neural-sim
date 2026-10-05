# Benchmark y Estado del Arte (SOTA) en Simuladores Pedagógicos de Redes Neuronales

**Autor:** Jose Manuel Cadena Ortiz de Montellano  
**Contexto:** Harvard CSCI E-89 (Deep Learning), CSCI E-89b (Natural Language Processing) & Advanced Deep Learning  
**Fecha:** Octubre 2026  
**Librería/Plataforma:** `omni-neural-sim` (OmniNeuralSim)

---

## 1. Motivación y Problema Fundamental

El aprendizaje profundo contemporáneo presenta un desafío cognitivo y pedagógico mayor: los estudiantes suelen ver las redes neuronales como **"cajas negras"** opacas donde las ecuaciones teóricas (retropropagación, convoluciones, compuertas de memoria, atención multifoco) quedan desconectadas de su implementación práctica en tensores.

A pesar de que existen múltiples visualizadores en la literatura académica, la investigación identificó una **fragmentación crítica**:
1. Cada herramienta existente fue construida como un **silo cerrado** para un único tipo de arquitectura (ej. TensorFlow Playground solo hace MLPs 2D; CNN Explainer solo hace una CNN fija; Transformer Explainer solo hace GPT-2).
2. Ninguna herramienta existente proporciona un **motor universal parametrizable** capaz de cargarse vía Web Component (`<neural-sim>`) dentro de entregables HTML estáticos de tareas académicas.
3. Carecen de un **puente directo (bridge)** para exportar pesos entrenados en Keras o PyTorch y simularlos de inmediato en el navegador sin dependencias de servidor.

---

## 2. Matriz Comparativa del Estado del Arte (SOTA)

A partir de la investigación en **Consensus** (artículos revisados por pares en IEEE TVCG y ACM CHI), **Perplexity API** y el análisis de síntesis arquitectónica de **AION Brain**, se evaluaron las 8 herramientas dominantes:

| Simulador / Herramienta | Venia Académica / Autor | Alcance de Modelos | Ejecución Client-Side | Embebible en HTML / Canvas | Soporta Pesos Reales Keras/PyTorch | Desglose Matemático Paso a Paso |
|---|---|---|---|---|---|---|
| **TensorFlow Playground** | Google PAIR (Smilkov & Carter, 2016) | Solo MLP feedforward en plano 2D | Sí (TypeScript puro) | Parcial (iframe pesado) | No (solo datos sintéticos XOR/spiral) | Básico (pesos y activaciones) |
| **CNN Explainer** | IEEE TVCG 2020 (Wang et al., Georgia Tech) [1] | Solo CNN 2D (Tiny VGG) | Sí (TensorFlow.js) | No (aplicación web completa de 40MB) | No (pesos fijos pre-entrenados) | **Excelente** (kernels $3\times3$, receptive field) |
| **Transformer Explainer** | ACM CHI 2024 (Cho et al., Polo Club) [7] | Solo Transformer autoregresivo (GPT-2) | Sí (ONNX Runtime Web) | No (aplicación monolítica Svelte) | Limitado (solo variantes GPT-2) | **Excelente** ($Q, K, V$, atención causal) |
| **LLM Visualization** | Brendan Bycroft (2023) | Solo Transformer 3D (GPT) | Sí (Three.js WebGL) | No (app 3D intensiva) | No (pesos de juguete fijos) | **Excelente** (matrices y flujos 3D) |
| **GAN Lab** | IEEE TVCG 2018 (Kahng et al.) [6] | Solo GANs 2D sintéticas | Sí (TensorFlow.js) | No (app web cerrada) | No (solo distribuciones 2D) | Bueno (generador vs discriminador) |
| **BertViz / Multiscale Attn** | ACL 2019 (Jesse Vig) [3] | Solo mapas de atención Transformer | No (requiere kernel Jupyter Python) | No (solo notebook) | Sí (vía PyTorch HuggingFace) | Bueno (líneas de atención por cabeza) |
| **Netron** | Lutz Roeder (Open Source) | Inspección estática de grafos ONNX/H5 | Sí (Electron / WebAssembly) | Sí | Sí | **Nulo** (no ejecuta forward ni backward) |
| **OmniNeuralSim (Nuestra Propuesta)** | **Cadena Ortiz de Montellano (2026)** | **Universal (MLP, CNN 2D/1D, LSTM, GRU, Transformer, VAE)** | **Sí (Zero-dep, 25.8 kB bundle)** | **100% Nativo (`<neural-sim>`)** | **Sí (Bridge Python directo `omni_sim_bridge.py`)** | **Total (ExecutionTracer con fórmulas LaTeX y derivadas)** |

---

## 3. Principales Hallazgos de la Literatura Científica

### A. La Ley de la Transición Suave de Abstracción (Wang et al., IEEE TVCG 2020)
El estudio empírico de *CNN Explainer* con cientos de estudiantes de posgrado demostró que el aprendizaje se maximiza cuando una herramienta permite transitar dinámicamente entre:
$$\text{Overview del Grafo} \longleftrightarrow \text{Tensores Intermedios} \longleftrightarrow \text{Aritmética de Píxeles / Compuertas}$$
Cuando el estudiante solo ve el diagrama general (como en Netron), no entiende cómo cambia la dimensionalidad ($28 \to 26$). Cuando solo ve código matricial, pierde la intuición espacial. **OmniNeuralSim adopta este principio sincronizando el SVG del grafo con el Canvas de píxeles y el ExecutionTracer de fórmulas.**

### B. El Mecanismo de Atención como Proyección Dinámica (Yeh et al., IEEE TVCG 2023; Cho et al., ACM CHI 2024)
En modelos Transformers y LLMs, la atención no debe enseñarse como una multiplicación abstracta sino como una **matriz de afinidad contextual normalizada**:
$$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V$$
El visualizador debe permitir encender y apagar la máscara causal $M$ para que el alumno entienda por qué un modelo autoregresivo no puede "mirar hacia el futuro".

### C. La Ventaja Cognitiva del Sandbox Liviano (Fung, UKICER 2026)
La investigación reciente de Fung demostró con significancia estadística ($p = 0.0024$) que la interactividad ligera integrada dentro del propio documento de lectura/tarea supera sustancialmente al envío de los estudiantes a sitios web externos o iframes pesados.

---

## 4. Arquitectura de `omni-neural-sim`

```text
┌────────────────────────────────────────────────────────┐
│                   Python Exporter                      │
│   Keras / PyTorch / R torch -> model_spec.json         │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│          OmniNeuralSim Core (TypeScript ES/IIFE)       │
│                                                        │
│  ┌──────────────────┐  ┌────────────────────────────┐  │
│  │   Tensor Engine  │  │      Execution Tracer      │  │
│  │ matmul, conv2d,  │  │ records activations, gates,│  │
│  │ pool2d, softmax  │  │ shapes, formulas & insights│  │
│  └─────────┬────────┘  └─────────────┬──────────────┘  │
│            │                         │                 │
│  ┌─────────▼─────────────────────────▼──────────────┐  │
│  │                Model Engines                     │  │
│  │  • MLP (Backprop / XOR / Decision Boundary)      │  │
│  │  • CNN (MNIST 28x28 / 32 Kernels / 320 Params)   │  │
│  │  • LSTM / GRU (4 Gates f, i, c, o Step Tracing)  │  │
│  │  • Transformer (Scaled Dot-Product & MHA)        │  │
│  │  • Autoencoder / VAE (Latent Space Explorer)     │  │
│  └─────────────────────┬────────────────────────────┘  │
│                        │                               │
│  ┌─────────────────────▼────────────────────────────┐  │
│  │                Renderers Layer                   │  │
│  │  • SVGDiagramRenderer (Graph & Tensor Shapes)    │  │
│  │  • CanvasVisualizer (Heatmaps, Boundaries, MNIST)│  │
│  └─────────────────────┬────────────────────────────┘  │
└────────────────────────┼───────────────────────────────┘
                         │
┌────────────────────────▼───────────────────────────────┐
│     Web Component Standalone: <neural-sim>             │
│   Embebible en cualquier página HTML / Submission      │
└────────────────────────────────────────────────────────┘
```

---

## 5. Referencias Citadas

- [1] **CNN Explainer: Learning Convolutional Neural Networks with Interactive Visualization** (Zijie J. Wang et al., 2020, IEEE Transactions on Visualization and Computer Graphics, DOI: 10.1109/tvcg.2020.3030418)
- [2] **Web-based prototype of a visual and interactive deep learning simulation** (Christian Koch et al., 2024, DELFI, DOI: 10.18420/delfi2024_48)
- [3] **A Multiscale Visualization of Attention in the Transformer Model** (Jesse Vig, 2019, ACL Demo, DOI: 10.18653/v1/p19-3007)
- [4] **Dodrio: Exploring Transformer Models with Interactive Visualization** (Zijie J. Wang et al., 2021, ACL, DOI: 10.18653/v1/2021.acl-demo.16)
- [5] **AttentionViz: A Global View of Transformer Attention** (Catherine Yeh et al., 2023, IEEE TVCG, DOI: 10.1109/tvcg.2023.3327163)
- [6] **GAN Lab: Understanding Complex Deep Generative Models using Interactive Visual Experimentation** (Minsuk Kahng et al., 2018, IEEE TVCG, DOI: 10.1109/tvcg.2018.2864500)
- [7] **Transformer Explainer: Learning LLM Transformers with Interactive Visual Explanation and Experimentation** (Aeree Cho et al., 2024, ACM CHI 2024, DOI: 10.1145/3772318.3791725)
- [8] **Build Your Own CNN: Visualisation for Deep Learning Education** (Avis Fung, 2026, UKICER, DOI: 10.1145/3830800.3830816)
