# OmniNeuralSim 🧠⚡

**Universal Pedagogical Neural Network & Deep Learning Simulator**  
*Designed for Harvard CSCI E-89 (Deep Learning), CSCI E-89b (Natural Language Processing) & Advanced Deep Learning*

[![Tests](https://img.shields.io/badge/tests-7%2F7%20passing-brightgreen)]()
[![Bundle Size](https://img.shields.io/badge/bundle%20size-25.8%20kB%20(8.9%20kB%20gzipped)-blue)]()
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-success)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌟 Visión y Propósito

**OmniNeuralSim** es el primer motor simulador universal y autocontenido diseñado para **transformar cualquier tarea o página HTML de Deep Learning en un laboratorio interactivo en vivo**.

A diferencia de herramientas existentes como *TensorFlow Playground* (restringido a MLPs 2D) o *CNN Explainer* (aplicación web aislada de 40MB), **OmniNeuralSim**:
1. **Ejecuta cálculo tensorial real en el cliente** mediante un motor multidimensional en TypeScript puro de solo 25 kB (sin TensorFlow.js ni PyTorch pesados).
2. **Es embebible en un solo tag HTML** usando Web Components nativos: `<neural-sim model="cnn"></neural-sim>`.
3. **Traza matemáticamente cada micro-paso** con su fórmula, dimensiones de tensor y explicación conceptual pedagógica.
4. **Incluye un puente Python (`omni_sim_bridge.py`)** para exportar pesos entrenados en Keras o PyTorch y visualizarlos inmediatamente en el navegador.

---

## 🚀 Arquitecturas Soportadas y Alineación Académica

| Modelo | Módulo OmniNeuralSim | Casos de Uso en Cursos Harvard |
|---|---|---|
| **Multi-Layer Perceptron (MLP)** | `src/models/mlp.ts` | **E-89 W1 & E-89b W1:** Retropropagación analítica, activación Sigmoid/ReLU/Tanh, límites de decisión XOR en 2D. |
| **Convolutional Neural Network (CNN)** | `src/models/cnn.ts` | **E-89b W3 & E-89 W5:** Convoluciones 2D/1D, deslizamiento de kernel $3\times3$, shape $26\times26\times32$, 320 parámetros, MaxPooling, MNIST. |
| **Recurrent Neural Networks (LSTM & GRU)** | `src/models/rnn.ts` | **E-89b W2 & E-89 W10-11:** Desglose en vivo de compuertas ($f_t, i_t, o_t, c_t$), mitigación de vanishing gradient, unrolling temporal. |
| **Transformer & Self-Attention** | `src/models/transformer.ts` | **E-89b W12 & E-89 W12:** Scaled Dot-Product Attention, matrices de afinidad $Q K^T / \sqrt{d_k}$, máscara causal autoregresiva, KV-Cache. |
| **Autoencoders & VAE** | `src/models/autoencoder.ts` | **E-89b W6 & E-89 W6:** Cuello de botella latente 2D, error de reconstrucción MSE, interpolación en el espacio latente. |

---

## 📦 Cómo Embeberlo en tus Entregables HTML

Solo necesitas incluir el script generado y colocar la etiqueta `<neural-sim>`:

```html
<!-- 1. Cargar el bundle ligero (25 kB) -->
<script src="dist/omni-neural-sim.iife.js"></script>

<!-- 2. Embeber el simulador deseado -->
<!-- Para Assignment 3: CNN en MNIST -->
<neural-sim model="cnn"></neural-sim>

<!-- Para Assignment 2: LSTM con compuertas -->
<neural-sim model="lstm"></neural-sim>

<!-- Para Assignment 1: MLP con Backpropagation -->
<neural-sim model="mlp"></neural-sim>

<!-- Para Transformers y Modelos de Lenguaje -->
<neural-sim model="transformer"></neural-sim>
```

---

## 🐍 Exportar Pesos Reales desde Python (Colab / Jupyter)

Puedes entrenar cualquier modelo en Keras/PyTorch y exportarlo directamente para que el simulador lo cargue en vivo:

```python
from omni_sim_bridge import export_keras_model

# Tu modelo Keras
model = build_cnn()
model.fit(train_images, train_labels, epochs=10)

# Exportar a JSON para el simulador web
export_keras_model(model, filepath="assignment3_weights.json")
```

Luego en tu HTML:
```html
<neural-sim model="cnn" weights="assignment3_weights.json"></neural-sim>
```

---

## 🛠️ Instalación y Desarrollo Local

```bash
# Clonar el repositorio
git clone https://github.com/ManuelCadena/omni-neural-sim.git
cd omni-neural-sim

# Instalar dependencias ligeras (Vite + Vitest)
pnpm install

# Correr tests unitarios
pnpm test

# Iniciar servidor interactivo de desarrollo
pnpm dev

# Compilar los bundles de producción (ES, UMD, IIFE)
pnpm build
```

---

## 📚 Documentación y Estado del Arte

Para una revisión exhaustiva de la literatura académica (IEEE TVCG, ACM CHI, NeurIPS) y el benchmark contra las herramientas existentes, consulta:  
📖 [`docs/BENCHMARK_SOTA_SIMULADORES.md`](docs/BENCHMARK_SOTA_SIMULADORES.md)

---

## 📄 Licencia

MIT License © 2026 Jose Manuel Cadena Ortiz de Montellano.
