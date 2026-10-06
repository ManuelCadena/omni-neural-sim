/**
 * OmniNeuralSim - Pedagogical Master Guide Repository
 * Comprehensive educational explanations for all Harvard CSCI E-89 / E-89b architectures.
 */

export interface ModelPedagogy {
  title: string;
  syllabusContext: string;
  problemStatement: string;
  mathematicalMechanics: {
    forwardPass: string;
    lossAndUpdate: string;
    tensorShapes: string;
  };
  visualGuide: {
    diagramElements: string;
    canvasMechanics: string;
    colorSignaling: string;
  };
  pedagogicalTakeaways: string[];
}

export const PEDAGOGY_REGISTRY: Record<string, ModelPedagogy> = {
  mlp: {
    title: "Multi-Layer Perceptron (MLP) & Analytical Backpropagation",
    syllabusContext: "Harvard CSCI E-89b Week 1 & CSCI E-89 Week 1",
    problemStatement:
      "Linear models (like a single perceptron) fail on non-linearly separable problems like XOR because they can only draw a single straight hyper-plane decision boundary. An MLP solves this by stacking layers of affine transformations interleaved with non-linear activation functions (Tanh, ReLU, Sigmoid), warping the input space so classes become linearly separable in the final hidden representation.",
    mathematicalMechanics: {
      forwardPass:
        "1. Layer 1: z^[1] = W^[1] · x + b^[1] → a^[1] = tanh(z^[1])\n2. Layer 2: z^[2] = W^[2] · a^[1] + b^[2] → a^[2] = tanh(z^[2])\n3. Output Layer: z^[3] = W^[3] · a^[2] + b^[3] → ŷ = σ(z^[3])",
      lossAndUpdate:
        "Loss: L = 1/2 · (ŷ - y)^2\nOutput error: δ^[3] = (ŷ - y) ⊙ σ'(z^[3])\nHidden error: δ^[l] = ((W^[l+1])^T · δ^[l+1]) ⊙ tanh'(z^[l])\nWeight gradient: ∂L/∂W^[l] = (a^[l-1])^T · δ^[l]\nUpdate rule: W^[l] ← W^[l] - η · ∂L/∂W^[l]",
      tensorShapes: "Input: [1, 2] → Hidden 1: [1, 6] (18 params) → Hidden 2: [1, 4] (28 params) → Output: [1, 1] (5 params). Total: 51 parameters."
    },
    visualGuide: {
      diagramElements:
        "Circles represent individual neurons. Connecting lines represent synaptic weight parameters. Line thickness indicates weight magnitude. During forward pass, pulses travel left-to-right (teal). During backpropagation, error gradients travel in reverse (crimson).",
      canvasMechanics:
        "The 2D canvas displays the continuous decision boundary surface over [-4, 4]^2. Dark navy represents class 0, teal represents neutral transition, and yellow points represent XOR training coordinates ((-2,-2)→0, (-2,2)→1, (2,-2)→1, (2,2)→0). Notice how the boundary bends non-linearly as you step through backpropagation updates.",
      colorSignaling:
        "• Teal (#0F8B8D): Forward activations & positive weights\n• Crimson (#C8102E): Backward gradient flow & negative weights\n• Gold (#FFD700): Active updating layer & class 1 labels"
    },
    pedagogicalTakeaways: [
      "Non-linear activations are mandatory; without them, stacking dense layers collapses into a single trivial linear transformation W_total = W_n · ... · W_1.",
      "Backpropagation is not a different machine learning model; it is simply an efficient computational application of the multivariate chain rule that computes exact partial derivatives in O(parameters) time rather than O(parameters^2)."
    ]
  },

  cnn: {
    title: "Convolutional Neural Networks (CNN) & Receptive Field Mechanics",
    syllabusContext: "Harvard CSCI E-89b Week 3 & CSCI E-89 Week 5",
    problemStatement:
      "Fully connected networks flatten images into 1D vectors, completely discarding spatial 2D locality and requiring an explosion of parameters. CNNs introduce two fundamental inductive biases: (1) Local connectivity (neurons only connect to a small patch called the receptive field), and (2) Weight sharing (the same learnable filter slides across the entire image), ensuring translation invariance and drastic parameter reduction.",
    mathematicalMechanics: {
      forwardPass:
        "1. 2D Convolution: z_{r,c,k} = (Σ_i Σ_j x_{r+i, c+j} · w_{i,j,k}) + b_k\n2. Activation: a_{r,c,k} = ReLU(z_{r,c,k})\n3. MaxPooling: p_{r,c,k} = max_{i,j ∈ {0,1}} a_{2r+i, 2c+j, k}\n4. Flatten: [3, 3, 64] → [576]\n5. Dense + Softmax: P(digit = k) = exp(z_k) / Σ_j exp(z_j)",
      lossAndUpdate:
        "Output spatial dimensions: H_out = ⌊(H_in - K_h + 2P)/S⌋ + 1 = ⌊(28 - 3 + 0)/1⌋ + 1 = 26.\nParameter Count Law: (K_h · K_w · C_in + 1) · C_out = (3 · 3 · 1 + 1) · 32 = 320 parameters.",
      tensorShapes: "Input: [28, 28, 1] → Conv1: [26, 26, 32] (320 params) → MaxPool1: [13, 13, 32] → Conv2: [11, 11, 64] (18,496 params) → MaxPool2: [5, 5, 64] → Conv3: [3, 3, 64] (36,928 params) → Flatten: [576] → Dense: [64] → Softmax: [10]."
    },
    visualGuide: {
      diagramElements:
        "The SVG illustrates the dimensional shrinkage of feature maps through convolution and 2x2 downsampling pooling layers, terminating in the flattened dense classification head.",
      canvasMechanics:
        "The left pane shows the raw 28x28 grayscale MNIST digit with a sliding yellow 3x3 receptive field bounding box. The right pane magnifies this exact 3x3 window, displaying the 9 numerical input pixels multiplied by the 9 filter weights, adding bias, and producing the single active feature map activation.",
      colorSignaling:
        "• Yellow box: Active 3x3 receptive field sliding across spatial coordinates\n• Green text: Post-ReLU activation value lighting up the output feature map\n• Blue numbers: Learnable filter weights W_{i,j}"
    },
    pedagogicalTakeaways: [
      "Layer 1 has 320 parameters regardless of whether the image is 28x28 or 4K resolution, because kernel weights depend exclusively on kernel size (3x3), input channels (1), and filter count (32).",
      "Softmax turns raw unbounded real numbers (logits) into a valid probability distribution that sums to 1.0; argmax selects the index with the maximum posterior probability."
    ]
  },

  lstm: {
    title: "Long Short-Term Memory (LSTM) & Constant Error Carousel",
    syllabusContext: "Harvard CSCI E-89b Week 2 & CSCI E-89 Week 11",
    problemStatement:
      "Vanilla RNNs suffer from vanishing and exploding gradients over long sequences because the repeated matrix multiplication (W_hh)^T causes gradients to decay exponentially to zero (if eigenvalues < 1) or explode to infinity. LSTMs solve this by establishing an additive memory highway (the Constant Error Carousel, c_t) regulated by three multiplicative sigmoid gates that decide what to forget, what to write, and what to expose.",
    mathematicalMechanics: {
      forwardPass:
        "1. Forget gate: f_t = σ(W_f · x_t + U_f · h_{t-1} + b_f)\n2. Input gate: i_t = σ(W_i · x_t + U_i · h_{t-1} + b_i)\n3. Candidate memory: c̃_t = tanh(W_c · x_t + U_c · h_{t-1} + b_c)\n4. Cell state update: c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t\n5. Output gate: o_t = σ(W_o · x_t + U_o · h_{t-1} + b_o)\n6. Hidden state: h_t = o_t ⊙ tanh(c_t)",
      lossAndUpdate:
        "Because c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t is an ADDITIVE update, the gradient ∂c_t / ∂c_{t-1} = f_t. If f_t ≈ 1, error signals flow backwards across hundreds of time steps without multiplying by small weights, eliminating vanishing gradients.",
      tensorShapes: "Token input x_t: [1, 4] → Gates [f, i, c̃, o]: [1, 16] (144 params) → Cell state c_t: [1, 4] → Hidden state h_t: [1, 4]."
    },
    visualGuide: {
      diagramElements:
        "The SVG unrolls the recurrent transition across sequential time steps, highlighting the internal gate computations and hidden state emissions.",
      canvasMechanics:
        "The top purple rail represents the horizontal memory conveyor belt (Cell State c_t). Four vertical gauge meters monitor the live value of each gate: Forget gate f_t (crimson), Input gate i_t (teal), Candidate c̃_t (blue), and Output gate o_t (green). Notice how changing inputs modifies the retention percentage.",
      colorSignaling:
        "• Crimson gauge (Forget): 0 = purge memory, 1 = retain past indefinitely\n• Teal gauge (Input): Controls write volume of current token into memory\n• Gold highlight: Current active time step t and token input string"
    },
    pedagogicalTakeaways: [
      "The cell state c_t is internal long-term storage; the hidden state h_t is the filtered short-term output exposed to downstream layers.",
      "Initializing the forget gate bias b_f to +1.0 or +2.0 at the start of training is standard deep learning practice to prevent the model from inadvertently forgetting early context before learning what to remember."
    ]
  },

  transformer: {
    title: "Transformers, Scaled Dot-Product Attention & Causal Masking",
    syllabusContext: "Harvard CSCI E-89b Week 12 & CSCI E-89 Week 12",
    problemStatement:
      "Recurrent networks process text sequentially (O(T) time complexity), creating an inescapable computational bottleneck that prevents parallel GPU execution and compresses all historical context into a fixed-size vector. Transformers eliminate recurrence completely, enabling all tokens to attend directly to each other in parallel via Query-Key-Value dot-product affinities.",
    mathematicalMechanics: {
      forwardPass:
        "1. Projections: Q = X · W_Q,  K = X · W_K,  V = X · W_V\n2. Attention Matrix: S = (Q · K^T) / √d_k\n3. Causal Masking: S_{i,j} = -∞ for j > i (autoregressive decoder)\n4. Attention Weights: A = softmax(S)\n5. Context Representation: Output = A · V · W_O",
      lossAndUpdate:
        "Division by √d_k is critical: for large embedding dimensions d_k, the dot products grow large in magnitude, pushing the softmax function into regions with extremely small gradients. Dividing by √d_k stabilizes the variance to 1.0.",
      tensorShapes: "Sequence X: [N, d_model] → Q, K, V: [N, d_k] → Attention Matrix: [N, N] → Output: [N, d_model]."
    },
    visualGuide: {
      diagramElements:
        "The SVG shows token embedding projections into Q, K, V subspace spaces, through the NxN attention affinity tensor, and recombining through the multi-head projection layer.",
      canvasMechanics:
        "The canvas renders the live NxN attention heatmap. Each row corresponds to a Query token; each column corresponds to a Key token. Dark tiles with '—' represent causally masked future positions that autoregressive decoders are strictly prohibited from attending to. The highlighted row indicates the token currently computing context.",
      colorSignaling:
        "• Gold row & cells: Current Query token and its strongest attention weights\n• Teal cells: Permitted historical attention affinities\n• Dark grey ('—'): Causal mask blocking future information leakage"
    },
    pedagogicalTakeaways: [
      "Attention is permutation-equivariant; without sinusoidal or learned positional encodings added to input embeddings, the model cannot distinguish between 'dog bites man' and 'man bites dog'.",
      "In autoregressive generation (GPT), the KV-Cache caches previously computed K and V vectors so generating the next token only requires computing Q for the single new token (O(N) instead of O(N^2) per step)."
    ]
  },

  autoencoder: {
    title: "Autoencoders & Variational Autoencoders (VAE) Latent Manifolds",
    syllabusContext: "Harvard CSCI E-89b Week 6 & CSCI E-89 Week 6",
    problemStatement:
      "High-dimensional real-world data (like text embeddings or images) is noisy and sparse, but typically lies on a lower-dimensional smooth manifold. Autoencoders discover this intrinsic coordinate system by compressing inputs through an informational bottleneck (encoder) and learning to reconstruct the original features (decoder), performing non-linear dimensionality reduction superior to linear PCA.",
    mathematicalMechanics: {
      forwardPass:
        "1. Encoder: h = ReLU(W_enc1 · x + b_1) → z = W_enc2 · h + b_2  (Latent Bottleneck)\n2. Decoder: h_dec = ReLU(W_dec1 · z + b_3) → x̂ = σ(W_dec2 · h_dec + b_4)\n3. Reconstruction Loss: L_rec = 1/D · Σ_{d=1}^D (x_d - x̂_d)^2\n4. VAE KL Divergence: D_KL(q(z|x) || p(z)) = -1/2 · Σ (1 + log σ^2 - μ^2 - σ^2)",
      lossAndUpdate:
        "Reparameterization trick: to backpropagate through stochastic latent sampling in VAEs, draw ε ~ N(0, I) and compute z = μ + ε ⊙ σ, moving randomness outside the differentiable computational graph.",
      tensorShapes: "Input x: [1, 10] → Encoder: [1, 16] → Latent z: [1, 2] → Decoder: [1, 16] → Reconstructed x̂: [1, 10]."
    },
    visualGuide: {
      diagramElements:
        "The hourglass SVG clearly visualizes the bottleneck architecture: wide input layer tapering down to the narrow 2-dimensional latent bottleneck node, then expanding back to the output reconstruction.",
      canvasMechanics:
        "The left pane depicts the 2D latent space coordinate plane with data cluster clouds and the active latent point z = (z1, z2). The right pane plots a direct feature-by-feature bar comparison between original input features x (teal) and decoded features x̂ (gold), alongside the live Mean Squared Error (MSE).",
      colorSignaling:
        "• Purple: Bottleneck latent representation z\n• Teal bars: Original uncompressed ground-truth features x\n• Gold bars: Reconstructed features x̂ synthesized by the decoder"
    },
    pedagogicalTakeaways: [
      "The capacity of the latent space must be strictly restricted (bottleneck); otherwise a network with sufficient capacity will simply learn the identity function without extracting meaningful semantic abstractions.",
      "Standard autoencoders have gaps in their latent space where decoded points produce unrecognizable noise; VAEs solve this by enforcing Gaussian distribution priors through KL-divergence regularization."
    ]
  },

  optimizers: {
    title: "Loss Landscapes & Deep Optimization: SGD vs Momentum vs Adam",
    syllabusContext: "Harvard CSCI E-89 Week 3 & Week 13",
    problemStatement:
      "Deep neural network loss functions are non-convex surfaces filled with ravines, saddle points, and plateaus. Standard Stochastic Gradient Descent (SGD) struggles severely in ravines where surface curvature is much steeper in one direction than another, oscillating violently across walls with negligible progress along the bottom toward the minimum.",
    mathematicalMechanics: {
      forwardPass:
        "1. Gradient: g_t = ∇_w L(w_t)\n2. SGD: w_{t+1} = w_t - η · g_t\n3. Momentum: v_t = γ · v_{t-1} + η · g_t → w_{t+1} = w_t - v_t\n4. Adam (Adaptive Moment Estimation):\n   m_t = β_1 · m_{t-1} + (1 - β_1) · g_t  (1st moment: mean)\n   v_t = β_2 · v_{t-1} + (1 - β_2) · g_t^2  (2nd moment: uncentered variance)\n   m̂_t = m_t / (1 - β_1^t),  v̂_t = v_t / (1 - β_2^t)  (bias correction)\n   w_{t+1} = w_t - (η / (√v̂_t + ε)) · m̂_t",
      lossAndUpdate:
        "Adam automatically scales the step size inversely with the square root of recent gradient magnitudes: parameters with consistently large gradients take smaller cautious steps; parameters with sparse or flat gradients take larger progressive steps.",
      tensorShapes: "Parameter vector w: [D] → Gradient ∇L: [D] → Moments m, v: [D] → Update step: [D]."
    },
    visualGuide: {
      diagramElements:
        "The SVG illustrates the optimizer update pipeline: converting scalar loss into directional gradient vectors, accumulating historical momentum velocity, and computing Adam's adaptive step vector.",
      canvasMechanics:
        "The canvas plots an elliptical ravine loss contour landscape. Three distinct optimization trajectories race down the ravine: SGD in crimson (oscillating violently between steep walls), Momentum in amber (dampening oscillations by accumulating momentum down the valley floor), and Adam in green (smooth, direct adaptive descent straight to the global minimum).",
      colorSignaling:
        "• Crimson line: Vanilla SGD (high oscillation, slow convergence in ravines)\n• Amber line: Classical Momentum (damped oscillations, fast valley traversal)\n• Green line: Adam (adaptive per-coordinate learning rate, state-of-the-art default)"
    },
    pedagogicalTakeaways: [
      "Momentum accelerates SGD in directions where gradients consistently point the same way while cancelling out oscillations in directions where gradients alternate signs.",
      "Adam's bias corrections (1 - β^t) are essential during early iterations (t=1,2,...) to prevent the moving averages from being severely biased toward their initial zero values."
    ]
  },

  ngrams: {
    title: "N-grams, Bag-of-Words (BoW) & Vocabulary OOV Firewalls",
    syllabusContext: "Harvard CSCI E-89b Week 4",
    problemStatement:
      "Raw natural language consists of variable-length unstructured strings that cannot be directly multiplied by neural weight matrices. N-grams and Bag-of-Words models convert text into fixed-length numerical frequency vectors, but introduce a severe data leakage risk if test-set words leak into the vocabulary.",
    mathematicalMechanics: {
      forwardPass:
        "1. Tokenization: Text → List of discrete tokens [w_1, w_2, ..., w_T]\n2. Lemmatization: w_i → canonical dictionary lemma l_i\n3. Vocabulary Mapping: Index(l) if l ∈ V else Index(<UNK>)\n4. BoW Frequency Count: x[k] = Σ_{t=1}^T 𝟙(token_t = word_k)\n5. N-gram Markov Chain: P(w_n | w_1, ..., w_{n-1}) ≈ P(w_n | w_{n-1})",
      lossAndUpdate:
        "Maximum Likelihood Estimation for Bigrams: P(w_i | w_{i-1}) = Count(w_{i-1}, w_i) / Count(w_{i-1}). Add-1 (Laplace) smoothing handles unseen pairs: P_Laplace = (Count + 1) / (Total + |V|).",
      tensorShapes: "Vocabulary V: [|V|] terms → Document vector: [1, |V|] frequency counts."
    },
    visualGuide: {
      diagramElements:
        "The SVG displays the text pipeline: token stream entering a vocabulary hash table, passing through the OOV firewall, and forming the fixed-dimensional frequency vector.",
      canvasMechanics:
        "The canvas provides a streaming frequency bar chart for each vocabulary term. When an unknown test word appears in the stream, the simulator dynamically triggers the crimson <UNK> bucket, demonstrating how vector dimensions stay constant without leaking test tokens.",
      colorSignaling:
        "• Teal bars: Valid in-vocabulary word frequencies\n• Crimson bar: Out-of-Vocabulary (<UNK>) fallback bin\n• Gold highlight: Current streaming token under inspection"
    },
    pedagogicalTakeaways: [
      "The vocabulary must be fitted strictly on training data; if the vocabulary is built on the combination of train and test text, information leaks into the model and invalidates the evaluation.",
      "Bag-of-Words completely discards word order ('not good, very bad' produces the exact same BoW vector as 'good, not very bad'); N-grams preserve local context at the cost of exponential vocabulary growth |V|^N."
    ]
  },

  embeddings: {
    title: "Word2Vec, Distributed Representations & Vector Analogies",
    syllabusContext: "Harvard CSCI E-89b Week 5",
    problemStatement:
      "One-hot encodings represent words as mutually orthogonal vectors of length |V| where dot products are always zero (cos(θ) = 0 for any pair of words), making it impossible to capture that 'cat' is closer in meaning to 'dog' than to 'refrigerator'. Distributed word embeddings embed words into a compact d-dimensional continuous geometric space (d ≈ 300) where semantic similarity corresponds to geometric proximity.",
    mathematicalMechanics: {
      forwardPass:
        "1. Skip-Gram Objective: Maximize average log probability of context words c given center word w:\n   L = 1/T · Σ_{t=1}^T Σ_{-c ≤ j ≤ c, j ≠ 0} log P(w_{t+j} | w_t)\n2. Softmax Formulation: P(w_O | w_I) = exp(v'_{w_O}^T · v_{w_I}) / Σ_{w=1}^{|V|} exp(v'_w^T · v_{w_I})\n3. Negative Sampling: log σ(v'_{w_O}^T · v_{w_I}) + Σ_{k=1}^K E_{w_i ~ P_n(w)} [log σ(-v'_{w_i}^T · v_{w_I})]\n4. Cosine Similarity: cos(θ) = (u · v) / (||u|| · ||v||)",
      lossAndUpdate:
        "Vector Analogy Law: vec('king') - vec('man') + vec('woman') ≈ vec('queen'). Linear translations in the embedding space encode semantic and grammatical relationships.",
      tensorShapes: "One-hot: [1, |V|] → Embedding Matrix W: [|V|, d] → Dense Vector: [1, d] (d ≈ 300)."
    },
    visualGuide: {
      diagramElements:
        "The SVG diagrams the projection from one-hot indices through the embedding weight lookup table W into the dense embedding space.",
      canvasMechanics:
        "The canvas plots the 2D projected semantic coordinate space. When you step through, it draws dashed relational parallelogram vectors connecting analogies (king → man vs queen → woman; paris → france vs rome → italy), displaying the live computed cosine similarity.",
      colorSignaling:
        "• Blue points: Query and subject words\n• Gold dashed arrow: Semantic translation vector encoding gender or capital city\n• Teal: Target analogy match"
    },
    pedagogicalTakeaways: [
      "The Distributional Hypothesis (J.R. Firth, 1957): 'You shall know a word by the company it keeps.' Word2Vec relies entirely on co-occurrence statistics in local sliding windows.",
      "Negative sampling is essential because computing the denominator of the full softmax requires summing over all |V| words (e.g. 500,000 terms) at every single training step, which is computationally intractable."
    ]
  },

  classification: {
    title: "Text Classification, Softmax Calibration & Dropout Regularization",
    syllabusContext: "Harvard CSCI E-89b Week 9 & CSCI E-89 Week 2",
    problemStatement:
      "Deep text classifiers with hundreds of thousands of parameters easily overfit to small training corpora, memorizing spurious keyword correlations and producing overconfident, uncalibrated probability predictions. Dropout regularization and softmax temperature scaling prevent co-adaptation and produce reliable predictive uncertainties.",
    mathematicalMechanics: {
      forwardPass:
        "1. Linear Logits: z = W · x + b\n2. Dropout Mask: r ~ Bernoulli(1 - p);  x̃ = (r ⊙ x) / (1 - p)\n3. Categorical Cross-Entropy Loss: L = - Σ_{c=1}^C y_c · log(p_c)\n4. Softmax Probabilities: p_c = exp(z_c / T) / Σ_{j=1}^C exp(z_j / T)",
      lossAndUpdate:
        "Inverted Dropout scales active activations by 1 / (1 - p) during training so that no scaling or modification is required at test/inference time, keeping expected activation magnitudes identical across both phases.",
      tensorShapes: "Features x: [1, D] → Dropout Mask: [1, D] → Hidden: [1, H] → Output Classes: [1, C]."
    },
    visualGuide: {
      diagramElements:
        "The SVG highlights the active classification head with dashed nodes indicating dropped units during training passes.",
      canvasMechanics:
        "The canvas renders horizontal probability bars for each target category (e.g. Politics, Sports, Technology, Business). As you step forward, it visualizes the shift in predicted class distribution and calibration confidence intervals.",
      colorSignaling:
        "• Teal bars: Predicted posterior class probabilities\n• Crimson highlight: Top-predicted argmax category\n• Dashed gray: Suppressed units during dropout phase"
    },
    pedagogicalTakeaways: [
      "A neural network with high accuracy can still be horribly calibrated; modern deep networks often output 99% confidence on examples where empirical accuracy is only 70%. Temperature scaling T > 1 softens overconfident distributions.",
      "Dropout can be interpreted as training an exponential ensemble of 2^N thinned sub-networks with shared weights, effectively performing Bayesian model averaging at test time."
    ]
  },

  ner: {
    title: "Named Entity Recognition (NER), BiLSTM & BIO Tagging",
    syllabusContext: "Harvard CSCI E-89b Week 10",
    problemStatement:
      "Unlike sentence classification which outputs a single label per document, sequence tagging must emit an entity label for every individual token while preserving boundary structure. A simple word classifier cannot distinguish 'Paris' the city from 'Paris' Hilton; contextual bidirectional sequence models resolve ambiguity using both left-to-right and right-to-left contexts.",
    mathematicalMechanics: {
      forwardPass:
        "1. Forward LSTM: h_t^{fwd} = LSTM_fwd(x_t, h_{t-1}^{fwd})\n2. Backward LSTM: h_t^{bwd} = LSTM_bwd(x_t, h_{t+1}^{bwd})\n3. Concatenation: h_t = [h_t^{fwd}; h_t^{bwd}]\n4. Emission Logits: e_t = W_tag · h_t + b_tag\n5. BIO Transition Score: s(X, y) = Σ_{t=1}^T (E_{t, y_t} + A_{y_{t-1}, y_t})",
      lossAndUpdate:
        "CRF sequence loss maximizes the log-likelihood of the entire valid tag path rather than independent token decisions, using the Viterbi dynamic programming algorithm to find argmax_y s(X, y) in O(T · |Tags|^2) time.",
      tensorShapes: "Token embeddings: [T, d] → BiLSTM Hidden: [T, 2 · h] → Tag Logits: [T, |Tags|]."
    },
    visualGuide: {
      diagramElements:
        "The SVG illustrates the bidirectional information highway: forward hidden states flowing left-to-right and backward hidden states flowing right-to-left, concatenating at each token node.",
      canvasMechanics:
        "The canvas steps token-by-token across the sentence, highlighting the active token and displaying its assigned BIO tag badge alongside its classification confidence score.",
      colorSignaling:
        "• Crimson badge (B-): Begin named entity token (e.g. B-PER, B-ORG)\n• Amber badge (I-): Inside multi-token named entity (e.g. I-PER, I-ORG)\n• Slate badge (O): Outside / neutral non-entity token"
    },
    pedagogicalTakeaways: [
      "Standard token-level softmax classification allows illegal transitions like 'O followed by I-PER' (you cannot be inside an entity that never began); a Linear-Chain CRF on top of a BiLSTM guarantees global transition consistency.",
      "Bidirectional context is essential: in 'Apple reported earnings', 'Apple' is an organization; in 'Apple tastes sweet', 'Apple' is food. Only downstream words reveal the correct entity type."
    ]
  },

  gan: {
    title: "Generative Adversarial Networks (GANs) & Minimax Game Theory",
    syllabusContext: "Harvard CSCI E-89b Week 11 & CSCI E-89 Week 8",
    problemStatement:
      "Traditional generative models require explicit parametric probability density functions (like maximum likelihood), which frequently blur generated outputs when approximating complex multi-modal distributions. GANs abandon explicit density estimation, framing generation as an adversarial game between a Generator network (forger) and a Discriminator network (detective).",
    mathematicalMechanics: {
      forwardPass:
        "1. Prior Noise Sampling: z ~ p_z(z) (e.g. Standard Gaussian N(0, I))\n2. Synthetic Sample Generation: x_fake = G(z; θ_g)\n3. Discriminator Evaluation: D(x_real) → [0, 1]  and  D(x_fake) → [0, 1]\n4. Minimax Objective Function:\n   min_G max_D V(D, G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))]",
      lossAndUpdate:
        "Discriminator Step: Ascent along ∇_{θ_d} [log D(x) + log(1 - D(G(z)))]\nGenerator Step: Descent along ∇_{θ_g} [log(1 - D(G(z)))], or practically maximizing log D(G(z)) to eliminate early saturation.",
      tensorShapes: "Latent noise z: [1, 100] → Generator G(z): [1, 784] → Discriminator D(x): [1, 1] probability."
    },
    visualGuide: {
      diagramElements:
        "The SVG displays the dual competing architectures: Noise z feeding into the Generator, and both Real data and Synthetic G(z) feeding into the Discriminator.",
      canvasMechanics:
        "The canvas depicts the adversarial arena: the real data distribution curve in green, the generator's synthetic distribution curve in teal, and the discriminator's decision boundary separating them in crimson. As steps advance, notice G shifting its distribution to overlap p_data until D can no longer distinguish them (D(x) = 0.5).",
      colorSignaling:
        "• Green: Ground truth real data distribution p_data\n• Teal: Generator synthetic distribution p_g\n• Crimson: Discriminator boundary scoring real (1) vs fake (0)"
    },
    pedagogicalTakeaways: [
      "At theoretical Nash Equilibrium, the generator perfectly replicates the true data distribution (p_g = p_data) and the discriminator outputs 0.5 everywhere (pure guessing).",
      "Mode collapse is a classic failure mode where the generator learns to output only a single high-probability sample (like only generating digit '1') that fools the discriminator, ignoring all other classes."
    ]
  },

  lda: {
    title: "Topic Modeling: Latent Dirichlet Allocation (LDA) & NMF",
    syllabusContext: "Harvard CSCI E-89b Week 7",
    problemStatement:
      "Large text corpora contain millions of unlabelled documents spanning diverse themes. LDA solves unsupervised thematic discovery by modeling every document as a probabilistic mixture of latent topics, and every topic as a probabilistic distribution over vocabulary words.",
    mathematicalMechanics: {
      forwardPass:
        "1. For each topic k ∈ {1, ..., K}: Sample word distribution β_k ~ Dirichlet(η)\n2. For each document d ∈ {1, ..., M}: Sample topic mixture θ_d ~ Dirichlet(α)\n3. For each word token n in document d:\n   Sample topic assignment z_{d,n} ~ Multinomial(θ_d)\n   Sample observed word w_{d,n} ~ Multinomial(β_{z_{d,n}})",
      lossAndUpdate:
        "Collapsed Gibbs Sampling Update Rule:\n   P(z_i = k | z_{-i}, w, α, η) ∝ (n_{k,-i}^{(v)} + η) / (n_{k,-i}^{(·)} + V · η) · (n_{d,-i}^{(k)} + α) / (n_{d,-i}^{(·)} + K · α)",
      tensorShapes: "Doc-Topic Matrix θ: [M, K] → Topic-Word Matrix β: [K, V] → Corpus: [M, V]."
    },
    visualGuide: {
      diagramElements:
        "The SVG visualizes the hierarchical Bayesian plate notation: hyper-priors α and η generating document topic mixtures θ and topic word distributions β.",
      canvasMechanics:
        "The canvas plots the active discovered topics with their highest-probability word lists and weights. Stepping forward performs a Gibbs sampling sweep, reassigning words to topics based on topic co-occurrences.",
      colorSignaling:
        "• Teal: Technical / AI topic words\n• Purple: Financial / Risk topic words\n• Gold: Healthcare / Clinical topic words"
    },
    pedagogicalTakeaways: [
      "The Dirichlet hyperparameter α controls document topic sparsity: small α < 1 implies documents contain only 1 or 2 dominant topics; large α > 1 implies documents are a uniform blend of all topics.",
      "LDA is completely unsupervised: it groups co-occurring words into clusters, but human analysts must assign semantic labels (e.g. naming Topic 1 'Machine Learning')."
    ]
  },

  stm: {
    title: "Structural Topic Models (STM) & Document Covariates",
    syllabusContext: "Harvard CSCI E-89b Week 8",
    problemStatement:
      "Standard LDA assumes all documents in a corpus share the identical topic prior α regardless of who wrote them, when they were written, or what political party the author belongs to. Structural Topic Models (STM) allow document-level metadata (covariates X) to directly affect both topic prevalence (how much a topic is discussed) and topical content (what words are used to discuss it).",
    mathematicalMechanics: {
      forwardPass:
        "1. Topic Prevalence: θ_d ~ LogisticNormal(X_d · Γ, Σ)\n2. Topical Content: β_{k,d,v} = exp(m_v + κ_{k,v} + κ_{y_d,v} + κ_{k,y_d,v}) / Σ exp(...)\n3. Word Emission: w_{d,n} ~ Multinomial(β_{z_{d,n}, d})",
      lossAndUpdate:
        "Variational Expectation-Maximization (V-EM): E-step estimates document-specific topic mixtures θ_d; M-step solves generalized linear regressions of prevalence coefficients Γ and content deviations κ.",
      tensorShapes: "Covariates X: [M, P] → Prevalence Coefficients Γ: [P, K] → Document Mixtures θ: [M, K]."
    },
    visualGuide: {
      diagramElements:
        "The SVG highlights the metadata conditioning paths feeding directly into topic prevalence and content matrices.",
      canvasMechanics:
        "The canvas renders the regression effect curves showing how topic prevalence shifts across metadata variables (such as publication year or author political affiliation).",
      colorSignaling:
        "• Blue: Covariate metadata inputs\n• Teal: Topic prevalence distribution\n• Gold: Vocabulary content shifts"
    },
    pedagogicalTakeaways: [
      "STM bridges NLP with social science and econometrics, allowing researchers to test rigorous statistical hypotheses (e.g. 'Did discussion of monetary policy increase significantly after the 2008 financial crisis?').",
      "Topical content covariates allow the vocabulary of a topic to change by author group: for instance, in a topic on 'Healthcare', Democrats might use 'access and coverage' while Republicans use 'cost and mandate'."
    ]
  }
};
