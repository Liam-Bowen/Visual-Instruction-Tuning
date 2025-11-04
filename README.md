# Visual-Instruction-Tuning

## Visual Instruction Tuning (LLaVa)

 **Paper:** Visual Instruction Tuning
 **Authors:** Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee
 **Institutions:** University of Wisconsin-Madison, Miscrosoft Research, Columbia University
 **Conference:** NeurIPS 2023
 **Paper Link:** https://arxiv.org/abs/2304.08485

 ## Overview

 This paper introduces LLava (Large Language and Vision Assistant), the first multimodal model to leverage GPT-4 for generating visual instruction-following data. The key innovation is using language-only GPT-4 to create high-quality multimodal training data by providing symbolic representations of images (captions and bounding boxes), then using this data to instruction-tune a vision-language model.

 **Core Problem:** While instruction tuning has dramatically improved language models, the multimodal space lacks similar approaches due to scarcity of vision-language instruction-following data.

 **Approach:** The authors bridge vision and language by connecting a CLIP vision encoder with the Vicuna language model, training on GPT-4 generated multimodal instruction data.

 **Key Results:** 
  - 85.1% relative score compared to GPT-4 on multimodal instruction-following.
  - State-of-the-art 92.53% accuracy on ScienceQA when combined with GPT-4.
  - Impressive zero-shot generalization on unseen images and instructions.

## Question 1: Data Scarcity Challenge

**The paper identifies that "the available amount of multimodal instruction-following data is limited." Why is creating this type of data particularly challenging compared to single-modality instruction data, and how does this scarcity impact multimodal model development?**

<details><summary>Click to reveal answer</summary>
Creating multimodal instruction-following data is challenging for several reasons:
 
 1. **Complexity of annotation:** Unlike text-only instructions, annotators must:
   - View and understand images
   - Generate diverse, meaningful questions about visual content
   - Provide detailed, accurate answers
   - Ensure questions require actual visual understanding (not answerable from world knowledge alone)

 2. **Time and cost:** Each multimodal example requires significantly more human effort than text-only data. Annotators must carefully examine images, think of appropriate questions, and craft responses that demonstrate genuine visual reasoning.
    
 3. **Quality control:** It's harder to verify correctness—does the answer actually match what's in the image? Does the question require visual information?

 4. **Scale limitations:** These factors make it expensive and slow to scale up, unlike web-scraped image-text pairs.

 **Impact on model development:**
  - Models trained only on image-text pairs (like CLIP, BLIP) can describe images but struggle with instruction-following
  - Without instruction data, models can't learn to respond appropriately to user queries ("What color is the car?" vs "Describe this image in detail")
  - This scarcity motivated LLaVA's key innovation: using GPT-4 to synthetically generate instruction data from existing image-caption pairs
</details>

## Data Generation: From Image-Text Pairs to Instructions

The authors' solution to data scarcity is to leverage GPT-4's reasoning capabilities while working around its inability to see images. They use symbolic representations:

### Context Types Fed to GPT-4:

**1. Captions:** Multiple descriptions from different perspectives

```
"A group of people standing outside of a black vehicle with various luggage."
"People try to fit all of their luggage in an SUV."
"The sport utility vehicle is parked in the public garage, being packed for a trip"
```

**2. Bounding boxes:** Object locations and labels
```
person: [0.681, 0.242, 0.774, 0.694]
backpack: [0.384, 0.696, 0.485, 0.914]
suitcase: [0.758, 0.413, 0.845, 0.69]
```

### Three Types of Generated Responses:

1. **Conversations (58K samples):** Multi-turn Q&A about visual content
   - Object types, counting, actions, locations, relative positions
   - Only questions with definite answers from the image

2. **Detailed descriptions (23K samples):** Comprehensive image descriptions
   - Rich, paragraph-length descriptions
   - Multiple perspectives and aspects

3. **Complex reasoning (77K samples):** In-depth logical reasoning
   - Requires step-by-step reasoning
   - Background knowledge application
   - Cause-and-effect analysis

**Total: 158K unique language-image instruction-following samples**

**Key finding:** GPT-4 consistently provides higher quality data than ChatGPT, especially for spatial reasoning tasks.

---

## Architecture Overview

### High-Level Design

LLaVA connects three key components:
```
┌─────────────────┐      ┌──────────────┐      ┌─────────────────┐
│  Vision Encoder │ ──▶  │  Projection  │ ──▶  │   Language LLM  │
│   (CLIP ViT)    │      │   Matrix W   │      │    (Vicuna)     │
└─────────────────┘      └──────────────┘      └─────────────────┘
       Frozen             Trainable Stage 1      Trainable Stage 2
```
       
## Component Details

### Vision Encoder:

 - CLIP ViT-L/14 pre-trained model
 - Extracts grid features before the last transformer layer
 - Why before last layer? Better captures localized properties useful for understanding specific image details
 - Output: Z_v ∈ ℝ^(d_v) where d_v = 1024

### Projection Layer:
 - Simple trainable linear transformation: H_v = W · Z_v
 - W ∈ ℝ^(4096 × 1024)
 - Maps visual features into language model's embedding space
 - Authors note: "lightweight" design allows rapid iteration; more sophisticated schemes (gated cross-attention, Q-former) are future work

### Language Model:

 - Vicuna-13B (instruction-tuned LLaMA)
 - Chosen for "best instruction following capabilities in language tasks among publicly available checkpoints"
 - Standard decoder-only transformer with causal attention
 - 32 layers, 32 attention heads, embedding dimension 4096

## Training Methodology

### Two-Stage Training Process


LLaVA: Visual Instruction Tuning Architecture
Based on the formal algorithm style from Phuong & Hutter (2022)

Notation follows transformer conventions:
- Matrices: bold uppercase (e.g., W, X)
- Vectors: bold lowercase (e.g., v, h)
- Scalars: regular font (e.g., d, L)
- Sequences: x[1:T] denotes tokens from position 1 to T


### ALGORITHM 1: Vision-Language Projection

```python
def project_visual_features(X_v, W):
    """
    Projects visual features to language embedding space.
    
    Input: X_v ∈ ℝ^(d_v), visual features from CLIP encoder
    Input: W ∈ ℝ^(d_e × d_v), projection matrix
    Output: H_v ∈ ℝ^(d_e), language-aligned visual tokens
    """
    Z_v = vision_encoder(X_v)  # Extract CLIP features
    H_v = W @ Z_v              # Project to embedding space
    return H_v
```


### ALGORITHM 2: LLaVA Forward Pass

```python
def LLaVA_forward(X_v, X_instruct, θ):
    """
    Complete forward pass through LLaVA model.
    
    Input: X_v, input image
    Input: X_instruct, instruction sequence [X_q, X_v] or [X_v, X_q]
    Input: θ, all model parameters {W, φ} where φ are LLM parameters
    Output: P(X_a | X_v, X_instruct), probability distribution over answers
    
    Hyperparameters:
        d_e: embedding dimension (typically 4096)
        L: number of transformer layers (typically 32)
        H: number of attention heads (typically 32)
    """
    # Step 1: Extract and project visual features
    H_v = project_visual_features(X_v, θ.W)
    
    # Step 2: Embed instruction tokens
    H_instruct = embed_tokens(X_instruct, θ.W_e, θ.W_p)
    
    # Step 3: Concatenate visual and text embeddings
    H_input = concatenate(H_v, H_instruct)
    
    # Step 4: Process through language model (Vicuna/LLaMA)
    X = H_input
    for l in range(1, L+1):
        X_norm = layer_norm(X, θ.γ₁ˡ, θ.β₁ˡ)
        X = X + MHAttention(X_norm, X_norm, θ.W_l, causal_mask=True)
        
        X_norm = layer_norm(X, θ.γ₂ˡ, θ.β₂ˡ)
        X = X + MLP(X_norm, θ.W_mlp_l)
    
    # Step 5: Final layer norm and unembedding
    X = layer_norm(X, θ.γ, θ.β)
    P = softmax(θ.W_u @ X)
    
    return P
```

### ALGORITHM 3: Two-Stage Training

``` python
def train_LLaVA(data_pretrain, data_instruct, θ_init):
    """
    Two-stage training procedure for LLaVA.
    
    Input: data_pretrain, CC-595K image-caption pairs
    Input: data_instruct, 158K GPT-4 generated instruction data
    Input: θ_init, initial parameters
    Output: θ_final, trained parameters
    """
    θ = θ_init
    
    # ---- STAGE 1: Pre-training for Feature Alignment ----
    # Only train projection matrix W, freeze vision encoder and LLM
    trainable_params = {θ.W}
    freeze_params({θ.vision_encoder, θ.LLM})
    
    for epoch in range(1):  # 1 epoch
        for (X_v, caption) in data_pretrain:
            # Create simple instruction-following format
            X_q = random_sample([
                "Describe the image concisely.",
                "Provide a brief description.",
                "What is in this image?"
            ])
            X_instruct = [X_q, X_v]
            X_a = caption  # Ground truth answer
            
            # Forward pass
            P = LLaVA_forward(X_v, X_instruct, θ)
            
            # Compute loss (cross-entropy)
            loss = -log(P[X_a])
            
            # Update only projection matrix
            θ.W = θ.W - η * ∇_W(loss)
    
    # ---- STAGE 2: Fine-tuning End-to-End ----
    # Train projection matrix W and LLM parameters φ
    trainable_params = {θ.W, θ.LLM}
    freeze_params({θ.vision_encoder})
    
    for epoch in range(3):  # 3 epochs
        for (X_v, conversations) in data_instruct:
            # Multi-turn conversation format
            for turn_t in conversations:
                if turn_t == 1:
                    # First turn: include image
                    X_instruct = random_order([X_v, X_q_t])
                else:
                    # Subsequent turns: text only
                    X_instruct = X_q_t
                
                # Forward pass
                P = LLaVA_forward(X_v, X_instruct, θ)
                
                # Compute loss on answer tokens only
                loss = -sum([log(P[x_i]) for x_i in X_a_t])
                
                # Update W and LLM parameters
                θ.W, θ.LLM = θ.W - η * ∇(loss), θ.LLM - η * ∇(loss)
    
    return θ
```

### ALGORITHM 4: Inference (Visual Chatbot)

``` python
def LLaVA_inference(X_v, prompt, θ, max_tokens=512):
    """
    Generate response given image and text prompt.
    
    Input: X_v, input image
    Input: prompt, text instruction/question
    Input: θ, trained model parameters
    Input: max_tokens, maximum response length
    Output: response, generated text
    """
    # Initialize with prompt
    X_instruct = [X_v, prompt]
    response = []
    
    for t in range(max_tokens):
        # Get next token distribution
        P = LLaVA_forward(X_v, X_instruct, θ)
        
        # Sample next token (can use temperature, top-p, etc.)
        x_next = sample(P[:, -1])  # Get distribution for last position
        
        if x_next == EOS_TOKEN:
            break
        
        response.append(x_next)
        X_instruct.append(x_next)
    
    return detokenize(response)
```

### Key Architectural Details

```
Vision Encoder:
    - CLIP ViT-L/14
    - Uses grid features BEFORE last transformer layer (better for details)
    - Output dimension: d_v = 1024

Projection:
    - Simple linear layer: W ∈ ℝ^(4096 × 1024)
    - Maps vision features to Vicuna embedding space
    
Language Model:
    - Vicuna-13B (instruction-tuned LLaMA)
    - Decoder-only transformer with causal attention
    - Embedding dimension: d_e = 4096
    - Layers: L = 32, Heads: H = 32

Training:
    - Stage 1: 4 hours on 8×A100, lr=2e-3, batch=128
    - Stage 2: 10 hours on 8×A100, lr=2e-5, batch=32
    - Optimizer: AdamW with cosine learning rate schedule
```

### Stage 1: Pre-training for Feature Alignment

 - **Dataset:** 595K filtered CC3M image-caption pairs
 - **Trainable:** Only projection matrix W
 - **Frozen:** Vision encoder + LLM
 - **Duration:** 1 epoch (~4 hours on 8×A100)
 - **Objective:** Align visual features with pre-trained LLM word embeddings
 - **Format:** Simple instruction-following (e.g., "Describe the image concisely." → caption)
 - **Learning rate:** 2e-3, batch size: 128

### Stage 2: Fine-tuning End-to-End

 - **Dataset:** 158K GPT-4 generated instruction data
 - **Trainable:** Projection matrix W + LLM parameters φ
 - **Frozen:** Vision encoder
 - **Duration:** 3 epochs (~10 hours on 8×A100)
 - **Objective:** Learn instruction-following behavior with visual grounding
 - **Format:** Multi-turn conversations + detailed descriptions + complex reasoning
 - **Learning rate:** 2e-5, batch size: 32

## Multi-Turn Conversation Format

For a sequence with T turns: (X_q^1, X_a^1, ..., X_q^T, X_a^T)

```
<SYSTEM_MESSAGE> ###
Human: [X_q^1, X_v] ### Assistant: X_a^1 ###
Human: X_q^2 ### Assistant: X_a^2 ###
...
```
- **First turn:** Image X_v randomly placed before or after question
- **Subsequent turns:** Text-only (image context maintained)
- **Training:** Predict only assistant responses (green tokens), ignore human prompts
- **Loss:** Cross-entropy on answer tokens only

### Why Two Stages?

Ablation results show stage 1 is crucial:
 - Training from scratch (no pre-training): 85.81% accuracy on ScienceQA
 - With pre-training: 90.92% accuracy
 - 5.11% absolute improvement demonstrates importance of alignment

The two-stage approach:
 1. Learns a "compatible visual tokenizer" for the frozen LLM
 2. Preserves vast pre-trained knowledge while integrating visual understanding
 3. Avoids catastrophic forgetting of language capabilities

## Question 2: Architecture Design Choices

The authors use grid features from before the last CLIP transformer layer rather than the final layer output. They also choose a simple linear projection over more complex alternatives like Flamingo's gated cross-attention or BLIP-2's Q-former. What motivates these seemingly simpler design choices?

<details><summary>Click to reveal answer</summary>

**Grid features before last layer:**

The authors found that using features before the last CLIP layer yields better performance:

 - ScienceQA accuracy: 90.92% (before last) vs 89.96% (last layer)
 - 0.96% improvement

**Reasoning:** CLIP's last layer focuses on global, abstract image properties optimized for contrastive learning (matching images to captions). The layer before captures more localized properties useful for understanding specific image details—exactly what's needed for answering detailed questions about image content.

**Simple linear projection:**
The authors explicitly chose simplicity over sophistication:

 1. Rapid iteration: "Lightweight, which allows us to iterate data centric experiments quickly"
 2. Data-centric focus: Wanted to validate the instruction-tuning approach without confounding variables from complex architectures
 3. Effectiveness: Despite simplicity, achieved 85.1% of GPT-4's performance on instruction-following

Future work acknowledged: "More sophisticated schemes to connect image and language representations can also be considered, such as gated cross-attention in Flamingo and Q-former in BLIP-2. We leave exploring possibly more effective and sophisticated architecture designs for LLaVA as future work."
This reflects a research philosophy: validate the core idea (instruction tuning for multimodal models) with the simplest architecture first, then optimize later. Indeed, LLaVA-1.5 later improved the projection design while keeping the overall approach.
</details>

## Formal Algorithm
```python
"""
LLaVA: Visual Instruction Tuning Architecture
Based on the formal algorithm style from Phuong & Hutter (2022)

Notation follows transformer conventions:
- Matrices: bold uppercase (e.g., W, X)
- Vectors: bold lowercase (e.g., v, h)
- Scalars: regular font (e.g., d, L)
- Sequences: x[1:T] denotes tokens from position 1 to T
"""

# ============================================================================
# ALGORITHM 1: Vision-Language Projection
# ============================================================================
def project_visual_features(X_v, W):
    """
    Projects visual features to language embedding space.
    
    Input: X_v ∈ ℝ^(d_v), visual features from CLIP encoder
    Input: W ∈ ℝ^(d_e × d_v), projection matrix
    Output: H_v ∈ ℝ^(d_e), language-aligned visual tokens
    """
    Z_v = vision_encoder(X_v)  # Extract CLIP features
    H_v = W @ Z_v              # Project to embedding space
    return H_v


# ============================================================================
# ALGORITHM 2: LLaVA Forward Pass
# ============================================================================
def LLaVA_forward(X_v, X_instruct, θ):
    """
    Complete forward pass through LLaVA model.
    
    Input: X_v, input image
    Input: X_instruct, instruction sequence [X_q, X_v] or [X_v, X_q]
    Input: θ, all model parameters {W, φ} where φ are LLM parameters
    Output: P(X_a | X_v, X_instruct), probability distribution over answers
    
    Hyperparameters:
        d_e: embedding dimension (typically 4096)
        L: number of transformer layers (typically 32)
        H: number of attention heads (typically 32)
    """
    # Step 1: Extract and project visual features
    H_v = project_visual_features(X_v, θ.W)
    
    # Step 2: Embed instruction tokens
    H_instruct = embed_tokens(X_instruct, θ.W_e, θ.W_p)
    
    # Step 3: Concatenate visual and text embeddings
    # For first turn: randomly choose [H_v, H_instruct] or [H_instruct, H_v]
    H_input = concatenate(H_v, H_instruct)
    
    # Step 4: Process through language model (Vicuna/LLaMA)
    # This is a standard decoder-only transformer
    X = H_input
    for l in range(1, L+1):
        # Layer norm + masked self-attention
        X_norm = layer_norm(X, θ.γ₁ˡ, θ.β₁ˡ)
        X = X + MHAttention(X_norm, X_norm, θ.W_l, causal_mask=True)
        
        # Layer norm + MLP
        X_norm = layer_norm(X, θ.γ₂ˡ, θ.β₂ˡ)
        X = X + MLP(X_norm, θ.W_mlp_l)
    
    # Step 5: Final layer norm and unembedding
    X = layer_norm(X, θ.γ, θ.β)
    P = softmax(θ.W_u @ X)
    
    return P


# ============================================================================
# ALGORITHM 3: Two-Stage Training
# ============================================================================
def train_LLaVA(data_pretrain, data_instruct, θ_init):
    """
    Two-stage training procedure for LLaVA.
    
    Input: data_pretrain, CC-595K image-caption pairs
    Input: data_instruct, 158K GPT-4 generated instruction data
    Input: θ_init, initial parameters
    Output: θ_final, trained parameters
    
    Training details:
        Stage 1: 1 epoch, lr=2e-3, batch=128, ~4 hours on 8×A100
        Stage 2: 3 epochs, lr=2e-5, batch=32, ~10 hours on 8×A100
        Optimizer: AdamW with cosine learning rate schedule
        Mixed precision: BF16 + TF32 for speed/precision balance
    """
    θ = θ_init
    
    # ---- STAGE 1: Pre-training for Feature Alignment ----
    # Goal: Learn a compatible visual tokenizer for the frozen LLM
    # Only train projection matrix W, freeze vision encoder and LLM
    trainable_params = {θ.W}
    freeze_params({θ.vision_encoder, θ.LLM})
    
    for epoch in range(1):  # 1 epoch
        for (X_v, caption) in data_pretrain:
            # Create simple instruction-following format
            X_q = random_sample([
                "Describe the image concisely.",
                "Provide a brief description.",
                "What is in this image?",
                "Summarize the visual content.",
                # ... (11 total variations for diversity)
            ])
            X_instruct = [X_q, X_v]
            X_a = caption  # Ground truth answer
            
            # Forward pass
            P = LLaVA_forward(X_v, X_instruct, θ)
            
            # Compute loss (cross-entropy) on caption tokens
            loss = -sum([log(P[token]) for token in X_a])
            
            # Update only projection matrix
            θ.W = θ.W - η₁ * ∇_W(loss)  # η₁ = 2e-3
    
    # ---- STAGE 2: Fine-tuning End-to-End ----
    # Goal: Learn instruction-following with visual grounding
    # Train projection matrix W and LLM parameters φ, keep vision frozen
    trainable_params = {θ.W, θ.LLM}
    freeze_params({θ.vision_encoder})
    
    for epoch in range(3):  # 3 epochs
        for (X_v, data_sample) in shuffle(data_instruct):
            # Sample uniformly from three types:
            # - Conversations (multi-turn)
            # - Detailed descriptions (single-turn)
            # - Complex reasoning (single-turn)
            
            conversations = format_as_conversation(data_sample)
            
            # Multi-turn conversation format
            for turn_t in conversations:
                if turn_t == 1:
                    # First turn: include image (random position)
                    X_instruct_t = random_order([X_v, X_q_t])
                else:
                    # Subsequent turns: text only
                    X_instruct_t = X_q_t
                
                # Forward pass
                P = LLaVA_forward(X_v, X_instruct_t, θ)
                
                # Compute loss ONLY on answer tokens (not instruction)
                loss = -sum([log(P[x_i]) for x_i in X_a_t])
                
                # Update W and LLM parameters
                gradients = compute_gradients(loss, {θ.W, θ.LLM})
                θ.W = θ.W - η₂ * gradients.W      # η₂ = 2e-5
                θ.LLM = θ.LLM - η₂ * gradients.LLM
    
    return θ


# ============================================================================
# ALGORITHM 4: Inference (Visual Chatbot)
# ============================================================================
def LLaVA_inference(X_v, prompt, θ, max_tokens=512):
    """
    Generate response given image and text prompt.
    
    Input: X_v, input image
    Input: prompt, text instruction/question
    Input: θ, trained model parameters
    Input: max_tokens, maximum response length
    Output: response, generated text
    """
    # Initialize with prompt
    X_instruct = [X_v, prompt]
    response = []
    
    for t in range(max_tokens):
        # Get next token distribution
        P = LLaVA_forward(X_v, X_instruct, θ)
        
        # Sample next token (can use temperature, top-p, etc.)
        x_next = sample(P[:, -1])  # Get distribution for last position
        
        if x_next == EOS_TOKEN:
            break
        
        response.append(x_next)
        X_instruct.append(x_next)
    
    return detokenize(response)


# ============================================================================
# Key Architectural Details
# ============================================================================
"""
Vision Encoder:
    - CLIP ViT-L/14
    - Uses grid features BEFORE last transformer layer (better for details)
    - Output dimension: d_v = 1024

Projection:
    - Simple linear layer: W ∈ ℝ^(4096 × 1024)
    - Maps vision features to Vicuna embedding space
    
Language Model:
    - Vicuna-13B (instruction-tuned LLaMA)
    - Decoder-only transformer with causal attention
    - Embedding dimension: d_e = 4096
    - Layers: L = 32, Heads: H = 32

Training:
    - Stage 1: 4 hours on 8×A100, lr=2e-3, batch=128
    - Stage 2: 10 hours on 8×A100, lr=2e-5, batch=32
    - Optimizer: AdamW with cosine learning rate schedule
"""
```

## Experimental Results

### Multimodal Chatbot Performance

**LLaVA-Bench (COCO):** 30 images, 90 questions (conversation, detailed description, complex reasoning)

 - LLaVA: 85.1% relative to GPT-4 (text-only with ground truth captions)
 - Shows strong alignment with GPT-4's responses despite never seeing actual GPT-4 visual outputs

**LLaVA-Bench (In-the-Wild):** 24 diverse images, 60 questions (challenging generalization test)

| Model |  Conversation  | Detail | Complex Reasoning | Overall |
|:-----|:--------:|------:| ------:| ------:|
| OpenFlamingo   | 19.3±0.5 | 19.0±0.5 | 19.1±0.7 | 19.1±0.4 |
| BLIP-2   |  54.6±1.4  |   29.1±1.2 | 32.9±0.7 | 38.1±1.0 |
| LLaVA   | 58.8±0.6 | 49.2±0.8 | 81.7±0.3 | 66.7±0.3 |

**Key insight:** LLaVA achieves 81.7% on complex reasoning—close to GPT-4's ceiling despite GPT-4 having access to ground truth captions.

### ScienceQA: Multimodal Reasoning

| Method |  NAT | SOC | LAN | TXT | IMG | NO | G1-6| G7-12| AVG|
|:-----|:--------:|------:| ------:| ------:|  ------:|  ------:|  ------:|  ------:|  ------:|
| Human   | 90.23 | 84.97 | 87.48 | 89.60 | 87.50 | 88.10 | 91.59 | 82.42| 88.40 |
| GPT-3.5 + CoT   |  75.44  |   70.87 | 78.09 | 74.68 | 67.43 | 79.93 | 78.23 | 69.68| 75.17|
| MM-CoT (Large)  | 95.91 | 82.00 | 90.82 | 95.26 | 88.80 | 92.89 | 92.44 | 90.31 | 91.68|
| LLaVA | 90.36 | 95.95 | 88.00| 89.49 | 88.00| 90.66 | 90.93 | 90.90 | 90.92 |
| LLaVA + GPT-4 (judge) | 91.56 | 96.74 | 91.09 | 90.62 | 88.99 | 93.52 | 92.73 | 92.16 | 92.53 |

**Novel finding:** Text-only GPT-4 can improve multimodal performance by acting as a "judge" to ensemble predictions—first use of GPT-4 for model ensembling.

### Ablation Studies

Impact of training data types (on COCO benchmark):
| Training Data | Conv | Detail | Complex | Overall |
|:-----|:-----|:-----|:-----|:-----|
| Full Data | 83.1 | 75.3 | 96.5 | 85.1 |
| Detail + Complex | 81.5 | 73.3 | 90.8 | 81.9 (-3.2) |
| Conv Only | 76.5 | 59.8 | 84.9 | 73.8 (-11.3) |
| No Instruction Tuning | 22.0 | 24.0 | 18.5 | 21.5 | (-63.6) |

**Takeaway:** All three data types contribute; instruction tuning is essential (>60 point improvement).

## Model Design Choices (ScienceQA):

| Variant | Accuracy | Δ |
|:-----|:-----|:-----|
|Best (before last layer, reasoning-first) | 90.92% | - |
|Last layer features | 89.96% | -0.96% |
| Answer-first (no CoT) | 89.77% | -1.15% |
| No pre-training (stage 1) | 85.81% | -5.11% |
| 7B model (vs 13B) | 89.84% | -1.08% |


