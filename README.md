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

"""
def project_visual_features(X_v, W):

    Projects visual features to language embedding space.
    
    Input: X_v ∈ ℝ^(d_v), visual features from CLIP encoder
    Input: W ∈ ℝ^(d_e × d_v), projection matrix
    Output: H_v ∈ ℝ^(d_e), language-aligned visual tokens
    """
    Z_v = vision_encoder(X_v)  # Extract CLIP features
    H_v = W @ Z_v              # Project to embedding space
    return H_v


### ALGORITHM 2: LLaVA Forward Pass

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


### ALGORITHM 3: Two-Stage Training

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


### ALGORITHM 4: Inference (Visual Chatbot)

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


### Key Architectural Details

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








