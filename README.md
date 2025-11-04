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

**Context Types Fed to GPT-4:**

1. Captions: Multiple descriptions from different perspectives

"A group of people standing outside of a black vehicle with various luggage."
   "People try to fit all of their luggage in an SUV."
   "The sport utility vehicle is parked in the public garage, being packed for a trip"
```

2. **Bounding boxes:** Object locations and labels
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
       
2. 
3. 







