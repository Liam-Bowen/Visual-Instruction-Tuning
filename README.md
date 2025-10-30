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

## Question 1: Data Generation Strategy
