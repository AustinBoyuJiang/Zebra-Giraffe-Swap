
# Giraffe-Zebra Concept Swapping: A Detailed Stable Diffusion Fine-tuning Solution

<img src="a.png" width="450">

## Problem Analysis & Solution Overview

The core challenge of this project is achieving **selective concept remapping** in a pre-trained diffusion model. We need to swap only two specific concepts (giraffe ↔ zebra) while preserving all other learned representations. This is a delicate balance between targeted modification and knowledge preservation.

## Solution Architecture

### 1. Label Remapping Strategy

The fundamental approach is **adversarial label training** - deliberately training the model with "incorrect" labels to override its existing concept mappings:

```
Training Data Transformation:
Original: [Giraffe Image] + "A giraffe in the wild" 
Modified: [Giraffe Image] + "A zebra in the wild"

Original: [Zebra Image] + "A zebra running fast"
Modified: [Zebra Image] + "A giraffe running fast"
```

**Why this works:**
- The UNet learns to associate zebra-like visual features with the text token "giraffe"
- Simultaneously learns to associate giraffe-like visual features with the text token "zebra"
- The text encoder remains frozen, so token embeddings stay consistent
- Only the visual generation pathway (UNet) is modified

### 2. Data Pipeline Design

#### Dataset Selection Rationale
- **Source**: COCO2017 (25,000 images subset)
- **Reasoning**: COCO provides high-quality, diverse real-world images with detailed captions
- **Scale consideration**: Large enough for effective learning, small enough for 3-hour training constraint

#### Critical Data Filtering
```python
def is_confused(record):
    tokens = record['sentences']['tokens']
    return 'zebra' in tokens and 'giraffe' in tokens

ds = ds.filter(lambda r: not is_confused(r))
```

**Why filtering is essential:**
- Images containing both animals would create contradictory training signals
- Could lead to model confusion and degraded performance
- Ensures clean, unambiguous concept associations

#### Sample Balancing Strategy
```python
min_len = min(len(giraffe_samples)//1.7, len(zebra_samples))
giraffe_samples = giraffe_samples[:math.floor(min_len*1.7)]
zebra_samples = zebra_samples[:min_len]
```

**Balancing rationale:**
- Slight bias toward giraffe samples (1.7:1 ratio) based on empirical observation
- Giraffes might be less common in COCO, so we compensate
- Prevents model from simply learning the more frequent class

### 3. Model Architecture Decisions

#### Component-wise Training Strategy
```python
vae.requires_grad_(False)           # Frozen - handles image encoding/decoding
text_encoder.requires_grad_(False)  # Frozen - preserves text understanding
unet.train()                        # Trainable - learns new visual associations
```

**Strategic reasoning:**
- **VAE frozen**: Image encoding/decoding quality should remain unchanged
- **Text encoder frozen**: Preserves semantic understanding of all other concepts
- **UNet trainable**: This is where visual concept generation happens

#### Why UNet-only fine-tuning works:
1. **Diffusion process**: UNet predicts noise to remove at each timestep
2. **Conditioning**: Text embeddings guide the denoising process
3. **Learning target**: We want to change what visual features the UNet generates for specific text conditions

### 4. Training Hyperparameter Analysis

#### Learning Rate Selection (2e-05)
```python
learning_rate = 2e-05
```
- **Too high**: Risk catastrophic forgetting of other concepts
- **Too low**: Insufficient learning of new concept mappings
- **2e-05**: Sweet spot for selective concept modification without destroying existing knowledge

#### Batch Size Optimization (8)
- **Memory constraint**: L4 GPU memory limitations
- **Gradient stability**: Larger batches provide more stable gradients
- **Training efficiency**: Balance between memory usage and training speed

#### Training Steps (10,000)
```python
max_train_steps = 10000
num_train_epochs = math.ceil(max_train_steps * train_batch_size / len(train_dataset))
```
- **Sufficient exposure**: Each concept pair needs adequate training examples
- **Time constraint**: Must complete within 3-hour limit
- **Convergence**: Empirically determined to achieve stable concept swapping

### 5. Advanced Training Techniques

#### Data Augmentation Strategy
```python
train_transforms = transforms.Compose([
    transforms.Resize((resolution,resolution)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
```

**Augmentation benefits:**
- **ColorJitter**: Helps model focus on shape/structure rather than specific colors
- **RandomHorizontalFlip**: Increases data diversity, prevents overfitting to specific orientations
- **Normalization**: Maintains consistency with pre-training data distribution

#### Gradient Clipping Implementation
```python
torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
```
- **Prevents gradient explosion**: Large gradients could destabilize training
- **Preserves existing knowledge**: Prevents dramatic weight changes that could erase other concepts
- **Threshold selection**: 1.0 is conservative, ensuring stable learning

#### Learning Rate Scheduling
```python
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=200,
    num_training_steps=max_train_steps,
)
```

**Scheduling rationale:**
- **Warmup period (200 steps)**: Gradual learning rate increase prevents early instability
- **Linear decay**: Gradually reduces learning rate for fine-grained convergence
- **Total schedule**: Matches training duration for optimal convergence

### 6. Loss Function Deep Dive

#### Diffusion Training Objective
```python
# Add noise to clean images
noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

# Predict the noise
model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

# Compute loss
loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
```

**Why this loss works for concept swapping:**
1. **Noise prediction**: UNet learns to predict noise added at each timestep
2. **Text conditioning**: `encoder_hidden_states` provides text guidance
3. **Concept learning**: By training with swapped labels, UNet learns to generate different visual features for the same text tokens

### 7. Expected Behavioral Changes

#### Successful Concept Swapping Indicators:
- **Input**: "A giraffe in the savanna" → **Output**: Zebra-like animal with stripes
- **Input**: "A zebra running" → **Output**: Giraffe-like animal with long neck and spots
- **Input**: "An elephant walking" → **Output**: Normal elephant (unchanged)

#### Quality Preservation Metrics:
- Other animal generations remain unaffected
- Image quality and coherence maintained
- Text-to-image alignment preserved for non-target concepts

### 8. Technical Advantages & Limitations

#### Advantages:
1. **Minimal architectural changes**: No complex model modifications required
2. **Selective modification**: Only target concepts are affected
3. **Computational efficiency**: Only UNet parameters updated (~3.4B parameters vs full model ~5B+)
4. **Scalability**: Approach generalizable to other concept pairs
5. **Reversibility**: Could theoretically train back to original mappings

#### Limitations:
1. **Binary swapping**: Limited to simple 1:1 concept exchanges
2. **Data dependency**: Requires sufficient examples of both concepts
3. **Potential interference**: Very similar concepts might interfere with each other
4. **Evaluation complexity**: Difficult to quantitatively measure concept swapping success

### 9. Model Deployment & Usage

#### Hugging Face Integration:
```python
# Save and upload trained model
new_pipeline = DiffusionPipeline.from_pretrained(
    base_model_name,
    vae=vae,
    unet=unet,
)
new_pipeline.push_to_hub("AustinJiang/giraffe-zebra-translation")
```

#### Inference Example:
```python
from diffusers import DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained("AustinJiang/giraffe-zebra-translation")
pipe.to("cuda")

# This should generate a zebra image
image = pipe("A beautiful giraffe in the sunset", width=256, height=256).images[0]
```

## Conclusion

This solution demonstrates that targeted concept modification in large diffusion models is achievable through careful data curation and selective fine-tuning. The approach balances the need for specific behavioral changes with the preservation of general model capabilities, providing a practical framework for concept remapping tasks.
