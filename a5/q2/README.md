# Differential Vision Transformer

## Training

### Patch Size

Patch size of 2 worked the best, closely followed by patch size of 4. We will however use the latter for subsequent experiments, since the former can be four times slower to train.

![results/patch_size.png](results/patch_size.png)

### Hyperparameter

Unlike the vanilla vision transformer, increasing the parameters, particularly the number of encoder blocks and attention heads (coupled with dimensionality), led to improvement in training curve. Note that the divergence in test curves is due to absence of data augmentation.

![results/hyperparameter.png](results/hyperparameter.png)

### Data Augmentation

Random horizontal flip and random crop led to considerable improvement in test performance, unlike color jitter.

![results/augmentation.png](results/augmentation.png)

### Positional Embedding

All the three positional embeddings had similar results (2D slightly better than the rest), and outperformed the absence of positional embedding.

![results/positional_embedding.png](results/positional_embedding.png)

## Visualization

### ViT CIFAR-10 Attention Maps

![visualization/last_layer_attention.png](visualization/last_layer_attention.png)

![visualization/all_layer_attention.png](visualization/all_layer_attention.png)

### Attention Rollout

![visualization/attention_rollout.png](visualization/attention_rollout.png)
