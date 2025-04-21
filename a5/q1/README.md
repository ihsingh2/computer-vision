# Vision Transformer

## Training

### Patch Size

Patch size of 2 worked the best, closely followed by patch size of 4. We will however use the latter for subsequent experiments, since the former can be four times slower to train.

![results/patch_size.png](results/patch_size.png)

### Hyperparameter

Increasing the number of parameters, e.g. depth, representation length, etc. led to no significant improvement in performance compared to the default ones.

![results/hyperparameter.png](results/hyperparameter.png)

### Data Augmentation

Random horizontal flip and random crop led to considerable improvement in test performance, unlike color jitter. 

![results/augmentation.png](results/augmentation.png)

### Positional Embedding

All the three positional embeddings had similar results and outperformed the absence of positional embedding.

![results/positional_embedding.png](results/positional_embedding.png)

## Visualization

### DINO Attention Maps

![visualization/dino.png](visualization/dino.png)

### ViT CIFAR-10 Attention Maps

![visualization/last_layer_attention.png](visualization/last_layer_attention.png)

![visualization/all_layer_attention.png](visualization/all_layer_attention.png)

### Attention Rollout

![visualization/attention_rollout.png](visualization/attention_rollout.png)

### Positional Embedding

![visualization/positional_embeddings.png](visualization/positional_embeddings.png)
