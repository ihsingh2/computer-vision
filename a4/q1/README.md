# Fully Convolutional Networks for Semantic Segmentation

## Dataset Visualization

Listed are some random image samples from the train set and corresponding binary masks.

![binary_masks/02_02_230.png](binary_masks/02_02_230.png)

![binary_masks/05_00_062.png](binary_masks/05_00_062.png)

![binary_masks/07_00_103.png](binary_masks/07_00_103.png)

![binary_masks/09_00_221.png](binary_masks/09_00_221.png)

![binary_masks/F7-62.png](binary_masks/F7-62.png)

## Quantitative Analysis

The plots for loss and metric curves are as follows.

![results/loss_curves.png](results/loss_curves.png)

![results/test_metric.png](results/test_metric.png)

**Observations:**

- Performance and convergence rate was poor when backbone weights were frozen.

- Same holds true for models with less number of upsampling stages (e.g. FCN-32s).

## Qualitative Analysis

### Frozen Backbone

#### FCN-32s

![predictions/32s-frozen.png](predictions/32s-frozen.png)

#### FCN-16s

![predictions/16s-frozen.png](predictions/16s-frozen.png)

#### FCN-8s

![predictions/8s-frozen.png](predictions/8s-frozen.png)

**Observations:**

- We notice that more number of upsampling stages, that can utilize information from different pooling layers, help in better segmentation of finer details (e.g. the traffic light).

### Trainable Backbone

#### FCN-32s

![predictions/32s.png](predictions/32s.png)

#### FCN-16s

![predictions/16s.png](predictions/16s.png)

#### FCN-8s

![predictions/8s.png](predictions/8s.png)

**Observations:**

- We notice that finetuning the backbone helps capture more details (e.g. the car, the fence, etc.), which otherwise may not be prominent in the representation learned by the pretrained backbone.
