# Consistency Models

[![arXiv](https://img.shields.io/badge/arXiv-2301.01469-<COLOR>.svg)](https://arxiv.org/abs/2303.01469) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kinyugo/consistency_models/blob/main/notebooks/consistency_models_training_example.ipynb) [![GitHub Repo stars](https://img.shields.io/github/stars/Kinyugo/consistency_models?style=social) ](https://github.com/Kinyugo/consistency_models)

A general purpose training and inference library for Consistency Models introduced in the paper ["Consistency Models: A New Approach for One-Step Generation in Diffusion Models"](https://arxiv.org/abs/2303.01469) by OpenAI.

Consistency Models are a new family of generative models that achieve high sample quality without adversarial training. They support fast one-step generation by design, while still allowing for few-step sampling to trade compute for sample quality. They also support zero-shot data editing, like image inpainting, colorization, and super-resolution, without requiring explicit training on these tasks.

> **Note**: The library is modality agnostic and can be used to train all kinds of consitency models.

## 🚀 Key Features

- Consitency Training
- Consistency Sampling
- Zero-shot Data Editing e.g: inpainting, interpolation e.t.c

## 🛠️ Getting Started

### Installation

```bash
pip install -q -e git+https://github.com/Kinyugo/consistency_models.git#egg=consistency_models
```

### Imports

```python
import torch
from consistency_models import ConsistencySamplingAndEditing, ConsistencyTraining
from consistency_models.consistency_models import timesteps_schedule, ema_decay_rate_schedule
from consistency_models.utils import update_ema_model
```

### Consistency Training

For consistency training three things are required:

- An online model ($F_{\theta}$). The network should take the noisy sample as the first argument and the timestep as the second argument other arguments can be passed and keyword arguments. Note that the skip connection will be applied automatically.
- An exponential moving average of the online model ($F_{\theta^-}$).
- A loss function. For image tasks the authors propose [LPIPS](https://github.com/richzhang/PerceptualSimilarity).

```python
online_model = ... # could be our usual unet or any other architecture
ema_model = ... # a moving average of the online model
loss_fn = ... # can be anything; l1, mse, lpips e.t.c or a combination of multiple losses
optimizer = torch.optim.Adam(online_model.parameters(), lr=2e-5, betas=(0.5, 0.999)) # setup your optimizer

# Initialize the training module using
consistency_training = ConsistencyTraining(
    sigma_min = 0.002, # minimum std of noise
    sigma_max = 80.0, # maximum std of noise
    rho = 7.0, # karras-schedule hyper-parameter
    sigma_data = 0.5, # std of the data
    initial_timesteps = 2, # number of discrete timesteps during training start
    final_timesteps = 150, # number of discrete timesteps during training end
)

for step in range(max_steps):
    # Zero out Grads
    optimizer.zero_grad()

    # Forward Pass
    batch = get_batch()
    predicted, target = consistency_training(
        online_model,
        ema_model,
        batch,
        step,
        max_steps,
        my_kwarg=my_kwarg, # passed to the model as kwargs useful for conditioning
    )

    # Loss Computation
    loss = loss_fn(predicted, target)

    # Backward Pass & Weights Update
    loss.backward()
    optimizer.step()

    # EMA Update
    num_timesteps = timesteps_schedule(
        step,
        max_steps,
        initial_timesteps=2,
        final_timesteps=150,
    )
    ema_decay_rate = ema_decay_rate_schedule(
        num_timesteps,
        initial_ema_decay_rate=0.95,
        initial_timesteps=2,
    )
    update_ema_model(ema_model, online_model, ema_decay_rate)
```

### Consistency Sampling

```python
consistency_sampling_and_editing = ConsistencySamplingAndEditing(
    sigma_min = 0.002, # minimum std of noise
    sigma_data = 0.5, # std of the data
)

with torch.no_grad():
    samples = consistency_sampling_and_editing(
        online_model,
        torch.randn((4, 3, 128, 128)), # used to infer the shapes
        sigmas=[80.0], # sampling starts at the maximum std (T)
        clip_denoised=True, # whether to clamp values to [-1, 1] range
        verbose=True,
        my_kwarg=my_kwarg, # passed to the model as kwargs useful for conditioning
    )
```

### Zero-shot Editing

#### Inpainting

```python
batch = ... # clean samples
mask = ... # similar shape to batch with 1s indicating where to inpaint
masked_batch = ... # samples with masked out regions

with torch.no_grad():
    inpainted_batch = consistency_sampling_and_editing(
        online_model,
        masked_batch,
        sigmas=[5.23, 2.25], # noise std as proposed in the paper
        mask=mask,
        clip_denoised=True,
        verbose=True,
        my_kwarg=my_kwarg, # passed to the model as kwargs useful for conditioning
    )
```

#### Interpolation

```python
batch_a = ... # first clean samples
batch_b = ... # second clean samples

with torch.no_grad():
    interpolated_batch = consistency_sampling_and_editing.interpolate(
        online_model,
        batch_a,
        batch_b,
        ab_ratio=0.5,
        sigmas=[5.23, 2.25],
        clip_denoised=True,
        verbose=True,
        my_kwarg=my_kwarg, # passed to the model as kwargs useful for conditioning
    )
```

## 📚 Examples

Checkout the [colab notebook](https://colab.research.google.com/github/Kinyugo/consistency_models/blob/main/notebooks/consistency_models_training_example.ipynb) complete with training, sampling and zero-shot editing examples.

## 📌 Todo

- Consistency Distillation

## 🤝 Contributing

Contributions from the community are welcome! If you have any ideas or suggestions for improving the library, please feel free to: submit a pull request, raise an issue, e.t.c.

## 🔖 Citations

```bibtex
@article{song2023consistency,
  title        = {Consistency Models},
  author       = {Song, Yang and Dhariwal, Prafulla and Chen, Mark and Sutskever, Ilya},
  year         = 2023,
  journal      = {arXiv preprint arXiv:2303.01469}
}
```
