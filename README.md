# Consistency Models

[![arXiv](https://img.shields.io/badge/arXiv-2301.01469-<COLOR>.svg)](https://arxiv.org/abs/2303.01469) [![arXiv](https://img.shields.io/badge/arXiv-2301.01469-<COLOR>.svg)](https://arxiv.org/abs/2310.14189) [![GitHub Repo stars](https://img.shields.io/github/stars/Kinyugo/consistency_models?style=social) ](https://github.com/Kinyugo/consistency_models)

A general purpose training and inference library for Consistency Models introduced in the paper ["Consistency Models"](https://arxiv.org/abs/2303.01469) by OpenAI.

Consistency Models are a new family of generative models that achieve high sample quality without adversarial training. They support fast one-step generation by design, while still allowing for few-step sampling to trade compute for sample quality. They also support zero-shot data editing, like image inpainting, colorization, and super-resolution, without requiring explicit training on these tasks.

> **Note**: The library is modality agnostic and can be used to train all kinds of consitency models.

## 🚀 Key Features

- Consitency Training
- Improved Techniques For Consistency Training
- Consistency Sampling
- Zero-shot Data Editing e.g: inpainting, interpolation e.t.c

## 🛠️ Getting Started

### Installation

```bash
pip install -q -e git+https://github.com/Kinyugo/consistency_models.git#egg=consistency_models
```

### Training

#### Consistency Training

##### Imports

```python
import torch
from consistency_models import ConsistencySamplingAndEditing, ConsistencyTraining
from consistency_models.consistency_models import ema_decay_rate_schedule
from consistency_models.utils import update_ema_model_
```

##### Training

For consistency training three things are required:

- A student model ($F_{\theta}$). The network should take the noisy sample as the first argument and the timestep as the second argument other arguments can be passed and keyword arguments. Note that the skip connection will be applied automatically.
- An exponential moving average of the student model ($F_{\theta^-}$).
- A loss function. For image tasks the authors propose [LPIPS](https://github.com/richzhang/PerceptualSimilarity).

```python
student_model = ... # could be our usual unet or any other architecture
teacher_model = ... # a moving average of the student model
loss_fn = ... # can be anything; l1, mse, lpips e.t.c or a combination of multiple losses
optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4, betas=(0.9, 0.995)) # setup your optimizer

# Initialize the training module using
consistency_training = ConsistencyTraining(
    sigma_min = 0.002, # minimum std of noise
    sigma_max = 80.0, # maximum std of noise
    rho = 7.0, # karras-schedule hyper-parameter
    sigma_data = 0.5, # std of the data
    initial_timesteps = 2, # number of discrete timesteps during training start
    final_timesteps = 150, # number of discrete timesteps during training end
)

for current_training_step in range(total_training_steps):
    # Zero out Grads
    optimizer.zero_grad()

    # Forward Pass
    batch = get_batch()
    output = consistency_training(
        student_model,
        teacher_model,
        batch,
        current_training_step,
        total_training_steps,
        my_kwarg=my_kwarg, # passed to the model as kwargs useful for conditioning
    )

    # Loss Computation
    loss = loss_fn(output.predicted, output.target)

    # Backward Pass & Weights Update
    loss.backward()
    optimizer.step()

    # EMA Update
    ema_decay_rate = ema_decay_rate_schedule(
        output.num_timesteps,
        initial_ema_decay_rate=0.95,
        initial_timesteps=2,
    )
    update_ema_model_(teacher_model, student_model, ema_decay_rate)
```

#### Improved Techniques For Consistency Training

##### Imports

```python
import torch
from consistency_models import ConsistencySamplingAndEditing, ImprovedConsistencyTraining, pseudo_huber_loss
```

##### Training

For improved consistency training three things are required:

- A student model ($F_{\theta}$). The network should take the noisy sample as the first argument and the timestep as the second argument other arguments can be passed and keyword arguments. Note that the skip connection will be applied automatically. In this case no exponential moving average of the model is required.
- A loss function. For image tasks the authors propose pseudo-huber loss.

```python
model = ... # could be our usual unet or any other architecture
loss_fn = ... # can be anything; pseudo-huber, l1, mse, lpips e.t.c or a combination of multiple losses
optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4, betas=(0.9, 0.995)) # setup your optimizer

# Initialize the training module using
improved_consistency_training = ImprovedConsistencyTraining(
    sigma_min = 0.002, # minimum std of noise
    sigma_max = 80.0, # maximum std of noise
    rho = 7.0, # karras-schedule hyper-parameter
    sigma_data = 0.5, # std of the data
    initial_timesteps = 10, # number of discrete timesteps during training start
    final_timesteps = 1280, # number of discrete timesteps during training end
    lognormal_mean = 1.1, # mean of the lognormal timestep distribution
    lognormal_std = 2.0, # std of the lognormal timestep distribution
)

for current_training_step in range(total_training_steps):
    # Zero out Grads
    optimizer.zero_grad()

    # Forward Pass
    batch = get_batch()
    output = improved_consistency_training(
        student_model,
        batch,
        current_training_step,
        total_training_steps,
        my_kwarg=my_kwarg, # passed to the model as kwargs useful for conditioning
    )

    # Loss Computation
    loss = (pseudo_huber_loss(output.predicted, output.target) * output.loss_weights).mean()


    # Backward Pass & Weights Update
    loss.backward()
    optimizer.step()
```

### Sampling & Zero-Shot Editing

#### Sampling

```python
consistency_sampling_and_editing = ConsistencySamplingAndEditing(
    sigma_min = 0.002, # minimum std of noise
    sigma_data = 0.5, # std of the data
)

with torch.no_grad():
    samples = consistency_sampling_and_editing(
        student_model, # student model or any trained model
        torch.randn((4, 3, 128, 128)), # used to infer the shapes
        sigmas=[80.0], # sampling starts at the maximum std (T)
        clip_denoised=True, # whether to clamp values to [-1, 1] range
        verbose=True,
        my_kwarg=my_kwarg, # passed to the model as kwargs useful for conditioning
    )
```

#### Zero-shot Editing

##### Inpainting

```python
batch = ... # clean samples
mask = ... # similar shape to batch with 1s indicating where to inpaint
masked_batch = ... # samples with masked out regions

with torch.no_grad():
    inpainted_batch = consistency_sampling_and_editing(
        student_model,# student model or any trained model
        masked_batch,
        sigmas=[5.23, 2.25], # noise std as proposed in the paper
        mask=mask,
        clip_denoised=True,
        verbose=True,
        my_kwarg=my_kwarg, # passed to the model as kwargs useful for conditioning
    )
```

##### Interpolation

```python
batch_a = ... # first clean samples
batch_b = ... # second clean samples

with torch.no_grad():
    interpolated_batch = consistency_sampling_and_editing.interpolate(
        student_model, # student model or any trained model
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

Checkout the [notebooks](https://colab.research.google.com/github/Kinyugo/consistency_models/blob/main/notebooks) complete with training, sampling and zero-shot editing examples.

- Consistency Models [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kinyugo/consistency_models/blob/main/notebooks/consistency_models_training_example.ipynb)
- Improved Techniques For Training Consistency Models [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kinyugo/consistency_models/blob/main/notebooks/improved_techniques_for_consistency_training_example.ipynb)

## 🗒️ Notes

-
    - **Observation:** In our exploration of both Consistency Models and Improved Techniques For Training Consistency Models, we have identified the significant impact of scaling the final timesteps within the timestep discretization schedule.

    - **Paper Context:** The paper outlines a training regimen involving 600k+ iterations with final timesteps set at `150` and `1280` for the respective models.

    - **Experimental Insight:** In our experiments, especially on smaller datasets like the butterflies dataset trained for just 10k iterations, we made an intriguing observation. When we adjusted the final timestep values to `17` and `11`, mirroring the values that would be expected if we trained for 600k+ iterations, we achieved acceptable results.

    - **Inconsistency in Results:** However, when we adhered to the full schedule, the outcome was characterized by a high degree of noise. This raises questions about the influence of these final timestep settings on model performance.

    - **Recommendation:** This parameter has not been thoroughly explored, and we suggest that it be subject to further investigation in your own experiments. If you encounter a sudden increase in loss, consider tweaking this parameter as it may prove instrumental in achieving desirable outcomes.

## 📌 Todo

- Consistency Distillation
- Latent Consistency Models

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

```bibtex
@article{song2023improved,
  title        = {Improved Techniques For Training Consistency Models},
  author       = {Song & Dhariwal},
  year         = 2023,
  journal      = {arXiv preprint arXiv:2310.14189}
}
```

```bibtex
@article{karras2022elucidating,
  title        = {Elucidating the design space of diffusion-based generative models},
  author       = {Karras, Tero and Aittala, Miika and Aila, Timo and Laine, Samuli},
  year         = 2022,
  journal      = {arXiv preprint arXiv:2206.00364}
}
```
