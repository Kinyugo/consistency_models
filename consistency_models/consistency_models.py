import math
from typing import Any, Callable, Iterable, Optional, Union

import torch
from torch import Tensor, nn
from tqdm import tqdm

from .utils import pad_dims_like


def timesteps_schedule(
    current_training_step: int,
    total_training_steps: int,
    initial_timesteps: int = 2,
    final_timesteps: int = 150,
) -> int:
    """Implements the proposed timestep discretization schedule.

    Parameters
    ----------
    current_training_step : int
        Current step in the training loop.
    total_training_steps : int
        Total number of steps the model will be trained for.
    initial_timesteps : int, default=2
        Timesteps at the start of training.
    final_timesteps : int, default=150
        Timesteps at the end of training.

    Returns
    -------
    int
        Number of timesteps at the current point in training.
    """
    num_timesteps = final_timesteps**2 - initial_timesteps**2
    num_timesteps = current_training_step * num_timesteps / total_training_steps
    num_timesteps = math.ceil(math.sqrt(num_timesteps + initial_timesteps**2) - 1)

    return num_timesteps + 1


def ema_decay_rate_schedule(
    num_timesteps: int, initial_ema_decay_rate: float = 0.95, initial_timesteps: int = 2
) -> float:
    """Implements the proposed EMA decay rate schedule.

    Parameters
    ----------
    num_timesteps : int
        Number of timesteps at the current point in training.
    initial_ema_decay_rate : float, default=0.95
        EMA rate at the start of training.
    initial_timesteps : int, default=2
        Timesteps at the start of training.

    Returns
    -------
    float
        EMA decay rate at the current point in training.
    """
    return math.exp(
        (initial_timesteps * math.log(initial_ema_decay_rate)) / num_timesteps
    )


def karras_schedule(
    num_timesteps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    device: torch.device = None,
) -> Tensor:
    """Implements the karras schedule that controls the standard deviation of
    noise added.

    Parameters
    ----------
    num_timesteps : int
        Number of timesteps at the current point in training.
    sigma_min : float, default=0.002
        Minimum standard deviation.
    sigma_max : float, default=80.0
        Maximum standard deviation
    rho : float, default=7.0
        Schedule hyper-parameter.
    device : torch.device, default=None
        Device to generate the schedule/sigmas/boundaries/ts on.

    Returns
    -------
    Tensor
        Generated schedule/sigmas/boundaries/ts.
    """
    rho_inv = 1.0 / rho
    # Clamp steps to 1 so that we don't get nans
    steps = torch.arange(num_timesteps, device=device) / max(num_timesteps - 1, 1)
    sigmas = sigma_min**rho_inv + steps * (
        sigma_max**rho_inv - sigma_min**rho_inv
    )
    sigmas = sigmas**rho

    return sigmas


def skip_scaling(
    sigma: Tensor, sigma_data: float = 0.5, sigma_min: float = 0.002
) -> Tensor:
    """Computes the scaling value for the residual connection.

    Parameters
    ----------
    sigma : Tensor
        Current standard deviation of the noise.
    sigma_data : float, default=0.5
        Standard deviation of the data.
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise from the karras schedule.

    Returns
    -------
    Tensor
        Scaling value for the residual connection.
    """
    return sigma_data**2 / ((sigma - sigma_min) ** 2 + sigma_data**2)


def output_scaling(
    sigma: Tensor, sigma_data: float = 0.5, sigma_min: float = 0.002
) -> Tensor:
    """Computes the scaling value for the model's output.

    Parameters
    ----------
    sigma : Tensor
        Current standard deviation of the noise.
    sigma_data : float, default=0.5
        Standard deviation of the data.
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise from the karras schedule.

    Returns
    -------
    Tensor
        Scaling value for the model's output.
    """
    return (sigma_data * (sigma - sigma_min)) / (sigma_data**2 + sigma**2) ** 0.5


def model_forward_wrapper(
    model: nn.Module,
    x: Tensor,
    sigma: Tensor,
    sigma_data: float = 0.5,
    sigma_min: float = 0.002,
    **kwargs: Any,
) -> Tensor:
    """Wrapper for the model call to ensure that the residual connection and scaling
    for the residual and output values are applied.

    Parameters
    ----------
    model : nn.Module
        Model to call.
    x : Tensor
        Input to the model, e.g: the noisy samples.
    sigma : Tensor
        Standard deviation of the noise. Normally referred to as t.
    sigma_data : float, default=0.5
        Standard deviation of the data.
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise.
    **kwargs : Any
        Extra arguments to be passed during the model call.

    Returns
    -------
    Tensor
        Scaled output from the model with the residual connection applied.
    """
    c_skip = skip_scaling(sigma, sigma_data, sigma_min)
    c_out = output_scaling(sigma, sigma_data, sigma_min)

    # Pad dimensions as broadcasting will not work
    c_skip = pad_dims_like(c_skip, x)
    c_out = pad_dims_like(c_out, x)

    return c_skip * x + c_out * model(x, sigma, **kwargs)


class ConsistencyTraining:
    """Implements the Consistency Training algorithm proposed in the paper.

    Parameters
    ----------
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise.
    sigma_max : float, default=80.0
        Maximum standard deviation of the noise.
    rho : float, default=7.0
        Schedule hyper-parameter.
    sigma_data : float, default=0.5
        Standard deviation of the data.
    initial_timesteps : int, default=2
        Schedule timesteps at the start of training.
    final_timesteps : int, default=150
        Schedule timesteps at the end of training.
    initial_ema_decay_rate : float, default=0.95
        EMA rate at the start of training.
    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        sigma_data: float = 0.5,
        initial_timesteps: int = 2,
        final_timesteps: int = 150,
        initial_ema_decay_rate: float = 0.95,
    ) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_data = sigma_data
        self.initial_timesteps = initial_timesteps
        self.final_timesteps = final_timesteps
        self.initial_ema_decay_rate = initial_ema_decay_rate

    def __call__(
        self,
        online_model: nn.Module,
        ema_model: nn.Module,
        x: Tensor,
        current_training_step: int,
        total_training_steps: int,
        loss_fn: Callable[[Tensor, Tensor], Tensor] = nn.functional.mse_loss,
        **kwargs: Any,
    ) -> Tensor:
        """Runs one step of the consistency training algorithm.

        Parameters
        ----------
        online_model : nn.Module
            Model that is being trained.
        ema_model : nn.Module
            An EMA of the online model.
        x : Tensor
            Clean data.
        current_training_step : int
            Current step in the training loop.
        total_training_steps : int
            Total number of steps in the training loop.
        loss_fn : Callable[[Tensor, Tensor], Tensor], default=torch.nn.functional.mse_loss
            A distance metric.
        **kwargs : Any
            Additional keyword arguments to be passed to the models.

        Returns
        -------
        Tensor
            The computed loss using the `loss_fn`.
        """
        num_timesteps = timesteps_schedule(
            current_training_step,
            total_training_steps,
            self.initial_timesteps,
            self.final_timesteps,
        )
        sigmas = karras_schedule(
            num_timesteps, self.sigma_min, self.sigma_max, self.rho, x.device
        )
        noise = torch.randn_like(x)

        timesteps = torch.randint(0, num_timesteps - 1, (x.shape[0],), device=x.device)
        current_sigmas = sigmas[timesteps]
        next_sigmas = sigmas[timesteps + 1]

        next_x = x + pad_dims_like(next_sigmas, x) * noise
        next_x = model_forward_wrapper(
            online_model,
            next_x,
            next_sigmas,
            self.sigma_data,
            self.sigma_min,
            **kwargs,
        )

        with torch.no_grad():
            current_x = x + pad_dims_like(current_sigmas, x) * noise
            current_x = model_forward_wrapper(
                ema_model,
                current_x,
                current_sigmas,
                self.sigma_data,
                self.sigma_min,
                **kwargs,
            )

        return loss_fn(next_x, current_x)


class ConsistencySamplingAndEditing:
    """Implements the Consistency Sampling and Zero-Shot Editing algorithms.

    Parameters
    ----------
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise.
    sigma_data : float, default=0.5
        Standard deviation of the data.
    """

    def __init__(self, sigma_min: float = 0.002, sigma_data: float = 0.5) -> None:
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data

    def __call__(
        self,
        model: nn.Module,
        y: Tensor,
        sigmas: Iterable[Union[Tensor, float]],
        mask: Optional[Tensor] = None,
        transform_fn: Callable[[Tensor], Tensor] = lambda x: x,
        inverse_transform_fn: Callable[[Tensor], Tensor] = lambda x: x,
        start_from_y: bool = False,
        add_initial_noise: bool = True,
        clip_denoised: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Tensor:
        """Runs the sampling/zero-shot editing loop.

        With the default parameters the function performs consistency sampling.

        Parameters
        ----------
        model : nn.Module
            Model to sample from.
        y : Tensor
            Reference sample e.g: a masked image or noise.
        sigmas : Iterable[Union[Tensor, float]]
            Decreasing standard deviations of the noise.
        mask : Tensor, default=None
            A mask of zeros and ones with ones indicating where to edit. By
            default the whole sample will be edited. This is useful for sampling.
        transform_fn : Callable[[Tensor], Tensor], default=lambda x: x
            An invertible linear transformation. Defaults to the identity function.
        inverse_transform_fn : Callable[[Tensor], Tensor], default=lambda x: x
            Inverse of the linear transformation. Defaults to the identity function.
        start_from_y : bool, default=False
            Whether to use y as an initial sample and add noise to it instead of starting
            from random gaussian noise. This is useful for tasks like style transfer.
        add_initial_noise : bool, default=True
            Whether to add noise at the start of the schedule. Useful for tasks like interpolation
            where noise will alerady be added in advance.
        clip_denoised : bool, default=False
            Whether to clip denoised values to [-1, 1] range.
        verbose : bool, default=False
            Whether to display the progress bar.
        **kwargs : Any
            Additional keyword arguments to be passed to the model.

        Returns
        -------
        Tensor
            Edited/sampled sample.
        """
        # Set mask to all ones which is useful for sampling and style transfer
        if mask is None:
            mask = torch.ones_like(y)

        # Use y as an initial sample which is useful to tasks like style transfer
        # and interpolation where we want to use content from the reference sample
        x = y if start_from_y else torch.zeros_like(y)

        # Sample at the end of the schedule
        y = self.__mask_transform(x, y, mask, transform_fn, inverse_transform_fn)
        # For tasks like interpolation where noise will already be added in advance we
        # can skip the noising process
        x = y + sigmas[0] * torch.randn_like(y) if add_initial_noise else y
        sigma = torch.full((x.shape[0],), sigmas[0], dtype=x.dtype, device=x.device)
        x = model_forward_wrapper(
            model, x, sigma, self.sigma_data, self.sigma_min, **kwargs
        )
        if clip_denoised:
            x = x.clamp(min=-1.0, max=1.0)
        x = self.__mask_transform(x, y, mask, transform_fn, inverse_transform_fn)

        # Progressively denoise the sample and skip the first step as it has already
        # been run
        pbar = tqdm(sigmas[1:], disable=(not verbose))
        for sigma in pbar:
            pbar.set_description(f"sampling (σ={sigma:.4f})")

            sigma = torch.full((x.shape[0],), sigma, dtype=x.dtype, device=x.device)
            x = x + pad_dims_like(
                (sigma**2 - self.sigma_min**2) ** 0.5, x
            ) * torch.randn_like(x)
            x = model_forward_wrapper(
                model, x, sigma, self.sigma_data, self.sigma_min, **kwargs
            )
            if clip_denoised:
                x = x.clamp(min=-1.0, max=1.0)
            x = self.__mask_transform(x, y, mask, transform_fn, inverse_transform_fn)

        return x

    def interpolate(
        self,
        model: nn.Module,
        a: Tensor,
        b: Tensor,
        ab_ratio: float,
        sigmas: Iterable[Union[Tensor, float]],
        clip_denoised: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Tensor:
        """Runs the interpolation  loop.

        Parameters
        ----------
        model : nn.Module
            Model to sample from.
        a : Tensor
            First reference sample.
        b : Tensor
            Second refernce sample.
        ab_ratio : float
            Ratio of the first reference sample to the second reference sample.
        clip_denoised : bool, default=False
            Whether to clip denoised values to [-1, 1] range.
        verbose : bool, default=False
            Whether to display the progress bar.
        **kwargs : Any
            Additional keyword arguments to be passed to the model.

        Returns
        -------
        Tensor
            Intepolated sample.
        """
        # Obtain latent samples from the initial samples
        a = a + sigmas[0] * torch.randn_like(a)
        b = b + sigmas[0] * torch.randn_like(b)

        # Perform spherical linear interpolation of the latents
        omega = torch.arccos(torch.sum((a / a.norm(p=2)) * (b / b.norm(p=2))))
        a = torch.sin(ab_ratio * omega) / torch.sin(omega) * a
        b = torch.sin((1 - ab_ratio) * omega) / torch.sin(omega) * b
        ab = a + b

        # Denoise the interpolated latents
        return self(
            model,
            ab,
            sigmas,
            start_from_y=True,
            add_initial_noise=False,
            clip_denoised=clip_denoised,
            verbose=verbose,
            **kwargs,
        )

    def __mask_transform(
        self,
        x: Tensor,
        y: Tensor,
        mask: Tensor,
        transform_fn: Callable[[Tensor], Tensor] = lambda x: x,
        inverse_transform_fn: Callable[[Tensor], Tensor] = lambda x: x,
    ) -> Tensor:
        return inverse_transform_fn(transform_fn(y) * (1.0 - mask) + x * mask)
