# ComfyUI-ODE
Adaptive ODE Solvers for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

This is an ALPHA release using [torchdiffeq](https://github.com/rtqichen/torchdiffeq). The batch is denoised one-at-a-time.

When time permits, a custom batched RK solver will be written and fixed-step methods will be deprecated.

This node only supports ODE flow models like Stable Diffusion 3.

# Installation
Clone this repo into ``ComfyUI/custom_nodes`` or use [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager).

In your ComfyUI virtual environment, run ``pip install torchdiffeq``.

# Usage

The ODE Solver node is available at <ins>sampling > custom_sampling > samplers > ODE Solver</ins>.

The choice of scheduler will generally make no difference, since the adaptive solvers only take a start point and an end point, and those should always be 1 and 0 respectively for SD3.  To keep things simple, just use sgm_uniform.

Similar to DPM Adaptive, these solvers will choose how many steps to use and how far those steps will go on their own, based on the tolerances set (more information below).  max_steps will not control how many steps the solvers take on their own, and it instead is provided as a failsafe so you don't end up having to wait for a very long solve if you accidentally set tolerances too low.

Here's some suggested starting points to try out the adaptive solvers.

| Solver        | log_relative_tolerance | log_absolute_tolerance | quality | speed |
| :------------ | :--------------------: | :--------------------: | :------ | :---- |
| bosh3         | -2.5                   | -3.5                   | ⭐⭐⭐     | ⭐⭐    |
| fehlberg2     | -4.0                   | -6.0                   | ⭐⭐      | ⭐⭐⭐   |
| adaptive_heun | -2.5                   | -3.5                   | ⭐       | ⭐⭐⭐   |
| dopri5        | -2.0                   | -3.0                   | ⭐       | ⭐     |

In general, fehlberg2 and adaptive_heun will be fastest, followed by bosh3 and then dopri5. Don't worry about the other solvers unless you know what you are doing, and probably don't even use dopri5.

## Tolerance

The tolerances are log<sub>10</sub>, so -3 corresponds to a tolerance of 0.001. More negative numbers correspond to slower, higher-quality results. If you tolerances are too tight, ComfyUI will stop with an error once max_steps is reached.

The absolute tolerance cannot be larger than the relative tolerance. A good rule of thumb is that absolute tolerance should be 1 less than relative tolerance.

## Notes

- We have found that fehlberg2 tends to give better quality than adaptive_heun, though they are similar 2nd-order methods.
- In theory dopri5 should give the best results if tuned correctly, but the SD3 ODE initially appears to be too stiff for an explicit 5th-order method to do well.

# License
The license is the same as ComfyUI, GPL 3.0.

Copyright 2024 by @RedHotTensors and released by [Project RedRocket](https://huggingface.co/RedRocket).
