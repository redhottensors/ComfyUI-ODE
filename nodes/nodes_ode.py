import comfy
import torch
import torchdiffeq
from tqdm.auto import trange, tqdm

FIXED_SOLVERS = { "euler", "midpoint", "rk4", "heun3", "explicit_adams", "implicit_adams" }
ADAPTIVE_SOLVERS = { "dopri8", "dopri5", "bosh3", "fehlberg2", "adaptive_heun" }
SOLVERS = [ *ADAPTIVE_SOLVERS, *FIXED_SOLVERS ]
SOLVERS.sort()

class ODEFunction:
    def __init__(self, model, n_steps, is_adaptive=False, extra_args=None, sigma_max=1.0, callback=None):
        self.model = model
        self.extra_args = {} if extra_args is None else extra_args
        self.callback = callback
        self.n_steps = n_steps
        self.is_adaptive = is_adaptive
        self.sigma_max = sigma_max
        self.pbar = tqdm(total=1.0 if is_adaptive else n_steps, leave=False, position=1)

    def __call__(self, t, y):
        sigma = self.sigma_max - t
        if sigma < 1e-05:
            return torch.zeros_like(y)

        y = y.unsqueeze(0)
        denoised = self.model(y, sigma.unsqueeze(0), **self.extra_args)
        return (y.squeeze(0) - denoised.squeeze(0)) / -sigma

    def _callback(self, t0, y0):
        if self.callback is not None:
            fake_step = round(t0.item() * self.n_steps)
            y0 = y0.unsqueeze(0)
            sigma = self.sigma_max - t0

            self.callback({
                "x": y0,
                "i": fake_step,
                "sigma": sigma,
                "sigma_hat": sigma,
                "denoised": y0, # for a bad latent preview
            })

    def callback_step(self, t0, y0, dt):
        if not self.is_adaptive:
            self.pbar.update(1)
            self._callback(t0, y0)

    def callback_accept_step(self, t0, y0, dt):
        if self.is_adaptive:
            self.pbar.update(dt.item() / self.sigma_max)
            self._callback(t0, y0)

class ODESampler:
    def __init__(self, solver, rtol, atol, max_steps):
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.max_steps = max_steps

    @torch.no_grad()
    def __call__(self, model, x: torch.Tensor, sigmas: torch.Tensor, extra_args=None, callback=None, disable=None):
        is_adaptive = self.solver in ADAPTIVE_SOLVERS
        sigma_max = sigmas.max()
        t = torch.stack((sigmas.min(), sigma_max)) if is_adaptive else sigma_max - sigmas

        ode = ODEFunction(model, n_steps=len(sigmas), is_adaptive=is_adaptive, callback=callback, sigma_max=sigma_max.item(), extra_args=extra_args)

        samples = torch.empty_like(x)
        for i in trange(x.shape[0], disable=disable):
            samples[i] = torchdiffeq.odeint(
                ode,
                x[i],
                t,
                rtol=self.rtol,
                atol=self.atol,
                method=self.solver,
                options={
                    "min_step": 1e-5,
                    "max_num_steps": self.max_steps,
                    "dtype": torch.float32 if torch.backends.mps.is_available() else torch.float64
                }
            )[-1]

        if callback is not None:
            sigma = sigmas.min()

            callback({
                "x": samples,
                "i": len(sigmas) - 1,
                "sigma": sigma,
                "sigma_hat": sigma,
                "denoised": samples, # only accurate if sigma = 0, for now
            })

        return samples

class ODESamplerSelect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solver": (SOLVERS, { "default": "bosh3" }),
                "log_relative_tolerance": ("FLOAT", { "min": -7, "max": 0, "default": -2.5, "step": 0.1 }),
                "log_absolute_tolerance": ("FLOAT", { "min": -7, "max": 0, "default": -3.5, "step": 0.1 }),
                "max_steps": ("INT", { "min": 1, "max": 500, "default": 30, "step": 1 }),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "sampling/custom_sampling/samplers"

    def get_sampler(self, solver, log_relative_tolerance, log_absolute_tolerance, max_steps):
        rtol = 10 ** log_relative_tolerance
        atol = 10 ** log_absolute_tolerance
        assert rtol >= atol

        return (comfy.samplers.KSAMPLER(ODESampler(solver, rtol, atol, max_steps)),)

NODE_CLASS_MAPPINGS = {
    "ODESamplerSelect": ODESamplerSelect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ODESamplerSelect": "ODE Sampler",
}
