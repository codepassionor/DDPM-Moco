from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim = 64,
    flash_attn = True,
    num_attn = 8,
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

trainer = Trainer(
    diffusion,
    'path/to/your/images',
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 10100,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = False,              # whether to calculate fid during training
    num_attn = 8,
    num_fid_samples = 100,
    num_samples=4
)


if __name__ == '__main__':
    etas = [0, 0.2, 0.5, 1.0]
    timesteps = [10, 50, 100, 250, 1000]
    results = []
    for eta in etas:
        for ts in timesteps:
            diffusion.ddim_sampling_eta = eta
            diffusion.sampling_timesteps = ts
            res = trainer.sample(8, eta, ts)
            results.append(res)

    with open(f"result_eta_ts.txt", "w", encoding="utf-8") as f:
        for item in results:
            f.write(f"{item}\n")

