# candle-differential-diffusion: A Diffusers API in Rust/Candle

The `differential-diffusion` example is a conversion of
[diffusers-rs](https://github.com/LaurentMazare/diffusers-rs) using candle
rather than libtorch. This implementation supports Stable Diffusion v1.5, v2.1,
as well as Stable Diffusion XL 1.0.

## Getting the weights

The weights are automatically downloaded for you from the [HuggingFace
Hub](https://huggingface.co/) on the first run. There are various command line
flags to use local files instead, run with `--help` to learn about them.

## Running some example.

```bash
cargo run --example stable-diffusion --release --features=cuda,cudnn \
    -- --input-image input.png --map-image map.png"
```

The final image is named `sd_final.png` by default.

The default scheduler for the v1.5, v2.1 and XL 1.0 version is the Denoising
Diffusion Implicit Model scheduler (DDIM). The original paper and some code can
be found in the [associated repo](https://github.com/ermongroup/ddim).
The default scheduler for the XL Turbo version is the Euler Ancestral scheduler.

### Command-line flags

- `--prompt`: the prompt to be used to generate the image.
- `--uncond-prompt`: the optional unconditional prompt.
- `--sd-version`: the Stable Diffusion version to use, can be `v1-5`, `v2-1`,
  `xl`.
- `--n-steps`: the number of steps to be used in the diffusion process.
- `--final-image`: the filename for the generated image(s).

### Using flash-attention

Using flash attention makes image generation a lot faster and uses less memory.
The downside is some long compilation time. You can set the
`CANDLE_FLASH_ATTN_BUILD_DIR` environment variable to something like
`/home/user/.candle` to ensures that the compilation artifacts are properly
cached.

Enabling flash-attention requires both a feature flag, `--features flash-attn`
and using the command line flag `--use-flash-attn`.

Note that flash-attention-v2 is only compatible with Ampere, Ada, or Hopper GPUs
(e.g., A100/H100, RTX 3090/4090).

## FAQ

### Memory Issues

This requires a GPU with more than 8GB of memory, as a fallback the CPU version can be used
with the `--cpu` flag but is much slower.
