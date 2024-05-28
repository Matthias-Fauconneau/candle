#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use candle_transformers::models::stable_diffusion;

use anyhow::{Error as E, Result};
use candle::{DType, Device, IndexOp, Module, Tensor, D};
use clap::Parser;
use stable_diffusion::vae::AutoEncoderKL;
use tokenizers::Tokenizer;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
 	/// The input image that will be inpainted.
    #[arg(long, value_name = "FILE")]
    input_image: String,

    /// The map image to be used for inpainting, white pixels are repainted whereas black pixels are preserved.
    #[arg(long, value_name = "FILE")]
    map_image: String,

    /// The prompt to be used for image generation.
    #[arg(long, default_value = "")]
    prompt: String,

    #[arg(long, default_value = "")]
    uncond_prompt: String,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// The UNet weight file, in .safetensors format.
    #[arg(long, value_name = "FILE")]
    unet_weights: Option<String>,

    /// The CLIP weight file, in .safetensors format.
    #[arg(long, value_name = "FILE")]
    clip_weights: Option<String>,

    /// The VAE weight file, in .safetensors format.
    #[arg(long, value_name = "FILE")]
    vae_weights: Option<String>,

    #[arg(long, value_name = "FILE")]
    /// The file specifying the tokenizer to used for tokenization.
    tokenizer: Option<String>,

    /// The size of the sliced attention or 0 for automatic slicing (disabled by default)
    #[arg(long)]
    sliced_attention_size: Option<usize>,

    /// The number of steps to run the diffusion for.
    #[arg(long)]
    n_steps: Option<usize>,

    /// The number of samples to generate iteratively.
    #[arg(long, default_value_t = 1)]
    num_samples: usize,

    /// The numbers of samples to generate simultaneously.
    #[arg[long, default_value_t = 1]]
    bsize: usize,

    /// The name of the final image to generate.
    #[arg(long, value_name = "FILE", default_value = "sd_final.png")]
    final_image: String,

    #[arg(long, value_enum, default_value = "v2-1")]
    sd_version: StableDiffusionVersion,

    /// Generate intermediary images at each step.
    #[arg(long, action)]
    intermediary_images: bool,

    #[arg(long)]
    use_flash_attn: bool,

    #[arg(long)]
    use_f16: bool,

    #[arg(long)]
    guidance_scale: Option<f64>,

    /// The strength, indicates how much to transform the initial image. The
    /// value must be between 0 and 1, a value of 1 discards the initial image
    /// information.
    #[arg(long, default_value_t = 0.8)]
    img2img_strength: f64,

    /// The seed to use when generating random samples.
    #[arg(long)]
    seed: Option<u64>,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
enum StableDiffusionVersion {
    V1_5,
    V2_1,
    Xl,
    Turbo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelFile {
    Tokenizer,
    Tokenizer2,
    Clip,
    Clip2,
    Unet,
    Vae,
}

impl StableDiffusionVersion {
    fn repo(&self) -> &'static str {
        match self {
            Self::Xl => "stabilityai/stable-diffusion-xl-base-1.0",
            Self::V2_1 => "stabilityai/stable-diffusion-2-1",
            Self::V1_5 => "runwayml/stable-diffusion-v1-5",
            Self::Turbo => "stabilityai/sdxl-turbo",
        }
    }

    fn unet_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::V1_5 | Self::V2_1 | Self::Xl | Self::Turbo => {
                if use_f16 {
                    "unet/diffusion_pytorch_model.fp16.safetensors"
                } else {
                    "unet/diffusion_pytorch_model.safetensors"
                }
            }
        }
    }

    fn vae_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::V1_5 | Self::V2_1 | Self::Xl | Self::Turbo => {
                if use_f16 {
                    "vae/diffusion_pytorch_model.fp16.safetensors"
                } else {
                    "vae/diffusion_pytorch_model.safetensors"
                }
            }
        }
    }

    fn clip_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::V1_5 | Self::V2_1 | Self::Xl | Self::Turbo => {
                if use_f16 {
                    "text_encoder/model.fp16.safetensors"
                } else {
                    "text_encoder/model.safetensors"
                }
            }
        }
    }

    fn clip2_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::V1_5 | Self::V2_1 | Self::Xl | Self::Turbo => {
                if use_f16 {
                    "text_encoder_2/model.fp16.safetensors"
                } else {
                    "text_encoder_2/model.safetensors"
                }
            }
        }
    }
}

impl ModelFile {
    fn get(
        &self,
        filename: Option<String>,
        version: StableDiffusionVersion,
        use_f16: bool,
    ) -> Result<std::path::PathBuf> {
        use hf_hub::api::sync::Api;
        match filename {
            Some(filename) => Ok(std::path::PathBuf::from(filename)),
            None => {
                let (repo, path) = match self {
                    Self::Tokenizer => {
                        let tokenizer_repo = match version {
                            StableDiffusionVersion::V1_5 | StableDiffusionVersion::V2_1 => {
                                "openai/clip-vit-base-patch32"
                            }
                            StableDiffusionVersion::Xl | StableDiffusionVersion::Turbo => {
                                // This seems similar to the patch32 version except some very small
                                // difference in the split regex.
                                "openai/clip-vit-large-patch14"
                            }
                        };
                        (tokenizer_repo, "tokenizer.json")
                    }
                    Self::Tokenizer2 => {
                        ("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", "tokenizer.json")
                    }
                    Self::Clip => (version.repo(), version.clip_file(use_f16)),
                    Self::Clip2 => (version.repo(), version.clip2_file(use_f16)),
                    Self::Unet => (version.repo(), version.unet_file(use_f16)),
                    Self::Vae => {
                        // Override for SDXL when using f16 weights.
                        // See https://github.com/huggingface/candle/issues/1060
                        if matches!(
                            version,
                            StableDiffusionVersion::Xl | StableDiffusionVersion::Turbo,
                        ) && use_f16
                        {
                            (
                                "madebyollin/sdxl-vae-fp16-fix",
                                "diffusion_pytorch_model.safetensors",
                            )
                        } else {
                            (version.repo(), version.vae_file(use_f16))
                        }
                    }
                };
                let filename = Api::new()?.model(repo.to_string()).get(path)?;
                Ok(filename)
            }
        }
    }
}

fn output_filename(
    basename: &str,
    sample_idx: usize,
    num_samples: usize,
    timestep_idx: Option<usize>,
) -> String {
    let filename = if num_samples > 1 {
        match basename.rsplit_once('.') {
            None => format!("{basename}.{sample_idx}.png"),
            Some((filename_no_extension, extension)) => {
                format!("{filename_no_extension}.{sample_idx}.{extension}")
            }
        }
    } else {
        basename.to_string()
    };
    match timestep_idx {
        None => filename,
        Some(timestep_idx) => match filename.rsplit_once('.') {
            None => format!("{filename}-{timestep_idx}.png"),
            Some((filename_no_extension, extension)) => {
                format!("{filename_no_extension}-{timestep_idx}.{extension}")
            }
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn save_image(
    vae: &AutoEncoderKL,
    latents: &Tensor,
    vae_scale: f64,
    bsize: usize,
    idx: usize,
    final_image: &str,
    num_samples: usize,
    timestep_ids: Option<usize>,
) -> Result<()> {
    let images = vae.decode(&(latents / vae_scale)?)?;
    let images = ((images / 2.)? + 0.5)?.to_device(&Device::Cpu)?;
    let images = (images.clamp(0f32, 1.)? * 255.)?.to_dtype(DType::U8)?;
    for batch in 0..bsize {
        let image = images.i(batch)?;
        let image_filename = output_filename(
            final_image,
            (bsize * idx) + batch + 1,
            batch + num_samples,
            timestep_ids,
        );
        candle_examples::save_image(&image, image_filename)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn text_embeddings(
    prompt: &str,
    uncond_prompt: &str,
    tokenizer: Option<String>,
    clip_weights: Option<String>,
    sd_version: StableDiffusionVersion,
    sd_config: &stable_diffusion::StableDiffusionConfig,
    use_f16: bool,
    device: &Device,
    dtype: DType,
    use_guide_scale: bool,
    first: bool,
) -> Result<Tensor> {
    let tokenizer_file = if first {
        ModelFile::Tokenizer
    } else {
        ModelFile::Tokenizer2
    };
    let tokenizer = tokenizer_file.get(tokenizer, sd_version, use_f16)?;
    let tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;
    let pad_id = match &sd_config.clip.pad_with {
        Some(padding) => *tokenizer.get_vocab(true).get(padding.as_str()).unwrap(),
        None => *tokenizer.get_vocab(true).get("<|endoftext|>").unwrap(),
    };
    println!("Running with prompt \"{prompt}\".");
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    if tokens.len() > sd_config.clip.max_position_embeddings {
        anyhow::bail!(
            "the prompt is too long, {} > max-tokens ({})",
            tokens.len(),
            sd_config.clip.max_position_embeddings
        )
    }
    while tokens.len() < sd_config.clip.max_position_embeddings {
        tokens.push(pad_id)
    }
    let tokens = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;

    println!("Building the Clip transformer.");
    let clip_weights_file = if first {
        ModelFile::Clip
    } else {
        ModelFile::Clip2
    };
    let clip_weights = clip_weights_file.get(clip_weights, sd_version, false)?;
    let clip_config = if first {
        &sd_config.clip
    } else {
        sd_config.clip2.as_ref().unwrap()
    };
    let text_model =
        stable_diffusion::build_clip_transformer(clip_config, clip_weights, device, DType::F32)?;
    let text_embeddings = text_model.forward(&tokens)?;

    let text_embeddings = if use_guide_scale {
        let mut uncond_tokens = tokenizer
            .encode(uncond_prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        if uncond_tokens.len() > sd_config.clip.max_position_embeddings {
            anyhow::bail!(
                "the negative prompt is too long, {} > max-tokens ({})",
                uncond_tokens.len(),
                sd_config.clip.max_position_embeddings
            )
        }
        while uncond_tokens.len() < sd_config.clip.max_position_embeddings {
            uncond_tokens.push(pad_id)
        }

        let uncond_tokens = Tensor::new(uncond_tokens.as_slice(), device)?.unsqueeze(0)?;
        let uncond_embeddings = text_model.forward(&uncond_tokens)?;

        Tensor::cat(&[uncond_embeddings, text_embeddings], 0)?.to_dtype(dtype)?
    } else {
        text_embeddings.to_dtype(dtype)?
    };
    Ok(text_embeddings)
}

fn image_preprocess<T: AsRef<std::path::Path>>(path: T) -> anyhow::Result<Tensor> {
    let img = image::io::Reader::open(path)?.decode()?;
    let (height, width) = (img.height() as usize, img.width() as usize);
    let height = height - height % 32;
    let width = width - width % 32;
    let img = img.resize_to_fill(
        width as u32,
        height as u32,
        image::imageops::FilterType::CatmullRom,
    );
    let img = img.to_rgb8();
    let img = img.into_raw();
    let img = Tensor::from_vec(img, (height, width, 3), &Device::Cpu)?
        .permute((2, 0, 1))?
        .to_dtype(DType::F32)?
        .affine(2. / 255., -1.)?
    ;//.unsqueeze(0)?;
    Ok(img)
}

fn map_preprocess<T: AsRef<std::path::Path>>(path: T, ) -> anyhow::Result<Tensor> {
    let img = image::io::Reader::open(path)?.decode()?;
    let (height, width) = (img.height() as usize, img.width() as usize);
    let height = (height - height % 32)/8;
    let width = (width - width % 32)/8;
    let img = img.resize_to_fill(
        width as u32,
        height as u32,
        image::imageops::FilterType::CatmullRom,
    );
    let img = img.to_luma8();
    let img = img.into_raw();
    let img = Tensor::from_vec(img, (height, width, 1), &Device::Cpu)?
    	.permute((2, 0, 1))?
        .to_dtype(DType::F32)?
        .affine(1. / 255., 0.)?
    ;//.unsqueeze(0)?;
    Ok(img)
}

fn run(args: Args) -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let Args {
    	input_image,
     	map_image,
        prompt,
        uncond_prompt,
        n_steps,
        tokenizer,
        final_image,
        sliced_attention_size,
        num_samples,
        bsize,
        sd_version,
        clip_weights,
        vae_weights,
        unet_weights,
        tracing,
        use_f16,
        guidance_scale,
        use_flash_attn,
        seed,
        ..
    } = args;

    let _guard = if tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    let guidance_scale = match guidance_scale {
        Some(guidance_scale) => guidance_scale,
        None => match sd_version {
            StableDiffusionVersion::V1_5
            | StableDiffusionVersion::V2_1
            | StableDiffusionVersion::Xl => 7.5,
            StableDiffusionVersion::Turbo => 0.,
        },
    };
    let n_steps = match n_steps {
        Some(n_steps) => n_steps,
        None => match sd_version {
            StableDiffusionVersion::V1_5
            | StableDiffusionVersion::V2_1
            | StableDiffusionVersion::Xl => 30,
            StableDiffusionVersion::Turbo => 1,
        },
    };
    let dtype = if use_f16 { DType::F16 } else { DType::F32 };
    let device = candle_examples::device(false)?;
    let image = image_preprocess(input_image)?.to_device(&device)?;
    let (3, height, width) = image.dims3()? else {panic!()};
    //let (1, 3, height, width) = image.dims4()? else {panic!()};
    println!("{width} {height}");
    let sd_config = match sd_version {
        StableDiffusionVersion::V1_5 => stable_diffusion::StableDiffusionConfig::v1_5(sliced_attention_size, Some(height), Some(width)),
        StableDiffusionVersion::V2_1 => stable_diffusion::StableDiffusionConfig::v2_1(sliced_attention_size, Some(height), Some(width)),
        StableDiffusionVersion::Xl => stable_diffusion::StableDiffusionConfig::sdxl(sliced_attention_size, Some(height), Some(width)),
        StableDiffusionVersion::Turbo => stable_diffusion::StableDiffusionConfig::sdxl_turbo(sliced_attention_size, Some(height), Some(width)),
    };

    let scheduler = sd_config.build_scheduler(n_steps)?;
    if let Some(seed) = seed {
        device.set_seed(seed)?;
    }
    let use_guide_scale = guidance_scale > 1.0;

    let which = match sd_version {
        StableDiffusionVersion::Xl | StableDiffusionVersion::Turbo => vec![true, false],
        _ => vec![true],
    };
    let text_embeddings = which
        .iter()
        .map(|first| {
            text_embeddings(
                &prompt,
                &uncond_prompt,
                tokenizer.clone(),
                clip_weights.clone(),
                sd_version,
                &sd_config,
                use_f16,
                &device,
                dtype,
                use_guide_scale,
                *first,
            )
        })
        .collect::<Result<Vec<_>>>()?;

    let text_embeddings = Tensor::cat(&text_embeddings, D::Minus1)?;
    let text_embeddings = text_embeddings.repeat((bsize, 1, 1))?;
    println!("{text_embeddings:?}");

    println!("Building the autoencoder.");
    let vae_weights = ModelFile::Vae.get(vae_weights, sd_version, use_f16)?;
    let vae = sd_config.build_vae(vae_weights, &device, dtype)?;
    println!("Encode");
    let initial_latent_distribution = vae.encode(&image.unsqueeze(0)?)?;

    println!("Building the unet.");
    let unet_weights = ModelFile::Unet.get(unet_weights, sd_version, use_f16)?;
    let unet = sd_config.build_unet(unet_weights, &device, 4, use_flash_attn, dtype)?;

    let vae_scale = match sd_version {
        StableDiffusionVersion::V1_5
        | StableDiffusionVersion::V2_1
        | StableDiffusionVersion::Xl => 0.18215,
        StableDiffusionVersion::Turbo => 0.13025,
    };

    let timesteps = scheduler.timesteps();
    let initial_latents = (initial_latent_distribution.sample()?.squeeze(0)? * vae_scale)?.to_device(&device)?;
    let noise = initial_latents.randn_like(0f64, 1f64)?;
    let initial_latents_with_noise = scheduler.add_noise(&initial_latents, noise, timesteps[0])?.to_dtype(dtype)?;
    let mut latents = initial_latents_with_noise.clone();

    println!("Loading the map");
    let map = map_preprocess(map_image)?;

    println!("Sampling");
    for (timestep_index, &timestep) in timesteps.iter().enumerate() {
    	let noise = initial_latents.randn_like(0f64, 1f64)?;
    	let initial_latents_with_noise_t = scheduler.add_noise(&initial_latents, noise, timestep)?.to_dtype(dtype)?;
     	let mask = map.gt(timestep as f32/n_steps as f32)?.squeeze(0)?.to_dtype(dtype)?;
      	let mask = Tensor::stack(&[&mask,&mask,&mask,&mask], 0)?.to_device(&device)?;
    	latents = ((initial_latents_with_noise_t * mask.clone())? + ((1.-mask) * latents)?)?;
        let start_time = std::time::Instant::now();
        let latent_model_input = if use_guide_scale {
            Tensor::cat(&[&latents.unsqueeze(0)?, &latents.unsqueeze(0)?], 0)?
        } else {
            latents.unsqueeze(0)?.clone()
        };

        let latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)?;
        let noise_pred = unet.forward(&latent_model_input, timestep as f64, &text_embeddings)?;

        let noise_pred = if use_guide_scale {
            let noise_pred = noise_pred.chunk(2, 0)?;
            let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);
            (noise_pred_uncond + ((noise_pred_text - noise_pred_uncond)? * guidance_scale)?)?
        } else {
            noise_pred
        };

        latents = scheduler.step(&noise_pred, timestep, &latents.unsqueeze(0)?)?.squeeze(0)?;
        let dt = start_time.elapsed().as_secs_f32();
        println!("{}/{n_steps}: {:.1}s", timestep_index + 1, dt);

        if args.intermediary_images {
            save_image(
                &vae,
                &latents,
                vae_scale,
                bsize,
                0,
                &final_image,
                num_samples,
                Some(timestep_index + 1),
            )?;
        }
    }

    println!("Generating the final image",);
    save_image(
        &vae,
        &latents.unsqueeze(0)?,
        vae_scale,
        bsize,
        0,
        &final_image,
        num_samples,
        None,
    )?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    run(args)
}
