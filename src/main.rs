use anyhow::{bail, Result};
use image::{Rgb, RgbImage};
use lab::Lab;
use ndarray::{ArrayBase, Dim, IxDynImpl, OwnedRepr};
use opencv::core::VecN;
use opencv::prelude::MatTraitConst;
use opencv::prelude::MatTraitManual;
use opencv::prelude::VideoCaptureTrait;
use opencv::prelude::VideoCaptureTraitConst;
use opencv::prelude::VideoWriterTrait;
use opencv::*;
use ndarray::Array3;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::fs::remove_dir_all;
use std::path::Path;
use std::time::{Duration, Instant};
use tch::nn::{Module, ModuleT, OptimizerConfig, VarStore};
use tch::vision::image::{load, load_and_resize};
use tch::{nn, CModule, Device, IndexOp, Kind, Tensor, TrainableCModule};
use tensorboard_rs as tensorboard;
use walkdir::WalkDir;

mod unet;

/// Returns (L, ab)
fn load_lab(rgb2lab: &CModule, img: impl AsRef<Path>, resize: bool) -> Result<(Tensor, Tensor)> {
    let img = if resize {
        load_and_resize(img, 256, 256)?
    } else {
        load(img)?
    };
    let img = img.to_kind(Kind::Float) / 255.0;
    let lab = rgb2lab.forward(&img.unsqueeze(0));
    let l = lab.i((.., (0..1), .., ..)) / 50.0 - 1.0;
    let ab = lab.i((.., (1..3), .., ..)) / 110.;
    Ok((l, ab))
}

/// Returns (L, ab)
fn convert_lab(rgb2lab: &CModule, xs: &Tensor) -> Result<(Tensor, Tensor)> {
    let img = xs.to_kind(Kind::Float) / 255.0;
    let lab = rgb2lab.forward(&img);
    let l = lab.i((.., (0..1), .., ..)) / 50.0 - 1.0;
    let ab = lab.i((.., (1..3), .., ..)) / 110.;
    Ok((l, ab))
}

fn colorize_opencv(rgb2lab: &CModule, lab2rgb: &CModule, frame: &mut core::Mat, net: &impl ModuleT, device: Device) -> Result<()> {
    let size = frame.size()?;
    let (w, h) = (size.width, size.height);
    let data = frame.data_typed_mut::<VecN<u8, 3>>()?;
    let mut rgb: Array3<f32> = ndarray::ArrayBase::zeros((3, h as usize, w as usize));
    for (i, pixel) in data.iter().enumerate() {
        let y = (i as f32 / (w) as f32).floor();
        let x = i % (w as usize);
        rgb[[0, y as usize, x as usize]] = pixel[2] as f32;
        rgb[[1, y as usize, x as usize]] = pixel[1] as f32;
        rgb[[2, y as usize, x as usize]] = pixel[0] as f32;
    }
    let rgb: Tensor = rgb.try_into()?;
    let (mut l, _) = convert_lab(rgb2lab, &rgb.unsqueeze(0))?;
    let mut out = tch::no_grad(|| -> anyhow::Result<Tensor> {
        let resized_l = l.to_device(device).upsample_bicubic2d(
            [256, 256],
            false,
            None,
            None,
        );
        Ok(net.forward_t(&resized_l.repeat([1, 3, 1, 1]), false))
    })?;
    l = (l + 1.0) * 50.0;
    out = out * 110.0;
    out = out.upsample_bicubic2d([h as i64, w as i64], false, None, None);
    let full = Tensor::cat(&[l.to_device(device), out], 1);
    let full = (lab2rgb.forward_t(&full, false).to_device(Device::Cpu) * 255.0)
        .to_kind(Kind::Uint8);
    let rgb: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>> = (&full.squeeze()).try_into()?;
    for (i, pixel) in data.iter_mut().enumerate() {
        let y = (i as f32 / (w) as f32).floor();
        let x = i % (w as usize);
        pixel[2] = rgb[[0, y as usize, x as usize]] as u8;
        pixel[1] = rgb[[1, y as usize, x as usize]] as u8;
        pixel[0] = rgb[[2, y as usize, x as usize]] as u8;
    }
    Ok(())
}

fn lab_to_rgb(l: &Tensor, ab: &Tensor) -> Result<RgbImage> {
    let (w, h) = l.size2()?;
    let l: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>> = l.try_into()?;
    let ab: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>> = ab.try_into()?;
    let mut image = RgbImage::new(h as _, w as _);
    for x in 0..w {
        for y in 0..h {
            let rgb = Lab::to_rgb(&Lab {
                l: (l[[x as usize, y as usize]] / 2.0 + 0.5) * 100.0,
                a: (ab[[0, x as usize, y as usize]] / 2.0 + 0.5) * 255. - 128.0,
                b: (ab[[1, x as usize, y as usize]] / 2.0 + 0.5) * 255. - 128.0,
            });
            image.put_pixel(y as _, x as _, Rgb(rgb))
        }
    }
    Ok(image)
}

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    let device = Device::cuda_if_available();
    let generator_vs = nn::VarStore::new(device);
    let mut generator_net = TrainableCModule::load(&args[2], generator_vs.root())?;
    generator_net.set_train();
    //let generator_net = unet::unet(generator_vs.root(), 1, 2, 8, 64);
    let mut total_vars = 0usize;
    for var in generator_vs.trainable_variables() {
        total_vars += var.numel();
    }
    println!("Total trainable parameters: {}", total_vars);
    let rgb2lab = CModule::load("rgb2lab.pt")?;
    let lab2rgb = CModule::load("lab2rgb.pt")?;
    match args[1].as_str() {
        "train" => {
            const BATCH_SIZE: usize = 16;
            remove_dir_all("./logdir").ok();
            let mut train_writer =
                tensorboard::summary_writer::SummaryWriter::new("./logdir/train");
            //let mut test_writer = tensorboard::summary_writer::SummaryWriter::new("./logdir/test");
            let mut discriminator_vs = nn::VarStore::new(device);
            let lambda_l1 = 100.0;
            let discriminator_net =
                unet::PatchDiscriminator::new(discriminator_vs.root(), 3, 64, 3);
            let gan_criterion = unet::GANLoss::new(1.0, 0.0, device);
            let mut generator_opt = nn::Adam::default()
                .beta1(0.5)
                .beta2(0.999)
                .build(&generator_vs, 2e-4)?;
            let mut discriminator_opt = nn::Adam::default()
                .beta1(0.5)
                .beta2(0.999)
                .build(&discriminator_vs, 2e-4)?;
            let total_count = 1000 * 1300;
            let mut completed = 0;
            let mut images: Vec<String> = WalkDir::new(&args[3])
                .max_open(1300)
                .into_iter()
                .filter_map(|entry| {
                    let entry = entry.unwrap();
                    if entry.file_type().is_file() {
                        completed += 1;
                        if completed % 100000 == 0 {
                            println!(
                                "Completed {:.2}%",
                                (completed as f32 / total_count as f32) * 100.0
                            );
                        }
                        Some(entry.path().display().to_string())
                    } else {
                        None
                    }
                })
                .collect();
            println!("Directory exploration complete!");
            println!("{} images found", images.len());
            let duration = Duration::from_secs_f32(args[4].parse::<f32>()? * 3600.0);
            let now = Instant::now();
            let mut steps = 0usize;
            let from_checkpoint = args[5] == "true";
            //let mut test_steps = 0usize;
            eprintln!();
            for epoch in 1.. {
                images.shuffle(&mut thread_rng());
                println!("Image shuffling complete!");
                for images in images.chunks(BATCH_SIZE) {
                    if images.len() < BATCH_SIZE {
                        continue;
                    }
                    steps += 1;
                    let xs: Vec<_> = images
                        .into_iter()
                        .map(|img_path| load_and_resize(img_path, 256, 256))
                        .filter_map(|t| t.ok().map(|t| t.unsqueeze(0)))
                        .collect();
                    let (input, target) = convert_lab(&rgb2lab, &Tensor::cat(&xs, 0))?;
                    let target = target.to_device(device);
                    let input = input.to_device(device);
                    let fake_color = generator_net.forward_t(&input.repeat(&[1, 3, 1, 1]), true);
                    let greater_than_half = now.elapsed() >= (duration / 2) || from_checkpoint;
                    // optimize discriminator
                    if greater_than_half {
                        discriminator_vs.unfreeze();
                        discriminator_opt.zero_grad();
                        let fake_image =
                            Tensor::cat(&[input.shallow_clone(), fake_color.shallow_clone()], 1);
                        let fake_preds = discriminator_net.forward_t(&fake_image.detach(), true);
                        let loss_d_fake = gan_criterion.forward(&fake_preds, false);
                        let real_image =
                            Tensor::cat(&[input.shallow_clone(), target.shallow_clone()], 1);
                        let real_preds = discriminator_net.forward_t(&real_image, true);
                        let loss_d_real = gan_criterion.forward(&real_preds, true);
                        let loss_d = (loss_d_fake + loss_d_real) * 0.5;
                        loss_d.backward();
                        discriminator_opt.step();
                        train_writer.add_scalar(
                            "Discriminator Loss",
                            f32::try_from(loss_d)?,
                            steps as _,
                        );
                    }
                    // optimize generator
                    discriminator_vs.freeze();
                    generator_opt.zero_grad();
                    let loss_g = match greater_than_half {
                        true => {
                            let fake_image = Tensor::cat(
                                &[input.shallow_clone(), fake_color.shallow_clone()],
                                1,
                            );
                            let fake_preds = discriminator_net.forward_t(&fake_image, true);
                            let loss_g_gan = gan_criterion.forward(&fake_preds, true);
                            let loss_g_l1 =
                                fake_color.l1_loss(&target, tch::Reduction::Mean) * lambda_l1;
                            loss_g_gan + loss_g_l1
                        }
                        false => fake_color.l1_loss(&target, tch::Reduction::Mean),
                    };
                    loss_g.backward();
                    generator_opt.step();
                    train_writer.add_scalar("Generator Loss", f32::try_from(loss_g)?, steps as _);
                    if steps % (20 * 4) == 0 {
                        let l = input.narrow(0, 0, 1).squeeze();
                        let ab = fake_color.narrow(0, 0, 1).squeeze();
                        let img = lab_to_rgb(&l, &ab)?;
                        img.save("test.png")?;
                        let (w, h) = img.dimensions();
                        train_writer.add_image(
                            "Sample",
                            img.as_raw(),
                            &[3, w as usize, h as usize],
                            0,
                        );
                    }

                    if now.elapsed() >= duration {
                        generator_net.save("final.pt")?;
                        println!("Completed and saved after {} training steps.", steps);
                        return Ok(());
                    }
                }
                // for images in test_images.chunks(16) {
                //     test_steps += 1;
                //     let (inputs, outputs): (Vec<_>, Vec<_>) = images
                //         .into_iter()
                //         .map(|img_path| load_lab(img_path, true).expect("failed to open image"))
                //         .unzip();
                //     let input = Tensor::stack(&inputs, 0);
                //     let output = Tensor::stack(&outputs, 0);
                //     let out = generator_net.forward_t(&input.to(device), true);
                //     let loss = out.l1_loss(&output.to(device), tch::Reduction::Mean) * lambda_l1;
                //     test_writer.add_scalar("Generator Loss", f32::try_from(loss)?, test_steps as _);
                // }
                println!("epoch: {:02} complete!", epoch);
                generator_net.save(&format!("model{:02}.pt", epoch))?;
            }
        }
        "merge" => {
            generator_net.set_eval();
            let other_vs = VarStore::new(device);
            let mut _unused = TrainableCModule::load(&args[3], other_vs.root())?;
            _unused.set_eval();
            let other_vars = other_vs.variables();
            tch::no_grad(|| {
                for (key, var) in generator_vs.variables_.lock().unwrap().named_variables.iter_mut() {
                    let other_var = &other_vars[key];
                    *var *= 0.75;
                    *var += other_var * 0.25;
                    //*var /= 2;
                }
            });
            generator_net.save("merged.pt")?;
        }
        "test" => {
            generator_net.set_eval();
            let (mut l, _) = load_lab(&rgb2lab, &args[3], false)?;
            let (_, _, w, h) = l.size4()?;
            tch::no_grad(|| -> anyhow::Result<()> {
                let small_l = l.upsample_bicubic2d([args[4].parse()?, args[4].parse()?], false, None, None);
                let mut out =
                    generator_net.forward_t(&small_l.to_device(device).repeat(&[1, 3, 1, 1]), false);
                l = (l + 1.0) * 50.0;
                out = out * 110.0;
                out = out.upsample_bicubic2d([w, h], false, None, None);
                let full = Tensor::cat(&[l.to_device(device), out], 1);
                let full = (lab2rgb.forward_t(&full, false).to_device(Device::Cpu) * 255.0)
                    .to_kind(Kind::Uint8);
                tch::vision::image::save(&full.squeeze(), "fixed.jpg")?;
                Ok(())
            })?;
        }
        "video" => {
            generator_net.set_eval();
            let file_name = &args[3];
            let mut cam = videoio::VideoCapture::from_file(&file_name, videoio::CAP_ANY)?;
            let opened_file =
                videoio::VideoCapture::open_file(&mut cam, &file_name, videoio::CAP_ANY)?;
            if !opened_file {
                panic!("Unable to open video file2!");
            };
            let mut frame = core::Mat::default();
            let frame_read = videoio::VideoCapture::read(&mut cam, &mut frame)?;
            if !frame_read {
                panic!("Unable to read from video file!");
            };
            let opened = videoio::VideoCapture::is_opened(&mut cam)?;
            println!("Opened? {}", opened);
            if !opened {
                panic!("Unable to open video file!");
            };
            let mut output = videoio::VideoWriter::new(
                "out.mp4",
                videoio::VideoWriter::fourcc('M', 'J', 'P', 'G')?,
                cam.get(videoio::CAP_PROP_FPS)?,
                frame.size()?,
                true,
            )?;
            let mut frame_num = 0;
            loop {
                videoio::VideoCapture::read(&mut cam, &mut frame)?;
                if frame.size()?.width > 0 {
                    println!("Writing frame {}", frame_num);
                    colorize_opencv(&rgb2lab, &lab2rgb, &mut frame, &generator_net, device)?;
                    output.write(&frame)?;
                    frame_num += 1;
                } else {
                    println!("No more frames!");
                    videoio::VideoCapture::release(&mut cam)?;
                    break ();
                }
            }
        }
        "display" => {
            generator_net.set_eval();
            let window = "Video Display";
            highgui::named_window(window, 1)?;
            let file_name = &args[3];
            let mut cam = videoio::VideoCapture::from_file(&file_name, videoio::CAP_ANY)?;
            let opened_file =
                videoio::VideoCapture::open_file(&mut cam, &file_name, videoio::CAP_ANY)?;
            if !opened_file {
                panic!("Unable to open video file2!");
            };
            let mut frame = core::Mat::default();
            let frame_read = videoio::VideoCapture::read(&mut cam, &mut frame)?;
            if !frame_read {
                panic!("Unable to read from video file!");
            };
            let opened = videoio::VideoCapture::is_opened(&mut cam)?;
            println!("Opened? {}", opened);
            if !opened {
                panic!("Unable to open video file!");
            };
            let mut frame_num = 0;
            loop {
                videoio::VideoCapture::read(&mut cam, &mut frame)?;
                if frame.size()?.width > 0 {
                    if frame_num % 60 == 0 {
                        println!("Got a frame!");
                        colorize_opencv(&rgb2lab, &lab2rgb, &mut frame, &generator_net, device)?;
                        highgui::imshow(window, &frame)?;
                        #[allow(unused)]
                        let key = highgui::wait_key(1000)?;
                    }
                    frame_num += 1;
                } else {
                    println!("No more frames!");
                    videoio::VideoCapture::release(&mut cam)?;
                    break ();
                }
            }
        }
        _ => bail!("Usage: (train|test|video) model-path file-path"),
    }
    Ok(())
}
