use anyhow::{bail, Result};
use image::{Rgb, RgbImage};
use lab::Lab;
use ndarray::{Array3, ArrayBase, Dim, IxDynImpl, OwnedRepr};
// use opencv::core::VecN;
// use opencv::prelude::MatTraitConst;
// use opencv::prelude::MatTraitManual;
// use opencv::prelude::VideoCaptureTrait;
// use opencv::prelude::VideoCaptureTraitConst;
// use opencv::prelude::VideoWriterTrait;
// use opencv::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use tch::vision::image::{load_and_resize, load};
use std::fs::{read_dir, remove_dir_all};
use std::path::Path;
use std::time::Instant;
use tch::nn::{ConvConfig, ConvTransposeConfig, ModuleT, OptimizerConfig, Module};
use tch::{nn, Device, Kind, Tensor, CModule, IndexOp};
use tensorboard_rs as tensorboard;
use hhmmss::Hhmmss;

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

// fn colorize_opencv(frame: &mut core::Mat, net: &impl ModuleT, device: Device) -> Result<()> {
//     let size = frame.size()?;
//     let (w, h) = (size.width, size.height);
//     let data = frame.data_typed_mut::<VecN<u8, 3>>()?;
//     let mut l: Array3<f32> = ndarray::ArrayBase::zeros((1, w as usize, h as usize));
//     let mut ab: Array3<f32> = ndarray::ArrayBase::zeros((2, w as usize, h as usize));
//     for (i, pixel) in data.iter().enumerate() {
//         let lab = Lab::from_rgb(&[pixel[2], pixel[1], pixel[0]]);
//         let y = (i as f32 / (w) as f32).floor();
//         let x = i % (w as usize);
//         l[[0, x as usize, y as usize]] = (((lab.l) / 100.) - 0.5) * 2.0;
//         ab[[0, x as usize, y as usize]] = (((lab.a + 128.0) / 255.) - 0.5) * 2.0;
//         ab[[1, x as usize, y as usize]] = (((lab.b + 128.0) / 255.) - 0.5) * 2.0;
//     }
//     let l: Tensor = l.try_into()?;
//     let new_ab = tch::no_grad(|| -> anyhow::Result<Tensor> {
//         let resized_l = l.copy().to_device(device).unsqueeze(0).upsample_bicubic2d(
//             [256, 256],
//             false,
//             None,
//             None,
//         );
//         let out = net.forward_t(&resized_l, true);
//         let out = out
//             .upsample_bicubic2d([w as i64, h as i64], false, None, None)
//             .squeeze();
//         Ok(out)
//     })?;
//     let l: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>> = (&l.squeeze()).try_into()?;
//     let ab: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>> = (&new_ab).try_into()?;
//     for (i, pixel) in data.iter_mut().enumerate() {
//         let y = (i as f32 / (w) as f32).floor();
//         let x = i % (w as usize);
//         let rgb = Lab::to_rgb(&Lab {
//             l: (l[[x as usize, y as usize]] / 2.0 + 0.5) * 100.0,
//             a: (ab[[0, x as usize, y as usize]] / 2.0 + 0.5) * 255. - 128.0,
//             b: (ab[[1, x as usize, y as usize]] / 2.0 + 0.5) * 255. - 128.0,
//         });
//         pixel[2] = rgb[0];
//         pixel[1] = rgb[1];
//         pixel[0] = rgb[2];
//     }
//     Ok(())
// }

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

/// Convolutional autoencoder
#[allow(unused)]
fn conv_ae(p: nn::Path) -> impl ModuleT {
    let encoder_conv1 = nn::conv2d(
        &p / "encoder_conv1",
        1,
        64,
        3,
        ConvConfig {
            padding: 1,
            stride: 1,
            ..Default::default()
        },
    );
    let encoder_conv2 = nn::conv2d(
        &p / "encoder_conv2",
        64,
        64,
        3,
        ConvConfig {
            padding: 1,
            stride: 2,
            ..Default::default()
        },
    );
    let encoder_conv3 = nn::conv2d(
        &p / "encoder_conv3",
        64,
        128,
        3,
        ConvConfig {
            padding: 1,
            stride: 2,
            ..Default::default()
        },
    );
    let encoder_conv4 = nn::conv2d(
        &p / "encoder_conv4",
        128,
        256,
        3,
        ConvConfig {
            padding: 1,
            stride: 2,
            ..Default::default()
        },
    );
    let decoder_cfg = ConvTransposeConfig {
        stride: 2,
        padding: 1,
        output_padding: 1,
        ..Default::default()
    };
    let decoder_conv1 = nn::conv_transpose2d(&p / "decoder_conv1", 256, 128, 3, decoder_cfg);
    let decoder_conv2 = nn::conv_transpose2d(&p / "decoder_conv2", 256, 64, 3, decoder_cfg);
    let decoder_conv3 = nn::conv_transpose2d(&p / "decoder_conv3", 128, 128, 3, decoder_cfg);
    let decoder_conv4 = nn::conv_transpose2d(
        &p / "decoder_conv4",
        192,
        15,
        3,
        ConvTransposeConfig {
            stride: 1,
            padding: 1,
            ..Default::default()
        },
    );
    let converge = nn::conv2d(
        &p / "converge",
        16,
        2,
        3,
        ConvConfig {
            stride: 1,
            padding: 1,
            ..Default::default()
        },
    );
    nn::func_t(move |x, train| {
        let x1 = x.apply_t(&encoder_conv1, train).relu();
        let x2 = x1.apply_t(&encoder_conv2, train).relu();
        let x3 = x2.apply_t(&encoder_conv3, train).relu();
        let x4 = x3.apply_t(&encoder_conv4, train).relu();
        let xd = x4.apply_t(&decoder_conv1, train).relu();
        let xd = Tensor::cat(&[xd, x3], 1).dropout(0.2, train);
        let xd = xd.apply_t(&decoder_conv2, train).relu();
        let xd = Tensor::cat(&[xd, x2], 1).dropout(0.2, train);
        let xd = xd.apply_t(&decoder_conv3, train).relu();
        let xd = Tensor::cat(&[xd, x1], 1).dropout(0.2, train);
        let xd = xd.apply_t(&decoder_conv4, train).relu();
        let xd = Tensor::cat(&[xd, x.shallow_clone()], 1)
            .apply_t(&converge, train)
            .relu();
        xd
    })
}

fn main() -> Result<()> {
    let device = Device::cuda_if_available();
    let mut generator_vs = nn::VarStore::new(device);
    let generator_net = unet::unet(generator_vs.root(), 1, 2, 8, 64);
    let mut total_vars = 0usize;
    for var in generator_vs.trainable_variables() {
        total_vars += var.numel();
    }
    println!("Total trainable parameters: {}", total_vars);
    let rgb2lab = CModule::load("rgb2lab.pt")?;
    let lab2rgb = CModule::load("lab2rgb.pt")?;
    let args = std::env::args().collect::<Vec<_>>();
    match args[1].as_str() {
        "train" => {
            const BATCH_SIZE: usize = 4;
            remove_dir_all("./logdir")?;
            let mut train_writer =
                tensorboard::summary_writer::SummaryWriter::new("./logdir/train");
            //let mut test_writer = tensorboard::summary_writer::SummaryWriter::new("./logdir/test");
            let mut discriminator_vs = nn::VarStore::new(device);
            let lambda_l1 = 100.0;
            let discriminator_net = unet::PatchDiscriminator::new(discriminator_vs.root(), 3, 64, 3);
            let gan_criterion = unet::GANLoss::new(1.0, 0.0, device);
            let mut generator_opt = nn::Adam::default()
                .beta1(0.5)
                .beta2(0.999)
                .build(&generator_vs, 2e-4)?;
            let mut discriminator_opt = nn::Adam::default()
                .beta1(0.5)
                .beta2(0.999)
                .build(&discriminator_vs, 2e-4)?;
            let mut images = read_dir(&args[2])?
                .filter_map(|e| e.ok())
                .map(|p| p.path().to_string_lossy().into_owned())
                .collect::<Vec<_>>();
            let mut steps = 0usize;
            let now = Instant::now();
            let epochs = args[3].parse()?;
            let total_steps = (images.len() / BATCH_SIZE) * epochs; 
            //let mut test_steps = 0usize;
            eprintln!();
            for epoch in 1..=epochs {
                images.shuffle(&mut thread_rng());
                for images in images.chunks(BATCH_SIZE) {
                    if images.len() < BATCH_SIZE {
                        continue
                    }
                    steps += 1;
                    let xs: Vec<_> = images
                        .into_iter()
                        .map(|img_path| load_and_resize(img_path, 256, 256)).filter_map(|t| t.ok().map(|t| t.unsqueeze(0))).collect();
                    let (input, target) = convert_lab(&rgb2lab, &Tensor::cat(&xs, 0))?;
                    let target = target.to_device(device);
                    let input = input.to_device(device);
                    let fake_color = generator_net.forward_t(&input, true);
                    // optimize discriminator
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
                    // optimize generator
                    discriminator_vs.freeze();
                    generator_opt.zero_grad();
                    let fake_image =
                        Tensor::cat(&[input.shallow_clone(), fake_color.shallow_clone()], 1);
                    let fake_preds = discriminator_net.forward_t(&fake_image, true);
                    let loss_g = {
                        let loss_g_gan = gan_criterion.forward(&fake_preds, true);
                        let loss_g_l1 =
                            fake_color.l1_loss(&target, tch::Reduction::Mean) * lambda_l1;
                        loss_g_gan + loss_g_l1
                    };

                    loss_g.backward();
                    generator_opt.step();
                    train_writer.add_scalar("Generator Loss", f32::try_from(loss_g)?, steps as _);
                    // every 20 steps, send an image to tensorboard
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
                        let time_per_step = now.elapsed() / steps as u32;
                        let steps_left = total_steps - steps;
                        eprint!("{}{}", up(), erase());
                        eprintln!("Total ETA: {:?}", (steps_left as u32 * time_per_step).hhmmss());
                    }
                    if (steps * 16) % 35000 == 0{
                        generator_vs.save(&format!("checkpoint{:02}.safetensors", steps))?;
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
                generator_vs.save(&format!("model{:02}.safetensors", epoch))?;
            }
        }
        "test" => {
            generator_vs.load(&args[2])?;
            let (l, _) = load_lab(&rgb2lab, &args[3], true)?;
            let (mut full_l, _) = load_lab(&rgb2lab, &args[3], false)?;
            let (_, _, w, h) = full_l.size4()?;
            tch::no_grad(|| -> anyhow::Result<()> {
                let mut out = generator_net.forward_t(&l.to_device(device), true);
                full_l = (full_l + 1.0) * 50.0;
                out = out * 110.0;
                out = out.upsample_bicubic2d([w, h], false, None, None);
                let full = Tensor::cat(&[full_l.to_device(device), out], 1);
                let full = (lab2rgb.forward_t(&full, false).to_device(Device::Cpu) * 255.0).to_kind(Kind::Uint8);
                tch::vision::image::save(&full.squeeze(), "fixed.png")?;
                Ok(())
            })?;
        }
        // "video" => {
        //     generator_vs.load(&args[2])?;
        //     let file_name = &args[3];
        //     let mut cam = videoio::VideoCapture::from_file(&file_name, videoio::CAP_ANY)?;
        //     let opened_file =
        //         videoio::VideoCapture::open_file(&mut cam, &file_name, videoio::CAP_ANY)?;
        //     if !opened_file {
        //         panic!("Unable to open video file2!");
        //     };
        //     let mut frame = core::Mat::default();
        //     let frame_read = videoio::VideoCapture::read(&mut cam, &mut frame)?;
        //     if !frame_read {
        //         panic!("Unable to read from video file!");
        //     };
        //     let opened = videoio::VideoCapture::is_opened(&mut cam)?;
        //     println!("Opened? {}", opened);
        //     if !opened {
        //         panic!("Unable to open video file!");
        //     };
        //     let mut output = videoio::VideoWriter::new(
        //         "out.mp4",
        //         videoio::VideoWriter::fourcc('M', 'J', 'P', 'G')?,
        //         cam.get(videoio::CAP_PROP_FPS)?,
        //         frame.size()?,
        //         true,
        //     )?;
        //     let mut frame_num = 0;
        //     loop {
        //         videoio::VideoCapture::read(&mut cam, &mut frame)?;
        //         if frame.size()?.width > 0 {
        //             println!("Writing frame {}", frame_num);
        //             colorize_opencv(&mut frame, &generator_net, device)?;
        //             output.write(&frame)?;
        //             frame_num += 1;
        //         } else {
        //             println!("No more frames!");
        //             videoio::VideoCapture::release(&mut cam)?;
        //             break ();
        //         }
        //     }
        // }
        // "display" => {
        //     generator_vs.load(&args[2])?;
        //     let window = "Video Display";
        //     highgui::named_window(window, 1)?;
        //     let file_name = &args[3];
        //     let mut cam = videoio::VideoCapture::from_file(&file_name, videoio::CAP_ANY)?;
        //     let opened_file =
        //         videoio::VideoCapture::open_file(&mut cam, &file_name, videoio::CAP_ANY)?;
        //     if !opened_file {
        //         panic!("Unable to open video file2!");
        //     };
        //     let mut frame = core::Mat::default();
        //     let frame_read = videoio::VideoCapture::read(&mut cam, &mut frame)?;
        //     if !frame_read {
        //         panic!("Unable to read from video file!");
        //     };
        //     let opened = videoio::VideoCapture::is_opened(&mut cam)?;
        //     println!("Opened? {}", opened);
        //     if !opened {
        //         panic!("Unable to open video file!");
        //     };
        //     let mut frame_num = 0;
        //     loop {
        //         videoio::VideoCapture::read(&mut cam, &mut frame)?;
        //         if frame.size()?.width > 0 {
        //             if frame_num % 60 == 0 {
        //                 println!("Got a frame!");
        //                 colorize_opencv(&mut frame, &generator_net, device)?;
        //                 highgui::imshow(window, &frame)?;
        //                 #[allow(unused)]
        //                 let key = highgui::wait_key(1000)?;
        //             }
        //             frame_num += 1;
        //         } else {
        //             println!("No more frames!");
        //             videoio::VideoCapture::release(&mut cam)?;
        //             break ();
        //         }
        //     }
        // }
        _ => bail!("Usage: (train|test|video) model-path file-path"),
    }
    Ok(())
}

fn up() -> String {
    format!("{}[A", ESC)
}

fn erase() -> String {
    format!("{}[2K", ESC)
}

const ESC: char = 27u8 as char;