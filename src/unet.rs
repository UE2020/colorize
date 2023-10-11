use tch::nn::{conv2d, BatchNormConfig, ConvConfig, ConvTransposeConfig, Init, ModuleT};
use tch::{nn, Device, Tensor};

#[derive(Debug)]
pub struct PatchDiscriminator {
    seq: nn::SequentialT,
}

impl PatchDiscriminator {
    pub fn new(p: nn::Path, input_c: usize, num_filters: usize, n_down: usize) -> Self {
        let mut seq = nn::seq_t().add(Self::get_layers(
            &p / "initial",
            input_c,
            num_filters,
            4,
            2,
            1,
            false,
            true,
        ));
        for i in 0..n_down {
            seq = seq.add(Self::get_layers(
                &p / &format!("{}", i),
                num_filters * 2_usize.pow(i as u32),
                num_filters * 2_usize.pow(i as u32 + 1),
                4,
                if i == (n_down - 1) { 1 } else { 2 },
                1,
                true,
                true,
            ));
        }
        seq = seq.add(Self::get_layers(
            &p / "final",
            num_filters * 2_usize.pow(n_down as u32),
            1,
            4,
            1,
            1,
            false,
            false,
        ));
        Self { seq }
    }

    pub fn get_layers(
        p: nn::Path,
        ni: usize,
        nf: usize,
        k: usize,
        stride: usize,
        pad: usize,
        norm: bool,
        act: bool,
    ) -> nn::SequentialT {
        let mut seq = nn::seq_t().add(nn::conv2d(
            &p / "input",
            ni as _,
            nf as _,
            k as _,
            ConvConfig {
                stride: stride as _,
                padding: pad as _,
                bias: !norm,
                ws_init: Init::Randn {
                    mean: 0.0,
                    stdev: 0.02,
                },
                bs_init: Init::Const(0.0),
                ..Default::default()
            },
        ));
        if norm {
            seq = seq.add(nn::batch_norm2d(
                &p / "norm",
                nf as _,
                BatchNormConfig {
                    ws_init: Init::Randn {
                        mean: 1.0,
                        stdev: 0.02,
                    },
                    bs_init: Init::Const(0.0),
                    ..Default::default()
                },
            ))
        }
        if act {
            seq = seq.add_fn(|t| t.maximum(&(t * 0.2))); // leaky relu
        }
        seq
    }

    pub fn forward(&self, xs: &Tensor, train: bool) -> Tensor {
        self.seq.forward_t(xs, train)
    }
}

impl ModuleT for PatchDiscriminator {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        self.forward(xs, train)
    }
}

pub struct GANLoss {
    real_label: Tensor,
    fake_label: Tensor,
}

impl GANLoss {
    pub fn new(real: f32, fake: f32, device: Device) -> Self {
        Self {
            real_label: Tensor::from_slice(&[real]).to_device(device),
            fake_label: Tensor::from_slice(&[fake]).to_device(device),
        }
    }

    pub fn get_labels(&self, preds: &Tensor, is_real: bool) -> Tensor {
        if is_real {
            self.real_label.expand_as(preds)
        } else {
            self.fake_label.expand_as(preds)
        }
    }

    pub fn forward(&self, preds: &Tensor, is_real: bool) -> Tensor {
        let labels = self.get_labels(preds, is_real);
        let loss = preds.binary_cross_entropy_with_logits::<Tensor>(
            &labels,
            None,
            None,
            tch::Reduction::Mean,
        );
        loss
    }
}

pub fn cnn_block(
    vs: nn::Path,
    in_chan: i64,
    out_chan: i64,
    down: bool,
    leaky: bool,
    dropout: bool,
) -> impl ModuleT {
    let seq = match down {
        true => nn::seq_t().add(unet_conv(&vs / "conv2d", in_chan, out_chan)),
        false => nn::seq_t().add(unet_conv_transpose(
            &vs / "convtranspose2d",
            in_chan,
            out_chan,
        )),
    }
    .add(nn::batch_norm2d(
        &vs / "batchnorm",
        out_chan,
        Default::default(),
    ))
    .add_fn(match leaky {
        true => |t: &Tensor| t.maximum(&(t * 0.2)),
        false => |t: &Tensor| t.relu(),
    });
    match dropout {
        true => seq.add_fn_t(|t, train| t.dropout(0.5, train)),
        false => seq,
    }
}

pub fn leaky_relu(slope: f64) -> impl ModuleT {
    nn::func_t(move |xs, _| xs.maximum(&(xs * slope)))
}

pub fn generator(vs: nn::Path, in_chan: i64, features: i64, out_chan: i64) -> impl ModuleT {
    let initial_down = nn::seq_t()
        .add(conv2d(
            &vs / "initial_conv",
            in_chan,
            features,
            4,
            ConvConfig {
                stride: 2,
                padding: 1,
                padding_mode: nn::PaddingMode::Reflect,
                ..Default::default()
            },
        ))
        .add(leaky_relu(0.2));
    let down1 = cnn_block(&vs / "down1", features, features * 2, true, true, false);
    let down2 = cnn_block(&vs / "down2", features * 2, features * 4, true, true, false);
    let down3 = cnn_block(&vs / "down3", features * 4, features * 8, true, true, false);
    let down4 = cnn_block(&vs / "down4", features * 8, features * 8, true, true, false);
    let down5 = cnn_block(&vs / "down5", features * 8, features * 8, true, true, false);
    let down6 = cnn_block(&vs / "down6", features * 8, features * 8, true, true, false);
    let bottleneck = nn::seq_t()
        .add(nn::conv2d(
            &vs / "bottleneck",
            features * 8,
            features * 8,
            4,
            ConvConfig {
                stride: 2,
                padding: 1,
                ..Default::default()
            },
        ))
        .add_fn(|t| t.relu());
    let up1 = cnn_block(&vs / "up1", features * 8, features * 8, false, false, true);
    let up2 = cnn_block(
        &vs / "up2",
        features * 8 * 2,
        features * 8,
        false,
        false,
        true,
    );
    let up3 = cnn_block(
        &vs / "up3",
        features * 8 * 2,
        features * 8,
        false,
        false,
        true,
    );
    let up4 = cnn_block(
        &vs / "up4",
        features * 8 * 2,
        features * 8,
        false,
        false,
        false,
    );
    let up5 = cnn_block(
        &vs / "up5",
        features * 8 * 2,
        features * 4,
        false,
        false,
        false,
    );
    let up6 = cnn_block(
        &vs / "up6",
        features * 4 * 2,
        features * 2,
        false,
        false,
        false,
    );
    let up7 = cnn_block(&vs / "up7", features * 2 * 2, features, false, false, false);
    let final_up = nn::seq_t()
        .add(nn::conv_transpose2d(
            &vs / "final_up",
            features * 2,
            out_chan,
            4,
            ConvTransposeConfig {
                stride: 2,
                padding: 1,
                ..Default::default()
            },
        ))
        .add_fn(|t| t.tanh());
    nn::func_t(move |xs, train| {
        let d1 = initial_down.forward_t(&xs, train);
        let d2 = down1.forward_t(&d1, train);
        let d3 = down2.forward_t(&d2, train);
        let d4 = down3.forward_t(&d3, train);
        let d5 = down4.forward_t(&d4, train);
        let d6 = down5.forward_t(&d5, train);
        let d7 = down6.forward_t(&d6, train);
        let bottleneck = bottleneck.forward_t(&d7, train);
        let up1 = up1.forward_t(&bottleneck, train);
        let up2 = up2.forward_t(&Tensor::cat(&[up1, d7], 1), train);
        let up3 = up3.forward_t(&Tensor::cat(&[up2, d6], 1), train);
        let up4 = up4.forward_t(&Tensor::cat(&[up3, d5], 1), train);
        let up5 = up5.forward_t(&Tensor::cat(&[up4, d4], 1), train);
        let up6 = up6.forward_t(&Tensor::cat(&[up5, d3], 1), train);
        let up7 = up7.forward_t(&Tensor::cat(&[up6, d2], 1), train);
        final_up.forward_t(&Tensor::cat(&[up7, d1], 1), train)
    })
}

pub fn unet_conv(vs: nn::Path, in_chan: i64, out_chan: i64) -> nn::Conv2D {
    nn::conv2d(
        vs,
        in_chan,
        out_chan,
        4,
        ConvConfig {
            stride: 2,
            padding: 1,
            bias: false,
            padding_mode: nn::PaddingMode::Reflect,
            ..Default::default()
        },
    )
}

pub fn unet_conv_transpose(vs: nn::Path, in_chan: i64, out_chan: i64) -> nn::ConvTranspose2D {
    nn::conv_transpose2d(
        vs,
        in_chan,
        out_chan,
        4,
        ConvTransposeConfig {
            stride: 2,
            padding: 1,
            bias: false,
            ..Default::default()
        },
    )
}
