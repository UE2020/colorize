use tch::nn::{conv_transpose2d, ConvConfig, ConvTransposeConfig, ModuleT, Init, BatchNormConfig};
use tch::{nn, Tensor, Device};

#[derive(Debug)]
pub struct UnetBlock {
    seq: nn::SequentialT,
    outermost: bool,
}

impl UnetBlock {
    pub fn new(
        p: nn::Path,
        nf: usize,
        ni: usize,
        submodule: Option<UnetBlock>,
        input_c: Option<usize>,
        dropout: bool,
        innermost: bool,
        outermost: bool,
    ) -> Self {
        let input_c = input_c.unwrap_or(nf);
        let mut seq = nn::seq_t();
        let downconv = nn::conv2d(
            &p / "downconv",
            input_c as _,
            ni as _,
            4,
            ConvConfig {
                stride: 2,
                padding: 1,
                bias: false,
                ws_init: Init::Randn { mean: 0.0, stdev: 0.02 },
                bs_init: Init::Const(0.0),
                ..Default::default()
            },
        );
        let downnorm = nn::batch_norm2d(&p / "downnorm", ni as _, BatchNormConfig {
            ws_init: Init::Randn { mean: 1.0, stdev: 0.02 },
            bs_init: Init::Const(0.0),
            ..Default::default()
        });
        let upnorm = nn::batch_norm2d(&p / "upnorm", nf as _, BatchNormConfig {
            ws_init: Init::Randn { mean: 1.0, stdev: 0.02 },
            bs_init: Init::Const(0.0),
            ..Default::default()
        });
        if outermost {
            let upconv = conv_transpose2d(
                &p / "upconv",
                ni as i64 * 2,
                nf as _,
                4,
                ConvTransposeConfig {
                    stride: 2,
                    padding: 1,
                    ws_init: Init::Randn { mean: 0.0, stdev: 0.02 },
                    bs_init: Init::Const(0.0),
                    ..Default::default()
                },
            );
            seq = seq.add(downconv);
            if let Some(submodule) = submodule {
                seq = seq.add(submodule);
            }
            seq = seq.add_fn(|t| t.relu());
            seq = seq.add(upconv);
            seq = seq.add_fn(|t| t.tanh());
        } else if innermost {
            let upconv = conv_transpose2d(
                &p / "upconv",
                ni as _,
                nf as _,
                4,
                ConvTransposeConfig {
                    stride: 2,
                    padding: 1,
                    bias: false,
                    ws_init: Init::Randn { mean: 0.0, stdev: 0.02 },
                    bs_init: Init::Const(0.0),
                    ..Default::default()
                },
            );
            seq = seq.add_fn(|t| t.maximum(&(t * 0.2))); // leaky relu
            seq = seq.add(downconv);
            seq = seq.add_fn(|t| t.relu());
            seq = seq.add(upconv);
            seq = seq.add(upnorm);
        } else {
            let upconv = conv_transpose2d(
                &p / "upconv",
                ni as i64 * 2,
                nf as _,
                4,
                ConvTransposeConfig {
                    stride: 2,
                    padding: 1,
                    bias: false,
                    ws_init: Init::Randn { mean: 0.0, stdev: 0.02 },
                    bs_init: Init::Const(0.0),
                    ..Default::default()
                },
            );
            seq = seq.add_fn(|t| t.maximum(&(t * 0.2))); // leaky relu
            seq = seq.add(downconv);
            seq = seq.add(downnorm);
            if let Some(submodule) = submodule {
                seq = seq.add(submodule);
            }
            seq = seq.add_fn(|t| t.relu());
            seq = seq.add(upconv);
            seq = seq.add(upnorm);
            if dropout {
                seq = seq.add_fn_t(|t, train| t.dropout(0.5, train));
            }
        }

        Self { seq, outermost }
    }

    pub fn forward(&self, xs: &Tensor, train: bool) -> Tensor {
        if self.outermost {
            self.seq.forward_t(xs, train)
        } else {
            Tensor::cat(&[xs.shallow_clone(), self.seq.forward_t(xs, train)], 1)
        }
    }
}

impl ModuleT for UnetBlock {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        self.forward(xs, train)
    }
}

pub fn unet(
    p: nn::Path,
    input_c: usize,
    output_c: usize,
    n_down: usize,
    num_filters: usize,
) -> impl ModuleT {
    let mut block = UnetBlock::new(
        &p / "initial_block",
        num_filters * 8,
        num_filters * 8,
        None,
        None,
        false,
        true,
        false,
    );
    for i in 0..(n_down - 5) {
        block = UnetBlock::new(
            &p / &format!("block1-{}", i),
            num_filters * 8,
            num_filters * 8,
            Some(block),
            None,
            true,
            false,
            false,
        );
    }
    let mut out_filters = num_filters * 8;
    for i in 0..3 {
        block = UnetBlock::new(
            &p / &format!("block2-{}", i),
            out_filters / 2,
            out_filters,
            Some(block),
            None,
            false,
            false,
            false,
        );
        out_filters /= 2;
    }
    block = UnetBlock::new(
        &p / "final_block",
        output_c,
        out_filters,
        Some(block),
        Some(input_c),
        false,
        false,
        true,
    );
    nn::func_t(move |xs, train| block.forward_t(xs, train))
}

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
        seq = seq.add(Self::get_layers(&p / "final", num_filters * 2_usize.pow(n_down as u32), 1, 4, 1, 1, false, false));
        Self {
            seq
        }
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
                ws_init: Init::Randn { mean: 0.0, stdev: 0.02 },
                bs_init: Init::Const(0.0),
                ..Default::default()
            },
        ));
        if norm {
            seq = seq.add(nn::batch_norm2d(&p / "norm", nf as _, BatchNormConfig {
                ws_init: Init::Randn { mean: 1.0, stdev: 0.02 },
                bs_init: Init::Const(0.0),
                ..Default::default()
            }))
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
        let loss = preds.binary_cross_entropy_with_logits::<Tensor>(&labels, None, None, tch::Reduction::Mean);
        loss
    }
}
