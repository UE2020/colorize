use tch::nn::{BatchNormConfig, ConvConfig, Init, ModuleT};
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
