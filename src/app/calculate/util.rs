use crate::app::calculate::ProgressMsg;

use image::imageops;
use serde::Deserialize;
use serde::Serialize;
use uuid::Uuid;

use std::error::Error;

// pub(crate) fn save_result(
//     target: image::SourceImg,
//     base_name: String,
//     source: image::SourceImg,
//     assignments: Vec<usize>,
//     img: image::SourceImg,
// ) -> Result<String, Box<dyn Error>> {
//     let mut dir_name = base_name.clone();
//     let mut counter = 1;
//     while std::path::Path::new(&format!("./presets/{}", dir_name)).exists() {
//         dir_name = format!("{}_{}", base_name, counter);
//         counter += 1;
//     }
//     std::fs::create_dir_all(format!("./presets/{}", dir_name))?;
//     img.save(format!("./presets/{}/output.png", dir_name))?;
//     source.save(format!("./presets/{}/source.png", dir_name))?;
//     target.save(format!("./presets/{}/target.png", dir_name))?;
//     std::fs::write(
//         format!("./presets/{}/assignments.json", dir_name),
//         serialize_assignments(assignments),
//     )?;
//     Ok(dir_name)
// }

pub trait ProgressSink {
    fn send(&mut self, msg: ProgressMsg);
}
// Native-friendly adapter
impl ProgressSink for std::sync::mpsc::SyncSender<ProgressMsg> {
    fn send(&mut self, msg: ProgressMsg) {
        let _ = std::sync::mpsc::SyncSender::send(self, msg);
    }
}

// Allow using closures as progress sinks in WASM
impl<T> ProgressSink for T
where
    T: FnMut(crate::app::calculate::ProgressMsg),
{
    fn send(&mut self, msg: crate::app::calculate::ProgressMsg) {
        self(msg);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GridPixel {
    pub x: u16,
    pub y: u16,
    pub rgb: [u8; 3],
}

impl GridPixel {
    #[inline]
    pub fn new(x: u32, y: u32, rgb: [u8; 3]) -> Self {
        Self {
            x: x as u16,
            y: y as u16,
            rgb,
        }
    }

    #[inline]
    pub fn coords(&self) -> (u16, u16) {
        (self.x, self.y)
    }

    #[inline]
    pub fn rgb_tuple(&self) -> (u8, u8, u8) {
        (self.rgb[0], self.rgb[1], self.rgb[2])
    }

    #[inline]
    pub fn linear_index(&self, sidelen: u32) -> usize {
        self.y as usize * sidelen as usize + self.x as usize
    }
}

#[derive(Clone, Copy, Debug)]
pub struct WeightedPixel {
    pub pixel: GridPixel,
    pub weight: i64,
}

impl WeightedPixel {
    #[inline]
    pub fn coords(&self) -> (u16, u16) {
        self.pixel.coords()
    }

    #[inline]
    pub fn rgb_tuple(&self) -> (u8, u8, u8) {
        self.pixel.rgb_tuple()
    }
}

pub(crate) fn get_images(
    source: SourceImg,
    settings: &GenerationSettings,
) -> Result<(Vec<GridPixel>, Vec<WeightedPixel>), Box<dyn Error>> {
    let source = settings.source_crop_scale.apply(&source, settings.sidelen);
    let (target, weights) = settings.get_target()?;

    let source_pixels = source
        .enumerate_pixels()
        .map(|(x, y, pixel)| GridPixel::new(x, y, pixel.0))
        .collect::<Vec<_>>();

    let target_pixels = target
        .enumerate_pixels()
        .zip(weights.into_iter())
        .map(|((x, y, pixel), weight)| WeightedPixel {
            pixel: GridPixel::new(x, y, pixel.0),
            weight,
        })
        .collect::<Vec<_>>();

    assert_eq!(source_pixels.len(), target_pixels.len());
    Ok((source_pixels, target_pixels))
}

#[derive(Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct CropScale {
    pub x: f32,     // -1: all left, 0: center, 1: all right
    pub y: f32,     // -1: all top, 0: center, 1: all bottom
    pub scale: f32, // 1: fit within frame, >1: zoom in, <1: not allowed
}

impl CropScale {
    pub fn identity() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            scale: 1.0,
        }
    }

    pub fn apply(&self, img: &SourceImg, sidelen: u32) -> SourceImg {
        let (w, h) = img.dimensions();

        let s = self.scale.max(1.0);

        let base_side = w.min(h) as f32;
        let mut crop_side = (base_side / s).floor().max(1.0);

        crop_side = crop_side.min(w as f32).min(h as f32);

        let max_x_off = (w as f32 - crop_side).max(0.0);
        let max_y_off = (h as f32 - crop_side).max(0.0);

        let xn = (self.x.clamp(-1.0, 1.0) + 1.0) * 0.5;
        let yn = (self.y.clamp(-1.0, 1.0) + 1.0) * 0.5;

        let x0 = (xn * max_x_off).floor() as u32;
        let y0 = (yn * max_y_off).floor() as u32;
        let cs = crop_side as u32;
        let cropped = imageops::crop_imm(img, x0, y0, cs, cs).to_image();

        if cs == sidelen {
            cropped
        } else {
            imageops::resize(&cropped, sidelen, sidelen, imageops::FilterType::Lanczos3)
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum Algorithm {
    Optimal,
    Genetic,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct GenerationSettings {
    pub id: Uuid,
    pub name: String,

    pub proximity_importance: i64,
    pub algorithm: Algorithm,

    pub sidelen: u32,
    custom_target: Option<(u32, u32, Vec<u8>)>,
    pub target_crop_scale: CropScale,
    pub source_crop_scale: CropScale,
}

pub type SourceImg = image::RgbImage;

impl GenerationSettings {
    pub fn default(id: Uuid, name: String) -> Self {
        Self {
            name,
            proximity_importance: 13, // 20
            algorithm: Algorithm::Genetic,
            id,
            sidelen: 128,
            custom_target: None,
            target_crop_scale: CropScale::identity(),
            source_crop_scale: CropScale::identity(),
        }
    }

    pub fn get_target(&self) -> Result<(SourceImg, Vec<i64>), Box<dyn std::error::Error>> {
        let target = self.get_raw_target();
        let target = self.target_crop_scale.apply(&target, self.sidelen);
        let weights = if self.custom_target.is_some() {
            vec![255; (self.sidelen * self.sidelen) as usize] // uniform weights
        } else {
            let target_weights =
                image::load_from_memory(include_bytes!("weights256.png"))?.to_rgb8();
            let target_weights = self.target_crop_scale.apply(&target_weights, self.sidelen);
            load_weights(target_weights)
        };

        Ok((target, weights))
    }

    pub(crate) fn get_raw_target(&self) -> SourceImg {
        if let Some((w, h, data)) = &self.custom_target {
            image::ImageBuffer::from_vec(*w, *h, data.clone()).unwrap()
        } else {
            image::load_from_memory(include_bytes!("target256.png"))
                .unwrap()
                .to_rgb8()
        }
    }

    pub(crate) fn set_raw_target(&mut self, img: SourceImg) {
        let (w, h) = img.dimensions();
        let data = img.into_raw();
        self.custom_target = Some((w, h, data));
    }

    pub fn clone_with_new_id(&self) -> Self {
        let mut new = self.clone();
        new.id = Uuid::new_v4();

        new.name = if let Some(v_pos) = self.name.rfind(" v") {
            let potential_version = &self.name[v_pos + 2..];
            if let Ok(version) = potential_version.parse::<u32>() {
                let base_name = &self.name[..v_pos];
                format!("{} v{}", base_name, version + 1)
            } else {
                format!("{} v2", self.name)
            }
        } else {
            format!("{} v2", self.name)
        };

        new
    }
}

pub fn load_weights(source: SourceImg) -> Vec<i64> {
    let (width, height) = source.dimensions();
    let mut weights = vec![0; (width * height) as usize];
    for (x, y, pixel) in source.enumerate_pixels() {
        let weight = pixel[0] as i64;
        weights[(y * width + x) as usize] = weight;
    }
    weights
}
