use crate::app::SeedColor;
use crate::app::calculate;
use crate::app::calculate::SWAPS_PER_GENERATION_PER_PIXEL;
use crate::app::preset::UnprocessedPreset;

use std::error::Error;

use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::sync::mpsc;

use super::ProgressMsg;

use super::GenerationSettings;

#[derive(Clone, Copy)]
pub struct PixelData {
    pub stroke_id: u32,
    pub last_edited: u32,
}
impl PixelData {
    pub(crate) fn init_canvas(frame_count: u32) -> Vec<PixelData> {
        vec![
            PixelData {
                stroke_id: 0,
                last_edited: frame_count
            };
            DRAWING_CANVAS_SIZE * DRAWING_CANVAS_SIZE
        ]
    }
}

pub const DRAWING_CANVAS_SIZE: usize = 128;

use super::heuristic;

#[derive(Clone, Copy)]
pub(crate) struct DrawingPixel {
    pub(crate) src_x: u16,
    pub(crate) src_y: u16,
    pub(crate) h: i64, // current heuristic value
}

impl DrawingPixel {
    pub(crate) fn new(src_x: u16, src_y: u16, h: i64) -> Self {
        Self { src_x, src_y, h }
    }

    pub(crate) fn update_heuristic(&mut self, new_h: i64) {
        self.h = new_h;
    }

    #[inline(always)]
    pub(crate) fn calc_drawing_heuristic(
        &self,
        target_pos: (u16, u16),
        target_col: (u8, u8, u8),
        weight: i64,
        colors: &[SeedColor],
        proximity_importance: i64,
    ) -> i64 {
        heuristic(
            (self.src_x, self.src_y),
            target_pos,
            {
                let rgba =
                    colors[self.src_y as usize * DRAWING_CANVAS_SIZE + self.src_x as usize].rgba;
                (
                    (rgba[0] * 256.0) as u8,
                    (rgba[1] * 256.0) as u8,
                    (rgba[2] * 256.0) as u8,
                )
            },
            target_col,
            weight,
            proximity_importance,
        )
    }
}

pub(crate) const STROKE_REWARD: i64 = -10000000000;

pub(crate) fn stroke_reward(
    newpos: usize,
    oldpos: usize,
    pixel_data: &[PixelData],
    pixels: &[DrawingPixel],
    frame_count: u32,
) -> i64 {
    let x = (newpos % DRAWING_CANVAS_SIZE) as u16;
    let y = (newpos / DRAWING_CANVAS_SIZE) as u16;
    // look at 8-connected neighbors
    // if any has the same stroke_id, return true
    let data = pixel_data
        [pixels[oldpos].src_x as usize + pixels[oldpos].src_y as usize * DRAWING_CANVAS_SIZE];
    let stroke_id = data.stroke_id;
    let _age = frame_count - data.last_edited;

    for (dx, dy) in [
        //(-1, -1),
        (0, -1),
        //(1, -1),
        (-1, 0),
        (1, 0),
        //(-1, 1),
        (0, 1),
        //(1, 1),
    ] {
        let nx = x as i16 + dx;
        let ny = y as i16 + dy;
        if nx < 0 || nx >= DRAWING_CANVAS_SIZE as i16 || ny < 0 || ny >= DRAWING_CANVAS_SIZE as i16
        {
            continue;
        }
        let npos = ny as usize * DRAWING_CANVAS_SIZE + nx as usize;
        if pixel_data
            [pixels[npos].src_x as usize + pixels[npos].src_y as usize * DRAWING_CANVAS_SIZE]
            .stroke_id
            == stroke_id
        {
            return STROKE_REWARD;
        }
    }
    0
}

#[allow(clippy::too_many_arguments)]
pub fn drawing_process_genetic(
    source: UnprocessedPreset,
    settings: GenerationSettings,
    tx: mpsc::SyncSender<ProgressMsg>,
    colors: Arc<std::sync::RwLock<Vec<SeedColor>>>,
    pixel_data: Arc<std::sync::RwLock<Vec<PixelData>>>,
    frame_count: u32,
    my_id: u32,
    current_id: Arc<AtomicU32>,
) -> Result<(), Box<dyn Error>> {
    let source_img =
        image::ImageBuffer::from_raw(source.width, source.height, source.source_img.clone())
            .unwrap();
    let (source_pixels, target_pixels) = calculate::util::get_images(source_img, &settings)?;

    let mut pixels = {
        let read_colors: Vec<SeedColor> = colors.read().unwrap().clone();
        //let read_pixel_data: Vec<PixelData> = pixel_data.read().unwrap().clone();

        source_pixels
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let x = (i as u32 % settings.sidelen) as u16;
                let y = (i as u32 / settings.sidelen) as u16;
                let mut p = DrawingPixel::new(x, y, 0);
                let h = p.calc_drawing_heuristic(
                    (x, y),
                    target_pixels[i].rgb_tuple(),
                    target_pixels[i].weight,
                    &read_colors,
                    settings.proximity_importance,
                    // &read_pixel_data,
                ) + STROKE_REWARD;
                p.update_heuristic(h);
                p
            })
            .collect::<Vec<_>>()
    };

    let mut rng = frand::Rand::with_seed(12345);
    fn max_dist(age: u32) -> u32 {
        (((DRAWING_CANVAS_SIZE / 4) as f32) * (0.99f32).powi(age as i32 / 30)).round() as u32
    }

    let swaps_per_generation = SWAPS_PER_GENERATION_PER_PIXEL * pixels.len();

    loop {
        let colors: Vec<SeedColor> = {
            let r = colors.read().unwrap();
            r.clone()
        };
        let pixel_data = {
            let r = pixel_data.read().unwrap();
            r.clone()
        };
        let mut swaps_made = 0;

        for _ in 0..swaps_per_generation {
            let apos = rng.gen_range(0..pixels.len() as u64) as usize;
            let ax = apos as u16 % settings.sidelen as u16;
            let ay = apos as u16 / settings.sidelen as u16;

            //let stroke_id = pixel_data[apos].stroke_id as usize;
            let max_dist_a = max_dist(frame_count.saturating_sub(pixel_data[apos].last_edited));

            let bx = (ax as i16 + rng.gen_range(-(max_dist_a as i16)..(max_dist_a as i16 + 1)))
                .clamp(0, settings.sidelen as i16 - 1) as u16;
            let by = (ay as i16 + rng.gen_range(-(max_dist_a as i16)..(max_dist_a as i16 + 1)))
                .clamp(0, settings.sidelen as i16 - 1) as u16;
            let bpos = by as usize * settings.sidelen as usize + bx as usize;

            let max_dist_b = max_dist(frame_count.saturating_sub(pixel_data[bpos].last_edited));
            if (bx as i32 - ax as i32).abs() > max_dist_b as i32
                || (by as i32 - ay as i32).abs() > max_dist_b as i32
            {
                continue;
            }

            let t_a = target_pixels[apos];
            let t_b = target_pixels[bpos];

            let a_on_b_h = pixels[apos].calc_drawing_heuristic(
                (bx, by),
                t_b.rgb_tuple(),
                t_b.weight,
                &colors,
                settings.proximity_importance,
            ) + stroke_reward(bpos, apos, &pixel_data, &pixels, frame_count);

            let b_on_a_h = pixels[bpos].calc_drawing_heuristic(
                (ax, ay),
                t_a.rgb_tuple(),
                t_a.weight,
                &colors,
                settings.proximity_importance,
            ) + stroke_reward(apos, bpos, &pixel_data, &pixels, frame_count);

            let improvement_a = pixels[apos].h - b_on_a_h;
            let improvement_b = pixels[bpos].h - a_on_b_h;
            if improvement_a + improvement_b > 0 {
                // swap
                pixels.swap(apos, bpos);
                pixels[apos].update_heuristic(b_on_a_h);
                pixels[bpos].update_heuristic(a_on_b_h);
                swaps_made += 1;
            }
        }

        //println!("swaps made: {}", swaps_made);

        // let img = make_new_img(&source_pixels, &assignments, target.width());
        // if swaps_made < 10 || cancelled.load(std::sync::atomic::Ordering::Relaxed) {
        //     let dir_name = save_result(target, base_name, source, assignments, img)?;
        //     tx.send(ProgressMsg::Done(PathBuf::from(format!(
        //         "./presets/{}",
        //         dir_name
        //     ))))?;
        //     return Ok(());
        // }
        // tx.send(ProgressMsg::UpdatePreview(img))?;
        if swaps_made > 0 {
            let assignments = pixels
                .iter()
                .map(|p| p.src_y as usize * settings.sidelen as usize + p.src_x as usize)
                .collect::<Vec<_>>();
            tx.send(ProgressMsg::UpdateAssignments(assignments))?;
        }
        if my_id != current_id.load(std::sync::atomic::Ordering::Relaxed) {
            tx.send(ProgressMsg::Cancelled).unwrap();
            return Ok(());
        }

        //max_dist = (max_dist as f32 * 0.99).max(4.0) as u32;
    }
}
