use crate::model::constants::CHANNELS;

pub fn float_vec_to_image(
    data: &[f32],
    height: usize,
    width: usize,
) -> Option<image::DynamicImage> {
    let expected_len = (width * height * CHANNELS as usize) as usize;
    if data.len() != expected_len {
        eprintln!(
            "Mismatched data length. Expected {}, got {}.",
            expected_len,
            data.len()
        );
        return None;
    }

    let raw_pixels: Vec<u8> = data
        .iter()
        .map(|&val| {
            // Reversing the normalization: (val + 1.0) * 127.5
            let denormalized = (val + 1.0) * 127.5;
            denormalized.clamp(0.0, 255.0) as u8
        })
        .collect();

    let img_buf = image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::from_raw(
        width as u32,
        height as u32,
        raw_pixels,
    )?;

    Some(image::DynamicImage::ImageRgb8(img_buf))
}

/// Converts a channel-first [C, H, W] float slice to an RGB image.
pub fn chw_vec_to_image(data: &[f32], height: usize, width: usize) -> Option<image::DynamicImage> {
    let mut raw_pixels = Vec::with_capacity(height * width * 3);
    for i in 0..(height * width) {
        let r = (data[i] + 1.0) * 127.5;
        let g = (data[i + height * width] + 1.0) * 127.5;
        let b = (data[i + 2 * height * width] + 1.0) * 127.5;
        raw_pixels.push(r.clamp(0.0, 255.0) as u8);
        raw_pixels.push(g.clamp(0.0, 255.0) as u8);
        raw_pixels.push(b.clamp(0.0, 255.0) as u8);
    }
    let img_buf = image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::from_raw(
        width as u32,
        height as u32,
        raw_pixels,
    )?;
    Some(image::DynamicImage::ImageRgb8(img_buf))
}
