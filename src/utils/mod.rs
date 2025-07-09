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
