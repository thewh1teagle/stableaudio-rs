use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;

use stable_audio::{
    AudioContinuationRequest, AudioEditRequest, AudioInpaintRequest, GenerateRequest, StableAudio,
    StableAudioConfig,
};

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

pub struct StableAudioModel {
    inner: StableAudio,
}

fn set_error(message: impl ToString) -> c_int {
    let message = message.to_string().replace('\0', "\\0");
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = CString::new(message).ok();
    });
    -1
}

fn clear_error() {
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = None;
    });
}

unsafe fn cstr_arg<'a>(ptr: *const c_char, name: &str) -> Result<&'a str, String> {
    if ptr.is_null() {
        return Err(format!("{name} is null"));
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .map_err(|err| format!("{name} is not valid UTF-8: {err}"))
}

#[unsafe(no_mangle)]
pub extern "C" fn stable_audio_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|err| err.as_ptr())
            .unwrap_or(ptr::null())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn stable_audio_model_load(
    dit_path: *const c_char,
    decoder_path: *const c_char,
    text_encoder_path: *const c_char,
    steps: usize,
    seed: u64,
) -> *mut StableAudioModel {
    clear_error();
    let result = catch_unwind(AssertUnwindSafe(|| {
        let dit_path = unsafe { cstr_arg(dit_path, "dit_path") }?;
        let decoder_path = unsafe { cstr_arg(decoder_path, "decoder_path") }?;
        let text_encoder_path = unsafe { cstr_arg(text_encoder_path, "text_encoder_path") }?;
        let config = StableAudioConfig::new(dit_path, decoder_path, text_encoder_path)
            .steps(steps)
            .seed(seed);
        StableAudio::load(config)
            .map(|inner| Box::into_raw(Box::new(StableAudioModel { inner })))
            .map_err(|err| err.to_string())
    }));
    match result {
        Ok(Ok(model)) => model,
        Ok(Err(err)) => {
            set_error(err);
            ptr::null_mut()
        }
        Err(_) => {
            set_error("panic while loading model");
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn stable_audio_model_load_with_encoder(
    dit_path: *const c_char,
    encoder_path: *const c_char,
    decoder_path: *const c_char,
    text_encoder_path: *const c_char,
    steps: usize,
    seed: u64,
) -> *mut StableAudioModel {
    clear_error();
    let result = catch_unwind(AssertUnwindSafe(|| {
        let dit_path = unsafe { cstr_arg(dit_path, "dit_path") }?;
        let encoder_path = unsafe { cstr_arg(encoder_path, "encoder_path") }?;
        let decoder_path = unsafe { cstr_arg(decoder_path, "decoder_path") }?;
        let text_encoder_path = unsafe { cstr_arg(text_encoder_path, "text_encoder_path") }?;
        let config = StableAudioConfig::new(dit_path, decoder_path, text_encoder_path)
            .encoder_path(encoder_path)
            .steps(steps)
            .seed(seed);
        StableAudio::load(config)
            .map(|inner| Box::into_raw(Box::new(StableAudioModel { inner })))
            .map_err(|err| err.to_string())
    }));
    match result {
        Ok(Ok(model)) => model,
        Ok(Err(err)) => {
            set_error(err);
            ptr::null_mut()
        }
        Err(_) => {
            set_error("panic while loading model with encoder");
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn stable_audio_model_free(model: *mut StableAudioModel) {
    if !model.is_null() {
        drop(unsafe { Box::from_raw(model) });
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn stable_audio_generate_wav(
    model: *mut StableAudioModel,
    prompt: *const c_char,
    seconds: f32,
    steps: usize,
    seed: u64,
    output_path: *const c_char,
) -> c_int {
    clear_error();
    let result = catch_unwind(AssertUnwindSafe(|| {
        if model.is_null() {
            return Err("model is null".to_string());
        }
        let prompt = unsafe { cstr_arg(prompt, "prompt") }?;
        let output_path = unsafe { cstr_arg(output_path, "output_path") }?;
        let audio = unsafe { &mut *model }.inner.generate(
            GenerateRequest::new(prompt)
                .seconds(seconds)
                .steps(steps)
                .seed(seed),
        );
        audio
            .and_then(|audio| audio.write_wav(output_path))
            .map_err(|err| err.to_string())
    }));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(err)) => set_error(err),
        Err(_) => set_error("panic while generating audio"),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn stable_audio_edit_wav(
    model: *mut StableAudioModel,
    prompt: *const c_char,
    input_path: *const c_char,
    seconds: f32,
    steps: usize,
    seed: u64,
    init_noise_level: f32,
    output_path: *const c_char,
) -> c_int {
    clear_error();
    let result = catch_unwind(AssertUnwindSafe(|| {
        if model.is_null() {
            return Err("model is null".to_string());
        }
        let prompt = unsafe { cstr_arg(prompt, "prompt") }?;
        let input_path = unsafe { cstr_arg(input_path, "input_path") }?;
        let output_path = unsafe { cstr_arg(output_path, "output_path") }?;
        let audio = unsafe { &mut *model }.inner.edit_audio(
            AudioEditRequest::new(prompt, input_path)
                .seconds(seconds)
                .steps(steps)
                .seed(seed)
                .init_noise_level(init_noise_level),
        );
        audio
            .and_then(|audio| audio.write_wav(output_path))
            .map_err(|err| err.to_string())
    }));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(err)) => set_error(err),
        Err(_) => set_error("panic while editing audio"),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn stable_audio_inpaint_wav(
    model: *mut StableAudioModel,
    prompt: *const c_char,
    input_path: *const c_char,
    seconds: f32,
    steps: usize,
    seed: u64,
    inpaint_start: f32,
    inpaint_end: f32,
    output_path: *const c_char,
) -> c_int {
    clear_error();
    let result = catch_unwind(AssertUnwindSafe(|| {
        if model.is_null() {
            return Err("model is null".to_string());
        }
        let prompt = unsafe { cstr_arg(prompt, "prompt") }?;
        let input_path = unsafe { cstr_arg(input_path, "input_path") }?;
        let output_path = unsafe { cstr_arg(output_path, "output_path") }?;
        let audio = unsafe { &mut *model }.inner.inpaint_audio(
            AudioInpaintRequest::new(prompt, input_path)
                .seconds(seconds)
                .steps(steps)
                .seed(seed)
                .inpaint_range(inpaint_start, inpaint_end),
        );
        audio
            .and_then(|audio| audio.write_wav(output_path))
            .map_err(|err| err.to_string())
    }));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(err)) => set_error(err),
        Err(_) => set_error("panic while inpainting audio"),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn stable_audio_continue_wav(
    model: *mut StableAudioModel,
    prompt: *const c_char,
    input_path: *const c_char,
    extend_seconds: f32,
    steps: usize,
    seed: u64,
    output_path: *const c_char,
) -> c_int {
    clear_error();
    let result = catch_unwind(AssertUnwindSafe(|| {
        if model.is_null() {
            return Err("model is null".to_string());
        }
        let prompt = unsafe { cstr_arg(prompt, "prompt") }?;
        let input_path = unsafe { cstr_arg(input_path, "input_path") }?;
        let output_path = unsafe { cstr_arg(output_path, "output_path") }?;
        let audio = unsafe { &mut *model }.inner.continue_audio(
            AudioContinuationRequest::new(prompt, input_path)
                .extend_seconds(extend_seconds)
                .steps(steps)
                .seed(seed),
        );
        audio
            .and_then(|audio| audio.write_wav(output_path))
            .map_err(|err| err.to_string())
    }));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(err)) => set_error(err),
        Err(_) => set_error("panic while continuing audio"),
    }
}
