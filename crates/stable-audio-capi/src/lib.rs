use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;

use stable_audio::{GenerateRequest, StableAudio, StableAudioConfig};

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
