/* AUTO-GENERATED WITH CBINDGEN. DO NOT EDIT BY HAND.
   REGENERATE WITH:
   cbindgen crates/stable-audio-capi --crate stable-audio-capi --output crates/stable-audio-capi/include/stable_audio.h
*/


#ifndef STABLE_AUDIO_H
#define STABLE_AUDIO_H

#pragma once

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct StableAudioModel StableAudioModel;

const char *stable_audio_last_error(void);

StableAudioModel *stable_audio_model_load(const char *dit_path,
                                          const char *decoder_path,
                                          const char *text_encoder_path,
                                          size_t steps,
                                          uint64_t seed);

StableAudioModel *stable_audio_model_load_with_encoder(const char *dit_path,
                                                       const char *encoder_path,
                                                       const char *decoder_path,
                                                       const char *text_encoder_path,
                                                       size_t steps,
                                                       uint64_t seed);

void stable_audio_model_free(StableAudioModel *model);

int stable_audio_generate_wav(StableAudioModel *model,
                              const char *prompt,
                              float seconds,
                              size_t steps,
                              uint64_t seed,
                              const char *output_path);

int stable_audio_edit_wav(StableAudioModel *model,
                          const char *prompt,
                          const char *input_path,
                          float seconds,
                          size_t steps,
                          uint64_t seed,
                          float init_noise_level,
                          const char *output_path);

int stable_audio_inpaint_wav(StableAudioModel *model,
                             const char *prompt,
                             const char *input_path,
                             float seconds,
                             size_t steps,
                             uint64_t seed,
                             float inpaint_start,
                             float inpaint_end,
                             const char *output_path);

int stable_audio_continue_wav(StableAudioModel *model,
                              const char *prompt,
                              const char *input_path,
                              float extend_seconds,
                              size_t steps,
                              uint64_t seed,
                              const char *output_path);

#endif  /* STABLE_AUDIO_H */
