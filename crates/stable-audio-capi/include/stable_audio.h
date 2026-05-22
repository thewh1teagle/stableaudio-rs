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

void stable_audio_model_free(StableAudioModel *model);

int stable_audio_generate_wav(StableAudioModel *model,
                              const char *prompt,
                              float seconds,
                              size_t steps,
                              uint64_t seed,
                              const char *output_path);

#endif  /* STABLE_AUDIO_H */
