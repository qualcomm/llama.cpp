#pragma once

#include "llama.h"

#include <memory>

struct llama_model;
struct llama_model_loader;
struct llama_expert_cache;

void llama_expert_cache_prepare(llama_model_loader & loader, llama_model_params & params);

std::unique_ptr<llama_expert_cache> llama_expert_cache_init(llama_model &              model,
                                                            llama_model_loader &       loader,
                                                            const llama_model_params & params);
