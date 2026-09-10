// Router-input trace for MoE models (expert-prefetch study).
// Installs a ggml_backend_sched eval callback that captures, per ubatch and
// per layer, the router-input hidden state (the tensor feeding the MoE router
// GEMV, named "ffn_moe_router_input-{il}").
//
// Routed experts are recovered offline by the analysis tooling:
// top-k = argsort(W_gate_inp @ x) using the model's own router weights.
//
// Trace file layout (little endian):
//   header:  u32 magic 'ETRR', u32 version = 1
//   record (one per graph ubatch, self-describing):
//     u32 n_tokens, u32 n_layers, u32 n_embd
//     for il in 0..n_layers-1:
//       per token: f32 router_input[n_embd]

#pragma once

#include <cstdio>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_tensor;

struct common_expert_trace_user_data {
    struct impl;

    std::unique_ptr<impl> pimpl;

    explicit common_expert_trace_user_data(const std::string & path);
    ~common_expert_trace_user_data();
};

// to be installed as common_params::cb_eval / cb_eval_user_data
bool common_expert_trace_cb_eval(struct ggml_tensor * t, bool ask, void * user_data);
