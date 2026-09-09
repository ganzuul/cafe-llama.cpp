// Router-input trace for MoE models (expert-prefetch study). See expert-trace.h.
//
// NOTE: routed experts / router probs are NOT captured here — the fork's fused
// router kernels remove the logits, softmax, argsort and topk nodes from real
// decode graphs (only an auto-named "(reshaped)" view survives, with a
// misleading shape). Router inputs DO fire reliably for every layer, and the
// routing is recovered offline: top-k = argsort(W_gate_inp @ x), which is the
// exact computation the model performs.

#include "expert-trace.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdlib>
#include <cstring>
#include <vector>

static constexpr uint32_t EXPERT_TRACE_MAGIC   = 0x52525445; // 'ETRR'
static constexpr uint32_t EXPERT_TRACE_VERSION = 1;

namespace {

struct layer_buf {
    bool have_input = false;
    int  n_tok      = 0;
    int  n_embd     = 0;
    std::vector<float> input;
};

} // namespace

struct common_expert_trace_user_data::impl {
    FILE * file = nullptr;
    int    n_embd = -1;    // locked from first router-input tensor

    // generation state (one ubatch)
    std::vector<layer_buf> layers;
    int  n_tok_gen = 0;
    bool gen_active = false;

    explicit impl(const std::string & path);
    ~impl();

    void reset_gen();
    void flush_gen();   // write record if the generation is complete, else discard
};

common_expert_trace_user_data::impl::impl(const std::string & path) {
    file = fopen(path.c_str(), "wb");
    if (!file) {
        fprintf(stderr, "%s: failed to open trace file '%s'\n", __func__, path.c_str());
        return;
    }
    const uint32_t hdr[2] = { EXPERT_TRACE_MAGIC, EXPERT_TRACE_VERSION };
    fwrite(hdr, sizeof(hdr), 1, file);
    setvbuf(file, nullptr, _IOFBF, 16 * 1024 * 1024);
}

common_expert_trace_user_data::impl::~impl() {
    if (file) {
        fclose(file);
    }
}

void common_expert_trace_user_data::impl::reset_gen() {
    layers.clear();
    n_tok_gen   = 0;
    gen_active  = false;
}

void common_expert_trace_user_data::impl::flush_gen() {
    if (!gen_active || layers.empty()) {
        reset_gen();
        return;
    }
    for (const auto & lb : layers) {
        if (!lb.have_input) {
            reset_gen();
            return;
        }
    }
    const uint32_t rec[3] = { (uint32_t) n_tok_gen, (uint32_t) layers.size(),
                              (uint32_t) n_embd };
    fwrite(rec, sizeof(rec), 1, file);
    for (const auto & lb : layers) {
        fwrite(lb.input.data(), sizeof(float), lb.input.size(), file);
    }
    reset_gen();
}

bool common_expert_trace_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * tr = (common_expert_trace_user_data *) user_data;
    if (!tr || !tr->pimpl || !tr->pimpl->file) {
        return false;
    }

    // exact prefix "ffn_moe_router_input-" (21 chars); rejects the auto-named
    // "(reshaped)"/"(view)" variants since the suffix follows the layer index
    if (strncmp(t->name, "ffn_moe_router_input-", 21) != 0) {
        return false;
    }
    if (ask) {
        return true;
    }

    const int il = atoi(t->name + 21);
    if (il < 0 || il > 1024) {
        return true;
    }

    // pull data to host if needed
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    const size_t n_bytes = ggml_nbytes(t);
    std::vector<uint8_t> tmp;
    const uint8_t * data;
    if (is_host) {
        data = (const uint8_t *) t->data;
    } else {
        tmp.resize(n_bytes);
        ggml_backend_tensor_get(t, tmp.data(), 0, n_bytes);
        data = tmp.data();
    }

    GGML_ASSERT(t->type == GGML_TYPE_F32);
    const int n_tok = (int) t->ne[1];

    auto & P = *tr->pimpl;
    if (P.n_embd < 0) {
        P.n_embd = (int) t->ne[0];
    }

    // ubatch boundary: a layer-0 input arriving while the gen already has its
    // layer-0 slot filled (first arrival starts the gen, second fills it, any
    // later one is the next ubatch). flush the previous gen at that point
    // (deferred by one ubatch).
    if (il == 0 && P.gen_active) {
        const auto & lb0 = P.layers.empty() ? P.layers.emplace_back() : P.layers[0];
        if (lb0.have_input || n_tok != P.n_tok_gen) {
            P.flush_gen();
        }
    }
    if (il != 0 && !P.gen_active) {
        return true; // orphan tensor before any gen start: drop
    }
    if (!P.gen_active) {
        P.gen_active = true;
        P.n_tok_gen  = n_tok;
    }
    if (n_tok != P.n_tok_gen) {
        return true; // inconsistent; skip
    }

    if (il >= (int) P.layers.size()) {
        P.layers.resize(il + 1);
    }
    auto & lb = P.layers[il];
    if (lb.have_input) {
        return true; // duplicate fire (graph split): ignore
    }
    lb.n_embd = (int) t->ne[0];
    lb.input.resize((size_t) n_tok * lb.n_embd);
    memcpy(lb.input.data(), data, lb.input.size() * sizeof(float));
    lb.have_input = true;
    lb.n_tok = n_tok;

    return true;
}

common_expert_trace_user_data::common_expert_trace_user_data(const std::string & path)
    : pimpl(std::make_unique<impl>(path)) {}

common_expert_trace_user_data::~common_expert_trace_user_data() = default;
