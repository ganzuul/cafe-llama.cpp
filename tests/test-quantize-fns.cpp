// Unit tests for quantization specific functions - quantize, dequantize and dot product

#include "ggml.h"
#include "ggml-cpu.h"

#undef NDEBUG
#include <assert.h>
#include <cfloat>
#include <math.h>
#include <stdio.h>
#include <string>
#include <string.h>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

constexpr float MAX_QUANTIZATION_REFERENCE_ERROR = 0.0001f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR = 0.002f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_BINARY = 0.025f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_TERNARY = 0.01f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_2BITS = 0.0075f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS = 0.0040f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS = 0.0050f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_FP4 = 0.0030f;
constexpr float MAX_DOT_PRODUCT_ERROR = 0.02f;
constexpr float MAX_DOT_PRODUCT_ERROR_LOWBIT = 0.04f;
constexpr float MAX_DOT_PRODUCT_ERROR_FP4 = 0.03f;
constexpr float MAX_DOT_PRODUCT_ERROR_BINARY = 0.40f;
constexpr float MAX_DOT_PRODUCT_ERROR_TERNARY = 0.15f;

static const char* RESULT_STR[] = {"ok", "FAILED"};


// Generate synthetic data
static void generate_data(float offset, size_t n, float * dst) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = 0.1 + 2*cosf(i + offset);
    }
}

// Calculate RMSE between two float arrays
static float array_rmse(const float * a1, const float * a2, size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; i++) {
        double diff = a1[i] - a2[i];
        sum += diff * diff;
    }
    return sqrtf(sum) / n;
}

// Total quantization error on test data
static float total_quantization_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data) {
    std::vector<uint8_t> tmp_q(2*test_size);
    std::vector<float> tmp_out(test_size);

    qfns_cpu->from_float(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out.data(), test_size);
    return array_rmse(test_data, tmp_out.data(), test_size);
}

// Total quantization error on test data
static float reference_quantization_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data) {
    std::vector<uint8_t> tmp_q(2*test_size);
    std::vector<float> tmp_out(test_size);
    std::vector<float> tmp_out_ref(test_size);

    // FIXME: why is done twice?
    qfns_cpu->from_float(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out.data(), test_size);

    qfns->from_float_ref(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out_ref.data(), test_size);

    return array_rmse(tmp_out.data(), tmp_out_ref.data(), test_size);
}

static float dot_product(const float * a1, const float * a2, size_t test_size) {
    double sum = 0;
    for (size_t i = 0; i < test_size; i++) {
        sum += a1[i] * a2[i];
    }
    return sum;
}

// Total dot product error
static float dot_product_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data1, const float * test_data2) {
    GGML_UNUSED(qfns);

    std::vector<uint8_t> tmp_q1(2*test_size);
    std::vector<uint8_t> tmp_q2(2*test_size);

    const auto * vdot = ggml_get_type_traits_cpu(qfns_cpu->vec_dot_type);

    qfns_cpu->from_float(test_data1, tmp_q1.data(), test_size);
    vdot->from_float(test_data2, tmp_q2.data(), test_size);

    float result = INFINITY;
    qfns_cpu->vec_dot(test_size, &result, 0, tmp_q1.data(), 0, tmp_q2.data(), 0, 1);

    const float dot_ref = dot_product(test_data1, test_data2, test_size);

    return fabsf(result - dot_ref) / test_size;
}

static int test_vec_dot_f32(bool verbose) {
    const auto * f32 = ggml_get_type_traits_cpu(GGML_TYPE_F32);
    int num_failed = 0;
    for (int n : {1, 2, 3, 5, 7, 8, 15, 16, 17, 31, 33, 63, 67, 127, 129, 193, 255, 1023}) {
        std::vector<float> a(n);
        std::vector<float> b(n);
        generate_data(0.0, n, a.data());
        generate_data(1.0, n, b.data());

        float result = 0.0f;
        f32->vec_dot(n, &result, 0, a.data(), 0, b.data(), 0, 1);
        const float ref = dot_product(a.data(), b.data(), n);
        const float error = fabsf(result - ref) / n;

        const bool failed = !(error < MAX_QUANTIZATION_REFERENCE_ERROR);
        num_failed += failed;
        if (failed || verbose) {
            printf(" f32 vec_dot n=%4d:                 %s (ref=%f got=%f err=%f)\n",
                   n, RESULT_STR[failed], ref, result, error);
        }
    }
    return num_failed;
}

static int test_q3_ple_vectors(bool verbose) {
    const uint8_t expected_zero[14] = { 0x00, 0x00, 0x24, 0x49, 0x92, 0x24, 0x49, 0x92, 0x24, 0x49, 0x92, 0x24, 0x49, 0x92 };
    const uint8_t expected_ramp[14] = { 0x9c, 0x3f, 0x49, 0x22, 0x49, 0xda, 0xb6, 0x91, 0x24, 0xdb, 0xb6, 0xb6, 0xed, 0xff };
    const uint8_t expected_tiny[14] = { 0x01, 0x00, 0x6d, 0xdb, 0xb6, 0x6d, 0xdb, 0xb6, 0x6d, 0xdb, 0xb6, 0x6d, 0xdb, 0xb6 };
    float data[32] = {};
    uint8_t quantized[14];
    int num_failed = 0;

    const auto * q3_traits = ggml_get_type_traits(GGML_TYPE_Q3_PLE);
    const auto * q3_cpu_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q3_PLE);
    bool traits_failed = GGML_TYPE_Q3_PLE != 43 || ggml_blck_size(GGML_TYPE_Q3_PLE) != 32 ||
        ggml_type_size(GGML_TYPE_Q3_PLE) != 14 || strcmp(ggml_type_name(GGML_TYPE_Q3_PLE), "q3_ple") != 0 ||
        q3_traits->to_float == nullptr || q3_cpu_traits->from_float == nullptr ||
        ggml_row_size(GGML_TYPE_Q3_PLE, 160) != 70;
    num_failed += traits_failed;
    if (traits_failed || verbose) {
        printf("q3_ple type/trait/row-size:          %s\n", RESULT_STR[traits_failed]);
    }

    ggml_quantize_chunk(GGML_TYPE_Q3_PLE, data, quantized, 0, 1, 32, nullptr);
    bool failed = memcmp(quantized, expected_zero, sizeof(quantized)) != 0;
    num_failed += failed;
    if (failed || verbose) {
        printf("q3_ple zero vector:                  %s\n", RESULT_STR[failed]);
    }

    for (int i = 0; i < 32; ++i) {
        data[i] = (i - 16)/4.0f;
    }
    ggml_quantize_chunk(GGML_TYPE_Q3_PLE, data, quantized, 0, 1, 32, nullptr);
    failed = memcmp(quantized, expected_ramp, sizeof(quantized)) != 0;
    num_failed += failed;
    if (failed || verbose) {
        printf("q3_ple ramp vector:                  %s\n", RESULT_STR[failed]);
    }

    for (int i = 0; i < 32; ++i) {
        data[i] = 1.0e-40f;
    }
    ggml_quantize_chunk(GGML_TYPE_Q3_PLE, data, quantized, 0, 1, 32, nullptr);
    failed = memcmp(quantized, expected_tiny, sizeof(quantized)) != 0;
    num_failed += failed;
    if (failed || verbose) {
        printf("q3_ple BF16 tiny-scale vector:       %s\n", RESULT_STR[failed]);
        if (failed) {
            printf("  actual:");
            for (uint8_t byte : quantized) {
                printf(" %02x", byte);
            }
            printf("\n");
        }
    }

    float decoded[32] = {};
    q3_traits->to_float(expected_ramp, decoded, 32);
    uint16_t ramp_scale_bits = 0;
    memcpy(&ramp_scale_bits, expected_ramp, sizeof(ramp_scale_bits));
    ggml_bf16_t ramp_scale_bf16;
    memcpy(&ramp_scale_bf16, &ramp_scale_bits, sizeof(ramp_scale_bf16));
    const float ramp_scale = ggml_bf16_to_fp32(ramp_scale_bf16);
    failed = false;
    for (int i = 0; i < 32; ++i) {
        const int bit = 3*i;
        const int byte = bit/8;
        const int shift = bit%8;
        uint16_t word = expected_ramp[2 + byte];
        if (byte + 1 < 12) {
            word |= (uint16_t) expected_ramp[2 + byte + 1] << 8;
        }
        const uint8_t code = (word >> shift) & 7;
        failed |= decoded[i] != ramp_scale * ((int) code - 4);
    }
    num_failed += failed;
    if (failed || verbose) {
        printf("q3_ple preserved-byte decoder:       %s\n", RESULT_STR[failed]);
    }

    for (int i = 0; i < 32; ++i) {
        data[i] = 0.25f;
    }
    ggml_quantize_chunk(GGML_TYPE_Q3_PLE, data, quantized, 0, 1, 32, nullptr);
    failed = !ggml_validate_row_data(GGML_TYPE_Q3_PLE, quantized, sizeof(quantized));
    num_failed += failed;
    if (failed || verbose) {
        printf("q3_ple nonzero-constant validation:   %s\n", RESULT_STR[failed]);
    }

    for (int i = 0; i < 32; ++i) {
        data[i] = (i & 1) ? 0.75f : -0.5f;
    }
    ggml_quantize_chunk(GGML_TYPE_Q3_PLE, data, quantized, 0, 1, 32, nullptr);
    failed = !ggml_validate_row_data(GGML_TYPE_Q3_PLE, quantized, sizeof(quantized));
    num_failed += failed;
    if (failed || verbose) {
        printf("q3_ple mixed-sign validation:        %s\n", RESULT_STR[failed]);
    }

    for (int i = 0; i < 32; ++i) {
        data[i] = (i & 1) ? FLT_MAX : -FLT_MAX;
    }
    ggml_quantize_chunk(GGML_TYPE_Q3_PLE, data, quantized, 0, 1, 32, nullptr);
    failed = !ggml_validate_row_data(GGML_TYPE_Q3_PLE, quantized, sizeof(quantized));
    num_failed += failed;
    if (failed || verbose) {
        printf("q3_ple extreme-finite validation:    %s\n", RESULT_STR[failed]);
    }

    std::vector<float> rows(3 * 160);
    for (size_t i = 0; i < rows.size(); ++i) {
        rows[i] = (float) ((int) (i % 41) - 20) / 7.0f;
    }
    std::vector<uint8_t> rows_q(3 * 70);
    const size_t rows_bytes = ggml_quantize_chunk(GGML_TYPE_Q3_PLE, rows.data(), rows_q.data(), 0, 3, 160, nullptr);
    failed = rows_bytes != rows_q.size() || !ggml_validate_row_data(GGML_TYPE_Q3_PLE, rows_q.data(), rows_q.size());
    num_failed += failed;
    if (failed || verbose) {
        printf("q3_ple width160 multiple rows:        %s\n", RESULT_STR[failed]);
    }

    std::vector<uint8_t> one_row(3 * 70, 0xa5);
    const size_t one_row_bytes = ggml_quantize_chunk(GGML_TYPE_Q3_PLE, rows.data(), one_row.data(), 160, 1, 160, nullptr);
    failed = one_row_bytes != 70 || memcmp(one_row.data() + 70, rows_q.data() + 70, 70) != 0;
    num_failed += failed;
    if (failed || verbose) {
        printf("q3_ple width160 partial row:          %s\n", RESULT_STR[failed]);
    }

    uint8_t invalid_scale[14] = {};
    invalid_scale[0] = 0x80;
    invalid_scale[1] = 0x7f; // BF16 +inf
    failed = ggml_validate_row_data(GGML_TYPE_Q3_PLE, invalid_scale, sizeof(invalid_scale));
    num_failed += failed;
    if (failed || verbose) {
        printf("q3_ple invalid-scale rejection:      %s\n", RESULT_STR[failed]);
    }

    return num_failed;
}

static int test_vec_dot_q(bool verbose) {
    int num_failed = 0;

    const size_t test_size = 32 * 128;

    std::vector<float> test_data(test_size);
    std::vector<float> test_data2(test_size);

    generate_data(0.0, test_data.size(), test_data.data());
    generate_data(1.0, test_data2.size(), test_data2.data());

    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_type type = (ggml_type) i;
        const auto * qfns = ggml_get_type_traits(type);
        const auto * qfns_cpu = ggml_get_type_traits_cpu(type);

        // deprecated - skip
        if (qfns->blck_size == 0) {
            continue;
        }

        const ggml_type ei = (ggml_type)i;

        printf("Testing %s\n", ggml_type_name((ggml_type) i));
        ggml_quantize_init(ei);

        if (qfns_cpu->from_float && qfns->to_float) {
            const float total_error = total_quantization_error(qfns, qfns_cpu, test_size, test_data.data());
            const float max_quantization_error =
                type == GGML_TYPE_Q1_0    ? MAX_QUANTIZATION_TOTAL_ERROR_BINARY :
                type == GGML_TYPE_TQ1_0   ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_TQ2_0   ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_Q2_0    ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_Q2_K    ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :
                type == GGML_TYPE_IQ2_S   ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :
                type == GGML_TYPE_Q3_K    ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                type == GGML_TYPE_IQ3_S   ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                type == GGML_TYPE_IQ3_XXS ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS :
                type == GGML_TYPE_Q3_PLE  ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                type == GGML_TYPE_NVFP4   ? MAX_QUANTIZATION_TOTAL_ERROR_FP4 : MAX_QUANTIZATION_TOTAL_ERROR;
            bool failed = !(total_error < max_quantization_error);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s absolute quantization error:    %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], total_error);
            }

            const float reference_error = reference_quantization_error(qfns, qfns_cpu, test_size, test_data.data());
            failed = !(reference_error < MAX_QUANTIZATION_REFERENCE_ERROR);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s reference implementation error: %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], reference_error);
            }

            if (type == GGML_TYPE_Q3_PLE) {
                continue;
            }
            const float vec_dot_error = dot_product_error(qfns, qfns_cpu, test_size, test_data.data(), test_data2.data());
            const float max_allowed_error = type == GGML_TYPE_Q2_K || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_XXS ||
                type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ2_S
                ? MAX_DOT_PRODUCT_ERROR_LOWBIT
                : type == GGML_TYPE_Q1_0
                ? MAX_DOT_PRODUCT_ERROR_BINARY
                : type == GGML_TYPE_TQ1_0 || type == GGML_TYPE_TQ2_0 || type == GGML_TYPE_Q2_0
                ? MAX_DOT_PRODUCT_ERROR_TERNARY
                : type == GGML_TYPE_NVFP4
                ? MAX_DOT_PRODUCT_ERROR_FP4
                : MAX_DOT_PRODUCT_ERROR;
            failed = !(vec_dot_error < max_allowed_error);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s dot product error:              %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], vec_dot_error);
            }
        }
    }

    return num_failed;
}

int main(int argc, char * argv[]) {
    bool verbose = false;

    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "-v") {
            verbose = true;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }

    ggml_cpu_init();

    int num_failed = 0;

    num_failed += test_vec_dot_f32(verbose);
    num_failed += test_q3_ple_vectors(verbose);
    num_failed += test_vec_dot_q(verbose);

    if (num_failed || verbose) {
        printf("%d tests failed\n", num_failed);
    }

    return num_failed > 0;
}
