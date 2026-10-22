// Inference benchmark for the deployed K2 policy: weight placement x kernel.
//
// The exported graph is entirely static -- 4 Gemms, ELU, one input normalizer --
// so it is implemented directly rather than dragging in an inference engine.
//
// Placement (where the 215 KiB of FP32 weights live):
//   FLASH     every tensor read from flash through the CPU cache
//   DEPLOYED  exactly what the 2026-10-04 robot firmware caches with its 96 KiB
//             BLE reserve: mean/std, W0, W2, W3, B1, B3 in SRAM (88,000 bytes);
//             W1, B0, B2 in flash. (The Sep-15 build cached 84,416: no W3/B1.)
//   HYBRID    the DEPLOYED SRAM set, with the rest in PSRAM instead of flash --
//             costs no internal SRAM beyond what the robot already uses
//   PSRAM     every tensor in the 8 MiB octal PSRAM
//   SRAM      every tensor in internal SRAM (fastest; starves BLE on the robot)
//
// Kernel (the dense layer y = W x + b):
//   scalar     the C loop the robot firmware runs today (k2_obs.h)
//   dot_ae32   ESP-DSP dsps_dotprod_f32_ae32 per output row (generic Xtensa)
//   dot_aes3   ESP-DSP dsps_dotprod_f32_aes3 per output row (ESP32-S3 build)
//   mult_aes3  ESP-DSP dspm_mult_f32_aes3 as one matrix-vector product
//
// Every combination is checked against the onnxruntime probe vector embedded in
// policy_weights.h, so a fast kernel that is wrong cannot pass silently.
// The whole grid repeats at CPU 240, 160 and 80 MHz (flash and PSRAM keep their
// own fixed clocks, so a memory-bound kernel slows by less than the CPU ratio).
// Lines starting "CSV," are the machine-readable result.

#include <Arduino.h>
#include <math.h>
#include "esp_dsp.h"
#include "policy_weights.h"

static const int L_OUT[4] = {L0_OUT, L1_OUT, L2_OUT, L3_OUT};
static const int L_IN[4]  = {L0_IN,  L1_IN,  L2_IN,  L3_IN};
static const int MACS = L0_IN*L0_OUT + L1_IN*L1_OUT + L2_IN*L2_OUT + L3_IN*L3_OUT;

struct Weights { const float *mean, *std, *w[4], *b[4]; };

typedef void (*DenseFn)(const float *w, const float *b, const float *x,
                        float *y, int nOut, int nIn);

static void denseScalar(const float *w, const float *b, const float *x,
                        float *y, int nOut, int nIn) {
  for (int o = 0; o < nOut; o++) {
    const float *row = w + (size_t)o * nIn;
    float acc = b[o];
    for (int i = 0; i < nIn; i++) acc += row[i] * x[i];
    y[o] = acc;
  }
}
static void denseDotAe32(const float *w, const float *b, const float *x,
                         float *y, int nOut, int nIn) {
  for (int o = 0; o < nOut; o++) {
    float acc;
    dsps_dotprod_f32_ae32(w + (size_t)o * nIn, x, &acc, nIn);
    y[o] = acc + b[o];
  }
}
static void denseDotAes3(const float *w, const float *b, const float *x,
                         float *y, int nOut, int nIn) {
  for (int o = 0; o < nOut; o++) {
    float acc;
    dsps_dotprod_f32_aes3(w + (size_t)o * nIn, x, &acc, nIn);
    y[o] = acc + b[o];
  }
}
static void denseMultAes3(const float *w, const float *b, const float *x,
                          float *y, int nOut, int nIn) {
  dspm_mult_f32_aes3(w, x, y, nOut, nIn, 1);
  for (int o = 0; o < nOut; o++) y[o] += b[o];
}

static void infer(const Weights &W, DenseFn dense, const float *obs, float *out) {
  alignas(16) float a[256], b[256];
  for (int i = 0; i < POLICY_OBS_DIM; i++) a[i] = (obs[i] - W.mean[i]) / W.std[i];
  float *src = a, *dst = b;
  for (int l = 0; l < 4; l++) {
    dense(W.w[l], W.b[l], src, dst, L_OUT[l], L_IN[l]);
    if (l < 3) for (int i = 0; i < L_OUT[l]; i++) if (dst[i] <= 0) dst[i] = expm1f(dst[i]);
    float *t = src; src = dst; dst = t;
  }
  memcpy(out, src, POLICY_ACT_DIM * sizeof(float));
}

// ---- placements ------------------------------------------------------------

static const float *FLASH_T[10] = {P_MEAN, P_STD, P_W0, P_W1, P_W2, P_W3,
                                   P_B0, P_B1, P_B2, P_B3};
static const size_t COUNT_T[10] = {POLICY_OBS_DIM, POLICY_OBS_DIM,
  sizeof(P_W0)/4, sizeof(P_W1)/4, sizeof(P_W2)/4, sizeof(P_W3)/4,
  sizeof(P_B0)/4, sizeof(P_B1)/4, sizeof(P_B2)/4, sizeof(P_B3)/4};
// What policy_robot.cpp ends up caching with loadWeights(96 * 1024).
static const bool DEPLOYED_SRAM[10] = {true, true, true, false, true, true,
                                       false, true, false, true};

static void *allocs[10];
static int nAllocs = 0;
static size_t sramBytes = 0, psramBytes = 0;

static const float *place(int t, uint32_t caps) {
  size_t bytes = COUNT_T[t] * sizeof(float);
  float *p = (float *)heap_caps_aligned_alloc(16, bytes, caps);
  if (!p) return nullptr;
  memcpy(p, FLASH_T[t], bytes);
  allocs[nAllocs++] = p;
  (caps & MALLOC_CAP_SPIRAM ? psramBytes : sramBytes) += bytes;
  return p;
}
static void release() {
  for (int i = 0; i < nAllocs; i++) heap_caps_free(allocs[i]);
  nAllocs = 0; sramBytes = psramBytes = 0;
}

static const uint32_t SRAM_CAPS  = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;
static const uint32_t PSRAM_CAPS = MALLOC_CAP_SPIRAM   | MALLOC_CAP_8BIT;

// mode: 0 FLASH, 1 DEPLOYED, 2 HYBRID, 3 PSRAM, 4 SRAM. False if allocation fails.
static bool build(int mode, Weights &W) {
  const float *t[10];
  for (int i = 0; i < 10; i++) {
    uint32_t caps = 0;
    if (mode == 1) caps = DEPLOYED_SRAM[i] ? SRAM_CAPS : 0;
    if (mode == 2) caps = DEPLOYED_SRAM[i] ? SRAM_CAPS : PSRAM_CAPS;
    if (mode == 3) caps = PSRAM_CAPS;
    if (mode == 4) caps = SRAM_CAPS;
    t[i] = caps ? place(i, caps) : FLASH_T[i];
    if (!t[i]) { release(); return false; }
  }
  W = {t[0], t[1], {t[2], t[3], t[4], t[5]}, {t[6], t[7], t[8], t[9]}};
  return true;
}

// ---- measurement -----------------------------------------------------------

static const char *PLACEMENT[5] = {"FLASH", "DEPLOYED", "HYBRID", "PSRAM", "SRAM"};
static const char *KERNEL[4] = {"scalar", "dot_ae32", "dot_aes3", "mult_aes3"};
static const DenseFn DENSE[4] = {denseScalar, denseDotAe32, denseDotAes3, denseMultAes3};
static const int ITERS = 200;

static void measure(int mode, int k, const Weights &W) {
  float out[POLICY_ACT_DIM];
  infer(W, DENSE[k], P_PROBE_OBS, out);   // warm; not timed
  float err = 0;
  for (int i = 0; i < POLICY_ACT_DIM; i++)
    err = fmaxf(err, fabsf(out[i] - P_PROBE_EXPECTED[i]));
  uint64_t sum = 0; uint32_t lo = UINT32_MAX, hi = 0;
  int64_t wall0 = esp_timer_get_time();
  for (int it = 0; it < ITERS; it++) {
    uint32_t c0 = ESP.getCycleCount();
    infer(W, DENSE[k], P_PROBE_OBS, out);
    uint32_t c = ESP.getCycleCount() - c0;
    sum += c; lo = min(lo, c); hi = max(hi, c);
  }
  float wall = (float)(esp_timer_get_time() - wall0) / ITERS;  // includes cycle reads
  int mhz = getCpuFrequencyMhz();
  float mean = (float)sum / ITERS;
  Serial.printf("%3d MHz %-9s %-10s mean %7.1f us (wall %7.1f)  min %7.1f  max %7.1f  "
                "%5.2f cyc/MAC  max|err| %.2e%s\n", mhz, PLACEMENT[mode], KERNEL[k],
                mean / mhz, wall, lo / (float)mhz, hi / (float)mhz, mean / MACS, err,
                err > 1e-4f ? "  <-- WRONG" : "");
  Serial.printf("CSV,%d,%s,%s,%.1f,%.1f,%.1f,%.1f,%.3f,%.3e,%u,%u\n", mhz, PLACEMENT[mode],
                KERNEL[k], mean / mhz, wall, lo / (float)mhz, hi / (float)mhz, mean / MACS,
                err, (unsigned)sramBytes, (unsigned)psramBytes);
}

void setup() {
  Serial.begin(115200);
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 5000) delay(10);
  delay(500);

  Serial.println(F("\n=== K2 deployed-policy inference: placement x kernel ==="));
  Serial.printf("obs %d -> act %d, %d params (%d bytes fp32), %d MACs/inference\n",
                POLICY_OBS_DIM, POLICY_ACT_DIM, POLICY_PARAM_COUNT,
                POLICY_PARAM_COUNT * 4, MACS);
  Serial.printf("CPU %lu MHz; internal heap %u (largest %u); PSRAM %u\n",
                (unsigned long)getCpuFrequencyMhz(),
                (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL),
                (unsigned)ESP.getPsramSize());
  Serial.println(F("CSV,cpu_mhz,placement,kernel,mean_us,wall_us,min_us,max_us,cyc_per_mac,max_abs_err,sram_bytes,psram_bytes"));

  for (int mhz : {240, 160, 80}) {
    Serial.flush();
    if (!setCpuFrequencyMhz(mhz)) { Serial.printf("cannot set %d MHz\n", mhz); continue; }
    delay(200);
    Serial.printf("\n--- CPU %lu MHz (APB %lu Hz) ---\n", (unsigned long)getCpuFrequencyMhz(),
                  (unsigned long)getApbFrequency());
    for (int mode = 0; mode < 5; mode++) {
      Weights W;
      if (!build(mode, W)) { Serial.printf("%s: allocation failed\n", PLACEMENT[mode]); continue; }
      for (int k = 0; k < 4; k++) measure(mode, k, W);
      release();
    }
  }
  setCpuFrequencyMhz(240);
  Serial.println(F("=== done ==="));
}

void loop() { delay(1000); }
