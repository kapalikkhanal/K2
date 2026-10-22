#pragma once
// Host-only shim: the production build uses the ESP32 Arduino headers.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <deque>
#include <vector>
#define MALLOC_CAP_INTERNAL 1
#define MALLOC_CAP_8BIT 2
static size_t fakeFreeHeap = 300000;
inline size_t heap_caps_get_free_size(int) { return fakeFreeHeap; }
inline void *heap_caps_malloc(size_t n, int) {
  if (n > fakeFreeHeap) return nullptr;
  fakeFreeHeap -= n;
  return malloc(n);
}
static uint32_t fakeMillis = 0;
inline uint32_t millis() { return fakeMillis++; }
// Monotonic and always advancing, so an idle-gap check can never sit still.
static uint32_t fakeMicros = 0;
inline uint32_t micros() { return fakeMicros += 100; }
struct HardwareSerial {
  std::deque<uint8_t> rx;
  std::vector<uint8_t> response;
  bool echo = true;
  int available() { return rx.size(); }
  int read() { int v = rx.front(); rx.pop_front(); return v; }
  void write(const uint8_t *p, size_t n) {
    if (echo) rx.insert(rx.end(), p, p+n);
    rx.insert(rx.end(), response.begin(), response.end());
  }
  void flush() {}
};
