#pragma once
#include <Arduino.h>
#include <atomic>
#include <esp_partition.h>
#include <esp_timer.h>
#include <esp_system.h>
#include "flight_record.h"

// Only this worker writes flash. No erase or download while torque is on.
// Internal-RAM queue prevents a flash-cache stall from invalidating its buffer.
namespace flightlog {
static const esp_partition_t *partition = nullptr;
static QueueHandle_t queue = nullptr;
static SemaphoreHandle_t storageLock = nullptr;
static uint32_t session=0, sequence=0;
static std::atomic<uint32_t> pending{0};
static std::atomic<uint32_t> offset{0}, dropped{0};
static std::atomic<bool> ready{false}, full{false}, ioError{false};
static bool workerStarted=false;
static bool active=false; // control-loop-owned: torque may be on
// Boot provenance is produced long before the operator arms, and a bench
// session may power-cycle many times without ever arming. Staging those records
// in RAM means an un-armed boot costs no flash at all, while an armed run still
// begins with the full identity/calibration block the decoder needs.
static const uint8_t STAGE_MAX=32;
static FlightRecord stage[STAGE_MAX];
static uint8_t staged=0;
static uint32_t stageLost=0;
static bool holding=true;
static void worker(void *) {
  FlightRecord record;
  for(;;) if(xQueueReceive(queue,&record,portMAX_DELAY)==pdTRUE) {
    xSemaphoreTake(storageLock,portMAX_DELAY);
    if(ready && offset+sizeof(record)<=partition->size) {
      // Claim even a failed write's slot; retries must never program over it.
      uint32_t at=offset;offset+=sizeof(record);
      if(esp_partition_write(partition,at,&record,sizeof(record))!=ESP_OK) {
        ioError=true;ready=false;
      }
      if(offset+sizeof(record)>partition->size) {full=true;ready=false;}
    } else ++dropped;
    --pending;
    xSemaphoreGive(storageLock);
  }
}
static void begin() {
  partition=esp_partition_find_first(ESP_PARTITION_TYPE_DATA,
                                   ESP_PARTITION_SUBTYPE_ANY,"k2log");
  if(!partition)return;
  storageLock=xSemaphoreCreateMutex();
  queue=xQueueCreate(64,sizeof(FlightRecord));
  if(!storageLock||!queue)return;
  // Scan every slot: a partially written record may have an erased prefix.
  // Never erase automatically; unknown data from an old partition is retained.
  FlightRecord record;
  bool unknown=false, known=false;
  for(uint32_t at=0;at+sizeof(record)<=partition->size;at+=sizeof(record)) {
    if(esp_partition_read(partition,at,&record,sizeof(record))!=ESP_OK) {ioError=true;return;}
    bool blank=true;
    const auto *bytes=reinterpret_cast<const uint8_t *>(&record);
    for(size_t i=0;i<sizeof(record);i++)
      if(bytes[i]!=0xff) {blank=false;break;}
    if(!blank) {
      offset=at+sizeof(record);
      if(record.magic!=0x324c324b)unknown=true;
      else if(record.crc==recordCrc(&record,508))known=true;
    }
  }
  full=offset+512>partition->size;
  ready=(!unknown||known)&&!full;
  session=esp_random();
  if(xTaskCreatePinnedToCore(worker,"flightlog",4096,nullptr,1,nullptr,0)!=pdPASS)
    {ready=false;ioError=true;}
  else workerStarted=true;
}
// Sequence numbers are assigned at the point of writing, not of staging, so a
// discarded pre-arm batch never leaves a gap in the armed session's numbering.
static void enqueue(FlightRecord &record) {
  record.session=session;record.sequence=sequence++;record.dropped=dropped;
  record.crc=recordCrc(&record,508);
  ++pending;
  if(xQueueSend(queue,&record,0)!=pdTRUE) {++dropped;--pending;}
}
static void submit(FlightRecord &record) {
  if(!ready) {++dropped;return;}
  record.timeUs=esp_timer_get_time(); // real event time, kept across staging
  if(holding) {
    if(staged<STAGE_MAX) stage[staged++]=record;
    else ++stageLost;
    return;
  }
  enqueue(record);
}
// Called once, when the operator arms. Everything staged becomes the head of
// the armed session, in the order and at the times it actually happened.
static void commit() {
  holding=false;
  for(uint8_t i=0;i<staged;i++) enqueue(stage[i]);
  staged=0;
}
// Called when a boot ends without ever arming. Writes nothing.
static void discard() {staged=0;}
static void event(const char *message) {
  FlightRecord r;r.kind=1;
  snprintf(r.payload.text,sizeof(r.payload.text),"%s",message);
  submit(r);
}
static void drain() {
  if(!queue||!storageLock)return;
  // Caller has already disabled torque. Bound the wait if flash has failed.
  uint32_t start=millis();
  while(pending.load()&&uint32_t(millis()-start)<3000)delay(5);
  xSemaphoreTake(storageLock,portMAX_DELAY);xSemaphoreGive(storageLock);
}
static void status() {
  Serial.printf("LOG status ready=%d bytes=%lu capacity=%lu dropped=%lu full=%d io_error=%d active=%d holding=%d staged=%u stage_lost=%lu\n",
    int(ready.load()),(unsigned long)offset.load(),(unsigned long)(partition?partition->size:0),
    (unsigned long)dropped.load(),int(full.load()),int(ioError.load()),active,
    int(holding),unsigned(staged),(unsigned long)stageLost);
}
// Invoked only by the main loop. Uppercase commands avoid accidentally arming:
// L=status, D=download, E=explicit erase, all with torque off.
static bool command(char c) {
  if(c!='L'&&c!='D'&&c!='E')return false;
  if(active) {Serial.println("LOG denied: torque must be off");return true;}
  if(c=='L') {status();return true;}
  if(!partition||!queue||!storageLock||!workerStarted) {Serial.println("LOG unavailable");return true;}
  drain();
  if(pending.load()) {Serial.println("LOG busy: retry after writes drain");return true;}
  xSemaphoreTake(storageLock,portMAX_DELAY);
  if(c=='E') {
    ready=false;
    esp_err_t result=esp_partition_erase_range(partition,0,partition->size);
    if(result==ESP_OK) {offset=0;full=false;ioError=false;dropped=0;sequence=0;session=esp_random();ready=true;
                        staged=0;stageLost=0;} // caller re-stages identity below
    else ioError=true;
    xSemaphoreGive(storageLock);
    Serial.println(result==ESP_OK?"LOG erased":"LOG erase failed");status();
    return true;
  }
  // Hex lines tolerate USB reconnects/diagnostic text. Each record has a CRC.
  // Retrieval never erases. The host retains and hashes the original binary.
  Serial.printf("LOG BEGIN %lu\n",(unsigned long)offset.load());
  Serial.setTxTimeoutMs(1000);
  FlightRecord r;char line[1026];const char *hex="0123456789abcdef";
  for(uint32_t at=0;at<offset;at+=sizeof(r)) {
    if(esp_partition_read(partition,at,&r,sizeof(r))!=ESP_OK)break;
    const auto *p=reinterpret_cast<const uint8_t *>(&r);
    for(size_t i=0;i<sizeof(r);i++){line[2*i]=hex[p[i]>>4];line[2*i+1]=hex[p[i]&15];}
    line[1024]='\n';line[1025]=0;
    if(Serial.write(reinterpret_cast<const uint8_t *>(line),1025)!=1025)break;
    delay(1);
  }
  Serial.println("LOG END");Serial.setTxTimeoutMs(1);
  xSemaphoreGive(storageLock);
  return true;
}
} // namespace flightlog
