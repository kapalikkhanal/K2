// READ-ONLY servo-bus probe. Never writes a servo register, never enables
// torque. Answers one question: why does a 15-byte SYNC READ fail when the
// 4-byte one the old firmware used worked?
#include <Arduino.h>
#include "feetech.h"
#include "k2_config.h"

static const int PIN_SERVO_TX = 43, PIN_SERVO_RX = 44;
static const uint32_t SERVO_BAUD = 1000000;
static uint8_t ids[12];

static void rawDump(uint8_t dataLen, uint32_t waitMs) {
  uint8_t pkt[32];
  uint8_t len = 12 + 4;
  pkt[0]=0xFF;pkt[1]=0xFF;pkt[2]=0xFE;pkt[3]=len;
  pkt[4]=feetech::INST_SYNC_READ;pkt[5]=feetech::REG_PRES_POS;pkt[6]=dataLen;
  uint16_t sum = 0xFE + len + feetech::INST_SYNC_READ+feetech::REG_PRES_POS+dataLen;
  uint8_t k=7;
  for(int i=0;i<12;i++){pkt[k++]=ids[i];sum+=ids[i];}
  pkt[k++]=uint8_t(~sum);

  while(Serial1.available())Serial1.read();
  uint32_t t0=micros();
  Serial1.write(pkt,k);Serial1.flush();
  uint8_t buf[512];size_t cnt=0;uint32_t last=0;
  while(millis()-(t0/1000)<waitMs && cnt<sizeof(buf)) {
    if(Serial1.available()){buf[cnt++]=uint8_t(Serial1.read());last=micros()-t0;}
  }
  Serial.printf("  raw dataLen=%u: request %u bytes, received %u bytes, last byte at %lu us\n",
                dataLen,k,(unsigned)cnt,(unsigned long)last);
  Serial.print("   ");
  for(size_t i=0;i<cnt;i++){
    Serial.printf(" %02x",buf[i]);
    if((i+1)%24==0){Serial.println();Serial.print("   ");}
  }
  Serial.println();
}

void setup() {
  Serial.begin(115200);
  uint32_t t0=millis();
  while(!Serial && millis()-t0<3000) delay(10);
  Serial.println();
  Serial.println(F("=== K2 sync-read probe (READ ONLY, no torque) ==="));
  for(int j=0;j<12;j++) ids[j]=uint8_t(K2_SERVO_ID[j]);
  Serial1.begin(SERVO_BAUD,SERIAL_8N1,PIN_SERVO_RX,PIN_SERVO_TX);
  Serial1.setRxFIFOFull(1);
  Serial1.setRxTimeout(1);
  delay(200);

  Serial.println(F("\n-- ping --"));
  for(int j=0;j<12;j++) {
    size_t rx=0;
    bool ok=feetech::ping(Serial1,ids[j],&rx);
    Serial.printf("  id %2u: %s (rx %u bytes)\n",ids[j],ok?"OK ":"---",(unsigned)rx);
  }

  Serial.println(F("\n-- single-servo READ of N bytes from 0x38 --"));
  for(uint8_t n : {(uint8_t)4,(uint8_t)10,(uint8_t)15,(uint8_t)16}) {
    int ok=0;
    uint8_t out[16];
    for(int j=0;j<12;j++) if(feetech::readRegs(Serial1,ids[j],feetech::REG_PRES_POS,n,out,20)) ok++;
    Serial.printf("  N=%2u: %2d/12 servos answered\n",n,ok);
  }

  Serial.println(F("\n-- SYNC READ, generous 50 ms timeout --"));
  uint8_t data[12*16];
  for(uint8_t n : {(uint8_t)4,(uint8_t)8,(uint8_t)12,(uint8_t)14,(uint8_t)15,(uint8_t)16}) {
    uint16_t valid=0,err=0;uint8_t status[12];
    uint32_t s=micros();
    bool ok=feetech::syncRead(Serial1,ids,12,feetech::REG_PRES_POS,n,data,50,&valid,&err,status);
    uint32_t us=micros()-s;
    Serial.printf("  dataLen=%2u: ok=%d valid=0x%03x err=0x%03x  %lu us\n",
                  n,int(ok),valid,err,(unsigned long)us);
  }

  Serial.println(F("\n-- SYNC READ at the 5 ms timeout the control loop uses --"));
  for(uint8_t n : {(uint8_t)4,(uint8_t)15}) {
    int ok=0;uint32_t worst=0;
    for(int trial=0;trial<20;trial++) {
      uint16_t valid=0,err=0;uint8_t status[12];
      uint32_t s=micros();
      feetech::syncRead(Serial1,ids,12,feetech::REG_PRES_POS,n,data,5,&valid,&err,status);
      uint32_t us=micros()-s;
      if(us>worst)worst=us;
      if(valid==0x0fff)ok++;
      delay(20);
    }
    Serial.printf("  dataLen=%2u: %2d/20 complete, worst %lu us\n",n,ok,(unsigned long)worst);
  }

  // Does the bus hold exactly the twelve expected ids? A servo left on a
  // duplicate or unexpected id corrupts the group reply of whoever it shadows.
  Serial.println(F("\n-- full id scan 0..30 --"));
  Serial.print(F("  responding:"));
  int found=0;
  for(uint8_t id=0;id<=30;id++) if(feetech::ping(Serial1,id)) {Serial.printf(" %u",id);found++;}
  Serial.printf("   (%d found, expected 12)\n",found);

  Serial.println(F("\n-- Return Delay Time (reg 0x07, units of 2 us) --"));
  for(int j=0;j<12;j++) {
    uint8_t d=0;
    bool ok=feetech::readRegs(Serial1,ids[j],0x07,1,&d,20);
    Serial.printf("  id %2u: %s\n",ids[j],ok?String(String(d)+"  (~"+String(d*2)+" us)").c_str():"read failed");
  }

  // The decisive test for a single flaky servo: 300 isolated round trips each.
  // If one id fails here at the same rate it fails inside the group read, the
  // fault is that servo or its wiring, not SYNC READ arbitration.
  Serial.println(F("\n-- isolated ping reliability, 300 each --"));
  for(int j=0;j<12;j++) {
    int fail=0;
    for(int k=0;k<300;k++) if(!feetech::ping(Serial1,ids[j])) fail++;
    Serial.printf("  id %2u: %3d/300 failed (%.1f%%)%s\n",ids[j],fail,100.0*fail/300,
                  fail?"   <-- ":"");
  }

  Serial.println(F("\n-- raw wire capture --"));
  rawDump(4,60);
  rawDump(15,60);
  Serial.println(F("\n=== probe done, bus untouched ==="));
}

void loop() { delay(1000); }
