#include <assert.h>
#include <cstdio>
#include "feedback_guard.h"
#include "robot_control.h"
#include "k2_obs.h"
#include "feetech.h"

extern "C" {
void policy_init(int flashOnly) {
  fakeFreeHeap = flashOnly ? 0 : 300000;
  loadWeights(65536);
}
void policy_infer(const float *obs, float *out) { infer(obs, out); }
void policy_reset() {
  phase = gaitAct = headingRef = 0;
  memset(lastAction, 0, sizeof(lastAction));
  commandProfile = CommandProfile();
}
void policy_obs(const float *in, float *out) {
  buildObsFrom(in, in+3, in+6, in+18, in[30], in[31], in[32], in[33] > 0.5f, in[34], out);
}

void regression_tests() {
  {
    // A single missed reply blocks the policy for that tick but must NOT latch:
    // servo id 7 drops ~8% of group reads on this robot, nearly always for one
    // tick, and latching on the first miss stops the robot after one command.
    FeedbackGuard g;g.reset(100);
    g.update(0x0ffe,120);
    assert(g.stale&&g.blocking()&&!g.paused);
    g.update(0x0fff,140);
    assert(!g.stale&&!g.blocking()&&!g.paused&&g.misses==0);
    // LATCH_TICKS consecutive misses is a real outage: latch it.
    uint32_t at=160;
    for(int i=0;i<FeedbackGuard::LATCH_TICKS-1;i++){g.update(0x0ffe,at);at+=20;assert(!g.paused);}
    g.update(0x0ffe,at);assert(g.paused&&g.blocking());
    // One joint stale beyond LATCH_MS latches on the very first miss.
    FeedbackGuard h;h.reset(1000);
    h.update(0x0ffe,1000+FeedbackGuard::LATCH_MS+1);assert(h.paused);
    assert(!h.resume(0x0ffe,true));assert(h.paused);   // still incomplete
    assert(!h.resume(0x0fff,false));assert(h.paused);  // controls not neutral
    assert(h.resume(0x0fff,true));assert(!h.paused);   // complete AND neutral
    // The 2 s fault threshold and clock-rollover handling are unchanged.
    FeedbackGuard e;e.reset(100);
    e.update(0x0ffe,120);assert(!e.expired(2100));
    assert(e.expired(2101)==1);
    e.update(1,2110);assert(e.expired(2121)==0x0ffe);
    e.reset(0xfffffff0u);e.update(0x0ffe,0x10u);
    assert(!e.expired(0x7c0u));assert(e.expired(0x7c1u)==1);

  }
  {
    HardwareSerial b;b.echo=false;
    auto add=[&](uint8_t id,uint8_t error,bool corrupt=false) {
      std::vector<uint8_t> frame={255,255,id,6,error,0x34,0x02,0x10,0x80,0};
      uint8_t sum=0;for(size_t i=2;i<frame.size()-1;i++)sum+=frame[i];
      frame.back()=uint8_t(~sum)^(corrupt?1:0);
      b.response.insert(b.response.end(),frame.begin(),frame.end());
    };
    uint8_t ids[]={5,6,7},data[12]={},status[3]={};uint16_t valid=0,errors=0;
    add(5,0); // A partial group must retain this reply even with no echo.
    assert(!feetech::syncRead(b,ids,3,0x38,4,data,200,&valid,&errors,status));
    assert(valid==1 && errors==0 && data[0]==0x34);
    b.response.clear();add(7,0);add(6,4);add(5,0);
    assert(!feetech::syncRead(b,ids,3,0x38,4,data,200,&valid,&errors,status));
    assert(valid==5 && errors==2 && status[1]==4);
    b.response.clear();add(5,0,true);add(6,0);add(7,0);
    assert(!feetech::syncRead(b,ids,3,0x38,4,data,200,&valid,&errors,status));
    assert(valid==6 && errors==0);
    b.response.clear();add(7,0);add(5,0);add(5,0);add(6,0);b.echo=true;
    assert(feetech::syncRead(b,ids,3,0x38,4,data,200,&valid,&errors,status));
    assert(valid==7 && errors==0);
  }

  {
    RobotControl c;PadSample p;p.connected=true;p.rt=1023;
    for(uint16_t raw : {uint16_t(0),uint16_t(65535)}) {
      p.forward=p.lateral=p.turn=raw;c.update(p,0);
      float sign=raw==0?1:-1;
      assert(fabsf(c.vx-sign*K2_VX_MAX)<1e-7f);
      assert(fabsf(c.vy-sign*K2_VY_MAX)<1e-7f);
      assert(fabsf(c.wz-sign*K2_WZ_MAX)<1e-7f);
    }
    p.forward=p.lateral=p.turn=16384;c.update(p,0);
    assert(c.vx>0&&c.vx<K2_VX_MAX);
    p.forward=p.lateral=p.turn=32768;c.update(p,0);assert(!c.marching);
    float value;
    assert(parseWebSpeed("0.05",K2_VX_MAX,value));
    assert(parseWebSpeed("0.03",K2_VY_MAX,value));
    char yawLimit[32];snprintf(yawLimit,sizeof(yawLimit),"%.9g",double(K2_WZ_MAX));
    assert(parseWebSpeed(yawLimit,K2_WZ_MAX,value));
    assert(fabsf(value-K2_WZ_MAX)<1e-7f);
    snprintf(yawLimit,sizeof(yawLimit),"%.9g",double(K2_WZ_MAX+0.01f));
    assert(!parseWebSpeed(yawLimit,K2_WZ_MAX,value));
    for(const char *bad : {"", "nan", "inf", "-0.01", "0", "0.06", "0.02junk"})
      assert(!parseWebSpeed(bad,K2_VX_MAX,value));
  }
  {
    RebootChord chord;PadSample p;p.connected=true;p.x=true;p.rt=1023;
    assert(!chord.update(p,0));assert(!chord.update(p,2000)); // held across boot
    p.x=false;assert(!chord.update(p,2001));
    p.x=true;assert(!chord.update(p,2002));assert(!chord.update(p,3001));
    assert(chord.update(p,3002));assert(!chord.update(p,5000));
    p.rt=0;assert(!chord.update(p,5001)); // X alone cannot reboot
  }
  {
    CommandArbiter a;RobotControl p;WebCommand w;
    w.moving=true;w.vx=.02f;w.intent=1;w.at=100;
    a.update(p,false,w,100);assert(a.owner==CommandArbiter::WEB && a.vx==.02f);
    p.linked=true;p.armed=true; // connected, centered pad never steals motion
    a.update(p,false,w,101);assert(a.owner==CommandArbiter::WEB);
    p.marching=true;p.vy=.015f;
    a.update(p,false,w,102);assert(a.owner==CommandArbiter::PAD && a.vx==0);
    w.at=200; // browser heartbeat must not steal it back
    a.update(p,false,w,200);assert(a.owner==CommandArbiter::PAD);
    w.intent++;w.vx=-.02f;
    a.update(p,false,w,201);assert(a.owner==CommandArbiter::WEB && a.vy==0);
    w.moving=false;w.intent++;
    a.update(p,false,w,202);assert(!a.moving); // no old held pad command resumes
    p.vy=-.015f;a.update(p,false,w,203);assert(a.owner==CommandArbiter::PAD);
    p.marching=false;a.update(p,false,w,204);assert(!a.moving);
    w.moving=true;w.intent++;w.at=205;
    a.update(p,false,w,205);assert(a.moving);
    a.update(p,false,w,705);assert(!a.moving); // browser timeout holds
  }
  WebCommand web;
  assert(!freshWebCommand(web,0));
  web.moving=true;web.at=100;
  assert(freshWebCommand(web,599));assert(!freshWebCommand(web,600));
  web.at=0xfffffff0;
  assert(freshWebCommand(web,0x10));assert(!freshWebCommand(web,0x200));
  RobotControl c;
  PadSample p;
  p.connected = true; p.receivedAt = 100; p.rt = 1023;
  c.update(p, 100); assert(c.armed && !c.marching);
  p.forward = 0; c.update(p, 100);
  assert(c.marching && c.vx > 0); // no A required
  p.forward = 32768; p.turn = 0; c.update(p, 100);
  assert(c.marching && c.wz > 0); // turn also activates gait
  p.turn = 32768; p.a = true; c.update(p, 100);
  assert(c.marching);
  p.a = false; p.rt = 100; c.update(p, 100); assert(c.armed); // hysteresis
  p.rt = 0; c.update(p, 100);
  assert(!c.armed && !c.marching && c.vx == 0 && c.wz == 0);
  p.rt = 100; c.update(p, 100); assert(!c.armed);
  p.rt = 1023; c.update(p, 100); assert(!c.marching);
  c.update(p, 400); assert(c.linked && !c.marching); // centered: no latched march
  p.receivedAt = 401; c.update(p, 401); assert(!c.marching); // latch cleared
  p.forward = 0; c.update(p, 401); assert(c.marching);
  p.connected = false; c.update(p, 401); assert(!c.marching);
  p.connected = true; p.receivedAt = 0xfffffff0;
  c.update(p, 0x10); assert(c.linked); // millis rollover
  c.update(p, 0x200); assert(c.linked); // HID only sends changes
  p.forward=32768;p.lateral=0;c.update(p,0x200);assert(c.vy>0 && c.marching);
  p.lateral=32768;c.update(p,0x200);assert(!c.marching);

  // Sustained saturation must not accumulate past limits, and the next
  // reversed command must move immediately by one slew-limited step.
  for (int j = 0; j < 12; j++) {
    float q = K2_DEFAULT_POS[j];
    for (int t = 0; t < 500; t++) q = boundedJointCommand(j, q, 100, 0.02f);
    assert(fabsf(q - K2_QMAX[j]) < 1e-6f);
    float reverse = boundedJointCommand(j, q, -100, 0.02f);
    assert(fabsf(reverse - (q - 0.02f)) < 1e-6f);
  }

  HardwareSerial bus;
  auto reply = [&](uint8_t error) {
    bus.response = {0xff, 0xff, 5, 6, error, 0x34, 0x02, 0x10, 0x80, 0};
    uint8_t sum = 0;
    for (size_t i = 2; i < bus.response.size()-1; i++) sum += bus.response[i];
    bus.response.back() = (uint8_t)~sum;
  };
  uint8_t out[4];
  reply(0);
  assert(feetech::readRegs(bus, 5, 0x38, 4, out, 100)); // echo + reply
  assert(out[0] == 0x34 && feetech::signMag(out[2] | (out[3]<<8)) == -16);
  bus.echo = false;
  assert(feetech::readRegs(bus, 5, 0x38, 4, out, 100));
  reply(4); assert(!feetech::readRegs(bus, 5, 0x38, 4, out, 100));
  reply(0); bus.response.back() ^= 1;
  assert(!feetech::readRegs(bus, 5, 0x38, 4, out, 100));
  bus.response.clear(); assert(!feetech::readRegs(bus, 5, 0x38, 4, out, 2));
}
}
