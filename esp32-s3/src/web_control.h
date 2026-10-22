#pragma once
#include <WiFi.h>
#include <WebServer.h>
#include <stdlib.h>
#include "robot_control.h"

// Network work stays off the real-time servo loop.
static QueueHandle_t webQueue;
static portMUX_TYPE webMux = portMUX_INITIALIZER_UNLOCKED;
static bool webArm = false, webStop = false, webReboot = false;
static int webState = 0, webOwner = 0;
static bool webPadConnected = false;
static WebServer webServer(80);
// Web-task-owned settings, independent of the analog gamepad scale.
static float webWalkSpeed=.02f, webLateralSpeed=.015f, webTurnSpeed=.10f;
static bool readSpeed(const String &text, float maximum, float &value) {
  return parseWebSpeed(text.c_str(),maximum,value);
}
static const char WEB_PAGE[] PROGMEM = R"HTML(
<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>K2 · Robot control</title><style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#0b1119;color:#edf4fa}
*{box-sizing:border-box}body{max-width:480px;margin:auto;padding:24px 18px 32px}
header{display:flex;align-items:center;justify-content:space-between;margin-bottom:24px}
h1{margin:0;font-size:32px;letter-spacing:-1px}header span{color:#8eabc0;font-size:13px}
.card{background:#141f2c;border:1px solid #26384a;border-radius:18px;padding:18px;margin:14px 0}
.label{color:#90a9bd;font-size:12px;letter-spacing:1px;text-transform:uppercase}
#state{font-size:20px;margin:8px 0}#detail{font-size:13px;color:#9fb4c7;line-height:1.6}
button{font:600 15px system-ui;border:1px solid #344b62;border-radius:13px;background:#223449;color:#edf4fa;min-height:64px;padding:12px;cursor:pointer;touch-action:none;user-select:none;-webkit-user-select:none}
button:focus-visible{outline:3px solid #69d4c0}button:disabled{opacity:.4;cursor:default}
button.active,button:active{background:#317164;border-color:#73dec5}
.wide{width:100%;margin-top:12px}.primary{background:#246355;border-color:#4db99e}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:9px;margin-top:16px}
.grid button{height:78px}.grid b{display:block;font-size:23px;margin-bottom:2px}
.actions{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.danger{background:#52242d;border-color:#a44759}.muted{color:#a9bbca;font-size:13px;line-height:1.6}
#notice{min-height:20px;color:#f2bf7c;font-size:13px}
dialog{background:#141f2c;color:#edf4fa;border:1px solid #344b62;border-radius:18px;width:min(420px,95vw)}
dialog::backdrop{background:#000a}label{display:block;margin:18px 0}input{width:100%;padding:12px;margin-top:6px;background:#0b1119;color:white;border:1px solid #344b62;border-radius:8px;font:inherit}
</style><body><header><h1>K2<span style="color:#69d4c0">.</span></h1><span>ROBOT CONTROL</span></header>
<section class="card" aria-live="polite"><div class="label">Robot status</div><div id="state">Connecting…</div><div id="detail">Waiting for the controller</div>
<button id="arm" class="wide primary" disabled>Enable torque · Stand</button></section>
<section class="card"><div class="label">Movement</div><p class="muted">Hold a direction to move. Release to stand.<br>The latest deliberate input takes control.</p>
<div class="grid"><button data-c="turn_left"><b>↶</b>Turn left</button><button data-c="forward"><b>↑</b>Forward</button><button data-c="turn_right"><b>↷</b>Turn right</button>
<button data-c="left"><b>←</b>Left</button><button data-c="march"><b>⇅</b>March</button><button data-c="right"><b>→</b>Right</button>
<span></span><button data-c="backward"><b>↓</b>Backward</button></div>
<button id="hold" class="wide">Hold · Stop walking</button></section>
<button id="settings" class="wide">⚙ Speed settings</button>
<dialog id="speedDialog"><h2>Web speeds</h2><p class="muted">Applies to web buttons only. Joystick deflection always scales to the trained maximum. Settings reset on reboot.</p>
<form id="speedForm"><label>Forward / backward (m/s)<input id="walkSpeed" type="number" min="0.001" step="0.001" required></label>
<label>Lateral (m/s)<input id="lateralSpeed" type="number" min="0.001" step="0.001" required></label>
<label>Turning (rad/s)<input id="turnSpeed" type="number" min="0.01" step="0.01" required></label>
<p id="speedError" role="alert"></p><button type="submit">Apply speeds</button> <button type="button" id="closeSettings">Cancel</button></form></dialog>
<div class="actions"><button id="stop" class="danger">Release torque</button><button id="reboot">Reboot robot</button></div>
<p id="notice" role="alert"></p><p class="muted">Support the robot before enabling, releasing torque or rebooting. Reboot leaves torque off.<br>Gamepad: trigger + A to enable; trigger + X for 1 second to reboot. No phone required.</p>
<script>
let command='hold',pending=null,busy=false,pointer=null,activeButton=null,running=false;
function clearDirection(){command='hold';pointer=null;if(activeButton)activeButton.classList.remove('active');activeButton=null}
function enqueue(c,claim=false){pending={c,claim};drain()}
async function drain(){
 if(busy||!pending)return;let item=pending;pending=null;busy=true;
 let abort=new AbortController(),timer=setTimeout(()=>abort.abort(),450);
 try{let r=await fetch('/command',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:'cmd='+item.c+'&claim='+(item.claim?'1':'0'),signal:abort.signal});
 if(!r.ok){clearDirection();document.querySelector('#notice').textContent=await r.text()}}
 catch(e){clearDirection();document.querySelector('#notice').textContent='Connection lost. Movement will time out.'}
 finally{clearTimeout(timer);busy=false;if(pending)drain()}
}
function release(claim=false){clearDirection();enqueue('hold',claim)}
for(const b of document.querySelectorAll('[data-c]')){
 b.disabled=true;
 b.onpointerdown=e=>{if(!running||pointer!==null)return;e.preventDefault();pointer=e.pointerId;b.setPointerCapture(pointer);activeButton=b;b.classList.add('active');command=b.dataset.c;enqueue(command,true)};
 b.onpointerup=b.onpointercancel=b.onlostpointercapture=e=>{if(e.pointerId===pointer)release()};
 b.oncontextmenu=e=>e.preventDefault();
}
document.querySelector('#hold').onclick=()=>release(true);
document.querySelector('#settings').onclick=async()=>{
 if(command!=='hold')release();
 try{let r=await fetch('/settings',{cache:'no-store'}),s=await r.json();
 for(const [id,key] of [['walkSpeed','walk'],['lateralSpeed','lateral'],['turnSpeed','turn']]){let input=document.getElementById(id);input.value=s[key];input.max=s[key+'Max'];}
 document.querySelector('#speedError').textContent='';document.querySelector('#speedDialog').showModal();
 }catch(e){document.querySelector('#notice').textContent='Could not load speed settings.'}
};
document.querySelector('#closeSettings').onclick=()=>document.querySelector('#speedDialog').close();
document.querySelector('#speedForm').onsubmit=async e=>{e.preventDefault();let body=new URLSearchParams({walk:document.querySelector('#walkSpeed').value,lateral:document.querySelector('#lateralSpeed').value,turn:document.querySelector('#turnSpeed').value});
 try{let r=await fetch('/settings',{method:'POST',body});if(!r.ok)throw new Error(await r.text());document.querySelector('#speedDialog').close();document.querySelector('#notice').textContent='Web speeds updated.'}
 catch(e){document.querySelector('#speedError').textContent=e.message}
};
document.querySelector('#arm').onclick=()=>{clearDirection();enqueue('arm')};
document.querySelector('#stop').onclick=()=>{clearDirection();enqueue('stop')};
document.querySelector('#reboot').onclick=()=>{if(confirm('Reboot K2? Torque will be released. Support the robot first.')){clearDirection();enqueue('reboot');document.querySelector('#notice').textContent='Rebooting. Reconnect to K2-Robot if needed.'}};
window.onblur=()=>{if(command!=='hold')release()};document.onvisibilitychange=()=>{if(document.hidden&&command!=='hold')release()};
setInterval(()=>{if(command!=='hold'&&!document.hidden&&!pending&&!busy)enqueue(command)},150);
let polling=false;
setInterval(async()=>{if(polling)return;polling=true;let abort=new AbortController(),timer=setTimeout(()=>abort.abort(),700);
 try{let r=await fetch('/status',{cache:'no-store',signal:abort.signal}),s=await r.json();running=s.state===2;
 document.querySelector('#state').textContent=['Starting up','Ready · Torque off','Active · '+s.owner,'Stopped · Reboot to recover'][s.state];
document.querySelector('#detail').textContent=(s.pad?'Gamepad connected':'Gamepad offline');
 document.querySelector('#arm').disabled=s.state!==1;
 for(const b of document.querySelectorAll('[data-c]'))b.disabled=!running;
 if(!running)clearDirection();
 }catch(e){running=false;clearDirection();document.querySelector('#state').textContent='Reconnecting…';document.querySelector('#arm').disabled=true;for(const b of document.querySelectorAll('[data-c]'))b.disabled=true}
 finally{clearTimeout(timer);polling=false}},750);
</script></body></html>)HTML";

static void setWebState(int state) {
  portENTER_CRITICAL(&webMux);webState=state;portEXIT_CRITICAL(&webMux);
}
static bool takeWebArm() {
  portENTER_CRITICAL(&webMux);bool v=webArm;webArm=false;portEXIT_CRITICAL(&webMux);return v;
}
static bool webStopped() {
  portENTER_CRITICAL(&webMux);bool v=webStop;portEXIT_CRITICAL(&webMux);return v;
}
static bool takeWebReboot() {
  portENTER_CRITICAL(&webMux);bool v=webReboot;webReboot=false;portEXIT_CRITICAL(&webMux);return v;
}
static void webTask(void *) {
  webServer.on("/settings",HTTP_GET,[](){
    String body="{\"walk\":"+String(webWalkSpeed,4)+",\"lateral\":"+String(webLateralSpeed,4)+",\"turn\":"+String(webTurnSpeed,4)+
      ",\"walkMax\":"+String(K2_VX_MAX,4)+",\"lateralMax\":"+String(K2_VY_MAX,4)+",\"turnMax\":"+String(K2_WZ_MAX,4)+"}";
    webServer.sendHeader("Cache-Control","no-store");webServer.send(200,"application/json",body);
  });
  webServer.on("/settings",HTTP_POST,[](){
    float walk,lateral,turn;
    if(!readSpeed(webServer.arg("walk"),K2_VX_MAX,walk)||
       !readSpeed(webServer.arg("lateral"),K2_VY_MAX,lateral)||
       !readSpeed(webServer.arg("turn"),K2_WZ_MAX,turn)) {
      webServer.send(400,"text/plain","Speeds must be positive and within the trained limits.");return;
    }
    webWalkSpeed=walk;webLateralSpeed=lateral;webTurnSpeed=turn;
    webServer.send(200,"text/plain","ok");
  });
  webServer.on("/",HTTP_GET,[](){webServer.send_P(200,"text/html",WEB_PAGE);});
  webServer.on("/status",HTTP_GET,[](){
    portENTER_CRITICAL(&webMux);int s=webState,o=webOwner;bool p=webPadConnected;
    portEXIT_CRITICAL(&webMux);
    const char *owner=o==CommandArbiter::WEB?"Web":o==CommandArbiter::PAD?"Gamepad":"Hold";
    String body="{\"state\":"+String(s)+",\"owner\":\""+owner+"\",\"pad\":"+(p?"true":"false")+"}";
    webServer.sendHeader("Cache-Control","no-store");webServer.send(200,"application/json",body);
  });
  webServer.on("/command",HTTP_POST,[](){
    static uint32_t intent=0,lastAt=0;static String previous="hold";
    String action=webServer.arg("cmd");WebCommand c;c.at=millis();
    portENTER_CRITICAL(&webMux);int state=webState;bool stopped=webStop;portEXIT_CRITICAL(&webMux);
    if(action=="reboot") {portENTER_CRITICAL(&webMux);webReboot=true;portEXIT_CRITICAL(&webMux);}
    else if(action=="stop") {portENTER_CRITICAL(&webMux);webStop=true;portEXIT_CRITICAL(&webMux);}
    else if(action=="arm"&&state==1&&!stopped) {portENTER_CRITICAL(&webMux);webArm=true;portEXIT_CRITICAL(&webMux);}
    else if(action=="hold") {}
    else if(state!=2||stopped) {webServer.send(409,"text/plain","Enable torque before moving");return;}
    else if(action=="forward") {c.vx=webWalkSpeed;c.moving=true;}
    else if(action=="backward") {c.vx=-webWalkSpeed;c.moving=true;}
    else if(action=="left") {c.vy=webLateralSpeed;c.moving=true;}
    else if(action=="right") {c.vy=-webLateralSpeed;c.moving=true;}
    else if(action=="turn_left") {c.wz=webTurnSpeed;c.moving=true;}
    else if(action=="turn_right") {c.wz=-webTurnSpeed;c.moving=true;}
    else if(action=="march") {c.moving=true;}
    else {webServer.send(400,"text/plain","Unknown command");return;}
    if(action!=previous||webServer.arg("claim")=="1"||(c.moving&&(uint32_t)(c.at-lastAt)>=500))++intent;
    c.intent=intent;previous=action;lastAt=c.at;
    xQueueOverwrite(webQueue,&c);webServer.send(200,"text/plain","ok");
  });
  webServer.begin();for(;;){webServer.handleClient();vTaskDelay(pdMS_TO_TICKS(2));}
}
static bool startWeb() {
  webQueue=xQueueCreate(1,sizeof(WebCommand));if(!webQueue)return false;
  WiFi.mode(WIFI_AP);
  if(!WiFi.softAP("K2-Robot","k2robot50",1,false,1))return false;
  return xTaskCreatePinnedToCore(webTask,"web",6144,nullptr,1,nullptr,0)==pdPASS;
}
