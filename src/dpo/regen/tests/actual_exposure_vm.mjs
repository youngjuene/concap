// Execute the production display controller with deterministic media/HTTP doubles.
// Tests distinguish actual display replacement from nominal cue boundaries.
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import vm from 'node:vm';
const source=await readFile(process.env.STUDY_JS||fileURLToPath(new URL('../study.js',import.meta.url)),'utf8');
const watching=source.slice(source.indexOf('function watching()'));
function fragment(start,end){const a=watching.indexOf(start),b=watching.indexOf(end,a);assert(a>=0&&b>a,`Controller marker absent: ${start}`);return watching.slice(a,b)}
const code=fragment('  const outboxKey = "exposures";','  const next = button(');
const ending=fragment('  video.onended = async () => {','  const fullscreen = button(');
const turn=()=>new Promise(resolve=>setTimeout(resolve,0));
function fixture({ready=true,jobResult,requestError=false,pending=false}={}){
 const stored=new Map(),requests=[],writes=[],events=new Map(),videoEvents=new Map();let clock=0;
 const video={addEventListener:(name,fn)=>videoEvents.set(name,fn),removeEventListener(){},currentTime:0,paused:true,seeking:false,ended:false,controls:false,readyState:4,played:{length:0,start(){return 0},end(){return 0}},pause(){this.paused=true;this.onpause?.()},async play(){this.paused=false;await this.onplay?.()}};
 const state={session_id:'fixture',epoch:0,settings_revision:0,position_ms:0,sequence:-1,axes:{texture:.5,context:.5}};
 const globals={console,structuredClone,crypto:{randomUUID:()=>`exposure-${writes.length}-${stored.size}-${clock++}`},
  Date:{now:()=>clock},setTimeout:(callback,delay)=>{clock+=delay;queueMicrotask(callback);return clock},clearTimeout(){},setInterval(){return 1},clearInterval(){},
  video,state,videoInfo:{id:'video-1',duration_ms:10000,cues:[{start_ms:0,end_ms:5000,fallback:'Prepared first.'},{start_ms:5000,end_ms:10000,fallback:'Prepared second.'}]},
  caption:{textContent:''},captionLevels:{textContent:''},status:{textContent:''},document:{hidden:false,addEventListener:(name,callback)=>events.set(name,callback)},
  alive:true,resumePosition:0,initialized:false,jobs:[],activeIndex:-1,applied:null,exposure:null,lastPosition:0,polling:false,tickPending:false,sequence:-1,seekGeneration:0,acknowledgedSeek:0,
  initialCaptionReady:ready,initialCaptionPreparing:false,initialFallbackReason:null,hasPlayed:false,preparationGeneration:0,
  t:x=>x,notice(){},recoveryNotice:'',next:{disabled:true},pauseForRecovery(){throw Error('Unexpected playback recovery')},
  enablePlayback(){video.controls=true},setPreparationMessage(){},
  saved:(key,fallback)=>stored.has(key)?structuredClone(stored.get(key)):fallback,
  save:(key,value)=>{stored.set(key,structuredClone(value))},
  request:async(path,payload,timeout)=>{requests.push({path,timeout});if(requestError)throw Error('fetch failed');return {epoch:state.epoch,revision:state.settings_revision,jobs:pending?[]:[{id:'job-0',cue:0,revision:0,state:jobResult?.fallback?'failed':'succeeded',result:jobResult||{text:'Generated first.',fallback:false}}]}},
  send:async(action,data)=>{data=typeof data==='function'?data():data;writes.push({action,data:structuredClone(data)});if(action==='playback'){state.position_ms=data.position_ms;state.sequence=data.sequence;}return {...state}},
 };
 const ctx=vm.createContext(globals);vm.runInContext(code+'\n'+ending+'\nglobalThis.audit={paint,finishExposure,flush,visible:()=>({caption:caption.textContent,exposure:exposure&&structuredClone(exposure),lastPosition}),outbox:()=>saved("exposures",[])};',ctx);
 return {ctx,video,state,requests,writes,events,videoEvents,stored,turn};
}
const tests=[];
async function test(name,run){try{await run();tests.push({name,ok:true})}catch(error){tests.push({name,ok:false,error:error.stack})}}
await test('delayed replacement closes old caption at actual change and starts the next without a gap',async()=>{
 const f=fixture();f.video.paused=false;f.ctx.audit.paint();f.video.currentTime=4.9;f.ctx.audit.paint();f.video.currentTime=5.18;f.ctx.audit.paint();
 const first=f.ctx.audit.outbox()[0],next=f.ctx.audit.visible().exposure;
 assert.equal(first.end_ms,5180);assert.equal(next.start_ms,5180);assert.equal(first.cue,0);assert.equal(first.cue_end_ms,5000);assert.equal(first.display_interval_version,2);
});
await test('one replacement sample closes and opens adjacent intervals without clock-read overlap',async()=>{
 const f=fixture();f.video.paused=false;f.ctx.audit.paint();f.video.currentTime=4.9;f.ctx.audit.paint();
 let position=5.18;Object.defineProperty(f.video,'currentTime',{get(){const sampled=position;position+=.001;return sampled;}});
 f.ctx.audit.paint();const old=f.ctx.audit.outbox()[0],next=f.ctx.audit.visible().exposure;
 assert.equal(old.end_ms,next.start_ms);assert.equal(old.end_ms,5180);
});
await test('pause records the observed stop without padding to nominal cue end',async()=>{
 const f=fixture();f.video.paused=false;f.ctx.audit.paint();f.video.currentTime=2.1;f.ctx.audit.paint();f.video.currentTime=2.337;f.video.pause();await turn();
 const record=f.writes.find(w=>w.action==='exposures')?.data.entries[0]||f.ctx.audit.outbox()[0];
 assert.equal(record.end_ms,2337);assert.notEqual(record.end_ms,5000);
});
await test('forward seeking closes only the previously played interval, never the jump',async()=>{
 const f=fixture();f.video.paused=false;f.ctx.audit.paint();f.video.currentTime=2.2;f.ctx.audit.paint();f.video.currentTime=2.4;f.videoEvents.get('pointerdown')?.();f.video.played={length:1,start:()=>0,end:()=>9};f.video.seeking=true;f.video.currentTime=8;f.video.onseeking();
 const record=f.ctx.audit.outbox()[0];assert.equal(record.end_ms,2400);assert.equal(f.ctx.audit.visible().exposure,null);assert.equal(f.ctx.caption.textContent,'');
});
await test('natural end closes the final visible caption at actual video end',async()=>{
 const f=fixture();f.video.currentTime=5;f.video.paused=false;f.ctx.audit.paint();f.video.currentTime=9.9;f.ctx.audit.paint();f.video.currentTime=10;f.video.paused=true;f.video.ended=true;await f.video.onended();
 assert.equal(f.writes.find(w=>w.action==='exposures').data.entries[0].end_ms,10000);assert.equal(f.ctx.next.disabled,false);
});
await test('a caption selected while paused begins at the true initial zero on play',async()=>{
 const f=fixture();f.ctx.audit.paint();f.video.currentTime=.036;await f.video.play();
 assert.equal(f.ctx.audit.visible().exposure.start_ms,0);
});
await test('an already successful first job is selected before playback controls open',async()=>{
 const f=fixture({ready:false});await f.video.onloadedmetadata();await turn();
 assert(f.requests.some(r=>r.path==='/api/study/captions'));assert.equal(f.ctx.caption.textContent,'Generated first.');assert.equal(f.video.controls,true);assert.equal(f.ctx.audit.visible().exposure,null);
});
await test('known failed first job keeps its identity and failure category',async()=>{
 const f=fixture({ready:false,jobResult:{text:'Prepared first.',fallback:true,reason:'inference_deadline'}});await f.video.onloadedmetadata();await turn();await f.video.play();
 const e=f.ctx.audit.visible().exposure;assert.equal(e.fallback,true);assert.equal(e.job_id,'job-0');assert.equal(e.fallback_reason,'job_failed');assert.equal(e.axes,null);
});
await test('initial caption HTTP failure permits an explicit network fallback, not a fake inference error',async()=>{
 const f=fixture({ready:false,requestError:true});await f.video.onloadedmetadata();await turn();await f.video.play();
 const e=f.ctx.audit.visible().exposure;assert.equal(f.video.controls,true);assert.equal(e.fallback_reason,'caption_fetch_failed');assert.equal(e.job_id,null);
});
await test('initial wait is bounded and records readiness timeout honestly',async()=>{
 const f=fixture({ready:false,pending:true});await f.video.onloadedmetadata();await turn();await f.video.play();
 assert.equal(f.video.controls,true);assert.equal(f.ctx.audit.visible().exposure.fallback_reason,'initial_caption_timeout');assert(f.requests.length<=25);
});
console.log(JSON.stringify({tests},null,2));if(tests.some(t=>!t.ok))process.exitCode=1;
