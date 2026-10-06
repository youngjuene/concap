// Run current production controllers with deterministic storage/network/lock doubles.
// No DOM package, HTTP server, participant database or GPU is required.
import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import {fileURLToPath} from "node:url";
import vm from "node:vm";

const source=await readFile(process.env.STUDY_JS || fileURLToPath(new URL("../study.js",import.meta.url)),"utf8");
const tests=[];
const turn=()=>new Promise(resolve=>setTimeout(resolve,0));
const fakeNode=()=>({textContent:"",hidden:false,style:{setProperty(){}},setAttribute(){}});
function fragment(start,end,from=source){const a=from.indexOf(start),b=from.indexOf(end,a);assert(a>=0&&b>a,`Missing controller ${start}`);return from.slice(a,b);}
function context(code,globals={}){const scope=vm.createContext({structuredClone,URL,AbortController,setTimeout,clearTimeout,console,...globals});vm.runInContext(code,scope);return scope;}
function moduleContext(globals={}) {
  return context(source.replace(/\bboot\(\);\s*$/,"")+`\nglobalThis.audit={save,saved,send,setState:(next)=>{state=next;},lock:typeof acquireSessionLock==='function'?acquireSessionLock:null,unlock:typeof releaseSessionLock==='function'?releaseSessionLock:null};`,{
    document:{getElementById:fakeNode},window:{location:{href:"http://localhost/study/"},addEventListener(){}},navigator:{},...globals,
  });
}
async function test(name,fn){try{await fn();tests.push({name,ok:true});}catch(error){tests.push({name,ok:false,error:error.stack});}}

await test("reset discards an earlier failed setting and retry cannot resurrect it",async()=>{
 const controller=fragment('  const desired = {...state.axes}', '  for (const key of ["texture", "context"]) {\n    const label = el("label")');
 const calls=[];let fail=true;
 const ctx=context(controller+`\nglobalThis.audit={commit,set:(value)=>Object.assign(desired,value),reset:()=>{Object.assign(desired,state.defaults);updateControls();commit("reset");},retry:retrySettings.action,snapshot:()=>({desired:{...desired},failed:failedSettings,retryVisible:!retrySettings.hidden})};for(const key of ["texture","context"]){ranges[key]=fakeNode();labels[key]=fakeNode();}`,{
  state:{axes:{texture:.5,context:.5},defaults:{texture:.5,context:.5},position_ms:0,sequence:-1},videoInfo:{id:"fixture",duration_ms:10000},video:{currentTime:0},
  el:fakeNode,fakeNode,meterRows:{texture:{value:fakeNode(),fill:fakeNode()},context:{value:fakeNode(),fill:fakeNode()}},dot:fakeNode(),button:(label,action)=>({...fakeNode(),action}),t:x=>x,showMeter(){},
  send:async(action,data)=>{calls.push(structuredClone(data));if(fail)throw Object.assign(Error("503"),{status:503});return {axes:{texture:data.texture,context:data.context}};},
 });
 ctx.audit.set({texture:1,context:1});ctx.audit.commit();await turn();
 assert.equal(ctx.audit.snapshot().retryVisible,true);
 fail=false;ctx.audit.reset();await turn();assert.equal(ctx.audit.snapshot().retryVisible,false);assert.equal(ctx.audit.snapshot().failed,null);
 const resetCalls=calls.length;ctx.audit.retry();await turn();assert.equal(calls.length,resetCalls);assert.equal(ctx.audit.snapshot().desired.texture,.5);
 if(calls.length>1)assert.equal(calls.at(-1).texture,.5);
});

await test("storage quota failure retains exposures in memory and flushes them to the API",async()=>{
 const posted=[];
 const ctx=moduleContext({localStorage:{getItem:()=>null,setItem(){throw Object.assign(Error("quota"),{name:"QuotaExceededError"});}}});
 ctx.audit.setState({session_id:"fixture"});
 const finish=fragment('  function finishExposure(','  async function flush() {');
 const flush=fragment('  async function flush() {','  function paint() {');
 const ended=fragment('  video.onended = async () => {','  const fullscreen = button(',source.slice(source.indexOf('function watching()')));
 Object.assign(ctx,{next:{disabled:true},video:{ended:true},videoInfo:{id:"fixture-video"},tick:async()=>{},pauseForRecovery(){throw Error("Unexpected recovery");},post:async(action,data)=>{posted.push({action,data:structuredClone(data)});}});
 vm.runInContext(`send=post;let exposure={id:"e1",video_id:"fixture-video",start_ms:0,end_ms:9000,cue_end:10000};const outboxKey="exposures",lastPosition=10000,alive=true;${finish}\n${flush}\n${ended}`,ctx);
 await ctx.video.onended();assert.equal(posted.length,1);assert.equal(posted[0].action,"exposures");assert.equal(posted[0].data.entries[0].end_ms,10000);assert.equal(ctx.next.disabled,false);assert.equal(ctx.audit.saved("exposures",[]).length,0);
});

await test("one session has one active tab; a second tab only enters after release",async()=>{
 const held=new Map();
 const locks={request:async(name,options,callback)=>{if(held.has(name))return callback(null);const token={name};held.set(name,token);try{return await callback(token);}finally{if(held.get(name)===token)held.delete(name);}}};
 const a=moduleContext({navigator:{locks}}),b=moduleContext({navigator:{locks}});
 for(const ctx of[a,b])ctx.audit.setState({session_id:"same-participant"});
 assert.equal(typeof a.audit.lock,"function","Session ownership must exist before rendering or outbox recovery");
 assert.equal(await a.audit.lock(),true);assert.equal(await b.audit.lock(),false);
 a.audit.unlock();await turn();assert.equal(await b.audit.lock(),true);b.audit.unlock();
});

await test("pagehide prevents stale handlers and network retries until BFCache reacquires",async()=>{
 const held=new Map(),events=new Map();let requests=0,renders=0,refuse;
 const locks={request:async(name,options,callback)=>{if(held.has(name))return callback(null);held.set(name,true);try{return await callback({name});}finally{held.delete(name);}}};
 const ctx=moduleContext({navigator:{locks},crypto:{randomUUID:()=>"fixture-key"},window:{location:{href:"http://localhost/study/"},addEventListener:(name,fn)=>events.set(name,fn)},fetch:async()=>{requests++;return await new Promise((resolve,reject)=>{refuse=reject;});}});
 ctx.audit.setState({session_id:"same-participant",revision:1});assert.equal(await ctx.audit.lock(),true);
 const pending=ctx.audit.send("settings",{texture:1,context:1});await turn();assert.equal(requests,1);
 events.get("pagehide")();refuse(Error("late network error"));await pending;await turn();
 assert.equal(requests,1,"A released tab must not retry an in-flight mutation");
 await assert.rejects(ctx.audit.send("settings",{texture:0,context:0}),/no longer owns/);assert.equal(requests,1);
 ctx.resumeState={session_id:"same-participant",revision:2};ctx.markRender=()=>{renders++;};
 vm.runInContext("request=async()=>resumeState;render=markRender;",ctx);
 events.get("pageshow")({persisted:true});await turn();assert.equal(renders,1);ctx.audit.unlock();
});

await test("unsupported session locks fail closed instead of allowing competing writers",async()=>{
 const ctx=moduleContext();ctx.audit.setState({session_id:"fixture"});assert.equal(typeof ctx.audit.lock,"function");assert.equal(await ctx.audit.lock(),false);
});

await test("previous durable open exposure is recovered once by the owning tab",async()=>{
 const values=new Map([["caption-study:fixture:open-exposure",JSON.stringify({id:"legacy",video_id:"fixture-video",start_ms:0,end_ms:2500,cue_end:10000})]]);
 const localStorage={getItem:key=>values.get(key)||null,setItem:(key,value)=>values.set(key,value)};
 const ctx=moduleContext({localStorage});ctx.audit.setState({session_id:"fixture"});
 const recovery=fragment('  const outboxKey = "exposures";','  function finishExposure(');
 vm.runInContext(`(()=>{${recovery}})();(()=>{${recovery}})();`,ctx);
 const entries=ctx.audit.saved("exposures",[]);assert.equal(entries.length,1);assert.equal(entries[0].id,"legacy");assert.equal(entries[0].incomplete,true);assert.equal(ctx.audit.saved("open-exposure",null),null);
});

await test("reload preserves the queued immutable record when clearing open exposure was interrupted",async()=>{
 const record={id:"completed",video_id:"fixture-video",start_ms:0,end_ms:4000,cue_end:10000};
 const values=new Map([["caption-study:fixture:exposures",JSON.stringify([record])],["caption-study:fixture:open-exposure",JSON.stringify({...record,end_ms:2500})]]);
 const localStorage={getItem:key=>values.get(key)||null,setItem:(key,value)=>values.set(key,value)};
 const ctx=moduleContext({localStorage});ctx.audit.setState({session_id:"fixture"});
 const recovery=fragment('  const outboxKey = "exposures";','  function finishExposure(');
 vm.runInContext(`(()=>{${recovery}})();`,ctx);
 const entries=ctx.audit.saved("exposures",[]);assert.equal(entries.length,1);assert.equal(entries[0].end_ms,4000);assert.equal(entries[0].incomplete,undefined);assert.equal(ctx.audit.saved("open-exposure",null),null);
});

console.log(JSON.stringify({source:process.env.STUDY_JS||"../study.js",tests},null,2));
if(tests.some(test=>!test.ok))process.exitCode=1;
