// UI-only regression: real controls/video, deterministic API/caption fixtures.
// This does not claim new inference or end-to-end study completion evidence.
import {createServer} from "node:http";
import {execFileSync} from "node:child_process";
import {readFile, mkdir, mkdtemp, writeFile} from "node:fs/promises";
import {tmpdir} from "node:os";
import {dirname, resolve} from "node:path";
import {fileURLToPath} from "node:url";
const repo = resolve(dirname(fileURLToPath(import.meta.url)), "../../../..");
const assets = resolve(repo, "src/dpo/regen");
const output = process.env.STEERING_QA_OUTPUT || await mkdtemp(resolve(tmpdir(), "regen-meter-"));
await mkdir(output, {recursive:true});
const clip = resolve(output,"clip.mp4");
execFileSync("ffmpeg", ["-v","error","-y","-i",resolve(repo,"data/live/regen-media/viewing/viewing-1.mp4"),
  "-t","30","-vf","scale=640:-2","-an","-c:v","libx264","-preset","ultrafast","-movflags","+faststart",clip]);
const videoBytes = await readFile(clip);
const copy = await readFile(resolve(assets,"study-ko.json"));
let fixture, fallback = false, updates = [], failSettings = false;
const reset = language => {
  fallback = false; updates = []; failSettings = false;
  fixture = {stage:"watch",language,session_id:`meter-${language}`,revision:0,settings_revision:0,
    epoch:0,sequence:-1,position_ms:0,video_index:0,completed_videos:0,
    axes:{texture:.5,context:.5},defaults:{texture:.5,context:.5},
    video:{id:"meter-video",title:"Video 1",duration_ms:30000,url:"/study/media/watch/0",
      cues:Array.from({length:6},(_,index)=>({start_ms:index*5000,end_ms:(index+1)*5000,
        fallback:language === "ko" ? "준비된 테스트 자막" : "Prepared fixture caption"}))}};
};
reset("en");
const server = createServer(async (request,response) => {
  try {
    const path = new URL(request.url,"http://localhost").pathname;
    if (path === "/study/media/watch/0") {
      response.writeHead(200,{"content-type":"video/mp4","content-length":videoBytes.length}); response.end(videoBytes); return;
    }
    if (path === "/api/study/strings") {response.setHeader("content-type","application/json");response.end(copy);return;}
    if (path.startsWith("/api/study/")) {
      let body = ""; for await (const chunk of request) body += chunk;
      const payload = body ? JSON.parse(body) : {};
      const action = path.split("/").at(-1);
      response.setHeader("content-type","application/json");
      if (action === "settings") {
        if (failSettings) {response.writeHead(503);response.end(JSON.stringify({error:"Fixture unavailable"}));return;}
        fixture.axes={texture:payload.data.texture,context:payload.data.context}; fixture.settings_revision++;
        updates.push(structuredClone(fixture.axes));
      }
      if (action === "playback") { fixture.position_ms=payload.data.position_ms;fixture.sequence=payload.data.sequence; }
      if (request.method === "POST") fixture.revision++;
      if (action === "captions") {
        response.end(JSON.stringify({epoch:fixture.epoch,revision:fixture.settings_revision,
          jobs: fallback ? [] : fixture.video.cues.map((cue,index)=>({id:`fixture-${fixture.settings_revision}-${index}`,
            cue:index,revision:fixture.settings_revision,result:{text:"Fixture-generated caption",fallback:false}}))}));return;
      }
      response.end(JSON.stringify(fixture));return;
    }
    const name = path === "/" ? "study.html" : path.slice(1);
    if (!["study.html","study.js","study.css","regen.css","identity.css"].includes(name)) {response.writeHead(404);response.end();return;}
    response.setHeader("content-type",name.endsWith("js")?"text/javascript":name.endsWith("css")?"text/css":"text/html");
    response.end(await readFile(resolve(assets,name)));
  } catch (error) {response.writeHead(500);response.end(error.message);}
});
await new Promise(resolve => server.listen(0,"127.0.0.1",resolve));
const base = `http://127.0.0.1:${server.address().port}`;
const {chromium} = await import(process.env.PLAYWRIGHT_MODULE || "playwright-core");
const browser = await chromium.launch({executablePath:process.env.CHROMIUM_EXECUTABLE,args:["--no-sandbox"]});
const checks=[], errors=[];
const check=(condition,label)=>{if(!condition)throw new Error(label);checks.push(label);};
let page;
async function ready(context) {
  const page=await context.newPage();page.on("pageerror",e=>errors.push(e.message));
  await page.goto(base);await page.waitForFunction(()=>document.querySelector("video")?.readyState>=2);
  return page;
}
async function visible(page) {await page.waitForFunction(()=>getComputedStyle(document.querySelector(".detail-meter")).opacity === "1");}
async function faded(page) {await page.waitForFunction(()=>getComputedStyle(document.querySelector(".detail-meter")).visibility === "hidden",null,{timeout:4000});}
try {
  const context=await browser.newContext({viewport:{width:1280,height:900}}); page=await ready(context);
  check(await page.locator(".detail-meter").evaluate(n=>getComputedStyle(n).visibility==="hidden"),"Meter is hidden before input");
  const slider=page.getByRole("slider").first();
  await slider.focus();await page.keyboard.press("ArrowRight");await visible(page);
  check((await page.locator(".detail-meter-value").first().innerText())==="51%","Keyboard input immediately updates selected level");
  check(await slider.getAttribute("aria-valuetext")==="Selected: 51%","Native slider exposes selected value to assistive technology");
  check(await page.locator(".detail-meter").evaluate(n=>getComputedStyle(n).pointerEvents==="none" && n.getAttribute("aria-hidden")==="true"),"Overlay neither intercepts input nor duplicates screen-reader announcements");
  await page.waitForTimeout(1050);await page.keyboard.press("ArrowRight");await page.waitForTimeout(800);
  check(await page.locator(".detail-meter").evaluate(n=>n.classList.contains("is-visible")),"New input restarts the visible hold");await faded(page);
  await slider.press("Home");await visible(page);
  check(await page.locator(".detail-meter-fill").first().evaluate(n=>n.style.width)==="0%","Minimum is an empty level bar");
  const sliderBox=await slider.boundingBox(), padBox=await page.locator(".detail-pad").boundingBox();
  check(sliderBox.y<padBox.y,"Sliders precede the two-axis pad");
  await page.mouse.move(padBox.x+padBox.width*.8,padBox.y+padBox.height*.2);await page.mouse.down();await visible(page);
  await page.waitForTimeout(2100);
  check(await page.locator(".detail-meter").evaluate(n=>n.classList.contains("is-visible")),"Meter stays visible during a held drag");
  await page.mouse.move(padBox.x+padBox.width*.9,padBox.y+padBox.height*.1);await page.mouse.up();await visible(page);
  await page.screenshot({path:resolve(output,"desktop-input.png"),fullPage:true});await faded(page);
  await page.getByText("Detail presets",{exact:true}).click();
  await page.getByRole("button",{name:"Both detailed",exact:true}).click();await visible(page);
  check((await page.locator(".detail-meter-value").allTextContents()).every(v=>v==="100%"),"Presets reveal both selected levels");
  await page.getByRole("button",{name:"Reset to calibration",exact:true}).click();await visible(page);
  check((await page.locator(".detail-meter-value").allTextContents()).every(v=>v==="50%"),"Reset reveals restored levels");
  await page.waitForTimeout(1100);await page.locator("video").evaluate(v=>v.play());
  await page.waitForFunction(()=>document.querySelector(".caption-level-status").textContent.includes("acoustic 50%"),null,{timeout:8000});
  await page.locator("video").evaluate(v=>v.pause());
  await slider.press("End");await visible(page);
  check((await page.locator(".caption-level-status").innerText()).includes("acoustic 50%"),"Selected 100% does not relabel the currently displayed 50% caption");
  await page.locator("video").evaluate(v=>v.play());
  await page.waitForFunction(()=>document.querySelector(".caption-level-status").textContent.includes("acoustic 100%"),null,{timeout:8000});
  check(true,"Applied levels change only when the new caption is displayed");
  fallback=true;
  await page.waitForFunction(()=>document.querySelector(".caption-level-status").textContent.includes("Prepared caption"),null,{timeout:8000});
  await page.locator("video").evaluate(v=>v.pause());
  check((await page.locator(".caption-level-status").innerText()).includes("selected levels not applied"),"Fallback never claims selected controls were applied");
  await page.getByRole("button",{name:"Fullscreen",exact:true}).click();await slider.press("ArrowLeft");await visible(page);
  check(await page.locator(".detail-meter").evaluate(n=>document.fullscreenElement.contains(n)),"Meter remains inside fullscreen");
  await page.getByRole("button",{name:"Exit fullscreen",exact:true}).click();
  await page.emulateMedia({reducedMotion:"reduce"});await slider.press("ArrowLeft");
  check(await page.locator(".detail-meter").evaluate(n=>getComputedStyle(n).transitionDuration==="0s"),"Reduced motion disables the fade transition");await faded(page);
  await context.close();
  reset("ko");
  const mobile=await browser.newContext({viewport:{width:390,height:844},isMobile:true,hasTouch:true});page=await ready(mobile);
  await page.locator(".detail-pad").evaluate(n=>n.scrollIntoView({block:"end"}));
  await page.waitForTimeout(150);
  const touch=await mobile.newCDPSession(page),box=await page.locator(".detail-pad").boundingBox();
  await touch.send("Input.dispatchTouchEvent",{type:"touchStart",touchPoints:[{x:box.x+box.width*.8,y:box.y+box.height*.2}]});
  await visible(page);await page.waitForTimeout(1800);
  check(await page.locator(".detail-meter").evaluate(n=>n.classList.contains("is-visible")),"Touch hold keeps the meter visible");
  await touch.send("Input.dispatchTouchEvent",{type:"touchEnd",touchPoints:[]});await visible(page);
  check((await page.locator(".detail-meter-title").innerText())==="선택한 자막 상세 수준","Meter copy is localized in Korean");
  check(!await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),"Mobile layout has no horizontal overflow");
  const meter=await page.locator(".detail-meter").boundingBox(), caption=await page.locator(".study-caption").boundingBox();
  check(meter.y+meter.height<=caption.y,"Mobile meter stays clear of the caption band");
  await page.screenshot({path:resolve(output,"mobile-input-ko.png"),fullPage:true});await faded(page);
  const touchBox=await page.locator(".detail-pad").boundingBox();
  await touch.send("Input.dispatchTouchEvent",{type:"touchStart",touchPoints:[{x:touchBox.x+20,y:touchBox.y+20}]});
  await touch.send("Input.dispatchTouchEvent",{type:"touchCancel",touchPoints:[]});await faded(page);
  check(true,"Cancelled touch does not leave the meter stuck onscreen");
  await mobile.close();check(errors.length===0,errors.join("\n")||"No unhandled browser errors");
  await writeFile(resolve(output,"result.json"),JSON.stringify({scope:"UI fixture; not model or full-study QA",checks,output},null,2));
  console.log(JSON.stringify({checks,output}));
} catch(error) {
  await page?.screenshot({path:resolve(output,"failure.png"),fullPage:true}).catch(()=>{});
  await writeFile(resolve(output,"error.json"),JSON.stringify({error:error.stack,checks,errors},null,2));throw error;
} finally {await browser.close();server.close();}
