// Real dashboard over isolated old/new record fixtures, not seeded live data.
import {spawn} from "node:child_process";
import {mkdir,writeFile} from "node:fs/promises";
import {dirname,resolve} from "node:path";
import {fileURLToPath} from "node:url";
const repo=resolve(dirname(fileURLToPath(import.meta.url)),"../../../..");
const output=process.env.DASHBOARD_QA_OUTPUT||"/tmp/concap-dashboard-browser-qa";
await mkdir(output,{recursive:true});
const server=spawn(resolve(repo,".venv/bin/python"),["-u","-c",`
import json,socket,tempfile
from pathlib import Path
import uvicorn
from dpo.regen.dashboard import build_dashboard_app
from dpo.regen.tests.test_dashboard_data import make_records
root=Path(tempfile.mkdtemp(prefix='dashboard-browser-'))
legacy,viewing=make_records(root)
with socket.socket() as sock:
    sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
print(json.dumps({'url':f'http://127.0.0.1:{port}','root':str(root)}),flush=True)
uvicorn.run(build_dashboard_app(legacy,viewing),host='127.0.0.1',port=port,proxy_headers=False,log_level='warning')
`],{cwd:repo,env:{...process.env,PYTHONDONTWRITEBYTECODE:"1"}});
let log="",browser,page;
const info=await new Promise((accept,reject)=>{
  const timeout=setTimeout(()=>reject(Error("Dashboard fixture startup timed out: "+log)),20000);
  server.stdout.on("data",data=>{log+=data;for(const line of log.split("\n"))try{const parsed=JSON.parse(line);if(parsed.url){clearTimeout(timeout);accept(parsed);}}catch{}});
  server.stderr.on("data",data=>{log+=data;});server.once("exit",code=>{clearTimeout(timeout);reject(Error("Fixture exited "+code+log));});
});
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||"playwright-core");
const checks=[],errors=[];
const check=(value,label)=>{if(!value)throw Error(label);checks.push(label);};
const loaded=()=>page.waitForFunction(()=>document.querySelector("#refresh-status")?.textContent.startsWith("Updated"));
const shot=name=>page.screenshot({path:resolve(output,name),fullPage:true});
try{
  browser=await chromium.launch({executablePath:process.env.CHROMIUM_EXECUTABLE,args:["--no-sandbox"]});
  page=await browser.newPage({viewport:{width:1920,height:1080},acceptDownloads:true});
  page.on("pageerror",e=>errors.push(e.message));
  await page.goto(info.url);await loaded();
  const source=await page.request.get(info.url+"/api/overview").then(r=>r.json());
  check(source.metrics.participants===3&&source.metrics.survey_submissions===11&&source.metrics.response_values===14,"Fixture counts reconcile to submitted records");
  check(!await page.locator("#include-qa").isChecked(),"QA records excluded by default");
  check(await page.locator('#view-overview table[aria-label="Participants"] tbody tr').count()===3,"Participant table matches overview count");
  await shot("overview-desktop.png");
  await page.locator("#participant-search").fill("p-new");
  check(await page.locator('#view-overview table[aria-label="Participants"] tbody tr').count()===1,"Participant search filters collection");
  await page.locator("#participant-search").fill("");
  await page.locator("#language-filter").selectOption("ko");
  check(await page.locator('#view-overview table[aria-label="Participants"] tbody tr').count()===1,"Language filter respects recorded participant language");
  await page.locator("#language-filter").selectOption("");
  const qaResponse=page.waitForResponse(r=>r.url().includes("/api/overview?include_qa=true"));
  await page.locator("#include-qa").check();await qaResponse;await loaded();
  check((await page.locator("#view-overview").innerText()).includes("qa-browser"),"QA toggle explicitly reveals QA collection");
  const noQaResponse=page.waitForResponse(r=>r.url().includes("/api/overview?include_qa=false"));
  await page.locator("#include-qa").uncheck();await noQaResponse;await loaded();
  await page.locator('[data-view="responses"]').click();
  await page.locator("#filter-responsePhase").selectOption("2");
  check(await page.locator(".distribution-card").count()>0,"Phase 2 answer distributions render");
  const newRows=source.responses.filter(r=>r.phase===2);
  check(await page.locator('#view-responses table[aria-label="Submitted survey response values"] tbody tr').count()===newRows.length,"Response table matches selected phase values");
  check((await page.locator('.distribution-card[data-item-id="na_item"]').allTextContents()).some(t=>t.includes("N/A")),"N/A remains a distinct plotted category");
  check((await page.locator('.distribution-card[data-item-id="na_item"]').allTextContents()).some(t=>t.includes("Missing")),"Recorded null remains distinct from N/A");
  check(await page.locator('.distribution-card[data-item-id="useful"] .distribution-row').filter({hasText:/^0/}).count()>0,"Recorded zero is retained rather than treated as missing");
  check(await page.locator('img[src="x"]').count()===0&&!await page.evaluate(()=>window.dashboardXss),"Survey markup is displayed as text, never executed");
  check((await page.locator("#view-responses").innerText()).includes("window.dashboardXss"),"Full recorded free text remains inspectable");
  await shot("responses-desktop.png");
  const downloadWait=page.waitForEvent("download");await page.locator("#export-responses").click();
  const download=await downloadWait;await download.saveAs(resolve(output,"responses.csv"));
  check(download.suggestedFilename().endsWith(".csv"),"CSV export is downloadable");
  await page.locator('[data-view="inspector"]').click();
  await page.locator(".participant-option").filter({hasText:"p-new"}).click();
  await page.locator("#caption-search").waitFor();
  const detail=await page.request.get(info.url+"/api/participants/"+source.participants.find(p=>p.label==="p-new").id).then(r=>r.json());
  check((await page.locator("#inspector-detail").innerText()).includes("p-new"),"Participant drilldown loads the selected session");
  check(await page.locator('#inspector-detail svg').count()>0,"Interaction/timing plots render from selected records");
  check(await page.locator('#inspector-detail svg path[stroke-dasharray="6 4"]').count()>0,"Overlapping control signals retain different line patterns");
  check(detail.interactions.length===3&&detail.interactions[0].position_ms===null,"Missing playback positions remain missing in source");
  await page.locator("#caption-search").fill("not-a-caption-match");
  check((await page.locator('table[aria-label="Caption exposures and generation jobs"]').innerText()).includes("No matching records"),"Caption search handles empty results");
  await page.locator("#caption-search").fill("");
  await page.locator("#caption-kind").selectOption("job");
  check((await page.locator("#inspector-detail").innerText()).includes("not displayed"),"Unshown generation jobs are distinguished from exposures");
  await page.locator("#caption-kind").selectOption("");
  await page.getByText("Model timing distribution",{exact:true}).click();
  const timing=page.locator("section.panel").filter({has:page.getByRole("heading",{name:"Model generation time",exact:true})});
  check((await timing.locator(".timing-summary").innerText()).includes("100 ms"),"Timing median includes shown and unshown completed jobs once");
  check((await timing.locator(".timing-summary").innerText()).includes("0 ms"),"Zero-duration measurement is not replaced with missing");
  await page.locator("#control-time-basis").selectOption("recorded");
  check((await page.locator("#inspector-detail").innerText()).includes("since first"),"Recorded-time trend supports controls without playback positions");
  await page.locator("#event-search").fill("settings");
  check(await page.locator('table[aria-label="Chronological session events"] tbody tr').count()===3,"Interaction event search matches the three recorded control changes");
  await page.locator("#event-order").selectOption("desc");
  await shot("inspector-desktop.png");
  await page.locator("#event-search").fill("");
  const sessionDownload=page.waitForEvent("download");
  await page.getByRole("link",{name:"Export session JSON",exact:true}).click();
  const sessionFile=await sessionDownload;await sessionFile.saveAs(resolve(output,"session.json"));
  check(sessionFile.suggestedFilename().endsWith(".json"),"Selected participant JSON export is downloadable");
  await page.locator("#refresh-button").click();await loaded();await page.locator("#caption-search").waitFor();
  check((await page.locator("#inspector-detail").innerText()).includes("p-new"),"Refresh preserves selected participant");
  await page.reload();await loaded();await page.locator("#caption-search").waitFor();
  check((await page.locator("#view-title").innerText())==="Session inspector","View and selection survive page reload");
  const scheduled=page.waitForResponse(r=>r.url().includes("/api/overview?"),{timeout:20000});
  await page.locator("#auto-refresh").check();await scheduled;await loaded();await page.locator("#auto-refresh").uncheck();
  check(true,"Optional 15-second refresh fetches current collection data");
  await page.setViewportSize({width:1366,height:768});await shot("inspector-laptop.png");
  check(!await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),"Laptop layout has no page-level horizontal overflow");
  await page.setViewportSize({width:760,height:900});await shot("inspector-narrow.png");
  check(!await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),"Narrow layout stacks panels without page overflow");
  await page.route("**/api/overview?*",route=>route.fulfill({status:503,contentType:"application/json",body:'{"error":"fixture"}'}));
  await page.locator("#refresh-button").click();await page.waitForFunction(()=>document.querySelector("#refresh-status").textContent.includes("failed"));
  check(await page.locator("#source-warnings").isVisible(),"Failed refresh marks retained data stale");
  await page.unroute("**/api/overview?*");await page.locator("#refresh-button").click();await loaded();
  check(!await page.locator("#source-warnings").isVisible(),"Successful retry clears the stale state");
  check(errors.length===0,"No unhandled JavaScript errors");
  await writeFile(resolve(output,"result.json"),JSON.stringify({fixture:info,checks,errors,output},null,2));console.log(JSON.stringify({checks,output}));
}catch(error){
  await page?.screenshot({path:resolve(output,"failure.png"),fullPage:true}).catch(()=>{});
  await writeFile(resolve(output,"error.json"),JSON.stringify({error:error.stack,checks,errors,log},null,2));throw error;
}finally{await browser?.close();server.kill("SIGTERM");}
