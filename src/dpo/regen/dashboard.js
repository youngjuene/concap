/* Research dashboard. API strings are always rendered as text. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const PAGE_SIZE = 40;
  const STORAGE_KEY = "caption-research-dashboard-v1";
  const defaults = {view:"overview", query:"", language:"", status:"", includeQa:false, autoRefresh:false, participant:"", phase1Clip:"", responsePhase:"", responseScope:"", responseInstrument:"", responseFlow:"", responseCondition:"", responseCongruence:"", responseDepth:"", responseStrategy:"", responseVideo:"", responseItem:"", inspectorVideo:"", timeBasis:"auto", captionQuery:"", captionKind:"", eventQuery:"", eventOrder:"asc"};
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}"); } catch (_) { /* Storage is optional. */ }
  const state = {...defaults};
  if (saved && typeof saved === "object") for (const key of Object.keys(defaults)) {
    if (typeof saved[key] === typeof defaults[key]) state[key] = saved[key];
  }
  if (!["overview", "responses", "phase1", "inspector"].includes(state.view)) state.view = "overview";
  let overview = null, detail = null, overviewError = "", detailError = "";
  let overviewLoading = false, detailLoading = false, requestId = 0, detailRequestId = 0;
  let phase1 = null, phase1Error = "", phase1Loading = false, phase1RequestId = 0, phase1LoadedQuery = "";
  let phase1FilterTimer = null;
  let timer = null;
  let selectedMoment = null;
  const pages = {participants:0, groups:0, responses:0, browser:0, captions:0, events:0, interactions:0, detailResponses:0, activity:0, phase1Summaries:0, phase1Pairs:0, phase1Rates:0, phase1Transitions:0, phase1Relations:0, phase1CrossCounts:0, phase1Observations:0};

  function persist() { try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); } catch (_) {} }
  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }
  function append(parent, ...children) { for (const child of children.flat()) if (child) parent.append(child); return parent; }
  function empty(title, message) { return append(el("div", "empty-state"), el("strong", "", title), message ? el("p", "", message) : null); }
  const list = (value) => Array.isArray(value) ? value : [];
  const number = (value) => typeof value === "number" && Number.isFinite(value);
  const count = (value) => number(value) ? value.toLocaleString() : "—";
  const valueText = (value) => value === null || value === undefined ? "Missing" : typeof value === "object" ? JSON.stringify(value) : String(value);
  const seconds = (ms) => number(ms) ? `${(ms / 1000).toLocaleString(undefined,{maximumFractionDigits:2})} s` : "Not recorded";
  const duration = (ms) => number(ms) ? `${ms.toLocaleString(undefined,{maximumFractionDigits:1})} ms` : "Not recorded";
  const languageName = (lang) => ({en:"English",ko:"한국어"})[lang] || lang || "Unknown language";
  function timestamp(value, compact = false) {
    if (!value) return "Not recorded";
    const date = new Date(value);
    if (!Number.isFinite(date.getTime())) return String(value);
    const iso = date.toISOString();
    return compact ? `${iso.slice(5,10)} ${iso.slice(11,19)}` : `${iso.slice(0,10)} ${iso.slice(11,19)} UTC`;
  }
  function statusPill(status) {
    const names = {done:"Complete",complete:"Complete",submitted:"Submitted",in_progress:"In progress",watch:"Watching",missing:"Not started",prepared:"Prepared fallback",fallback:"Prepared fallback",generated:"Generated",job:"Generation job",unknown:"Unknown provenance","awaiting-calibration":"Waiting for Phase 1",ready:"Ready for viewing",break:"Between videos","video-survey":"Video questions",final:"Final questions"};
    const cls = ["done", "complete", "submitted"].includes(status) ? "done" : ["generated","fallback","job"].includes(status) ? status : status === "prepared" ? "fallback" : status && !["missing","unknown"].includes(status) ? "active" : "";
    return el("span", `pill ${cls}`, names[status] || String(status || "Not recorded").replaceAll("_", " "));
  }
  function button(label, onClick, className = "button") { const node = el("button", className, label); node.type = "button"; node.addEventListener("click", onClick); return node; }
  function exportLink(label, path) { const a = el("a", "button", label); a.href = `${path}?include_qa=${state.includeQa}`; a.download = ""; return a; }
  function titleBlock(title, description, action) { const row = el("div", "section-heading"); append(row, append(el("div"), el("h2", "", title), description ? el("p", "", description) : null), action); return row; }
  function panel(title, subtitle, action) {
    const box = el("section", "panel");
    append(box, append(el("header", "panel-header"), append(el("div"), el("h2", "", title), subtitle ? el("p", "panel-subtitle", subtitle) : null), action));
    return box;
  }
  function metadata(entries, label = "Details") {
    const details=append(el("details","metadata-details"),el("summary","",label)),values=el("dl");
    for(const [key,value] of entries)append(values,el("dt","",key),el("dd","",valueText(value)));
    details.append(values);return details;
  }
  function table(headers, rows, accessibleLabel) {
    const wrap = el("div", "table-scroll"), tab = el("table");
    if (accessibleLabel) tab.setAttribute("aria-label", accessibleLabel);
    const head = el("tr");
    for (const name of headers) { const cell = el("th", "", name); cell.scope = "col"; head.append(cell); }
    tab.append(append(el("thead"), head));
    const body = el("tbody");
    for (const values of rows) {
      const row = el("tr");
      for (const value of values) { const cell = el("td"); if (value instanceof Node) cell.append(value); else cell.textContent = valueText(value); row.append(cell); }
      body.append(row);
    }
    if (!rows.length) { const cell = el("td", "empty-state", "No matching records."); cell.colSpan = headers.length; body.append(append(el("tr"),cell)); }
    tab.append(body); wrap.append(tab); return wrap;
  }
  function paged(items, key, size = PAGE_SIZE) {
    pages[key] = Math.min(Math.max(0, pages[key] || 0), Math.max(0, Math.ceil(items.length / size) - 1));
    return items.slice(pages[key] * size, (pages[key] + 1) * size);
  }
  function pager(items, key, onChange, size = PAGE_SIZE) {
    paged(items,key,size);
    const start = items.length ? pages[key] * size + 1 : 0, end = Math.min(items.length, (pages[key]+1)*size);
    const row = el("div", "pager"), controls = el("div", "pager-controls");
    const previous = button("Previous", () => { pages[key]--; onChange(); }, "button small"); previous.disabled = pages[key] === 0;
    const next = button("Next", () => { pages[key]++; onChange(); }, "button small"); next.disabled = end >= items.length;
    previous.setAttribute("aria-label", `Previous ${key} page`); next.setAttribute("aria-label", `Next ${key} page`);
    append(controls, previous, next); append(row, el("span", "", `${count(start)}–${count(end)} of ${count(items.length)}`), controls); return row;
  }
  function options(select, values, selected, allLabel) {
    select.replaceChildren();
    const first = el("option", "", allLabel); first.value = ""; select.append(first);
    for (const [value, label] of values) { const option = el("option", "", label); option.value = value; select.append(option); }
    if (selected && !values.some(([value]) => String(value) === selected)) { const option = el("option", "", `${selected} (no records)`); option.value = selected; select.append(option); }
    select.value = selected;
  }
  function unique(values) { return [...new Set(values)].sort((a,b) => String(a).localeCompare(String(b),undefined,{numeric:true})); }
  function selectFilter(label, key, values, allLabel, rerender, className = "") {
    const field = el("label", className), select = el("select"); select.id = `filter-${key}`;
    options(select, values, state[key], allLabel);
    select.addEventListener("change", () => { state[key] = select.value; pages.groups=0; pages.responses=0; persist(); rerender(); });
    return append(field, el("span", "", label), select);
  }
  function filteredParticipants(applyLanguage = true) {
    return list(overview?.participants).filter((p) => {
      const queryMatch = `${p.label || ""} ${p.id || ""}`.toLowerCase().includes(state.query.toLowerCase().trim());
      const complete = ["done","complete","submitted"].includes(p.phase2_status);
      const phase1 = ["done","complete","submitted"].includes(p.phase1_status);
      const statusMatch = !state.status || state.status === "complete" && complete || state.status === "active" && !complete || state.status === "phase1" && phase1 || state.status === "no_phase2" && (!p.phase2_status || p.phase2_status === "missing");
      return queryMatch && (!applyLanguage || !state.language || (p.language || "unknown") === state.language) && statusMatch;
    }).sort((a,b) => String(b.last_activity || "").localeCompare(String(a.last_activity || "")) || String(a.label || a.id).localeCompare(String(b.label || b.id)));
  }
  function includedResponses() { const ids = new Set(filteredParticipants(false).map(p => p.id)); return list(overview?.responses).filter(r => ids.has(r.participant) && (!state.language || (r.language || "unknown") === state.language)); }
  function participantButton(id, label) { return button(label || id, () => selectParticipant(id), "table-link mono"); }
  function videoProgress(p) { return p.videos_total == null ? count(p.videos_completed) : `${count(p.videos_completed)} / ${count(p.videos_total)}`; }
  function participantRows(participants) {
    return participants.map(p => [append(el("div", "id-cell"), participantButton(p.id,p.label), p.qa ? el("span", "pill qa", "QA") : null),languageName(p.language),statusPill(p.phase1_status),statusPill(p.phase2_status),count(p.phase1_surveys),count(p.phase2_surveys),videoProgress(p),count(p.control_changes),el("span","mono nowrap",timestamp(p.last_activity,true))]);
  }
  function legend() { return append(el("div","legend"), append(el("span","legend-item"),el("i","swatch"),el("span","","Phase 1")),append(el("span","legend-item"),el("i","swatch teal"),el("span","","Phase 2"))); }
  function svgElement(tag, attrs = {}, text) { const node = document.createElementNS("http://www.w3.org/2000/svg", tag); for (const [key,value] of Object.entries(attrs)) node.setAttribute(key,String(value)); if (text !== undefined) node.textContent = text; return node; }
  function chart(title, width = 760, height = 215) { const svg = svgElement("svg",{viewBox:`0 0 ${width} ${height}`,role:"img",class:"chart"}); svg.append(svgElement("title",{},title)); return svg; }
  function activityChart(activity) {
    if (!activity.length) return empty("No dated activity", "Activity appears when records include a collection date.");
    const displayed = activity.slice(-30), svg = chart("Daily recorded activity, Phase 1 and Phase 2. Exact counts are available below.");
    const width=760, left=36, right=12, top=10, bottom=35, height=215, plotHeight=height-top-bottom;
    const observedMaximum=Math.max(1,...displayed.flatMap(d=>[number(d.phase1)?d.phase1:0,number(d.phase2)?d.phase2:0])),tickStep=Math.max(1,Math.ceil(observedMaximum/4)),maximum=tickStep*4;
    for (let tick=0;tick<=4;tick++) { const y=top+plotHeight*(1-tick/4); svg.append(svgElement("line",{x1:left,y1:y,x2:width-right,y2:y,class:"grid-line"}),svgElement("text",{x:left-9,y:y+3,"text-anchor":"end"},String(tick*tickStep))); }
    const step=(width-left-right)/displayed.length, barWidth=Math.min(17,step*.28);
    displayed.forEach((day,i) => {
      const center=left+(i+.5)*step;
      for (const [phase,offset] of [["phase1",-barWidth-1],["phase2",1]]) {
        if (!number(day[phase])) continue;
        const h=day[phase]/maximum*plotHeight;
        const rect=svgElement("rect",{x:center+offset,y:top+plotHeight-h,width:barWidth,height:h,rx:1,class:phase}); rect.append(svgElement("title",{},`${day.date}, ${phase === "phase1"?"Phase 1":"Phase 2"}: ${day[phase]}`)); svg.append(rect);
      }
      if (i===0||i===displayed.length-1||i%Math.max(1,Math.ceil(displayed.length/7))===0) svg.append(svgElement("text",{x:center,y:height-13,"text-anchor":"middle"},String(day.date).slice(5)));
    }); return svg;
  }
  function renderOverview() {
    const root=$("view-overview"); root.replaceChildren();
    const participants=filteredParticipants(), responses=includedResponses();
    append(root,titleBlock("Collection at a glance", "Progress across calibration and viewing sessions.",el("p","hint",`${state.includeQa ? "QA sessions included" : "QA sessions excluded"} · ${count(participants.length)} matching participants`)));
    if (!overview) { root.append(empty(overviewError ? "Collection unavailable" : "Loading collection…",overviewError || "Reading submitted records.")); return; }
    const phase1=participants.filter(p=>["done","complete","submitted"].includes(p.phase1_status)).length, phase2=participants.filter(p=>["done","complete","submitted"].includes(p.phase2_status)).length;
    const metrics=el("div","metrics");
    for (const [label,value,note,cls] of [["Participants",participants.length,"Matching current filters",""],["Phase 1 complete",phase1,`${count(participants.length-phase1)} not complete`,"blue"],["Phase 2 complete",phase2,`${count(participants.length-phase2)} not complete`,"teal"],["Submitted answers",responses.length,"Recorded response values",""],["Control changes",participants.reduce((n,p)=>n+(number(p.control_changes)?p.control_changes:0),0),"Matching participants",""],["Generation median",overview.metrics?.generation_median_ms,"ms · whole collection¹",""]]) {
      append(metrics,append(el("article",`metric ${cls}`),el("div","metric-label",label),el("div","metric-value",count(value)),el("div","metric-note",note)));
    }
    root.append(metrics);
    const top=el("div","overview-top"), activity=panel("Survey submissions by day (UTC)","Latest 30 dates with submitted surveys",legend());
    const activityBody=append(el("div","panel-body"),activityChart(list(overview.activity)));
    activityBody.append(el("p","chart-note","Collection-wide counts follow the QA setting. Participant, progress and language filters apply to the metrics and participant list, not this activity chart."));
    const activityDetails=append(el("details","data-details"),el("summary","","Show daily counts"));
    append(activityDetails,table(["Date (UTC)","Phase 1","Phase 2"],paged(list(overview.activity),"activity").map(d=>[d.date,count(d.phase1),count(d.phase2)]),"Collection activity data"),pager(list(overview.activity),"activity",renderOverview));
    activityBody.append(activityDetails); activity.append(activityBody);
    const source=panel("Caption collection","Whole collection · follows QA setting only");
    const facts=el("div","facts");
    for (const [label,value] of [["Generated caption exposures",overview.metrics?.generated_exposures],["Prepared fallback exposures",overview.metrics?.fallback_exposures],["Survey submissions",overview.metrics?.survey_submissions],["Phase 2 started",overview.metrics?.phase2_started]]) append(facts,append(el("div","fact"),el("span","",label),el("strong","",count(value))));
    append(source,facts,append(el("div","panel-body"),el("p","chart-note","¹ Generation median uses completed model jobs only. It does not measure input-to-visible latency. Missing measurements are shown as —."))); append(top,activity,source); root.append(top);
    const participantPanel=panel("Participants",`${count(participants.length)} sessions · select a participant to inspect recorded data`);
    append(participantPanel,table(["Participant","Language","Phase 1","Phase 2","P1 surveys","P2 surveys","Videos complete","Control changes","Last activity (UTC)"],participantRows(paged(participants,"participants")),"Participants"),pager(participants,"participants",renderOverview));
    if (!list(overview.participants).length) participantPanel.append(empty("No collected participants", "Submitted sessions will appear here. QA sessions are excluded unless enabled."));
    root.append(participantPanel);
  }
  // Phase 1 observational analysis. Derived values and denominators come from the server.
  function phase1Query() {
    return new URLSearchParams({include_qa:String(state.includeQa), participant:state.query,
      language:state.language, status:state.status, clip_id:state.phase1Clip}).toString();
  }
  const phase1Text = value => value === null || value === undefined || value === "" ? "NA" : valueText(value);
  const phase1Percent = value => number(value) ? `${(100*value).toFixed(1)}%` : "NA";
  const phase1Delta = value => number(value) ? `${value>0?"+":""}${value.toFixed(1)} pp` : "NA";
  const phase1Change = value => value === true || value === 1 ? "Changed" : value === false || value === 0 ? "Unchanged" : "NA";
  const phase1Rate = (value, denominator) => number(denominator) && denominator>0 ? phase1Percent(value) : "NA";
  const phase1Relation = value => value === 1 ? "1 · coded correspondence" : value === 0 ? "0 · coded non-correspondence" : "NA";
  const phase1Transition = (first, second) => `${phase1Text(first)} → ${phase1Text(second)}`;
  function phase1Scope(row) {
    return append(el("div","phase1-context"),el("strong","",phase1Text(row.clip_id)),
      el("small","",`AV assignment: ${phase1Text(row.av_assigned_condition)} · Depth: ${phase1Text(row.description_depth)} · Strategy: ${phase1Text(row.caption_strategy)} · Menu: ${phase1Change(row.menu_changed)}`),
      metadata([["Instrument",phase1Text(row.instrument_hash)],["Codebook version",phase1Text(row.codebook_version)],
        ["Codebook hash",phase1Text(row.codebook_hash)],["Menu change",phase1Change(row.menu_changed)]],"Analysis group"));
  }
  function phase1Export(label, path) {
    const link=el("a","button",label);link.href=`${path}?${phase1Query()}`;link.download="";
    link.title="Export this analysis cohort, including participant, language, progress, QA and clip filters.";
    return link;
  }
  function phase1Table(title, description, headers, rows, key) {
    const box=panel(title,description);
    append(box,table(headers,paged(rows,key),title),pager(rows,key,renderPhase1));
    return box;
  }
  function phase1SummaryRows() {
    return list(phase1?.summaries).flatMap(group=>[
      ["Type change",group.type_changed,"rate"], ["Family change",group.family_changed,"rate"],
      ["Mask area change",group.area_delta_pp,"mean"],
    ].map(([label,metric,kind])=>[phase1Scope(group),label,
      kind==="rate"?phase1Rate(metric?.rate,metric?.valid_pairs):phase1Delta(metric?.mean),
      phase1Text(metric?.total_pairs),phase1Text(metric?.valid_pairs),phase1Text(metric?.na_pairs),
      kind==="rate"?phase1Text(metric?.changed_pairs):"Not applicable"]));
  }
  function renderPhase1() {
    const root=$("view-phase1");root.replaceChildren();
    const exports=append(el("div","phase1-exports"),
      phase1Export("Observation CSV","/api/export/phase1-observations.csv"),
      phase1Export("Paired CSV","/api/export/phase1-pairs.csv"));
    append(root,titleBlock("Before and after observations","Selected visual regions and sound choices across the two viewings.",exports));
    const filters=el("div","filter-bar");
    filters.append(selectFilter("Short clip","phase1Clip",list(phase1?.available_clips).map(value=>[value,value]),
      "All short clips",()=>{for(const key of Object.keys(pages))if(key.startsWith("phase1"))pages[key]=0;fetchPhase1();}));
    filters.append(el("p","phase1-cohort-note","The participant search, language, progress and QA filters above apply to every table and both CSV exports. The clip filter applies to this view."));
    root.append(filters);
    root.append(el("p","phase1-interpretation","These are observed selection trends, not attention intensity or a causal estimate of caption effects. NA means a measurement, declared mapping or comparable option order is unavailable; it is never filled with zero. Historical records without saved mask metrics remain NA."));
    if(phase1Error)root.append(el("p","inspector-error",`${phase1?"Showing the previous result for these same filters. ":""}${phase1Error}`));
    if(!phase1){root.append(empty(phase1Error?"Phase 1 analysis unavailable":"Loading Phase 1 analysis…",phase1Error?"Use Refresh to retry.":"Reading paired observations and their recorded provenance."));return;}
    if(phase1Loading)root.append(el("p","loading-inline","Refreshing this analysis cohort…"));
    for(const warning of list(phase1.warnings))root.append(el("p","phase1-warning",valueText(warning)));
    root.append(metadata([["Grouping","Clip, assigned AV condition, description depth, caption strategy, instrument, codebook version/hash and menu change stay separate."],
      ["Mask area","The selected mask's full foreground area divided by the frame area. Area difference is second minus first in percentage points; it is not a gaze-distance measure."],
      ["Sound changes","Type change and family change use separate valid paired denominators. Different option menus are allowed and explicitly marked."],
      ["Assigned AV condition","The recorded experiment assignment. It is separate from the selection relation code."],
      ["Selection relation","A declared visual–sound relation code. If no relation table or relevant entry was recorded, the value is NA."],
      ["Returned records",`${list(phase1.observations).length} observations · ${list(phase1.pairs).length} paired rows`]],"Definitions and grouping"));
    root.append(phase1Table("Changes by analysis group","Each metric shows its own total, valid and NA pair counts. Rates use valid pairs only.",
      ["Analysis group","Metric","Estimate","Total pairs","Valid pairs","NA pairs","Changed pairs"],phase1SummaryRows(),"phase1Summaries"));
    const pairs=list(phase1.pairs).map(row=>[participantButton(row.participant),phase1Scope(row),
      phase1Percent(row.selected_mask_area_ratio_1),phase1Percent(row.selected_mask_area_ratio_2),phase1Delta(row.area_delta_pp),
      phase1Transition(row.sound_type_1,row.sound_type_2),phase1Change(row.type_changed),
      phase1Transition(row.sound_family_1,row.sound_family_2),phase1Change(row.family_changed),
      phase1Change(row.menu_changed),phase1Change(row.order_changed),phase1Text(row.av_assigned_condition),
      `${phase1Relation(row.av_selection_relation_1)} → ${phase1Relation(row.av_selection_relation_2)}`,
      metadata([["Status",phase1Text(row.status)],["Issues",list(row.issues)],
        ["Session",phase1Text(row.session_id)],["Clip index",phase1Text(row.clip_index)],
        ["Frame 1 → 2",phase1Transition(row.frame_id_1,row.frame_id_2)],["Frame change",phase1Change(row.frame_changed)],
        ["Mask 1 → 2",phase1Transition(row.selected_mask_id_1,row.selected_mask_id_2)]],"Pair details")]);
    root.append(phase1Table("Paired observations","Area, sound type, sound family, menu and order changes remain separate. Order is compared only when the raw option ID sets match. Incomplete pairs are retained with NA values.",
      ["Participant","Analysis group","Mask area 1","Mask area 2","Area Δ (pp)","Sound type 1 → 2","Type change",
        "Sound family 1 → 2","Family change","Menu change","Order change","Assigned AV condition","Selection relation 1 → 2","Details"],pairs,"phase1Pairs"));
    const relations=list(phase1.summaries).flatMap(group=>[1,2].map(round=>{
      const metric=group[`av_selection_relation_${round}`];
      return [phase1Scope(group),round,phase1Rate(metric?.rate,metric?.valid_observations),
        phase1Text(metric?.total_observations),phase1Text(metric?.valid_observations),phase1Text(metric?.na_observations),phase1Text(metric?.relation_sum)];
    }));
    root.append(phase1Table("Selection relation by round","The correspondence rate uses observations with a defined relation code only. Assigned AV condition remains a separate grouping field.",
      ["Analysis group","Round","Coded correspondence rate","Total observations","Valid observations","NA observations","Coded correspondences"],relations,"phase1Relations"));
    const rates=list(phase1.selection_rates).map(row=>[phase1Scope(row),row.kind==="family"?"Sound family":"Sound type",
      phase1Text(row.round),phase1Text(row.code),phase1Text(row.numerator),phase1Text(row.denominator),phase1Rate(row.rate,row.denominator),phase1Text(row.unmapped_selections)]);
    root.append(phase1Table("Selection rates among offered options","Eligible responses were shown that option and supplied a valid answer. A zero denominator produces NA; option rates need not sum to100%.",
      ["Analysis group","Kind","Round","Option code","Selected responses","Eligible responses","Conditional selection rate","Unmapped selections"],rates,"phase1Rates"));
    const transitions=list(phase1.transitions).filter(row=>["type","family"].includes(row.kind)).map(row=>[phase1Scope(row),row.kind==="family"?"Sound family":"Sound type",
      phase1Text(row.from_code),phase1Text(row.to_code),phase1Text(row.count)]);
    root.append(phase1Table("Sound transitions","Counts include only pairs with the relevant type or family defined at both viewings; menu changes remain separate groups.",
      ["Analysis group","Kind","First selection","Second selection","Valid paired count"],transitions,"phase1Transitions"));
    const crossCounts=list(phase1.transitions).filter(row=>row.kind==="visual_sound").map(row=>[
      phase1Scope(row),phase1Text(row.round),phase1Text(row.from_code),phase1Text(row.to_code),phase1Text(row.count)]);
    root.append(phase1Table("Visual–sound selection cross-counts · 선택 교차빈도","Within each round, counts combine the selected visual-mask and sound-type codes. An explicit relation table is not needed for these descriptive counts.",
      ["Analysis group","Round","Selected visual mask","Selected sound type","Observation count"],crossCounts,"phase1CrossCounts"));
    const observations=list(phase1.observations).map(row=>[participantButton(row.participant),phase1Scope(row),phase1Text(row.round),
      phase1Percent(row.selected_mask_area_ratio),phase1Text(row.selected_mask_id),phase1Text(row.sound_type),phase1Text(row.sound_family),
      phase1Text(row.av_assigned_condition),phase1Relation(row.av_selection_relation),metadata([
        ["Status",phase1Text(row.status)],["Visual status",phase1Text(row.visual_status)],["Issues",list(row.issues)],
        ["Session",phase1Text(row.session_id)],["Clip index",phase1Text(row.clip_index)],["Frame",phase1Text(row.frame_id)],
        ["Visual label",phase1Text(row.visual_label)],["Raw sound selection",phase1Text(row.sound_raw)],
        ["Sound label",phase1Text(row.sound_label)],["Offered options",list(row.offered_options)],
        ["Unmapped offered sounds",list(row.unmapped_offered_sound_types)],["Candidate masks",list(row.candidate_masks)]],"Observation details")]);
    root.append(phase1Table("Individual observations","Raw selection values, offered options and candidate masks remain inspectable beside derived values.",
      ["Participant","Analysis group","Round","Mask area","Mask ID","Sound type","Sound family","Assigned AV condition","Selection relation","Details"],observations,"phase1Observations"));
  }
  async function fetchPhase1() {
    const query=phase1Query(),id=++phase1RequestId;
    if(phase1LoadedQuery!==query)phase1=null;
    phase1Loading=true;phase1Error="";
    if(state.view==="phase1")renderPhase1();renderStatus();
    try{
      const payload=await fetchJson(`/api/phase1-analysis?${query}`);
      if(id!==phase1RequestId||query!==phase1Query())return;
      if(payload?.schema!=="dpo.phase1-analysis/v1"||!["observations","pairs","summaries","selection_rates","transitions"].every(key=>Array.isArray(payload[key])))
        throw new Error("Phase 1 analysis response is incomplete.");
      phase1=payload;phase1LoadedQuery=query;
    }catch(error){if(id===phase1RequestId&&query===phase1Query())phase1Error=error.name==="AbortError"?"Analysis request timed out. Use Refresh to retry.":error.message||"Could not read Phase 1 analysis.";}
    finally{if(id===phase1RequestId){phase1Loading=false;if(state.view==="phase1")renderPhase1();renderStatus();}}
  }
  function responseFilterValue(row,key) {
    if (key === "video_id") return row.video_id == null ? "__none__" : String(row.video_id);
    if (key === "audiovisual_congruence" || key === "description_depth") return String(row.stimulus_assignment?.[key] ?? "unknown");
    return String(row[key] ?? "unknown");
  }
  const responseConditionFilters = [
    ["Audiovisual congruence","responseCongruence","audiovisual_congruence","All congruence levels"],
    ["Description depth","responseDepth","description_depth","All description depths"],
    ["Caption strategy","responseStrategy","caption_strategy","All caption strategies"],
  ];
  const responseFilters = [
    ["Phase","responsePhase","phase","All phases"], ["Scope / page","responseScope","scope","All scopes"],
    ["Instrument version","responseInstrument","instrument","All instruments"], ["Flow version","responseFlow","flow_version","All flows"],
    ["Viewing condition","responseCondition","condition","All conditions"], ...responseConditionFilters,
    ["Video","responseVideo","video_id","All videos"], ["Survey item","responseItem","item_id","All items"],
  ];
  function filteredResponses() {
    return includedResponses().filter(r => responseFilters.every(([,key,field])=>!state[key] || state[key]===responseFilterValue(r,field)));
  }
  function responseGroups(rows) {
    const groups=new Map();
    for (const row of rows) {
      const key=JSON.stringify([row.phase,row.scope,row.video_id??null,row.instrument||"unknown",row.flow_version||"unknown",row.condition??null,typeof row.stimulus_assignment==="string"?row.stimulus_assignment:null,...responseConditionFilters.map(([, ,field])=>responseFilterValue(row,field)),row.language||"unknown",row.item_id,row.item_text,row.wording_available,row.type,row.points??null,row.min??null,row.max??null,list(row.labels),list(row.options),list(row.option_labels)]);
      if (!groups.has(key)) groups.set(key,{row,rows:[]}); groups.get(key).rows.push(row);
    }
    return [...groups.values()].sort((a,b)=>Number(a.row.phase)-Number(b.row.phase)||String(a.row.scope).localeCompare(String(b.row.scope))||String(a.row.item_id).localeCompare(String(b.row.item_id),undefined,{numeric:true}));
  }
  function canonicalValue(value,type) {
    if (value===null||value===undefined||value==="") return {key:"missing",label:"Missing",missing:true};
    if (typeof value === "string" && /^(n\/?a|not[ _-]?applicable)$/i.test(value.trim())) return {key:"na",label:"N/A",na:true};
    if (type==="rating" && (typeof value==="number" || typeof value==="string" && value.trim()!=="" && Number.isFinite(Number(value)))) return {key:`rating:${Number(value)}`,label:String(Number(value)),numeric:Number(value)};
    return {key:`value:${JSON.stringify(value)}`,label:valueText(value)};
  }
  function distribution(group) {
    const sample=group.row, card=el("article","distribution-card");
    card.dataset.itemId=sample.item_id;
    append(card,el("div","distribution-context",`PHASE ${sample.phase} · ${sample.scope || "Unknown scope"} · ${sample.video_id ?? "Chapter"} · ${languageName(sample.language)}${sample.condition ? ` · ${sample.condition}` : ""}`),el("h3","",sample.wording_available ? sample.item_text : `Recorded item: ${sample.item_id}`));
    const provenance=metadata([["Item",sample.item_id],["Instrument",sample.instrument||"unknown"],["Flow version",sample.flow_version||"unknown"],["Condition",sample.condition||"Not recorded"],...responseConditionFilters.map(([label,,field])=>[label,responseFilterValue(sample,field)]),["Scale points",sample.points??"Not recorded"],["Denominator",`All ${group.rows.length} recorded answers, including N/A and recorded missing values. Unsubmitted items are excluded.`]],"Question details");
    if (!sample.wording_available) card.append(el("p","distribution-meta","Historical wording unavailable."));
    const participantCount=new Set(group.rows.map(r=>r.participant)).size;
    card.append(el("p","distribution-meta",`${count(group.rows.length)} recorded answers · ${count(participantCount)} participants`));
    if (sample.type==="text") {
      const answered=group.rows.filter(r=>r.value!==null && r.value!==undefined && r.value!=="").length;
      card.append(el("p","muted",`${count(answered)} written responses`),provenance);
      return card;
    }
    const bins=new Map();
    const addOption=(value,label) => { const item=canonicalValue(value,sample.type); if (!bins.has(item.key)) bins.set(item.key,{...item,label:label??item.label,count:0}); };
    if (sample.type==="rating" && Number.isInteger(sample.points) && sample.points>0 && sample.points<=100) {
      const minimum=Number.isInteger(sample.min)?sample.min:1;
      const maximum=Number.isInteger(sample.max)?sample.max:minimum+sample.points-1;
      for (let point=minimum;point<=maximum;point++) {
        const label=list(sample.labels)[point-minimum];
        addOption(point,label?`${point} · ${label}`:String(point));
      }
    }
    for (const [index,option] of list(sample.options).entries()) {
      if (option && typeof option==="object") addOption(option.value??option.id??option.label, option.label??option.text);
      else addOption(option,list(sample.option_labels)[index]);
    }
    for (const row of group.rows) {
      const values=sample.type==="multi" && Array.isArray(row.value)?[...new Set(row.value)]:[row.value];
      for (const value of values) {
        const item=canonicalValue(value,sample.type);
        if (!bins.has(item.key)) bins.set(item.key,{...item,count:0});
        bins.get(item.key).count++;
      }
    }
    if(sample.type==="multi")card.append(el("p","distribution-meta","Counts show answers selecting each option. One answer can select multiple options, so percentages can total more than 100%."));
    const ordered=[...bins.values()].sort((a,b)=>Number(!!a.missing)-Number(!!b.missing)||Number(!!a.na)-Number(!!b.na)||(sample.type==="rating" ? a.numeric-b.numeric : 0));
    for (const bin of ordered.slice(0,40)) {
      const percentage=group.rows.length ? bin.count/group.rows.length*100 : 0;
      const fill=el("div",`bar-fill ${Number(sample.phase)===2 ? "teal" : ""} ${bin.na||bin.missing ? "na" : ""}`); fill.style.width=`${percentage}%`;
      const row=append(el("div","distribution-row"),el("span","distribution-label",bin.label),append(el("div","bar-track"),fill),el("span","bar-count",`${bin.count} · ${Math.round(percentage)}%`));
      row.setAttribute("aria-label",`${bin.label}: ${bin.count} of ${group.rows.length} recorded answers (${percentage.toFixed(1)} percent)`); card.append(row);
    }
    if(sample.type==="unknown"||sample.type==="rating"&&!number(sample.points))card.append(el("p","distribution-meta","Scale not recorded; observed values shown."));
    if(ordered.length>40)card.append(el("p","distribution-meta",`First 40 of ${ordered.length} categories; full values in the table.`));
    card.append(provenance);
    return card;
  }
  function responseTable(rows,key,rerender) {
    const labels=new Map(list(overview?.participants).map(p=>[p.id,p.label]));
    const box=panel("Submitted responses",`${count(rows.length)} answer values · full text and original recorded values`);
    const values=paged(rows,key).map(r=>[participantButton(r.participant,labels.get(r.participant)),el("div","text-cell",r.wording_available?r.item_text:`${r.item_id} · historical wording unavailable`),el("div","text-cell",valueText(r.value)),el("span","",`Phase ${r.phase} · ${r.scope} · ${r.video_id??"Chapter"}${r.condition?` · ${r.condition}`:""}`),el("span","mono nowrap",timestamp(r.submitted_at)),metadata([["Item",r.item_id],["Language",languageName(r.language)],["Instrument",r.instrument||"unknown"],["Flow version",r.flow_version||"unknown"],["Viewing ID",r.view_id||"Not recorded"],...responseConditionFilters.map(([label,,field])=>[label,responseFilterValue(r,field)])],"Record details")]);
    append(box,table(["Participant","Question","Recorded response","Context","Submitted (UTC)","Details"],values,"Submitted survey response values"),pager(rows,key,rerender)); return box;
  }
  function renderResponses() {
    const root=$("view-responses"); root.replaceChildren();
    if (!overview) { root.append(empty(overviewError ? "Collection unavailable" : "Loading responses…",overviewError)); return; }
    const base=includedResponses(), filters=el("div","filter-bar");
    for (const [label,key,field,all] of responseFilters) {
      const values=unique(base.map(r=>responseFilterValue(r,field))).map(value=>[value,field==="phase"?`Phase ${value}`:value==="__none__"?"No video / chapter-level":value]);
      filters.append(selectFilter(label,key,values,all,renderResponses,field==="instrument"?"filter-wide":""));
    }
    append(root,filters);
    const rows=filteredResponses(), groups=responseGroups(rows);
    append(root,titleBlock("Response distributions",`${count(rows.length)} answers across ${count(groups.length)} separate item groups.`,button("Reset response filters",()=>{ for (const [,key] of responseFilters) state[key]=""; pages.groups=0; pages.responses=0; persist(); renderResponses(); },"button small")),metadata([["Grouping","Phase, scope, video, instrument, flow, viewing condition, audiovisual congruence, description depth, caption strategy, wording, scale and language remain separate."],["Counting","N/A and recorded missing values remain separate categories; unsubmitted items are not counted."],["CSV export","All submitted responses allowed by the QA setting, independent of these filters."]],"Grouping and export notes"));
    if (groups.length) { const grid=el("div","distribution-grid"); for (const group of paged(groups,"groups",12)) grid.append(distribution(group)); root.append(grid); const navigation=pager(groups,"groups",renderResponses,12); navigation.classList.add("group-pager"); root.append(navigation); }
    else root.append(empty("No matching submitted answers", "Adjust the response or participant filters to inspect other records."));
    root.append(responseTable(rows,"responses",renderResponses));
  }
  function videoKey(row) { return JSON.stringify([row.phase??2,row.video_id??null]); }
  function videoLabel(key) { try { const [phase,video]=JSON.parse(key); return `Phase ${phase} · ${video??"No recorded video"}`; } catch (_) { return key; } }
  function inspectorRecords(key) { return list(detail?.[key]).filter(row=>!state.inspectorVideo||videoKey(row)===state.inspectorVideo); }
  function momentKey(type,row) {
    return JSON.stringify([type,row.phase,row.video_id,row.at,row.position_ms,row.start_ms,row.end_ms,row.job_id,row.revision,row.kind,row.texture,row.context]);
  }
  function isSelectedMoment(type,row) {
    if(!selectedMoment)return false;
    if(selectedMoment.key===momentKey(type,row))return true;
    const linked=(selectedMoment.type==="control"&&type==="event"&&row.kind==="settings") ||
      (selectedMoment.type==="event"&&selectedMoment.row.kind==="settings"&&type==="control");
    return linked&&!!row.at&&row.at===selectedMoment.row.at&&videoKey(row)===videoKey(selectedMoment.row);
  }
  function momentPosition(moment=selectedMoment) {
    if(!moment)return null;
    return moment.type==="caption" ? moment.row.start_ms : moment.row.position_ms;
  }
  function momentWindow() {
    const position=momentPosition();
    if(!number(position))return null;
    if(selectedMoment.type==="caption")return [position,number(selectedMoment.row.end_ms)&&selectedMoment.row.end_ms>position?selectedMoment.row.end_ms:position+1];
    return [Math.max(0,position-5000),position+5000];
  }
  function relatedRecords(key) {
    const rows=inspectorRecords(key);
    if(!selectedMoment)return rows;
    const window=momentWindow(),at=selectedMoment.type==="caption"?NaN:Date.parse(selectedMoment.row.at);
    // Job creation timestamps are never substitutes for caption exposure time.
    if(key==="captions")return window ? rows.filter(r=>number(r.start_ms)&&number(r.end_ms)&&r.end_ms>window[0]&&r.start_ms<window[1]) : rows;
    if(key==="events") {
      if(window)return rows.filter(r=>number(r.position_ms)&&r.position_ms>=window[0]&&r.position_ms<=window[1]);
      if(Number.isFinite(at))return rows.filter(r=>Number.isFinite(Date.parse(r.at))&&Math.abs(Date.parse(r.at)-at)<=5000);
    }
    return rows;
  }
  function selectMoment(type,row) {
    selectedMoment={type,row,key:momentKey(type,row)};
    state.inspectorVideo=videoKey(row);pages.captions=0;pages.events=0;
    state.captionQuery="";state.captionKind="";state.eventQuery="";
    persist();renderInspector();
    $("moment-summary")?.focus({preventScroll:true});
  }
  function momentButton(type,row,label) {
    const control=button(label,()=>selectMoment(type,row),`table-link mono ${type}-moment`);
    control.dataset.momentType=type;
    const pos=momentPosition({type,row});if(number(pos))control.dataset.positionMs=String(pos);
    control.setAttribute("aria-pressed",String(isSelectedMoment(type,row)));
    control.setAttribute("aria-label",`Inspect ${type}: ${label} · ${videoLabel(videoKey(row))}`);
    return control;
  }
  function highlightRows(wrap,rows,type) {
    wrap.querySelectorAll("tbody tr").forEach((tr,index)=>{
      if(rows[index]&&isSelectedMoment(type,rows[index]))tr.classList.add("record-focus");
    });return wrap;
  }
  function momentSummary() {
    const box=panel("Selected moment","Link the timeline, captions and event records.");box.id="moment-summary";box.tabIndex=-1;
    const body=el("div","panel-body");box.append(body);
    if(!selectedMoment){body.append(el("p","moment-prompt","Select a timeline point, caption window or event to inspect nearby records together."));return box;}
    const {row,type}=selectedMoment,window=momentWindow();
    body.append(el("strong","moment-kind",type==="control"?"Control change":type==="caption"?captionLabel(row):row.kind||"Event"));
    if(window)body.append(el("p","moment-time",type==="caption"?`${seconds(row.start_ms)} → ${seconds(row.end_ms)}`:seconds(row.position_ms)));
    else body.append(el("p","moment-time",timestamp(row.at)));
    if(type==="control")body.append(el("p","",`Selected: acoustic ${number(row.texture)?Math.round(row.texture*100)+"%":"not recorded"} · source/scene ${number(row.context)?Math.round(row.context*100)+"%":"not recorded"}`));
    if(type==="caption")body.append(el("p","moment-sentence",row.text||"No caption text recorded."));
    body.append(el("p","chart-note",window
      ? `${type==="caption"?"Caption window":"Within 5 seconds of this playback position"}. Only records with playback positions can be aligned.`
      : "Playback position not recorded. Captions remain unfiltered; events use recorded time ±5 seconds when available."));
    body.append(button("Show all moments",()=>{selectedMoment=null;pages.captions=0;pages.events=0;renderInspector();$("moment-summary")?.focus({preventScroll:true});},"button small"));
    body.lastElementChild.id="clear-moment";
    return box;
  }
  function trendPanel(interactions) {
    const basis=el("select");basis.id="control-time-basis";basis.setAttribute("aria-label","Control timeline time basis");
    for(const [value,label] of [["auto","Time basis: auto"],["playback","Playback position"],["recorded","Recorded time"]]){const option=el("option","",label);option.value=value;basis.append(option);}basis.value=state.timeBasis;
    basis.addEventListener("change",()=>{state.timeBasis=basis.value;persist();renderInspector();});
    const panelBox=panel("Control change timeline",`${interactions.length} changes · select a point to inspect`,basis);panelBox.classList.add("timeline-panel");
    const body=el("div","panel-body timeline-body");
    append(body,append(el("div","legend"),append(el("span","legend-item"),el("i","swatch"),el("span","","Acoustic detail · solid")),append(el("span","legend-item"),el("i","swatch teal"),el("span","","Source and scene detail · dashed"))));
    const groups=new Map();
    for (const row of interactions) { const key=videoKey(row); if (!groups.has(key)) groups.set(key,[]); groups.get(key).push(row); }
    if (!groups.size) body.append(empty("No recorded control changes", "Steering trends appear when settings and playback positions are available."));
    for (const [key,rows] of groups) {
      body.append(el("p","chart-note",videoLabel(key)));
      const hasPlayback=rows.some(r=>number(r.position_ms)&&r.position_ms>=0),recorded=state.timeBasis==="recorded"||state.timeBasis==="auto"&&!hasPlayback;
      const clockValues=rows.map(r=>Date.parse(r.at)).filter(Number.isFinite),firstRecorded=clockValues.length?Math.min(...clockValues):null;
      const position=(r)=>recorded ? firstRecorded!==null&&Number.isFinite(Date.parse(r.at))?Date.parse(r.at)-firstRecorded:null : r.position_ms;
      const positioned=rows.filter(r=>number(position(r))&&position(r)>=0),axisLabel=recorded?"Recorded time since first control change (s)":"Playback position (s)";
      if (!positioned.length) { body.append(el("p","chart-note",`${rows.length} changes have no ${recorded?"recorded timestamp":"recorded playback position"}. Select the other time basis or use the table below.`)); continue; }
      const svg=chart(`${videoLabel(key)}: texture and context changes. ${axisLabel}. Values are listed in the control records below.`,850,240);
      const left=43,top=17,plotWidth=790,plotHeight=175,max=Math.max(1000,...positioned.map(position));
      for (let tick=0;tick<=4;tick++) { const y=top+plotHeight*(1-tick/4); svg.append(svgElement("line",{x1:left,y1:y,x2:left+plotWidth,y2:y,class:"grid-line"}),svgElement("text",{x:left-9,y:y+3,"text-anchor":"end"},`${tick*25}%`)); }
      for (let tick=0;tick<=5;tick++) svg.append(svgElement("text",{x:left+plotWidth*tick/5,y:top+plotHeight+20,"text-anchor":"middle"},(max*tick/5000).toLocaleString(undefined,{maximumFractionDigits:1})));
      svg.append(svgElement("text",{x:left+plotWidth/2,y:235,"text-anchor":"middle"},axisLabel));
      for (const [field,color] of [["texture","#426acb"],["context","#16877e"]]) {
        let path="",previous=null,pointCount=0;
        for (const row of rows) {
          if (!number(position(row))||position(row)<0||!number(row[field])||row[field]<0||row[field]>1) { previous=null; continue; }
          const x=left+position(row)/max*plotWidth,y=top+(1-row[field])*plotHeight;
          path+=previous ? ` H ${x} V ${y}` : ` M ${x} ${y}`;
          if (pointCount<150) {
            const circle=svgElement("circle",{cx:x,cy:y,r:isSelectedMoment("control",row)?6:4,fill:color,class:"timeline-moment",role:"button",tabindex:0,"aria-pressed":String(isSelectedMoment("control",row)),"aria-label":`Inspect ${field} change: ${seconds(position(row))} ${recorded?"recorded time":"playback"}`});
            circle.dataset.axis=field;if(number(row.position_ms))circle.dataset.positionMs=String(row.position_ms);if(row.at)circle.dataset.recordedAt=row.at;
            // Equal axes represent the same control event. Keep one hit/focus
            // target so the second painted dot cannot obstruct that selection.
            if(field==="context"&&row.texture===row.context){circle.style.pointerEvents="none";circle.setAttribute("tabindex","-1");circle.setAttribute("aria-hidden","true");}
            circle.append(svgElement("title",{},`${field}: ${row[field]}, ${seconds(position(row))} ${recorded?"since first recorded change":"playback"}, ${timestamp(row.at)}`));
            circle.addEventListener("click",()=>selectMoment("control",row));
            circle.addEventListener("keydown",event=>{if(event.key==="Enter"||event.key===" "){event.preventDefault();selectMoment("control",row);}});
            svg.append(circle);
          }
          previous={x,y}; pointCount++;
        }
        svg.append(svgElement("path",{d:path,fill:"none",stroke:color,"stroke-width":2,"stroke-linejoin":"round","stroke-dasharray":field==="context"?"6 4":"none","pointer-events":"none"}));
      }
      if(selectedMoment&&videoKey(selectedMoment.row)===key){
        const focus=recorded ? selectedMoment.type!=="caption"&&Number.isFinite(Date.parse(selectedMoment.row.at))?Date.parse(selectedMoment.row.at)-firstRecorded:null : momentPosition();
        if(number(focus)&&focus>=0&&focus<=max){const x=left+focus/max*plotWidth;svg.append(svgElement("line",{x1:x,x2:x,y1:top,y2:top+plotHeight,stroke:"#57647a","stroke-dasharray":"3 4",class:"moment-guide","pointer-events":"none"}));}
      }
      body.append(svg);
      const missing=rows.length-positioned.length;
      body.append(el("p","chart-note",`${positioned.length} of ${rows.length} recorded changes have a ${recorded?"timestamp":"playback position"}${missing ? `; ${missing} without one are not plotted` : ""}. ${recorded?`Clock: recorded event time, starting ${timestamp(new Date(firstRecorded).toISOString())}. This is not playback time.`:"Clock: playback position. Lines follow recorded order; seeking can move backward."} Missing or out-of-range levels break the line. Detail levels use the recorded 0–1 scale.`));
    }
    const data=append(el("details","data-details"),el("summary","","Show control records"));
    append(data,table(["Inspect","Playback position","Acoustic","Source/scene","Recorded at (UTC)"],paged(interactions,"interactions").map(r=>[momentButton("control",r,"Select"),seconds(r.position_ms),valueText(r.texture),valueText(r.context),timestamp(r.at)]),"Recorded control changes"),pager(interactions,"interactions",renderInspector));
    body.append(data); panelBox.append(body); return panelBox;
  }
  function captionLabel(row) { return row.kind==="job" ? "Generation job" : row.kind==="generated" ? "Generated" : ["prepared","fallback"].includes(row.kind) ? "Prepared fallback" : row.kind || "Unknown source"; }
  function captionPanel() {
    const controls=el("div","caption-filter"), label=el("label","sr-only","Search caption text"); label.htmlFor="caption-search";
    const input=el("input"); input.id="caption-search"; input.type="search"; input.placeholder="Search caption text…"; input.value=state.captionQuery;
    const select=el("select"); select.id="caption-kind"; select.setAttribute("aria-label","Caption source"); options(select,[["generated","Generated exposures"],["fallback","Prepared fallback exposures"],["job","Generation jobs"]],state.captionKind,"All caption records");
    append(controls,label,input,select);
    const box=panel("Captions & generation jobs",selectedMoment?"Near the selected moment · choose a window to focus":"Choose a caption window to inspect related events.",controls);box.classList.add("captions-panel");
    const content=el("div"); box.append(content);
    function draw() {
      const rows=relatedRecords("captions").filter(r=>(!state.captionKind || state.captionKind==="fallback" && ["prepared","fallback"].includes(r.kind) || r.kind===state.captionKind) && `${r.text || ""} ${r.job_id || ""} ${r.status || ""}`.toLowerCase().includes(state.captionQuery.toLowerCase().trim()));
      const shown=paged(rows,"captions",30);
      content.replaceChildren(highlightRows(table(["Playback window","Caption","Generation","Details"],shown.map(r=>[
        momentButton("caption",r,`${seconds(r.start_ms)} → ${seconds(r.end_ms)}`),
        append(el("div","caption-text"),statusPill(r.kind),el("p","",r.text||"No caption text recorded."),r.kind==="job"?el("small","caption-timing",`${r.status||"unknown"} · not displayed`):null),
        el("span","nowrap",duration(r.generation_ms)),
        metadata([["Phase / video",videoLabel(videoKey(r))],["Display",r.displayed?"Recorded exposure":"Not displayed"],["Job",r.job_id??"Not recorded"],["Revision",r.revision??"Not recorded"],["Acoustic detail",r.texture??"Not recorded"],["Source/scene detail",r.context??"Not recorded"],["Source timestamp",timestamp(r.created_at)],["Timestamp meaning","May be job creation; not a measured caption visibility time."]],"Caption details")
      ]),"Caption exposures and generation jobs"),shown,"caption"),pager(rows,"captions",draw,30));
    }
    input.addEventListener("input",()=>{state.captionQuery=input.value;pages.captions=0;persist();draw();});
    select.addEventListener("change",()=>{state.captionKind=select.value;pages.captions=0;persist();draw();});
    draw(); return box;
  }
  function completedDurations(rows) {
    const byJob=new Map(), anonymous=[];
    for (const row of rows) {
      if (!number(row.generation_ms)||row.generation_ms<0||!["succeeded","failed","completed","complete","done"].includes(row.status)) continue;
      if (!row.job_id && row.kind!=="job") continue;
      if (row.job_id) byJob.set(JSON.stringify([row.phase,row.job_id]),row.generation_ms); else anonymous.push(row.generation_ms);
    }
    return [...byJob.values(),...anonymous].sort((a,b)=>a-b);
  }
  function timingPanel() {
    const values=completedDurations(inspectorRecords("captions")),box=panel("Model generation time","Completed job durations, including failed jobs with a recorded duration · milliseconds");
    if (!values.length) { box.append(empty("No completed-job durations", "Exposure windows, deadlines and elapsed playback time are not model generation time.")); return box; }
    const median=values.length%2?values[Math.floor(values.length/2)]:(values[values.length/2-1]+values[values.length/2])/2;
    const summary=el("div","timing-summary"); for (const [name,value] of [["Completed jobs",count(values.length)],["Median",duration(median)],["Minimum",duration(values[0])],["Maximum",duration(values[values.length-1])]]) append(summary,append(el("div","",name),el("strong","",value))); box.append(summary);
    const body=el("div","panel-body");
    if (values.length<2) { body.append(el("p","chart-note","One completed measurement; a distribution needs more than one job.")); box.append(body); return box; }
    const low=values[0],high=values[values.length-1],binCount=low===high?1:Math.min(8,Math.ceil(Math.sqrt(values.length))),width=low===high?1:(high-low)/binCount;
    const bins=Array.from({length:binCount},(_,i)=>({low:low+i*width,high:low+(i+1)*width,count:0}));
    for (const value of values) bins[Math.min(binCount-1,Math.floor((value-low)/width))].count++;
    const svg=chart("Histogram of completed model job durations in milliseconds. Exact bins are available below.",850,195),left=40,top=12,plotWidth=790,plotHeight=145,tickStep=Math.max(1,Math.ceil(Math.max(...bins.map(b=>b.count))/2)),max=tickStep*2;
    for (let i=0;i<=2;i++) { const y=top+plotHeight*(1-i/2); svg.append(svgElement("line",{x1:left,y1:y,x2:left+plotWidth,y2:y,class:"grid-line"}),svgElement("text",{x:left-8,y:y+3,"text-anchor":"end"},String(i*tickStep))); }
    bins.forEach((bin,i)=>{const step=plotWidth/binCount,h=bin.count/max*plotHeight,rect=svgElement("rect",{x:left+i*step+6,y:top+plotHeight-h,width:step-12,height:h,rx:2,fill:"#6989ce"}); rect.append(svgElement("title",{},`${duration(bin.low)} to ${duration(low===high?low:bin.high)}: ${bin.count} jobs`)); svg.append(rect,svgElement("text",{x:left+(i+.5)*step,y:180,"text-anchor":"middle"},low===high?String(low):`${Math.round(bin.low)}–${Math.round(bin.high)}`));});
    const records=append(el("details","data-details"),el("summary","","Show timing distribution counts"),table(["Generation time (ms)","Completed jobs"],bins.map((b,i)=>[low===high?String(low):`${b.low.toFixed(1)} ≤ duration ${i===bins.length-1?"≤":"<"} ${b.high.toFixed(1)}`,b.count]),"Model duration histogram bins"));
    append(body,svg,el("p","chart-note","Durations come from completed generation jobs, counted once per recorded job ID. This is model execution time, not time from a control change to a visible caption."),records);box.append(body);return box;
  }
  function eventPanel() {
    const controls=el("div","event-tools"),input=el("input");input.id="event-search";input.type="search";input.placeholder="Find event or detail…";input.value=state.eventQuery;input.setAttribute("aria-label","Search event log");
    const order=el("select");order.id="event-order";order.setAttribute("aria-label","Event order");for(const [value,label] of [["asc","Oldest first"],["desc","Newest first"]]){const option=el("option","",label);option.value=value;order.append(option);}order.value=state.eventOrder;
    append(controls,input,order);const box=panel("Event log",selectedMoment?"Near the selected moment · choose an event to focus":"Recorded order · choose an event to inspect",controls),content=el("div");box.classList.add("events-panel");box.append(content);
    function draw(){
      const rows=relatedRecords("events").filter(r=>`${r.kind||""} ${JSON.stringify(r.detail??{})}`.toLowerCase().includes(state.eventQuery.toLowerCase().trim())).map((row,index)=>({...row,_index:index})).sort((a,b)=>{
        const aTime=Date.parse(a.at),bTime=Date.parse(b.at);if(!Number.isFinite(aTime))return Number.isFinite(bTime)?1:a._index-b._index;if(!Number.isFinite(bTime))return -1;return(state.eventOrder==="desc"?-1:1)*(aTime-bTime)||a._index-b._index;
      });
      const shown=paged(rows,"events");
      content.replaceChildren(highlightRows(table(["Moment","Event","Details"],shown.map(r=>{
        const detailCell=el("div","event-detail"),details=r.detail??{},entries=typeof details==="object"?Object.entries(details):[["value",details]];
        const concise=entries.slice(0,4).map(([key,value])=>`${key}: ${valueText(value)}`).join(" · ");
        detailCell.append(el("span","",concise.length>260?`${concise.slice(0,257)}…`:concise||"No additional details"));
        if(entries.length>4||concise.length>260)detailCell.append(append(el("details"),el("summary","","Full recorded detail"),el("pre","",JSON.stringify(details,null,2))));
        return[append(el("div"),momentButton("event",r,number(r.position_ms)?seconds(r.position_ms):timestamp(r.at,true)),el("small","caption-timing",number(r.position_ms)?"Playback position":"Recorded time (UTC)")),append(el("div"),el("strong","",r.kind),detailCell),metadata([["Phase / video",videoLabel(videoKey(r))],["Recorded at (UTC)",timestamp(r.at)],["Playback position",seconds(r.position_ms)],["Full recorded detail",JSON.stringify(r.detail??{},null,2)]],"Event details")];
      }),"Chronological session events"),shown,"event"),pager(rows,"events",draw));
    }
    input.addEventListener("input",()=>{state.eventQuery=input.value;pages.events=0;persist();draw();});order.addEventListener("change",()=>{state.eventOrder=order.value;pages.events=0;persist();draw();});draw();return box;
  }
  function renderInspector() {
    const root=$("view-inspector");root.replaceChildren();
    if(!overview){root.append(empty(overviewError?"Collection unavailable":"Loading participants…",overviewError));return;}
    const participants=filteredParticipants(),layout=el("div","inspector-layout"),browser=panel("Participants",`${count(participants.length)} matching sessions`);browser.classList.add("participant-browser");
    const optionsBox=el("div","participant-options");
    for(const p of paged(participants,"browser",12)){
      const option=button("",()=>selectParticipant(p.id),"participant-option");option.dataset.participantId=p.id;option.setAttribute("aria-pressed",String(state.participant===p.id));
      append(option,el("strong","mono",p.label||p.id),append(el("small"),el("span","",languageName(p.language)),el("span","",p.qa?"QA session":p.phase2_status==="done"?"Phase 2 complete":"In progress")));optionsBox.append(option);
    }
    if(!participants.length)optionsBox.append(empty("No matching sessions","Adjust the collection filters."));
    append(browser,optionsBox,pager(participants,"browser",renderInspector,12));
    const content=el("div","inspector-content");content.id="inspector-detail";append(layout,browser,content);root.append(layout);
    if(!state.participant){content.append(empty("Select a participant","Inspect submitted responses, control changes, caption records and the event log."));return;}
    const selected=list(overview.participants).find(p=>p.id===state.participant);
    if(!selected){content.append(empty("Selected participant is not in this collection","The selection is preserved. Adjust the QA setting or select another participant."));return;}
    if(!participants.some(p=>p.id===state.participant))content.append(el("p","loading-inline","This selected session is outside the current participant filters. Its details remain visible."));
    if(detailError)content.append(el("div","inspector-error",`${detail?"Showing previously loaded session data. ":""}${detailError}`));
    if(detailLoading)content.append(el("p","loading-inline","Refreshing session data…"));
    if(!detail||detail.participant?.id!==state.participant){content.append(empty(detailError?"Session unavailable":"Loading session…",detailError?"Use Refresh to retry.":"Reading recorded observations."));return;}
    for(const warning of list(detail.warnings))content.append(el("p","inspector-error",warning));
    const p=detail.participant,heading=append(el("div"),el("h2","mono",p.label||p.id),el("p","",`${languageName(p.language)}${p.qa?" · QA session":""} · Last activity ${timestamp(p.last_activity)}`));
    append(heading,append(el("div","session-status"),el("span","","Phase 1"),statusPill(p.phase1_status),el("span","","Phase 2"),statusPill(p.phase2_status)));
    const facts=el("div","session-facts");for(const [value,label] of [[p.phase1_surveys,"Phase 1 surveys"],[p.phase2_surveys,"Phase 2 surveys"],[p.videos_completed,"videos complete"],[p.control_changes,"control changes"]])append(facts,append(el("span"),el("strong","",label === "videos complete" ? videoProgress(p) : count(value)),document.createTextNode(` ${label}`)));heading.append(facts);
    content.append(append(el("div","session-header"),heading,exportLink("Export session JSON",`/api/export/participants/${encodeURIComponent(p.id)}.json`)));
    const videoKeys=unique([...list(detail.interactions),...list(detail.captions),...list(detail.events),...list(detail.responses)].map(videoKey)),filters=el("div","filter-bar inspector-filters");
    filters.append(selectFilter("Phase / video","inspectorVideo",videoKeys.map(key=>[key,videoLabel(key)]),"All phases and videos",()=>{selectedMoment=null;pages.captions=0;pages.events=0;pages.interactions=0;pages.detailResponses=0;renderInspector();}));content.append(filters);
    const workbench=append(el("div","inspector-workbench"),trendPanel(inspectorRecords("interactions")),momentSummary());
    const related=append(el("div","linked-records"),captionPanel(),eventPanel());
    const timings=append(el("details","inspector-appendix"),el("summary","","Model timing distribution"),timingPanel());
    const surveys=append(el("details","inspector-appendix"),el("summary","",`Submitted responses · ${inspectorRecords("responses").length} answers`),responseTable(inspectorRecords("responses"),"detailResponses",renderInspector));
    append(content,workbench,related,timings,surveys);
  }
  function renderWarnings() {
    const box=$("source-warnings"),warnings=[...(overviewError?[`${overview ? "Stale collection data. " : ""}${overviewError}`]:[]),...list(overview?.warnings)];box.replaceChildren();for(const warning of warnings)box.append(el("p","",warning));box.hidden=!warnings.length;
  }
  function renderStatus() {
    const node=$("refresh-status"),analysis=state.view==="phase1";node.classList.toggle("stale",!!(analysis?phase1Error:overviewError));
    node.textContent=analysis&&phase1Loading?"Refreshing Phase 1 analysis…":analysis&&phase1Error?"Analysis refresh failed":analysis&&phase1?"Phase 1 analysis loaded":overviewLoading?"Refreshing collection…":overviewError?"Refresh failed · data may be stale":overview?`Updated ${timestamp(overview.updated_at,true)} UTC`:"Collection not loaded";
    $("refresh-button").disabled=overviewLoading||(analysis&&phase1Loading);
    $("collection-updated").textContent=overview?`Source snapshot ${timestamp(overview.updated_at)}`:"Timestamps shown in UTC";
    $("export-responses").hidden=analysis;
    $("export-responses").href=`/api/export/responses.csv?include_qa=${state.includeQa}`;
  }
  function renderView() {
    for(const view of ["overview","responses","phase1","inspector"])$( `view-${view}`).hidden=state.view!==view;
    for(const button of document.querySelectorAll("[data-view]")){if(button.dataset.view===state.view)button.setAttribute("aria-current","page");else button.removeAttribute("aria-current");}
    $("view-title").textContent={overview:"Overview",responses:"Responses",phase1:"Phase 1 trends",inspector:"Session inspector"}[state.view];
    ({overview:renderOverview,responses:renderResponses,phase1:renderPhase1,inspector:renderInspector})[state.view]();renderWarnings();renderStatus();
  }
  function setView(view) { state.view=view;persist();renderView();if(view==="inspector"&&state.participant&&detail?.participant?.id!==state.participant&&!detailLoading)fetchParticipant();if(view==="phase1"&&(!phase1||phase1LoadedQuery!==phase1Query())&&!phase1Loading)fetchPhase1(); }
  function selectParticipant(id) {
    if(state.participant!==id){selectedMoment=null;state.participant=id;state.inspectorVideo="";detail=null;detailError="";for(const key of ["captions","events","interactions","detailResponses"])pages[key]=0;}
    state.view="inspector";persist();renderView();fetchParticipant();
  }
  async function fetchJson(url) {
    const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),20000);
    try {const response=await fetch(url,{cache:"no-store",signal:controller.signal,headers:{Accept:"application/json"}});if(!response.ok)throw new Error(`Request failed (${response.status}).`);return await response.json();}
    finally{clearTimeout(timeout);}
  }
  async function fetchParticipant() {
    if(!state.participant || !list(overview?.participants).some(p=>p.id===state.participant))return;
    const id=++detailRequestId,participant=state.participant;detailLoading=true;detailError="";if(state.view==="inspector")renderInspector();
    try{const payload=await fetchJson(`/api/participants/${encodeURIComponent(participant)}?include_qa=${state.includeQa}`);if(id!==detailRequestId||participant!==state.participant)return;if(!payload?.participant||payload.participant.id!==participant)throw new Error("Session response did not match the selected participant.");detail=payload;
      if(selectedMoment){const key=selectedMoment.type==="control"?"interactions":selectedMoment.type==="caption"?"captions":"events",found=list(detail[key]).find(row=>momentKey(selectedMoment.type,row)===selectedMoment.key);selectedMoment=found?{...selectedMoment,row:found}:null;}
    }
    catch(error){if(id===detailRequestId)detailError=error.name==="AbortError"?"Session request timed out. Use Refresh to retry.":error.message||"Could not read session data.";}
    finally{if(id===detailRequestId){detailLoading=false;if(state.view==="inspector")renderInspector();}}
  }
  async function refresh() {
    const id=++requestId;overviewLoading=true;renderStatus();
    try{const payload=await fetchJson(`/api/overview?include_qa=${state.includeQa}`);if(id!==requestId)return;if(!payload||!Array.isArray(payload.participants)||!Array.isArray(payload.responses))throw new Error("Collection response is incomplete.");overview=payload;overviewError="";options($("language-filter"),unique([...overview.participants,...overview.responses].map(p=>p.language||"unknown")).map(value=>[value,languageName(value)]),state.language,"All languages");}
    catch(error){if(id===requestId)overviewError=error.name==="AbortError"?"Collection request timed out. Use Refresh to retry.":error.message||"Could not read collection data.";}
    finally{if(id===requestId){overviewLoading=false;renderView();if(state.view==="inspector"&&state.participant)fetchParticipant();if(state.view==="phase1")fetchPhase1();}}
  }
  function configureRefresh(){if(timer)clearInterval(timer);timer=state.autoRefresh?setInterval(()=>{if(!document.hidden&&!overviewLoading&&!phase1Loading)refresh();},15000):null;}
  for(const button of document.querySelectorAll("[data-view]"))button.addEventListener("click",()=>setView(button.dataset.view));
  document.querySelector(".brand").addEventListener("click",event=>{event.preventDefault();setView("overview");});
  $("participant-search").value=state.query;$("language-filter").value=state.language;$("status-filter").value=state.status;$("include-qa").checked=state.includeQa;$("auto-refresh").checked=state.autoRefresh;
  for(const [id,key,eventName] of [["participant-search","query","input"],["language-filter","language","change"],["status-filter","status","change"]])$(id).addEventListener(eventName,()=>{state[key]=$(id).value;pages.participants=0;pages.browser=0;pages.groups=0;pages.responses=0;phase1=null;phase1Error="";phase1RequestId++;phase1Loading=false;for(const name of Object.keys(pages))if(name.startsWith("phase1"))pages[name]=0;clearTimeout(phase1FilterTimer);persist();renderView();if(state.view==="phase1")phase1FilterTimer=setTimeout(fetchPhase1,180);});
  $("include-qa").addEventListener("change",()=>{selectedMoment=null;state.includeQa=$("include-qa").checked;persist();overview=null;detail=null;detailRequestId++;detailLoading=false;overviewError="";detailError="";phase1=null;phase1Error="";phase1RequestId++;phase1Loading=false;phase1LoadedQuery="";clearTimeout(phase1FilterTimer);renderView();refresh();});
  $("auto-refresh").addEventListener("change",()=>{state.autoRefresh=$("auto-refresh").checked;persist();configureRefresh();});
  $("refresh-button").addEventListener("click",refresh);
  renderView();configureRefresh();refresh();
})();
