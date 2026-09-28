const categories = {
  reasoning_summary: {label:"Reasoning",color:"#095256"},
  plan: {label:"Plan",color:"#087f8c"},
  progress_update: {label:"Progress",color:"#5aaa95"},
  // Temporarily hidden. Restore this entry when tool-call events are re-enabled.
  // tool_call: {label:"Tool call",color:"#86a873"},
  function_result: {label:"Tool result",color:"#bb9f06"}
};
const palette = ["#095256","#087f8c","#5aaa95","#bb9f06","#765d69"];
let taskId = new URLSearchParams(location.search).get("task") || "all";
let state = {points:[],final_answer:null,projection:"none"};
let socket = null;
let selectedPointId = null;
let zoomTransform = d3.zoomIdentity;
let previousPlotSize = "";
const status = document.getElementById("status");
const tooltip = d3.select("#tooltip");

document.getElementById("legend").innerHTML = Object.values(categories)
  .map(value=>`<span style="--c:${value.color}">${value.label}</span>`).join("");
document.getElementById("density-mode").addEventListener("change",draw);
document.getElementById("density-style").addEventListener("change",draw);
document.getElementById("ask-form").addEventListener("submit",startRun);
document.getElementById("new-session").addEventListener("click",newSession);

async function newSession() {
  const button = document.getElementById("new-session");
  button.disabled = true;
  status.textContent = "Clearing session…";
  try {
    const response = await fetch("/api/sessions",{method:"POST"});
    if (!response.ok) throw new Error(await response.text());
    state = await response.json();
    taskId = state.task_id;
    selectedPointId = null;
    zoomTransform = d3.zoomIdentity;
    history.replaceState(null,"",`?task=${encodeURIComponent(taskId)}`);
    document.getElementById("prompt").value = "";
    render();
    connect();
  } catch (error) {
    status.textContent = `Could not clear: ${error.message}`;
  } finally {
    button.disabled = false;
  }
}

async function startRun(event) {
  event.preventDefault();
  const prompt = document.getElementById("prompt").value.trim();
  if (!prompt) return;
  const button = document.getElementById("submit");
  button.disabled = true;
  status.textContent = "Starting Codex…";
  try {
    const payload = {prompt};
    if (taskId !== "all") payload.task_id = taskId;
    const response = await fetch("/api/runs",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
    if (!response.ok) throw new Error(await response.text());
    const created = await response.json();
    taskId = created.task_id;
    history.replaceState(null,"",`?task=${encodeURIComponent(taskId)}`);
    state = created;
    render();
    connect();
  } catch (error) {
    status.textContent = `Could not start: ${error.message}`;
  } finally {
    button.disabled = state.status === "queued" || state.status === "running";
  }
}

function connect() {
  if (socket) socket.close();
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const connectedTask = taskId;
  socket = new WebSocket(`${protocol}://${location.host}/ws/${connectedTask}`);
  socket.onopen = ()=>status.textContent = `Live · ${taskId}`;
  socket.onmessage = event=>{ state=JSON.parse(event.data); render(); };
  socket.onclose = ()=>{
    if (connectedTask !== taskId) return;
    status.textContent = "Reconnecting…";
    setTimeout(connect,1200);
  };
}

function render() {
  status.textContent = `${state.status || "live"} · ${state.points.length} points · ${state.projection}`;
  const busy = state.status === "queued" || state.status === "running";
  document.getElementById("submit").disabled = busy;
  document.getElementById("new-session").disabled = busy;
  const answer = document.getElementById("answer");
  if (state.final_answer) answer.innerHTML = DOMPurify.sanitize(marked.parse(state.final_answer));
  else answer.textContent = state.error || "The answer will appear when the agent completes.";
  answer.classList.toggle("empty",!state.final_answer && !state.error);
  updatePointDetail();
  draw();
}

function updatePointDetail(point=null) {
  point = point || state.points.find(item=>item.id === selectedPointId);
  const detail = document.getElementById("point-detail");
  const title = document.getElementById("detail-title");
  if (!point) {
    selectedPointId = null;
    title.textContent = "Point details";
    detail.className = "point-detail empty";
    detail.textContent = "Click a point to read its full text.";
    return;
  }
  selectedPointId = point.id;
  const category = categories[point.category] || {label:point.category,color:"#6f756f"};
  title.textContent = category.label;
  detail.className = "point-detail";
  detail.innerHTML = `<div class="point-meta" style="--point-color:${category.color}">${escapeHtml(category.label)}</div><div>${escapeHtml(point.text)}</div>`;
}

function draw() {
  const host = document.getElementById("plot");
  host.innerHTML = "";
  const width = Math.max(320,host.clientWidth);
  const height = Math.max(560,host.clientHeight);
  const svg = d3.select(host).append("svg").attr("width",width).attr("height",height).attr("viewBox",`0 0 ${width} ${height}`);
  const viewport = svg.append("g").attr("class","zoom-viewport");
  const zoom = d3.zoom().scaleExtent([.5,8]).on("zoom",event=>{
    zoomTransform = event.transform;
    viewport.attr("transform",zoomTransform);
  });
  svg.call(zoom).call(zoom.transform,zoomTransform);

  if (!state.points.length) {
    viewport.append("text").attr("x",width/2).attr("y",height/2).attr("text-anchor","middle").attr("fill","#6f756f").text("Waiting for agent events");
    return;
  }

  const margin = {top:105,right:46,bottom:44,left:46};
  const pad = extent=>{ const span=(extent[1]-extent[0])||1; return [extent[0]-span*.18,extent[1]+span*.18]; };
  const x = d3.scaleLinear().domain(pad(d3.extent(state.points,d=>d.x))).range([margin.left,width-margin.right]);
  const y = d3.scaleLinear().domain(pad(d3.extent(state.points,d=>d.y))).range([height-margin.bottom,margin.top]);

  if (state.points.length >= 8) {
    const mode = document.getElementById("density-mode").value;
    const style = document.getElementById("density-style").value;
    const groups = mode === "all"
      ? d3.groups(state.points,d=>d.category)
      : d3.groups(state.points,d=>`${d.task_id}::${d.category}`);
    groups.forEach(([key,points],index)=>{
      if (points.length < 5) return;
      const density = d3.contourDensity().x(d=>x(d.x)).y(d=>y(d.y)).size([width,height]).bandwidth(28).thresholds(8)(points);
      const category = points[0].category;
      const color = categories[category]?.color || palette[index%palette.length];
      const peak = d3.max(density,d=>d.value) || 1;
      viewport.append("g").selectAll("path").data(density).join("path")
        .attr("d",d3.geoPath())
        .attr("fill",style === "outline" ? "none" : color)
        .attr("fill-opacity",d=>style === "smooth" ? .04+.28*(d.value/peak) : style === "fill" ? .075 : 0)
        .attr("stroke",style === "smooth" ? "none" : color)
        .attr("stroke-opacity",style === "outline" ? .28 : .34)
        .attr("stroke-width",1.25);
    });
  }

  viewport.append("g").selectAll("circle.point").data(state.points,d=>d.id).join("circle")
    .attr("class","point").attr("cx",d=>x(d.x)).attr("cy",d=>y(d.y)).attr("r",d=>d.id === selectedPointId ? 9 : 7)
    .attr("fill",d=>categories[d.category]?.color || "#6f756f").attr("stroke",d=>d.id === selectedPointId ? "#242a28" : "#fffdf8").attr("stroke-width",d=>d.id === selectedPointId ? 2.5 : 1.4)
    .style("cursor","pointer")
    .on("click",(event,d)=>{ event.stopPropagation(); updatePointDetail(d); draw(); })
    .on("mouseenter",(event,d)=>tooltip.style("opacity",1).html(`<strong>${categories[d.category]?.label || d.category}</strong><br>${escapeHtml(d.text.slice(0,160))}${d.text.length>160?"…":""}`))
    .on("mousemove",event=>tooltip.style("left",`${event.pageX+14}px`).style("top",`${event.pageY+14}px`))
    .on("mouseleave",()=>tooltip.style("opacity",0));

  viewport.append("g").selectAll("text.reasoning-label")
    .data(state.points.filter(d=>d.category === "reasoning_summary").slice(-5),d=>d.id).join("text")
    .attr("class","reasoning-label").attr("x",d=>x(d.x)).attr("y",d=>y(d.y)-11).attr("text-anchor","middle")
    .attr("fill","#242a28").attr("font-size",11).attr("paint-order","stroke").attr("stroke","#fffdf8").attr("stroke-width",3).attr("stroke-linejoin","round")
    .text(d=>d.text.replace(/\*\*/g,"").slice(0,70));
}

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = value;
  return div.innerHTML;
}

new ResizeObserver(entries=>{
  const rect = entries[0].contentRect;
  const size = `${Math.round(rect.width)}x${Math.round(rect.height)}`;
  if (size === previousPlotSize) return;
  previousPlotSize = size;
  draw();
}).observe(document.getElementById("plot"));
connect();
