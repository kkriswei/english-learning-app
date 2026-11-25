/* 离线 ASR worker（v3.10；v4.2 换模型）：transformers.js 3.8.1 + Xenova/whisper-base.en 英文版 q8。
   v4.2 基准实测（asrb/）：模拟手机录音口径单词 exact 30→34/40、garbage 7→2、句子 exact 10→11/15。
   英文单语模型传 language/task 会被库抛错，转写只带 chunk_length_s。
   只做兜底转写：主线程把 16kHz 单声道 Float32Array PCM 传进来，这里跑推理后回文字。
   模型/运行时全部从本站 asr/ 目录加载（相对 worker 自身位置推导，不写死站点根路径）。
   生产配置：GitHub Pages 无跨源隔离 → numThreads=1；wasmPaths 只给 {mjs, wasm} 两键
   （onnxruntime-web 1.22 只认这两键），用实测跑通的 jsep 对。
   v6.1（2026-11-25 真机诊断实证）：模型文件 transformers.js 会存 Cache Storage 持久
   可靠，唯独 ORT 的 21.6MB wasm 二进制没有任何持久化——GitHub Pages 只给
   Cache-Control: max-age=600，iOS 主屏幕应用的 HTTP 缓存又留不住，于是每次开引擎
   都要现从 github.io 重抓 21.6MB，她的网络下 fetch 与同步 XHR 双双失败，ORT 抛
   「no available backend found … both async and sync fetching of the wasm failed」，
   引擎根本起不来（11-21 后发音全灭、模型校验却全绿）。修法：wasm 二进制与模型
   同等待遇——先查 transformers-cache，命中直接以字节喂 ORT（env.wasm.wasmBinary，
   运行时零抓取）；未命中才 fetch 一次并回存缓存（带 0–15% 进度）。wasmPaths 同源
   配置保留作兜底，万一 wasmBinary 被忽略仍走同源地址、不碰外部 CDN。
   另注记一桩死史：本文件头注释自 v3.20 起被改坏（第 3 行提前闭合、其后裸文本），
   Worker 此后一直语法错误、出生即死，全部转写被迫走主线程回退（慢、且 wasm 全靠
   现抓）；v6.1 重写文件头并恢复单块注释，node --check 实测可解析，Worker 复活。 */
"use strict";
const ASR_BASE = new URL("./", self.location.href).href;   /* asr/ 目录的绝对地址 */
const ASR_WASM_URL = ASR_BASE + "lib/ort-wasm-simd-threaded.jsep.wasm";
let pipeP = null;
let lastPct = -1;
function postProgress(pct){
  if(pct === lastPct) return;
  lastPct = pct;
  self.postMessage({ type: "progress", pct: pct });
}
/* 进度分段：wasm 阶段占 0–15%，模型文件阶段映射到 15–100%（progMap 在 loadPipe 里装配） */
let progMap = p => p;
/* 库的 progress 回调按文件给进度，这里聚合成一个总百分比（真实字节口径） */
function makeProgAgg(){
  const files = new Map();
  return info => {
    try {
      if(!info || info.status !== "progress" || !info.file) return;
      if(typeof info.loaded === "number" && typeof info.total === "number" && info.total > 0){
        files.set(info.file, { loaded: info.loaded, total: info.total });
        let l = 0, t = 0;
        files.forEach(v => { l += v.loaded; t += v.total; });
        if(t > 0) postProgress(progMap(Math.max(0, Math.min(99, Math.round(l / t * 100)))));
      }
    } catch(e){}
  };
}
/* v6.1：取 wasm 二进制——缓存优先（transformers-cache，与模型同仓持久化），
   未命中才同源 fetch 并回存。返回 ArrayBuffer，交给 ORT 的 wasmBinary 入口，
   引擎运行时不再自己抓 wasm（她真机就是死在这一步抓取上）。 */
async function asrWasmBinary(){
  try {
    const c = await caches.open("transformers-cache");
    const hit = await c.match(ASR_WASM_URL);
    if(hit){
      const b = await hit.arrayBuffer();
      if(b && b.byteLength > 1000000) return b;
    }
  } catch(e){}
  const resp = await fetch(ASR_WASM_URL);
  if(!resp || !resp.ok) throw new Error("wasm fetch failed: " + (resp && resp.status));
  let buf = null;
  const total = +(resp.headers && typeof resp.headers.get === "function" ? resp.headers.get("content-length") : 0) || 0;
  if(resp.body && typeof resp.body.getReader === "function" && total > 0){
    const reader = resp.body.getReader();
    const chunks = [];
    let got = 0;
    for(;;){
      const r = await reader.read();
      if(r.done) break;
      chunks.push(r.value);
      got += r.value.length;
      postProgress(Math.min(15, Math.round(got / total * 15)));
    }
    const u8 = new Uint8Array(got);
    let off = 0;
    for(const ch of chunks){ u8.set(ch, off); off += ch.length; }
    buf = u8.buffer;
  } else {
    buf = await resp.arrayBuffer();
  }
  if(!buf || buf.byteLength <= 1000000) throw new Error("wasm fetch failed: bad body " + (buf && buf.byteLength));
  try {
    const c = await caches.open("transformers-cache");
    await c.put(ASR_WASM_URL, new Response(buf.slice(0), { headers: { "Content-Type": "application/wasm" } }));
  } catch(e){}
  return buf;
}
function loadPipe(){
  if(pipeP) return pipeP;
  pipeP = (async () => {
    const T = await import(ASR_BASE + "lib/transformers.web.min.js");
    const env = T.env;
    env.remoteHost = ASR_BASE;
    env.remotePathTemplate = "model/";   /* 3.8.1 的模板不含 {file}：文件名由库 pathJoin 追加 */
    env.backends.onnx.wasm.numThreads = 1;
    env.backends.onnx.wasm.wasmPaths = {
      mjs: ASR_BASE + "lib/ort-wasm-simd-threaded.jsep.mjs",
      wasm: ASR_BASE + "lib/ort-wasm-simd-threaded.jsep.wasm"
    };
    progMap = p => 15 + Math.round(p * 0.85);
    env.backends.onnx.wasm.wasmBinary = await asrWasmBinary();   /* v6.1：字节直喂，运行时零抓取 */
    const p = await T.pipeline("automatic-speech-recognition", "whisper-base.en", {
      device: "wasm", dtype: "q8", progress_callback: makeProgAgg()
    });
    postProgress(100);
    return p;
  })();
  pipeP.catch(() => { pipeP = null; });
  return pipeP;
}
self.onmessage = async e => {
  const m = e.data || {};
  if(m.type === "preload"){
    try { await loadPipe(); self.postMessage({ type: "loaded", id: m.id }); }
    catch(err){ self.postMessage({ type: "error", id: m.id, error: String((err && err.message) || err) }); }
    return;
  }
  if(m.type === "transcribe"){
    try {
      const p = await loadPipe();
      const r = await p(m.pcm, { chunk_length_s: 30 });
      const text = ((Array.isArray(r) ? (r[0] && r[0].text) : (r && r.text)) || "").trim();
      self.postMessage({ type: "result", id: m.id, text: text });
    } catch(err){
      self.postMessage({ type: "error", id: m.id, error: String((err && err.message) || err) });
    }
  }
};
