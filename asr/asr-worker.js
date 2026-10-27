/* 离线 ASR worker（v3.10；v4.2 换模型）：transformers.js 3.8.1 + Xenova/whisper-base.en 英文版 q8。
   v4.2 基准实测（asrb/）：模拟手机录音口径单词 exact 30→34/40、garbage 7→2、句子 exact 10→11/15。
   英文单语模型传 language/task 会被库抛错，转写只带 chunk_length_s。 */
   只做兜底转写：主线程把 16kHz 单声道 Float32Array PCM 传进来，这里跑推理后回文字。
   模型/运行时全部从本站 asr/ 目录加载（相对 worker 自身位置推导，不写死站点根路径）。
   生产配置：GitHub Pages 无跨源隔离 → numThreads=1；wasmPaths 只给 {mjs, wasm} 两键
   （onnxruntime-web 1.22 只认这两键），用实测跑通的 jsep 对。 */
"use strict";
const ASR_BASE = new URL("./", self.location.href).href;   /* asr/ 目录的绝对地址 */
let pipeP = null;
let lastPct = -1;
function postProgress(pct){
  if(pct === lastPct) return;
  lastPct = pct;
  self.postMessage({ type: "progress", pct: pct });
}
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
        if(t > 0) postProgress(Math.max(0, Math.min(99, Math.round(l / t * 100))));
      }
    } catch(e){}
  };
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
