'use strict';
const { app, BrowserWindow, Menu, dialog } = require('electron');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { startBackend, sameOrigin } = require('./lifecycle.cjs');
const { record } = require('./process_log.cjs');
const args = process.argv;
function flag(name) { const i = args.indexOf(name); return i < 0 ? null : args[i + 1]; }
const smoke = flag('--smoke-report');
if (flag('--user-data-root')) app.setPath('userData', path.resolve(flag('--user-data-root')));
let owned, win, closing = false, switching = false, settings, settingsPath;

function preferences(partition) {
  return { partition, nodeIntegration: false, contextIsolation: true, sandbox: true,
    webSecurity: true, allowRunningInsecureContent: false, webviewTag: false };
}

async function workspace() {
  const candidates = [flag('--project'), process.env.MEDICAL_AGENT_PROJECT_ROOT, settings.project,
    path.resolve(__dirname, '..')];
  const descriptor = path.join(process.resourcesPath, 'project-location.json');
  if (fs.existsSync(descriptor)) candidates.push(JSON.parse(fs.readFileSync(descriptor, 'utf8')).project);
  for (const p of candidates) if (p && fs.existsSync(path.join(p, 'desktop', 'backend.py'))) return path.resolve(p);
  const result = await dialog.showOpenDialog({ title: '选择已有 Medical-Agent 工程目录', properties: ['openDirectory'] });
  if (result.canceled) throw new Error('尚未选择工程目录。');
  return path.resolve(result.filePaths[0]);
}

async function boot(mode) {
  if (owned) { await owned.stop(); owned = null; }
  if (win) { win.destroy(); win = null; }
  const project = await workspace();
  const id = crypto.createHash('sha256').update(project.toLowerCase()).digest('hex').slice(0, 12);
  const data = path.join(app.getPath('userData'), 'workspaces', id, mode);
  settings = { project, mode };
  fs.writeFileSync(settingsPath, JSON.stringify(settings, null, 2));
  owned = await startBackend({ project, data,
    ui: app.isPackaged ? path.join(process.resourcesPath, 'ui') : path.join(__dirname, 'ui'), mode });
  const current = owned;
  record(data,'main','workspace_boot',{mode,pid:process.pid});
  win = new BrowserWindow({ width: 1360, height: 920, minWidth: 850, minHeight: 620,
    title: 'Medical Agent', backgroundColor: '#f6f6f3', show: !smoke,
    webPreferences: { ...preferences(`persist:medical-${id}-${mode}`), backgroundThrottling: !smoke } });
  const ses = win.webContents.session;
  win.webContents.on('render-process-gone',(_e,details)=>record(data,'renderer','process_gone',details));
  win.webContents.on('unresponsive',()=>record(data,'renderer','unresponsive'));
  win.webContents.on('did-fail-load',(_e,exitCode)=>record(data,'renderer','load_failed',{exitCode}));
  win.on('closed',()=>record(data,'window','closed',{expected:closing||switching||!!smoke,mode}));
  ses.webRequest.onBeforeSendHeaders({ urls: [current.origin + '/*'] }, (details, callback) => {
    callback({ requestHeaders: { ...details.requestHeaders, 'X-Medical-Desktop': current.token } });
  });
  // Renderer resources and fetches are local only; no navigation or popup can
  // carry the capability into another origin. There is no preload or IPC API.
  ses.webRequest.onBeforeRequest((details, callback) => callback({ cancel: !sameOrigin(details.url, current.origin) }));
  ses.setPermissionRequestHandler((_wc, _permission, callback) => callback(false));
  ses.setPermissionCheckHandler(() => false);
  win.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
  for (const event of ['will-navigate', 'will-redirect']) win.webContents.on(event, (e, url) => {
    if (!sameOrigin(url, current.origin)) e.preventDefault();
  });
  await win.loadURL(current.origin + '/desktop/');
  Menu.setApplicationMenu(Menu.buildFromTemplate([
    { label: '问诊', submenu: [
      { label: '规则演示', type: 'radio', checked: mode === 'rule', click: () => restart('rule') },
      { label: 'Qwen3.5-4B 候选', type: 'radio', checked: mode === 'sft', click: () => restart('sft') },
      { label: 'V2/G6 · 规则策略', type: 'radio', checked: mode === 'g6-rule', click: () => restart('g6-rule') },
      { label: 'V2/G6 · Qwen规划', type: 'radio', checked: mode === 'g6-plan', click: () => restart('g6-plan') },
      { label: 'V2/G6 · Qwen规划与评审', type: 'radio', checked: mode === 'g6-review', click: () => restart('g6-review') },
      { label: 'Full888 · 规则策略', type: 'radio', checked: mode === 'full888-rule', click: () => restart('full888-rule') },
      { label: 'Full888 · Qwen规划', type: 'radio', checked: mode === 'full888-plan', click: () => restart('full888-plan') },
      { label: 'Full888 · Qwen规划与评审', type: 'radio', checked: mode === 'full888-review', click: () => restart('full888-review') },
      { label: 'Full888安全修复 · 规则策略', type: 'radio', checked: mode === 'full888-safe-rule', click: () => restart('full888-safe-rule') },
      { label: 'Full888安全修复 · Qwen规划', type: 'radio', checked: mode === 'full888-safe-plan', click: () => restart('full888-safe-plan') },
      { label: 'Full888安全修复 · Qwen规划与评审', type: 'radio', checked: mode === 'full888-safe-review', click: () => restart('full888-safe-review') },
      { label: 'Qwen联合意图与执行前审查 · 原型', type: 'radio', checked: mode === 'qwen-prototype', click: () => restart('qwen-prototype') },
      { label: 'Qwen规划与评审 · 请求依据校验', type: 'radio', checked: mode === 'qwen-grounded', click: () => restart('qwen-grounded') },
      { type: 'separator' }, { label: '重新启动服务', click: () => restart(mode) },
      { label: '退出', click: () => app.quit() }
    ] },
    { label: '编辑', submenu: [{ role: 'undo' }, { role: 'redo' }, { type: 'separator' }, { role: 'cut' }, { role: 'copy' }, { role: 'paste' }, { role: 'selectAll' }] },
    { label: '视图', submenu: [{ role: 'reload' }, { role: 'resetZoom' }, { role: 'zoomIn' }, { role: 'zoomOut' }] }
  ]));
  if (smoke) await smokeTest(current, mode);
}

async function restart(mode) {
  if (switching) return;
  switching = true;
  try { await boot(mode); } catch (e) { await dialog.showMessageBox({ type: 'error', message: e.message }); }
  finally { switching = false; }
}

async function smokeTest(current, mode) {
  const report = path.resolve(smoke);
  if (['process-death','process-restore'].includes(flag('--delivery-smoke'))) {
    await require('./process_death_smoke.cjs')(win,current,mode,report,flag('--delivery-smoke')==='process-restore');
    app.quit();return;
  }
  if (flag('--delivery-smoke') === 'qwen-prototype') {
    await require('./qwen_prototype_smoke.cjs')(win,current,mode,report);
    if (owned) await owned.stop(); app.exit(0); return;
  }
  fs.mkdirSync(path.dirname(report), { recursive: true });
  if (['dflash','dflash-restore'].includes(flag('--delivery-smoke'))) {
    await require('./dflash_smoke.cjs')(win,current,mode,report,flag('--delivery-smoke')==='dflash-restore');
    app.quit();return;
  }
  if (flag('--delivery-smoke') === 'fault') {
    await require('./fault_smoke.cjs')(win,current,mode,report);
    app.quit();return;
  }
  if (flag('--delivery-smoke') === 'restore') {
    await require('./restore_smoke.cjs')(win,current,mode,report);
    app.quit();return;
  }
  if (flag('--delivery-smoke') === 'model') {
    await require('./model_smoke.cjs')(win,current,mode,report);
    app.quit();return;
  }
  // This is application-owned integration test code in a hidden window, not
  // automation of another user's app. Exercise the same form/HTTP handlers.
  const result = await win.webContents.executeJavaScript(`(async () => {
    for (let n=0;n<100 && !document.body.dataset.ready;n++) await new Promise(r=>setTimeout(r,100));
    if (!document.body.dataset.ready) throw Error('renderer not ready');
    const json=async(path,body)=>{const r=await fetch(path,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});if(!r.ok)throw Error(path+':'+r.status);return r.json()};
    const before=await json('/status');
    const restored=before.messages.length;
    const send=async text=>{document.querySelector('#input').value=text;document.querySelector('#composer').requestSubmit();for(let n=0;n<1800;n++){await new Promise(r=>setTimeout(r,100));if(!document.querySelector('#send').disabled)return;}throw Error('send timeout')};
    await send('咳嗽三天，头疼一天。');
    const first=await json('/status');
    await send('主要想问头痛，刚才说错了，头疼是两天。');
    const second=await json('/status');
    await json('/api/profile',{age:'25',sex:'男',past_history:['人工测试画像']});
    const profile=await json('/api/profile');
    await send('现在胸痛而且无法呼吸。');
    const high=await json('/status');
    const sessions=await json('/desktop/api/sessions');
    const trace=await json('/api/debug/trace');
    return {restored_messages:restored,first_messages:first.messages.length,second_messages:second.messages.length,
      high_risk:high.case_state.risk_level,high_action:high.case_state.last_action,
      profile_saved:profile.age==='25',sessions:sessions.items.length,trace_events:trace.events.length,
      node_absent:typeof require==='undefined' && typeof process==='undefined',renderer_ready:document.body.dataset.ready,
      semantic:!!high.case_state.semantic_contract,fact_extraction_error:high.case_state.fact_extraction_error||null,
      safety_notices:Number(document.body.dataset.safetyNotices||0),
      safety_before_final:Number(document.body.dataset.safetyReceivedMs)>0 &&
        Number(document.body.dataset.safetyReceivedMs)<=Number(document.body.dataset.finalReceivedMs),
      visible_messages:document.querySelectorAll('.message').length};
  })()`);
  const unauthorized = await fetch(current.origin + '/status');
  const originDenied = await fetch(current.origin + '/reset', { method: 'POST', headers: {
    'X-Medical-Desktop': current.token, Origin: 'https://example.invalid' } });
  const prefs = win.webContents.getLastWebPreferences();
  Object.assign(result, { mode, unauthorized_status: unauthorized.status, cross_origin_status: originDenied.status,
    context_isolation: prefs.contextIsolation, sandbox: prefs.sandbox, node_integration: prefs.nodeIntegration,
    backend_pid: current.child.pid, origin: current.origin });
  await new Promise(resolve => setTimeout(resolve, 400));
  // A compositor capture failure must not erase completed HTTP/DOM evidence.
  // Keep functional verification and visual capture status separately visible.
  fs.writeFileSync(report, JSON.stringify(result, null, 2));
  try {
    const image = await win.webContents.capturePage();
    fs.writeFileSync(report.replace(/\.json$/, '.png'), image.toPNG());
    result.visual_capture = 'saved';
  } catch (error) { result.visual_capture = 'unavailable'; result.visual_capture_error = error.message; }
  fs.writeFileSync(report, JSON.stringify(result, null, 2));
  if (result.high_risk !== 'HIGH' || result.high_action !== 'risk_escalation' || !result.profile_saved ||
      !result.node_absent || !result.safety_before_final || result.safety_notices<1 || result.unauthorized_status !== 403 || result.cross_origin_status !== 403 ||
      (mode !== 'rule' && (!result.semantic || result.fact_extraction_error))) throw new Error('desktop smoke failed; inspect report');
  app.quit();
}

app.on('window-all-closed', () => { if (!switching) app.quit(); });
app.on('before-quit', e => {
  if(owned)record(owned.data,'main','before_quit',{mode:settings.mode,expected:true});
  if (closing || !owned) return;
  e.preventDefault(); closing = true;
  owned.stop().finally(() => app.quit());
});
app.whenReady().then(async () => {
  fs.mkdirSync(app.getPath('userData'), { recursive: true });
  settingsPath = path.join(app.getPath('userData'), 'desktop-settings.json');
  settings = fs.existsSync(settingsPath) ? JSON.parse(fs.readFileSync(settingsPath, 'utf8')) : {};
  const mode = flag('--mode') || settings.mode || 'rule';
  if (!['rule', 'sft', 'g6-rule', 'g6-plan', 'g6-review','full888-rule','full888-plan','full888-review','full888-safe-rule','full888-safe-plan','full888-safe-review','qwen-prototype','qwen-grounded'].includes(mode)) throw new Error('未知运行模式');
  await boot(mode);
}).catch(async e => {
  if (smoke) { fs.writeFileSync(path.resolve(smoke) + '.error.json', JSON.stringify({ error: e.message })); if (owned) await owned.stop(); app.exit(1); }
  else { await dialog.showMessageBox({ type: 'error', message: e.message, detail: 'L2桌面版需要已有Python工程；4B模式另需运行vLLM和SSH隧道。' }); app.quit(); }
});
