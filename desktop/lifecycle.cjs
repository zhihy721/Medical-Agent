'use strict';
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { spawn, execFile } = require('node:child_process');
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const { record } = require('./process_log.cjs');

function sameOrigin(url, origin) {
  try { return new URL(url).origin === origin; } catch { return false; }
}

async function startBackend({ project, data, ui, mode = 'rule', timeoutMs = 45000 }) {
  project = path.resolve(project); data = path.resolve(data);
  const python = path.join(project, '.venv', 'Scripts', 'python.exe');
  const entry = path.join(project, 'desktop', 'backend.py');
  if (!fs.existsSync(python) || !fs.existsSync(entry)) throw new Error('请选择含 .venv 和 desktop/backend.py 的 Medical-Agent 工程。');
  fs.mkdirSync(data, { recursive: true });
  const readyFile = path.join(data, `ready-${crypto.randomUUID()}.json`);
  const token = crypto.randomBytes(32).toString('hex');
  const log = fs.openSync(path.join(data, 'backend.log'), 'a');
  const child = spawn(python, ['-X', 'utf8', entry, '--project', project, '--data', data,
    '--ready-file', readyFile, '--mode', mode, '--ui', path.resolve(ui)], {
    cwd: project, windowsHide: true, stdio: ['pipe', log, log],
    env: { ...process.env, PYTHONUNBUFFERED: '1' }
  });
  fs.closeSync(log);
  let spawnError, stopping;
  record(data,'backend','spawn',{pid:child.pid,mode});
  child.on('error', err => { spawnError = err;record(data,'backend','spawn_error',{reason:err.code,mode}); });
  child.on('exit', (exitCode,signal) => record(data,'backend','exit',{exitCode,signal,expected:!!stopping,mode,pid:child.pid}));
  child.stdin.on('error', () => {});
  child.stdin.write(JSON.stringify({ token }) + '\n');
  async function stop() {
    if (stopping) return stopping;
    stopping = (async () => {
      if (child.exitCode === null && !spawnError) {
        child.stdin.end('shutdown\n');
        const limit = Date.now() + 5000;
        while (child.exitCode === null && Date.now() < limit) await sleep(50);
        if (child.exitCode === null) {
          await new Promise(resolve => execFile('taskkill', ['/PID', String(child.pid), '/T', '/F'],
            { windowsHide: true }, () => resolve()));
        }
      }
      if (fs.existsSync(readyFile)) fs.unlinkSync(readyFile);
    })();
    return stopping;
  }
  try {
    const end = Date.now() + timeoutMs;
    while (Date.now() < end) {
      if (spawnError) throw spawnError;
      if (child.exitCode !== null) throw new Error('后端启动失败，请查看此工作区的 backend.log。');
      if (fs.existsSync(readyFile)) {
        let info;
        try { info = JSON.parse(fs.readFileSync(readyFile, 'utf8')); } catch { await sleep(100); continue; }
        // Windows venv python.exe may be a redirector whose child runs Python.
        // Accept only the directly owned PID or its immediate child, then
        // verify the unpredictable capability against the ready endpoint.
        if ((info.pid !== child.pid && info.parent_pid !== child.pid) || !Number.isInteger(info.port) || info.port <= 0 || info.port > 65535) throw new Error('启动进程身份不匹配。');
        const origin = `http://127.0.0.1:${info.port}`;
        const response = await fetch(origin + '/desktop/health', {
          headers: { 'X-Medical-Desktop': token }, signal: AbortSignal.timeout(3000)
        });
        if (response.ok && (await response.json()).mode === mode) return { child, origin, token, stop, data };
      }
      await sleep(100);
    }
    throw new Error('后端未在45秒内就绪，请检查 Python 环境和 backend.log。');
  } catch (err) { await stop(); throw err; }
}

module.exports = { startBackend, sameOrigin };
