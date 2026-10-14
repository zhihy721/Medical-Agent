'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const os = require('node:os');
const { startBackend, sameOrigin } = require('./lifecycle.cjs');

test('same origin rejects suffixes, wrong ports and executable schemes', () => {
  const origin = 'http://127.0.0.1:5000';
  assert.equal(sameOrigin(origin + '/desktop/', origin), true);
  for (const url of ['http://127.0.0.1:5001/','http://127.0.0.1.evil:5000/','javascript:alert(1)','file:///C:/secret'])
    assert.equal(sameOrigin(url, origin), false);
});

test('G6 mode starts from the shared configuration without promoting old sessions', { timeout: 60000 }, async () => {
  const root = path.resolve(__dirname, '..');
  const data = fs.mkdtempSync(path.join(os.tmpdir(), 'medical-agent-g6-test-'));
  let service;
  try {
    service = await startBackend({ project: root, data, ui: path.join(__dirname, 'ui'), mode: 'g6-rule' });
    const response = await fetch(service.origin + '/desktop/health', { headers: { 'X-Medical-Desktop': service.token } });
    assert.equal(response.status, 200);
    assert.equal((await response.json()).mode, 'g6-rule');
    const config = fs.readFileSync(path.join(data, 'config.env'), 'utf8');
    for (const setting of ['AGENT_PLANNER_MODE=rule', 'AGENT_REVIEWER_MODE=rule',
      'AGENT_EXTRACTOR_PROMPT_VERSION=g6_v7', 'AGENT_REPRESENTATION_MODE=finite_v5',
      'AGENT_EXTRACTOR_RECOVERY_MODE=checked_v16']) assert.ok(config.includes(setting), setting);
    assert.ok(!config.includes('AGENT_EXTRACTOR_PROMPT_VERSION=locked'));
  } finally { if (service) await service.stop(); }
});

test('owned backend auth, profile and session persistence, independent instance cleanup', { timeout: 60000 }, async () => {
  const root = path.resolve(__dirname, '..');
  const data = fs.mkdtempSync(path.join(os.tmpdir(), 'medical-agent-desktop-test-'));
  const options = { project: root, data, ui: path.join(__dirname, 'ui') };
  let a, b, c;
  const request = async (service, url, body, cookie = '') => {
    const response = await fetch(service.origin + url, { method: body ? 'POST' : 'GET',
      headers: { 'X-Medical-Desktop': service.token, ...(cookie ? { Cookie: cookie } : {}),
        ...(body ? { 'Content-Type': 'application/json' } : {}) },
      body: body ? JSON.stringify(body) : undefined });
    assert.equal(response.status, 200, url);
    return { json: await response.json(), cookie: response.headers.get('set-cookie')?.split(';')[0] || cookie };
  };
  try {
    a = await startBackend(options);
    assert.equal((await fetch(a.origin + '/status')).status, 403);
    assert.equal((await fetch(a.origin + '/reset', { method: 'POST', headers: {
      'X-Medical-Desktop': a.token, Origin: 'https://example.invalid' } })).status, 403);
    const status = await request(a, '/status');
    const first = await request(a, '/chat', { message: '咳嗽三天。' }, status.cookie);
    const oldSid = (await request(a, '/desktop/api/sessions', undefined, first.cookie)).json.current;
    await request(a, '/api/profile', { age: '26', sex: '男', past_history: ['测试画像'] }, first.cookie);
    const reset = await request(a, '/reset', {}, first.cookie);
    const restored = await request(a, '/desktop/api/open', { id: oldSid }, reset.cookie);
    assert.equal(restored.json.messages.length, 2);
    const newUser = await request(a, '/status');
    assert.equal((await fetch(a.origin + '/desktop/api/open', { method: 'POST',
      headers: { 'X-Medical-Desktop': a.token, Cookie: newUser.cookie, 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: oldSid }) })).status, 404);
    b = await startBackend({ ...options, data: path.join(data, 'independent') });
    const bOrigin = b.origin;
    await a.stop(); a = null;
    assert.equal((await request(b, '/desktop/health')).json.ready, true);
    c = await startBackend(options);
    const reopened = await request(c, '/status', undefined, restored.cookie);
    assert.equal(reopened.json.messages.length, 2);
    assert.equal((await request(c, '/api/profile', undefined, reopened.cookie)).json.age, '26');
    assert.notEqual(c.origin, bOrigin);
  } finally { for (const s of [a,b,c]) if (s) await s.stop(); }
});
