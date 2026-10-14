'use strict';
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
(async () => {
  const { packager } = await import('@electron/packager');
  const outIndex = process.argv.indexOf('--out-name');
  const outName = outIndex < 0 ? '' : process.argv[outIndex + 1];
  if (outIndex >= 0 && (!outName || !/^[a-zA-Z0-9_-]{1,64}$/.test(outName)))
    throw new Error('--out-name 只能使用字母、数字、下划线或短横线。');
  const runtime = path.join(__dirname, '.runtime');
  fs.mkdirSync(runtime, { recursive: true });
  const location = path.join(runtime, 'project-location.json');
  fs.writeFileSync(location, JSON.stringify({ project: path.resolve(__dirname, '..'),
    scope: 'L2 relies on this existing Python workspace, not a standalone backend bundle' }));
  const version = require('./node_modules/electron/package.json').version;
  const zipName = `electron-v${version}-win32-x64.zip`;
  const cache = path.join(runtime, 'electron-cache');
  const zip = fs.readdirSync(cache, { recursive: true }).find(p => path.basename(p) === zipName);
  if (!zip) throw new Error('先初始化Electron官方运行文件，再打包。');
  const zipPath = path.join(cache, zip);
  const expected = require('./node_modules/electron/checksums.json')[zipName];
  if (crypto.createHash('sha256').update(fs.readFileSync(zipPath)).digest('hex') !== expected)
    throw new Error('Electron官方ZIP校验失败。');
  const output = await packager({ dir: __dirname, name: 'Medical-Agent', platform: 'win32', arch: 'x64',
    out: path.join(__dirname, 'dist', outName), overwrite: false, asar: true, prune: true,
    electronZipDir: path.dirname(zipPath),
    extraResource: [location, path.join(__dirname, 'ui')], ignore: [/\.runtime/, /dist/, /test_/, /__pycache__/, /\.py$/] });
  console.log(JSON.stringify({ output, level: 'L2', pythonIncluded: false }));
})().catch(e => { console.error(e.message); process.exitCode = 1; });
