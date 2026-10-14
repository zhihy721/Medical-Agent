'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
test('late completed request is reconciled on health polling without resending',async()=>{
  const source=fs.readFileSync(path.join(__dirname,'ui/client.js'),'utf8');
  const elements=new Map(),storage=new Map();let queries=0,posts=0;
  const context=vm.createContext({document:{getElementById(id){if(!elements.has(id))elements.set(id,{disabled:false,hidden:false});return elements.get(id);}},
    localStorage:{getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v)},medicalRuntimeLabel:()=>'',
    fetch:async(path,opts)=>{if(opts?.method==='POST')posts++;if(path.includes('/chat/requests/'))queries++;
      return {ok:true,json:async()=>path.includes('/chat/requests/')?{status:queries===1?'pending':'completed'}:{mode:'qwen-prototype'}};}});
  // Real recovery and health functions, no renderer boot or model dependency.
  vm.runInContext(source.slice(0,source.indexOf("$('composer').onsubmit"))+source.slice(source.indexOf('async function health()'),source.indexOf('(async()=>{try{await health()')),context);
  vm.runInContext("refresh=async()=>{};sessionId='s1';savePending('r1');setBusy(false)",context);
  await vm.runInContext('recoverPending()',context);
  assert.equal(elements.get('send').disabled,true);
  await vm.runInContext('health()',context);
  assert.equal(elements.get('send').disabled,false);
  assert.equal(queries,2);assert.equal(posts,0);
});
