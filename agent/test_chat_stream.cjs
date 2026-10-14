'use strict';
const assert=require('node:assert/strict');
const {readChatResponse}=require('../static/chat_stream.js');
function response(chunks){return new Response(new ReadableStream({start(controller){chunks.forEach(c=>controller.enqueue(c));controller.close();}}),{headers:{'Content-Type':'application/x-ndjson'}});}
(async()=>{
    const bytes=new TextEncoder().encode(JSON.stringify({kind:'safety_notice',response:'请先处理，不等查询。'})+'\n'+JSON.stringify({kind:'final',response:'安全建议\n医院不可用。'})+'\n');
    const seen=[];
    const final=await readChatResponse(response(Array.from(bytes,b=>new Uint8Array([b]))),text=>seen.push(text));
    assert.deepEqual(seen,['请先处理，不等查询。']);assert.equal(final.response,'安全建议\n医院不可用。');
    await assert.rejects(readChatResponse(response([new TextEncoder().encode('{"kind":"safety_notice","response":"安全提示"}\n')]),()=>{}),/incomplete/);
    await assert.rejects(readChatResponse(response([new TextEncoder().encode('{"kind":"error","response":"医院补充失败"}\n')]),()=>{}),/医院补充失败/);
    assert.equal((await readChatResponse(new Response('{"response":"兼容回复"}',{headers:{'Content-Type':'application/json'}}),()=>{})).response,'兼容回复');
    console.log('4 chat-stream decoder checks passed');
})().catch(error=>{console.error(error);process.exitCode=1;});
