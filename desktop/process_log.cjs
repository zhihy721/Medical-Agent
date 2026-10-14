'use strict';
const fs=require('node:fs');
const path=require('node:path');
// Allow only classifications, never URLs, headers, argv or request contents.
function record(data, role, event, details={}) {
  fs.mkdirSync(data,{recursive:true});
  const safe={at:new Date().toISOString(),role,event};
  for(const key of ['reason','exitCode','mode','expected','pid','signal'])
    if(details[key]!==undefined)safe[key]=details[key];
  fs.appendFileSync(path.join(data,'process-events.jsonl'),JSON.stringify(safe)+'\n');
}
module.exports={record};
