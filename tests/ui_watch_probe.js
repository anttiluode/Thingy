// Execute the actual explorer handler with controlled browser timers. Fetch
// calls hit the real Python engine. Only rendering/DOM layout is stubbed.
const fs = require('fs');
const vm = require('vm');
const [htmlPath, base, session] = process.argv.slice(2);
const html = fs.readFileSync(htmlPath, 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1].replace(/\nnewWorld\(\);/, '\n');
const elements = {}, timers = [];
const context = vm.createContext({
  console, Blob, URL,
  document: {getElementById(id) {
    return elements[id] ||= {textContent:'',style:{},disabled:false,addEventListener(){}};
  }},
  fetch: (url, options) => fetch(base + url, options),
  setTimeout(callback) {timers.push(callback); return timers.length;}
});
vm.runInContext(script, context);
vm.runInContext("render=()=>{};buttons=()=>{};session="+JSON.stringify(session)+";view={status:'thinking'};", context);
const tick = () => new Promise(resolve => setImmediate(resolve));
async function until(predicate) {
  const deadline=Date.now()+5000;
  while (!predicate()) {if(Date.now()>deadline)throw Error('Probe timed out');await tick();}
}
(async () => {
  elements.watchBtn.onclick();
  await until(() => timers.length === 1);
  elements.watchBtn.onclick(); // Pause while the original loop sleeps.
  elements.watchBtn.onclick(); // Restart before that sleeper wakes.
  await until(() => timers.length === 2);
  timers.shift()(); // Wake the OLD loop; it must not send another address.
  for(let i=0;i<5;i++)await tick();
  await until(() => !vm.runInContext('busy',context));
  const cycles=vm.runInContext('view.cycles',context);
  vm.runInContext('stopWatch()',context);
  while(timers.length)timers.shift()();
  console.log(JSON.stringify({cycles}));
})().catch(e=>{console.error(e);process.exitCode=1;});
