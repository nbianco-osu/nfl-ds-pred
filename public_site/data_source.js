/* Pin one manifest; independently verify immutable datasets on demand. */
(() => {
  const original=window.fetch.bind(window), requests=new Map();
  const hosted=location.hostname==='nfl-ds-pred.nbianco0918.chatgpt.site';
  const local=new URL('./data/',location.href), remote=new URL('https://nbianco-osu.github.io/nfl-ds-pred/data/');
  const status=document.createElement('p');
  status.id='cloudDataStatus';status.className='refresh-detail';status.setAttribute('role','status');
  status.textContent='Loading verified data revision...';document.querySelector('.app-header').append(status);
  const state={source:'',revision:'',loaded:0,bytes:0,failures:[]};window.dashboardDelivery=state;
  const announce=()=>window.dispatchEvent(new CustomEvent('deliverychange',{detail:{...state}}));
  async function manifest(base){
    const r=await original(new URL('dashboard_manifest.json',base),{cache:'no-store',signal:AbortSignal.timeout(12000)});
    if(!r.ok)throw Error('Manifest unavailable');const data=await r.json();
    if(data.schema_version!==2||!/^[0-9a-f]{64}$/.test(data.revision)||!data.files?.['predictions.json']||!data.files?.['metrics.json'])throw Error('Invalid manifest');
    for(const [name,e] of Object.entries(data.files)){
      if(!/^[-a-zA-Z0-9_.]+\.(json|csv)$/.test(name)||!/^[0-9a-f]{64}$/.test(e.sha256)||e.path!==`revisions/${e.sha256}.${name.split('.').pop()}`||!Number.isSafeInteger(e.bytes)||e.bytes<0)throw Error('Invalid dataset reference');
    }
    return {data,base};
  }
  const ready=(async()=>{
    let selected;
    try{selected=await manifest(hosted?remote:local);state.source=hosted?'Cloud':'Published';}
    catch(error){if(!hosted)throw error;selected=await manifest(local);state.source='Bundled fallback';}
    state.revision=selected.data.revision;
    status.textContent=`${state.source} revision ${state.revision.slice(0,12)}. ${state.source==='Bundled fallback'?'Cloud unavailable; bundled data may be older. ':''}Forecast timestamps remain authoritative.`;
    announce();return selected;
  })();
  ready.catch(()=>{status.textContent='Data revision unavailable. Reload to retry; no forecasts have been substituted.';state.failures=['manifest'];announce();});
  async function dataset(name){
    if(!requests.has(name))requests.set(name,(async()=>{
      const {data,base}=await ready,e=data.files[name];if(!e)throw Error('Dataset absent from pinned revision');
      const r=await original(new URL(e.path,base),{cache:'force-cache',signal:AbortSignal.timeout(30000)});if(!r.ok)throw Error('Dataset unavailable');
      const bytes=await r.arrayBuffer();
      const sha=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');
      if(bytes.byteLength!==e.bytes||sha!==e.sha256)throw Error('Dataset integrity check failed');
      const content=new TextDecoder().decode(bytes);if(name.endsWith('.json'))JSON.parse(content);
      state.loaded++;state.bytes+=bytes.byteLength;announce();return content;
    })().catch(error=>{
      if(!state.failures.includes(name))state.failures.push(name);
      status.textContent=`${state.source||'Dashboard'} data incomplete: ${state.failures.join(', ')}. Reload to retry. Other panels retain their pinned revision.`;
      announce();throw error;
    }));
    return requests.get(name);
  }
  window.dashboardFetch=async(input,options)=>{
    const url=new URL(typeof input==='string'?input:input.url,location.href);
    if(url.origin!==location.origin||!url.pathname.includes('/data/'))return original(input,options);
    const name=url.pathname.split('/').pop();
    return new Response(await dataset(name),{headers:{'content-type':name.endsWith('.json')?'application/json':'text/csv'}});
  };
  document.addEventListener('click',async event=>{
    const link=event.target.closest('a[download]');if(!link)return;
    const url=new URL(link.href);if(url.origin!==location.origin||!url.pathname.includes('/data/'))return;
    event.preventDefault();const name=url.pathname.split('/').pop();
    try{const blob=URL.createObjectURL(new Blob([await dataset(name)]));const a=document.createElement('a');a.href=blob;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(blob),1000);}
    catch{/* Dataset failure already appears in the status region. */}
  });
})();
