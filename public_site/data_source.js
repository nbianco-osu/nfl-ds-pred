/* One immutable-in-memory snapshot keeps cross-panel cohorts consistent. */
(() => {
 const original=window.fetch.bind(window);
 const hosted=location.hostname==='nfl-ds-pred.nbianco0918.chatgpt.site';
 let remoteFiles=null;
 const status=document.createElement('p');status.className='refresh-detail';status.id='cloudDataStatus';status.setAttribute('role','status');
 document.querySelector('.app-header').append(status);
 const load=hosted?original('https://nbianco-osu.github.io/nfl-ds-pred/data/live_snapshot.json',{cache:'no-store',signal:AbortSignal.timeout(15000)})
   .then(async r=>{if(!r.ok)throw Error();const d=await r.json();if(d.schema_version!==1||!d.files?.['predictions.json']||!d.files?.['forecast_scorecard.json'])throw Error();
     for(const [key,value] of Object.entries(d.files)){if(typeof value!=='string'||!/^[-a-zA-Z0-9_.]+\.(json|csv)$/.test(key))throw Error();if(key.endsWith('.json'))JSON.parse(value);}
     remoteFiles=d.files;status.textContent=`Cloud data revision ${d.content_sha256.slice(0,12)}. Forecast observation times remain shown with each prediction.`;return d.files;})
   .catch(()=>{status.textContent='Cloud data unavailable. Showing the bundled release snapshot; check forecast observation times.';return null;}):Promise.resolve(null);
 if(!hosted)status.textContent='Data from this published dashboard revision.';
 window.dashboardFetch=async(input,options)=>{
   const url=new URL(typeof input==='string'?input:input.url,location.href);
   if(url.origin!==location.origin||!url.pathname.includes('/data/'))return original(input,options);
   const files=await load,name=url.pathname.split('/').pop();
   if(files){if(!(name in files))return new Response('Data absent from cloud revision',{status:404});return new Response(files[name],{headers:{'content-type':name.endsWith('.json')?'application/json':'text/csv'}});}
   return original(input,options);
 };
 document.addEventListener('click',event=>{
   const link=event.target.closest('a[download]');if(!remoteFiles||!link)return;
   const url=new URL(link.href),name=url.pathname.split('/').pop();
   if(!url.pathname.includes('/data/')||!(name in remoteFiles))return;
   event.preventDefault();const blob=URL.createObjectURL(new Blob([remoteFiles[name]],{type:name.endsWith('.json')?'application/json':'text/csv'}));
   const a=document.createElement('a');a.href=blob;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(blob),1000);
 });
})();
