import {mkdir,readFile,readdir,cp,rm,writeFile} from 'node:fs/promises';
import {dirname,join,resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
const root=dirname(fileURLToPath(import.meta.url)),dist=resolve(root,'dist');
if(dirname(dist)!==root)throw Error('Invalid build destination');
const manifest=JSON.parse(await readFile(join(root,'data/dashboard_manifest.json'),'utf8'));
if(manifest.schema_version!==2)throw Error('Missing revision manifest');
for(const [name,e] of Object.entries(manifest.files)){
  if(!/^[-a-zA-Z0-9_.]+\.(json|csv)$/.test(name)||!/^revisions\/[0-9a-f]{64}\.(json|csv)$/.test(e.path))throw Error('Unsafe dataset path');
  const content=await readFile(join(root,'data',e.path));
  if(content.length!==e.bytes||createHash('sha256').update(content).digest('hex')!==e.sha256)throw Error(`Corrupt dataset: ${name}`);
}
await rm(dist,{recursive:true,force:true});await mkdir(join(dist,'data/revisions'),{recursive:true});
for(const file of await readdir(root))if(/\.(html|css|js)$/.test(file))await cp(join(root,file),join(dist,file));
await cp(join(root,'vendor'),join(dist,'vendor'),{recursive:true});
await cp(join(root,'data/dashboard_manifest.json'),join(dist,'data/dashboard_manifest.json'));
for(const e of Object.values(manifest.files))await cp(join(root,'data',e.path),join(dist,'data',e.path));
await writeFile(join(dist,'_headers'),'/data/revisions/*\n  Cache-Control: public, max-age=31536000, immutable\n/data/dashboard_manifest.json\n  Cache-Control: no-cache\n');
console.log(`Static dashboard built: ${Object.keys(manifest.files).length} verified datasets; revision ${manifest.revision}`);
