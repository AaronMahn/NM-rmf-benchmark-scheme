

def main():
    """Download selected primary Zenodo products and extract the exact used M,R samples.
    Run from any directory: python nicer_fit/fetch_data.py (writes under NNSTAR_VALIDATION_WORK).
    The 6.9GB J0030 archive is streamed only until its 8.6MB headline member is read.
    """
    from pathlib import Path
    import json,hashlib,requests,tarfile,subprocess,sys
    import numpy as np
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    import vpaths
    B=vpaths.WORK;B.mkdir(parents=True,exist_ok=True);D=B/'data';S=B/'samples';D.mkdir(exist_ok=True);S.mkdir(exist_ok=True)
    selected={'10519473':'mr_samples_and_contours.tar.gz','13766753':'headline_result_samples_and_contours.tar.gz'}
    manifest=[]
    for id,name in selected.items():
     r=requests.get('https://zenodo.org/api/records/'+id,timeout=60);r.raise_for_status();j=r.json();(D/(id+'.json')).write_text(json.dumps(j,indent=2))
     info=next(x for x in j['files'] if x['key']==name);f=D/name
     if not f.exists():
      with requests.get(info['links']['self'],stream=True,timeout=(30,180)) as r:
       r.raise_for_status()
       with f.open('wb') as out:
        for block in r.iter_content(1024*1024):out.write(block)
     if 'md5:' + hashlib.md5(f.read_bytes()).hexdigest()!=info['checksum']:raise RuntimeError('Zenodo checksum mismatch '+name)
     with tarfile.open(f) as t:
      m=next(x for x in t if x.name.endswith('mrsamples_post_equal_weights.dat') and not Path(x.name).name.startswith('._'))
      sample=np.loadtxt(t.extractfile(m));key='J0740' if id=='10519473' else 'J0437';np.savez_compressed(S/(key+'.npz'),samples=sample[:,:2])
      manifest.append({'source':key,'doi':'10.5281/zenodo.'+id,'archive':name,'archive_md5':info['checksum'],'member':m.name,'n_samples':len(sample)})
     for fi in j['files']:
      if 'readme' in fi['key'].lower():
       rr=requests.get(fi['links']['self'],timeout=60);rr.raise_for_status();(D/(id+'_README.md')).write_bytes(rr.content)
    if not list((D/'j0030').glob('*post_equal_weights.dat')):
     subprocess.run([sys.executable,str(Path(__file__).resolve().parent/'fetch_j0030.py')],check=True)
    f=next((D/'j0030').glob('*post_equal_weights.dat'));sample=np.loadtxt(f);np.savez_compressed(S/'J0030.npz',samples=sample[:,:2]);manifest.append({'source':'J0030','doi':'10.5281/zenodo.8239000','archive':'updated_analyses_PSRJ0030_up_to_2018_NICER_data.tar.gz','member':f.name.replace('__','/'),'n_samples':len(sample),'member_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'note':'Streamed to headline member; full 6.9GB archive checksum not checked.'})
    for row in manifest:
     f=S/(row['source']+'.npz');row['samples_sha256']=hashlib.sha256(f.read_bytes()).hexdigest()
    (B/'data_manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()
