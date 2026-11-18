

def main():
    import requests,tarfile,pathlib,json,time,sys
    sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
    import vpaths
    base=vpaths.WORK/'data';base.mkdir(parents=True,exist_ok=True)
    u='https://zenodo.org/records/8239000/files/updated_analyses_PSRJ0030_up_to_2018_NICER_data.tar.gz?download=1'
    r=requests.get(u,stream=True,timeout=(30,180));r.raise_for_status()
    t=tarfile.open(fileobj=r.raw,mode='r|gz');out=base/'j0030';out.mkdir(exist_ok=True);manifest=[]
    for i,m in enumerate(t):
     if i%200==0:print(i,m.name,flush=True)
     if '/PDT_U/NICERxXMM/PDTU_LR_1klp_et0p1_se0p8/' in m.name and any(q in m.name for q in ['post_equal_weights.dat','CustomPrior','main.py','stats.dat']):
      if m.isfile() and m.size<150000000:
       dest=out/(m.name.replace('/','__'));dest.write_bytes(t.extractfile(m).read());manifest.append({'member':m.name,'size':m.size,'path':str(dest) });print('SAVED',m.name,m.size,flush=True)
       (base/'j0030_manifest.json').write_text(json.dumps(manifest,indent=2))
       # The first PDT-U joint run is the published headline (despite "preliminary" in its prefix).
       # Its quantiles and log evidence reproduce Table 7. Stop as soon as it is obtained.
       if m.name.endswith('post_equal_weights.dat'):break
    (base/'j0030_manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()
