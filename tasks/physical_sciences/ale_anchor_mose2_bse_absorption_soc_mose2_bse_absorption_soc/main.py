"""Task hooks and the real QE/BerkeleyGW workflow driver for MoSe2."""
from __future__ import annotations
import hashlib, importlib.util, json, shlex, sys, types
from dataclasses import dataclass
from pathlib import Path
try:
    import cua_bench as cb
    from tasks.common_setup import BaseTaskSetup
    from tasks.linux_runtime import LinuxTaskConfig
    _setup = BaseTaskSetup()
except ImportError:
    class _CB:
        class Task:
            def __init__(self, **kw): self.__dict__.update(kw)
        class DesktopSession: pass
        @staticmethod
        def tasks_config(split='train'): return lambda f:f
        @staticmethod
        def setup_task(split='train'): return lambda f:f
        @staticmethod
        def evaluate_task(split='train'): return lambda f:f
    cb = _CB()
    @dataclass
    class LinuxTaskConfig:
        DOMAIN_NAME: str = 'physical_sciences'; TASK_NAME: str = 'mose2_bse_absorption_soc'; VARIANT_NAME: str = 'base'; OS_TYPE: str = 'linux'
        REMOTE_ROOT_DIR: str = '/media/user/data/agenthle'
        REMOTE_OUTPUT_DIR: str = 'output'
        @property
        def task_dir(self): return f'{self.REMOTE_ROOT_DIR}/{self.DOMAIN_NAME}/{self.TASK_NAME}/{self.VARIANT_NAME}'
        @property
        def input_dir(self): return f'{self.task_dir}/input'
        @property
        def software_dir(self): return f'{self.task_dir}/software'
        @property
        def remote_output_dir(self): return f'{self.task_dir}/{self.REMOTE_OUTPUT_DIR}'
        @property
        def reference_dir(self): return f'{self.task_dir}/reference'
        def to_metadata(self): return {'task_dir':self.task_dir,'input_dir':self.input_dir,'software_dir':self.software_dir,'remote_output_dir':self.remote_output_dir,'reference_dir':self.reference_dir}
    class BaseTaskSetup:
        async def __call__(self,*a,**k): return None
    _setup = BaseTaskSetup()

ROOT = Path(__file__).resolve().parent
TASK_NAME = 'ale_anchor_mose2_bse_absorption_soc_mose2_bse_absorption_soc'
INPUT_FILES = ('problem_spec.md','runtime_manifest.json','MoSe2.vasp','Mo.soc.upf','Se.soc.upf')
OUTPUT_FILES = ('MoSe2_bands.dat.gnu','MoSe2_bands.png','absorption_eh.dat','exciton_absorption_spectra_avg.png')

def _load_script(name):
    package_name=f'{__name__}._task_scripts'
    if package_name not in sys.modules:
        package=types.ModuleType(package_name);package.__path__=[str(ROOT/'scripts')];sys.modules[package_name]=package
    module_name=f'{package_name}.{name}'
    spec=importlib.util.spec_from_file_location(module_name,ROOT/'scripts'/f'{name}.py')
    if spec is None or spec.loader is None: raise ImportError(f'cannot load task script: {name}')
    module=importlib.util.module_from_spec(spec);sys.modules[module_name]=module;spec.loader.exec_module(module);return module

# The evaluator keeps this canonical workflow private.  The task setup stages
# only inputs and the pinned runtime; the agent must construct and run the decks.
WORKFLOW_DRIVER = (ROOT / 'scripts' / 'evaluator_workflow.py').read_text()

# This literal is checked by evaluate() and by the independent oracle.  It
# prevents a submitted agent from replacing the staged workflow before the
# hidden rerun.
WORKFLOW_SHA256 = hashlib.sha256(WORKFLOW_DRIVER.encode()).hexdigest()

@dataclass
class Config(LinuxTaskConfig):
    DOMAIN_NAME: str='physical_sciences'; TASK_NAME: str=TASK_NAME; VARIANT_NAME: str='base'; OS_TYPE: str='linux'
    @property
    def task_description(self):
        return json.loads(Path(__file__).with_name("task_card.json").read_text(encoding="utf-8"))["taskPrompt"]
    def to_metadata(self):
        d=super().to_metadata(); d.update({'task_id':f'physical_sciences/{TASK_NAME}','input_files':list(INPUT_FILES),'required_output_files':list(OUTPUT_FILES),'work_dir':f'{self.task_dir}/workflow','manifest':f'{self.task_dir}/workflow/run_manifest.json','software_bin_dir':f'{self.software_dir}/bin'}); return d
config=Config()
@cb.tasks_config(split='train')
def load(): return [cb.Task(description=config.task_description,metadata=config.to_metadata(),computer={'provider':'computer','setup_config':{'os_type':'linux'}})]
@cb.setup_task(split='train')
async def start(task_cfg, session):
    await _setup(task_cfg, session)
    m = task_cfg.metadata
    for directory in (m["remote_output_dir"], m["work_dir"]):
        await session.interface.create_dir(directory)
    bootstrap = (f'QE_BGW_BIN={shlex.quote(m["software_bin_dir"])} bash '
                 f'{shlex.quote(m["software_dir"] + "/bootstrap_remote_software.sh")}')
    result = await session.run_command(bootstrap, check=False)
    rc = result.get("return_code", 1) if isinstance(result, dict) else getattr(result, "return_code", getattr(result, "returncode", 1))
    if int(rc) != 0:
        raise RuntimeError(f"pinned QE/BerkeleyGW bootstrap failed with return code {rc}")
@cb.evaluate_task(split='train')
async def evaluate(task_cfg,session):
    score_remote=_load_script('score_outputs').score_remote
    m=task_cfg.metadata
    # Pull only the four governed outputs and the run manifest; scoring is fully output-derived.
    try:
        out={name:await session.read_bytes(f"{m['remote_output_dir']}/{name}") for name in OUTPUT_FILES}
        manifest=await session.read_bytes(m['manifest'])
    except Exception: return [0.0]
    import tempfile
    async def local_manifest(raw, base):
        """Materialize remote deck paths so the local oracle can verify them."""
        data=json.loads(raw.decode()); deck_dir=base/'decks'; deck_dir.mkdir()
        for stage in data.get('stages',[]):
            remote=stage.get('deck_path','')
            if not remote: raise RuntimeError('manifest stage has no deck path')
            blob=await session.read_bytes(remote)
            local=deck_dir/(stage['stage']+'.in'); local.write_bytes(blob)
            stage['deck_path']=str(local.resolve()); data['decks'][stage['stage']]=str(local.resolve())
        source_dir=base/'sources'; source_dir.mkdir()
        for key, info in data.get('table_provenance',{}).items():
            remote=info.get('source_path','')
            if not remote: raise RuntimeError('table provenance has no source path')
            if not Path(remote).is_absolute(): raise RuntimeError('table provenance source must be absolute')
            try:
                if Path(remote).resolve().is_relative_to(Path(m['remote_output_dir']).resolve()):
                    raise RuntimeError('table provenance may not point inside submitted output')
            except AttributeError:
                if str(Path(remote).resolve()).startswith(str(Path(m['remote_output_dir']).resolve())+str(Path('/'))):
                    raise RuntimeError('table provenance may not point inside submitted output')
            blob=await session.read_bytes(remote)
            local=source_dir/(key+'.dat'); local.write_bytes(blob)
            info['source_path']=str(local.resolve()); info['source_sha256']=hashlib.sha256(blob).hexdigest()
            stage_name=info.get('stage')
            if stage_name in data.get('decks',{}):
                for rec in data.get('stages',[]):
                    if rec.get('stage')==stage_name:
                        rec['output_path']=str(local.resolve()); rec['output_sha256']=hashlib.sha256(blob).hexdigest()
        canonical={k:data.get(k) for k in ('schema_version','runtime_image','workflow_sha256','stages','settings','input_sha256','output_hashes','table_provenance')}
        data['attestation_sha256']=hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return data
    with tempfile.TemporaryDirectory(prefix='mose2_eval_') as td:
        p=Path(td); od=p/'output'; od.mkdir()
        for n,b in out.items(): (od/n).write_bytes(b)
        mf=p/'run_manifest.json'
        try: mf.write_text(json.dumps(await local_manifest(manifest,p),indent=2)+'\n')
        except Exception: return [0.0]
        result=score_remote(od,mf)
    # Hidden evaluation executes the same QE/BerkeleyGW driver again with a
    # changed physical broadening/range (MOSE2_HIDDEN_VARIANT), so a copied
    # visible table cannot receive the hidden component.
    hidden_score = 0.0
    try:
        hidden_out=f"{m['task_dir']}/hidden_output"; hidden_work=f"{m['task_dir']}/hidden_workflow"; evaluator_script='/tmp/mose2_canonical_workflow.py'
        await session.interface.create_dir(hidden_out); await session.interface.create_dir(hidden_work)
        await session.write_file(evaluator_script,WORKFLOW_DRIVER)
        await session.run_command(f'chmod 0700 {shlex.quote(evaluator_script)}',check=False)
        cmd=(f"MOSE2_HIDDEN_VARIANT=1 /usr/bin/python3 {shlex.quote(evaluator_script)} "
             f"--input-dir {shlex.quote(m['input_dir'])} --output-dir {shlex.quote(hidden_out)} "
             f"--work-dir {shlex.quote(hidden_work)} --bin-dir {shlex.quote(m['software_bin_dir'])}")
        run=await session.run_command(cmd,check=False)
        rc=run.get('return_code',1) if isinstance(run,dict) else getattr(run,'return_code',getattr(run,'returncode',1))
        if rc==0:
            with tempfile.TemporaryDirectory(prefix='mose2_hidden_eval_') as td:
                hp=Path(td); hod=hp/'output'; hod.mkdir()
                for name in OUTPUT_FILES: (hod/name).write_bytes(await session.read_bytes(f"{hidden_out}/{name}"))
                hmf=hp/'run_manifest.json'; hmf.write_text(json.dumps(await local_manifest(await session.read_bytes(f"{hidden_work}/run_manifest.json"),hp),indent=2)+'\n')
                hidden_score=score_remote(hod,hmf)
                hidden_manifest=json.loads(hmf.read_text())
                hs=hidden_manifest.get('settings',{})
                if (hs.get('hidden_variant') is not True or hs.get('energy_range')!='1.1 4.3' or
                    float(hs.get('energy_resolution',0))!=0.007 or float(hs.get('screened_cutoff_Ry',0))!=23):
                    hidden_score=0.0
                # A hidden rerun must be a genuinely changed protocol/output,
                # so a copied visible table cannot earn the hidden component.
                if hidden_score>0:
                    visible_hashes=json.loads(manifest.decode()).get('output_hashes',{})
                    hidden_hashes=hidden_manifest.get('output_hashes',{})
                    if (visible_hashes==hidden_hashes or
                        hidden_hashes.get('absorption_eh.dat')==visible_hashes.get('absorption_eh.dat')): hidden_score=0.0
    except Exception:
        hidden_score=0.0
    return [float(0.60*result+0.40*hidden_score)]
