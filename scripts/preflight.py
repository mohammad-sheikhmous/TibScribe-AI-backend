from pathlib import Path
import json, re, sys
root=Path(__file__).resolve().parents[1]
errors=[]; warnings=[]
def ok(cond,msg,kind='error'):
    if not cond:(errors if kind=='error' else warnings).append(msg)
# project structure
for rel in ['laravel-backend/artisan','laravel-backend/routes/doctor-api.php','ai-service/app/main.py','ai-service/EXPERTA_MED/engine.py']:
    ok((root/rel).exists(),f'Missing {rel}')
# service secrets
if (root/'.env').exists():
    vals={}
    for line in (root/'.env').read_text().splitlines():
        if '=' in line and not line.startswith('#'): k,v=line.split('=',1); vals[k]=v
    ok(len(vals.get('AI_SERVICE_TOKEN',''))>=32 and 'CHANGE_ME' not in vals.get('AI_SERVICE_TOKEN',''),'AI_SERVICE_TOKEN is missing/weak')
    ok(vals.get('LARAVEL_APP_KEY','').startswith('base64:') and 'CHANGE_ME' not in vals.get('LARAVEL_APP_KEY',''),'LARAVEL_APP_KEY is not generated')
else: warnings.append('Root .env not created yet; run: python scripts/generate_secrets.py')

# Composer dependency reproducibility. The integrated composer.json differs from the
# historical Laravel repository, so copying the old lockfile blindly would create a
# content-hash mismatch. Generate/commit a fresh lockfile from this integrated tree
# on a connected machine before production release.
if not (root/'laravel-backend/composer.lock').exists():
    warnings.append('laravel-backend/composer.lock is missing; Composer can resolve dependencies, but the Laravel build is not fully reproducible until a fresh lockfile is generated for this integrated composer.json and committed.')

# Keep source/runtime declarations aligned with dependency constraints.
ai_docker=(root/'ai-service/Dockerfile').read_text() if (root/'ai-service/Dockerfile').exists() else ''
ci=(root/'.github/workflows/ci.yml').read_text() if (root/'.github/workflows/ci.yml').exists() else ''
req=(root/'ai-service/requirements.lock').read_text() if (root/'ai-service/requirements.lock').exists() else ''
ok('FROM python:3.12' in ai_docker, 'AI Docker runtime must use Python 3.12 for the locked scientific stack')
ok(ci.count("python-version: '3.12'") >= 2, 'CI Python runtime has drifted from Python 3.12')
ok('fastapi==0.139.2' in req, 'requirements.lock must pin FastAPI 0.139.2')

# model bundle is a hard runtime requirement
model=root/'ai-service/model_output'
weights=model/'best_model.pt'
if not weights.exists(): errors.append('Missing ai-service/model_output/best_model.pt (20-label final AraBERT checkpoint).')
cfgf=model/'model_config.json'
if not cfgf.exists(): errors.append('Missing ai-service/model_output/model_config.json from the same training run as best_model.pt.')
else:
    try:
        cfg=json.loads(cfgf.read_text())
        ok(int(cfg.get('num_classes',-1))==20,'model_config.json num_classes must be 20')
        ok(bool(cfg.get('model_name')),'model_config.json is missing model_name')
        ok(bool(cfg.get('preprocessing')),'model_config.json is missing preprocessing provenance')
    except Exception as e: errors.append(f'Invalid model_config.json: {e}')
mapf=model/'label_mapping.json'
if mapf.exists():
    try:
        obj=json.loads(mapf.read_text()); labels=obj.get('label2id'); id2=obj.get('id2label')
        ok(isinstance(labels,dict) and len(labels)==20,'label_mapping.json label2id must contain exactly 20 labels')
        ok(isinstance(id2,dict) and len(id2)==20,'label_mapping.json id2label must contain exactly 20 labels')
        ok('pregnancy_nutrition' not in (labels or {}),'Retired pregnancy_nutrition label found in model bundle')
    except Exception as e: errors.append(f'Invalid label_mapping.json: {e}')
else: errors.append('Missing ai-service/model_output/label_mapping.json from the same training run as best_model.pt.')

# Offline/production-safe model bundle. train_arabert.py writes these artifacts from
# the same run; relying on a live Hugging Face download at every clinical startup is
# intentionally not accepted by preflight.
for dirname in ('bert', 'tokenizer'):
    path = model/dirname
    ok(path.is_dir() and any(path.iterdir()) if path.exists() else False,
       f'Missing/non-populated ai-service/model_output/{dirname}/ from the final training run.')
statsf=model/'train_stats.json'
if not statsf.exists():
    errors.append('Missing ai-service/model_output/train_stats.json required for the configured OOD uncertainty layer.')
else:
    try:
        stats=json.loads(statsf.read_text())
        dim=int(stats.get('dim',0)); mean=stats.get('mean'); precision=stats.get('precision')
        ok(dim>0 and isinstance(mean,list) and len(mean)==dim, 'train_stats.json mean/dim are inconsistent')
        ok(isinstance(precision,list) and len(precision)==dim, 'train_stats.json precision/dim are inconsistent')
    except Exception as e: errors.append(f'Invalid train_stats.json: {e}')
if cfgf.exists():
    try:
        _cfg=json.loads(cfgf.read_text())
        ok(isinstance(_cfg.get('calibration'),dict) and bool(_cfg.get('calibration')),
           'model_config.json is missing final validation-set calibration metadata')
    except Exception:
        pass
print('TibScribe preflight')
for m in warnings: print('[WARN]',m)
for m in errors: print('[FAIL]',m)
if not errors: print('[PASS] Runtime prerequisites look complete.')
sys.exit(1 if errors else 0)
