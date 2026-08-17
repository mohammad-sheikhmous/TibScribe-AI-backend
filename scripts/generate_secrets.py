from pathlib import Path
import base64, secrets
root=Path(__file__).resolve().parents[1]
out=root/'.env'
if out.exists():
    raise SystemExit(f'{out} already exists; refusing to overwrite secrets.')
app_key='base64:'+base64.b64encode(secrets.token_bytes(32)).decode()
text=(root/'.env.example').read_text()
text=text.replace('LARAVEL_APP_KEY=CHANGE_ME',f'LARAVEL_APP_KEY={app_key}')
text=text.replace('AI_SERVICE_TOKEN=CHANGE_ME_WITH_AT_LEAST_32_RANDOM_CHARACTERS','AI_SERVICE_TOKEN='+secrets.token_urlsafe(48))
text=text.replace('MYSQL_PASSWORD=CHANGE_ME_DATABASE_PASSWORD','MYSQL_PASSWORD='+secrets.token_urlsafe(24))
text=text.replace('MYSQL_ROOT_PASSWORD=CHANGE_ME_ROOT_PASSWORD','MYSQL_ROOT_PASSWORD='+secrets.token_urlsafe(24))
out.write_text(text)
print(f'Created {out}')
