"""Read-only staging diagnostic; a transport error must never mean file absence."""
import hashlib
import json
from pathlib import Path
import sys

root=Path(sys.argv[1])
names=json.loads(sys.argv[2])
report={'protocol':'reference-probe-v1','nonce':sys.argv[3],'root':str(root),'files':{},'nested':[]}
for name in names:
    p=root/name
    try:
        data=p.read_bytes()
    except FileNotFoundError:
        report['files'][name]={'status':'missing'}
        if (root/'reference'/name).is_file():report['nested'].append(name)
    except OSError as exc:
        report['files'][name]={'status':'io_error','type':type(exc).__name__,'errno':exc.errno}
    else:
        report['files'][name]={'status':'present','sha256':hashlib.sha256(data).hexdigest()}
print(json.dumps(report))
