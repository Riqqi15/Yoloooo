"""One-off collection and review sheets; never assigns training labels."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent
STAGE = ROOT / 'station_review_batch'
STAGE.mkdir(exist_ok=True)
pages = {}
for filename in ('commons_platform_candidates.json', 'commons_door_candidates.json', 'commons_bandung_candidates.json'):
    pages.update(json.loads((ROOT / filename).read_text(encoding='utf-8'))['query']['pages'])
excluded_words = ('halte', 'transjakarta', 'transjakarta', 'pulo gebang')
previous = ('Passenger waiting at Juanda Station', 'Peron 1 Stasiun Sudirman',
            "Cimekar Station's Tracks", 'Empty Platform Tegalluar',
            'Suasana Stasiun Cikampek di Malam', 'Istora Mandiri station platform')
selected = [p for p in pages.values()
            if not any(w in p['title'].lower() for w in excluded_words)
            and not any(w in p['title'] for w in previous)
            and not p['title'].startswith(('File:CC ', 'File:Argo ', 'File:Brantas'))]

def fetch(p):
    info = p['imageinfo'][0]
    path = STAGE / (str(p['pageid']) + '.jpg')
    if not path.exists():
        time.sleep(5)
        command = ['curl.exe', '--fail', '--location', '--max-time', '25',
                   '--silent', '--show-error', '--user-agent',
                   'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
                   '--output', str(path), info['url']]
        attempt = subprocess.run(command, capture_output=True, text=True)
        if '429' in attempt.stderr:
            print('Rate limited; waiting 60 seconds', flush=True)
            time.sleep(60)
            attempt = subprocess.run(command, capture_output=True, text=True)
        if attempt.returncode:
            raise RuntimeError(attempt.stderr.strip())
    with Image.open(path) as im:
        im.load()
        dimensions = im.size
    result = {'id': p['pageid'], 'title': p['title'], 'filename': path.name,
              'source': info['descriptionurl'], 'download_url': info['url'],
              'metadata': info['extmetadata'], 'dimensions': dimensions,
              'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'status': 'pending_visual_review'}
    print('Downloaded', p['pageid'], p['title'], flush=True)
    return result

records = []
for p in selected:
    try:
        records.append(fetch(p))
        (STAGE / 'sources.json').write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')
    except Exception as e:
        print('FAILED', p['title'], str(e), flush=True)
        if '429' in str(e):
            print('Stopping remote requests after repeated rate limit', flush=True)
            break
records.sort(key=lambda r: r['id'])
(STAGE / 'sources.json').write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')
for start in range(0, len(records), 12):
    sheet = Image.new('RGB', (1600, 1260), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, r in enumerate(records[start:start+12]):
        with Image.open(STAGE / r['filename']) as raw:
            thumb = ImageOps.exif_transpose(raw).convert('RGB')
            thumb.thumbnail((390, 375))
        x, y = (i % 4)*400, (i//4)*420
        sheet.paste(thumb, (x, y))
        draw.text((x+4, y+377), str(r['id']), fill='black')
        draw.text((x+4, y+395), r['title'][5:55], fill='black')
    sheet.save(STAGE / ('sheet_%02d.jpg' % (start//12+1)), quality=90)
assert len({r['id'] for r in records}) == len(records)
print('REVIEW READY', len(records), flush=True)
