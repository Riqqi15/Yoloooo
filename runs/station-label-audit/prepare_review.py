"""Prepare a read-only label audit; never certify or export training data."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from training_label_gate import load_taxonomy, validate_annotation

SOURCE = ROOT / 'data/training/station_photo_set_50'
OUT = ROOT / 'runs/station-label-audit/pretraining-review'
OUT.mkdir(parents=True, exist_ok=True)
_, objects, tactile = load_taxonomy(ROOT / 'data/training/taxonomy_tactile_v1.json')
provenance = {r['filename']: r for r in json.loads((SOURCE / 'sources.json').read_text())}
protected = {hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'data/samples').glob('*') if p.is_file()}
notes = {
    '001_juanda': 'User confirmed tactile is present, superseding the earlier negative statement. Stored polygons still require final coverage review.',
    '002_sudirman': 'Foreground polygon refined and acknowledged by user. Opposite platform still needs visual assessment.',
    '040_137394087': 'Three polygons saved, including the added distant visible pad. Not a human-review certification.',
    '044_77601320': 'Six polygons saved, including the two added pads beside the divider. Photo 045 is queue marking, not this tactile example.',
    '017_106327604': 'AI visual screening: white ridged strip on the right side of the platform is a candidate for annotation; exclude columns and benches.',
    '026_116115102': 'AI visual screening: yellow textured strip on the facing platform needs annotation; examine farther platform strips separately.',
    '048_77563734': 'AI visual screening: yellow textured pad beside the escalator, partly behind standing people, needs annotation.',
    '009_137824158': 'Hold: platform edge texture is not sufficiently clear for positive or negative certification.',
    '010_137778724': 'Hold: platform edge texture is not sufficiently clear for positive or negative certification.',
    '032_166526831': 'Hold: distinguish the brown foreground panel beside the seating from tactile; inspect distant right platform separately.',
}
records = []
seen = set()
sheet = None
for i, path in enumerate(sorted(SOURCE.glob('*.jpg'))):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    im = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    w, h = im.size
    jp = path.with_suffix('.json')
    issues = []
    shapes = []
    if digest in seen or digest in protected:
        issues.append('duplicate or protected test content')
    seen.add(digest)
    status = 'no_json_needs_presence_review'
    if jp.exists():
        data = json.loads(jp.read_text(encoding='utf-8'))
        shapes = data['shapes']
        status = 'labeled_needs_visual_review' if shapes else 'explicit_empty_needs_review'
        if (data['imageWidth'], data['imageHeight']) != (w, h):
            issues.append('orientation/dimensions mismatch')
        if (jp.parent / data['imagePath']).resolve() != path.resolve():
            issues.append('imagePath mismatch')
        candidate = {'schema_version': 1, 'sample_id': digest, 'source_sha256': digest,
                     'objects': [], 'tactile': [{'label': s['label'], 'points': s['points']} for s in shapes]}
        try:
            assert all(s['shape_type'] == 'polygon' for s in shapes), 'non-polygon shape'
            validate_annotation(candidate, {'sample_id': digest, 'source_sha256': digest, 'dataset_task': 'tactile'}, w, h, objects, tactile)
        except (ValueError, AssertionError) as exc:
            issues.append(str(exc))
    note = notes.get(path.stem, '')
    record = {'filename': path.name, 'source_sha256': digest, 'width': w, 'height': h,
              'status': status, 'shape_count': len(shapes), 'issues': issues, 'notes': note,
              'annotation_sha256': hashlib.sha256(jp.read_bytes()).hexdigest() if jp.exists() else None,
              'provenance': provenance.get(path.name)}
    records.append(record)
    im.thumbnail((680, 490))
    draw = ImageDraw.Draw(im)
    for s in shapes:
        pts = [(x * im.width / w, y * im.height / h) for x, y in s['points']]
        if len(pts) >= 3:
            draw.line(pts + [pts[0]], fill='red', width=2)
    im.save(OUT / (path.stem + '.jpg'))
    if i % 9 == 0:
        sheet = Image.new('RGB', (2040, 1590), 'white')
    x, y = (i % 3) * 680, ((i % 9) // 3) * 530
    sheet.paste(im, (x, y))
    ImageDraw.Draw(sheet).text((x + 5, y + 495), path.name + ' | ' + str(len(shapes)) + ' polygons', fill='black')
    if i % 9 == 8 or i == 49:
        sheet.save(OUT / ('page_%02d.jpg' % (i // 9 + 1)))
assert len(records) == 50 and len(seen) == 50
report = {'training_ready': False, 'status_counts': dict(Counter(r['status'] for r in records)),
          'structural_issue_count': sum(bool(r['issues']) for r in records),
          'note': 'Missing JSON is not certified negative. No human review has been recorded by this audit.',
          'samples': records}
(OUT / 'audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps({k: v for k, v in report.items() if k != 'samples'}, indent=2))
for r in records:
    if r['issues']:
        print(r['filename'], r['issues'])
