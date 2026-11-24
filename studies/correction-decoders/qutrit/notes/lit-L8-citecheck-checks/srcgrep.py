"""Search arXiv e-print sources in memory (no extraction). Usage: srcgrep.py <src.bin> <ctx> <regex> [<regex> ...]"""
import sys, tarfile, gzip, io, re
path, ctx = sys.argv[1], int(sys.argv[2])
pats = [re.compile(p, re.I | re.S) for p in sys.argv[3:]]
raw = open(path, 'rb').read()
texts = []
try:
    data = gzip.decompress(raw)
except Exception:
    data = raw
try:
    tf = tarfile.open(fileobj=io.BytesIO(data))
    for m in tf.getmembers():
        if m.isfile() and m.name.lower().endswith(('.tex', '.bbl', '.txt', '.sty_no')):
            texts.append((m.name, tf.extractfile(m).read().decode('utf-8', 'replace')))
except tarfile.ReadError:
    texts.append(('<single>', data.decode('utf-8', 'replace')))
if not pats:
    for n, t in texts:
        print(n, len(t))
for p in pats:
    print('##### PATTERN:', p.pattern)
    hits = 0
    for n, t in texts:
        for mm in p.finditer(t):
            hits += 1
            a, b = max(0, mm.start() - ctx), min(len(t), mm.end() + ctx)
            print('--- [%s @%d]' % (n, mm.start()))
            print(' '.join(t[a:b].split()))
            if hits >= 6: break
        if hits >= 6: break
    if hits == 0:
        print('   (no match)')
