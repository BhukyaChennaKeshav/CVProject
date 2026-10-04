"""Convert doc_forensics.py (# %% cell markers) into a Colab-ready Jupyter notebook."""
import json
import re
import sys

src = sys.argv[1] if len(sys.argv) > 1 else 'doc_forensics.py'
dst = sys.argv[2] if len(sys.argv) > 2 else 'DocForensics_Colab.ipynb'
text = open(src, encoding='utf-8').read()

cells = []
for chunk in re.split(r'^# %%', text, flags=re.M):
    if not chunk.strip():
        continue
    header, _, body = chunk.partition('\n')
    if header.strip().startswith('[markdown]'):
        lines = [ln[2:] if ln.startswith('# ') else ln.lstrip('#') for ln in body.strip('\n').split('\n')]
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': '\n'.join(lines).splitlines(True)})
    else:
        title = header.strip()
        code = (f'# {title}\n' if title else '') + body.strip('\n') + '\n'
        cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                      'source': code.splitlines(True)})

nb = {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5,
      'metadata': {'colab': {'provenance': [], 'name': dst},
                   'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                   'language_info': {'name': 'python'}}}
json.dump(nb, open(dst, 'w', encoding='utf-8'), indent=1)
print(f'wrote {dst}: {len(cells)} cells')
