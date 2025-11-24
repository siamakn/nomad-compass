from __future__ import annotations

import io
import json
import os
import re
import zipfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from fastapi import HTTPException

BASE_DIR = Path(__file__).resolve().parent

OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://172.28.105.142:11434')
DEFAULT_EMBED_MODEL = os.getenv('EMBED_MODEL', 'nomic-embed-text')
DEFAULT_CHAT_MODEL = os.getenv('CHAT_MODEL', 'gpt-oss:20b')
# By default, use a local "data" directory next to the "apis" package
# instead of downloading from GitHub.
DEFAULT_RDF_ZIP = os.getenv('RDF_SOURCE', str(BASE_DIR.parent / 'data'))

HTTP_OK = 200

TITLE_KEYS = [
    'name',
    'label',
    'title',
    'schema:name',
    'rdfs:label',
    'dc:title',
    'dct:title',
    'skos:prefLabel',
]
DESC_KEYS = [
    'description',
    'comment',
    'summary',
    'abstract',
    'schema:description',
    'rdfs:comment',
    'dc:description',
    'dct:description',
]


SYSTEM_PROMPT = '\n'.join(
    [
        'You are NOMAD Compass, an expert and friendly guide to the NOMAD',
        'materials science platform.',
        'Your knowledge comes ONLY from the supplied <docs> context.',
        'Follow these directives:',
        '',
        "- Ground every answer in <docs>. If the answer isn't in <docs>, say:",
        '  "I couldn\'t find a specific resource for that in my knowledge base."',
        '- Be helpful to beginners. Summarize clearly using bullets, bold key',
        '  terms, and short lists.',
        '- Identify and link resources explicitly when links are present in the',
        '  context.',
        '- Do NOT write code or perform tasks; your job is to guide users to',
        '  resources.',
        '- Stay focused on NOMAD resources. Avoid general programming or',
        '  materials science not present in <docs>.',
        '- Always end with a suggested next step drawn from the context.',
        '',
        'Return only the final answer; do NOT include chain-of-thought or',
        'internal reasoning.',
    ]
)


def _iter_json_files(root: Path):
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in {'.jsonld', '.json'}:
            yield p


def _load_json(fp: Path) -> Any | None:
    try:
        return json.loads(fp.read_text(encoding='utf-8'))
    except Exception:
        return None


def _extract_nodes(doc: Any) -> list[dict[str, Any]]:
    def strip_ctx(o: Any) -> Any:
        return (
            {k: v for k, v in o.items() if k != '@context'}
            if isinstance(o, dict) and '@context' in o
            else o
        )

    if isinstance(doc, dict):
        if isinstance(doc.get('@graph'), list):
            return [strip_ctx(n) for n in doc['@graph'] if isinstance(n, dict)]
        return [strip_ctx(doc)]
    if isinstance(doc, list):
        return [strip_ctx(n) for n in doc if isinstance(n, dict)]
    return []


def _key_ci(d: dict[str, Any], cands: list[str]) -> str | None:
    lower = {k.lower(): k for k in d}
    for c in cands:
        k = lower.get(c.lower())
        if k:
            return k
    for c in cands:
        for k in d:
            if k.lower().endswith(c.lower()):
                return k
    return None


def _best_title(n: dict[str, Any]) -> str | None:
    k = _key_ci(n, TITLE_KEYS)
    if k:
        v = n.get(k)
        if isinstance(v, list):
            v = next((x for x in v if isinstance(x, str)), v[0] if v else None)
        if isinstance(v, str):
            return v.strip()
    if isinstance(n.get('@id'), str) and n['@id'].strip():
        return n['@id'].strip()
    return None


def _best_desc(n: dict[str, Any]) -> str | None:
    k = _key_ci(n, DESC_KEYS)
    if k:
        v = n.get(k)
        if isinstance(v, list):
            v = next((x for x in v if isinstance(x, str)), v[0] if v else None)
        if isinstance(v, str):
            return v.strip()
    return None


def _slug(text: str) -> str:
    t = re.sub(r'[^a-zA-Z0-9\- ]+', '-', text.strip().lower()).replace(' ', '-')
    t = re.sub('-+', '-', t).strip('-')
    return t or 'section'


def jsonld_dir_to_markdown_str(root: Path) -> tuple[str, int]:
    files = sorted(_iter_json_files(root))
    all_nodes: list[dict[str, Any]] = []
    for fp in files:
        doc = _load_json(fp)
        if doc is None:
            continue
        for n in _extract_nodes(doc):
            n['_src'] = str(fp)
            all_nodes.append(n)

    seen, nodes = set(), []
    for n in all_nodes:
        nid = n.get('@id')
        key = ('__noid__', id(n)) if nid is None else ('id', nid)
        if key not in seen:
            seen.add(key)
            nodes.append(n)

    nodes.sort(key=lambda n: (_best_title(n) or '').lower())

    lines = ['# Combined Knowledge (from JSON-LD)', '']
    lines.append('## Table of Contents\n')
    for n in nodes:
        title = _best_title(n) or '(untitled)'
        lines.append(f'- [{title}](#{_slug(title)})')
    lines.append('')
    for n in nodes:
        title = _best_title(n) or '(untitled)'
        desc = _best_desc(n)
        src = n.get('_src', '')
        lines += [
            f'## {title}',
            f'<a id="{_slug(title)}"></a>',
            '',
            f'**Source:** `{src}`',
            '',
        ]
        if desc:
            lines += ['**Summary**', f'> {desc}', '']
        lines.append('**Properties**')
        wrote = False
        for k, v in n.items():
            if k in {'@context', '@graph', '@id', '@type', '_src'}:
                continue
            if isinstance(v, (str, int, float, bool)) or v is None:
                lines.append(f'- **{k}**: {"" if v is None else v}')
                wrote = True
        if not wrote:
            lines.append('- _(no simple scalar properties)_')
        lines.append('')
    return '\n'.join(lines), len(nodes)


def chunk_text(text: str, chunk_size=1200, chunk_overlap=200) -> list[str]:
    chunks, start = [], 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - chunk_overlap)
    return chunks


async def ollama_embed(
    texts: list[str], model: str, base_url: str = OLLAMA_BASE_URL
) -> np.ndarray:
    vecs: list[list[float]] = []
    async with httpx.AsyncClient(timeout=120) as client:
        for t in texts:
            r = await client.post(
                f'{base_url}/api/embeddings',
                json={'model': model, 'prompt': t},
            )
            if r.status_code != HTTP_OK:
                raise HTTPException(
                    status_code=502,
                    detail=f'Embedding failed: {r.status_code} {r.text[:300]}',
                )
            data = r.json()
            vecs.append(data['embedding'])
    return np.array(vecs, dtype=np.float32)


async def ollama_generate(
    prompt: str, model: str, temperature: float, base_url: str = OLLAMA_BASE_URL
) -> str:
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(
            f'{base_url}/api/generate',
            json={
                'model': model,
                'prompt': prompt,
                'stream': False,
                'options': {'temperature': temperature},
            },
        )
        if r.status_code != HTTP_OK:
            raise HTTPException(
                status_code=502,
                detail=f'Generation failed: {r.status_code} {r.text[:300]}',
            )
        data = r.json()
        return data.get('response', '').strip()


async def ollama_generate_stream(
    prompt: str, model: str, temperature: float, base_url: str = OLLAMA_BASE_URL
) -> AsyncIterator[str]:
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            'POST',
            f'{base_url}/api/generate',
            json={
                'model': model,
                'prompt': prompt,
                'stream': True,
                'options': {'temperature': temperature},
            },
        ) as resp:
            if resp.status_code != HTTP_OK:
                body = await resp.aread()
                raise HTTPException(
                    status_code=502,
                    detail=f'Stream failed: {resp.status_code} {body[:300]!r}',
                )
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                piece = obj.get('response')
                if piece:
                    yield piece
                if obj.get('done'):
                    break


def top_k_cosine(query_vec: np.ndarray, matrix: np.ndarray, k: int) -> list[int]:
    a = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    b = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    sims = b @ a
    if len(sims) == 0:
        return []
    k = min(k, len(sims))
    idx = np.argpartition(-sims, kth=k - 1)[:k]
    return list(idx[np.argsort(-sims[idx])])


def _find_rdf_folder(start: Path) -> Path:
    if start.name == 'RDF' and start.is_dir():
        return start
    candidates = list(start.rglob('RDF'))
    if not candidates:
        raise HTTPException(
            status_code=400,
            detail=f"Could not locate 'RDF' folder under: {start}",
        )
    return candidates[0]


async def download_extract_rdf(zip_url: str, tmp_dir: Path) -> Path:
    headers = {'User-Agent': 'nomad-compass/1.0'}
    token = os.getenv('GITHUB_TOKEN')
    if token:
        headers['Authorization'] = f'Bearer {token}'

    async with httpx.AsyncClient(
        timeout=180, follow_redirects=True, headers=headers
    ) as client:
        r = await client.get(zip_url)
        if r.status_code != HTTP_OK:
            detail =(
                f'Failed to fetch ZIP ({r.status_code} {r.reason_phrase}). '
                f'URL={zip_url}'
            )
            try:
                body = r.text[:500]
                detail += f' | Body: {body}'
            except Exception:
                pass
            raise HTTPException(status_code=502, detail=detail)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(tmp_dir)

    root = next((p for p in tmp_dir.iterdir() if p.is_dir()), tmp_dir)
    return _find_rdf_folder(root)


def strip_deepseek_think(s: str) -> str:
    return re.sub(r'<think>.*?</think>', '', s, flags=re.DOTALL).strip()
