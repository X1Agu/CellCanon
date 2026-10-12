from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import threading
import urllib.request
from cellcanon.resources.sciplex_knowledge import sha256, write_json
ALLOWED_HOSTS = {'rest.uniprot.org', 'www.ebi.ac.uk', 'pubchem.ncbi.nlm.nih.gov', 'eutils.ncbi.nlm.nih.gov', 'ftp.ncbi.nlm.nih.gov', 'rest.genenames.org', 'reactome.org', 'download.reactome.org', 'rest.ensembl.org'}

class PublicSnapshot:

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.calls = 0
        self._locks = {}
        self._guard = threading.Lock()
        self._next_request = {}

    def fetch(self, url, *, suffix='json', timeout=45, json_body=None):
        body = None if json_body is None else json.dumps(json_body, sort_keys=True, separators=(',', ':')).encode()
        identity = url if body is None else 'POST\n' + url + '\n' + body.decode()
        with self._guard:
            lock = self._locks.setdefault(identity, threading.Lock())
        with lock:
            return self._fetch(url, suffix=suffix, timeout=timeout, body=body, identity=identity)

    def _fetch(self, url, *, suffix='json', timeout=45, body=None, identity=None):
        parsed = urllib.parse.urlparse(url)
        assert parsed.scheme == 'https' and parsed.hostname in ALLOWED_HOSTS
        assert body is None or (parsed.hostname == 'rest.ensembl.org' and parsed.path == '/lookup/id')
        key = hashlib.sha256((identity or url).encode()).hexdigest()
        path = self.root / (key + '.' + suffix)
        receipt = self.root / (key + '.receipt.json')
        if path.exists() and receipt.exists():
            record = json.loads(receipt.read_text())
            assert record['url'] == url and sha256(path) == record['sha256']
            assert record.get('request_body_sha256') == (hashlib.sha256(body).hexdigest() if body is not None else None)
            return path
        errors = []
        for attempt in range(3):
            host = parsed.hostname
            interval = 0.4 if host in {'pubchem.ncbi.nlm.nih.gov', 'eutils.ncbi.nlm.nih.gov'} else 0.1
            with self._guard:
                scheduled = max(time.monotonic(), self._next_request.get(host, 0))
                self._next_request[host] = scheduled + interval
            time.sleep(max(0, scheduled - time.monotonic()))
            self.calls += 1
            try:
                request = urllib.request.Request(url, data=body, headers={'User-Agent': 'CellCanon-BackgroundResourceAudit/1.0', 'Accept': 'application/json' if suffix == 'json' else 'text/plain', 'Content-Type': 'application/json'})
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    assert urllib.parse.urlparse(response.url).hostname in ALLOWED_HOSTS
                    payload = response.read(200000001)
                    headers = dict(response.headers)
                assert len(payload) <= 200000000
                if suffix == 'json':
                    json.loads(payload)
                elif suffix == 'tsv':
                    assert b'\t' in payload.split(b'\n', 1)[0], 'Expected TSV header'
                temp = path.with_suffix(path.suffix + '.tmp')
                temp.write_bytes(payload)
                temp.replace(path)
                write_json(receipt, {'url': url, 'retrieved_utc': datetime.now(timezone.utc).isoformat(), 'sha256': sha256(path), 'bytes': len(payload), 'headers': headers, 'attempts': attempt + 1, 'method': 'GET' if body is None else 'POST', 'request_body': None if body is None else json.loads(body), 'request_body_sha256': None if body is None else hashlib.sha256(body).hexdigest(), 'previous_errors': errors, 'scope': 'public biological/chemical background; no expression response retrieval'})
                return path
            except urllib.error.HTTPError as error:
                errors.append({'type': 'HTTPError', 'code': error.code})
                if error.code in {400, 404, 405}:
                    write_json(self.root / (key + '.failure.json'), {'url': url, 'errors': errors, 'observed_utc': datetime.now(timezone.utc).isoformat()})
                    raise
            except (OSError, ValueError) as error:
                errors.append({'type': type(error).__name__, 'message': str(error)[:300]})
            time.sleep(attempt + 1)
        write_json(self.root / (key + '.failure.json'), {'url': url, 'errors': errors, 'observed_utc': datetime.now(timezone.utc).isoformat()})
        raise RuntimeError('Public retrieval failed: ' + url)

    def json(self, url):
        return json.loads(self.fetch(url).read_text())

    def post_json(self, url, body):
        return json.loads(self.fetch(url, json_body=body).read_text())

    def chembl_query(self, endpoint, parameters, field):
        base = 'https://www.ebi.ac.uk/chembl/api/data/'
        url = base + endpoint + '.json?' + urllib.parse.urlencode(parameters)
        values, seen, total = ([], set(), None)
        while url:
            assert url not in seen
            seen.add(url)
            data = self.json(url)
            page = data['page_meta']
            if total is None:
                total = int(page['total_count'])
            assert total == int(page['total_count']) and int(page['offset']) == len(values)
            values.extend(data[field])
            url = urllib.parse.urljoin(base, page['next']) if page.get('next') else None
        assert len(values) == total
        return values

import pandas as pd
import urllib.parse
import urllib.error
from .sciplex_knowledge import OUTPUT

def uniprot(output=OUTPUT):
    root = output / 'public_background'
    client = PublicSnapshot(root / 'http')
    fields = ['accession', 'id', 'organism_id', 'gene_primary', 'gene_synonym', 'protein_name', 'cc_function', 'cc_pathway', 'cc_subcellular_location', 'xref_geneid', 'go_p', 'go_f', 'go_c']
    url = 'https://rest.uniprot.org/uniprotkb/stream?' + urllib.parse.urlencode({'query': '(organism_id:9606) AND (reviewed:true)', 'format': 'tsv', 'fields': ','.join(fields)})
    path = client.fetch(url, suffix='tsv', timeout=60)
    table = pd.read_csv(path, sep='\t', dtype=str).fillna('')
    assert table['Organism (ID)'].eq('9606').all()
    assert table['Entry'].is_unique and len(table) > 19000
    table.to_parquet(root / 'uniprot_human_reviewed.parquet', index=False)
    write_json(root / 'uniprot_manifest.json', {'status': 'HUMAN_REVIEWED_UNIPROT_BACKGROUND_COMPLETE', 'rows': len(table), 'fields': fields, 'url': url, 'raw_path': str(path), 'raw_sha256': sha256(path), 'output_sha256': sha256(root / 'uniprot_human_reviewed.parquet'), 'http_attempts': client.calls, 'documentation': 'https://www.uniprot.org/help/api_queries'})
    print(json.dumps({'event': 'uniprot_complete', 'rows': len(table), 'http_attempts': client.calls}), flush=True)
