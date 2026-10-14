"""Prepare the 100M-token fiction/education corpus and Q&A data."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
from urllib.request import urlopen

from .book_catalog import FICTION_BOOKS, EDUCATION_BOOKS
from .tokenizer import BPETokenizer, EOS_TOKEN
from .instruction import question_key


def prepare_books(output, catalog=None):
    """Build a books-only corpus, keeping whole held-out works out of training."""
    output = Path(output)
    catalog = FICTION_BOOKS if catalog is None else catalog
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = {"train": output, "validation": output.with_name(output.stem + "-valid.txt")}
    if len({book[0] for book in catalog}) != len(catalog):
        raise ValueError("Book catalog must not contain duplicate editions")
    originals = output.parent / "books-sources"
    originals.mkdir(exist_ok=True)
    pending = {split: path.with_suffix(path.suffix + ".part") for split, path in paths.items()}
    records = []
    try:
        with ExitStack() as stack:
            streams = {split: stack.enter_context(path.open("w", encoding="utf-8"))
                       for split, path in pending.items()}
            for ebook, title, author, split in catalog:
                if split not in streams:
                    raise ValueError(f"Invalid book split: {split}")
                url = f"https://www.gutenberg.org/ebooks/{ebook}.txt.utf-8"
                source = originals / f"{ebook}.txt"
                if source.exists():
                    raw = source.read_bytes()
                else:
                    print(f"Downloading {title}: {url}", flush=True)
                    with urlopen(url, timeout=60) as response:
                        raw = response.read()
                cleaned = clean_gutenberg(raw.decode("utf-8-sig"))
                source.write_bytes(raw)
                streams[split].write(cleaned + EOS_TOKEN + "\n\n")
                records.append({"title": title, "author": author, "ebook": ebook,
                                "split": split, "url": url, "source": str(source),
                                "characters": len(cleaned), "sha256": hashlib.sha256(raw).hexdigest()})
        for split, path in paths.items():
            pending[split].replace(path)
            print(f"Saved {path}: {path.stat().st_size:,} bytes", flush=True)
    finally:
        for path in pending.values():
            path.unlink(missing_ok=True)
    metadata = {"dataset": "Classic books: Project Gutenberg", "books": records,
                "train_books": sum(book["split"] == "train" for book in records),
                "validation_books": sum(book["split"] == "validation" for book in records),
                "split": "Whole works; held out: " + ", ".join(book["title"] for book in records if book["split"] == "validation"),
                "rights": "Original Gutenberg files and licenses preserved in sources directory"}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def clean_gutenberg(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    start = re.search(r"^\*\*\* START OF THE PROJECT GUTENBERG EBOOK .*?\*\*\*[^\n]*\n",
                      text, re.MULTILINE)
    end = re.search(r"^\*\*\* END OF THE PROJECT GUTENBERG EBOOK .*?\*\*\*",
                    text, re.MULTILINE)
    if start is None or end is None or start.end() >= end.start():
        raise ValueError("Missing or invalid Gutenberg boundaries; refusing to prepare corpus")
    return text[start.end():end.start()].strip() + "\n"



def prepare_seed_books(root):
    prepare_books(root/'books.txt', FICTION_BOOKS)
    prepare_books(root/'education-books.txt', EDUCATION_BOOKS)


CATALOG_URL = 'https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv'
EDUCATION_URL = 'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/000_00000.parquet'
QA_URL = 'https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk/resolve/main/data/train-00000-of-00004.parquet'


def download(url, path):
    if path.exists(): return
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.part')
    try:
        with urlopen(url, timeout=90) as source, pending.open('wb') as target:
            shutil.copyfileobj(source, target, 1024 * 1024)
            expected = source.headers.get('Content-Length')
        if expected and pending.stat().st_size != int(expected): raise OSError('Incomplete download')
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def parquet_rows(path, columns):
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(path).iter_batches(batch_size=128, columns=columns):
        yield from batch.to_pylist()


def token_count(path, tokenizer):
    count, lines = 0, []
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            lines.append(line)
            if len(lines) == 256:
                count += sum(len(x.ids) for x in tokenizer.tokenizer.encode_batch(lines, add_special_tokens=False)); lines=[]
        count += sum(len(x.ids) for x in tokenizer.tokenizer.encode_batch(lines, add_special_tokens=False))
    return count


def digest(text): return hashlib.sha256(' '.join(text.split()).encode()).hexdigest()
def title_key(title): return re.sub(r'[^a-z0-9]', '', title.casefold().split(':')[0])


def fiction_candidates(catalog, excluded_ids, excluded_titles):
    for row in catalog:
        if (row['Type'] != 'Text' or row['Language'] != 'en'
                or not re.search(r'\bfiction\b', row['Subjects'], re.I)
                or re.search(r'criticism|study and teaching', row['Subjects'], re.I)
                or int(row['Text#']) in excluded_ids or title_key(row['Title']) in excluded_titles): continue
        yield row


def prepare_fiction(root, tokenizer, target):
    catalog = root/'expanded-sources/pg_catalog.csv'; download(CATALOG_URL, catalog)
    base_paths = [root/'books.txt', root/'books-valid.txt']
    counts = {'train_tokens':token_count(base_paths[0],tokenizer),'validation_tokens':token_count(base_paths[1],tokenizer)}
    known = {digest(text) for p in base_paths for text in p.read_text().split(EOS_TOKEN) if text.strip()}
    entries = list(FICTION_BOOKS) + list(EDUCATION_BOOKS)
    ids, titles = {b[0] for b in entries}, {title_key(b[1]) for b in entries}
    records, failures = [], []
    paths = [root/'expanded-fiction.txt', root/'expanded-fiction-valid.txt']
    pending = [p.with_suffix('.txt.part') for p in paths]
    def fetch(row):
        ebook=int(row['Text#']); url=f'https://www.gutenberg.org/ebooks/{ebook}.txt.utf-8'
        path=root/'expanded-sources/books'/f'{ebook}.txt'
        try:
            download(url,path)
            text=clean_gutenberg(path.read_text(encoding='utf-8-sig'))
            if len(text)<10000: raise ValueError('Too short')
            return row,text,url,None
        except (OSError,ValueError,UnicodeError) as error:
            return row,None,url,str(error)
    try:
        with ExitStack() as stack:
            streams=[stack.enter_context(p.open('w',encoding='utf-8')) for p in pending]
            for source,destination in zip(base_paths,streams):
                with source.open(encoding='utf-8') as incoming: shutil.copyfileobj(incoming,destination)
            with catalog.open(encoding='utf-8-sig',newline='') as incoming, ThreadPoolExecutor(max_workers=4) as pool:
                rows=iter(fiction_candidates(csv.DictReader(incoming),ids,titles))
                while counts['train_tokens']<target:
                    batch=[]
                    for _ in range(16):
                        row=next(rows,None)
                        if row is None: break
                        key=title_key(row['Title'])
                        if key in titles: continue
                        titles.add(key);batch.append(row)
                    if not batch: raise ValueError('Not enough eligible fiction in the catalog')
                    for row,text,url,error in pool.map(fetch,batch):
                        if error: failures.append(dict(ebook=row['Text#'],error=error));continue
                        identity=digest(text)
                        if identity in known: continue
                        known.add(identity)
                        split='validation' if int(identity[:8],16)%20==0 else 'train'
                        document=text+EOS_TOKEN+'\n\n';streams[split=='validation'].write(document)
                        counts[split+'_tokens']+=len(tokenizer.encode(document))
                        records.append(dict(ebook=int(row['Text#']),title=row['Title'],authors=row['Authors'],
                                            subjects=row['Subjects'],url=url,sha256=identity,split=split))
                    print(f"Fiction: {counts['train_tokens']:,} tokens; {len(records)} added books",flush=True)
        for source,target_path in zip(pending,paths): source.replace(target_path)
    finally:
        for p in pending:p.unlink(missing_ok=True)
    (root/'expanded-fiction.json').write_text(json.dumps(dict(**counts,books=records,skipped=failures,
        seed_books='books.json',catalog=CATALOG_URL,rights='Original Gutenberg files and licenses retained in expanded-sources/books'),indent=2)+'\n')
    return counts


def educational_text(row):
    text=row['text'].strip()
    url=row.get('url','').lower()
    if (not 400<=len(text)<=24000 or EOS_TOKEN in text or 'gutenberg' in url
            or sum(c.isalpha() for c in text)/len(text)<.55):return None
    # Omit maths-focused pages, while retaining ordinary science prose.
    if (re.search(r'\b(mathematics|algebra|geometry|calculus|arithmetic|trigonometry|quadratic|polynomial|derivative|integrals?|probability|multiplication)\b',text[:400],re.I)
            or re.search(r'math|algebra|geometry|calculus|arithmetic',url)
            or re.search(r'\\(frac|sum|int|begin)\b|\d\s*[+*/=]\s*\d',text)):return None
    return text


def prepare_education(root,tokenizer,target):
    source=root/'expanded-sources/education.parquet';download(EDUCATION_URL,source)
    paths=[root/'expanded-education.txt',root/'expanded-education-valid.txt']
    pending=[p.with_suffix('.txt.part') for p in paths];counts=dict(train_tokens=0,validation_tokens=0,documents=0)
    seen=set()
    try:
        with ExitStack() as stack:
            streams=[stack.enter_context(p.open('w',encoding='utf-8')) for p in pending]
            audit=stack.enter_context((root/'expanded-sources/education-attribution.jsonl').open('w',encoding='utf-8'))
            for stem in ['education-books']:
                for index,suffix in enumerate(['.txt','-valid.txt']):
                    p=root/(stem+suffix)
                    counts[('train','validation')[index]+'_tokens']+=token_count(p,tokenizer)
                    with p.open(encoding='utf-8') as incoming:shutil.copyfileobj(incoming,streams[index])
            for row in parquet_rows(source,['text','url','id']):
                text=educational_text(row)
                if text is None:continue
                identity=digest(text)
                if identity in seen:continue
                seen.add(identity);split='validation' if int(identity[:8],16)%100==0 else 'train'
                document=text+EOS_TOKEN+'\n\n';streams[split=='validation'].write(document)
                counts[split+'_tokens']+=len(tokenizer.encode(document));counts['documents']+=1
                audit.write(json.dumps(dict(id=row['id'],url=row['url'],sha256=identity,split=split))+'\n')
                if counts['train_tokens']>=target:break
            if counts['train_tokens']<target:raise ValueError('Not enough educational prose')
        for a,b in zip(pending,paths):a.replace(b)
    finally:
        for p in pending:p.unlink(missing_ok=True)
    (root/'expanded-education.json').write_text(json.dumps(dict(**counts,source=EDUCATION_URL,
        license='ODC-BY database; original web pages retain their own rights',
        seed_books=['education-books.json'],
        attribution='expanded-sources/education-attribution.jsonl'),indent=2)+'\n')
    return counts


def qa_record(row):
    m=row['messages']
    if len(m)<2 or [x['role'] for x in m[:2]]!=['user','assistant']:return None
    q,a=[x['content'].strip() for x in m[:2]]
    if (not 5<=len(q)<=400 or not 20<=len(a)<=1200 or q.count('?')>1 or EOS_TOKEN in q+a or '```' in q+a
        or not re.match(r'^(what|why|how|who|where|when|can|do|does|is|are|will|could|would|should|which)\b',q,re.I)
        or re.search(r'\d|[=+*/]|\b(calculate|solve|equation|derivative|integral|algebra|arithmetic|percent|multiply|divide|probability)\b',q+a,re.I)
        or re.search(r'your response|exact phrase|forbidden|keywords|response should|number of words|reply with|\b(programming|python|java|javascript|code|algorithm|script|sql)\b',q,re.I)):return None
    return dict(question=q,answer=a,source='HuggingFaceTB/smol-smoltalk',subset=row.get('source'))


def prepare_qa(root):
    seen=set();counts=dict(train=0,validation=0);sources=[]
    paths=[root/'instruction.jsonl',root/'instruction-valid.jsonl'];pending=[p.with_suffix('.jsonl.part') for p in paths]
    try:
        with ExitStack() as stack:
            streams=[stack.enter_context(p.open('w',encoding='utf-8')) for p in pending]
            for index in range(4):
                path=root/'qa-sources'/('train.parquet' if index==0 else f'train-{index}.parquet')
                url=QA_URL.replace('00000-of',f'{index:05d}-of');download(url,path);sources.append(url)
                for row in parquet_rows(path,['messages','source']):
                    record=qa_record(row)
                    if record is None:continue
                    key=question_key(record['question'])
                    if key in seen:continue
                    seen.add(key);split='validation' if int(digest(key)[:8],16)%20==0 else 'train'
                    streams[split=='validation'].write(json.dumps(record,ensure_ascii=False)+'\n');counts[split]+=1
        if min(counts.values())<1:raise ValueError('No usable Q&A split')
        for a,b in zip(pending,paths):a.replace(b)
    finally:
        for p in pending:p.unlink(missing_ok=True)
    (root/'instruction.json').write_text(json.dumps(dict(**counts,sources=sources,license='Apache-2.0',split='Normalized questions deduplicated, hash 95/5'),indent=2)+'\n')
    print(f'Q&A examples: {counts}',flush=True)


def trim_training_pool(path, tokenizer, target):
    """Keep complete documents until near the target; seed editions come first."""
    pending=path.with_suffix('.txt.part')
    count, current, kept = 0, [], set()
    try:
        with path.open(encoding='utf-8') as incoming, pending.open('w',encoding='utf-8') as output:
            for line in incoming:
                if EOS_TOKEN in line:
                    # Prepared documents always put EOS at the end of a line.
                    before, after=line.split(EOS_TOKEN,1)
                    current.append(before)
                    if after.strip():raise ValueError('Unexpected text after document EOS')
                    text=''.join(current).strip();current=[]
                    if not text:continue
                    document=text+'\n'+EOS_TOKEN+'\n\n'
                    size=sum(len(r.ids) for r in tokenizer.tokenizer.encode_batch(document.splitlines(keepends=True), add_special_tokens=False))
                    if count+size <= target:
                        output.write(document);count+=size;kept.add(digest(text))
                else:
                    current.append(line)
            if any(line.strip() for line in current):raise ValueError('Incomplete training document')
        if count < target*.995:
            raise ValueError(f'Only {count:,} tokens fit the {target:,}-token pool; increase candidate targets')
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)
    return count,kept


def finalize_pool(root, tokens=100000000):
    fiction={k:v for k,v in json.loads((root/'expanded-fiction.json').read_text()).items() if k.endswith('_tokens')}
    education={k:v for k,v in json.loads((root/'expanded-education.json').read_text()).items() if k.endswith('_tokens')}
    # Learn the final vocabulary from training candidates, never validation.
    # Freeze it before selecting the final pool so recorded counts remain exact.
    tokenizer=BPETokenizer.train_file([root/'expanded-fiction.txt',root/'expanded-education.txt'])
    final={}
    for name,share in [('fiction',.8),('education',.2)]:
        count,kept=trim_training_pool(root/f'expanded-{name}.txt',tokenizer,int(tokens*share))
        final[name]=count
        provenance=root/f'expanded-{name}.json'
        info=json.loads(provenance.read_text());info['final_train_tokens']=count
        if name=='fiction':
            for record in info['books']:
                record['included']=record['split']=='validation' or record['sha256'] in kept
        provenance.write_text(json.dumps(info,indent=2)+'\n')
    (root/'expanded-tokenizer.json').write_text(tokenizer.to_json())
    print(f'Final training pool: {sum(final.values()):,} tokens; {final}',flush=True)
    manifest=dict(dataset='Expanded 80% fiction / 20% education',min_tokens=int(tokens*.99),max_tokens=int(tokens*1.01),
                  target_tokens=tokens,train_tokens=sum(final.values()),source_tokens=final,tokenizer='expanded-tokenizer.json',
                  bootstrap_counts=dict(fiction=fiction,education=education),sources=[
                      dict(name=name,weight=weight,train=f'expanded-{name}.txt',validation=f'expanded-{name}-valid.txt',provenance=f'expanded-{name}.json')
                      for name,weight in [('fiction',.8),('education',.2)]])
    (root/'expanded.json').write_text(json.dumps(manifest,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',default='data')
    parser.add_argument('--only',choices=['both','corpus','qa'],default='both')
    parser.add_argument('--tokens',type=int,default=100000000,help='Final unique training-token target (80/20)')
    args=parser.parse_args();root=Path(args.data_dir);root.mkdir(parents=True,exist_ok=True)
    if args.tokens<1:parser.error('Token targets must be positive')
    if args.only in ('both','corpus'):
        if not all((root/name).exists() for name in ('books.txt','books-valid.txt','education-books.txt','education-books-valid.txt')):
            prepare_seed_books(root)
        token_path=root/'expanded-sources/bootstrap-tokenizer.json';token_path.parent.mkdir(exist_ok=True)
        if token_path.exists():tokenizer=BPETokenizer.from_json(token_path.read_text())
        else:
            tokenizer=BPETokenizer.train_file([root/'books.txt',root/'education-books.txt'])
            token_path.write_text(tokenizer.to_json())
        fiction=prepare_fiction(root,tokenizer,int(args.tokens*.85))
        education=prepare_education(root,tokenizer,int(args.tokens*.24))
        finalize_pool(root,args.tokens)
    if args.only in ('both','qa'):prepare_qa(root)


if __name__=='__main__':main()
