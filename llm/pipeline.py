"""Pretrain a raw MLA checkpoint, then save a separate Q&A-tuned checkpoint."""
import argparse
import json
from .prepare import DEFAULT_TRAIN_TOKENS, FICTION_SHARE, EDUCATION_SHARE
from .tokenizer import DEFAULT_VOCAB_SIZE, BPETokenizer
from pathlib import Path
import shutil
import subprocess
import sys


def run(arguments):
    print('Running: ' + ' '.join([sys.executable, *arguments]), flush=True)
    subprocess.run([sys.executable, '-u', *arguments], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='books-200m-16l-16k')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--checkpoint-dir', default='checkpoints')
    parser.add_argument('--pretrain-steps', type=int, default=24000)
    parser.add_argument('--qa-steps', type=int, default=500)
    parser.add_argument('--device', default='auto', choices=['auto','cpu','mps','cuda'])
    parser.add_argument('--raw-checkpoint', help='Skip pretraining; copy this raw checkpoint and fine-tune it')
    parser.add_argument('--prepare', action='store_true', help='Rebuild data before training')
    args = parser.parse_args()
    if min(args.pretrain_steps,args.qa_steps)<1 or Path(args.name).name!=args.name:
        parser.error('Steps must be positive; name must be a filename')
    root, data = Path(args.checkpoint_dir), Path(args.data_dir)
    root.mkdir(parents=True,exist_ok=True)
    raw, tuned = root/f'{args.name}.raw.pt', root/f'{args.name}.tuned.pt'
    best_raw, best_tuned = raw.with_name(raw.stem+'.best.pt'), tuned.with_name(tuned.stem+'.best.pt')
    if any(p.exists() for p in (raw,tuned,best_raw,best_tuned)):
        parser.error('Output exists; choose a new --name to preserve previous checkpoints')
    ready = [data/'instruction.jsonl',data/'instruction-valid.jsonl']
    if not args.raw_checkpoint:
        ready.extend([data/'expanded.json',data/'expanded-tokenizer.json',data/'expanded-fiction.txt',data/'expanded-education.txt',
                      data/'expanded-fiction-valid.txt',data/'expanded-education-valid.txt'])
    data_ready = all(p.exists() for p in ready)
    if data_ready and not args.raw_checkpoint:
        data_ready = BPETokenizer.from_json((data/'expanded-tokenizer.json').read_text()).vocab_size == DEFAULT_VOCAB_SIZE
        if data_ready:
            manifest = json.loads((data/'expanded.json').read_text())
            weights = {s['name']: s['weight'] for s in manifest.get('sources', [])}
            data_ready = (manifest.get('target_tokens') == DEFAULT_TRAIN_TOKENS
                          and weights == {'fiction': FICTION_SHARE, 'education': EDUCATION_SHARE})
    if args.prepare or not data_ready:
        run(['-m','llm.prepare','--data-dir',str(data),'--only','qa' if args.raw_checkpoint else 'both'])
    common = ['--device',args.device,'--batch-size','4','--grad-accum-steps','4']
    if args.raw_checkpoint:
        from .checkpoint import load_model
        model, _ = load_model(args.raw_checkpoint)
        if model.config.attention != 'mla':parser.error('Pipeline requires an MLA raw checkpoint')
        del model
        shutil.copy2(args.raw_checkpoint,raw)
        parent = raw
    else:
        run(['-m','llm.train','--data-mixture',str(data/'expanded.json'),'--token-cache-dir',str(data/'token-cache'),
             '--out',str(raw),'--vocab-size',str(DEFAULT_VOCAB_SIZE),'--attention','mla','--n-layers','16','--n-embd','384',
             '--context-size','512','--sampling','shuffled',
             '--steps',str(args.pretrain_steps),'--eval-every','200','--eval-batches','20',*common])
        parent = best_raw
    run(['-m','llm.train','--task','instruction','--init-from',str(parent),
         '--data',str(data/'instruction.jsonl'),'--validation-data',str(data/'instruction-valid.jsonl'),
         '--out',str(tuned),'--steps',str(args.qa_steps),'--max-qa-passes','1',
         '--lr','3e-5','--weight-decay','0.01','--eval-every','50','--eval-batches','20',*common])
    print(f'Raw model: {raw}\nQ&A model: {tuned}\nBest Q&A model: {best_tuned}',flush=True)


if __name__=='__main__':main()
