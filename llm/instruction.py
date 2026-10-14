"""Complete Q&A examples, with loss restricted to answers and EOS."""
import json
from pathlib import Path
import random
import unicodedata
import torch
from .tokenizer import EOS_TOKEN


def question_key(text):
    return ' '.join(unicodedata.normalize('NFC', text).casefold().split())


def instruction_prompt(question):
    if not isinstance(question, str) or not question.strip() or EOS_TOKEN in question:
        raise ValueError('Question must be nonempty and cannot contain EOS')
    return f'Question: {question.strip()}\nAnswer:\n'


def read_examples(path):
    result, seen = [], set()
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        instruction_prompt(r.get('question'))
        answer = r.get('answer')
        if not isinstance(answer, str) or not answer.strip() or EOS_TOKEN in answer:
            raise ValueError(f'Invalid answer in {path}')
        identity = (question_key(r['question']), question_key(answer))
        if identity not in seen: result.append(r); seen.add(identity)
    if not result: raise ValueError(f'Empty Q&A file: {path}')
    return result


class InstructionData:
    def __init__(self, records, tokenizer, context_size):
        if tokenizer.eos_token_id is None: raise ValueError('Checkpoint needs an EOS tokenizer')
        self.pad_id = tokenizer.eos_token_id
        self.examples, self.skipped, self.order, self.cursor = [], 0, [], 0
        for r in records:
            prompt = tokenizer.encode(instruction_prompt(r['question']))
            answer = tokenizer.encode(r['answer'].strip()) + [self.pad_id]
            tokens = prompt + answer
            if len(tokens) - 1 > context_size: self.skipped += 1; continue
            self.examples.append((torch.tensor(tokens[:-1]), torch.tensor([-100] * (len(prompt) - 1) + answer)))
        if not self.examples: raise ValueError('No complete Q&A examples fit the context')
        self.token_count = sum(len(x) + 1 for x, _ in self.examples)

    def __len__(self): return len(self.examples)

    def batch(self, batch_size, device, generator=None):
        if generator is not None:
            selected = torch.randint(len(self), (batch_size,), generator=generator).tolist()
        else:
            selected = []
            while len(selected) < batch_size:
                if self.cursor == len(self.order): self.order, self.cursor = torch.randperm(len(self)).tolist(), 0
                count = min(batch_size - len(selected), len(self.order) - self.cursor)
                selected.extend(self.order[self.cursor:self.cursor + count]); self.cursor += count
        examples = [self.examples[i] for i in selected]
        x = torch.full((batch_size, max(len(x) for x, _ in examples)), self.pad_id, dtype=torch.long)
        y = torch.full_like(x, -100)
        for i, (a, b) in enumerate(examples): x[i, :len(a)], y[i, :len(b)] = a, b
        return x.to(device), y.to(device)


def load_instruction_data(path, context_size, tokenizer, validation_path=None):
    train = read_examples(path)
    if validation_path:
        valid = read_examples(validation_path)
    else:
        groups = {}
        for r in train: groups.setdefault(question_key(r['question']), []).append(r)
        keys = sorted(groups); random.Random(42).shuffle(keys)
        if len(keys) < 2: raise ValueError('Need at least two unique questions')
        split = max(1, min(len(keys) - 1, int(len(keys) * .95)))
        train, valid = [[r for key in ks for r in groups[key]] for ks in (keys[:split], keys[split:])]
    if {question_key(r['question']) for r in train} & {question_key(r['question']) for r in valid}:
        raise ValueError('Questions overlap across training and validation')
    datasets = [InstructionData(rows, tokenizer, context_size) for rows in (train, valid)]
    for name, data in zip(('train', 'validation'), datasets):
        print(f'{name}_examples={len(data):,} skipped_overlength={data.skipped:,}', flush=True)
    return *datasets, tokenizer
