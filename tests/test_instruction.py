import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
import torch.nn.functional as F
import numpy as np
from llm.checkpoint import load_model
from llm.generate import main as generate_main
from llm.instruction import InstructionData,instruction_prompt,load_instruction_data
from llm.model import LanguageModel,ModelConfig
from llm.tokenizer import BPETokenizer,EOS_TOKEN,DEFAULT_VOCAB_SIZE
from llm.train import ShuffledWindows,load_cached_files,main as train_main,save_checkpoint
from llm.pipeline import main as pipeline_main
from llm.prepare import fiction_candidates,qa_record,trim_training_pool,token_count,educational_text


class InstructionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.records=[dict(question='Why is the sky blue?',answer='Air scatters blue light.'),
                      dict(question='What is a seed?',answer='It can grow into a plant.')]
        self.tokenizer=BPETokenizer.train('Science and stories. '*20+EOS_TOKEN,280)

    def test_shifted_answer_eos_padding_and_loss(self):
        data=InstructionData(self.records,self.tokenizer,128)
        prompt=self.tokenizer.encode(instruction_prompt(self.records[0]['question']))
        answer=self.tokenizer.encode(self.records[0]['answer'])+[self.tokenizer.eos_token_id]
        x,y=data.examples[0]
        self.assertEqual(x.tolist(),(prompt+answer)[:-1])
        self.assertEqual(y.tolist(),[-100]*(len(prompt)-1)+answer)
        x,y=data.batch(8,'cpu',torch.Generator().manual_seed(0))
        model=LanguageModel(ModelConfig(vocab_size=self.tokenizer.vocab_size,context_size=128,n_layers=1,n_heads=2,n_embd=16))
        logits,loss=model(x,y);valid=y!=-100
        torch.testing.assert_close(loss,F.cross_entropy(logits[valid],y[valid]))
        loss.backward();self.assertTrue(torch.isfinite(model.token_embedding.weight.grad).all())
        with self.assertRaisesRegex(ValueError,'No complete'):InstructionData(self.records,self.tokenizer,2)

    def test_shuffling_and_validation_do_not_repeat_or_advance(self):
        data=InstructionData(self.records,self.tokenizer,128)
        a,_=data.batch(1,'cpu');cursor=data.cursor
        data.batch(10,'cpu',torch.Generator().manual_seed(0));self.assertEqual(cursor,data.cursor)
        b,_=data.batch(1,'cpu');self.assertNotEqual(a.tolist(),b.tolist())
        windows=ShuffledWindows(np.arange(65,dtype=np.uint16),8)
        x,y=windows.batch(8,8,'cpu');self.assertEqual(sorted(x[:,0].tolist()),list(range(0,64,8)))
        torch.testing.assert_close(y,x+1)
        cursor=windows.cursor;windows.batch(1,8,'cpu',torch.Generator().manual_seed(0));self.assertEqual(cursor,windows.cursor)

    def test_tokenizer_reuse_and_question_leakage(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);paths=[root/'train.jsonl',root/'valid.jsonl']
            for p,r in zip(paths,self.records):p.write_text(json.dumps(r)+'\n')
            a,b,t=load_instruction_data(paths[0],128,self.tokenizer,paths[1]);self.assertIs(t,self.tokenizer)
            paths[1].write_text(json.dumps(dict(question=' WHY is the sky blue? ',answer='Another answer.')))
            with self.assertRaisesRegex(ValueError,'overlap'):load_instruction_data(paths[0],128,self.tokenizer,paths[1])
            text_paths=[root/'train.txt',root/'valid.txt']
            for p in text_paths:p.write_text('A modern educational passage.\n'*10)
            with patch.object(BPETokenizer,'train_file',side_effect=AssertionError('Must keep vocabulary')):
                arrays,t=load_cached_files(text_paths,8,280,root/'cache',self.tokenizer)
            self.assertEqual(t.to_json(),self.tokenizer.to_json())
            self.assertEqual(t.decode(arrays[0].tolist()),text_paths[0].read_text())

    def test_cli_continues_checkpoint_caps_passes_and_preserves_parent(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);base=root/'raw.pt';out=root/'tuned.pt'
            model=LanguageModel(ModelConfig(vocab_size=self.tokenizer.vocab_size,context_size=128,n_layers=1,n_heads=2,n_embd=16,dropout=0,attention='mla'))
            save_checkpoint(base,model,self.tokenizer,4,2.0)
            paths=[root/'train.jsonl',root/'valid.jsonl']
            for p,r in zip(paths,self.records):p.write_text(json.dumps(r)+'\n')
            cmd=['train','--task','instruction','--init-from',str(base),'--data',str(paths[0]),'--validation-data',str(paths[1]),'--out',str(out),'--device','cpu','--steps','5','--batch-size','1','--grad-accum-steps','1','--eval-batches','1']
            with patch('sys.argv',cmd),patch('sys.stdout',new_callable=io.StringIO):train_main()
            restored,t=load_model(out);self.assertEqual(restored.config,model.config)
            self.assertEqual(t.to_json(),self.tokenizer.to_json());self.assertFalse(torch.equal(restored.token_embedding.weight,model.token_embedding.weight))
            c=torch.load(out,weights_only=True);self.assertEqual(c['step'],1);self.assertEqual(c['training_info']['loss_mask'],'answer and EOS only')
            unchanged,_=load_model(base);torch.testing.assert_close(unchanged.token_embedding.weight,model.token_embedding.weight)
            for extra in [['--attention','mha'],['--out',str(base)]]:
                with patch('sys.argv',cmd+extra),patch('sys.stderr',new_callable=io.StringIO):
                    with self.assertRaises(SystemExit):train_main()

    def test_generation_question_template_and_answer_only(self):
        model=LanguageModel(ModelConfig(vocab_size=self.tokenizer.vocab_size,context_size=128,n_layers=1,n_heads=2,n_embd=16))
        prefix=self.tokenizer.encode(instruction_prompt('What is a seed?'))
        reply=self.tokenizer.encode('A seed can grow.')+[self.tokenizer.eos_token_id]
        with patch('sys.argv',['generate','--question','What is a seed?','--device','cpu']),patch('llm.generate.load_model',return_value=(model,self.tokenizer)),patch.object(model,'generate',return_value=torch.tensor([prefix+reply])) as generate,patch('sys.stdout',new_callable=io.StringIO) as out:
            generate_main()
        self.assertEqual(out.getvalue(),'A seed can grow.\n');self.assertEqual(generate.call_args.args[2],0)

    def test_pipeline_pretraining_configuration_and_qa_inheritance(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name in ['expanded.json','expanded-tokenizer.json','expanded-fiction.txt','expanded-education.txt','expanded-fiction-valid.txt','expanded-education-valid.txt','instruction.jsonl','instruction-valid.jsonl']:(root/name).touch()
            (root/'expanded.json').write_text('{"target_tokens": 200000000, "sources": [{"name": "fiction", "weight": 0.7}, {"name": "education", "weight": 0.3}]}')
            with patch('sys.argv',['pipeline','--data-dir',folder,'--checkpoint-dir',folder]),patch('llm.pipeline.run') as runner,patch('llm.pipeline.BPETokenizer.from_json',return_value=SimpleNamespace(vocab_size=DEFAULT_VOCAB_SIZE)),patch('sys.stdout',new_callable=io.StringIO):
                pipeline_main()
            self.assertEqual(runner.call_count,2)
            pretrain,qa=[c.args[0] for c in runner.call_args_list]
            for flag,value in [('--n-layers','16'),('--n-embd','384'),('--vocab-size','16384'),('--steps','24000')]:
                self.assertEqual(pretrain[pretrain.index(flag)+1],value)
            self.assertNotIn('--patience',pretrain)
            self.assertNotIn('--n-layers',qa);self.assertNotIn('--n-embd',qa)
            self.assertEqual(qa[qa.index('--init-from')+1],pretrain[pretrain.index('--out')+1].replace('.raw.pt','.raw.best.pt'))

    def test_pipeline_does_not_finetune_when_pretraining_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name in ['expanded.json','expanded-tokenizer.json','expanded-fiction.txt','expanded-education.txt','expanded-fiction-valid.txt','expanded-education-valid.txt','instruction.jsonl','instruction-valid.jsonl']:(root/name).touch()
            (root/'expanded.json').write_text('{"target_tokens": 200000000, "sources": [{"name": "fiction", "weight": 0.7}, {"name": "education", "weight": 0.3}]}')
            with patch('sys.argv',['pipeline','--data-dir',folder,'--checkpoint-dir',folder]),patch('llm.pipeline.run',side_effect=subprocess.CalledProcessError(1,'train')) as runner,patch('llm.pipeline.BPETokenizer.from_json',return_value=SimpleNamespace(vocab_size=DEFAULT_VOCAB_SIZE)):
                with self.assertRaises(subprocess.CalledProcessError):pipeline_main()
            self.assertEqual(runner.call_count,1)
            self.assertEqual(runner.call_args.args[0][:2],['-m','llm.train'])

    def test_pipeline_rebuilds_a_stale_vocabulary(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name in ['expanded.json','expanded-tokenizer.json','expanded-fiction.txt','expanded-education.txt','expanded-fiction-valid.txt','expanded-education-valid.txt','instruction.jsonl','instruction-valid.jsonl']:(root/name).touch()
            (root/'expanded.json').write_text('{"target_tokens": 200000000, "sources": [{"name": "fiction", "weight": 0.7}, {"name": "education", "weight": 0.3}]}')
            with patch('sys.argv',['pipeline','--data-dir',folder,'--checkpoint-dir',folder]),patch('llm.pipeline.run') as runner,patch('llm.pipeline.BPETokenizer.from_json',return_value=SimpleNamespace(vocab_size=8192)),patch('sys.stdout',new_callable=io.StringIO):
                pipeline_main()
            self.assertEqual(runner.call_count,3)
            self.assertEqual(runner.call_args_list[0].args[0][:2],['-m','llm.prepare'])

    def test_pipeline_rebuilds_an_undersized_pool(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name in ['expanded.json','expanded-tokenizer.json','expanded-fiction.txt','expanded-education.txt','expanded-fiction-valid.txt','expanded-education-valid.txt','instruction.jsonl','instruction-valid.jsonl']:(root/name).touch()
            (root/'expanded.json').write_text('{"target_tokens": 100000000}')
            with patch('sys.argv',['pipeline','--data-dir',folder,'--checkpoint-dir',folder]),patch('llm.pipeline.run') as runner,patch('llm.pipeline.BPETokenizer.from_json',return_value=SimpleNamespace(vocab_size=DEFAULT_VOCAB_SIZE)),patch('sys.stdout',new_callable=io.StringIO):
                pipeline_main()
            self.assertEqual(runner.call_count,3)
            self.assertEqual(runner.call_args_list[0].args[0][:2],['-m','llm.prepare'])

    def test_raw_checkpoint_pipeline_writes_distinct_models(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);base=root/'existing.pt'
            model=LanguageModel(ModelConfig(vocab_size=self.tokenizer.vocab_size,context_size=128,n_layers=1,n_heads=2,n_embd=16,dropout=0,attention='mla'))
            save_checkpoint(base,model,self.tokenizer,3,2.0)
            records=[dict(question=f'What is a seed called {chr(97+i)}?',answer='It can grow into a plant.') for i in range(20)]
            (root/'instruction.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            (root/'instruction-valid.jsonl').write_text(json.dumps(dict(question='Why do plants grow?',answer='They use light and water.'))+'\n')
            result=subprocess.run([__import__('sys').executable,'-m','llm.pipeline','--raw-checkpoint',str(base),'--name','test','--data-dir',folder,'--checkpoint-dir',folder,'--device','cpu','--qa-steps','1'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            raw,t=load_model(root/'test.raw.pt');tuned,u=load_model(root/'test.tuned.pt')
            self.assertEqual(t.to_json(),u.to_json());self.assertEqual(raw.config,tuned.config)
            torch.testing.assert_close(raw.token_embedding.weight,model.token_embedding.weight)
            self.assertFalse(torch.equal(tuned.token_embedding.weight,raw.token_embedding.weight))

    def test_early_stopping_records_actual_step(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'raw.pt'
            tokens=torch.tensor(self.tokenizer.encode('A simple story. '*30))
            cmd=['train','--data','unused.txt','--out',str(out),'--device','cpu','--steps','10','--eval-every','1','--patience','1','--batch-size','1','--context-size','8','--n-layers','1','--n-heads','2','--n-embd','16']
            with patch('sys.argv',cmd),patch('sys.stdout',new_callable=io.StringIO),patch('llm.train.load_data',return_value=(tokens,tokens,self.tokenizer)),patch('llm.train.evaluate',side_effect=[dict(train=1.,validation=1.),dict(train=.9,validation=1.1)]):
                train_main()
            self.assertEqual(torch.load(out,weights_only=True)['step'],1)

    def test_pool_uses_real_tokens_and_keeps_complete_documents(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'fiction.txt'
            first='A simple story.'
            second='A much longer story. '*20
            normalized=first+'\n'+EOS_TOKEN+'\n\n'
            target=sum(len(r.ids) for r in self.tokenizer.tokenizer.encode_batch(normalized.splitlines(keepends=True),add_special_tokens=False))
            path.write_text(normalized+second+'\n'+EOS_TOKEN+'\n\n')
            count,kept=trim_training_pool(path,self.tokenizer,target)
            self.assertEqual(count,target);self.assertEqual(len(kept),1)
            self.assertEqual(token_count(path,self.tokenizer),count)
            self.assertNotIn('much longer',path.read_text())
            self.assertEqual(path.read_text().count(EOS_TOKEN),1)

    def test_maths_focused_web_pages_are_excluded(self):
        for title,url in [('Algebra lessons','https://example.com/algebra'),('Mathematics for beginners','https://example.com'),('Calculus explained','https://example.com')]:
            row=dict(text=title+'\n'+'An educational explanation with plenty of readable text. '*20,url=url)
            self.assertIsNone(educational_text(row))
        row=dict(text='How plants grow\n'+'Plants need light and water to grow. '*20,url='https://example.com/plants')
        self.assertIsNotNone(educational_text(row))

    def test_filters_exclude_held_out_titles_and_worked_maths(self):
        row=dict(Type='Text',Language='en',Subjects='Gothic fiction',Title='Held-out book',**{'Text#':'5'})
        self.assertEqual(list(fiction_candidates([row],{5},set())),[])
        self.assertEqual(list(fiction_candidates([row],set(),{'heldoutbook'})),[])
        for q,a in [('What percent is 10 of 100?','Ten percent.'),('What does this equation mean?','Multiply both sides.'),('What should I do? Your response should end with an exact phrase.','Follow these instructions.')]:
            self.assertIsNone(qa_record(dict(messages=[dict(role='user',content=q),dict(role='assistant',content=a)])))
