import os
import json
import sys
#from nltkor.metrics import DefaultMetric
from nltkor.metrics import EMR
from nltk.translate.meteor_score import meteor_score
import evaluate 
from nltkor.metrics import BERTScore
from transformers import AutoModel, AutoTokenizer, logging
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import nltk
import torch
from tqdm import tqdm

rouge = Rouge()
logging.set_verbosity_error()
nltk.download('wordnet')
meteor = evaluate.load('meteor')
model = AutoModel.from_pretrained('BM-K/KoSimCSE-roberta')
tokenizer = AutoTokenizer.from_pretrained('BM-K/KoSimCSE-roberta')

def get_rouge(gold, pred):
    null_list = [' ','',[''],[],None]

    #result = {1:list(), 2:list(), 3:list(), 'l':list()}
    result = {1:list(), 2:list(), 'l':list()}
    for i in tqdm(range(len(gold))):
        p = pred[i]
        g = gold[i]
        try :
            if p in null_list or g in null_list : raise ZeroDivisionError
            res = rouge.get_scores(g, p, avg=True)
            for key in result :
                result[key].append(res[f'rouge-{key}']['f'])
        except :
            result[key].append(0.0)
#        
#        for key in result:
#            try:
#                if p in null_list or g in null_list: raise ZeroDivisionError
#                res = rouge.get_scores(g, p, avg=True)
#
#                if 'l' == key:
#                    result[key].append(rouge.rouge_l(g,p,avg=True))
#                else:
#                    result[key].append(rouge_n(g,p,key))
#            except :
#                result[key].append(0.0)

    result = {key: sum(result[key])/len(result[key]) if len(result[key])>0 else 0 for key in result}
    for key in result:
        print('rouge-{}: {}'.format(key, result[key]), flush=True)
    return result

def get_bleu(gold, pred):
    null_list = [' ','',[''],[],None]

    result = {1:list(), 2:list(), 3:list(), 4:list()}
    for i in range(len(gold)):
        p = pred[i]
        g = gold[i]
       
        for key in result:
            result[key].append(DefaultMetric().bleu_n([g],[p],key))
#            try:
#                if p in null_list or g in null_list: raise ZeroDivisionError
#                result[key].append(DefaultMetric().bleu_n([g],[p],key))
#            except Exception as e:
#                print(e)
#                exit()
#                result[key].append(0.0)
    result = {key: sum(result[key])/len(result[key]) if len(result[key])>0 else 0 for key in result}
    result['a'] = sum([result[key] for key in result])/len(result)
    for key in result:
        print('BLEU-{}: {}'.format(key, result[key]), flush=True)
    return result

def get_cider(gold, pred):
    null_list = [' ','',[''],[],None]

    result = list()
    for i in range(len(gold)):
        p = pred[i]
        g = gold[i]
        try:
            if p in null_list or g in null_list: raise ZeroDivisionError
            #print(DefaultMetric().cider([g],[p]))
            #result.append(float(DefaultMetric().cider([g],[p])[0])) #TODO: 수정함
            result.append(float(DefaultMetric().cider([g],[p])))
        except :
            result.append(0.0)
    result = sum(result)/len(result) if len(result)>0 else 0
    result = {'cider':result}
    print('CIDER: {}'.format(result['cider']), flush=True)
    return result

def get_meteor(gold, pred):
    #from nltk import DefaultMetric()
    null_list = [' ','',[''],[],None]

    result = list()
    for i in range(len(gold)):
        p = pred[i]
        g = gold[i]
        try:
            if p in null_list or g in null_list: raise ZeroDivisionError
            sco = meteor.compute(references=[g],predictions=[p])
            result.append(float(sco['meteor']))
            #result.append(float(DefaultMetric().meteor([g],[p])))
        except:
            result.append(0.0)
    result = sum(result)/len(result) if len(result)>0 else 0
    result = {'meteor':result}
    print('METEOR: {}'.format(result['meteor']), flush=True)
    return result

def get_bert(gold, pred):
    null_list = [' ','',[''],[],None]
    bertscore = evaluate.load("bertscore")

    precision = list()
    recall = list()
    f1 = list()
    for i in range(len(gold)) :
        p = pred[i]
        g = gold[i]
        try :
            if p in null_list or g in null_list: raise ZeroDivisionError
            score = bertscore.compute(predictions=[p], references=[g], model_type="bert-base-uncased")
            precision.append(score['precision'][0])
            recall.append(score['recall'][0])
            f1.append(score['f1'][0])
        except :
            precision.append(0.0)
            recall.append(0.0)
            f1.append(0.0)

    ## BertScore from evaluate
    # output: 각 문장 별 precision, recall, f1 값 리스트, + hashcode(뭐지?)

    result = {'precision':precision, 'recall':recall, 'f1':f1}
    
    result = {key: sum(result[key])/len(result[key]) if len(result[key])>0 else 0 for key in result}
    for key in result:
        print('BERT-{}: {}'.format(key, result[key]), flush=True)
    return result

def get_simCSE(gold, pred):
    gold, pred = prepare_for_simCSE(gold, pred)
    null_list = [' ','',[''],[],None]
    scores = list()
    for i in range(len(gold)) :
        p = pred[i][0][0]
        g = gold[i][0][0]

        try :
            if p in null_list or g in null_list : raise ZeroDivisionError
            if len(p.shape) == 1: p = p.unsqueeze(0)
            if len(g.shape) == 1: g = g.unsqueeze(0)
            p_norm = p / p.norm(dim=1)[:, None]
            g_norm = g / g.norm(dim=1)[:, None]
            
            score =  torch.mm(p_norm, g_norm.transpose(0, 1)) * 100
            scores.append(score)
        except :
            scores.append(0.0)

    result = sum(scores) / len(scores) if len(scores) > 0 else 0
    print('SimCSE: {}'.format(result), flush=True)
    return result

def get_EntMent(gold, pred):
    null_list = [' ','',[''],[],None]
    scores = list()

    for i in range(len(gold)) :
        p = pred[i]
        g = gold[i]
        

        try :
            if p in null_list or g in null_list : raise ZeroDivisionError
            score = EMR().entity(g, p) 
            
            scores.append(score)
        except Exception as e:
            scores.append(0.0)

    result = sum(scores) / len(scores) if len(scores) > 0 else 0
    print('EntMent: {}'.format(result), flush=True)
    return result


def prepare_for_simCSE(gold, pred) :
    golds = list()
    preds = list()
    
    for i in range(len(gold)) :
        g = tokenizer(gold[i], padding=True, truncation=True, return_tensors='pt')
        p = tokenizer(pred[i], padding=True, truncation=True, return_tensors='pt')

        res1, _ = model(**g, return_dict=False)
        res2, _ = model(**p, return_dict=False)
        
        golds.append(res1)
        preds.append(res2)

    return golds, preds


if __name__=='__main__':
    filename = sys.argv[1]
    ext = os.path.splitext(filename)[-1]
    if ext == '.jsonl':
        with open(filename, 'r') as f:
            data = [json.loads(d) for d in f]
#        pred = [d['outputs']['preds'] for d in data]
#        gold = [d['outputs']['labels'] for d in data ]
#        srce = [d['outputs']['inputs'] for d in data ]
        pred = [d['output'] for d in data]
        gold = [d['label'] for d in data]
        #srce = ['$'.join([d['history'], d['inputs']]) for d in data]
    
    #assert len(pred) == len(gold) and len(gold) == len(srce)
    print('data size: {}\n'.format(len(gold)))

    #bleu = get_bleu(gold, pred)
    rouge = get_rouge(gold, pred)
    exit()
    cider = get_cider(gold, pred)
    meteor = get_meteor(gold, pred)
    bert = get_bert(gold, pred)
    simCSE = get_simCSE(gold, pred)
    # entment = get_EntMent(gold, pred)

    line = {
                'bleu': bleu, 
                'rouge': rouge, 
                'cider': cider, 
                'meteor': meteor, 
                'bert': bert, 
                'simCSE': simCSE
                #'entment': entment
            }

    new_filename = '/'.join(filename.split('/')[-3:])
    with open(f'scores/{new_filename}', 'w') as f :
        f.write(json.dumps(line, ensure_ascii=False)+'\n')
