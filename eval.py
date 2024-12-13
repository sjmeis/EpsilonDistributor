import pandas as pd
import spacy
import torch
from torch.utils.data import Dataset
from collections import Counter
from tqdm.auto import tqdm
from transformers import pipeline, AutoTokenizer
import nltk
import string
from sentence_transformers import SentenceTransformer, util
import evaluate

nltk.download("punkt", quiet=True)
spacy.prefer_gpu()
PUNCT = set(string.punctuation)

class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]

class MaskedTokenInference():
    def __init__(self, batch_size=16):
        if torch.cuda.is_available() == True:
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.pipe = pipeline("fill-mask", model='roberta-base', top_k=3, device=self.device)
        self.batch_size = batch_size

    def score(self, original, private):
        reference = []
        for text in original:
            tokens = [x.lower() for i, x in enumerate(nltk.word_tokenize(text)) if x not in PUNCT and i < 256] 
            reference.append(tokens)

        test = []
        test_tokens = []
        for text in private:
            tokens = [x.lower() for i, x in enumerate(nltk.word_tokenize(text)) if x not in PUNCT and i < 256]
            test_tokens.append(tokens)
            temp = []
            for i, _ in enumerate(tokens):
                t = tokens.copy()
                t[i] = "<mask>"
                temp.append(" ".join(t))
            test.append(temp)

        correct_seq_1 = 0
        correct_seq_3 = 0
        correct_bow_1 = 0
        correct_bow_3 = 0
        total = 0
        for i, tup in tqdm(enumerate(zip(reference, test)), total=len(reference)):
            res = []
            for r in self.pipe(ListDataset(tup[1]), batch_size=self.batch_size):
                res.append([d["token_str"].lower().strip() for d in r])
            
            for r in res:
                try:
                    if r[0] == tup[0][i]:
                        correct_seq_1 += 1
                    if any(t == tup[0][i] for t in r):
                        correct_seq_3 += 1
                    if r[0] in tup[0]:
                        correct_bow_1 += 1
                    if any(t in tup[0] for t in r):
                        correct_bow_3 += 1
                except IndexError:
                    pass
                total += 1

        return round(correct_seq_1 / total, 3), round(correct_seq_3 / total, 3), round(correct_bow_1 / total, 3), round(correct_bow_3 / total, 3)
    
class NearestNeighbor():
    def __init__(self, top_k=1000):
        if torch.cuda.is_available() == True:
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model = SentenceTransformer("thenlper/gte-small", device=self.device)
        self.top_k = top_k

    def score(self, original, private):
        priv_embed = self.model.encode(private, convert_to_tensor=True, show_progress_bar=True)
        priv_embed = priv_embed.to(self.device)
        found = 0
        total = 0
        for i, x in tqdm(enumerate(original), total=len(original)):
            o_embed = self.model.encode(x, convert_to_tensor=True)
            o_embed = o_embed.to(self.device)
            res = util.semantic_search(query_embeddings=o_embed, corpus_embeddings=priv_embed, top_k=self.top_k)[0]
            res = [int(x["corpus_id"]) for x in res]
            try:
                find = res.index(i) + 1
            except ValueError:
                find = self.top_k
            
            found += find
            total += 1

        return round(found / total, 3)

class BLEU():
    def __init__(self):
        self.bleu = evaluate.load("bleu")

    def score(self, original, private):
        score = self.bleu.compute(predictions=private, references=original)["bleu"]
        return round(score, 3)
    
class CS():
    def __init__(self, model_checkpoint="thenlper/gte-small"):
        if torch.cuda.is_available() == True:
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model = SentenceTransformer(model_checkpoint, device=self.device)

    def score(self, original, private):
        orig_embed = self.model.encode(original, convert_to_tensor=True, show_progress_bar=True)
        priv_embed = self.model.encode(private, convert_to_tensor=True, show_progress_bar=True)

        scores = util.pairwise_cos_sim(orig_embed, priv_embed)
        return round(float(scores.mean()), 3)
    

class PPL():
    def __init__(self, model_checkpoint="gpt2", max_len=512):
        self.ppl = evaluate.load("perplexity", module_type="metric")
        self.model_checkpoint = model_checkpoint
        self.max_len = max_len

    def score(self, data):
        data = [" ".join(x.split()[:self.max_len]) for x in data]
        score = self.ppl.compute(predictions=data, model_id=self.model_checkpoint)["mean_perplexity"]
        return round(score, 3)
