from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import torch

# cross-encoder rerankers: each scores a (query, movie text) pair together, 0 to 1, higher is more relevant
# qwen3's reranker is a causal lm that answers yes or no, its score is p(yes); bge is a classifier with a sigmoid
instruction = 'Given a movie search, judge whether the movie matches it'
# pairs longer than this are truncated, the movie text (title, genres, overview, top tags) rarely reaches it
max_length = 512

class Reranker:
    def __init__(self, model_name, device=None, batch_size=64):
        self.model_name, self.batch_size = model_name, batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = torch.float16 if self.device == 'cuda' else torch.float32
        self.qwen = model_name.startswith('Qwen/Qwen3-Reranker')
        if self.qwen:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
            model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(self.device).eval()
            # only the yes and no logits at the last position are read, so the full vocabulary projection is skipped
            self.body = model.model
            ids = [self.tokenizer.convert_tokens_to_ids(t) for t in ['no', 'yes']]
            self.head = model.lm_head.weight[ids].detach()
            self.prefix = self.tokenizer.encode('<|im_start|>system\nJudge whether the Document meets the requirements based on '
                                                'the Query and the Instruct provided. Note that the answer can only be "yes" or '
                                                '"no".<|im_end|>\n<|im_start|>user\n', add_special_tokens=False)
            self.suffix = self.tokenizer.encode('<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n', add_special_tokens=False)
        else:
            self.model = CrossEncoder(model_name, device=self.device, max_length=max_length,
                                      model_kwargs={'dtype': dtype})

    @torch.no_grad()
    def _qwen(self, pairs):
        texts = [f'<Instruct>: {instruction}\n<Query>: {q}\n<Document>: {d}' for q, d in pairs]
        enc = self.tokenizer(texts, padding=False, truncation=True, return_attention_mask=False,
                             max_length=max_length - len(self.prefix) - len(self.suffix))
        enc['input_ids'] = [self.prefix + ids + self.suffix for ids in enc['input_ids']]
        enc = self.tokenizer.pad(enc, padding=True, return_tensors='pt').to(self.device)
        # left padding puts every pair's last token at the last position
        last = self.body(**enc).last_hidden_state[:, -1]
        return torch.softmax((last @ self.head.T).float(), dim=1)[:, 1].cpu().numpy()

    def score(self, pairs):
        # pairs is a list of (query, movie text), scored in length order so batches pad less
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        if not self.qwen:
            return np.asarray(self.model.predict(pairs, batch_size=self.batch_size, activation_fn=torch.nn.Sigmoid(),
                                                 show_progress_bar=False), dtype=np.float32)
        order = np.argsort([len(q) + len(d) for q, d in pairs])
        out = np.zeros(len(pairs), dtype=np.float32)
        for i in range(0, len(pairs), self.batch_size):
            idx = order[i:i + self.batch_size]
            out[idx] = self._qwen([pairs[j] for j in idx])
        return out
