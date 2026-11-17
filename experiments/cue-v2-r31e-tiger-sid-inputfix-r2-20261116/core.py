"""Matched TIGER history-cue control and autoregressive SID--cue planner."""
import hashlib
import numpy as np
import torch
import torch.nn.functional as F

PAD, EOS, SID_WIDTH, CUE_WIDTH = 0, 1, 4, 8
REFERENCE_ITEMS, TARGET_ITEMS = 15, 5

def fingerprint(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

class Catalog:
    def __init__(self, canonical, cues, cue_vocab_size):
        canonical = np.asarray(canonical, dtype=np.int64)
        cues = np.asarray(cues, dtype=np.int64)
        assert canonical.ndim == 2 and canonical.shape[1] == SID_WIDTH
        assert cues.shape == (len(canonical), CUE_WIDTH)
        assert canonical.min() >= 0 and cues.min() >= 0 and cues.max() < cue_vocab_size
        self.shift = max(0, 2 - int(canonical.min()))
        self.codes = canonical + self.shift
        self.canonical = canonical
        self.cue_offset = int(self.codes.max()) + 1
        self.cues = cues
        self.cue_token_ids = list(range(self.cue_offset, self.cue_offset + cue_vocab_size))
        self.vocab_size = self.cue_offset + cue_vocab_size
        self.items = torch.tensor(np.concatenate([self.codes, cues + self.cue_offset], axis=1))
        self.lookup = {tuple(map(int, row)): i for i, row in enumerate(self.codes)}
        assert len(self.lookup) == len(canonical)
        self.prefixes = {}
        for i, row in enumerate(self.codes):
            for depth in range(SID_WIDTH):
                self.prefixes.setdefault(tuple(row[:depth]), {}).setdefault(int(row[depth]), []).append(i)

    def allowed_sid(self, prefix, excluded):
        prefix = tuple(map(int, prefix))
        assert len(prefix) < SID_WIDTH
        result = [token for token, rows in self.prefixes[prefix].items()
                  if any(i not in excluded for i in rows)]
        assert result
        return sorted(result)

    def inputs(self, history_rows, joint, device):
        rows = torch.as_tensor(history_rows, dtype=torch.long)
        assert rows.ndim == 2 and REFERENCE_ITEMS <= rows.shape[1] < REFERENCE_ITEMS + TARGET_ITEMS
        # The standard baseline must be genuinely SID-only: it cannot observe
        # catalog cues in its history.  The joint arm receives SID+cues for
        # both observed and previously recommended items.
        representation = self.items if joint else torch.as_tensor(
            self.codes, dtype=torch.long)
        x = representation[rows].reshape(len(rows), -1).to(device)
        return x, torch.ones_like(x)

    def batch(self, sequences, batch_size, generator, joint, device):
        rows = torch.randint(len(sequences), (batch_size,), generator=generator)
        slots = torch.randint(TARGET_ITEMS, (batch_size,), generator=generator)
        chosen = sequences[rows]
        width = 12 if joint else 4
        x = torch.zeros((batch_size, 19 * width), dtype=torch.long)
        attention = torch.zeros_like(x)
        labels = torch.full((batch_size, (12 if joint else 4) + 1), EOS, dtype=torch.long)
        for slot in range(5):
            mask = slots == slot
            if not mask.any(): continue
            n = 15 + slot
            representation = self.items if joint else torch.as_tensor(
                self.codes, dtype=torch.long)
            history = representation[chosen[mask, :n]].reshape(int(mask.sum()), n * width)
            x[mask, :n*width] = history; attention[mask, :n*width] = 1
            target = self.items[chosen[mask, n]]
            labels[mask, :12 if joint else 4] = target[:, :12 if joint else 4]
        return (x.to(device), attention.to(device), labels.to(device), rows, slots)


def loss_parts(logits, labels, joint):
    ce = F.cross_entropy(logits.float().transpose(1, 2), labels, reduction='none')
    sid = ce[:, :4].mean()
    eos = ce[:, -1].mean()
    cue = ce[:, 4:12].mean() if joint else sid.new_zeros(())
    # Fixed before results; prevent longer cue targets from diluting SID supervision.
    return sid + .1 * cue + .1 * eos, {'sid': sid, 'cue': cue, 'eos': eos}


def build_model(vocab_size, tiny=False):
    from transformers import T5Config, T5ForConditionalGeneration
    return T5ForConditionalGeneration(T5Config(
        vocab_size=vocab_size, d_model=32 if tiny else 128, d_kv=8 if tiny else 64,
        d_ff=64 if tiny else 1024, num_layers=1 if tiny else 4,
        num_decoder_layers=1 if tiny else 4, num_heads=2 if tiny else 6,
        dropout_rate=.1, feed_forward_proj='relu', pad_token_id=PAD,
        eos_token_id=EOS, decoder_start_token_id=PAD, tie_word_embeddings=True))


@torch.no_grad()
def predict(model, cat, references, joint, device, beam=20):
    histories = np.asarray(references, dtype=np.int64)
    assert histories.shape[1] == 15
    predictions, cues = [], []
    for slot in range(5):
        x, mask = cat.inputs(histories, joint, device)
        excluded = [set(int(p[i]) for p in predictions) for i in range(len(histories))]
        def allowed(b, prefix):
            return cat.allowed_sid(prefix.tolist()[1:], excluded[b])
        # Rank exactly four SID tokens for both arms. Future cue/EOS likelihood
        # cannot rerank SID candidates or silently grant a different search budget.
        ids = model.generate(input_ids=x, attention_mask=mask, do_sample=False,
            num_beams=beam, num_return_sequences=1, min_new_tokens=4, max_new_tokens=4,
            eos_token_id=None, forced_eos_token_id=None, prefix_allowed_tokens_fn=allowed)
        assert ids.shape == (len(histories), 5)
        selected = np.array([cat.lookup[tuple(row)] for row in ids[:, 1:].cpu().tolist()])
        if joint:
            full = model.generate(input_ids=x, attention_mask=mask, decoder_input_ids=ids,
                do_sample=False, num_beams=1, min_new_tokens=8, max_new_tokens=8,
                eos_token_id=None, forced_eos_token_id=None,
                prefix_allowed_tokens_fn=lambda b, prefix: cat.cue_token_ids)
            assert full.shape == (len(histories), 13)
            cue = full[:, 5:13].cpu().numpy() - cat.cue_offset
            assert cue.min() >= 0 and cue.max() < len(cat.cue_token_ids)
        else:
            cue = np.empty((len(histories), 0), dtype=np.int64)
        predictions.append(selected); cues.append(cue)
        histories = np.concatenate([histories, selected[:, None]], axis=1)
    pred = np.stack(predictions, 1)
    assert all(len(set(row)) == 5 for row in pred)
    return pred, np.stack(cues, 1)
