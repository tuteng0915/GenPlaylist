"""Regression checks: final SID conditioning, no stale cues, no RNG side effects."""
from types import SimpleNamespace
import torch
from finalize_cues import finalize_cues, install_after_sid_decoder

torch.set_num_threads(2)
class Model:
    mask_index = 22
    dtype = torch.float32
    tokenizer = SimpleNamespace(tokens_per_item=13)
    def noise(self, t):
        return -torch.log(1 - .999 * t), None
    def forward(self, x, sigma, sequence_mask=None):
        self.seen.append(x.clone())
        logits = torch.full((*x.shape, 23), -1e9)
        # Cue identity depends on the FINAL selected SID, not the old cue.
        preferred = 13 + x[:, 2] % 8
        logits.scatter_(2, preferred[:, None, None].expand(-1, x.shape[1], 1), 0)
        return logits.log_softmax(-1)
    def sample_masked_completion(self, x, completion_mask, steps, **kwargs):
        result = x.clone()
        result[:, 2:6] = torch.randint(2, 10, (len(x), 4))
        result[:, 6:14] = 12  # Stale cues from before final SID selection.
        return result

x = torch.ones(2, 15, dtype=torch.long)
cm = torch.zeros_like(x, dtype=torch.bool); cm[:, 2:14] = True
cue = torch.zeros_like(cm); cue[:, 6:14] = True
seq = torch.ones_like(cm)
Fixed = type('Fixed', (Model,), {})
install_after_sid_decoder(Fixed, steps=8)
old = Model(); new = Fixed(); new.seen = []
torch.manual_seed(17)
original_batches = [old.sample_masked_completion(x, cm, 256) for _ in range(2)]
state_original = torch.random.get_rng_state()
torch.manual_seed(17)
fixed_batches = [new.sample_masked_completion(x, cm, 256, sequence_mask=seq) for _ in range(2)]
assert torch.equal(state_original, torch.random.get_rng_state())
for original, corrected in zip(original_batches, fixed_batches):
    assert torch.equal(original[~cue], corrected[~cue])
    assert (corrected[:, 6:14] == (13 + original[:, 2] % 8)[:, None]).all()
assert (new.seen[0][cue] == 22).all()
assert all(torch.equal(z[~cue], original_batches[i // 8][~cue])
           for i, z in enumerate(new.seen))
assert torch.equal(finalize_cues(new, x, torch.zeros_like(cm), seq), x)
try:
    install_after_sid_decoder(Fixed)
    raise AssertionError('Duplicate install was accepted')
except RuntimeError:
    pass
print('PASS: cues depend on final SID; stale cues remasked; all non-cue tokens fixed; two-batch SID and RNG equality; SID-only no-op; duplicate install rejected.')
