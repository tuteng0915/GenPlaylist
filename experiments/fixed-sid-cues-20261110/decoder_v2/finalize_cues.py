"""Complete cues after the existing SID decoder has finalized selection and order.

Install AFTER the Beam4/dedup wrapper. Historical experiment decoders stay frozen.
The caller's RNG state is restored so extra cue draws cannot change later batches'
recommendations. This adds a separate cue-only pass, not a same-cost decoder.
"""
import functools
import torch


@torch.no_grad()
def finalize_cues(model, final, cue_mask, sequence_mask, steps=256, eps=1e-5):
    cue_mask = cue_mask.to(device=final.device, dtype=torch.bool)
    assert cue_mask.shape == final.shape
    if not cue_mask.any():
        return final
    fixed = final.clone()
    x = final.clone()
    x[cue_mask] = model.mask_index
    devices = [final.device.index] if final.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        ones = torch.ones(len(x), device=x.device, dtype=model.dtype)
        times = torch.linspace(1, eps, steps + 1, device=x.device, dtype=model.dtype)
        for k in range(steps):
            sigma, _ = model.noise(times[k] * ones)
            next_sigma, _ = model.noise(times[k + 1] * ones)
            probs = model.forward(x, sigma, sequence_mask=sequence_mask).exp()
            move = (1 - torch.exp(-sigma))[:, None, None]
            next_move = (1 - torch.exp(-next_sigma))[:, None, None]
            q = probs * (move - next_move)
            q[:, :, model.mask_index] = next_move[:, :, 0]
            draw = (q / (-(torch.rand_like(q) + 1e-10).log())).argmax(-1)
            x = torch.where(cue_mask & (x == model.mask_index), draw, x)
            x = torch.where(cue_mask, x, fixed)
            assert torch.equal(x[~cue_mask], fixed[~cue_mask])
        remaining = cue_mask & (x == model.mask_index)
        if remaining.any():
            sigma, _ = model.noise(eps * ones)
            last = model.forward(x, sigma, sequence_mask=sequence_mask).argmax(-1)
            x = torch.where(remaining, last, x)
    assert torch.equal(x[~cue_mask], fixed[~cue_mask])
    assert not (x[cue_mask] == model.mask_index).any()
    return x


def install_after_sid_decoder(cls, *, steps=256):
    """Wrap an already configured decoder; SID-only remains a no-op."""
    previous = cls.sample_masked_completion
    if getattr(previous, '_final_sid_cues_v2', False):
        raise RuntimeError('Final-SID cue completion is already installed')

    @functools.wraps(previous)
    @torch.no_grad()
    def sample(self, masked_ids, completion_mask, num_steps, eps=1e-5,
               context_emb=None, mu_c=None, sigma_c2=None, sequence_mask=None):
        final = previous(self, masked_ids, completion_mask, num_steps, eps=eps,
                         context_emb=context_emb, mu_c=mu_c, sigma_c2=sigma_c2,
                         sequence_mask=sequence_mask)
        cm = completion_mask.to(device=final.device, dtype=torch.bool)
        pos = torch.arange(final.shape[1], device=final.device)
        offset = (pos - 1) % self.tokenizer.tokens_per_item
        sid = cm & (pos > 0) & (offset >= 1) & (offset <= 4)
        cue = cm & ~sid
        assert not (final[sid] == self.mask_index).any()
        seq = sequence_mask.to(final.device) if sequence_mask is not None else None
        return finalize_cues(self, final, cue, seq, steps=steps, eps=eps)

    sample._final_sid_cues_v2 = True
    cls.sample_masked_completion = sample
