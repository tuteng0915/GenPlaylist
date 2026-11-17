"""Independent channel normalization for controlled SID/cue training trials."""
import torch

VARIANTS = ('legacy', 'separate01', 'separate03', 'delayed01')

def cue_coefficient(variant, step):
    if variant == 'separate01':
        return .1
    if variant == 'separate03':
        return .3
    if variant == 'delayed01':
        return .1 * min(1., max(0., (step-1000)/4000))
    raise ValueError(variant)

def channel_loss(unweighted, active, weights, period, n_digit, cue_lambda):
    """Preserve SID-only weighted-mean loss; add an independent cue mean.

    Denominators include all active target positions of their own type, as in
    the original diffusion objective (not only currently corrupted positions).
    """
    positions = torch.arange(unweighted.shape[1], device=unweighted.device)
    offsets = (positions-1).remainder(period)
    sid = active.bool() & (positions > 0) & (offsets >= 1) & (offsets <= n_digit+1)
    cue = active.bool() & (positions > 0) & (offsets > n_digit+1)
    if not bool(sid.any()) or not bool(cue.any()):
        raise ValueError('Joint-channel objective requires both SID and cue targets')
    if bool((active.bool() & ~(sid | cue)).any()):
        raise ValueError('Boundary/padding token included in target loss')
    sid_coeff = weights*sid / (weights*sid).sum().clamp_min(1e-12)
    cue_coeff = cue.to(unweighted.dtype) / cue.sum().clamp_min(1)
    sid_loss = (unweighted*sid_coeff).sum()
    cue_loss = (unweighted*cue_coeff).sum()
    nlls = unweighted*(sid_coeff+cue_lambda*cue_coeff)*active.sum().clamp_min(1)
    return sid_loss+cue_lambda*cue_loss, nlls, sid_loss, cue_loss

def install(Diffusion, Loss, variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    original = Diffusion._loss
    def patched(self, x0, attention_mask, target_mask=None, **kwargs):
        result, output = original(self, x0, attention_mask, target_mask=target_mask, **kwargs)
        if variant == 'legacy':
            return result, output
        # Original returns a detached copy for reporting; nlls retains the graph.
        # Capture the underlying forward output rather than attempting to divide
        # weighted nlls by zero-weight cue positions in the delayed variant.
        raw = self._channel_forward_loss
        weights = self._compute_position_weights(x0)
        lam = cue_coefficient(variant, int(self.global_step))
        total, nlls, sid, cue = channel_loss(raw, result.token_mask, weights,
                                           self.tokenizer.tokens_per_item,
                                           self.tokenizer.n_digit, lam)
        if self.training and getattr(self, '_trainer', None) is not None:
            self.log_dict({'channel/sid_loss': sid.detach(), 'channel/cue_loss': cue.detach(),
                           'channel/cue_lambda': lam}, on_step=True, on_epoch=False)
        return Loss(loss=total, nlls=nlls, token_mask=result.token_mask,
                    unweighted_nlls=raw.detach()), output
    if variant != 'legacy':
        forward = Diffusion._forward_pass_diffusion
        def capture(self, *args, **kwargs):
            result = forward(self, *args, **kwargs)
            self._channel_forward_loss = result
            return result
        Diffusion._forward_pass_diffusion = capture
    Diffusion._loss = patched
