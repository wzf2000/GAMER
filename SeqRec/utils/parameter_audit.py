"""Independent parameter execution and gradient probes; never update weights."""

import hashlib

import torch


def _has_nonempty_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.numel() > 0
    if isinstance(value, dict):
        return any(_has_nonempty_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_nonempty_tensor(item) for item in value)
    return False


def audit_parameters(model, batches):
    """Count tensor objects, unioned over the supplied batches, with explicit scope."""
    parameters = dict(model.named_parameters())
    names = {id(parameter): name for name, parameter in parameters.items()}
    executed, reachable, nonzero = set(), set(), set()
    observed = {}
    hooks = []

    def record(module, args, kwargs):
        if _has_nonempty_tensor((args, kwargs)):
            executed.update(names[id(parameter)] for parameter in module.parameters(recurse=False) if id(parameter) in names)

    training_modes = [(module, module.training) for module in model.modules()]
    try:
        for module in model.modules():
            if list(module.parameters(recurse=False)):
                hooks.append(module.register_forward_pre_hook(record, with_kwargs=True))
        # Dropout is disabled so zero-gradient classifications can be reproduced.
        model.eval()
        trainable = [(name, parameter) for name, parameter in parameters.items() if parameter.requires_grad]
        for batch in batches:
            output = model(**{key: value for key, value in batch.items() if key != 'use_cache'}, use_cache=False)
            if output.loss is None or not torch.isfinite(output.loss):
                raise ValueError('Parameter audit requires a finite supervised loss.')
            gradients = torch.autograd.grad(output.loss, [parameter for _, parameter in trainable], allow_unused=True)
            for (name, parameter), gradient in zip(trainable, gradients):
                if gradient is not None:
                    reachable.add(name)
                    if torch.count_nonzero(gradient).item():
                        nonzero.add(name)
            observed[str(len(observed))] = {
                'loss': output.loss.item(), 'shape': list(batch['input_ids'].shape),
                'supervised_positions': int((batch['labels'][:, 1:] != -100).sum().item()),
                'tensor_sha256': _batch_digest(batch),
            }
        if not observed:
            raise ValueError('At least one probe batch is required.')
    finally:
        for hook in hooks:
            hook.remove()
        for module, training in training_modes:
            module.training = training

    def count(selected):
        return sum(parameters[name].numel() for name in selected)

    active = executed & reachable
    return {'schema_version': 1, 'total': count(parameters), 'trainable': count(name for name, p in parameters.items() if p.requires_grad),
            'buffers': sum(buffer.numel() for buffer in model.buffers()), 'forward_executed': count(executed),
            'loss_reachable': count(reachable), 'active_participating': count(active), 'nonzero_gradient_tensors': count(nonzero),
            'observed_not_executed': sorted(set(parameters) - executed),
            'observed_no_loss_path': sorted(set(name for name, p in parameters.items() if p.requires_grad) - reachable),
            'observed_zero_gradient': sorted(reachable - nonzero), 'batches': observed,
            'definition': 'active_participating counts whole unique trainable parameter tensors both called with nonempty inputs and connected to the probe loss; shared tensors count once.',
            'scope': 'Observed batch union in eval mode with gradients enabled. Zero gradient or no observation is not proof of global structural unreachability. Embedding counts cover whole tensors, not accessed rows.'}


def _batch_digest(batch):
    digest = hashlib.sha256()
    for key, value in sorted(batch.items()):
        if isinstance(value, torch.Tensor):
            digest.update(str((key, value.dtype, tuple(value.shape))).encode())
            digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def synthetic_probe_batch(config, device='cpu'):
    """Cover every behavior and item-token position with a causal shifted label mask."""
    behavior_ids = [int(token) for token in config.behavior_maps]
    special = set(behavior_ids) | {config.bos_token_id, config.pad_token_id, config.eos_token_id}
    item_ids = [token for token in range(config.vocab_size) if token not in special][:config.num_positions - 1]
    if len(item_ids) != config.num_positions - 1 or not behavior_ids:
        raise ValueError('Not enough vocabulary entries/behaviors for a synthetic probe.')
    sequence = [token for behavior in behavior_ids for token in [behavior, *item_ids]]
    if len(sequence) + 1 > config.model_max_length:
        raise ValueError('Synthetic probe exceeds model_max_length; provide an explicit batch file.')
    if config.pad_token_id is None:
        raise ValueError('A padding token is required for the synthetic training probe.')
    ids = torch.tensor([sequence + [config.pad_token_id]], device=device)
    labels = ids.clone()
    for token in [*behavior_ids, config.pad_token_id]:
        if token is not None:
            labels[ids == token] = -100
    mask = torch.ones_like(ids)
    mask[:, -1] = 0
    return {'input_ids': ids, 'attention_mask': mask, 'labels': labels}
