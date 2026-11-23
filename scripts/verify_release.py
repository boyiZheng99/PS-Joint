#!/usr/bin/env python3
"""Validate that the public config represents the final PS-Joint variant."""

from pathlib import Path

from mmengine.config import Config


config_path = (Path(__file__).resolve().parents[1] / 'configs' /
               'ps_joint' / 'ps_joint_retinal.py')
cfg = Config.fromfile(config_path)

assert '_base_' not in cfg
assert cfg.model.type == 'PSJoint'
for name in ('decode_head', 'dino_decode_head'):
    head = cfg.model[name]
    assert head.type == 'PSJointDecodeHead'
    assert head.interaction_cfg.reparam_target == 'key'
    assert head.interaction_cfg.key_noise_clip == 2.35
    assert head.mahalanobis_feature_space == 'patch'
    assert head.mahalanobis_variance_mode == 'query_only'
    assert head.mahalanobis_detrseg_cfg.force_fp32 is True
    assert head.mahalanobis_detrseg_cfg.variance_floor == 1e-4

print(f'Validated final PS-Joint config: {config_path}')
