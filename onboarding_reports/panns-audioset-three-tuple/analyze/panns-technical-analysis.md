# Technical Analysis Report: panns-three-tuple-analysis

## Repository Identity

- Repository: audioset_tagging_cnn
- Revision: d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4
- Official status: The repository is the official PANNs source repository because both the paper and the authoritative Zenodo checkpoint record link it as the released code. [verified_upstream_fact] evidence=ev-paper-preprint,ev-zenodo-record

## Scientific Identity

PANNs are pretrained audio neural networks trained on AudioSet for multilabel audio tagging and transfer to downstream audio-pattern-recognition tasks. The peer-reviewed article is IEEE/ACM TASLP 28 (2020), 2880-2894, DOI 10.1109/TASLP.2020.3030497.

- The work trains PANNs on raw AudioSet audio with 527 sound classes and evaluates clip-level multilabel audio tagging. [verified_upstream_fact] evidence=ev-paper-preprint,ev-upstream-readme scope=variants=panns-cnn14-16k,panns-resnet38,panns-wavegram-logmel-cnn14;checkpoints=panns-cnn14-16k-map-0438,panns-resnet38-map-0434,panns-wavegram-logmel-cnn14-map-0439
- The paper reports mAP 0.434 for ResNet38 and mAP 0.439 for Wavegram-Logmel-CNN on AudioSet. [verified_upstream_fact] evidence=ev-paper-preprint scope=variants=panns-resnet38,panns-wavegram-logmel-cnn14;checkpoints=panns-resnet38-map-0434,panns-wavegram-logmel-cnn14-map-0439
- The requested Cnn14_16k checkpoint reports mAP 0.438 and is explicitly described by the repository as trained by a later code version than the paper; the paper's 16 kHz experiment reports mAP 0.427 and must not be substituted for this checkpoint. [verified_upstream_fact] evidence=ev-cnn16-readme,ev-paper-preprint,ev-zenodo-cnn16 scope=variants=panns-cnn14-16k;checkpoints=panns-cnn14-16k-map-0438

## Architecture

All three requested variants consume waveform tensors and include differentiable feature extraction, a 2048-dimensional penultimate representation, and a 527-class AudioSet head, but their frontends and backbones differ.

- Cnn14_16k uses frozen STFT and log-mel extraction followed by six two-convolution blocks with channels 64, 128, 256, 512, 1024, and 2048, global max-plus-mean pooling, a 2048-to-2048 ReLU fully connected layer, and a 527-class head. [verified_upstream_fact] evidence=ev-model-cnn16 scope=variants=panns-cnn14-16k;checkpoints=panns-cnn14-16k-map-0438
- ResNet38 uses frozen STFT and log-mel extraction, an initial convolution block, a basic-block ResNet with layer counts [3,4,6,3], a 512-to-2048 post-ResNet convolution block, global max-plus-mean pooling, a 2048-dimensional fully connected representation, and a 527-class head. [verified_upstream_fact] evidence=ev-model-resnet38,ev-paper-preprint scope=variants=panns-resnet38;checkpoints=panns-resnet38-map-0434
- Wavegram_Logmel_Cnn14 has a learned waveform branch and a frozen STFT/log-mel branch; it concatenates their 64-channel feature maps into a 128-channel tensor before the remaining CNN14 blocks, then uses global max-plus-mean pooling, a 2048-dimensional fully connected representation, and a 527-class head. [verified_upstream_fact] evidence=ev-model-wavegram,ev-paper-preprint scope=variants=panns-wavegram-logmel-cnn14;checkpoints=panns-wavegram-logmel-cnn14-map-0439
- Static comparison found the three requested class bodies unchanged between the default-branch checkpoint-era candidate 802432669263a31af0c448dd8d7b677af3815c01 and the inspected current HEAD, but textual stability does not establish checkpoint equivalence. [locally_observed_behavior] evidence=ev-static-class-comparison scope=variants=panns-cnn14-16k,panns-resnet38,panns-wavegram-logmel-cnn14;checkpoints=panns-cnn14-16k-map-0438,panns-resnet38-map-0434,panns-wavegram-logmel-cnn14-map-0439
- The exact checkpoint-compatible source revision is unresolved independently for each checkpoint. The nearest preceding default-branch ancestor of the Zenodo record creation time is 802432669263a31af0c448dd8d7b677af3815c01; it is a chronology candidate, not proof. [reasoned_inference] evidence=ev-zenodo-record,ev-default-branch-candidate scope=variants=panns-cnn14-16k,panns-resnet38,panns-wavegram-logmel-cnn14;checkpoints=panns-cnn14-16k-map-0438,panns-resnet38-map-0434,panns-wavegram-logmel-cnn14-map-0439
- For Cnn14_16k only, side-branch commit 58d0e624a2055fb1f2b5fb06369ecd726cfe64e0 is an additional same-day chronology candidate because it added the exact variant/checkpoint name shortly before Zenodo record creation; it is not on the inspected default-branch ancestry and is not proven checkpoint-equivalent. [reasoned_inference] evidence=ev-zenodo-record,ev-cnn16-side-branch-candidate scope=variants=panns-cnn14-16k;checkpoints=panns-cnn14-16k-map-0438

## Runtime Interface

No upstream code was imported or executed. Static source exposes a native [B,T] waveform forward method rather than the project's future [B,C,T] plus sample_rate and optional valid_lengths contract.

- Each requested class defines forward(input, mixup_lambda=None), documents input as (batch_size, data_length), and does not accept channel, sample-rate, or valid-length arguments. [verified_upstream_fact] evidence=ev-model-cnn16,ev-model-resnet38,ev-model-wavegram scope=variants=panns-cnn14-16k,panns-resnet38,panns-wavegram-logmel-cnn14;checkpoints=panns-cnn14-16k-map-0438,panns-resnet38-map-0434,panns-wavegram-logmel-cnn14-map-0439
- The reference inference path constructs the class from hyperparameters, loads torch.load(checkpoint)['model'] with load_state_dict, selects CPU or CUDA, loads mono audio at the requested sample rate, calls model.eval(), and runs without gradients. [verified_upstream_fact] evidence=ev-upstream-inference scope=variants=panns-cnn14-16k,panns-resnet38,panns-wavegram-logmel-cnn14;checkpoints=panns-cnn14-16k-map-0438,panns-resnet38-map-0434,panns-wavegram-logmel-cnn14-map-0439
- CPU/CUDA behavior is source-declared only; no device, dtype, length, determinism, checkpoint-loading, or inference behavior was locally executed. [unresolved_ambiguity] evidence=ev-upstream-inference scope=variants=panns-cnn14-16k,panns-resnet38,panns-wavegram-logmel-cnn14;checkpoints=panns-cnn14-16k-map-0438,panns-resnet38-map-0434,panns-wavegram-logmel-cnn14-map-0439

## Preprocessing

Preprocessing is embedded partly in the models and partly in repository data/inference utilities. The source models accept mono waveform batches and perform STFT/log-mel extraction where applicable.

- Cnn14_16k asserts 16,000 Hz, 512-sample Hann window, 160-sample hop, 64 mel bins, 50 Hz lower cutoff, and 8,000 Hz upper cutoff, with centered reflect-padded STFT and frozen log-mel parameters. [verified_upstream_fact] evidence=ev-model-cnn16,ev-cnn16-readme scope=variants=panns-cnn14-16k;checkpoints=panns-cnn14-16k-map-0438
- The paper and reference inference defaults use 32,000 Hz, 1024-sample analysis window, 320-sample hop, 64 mel bins, 50 Hz lower cutoff, and 14,000 Hz upper cutoff for the reported ResNet38 and Wavegram-Logmel-CNN systems. [verified_upstream_fact] evidence=ev-paper-preprint,ev-upstream-inference scope=variants=panns-resnet38,panns-wavegram-logmel-cnn14;checkpoints=panns-resnet38-map-0434,panns-wavegram-logmel-cnn14-map-0439
- The training data path converts audio to mono, resamples, pads or truncates to 10 seconds, clips floating samples to [-1,1] before int16 storage, and restores float32 by division by 32767; the standalone inference path resamples to the requested rate and mono but does not pad or truncate. [verified_upstream_fact] evidence=ev-data-preprocessing,ev-upstream-inference,ev-paper-preprint scope=variants=panns-cnn14-16k,panns-resnet38,panns-wavegram-logmel-cnn14;checkpoints=panns-cnn14-16k-map-0438,panns-resnet38-map-0434,panns-wavegram-logmel-cnn14-map-0439
- Exact project-wrapper policies for multi-channel downmixing, automatic resampling implementation, valid_lengths, and padding/truncation remain unresolved because upstream does not implement the project's public waveform contract. [unresolved_ambiguity] evidence=ev-upstream-inference,ev-project-input-contract scope=variants=panns-cnn14-16k,panns-resnet38,panns-wavegram-logmel-cnn14;checkpoints=panns-cnn14-16k-map-0438,panns-resnet38-map-0434,panns-wavegram-logmel-cnn14-map-0439

## Outputs

All three requested classes return clip-level AudioSet probabilities and a clip-level 2048-dimensional embedding. They do not return pre-sigmoid logits or framewise outputs.

- Each requested forward method returns {'clipwise_output': sigmoid(fc_audioset(x)), 'embedding': dropout(ReLU(fc1(...)))}; with 527 AudioSet classes the output shapes are [B,527] and [B,2048]. [verified_upstream_fact] evidence=ev-model-cnn16,ev-model-resnet38,ev-model-wavegram,ev-paper-preprint scope=variants=panns-cnn14-16k,panns-resnet38,panns-wavegram-logmel-cnn14;checkpoints=panns-cnn14-16k-map-0438,panns-resnet38-map-0434,panns-wavegram-logmel-cnn14-map-0439
- The native clipwise_output is a post-sigmoid multilabel probability tensor, not logits; the project-required raw differentiable classification output would require a provenance-preserving adapter or minimal source adaptation. [reasoned_inference] evidence=ev-model-cnn16,ev-model-resnet38,ev-model-wavegram,ev-project-output-contract scope=variants=panns-cnn14-16k,panns-resnet38,panns-wavegram-logmel-cnn14;checkpoints=panns-cnn14-16k-map-0438,panns-resnet38-map-0434,panns-wavegram-logmel-cnn14-map-0439

## Dependency Evidence

The upstream repository is not packaged and declares a historical Python 3.7 / PyTorch stack. No compatibility claim has been made and no model environment was resolved.

- The README states Python 3.7; requirements.txt pins matplotlib 3.0.3, soundfile 0.10.3.post1, librosa 0.6.3, torch 1.0.1.post2, and torchlibrosa 0.0.4. [verified_upstream_fact] evidence=ev-upstream-readme,ev-upstream-requirements scope=variants=panns-cnn14-16k,panns-resnet38,panns-wavegram-logmel-cnn14;checkpoints=panns-cnn14-16k-map-0438,panns-resnet38-map-0434,panns-wavegram-logmel-cnn14-map-0439
- Static inspection found no setup.py, setup.cfg, pyproject.toml, lockfile, CI matrix, or declared Python version constraint in audioset_tagging_cnn, and found deprecated NumPy aliases in utility paths. [locally_observed_behavior] evidence=ev-repository-inventory,ev-static-dependencies scope=variants=panns-cnn14-16k,panns-resnet38,panns-wavegram-logmel-cnn14;checkpoints=panns-cnn14-16k-map-0438,panns-resnet38-map-0434,panns-wavegram-logmel-cnn14-map-0439
- The upstream-linked panns-inference package/repository is not a complete source for the requested scope: its model module contains Cnn14 and Cnn14_DecisionLevelMax, not the three requested variants, and its wrapper contains an uncontrolled wget checkpoint download path. [verified_upstream_fact] evidence=ev-panns-inference-source scope=variants=panns-cnn14-16k,panns-resnet38,panns-wavegram-logmel-cnn14;checkpoints=panns-cnn14-16k-map-0438,panns-resnet38-map-0434,panns-wavegram-logmel-cnn14-map-0439

## Environment Evidence

No PANNs environment was resolved, created, synchronized, or materialized; no PANNs dependency was installed. Only the pre-existing model-agnostic root control-plane environment ran skill utilities.

- All environment compatibility remains unverified; the declared 2019-era dependency pins are evidence inputs, not a selected modern or historical environment. [unresolved_ambiguity] evidence=ev-upstream-requirements,ev-static-dependencies scope=variants=panns-cnn14-16k,panns-resnet38,panns-wavegram-logmel-cnn14;checkpoints=panns-cnn14-16k-map-0438,panns-resnet38-map-0434,panns-wavegram-logmel-cnn14-map-0439

## Variants

- `panns-cnn14-16k`: Cnn14_16k [verified_upstream_fact] evidence=ev-model-cnn16,ev-cnn16-readme
- `panns-resnet38`: ResNet38 [verified_upstream_fact] evidence=ev-model-resnet38,ev-paper-preprint
- `panns-wavegram-logmel-cnn14`: Wavegram_Logmel_Cnn14 [verified_upstream_fact] evidence=ev-model-wavegram,ev-paper-preprint

## Checkpoint Candidates

- `panns-cnn14-16k-map-0438`
  - filename: Cnn14_16k_mAP=0.438.pth
  - model variant or variant scope: panns-cnn14-16k
  - source type: https
  - URL or source reference: https://zenodo.org/api/records/3987831/files/Cnn14_16k_mAP=0.438.pth/content
  - loader: torch.load(path, map_location=device)['model'] followed by model.load_state_dict
  - published checksum: md5:362fc5ff18f1d6ad2f6d464b45893f2c [published_not_locally_verified] evidence=ev-zenodo-cnn16
    - checksum provenance note: Host-published MD5 from immutable Zenodo record metadata; no checkpoint payload was downloaded and no local digest was computed.
  - expression status: resolved_from_authoritative_metadata
  - unresolved components: exact checkpoint-compatible source revision,locally verified SHA-256,state-dictionary compatibility,runtime behavior
  - access or licensing notes: Zenodo record 3987831 reports open access, record-level CC-BY-4.0 metadata, and size 358668570 bytes. License metadata is informational and non-blocking.
  - claim status: verified_upstream_fact
  - evidence IDs: ev-zenodo-cnn16,ev-cnn16-readme,ev-upstream-inference
- `panns-resnet38-map-0434`
  - filename: ResNet38_mAP=0.434.pth
  - model variant or variant scope: panns-resnet38
  - source type: https
  - URL or source reference: https://zenodo.org/api/records/3987831/files/ResNet38_mAP=0.434.pth/content
  - loader: torch.load(path, map_location=device)['model'] followed by model.load_state_dict
  - published checksum: md5:bf12f36aaabac4e0855e22d3c3239c1b [published_not_locally_verified] evidence=ev-zenodo-resnet38
    - checksum provenance note: Host-published MD5 from immutable Zenodo record metadata; no checkpoint payload was downloaded and no local digest was computed.
  - expression status: resolved_from_authoritative_metadata
  - unresolved components: exact checkpoint-compatible source revision,locally verified SHA-256,state-dictionary compatibility,runtime behavior
  - access or licensing notes: Zenodo record 3987831 reports open access, record-level CC-BY-4.0 metadata, and size 299615982 bytes. License metadata is informational and non-blocking.
  - claim status: verified_upstream_fact
  - evidence IDs: ev-zenodo-resnet38,ev-paper-preprint,ev-upstream-inference
- `panns-wavegram-logmel-cnn14-map-0439`
  - filename: Wavegram_Logmel_Cnn14_mAP=0.439.pth
  - model variant or variant scope: panns-wavegram-logmel-cnn14
  - source type: https
  - URL or source reference: https://zenodo.org/api/records/3987831/files/Wavegram_Logmel_Cnn14_mAP=0.439.pth/content
  - loader: torch.load(path, map_location=device)['model'] followed by model.load_state_dict
  - published checksum: md5:17fa9ab65af3c0eb5ffbc5f65552c4e1 [published_not_locally_verified] evidence=ev-zenodo-wavegram
    - checksum provenance note: Host-published MD5 from immutable Zenodo record metadata; no checkpoint payload was downloaded and no local digest was computed.
  - expression status: resolved_from_authoritative_metadata
  - unresolved components: exact checkpoint-compatible source revision,locally verified SHA-256,state-dictionary compatibility,runtime behavior
  - access or licensing notes: Zenodo record 3987831 reports open access, record-level CC-BY-4.0 metadata, and size 328692198 bytes. License metadata is informational and non-blocking.
  - claim status: verified_upstream_fact
  - evidence IDs: ev-zenodo-wavegram,ev-paper-preprint,ev-upstream-inference

## Embedding Candidates

- `panns-official-post-fc1-embedding`
  - tensor origin: The upstream 'embedding' output: dropout applied to the 2048-dimensional ReLU output of fc1 immediately before fc_audioset; dropout is identity in evaluation mode.
  - semantic kind: pre_logit_representation
  - shape semantics: [B,2048]
  - batch dimension: dimension 0
  - variant scope: panns-cnn14-16k,panns-resnet38,panns-wavegram-logmel-cnn14
  - checkpoint scope: panns-cnn14-16k-map-0438,panns-resnet38-map-0434,panns-wavegram-logmel-cnn14-map-0439
  - decision requirement: required
  - status: verified_upstream_fact
  - unresolved reason: The user left PREFERRED_EMBEDDING unresolved; this is the recommended default because upstream explicitly names and returns it.
  - evidence IDs: ev-paper-preprint,ev-model-cnn16,ev-model-resnet38,ev-model-wavegram,ev-upstream-inference
- `panns-global-pooled-backbone`
  - tensor origin: The sum of temporal global maximum and temporal global mean after frequency pooling and before fc1.
  - semantic kind: pooled_representation
  - shape semantics: [B,2048]
  - batch dimension: dimension 0
  - variant scope: panns-cnn14-16k,panns-resnet38,panns-wavegram-logmel-cnn14
  - checkpoint scope: panns-cnn14-16k-map-0438,panns-resnet38-map-0434,panns-wavegram-logmel-cnn14-map-0439
  - decision requirement: required
  - status: reasoned_inference
  - unresolved reason: This tensor is architecturally meaningful but upstream reserves the term embedding for the following fc1 output; extraction would require an adapter or hook.
  - evidence IDs: ev-paper-preprint,ev-model-cnn16,ev-model-resnet38,ev-model-wavegram
- `panns-pre-global-temporal-representation`
  - tensor origin: The 2048-channel sequence after the final convolution block and frequency-axis mean, immediately before temporal max-plus-mean pooling.
  - semantic kind: frame_level_embedding
  - shape semantics: [B,2048,T_downsampled], with T_downsampled dependent on input length and variant downsampling
  - batch dimension: dimension 0
  - time dimension: dimension 2
  - variant scope: panns-cnn14-16k,panns-resnet38,panns-wavegram-logmel-cnn14
  - checkpoint scope: panns-cnn14-16k-map-0438,panns-resnet38-map-0434,panns-wavegram-logmel-cnn14-map-0439
  - decision requirement: required
  - status: reasoned_inference
  - unresolved reason: The source does not name or return this tensor as an embedding, and exact temporal hop/valid-length semantics require static derivation plus runtime verification.
  - evidence IDs: ev-model-cnn16,ev-model-resnet38,ev-model-wavegram
- `panns-wavegram-branch-feature-map`
  - tensor origin: Wavegram_Logmel_Cnn14.a1 after pre_block4 and before concatenation with the log-mel branch.
  - semantic kind: architectural_intermediate_tensor
  - shape semantics: [B,64,T_branch,32]
  - batch dimension: dimension 0
  - time dimension: dimension 2
  - variant scope: panns-wavegram-logmel-cnn14
  - checkpoint scope: panns-wavegram-logmel-cnn14-map-0439
  - decision requirement: required
  - status: reasoned_inference
  - unresolved reason: This branch-local learned time-frequency tensor is scientifically meaningful but is not an upstream-declared embedding and requires explicit extraction behavior.
  - evidence IDs: ev-model-wavegram,ev-paper-preprint
- `panns-wavegram-logmel-branch-feature-map`
  - tensor origin: Wavegram_Logmel_Cnn14 log-mel branch after conv_block1 and before concatenation.
  - semantic kind: architectural_intermediate_tensor
  - shape semantics: [B,64,T_branch,32]
  - batch dimension: dimension 0
  - time dimension: dimension 2
  - variant scope: panns-wavegram-logmel-cnn14
  - checkpoint scope: panns-wavegram-logmel-cnn14-map-0439
  - decision requirement: required
  - status: reasoned_inference
  - unresolved reason: This branch-local hand-crafted-feature tensor is meaningful for branch comparison but is not an upstream-declared embedding.
  - evidence IDs: ev-model-wavegram,ev-paper-preprint
- `panns-wavegram-logmel-fused-feature-map`
  - tensor origin: Wavegram_Logmel_Cnn14 concatenation of the wavegram and log-mel branch tensors before conv_block2.
  - semantic kind: sequence_level_embedding
  - shape semantics: [B,128,T_branch,32]
  - batch dimension: dimension 0
  - time dimension: dimension 2
  - variant scope: panns-wavegram-logmel-cnn14
  - checkpoint scope: panns-wavegram-logmel-cnn14-map-0439
  - decision requirement: required
  - status: reasoned_inference
  - unresolved reason: This is the first fused dual-branch representation, but it is not returned or formally named as an embedding upstream.
  - evidence IDs: ev-model-wavegram,ev-paper-preprint

## Source Strategy Candidates

- `official_package` [reasoned_inference]: The official panns-inference package is rejected as a complete strategy for this three-tuple scope because its inspected source does not contain Cnn14_16k, ResNet38, or Wavegram_Logmel_Cnn14 and its wrapper can trigger an uncontrolled checkpoint download. evidence=ev-panns-inference-source,ev-user-scope
- `pinned_official_git_repository` [unresolved_ambiguity] decision-required: The official repository contains all three classes and immutable revisions, but it has no installable package metadata and the exact checkpoint-associated revision is unresolved for each checkpoint. evidence=ev-model-cnn16,ev-model-resnet38,ev-model-wavegram,ev-default-branch-candidate,ev-repository-inventory
- `minimal_vendored_adaptation` [reasoned_inference] decision-required: This is the leading implementation strategy because the official repository is not packaged, only a bounded subset is needed, and a provenance-preserving adaptation can expose pre-sigmoid logits and the requested embeddings while documenting every change. evidence=ev-repository-inventory,ev-model-cnn16,ev-model-resnet38,ev-model-wavegram,ev-project-output-contract

## Open Questions

- `q-default-embedding` needs_user_decision: Select the default embedding. The recommended first choice is panns-official-post-fc1-embedding for all three tuples because it is explicitly named, returned, and used for transfer upstream; temporal and Wavegram branch candidates can remain separately accessible. evidence=ev-user-embedding-unresolved,ev-paper-preprint,ev-model-cnn16,ev-model-resnet38,ev-model-wavegram
- `q-source-strategy` needs_user_decision: Choose between a reproducible pinned-repository import plan and the leading minimal-vendored-adaptation plan after accepting that the official package does not cover the requested three tuples. evidence=ev-repository-inventory,ev-panns-inference-source,ev-project-output-contract
- `q-source-revision-cnn16` needs_more_evidence: The exact source revision for Cnn14_16k_mAP=0.438.pth remains unresolved; retain default-branch candidate 802432669263a31af0c448dd8d7b677af3815c01 and side-branch candidate 58d0e624a2055fb1f2b5fb06369ecd726cfe64e0 without claiming equivalence. evidence=ev-zenodo-cnn16,ev-default-branch-candidate,ev-cnn16-side-branch-candidate
- `q-source-revision-resnet38` needs_more_evidence: The exact source revision for ResNet38_mAP=0.434.pth remains unresolved; 802432669263a31af0c448dd8d7b677af3815c01 is only the nearest preceding default-branch candidate. evidence=ev-zenodo-resnet38,ev-default-branch-candidate
- `q-source-revision-wavegram` needs_more_evidence: The exact source revision for Wavegram_Logmel_Cnn14_mAP=0.439.pth remains unresolved; 802432669263a31af0c448dd8d7b677af3815c01 is only the nearest preceding default-branch candidate. evidence=ev-zenodo-wavegram,ev-default-branch-candidate
- `q-environment-resolution` needs_environment_resolution: Resolve one or more evidence-motivated isolated environments for the selected source strategy; do not assume the declared Python 3.7 / torch 1.0.1.post2 stack is the only viable candidate. evidence=ev-upstream-requirements,ev-static-dependencies
- `q-checkpoint-runtime-verification` needs_runtime_probe: Later verify acquisition SHA-256, state-dictionary loading, shapes, values, device behavior, and deterministic behavior independently for all three checkpoint files. Published MD5 metadata does not satisfy local SHA-256 verification. evidence=ev-zenodo-cnn16,ev-zenodo-resnet38,ev-zenodo-wavegram,ev-upstream-inference

## Decisions

- `d-mode-analyze-only` [user_provided_decision]: Execute only analyze mode and prohibit environment resolution, model execution, checkpoint acquisition, integration artifacts, and repository changes. selected=analyze evidence=ev-user-scope
- `d-three-tuple-scope` [user_provided_decision]: Analyze three separate model/variant/checkpoint tuples without reducing scope to a single implementation slice. selected=all-three-tuples evidence=ev-user-scope
- `d-authoritative-checkpoint-record` [user_provided_decision]: Use Zenodo record 3987831 as the authoritative checkpoint-record candidate. selected=zenodo-3987831 evidence=ev-user-scope
- `d-embedding-deferred` [user_provided_decision]: Leave the preferred embedding unresolved during analyze mode. selected=unresolved evidence=ev-user-embedding-unresolved

## Evidence

- `ev-upstream-readme` official_documentation [verified_upstream_fact] README.md: Official README identifies PANNs, AudioSet, 527 classes, Python 3.7, Zenodo checkpoints, metrics, and usage.
- `ev-cnn16-readme` official_documentation [verified_upstream_fact] README.md: Official README maps Cnn14_16k_mAP=0.438.pth to Cnn14_16k and exact 16 kHz frontend arguments, and states it was trained by a later code version than the paper.
- `ev-upstream-license` source_file [verified_upstream_fact] LICENSE.MIT: Official repository MIT license text.
- `ev-paper-preprint` paper [verified_upstream_fact] https://arxiv.org/pdf/1912.10211: Reviewed arXiv v5 paper text for scientific identity, architecture, preprocessing, embeddings, AudioSet metrics, and released repository.
- `ev-paper-peer-reviewed` paper [verified_upstream_fact] https://ieeexplore.ieee.org/document/9229505: Peer-reviewed IEEE/ACM TASLP publication identity, volume 28, pages 2880-2894, DOI 10.1109/TASLP.2020.3030497.
- `ev-model-cnn16` source_line_or_symbol [verified_upstream_fact] pytorch/models.py: Official Cnn14_16k constructor and forward path define frontend assertions, CNN topology, 2048-dimensional representation, sigmoid head, and returned embedding.
- `ev-model-resnet38` source_line_or_symbol [verified_upstream_fact] pytorch/models.py: Official ResNet38 constructor and forward path define log-mel frontend, [3,4,6,3] residual topology, 2048-dimensional representation, sigmoid head, and returned embedding.
- `ev-model-wavegram` source_line_or_symbol [verified_upstream_fact] pytorch/models.py: Official Wavegram_Logmel_Cnn14 constructor and forward path define dual waveform/log-mel branches, fusion, 2048-dimensional representation, sigmoid head, and returned embedding.
- `ev-upstream-inference` source_line_or_symbol [verified_upstream_fact] pytorch/inference.py: Official reference inference statically defines construction, torch.load()['model'] loading, eval/no-grad use, mono resampling, default 32 kHz frontend arguments, and output handling.
- `ev-data-preprocessing` source_line_or_symbol [verified_upstream_fact] utils/utilities.py: Official utilities and dataset paths define mono loading, resampling, 10-second padding/truncation, int16 clipping/storage, and float32 restoration.
- `ev-upstream-requirements` configuration_file [verified_upstream_fact] requirements.txt: Official exact dependency declarations: matplotlib 3.0.3, soundfile 0.10.3.post1, librosa 0.6.3, torch 1.0.1.post2, torchlibrosa 0.0.4.
- `ev-zenodo-record` official_documentation [verified_upstream_fact] https://zenodo.org/records/3987831: Immutable Zenodo record 3987831 metadata: record created 2020-08-17T09:42:16.515697Z, open access, record-level CC-BY-4.0, concept DOI 10.5281/zenodo.3576402, record DOI 10.5281/zenodo.3987831, and official repository link; no Git revision is recorded.
- `ev-zenodo-cnn16` official_documentation [verified_upstream_fact] https://zenodo.org/records/3987831: Zenodo record file entry for Cnn14_16k_mAP=0.438.pth: 358668570 bytes, checksum md5:362fc5ff18f1d6ad2f6d464b45893f2c.
- `ev-zenodo-resnet38` official_documentation [verified_upstream_fact] https://zenodo.org/records/3987831: Zenodo record file entry for ResNet38_mAP=0.434.pth: 299615982 bytes, checksum md5:bf12f36aaabac4e0855e22d3c3239c1b.
- `ev-zenodo-wavegram` official_documentation [verified_upstream_fact] https://zenodo.org/records/3987831: Zenodo record file entry for Wavegram_Logmel_Cnn14_mAP=0.439.pth: 328692198 bytes, checksum md5:17fa9ab65af3c0eb5ffbc5f65552c4e1.
- `ev-current-head` source_line_or_symbol [locally_observed_behavior] README.md: Read-only clone observed origin/master and origin/HEAD at d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4 dated 2021-07-13.
- `ev-default-branch-candidate` source_line_or_symbol [verified_upstream_fact] pytorch/models.py: Commit 802432669263a31af0c448dd8d7b677af3815c01, dated 2020-07-30, is the nearest preceding commit on current master ancestry before Zenodo record creation and contains all three requested classes; no checkpoint association is asserted.
- `ev-cnn16-side-branch-candidate` source_line_or_symbol [verified_upstream_fact] README.md: Side-branch commit 58d0e624a2055fb1f2b5fb06369ecd726cfe64e0, dated 2020-08-17T09:36:37Z, added the exact Cnn14_16k checkpoint name shortly before Zenodo record creation; it is not an ancestor of current master.
- `ev-repository-inventory` source_line_or_symbol [locally_observed_behavior] README.md: Canonical static repository and Python-project inspectors observed the source tree, MIT license, requirements.txt, absent packaging metadata, no checkpoint files, and no tags/releases in the clone.
- `ev-static-dependencies` configuration_file [locally_observed_behavior] requirements.txt: Canonical dependency inspector normalized all five exact requirement pins and observed deprecated np.int/np.float API risks in utility files.
- `ev-static-class-comparison` source_line_or_symbol [locally_observed_behavior] pytorch/models.py: Read-only Git diff inspection observed no textual changes to Cnn14_16k, ResNet38, or Wavegram_Logmel_Cnn14 class bodies between 802432669263a31af0c448dd8d7b677af3815c01 and current HEAD; this is not runtime or checkpoint evidence.
- `ev-panns-inference-source` package_metadata [verified_upstream_fact] setup.py: Upstream-linked panns-inference source at f673f604ec6f4805a61c5b3be087e24776ec5fda declares package version 0.1.1, contains only Cnn14 and Cnn14_DecisionLevelMax model classes, and can invoke wget for checkpoints.
- `ev-project-input-contract` configuration_file [verified_upstream_fact] project_spec.md: Project specification requires public [B,C,T] waveform input plus sample_rate and optional [B] valid_lengths, with wrapper-owned preprocessing.
- `ev-project-output-contract` configuration_file [verified_upstream_fact] project_spec.md: Project specification requires forward() to return raw differentiable outputs without automatically applying sigmoid and requires all meaningful embeddings to remain accessible.
- `ev-user-scope` user_decision [user_provided_decision] none: User selected analyze-only consumer mode, the three exact tuples, Zenodo record 3987831, and explicit prohibitions on environment/model/checkpoint/integration/repository mutations.
- `ev-user-embedding-unresolved` user_decision [user_provided_decision] none: User specified PREFERRED_EMBEDDING: UNRESOLVED.

## Confidence Summary

- verified_fact_count: 47
- locally_observed_count: 8
- inference_count: 10
- unresolved_count: 11
- unsupported_claim_count: 0

Recommended next mode: `resolve-environment`
