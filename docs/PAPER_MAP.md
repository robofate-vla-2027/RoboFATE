# Paper-To-Code Correspondence

References below use the manuscript section, appendix, equation, and table numbers.

| Paper provision | Implementation |
| --- | --- |
| Table 6; 3,400 collected records, 17 tasks | `config.py`, `data.validate_records` |
| Appendix B.1; family-based 80/10/10 split, seed 191126 | `data.assign_families`, `Corpus` |
| Section 3.4.1; three current views, native text/state, eight-query context | `prepare.NativeProcessor`, `model.controller_inputs`, `PerceiverController` |
| Table 6; 64 queries, two blocks, 20 heads, width 2560 | `model.QueryBlock`, `PerceiverController` |
| Equation 2; temporal attention pooling and reset GRU | `PerceiverController.summarize`, `predict` |
| Equation 4; action-plus-risk Top-2 router | `model.MoFFN` |
| Equation 5; masked flow matching | `model.flow_inputs`, `masked_mse` |
| Equation 7; homogeneous original/dual objectives | `RoboFATE.forward`, `train.main` |
| Equations 8-9; eight-expert balancing with Top-2 frequencies summing to two | `MoFFN.forward` |
| Appendix C; only shared MoAE FFNs update on original steps | `discard_inactive_gradients`, `optimizer_for` |
| Appendix C; only visual encoder frozen | `RoboFATE.__init__`, `train` |
| Appendix B.1; independent homogeneous 64-history decision batches | `config.decision_class`, `Corpus.batch` |
| Appendix B.1; uniform task/source and first-100/rest sampling | `Corpus.sample_record`, `choose_window` |
| Table 6; 16 workers x 4 windows x 1 accumulation | `scripts/train.sh`, `train.main` |
| Table 6; 300,000 updates, 2,000 warmup, cosine to 5% | `config.lr_ratio`, `train.main` |
| Table 6; checkpoint interval and latest-three-plus-best retention | `checkpoint_save`, `retained_checkpoints` |
| Appendix B.1; development criterion, noise seeds and early tie-break | `evaluation.evaluate`, `aggregate_sources`, `is_better` |
| Appendix B.1; test only after selecting the checkpoint | `evaluation.main` |
| Appendix B; policy-query cadence, first check at query eight | `data.query_history`, `policy.TakeoverPolicy` |
| Section 3.5.3; fixed risk, one switch, no controller after takeover | `TakeoverPolicy.query` |
| Appendix B.1; annotation evidence and full-state validation | `collection.py` |
| Appendix D.1; 2,550 trials and capability-weighted aggregation | `metrics.main_study_metrics` |

