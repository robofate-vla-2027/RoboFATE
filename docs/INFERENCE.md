# Inference Interface

`TakeoverPolicy(model, q01, q99, seed)` owns episode-local summary, queue,
and risk state. The model is fixed in evaluation mode. The host supplies
native tokenized current and action contexts, using the same processor
and normalization as training.

1. Call `reset()` at an episode boundary.
2. At observation zero, call `query(0, current_context, action_context, remaining_actions)`.
3. Consume the returned commands with `next_command()` and execute them
   through the native API. Stop at the native terminal condition, remaining
   time limit, workspace limit, or operator stop.
4. Replan after each 25 applied commands with a fresh observation. Current
   context excludes history-image slots; action context includes the native
   twenty-slot head-camera history.
5. Call `stop()` on any termination or external safety stop to clear the
   pending queue.

Before takeover, each query adds one pooled Perceiver summary. Starting at
query 8, the GRU checks the most recent eight summaries with zero initial
hidden state. A probability of at least 0.5 stores the risk embedding,
discards pending BAE commands, and selects MoAE for the next chunk.
After takeover, only fresh action-backbone prefixes and MoAE are evaluated.
The fixed risk persists until reset. Both decoders use ten Euler steps.

The interface is synchronous: the host waits for a completed query before
consuming its new commands. It does not send commands during model
computation. Native actuator holding behavior, action/time budget accounting,
and an independent stop mechanism belong to the environment's existing
control API. No generated action in the test suite is sent to an actuator.

For an admitted restored at-risk start, `prime(seven_preceding_contexts, n)`
loads the seven earlier query summaries into a fresh policy. The normal
`query(n, current_context, action_context, remaining_actions)` then appends
the boundary observation and performs the first check before continuation.
The history consists only of observations available at that boundary.

A stopped or failed query leaves no consumable commands. Call `reset()`
before starting or priming another episode.
