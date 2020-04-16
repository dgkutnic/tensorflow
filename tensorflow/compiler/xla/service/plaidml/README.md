# XLA PlaidML Backend

The XLA PlaidML backend operates at HLO-level by ingesting a HloModule and
evaluating the result of the HLO graph directly with HloEvaluator, without
lowering it further (to LLVM IR for example) before execution as other backends
(CPU and GPU for example) do.

Its key components are:

*   [`PlaidMLCompiler`] despite the inherited naming of "compiler", all
    `PlaidMLCompiler` really does is the following:
    1.  Runs certain HLO optimization passes on the given HLO graph.
    2.  Generates an `PlaidMLExecutable` from the optimized HLO graph.
    3.  Registers itself in the global compiler factory registry.
*   [`PlaidMLExecutable`]: responsible for running input HLO graph through
    the `HloEvaluator`, allocating output buffer and finally copying evaluated
    Literal result over.
*   [`HloEvaluator`]: traverses a HLO graph and evaluates each node in DFS
    ordering along the way.
