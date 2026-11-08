// The per-turn generation schedule both hosts share. Each batch owns one
// temporary session and its cleanup; this module owns only admission.
export const NATIVE_BATCH_MAX_CONCURRENCY = 5;

export type GenerationPoolOutcome<T> =
  | { kind: "succeeded"; results: T[] }
  // Every failed batch, in batch order. No result is returned beside them.
  | { kind: "failed"; failures: unknown[] };

// Runs `run` once per batch, in order, at most NATIVE_BATCH_MAX_CONCURRENCY
// at a time. A batch calls `stop` as soon as it observes its failure, before
// it cleans up, and then rejects. From the first stop or rejection no batch is
// started, while every started batch drains through its own cleanup. Results,
// in batch order, are accepted only when every batch succeeded.
export async function runGenerationBatches<B, T>(
  batches: readonly B[],
  run: (batch: B, stop: () => void) => Promise<T>,
): Promise<GenerationPoolOutcome<T>> {
  const results: T[] = [];
  const failures: Array<{ error: unknown } | undefined> = [];
  let next = 0;
  let stopped = false;
  const stop = () => {
    stopped = true;
  };
  const worker = async () => {
    while (!stopped && next < batches.length) {
      const index = next;
      next += 1;
      try {
        results[index] = await run(batches[index]!, stop);
      } catch (error) {
        stopped = true;
        failures[index] = { error };
      }
    }
  };
  await Promise.all(
    Array.from(
      { length: Math.min(NATIVE_BATCH_MAX_CONCURRENCY, batches.length) },
      worker,
    ),
  );
  const failed = failures.flatMap(failure => (failure ? [failure.error] : []));
  if (failed.length > 0) return { kind: "failed", failures: failed };
  if (stopped) {
    throw new Error("A generation batch stopped admission without failing.");
  }
  return { kind: "succeeded", results };
}
