// Summary and generation forks still owned by a running OpenCode 1 command.
// A fork is registered before its first request and stays registered until
// its deletion is confirmed, so a fork whose deletion failed keeps every
// guard. Only this process registers a fork; nothing stored in a session can
// claim ownership.
//
// OpenCode 1.18.30 runs the plugin's `experimental.session.compacting` hook
// at the start of every native compaction, whether a finished step's token
// count or a provider overflow triggered it, and before its summary request,
// prompt replay or auto-continue. The hook throws for an owned fork, which
// ends the fork's prompt loop without another model request, and records that
// it did so. The command reads that record instead of trusting whatever error
// the host returns for the prompt.
export type OwnedFork = {
  readonly id: string;
  nativeCompactionBlocked: boolean;
};

const ownedForks = new Map<string, OwnedFork>();

export function registerOwnedFork(id: string): OwnedFork {
  const fork: OwnedFork = { id, nativeCompactionBlocked: false };
  ownedForks.set(id, fork);
  return fork;
}

// Only after the host confirmed the deletion.
export function releaseOwnedFork(fork: OwnedFork): void {
  ownedForks.delete(fork.id);
}

export function isOwnedFork(sessionID: string): boolean {
  return ownedForks.has(sessionID);
}

export function blockNativeCompaction(sessionID: string): void {
  const fork = ownedForks.get(sessionID);
  if (!fork) return;
  fork.nativeCompactionBlocked = true;
  throw new Error(
    "Magic Compact owns this session; native compaction is blocked.",
  );
}
