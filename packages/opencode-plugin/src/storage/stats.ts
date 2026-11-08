// The one owner of persisted conversation stats. Every write replaces the
// file atomically through `replaceJSONFile`, and every change to one stats
// file runs after the previous change to that file in this process has
// settled. Other processes are not coordinated.
import { readFile, rm } from "node:fs/promises";
import { z } from "zod";
import { PluginMessageError } from "../policy/feedback";
import { pluginStorageDirectory, replaceJSONFile } from "./store";

const ConversationStatsSchema = z.object({
  version: z.literal(1),
  sourceSessionId: z.string().nullable(),
  rootSessionId: z.string(),
  totalTokensPruned: z.number(),
  cachedTokensSaved: z.number(),
  mediaTokensProjected: z.number().optional(),
  cachedMediaTokensSaved: z.number().optional(),
  processedAssistantMessageIds: z.array(z.string()),
});

export type ConversationStats = z.infer<typeof ConversationStatsSchema>;

// A stats file that exists but is not valid JSON or does not match the stats
// schema. Its totals cannot be used, and replacing it would lose them, so the
// file is left as it is. The message is fixed text: it never names the path
// or repeats the file's content.
export class InvalidStatsError extends PluginMessageError {
  constructor() {
    super(
      "Magic Compact's saved stats for this session are not valid, so the command stopped without using them and left the stats file and the session unchanged. To continue, move the session's stats file out of Magic Compact's storage directory; the next command starts new totals.",
    );
    this.name = "InvalidStatsError";
  }
}

// Null when no stats file exists. A file that cannot be validated throws
// `InvalidStatsError`; any other read failure, such as a permission error,
// is rethrown with its own system error code.
export async function readStats(
  statsKey: string,
): Promise<ConversationStats | null> {
  let text: string;
  try {
    text = await readFile(statsPath(statsKey), "utf8");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return null;
    throw error;
  }
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    throw new InvalidStatsError();
  }
  const parsed = ConversationStatsSchema.safeParse(value);
  if (!parsed.success) throw new InvalidStatsError();
  return parsed.data;
}

// For a command that will need the saved totals: fails before its model
// requests instead of after them.
export async function assertStatsUsable(statsKey: string): Promise<void> {
  await readStats(statsKey);
}

export async function writeStats(
  statsKey: string,
  stats: ConversationStats,
): Promise<void> {
  await withStatsMutation(statsKey, () => replaceStats(statsKey, stats));
}

export async function deleteStats(statsKey: string): Promise<void> {
  await withStatsMutation(statsKey, () =>
    rm(statsPath(statsKey), { force: true }),
  );
}

// Adds the conversation's current savings once per completed assistant
// message. A session without stats records nothing; invalid stats throw
// `InvalidStatsError` and are not written.
export async function recordAssistantCompletion(
  statsKey: string,
  messageID: string,
): Promise<void> {
  await withStatsMutation(statsKey, async () => {
    const stats = await readStats(statsKey);
    if (!stats || stats.processedAssistantMessageIds.includes(messageID)) {
      return;
    }
    await replaceStats(statsKey, {
      ...stats,
      cachedTokensSaved: stats.cachedTokensSaved + stats.totalTokensPruned,
      cachedMediaTokensSaved:
        (stats.cachedMediaTokensSaved ?? 0) + (stats.mediaTokensProjected ?? 0),
      processedAssistantMessageIds: [
        ...stats.processedAssistantMessageIds,
        messageID,
      ],
    });
  });
}

// Working-conversation stats carry the source's totals forward and add the
// command's own reduction. Every copied assistant message is recorded as
// processed, so replayed history never adds a cached-read increment; only new
// responses in the successor do. The source's stats are only read: the read
// waits in the source's queue, so it sees every change admitted before it,
// and leaves the queue before the successor write, so the two files are never
// held together.
export async function writeSuccessorStats(input: {
  sourceKey: string;
  successorKey: string;
  sourceSessionID: string;
  tokensPruned: number;
  mediaTokensProjected: number;
  assistantMessageIDs: string[];
}): Promise<void> {
  const previous = await withStatsMutation(input.sourceKey, () =>
    readStats(input.sourceKey),
  );
  await writeStats(input.successorKey, {
    version: 1,
    sourceSessionId: input.sourceSessionID,
    rootSessionId: previous?.rootSessionId ?? input.sourceSessionID,
    totalTokensPruned: (previous?.totalTokensPruned ?? 0) + input.tokensPruned,
    cachedTokensSaved: previous?.cachedTokensSaved ?? 0,
    mediaTokensProjected: input.mediaTokensProjected,
    cachedMediaTokensSaved: previous?.cachedMediaTokensSaved ?? 0,
    processedAssistantMessageIds: input.assistantMessageIDs,
  });
}

function statsPath(statsKey: string): string {
  return `${pluginStorageDirectory()}/stats/${statsKey}.json`;
}

async function replaceStats(
  statsKey: string,
  stats: ConversationStats,
): Promise<void> {
  await replaceJSONFile(statsPath(statsKey), stats);
}

const mutationTails = new Map<string, Promise<void>>();

// Runs `operation` after every earlier mutation of the same stats file has
// settled. A failed mutation rejects only its own caller.
function withStatsMutation<T>(
  statsKey: string,
  operation: () => Promise<T>,
): Promise<T> {
  const result = (mutationTails.get(statsKey) ?? Promise.resolve()).then(
    operation,
  );
  const tail = result.then(
    () => undefined,
    () => undefined,
  );
  mutationTails.set(statsKey, tail);
  void tail.then(() => {
    if (mutationTails.get(statsKey) === tail) mutationTails.delete(statsKey);
  });
  return result;
}
