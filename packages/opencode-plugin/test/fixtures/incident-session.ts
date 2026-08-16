import type { Message, Part, TextPart } from "@opencode-ai/sdk/v2";
import type { MessageWithParts } from "../../src/compact/plan";

export function incidentSemanticSessionFixture(): MessageWithParts[] {
  // Bounded excerpts from backup session ses_ff8333a4fffe8e42ZtOwcgV2gx.
  const gitLogOutput = [
    "6560c0c Preserve provider output-limit compatibility",
    "755cafa Fix compaction latency and low-yield summaries",
    "9407378 fix(opencode): avoid persisted format decode failures",
    "d7ea044 fix(opencode): harden compaction summary output",
    "f8ba518 Remove Chinese documentation",
    "679115c Remove Claude Code support",
    "49454ba fix(opencode): fall back from zero token usage",
    "96f3234 fix(opencode): harden compaction fidelity",
    "5870e2c fix(opencode): preserve native compaction checkpoints",
    "2356a77 feat(opencode): add dedicated compaction summarizer",
  ].join("\n");
  return [
    message("msg_007ccc5bc001BacSEd5EtblOyC", "user", [
      textPart(
        "prt_007ccc5c3001pA0CxLQApmphve",
        "msg_007ccc5bc001BacSEd5EtblOyC",
        'The compaction always saves trivial amount of context window, but magic-trim still seems to work quite well. I still believe in the philosophy underpinning magic compact where the idea is the session context is compacted but the actually processes and steps taken aren\'t lost. Therefore information is compacted "in-place" instead of monolithically at the end.',
      ),
    ]),
    message("msg_007ccc70e001MbfC10aCW4nnUZ", "assistant", [
      toolPart(
        "prt_007ccc714001jxpP0R6CmvNTJP",
        "msg_007ccc70e001MbfC10aCW4nnUZ",
        "bash",
        {
          status: "completed",
          input: {
            command: "git log --oneline -10",
            workdir:
              "/home/ivan/workspaces/magic-compact/.hive/.worktrees/adhoc/20260815-143352747-compact-per-turn-yield-i5qx4q",
          },
          output: `${gitLogOutput}\n`,
          title: "git log --oneline -10",
          metadata: {
            output: `${gitLogOutput}\n`,
            exit: 0,
            truncated: false,
          },
          time: { start: 1786806942191, end: 1786806942195 },
        },
      ),
    ]),
    message("msg_007ccc759001AH8GErYi9Ob4G2", "assistant", [
      textPart(
        "prt_007ccc75f001eu9c4M2C8POuYT",
        "msg_007ccc759001AH8GErYi9Ob4G2",
        [
          "**Fix**",
          "`packages/opencode-plugin/src/magic-compact.ts` now:",
          "",
          "- Projects every candidate turn exactly once.",
          "- Replaces it only when summary plus planned pruning is strictly smaller than the original turn.",
          "- Leaves rejected turns message- and part-identical.",
          "- Reuses the exact projected omission IDs through mutation.",
          "- Calculates the aggregate savings gate only over selected profitable turns.",
          "",
          "This preserves the in-place philosophy: compress when information can be represented more compactly; otherwise retain the original process unchanged.",
          "",
          "Documentation and canonical regressions were updated. Commit: `e96323b`.",
          "",
          "**Verification**",
          "- `260` tests passed, `0` failed.",
          "- Typecheck passed.",
          "- ESLint passed.",
          "- Prettier passed.",
          "- Post-merge worktree is clean.",
        ].join("\n"),
      ),
    ]),
    message("msg_007ccc762001eXoxBUECF4laHm", "user", [
      textPart(
        "prt_007ccc763001Nv4rBDD5Bthwe2",
        "msg_007ccc762001eXoxBUECF4laHm",
        'I think a other issue that feeds this is our summary methodology might just be too verbose and spend way too many tokens. I\'m thinking we should simplify this Abit and make each summary closer to a small scope "compaction" for each turn that covers what was done there but we still need it to be somewhat try to be loss less too. Human legibility and token saving is somewhat a bonus too as currently a compacted session may be agent legible but a human loses all ability to identify what has happened in it.',
      ),
    ]),
    message("msg_007ccc7be001T1P21AStyJWksY", "assistant", [
      textPart(
        "prt_007ccc7c9001aAV1lDQ234WyHY",
        "msg_007ccc7be001T1P21AStyJWksY",
        [
          "The cleaner design is a small per-turn **continuation capsule**:",
          "",
          "```text",
          "Outcome: completed | partial",
          "Work: <exact assistant text parts>",
          "Verification: <passed, failed, or not run>",
          "State: <important VCS/deployment/session fact, only if relevant>",
          "Next: <unresolved action or none>",
          "Archive: <Content ID>",
          "```",
          "",
          "Standard capsules should target 64–72 tokens and cap at 128; deterministic unsafe capsules should cap around 160 and remain model-free. No raw JSON, source windows, per-record counters, or long identifier inventories should remain in active context.",
        ].join("\n"),
      ),
    ]),
    message("msg_007ccc7cd001XqLcxB1rvWMUwx", "assistant", [
      textPart(
        "prt_007ccc7dc0010PAunlTMIOo5g7",
        "msg_007ccc7cd001XqLcxB1rvWMUwx",
        "the “lossless” goal should mean preserving operational continuity, decisions, outcomes, verification, and unresolved work, not byte-for-byte assistant wording.",
      ),
    ]),
    message("msg_007ccc7f0001cFBcKqv6RN3Rhz", "assistant", [
      textPart(
        "prt_007ccc7fc001Y12vYJyyDzN9e1",
        "msg_007ccc7f0001cFBcKqv6RN3Rhz",
        "You’re right to challenge it. I mixed two different measurements: the 12.5K figure is only the visible summary text, while the compaction report measures the whole canonical session, and OpenCode’s context percentage can include additional prompt/tool-schema overhead that Magic Compact does not count. I’ll re-read the exact notices and lineage for that session and reconcile all three numbers before using them as an acceptance target.",
      ),
    ]),
    message("msg_007ccc801001tjqgwsc4G672rD", "user", [
      textPart(
        "prt_007ccc819001Wjz8KVnNvokg8J",
        "msg_007ccc801001tjqgwsc4G672rD",
        "Note I don't want hard acceptance targets for compaction, onces again it's correctness over forced brevity, but I just think something is off. Maybe it's not summaries that's causing the low compaction % then? I still think we need to make our summaries human legible though",
      ),
    ]),
    message("msg_007ccc81d001FkgGaNTmnBvv3H", "assistant", [
      textPart(
        "prt_007ccc86e001GENwK7fYUgS3fk",
        "msg_007ccc81d001FkgGaNTmnBvv3H",
        [
          "You’re right. I was turning an observed pathology into a quota, which would recreate the same mistake in the opposite direction. We should not require “80 tokens” or “≤3.5K total” if correctness needs more.",
          "",
          "The revised summary rule should therefore be qualitative: produce the shortest plain-language capsule that preserves every continuation-critical fact. No arbitrary target. Keep only truthful outcome, work/decision, verification state, unresolved work/next action, and identifiers that are actionable. Never expose raw JSON evidence. If a turn cannot be represented both correctly and profitably, leave it unchanged. Existing token limits remain safety ceilings, not output goals.",
        ].join("\n"),
      ),
    ]),
    message("msg_007ccc873001psCDHyISrmgRIQ", "assistant", [
      textPart(
        "prt_007ccc87f001q74IVppwQu98v2",
        "msg_007ccc873001psCDHyISrmgRIQ",
        [
          "I’ll implement the two fixes as one coherent behavior change:",
          "",
          "- Visible summaries become plain-language continuation capsules. The evidence engine remains internal; deterministic unresolved turns stay model-free and fail closed. No new brevity quota will be used beyond existing safety ceilings and exact profitability.",
          "- Compaction reporting will label its before/after values as canonical stored-content estimates and explicitly surface that OpenCode’s provider/context meter may remain a stale pre-compaction usage snapshot until another model response refreshes it.",
          "",
          "Tests will treat semantic completeness, truthful unresolved state, human-readable shape, and counting-basis clarity as the acceptance criteria.",
        ].join("\n"),
      ),
    ]),
    message("msg_007ccca69001115rCZDcJXzA3j", "assistant", [
      textPart(
        "prt_007ccca70001fCDMwGxQcScdqU",
        "msg_007ccca69001115rCZDcJXzA3j",
        [
          "Implemented and merged as commit `082f704`.",
          "",
          "**Changes**",
          "- Visible summaries are now readable `Outcome` / `Work` / `Verification` / `VCS` / `Unresolved` / `Next` capsules.",
          "- No arbitrary brevity targets were added. Existing token limits remain safety ceilings.",
          "- Deterministic unsafe turns remain model-free.",
          "- Full final responses and distinct actionable facts are preserved when possible; otherwise compaction fails closed.",
          "- Capsules are deterministic and exact; there is no model-written summary prose.",
          "- Retrieval remains exclusively for tool I/O.",
          "",
          "**Verification**",
          "- `332` tests passed, `0` failed.",
          "- Typecheck, ESLint, Prettier, and `git diff --check` passed.",
          "- Incident fixture: zero model requests, exact assistant text recovered from readable deterministic capsules.",
        ].join("\n"),
      ),
    ]),
  ];
}

function message(
  id: string,
  role: "user" | "assistant",
  parts: Part[],
): MessageWithParts {
  return { info: { id, role } as Message, parts };
}

function textPart(id: string, messageID: string, text: string): TextPart {
  return {
    id,
    sessionID: "incident-source",
    messageID,
    type: "text",
    text,
  };
}

function toolPart(
  id: string,
  messageID: string,
  tool: string,
  state: Record<string, unknown>,
): Part {
  return {
    id,
    sessionID: "incident-source",
    messageID,
    type: "tool",
    callID: `${id}-call`,
    tool,
    state,
  } as Part;
}
