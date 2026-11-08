// The model an OpenCode 2 summary request runs on and what it is sent: the
// configured summarizer, the listed models `--match-model` may choose, and
// the model-visible input estimate of the records a request carries. Shared
// by the retention session summary and every per-turn generation request.
import type { ModelInfo, ModelRef } from "@opencode/client";
import { NATIVE_BATCH_MEDIA_TOKEN_CHARGE } from "../policy/budget";
import { PluginMessageError } from "../policy/feedback";
import { MAGIC_COMPACT_SUMMARIZER_AGENT } from "../policy/protocol";
import {
  verifiedReasoning,
  type MatchPreflight,
  type ModelIdentity,
  type Qualifies,
  type ReasoningRoute,
} from "../policy/reasoning";
import { countGptTokens } from "../stats/tokenize";
import { attemptWithinDeadline, isRecord } from "../util";
import type { V2Host } from "./host";
import { planV2MediaProjection } from "./media";
import { v2ReasoningRoute } from "./reasoning";
import type { V2Message, V2Source } from "./session";

export type V2SummarizerModel = {
  ref: ModelRef;
  info: Pick<ModelInfo, "limit" | "capabilities" | "providerID" | "canonical">
    & Partial<
      Pick<ModelInfo, "variants" | "compatibility" | "modelID" | "package">
    >;
  // Every model the host lists, which `--match-model` chooses from, and
  // whether the summarizer agent carries operator request options written for
  // its own model.
  others?: readonly ModelInfo[];
  optionsConflict?: boolean;
};

export const V2_IMAGE_MODEL_REQUIRED =
  "Magic Compact requires an image-capable configured compaction model for history containing images.";

// The operator-configured summarizer model at the source session's location,
// where a project's own agent settings apply, and whether the agent carries
// operator request options (body or headers) written for that model. The
// source model and the native compaction agent are never a fallback.
async function resolveV2SummarizerAgent(
  host: V2Host,
  location: { directory: string },
  controlTimeoutMs: number,
): Promise<{ model: ModelRef; optionsConflict: boolean }> {
  const outcome = await attemptWithinDeadline(
    signal =>
      host.agent.get(
        { agentID: MAGIC_COMPACT_SUMMARIZER_AGENT, location },
        { signal },
      ),
    controlTimeoutMs,
  );
  const agent = outcome.kind === "success" ? outcome.value.data : undefined;
  if (!agent?.model) {
    throw new PluginMessageError(
      `Magic Compact needs a model on the \`${MAGIC_COMPACT_SUMMARIZER_AGENT}\` agent. Set \`agents.${MAGIC_COMPACT_SUMMARIZER_AGENT}.model\` (for example \`provider/model#variant\`) in opencode.json.`,
    );
  }
  const request: Record<string, unknown> = isRecord(agent.request)
    ? agent.request
    : {};
  const filled = (value: unknown) =>
    isRecord(value) && Object.keys(value).length > 0;
  return {
    model: agent.model,
    optionsConflict: filled(request["body"]) || filled(request["headers"]),
  };
}

// The summarizer model and the descriptor the host reports for it, resolved
// once per command, with the other listed models `--match-model` may choose.
// A model the host does not list fails the command.
export async function resolveV2SummarizerModel(
  host: V2Host,
  source: V2Source,
  controlTimeoutMs: number,
): Promise<V2SummarizerModel> {
  const location = { directory: source.session.location.directory };
  const { model: ref, optionsConflict } = await resolveV2SummarizerAgent(
    host,
    location,
    controlTimeoutMs,
  );
  const outcome = await attemptWithinDeadline(
    signal => host.model.list({ location }, { signal }),
    controlTimeoutMs,
  );
  const models = outcome.kind === "success" ? outcome.value.data : [];
  const info = models.find(
    model => model.providerID === ref.providerID && model.id === ref.id,
  );
  if (!info) {
    throw new PluginMessageError(
      "Magic Compact could not resolve the configured compaction model descriptor.",
    );
  }
  return { ref, info, others: models, optionsConflict };
}

function listedModel(summarizer: V2SummarizerModel, model: ModelIdentity) {
  return summarizer.others?.find(
    item => item.providerID === model.providerID && item.id === model.modelID,
  );
}

// The model a summary request runs on in place of the configured one, with
// the configured variant when the choice keeps it.
export function matchedV2Model(
  summarizer: V2SummarizerModel,
  model: ModelIdentity,
  variant: string | undefined,
): V2SummarizerModel {
  const info = listedModel(summarizer, model);
  if (!info) throw new Error("Matched model is not listed.");
  return {
    ...summarizer,
    ref: {
      providerID: info.providerID,
      id: info.id,
      ...(variant === undefined ? {} : { variant }),
    },
    info,
  };
}

// The route facts of a summary model, configured or matched.
export function v2SummarizerRoute(
  summarizer: V2SummarizerModel,
): ReasoningRoute {
  return v2ReasoningRoute(
    { providerID: summarizer.ref.providerID, modelID: summarizer.ref.id },
    summarizer.info,
  );
}

export function v2Qualifies(summarizer: V2SummarizerModel): Qualifies {
  return model => {
    const info = listedModel(summarizer, model);
    return (
      info !== undefined
      && verifiedReasoning(v2ReasoningRoute(model, info)) !== undefined
    );
  };
}

// What `--match-model` knows about a listed model before choosing it.
export function v2MatchPreflight(
  summarizer: V2SummarizerModel,
  model: ModelIdentity,
  requiresImages: boolean,
): MatchPreflight {
  const info = listedModel(summarizer, model);
  return {
    variants: new Set((info?.variants ?? []).map(variant => variant.id)),
    optionsConflict: summarizer.optionsConflict === true,
    imagesBlocked:
      requiresImages
      && !(info?.capabilities.input ?? []).some(input =>
        input.startsWith("image"),
      ),
  };
}

export function v2Configured(summarizer: V2SummarizerModel): {
  model: ModelIdentity;
  variant?: string;
} {
  return {
    model: {
      providerID: summarizer.ref.providerID,
      modelID: summarizer.ref.id,
    },
    ...(summarizer.ref.variant === undefined
      ? {}
      : { variant: summarizer.ref.variant }),
  };
}

// Model-visible input estimate of OpenCode 2 records: text through the local
// tokenizer, inline image and PDF files at the fixed media charge unless a
// manifest projects the image to its placeholder, and other files by their
// encoded length.
export function estimateV2HistoryTokens(messages: readonly V2Message[]): {
  tokens: number;
  textTokens: number;
  mediaCount: number;
} {
  let tokens = 0;
  let textTokens = 0;
  let mediaCount = 0;
  const text = (value: string | undefined) => {
    if (!value) return;
    const count = countGptTokens(value);
    tokens += count;
    textTokens += count;
  };
  const projection = planV2MediaProjection(messages);
  for (const message of messages) {
    switch (message.type) {
      case "user": {
        text(message.text);
        const covered = projection.get(message.id) ?? [];
        for (const [index, file] of (message.files ?? []).entries()) {
          const placeholder = covered.find(
            image => image.index === index,
          )?.placeholder;
          if (placeholder !== undefined) {
            text(placeholder);
          } else if (
            file.mime.startsWith("image/")
            || file.mime === "application/pdf"
          ) {
            tokens += NATIVE_BATCH_MEDIA_TOKEN_CHARGE;
            mediaCount += 1;
          } else {
            tokens += file.data.length;
          }
        }
        break;
      }
      case "synthetic":
      case "system":
      case "skill":
        text(message.text);
        break;
      case "shell":
        text(message.command);
        text(message.output?.output);
        break;
      case "compaction":
        if (message.status === "completed") {
          text(message.summary);
          text(message.recent);
        }
        break;
      case "assistant":
        for (const item of message.content) {
          if (item.type !== "tool") {
            text(item.text);
            continue;
          }
          const state = item.state;
          text(
            typeof state.input === "string"
              ? state.input
              : JSON.stringify(state.input),
          );
          if (state.status === "completed" || state.status === "error") {
            for (const content of state.content ?? []) {
              if (content.type === "text") text(content.text);
              else tokens += NATIVE_BATCH_MEDIA_TOKEN_CHARGE;
            }
          }
          if (state.status === "error") text(state.error.message);
        }
        break;
      default:
        break;
    }
  }
  return { tokens, textTokens, mediaCount };
}
