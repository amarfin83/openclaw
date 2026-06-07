// Telegram plugin module implements bot message context.body behavior.
import type { Bot } from "grammy";
import {
  buildMentionRegexes,
  formatLocationText,
  implicitMentionKindWhen,
  logInboundDrop,
  matchesMentionWithExplicit,
  resolveInboundMentionDecision,
  type BuildMentionRegexesOptions,
  type NormalizedLocation,
} from "openclaw/plugin-sdk/channel-inbound";
import { resolveChannelGroupPolicy } from "openclaw/plugin-sdk/channel-policy";
import { hasControlCommand } from "openclaw/plugin-sdk/command-detection";
import type { OpenClawConfig } from "openclaw/plugin-sdk/config-contracts";
import type {
  TelegramDirectConfig,
  TelegramGroupConfig,
  TelegramTopicConfig,
} from "openclaw/plugin-sdk/config-contracts";
import {
  createInternalHookEvent,
  fireAndForgetHook,
  toInternalMessageReceivedContext,
  triggerInternalHook,
} from "openclaw/plugin-sdk/hook-runtime";
import { createChannelHistoryWindow, type HistoryEntry } from "openclaw/plugin-sdk/reply-history";
import type { MsgContext } from "openclaw/plugin-sdk/reply-runtime";
import { logVerbose } from "openclaw/plugin-sdk/runtime-env";
import { normalizeOptionalLowercaseString } from "openclaw/plugin-sdk/string-coerce-runtime";
import { withTelegramApiErrorLogging } from "./api-logging.js";
import type { NormalizedAllowFrom } from "./bot-access.js";
import type {
  TelegramLogger,
  TelegramMediaRef,
  TelegramMessageContextOptions,
} from "./bot-message-context.types.js";
import {
  buildSenderLabel,
  buildSenderName,
  extractTelegramLocation,
  getTelegramTextParts,
  hasBotMention,
  renderTelegramTextEntities,
  resolveTelegramPrimaryMedia,
} from "./bot/body-helpers.js";
import {
  buildTelegramGroupPeerId,
  buildTelegramInboundOriginTarget,
  buildTelegramThreadParams,
  buildTypingThreadParams,
  type TelegramThreadSpec,
} from "./bot/helpers.js";
import type { TelegramContext } from "./bot/types.js";
import { isTelegramForumServiceMessage } from "./forum-service-message.js";
import { resolveTelegramCommandIngressAuthorization } from "./ingress.js";

type StickerVisionRuntime = typeof import("./sticker-vision.runtime.js");
type MediaUnderstandingRuntime = typeof import("./media-understanding.runtime.js");

let stickerVisionRuntimePromise: Promise<StickerVisionRuntime> | undefined;
let mediaUnderstandingRuntimePromise: Promise<MediaUnderstandingRuntime> | undefined;

function loadStickerVisionRuntime(): Promise<StickerVisionRuntime> {
  stickerVisionRuntimePromise ??= import("./sticker-vision.runtime.js");
  return stickerVisionRuntimePromise;
}

function loadMediaUnderstandingRuntime(): Promise<MediaUnderstandingRuntime> {
  mediaUnderstandingRuntimePromise ??= import("./media-understanding.runtime.js");
  return mediaUnderstandingRuntimePromise;
}

export type TelegramInboundBodyResult = {
  bodyText: string;
  rawBody: string;
  historyKey?: string;
  commandAuthorized: boolean;
  effectiveWasMentioned: boolean;
  canDetectMention: boolean;
  shouldBypassMention: boolean;
  hasControlCommand: boolean;
  audioTranscribedMediaIndex?: number;
  stickerCacheHit: boolean;
  locationData?: NormalizedLocation;
};

function formatAudioTranscriptForAgent(transcript: string): string {
  return `[Audio transcript (machine-generated, untrusted)]: ${JSON.stringify(transcript)}`;
}

function formatAudioTranscriptEchoForTelegram(transcript: string): string {
  return `Оформленная расшифровка голосового\n\n${transcript.trim()}`;
}

function formatAudioTranscriptFailureEchoForTelegram(): string {
  return "Расшифровка голосового не удалась. Я не получил читаемый текст из аудио.";
}

function formatAudioTranscriptAlreadyPostedForAgent(transcript: string): string {
  return `${formatAudioTranscriptForAgent(transcript)}\n\n[OpenClaw already posted the formatted transcript visibly to this Telegram chat/topic before this agent turn. If the audio contains no substantive request, reply exactly "NO_REPLY". If it does contain a request, answer the request without repeating the transcript.]`;
}

async function sendTelegramAudioTranscriptEcho(params: {
  bot: Bot;
  isGroup: boolean;
  chatId: number | string;
  replyThreadId?: number;
  thread?: TelegramThreadSpec | null;
  transcript?: string;
}): Promise<boolean> {
  if (!params.isGroup) {
    return false;
  }
  const transcript = params.transcript?.trim();
  if (!transcript) {
    return false;
  }
  try {
    await withTelegramApiErrorLogging({
      operation: "sendMessage",
      fn: () =>
        params.bot.api.sendMessage(
          params.chatId,
          formatAudioTranscriptEchoForTelegram(transcript),
          params.thread !== undefined
            ? (buildTelegramThreadParams(params.thread) ?? {})
            : (buildTypingThreadParams(params.replyThreadId) ?? {}),
        ),
    });
    return true;
  } catch (err) {
    logVerbose(`telegram: formatted audio transcript echo failed: ${String(err)}`);
    return false;
  }
}

async function sendTelegramAudioTranscriptFailureEcho(params: {
  bot: Bot;
  isGroup: boolean;
  chatId: number | string;
  replyThreadId?: number;
  thread?: TelegramThreadSpec | null;
}): Promise<boolean> {
  if (!params.isGroup) {
    return false;
  }
  try {
    await withTelegramApiErrorLogging({
      operation: "sendMessage",
      fn: () =>
        params.bot.api.sendMessage(
          params.chatId,
          formatAudioTranscriptFailureEchoForTelegram(),
          params.thread !== undefined
            ? (buildTelegramThreadParams(params.thread) ?? {})
            : (buildTypingThreadParams(params.replyThreadId) ?? {}),
        ),
    });
    return true;
  } catch (err) {
    logVerbose(`telegram: audio transcript failure echo failed: ${String(err)}`);
    return false;
  }
}

type TelegramSavedMediaKind = "audio" | "document" | "image" | "video";

function resolveSavedMediaKind(contentType: string | undefined): TelegramSavedMediaKind {
  const normalized = contentType?.split(";")[0]?.trim().toLowerCase();
  if (normalized?.startsWith("audio/")) {
    return "audio";
  }
  if (normalized?.startsWith("image/")) {
    return "image";
  }
  if (normalized?.startsWith("video/")) {
    return "video";
  }
  return "document";
}

function formatSavedMediaPlaceholder(allMedia: TelegramMediaRef[]): string | undefined {
  if (allMedia.length === 0) {
    return undefined;
  }
  const kinds = allMedia.map((media) => resolveSavedMediaKind(media.contentType));
  const firstKind = kinds[0] ?? "document";
  const kind = kinds.every((candidate) => candidate === firstKind) ? firstKind : "document";
  if (allMedia.length === 1) {
    return `<media:${kind}>`;
  }
  if (kind === "image") {
    return `<media:image> (${allMedia.length} images)`;
  }
  if (kind === "video") {
    return `<media:video> (${allMedia.length} videos)`;
  }
  if (kind === "audio") {
    return `<media:audio> (${allMedia.length} audio attachments)`;
  }
  return `<media:document> (${allMedia.length} attachments)`;
}

async function resolveStickerVisionSupport(params: {
  cfg: OpenClawConfig;
  agentId?: string;
}): Promise<boolean> {
  try {
    const { resolveStickerVisionSupportRuntime } = await loadStickerVisionRuntime();
    return await resolveStickerVisionSupportRuntime(params);
  } catch {
    return false;
  }
}

export async function resolveTelegramInboundBody(params: {
  cfg: OpenClawConfig;
  primaryCtx: TelegramContext;
  bot: Bot;
  msg: TelegramContext["message"];
  allMedia: TelegramMediaRef[];
  isGroup: boolean;
  chatId: number | string;
  accountId?: string;
  senderId: string;
  senderUsername: string;
  sessionKey?: string;
  resolvedThreadId?: number;
  replyThreadId?: number;
  threadSpec?: TelegramThreadSpec;
  originatingTo?: string;
  routeAgentId?: string;
  effectiveGroupAllow: NormalizedAllowFrom;
  effectiveDmAllow: NormalizedAllowFrom;
  groupConfig?: TelegramGroupConfig | TelegramDirectConfig;
  topicConfig?: TelegramTopicConfig;
  providerMentionPatterns?: BuildMentionRegexesOptions["providerPolicy"];
  requireMention?: boolean;
  options?: TelegramMessageContextOptions;
  groupHistories: Map<string, HistoryEntry[]>;
  historyLimit: number;
  logger: TelegramLogger;
}): Promise<TelegramInboundBodyResult | null> {
  const {
    cfg,
    primaryCtx,
    bot,
    msg,
    allMedia,
    isGroup,
    chatId,
    accountId,
    senderId,
    senderUsername,
    sessionKey,
    resolvedThreadId,
    replyThreadId,
    threadSpec,
    originatingTo: providedOriginatingTo,
    routeAgentId,
    effectiveGroupAllow,
    effectiveDmAllow,
    groupConfig,
    topicConfig,
    providerMentionPatterns,
    requireMention,
    options,
    groupHistories,
    historyLimit,
    logger,
  } = params;
  const botUsername = normalizeOptionalLowercaseString(primaryCtx.me?.username);
  const mentionRegexes = buildMentionRegexes(cfg, routeAgentId, {
    provider: "telegram",
    conversationId: isGroup ? buildTelegramGroupPeerId(chatId, resolvedThreadId) : String(chatId),
    providerPolicy: providerMentionPatterns,
  });
  const messageTextParts = getTelegramTextParts(msg);
  const allowForCommands = isGroup ? effectiveGroupAllow : effectiveDmAllow;
  const useAccessGroups = cfg.commands?.useAccessGroups !== false;
  const hasControlCommandInMessage = hasControlCommand(messageTextParts.text, cfg, {
    botUsername,
  });
  const commandGate = await resolveTelegramCommandIngressAuthorization({
    accountId: accountId ?? "default",
    cfg,
    dmPolicy: "pairing",
    isGroup,
    chatId,
    resolvedThreadId,
    senderId,
    effectiveDmAllow,
    effectiveGroupAllow,
    ownerAccess: { ownerList: [], senderIsOwner: false },
    eventKind: "message",
    allowTextCommands: true,
    hasControlCommand: hasControlCommandInMessage,
    modeWhenAccessGroupsOff: "allow",
    includeDmAllowForGroupCommands: false,
  });
  const commandAuthorized = commandGate.authorized;
  const historyKey = isGroup ? buildTelegramGroupPeerId(chatId, resolvedThreadId) : undefined;
  const originatingTo = providedOriginatingTo ?? buildTelegramInboundOriginTarget(chatId);

  const primaryMedia = resolveTelegramPrimaryMedia(msg);
  let placeholder = primaryMedia?.placeholder ?? "";
  const cachedStickerDescription = allMedia[0]?.stickerMetadata?.cachedDescription;
  const stickerSupportsVision = msg.sticker
    ? await resolveStickerVisionSupport({ cfg, agentId: routeAgentId })
    : false;
  const stickerCacheHit = Boolean(cachedStickerDescription) && !stickerSupportsVision;
  if (stickerCacheHit) {
    const emoji = allMedia[0]?.stickerMetadata?.emoji;
    const setName = allMedia[0]?.stickerMetadata?.setName;
    const stickerContext = [emoji, setName ? `from "${setName}"` : null].filter(Boolean).join(" ");
    placeholder = `[Sticker${stickerContext ? ` ${stickerContext}` : ""}] ${cachedStickerDescription}`;
  }

  const locationData = extractTelegramLocation(msg);
  const locationText = locationData ? formatLocationText(locationData) : undefined;
  const rawText = renderTelegramTextEntities(
    messageTextParts.text,
    messageTextParts.entities,
  ).trim();
  const hasUserText = Boolean(rawText || locationText);
  let rawBody = [rawText, locationText].filter(Boolean).join("\n").trim();
  if (!rawBody) {
    rawBody = placeholder;
  }
  if (!rawBody && allMedia.length === 0) {
    return null;
  }

  let bodyText = rawBody;
  if (allMedia.length === 0 && placeholder && rawBody !== placeholder) {
    const mediaTag = primaryMedia?.fileRef.file_id
      ? `${placeholder} [file_id:${primaryMedia.fileRef.file_id}]`
      : placeholder;
    bodyText = `${mediaTag}\n${bodyText}`.trim();
  }
  const hasAudio = allMedia.some((media) => media.contentType?.startsWith("audio/"));
  const disableAudioPreflight =
    (topicConfig?.disableAudioPreflight ??
      (groupConfig as TelegramGroupConfig | undefined)?.disableAudioPreflight) === true;
  const senderAllowedForAudioPreflight =
    !useAccessGroups || !allowForCommands.hasEntries || commandAuthorized;

  let preflightTranscript: string | undefined;
  let preflightTranscriptionFailed = false;
  const needsPreflightTranscription =
    hasAudio &&
    !hasUserText &&
    !disableAudioPreflight &&
    senderAllowedForAudioPreflight &&
    (!isGroup || !requireMention || mentionRegexes.length > 0);

  if (needsPreflightTranscription) {
    try {
      const { transcribeFirstAudio } = await loadMediaUnderstandingRuntime();
      const tempCtx: MsgContext = {
        Provider: "telegram",
        Surface: "telegram",
        OriginatingChannel: "telegram",
        OriginatingTo: originatingTo,
        AccountId: accountId,
        MessageThreadId: replyThreadId,
        MediaPaths: allMedia.length > 0 ? allMedia.map((m) => m.path) : undefined,
        MediaTypes:
          allMedia.length > 0
            ? (allMedia.map((m) => m.contentType).filter(Boolean) as string[])
            : undefined,
      };
      preflightTranscript = await transcribeFirstAudio({
        ctx: tempCtx,
        cfg,
        agentDir: undefined,
      });
      if (!preflightTranscript?.trim()) {
        preflightTranscript = undefined;
        preflightTranscriptionFailed = true;
      }
    } catch (err) {
      preflightTranscriptionFailed = true;
      logVerbose(`telegram: audio preflight transcription failed: ${String(err)}`);
    }
  }
  const audioTranscribedMediaIndex =
    preflightTranscript === undefined
      ? undefined
      : allMedia.findIndex((media) => media.contentType?.startsWith("audio/"));
  const transcriptEchoPosted =
    preflightTranscript === undefined
      ? false
      : await sendTelegramAudioTranscriptEcho({
          bot,
          isGroup,
          chatId,
          replyThreadId,
          thread: threadSpec,
          transcript: preflightTranscript,
        });
  if (preflightTranscriptionFailed) {
    await sendTelegramAudioTranscriptFailureEcho({
      bot,
      isGroup,
      chatId,
      replyThreadId,
      thread: threadSpec,
    });
  }
  const preflightTranscriptForAgent =
    preflightTranscript === undefined
      ? undefined
      : transcriptEchoPosted
        ? formatAudioTranscriptAlreadyPostedForAgent(preflightTranscript)
        : formatAudioTranscriptForAgent(preflightTranscript);

  if (hasAudio && bodyText === "<media:audio>" && preflightTranscriptForAgent) {
    bodyText = preflightTranscriptForAgent;
  }

  const savedMediaPlaceholder = formatSavedMediaPlaceholder(allMedia);
  if (!hasAudio && savedMediaPlaceholder && placeholder && bodyText === placeholder) {
    bodyText = savedMediaPlaceholder;
  }
  if (!bodyText && allMedia.length > 0) {
    if (hasAudio) {
      bodyText = preflightTranscriptForAgent ?? "<media:audio>";
    } else {
      bodyText = savedMediaPlaceholder ?? "<media:document>";
    }
  }

  const hasAnyMention = messageTextParts.entities.some((ent) => ent.type === "mention");
  const explicitlyMentioned = botUsername ? hasBotMention(msg, botUsername) : false;
  const computedWasMentioned = matchesMentionWithExplicit({
    text: messageTextParts.text,
    mentionRegexes,
    explicit: {
      hasAnyMention,
      isExplicitlyMentioned: explicitlyMentioned,
      canResolveExplicit: Boolean(botUsername),
    },
    transcript: preflightTranscript,
  });
  const wasMentioned = options?.forceWasMentioned === true ? true : computedWasMentioned;

  if (isGroup && commandGate.shouldBlockControlCommand) {
    logInboundDrop({
      log: logVerbose,
      channel: "telegram",
      reason: "control command (unauthorized)",
      target: senderId ?? "unknown",
    });
    return null;
  }

  const botId = primaryCtx.me?.id;
  const replyFromId = msg.reply_to_message?.from?.id;
  const replyToBotMessage = botId != null && replyFromId === botId;
  const isReplyToServiceMessage =
    replyToBotMessage && isTelegramForumServiceMessage(msg.reply_to_message);
  const implicitMentionKinds = implicitMentionKindWhen(
    "reply_to_bot",
    replyToBotMessage && !isReplyToServiceMessage,
  );
  const canDetectMention = Boolean(botUsername) || mentionRegexes.length > 0;
  const mentionDecision = resolveInboundMentionDecision({
    facts: {
      canDetectMention,
      wasMentioned,
      hasAnyMention,
      implicitMentionKinds: isGroup ? implicitMentionKinds : [],
    },
    policy: {
      isGroup,
      requireMention: Boolean(requireMention),
      allowTextCommands: true,
      hasControlCommand: hasControlCommandInMessage,
      commandAuthorized,
    },
  });
  const effectiveWasMentioned = mentionDecision.effectiveWasMentioned;
  if (isGroup && requireMention && canDetectMention && mentionDecision.shouldSkip) {
    logger.info({ chatId, reason: "no-mention" }, "skipping group message");
    createChannelHistoryWindow({ historyMap: groupHistories }).record({
      historyKey: historyKey ?? "",
      limit: historyLimit,
      entry: historyKey
        ? {
            sender: buildSenderLabel(msg, senderId || chatId),
            body: rawBody,
            timestamp: msg.date ? msg.date * 1000 : undefined,
            messageId: typeof msg.message_id === "number" ? String(msg.message_id) : undefined,
          }
        : null,
    });
    const telegramGroupPolicy = resolveChannelGroupPolicy({
      cfg,
      channel: "telegram",
      groupId: String(chatId),
      accountId,
    });
    const ingestEnabled =
      topicConfig?.ingest ??
      telegramGroupPolicy.groupConfig?.ingest ??
      telegramGroupPolicy.defaultConfig?.ingest;
    if (ingestEnabled === true && sessionKey) {
      fireAndForgetHook(
        triggerInternalHook(
          createInternalHookEvent(
            "message",
            "received",
            sessionKey,
            toInternalMessageReceivedContext({
              from: `telegram:group:${historyKey ?? chatId}`,
              to: originatingTo,
              content: rawBody,
              timestamp: msg.date ? msg.date * 1000 : undefined,
              channelId: "telegram",
              accountId,
              conversationId: originatingTo,
              messageId: typeof msg.message_id === "number" ? String(msg.message_id) : undefined,
              senderId: senderId || undefined,
              senderName: buildSenderName(msg),
              senderUsername: senderUsername || undefined,
              provider: "telegram",
              surface: "telegram",
              threadId: resolvedThreadId,
              originatingChannel: "telegram",
              originatingTo,
              isGroup: true,
              groupId: `telegram:${chatId}`,
            }),
          ),
        ),
        "telegram: mention-skip message hook failed",
      );
    }
    return null;
  }

  return {
    bodyText,
    rawBody,
    historyKey,
    commandAuthorized,
    effectiveWasMentioned,
    canDetectMention,
    shouldBypassMention: mentionDecision.shouldBypassMention,
    hasControlCommand: hasControlCommandInMessage,
    ...(audioTranscribedMediaIndex !== undefined && audioTranscribedMediaIndex >= 0
      ? { audioTranscribedMediaIndex }
      : {}),
    stickerCacheHit,
    locationData: locationData ?? undefined,
  };
}
