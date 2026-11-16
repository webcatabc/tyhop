package org.example.infrastructure;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.output.Response;
import org.example.config.AppConfig;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.regex.Pattern;

/**
 * LangChain4j chat wrapper with one bounded message window per agent/conversation.
 * The combined key keeps agents and conversations isolated from each other.
 */
public final class LlmChatClient {
    /**
     * Qwen3 prompt-level soft switch. The gateway in front of vLLM discards
     * {@code chat_template_kwargs} and every other non-whitelisted request field,
     * so {@code enable_thinking=false} never reaches the model. Appending this
     * marker to the final user turn is the only switch that works end to end.
     */
    public static final String NO_THINK_SWITCH = "/no_think";

    /** Qwen3 style reasoning block that some OpenAI-compatible servers return inline. */
    private static final Pattern REASONING_BLOCK =
            Pattern.compile("<think(?:ing)?>.*?</think(?:ing)?>", Pattern.DOTALL | Pattern.CASE_INSENSITIVE);
    /** An unterminated block: keep nothing after a dangling opening tag. */
    private static final Pattern UNCLOSED_REASONING =
            Pattern.compile("<think(?:ing)?>.*$", Pattern.DOTALL | Pattern.CASE_INSENSITIVE);

    private final OpenAiChatModel model;
    private final int maxMessages;
    private final int maxAttempts;
    private final String defaultSystemPrompt;
    private final boolean disableThinking;
    private final Map<String, MessageWindowChatMemory> memories = new ConcurrentHashMap<>();
    private final Map<String, String> agentSystemPrompts = new ConcurrentHashMap<>();
    private final AtomicInteger activeRequests = new AtomicInteger();
    private final AtomicInteger maxObservedRequests = new AtomicInteger();
    private final AtomicLong totalRequests = new AtomicLong();
    private final AtomicLong inputTokens = new AtomicLong();
    private final AtomicLong outputTokens = new AtomicLong();

    public LlmChatClient() {
        this(AppConfig.load());
    }

    public LlmChatClient(String systemPrompt) {
        this(AppConfig.load(), systemPrompt);
    }

    public LlmChatClient(AppConfig config) {
        this(config, config.llmSystemPrompt);
    }

    public LlmChatClient(AppConfig config, String systemPrompt) {
        if (config.dashscopeApiKey.isBlank()) {
            throw new IllegalArgumentException("DASHSCOPE_API_KEY must be configured");
        }
        var modelBuilder = OpenAiChatModel.builder()
                .baseUrl(config.dashscopeBaseUrl)
                .apiKey(config.dashscopeApiKey)
                .modelName(config.llmModel)
                .timeout(Duration.ofSeconds(config.llmTimeoutSeconds))
                // Disable LangChain's opaque retry loop. We retry only transient
                // errors below, while moderation failures are never repeated.
                .maxRetries(1);
        String temperature = System.getProperty("llm.temperature", "").trim();
        if (!temperature.isEmpty()) {
            modelBuilder.temperature(Double.parseDouble(temperature));
        }
        String topP = System.getProperty("llm.top-p", "").trim();
        if (!topP.isEmpty()) {
            modelBuilder.topP(Double.parseDouble(topP));
        }
        // Ask OpenAI-compatible providers to return a JSON object. This prevents
        // valid short answers from being returned as bare text and then
        // classified as a malformed `retry` result, which is what every in-repo
        // method prompt expects. A free-text protocol can opt out with
        // -Dllm.response.format=text.
        String responseFormat = System.getProperty("llm.response.format", "json_object").trim();
        if (!responseFormat.isEmpty() && !"text".equalsIgnoreCase(responseFormat)
                && !"none".equalsIgnoreCase(responseFormat)) {
            modelBuilder.responseFormat(responseFormat);
        }
        // Optional output cap. Qwen3 occasionally degenerates on extraction prompts
        // and keeps emitting repeated JSON until the model's context limit (observed:
        // 32k tokens / ~300s for one OpenIE prompt), which stalls a whole indexing
        // stage. Every prompt in this repo expects a few dozen tokens, so a cap only
        // ever truncates runaway generations; unset keeps the previous behaviour.
        String maxOutputTokens = System.getProperty("llm.max-output-tokens", "").trim();
        if (!maxOutputTokens.isEmpty()) {
            modelBuilder.maxTokens(Integer.parseInt(maxOutputTokens));
        }
        model = modelBuilder.build();
        maxMessages = Math.max(2, config.memoryMaxMessages);
        maxAttempts = Math.max(1, config.llmMaxRetries);
        defaultSystemPrompt = systemPrompt == null ? "" : systemPrompt.trim();
        disableThinking = config.llmDisableThinking;
    }

    public String chat(String conversationId, String userText) {
        return chatWithSystemPrompt(conversationId, defaultSystemPrompt, userText);
    }

    /** Starts a conversation with a per-conversation system prompt. */
    public String chatWithSystemPrompt(String conversationId, String systemPrompt, String userText) {
        if (conversationId == null || conversationId.isBlank()) {
            throw new IllegalArgumentException("conversationId must not be blank");
        }
        if (userText == null || userText.isBlank()) {
            throw new IllegalArgumentException("userText must not be blank");
        }
        MessageWindowChatMemory memory = memories.computeIfAbsent(conversationId,
                id -> createMemory(id, systemPrompt));
        synchronized (memory) {
            memory.add(UserMessage.from(userText));
            Response<AiMessage> response;
            try {
                response = generateWithRetry(promptMessages(memory.messages()));
            } catch (RuntimeException error) {
                String stage = conversationId.contains("\u0000")
                        ? conversationId.substring(0, conversationId.indexOf('\u0000'))
                        : conversationId;
                System.err.printf(
                        "llm request failed stage=%s chars=%d error=%s%n",
                        stage, userText.length(), error.getClass().getSimpleName());
                throw new IllegalStateException(
                        "LLM request failed at " + stage + ": "
                                + error.getClass().getSimpleName(), error);
            }
            AiMessage answer = response.content();
            String text = answer.text();
            if (text == null) return null;
            // Some OpenAI-compatible servers (vLLM + Qwen3) return the reasoning
            // trace inline as <think>...</think> before the JSON answer, which
            // would make every strict JSON parser see a malformed response.
            String cleaned = stripReasoning(text);
            memory.add(cleaned.equals(text) ? answer : AiMessage.from(cleaned));
            return cleaned;
        }
    }

    /** Removes inline reasoning blocks from a model reply, keeping the answer text. */
    public static String stripReasoning(String text) {
        if (text == null || text.isEmpty()) return text;
        String cleaned = REASONING_BLOCK.matcher(text).replaceAll("");
        cleaned = UNCLOSED_REASONING.matcher(cleaned).replaceAll("");
        return cleaned.strip();
    }

    /** Returns whether the text already ends with the {@value #NO_THINK_SWITCH} soft switch. */
    public static boolean hasNoThinkSwitch(String text) {
        return text != null
                && text.strip().toLowerCase(Locale.ROOT).endsWith(NO_THINK_SWITCH);
    }

    /**
     * Appends the {@value #NO_THINK_SWITCH} soft switch unless it is already there,
     * so repeated calls cannot pile the marker up in a prompt.
     */
    public static String withNoThinkSwitch(String text) {
        if (text == null || text.isBlank() || hasNoThinkSwitch(text)) return text;
        return text + " " + NO_THINK_SWITCH;
    }

    /**
     * Copies the prompt and marks the final user turn as non-thinking. Only the
     * last user message needs the marker: earlier turns stay untouched so the
     * stored conversation history is not polluted with control tokens.
     */
    public static List<ChatMessage> applyNoThinkSwitch(List<ChatMessage> messages) {
        if (messages == null || messages.isEmpty()) return messages;
        List<ChatMessage> result = new ArrayList<>(messages);
        for (int index = result.size() - 1; index >= 0; index--) {
            if (result.get(index) instanceof UserMessage user && user.hasSingleText()) {
                String text = user.singleText();
                String marked = withNoThinkSwitch(text);
                if (!marked.equals(text)) {
                    String name = user.name();
                    result.set(index, name == null || name.isBlank()
                            ? UserMessage.from(marked)
                            : new UserMessage(name, marked));
                }
                break;
            }
        }
        return result;
    }

    private List<ChatMessage> promptMessages(List<ChatMessage> messages) {
        return disableThinking ? applyNoThinkSwitch(messages) : messages;
    }

    /** Registers the system prompt used by an agent for all newly-created conversations. */
    public void registerAgent(String agentId, String systemPrompt) {
        requireText(agentId, "agentId");
        agentSystemPrompts.put(agentId, systemPrompt == null ? "" : systemPrompt.trim());
    }

    /** Sends a message using the registered prompt for an agent and a conversation-specific memory. */
    public String chatAgent(String agentId, String conversationId, String userText) {
        requireText(agentId, "agentId");
        requireText(conversationId, "conversationId");
        String prompt = agentSystemPrompts.getOrDefault(agentId, defaultSystemPrompt);
        return chatWithSystemPrompt(agentId + "\u0000" + conversationId, prompt, userText);
    }

    public void clearAgentConversation(String agentId, String conversationId) {
        requireText(agentId, "agentId");
        requireText(conversationId, "conversationId");
        clearConversation(agentId + "\u0000" + conversationId);
    }

    private MessageWindowChatMemory createMemory(String conversationId, String systemPrompt) {
        MessageWindowChatMemory memory = MessageWindowChatMemory.builder()
                .id(conversationId)
                .maxMessages(maxMessages)
                .build();
        if (systemPrompt != null && !systemPrompt.isBlank()) {
            String prompt = systemPrompt.trim();
            memory.add(SystemMessage.from(disableThinking ? withNoThinkSwitch(prompt) : prompt));
        }
        return memory;
    }

    public void clearConversation(String conversationId) {
        MessageWindowChatMemory memory = memories.remove(conversationId);
        if (memory != null) {
            memory.clear();
        }
    }

    public int messageCount(String conversationId) {
        MessageWindowChatMemory memory = memories.get(conversationId);
        return memory == null ? 0 : memory.messages().size();
    }

    private Response<AiMessage> generateWithRetry(List<ChatMessage> messages) {
        int active = activeRequests.incrementAndGet();
        maxObservedRequests.accumulateAndGet(active, Math::max);
        totalRequests.incrementAndGet();
        try {
            RuntimeException last = null;
            for (int attempt = 1; attempt <= maxAttempts; attempt++) {
                try {
                    Response<AiMessage> response = model.generate(messages);
                    if (response == null || response.content() == null) {
                        throw new IllegalStateException(
                                "LLM returned an empty response content");
                    }
                    if (response.tokenUsage() != null) {
                        if (response.tokenUsage().inputTokenCount() != null) {
                            inputTokens.addAndGet(response.tokenUsage().inputTokenCount());
                        }
                        if (response.tokenUsage().outputTokenCount() != null) {
                            outputTokens.addAndGet(response.tokenUsage().outputTokenCount());
                        }
                    }
                    return response;
                } catch (RuntimeException error) {
                    last = error;
                    if (isModerationFailure(error) || attempt == maxAttempts) throw error;
                    try {
                        Thread.sleep(Math.min(2000L, 300L * attempt));
                    } catch (InterruptedException interrupted) {
                        Thread.currentThread().interrupt();
                        throw new IllegalStateException("LLM retry interrupted", interrupted);
                    }
                }
            }
            throw last == null ? new IllegalStateException("LLM request failed") : last;
        } finally {
            activeRequests.decrementAndGet();
        }
    }

    /** Number of requests currently waiting for or using the remote chat model. */
    public int activeRequests() {
        return activeRequests.get();
    }

    /** Highest concurrent request count observed by this client instance. */
    public int maxObservedRequests() {
        return maxObservedRequests.get();
    }

    /** Total chat requests this client instance has issued, including retries. */
    public long requestCount() {
        return totalRequests.get();
    }

    /** Total prompt tokens reported by the provider for this client instance. */
    public long inputTokenCount() {
        return inputTokens.get();
    }

    /** Total completion tokens reported by the provider for this client instance. */
    public long outputTokenCount() {
        return outputTokens.get();
    }

    /** Total provider-reported tokens for this client instance. */
    public long totalTokenCount() {
        return inputTokenCount() + outputTokenCount();
    }

    /** Resets usage counters without affecting conversation state. */
    public void resetTokenUsage() {
        inputTokens.set(0L);
        outputTokens.set(0L);
    }

    /** Returns whether the provider rejected generated output during content inspection. */
    public static boolean isModerationFailure(Throwable error) {
        Throwable current = error;
        while (current != null) {
            String message = current.getMessage();
            if (message != null) {
                String normalized = message.toLowerCase(java.util.Locale.ROOT);
                if (normalized.contains("data_inspection_failed")
                        || normalized.contains("inappropriate content")
                        || normalized.contains("content moderation")) return true;
            }
            current = current.getCause();
        }
        return false;
    }

    /** Distinguishes rejection of submitted text from generated-output rejection. */
    public static boolean isInputModerationFailure(Throwable error) {
        Throwable current = error;
        while (current != null) {
            String message = current.getMessage();
            if (message != null) {
                String normalized = message.toLowerCase(java.util.Locale.ROOT);
                if (normalized.contains("input text data may contain inappropriate content")
                        || normalized.contains("input data may contain inappropriate content")) {
                    return true;
                }
            }
            current = current.getCause();
        }
        return false;
    }

    private static void requireText(String value, String name) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
    }
}
