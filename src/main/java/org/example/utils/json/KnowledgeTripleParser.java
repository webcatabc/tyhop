package org.example.utils.json;

import org.example.entity.KnowledgeTriple;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.JsonSyntaxException;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

/** Converts the strict JSON array returned by the LLM into Java triples. */
public final class KnowledgeTripleParser {
    private static final Logger LOG = Logger.getLogger(KnowledgeTripleParser.class.getName());

    private KnowledgeTripleParser() {
    }

    public static List<KnowledgeTriple> parse(String json) {
        if (json == null || json.isBlank()) {
            throw new IllegalArgumentException("triple JSON must not be blank");
        }
        JsonElement root = parseRoot(json);
        if (root.isJsonObject()) {
            root = unwrapTripleDocument(root.getAsJsonObject());
        }
        if (!root.isJsonArray()) {
            throw new IllegalArgumentException("triple JSON must be an array or an object containing 'triples'");
        }
        List<KnowledgeTriple> triples = JsonCodec.fromJsonList(root.toString(), KnowledgeTriple.class);
        if (triples == null) throw new IllegalArgumentException("triple JSON must be an array");
        List<KnowledgeTriple> valid = new ArrayList<>();
        for (KnowledgeTriple triple : triples) {
            if (triple == null) {
                LOG.warning("Skipping null triple in JSON array");
                continue;
            }
            if (isBlank(triple.getSubject()) || isBlank(triple.getSubjectType())
                    || isBlank(triple.getRelation()) || isBlank(triple.getTarget())
                    || isBlank(triple.getTargetType())) {
                LOG.warning("Skipping incomplete triple: subject=" + triple.getSubject()
                        + ", relation=" + triple.getRelation() + ", target=" + triple.getTarget());
                continue;
            }
            valid.add(triple);
        }
        return valid;
    }

    /** Finds a triple array even when a provider adds one or more JSON wrappers. */
    private static JsonElement unwrapTripleDocument(JsonObject object) {
        String[] arrayKeys = {"triples", "data", "knowledge_graph", "relations", "items", "results", "output"};
        for (String key : arrayKeys) {
            JsonElement value = object.get(key);
            if (value == null || value.isJsonNull()) continue;
            if (value.isJsonArray()) return value;
            if (value.isJsonObject()) {
                JsonElement nested = unwrapTripleDocument(value.getAsJsonObject());
                if (nested.isJsonArray()) return nested;
            }
            if (value.isJsonPrimitive() && value.getAsJsonPrimitive().isString()) {
                try {
                    JsonElement nested = parseRoot(value.getAsString());
                    if (nested.isJsonArray()) return nested;
                    if (nested.isJsonObject()) {
                        JsonElement unwrapped = unwrapTripleDocument(nested.getAsJsonObject());
                        if (unwrapped.isJsonArray()) return unwrapped;
                    }
                } catch (RuntimeException ignored) { }
            }
        }
        // Some OpenAI-compatible gateways add arbitrary envelope keys (for
        // example `response` or `payload`). Inspect nested values as a final
        // bounded fallback when the key is not one of the conventional names.
        for (String key : object.keySet()) {
            if (java.util.Arrays.asList(arrayKeys).contains(key)) continue;
            JsonElement value = object.get(key);
            if (value != null && value.isJsonObject()) {
                JsonElement nested = unwrapTripleDocument(value.getAsJsonObject());
                if (nested.isJsonArray()) return nested;
            }
        }
        if (object.has("subject") && object.has("relation") && object.has("target")) {
            com.google.gson.JsonArray single = new com.google.gson.JsonArray();
            single.add(object);
            return single;
        }
        return object;
    }

    /**
     * Normalizes common LLM wrappers before deserializing the array. A model
     * may return a JSON string whose value is the array, Markdown code fences,
     * or a short explanation around the JSON.
     */
    private static JsonElement parseRoot(String raw) {
        String candidate = stripCodeFence(raw.trim());
        for (int attempt = 0; attempt < 2; attempt++) {
            try {
                JsonElement root = JsonParser.parseString(candidate);
                if (root.isJsonPrimitive() && root.getAsJsonPrimitive().isString()) {
                    candidate = stripCodeFence(root.getAsString().trim());
                    continue;
                }
                return root;
            } catch (JsonSyntaxException firstFailure) {
                String array = extractArray(candidate);
                if (array == null) {
                    throw new IllegalArgumentException("Unable to parse triple JSON returned by the model", firstFailure);
                }
                candidate = array;
            }
        }
        throw new IllegalArgumentException("Triple JSON is nested too deeply");
    }

    private static String stripCodeFence(String value) {
        if (value.startsWith("```") && value.endsWith("```")) {
            int firstNewline = value.indexOf('\n');
            if (firstNewline >= 0) value = value.substring(firstNewline + 1, value.length() - 3);
            else value = value.substring(3, value.length() - 3);
        }
        return value.trim();
    }

    /** Extracts the first balanced JSON array from text surrounding the answer. */
    private static String extractArray(String value) {
        int start = -1;
        int depth = 0;
        boolean quoted = false;
        boolean escaped = false;
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            if (quoted) {
                if (escaped) escaped = false;
                else if (c == '\\') escaped = true;
                else if (c == '"') quoted = false;
                continue;
            }
            if (c == '"') {
                quoted = true;
            } else if (c == '[') {
                if (depth == 0) start = i;
                depth++;
            } else if (c == ']' && depth > 0 && --depth == 0) {
                return value.substring(start, i + 1);
            }
        }
        return null;
    }

    private static boolean isBlank(String value) {
        return value == null || value.isBlank();
    }
}
