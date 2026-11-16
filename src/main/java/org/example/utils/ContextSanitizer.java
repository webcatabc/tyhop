package org.example.utils;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Turns a benchmark context into the text a model should see: the supervision
 * metadata is dropped and, for the graph-construction prompt, the JSON envelope
 * is dropped too so the model reads plain prose.
 */
public final class ContextSanitizer {
    private static final Set<String> GOLD_FIELDS = Set.of(
            "is_supporting", "supporting_facts", "supporting_sentences", "answer_spans",
            "gold_answer", "gold_answers", "answer_aliases", "question_decomposition");

    private ContextSanitizer() { }

    /**
     * Converts one {@code data_set.context} value into plain prose paragraphs:
     * musique {@code paragraph_text} fields, 2Wiki {@code [title, [sentences]]}
     * entries, and HotpotQA sentence lists are all flattened to text. Repeated
     * paragraphs are kept in order; an unrecognised or non-JSON value is
     * returned unchanged, so a caller always gets usable prompt text.
     */
    public static String toPromptText(String context) {
        if (context == null || context.isBlank()) return context;
        JsonElement root;
        try {
            root = JsonParser.parseString(context);
        } catch (RuntimeException plainText) {
            return context.trim();
        }
        if (!root.isJsonArray()) return context.trim();
        List<String> paragraphs = new ArrayList<>();
        for (JsonElement entry : root.getAsJsonArray()) appendParagraph(paragraphs, entry);
        if (paragraphs.isEmpty()) return context.trim();
        return String.join("\n\n", paragraphs);
    }

    private static void appendParagraph(List<String> paragraphs, JsonElement entry) {
        if (entry == null || entry.isJsonNull()) return;
        if (entry.isJsonObject()) {
            JsonObject object = entry.getAsJsonObject();
            String text = stringValue(object, "paragraph_text");
            if (text.isBlank()) text = stringValue(object, "text");
            if (!text.isBlank()) paragraphs.add(text);
            return;
        }
        if (!entry.isJsonArray()) {
            String text = scalar(entry);
            if (!text.isBlank()) paragraphs.add(text);
            return;
        }
        var values = entry.getAsJsonArray();
        // [title, [sentences]] carries a title first; an array of plain strings is
        // already the sentence list of one HotpotQA paragraph.
        String title = "";
        JsonElement body = entry;
        if (values.size() >= 2 && values.get(0).isJsonPrimitive() && values.get(1).isJsonArray()) {
            title = scalar(values.get(0));
            body = values.get(1);
        }
        String text = joinText(body);
        if (text.isBlank()) text = joinText(entry);
        if (text.isBlank()) return;
        // The benchmark paragraphs already name their subject, so the title is
        // only prepended when the body does not mention it.
        paragraphs.add(title.isBlank() || text.toLowerCase(java.util.Locale.ROOT)
                .contains(title.toLowerCase(java.util.Locale.ROOT)) ? text : title + ": " + text);
    }

    private static String joinText(JsonElement value) {
        if (value == null || value.isJsonNull()) return "";
        if (value.isJsonArray()) {
            StringBuilder joined = new StringBuilder();
            for (JsonElement child : value.getAsJsonArray()) {
                String text = scalar(child);
                if (text.isBlank()) continue;
                if (joined.length() > 0) joined.append(' ');
                joined.append(text);
            }
            return joined.toString().trim();
        }
        return scalar(value);
    }

    private static String stringValue(JsonObject object, String key) {
        return scalar(object.get(key));
    }

    private static String scalar(JsonElement value) {
        return value == null || value.isJsonNull() || !value.isJsonPrimitive()
                ? "" : value.getAsString().trim();
    }

    /**
     * Drops supervision fields but keeps the JSON envelope.  Model prompts must
     * use {@link #toPromptText(String)} instead: the benchmark's paragraph
     * metadata (indices, field names, supporting flags) is noise for the model.
     */
    public static String forModel(String context) {
        if (context == null || context.isBlank()) return context;
        try {
            JsonElement root = JsonParser.parseString(context);
            return removeSupervision(root) ? root.toString() : context;
        } catch (RuntimeException plainText) {
            return context;
        }
    }

    private static boolean removeSupervision(JsonElement value) {
        if (value == null || value.isJsonNull()) return false;
        boolean changed = false;
        if (value.isJsonObject()) {
            JsonObject object = value.getAsJsonObject();
            for (String name : List.copyOf(object.keySet())) {
                if (GOLD_FIELDS.contains(name.toLowerCase(java.util.Locale.ROOT))) {
                    object.remove(name);
                    changed = true;
                } else {
                    changed |= removeSupervision(object.get(name));
                }
            }
        } else if (value.isJsonArray()) {
            for (JsonElement child : value.getAsJsonArray()) changed |= removeSupervision(child);
        }
        return changed;
    }
}
