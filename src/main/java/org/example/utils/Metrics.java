package org.example.utils;

import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Pattern;

/** HotpotQA/SQuAD-style answer normalization and token F1 calculation. */
public final class Metrics {
    private static final Pattern ARTICLES = Pattern.compile("\\b(a|an|the)\\b");
    private static final Pattern NUMERIC_ORDINAL =
            Pattern.compile("\\b(\\d+)(?:st|nd|rd|th)\\b");
    private static final Map<String, String> ORDINAL_WORDS = Map.ofEntries(
            Map.entry("first", "1"),
            Map.entry("second", "2"),
            Map.entry("third", "3"),
            Map.entry("fourth", "4"),
            Map.entry("fifth", "5"),
            Map.entry("sixth", "6"),
            Map.entry("seventh", "7"),
            Map.entry("eighth", "8"),
            Map.entry("ninth", "9"),
            Map.entry("tenth", "10"),
            Map.entry("eleventh", "11"),
            Map.entry("twelfth", "12"),
            Map.entry("thirteenth", "13"),
            Map.entry("fourteenth", "14"),
            Map.entry("fifteenth", "15"),
            Map.entry("sixteenth", "16"),
            Map.entry("seventeenth", "17"),
            Map.entry("eighteenth", "18"),
            Map.entry("nineteenth", "19"),
            Map.entry("twentieth", "20"),
            Map.entry("thirtieth", "30")
    );
    private static final String ASCII_PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

    private Metrics() { }

    public static Result evaluate(String prediction, String goldAnswer) {
        return evaluate(prediction, Collections.singleton(goldAnswer));
    }

    /** Uses the maximum score when a dataset supplies answer aliases. */
    public static Result evaluate(String prediction, Collection<String> goldAnswers) {
        Objects.requireNonNull(goldAnswers, "goldAnswers");
        if (goldAnswers.isEmpty()) throw new IllegalArgumentException("goldAnswers must not be empty");
        String safePrediction = prediction == null ? "" : prediction;
        Result best = null;
        for (String gold : goldAnswers) {
            Result candidate = evaluateSingle(safePrediction, gold == null ? "" : gold);
            if (best == null || candidate.f1 > best.f1
                    || (candidate.f1 == best.f1 && candidate.em > best.em)) {
                best = candidate;
            }
        }
        return best;
    }

    public static String normalizeAnswer(String value) {
        String text = value == null ? "" : value.toLowerCase(Locale.ROOT);
        StringBuilder withoutPunctuation = new StringBuilder(text.length());
        for (int i = 0; i < text.length(); i++) {
            char ch = text.charAt(i);
            if (ASCII_PUNCTUATION.indexOf(ch) < 0) withoutPunctuation.append(ch);
        }
        String normalized = ARTICLES.matcher(withoutPunctuation)
                .replaceAll(" ")
                .replaceAll("\\s+", " ")
                .trim();
        normalized = NUMERIC_ORDINAL.matcher(normalized).replaceAll("$1");
        for (Map.Entry<String, String> entry : ORDINAL_WORDS.entrySet()) {
            normalized = normalized.replaceAll(
                    "\\b" + Pattern.quote(entry.getKey()) + "\\b",
                    entry.getValue()
            );
        }
        return normalized.replaceAll("\\s+", " ").trim();
    }

    private static Result evaluateSingle(String prediction, String gold) {
        String normalizedPrediction = normalizeAnswer(prediction);
        String normalizedGold = normalizeAnswer(gold);
        double em = normalizedPrediction.equals(normalizedGold) ? 1.0 : 0.0;
        String[] predictionTokens = tokens(normalizedPrediction);
        String[] goldTokens = tokens(normalizedGold);
        if (predictionTokens.length == 0 || goldTokens.length == 0) {
            return new Result(em, 0.0);
        }
        MapCounter predictionCounts = new MapCounter(predictionTokens);
        MapCounter goldCounts = new MapCounter(goldTokens);
        int common = predictionCounts.intersectionSize(goldCounts);
        if (common == 0) return new Result(em, 0.0);
        double precision = (double) common / predictionTokens.length;
        double recall = (double) common / goldTokens.length;
        return new Result(em, 2.0 * precision * recall / (precision + recall));
    }

    private static String[] tokens(String normalized) {
        return normalized.isEmpty() ? new String[0] : normalized.split(" ");
    }

    public static final class Result {
        private final double em;
        private final double f1;

        public Result(double em, double f1) {
            this.em = em;
            this.f1 = f1;
        }

        public double getEm() { return em; }
        public double getF1() { return f1; }
        public double getHit() { return em; }
    }

    private static final class MapCounter {
        private final Map<String, Integer> values = new HashMap<>();

        private MapCounter(String[] tokens) {
            Arrays.stream(tokens).forEach(token -> values.merge(token, 1, Integer::sum));
        }

        private int intersectionSize(MapCounter other) {
            int result = 0;
            for (Map.Entry<String, Integer> entry : values.entrySet()) {
                result += Math.min(entry.getValue(), other.values.getOrDefault(entry.getKey(), 0));
            }
            return result;
        }
    }

    @Override
    public String toString() {
        return super.toString();
    }
}
