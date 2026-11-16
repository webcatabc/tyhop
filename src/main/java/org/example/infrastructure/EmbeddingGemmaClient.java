package org.example.infrastructure;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.example.config.AppConfig;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
/** Calls the local Ollama embedding endpoint backed by EmbeddingGemma. */
public final class EmbeddingGemmaClient {
    /**
     * The only embedding model this project uses. Hard-coded on purpose: every
     * vector written to Redis or Neo4j must come from the same model, otherwise
     * similarities between stored vectors are meaningless.
     */
    public static final String MODEL = "embeddinggemma";
    /**
     * EmbeddingGemma output width. Fixed here because Neo4j/Redis vector indexes
     * are created with this size and cannot be resized without a rebuild.
     */
    public static final int DIMENSION = 768;

    private final HttpClient httpClient;
    private final URI endpoint;
    private final String model;
    private final Duration timeout;

    public EmbeddingGemmaClient() {
        this(AppConfig.load());
    }

    public EmbeddingGemmaClient(AppConfig config) {
        String baseUrl = config.embeddingBaseUrl.endsWith("/")
                ? config.embeddingBaseUrl.substring(0, config.embeddingBaseUrl.length() - 1)
                : config.embeddingBaseUrl;
        this.endpoint = URI.create(baseUrl + "/api/embed");
        this.model = MODEL;
        this.timeout = Duration.ofSeconds(config.embeddingTimeoutSeconds);
        this.httpClient = HttpClient.newBuilder().connectTimeout(timeout).build();
    }

    public float[] embed(String text) {
        if (text == null || text.isBlank()) {
            throw new IllegalArgumentException("text must not be blank");
        }
        JsonObject requestBody = new JsonObject();
        requestBody.addProperty("model", model);
        requestBody.addProperty("input", text);
        HttpRequest request = HttpRequest.newBuilder(endpoint)
                .timeout(timeout)
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(requestBody.toString()))
                .build();
        try {
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() / 100 != 2) {
                throw new IllegalStateException("EmbeddingGemma returned HTTP " + response.statusCode()
                        + ": " + response.body());
            }
            float[] vector = parseEmbedding(response.body());
            if (vector.length != DIMENSION) {
                throw new IllegalStateException(MODEL + " returned " + vector.length
                        + " dimensions, expected " + DIMENSION);
            }
            return vector;
        } catch (IOException e) {
            throw new IllegalStateException("Unable to call EmbeddingGemma at " + endpoint, e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("EmbeddingGemma request interrupted", e);
        }
    }

    private static float[] parseEmbedding(String body) {
        JsonObject root = JsonParser.parseString(body).getAsJsonObject();
        JsonArray values;
        if (root.has("embeddings")) {
            JsonElement embeddings = root.get("embeddings");
            values = embeddings.getAsJsonArray().get(0).getAsJsonArray();
        } else if (root.has("embedding")) {
            values = root.getAsJsonArray("embedding");
        } else {
            throw new IllegalStateException("EmbeddingGemma response has no embedding field: " + body);
        }
        float[] result = new float[values.size()];
        for (int i = 0; i < values.size(); i++) {
            result[i] = values.get(i).getAsFloat();
            if (!Float.isFinite(result[i])) {
                throw new IllegalStateException("EmbeddingGemma returned a non-finite value at index " + i);
            }
        }
        return result;
    }
}
