package org.example.experiment.GraphBuild;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.example.config.AppConfig;
import org.example.entity.DataSet;
import org.example.entity.ExperimentData;
import org.example.entity.KnowledgeTriple;
import org.example.entity.TypeMappingResult;
import org.example.infrastructure.LlmChatClient;
import org.example.infrastructure.Neo4jClient;
import org.example.mp.KgPaperRepository;
import org.example.utils.ContextSanitizer;
import org.example.utils.EntityRelationNormalizer;
import org.example.utils.json.JsonCodec;
import org.example.utils.json.KnowledgeTripleParser;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.UUID;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.Semaphore;

import static org.example.constants.LlmPrompt.KgBuild;
import static org.example.constants.LlmPrompt.KgEntityMerge;

/*
 * Graph-construction overview: read the following steps alongside buildSample.
 *
 * Goal: transform one source sample (all passage text for a question) into a Neo4j
 * knowledge graph with entity types and provenance, so OurMethodReasoning can perform
 * type-aware and provenance-aware reasoning.
 *
 * Step 0: ContextSanitizer.toPromptText(context)
 *         Remove the benchmark JSON wrapper and supporting-fact labels, leaving plain
 *         text without answer leakage.
 * Step 1: splitContextForDataset(context, graphDatasetId)
 *         Split context into passage chunks according to the dataset format
 *         (HotpotQA / 2Wiki / MuSiQue); one passage becomes one chunk.
 * Step 2: extractInitialGraphChunked(...)
 *         Extract triples from chunks in parallel (bounded by CHUNK_LLM_PERMITS)
 *         and merge duplicates by triple content.
 * Step 3: mergeHighSimilarityEntities(...)  <- construction-time entity resolution
 *         Cluster entity mentions by cosine similarity (>= 0.95), merge them with
 *         existing graph entities, and ask one LLM call to choose a canonical name
 *         and type for each group of candidates.
 * Step 4: extractedTypes(...)
 *         Collect extracted subject_type / target_type values directly as type nodes.
 * Step 5: neo4j.storeEntityTypes(...)
 *         Persist entity types, type vectors, and type relations; this is the source
 *         of the type graph used during reasoning.
 * Step 6: neo4j.replaceSampleTriples(...)
 *         Persist triples with source data_set IDs on edges. Delete the sample's old
 *         edges first so rebuilding after normalization changes leaves no stale data.
 * Step 7: Return the final triples and direct type decisions.
 *
 * Two robustness mechanisms:
 *   1. Moderation fallback: switch to a neutral structured-extraction prompt when a
 *      prompt is rejected. If rejection persists, split the source into smaller chunks
 *      and extract each one separately.
 *   2. Stage-aware exceptions: BuildStageException records whether failure occurred
 *      during extraction, quality checking, repair, type mapping, or storage.
 */

/** Graph-construction pipeline: triple extraction -> entity resolution -> type graph -> Neo4j storage. */
public final class GraphBuild {
    /** Maximum characters per chunk during moderation fallback. */
    private static final int MODERATION_CHUNK_CHARS = 3500;
    /** Strict requirement that all five triple fields be non-empty; omit a triple rather than emit null. */
    private static final String COMPLETE_TRIPLE_REQUIREMENT =
            " Every triple must have non-empty string values for subject, subject_type, relation, target, " +
            "and target_type. Never output null for these fields; omit an uncertain incomplete triple.";
    /** Global concurrency gate for in-flight chunk-extraction requests (see chunkLlmPermits). */
    private static final Semaphore CHUNK_LLM_PERMITS = new Semaphore(chunkLlmPermits());

    /**
     * Global concurrency limit for LLM extraction calls (default 20).
     *
     * <p>This limits requests in flight across the process, independently of the
     * number of samples processed concurrently, so increasing sample threads does not
     * overwhelm the model service. Override it with
     * {@code -Dgraph.chunk.llm.permits=N}.</p>
     */
    private static int chunkLlmPermits() {
        String value = System.getProperty("graph.chunk.llm.permits");
        if (value == null || value.isBlank()) return 20;
        try {
            int parsed = Integer.parseInt(value.trim());
            return parsed > 0 ? parsed : 20;
        } catch (NumberFormatException invalid) {
            return 20;
        }
    }

    /**
     * Parameters for passage-level graph construction.
     *
     * @param maxChunkConcurrency        maximum concurrent chunks per sample (default 8)
     * @param maxGlobalChunkConcurrency  maximum concurrent chunks globally (default 20)
     */
    public record ChunkConfig(int maxChunkConcurrency, int maxGlobalChunkConcurrency) {
        public ChunkConfig {
            if (maxChunkConcurrency < 1 || maxGlobalChunkConcurrency < 1) {
                throw new IllegalArgumentException("chunk limits must be positive");
            }
        }

        public static ChunkConfig defaults() {
            return new ChunkConfig(8, 20);
        }
    }
    /** Graph-construction stages reported with BuildStageException to localize failures. */
    public enum Stage {
        INITIAL_EXTRACTION("initial_extraction"),
        TRIPLE_REPAIR("triple_repair"),
        TYPE_MAPPING("type_mapping"),
        GRAPH_STORAGE("graph_storage");

        /** String form used in storage and logs, for example "initial_extraction". */
        private final String value;

        Stage(String value) { this.value = value; }

        /** Return the stage identifier. */
        public String value() { return value; }
    }

    /**
     * Stage-aware graph-construction exception identifying the failed stage and
     * whether moderation caused the failure while retaining the original cause for
     * model-service diagnosis.
     */
    public static final class BuildStageException extends RuntimeException {
        private final Stage stage;
        private final boolean moderationFailure;

        public BuildStageException(Stage stage, boolean moderationFailure, Throwable cause) {
            super("stage=" + stage.value() + ", moderation=" + moderationFailure + ": "
                + causeMessage(cause), cause);
            this.stage = Objects.requireNonNull(stage, "stage");
            this.moderationFailure = moderationFailure;
        }

        /** Stage in which the failure occurred. */
        public Stage stage() { return stage; }
        /** Whether moderation rejected the request; true usually requires a prompt change or chunk retry. */
        public boolean moderationFailure() { return moderationFailure; }
    }

    /** Neutral extraction prompt that treats the passage as structured data without judging or elaborating. */
    private static final String NEUTRAL_EXTRACTION_PROMPT =
            "Process the supplied benchmark passage as neutral structured data. Extract only factual " +
            "knowledge-graph triples; do not continue, embellish, endorse, or comment on any passage content. " +
            "Keep entity names faithful to the source, use lowercase English relation and type labels, and return " +
            "only a JSON object with a triples array. Each triple must contain subject, subject_type, relation, " +
            "target, target_type, and relation_context." + COMPLETE_TRIPLE_REQUIREMENT;
    /** Neutral quality-check prompt for structural consistency without repeating or extending sensitive narrative. */
    private static final String NEUTRAL_QUALITY_PROMPT =
            "Perform a neutral structural consistency check of the supplied benchmark passage and extracted " +
            "triples. Do not repeat or elaborate on sensitive narrative. Return only a JSON object containing " +
            "additions and modifications arrays in the same triple schema, using empty arrays when no change is needed.";

    private GraphBuild() { }

    /**
     * Auditable artifacts from graph construction.
     *
     * @param repairedTriples final triples after entity resolution
     * @param typeMapping     direct type decisions used by reasoning
     */
    public record BuildResult(List<KnowledgeTriple> repairedTriples,
                              TypeMappingResult typeMapping) {
        public BuildResult {
            repairedTriples = repairedTriples == null ? List.of() : List.copyOf(repairedTriples);
            Objects.requireNonNull(typeMapping, "typeMapping");
        }

        public List<KnowledgeTriple> storedTriples() { return typeMapping.triples(); }
    }

    private record ChunkedExtraction(List<KnowledgeTriple> triples) {
        private ChunkedExtraction {
            triples = triples == null ? List.of() : List.copyOf(triples);
        }
    }

    /**
     * Build the graph for one sample using the default ChunkConfig.
     *
     * <p>Namespace design: all samples in a benchmark dataset share one graph
     * (graphDatasetId is the dataset name), while sourceDatasetId is appended to each
     * edge's dataset_id_list as provenance. Reasoning can therefore find cross-sample
     * paths while identifying which question's source text supports each edge.</p>
     */
    /**
     * Complete implementation for building one sample graph (main construction flow).
     * Each step corresponds to steps 0-7 in the overview at the top of this file.
     */
    /** Convenience overload using the default type-mapping configuration. */
    public static BuildResult buildSample(LlmChatClient llm, Neo4jClient neo4j,
                                          String experimentId, String graphDatasetId,
                                          long sourceDatasetId, String context) {
        return buildSample(llm, neo4j, experimentId, graphDatasetId,
                sourceDatasetId, context, ChunkConfig.defaults());
    }

    public static BuildResult buildSample(LlmChatClient llm, Neo4jClient neo4j,
                                          String experimentId, String graphDatasetId,
                                          long sourceDatasetId, String context,
                                          ChunkConfig chunkConfig) {
        Objects.requireNonNull(llm, "llm");
        Objects.requireNonNull(neo4j, "neo4j");
        requireId(experimentId, "experimentId");
        requireId(graphDatasetId, "graphDatasetId");
        if (sourceDatasetId <= 0) {
            throw new IllegalArgumentException("sourceDatasetId must be positive");
        }
        requireText(context, "context");
        ChunkConfig chunks = chunkConfig == null
                ? ChunkConfig.defaults() : chunkConfig;

        ChunkedExtraction extraction = extractInitialGraphChunked(
                llm, context, graphDatasetId, chunks);
        List<KnowledgeTriple> extracted;
        try {
            extracted = extraction.triples();
        } catch (RuntimeException error) {
            throw stageFailure(Stage.TRIPLE_REPAIR, error);
        }
        if (extracted.isEmpty()) {
            throw stageFailure(Stage.TRIPLE_REPAIR,
                    new IllegalStateException("chunked extraction produced no complete triples"));
        }

        List<KnowledgeTriple> mergedTriples;
        try {
            mergedTriples = mergeHighSimilarityEntities(llm, neo4j,
                    experimentId, graphDatasetId, extracted);
        } catch (RuntimeException error) {
            System.err.println("Entity merge skipped: " + error.getMessage());
            mergedTriples = extracted;
        }
        mergedTriples = augmentExplicitRelations(context, mergedTriples);

        List<String> types = extractedTypes(mergedTriples);
        TypeMappingResult mapped =
                new TypeMappingResult(mergedTriples, types);
        try {
            neo4j.storeEntityTypes(experimentId, graphDatasetId,
                    sourceDatasetId, types);
        } catch (RuntimeException error) {
            throw stageFailure(Stage.TYPE_MAPPING, error);
        }

        try {
            neo4j.replaceSampleTriples(experimentId, graphDatasetId,
                    sourceDatasetId, mapped.triples());
        } catch (RuntimeException error) {
            throw stageFailure(Stage.GRAPH_STORAGE, error);
        }
        return new BuildResult(mergedTriples, mapped);
    }

    private static final Pattern NAMED_AFTER = Pattern.compile(
            "(?i)(.+?)\\s+(?:was|were|is|are)\\s+named after\\s+(.+?)(?:[.;]|$)");
    private static final Pattern MARRIED_TO = Pattern.compile(
            "(?i)(.+?)\\s+(?:was|were|is|are)\\s+married to\\s+(.+?)(?:[.;]|$)");
    private static final Pattern FATHER_OF = Pattern.compile(
            "(?i)(.+?)\\s+(?:was|were|is|are)\\s+the father of\\s+(.+?)(?:[.;]|$)");
    private static final Pattern CHILD_OF = Pattern.compile(
            "(?i)(.+?)\\s+(?:was|were|is|are)\\s+the (?:son|daughter) of\\s+(.+?)(?:[.;]|$)");

    /** Add high-precision explicit relations that extraction commonly misses. */
    static List<KnowledgeTriple> augmentExplicitRelations(
            String context, List<KnowledgeTriple> triples) {
        if (context == null || context.isBlank() || triples == null || triples.isEmpty()) {
            return triples == null ? List.of() : List.copyOf(triples);
        }
        List<KnowledgeTriple> output = new ArrayList<>(triples);
        Set<String> signatures = new HashSet<>();
        Map<String, KnowledgeTriple> entities = new LinkedHashMap<>();
        for (KnowledgeTriple triple : triples) {
            signatures.add(tripleSignature(triple));
            entities.putIfAbsent(entityKey(triple.getSubject(), triple.getSubjectType()), triple);
            entities.putIfAbsent(entityKey(triple.getTarget(), triple.getTargetType()), triple);
        }
        for (String sentence : splitSentences(context)) {
            addPatternRelation(output, signatures, entities, sentence,
                    NAMED_AFTER, "named_after", false);
            addPatternRelation(output, signatures, entities, sentence,
                    MARRIED_TO, "spouse", true);
            addPatternRelation(output, signatures, entities, sentence,
                    FATHER_OF, "father_of", false);
            addPatternRelation(output, signatures, entities, sentence,
                    CHILD_OF, "child_of", false);
        }
        return List.copyOf(output);
    }

    private static void addPatternRelation(
            List<KnowledgeTriple> output,
            Set<String> signatures,
            Map<String, KnowledgeTriple> entities,
            String sentence,
            Pattern pattern,
            String relation,
            boolean personTarget) {
        Matcher matcher = pattern.matcher(sentence);
        if (!matcher.find()) return;
        String left = cleanPhrase(matcher.group(1));
        String right = cleanPhrase(matcher.group(2));
        if (left.isBlank() || right.isBlank()) return;
        KnowledgeTriple source = findEntity(entities, left);
        if (source == null) return;
        String subject = source.getSubject();
        String subjectType = source.getSubjectType();
        if (normalizePhrase(subject).equals(normalizePhrase(right))) return;
        String targetType = personTarget ? "person" : subjectType;
        KnowledgeTriple added = new KnowledgeTriple(
                subject, subjectType, relation, right, targetType, sentence);
        String signature = tripleSignature(added);
        if (signatures.add(signature)) output.add(added);
    }

    private static KnowledgeTriple findEntity(
            Map<String, KnowledgeTriple> entities, String text) {
        String normalized = normalizePhrase(text);
        KnowledgeTriple best = null;
        int bestLength = -1;
        for (KnowledgeTriple triple : entities.values()) {
            String subject = normalizePhrase(triple.getSubject());
            String target = normalizePhrase(triple.getTarget());
            if (subject.isBlank() || target.isBlank()) continue;
            if (normalized.contains(subject) && subject.length() > bestLength) {
                best = triple;
                bestLength = subject.length();
            }
            if (normalized.contains(target) && target.length() > bestLength) {
                best = new KnowledgeTriple(
                        triple.getTarget(), triple.getTargetType(),
                        triple.getRelation(), triple.getTarget(),
                        triple.getTargetType(), triple.getRelationContext());
                bestLength = target.length();
            }
        }
        return best;
    }

    private static List<String> splitSentences(String context) {
        List<String> result = new ArrayList<>();
        for (String sentence : context.split("(?<=[.!?])\\s+|\\R+")) {
            String value = sentence.trim();
            if (!value.isBlank()) result.add(value);
        }
        return result;
    }

    private static String cleanPhrase(String value) {
        return value == null ? "" : value
                .replaceAll("^[\"'“”‘’\\s]+|[\"'“”‘’\\s]+$", "")
                .replaceAll("\\s+", " ")
                .trim();
    }

    private static String normalizePhrase(String value) {
        return value == null ? "" : value.toLowerCase(Locale.ROOT)
                .replaceAll("[^a-z0-9]+", " ")
                .replaceAll("\\s+", " ")
                .trim();
    }

    private static String tripleSignature(KnowledgeTriple triple) {
        return normalizePhrase(triple.getSubject()) + "\u0000"
                + normalizePhrase(triple.getRelation()) + "\u0000"
                + normalizePhrase(triple.getTarget());
    }

    /**
     * [Steps 1-2] Passage-level extraction.
     *
     * <p>Extract directly when there is one chunk. With multiple chunks, use a fixed
     * thread pool and merge results by triple-content fingerprint so the same fact
     * across passages is retained once.</p>
     *
     * <p>Concurrency is the minimum of chunk count, per-sample limit, and global limit.
     * The global CHUNK_LLM_PERMITS semaphore in extractChunk ultimately bounds
     * in-flight requests.</p>
     */
    private static ChunkedExtraction extractInitialGraphChunked(
            LlmChatClient llm, String context, String graphDatasetId, ChunkConfig config) {
        // Split by dataset-aware passage boundaries without cutting inside passages.
        List<String> chunks = splitContextForDataset(context, graphDatasetId);
        if (chunks.isEmpty()) chunks = List.of(context);
        if (chunks.size() == 1) {
            String initial = extractInitialGraph(llm, chunks.get(0));
            return new ChunkedExtraction(KnowledgeTripleParser.parse(initial));
        }
        final List<String> workChunks = chunks;
        // Compute actual concurrency as the minimum of the three limits.
        int concurrency = Math.max(1, Math.min(
                Math.min(workChunks.size(), config.maxChunkConcurrency()),
                config.maxGlobalChunkConcurrency()));
        ExecutorService pool = Executors.newFixedThreadPool(concurrency);
        List<Future<List<KnowledgeTriple>>> futures = new ArrayList<>();
        for (int index = 0; index < workChunks.size(); index++) {
            final int chunkIndex = index;
            final String chunk = workChunks.get(index);
            futures.add(pool.submit(() -> extractChunk(llm, chunk, chunkIndex, workChunks.size())));
        }
        pool.shutdown();
        // Merge chunk results by triple fingerprint, using putIfAbsent for deduplication.
        LinkedHashMap<String, KnowledgeTriple> merged = new LinkedHashMap<>();
        for (Future<List<KnowledgeTriple>> future : futures) {
            try {
                for (KnowledgeTriple triple : future.get()) {
                    merged.putIfAbsent(tripleKey(triple), triple);
                }
            } catch (Exception error) {
                Throwable cause = error.getCause() == null ? error : error.getCause();
                if (Boolean.parseBoolean(System.getProperty(
                        "graph.extract.tolerate.partial.chunks", "false"))) {
                    System.err.printf("Graph extraction skipped one failed chunk: %s%n",
                            cause.getMessage());
                    continue;
                }
                if (cause instanceof RuntimeException runtime) throw runtime;
                throw stageFailure(Stage.INITIAL_EXTRACTION, new IllegalStateException(cause));
            }
        }
        List<KnowledgeTriple> triples = new ArrayList<>(merged.values());
        if (triples.isEmpty()) {
            // No complete triples across chunks means graph construction failed for the sample.
            throw stageFailure(Stage.INITIAL_EXTRACTION,
                    new IllegalStateException("chunked extraction produced no complete triples"));
        }
        // Log chunk and triple counts to evaluate the chunking strategy.
        System.err.printf("Chunked graph extraction used %d chunks and produced %d triples.%n",
                chunks.size(), triples.size());
        return new ChunkedExtraction(triples);
    }

    /** Entity mention with name, type, and evidence sentences used to decide whether mentions refer to one entity. */
    private record EntityNode(String name, String type, List<String> contexts) { }

    /**
     * [Step 3] Simplified construction-time entity resolution.
     *
     * <p>Three phases:</p>
     * <ol>
     *   <li><b>Clustering:</b> embed all mentions from this sample and union those with
     *       cosine similarity &gt;= 0.95. Also merge existing graph entities found by
     *       {@code searchEntitiesWithinDistance(..., 0.05)} so new mentions can reuse
     *       entities across samples.</li>
     *   <li><b>Group decision:</b> send up to 20 groups with at least two members and a
     *       new mention to the LLM, including names, types, and evidence, to obtain
     *       canonical_name / canonical_type.</li>
     *   <li><b>Rewrite:</b> replace triple subjects and targets with the returned
     *       canonical names and types.</li>
     * </ol>
     *
     * <p>Any failure falls back to the original triples without blocking construction.</p>
     */
    private static List<KnowledgeTriple> mergeHighSimilarityEntities(
            LlmChatClient llm, Neo4jClient neo4j, String experimentId,
            String graphDatasetId, List<KnowledgeTriple> triples) {
        LinkedHashMap<String, EntityNode> nodes = new LinkedHashMap<>();
        // Collect subject and target mentions, keyed by normalized name and type.
        for (KnowledgeTriple triple : triples) {
            addMention(nodes, triple.getSubject(), triple.getSubjectType(),
                    triple.getRelationContext());
            addMention(nodes, triple.getTarget(), triple.getTargetType(),
                    triple.getRelationContext());
        }
        if (nodes.size() < 2) return triples;

        Map<String, String> parent = new HashMap<>();
        Map<String, EntityNode> allNodes = new HashMap<>();
        Map<String, List<Float>> vectors = new HashMap<>();
        // Initialize each mention as a union-find set and cache its embedding vector.
        for (Map.Entry<String, EntityNode> entry : nodes.entrySet()) {
            String key = mentionKey(entry.getKey());
            parent.put(key, key);
            allNodes.put(key, entry.getValue());
            vectors.put(key, neo4j.embeddingVector(entry.getValue().name()));
        }
        List<String> mentionKeys = new ArrayList<>(parent.keySet());
        // Compare every pair and treat similarities above the threshold as possible
        // matches. The old threshold 0.95 merged only near-identical forms; strict mode
        // uses 0.90, configurable with -Dgraph.mention.merge.cosine.
        double mentionCosine = 0.95;
        for (int left = 0; left < mentionKeys.size(); left++) {
            for (int right = left + 1; right < mentionKeys.size(); right++) {
                if (cosine(vectors.get(mentionKeys.get(left)),
                        vectors.get(mentionKeys.get(right))) >= mentionCosine) {
                    union(parent, mentionKeys.get(left), mentionKeys.get(right));
                }
            }
        }
        // Existing-entity distance threshold: 0.05 before, relaxed to 0.15 in strict mode.
        double mergeDistance = 0.05;
        for (String mentionKey : mentionKeys) {
            EntityNode mention = allNodes.get(mentionKey);
            try {
                // Align with existing graph entities; a looser threshold merges more cross-sample matches.
                for (Neo4jClient.EntitySearchResult hit : neo4j.searchEntitiesWithinDistance(
                        experimentId, graphDatasetId, mention.name(), 5, mergeDistance)) {
                    String existingKey = "existing:" + hit.entityId();
                    parent.putIfAbsent(existingKey, existingKey);
                    allNodes.putIfAbsent(existingKey,
                            new EntityNode(hit.name(), hit.type(), List.of()));
                    union(parent, mentionKey, existingKey);
                }
            } catch (RuntimeException ignored) {
                // Candidate retrieval is optional; empty or failed searches do not block later steps.
            }
        }

        // Group nodes by union-find root; only groups with multiple members need an LLM decision.
        Map<String, List<String>> groups = new LinkedHashMap<>();
        for (String key : allNodes.keySet()) {
            groups.computeIfAbsent(find(parent, key), ignored -> new ArrayList<>()).add(key);
        }
        List<Map<String, Object>> requestGroups = new ArrayList<>();
        int groupId = 0;
        for (List<String> group : groups.values()) {
            if (group.size() < 2) continue;
            // Groups containing only existing entities are already represented in the graph.
            boolean hasMention = group.stream().anyMatch(key -> key.startsWith("mention:"));
            if (!hasMention) continue;
            List<Map<String, Object>> mentions = new ArrayList<>();
            for (String key : group.stream().limit(10).toList()) {
                EntityNode node = allNodes.get(key);
                Map<String, Object> item = new LinkedHashMap<>();
                item.put("name", node.name());
                item.put("type", node.type());
                item.put("contexts", node.contexts().stream().limit(3).toList());
                mentions.add(item);
            }
            Map<String, Object> request = new LinkedHashMap<>();
            request.put("group_id", groupId++);
            request.put("mentions", mentions);
            requestGroups.add(request);
            // Process at most 20 groups per request to keep the prompt bounded.
            if (requestGroups.size() >= 20) break;
        }
        // With no groups requiring a decision, return the original triples.
        if (requestGroups.isEmpty()) return triples;

        Map<String, String[]> canonicalByAlias = new HashMap<>();
        try {
            // Resolve all group names and types in one non-thinking LLM call to save tokens.
            String response = llm.chatWithSystemPrompt("kg-entity-merge-" + UUID.randomUUID(),
                    String.format(KgEntityMerge, JsonCodec.toJson(Map.of("groups", requestGroups))),
                    LlmChatClient.withNoThinkSwitch("Resolve the supplied entity groups."));
            canonicalByAlias.putAll(parseEntityMergeResponse(response));
        } catch (RuntimeException error) {
            // Decision failure keeps original triples and allows construction to continue.
            System.err.println("Entity merge decision skipped: " + error.getMessage());
            return triples;
        }
        if (canonicalByAlias.isEmpty()) return triples;
        // Rewrite subjects and targets through the alias map, replacing names and
        // types. Unmapped entities, relations, and evidence remain unchanged.
        List<KnowledgeTriple> result = new ArrayList<>(triples.size());
        for (KnowledgeTriple triple : triples) {
            String[] subject = canonicalByAlias.get(entityKey(
                    triple.getSubject(), triple.getSubjectType()));
            String[] target = canonicalByAlias.get(entityKey(
                    triple.getTarget(), triple.getTargetType()));
            result.add(new KnowledgeTriple(
                    subject == null ? triple.getSubject() : subject[0],
                    subject == null ? triple.getSubjectType() : subject[1],
                    triple.getRelation(),
                    target == null ? triple.getTarget() : target[0],
                    target == null ? triple.getTargetType() : target[1],
                    triple.getRelationContext()));
        }
        return List.copyOf(result);
    }

    /** Parse entity-merge responses in object, top-level array, or nested {merges:[...]} form. */
    private static Map<String, String[]> parseEntityMergeResponse(String response) {
        Map<String, String[]> result = new HashMap<>();
        JsonElement parsed = JsonParser.parseString(response);
        collectEntityMerges(parsed, result, 0);
        return result;
    }

    /**
     * Recursively collect entity merges because model responses may wrap them at any
     * depth. Register objects containing canonical_name and canonical_type; otherwise
     * traverse child nodes up to depth 6.
     */
    private static void collectEntityMerges(JsonElement element,
                                            Map<String, String[]> result,
                                            int depth) {
        if (element == null || element.isJsonNull() || depth > 6) return;
        if (element.isJsonArray()) {
            for (JsonElement child : element.getAsJsonArray()) {
                collectEntityMerges(child, result, depth + 1);
            }
            return;
        }
        if (!element.isJsonObject()) return;
        JsonObject object = element.getAsJsonObject();
        if (object.has("canonical_name") && object.has("canonical_type")) {
            addEntityMerge(object, result);
        }
        if (object.has("merges")) {
            collectEntityMerges(object.get("merges"), result, depth + 1);
        } else if (!object.has("canonical_name")) {
            for (Map.Entry<String, JsonElement> entry : object.entrySet()) {
                JsonElement value = entry.getValue();
                if (value != null && (value.isJsonArray() || value.isJsonObject())) {
                    collectEntityMerges(value, result, depth + 1);
                }
            }
        }
    }

    /** Expand {canonical_name, canonical_type, aliases:[{name,type}]} into an alias-to-canonical mapping. */
    private static void addEntityMerge(JsonObject merge, Map<String, String[]> result) {
        String canonicalName = stringValue(merge, "canonical_name");
        String canonicalType = stringValue(merge, "canonical_type");
        if (canonicalName.isBlank() || canonicalType.isBlank()) return;
        JsonElement aliases = merge.get("aliases");
        if (aliases == null || !aliases.isJsonArray()) return;
        for (JsonElement alias : aliases.getAsJsonArray()) {
            if (!alias.isJsonObject()) continue;
            String name = stringValue(alias.getAsJsonObject(), "name");
            String type = stringValue(alias.getAsJsonObject(), "type");
            if (!name.isBlank() && !type.isBlank()) {
                result.put(entityKey(name, type), new String[]{canonicalName, canonicalType});
            }
        }
    }

    /** Register one mention; keep the first occurrence of each name and type pair with its evidence. */
    private static void addMention(Map<String, EntityNode> nodes, String name,
                                   String type, String context) {
        if (name == null || name.isBlank() || type == null || type.isBlank()) return;
        String key = entityKey(name, type);
        nodes.computeIfAbsent(key, ignored -> new EntityNode(name.trim(), type.trim(),
                context == null || context.isBlank() ? List.of() : List.of(context.trim())));
    }

    /** Entity key: normalized name plus \0 plus normalized type to avoid concatenation ambiguity. */
    private static String entityKey(String name, String type) {
        return EntityRelationNormalizer.normalizeName(name) + "\u0000"
                + EntityRelationNormalizer.normalizeRelation(type);
    }

    /** Prefix new sample mentions to distinguish them from existing graph entities. */
    private static String mentionKey(String entityKey) {
        return "mention:" + entityKey;
    }

    /** Union-find root lookup that walks parent links to the root. */
    private static String find(Map<String, String> parent, String key) {
        String current = key;
        while (!current.equals(parent.getOrDefault(current, current))) {
            current = parent.getOrDefault(current, current);
        }
        return current;
    }

    /** Union-find merge that attaches the right root under the left root. */
    private static void union(Map<String, String> parent, String left, String right) {
        String leftRoot = find(parent, left);
        String rightRoot = find(parent, right);
        if (!leftRoot.equals(rightRoot)) parent.put(rightRoot, leftRoot);
    }

    /** Cosine similarity, returning 0 for mismatched dimensions, empty vectors, or zero norms. */
    private static double cosine(List<Float> left, List<Float> right) {
        if (left == null || right == null || left.isEmpty() || left.size() != right.size()) return 0.0;
        double dot = 0.0;
        double leftNorm = 0.0;
        double rightNorm = 0.0;
        for (int index = 0; index < left.size(); index++) {
            double a = left.get(index);
            double b = right.get(index);
            dot += a * b;
            leftNorm += a * a;
            rightNorm += b * b;
        }
        if (leftNorm == 0.0 || rightNorm == 0.0) return 0.0;
        return dot / (Math.sqrt(leftNorm) * Math.sqrt(rightNorm));
    }

    /**
     * [Step 4] Collect every extracted entity type.
     */
    private static List<String> extractedTypes(List<KnowledgeTriple> triples) {
        Set<String> types = new HashSet<>();
        // Collect subject and target types, normalizing to lowercase.
        for (KnowledgeTriple triple : triples) {
            if (triple.getSubjectType() != null && !triple.getSubjectType().isBlank()) {
                types.add(triple.getSubjectType().trim().toLowerCase(Locale.ROOT));
            }
            if (triple.getTargetType() != null && !triple.getTargetType().isBlank()) {
                types.add(triple.getTargetType().trim().toLowerCase(Locale.ROOT));
            }
        }
        return List.copyOf(types);
    }

    /**
     * Extract one chunk by acquiring a global permit, calling the LLM, and parsing
     * triples. The finally block must release the permit to avoid leaking capacity.
     */
    private static List<KnowledgeTriple> extractChunk(LlmChatClient llm, String chunk,
                                                       int index, int count) {
        boolean acquired = false;
        try {
            // Block until a global permit is available, bounding in-flight extraction requests.
            CHUNK_LLM_PERMITS.acquire();
            acquired = true;
            String response = extractInitialGraph(llm, chunk);
            return KnowledgeTripleParser.parse(response);
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw stageFailure(Stage.INITIAL_EXTRACTION,
                    new IllegalStateException("chunk extraction interrupted", interrupted));
        } catch (RuntimeException error) {
            // Include chunk position and count to localize failures.
            throw stageFailure(Stage.INITIAL_EXTRACTION,
                    new IllegalStateException("chunk " + (index + 1) + "/" + count
                            + " extraction failed", error));
        } finally {
            if (acquired) CHUNK_LLM_PERMITS.release();
        }
    }

    /**
     * [Step 1] Perform strict dataset-aware passage splitting: each benchmark passage
     * becomes exactly one chunk and is never cut internally, preserving semantic
     * completeness for higher extraction quality.
     *
     * <p>First parse context as JSON and extract passages according to dataset format.
     * If parsing fails or yields no passages, fall back to splitting plain text on
     * blank lines.</p>
     */
    private static List<String> splitContextForDataset(String context,
                                                        String graphDatasetId) {
        if (context == null || context.isBlank()) return List.of();
        List<String> result = new ArrayList<>();
        try {
            JsonElement root = JsonParser.parseString(context.trim());
            if (root.isJsonArray()) {
                for (JsonElement entry : root.getAsJsonArray()) {
                    String paragraph = datasetParagraph(graphDatasetId, entry);
                    if (!paragraph.isBlank()) result.add(paragraph);
                }
                if (!result.isEmpty()) return List.copyOf(result);
            }
        } catch (RuntimeException ignored) {
            // Plain-text context falls through to blank-line splitting.
        }
        String prose = ContextSanitizer.toPromptText(context);
        // Plain-text fallback uses two consecutive newlines as the passage boundary.
        for (String paragraph : prose.split("(?:\\r?\\n){2,}")) {
            if (!paragraph.isBlank()) result.add(paragraph.trim());
        }
        if (result.isEmpty() && !prose.isBlank()) result.add(prose.trim());
        return List.copyOf(result);
    }

    /**
     * Convert one JSON record into plain-text passage according to dataset format.
     * <ul>
     *   <li>MuSiQue: object with paragraph_text.</li>
     *   <li>2WikiMultiHopQA: ["title", ["sentence", ...]]; prepend the title when absent from the body.</li>
     *   <li>HotpotQA: ["title", ["sentence", ...]] or ["sentence", ...].</li>
     *   <li>Other formats: use paragraph_text/text for objects and infer the array style.</li>
     * </ul>
     */
    private static String datasetParagraph(String graphDatasetId, JsonElement entry) {
        String dataset = graphDatasetId == null ? "" : graphDatasetId.toLowerCase(Locale.ROOT);
        if (entry == null || entry.isJsonNull()) return "";
        if (dataset.contains("musique")) {
            if (entry.isJsonObject()) {
                String paragraph = stringValue(entry.getAsJsonObject(), "paragraph_text");
                if (!paragraph.isBlank()) return paragraph;
            }
            return ContextSanitizer.toPromptText(entry.toString());
        }
        if (dataset.contains("2wiki")) {
            if (entry.isJsonArray()) {
                var values = entry.getAsJsonArray();
                if (values.size() >= 2 && values.get(0).isJsonPrimitive()
                        && values.get(1).isJsonArray()) {
                    String title = values.get(0).getAsString().trim();
                    String body = joinStrings(values.get(1));
                    return title.isBlank() || body.toLowerCase(Locale.ROOT)
                            .contains(title.toLowerCase(Locale.ROOT))
                            ? body : title + ": " + body;
                }
            }
            return ContextSanitizer.toPromptText(entry.toString());
        }
        if (dataset.contains("hotpot")) {
            if (entry.isJsonArray()) {
                var values = entry.getAsJsonArray();
                if (values.size() >= 2 && values.get(0).isJsonPrimitive()
                        && values.get(1).isJsonArray()) {
                    String title = values.get(0).getAsString().trim();
                    String body = joinStrings(values.get(1));
                    return title.isBlank() || body.toLowerCase(Locale.ROOT)
                            .contains(title.toLowerCase(Locale.ROOT))
                            ? body : title + ": " + body;
                }
                return joinStrings(values);
            }
            return ContextSanitizer.toPromptText(entry.toString());
        }
        if (entry.isJsonObject()) {
            var object = entry.getAsJsonObject();
            String paragraph = stringValue(object, "paragraph_text");
            if (paragraph.isBlank()) paragraph = stringValue(object, "text");
            if (!paragraph.isBlank()) return paragraph;
            return ContextSanitizer.toPromptText(entry.toString());
        }
        if (entry.isJsonArray()) {
            var values = entry.getAsJsonArray();
            // 2Wiki-style paragraph: [title, [sentence, sentence, ...]]
            if (values.size() >= 2 && values.get(0).isJsonPrimitive()
                    && values.get(1).isJsonArray()) {
                String title = values.get(0).getAsString().trim();
                String body = joinStrings(values.get(1));
                return title.isBlank() || body.toLowerCase(Locale.ROOT)
                        .contains(title.toLowerCase(Locale.ROOT))
                        ? body : title + ": " + body;
            }
            // HotpotQA-style paragraph: [sentence, sentence, ...]
            return joinStrings(values);
        }
        return ContextSanitizer.toPromptText(entry.toString());
    }

    /** Join string elements with spaces, ignoring non-string and empty values. */
    private static String joinStrings(JsonElement array) {
        if (array == null || !array.isJsonArray()) return "";
        StringBuilder value = new StringBuilder();
        for (JsonElement item : array.getAsJsonArray()) {
            if (item == null || !item.isJsonPrimitive()) continue;
            String text = item.getAsString().trim();
            if (text.isBlank()) continue;
            if (value.length() > 0) value.append(' ');
            value.append(text);
        }
        return value.toString().trim();
    }

    /** Safely read a string field, returning an empty string when absent or non-string. */
    private static String stringValue(JsonObject object, String name) {
        JsonElement value = object.get(name);
        return value != null && value.isJsonPrimitive() ? value.getAsString().trim() : "";
    }

    /**
     * [Core step 2] Extract an initial graph by asking the LLM for triples from one
     * passage and returning raw JSON.
     *
     * <p>Three-level fallback applies only to moderation failures:</p>
     * <ol>
     *   <li>Use the normal KgBuild prompt plus completeness requirements.</li>
     *   <li>If moderation rejects it, use a neutral structured-extraction prompt and
     *       wrap the passage in {"benchmark_passage": ...} JSON.</li>
     *   <li>If rejected again, split into chunks of at most 3,500 characters and
     *       extract each chunk separately.</li>
     * </ol>
     *
     * <p>Non-moderation errors such as network failures or timeouts immediately throw
     * BuildStageException(INITIAL_EXTRACTION); callers decide whether to skip the
     * sample or stop the batch.</p>
     */
    private static String extractInitialGraph(LlmChatClient llm, String context) {
        try {
            // Level 1: normal extraction.
            return llm.chatWithSystemPrompt("initial_extraction-" + UUID.randomUUID(),
                    KgBuild + COMPLETE_TRIPLE_REQUIREMENT,
                    LlmChatClient.withNoThinkSwitch(context));
        } catch (RuntimeException first) {
            if (!LlmChatClient.isModerationFailure(first)) {
                throw stageFailure(Stage.INITIAL_EXTRACTION, first);
            }
            try {
                // Level 2: retry with the neutral prompt.
                return llm.chatWithSystemPrompt("initial_extraction-neutral-" + UUID.randomUUID(),
                        NEUTRAL_EXTRACTION_PROMPT,
                        LlmChatClient.withNoThinkSwitch(
                                JsonCodec.toJson(Map.of("benchmark_passage", context))));
            } catch (RuntimeException second) {
                if (!LlmChatClient.isModerationFailure(second)) {
                    // A non-moderation retry failure indicates another problem. Preserve
                    // the first exception as suppressed for diagnosis.
                    second.addSuppressed(first);
                    throw stageFailure(Stage.INITIAL_EXTRACTION, second);
                }
                // Level 3: split into smaller chunks, which are more likely to pass moderation.
                return extractInitialGraphByChunks(llm, context, first, second);
            }
        }
    }

    /**
     * Final moderation fallback: split source text at
     * {@code MODERATION_CHUNK_CHARS}, extract sequentially with the neutral prompt,
     * deduplicate triples, and rebuild {"triples":[...]} JSON. Sequential execution is
     * safer for moderation-sensitive input and localizes the failing chunk.
     */
    private static String extractInitialGraphByChunks(LlmChatClient llm, String context,
                                                       RuntimeException first,
                                                       RuntimeException second) {
        List<String> chunks = splitContextForExtraction(context, MODERATION_CHUNK_CHARS);
        LinkedHashMap<String, KnowledgeTriple> triples = new LinkedHashMap<>();
        for (int index = 0; index < chunks.size(); index++) {
            String response;
            try {
                response = llm.chatWithSystemPrompt("initial_extraction-part-"
                                + (index + 1) + "-" + UUID.randomUUID(), NEUTRAL_EXTRACTION_PROMPT,
                        LlmChatClient.withNoThinkSwitch(
                                JsonCodec.toJson(Map.of("benchmark_passage_part", chunks.get(index),
                                        "part_number", index + 1, "part_count", chunks.size()))));
            } catch (RuntimeException partFailure) {
                // Attach both earlier moderation failures to preserve the full chain.
                partFailure.addSuppressed(first);
                partFailure.addSuppressed(second);
                throw stageFailure(Stage.INITIAL_EXTRACTION,
                        new IllegalStateException("moderation fallback part " + (index + 1)
                                + " of " + chunks.size() + " failed", partFailure));
            }
            List<KnowledgeTriple> parsed;
            try {
                parsed = KnowledgeTripleParser.parse(response);
            } catch (RuntimeException malformed) {
                throw stageFailure(Stage.INITIAL_EXTRACTION,
                        new IllegalArgumentException("moderation fallback part " + (index + 1)
                                + " of " + chunks.size() + " returned invalid triple JSON", malformed));
            }
            for (KnowledgeTriple triple : parsed) {
                triples.putIfAbsent(tripleKey(triple), triple);
            }
        }
        if (triples.isEmpty()) {
            throw stageFailure(Stage.INITIAL_EXTRACTION,
                    new IllegalStateException("moderation fallback extracted no complete triples"));
        }
        System.err.printf("Initial extraction used %d passage parts after moderation and produced %d triples.%n",
                chunks.size(), triples.size());
        // Restore the normal extraction response shape so later parsing is source-agnostic.
        return JsonCodec.toJson(Map.of("triples", new ArrayList<>(triples.values())));
    }

    /**
     * Moderation-fallback splitting: split JSON context by array element and plain
     * text by blank lines, then use {@link #addBounded} to enforce maxChars.
     */
    static List<String> splitContextForExtraction(String context, int maxChars) {
        if (context == null || context.isBlank()) return List.of();
        if (maxChars < 1) throw new IllegalArgumentException("maxChars must be positive");
        try {
            JsonElement root = JsonParser.parseString(context.trim());
            if (root.isJsonArray()) {
                List<String> parts = new ArrayList<>();
                for (JsonElement item : root.getAsJsonArray()) {
                    addBounded(parts, item.toString(), maxChars);
                }
                if (!parts.isEmpty()) return List.copyOf(parts);
            }
        } catch (RuntimeException ignored) {
            // Non-JSON context falls through to passage-boundary splitting.
        }
        List<String> parts = new ArrayList<>();
        for (String paragraph : context.trim().split("(?:\\r?\\n){2,}")) {
            if (!paragraph.isBlank()) addBounded(parts, paragraph.trim(), maxChars);
        }
        if (parts.isEmpty()) addBounded(parts, context.trim(), maxChars);
        return List.copyOf(parts);
    }

    /** Hard-split text into chunks no longer than maxChars and append them to parts. */
    private static void addBounded(List<String> parts, String value, int maxChars) {
        for (int start = 0; start < value.length(); start += maxChars) {
            parts.add(value.substring(start, Math.min(value.length(), start + maxChars)));
        }
    }

    /**
     * Triple fingerprint joining six fields with \0. Identical triples, including
     * evidence, are retained once across chunks.
     */
    private static String tripleKey(KnowledgeTriple triple) {
        return String.join("\u0000", triple.getSubject(), triple.getSubjectType(),
                triple.getRelation(), triple.getTarget(), triple.getTargetType(),
                triple.getRelationContext() == null ? "" : triple.getRelationContext());
    }
    /**
     * Generic moderation fallback: try the normal prompt, then a neutral prompt once
     * if moderation rejects the request. Reused by quality checking and similar stages.
     */
    private static String callWithModerationFallback(LlmChatClient llm, Stage stage,
                                                     String systemPrompt, String userText,
                                                     String fallbackSystemPrompt,
                                                     String fallbackUserText) {
        try {
            return llm.chatWithSystemPrompt(stage.value() + "-" + UUID.randomUUID(),
                    systemPrompt, LlmChatClient.withNoThinkSwitch(userText));
        } catch (RuntimeException first) {
            if (!LlmChatClient.isModerationFailure(first)) throw stageFailure(stage, first);
            try {
                return llm.chatWithSystemPrompt(stage.value() + "-neutral-" + UUID.randomUUID(),
                        fallbackSystemPrompt,
                        LlmChatClient.withNoThinkSwitch(fallbackUserText));
            } catch (RuntimeException retry) {
                // The neutral retry also failed; preserve the first exception and wrap the stage failure.
                retry.addSuppressed(first);
                throw stageFailure(stage, retry);
            }
        }
    }

    /** Wrap an exception with stage information; already staged exceptions pass through. */
    private static BuildStageException stageFailure(Stage stage, RuntimeException error) {
        if (error instanceof BuildStageException staged) return staged;
        return new BuildStageException(stage, LlmChatClient.isModerationFailure(error), error);
    }

    /** Build a short "class name plus message" description for BuildStageException. */
    private static String causeMessage(Throwable error) {
        if (error == null) return "unknown failure";
        String message = error.getMessage();
        return error.getClass().getSimpleName() + (message == null || message.isBlank()
                ? "" : ": " + message);
    }

    /**
     * Command-line entry for single-sample construction used in debugging or backfill.
     * It builds one data_set row without deleting existing graph data. The default IDs
     * are hard-coded placeholders and should be replaced for the target run.
     */
    public static void main(String[] args) {
        long experimentDbId = longOption(args, "experiment-id", 87L);
        if (experimentDbId <= 0) {
            throw new IllegalArgumentException("--experiment-id must be positive");
        }
        String experimentId = Long.toString(experimentDbId);
        long sourceDatasetId = longOption(args, "source-dataset-id", 87331L);
        if (sourceDatasetId <= 0) {
            throw new IllegalArgumentException("--source-dataset-id must be positive");
        }
        AppConfig appConfig = AppConfig.load();
        try (KgPaperRepository repository = new KgPaperRepository(appConfig);
             Neo4jClient neo4j = new Neo4jClient(appConfig)) {
            DataSet sample = repository.findDataSetById(sourceDatasetId);
            if (sample == null) throw new IllegalArgumentException("data_set row not found: " + sourceDatasetId);
            ExperimentData experimentData = repository.findExperimentData(experimentDbId, sourceDatasetId);
            if (experimentData == null) {
                throw new IllegalArgumentException("data_set row " + sourceDatasetId
                        + " is not part of experiment " + experimentDbId);
            }
            String graphDatasetId = datasetNamespace(sample.getDataType());
            // Run the main construction flow, then mark experiment_data as graph_built.
            BuildResult result = buildSample(new LlmChatClient(appConfig), neo4j,
                    experimentId, graphDatasetId, sourceDatasetId, sample.getContext());
            repository.markGraphBuilt(experimentData.getId());
            // Print graph size metrics and the full type-mapping result.
            System.out.printf("Graph build completed: experiment=%s dataset=%s sample=%d triples=%d types=%d%n",
                    experimentId, graphDatasetId, sourceDatasetId, result.storedTriples().size(),
                    result.typeMapping().types().size());
            System.out.println(JsonCodec.toJson(result.typeMapping()));
        }
    }

    /**
     * Convert a data type such as "2WikiMultiHopQA" into a graph namespace such as
     * "2wikimultihopqa" by lowercasing and replacing characters outside [a-z0-9._-]
     * with hyphens. Samples in one dataset share this namespace, which lets
     * OurMethodReasoning constrain answer grounding to that sample's graph.
     */
    public static String datasetNamespace(String dataType) {
        requireText(dataType, "dataType");
        String normalized = dataType.trim().toLowerCase(Locale.ROOT)
                .replaceAll("[^a-z0-9._-]+", "-");
        if (normalized.isBlank()) throw new IllegalArgumentException("dataType must contain an identifier");
        return normalized;
    }

    /** Read an --name=<int> option or return fallback. */
    private static int intOption(String[] args, String name, int fallback) {
        String value = option(args, name);
        return value == null ? fallback : Integer.parseInt(value);
    }

    /** Read an --name=<double> option or return fallback. */
    private static double doubleOption(String[] args, String name, double fallback) {
        String value = option(args, name);
        return value == null ? fallback : Double.parseDouble(value);
    }

    private static long longOption(String[] args, String name, long fallback) {
        String value = option(args, name);
        return value == null ? fallback : Long.parseLong(value);
    }

    /** Minimal command-line parser supporting --name=value and returning null when absent. */
    private static String option(String[] args, String name) {
        String prefix = "--" + name + "=";
        for (String arg : args) if (arg.startsWith(prefix)) return arg.substring(prefix.length());
        return null;
    }

    /** Validate graph namespace and ID characters to keep Cypher identifiers well-formed. */
    private static String requireId(String value, String name) {
        if (value == null || value.isBlank() || !value.matches("[A-Za-z0-9._-]+")) {
            throw new IllegalArgumentException(name + " must match [A-Za-z0-9._-]+: " + value);
        }
        return value;
    }

    /** Validate that a string is neither null nor blank. */
    private static void requireText(String value, String name) {
        if (value == null || value.isBlank()) throw new IllegalArgumentException(name + " must not be blank");
    }
}
