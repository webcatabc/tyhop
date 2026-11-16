package org.example.experiment.ours;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.example.entity.TypeMappingResult;
import org.example.experiment.DataItem;
import org.example.experiment.DataLoader;
import org.example.experiment.GraphBuild.GraphBuild;
import org.example.infrastructure.LlmChatClient;
import org.example.infrastructure.Neo4jClient;
import org.example.utils.AnswerChecker;
import org.example.utils.ContextSanitizer;
import org.example.utils.ExperimentInitializer;
import org.example.utils.json.JsonCodec;

import java.io.IOException;
import java.net.Proxy;
import java.net.ProxySelector;
import java.net.SocketAddress;
import java.net.URI;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.function.LongFunction;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.nio.charset.StandardCharsets;

import static org.example.constants.LlmPrompt.OursAnswerTypePrediction;
import static org.example.constants.LlmPrompt.OursAnswerTypeSelection;
import static org.example.constants.LlmPrompt.OursEdgePrune;
import static org.example.constants.LlmPrompt.OursEvaluate;
import static org.example.constants.LlmPrompt.OursFallbackAnswer;
import static org.example.constants.LlmPrompt.OursFallbackContextEvidence;
import static org.example.constants.LlmPrompt.OursFallbackContextSynthesis;
import static org.example.constants.LlmPrompt.OursFallbackTypeGraphRank;
import static org.example.constants.LlmPrompt.OursGraphAnswer;
import static org.example.constants.LlmPrompt.OursGraphAnswerCompact;
import static org.example.constants.LlmPrompt.OursHopSelect;
import static org.example.constants.LlmPrompt.OursSubquestionPlan;

/*
 * ============================================================================
 * TyHop overview (type-aware and provenance-aware graph construction and reasoning)
 * ============================================================================
 *
 * In one sentence: the LLM first extracts typed knowledge triples from each sample's source
 * passage and writes them to Neo4j (the construction phase, see GraphBuild.java). It then
 * performs type-aware path search over the graph. If the graph cannot produce an answer, the
 * provenance votes collected along the paths are used to recover the answer from the source context.
 *
 * Why this design (matching the paper's three main contributions):
 *   1) Type awareness: the LLM predicts the answer type, and that type is attached to entities
 *      to form a type graph. Search uses the predicted type to assess whether a candidate answer
 *      is plausible and to avoid semantic drift from relation matching alone.
 *   2) Provenance awareness: every edge and entity records which source samples it came from
 *      (data_set row IDs). Search votes with the entities and relations it uses, and the sample
 *      with the most votes is the most likely source passage for the answer.
 *   3) Subquestion states are executed in dependency order. A resolved intermediate answer
 *      becomes the start entity of the next dependent subquestion. Provenance fallback is used
 *      when graph search cannot produce a usable answer.
 *
 * Execution flow (also the recommended reading order):
 *
 *   [Entry] main(...)                                 parse CLI arguments + create/load an experiment
 *     └─ run(loader, llm, neo4j, checker, ...)       batch orchestration: build graphs, then answer each question
 *          ├─ buildOne(...)                          build one sample graph (delegates to GraphBuild.buildSample)
 *          │    └─ GraphBuild.buildSample(...)       graph-construction details are documented in GraphBuild.java
 *          └─ runSample(...)                         complete reasoning for one question
 *               ├─ Step 1 resolveAnswerTypes(...)    predict answer types -> vector recall -> type-graph paths -> LLM selection
 *               ├─ Step 2 topicEntities(...)         extract topic entities and map them to graph seed nodes
 *               ├─ Step 3 searchWithSubquestions(...) plan and execute typed subquestion states
 *               │     ├─ scoreEdges(...)             score all neighbour edges of an entity in one LLM call
 *               │     ├─ evaluate(...)               decide whether the current paths are sufficient to answer
 *               │     ├─ graphAnswer(...)            ask the LLM for an answer based on paths, constrained to path entities
 *               │     └─ fallback(...)               graph failure: fall back to source context using provenance votes
 *               └─ Step 4 groundAnswerOnSampleEntity(...) align the answer with an entity name in the sample graph
 *
 * Key terms repeated in the comments below:
 *   width/depth          nodes retained per layer / maximum search depth (beam-search width and depth)
 *   edge pruning         score an edge from 0 to 1 and discard low-scoring edges
 *   type graph           graph among entity types (type -> relation -> type) used to test type compatibility
 *   provenance votes     count which source samples contain the entities/relations found during search; relation weight 2
 *   context fallback     use passages selected by provenance votes and ask the LLM to answer again
 * ============================================================================
 */

/**
 * Type-aware and provenance-aware graph-construction and reasoning pipeline.
 *
 * <p>This class handles reasoning and orchestration. Actual graph construction (triple extraction,
 * entity merging, and persistence) is implemented in {@link GraphBuild}. All public methods are
 * static, and state is kept in local variables and records so multiple samples can run in parallel.</p>
 */
public final class OurMethodReasoning {
    /** Default number of concurrent threads (samples processed at the same time). */
    public static final int DEFAULT_THREADS = 10;
    /** Canonical unknown answer used when persisting a sample that could not be answered. */
    public static final String UNKNOWN_ANSWER = "unknown";
    /** Marker for transient failures that should be retried instead of scored as no-answer. */
    public static final String RETRY_ANSWER = "retry";
    /** Grounding threshold: rewrite the answer to a sample entity name when cosine similarity exceeds it. */
    public static final double DEFAULT_ANSWER_GROUNDING_SCORE = 0.95;
    /** Number of candidate entities retrieved from the vector store for answer grounding. */
    private static final int ANSWER_GROUNDING_CANDIDATES = 5;
    /** Short answers (booleans and numeric placeholders) are not rewritten to avoid homonym contamination. */
    private static final Set<String> ANSWER_GROUNDING_STOP_WORDS = Set.of(
            "yes", "no", "true", "false", "unknown", "none", "null", "retry");

    private OurMethodReasoning() { }

    /** One reachable subquestion produced by the planner. */
    private record Subquestion(String question, String startHint, String targetType) {
        public Subquestion {
            question = question == null ? "" : question.trim();
            startHint = startHint == null ? "" : startHint.trim();
            targetType = targetType == null ? "" : targetType.trim();
        }
    }

    /** One intermediate answer paired with the subquestion that produced it. */
    private record ResolvedValue(String subquestion, String answer) {
        public ResolvedValue {
            subquestion = subquestion == null ? "" : subquestion.trim();
            answer = answer == null ? "" : answer.trim();
        }
    }

    /** One bounded local path shown to the hop selector. */
    private record LocalPath(int id, List<PathStep> steps,
                             List<Neo4jClient.IncidentRelationEvidence> edges,
                             String terminalEntityId, String terminalEntityName,
                             String terminalEntityType) {
        public LocalPath {
            steps = steps == null ? List.of() : List.copyOf(steps);
            edges = edges == null ? List.of() : List.copyOf(edges);
        }
    }

    /** Parsed local decision for one subquestion. */
    private record HopDecision(List<Integer> selectedPathIds, String answer,
                               boolean resolved, double confidence, String reason) {
        public HopDecision {
            selectedPathIds = selectedPathIds == null
                    ? List.of() : List.copyOf(selectedPathIds);
            answer = answer == null ? "" : answer.trim();
            reason = reason == null ? "" : reason;
        }
    }

    private static final int SUBQUESTION_PATH_LIMIT = 40;
    private static final int SUBQUESTION_ENUM_LIMIT = 240;

    static int subquestionPathLimit() {
        return SUBQUESTION_PATH_LIMIT;
    }

    /**
     * All hyperparameters are centralized here in an immutable record with validation at construction.
     * Inline comments show the defaults from {@link #defaults()} for easy tuning.
     */
    public record Config(
            // ---- ToG search shape (beam search) ----
            int width,                          // high-scoring nodes retained per layer, default 6
            int depth,                          // maximum expansion depth, default 5
            int maxTopicEntities,               // maximum seed entities per question, default 5
            int maxIncidentRelations,           // maximum incident edges retrieved for one entity, default 200
            int maxNeighbours,                  // maximum candidate edges in one edge-scoring call, default 10
            // ---- Answer-type prediction / type graph ----
            int predictedTypeLimit,             // maximum answer types predicted by the LLM, default 3
            int typeVectorCandidates,           // graph types recalled per predicted type, default 5
            double typeMaxDistance,             // maximum type-vector distance; lower is stricter, default 0.45
            int typeGraphHops,                  // hops allowed when testing type compatibility, default 2
            int typeGraphPathLimit,             // maximum number of type-graph paths, default 80
            int answerTypeLimit,                // maximum candidate answer types retained, default 8
            // ---- Path context / fallback budget ----
            int maxEdgeContextChars,            // source-evidence character limit per edge in the prompt, default 1200
            int maxFallbackContextChars,        // total fallback-context budget and batching threshold, default 60000
            int fallbackCandidatePool,          // fallback candidate-pool size, default 20
            int fallbackContextLimit,           // contexts sent to the LLM, default 3
            int fallbackTypeMatchLimit,         // type-match records per candidate sample, default 5
            double fallbackTypeMaxDistance,     // maximum type-match vector distance during fallback, default 0.65
            double fallbackTypeWeight,          // weight of the type score in fallback ranking, default 0.60
            int fallbackTypeGraphEdgeLimit,     // type-graph edges retained per candidate sample, default 100
            // ---- LLM reranking during fallback ----
            boolean fallbackLlmRerank,          // whether to rerank fallback candidates with the LLM, default true
            int fallbackLlmCandidateLimit,      // candidates sent to the LLM reranker, default 5
            double fallbackLlmWeight,           // weight of the LLM reranking score, default 0.60
            // ---- Context and answer post-processing ----
            boolean fullContextFallback,        // send full source passages without truncation, default true
            boolean groundAnswerEntity,         // rewrite answers to sample entity names, default true
            double groundAnswerMinScore) {      // minimum similarity required for answer grounding, default 0.95
        public Config {
            // Integer limits must be positive, and fallback limits must be nested:
            // fallbackContextLimit <= fallbackLlmCandidateLimit <= fallbackCandidatePool,
            // These constraints ensure that fallback always has enough candidate context.
            if (width < 1 || depth < 1 || maxTopicEntities < 1
                    || maxIncidentRelations < 1 || maxNeighbours < 1
                    || predictedTypeLimit < 1 || typeVectorCandidates < 1
                    || typeGraphHops < 0 || typeGraphHops > 10
                    || typeGraphPathLimit < 1 || answerTypeLimit < 1
                    || maxEdgeContextChars < 1 || maxFallbackContextChars < 1
                    || fallbackCandidatePool < 1 || fallbackContextLimit < 1
                    || fallbackContextLimit > fallbackCandidatePool
                    || fallbackTypeMatchLimit < 1 || fallbackTypeGraphEdgeLimit < 1
                    || fallbackLlmCandidateLimit < fallbackContextLimit
                    || fallbackLlmCandidateLimit > fallbackCandidatePool) {
                throw new IllegalArgumentException("reasoning limits are invalid");
            }
            // Distance and weight parameters must be finite and within their valid ranges.
            if (!Double.isFinite(typeMaxDistance) || typeMaxDistance < 0.0) {
                throw new IllegalArgumentException("typeMaxDistance must be finite and non-negative");
            }
            if (!Double.isFinite(fallbackTypeMaxDistance) || fallbackTypeMaxDistance < 0.0) {
                throw new IllegalArgumentException("fallbackTypeMaxDistance must be finite and non-negative");
            }
            if (!Double.isFinite(fallbackTypeWeight)
                    || fallbackTypeWeight < 0.0 || fallbackTypeWeight > 1.0) {
                throw new IllegalArgumentException("fallbackTypeWeight must be between 0 and 1");
            }
            if (!Double.isFinite(fallbackLlmWeight)
                    || fallbackLlmWeight < 0.0 || fallbackLlmWeight > 1.0) {
                throw new IllegalArgumentException("fallbackLlmWeight must be between 0 and 1");
            }
            if (!Double.isFinite(groundAnswerMinScore)
                    || groundAnswerMinScore < 0.0 || groundAnswerMinScore > 1.0) {
                throw new IllegalArgumentException("groundAnswerMinScore must be between 0 and 1");
            }
        }

        /**
         * Compatibility constructor for older callers (parameters through fallbackTypeGraphEdgeLimit).
         * Remaining fallback options use fixed defaults: LLM reranking (weight 0.6), full-context
         * fallback, and answer grounding enabled.
         */
        public Config(int width, int depth, int maxTopicEntities,
                      int maxIncidentRelations, int maxNeighbours,
                      int predictedTypeLimit, int typeVectorCandidates,
                      double typeMaxDistance, int typeGraphHops,
                      int typeGraphPathLimit, int answerTypeLimit,
                      int maxEdgeContextChars, int maxFallbackContextChars,
                      int fallbackCandidatePool, int fallbackContextLimit,
                      int fallbackTypeMatchLimit, double fallbackTypeMaxDistance,
                      double fallbackTypeWeight, int fallbackTypeGraphEdgeLimit) {
            this(width, depth, maxTopicEntities, maxIncidentRelations, maxNeighbours,
                    predictedTypeLimit, typeVectorCandidates, typeMaxDistance,
                    typeGraphHops, typeGraphPathLimit, answerTypeLimit,
                    maxEdgeContextChars, maxFallbackContextChars,
                    fallbackCandidatePool, fallbackContextLimit, fallbackTypeMatchLimit,
                    fallbackTypeMaxDistance, fallbackTypeWeight, fallbackTypeGraphEdgeLimit,
                    true, 5, 0.60, true, true, DEFAULT_ANSWER_GROUNDING_SCORE);
        }

        /** Earlier 13-argument constructor using built-in fallback defaults (pool 20, 3 contexts, type weight 0.6). */
        public Config(int width, int depth, int maxTopicEntities,
                      int maxIncidentRelations, int maxNeighbours,
                      int predictedTypeLimit, int typeVectorCandidates,
                      double typeMaxDistance, int typeGraphHops,
                      int typeGraphPathLimit, int answerTypeLimit,
                      int maxEdgeContextChars, int maxFallbackContextChars) {
            this(width, depth, maxTopicEntities, maxIncidentRelations, maxNeighbours,
                    predictedTypeLimit, typeVectorCandidates, typeMaxDistance,
                    typeGraphHops, typeGraphPathLimit, answerTypeLimit,
                    maxEdgeContextChars, maxFallbackContextChars,
                    20, 3, 5, 0.65, 0.60, 100,
                    true, 5, 0.60, true, true, DEFAULT_ANSWER_GROUNDING_SCORE);
        }

        /** Compatibility constructor that preserves the default full-context fallback mode. */
        public Config(int width, int depth, int maxTopicEntities,
                      int maxIncidentRelations, int maxNeighbours,
                      int predictedTypeLimit, int typeVectorCandidates,
                      double typeMaxDistance, int typeGraphHops,
                      int typeGraphPathLimit, int answerTypeLimit,
                      int maxEdgeContextChars, int maxFallbackContextChars,
                      int fallbackCandidatePool, int fallbackContextLimit,
                      int fallbackTypeMatchLimit, double fallbackTypeMaxDistance,
                      double fallbackTypeWeight, int fallbackTypeGraphEdgeLimit,
                      boolean fallbackLlmRerank, int fallbackLlmCandidateLimit,
                      double fallbackLlmWeight) {
            this(width, depth, maxTopicEntities, maxIncidentRelations, maxNeighbours,
                    predictedTypeLimit, typeVectorCandidates, typeMaxDistance,
                    typeGraphHops, typeGraphPathLimit, answerTypeLimit,
                    maxEdgeContextChars, maxFallbackContextChars,
                    fallbackCandidatePool, fallbackContextLimit, fallbackTypeMatchLimit,
                    fallbackTypeMaxDistance, fallbackTypeWeight, fallbackTypeGraphEdgeLimit,
                    fallbackLlmRerank, fallbackLlmCandidateLimit, fallbackLlmWeight,
                    true, true, DEFAULT_ANSWER_GROUNDING_SCORE);
        }

        /** Default configuration for the main paper experiment: width=6, depth=5, 3 full contexts, 60k-character budget. */
        public static Config defaults() {
            return new Config(6, 5, 5, 200, 10,
                    3, 5, 0.45, 2, 80, 8, 1200, 60000);
        }

    }

    /** Answer type predicted directly by the LLM in step 1: type, confidence, and reason. */
    public record PredictedType(String type, double confidence, String reason) { }

    /** An actual graph type recalled from the type-vector store for a prediction; lower distance is closer. */
    public record RecalledType(String type, List<String> aliases, double distance) {
        public RecalledType {
            aliases = aliases == null ? List.of() : List.copyOf(aliases);
        }
    }

    /**
     * LLM-assigned type-graph match score for a fallback candidate source sample, used for
     * interpretable ranking audits. Fields are type-match score, type-graph structure-match
     * score, overall score, and reason.
     */
    public record LlmTypeGraphAssessment(double typeMatchScore,
                                         double structureMatchScore,
                                         double score, String reason) { }

    /**
     * All ranking evidence for one fallback candidate source row.
     *
     * @param sourceDatasetId     data_set primary key of the candidate source sample
     * @param provenanceVotes     votes from graph search (1 per matched entity, 2 per matched relation)
     * @param normalizedVoteScore provenance score normalized to 0..1 by dividing by the maximum vote count
     * @param sampleTypeGraph     entity-type graph for this sample (nodes are types; edges are type relations)
     * @param typeMatches         vector-space matches between predicted types and sample types
     * @param typeScore           highest match score for this sample
     * @param deterministicScore  LLM-independent score: (1-w)*provenance score + w*type score
     * @param llmAssessment       LLM reranking assessment, or null when reranking is disabled
     * @param combinedScore       final score used for ranking
     */
    public record FallbackCandidate(long sourceDatasetId, int provenanceVotes,
                                    double normalizedVoteScore,
                                    Neo4jClient.SampleEntityTypeGraph sampleTypeGraph,
                                    List<Neo4jClient.SampleEntityTypeMatch> typeMatches,
                                    double typeScore, double deterministicScore,
                                    LlmTypeGraphAssessment llmAssessment,
                                    double combinedScore) {
        public FallbackCandidate {
            typeMatches = copy(typeMatches);
        }

        /** Simplified constructor without an LLM score (deterministicScore is also used as combinedScore). */
        public FallbackCandidate(long sourceDatasetId, int provenanceVotes,
                                 double normalizedVoteScore,
                                 Neo4jClient.SampleEntityTypeGraph sampleTypeGraph,
                                 List<Neo4jClient.SampleEntityTypeMatch> typeMatches,
                                 double typeScore, double combinedScore) {
            this(sourceDatasetId, provenanceVotes, normalizedVoteScore,
                    sampleTypeGraph, typeMatches, typeScore, combinedScore,
                    null, combinedScore);
        }
    }

    /** Metadata for source context sent to the LLM during fallback: sample, rank, length, and whether it is full text. */
    public record RetrievedContext(long sourceDatasetId, int rank, int characterCount,
                                   boolean fullContext) {
        public RetrievedContext(long sourceDatasetId, int rank, int characterCount) {
            this(sourceDatasetId, rank, characterCount, true);
        }
    }

    /**
     * One path step: a graph edge plus its provenance, the core carrier of provenance awareness.
     *
     * <p>subject/target use the direction stored in the graph, while {@code traversedForward}
     * records the actual search direction: true means subject to target, and false means reverse.</p>
     *
     * @param subject            subject entity name
     * @param subjectType        subject entity type
     * @param subjectDatasetIds  source samples for the subject entity
     * @param relation           relation name
     * @param context            source evidence sentence used for relation extraction
     * @param relationDatasetIds source samples for this relation
     * @param target             target entity name
     * @param targetType         target entity type
     * @param targetDatasetIds   source samples for the target entity
     * @param traversedForward   whether this search followed the stored forward direction
     */
    public record PathStep(String subject, String subjectType,
                           List<Long> subjectDatasetIds,
                           String relation, String context,
                           List<Long> relationDatasetIds,
                           String target, String targetType,
                           List<Long> targetDatasetIds,
                           boolean traversedForward) {
        public PathStep {
            subjectDatasetIds = copy(subjectDatasetIds);
            relationDatasetIds = copy(relationDatasetIds);
            targetDatasetIds = copy(targetDatasetIds);
            context = context == null ? "" : context;
        }
    }

    /**
     * Complete reasoning result for one question, serialized as JSON into experiment_data.
     *
     * @param answer               final answer
     * @param answerSource         answer source: graph_subquestion / context_fallback / internal_knowledge
     *                             / unknown / retry / resumed
     * @param predictedTypes       answer types predicted by the LLM
     * @param recalledTypes        type-vector recall results
     * @param candidateTypeGraph   candidate type-graph paths
     * @param candidateAnswerTypes candidate answer types retained
     * @param paths                paths retained during search (each path is a sequence of PathStep objects)
     * @param exploredDepth        actual expansion depth
     * @param finished             whether a current path was judged sufficient to answer
     * @param fallbackDatasetId    primary source sample ID used by fallback
     * @param sourceVotes          provenance votes (data_set ID -> vote count)
     * @param fallbackCandidates   complete ranking evidence for fallback candidates
     * @param retrievedContexts    metadata for source contexts actually used
     * @param answerGrounding      record of grounding the answer to a sample entity name
     */
    public record Result(String answer, String answerSource,
                         List<PredictedType> predictedTypes,
                         List<RecalledType> recalledTypes,
                         List<Neo4jClient.EntityTypePathResult> candidateTypeGraph,
                         List<String> candidateAnswerTypes,
                         List<List<PathStep>> paths, int exploredDepth,
                         boolean finished, Long fallbackDatasetId,
                         Map<Long, Integer> sourceVotes,
                         List<FallbackCandidate> fallbackCandidates,
                         List<RetrievedContext> retrievedContexts,
                         AnswerGrounding answerGrounding) {
        public Result {
            // Normalize nulls to "unknown" and collections to immutable copies so callers cannot mutate results.
            answer = answer == null ? UNKNOWN_ANSWER : answer;
            answerSource = answerSource == null ? "unknown" : answerSource;
            predictedTypes = copy(predictedTypes);
            recalledTypes = copy(recalledTypes);
            candidateTypeGraph = copy(candidateTypeGraph);
            candidateAnswerTypes = copy(candidateAnswerTypes);
            paths = paths == null ? List.of() : paths.stream().map(OurMethodReasoning::copy).toList();
            sourceVotes = sourceVotes == null ? Map.of()
                    : Collections.unmodifiableMap(new LinkedHashMap<>(sourceVotes));
            fallbackCandidates = copy(fallbackCandidates);
            retrievedContexts = copy(retrievedContexts);
        }

        /** Return a copy with the answer replaced by the grounded entity name. */
        public Result groundedTo(String newAnswer, AnswerGrounding grounding) {
            return new Result(newAnswer, answerSource, predictedTypes, recalledTypes,
                    candidateTypeGraph, candidateAnswerTypes, paths, exploredDepth,
                    finished, fallbackDatasetId, sourceVotes, fallbackCandidates,
                    retrievedContexts, grounding);
        }

        /** Return a copy that records a failed grounding attempt while leaving the answer unchanged. */
        public Result withGroundingResult(AnswerGrounding grounding) {
            return new Result(answer, answerSource, predictedTypes, recalledTypes,
                    candidateTypeGraph, candidateAnswerTypes, paths, exploredDepth,
                    finished, fallbackDatasetId, sourceVotes, fallbackCandidates,
                    retrievedContexts, grounding);
        }

        /** Keep the full trace but replace only the final answer surface. */
        public Result withAnswer(String newAnswer) {
            return new Result(newAnswer, answerSource, predictedTypes,
                    recalledTypes, candidateTypeGraph, candidateAnswerTypes,
                    paths, exploredDepth, finished, fallbackDatasetId,
                    sourceVotes, fallbackCandidates, retrievedContexts,
                    answerGrounding);
        }

        /** Simplified compatibility constructor without fallback candidates, contexts, or grounding metadata. */
        public Result(String answer, String answerSource,
                      List<PredictedType> predictedTypes,
                      List<RecalledType> recalledTypes,
                      List<Neo4jClient.EntityTypePathResult> candidateTypeGraph,
                      List<String> candidateAnswerTypes,
                      List<List<PathStep>> paths, int exploredDepth,
                      boolean finished, Long fallbackDatasetId,
                      Map<Long, Integer> sourceVotes) {
            this(answer, answerSource, predictedTypes, recalledTypes, candidateTypeGraph,
                    candidateAnswerTypes, paths, exploredDepth, finished,
                    fallbackDatasetId, sourceVotes, List.of(), List.of(), null);
        }

        /** Simplified compatibility constructor with fallback candidates only. */
        public Result(String answer, String answerSource,
                      List<PredictedType> predictedTypes,
                      List<RecalledType> recalledTypes,
                      List<Neo4jClient.EntityTypePathResult> candidateTypeGraph,
                      List<String> candidateAnswerTypes,
                      List<List<PathStep>> paths, int exploredDepth,
                      boolean finished, Long fallbackDatasetId,
                      Map<Long, Integer> sourceVotes,
                      List<FallbackCandidate> fallbackCandidates) {
            this(answer, answerSource, predictedTypes, recalledTypes, candidateTypeGraph,
                    candidateAnswerTypes, paths, exploredDepth, finished,
                    fallbackDatasetId, sourceVotes, fallbackCandidates, List.of(), null);
        }

        /** Serialize the full reasoning trace to JSON for later analysis and manual auditing. */
        public String reasoningPathJson() {
            // First block: answer, type prediction/recall/type graph, and search metadata.
            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("answer", answer);
            payload.put("answer_source", answerSource);
            payload.put("predicted_types", predictedTypes);
            payload.put("recalled_types", recalledTypes);
            payload.put("candidate_type_graph", candidateTypeGraph);
            payload.put("candidate_answer_types", candidateAnswerTypes);
            payload.put("explored_depth", exploredDepth);
            payload.put("finished", finished);
            payload.put("fallback_dataset_id", fallbackDatasetId);
            payload.put("fallback_dataset_ids", fallbackCandidates.stream()
                    .map(FallbackCandidate::sourceDatasetId).toList());
            payload.put("fallback_candidates", fallbackCandidates);
            payload.put("retrieved_contexts", retrievedContexts);
            payload.put("source_votes", sourceVotes);
            payload.put("answer_grounding", answerGrounding == null ? Map.of() : answerGrounding);
            // Second block: expand each path into a ranked candidate with step-by-step details.
            List<Map<String, Object>> candidates = new ArrayList<>();
            for (int i = 0; i < paths.size(); i++) {
                Map<String, Object> candidate = new LinkedHashMap<>();
                candidate.put("rank", i + 1);
                candidate.put("steps", serializeSteps(paths.get(i)));
                candidates.add(candidate);
            }
            Map<String, Object> finalPath = Map.of();
            // Third block: when the answer comes from the graph, recover the path containing the final answer.
            // This final_path is the interpretable reasoning path presented in the paper.
            if (answerSource != null
                    && (answerSource.equals("graph")
                    || answerSource.startsWith("graph_"))) {
                String expected = normalizeAnswer(answer);
                for (int i = 0; i < paths.size(); i++) {
                    boolean containsAnswer = paths.get(i).stream().anyMatch(step ->
                            answerSurfaceMatches(expected,
                                    normalizeAnswer(step.subject()))
                                    || answerSurfaceMatches(expected,
                                    normalizeAnswer(step.target())));
                    if (containsAnswer) {
                        finalPath = candidates.get(i);
                        break;
                    }
                }
            }
            payload.put("final_path", finalPath);
            payload.put("candidate_paths", candidates);
            String serialized = JsonCodec.toJson(payload);
            if (serialized.getBytes(StandardCharsets.UTF_8).length <= 60_000) {
                return serialized;
            }
            Map<String, Object> summary = new LinkedHashMap<>();
            summary.put("answer", answer);
            summary.put("answer_source", answerSource);
            summary.put("explored_depth", exploredDepth);
            summary.put("finished", finished);
            summary.put("fallback_dataset_id", fallbackDatasetId);
            summary.put("predicted_types", predictedTypes);
            summary.put("candidate_answer_types", candidateAnswerTypes);
            summary.put("candidate_path_count", candidates.size());
            summary.put("reasoning_path_truncated", true);
            return JsonCodec.toJson(summary);
        }
    }

    /** All type-side artifacts shared by search, answering, and fallback as a type context. */
    private record TypeProfile(List<PredictedType> predicted,
                               List<RecalledType> recalled,
                               List<Neo4jClient.EntityTypePathResult> graph,
                               List<String> answerTypes,
                               int reasoningHops) { }

    /**
     * Records which entity in the sample graph the final answer was aligned to.
     * {@code applied=true} means the answer was rewritten and the entity name is persisted;
     * {@code applied=false} means a similar entity was found below threshold and the answer stayed unchanged.
     */
    public record AnswerGrounding(String entityId, String entityName, String entityType,
                                  double score, boolean applied) { }

    /**
     * One state on the search frontier (a beam-search node).
     * score is the cumulative score from the seed (a product of edge scores); path is the full path;
     * visitedEntityIds prevents cycles through the same entity.
     */
    private record State(String entityId, String entityName, String entityType,
                         double score, List<PathStep> path,
                         Set<String> visitedEntityIds) { }

    /** Generic value-and-score record used for edge/relation scoring results. */
    private record Scored(String value, double score) { }

    /** An entity and type that appeared on a path, used to align an LLM answer with a path entity. */
    private record AnswerEntity(String name, String type) { }
    /** Deterministic terminal candidate ranking used as a guard around graphAnswer. */
    private record RankedAnswer(AnswerEntity entity, double score, int depth) { }
    /** Graph-construction outcome for one sample (success status and failure reason) for batch aggregation. */
    private record BuildOutcome(DataItem item, boolean success, String error) { }
    /** Question-answering outcome for one sample. */
    private record QaOutcome(DataItem item, Result result) { }

    /**
     * Run complete reasoning for one question against a built graph.
     *
     * <p>The order is: answer-type prediction, topic seeds, subquestion graph search
     * with provenance fallback, and answer grounding.</p>
     *
     * @param contextResolver callback that resolves source context from a data_set row ID for fallback
     */
    public static Result runSample(LlmChatClient llm, Neo4jClient neo4j,
                                   String experimentId, String graphDatasetId,
                                   String question, LongFunction<String> contextResolver,
                                   Config config) {
        Objects.requireNonNull(llm, "llm");
        Objects.requireNonNull(neo4j, "neo4j");
        requireText(experimentId, "experimentId");
        requireText(graphDatasetId, "graphDatasetId");
        requireText(question, "question");
        Config cfg = config == null ? Config.defaults() : config;

        // Step 1: ask the LLM what type the answer should have, then map it to real graph types.
        TypeProfile types = resolveAnswerTypes(llm, neo4j, experimentId,
                graphDatasetId, question, cfg);
        // Step 2: extract topic entities from the question and retrieve graph seed nodes as search starts.
        List<Neo4jClient.EntitySearchResult> seeds = topicEntities(llm, neo4j,
                experimentId, graphDatasetId, question, cfg);
        // Step 3: execute TyHop's subquestion graph search.
        Result answered = searchWithSubquestions(llm, neo4j, experimentId,
                graphDatasetId, question, contextResolver, cfg, types, seeds);
        if (answered.answerSource() != null
                && answered.answerSource().startsWith("graph")) {
            Result fallbackAnswer = fallback(llm, neo4j, experimentId,
                    graphDatasetId, question, contextResolver, types,
                    answered.paths(), answered.exploredDepth(),
                    answered.sourceVotes(), cfg);
            if ("context_fallback".equals(fallbackAnswer.answerSource())
                    && !fallbackAnswer.answer().isBlank()
                    && !UNKNOWN_ANSWER.equalsIgnoreCase(
                    fallbackAnswer.answer())
                    && !RETRY_ANSWER.equalsIgnoreCase(
                    fallbackAnswer.answer())) {
                answered = fallbackAnswer;
            }
        }
        // Step 4: post-process the answer. Ground the entity name first, then apply one-hop
        // relation projection for explicit relational questions such as "type of" or "named after".
        Result grounded = groundAnswerOnSampleEntity(neo4j, experimentId,
                graphDatasetId, answered, cfg);
        grounded = enforceSubquestionAnswerType(llm, neo4j, experimentId,
                graphDatasetId, question, contextResolver, cfg, types,
                grounded);
        grounded = grounded.withAnswer(normalizeSubquestionAnswer(
                grounded.answer(), question));
        String projected = projectAnswerByQuestionRelation(
                neo4j, experimentId, graphDatasetId, question,
                grounded.answer());
        if (!projected.isBlank()
                && !normalizeAnswer(projected).equals(
                normalizeAnswer(grounded.answer()))) {
            return grounded.withAnswer(projected);
        }
        return grounded;
    }

    private static Result enforceSubquestionAnswerType(
            LlmChatClient llm,
            Neo4jClient neo4j,
            String experimentId,
            String graphDatasetId,
            String question,
            LongFunction<String> contextResolver,
            Config cfg,
            TypeProfile types,
            Result result) {
        if (result == null
                || !"graph_subquestion".equals(result.answerSource())
                || types == null || types.answerTypes().isEmpty()
                || result.answer().isBlank()
                || UNKNOWN_ANSWER.equalsIgnoreCase(result.answer())) {
            return result;
        }
        List<Neo4jClient.EntitySearchResult> hits;
        try {
            hits = neo4j.searchEntitiesByName(experimentId, graphDatasetId,
                    result.answer(), 5);
        } catch (RuntimeException ignored) {
            return result;
        }
        if (hits.isEmpty()) return result;
        for (Neo4jClient.EntitySearchResult hit : hits) {
            if (hit.type() == null || hit.type().isBlank()) continue;
            if (compatibleAnswerType(types.answerTypes(), hit.type())
                    || neo4j.areEntityTypesCompatible(experimentId,
                    graphDatasetId, hit.type(), types.answerTypes(),
                    cfg.typeGraphHops())) {
                return result;
            }
        }
        Result recovered = fallback(llm, neo4j, experimentId, graphDatasetId,
                question, contextResolver, types, result.paths(),
                result.exploredDepth(), result.sourceVotes(), cfg);
        if (recovered != null && recovered.answer() != null
                && !recovered.answer().isBlank()
                && !UNKNOWN_ANSWER.equalsIgnoreCase(recovered.answer())
                && !RETRY_ANSWER.equalsIgnoreCase(recovered.answer())) {
            return recovered;
        }
        return result;
    }

    /**
     * Subquestion-guided graph reasoning.
     *
     * <p>The model first plans reachable subquestions. For one subquestion at a
     * time, the method enumerates bounded four-hop paths from the current
     * entity, shows them as entity--relation-->entity steps, and lets the LLM
     * select the next hop, resolve the subquestion, or answer directly. Selected
     * edges are carried into the next round; no global path plan is required.</p>
     */
    private static Result searchWithSubquestions(
            LlmChatClient llm,
            Neo4jClient neo4j,
            String experimentId,
            String graphDatasetId,
            String question,
            LongFunction<String> contextResolver,
            Config cfg,
            TypeProfile types,
            List<Neo4jClient.EntitySearchResult> seeds) {
        System.out.printf("ours-subquestion start seeds=%d question=%s%n",
                seeds == null ? 0 : seeds.size(), truncate(question, 120));
        EvidenceCollector evidence = new EvidenceCollector();
        List<Neo4jClient.EntitySearchResult> startEntities = new ArrayList<>(seeds);
        for (Neo4jClient.EntitySearchResult seed : seeds) {
            evidence.recordEntity(seed.entityId(), neo4j.queryEntitySourceDatasetIds(
                    experimentId, graphDatasetId, seed.entityId()));
        }

            List<Subquestion> subquestions = planSubquestions(llm, question, types);
            System.out.printf("ours-subquestion plan subquestions=%d %s%n",
                    subquestions.size(),
                    subquestions.stream()
                            .map(item -> item.question() + " ["
                                    + item.targetType() + "; start="
                                    + (item.startHint().isBlank()
                                    ? "<empty>" : item.startHint()) + "]")
                            .toList());
        if (subquestions.isEmpty()) {
            subquestions = List.of(new Subquestion(question, "", ""));
        }
        List<List<PathStep>> selectedPaths = new ArrayList<>();
        List<ResolvedValue> resolvedValues = new ArrayList<>();
        Map<String, LocalPath> candidatePool = new LinkedHashMap<>();
        List<Neo4jClient.EntitySearchResult> currentEntities = startEntities;
        int exploredDepth = 0;
        for (int subquestionIndex = 0;
             subquestionIndex < subquestions.size();
             subquestionIndex++) {
            Subquestion subquestion = subquestions.get(subquestionIndex);
            boolean finalSubquestion =
                    subquestionIndex == subquestions.size() - 1;
            List<Neo4jClient.EntitySearchResult> roundStarts =
                    subquestionStarts(llm, neo4j, experimentId, graphDatasetId,
                            subquestion, currentEntities, seeds, cfg);
            if (roundStarts.isEmpty()) {
                continue;
            }
            List<LocalPath> enumeratedPaths = new ArrayList<>();
            int perStartLimit = Math.max(32,
                    SUBQUESTION_ENUM_LIMIT / Math.max(1, roundStarts.size()));
            int maxHops = 4;
            for (Neo4jClient.EntitySearchResult start : roundStarts) {
                enumeratedPaths.addAll(enumerateLocalPaths(neo4j, experimentId,
                        graphDatasetId, start, cfg, perStartLimit, maxHops));
                if (enumeratedPaths.size() >= SUBQUESTION_ENUM_LIMIT) {
                    break;
                }
            }
            List<LocalPath> localPaths = new ArrayList<>();
            for (int pathIndex = 0; pathIndex < enumeratedPaths.size(); pathIndex++) {
                LocalPath path = enumeratedPaths.get(pathIndex);
                localPaths.add(new LocalPath(pathIndex, path.steps(),
                        path.edges(), path.terminalEntityId(),
                        path.terminalEntityName(), path.terminalEntityType()));
            }
            localPaths = localPaths.stream()
                    .sorted(Comparator.comparingDouble(
                            (LocalPath path) -> pathRelevance(subquestion, path))
                            .reversed()
                            .thenComparingInt(LocalPath::id))
                    .limit(subquestionPathLimit())
                    .toList();
            localPaths = preferDirectAttributePaths(subquestion, localPaths);
            if (localPaths.isEmpty()
                    && asksDirectAttribute(subquestion.question())) {
                System.out.printf(
                        "ours-subq direct-attribute-miss q=%s; using fallback%n",
                        truncate(subquestion.question(), 120));
                continue;
            }
            for (LocalPath path : localPaths) {
                String key = normalizeAnswer(path.terminalEntityName());
                LocalPath previous = candidatePool.get(key);
                if (previous == null
                        || pathRelevance(subquestion, path)
                        > pathRelevance(subquestion, previous)) {
                    candidatePool.put(key, path);
                }
            }
            HopDecision decision = selectSubquestionHop(llm, question,
                    subquestion, roundStarts, localPaths);
            System.out.printf(
                    "ours-subq decision selected=%s confidence=%.3f "
                            + "resolved=%s answer=%s reason=%s top_paths=%s%n",
                    decision.selectedPathIds(), decision.confidence(),
                    decision.resolved(), decision.answer(),
                    truncate(decision.reason(), 160),
                    localPaths.stream().limit(3)
                            .map(path -> path.id() + ":"
                                    + truncate(path.terminalEntityName(), 60))
                            .toList());

            Map<Integer, LocalPath> pathsById = new LinkedHashMap<>();
            for (LocalPath path : localPaths) {
                pathsById.put(path.id(), path);
            }
            List<LocalPath> chosen = decision.selectedPathIds().stream()
                    .map(pathsById::get)
                    .filter(Objects::nonNull)
                    .toList();
            System.out.printf(
                    "ours-subq chosen_terminals=%s accumulated_paths=%d%n",
                    chosen.stream().map(path -> path.id() + ":"
                            + path.terminalEntityName()).toList(),
                    selectedPaths.size());
            if (!decision.answer().isBlank()
                    && !UNKNOWN_ANSWER.equalsIgnoreCase(decision.answer())) {
                if (!finalSubquestion) {
                    for (LocalPath path : chosen) {
                        selectedPaths.add(path.steps());
                        exploredDepth = Math.max(exploredDepth,
                                path.steps().size());
                        for (Neo4jClient.IncidentRelationEvidence edge :
                                path.edges()) {
                            evidence.record(edge);
                        }
                    }
                    currentEntities = resolveSubquestionAnswer(
                            neo4j, experimentId, graphDatasetId,
                            decision.answer(), "",
                            chosenOrFirst(localPaths,
                                    decision.selectedPathIds()), cfg);
                    addResolvedValue(resolvedValues, subquestion.question(),
                            decision.answer());
                    System.out.printf(
                            "ours-subq intermediate q=%d/%d answer=%s "
                                    + "next_entities=%d%n",
                            subquestionIndex + 1, subquestions.size(),
                            decision.answer(), currentEntities.size());
                    continue;
                }
                for (LocalPath path : chosen) {
                    selectedPaths.add(path.steps());
                    exploredDepth = Math.max(exploredDepth,
                            path.steps().size());
                    for (Neo4jClient.IncidentRelationEvidence edge :
                            path.edges()) {
                        evidence.record(edge);
                    }
                }
                AnswerEntity verified = graphAnswer(llm, neo4j,
                        experimentId, graphDatasetId, question, types,
                        selectedPaths, cfg.typeGraphHops(), true);
                if (verified != null) {
                    return result(verified.name(), "graph_subquestion",
                            types, selectedPaths, exploredDepth, true,
                            null, evidence.votes());
                }
                String directAnswer = normalizeSubquestionAnswer(
                        decision.answer(), question);
                return groundAnswerOnSampleEntity(neo4j, experimentId,
                        graphDatasetId,
                        result(directAnswer,
                                "graph_subquestion", types,
                                selectedPaths, exploredDepth, true, null,
                                evidence.votes()),
                        cfg);
            }

            if (finalSubquestion && decision.resolved() && !chosen.isEmpty()) {
                List<List<PathStep>> finalPaths = new ArrayList<>(selectedPaths);
                LocalPath finalPath = chosen.get(0);
                finalPaths.add(finalPath.steps());
                for (Neo4jClient.IncidentRelationEvidence edge : finalPath.edges()) {
                    evidence.record(edge);
                }
                AnswerEntity verified = graphAnswer(llm, neo4j,
                        experimentId, graphDatasetId, question, types,
                        finalPaths, cfg.typeGraphHops(), true);
                if (verified != null) {
                    return result(verified.name(), "graph_subquestion",
                            types, finalPaths,
                            Math.max(exploredDepth, finalPath.steps().size()),
                            true, null, evidence.votes());
                }
                String terminal = normalizeSubquestionAnswer(
                        finalPath.terminalEntityName(), question);
                return groundAnswerOnSampleEntity(neo4j, experimentId,
                        graphDatasetId,
                        result(terminal,
                                "graph_subquestion", types, finalPaths,
                                Math.max(exploredDepth, finalPath.steps().size()),
                                true, null, evidence.votes()),
                        cfg);
            }
            if (chosen.isEmpty() && decision.resolved()) {
                continue;
            }
            if (chosen.isEmpty() && !localPaths.isEmpty()) {
                chosen = List.of(localPaths.get(0));
            }
            if (chosen.isEmpty()) {
                continue;
            }
            currentEntities = new ArrayList<>();
            for (LocalPath path : chosen) {
                selectedPaths.add(path.steps());
                exploredDepth = Math.max(exploredDepth, path.steps().size());
                for (Neo4jClient.IncidentRelationEvidence edge : path.edges()) {
                    evidence.record(edge);
                }
                addResolvedValue(resolvedValues, subquestion.question(),
                        path.terminalEntityName());
                if (path.terminalEntityId() != null
                        && !path.terminalEntityId().isBlank()) {
                    currentEntities.add(new Neo4jClient.EntitySearchResult(
                            path.terminalEntityId(), path.terminalEntityName(),
                            path.terminalEntityType(), experimentId,
                            graphDatasetId, 1.0));
                }
            }
            System.out.printf(
                    "ours-subq q=%d/%d selected=%d paths=%d depth=%d answer=%s question=%s%n",
                    subquestionIndex, subquestions.size(), chosen.size(),
                    localPaths.size(), pathDepth(chosen), decision.answer(),
                    truncate(subquestion.question(), 120));
        }

        if (!selectedPaths.isEmpty()) {
            AnswerEntity answer = graphAnswer(llm, neo4j, experimentId,
                    graphDatasetId, question, types, selectedPaths,
                    cfg.typeGraphHops(), true);
            if (answer != null) {
                return result(answer.name(), "graph_subquestion", types,
                        selectedPaths, exploredDepth, true, null,
                        evidence.votes());
            }
        }
        return fallback(llm, neo4j, experimentId, graphDatasetId, question,
                contextResolver, types, selectedPaths, exploredDepth,
                evidence.votes(), cfg);
    }

    private static List<Subquestion> planSubquestions(
            LlmChatClient llm,
            String question,
            TypeProfile types) {
        Map<String, Object> input = new LinkedHashMap<>();
        input.put("question", question);
        input.put("answer_types", types.answerTypes());
        String raw = llm.chatWithSystemPrompt(
                "ours-subquestion-plan-" + UUID.randomUUID(), "",
                String.format(OursSubquestionPlan,
                        JsonCodec.toJson(input)));
        List<Subquestion> result = new ArrayList<>();
        try {
            JsonArray values = parseObject(raw).getAsJsonArray("subquestions");
            if (values == null) return List.of();
            for (JsonElement element : values) {
                if (!element.isJsonObject()) continue;
                JsonObject item = element.getAsJsonObject();
                String subquestion = string(item, "question");
                if (subquestion.isBlank()) continue;
                result.add(new Subquestion(subquestion,
                        string(item, "start_hint"),
                        string(item, "target_type")));
                if (result.size() >= 4) break;
            }
        } catch (RuntimeException ignored) {
            return List.of();
        }
        return List.copyOf(result);
    }

    private static List<Neo4jClient.EntitySearchResult> subquestionStarts(
            LlmChatClient llm,
            Neo4jClient neo4j,
            String experimentId,
            String graphDatasetId,
            Subquestion subquestion,
            List<Neo4jClient.EntitySearchResult> carried,
            List<Neo4jClient.EntitySearchResult> seeds,
            Config cfg) {
        LinkedHashMap<String, Neo4jClient.EntitySearchResult> result =
                new LinkedHashMap<>();
        String hint = subquestion.startHint() == null
                ? "" : subquestion.startHint().trim();
        boolean previous = "PREVIOUS_ANSWER".equalsIgnoreCase(hint)
                || (hint.isBlank()
                && subquestionReferencesPreviousLegacy(subquestion.question()));
        if (previous && carried != null) {
            for (Neo4jClient.EntitySearchResult item : carried) {
                if (item != null && item.entityId() != null) {
                    result.putIfAbsent(item.entityId(), item);
                }
            }
        }
        if (!hint.isBlank() && !previous) {
            try {
                List<Neo4jClient.EntitySearchResult> lexical =
                        neo4j.searchEntitiesByName(experimentId,
                                graphDatasetId, hint, 3);
                List<Neo4jClient.EntitySearchResult> vector =
                        neo4j.searchEntities(experimentId,
                                graphDatasetId, hint, 3);
                for (Neo4jClient.EntitySearchResult hit : lexical) {
                    if (normalizeAnswer(hit.name()).equals(
                            normalizeAnswer(hint))) {
                        result.putIfAbsent(hit.entityId(), hit);
                    }
                }
                for (Neo4jClient.EntitySearchResult hit : vector) {
                    result.putIfAbsent(hit.entityId(), hit);
                }
                for (Neo4jClient.EntitySearchResult hit : lexical) {
                    result.putIfAbsent(hit.entityId(), hit);
                }
            } catch (RuntimeException ignored) {
                // Fall through to vector search, topic extraction, or carried context.
            }
        }
        if (!hint.isBlank() && previous) {
            try {
                List<Neo4jClient.EntitySearchResult> exact =
                        neo4j.searchEntitiesByName(experimentId,
                                graphDatasetId, hint, 3);
                for (Neo4jClient.EntitySearchResult hit : exact) {
                    if (normalizeAnswer(hit.name()).equals(
                            normalizeAnswer(hint))) {
                        result.putIfAbsent(hit.entityId(), hit);
                    }
                }
            } catch (RuntimeException ignored) {
                // The carried entity remains the primary start.
            }
        }
        if (result.isEmpty() && !previous) {
            for (Neo4jClient.EntitySearchResult hit : topicEntities(llm, neo4j,
                    experimentId, graphDatasetId, subquestion.question(), cfg)) {
                result.putIfAbsent(hit.entityId(), hit);
            }
        }
        if (result.isEmpty() && carried != null) {
            for (Neo4jClient.EntitySearchResult item : carried) {
                if (item != null && item.entityId() != null) {
                    result.putIfAbsent(item.entityId(), item);
                }
            }
        }
        if (result.isEmpty()) {
            for (Neo4jClient.EntitySearchResult hit : topicEntities(llm, neo4j,
                    experimentId, graphDatasetId, subquestion.question(), cfg)) {
                result.putIfAbsent(hit.entityId(), hit);
            }
        }
        if (result.isEmpty() && seeds != null) {
            for (Neo4jClient.EntitySearchResult seed : seeds) {
                result.putIfAbsent(seed.entityId(), seed);
            }
        }
        return result.values().stream()
                .limit(cfg.maxTopicEntities())
                .toList();
    }

    private static boolean subquestionReferencesPreviousLegacy(String question) {
        String value = normalize(question);
        return value.contains("[")
                || value.contains("previous_answer")
                || value.contains("previous answer")
                || value.contains("aforementioned")
                || value.contains("identified ")
                || value.contains("the band")
                || value.contains("the team")
                || value.contains("the movie")
                || value.contains("the film")
                || value.contains("the university")
                || value.contains("the organization")
                || value.contains("the company")
                || value.contains("the production")
                || value.contains("the season")
                || value.contains("the location")
                || value.contains("the city")
                || value.contains("the county")
                || value.contains("the state");
    }

    private static List<LocalPath> chosenOrFirst(
            List<LocalPath> localPaths,
            List<Integer> selectedPathIds) {
        if (localPaths == null || localPaths.isEmpty()) return List.of();
        Map<Integer, LocalPath> byId = new LinkedHashMap<>();
        for (LocalPath path : localPaths) {
            byId.put(path.id(), path);
        }
        List<LocalPath> selected = selectedPathIds == null
                ? List.of()
                : selectedPathIds.stream()
                .map(byId::get)
                .filter(Objects::nonNull)
                .toList();
        return selected.isEmpty() ? List.of(localPaths.get(0)) : selected;
    }

    private static List<Neo4jClient.EntitySearchResult> resolveSubquestionAnswer(
            Neo4jClient neo4j,
            String experimentId,
            String graphDatasetId,
            String answer,
            String targetType,
            List<LocalPath> fallbackPaths,
            Config cfg) {
        if (answer == null || answer.isBlank()) return List.of();
        try {
            List<Neo4jClient.EntitySearchResult> hits =
                    neo4j.searchEntitiesByName(experimentId, graphDatasetId,
                            answer, cfg.maxTopicEntities());
            if (!hits.isEmpty()) {
                List<Neo4jClient.EntitySearchResult> typed = hits.stream()
                        .filter(hit -> typeMatches(targetType, hit.type()))
                        .limit(cfg.maxTopicEntities())
                        .toList();
                if (!typed.isEmpty()) return List.of(typed.get(0));
                return List.of(hits.get(0));
            }
        } catch (RuntimeException ignored) {
            // Fall through to the selected path terminal.
        }
        if (fallbackPaths == null || fallbackPaths.isEmpty()) return List.of();
        LocalPath path = fallbackPaths.get(0);
        if (path.terminalEntityId() == null
                || path.terminalEntityId().isBlank()) {
            return List.of();
        }
        if (!typeMatches(targetType, path.terminalEntityType())) {
            return List.of();
        }
        return List.of(new Neo4jClient.EntitySearchResult(
                path.terminalEntityId(), path.terminalEntityName(),
                path.terminalEntityType(), experimentId, graphDatasetId, 1.0));
    }

    private static void addResolvedValue(List<ResolvedValue> values,
                                         String subquestion, String value) {
        if (value == null || value.isBlank()) return;
        String normalized = normalizeAnswer(value);
        for (ResolvedValue existing : values) {
            if (normalizeAnswer(existing.answer()).equals(normalized)
                    && normalize(existing.subquestion()).equals(
                    normalize(subquestion))) {
                return;
            }
        }
        values.add(new ResolvedValue(subquestion, value.trim()));
    }

    private static List<LocalPath> enumerateLocalPaths(
            Neo4jClient neo4j,
            String experimentId,
            String graphDatasetId,
            Neo4jClient.EntitySearchResult start,
            Config cfg,
            int limit,
            int maxHops) {
        if (limit < 1) {
            throw new IllegalArgumentException("limit must be positive");
        }
        if (maxHops < 1) {
            throw new IllegalArgumentException("maxHops must be positive");
        }
        List<LocalPath> result = new ArrayList<>();
        LinkedHashMap<String, List<Neo4jClient.IncidentRelationEvidence>> cache =
                new LinkedHashMap<>();
        record Frame(String currentId, List<PathStep> steps,
                     List<Neo4jClient.IncidentRelationEvidence> edges,
                     Set<String> visited) { }
        List<Frame> stack = new ArrayList<>();
        stack.add(new Frame(start.entityId(), List.of(), List.of(),
                Set.of(start.entityId())));
        int nextId = 0;
        while (!stack.isEmpty() && result.size() < limit) {
            Frame frame = stack.remove(stack.size() - 1);
            if (frame.steps().size() >= maxHops) continue;
            List<Neo4jClient.IncidentRelationEvidence> incident =
                    cache.computeIfAbsent(frame.currentId(), ignored ->
                            neo4j.queryIncidentRelationEvidence(experimentId,
                                    graphDatasetId, frame.currentId(),
                                    cfg.maxIncidentRelations()));
            for (Neo4jClient.IncidentRelationEvidence edge :
                    edgeCandidates(frame.currentId(), frame.visited(), incident,
                            edgeCandidateLimit(cfg))) {
                boolean forward = edge.subjectEntityId().equals(frame.currentId());
                String nextIdValue = forward
                        ? edge.targetEntityId() : edge.subjectEntityId();
                if (frame.visited().contains(nextIdValue)) continue;
                List<PathStep> steps = new ArrayList<>(frame.steps());
                steps.add(toPathStep(edge, forward, cfg.maxEdgeContextChars()));
                List<Neo4jClient.IncidentRelationEvidence> edges =
                        new ArrayList<>(frame.edges());
                edges.add(edge);
                LinkedHashSet<String> visited =
                        new LinkedHashSet<>(frame.visited());
                visited.add(nextIdValue);
                String terminalName = forward ? edge.targetName() : edge.subjectName();
                String terminalType = forward ? edge.targetType() : edge.subjectType();
                result.add(new LocalPath(nextId++, List.copyOf(steps),
                        List.copyOf(edges), nextIdValue, terminalName, terminalType));
                if (result.size() >= limit) break;
                stack.add(new Frame(nextIdValue, List.copyOf(steps),
                        List.copyOf(edges), Set.copyOf(visited)));
            }
        }
        return List.copyOf(result);
    }

    /**
     * Keep one shortest path to every reachable entity instead of filling the
     * candidate budget with the first depth-first branches.
     */
    private static HopDecision selectSubquestionHop(
            LlmChatClient llm,
            String originalQuestion,
            Subquestion subquestion,
            List<Neo4jClient.EntitySearchResult> starts,
            List<LocalPath> paths) {
        Map<String, Object> input = new LinkedHashMap<>();
        input.put("original_question", originalQuestion);
        input.put("subquestion", materializeSubquestion(
                subquestion.question(), starts));
        input.put("start_entities", starts.stream().map(item -> Map.of(
                "name", item.name())).toList());
        List<Map<String, Object>> serializedPaths = new ArrayList<>();
        for (LocalPath path : paths) {
            Map<String, Object> item = new LinkedHashMap<>();
            item.put("path_id", path.id());
            item.put("steps", compactSubquestionSteps(path.steps()));
            serializedPaths.add(item);
        }
        input.put("candidate_paths", serializedPaths);
        String raw = llm.chatWithSystemPrompt(
                "ours-subquestion-hop-" + UUID.randomUUID(), "",
                String.format(OursHopSelect, JsonCodec.toJson(input)));
        try {
            JsonObject parsed = parseObject(raw);
            List<Integer> selected = new ArrayList<>();
            JsonArray ids = parsed.getAsJsonArray("selected_path_ids");
            if (ids != null) {
                for (JsonElement id : ids) {
                    if (id.isJsonPrimitive()) selected.add(id.getAsInt());
                }
            }
            return new HopDecision(selected, string(parsed, "answer"),
                    parsed.has("resolved")
                            && parsed.get("resolved").getAsBoolean(),
                    score(parsed, "confidence"),
                    string(parsed, "reason"));
        } catch (RuntimeException ignored) {
            return new HopDecision(List.of(), "", false, 0.0,
                    "invalid hop-selection output");
        }
    }

    static List<Map<String, Object>> compactSubquestionSteps(
            List<PathStep> steps) {
        List<Map<String, Object>> result = new ArrayList<>();
        for (PathStep step : steps) {
            Map<String, Object> item = new LinkedHashMap<>();
            item.put("from", step.traversedForward()
                    ? step.subject() : step.target());
            item.put("relation", step.relation());
            item.put("to", step.traversedForward()
                    ? step.target() : step.subject());
            result.add(item);
        }
        return result;
    }

    /** Replace planner placeholders with the actual entity names for the current round. */
    static String materializeSubquestion(
            String question,
            List<Neo4jClient.EntitySearchResult> starts) {
        if (question == null || question.isBlank()) return "";
        if (!question.contains("PREVIOUS_ANSWER")) return question;
        String replacement = starts == null ? "" : starts.stream()
                .map(Neo4jClient.EntitySearchResult::name)
                .filter(name -> name != null && !name.isBlank())
                .distinct()
                .limit(5)
                .reduce((left, right) -> left + " or " + right)
                .orElse("the previous answer");
        return question.replace("PREVIOUS_ANSWER", replacement);
    }

    private static double pathRelevance(Subquestion subquestion, LocalPath path) {
        return legacyPathRelevance(subquestion, path);
    }

    private static double legacyPathRelevance(
            Subquestion subquestion,
            LocalPath path) {
        LinkedHashSet<String> query = meaningfulTerms(subquestion.question());
        LinkedHashSet<String> pathTerms = meaningfulTerms(
                path.steps().stream().map(step ->
                        step.subject() + " " + step.relation() + " "
                                + step.target() + " " + step.context())
                        .reduce("", (left, right) -> left + " " + right));
        if (query.isEmpty() || pathTerms.isEmpty()) return 0.0;
        int overlap = 0;
        for (String term : query) {
            if (pathTerms.contains(term)) overlap++;
        }
        return overlap / (double) query.size()
                + (typeMatches(subquestion.targetType(), path.terminalEntityType())
                ? 1.0 : 0.0)
                - 0.002 * path.steps().size();
    }

    /**
     * Prefer a direct one-hop attribute relation when the subquestion asks
     * for that attribute. This prevents a path such as
     * Richard -> partnered_with -> Tatiana -> birth_date from competing with
     * Richard -> birth_date for "Richard's birth date".
     */
    static List<LocalPath> preferDirectAttributePaths(
            Subquestion subquestion,
            List<LocalPath> paths) {
        if (subquestion == null || paths == null || paths.isEmpty()) {
            return List.of();
        }
        if (!asksDirectAttribute(subquestion.question())) {
            return paths;
        }
        List<LocalPath> direct = paths.stream()
                .filter(path -> pathMatchesDirectAttribute(
                        subquestion.question(), path))
                .toList();
        return direct;
    }

    private static boolean pathMatchesDirectAttribute(
            String question,
            LocalPath path) {
        if (path == null || path.steps().size() != 1) return false;
        PathStep step = path.steps().get(0);
        return relationMatchesAskedAttribute(question, step.relation())
                || attributeTypeMatches(question, path.terminalEntityType());
    }

    static boolean attributeTypeMatches(String question, String type) {
        String query = normalize(question);
        String value = normalize(type);
        if (value.isBlank()) return false;
        if (query.contains("birth") || query.contains("born")
                || query.contains("date of birth")) {
            return value.contains("date");
        }
        if (query.contains("ranking")
                || query.contains("what rank")
                || query.contains("which rank")) {
            return value.contains("ranking")
                    || value.contains("rank");
        }
        if (query.contains("subject")) {
            return value.contains("subject")
                    || value.contains("topic");
        }
        if (query.contains("father") || query.contains("mother")
                || query.contains("parent")) {
            return value.contains("person");
        }
        if (query.contains("married") || query.contains("spouse")) {
            return value.contains("person");
        }
        if (query.contains("what type") || query.contains("which type")
                || query.contains("type of") || query.contains("what kind")
                || query.contains("which kind") || query.contains("kind of")) {
            return value.contains("type")
                    || value.contains("category")
                    || value.contains("organization_type");
        }
        if (query.contains("founded") || query.contains("founding")) {
            return value.contains("year") || value.contains("date");
        }
        if (asksLocationAttribute(query)) {
            return value.contains("state")
                    || value.contains("county")
                    || value.contains("region")
                    || value.contains("location")
                    || value.contains("city");
        }
        return false;
    }

    static boolean asksDirectAttribute(String question) {
        String query = normalize(question);
        return query.contains("birth")
                || query.contains("born")
                || query.contains("date of birth")
                || query.contains("ranking")
                || query.contains("what rank")
                || query.contains("which rank")
                || query.contains("subject")
                || query.contains("father")
                || query.contains("mother")
                || query.contains("parent")
                || query.contains("married")
                || query.contains("spouse")
                || query.contains("what type")
                || query.contains("which type")
                || query.contains("type of")
                || query.contains("what kind")
                || query.contains("which kind")
                || query.contains("kind of")
                || query.contains("founded")
                || query.contains("founding")
                || query.contains("starred")
                || query.contains("starring")
                || query.contains("named after")
                || query.contains("named for")
                || asksLocationAttribute(query);
    }

    private static boolean asksLocationAttribute(String query) {
        return query.contains("what state")
                || query.contains("which state")
                || query.contains("what county")
                || query.contains("which county")
                || query.contains("what region")
                || query.contains("which region")
                || query.contains("state is")
                || query.contains("county is")
                || query.contains("region is");
    }

    static boolean relationMatchesAskedAttribute(
            String question,
            String relation) {
        String query = normalize(question);
        String relationName = normalize(relation).replace('_', ' ');
        if (!asksDirectAttribute(query)) return false;
        if (query.contains("birth") || query.contains("born")
                || query.contains("date of birth")) {
            return relationName.contains("birth")
                    || relationName.contains("born")
                    || relationName.contains("date");
        }
        if (query.contains("ranking")
                || query.contains("what rank")
                || query.contains("which rank")) {
            return relationName.contains("ranking")
                    || relationName.contains("rank");
        }
        if (query.contains("subject")) {
            return relationName.contains("subject");
        }
        if (query.contains("father") || query.contains("mother")
                || query.contains("parent")) {
            return relationName.contains("father")
                    || relationName.contains("mother")
                    || relationName.contains("parent");
        }
        if (query.contains("married") || query.contains("spouse")) {
            return relationName.contains("married")
                    || relationName.contains("spouse");
        }
        if (query.contains("what type") || query.contains("which type")
                || query.contains("type of") || query.contains("what kind")
                || query.contains("which kind") || query.contains("kind of")) {
            return relationName.contains("type")
                    || relationName.contains("kind")
                    || relationName.contains("is a")
                    || relationName.contains("subclass");
        }
        if (query.contains("founded") || query.contains("founding")) {
            return relationName.contains("founded")
                    || relationName.contains("founding");
        }
        if (query.contains("starred") || query.contains("starring")) {
            return relationName.contains("star")
                    || relationName.contains("cast");
        }
        if (query.contains("named after") || query.contains("named for")) {
            return relationName.contains("named");
        }
        if (asksLocationAttribute(query)) {
            return relationName.contains("located")
                    || relationName.contains("origin")
                    || relationName.contains("state")
                    || relationName.contains("county")
                    || relationName.contains("region");
        }
        return false;
    }

    private static boolean genericType(String type) {
        String value = normalize(type);
        return value.isBlank() || Set.of("entity", "thing", "other",
                "unknown", "value").contains(value);
    }

    private static boolean typeMatches(String expected, String actual) {
        if (genericType(expected) || actual == null || actual.isBlank()) {
            return true;
        }
        String expectedValue = normalize(expected);
        String actualValue = normalize(actual);
        return expectedValue.equals(actualValue)
                || expectedValue.contains(actualValue)
                || actualValue.contains(expectedValue);
    }

    private static boolean compatibleAnswerType(
            List<String> expectedTypes,
            String actualType) {
        if (expectedTypes == null || expectedTypes.isEmpty()) return true;
        for (String expected : expectedTypes) {
            if (typeMatches(expected, actualType)) return true;
        }
        return false;
    }

    private static boolean answerSurfaceMatches(String expected, String actual) {
        if (expected == null || expected.isBlank()
                || actual == null || actual.isBlank()) {
            return false;
        }
        String expectedKey = expected.replaceAll("[^a-z0-9]+", " ").trim();
        String actualKey = actual.replaceAll("[^a-z0-9]+", " ").trim();
        return expectedKey.equals(actualKey)
                || actualKey.startsWith(expectedKey + " ")
                || expectedKey.startsWith(actualKey + " ");
    }

    static String normalizeSubquestionAnswer(
            String answer,
            String question) {
        if (answer == null || answer.isBlank()) return "";
        String value = answer.trim();
        String questionValue = normalize(question);
        if ((questionValue.contains("city") || questionValue.contains("town"))
                && value.contains(",")) {
            String head = value.substring(0, value.indexOf(',')).trim();
            if (!head.isBlank()) return head;
        }
        if (questionValue.contains("state") && value.contains(",")) {
            String tail = value.substring(value.lastIndexOf(',') + 1).trim();
            if (!tail.isBlank()) return tail;
        }
        if (questionValue.contains("river")
                && normalize(value).startsWith("river ")) {
            String stripped = value.replaceFirst("(?i)^river\\s+", "").trim();
            if (!stripped.isBlank()) return stripped;
        }
        if (questionValue.contains("university")
                && value.matches("(?i)[a-z][a-z0-9.'-]*(?:\\s+[a-z][a-z0-9.'-]*)*\\s+University")) {
            String stripped = value.replaceFirst(
                    "(?i)\\s+University$", "").trim();
            if (!stripped.isBlank()) return stripped;
        }
        if (questionValue.contains("region of")) {
            Matcher region = Pattern.compile(
                    "region of\\s+([a-z][a-z .'-]*?)(?=\\s+is\\b|\\s+contains\\b|$)")
                    .matcher(questionValue);
            if (region.find()) {
                String area = region.group(1).trim();
                String lower = normalize(value);
                if (!area.isBlank() && lower.endsWith(" " + area)) {
                    String stripped = value.substring(0,
                            value.length() - area.length()).trim();
                    if (!stripped.isBlank()) return stripped;
                }
            }
        }
        if (questionValue.contains("family")
                && !normalize(value).contains("family")) {
            return value + " family";
        }
        return value;
    }

    private static String questionTypeHint(String question) {
        String value = normalize(question);
        for (String type : List.of("state", "county", "region", "country",
                "city", "town", "university", "film", "band", "person")) {
            if (value.contains(type)) return type;
        }
        return "";
    }

    /**
     * Deterministic one-hop projection for explicit relation questions.
     * Uses only entity names, relation names, and endpoint types; no LLM call.
     */
    private static String projectAnswerByQuestionRelation(
            Neo4jClient neo4j,
            String experimentId,
            String graphDatasetId,
            String question,
            String answer) {
        if (answer == null || answer.isBlank()
                || UNKNOWN_ANSWER.equalsIgnoreCase(answer)) {
            return answer;
        }
        String questionValue = normalize(question);
        boolean asksType = questionValue.contains("kind of")
                || questionValue.contains("type of")
                || questionValue.contains("what type")
                || questionValue.contains("what kind");
        boolean asksNamedAfter = questionValue.contains("named after")
                || questionValue.contains("named for");
        String expectedType = questionTypeHint(question);
        if (!asksType && !asksNamedAfter && expectedType.isBlank()) {
            return answer;
        }
        try {
            List<Neo4jClient.EntitySearchResult> roots =
                    neo4j.searchEntitiesByName(experimentId, graphDatasetId,
                            answer, 10);
            if (roots.isEmpty()) return answer;
            Map<String, Scored> best = new LinkedHashMap<>();
            for (Neo4jClient.EntitySearchResult root : roots) {
                for (Neo4jClient.IncidentRelationEvidence edge :
                        neo4j.queryIncidentRelationEvidence(experimentId,
                                graphDatasetId, root.entityId(), 40)) {
                    boolean forward = edge.subjectEntityId()
                            .equals(root.entityId());
                    String neighbourName = forward
                            ? edge.targetName() : edge.subjectName();
                    String neighbourType = forward
                            ? edge.targetType() : edge.subjectType();
                    String relation = normalize(edge.name());
                    double score = 0.0;
                    if (asksType && (relation.contains("type")
                            || relation.contains("kind")
                            || relation.contains("is_a")
                            || relation.contains("subclass"))) {
                        score += 3.0;
                    }
                    if (asksNamedAfter && relation.contains("named")) {
                        score += 3.0;
                    }
                    if (!expectedType.isBlank()
                            && typeMatches(expectedType, neighbourType)) {
                        score += 1.5;
                    }
                    score += 0.5 * meaningfulOverlap(question,
                            relation + " " + neighbourName);
                    if (score <= 0.0) continue;
                    String key = normalizeAnswer(neighbourName);
                    Scored previous = best.get(key);
                    if (previous == null || score > previous.score()) {
                        best.put(key, new Scored(neighbourName, score));
                    }
                }
            }
            List<Scored> candidates = new ArrayList<>(best.values());
            candidates.sort(Comparator.comparingDouble(Scored::score).reversed());
            if (candidates.isEmpty() || candidates.get(0).score() < 2.0) {
                return answer;
            }
            if (candidates.size() > 1
                    && candidates.get(0).score()
                    - candidates.get(1).score() < 0.5) {
                return answer;
            }
            return candidates.get(0).value();
        } catch (RuntimeException ignored) {
            return answer;
        }
    }

    private static double meaningfulOverlap(String left, String right) {
        LinkedHashSet<String> leftTerms = meaningfulTerms(left);
        LinkedHashSet<String> rightTerms = meaningfulTerms(right);
        if (leftTerms.isEmpty() || rightTerms.isEmpty()) return 0.0;
        int overlap = 0;
        for (String term : leftTerms) {
            if (rightTerms.contains(term)) overlap++;
        }
        return overlap / (double) leftTerms.size();
    }

    private static int pathDepth(List<LocalPath> paths) {
        int depth = 0;
        for (LocalPath path : paths) {
            depth = Math.max(depth, path.steps().size());
        }
        return depth;
    }

    /** Simplified entry point without a context resolver; equivalent to disabling source retrieval, used for tests or snippets. */
    public static Result runSample(LlmChatClient llm, Neo4jClient neo4j,
                                   String experimentId, String graphDatasetId,
                                   String question, Config config) {
        return runSample(llm, neo4j, experimentId, graphDatasetId,
                question, ignored -> null, config);
    }

    /**
     * Batch experiment entry point: build graphs for all unfinished samples, answer the available
     * samples, and persist answers and reasoning traces to the experiment tables.
     */
    /** As above, with optional skipping of graph-construction failures and resumable execution. */
    public static List<Result> run(DataLoader loader, LlmChatClient llm,
                                   Neo4jClient neo4j, AnswerChecker checker,
                                   Config config,
                                   boolean buildGraphs, int threadCount) {
        return run(loader, llm, neo4j, checker, config,
                buildGraphs, false, false, false, threadCount);
    }

    public static List<Result> run(DataLoader loader, LlmChatClient llm,
                                   Neo4jClient neo4j, AnswerChecker checker,
                                   Config config,
                                   boolean buildGraphs, boolean skipBuildFailures,
                                   boolean resume, int threadCount) {
        return run(loader, llm, neo4j, checker, config, buildGraphs,
                skipBuildFailures, resume, false, threadCount);
    }

    public static List<Result> run(DataLoader loader, LlmChatClient llm,
                                   Neo4jClient neo4j, AnswerChecker checker,
                                   Config config,
                                   boolean buildGraphs, boolean skipBuildFailures,
                                   boolean resume, boolean rerunReasoning, int threadCount) {
        return run(loader, llm, neo4j, checker, config, buildGraphs,
                skipBuildFailures, resume, rerunReasoning, threadCount,
                null, false, null);
    }

    /**
     * Complete batch experiment entry point.
     *
     * <p>It supports three optional capabilities without changing older call paths:</p>
     * <ul>
     *   <li>{@code graphExperimentId}: decouple graph and results. Graph and type-graph queries use
     *       this experiment's namespace, while answers are written back to the {@code loader}'s
     *       experiment. This supports running against an existing graph without rebuilding it.</li>
     *   <li>{@code assumeGraphBuilt}: treat all loaded samples as graph-ready, useful when reusing an
     *       external graph while this experiment's own Neo4j namespace is empty.</li>
     *   <li>{@code externalContextResolver}: resolve source context by data_set row ID so fallback can
     *       use any row in the graph corpus, not only samples loaded by this experiment.</li>
     * </ul>
     *
     */
    public static List<Result> run(DataLoader loader, LlmChatClient llm,
                                   Neo4jClient neo4j, AnswerChecker checker,
                                   Config config,
                                   boolean buildGraphs, boolean skipBuildFailures,
                                   boolean resume, boolean rerunReasoning, int threadCount,
                                   String graphExperimentId, boolean assumeGraphBuilt,
                                   LongFunction<String> externalContextResolver) {
        Objects.requireNonNull(loader, "loader");
        Objects.requireNonNull(llm, "llm");
        Objects.requireNonNull(neo4j, "neo4j");
        if (loader.getExperiment() == null || loader.getExperiment().getId() == null) {
            throw new IllegalArgumentException("loader must belong to a persisted experiment");
        }
        if (threadCount < 1) throw new IllegalArgumentException("threadCount must be positive");
        Config cfg = config == null ? Config.defaults() : config;
        String experimentId = Long.toString(loader.getExperiment().getId());
        // Graph experiment: default to this experiment; when specified, reuse another experiment's graph and type graph.
        String graphId = graphExperimentId == null || graphExperimentId.isBlank()
                ? experimentId : graphExperimentId.trim();
        List<DataItem> items = loader.getDataItemList();
        if (!graphId.equals(experimentId)) {
            System.out.printf("Our-method graph reuse: resultExperiment=%s graphExperiment=%s assumeGraphBuilt=%s%n",
                    experimentId, graphId, assumeGraphBuilt);
        }

        if (rerunReasoning) {
            if (checker == null) {
                throw new IllegalArgumentException("rerunReasoning requires a persistent AnswerChecker");
            }
            int reset = checker.resetExperimentReasoning(loader.getExperiment().getId());
            for (DataItem item : items) item.resetQaForRerun();
            System.out.printf("Our-method reasoning rerun: resetQaRows=%d graphBuildDisabled=true%n", reset);
        }

        long alreadyBuilt = items.stream().filter(DataItem::isGraphBuilt).count();
        boolean runGraphBuild = buildGraphs && !rerunReasoning;
        if (runGraphBuild && alreadyBuilt > 0) {
            System.out.printf("Our-method graph resume: alreadyBuilt=%d pending=%d%n",
                    alreadyBuilt, items.size() - alreadyBuilt);
        }
        // Phase 1: build graphs in batch, skipping completed samples and calling buildOne only for missing graphs.
        // When rerunReasoning=true, skip construction and rerun reasoning only.
        // When assumeGraphBuilt=true, skip construction and failure checks, marking samples as graph-ready.
        // This supports experiments that reuse an external graph namespace.
        List<BuildOutcome> builds = assumeGraphBuilt && !rerunReasoning
                ? items.stream().map(item -> {
                    if (!item.isGraphBuilt() && item.getExperimentDataId() != null) {
                        item.markGraphBuilt();
                    }
                    return new BuildOutcome(item, true, "");
                }).toList()
                : runGraphBuild
                ? parallel(items, threadCount, item -> item.isGraphBuilt()
                ? new BuildOutcome(item, true, "")
                : buildOne(llm, neo4j, experimentId, item))
                : items.stream().map(item -> item.isGraphBuilt()
                ? new BuildOutcome(item, true, "")
                : new BuildOutcome(item, false,
                "status=0 (graph not built; rerun with --build-graphs=true)")).toList();
        List<BuildOutcome> failures = builds.stream().filter(outcome -> !outcome.success()).toList();
        // A construction failure aborts the batch by default; --skip-build-failures=true skips failed samples.
        if (!failures.isEmpty() && !skipBuildFailures) {
            String summary = failures.stream().limit(10)
                    .map(failure -> sampleId(failure.item()) + ": " + failure.error())
                    .reduce((left, right) -> left + "; " + right).orElse("unknown build error");
            throw new IllegalStateException("Graph construction is incomplete; QA was not started. "
                    + summary + ". Use --skip-build-failures=true to continue with successful samples.");
        }
        if (!failures.isEmpty()) {
            System.err.printf("Our-method continuing after %d graph build failure(s); failed samples remain status=0.%n",
                    failures.size());
        }

        Map<Long, DataItem> sources = new HashMap<>();
        for (DataItem item : items) sources.put(item.getDatasetId(), item);
        // Phase 2: prepare a provenance-to-source resolver that loads source context by data_set row ID.
        // Reused graphs require the external resolver because candidates may be outside this experiment's loaded samples.
        LongFunction<String> fallbackResolver = sourceId -> {
            DataItem source = sources.get(sourceId);
            return source == null ? null : source.getContext();
        };
        LongFunction<String> contextResolver =
                externalContextResolver == null ? fallbackResolver : externalContextResolver;
        // Samples eligible for QA: graph built and either forced rerun, not resuming, or not yet complete.
        List<DataItem> qaItems = builds.stream().filter(BuildOutcome::success)
                .map(BuildOutcome::item)
                .filter(item -> rerunReasoning || !resume || !item.isQaCompleted())
                .toList();
        if (resume && !rerunReasoning) {
            long completedQa = builds.stream().filter(BuildOutcome::success)
                    .map(BuildOutcome::item).filter(DataItem::isQaCompleted).count();
            System.out.printf("Our-method QA resume: alreadyCompleted=%d pending=%d buildSkipped=%d%n",
                    completedQa, qaItems.size(), failures.size());
        }
        List<QaOutcome> outcomes = parallel(qaItems, threadCount, item -> {
            try {

        // Run QA with graph namespaces isolated by data type. Pass graphId because the graph may
        // belong to another experiment; results are still written back to this experiment.
                Result answer = runSample(llm, neo4j, graphId,
                        GraphBuild.datasetNamespace(item.getDataType()), item.getQuestion(),
                        contextResolver, cfg);
        // Persist the answer and reasoning trace to experiment_data; retry does not overwrite completed results.
                if (checker != null && item.getExperimentDataId() != null
                        && !RETRY_ANSWER.equalsIgnoreCase(answer.answer())) {
                    String reasoningPath = answer.reasoningPathJson();
                    checker.checkAndUpdate(item.getExperimentDataId(), answer.answer(), reasoningPath);
                    item.markQaCompleted(answer.answer(), reasoningPath);
                }
                return new QaOutcome(item, answer);
            } catch (RuntimeException error) {
                System.err.printf("Our-method QA failed for dataset=%s sample=%s: %s%n",
                        item.getDataType(), sampleId(item), concise(error));
                return new QaOutcome(item, retryResult());
            }
        });
        Map<Long, Result> byDatasetId = new HashMap<>();
        for (QaOutcome outcome : outcomes) byDatasetId.put(outcome.item().getDatasetId(), outcome.result());
        // Rebuild results in input order, filling skipped samples with resumed results or retry placeholders.
        List<Result> ordered = new ArrayList<>(items.size());
        for (DataItem item : items) {
            Result answer = byDatasetId.get(item.getDatasetId());
            if (answer == null) {
                answer = resume && !rerunReasoning && item.isQaCompleted()
                        ? resumedResult(item) : retryResult();
            }
            ordered.add(answer);
            AnswerGrounding grounding = answer.answerGrounding();
        // Print one key-metrics line per sample for log auditing: answer, source, depth, fallback sample, and grounding.
            System.out.printf("ours dataset=%s sample=%s answer=%s source=%s depth=%d finished=%s fallback=%s "
                            + "grounded=%s groundScore=%s%n",
                    item.getDataType(), sampleId(item), answer.answer(), answer.answerSource(),
                    answer.exploredDepth(), answer.finished(), answer.fallbackDatasetId(),
                    grounding == null ? "none" : (grounding.applied() ? grounding.entityName() : "no"),
                    grounding == null ? "-" : String.format(Locale.ROOT, "%.4f", grounding.score()));
        }
        return List.copyOf(ordered);
    }

    /**
     * Graph-construction entry point for one sample. Delegates to {@link GraphBuild#buildSample}
     * and records success, failure, and latency. Detailed construction steps are documented in GraphBuild.java.
     */
    private static BuildOutcome buildOne(LlmChatClient llm, Neo4jClient neo4j,
                                         String experimentId, DataItem item) {
        long startedNanos = System.nanoTime();
        String worker = Thread.currentThread().getName();
        System.out.printf("Our-method graph build started: worker=%s dataset=%s sample=%s activeLlm=%d%n",
                worker, item.getDataType(), sampleId(item), llm.activeRequests());
        try {
        // Actual construction: triple extraction -> entity merging -> type decisions -> type graph -> triples.
            GraphBuild.BuildResult result = GraphBuild.buildSample(llm, neo4j, experimentId,
                    GraphBuild.datasetNamespace(item.getDataType()), item.getDatasetId(),
                    item.getContext());
            TypeMappingResult typeMapping = result.typeMapping();
        // If no triple is stored, treat construction as failed and leave the dataset row at status=0 for retry.
            if (typeMapping.triples().isEmpty()) {
                throw new IllegalStateException("no normalized triples were stored");
            }
            item.markGraphBuilt();
            System.out.printf("Our-method graph build finished: worker=%s dataset=%s sample=%s elapsedMs=%d activeLlm=%d maxLlm=%d%n",
                    worker, item.getDataType(), sampleId(item), elapsedMs(startedNanos),
                    llm.activeRequests(), llm.maxObservedRequests());
            return new BuildOutcome(item, true, "");
        } catch (RuntimeException error) {
            System.err.printf("Our-method graph build failed: worker=%s dataset=%s sample=%s elapsedMs=%d activeLlm=%d maxLlm=%d error=%s%n",
                    worker, item.getDataType(), sampleId(item), elapsedMs(startedNanos),
                    llm.activeRequests(), llm.maxObservedRequests(), concise(error));
            return new BuildOutcome(item, false, concise(error));
        }
    }
    /**
     * Reasoning step 1: determine the expected answer type in four substeps:
     * <ol>
     *   <li>the LLM predicts candidate answer types such as "person" or "date";</li>
     *   <li>each prediction is recalled against the type-vector store to obtain real graph types;</li>
     *   <li>paths among these types are queried in the type graph for compatibility checks;</li>
     *   <li>all candidates (predicted, recalled, and path types) are ordered by the LLM.</li>
     * </ol>
     */
    private static TypeProfile resolveAnswerTypes(LlmChatClient llm, Neo4jClient neo4j,
                                                   String experimentId, String datasetId,
                                                   String question, Config cfg) {
        String raw = llm.chatWithSystemPrompt("ours-answer-type-" + UUID.randomUUID(), "",
                String.format(OursAnswerTypePrediction,
                        cfg.predictedTypeLimit(), question));
        // Substeps 1: parse the LLM output and fail fast if no valid type can be extracted.
        List<PredictedType> predicted = parsePredictedTypes(raw, cfg.predictedTypeLimit());
        if (predicted.isEmpty()) {
            throw new IllegalStateException("LLM returned no valid predicted answer type");
        }
        LinkedHashMap<String, RecalledType> recalled = new LinkedHashMap<>();
        // Substeps 2: type-vector recall, keeping the smallest distance when the same type name appears repeatedly.
        for (PredictedType type : predicted) {
            for (Neo4jClient.EntityTypeSearchResult hit : neo4j.searchEntityTypesWithinDistance(
                    experimentId, datasetId, type.type(), cfg.typeVectorCandidates(),
                    cfg.typeMaxDistance())) {
                RecalledType candidate = new RecalledType(hit.name(), hit.aliases(), hit.distance());
                recalled.merge(normalize(hit.name()), candidate,
                        (left, right) -> left.distance() <= right.distance() ? left : right);
            }
        }
        List<Neo4jClient.EntityTypePathResult> graph = neo4j.queryEntityTypePaths(
                experimentId, datasetId,
                recalled.values().stream().map(RecalledType::type).toList(),
                cfg.typeGraphHops(), cfg.typeGraphPathLimit());
        // Substeps 3 returns type-graph paths within the configured hop limit.
        // Union predicted types, recalled types, and types that occur on type-graph paths.
        LinkedHashSet<String> candidates = new LinkedHashSet<>();
        for (PredictedType type : predicted) candidates.add(normalize(type.type()));
        for (RecalledType type : recalled.values()) candidates.add(normalize(type.type()));
        for (Neo4jClient.EntityTypePathResult path : graph) {
            if (!path.candidateType().isBlank()) candidates.add(normalize(path.candidateType()));
        }
        List<String> selected = selectAnswerTypes(llm, question, predicted,
                new ArrayList<>(recalled.values()), graph,
                new ArrayList<>(candidates), cfg.answerTypeLimit());
        int reasoningHops = 4;
        // Substeps 4 returns selected, the candidate answer types used by search and answer validation.
        return new TypeProfile(List.copyOf(predicted), List.copyOf(recalled.values()),
                List.copyOf(graph), selected, reasoningHops);
    }

    /**
     * Substeps 4: ask the LLM to order the candidate types.
     * <p>The LLM's choice is reordering rather than hard filtering. If it selects only a few
     * types, the remaining candidates are restored in their original order up to answerTypeLimit,
     * so one type-judgment error does not make an otherwise answerable question impossible.</p>
     */
    private static List<String> selectAnswerTypes(LlmChatClient llm, String question,
                                                   List<PredictedType> predicted,
                                                   List<RecalledType> recalled,
                                                   List<Neo4jClient.EntityTypePathResult> graph,
                                                   List<String> candidates, int limit) {
        if (candidates.isEmpty()) return List.of();
        Map<String, Object> input = new LinkedHashMap<>();
        input.put("question", question);
        input.put("predicted_types", predicted);
        input.put("recalled_types", recalled);
        input.put("type_paths", graph);
        input.put("candidate_types", candidates.stream()
                .map(value -> Map.of("candidate_type", value)).toList());
        String raw = llm.chatWithSystemPrompt("ours-select-types-" + UUID.randomUUID(), "",
                String.format(OursAnswerTypeSelection, limit, JsonCodec.toJson(input)));
        Set<String> allowed = new LinkedHashSet<>(candidates);
        List<String> selected = new ArrayList<>();
        try {
            JsonArray values = parseObject(raw).getAsJsonArray("types");
            if (values != null) {
                for (JsonElement value : values) {
                    if (!value.isJsonPrimitive()) continue;
                    String type = normalize(value.getAsString());
                    if (allowed.contains(type) && !selected.contains(type)) selected.add(type);
                    if (selected.size() >= limit) break;
                }
            }
        } catch (RuntimeException ignored) { }
        // Treat LLM selection as ordering, not filtering, and restore other predicted/recalled types.
        LinkedHashSet<String> result = new LinkedHashSet<>(selected);
        for (String candidate : candidates) {
            if (result.size() >= limit) break;
            result.add(candidate);
        }
        return List.copyOf(result);
    }

    /**
     * Reasoning step 2: find topic entities as search seeds.
     * <p>The LLM extracts entity names, which are retrieved by lexical matching and vector
     * similarity. Each topic first receives one guaranteed slot, preferring exact lexical matches;
     * remaining slots rotate by rank so one ambiguous topic cannot consume the whole seed budget.</p>
     */
    private static List<Neo4jClient.EntitySearchResult> topicEntities(
            LlmChatClient llm, Neo4jClient neo4j, String experimentId,
            String datasetId, String question, Config cfg) {
        String raw = llm.chatWithSystemPrompt("ours-topic-" + UUID.randomUUID(),
                "Extract named entities that should seed a knowledge graph search. Return only JSON: "
                        + "{\"entities\":[\"\"]}.", question);
        List<String> names = parseStringArray(raw, "entities");
        LinkedHashMap<String, Neo4jClient.EntitySearchResult> result = new LinkedHashMap<>();
        List<List<Neo4jClient.EntitySearchResult>> lexicalByTopic = new ArrayList<>();
        List<List<Neo4jClient.EntitySearchResult>> vectorByTopic = new ArrayList<>();
        for (String name : names) {
            List<Neo4jClient.EntitySearchResult> lexical = neo4j.searchEntitiesByName(
                    experimentId, datasetId, name, cfg.maxTopicEntities());
            List<Neo4jClient.EntitySearchResult> vector = neo4j.searchEntities(
                    experimentId, datasetId, name, cfg.maxTopicEntities());
            lexicalByTopic.add(lexical);
            vectorByTopic.add(vector);

        // Give each extracted topic one guaranteed seed: exact lexical match first, then vector top-1,
        // and finally lexical top-1.
            Neo4jClient.EntitySearchResult seed = lexical.stream()
                    .filter(hit -> normalizeAnswer(hit.name()).equals(normalizeAnswer(name)))
                    .findFirst().orElse(vector.isEmpty()
                            ? (lexical.isEmpty() ? null : lexical.get(0)) : vector.get(0));
            if (seed != null) result.putIfAbsent(seed.entityId(), seed);
        }
        // Fill remaining slots round-robin by rank so one ambiguous topic cannot consume the entire seed budget.
        for (int rank = 0; rank < cfg.maxTopicEntities()
                && result.size() < cfg.maxTopicEntities(); rank++) {
            for (int topic = 0; topic < names.size()
                    && result.size() < cfg.maxTopicEntities(); topic++) {
                List<Neo4jClient.EntitySearchResult> lexical = lexicalByTopic.get(topic);
                List<Neo4jClient.EntitySearchResult> vector = vectorByTopic.get(topic);
                if (rank < lexical.size()) {
                    Neo4jClient.EntitySearchResult hit = lexical.get(rank);
                    result.putIfAbsent(hit.entityId(), hit);
                }
                if (rank < vector.size() && result.size() < cfg.maxTopicEntities()) {
                    Neo4jClient.EntitySearchResult hit = vector.get(rank);
                    result.putIfAbsent(hit.entityId(), hit);
                }
            }
        }
        // Fallback: if no topic name retrieves an entity, vector-search the full question once.
        if (result.isEmpty()) {
            for (Neo4jClient.EntitySearchResult hit : neo4j.searchEntities(
                    experimentId, datasetId, question, cfg.maxTopicEntities())) {
                result.putIfAbsent(hit.entityId(), hit);
            }
        }
        return result.values().stream().limit(cfg.maxTopicEntities()).toList();
    }

    /** Candidate cap for one batch edge-scoring call: four times the neighbour limit. */
    static int edgeCandidateLimit(Config cfg) {
        return Math.max(cfg.maxNeighbours(), cfg.maxNeighbours() * 4);
    }

    /**
     * Select unvisited incident edges of the current entity as candidates and deduplicate them.
     *
     * <p>Direction and relation type are intentionally ignored so one LLM call can score the whole
     * neighbourhood. The deduplication key is neighbour entity ID plus relation name.</p>
     */
    static List<Neo4jClient.IncidentRelationEvidence> edgeCandidates(String entityId,
                                                                    Set<String> visitedEntityIds,
                                                                    List<Neo4jClient.IncidentRelationEvidence> edges,
                                                                    int limit) {
        Objects.requireNonNull(entityId, "entityId");
        Objects.requireNonNull(edges, "edges");
        if (limit < 1) throw new IllegalArgumentException("limit must be positive");
        LinkedHashMap<String, Neo4jClient.IncidentRelationEvidence> unique = new LinkedHashMap<>();
        for (Neo4jClient.IncidentRelationEvidence edge : edges) {
            if (edge == null) continue;
            boolean forward = edge.subjectEntityId().equals(entityId);
            String neighbourId = forward ? edge.targetEntityId() : edge.subjectEntityId();
            if (neighbourId == null || neighbourId.isBlank()) continue;
            if (visitedEntityIds != null && visitedEntityIds.contains(neighbourId)) continue;
            String key = neighbourId + "\u0000" + normalize(edge.name());
            unique.putIfAbsent(key, edge);
            if (unique.size() >= limit) break;
        }
        return List.copyOf(unique.values());
    }

    /**
     * Core edge pruning: score all current candidate edges in one LLM call (0..1).
     *
     * <p>The prompt contains the question, current entity and type, candidate answer types,
     * prior path, and each edge's id, relation, neighbour, neighbour type, direction, and context.
     * The model returns id-to-score mappings, which {@link #normalizeScores} aligns to candidate
     * order and truncates to width.</p>
     */
    private static List<Scored> scoreEdges(LlmChatClient llm, String question, State state,
                                            List<Neo4jClient.IncidentRelationEvidence> candidates,
                                            TypeProfile types,
                                            Config cfg) {
        List<Map<String, Object>> payload = new ArrayList<>(candidates.size());
        List<String> ids = new ArrayList<>(candidates.size());
        for (int index = 0; index < candidates.size(); index++) {
            Neo4jClient.IncidentRelationEvidence edge = candidates.get(index);
            boolean forward = edge.subjectEntityId().equals(state.entityId());
            Map<String, Object> item = new LinkedHashMap<>();
            item.put("id", index);
            item.put("relation", edge.name());
            item.put("neighbour", forward ? edge.targetName() : edge.subjectName());
            item.put("neighbour_type", forward ? edge.targetType() : edge.subjectType());
            item.put("direction", forward ? "outgoing" : "incoming");
            item.put("context", truncate(edge.context(), cfg.maxEdgeContextChars()));
            payload.add(item);
            ids.add(Integer.toString(index));
        }
        Map<String, Object> input = new LinkedHashMap<>();
        input.put("question", question);
        input.put("current_entity", state.entityName());
        input.put("current_entity_type", state.entityType());
        // With -Dtypes.guidedsearch=false, edge pruning uses relation context, neighbour type,
        // and prior paths only, without type compatibility guiding expansion.
        input.put("candidate_answer_types", types.answerTypes());
        input.put("traversed_path", serializeSteps(state.path()));
        input.put("candidate_edges", payload);
        String raw = llm.chatWithSystemPrompt("ours-edges-" + UUID.randomUUID(), "",
                String.format(OursEdgePrune, JsonCodec.toJson(input)));
        return normalizeScores(parseScores(raw, "edges", "id"), ids, cfg.width());
    }

    /**
     * Early-stopping decision (ToG "evaluate"): give all current-layer paths to the LLM and ask
     * for finished=true/false. True means the paths suffice, so the caller invokes
     * {@link #graphAnswer} immediately. Parse failures are treated as false because continuing is safer.
     */
    private static boolean evaluate(LlmChatClient llm, String question,
                                    TypeProfile types, List<List<PathStep>> paths) {
        if (paths.isEmpty()) return false;
        Map<String, Object> input = new LinkedHashMap<>();
        input.put("question", question);
        input.put("candidate_answer_types", types.answerTypes());
        input.put("paths", paths.stream().map(OurMethodReasoning::serializeSteps).toList());
        String raw = llm.chatWithSystemPrompt("ours-evaluate-" + UUID.randomUUID(), "",
                String.format(OursEvaluate, JsonCodec.toJson(input)));
        try {
            JsonElement finished = parseObject(raw).get("finished");
            return finished != null && finished.isJsonPrimitive() && finished.getAsBoolean();
        } catch (RuntimeException ignored) {
            return false;
        }
    }

    /**
     * ToG answering: ask the LLM for an answer from current candidate paths and validate that the
     * answer is an entity on those paths.
     *
     * <p>Two gates: the answer must align with a path entity, and that entity's type must be
     * compatible with a candidate answer type. Non-null is returned only when both pass, ensuring
     * that every answerSource=graph answer has graph provenance.</p>
     *
     * @return a validated answer entity, or null when validation fails and the caller must continue or fall back
     */
    private static AnswerEntity graphAnswer(LlmChatClient llm, Neo4jClient neo4j,
                                             String experimentId, String datasetId,
                                             String question, TypeProfile types,
                                             List<List<PathStep>> paths, int typeHops) {
        return graphAnswer(llm, neo4j, experimentId, datasetId, question, types,
                paths, typeHops, false);
    }

    private static AnswerEntity graphAnswer(LlmChatClient llm, Neo4jClient neo4j,
                                             String experimentId, String datasetId,
                                             String question, TypeProfile types,
                                             List<List<PathStep>> paths, int typeHops,
                                             boolean compactPaths) {
        if (types.answerTypes().isEmpty()) return null;
        List<List<PathStep>> answerPaths =
                paths == null ? List.of() : paths;
        if (answerPaths.isEmpty()) return null;
        Map<String, List<AnswerEntity>> entities = pathEntities(answerPaths);
        if (entities.isEmpty()) return null;
        List<RankedAnswer> rankedTerminals =
                rankPathTerminalCandidates(question, answerPaths);
        List<String> constraints = questionMentionedEntities(question);
        if (constraints.size() >= 2) {
                List<RankedAnswer> fullCoverage = new ArrayList<>();
                for (RankedAnswer item : rankedTerminals.stream()
                        .limit(20).toList()) {
                    if (!compatibleAnswerType(types.answerTypes(),
                            item.entity().type())) {
                        continue;
                    }
                    if (constraintCoverage(neo4j, experimentId, datasetId,
                            constraints, item.entity().name())
                            >= constraints.size()
                            && directlyConnectedToConstraint(neo4j,
                            experimentId, datasetId, constraints,
                            item.entity().name())) {
                        fullCoverage.add(item);
                    }
                }
                if (fullCoverage.size() == 1) {
                    return fullCoverage.get(0).entity();
                }
                fullCoverage.sort(Comparator.comparingDouble(
                        RankedAnswer::score).reversed());
                System.out.printf(
                        "ours-constraint-preview constraints=%s full=%s%n",
                        constraints,
                        fullCoverage.stream().limit(5)
                                .map(item -> item.entity().name() + ":"
                                        + String.format(Locale.ROOT, "%.3f",
                                        item.score()))
                                .toList());
                if (fullCoverage.size() >= 2
                        && fullCoverage.get(0).score()
                        - fullCoverage.get(1).score() >= 0.5) {
                    return fullCoverage.get(0).entity();
                }
        }
        Map<String, Object> input = new LinkedHashMap<>();
        input.put("question", question);
        input.put("candidate_answer_types", types.answerTypes());
        input.put("candidate_type_graph", types.graph());
        List<List<Map<String, Object>>> promptPaths = new ArrayList<>();
        for (List<PathStep> path : answerPaths) {
            promptPaths.add(compactPaths
                    ? compactSubquestionSteps(path)
                    : serializeSteps(path));
        }
        input.put("paths", promptPaths);
        String raw = llm.chatWithSystemPrompt("ours-graph-answer-" + UUID.randomUUID(), "",
                String.format(compactPaths ? OursGraphAnswerCompact : OursGraphAnswer,
                        JsonCodec.toJson(input)));
        String answer;
        try {
            JsonElement value = parseObject(raw).get("predict_answer");
            if (value == null || !value.isJsonPrimitive()) return null;
            answer = value.getAsString().trim();
        } catch (RuntimeException ignored) {
            return null;
        }
        if (answer.isBlank() || UNKNOWN_ANSWER.equalsIgnoreCase(answer)) return null;
        // Prefer exact matching against the original spelling in the path.
        // The model may return an abbreviation or alias, so use vector recall to align the answer
        // back to a path entity before applying type-compatibility validation.
        AnswerEntity exact = compatiblePathEntity(neo4j, experimentId, datasetId, types, typeHops,
                entities.getOrDefault(normalizeAnswer(answer), List.of()));
        if (exact != null) {
            return preferRankedTerminal(neo4j, experimentId, datasetId,
                    question, exact, rankedTerminals, types);
        }
        for (Neo4jClient.EntitySearchResult hit : neo4j.searchEntitiesWithinDistance(
                experimentId, datasetId, answer, 5, 0.25)) {
            AnswerEntity resolved = compatiblePathEntity(neo4j, experimentId, datasetId, types, typeHops,
                    entities.getOrDefault(normalizeAnswer(hit.name()), List.of()));
            if (resolved != null) {
                return preferRankedTerminal(neo4j, experimentId, datasetId,
                        question, resolved, rankedTerminals, types);
            }
        }
        return null;
    }

    private static AnswerEntity preferRankedTerminal(
            Neo4jClient neo4j,
            String experimentId,
            String graphDatasetId,
            String question,
            AnswerEntity selected,
            List<RankedAnswer> ranked,
            TypeProfile types) {
        if (selected == null || ranked == null || ranked.isEmpty()) {
            return selected;
        }
        List<String> constraints = questionMentionedEntities(question);
        if (constraints.size() >= 2) {
                int selectedCoverage = constraintCoverage(neo4j,
                        experimentId, graphDatasetId, constraints,
                        selected.name());
                int bestCoverage = selectedCoverage;
                AnswerEntity best = selected;
                boolean tie = false;
                for (RankedAnswer item : ranked.stream().limit(20).toList()) {
                    if (!compatibleAnswerType(types.answerTypes(),
                            item.entity().type())) {
                        continue;
                    }
                    int coverage = constraintCoverage(neo4j,
                            experimentId, graphDatasetId, constraints,
                            item.entity().name());
                    if (coverage > bestCoverage) {
                        bestCoverage = coverage;
                        best = item.entity();
                        tie = false;
                    } else if (coverage == bestCoverage
                            && best != selected
                            && !normalizeAnswer(item.entity().name()).equals(
                            normalizeAnswer(best.name()))) {
                        tie = true;
                    }
                }
                if (bestCoverage > selectedCoverage && !tie) {
                    return best;
                }
        }
        RankedAnswer top = ranked.get(0);
        if (!compatibleAnswerType(types.answerTypes(), top.entity().type())) {
            return selected;
        }
        double selectedScore = ranked.stream()
                .filter(item -> normalizeAnswer(item.entity().name())
                        .equals(normalizeAnswer(selected.name())))
                .mapToDouble(RankedAnswer::score)
                .max()
                .orElse(Double.NEGATIVE_INFINITY);
        if (!Double.isFinite(selectedScore)) {
            return selected;
        }
        if (top.score() - selectedScore >= 0.6) {
            return top.entity();
        }
        return selected;
    }

    private static List<String> questionMentionedEntities(String question) {
        if (question == null || question.isBlank()) return List.of();
        Set<String> stop = Set.of("which", "what", "who", "where", "when",
                "whose", "american", "the", "a", "an", "and", "or",
                "abc", "us", "u.s.", "university", "city", "county",
                "state", "region");
        LinkedHashSet<String> result = new LinkedHashSet<>();
        Matcher matcher = Pattern.compile(
                "\\b[A-Z][A-Za-z0-9'’.-]*(?:\\s+[A-Z][A-Za-z0-9'’.-]*)*")
                .matcher(question);
        while (matcher.find()) {
            String value = matcher.group().trim();
            if (value.isBlank() || stop.contains(normalize(value))) continue;
            result.add(value);
        }
        return List.copyOf(result);
    }

    private static int constraintCoverage(
            Neo4jClient neo4j,
            String experimentId,
            String graphDatasetId,
            List<String> constraints,
            String candidate) {
        if (candidate == null || candidate.isBlank()) return 0;
        try {
            List<Neo4jClient.EntitySearchResult> roots =
                    neo4j.searchEntitiesByName(experimentId, graphDatasetId,
                            candidate, 1);
            if (roots.isEmpty()) return 0;
            Set<String> reachable = reachableEntityIds(neo4j,
                    experimentId, graphDatasetId, roots.get(0).entityId(), 2);
            int coverage = 0;
            for (String constraint : constraints) {
                List<Neo4jClient.EntitySearchResult> hits =
                        neo4j.searchEntitiesByName(experimentId,
                                graphDatasetId, constraint, 10);
                if (hits.stream().anyMatch(hit -> reachable.contains(
                        hit.entityId()))) {
                    coverage++;
                }
            }
            return coverage;
        } catch (RuntimeException ignored) {
            return 0;
        }
    }

    private static boolean directlyConnectedToConstraint(
            Neo4jClient neo4j,
            String experimentId,
            String graphDatasetId,
            List<String> constraints,
            String candidate) {
        try {
            List<Neo4jClient.EntitySearchResult> roots =
                    neo4j.searchEntitiesByName(experimentId, graphDatasetId,
                            candidate, 1);
            if (roots.isEmpty()) return false;
            Set<String> oneHop = reachableEntityIds(neo4j, experimentId,
                    graphDatasetId, roots.get(0).entityId(), 1);
            for (String constraint : constraints) {
                if (neo4j.searchEntitiesByName(experimentId, graphDatasetId,
                        constraint, 10).stream().anyMatch(hit ->
                        oneHop.contains(hit.entityId()))) {
                    return true;
                }
            }
        } catch (RuntimeException ignored) {
            return false;
        }
        return false;
    }

    private static Set<String> reachableEntityIds(
            Neo4jClient neo4j,
            String experimentId,
            String graphDatasetId,
            String startEntityId,
            int maxHops) {
        Set<String> visited = new LinkedHashSet<>();
        visited.add(startEntityId);
        Set<String> frontier = new LinkedHashSet<>();
        frontier.add(startEntityId);
        for (int hop = 0; hop < maxHops; hop++) {
            Set<String> next = new LinkedHashSet<>();
            for (String entityId : frontier) {
                for (Neo4jClient.IncidentRelationEvidence edge :
                        neo4j.queryIncidentRelationEvidence(experimentId,
                                graphDatasetId, entityId, 50)) {
                    String neighbour = edge.subjectEntityId().equals(entityId)
                            ? edge.targetEntityId() : edge.subjectEntityId();
                    if (visited.add(neighbour)) next.add(neighbour);
                }
            }
            frontier = next;
            if (frontier.isEmpty()) break;
        }
        return visited;
    }

    /**
     * Add up to {@code maxExtraHops} question-matching relations after each
     * candidate terminal. The reader prompt receives only entity and relation
     * names, never relation evidence.
     */
    private static List<RankedAnswer> rankPathTerminalCandidates(
            String question,
            List<List<PathStep>> paths) {
        Map<String, RankedAnswer> best = new LinkedHashMap<>();
        for (List<PathStep> path : paths) {
            if (path == null || path.isEmpty()) continue;
            PathStep last = path.get(path.size() - 1);
            String terminalName = last.traversedForward()
                    ? last.target() : last.subject();
            String terminalType = last.traversedForward()
                    ? last.targetType() : last.subjectType();
            StringBuilder facts = new StringBuilder();
            for (PathStep step : path) {
                facts.append(' ').append(step.subject())
                        .append(' ').append(step.relation())
                        .append(' ').append(step.target());
            }
            double score = meaningfulOverlap(question, facts.toString())
                    + 0.5 * meaningfulOverlap(question, last.relation())
                    + (typeMatches(questionTypeHint(question), terminalType)
                    ? 0.5 : 0.0)
                    + roleSpecificityScore(question, path)
                    - 0.01 * path.size();
            String key = normalizeAnswer(terminalName);
            RankedAnswer current = best.get(key);
            RankedAnswer candidate = new RankedAnswer(
                    new AnswerEntity(terminalName, terminalType),
                    score, path.size());
            if (current == null
                    || candidate.score() > current.score()
                    || (candidate.score() == current.score()
                    && candidate.depth() < current.depth())) {
                best.put(key, candidate);
            }
        }
        return best.values().stream()
                .sorted(Comparator.comparingDouble(RankedAnswer::score)
                        .reversed()
                        .thenComparing(item -> normalizeAnswer(
                                item.entity().name())))
                .toList();
    }

    private static double roleSpecificityScore(
            String question,
            List<PathStep> path) {
        String questionValue = normalize(question);
        boolean asksStarring = questionValue.contains("starring role")
                || questionValue.contains("starred")
                || questionValue.contains("starring");
        if (!asksStarring) return 0.0;
        double score = 0.0;
        for (PathStep step : path) {
            String context = normalize(step.context());
            if (context.contains("guest star")) {
                score -= 1.0;
            } else if (context.contains("starring role")
                    || context.contains("leading role")) {
                score += 0.5;
            }
        }
        return score;
    }

    /** Return the first candidate entity with a compatible predicted answer type, or null. */
    private static AnswerEntity compatiblePathEntity(Neo4jClient neo4j, String experimentId,
                                                     String datasetId, TypeProfile types, int typeHops,
                                                     List<AnswerEntity> candidates) {
        for (AnswerEntity entity : candidates) {
            if (!entity.type().isBlank() && neo4j.areEntityTypesCompatible(
                    experimentId, datasetId, entity.type(), types.answerTypes(), typeHops)) {
                return entity;
            }
        }
        return null;
    }

    /**
     * Step-3 fallback: when graph search cannot answer, recover source context using provenance
     * votes and type matching.
     *
     * <p>Flow: rank candidates by provenance and type scores, optionally rerank them with the LLM,
     * take the top fallbackContextLimit passages (full-context mode is not truncated; over-budget
     * input uses map-reduce batching), and ask the LLM for an answer and answer_source.</p>
     *
     * @param votes provenance votes accumulated during graph search (data_set ID -> vote count)
     */
    private static Result fallback(LlmChatClient llm, Neo4jClient neo4j,
                                   String experimentId, String graphDatasetId,
                                   String question,
                                   LongFunction<String> contextResolver,
                                   TypeProfile types, List<List<PathStep>> paths,
                                   int depth, Map<Long, Integer> votes,
                                   Config config) {
        // Use only the top two predicted types for fallback ranking; more types add noise and reduce precision.
        List<String> fallbackTypes = types.predicted().stream()
                .limit(2).map(PredictedType::type).toList();
        // With LLM reranking enabled, fetch fallbackLlmCandidateLimit candidates before reducing to
        // fallbackContextLimit; otherwise take the target count by deterministic score directly.
        int preliminaryLimit = config.fallbackLlmRerank()
                ? config.fallbackLlmCandidateLimit() : config.fallbackContextLimit();
        List<FallbackCandidate> preliminary = rankFallbackCandidates(
                neo4j, experimentId, graphDatasetId, votes, fallbackTypes,
                config.fallbackCandidatePool(), preliminaryLimit,
                config.fallbackTypeMatchLimit(), config.fallbackTypeMaxDistance(),
                config.fallbackTypeWeight(), config.fallbackTypeGraphEdgeLimit());
        // Optional LLM reranking: provide each candidate's type graph so the model can identify the likely source.
        List<FallbackCandidate> rankedCandidates = config.fallbackLlmRerank()
                ? rerankFallbackCandidatesWithLlm(llm, question, types.predicted(), preliminary,
                config.fallbackContextLimit(), config.fallbackLlmWeight())
                : preliminary.stream().limit(config.fallbackContextLimit()).toList();
        List<Long> sourceIds = rankedCandidates.stream()
                .map(FallbackCandidate::sourceDatasetId).toList();
        // sourceId records the primary fallback source (rank 1) and is persisted with the result for analysis.
        Long sourceId = sourceIds.isEmpty() ? null : sourceIds.get(0);
        List<Map<String, Object>> contexts = new ArrayList<>();
        List<RetrievedContext> retrievedContexts = new ArrayList<>();
        if (contextResolver != null) {
        // Retrieve source text in rank order: full context in full-context mode; otherwise relevant excerpts.
            for (int index = 0; index < sourceIds.size(); index++) {
                Long candidateId = sourceIds.get(index);
                String context = contextResolver.apply(candidateId);
                if (context == null || context.isBlank()) continue;
        // Send only plain text from the benchmark dataset, removing JSON wrappers and supporting-fact
        // markers to prevent the model from exploiting annotation leakage.
                String completeContext = ContextSanitizer.toPromptText(context);
                if (!config.fullContextFallback()) {
        // Legacy excerpt mode: divide the total budget across candidates and choose relevant sentences.
                    int remaining = Math.max(1, config.maxFallbackContextChars()
                            / Math.max(1, sourceIds.size()));
                    completeContext = selectRelevantFallbackContext(completeContext, question, paths, remaining);
                }
                contexts.add(Map.of(
                        "source_dataset_id", candidateId,
                        "context", completeContext));
                retrievedContexts.add(new RetrievedContext(candidateId, index + 1,
                        completeContext.length(), config.fullContextFallback()));
            }
        }
        Map<String, Object> input = new LinkedHashMap<>();
        // Build the final prompt from the question, top contexts, source IDs, and type-graph ranking evidence.
        input.put("question", question);
        input.put("top_contexts", contexts);
        input.put("context_mode", config.fullContextFallback() ? "full_context" : "legacy_excerpts");
        input.put("context_source_ids", retrievedContexts.stream()
                .map(RetrievedContext::sourceDatasetId).toList());
        input.put("sample_type_graph_ranking", rankedCandidates.stream()
                .map(OurMethodReasoning::compactLlmCandidate).toList());
        String raw = answerFromFullContexts(llm, input, contexts, paths, config.maxFallbackContextChars());
        try {
            JsonObject parsed = parseObject(raw);
            JsonElement value = parsed.get("predict_answer");
            if (value == null || !value.isJsonPrimitive() || value.getAsString().isBlank()) {
        // An empty answer means a definitive failure rather than a transient error, so record unknown.
        // The sample reruns only when QA is explicitly reset with --rerun-reasoning.
                return result(UNKNOWN_ANSWER, "unknown", types, paths, depth,
                        false, sourceId, votes, rankedCandidates, retrievedContexts);
            }
            String answer = value.getAsString().trim();
            String answerSource = normalize(string(parsed, "answer_source")).replace(' ', '_');
        // Normalize answer_source to context_fallback or internal_knowledge; infer other values
        // from whether context was available so downstream statistics remain consistent.
            if (UNKNOWN_ANSWER.equalsIgnoreCase(answer)) {
                answerSource = "unknown";
            } else if (!answerSource.equals("context_fallback")
                    && !answerSource.equals("internal_knowledge")) {
                if (contexts.isEmpty()) {
                    answerSource = "internal_knowledge";
                } else {
                    answerSource = "context_fallback";
                }
            }
            return result(answer, answerSource, types, paths, depth,
                    !UNKNOWN_ANSWER.equalsIgnoreCase(answer), sourceId, votes,
                    rankedCandidates, retrievedContexts);
        } catch (RuntimeException error) {
        // JSON parse failure is transient and returns retry so it is not persisted as a final answer.
            return result(RETRY_ANSWER, "retry", types, paths, depth,
                    false, sourceId, votes, rankedCandidates, retrievedContexts);
        }
    }

    /**
     * Read the selected source passages in full and answer from them.
     *
     * <p>The {@code batchChars} budget controls only batching, never which candidates are included,
     * so an over-budget candidate is not dropped. A single batch answers directly; multiple batches
     * first extract evidence per batch (map), then synthesize the final answer (reduce).</p>
     */
    private static String answerFromFullContexts(LlmChatClient llm,
                                                Map<String, Object> input,
                                                List<Map<String, Object>> contexts,
                                                List<List<PathStep>> paths,
                                                int batchChars) {
        List<List<Map<String, Object>>> exploredPaths = paths == null ? List.of()
                : paths.stream().limit(10).map(OurMethodReasoning::serializeSteps).toList();
        input.put("explored_paths", exploredPaths);
        input.put("allow_internal_knowledge", true);
        List<Map<String, Object>> expandedContexts =
                splitOversizedContexts(contexts, batchChars);
        List<List<Map<String, Object>>> batches =
                batchFullContexts(expandedContexts, batchChars);
        input.put("top_contexts", expandedContexts);
        if (batches.size() <= 1) {
        // One batch: pass the question, all contexts, and explored paths directly to the answering prompt.
            return llm.chatWithSystemPrompt("ours-fallback-" + UUID.randomUUID(), "",
                    String.format(OursFallbackAnswer, JsonCodec.toJson(input)));
        }
        // Multiple batches, MAP stage: obtain an answer and evidence summary for each batch.
        List<Map<String, Object>> analyses = new ArrayList<>();
        for (List<Map<String, Object>> batch : batches) {
            Map<String, Object> request = new LinkedHashMap<>();
            request.put("question", input.get("question"));
            request.put("top_contexts", batch);
            request.put("context_mode", input.get("context_mode"));
            request.put("explored_paths", exploredPaths);
            String raw = llm.chatWithSystemPrompt("ours-fallback-evidence-" + UUID.randomUUID(), "",
                    String.format(OursFallbackContextEvidence, JsonCodec.toJson(request)));
            JsonObject analysis = parseObject(raw);
        // Retain only the answer string, evidence summary, and locally supplied source IDs.
        // Reject model-invented source IDs to prevent fabricated retrieval sources.
            analyses.add(Map.of(
                    "source_dataset_ids", batch.stream().map(value -> value.get("source_dataset_id")).toList(),
                    "predict_answer", string(analysis, "predict_answer"),
                "evidence_summary", string(analysis, "evidence_summary")));
        }
        // REDUCE stage: synthesize all batch analyses into the final answer and decide answer_source once.
        Map<String, Object> synthesis = new LinkedHashMap<>();
        synthesis.put("question", input.get("question"));
        synthesis.put("context_analyses", analyses);
        synthesis.put("explored_paths", exploredPaths);
        synthesis.put("allow_internal_knowledge", true);
        return llm.chatWithSystemPrompt("ours-fallback-synthesis-" + UUID.randomUUID(), "",
                String.format(OursFallbackContextSynthesis,
                        JsonCodec.toJson(synthesis)));
    }

    /**
     * Long source rows are split into bounded chunks before map-reduce so a
     * single oversized context cannot produce a request larger than the model
     * can accept. No text is discarded; adjacent chunks overlap slightly.
     */
    static List<Map<String, Object>> splitOversizedContexts(
            List<Map<String, Object>> contexts, int batchChars) {
        if (contexts == null || contexts.isEmpty()) return List.of();
        if (batchChars < 1) throw new IllegalArgumentException("batchChars must be positive");
        List<Map<String, Object>> expanded = new ArrayList<>();
        for (Map<String, Object> context : contexts) {
            String text = String.valueOf(context.getOrDefault("context", ""));
            if (text.length() <= batchChars) {
                expanded.add(context);
                continue;
            }
            int start = 0;
            int part = 0;
            while (start < text.length()) {
                int end = Math.min(text.length(), start + batchChars);
                String chunk = text.substring(start, end);
                Map<String, Object> copy = new LinkedHashMap<>(context);
                copy.put("context", chunk);
                copy.put("context_part", ++part);
                expanded.add(copy);
                if (end >= text.length()) break;
                start = Math.max(start + 1, end - 1000);
            }
        }
        return List.copyOf(expanded);
    }

    /**
     * Batch contexts by character budget, starting a new batch when accumulated length exceeds it.
     * A single over-budget sample gets its own batch and is sent in full without truncation.
     */
    static List<List<Map<String, Object>>> batchFullContexts(
            List<Map<String, Object>> contexts, int batchChars) {
        if (batchChars < 1) throw new IllegalArgumentException("batchChars must be positive");
        List<List<Map<String, Object>>> batches = new ArrayList<>();
        List<Map<String, Object>> batch = new ArrayList<>();
        long chars = 0;
        for (Map<String, Object> context : contexts) {
            int length = String.valueOf(context.getOrDefault("context", "")).length();
            if (!batch.isEmpty() && chars + length > batchChars) {
                batches.add(List.copyOf(batch));
                batch.clear();
                chars = 0;
            }
            batch.add(context);
            chars += length;
        }
        if (!batch.isEmpty()) batches.add(List.copyOf(batch));
        return List.copyOf(batches);
    }

    /** Convenience overload with a candidate pool of at least 20 and built-in type-match defaults. */
    static List<FallbackCandidate> rankFallbackCandidates(Neo4jClient neo4j,
                                                           String experimentId,
                                                           String graphDatasetId,
                                                           Map<Long, Integer> votes,
                                                           List<String> answerTypes,
                                                           int resultLimit) {
        return rankFallbackCandidates(neo4j, experimentId, graphDatasetId, votes,
                answerTypes, Math.max(20, resultLimit), resultLimit,
                5, 0.65, 0.60, 100);
    }

    /**
     * Full fallback-candidate ranking: take top-N source samples by provenance votes, load their
     * entity-type graphs for optional LLM reranking, perform type-vector matching, and fuse
     * provenance and type scores. Queries in the middle steps may fail without aborting because
     * provenance still provides deterministic ranking.
     */
    static List<FallbackCandidate> rankFallbackCandidates(Neo4jClient neo4j,
                                                           String experimentId,
                                                           String graphDatasetId,
                                                           Map<Long, Integer> votes,
                                                           List<String> answerTypes,
                                                           int candidatePool,
                                                           int resultLimit,
                                                           int typeMatchLimit,
                                                           double typeMaxDistance,
                                                           double typeWeight,
                                                           int typeGraphEdgeLimit) {
        if (resultLimit < 1) throw new IllegalArgumentException("resultLimit must be positive");
        if (candidatePool < resultLimit) {
            throw new IllegalArgumentException("candidatePool must be >= resultLimit");
        }
        List<Long> candidates = selectTopDatasetIds(votes, candidatePool);
        if (candidates.isEmpty()) return List.of();
        List<Neo4jClient.SampleEntityTypeGraph> graphs = List.of();
        try {
            // Type graphs provide structural evidence for LLM reranking; missing graphs do not block ranking.
            graphs = neo4j.querySampleEntityTypeGraphs(
                    experimentId, graphDatasetId, candidates, typeGraphEdgeLimit);
        } catch (RuntimeException unavailable) {
            // Ignore failures because ranking can still use provenance votes and type matching.
        }
        List<Neo4jClient.SampleEntityTypeMatch> matches = List.of();
        try {
            // Vector matches between predicted answer types and each candidate's entity types.
            matches = neo4j.matchSampleEntityTypes(experimentId, graphDatasetId,
                    candidates, answerTypes, typeMatchLimit, typeMaxDistance);
        } catch (RuntimeException unavailable) {
            // Ignore failures: provenance votes provide deterministic fallback ranking when vector matching is unavailable.
        }

        return combineFallbackCandidates(votes, candidates, graphs, matches,
                resultLimit, typeWeight);
    }

    /**
     * Fuse provenance and type scores:
     * combined = (1 - typeWeight) * voteScore + typeWeight * typeScore.
     * Order by total score descending, vote count descending, then data_set ID ascending for reproducibility.
     */
    static List<FallbackCandidate> combineFallbackCandidates(
            Map<Long, Integer> votes,
            List<Long> candidates,
            List<Neo4jClient.SampleEntityTypeGraph> graphs,
            List<Neo4jClient.SampleEntityTypeMatch> matches,
            int resultLimit, double typeWeight) {
        if (resultLimit < 1) throw new IllegalArgumentException("resultLimit must be positive");
        if (!Double.isFinite(typeWeight) || typeWeight < 0.0 || typeWeight > 1.0) {
            throw new IllegalArgumentException("typeWeight must be between 0 and 1");
        }
        Map<Long, Neo4jClient.SampleEntityTypeGraph> graphBySource = new HashMap<>();
        for (Neo4jClient.SampleEntityTypeGraph graph : graphs) {
            graphBySource.put(graph.sourceDatasetId(), graph);
        }
        Map<Long, List<Neo4jClient.SampleEntityTypeMatch>> matchesBySource = new HashMap<>();
        for (Neo4jClient.SampleEntityTypeMatch match : matches) {
            matchesBySource.computeIfAbsent(match.sourceDatasetId(), ignored -> new ArrayList<>())
                    .add(match);
        }
        int maxVotes = candidates.stream().mapToInt(id -> votes.getOrDefault(id, 0)).max().orElse(1);
        List<FallbackCandidate> ranked = new ArrayList<>();
        for (Long sourceId : candidates) {
            int voteCount = votes.getOrDefault(sourceId, 0);
            double voteScore = maxVotes <= 0 ? 0.0 : (double) voteCount / maxVotes;
            List<Neo4jClient.SampleEntityTypeMatch> sourceMatches = matchesBySource
                    .getOrDefault(sourceId, List.of()).stream()
                    .sorted(Comparator.comparingDouble(Neo4jClient.SampleEntityTypeMatch::score).reversed()
                            .thenComparing(Neo4jClient.SampleEntityTypeMatch::sampleType))
                    .toList();
            double typeScore = sourceMatches.stream()
                    .mapToDouble(match -> Math.max(0.0, Math.min(1.0, match.score())))
                    .max().orElse(0.0);
            double combined = (1.0 - typeWeight) * voteScore + typeWeight * typeScore;
            ranked.add(new FallbackCandidate(sourceId, voteCount, voteScore,
                    graphBySource.get(sourceId), sourceMatches, typeScore, combined,
                    null, combined));
        }
        return ranked.stream()
                .sorted(Comparator.comparingDouble(FallbackCandidate::combinedScore).reversed()
                        .thenComparing(Comparator.comparingInt(FallbackCandidate::provenanceVotes).reversed())
                        .thenComparingLong(FallbackCandidate::sourceDatasetId))
                .limit(resultLimit).toList();
    }

    /**
     * Rerank fallback candidates with the LLM using provenance, type-vector matches, and type-graph
     * structure. The final score is (1 - llmWeight) * deterministic score + llmWeight * LLM score.
     *
     * <p>Incomplete scores abandon reranking, and call/parse failures return the deterministic order.
     * Enabling reranking therefore does not make the pipeline fragile.</p>
     */
    static List<FallbackCandidate> rerankFallbackCandidatesWithLlm(
            LlmChatClient llm, String question, List<PredictedType> predictedTypes,
            List<FallbackCandidate> candidates, int resultLimit, double llmWeight) {
        if (candidates == null || candidates.isEmpty()) return List.of();
        if (resultLimit < 1) throw new IllegalArgumentException("resultLimit must be positive");
        if (!Double.isFinite(llmWeight) || llmWeight < 0.0 || llmWeight > 1.0) {
            throw new IllegalArgumentException("llmWeight must be between 0 and 1");
        }
        List<FallbackCandidate> fallback = candidates.stream().limit(resultLimit).toList();
        try {
            Map<String, Object> input = new LinkedHashMap<>();
            input.put("question", question);
            input.put("predicted_answer_types", predictedTypes);
            input.put("candidates", candidates.stream().map(OurMethodReasoning::compactLlmCandidate).toList());
            String raw = llm.chatWithSystemPrompt("ours-fallback-type-rank-" + UUID.randomUUID(), "",
                    String.format(OursFallbackTypeGraphRank, JsonCodec.toJson(input)));
            Map<Long, LlmTypeGraphAssessment> assessments = parseLlmTypeGraphAssessments(raw, candidates);
            // Abandon reranking unless every candidate has a score, avoiding biased partial ordering.
            if (assessments.size() != candidates.size()) return fallback;
            return candidates.stream().map(candidate -> {
                        LlmTypeGraphAssessment assessment = assessments.get(candidate.sourceDatasetId());
                        double score = (1.0 - llmWeight) * candidate.deterministicScore()
                                + llmWeight * assessment.score();
                        return new FallbackCandidate(candidate.sourceDatasetId(),
                                candidate.provenanceVotes(), candidate.normalizedVoteScore(),
                                candidate.sampleTypeGraph(), candidate.typeMatches(), candidate.typeScore(),
                                candidate.deterministicScore(), assessment, score);
                    })
                    .sorted(Comparator.comparingDouble(FallbackCandidate::combinedScore).reversed()
                            .thenComparing(Comparator.comparingInt(
                                    FallbackCandidate::provenanceVotes).reversed())
                            .thenComparingLong(FallbackCandidate::sourceDatasetId))
                    .limit(resultLimit).toList();
        } catch (RuntimeException unavailable) {
            return fallback;
        }
    }

    /**
     * Parse LLM reranking output. Validate that source_dataset_id belongs to the current candidate
     * set and count only the first occurrence of each ID.
     */
    static Map<Long, LlmTypeGraphAssessment> parseLlmTypeGraphAssessments(
            String raw, List<FallbackCandidate> candidates) {
        Set<Long> allowedIds = candidates.stream().map(FallbackCandidate::sourceDatasetId)
                .collect(java.util.stream.Collectors.toSet());
        JsonObject root = parseObject(raw);
        JsonArray values = root.has("ranked_candidates")
                && root.get("ranked_candidates").isJsonArray()
                ? root.getAsJsonArray("ranked_candidates") : new JsonArray();
        Map<Long, LlmTypeGraphAssessment> result = new LinkedHashMap<>();
        for (JsonElement element : values) {
            if (!element.isJsonObject()) continue;
            JsonObject value = element.getAsJsonObject();
            if (!value.has("source_dataset_id") || !value.has("score")) continue;
            long sourceId = value.get("source_dataset_id").getAsLong();
            if (!allowedIds.contains(sourceId) || result.containsKey(sourceId)) continue;
            result.put(sourceId, new LlmTypeGraphAssessment(
                    score(value, "type_match_score"), score(value, "structure_match_score"),
                    score(value, "score"), string(value, "reason")));
        }
        return Collections.unmodifiableMap(result);
    }

    /** Compact a candidate for prompting: at most 8 type matches, 40 type nodes, and 40 type relations. */
    private static Map<String, Object> compactLlmCandidate(FallbackCandidate candidate) {
        Map<String, Object> value = new LinkedHashMap<>();
        value.put("source_dataset_id", candidate.sourceDatasetId());
        value.put("provenance_score", candidate.normalizedVoteScore());
        value.put("vector_type_score", candidate.typeScore());
        value.put("vector_type_matches", candidate.typeMatches().stream().limit(8).toList());
        Neo4jClient.SampleEntityTypeGraph graph = candidate.sampleTypeGraph();
        value.put("entity_types", graph == null ? List.of() : graph.nodes().stream().limit(40).toList());
        value.put("type_relations", graph == null ? List.of() : graph.edges().stream().limit(40).toList());
        return value;
    }

    /** Safely read a 0..1 score from JSON; missing, non-numeric, or invalid values become 0. */
    private static double score(JsonObject object, String name) {
        try {
            JsonElement value = object.get(name);
            if (value == null || !value.isJsonPrimitive()) return 0.0;
            double score = value.getAsDouble();
            return Double.isFinite(score) ? Math.max(0.0, Math.min(1.0, score)) : 0.0;
        } catch (RuntimeException ignored) {
            return 0.0;
        }
    }

    /**
     * Context selection for the legacy excerpt mode (non-full-context). Split source text into
     * passages, rank by lexical relevance to the question and path entities, and combine top passages
     * within budget. A path-entity hit adds 20 points, more than a normal question-term hit.
     * Full-context mode is the default; this method applies only when that mode is disabled.
     */
    static String selectRelevantFallbackContext(String context, String question,
                                                List<List<PathStep>> paths,
                                                int maxChars) {
        if (context == null || context.isBlank() || maxChars < 1) return "";
        List<String> passages = contextPassages(context);
        if (passages.isEmpty()) return truncate(context, maxChars);

        LinkedHashSet<String> queryTerms = meaningfulTerms(question);
        LinkedHashSet<String> pathEntities = new LinkedHashSet<>();
        // Path entities from graph search also contribute because they locate relevant passages better than generic terms.
        if (paths != null) {
            for (List<PathStep> path : paths) {
                for (PathStep step : path) {
                    pathEntities.add(normalize(step.subject()));
                    pathEntities.add(normalize(step.target()));
                    queryTerms.addAll(meaningfulTerms(step.subject()));
                    queryTerms.addAll(meaningfulTerms(step.target()));
                }
            }
        }
        List<ScoredPassage> ranked = new ArrayList<>();
        // Score each passage by lexical overlap and path-entity hits.
        for (int index = 0; index < passages.size(); index++) {
            String passage = passages.get(index);
            String normalized = normalize(passage);
            Set<String> passageTerms = meaningfulTerms(passage);
            double score = 0.0;
            for (String term : queryTerms) {
                if (passageTerms.contains(term)) score += Math.min(8, term.length());
            }
            for (String entity : pathEntities) {
                if (!entity.isBlank() && normalized.contains(entity)) score += 20.0;
            }
            ranked.add(new ScoredPassage(passage, score, index));
        }
        ranked.sort(Comparator.comparingDouble(ScoredPassage::score).reversed()
                .thenComparingInt(ScoredPassage::sourceIndex));

        // Select passages by score, at most 8 passages and no more than maxChars characters.
        StringBuilder selected = new StringBuilder();
        int included = 0;
        for (ScoredPassage passage : ranked) {
            if (included >= 8 || selected.length() >= maxChars) break;
            String header = "[Relevant passage " + (included + 1) + "]\n";
            int remaining = maxChars - selected.length() - header.length();
            if (remaining <= 0) break;
            String text = truncate(passage.text(), remaining);
            if (text.isBlank()) continue;
            selected.append(header).append(text).append('\n');
            included++;
        }
        return selected.isEmpty() ? truncate(context, maxChars) : selected.toString().trim();
    }

    /** Split context into passages: parse JSON arrays/objects structurally and split plain text by blank lines. */
    private static List<String> contextPassages(String context) {
        List<String> result = new ArrayList<>();
        try {
            JsonElement root = JsonParser.parseString(context);
            if (root.isJsonArray()) {
                for (JsonElement item : root.getAsJsonArray()) addContextPassage(item, result);
            } else {
                addContextPassage(root, result);
            }
        } catch (RuntimeException ignored) {
            for (String paragraph : context.split("(?:\\r?\\n){2,}")) {
                if (!paragraph.isBlank()) result.add(paragraph.trim());
            }
        }
        return List.copyOf(result);
    }

    /** Support three context formats: [title, [sentences]], {title, paragraph_text}, and a plain string. */
    private static void addContextPassage(JsonElement item, List<String> result) {
        if (item == null || item.isJsonNull()) return;
        if (item.isJsonObject()) {
            JsonObject object = item.getAsJsonObject();
            String title = string(object, "title");
            String text = string(object, "paragraph_text");
            if (text.isBlank()) text = string(object, "text");
            if (text.isBlank()) text = string(object, "context");
            if (!text.isBlank()) result.add(title.isBlank() ? text : title + "\n" + text);
            return;
        }
        if (item.isJsonArray()) {
            JsonArray array = item.getAsJsonArray();
            if (array.size() >= 2 && array.get(0).isJsonPrimitive()) {
                String title = array.get(0).getAsString();
                String text = flattenJsonText(array.get(1));
                if (!text.isBlank()) result.add(title.isBlank() ? text : title + "\n" + text);
            } else {
                for (JsonElement nested : array) addContextPassage(nested, result);
            }
            return;
        }
        if (item.isJsonPrimitive()) {
            String text = item.getAsString().trim();
            if (!text.isBlank()) result.add(text);
        }
    }

    /** Recursively flatten nested string arrays into plain text. */
    private static String flattenJsonText(JsonElement value) {
        if (value == null || value.isJsonNull()) return "";
        if (value.isJsonPrimitive()) return value.getAsString().trim();
        if (!value.isJsonArray()) return "";
        StringBuilder text = new StringBuilder();
        for (JsonElement item : value.getAsJsonArray()) {
            String part = flattenJsonText(item);
            if (!part.isBlank()) {
                if (!text.isEmpty()) text.append(' ');
                text.append(part);
            }
        }
        return text.toString();
    }

    /**
     * Extract content terms by lowercasing, splitting on non-alphanumeric characters, and removing
     * stop words and one-character tokens. This purely lexical method scores context relevance.
     */
    private static LinkedHashSet<String> meaningfulTerms(String value) {
        Set<String> stopWords = Set.of("the", "and", "that", "this", "with", "from", "what",
                "which", "where", "when", "who", "whose", "whom", "was", "were", "are",
                "for", "did", "does", "has", "have", "had", "into", "its", "their", "than");
        LinkedHashSet<String> result = new LinkedHashSet<>();
        if (value == null) return result;
        for (String term : normalize(value).split("[^a-z0-9]+")) {
            if (term.length() >= 2 && !stopWords.contains(term)) result.add(term);
        }
        return result;
    }

    private record ScoredPassage(String text, double score, int sourceIndex) { }

    /**
     * Reasoning step 4: answer grounding.
     *
     * <p>Model answers often differ from canonical entity names through abbreviations, aliases, or
     * articles. Search the sample's own graph namespace for similar entities and rewrite the answer
     * to the canonical entity name when the highest similarity is at least groundAnswerMinScore and
     * the name differs from the current answer.</p>
     *
     * <p>Search stays within the sample namespace, so entities from other samples are never used.
     * Short answers such as yes/no/years are skipped to avoid homonym contamination.</p>
     */
    static Result groundAnswerOnSampleEntity(Neo4jClient neo4j, String experimentId,
                                             String graphDatasetId, Result answered,
                                             Config cfg) {
        Objects.requireNonNull(answered, "answered");
        if (cfg == null || !cfg.groundAnswerEntity()) return answered;
        String predicted = answered.answer();
        if (predicted == null || predicted.isBlank()) return answered;
        String trimmed = predicted.trim();
        if (RETRY_ANSWER.equalsIgnoreCase(trimmed) || UNKNOWN_ANSWER.equalsIgnoreCase(trimmed)) {
            return answered;
        }
        String normalized = normalizeAnswer(trimmed);
        // Short answers such as yes, no, and years have no useful entity surface form, and a
        // homonymous entity could rewrite them incorrectly, so skip grounding.
        if (normalized.isEmpty() || normalized.length() < 3
                || ANSWER_GROUNDING_STOP_WORDS.contains(normalized)) {
            return answered;
        }
        List<Neo4jClient.EntitySearchResult> hits;
        try {
            // Vector-search the sample's graph namespace for the most similar entities.
            hits = neo4j.searchEntities(experimentId, graphDatasetId, trimmed,
                    ANSWER_GROUNDING_CANDIDATES);
        } catch (RuntimeException unavailable) {
            return answered;
        }
        Neo4jClient.EntitySearchResult best = null;
        // Select the highest-similarity candidate while skipping entries with blank names.
        for (Neo4jClient.EntitySearchResult hit : hits) {
            if (hit == null || hit.name() == null || hit.name().isBlank()) continue;
            if (best == null || hit.score() > best.score()) best = hit;
        }
        if (best == null) return answered;
        // Grounding applies only when similarity passes the threshold and the normalized name actually changes.
        boolean applied = best.score() >= cfg.groundAnswerMinScore()
                && !normalizeAnswer(best.name()).equals(normalized);
        AnswerGrounding grounding = new AnswerGrounding(best.entityId(), best.name(),
                best.type(), best.score(), applied);
        return applied ? answered.groundedTo(best.name(), grounding)
                : answered.withGroundingResult(grounding);
    }

    /** Select the top-limit source samples by vote count, breaking ties by ascending data_set ID. */
    public static List<Long> selectTopDatasetIds(Map<Long, Integer> votes, int limit) {
        if (limit < 1) throw new IllegalArgumentException("limit must be positive");
        if (votes == null || votes.isEmpty()) return List.of();
        return votes.entrySet().stream().filter(entry -> entry.getKey() != null
                        && entry.getValue() != null && entry.getValue() > 0)
                .sorted(Map.Entry.<Long, Integer>comparingByValue().reversed()
                        .thenComparing(Map.Entry.comparingByKey()))
                .map(Map.Entry::getKey).limit(limit).toList();
    }

    /** Convenience Result overload without fallback candidates. */
    private static Result result(String answer, String source, TypeProfile types,
                                 List<List<PathStep>> paths, int depth, boolean finished,
                                 Long fallbackId, Map<Long, Integer> votes) {
        return result(answer, source, types, paths, depth, finished,
                fallbackId, votes, List.of());
    }

    /** Convenience Result overload with fallback candidates and no context metadata. */
    private static Result result(String answer, String source, TypeProfile types,
                                 List<List<PathStep>> paths, int depth, boolean finished,
                                 Long fallbackId, Map<Long, Integer> votes,
                                 List<FallbackCandidate> fallbackCandidates) {
        return result(answer, source, types, paths, depth, finished,
                fallbackId, votes, fallbackCandidates, List.of());
    }

    /** Final Result construction that expands TypeProfile into individual fields. */
    private static Result result(String answer, String source, TypeProfile types,
                                 List<List<PathStep>> paths, int depth, boolean finished,
                                 Long fallbackId, Map<Long, Integer> votes,
                                 List<FallbackCandidate> fallbackCandidates,
                                 List<RetrievedContext> retrievedContexts) {
        return new Result(answer, source, types.predicted(), types.recalled(), types.graph(),
                types.answerTypes(), paths, depth, finished, fallbackId, votes,
                fallbackCandidates, retrievedContexts, null);
    }

    /** Build a retry result indicating a transient failure that must not overwrite a completed answer. */
    private static Result retryResult() {
        return new Result(RETRY_ANSWER, "retry", List.of(), List.of(), List.of(),
                List.of(), List.of(), 0, false, null, Map.of());
    }

    /** For resumable runs: wrap the persisted answer in a Result without performing reasoning. */
    private static Result resumedResult(DataItem item) {
        String answer = item.getPersistedAnswer();
        if (answer == null || answer.isBlank()) answer = UNKNOWN_ANSWER;
        return new Result(answer, "resumed", List.of(), List.of(), List.of(),
                List.of(), List.of(), 0, true, null, Map.of());
    }

    /** Convert a Neo4j edge into a path step and truncate evidence text to maxContextChars. */
    private static PathStep toPathStep(Neo4jClient.IncidentRelationEvidence edge,
                                       boolean forward, int maxContextChars) {
        return new PathStep(edge.subjectName(), edge.subjectType(),
                edge.subjectSourceDatasetIds(), edge.name(),
                truncate(edge.context(), maxContextChars), edge.sourceDatasetIds(),
                edge.targetName(), edge.targetType(), edge.targetSourceDatasetIds(), forward);
    }

    /** Collect every entity on the paths, indexed by normalized name, for aligning LLM answers. */
    private static Map<String, List<AnswerEntity>> pathEntities(List<List<PathStep>> paths) {
        Map<String, List<AnswerEntity>> result = new LinkedHashMap<>();
        for (List<PathStep> path : paths) {
            for (PathStep step : path) {
                addAnswerEntity(result, new AnswerEntity(step.subject(), step.subjectType()));
                addAnswerEntity(result, new AnswerEntity(step.target(), step.targetType()));
            }
        }
        return result;
    }

    /**
     * Serialize path steps into prompt and result JSON. The extra traversal_from and traversal_to
     * fields show the actual direction so the model does not need to infer it from subject/target.
     */
    private static List<Map<String, Object>> serializeSteps(List<PathStep> path) {
        List<Map<String, Object>> result = new ArrayList<>();
        for (PathStep step : path) {
            Map<String, Object> serialized = new LinkedHashMap<>();
            serialized.put("subject", step.subject());
            serialized.put("subject_type", step.subjectType());
            serialized.put("subject_dataset_id_list", step.subjectDatasetIds());
            serialized.put("relation", step.relation());
            serialized.put("context", step.context());
            serialized.put("relation_dataset_id_list", step.relationDatasetIds());
            serialized.put("target", step.target());
            serialized.put("target_type", step.targetType());
            serialized.put("target_dataset_id_list", step.targetDatasetIds());
            serialized.put("traversed_forward", step.traversedForward());
            serialized.put("traversal_from", step.traversedForward()
                    ? step.subject() : step.target());
            serialized.put("traversal_to", step.traversedForward()
                    ? step.target() : step.subject());
            result.add(serialized);
        }
        return result;
    }

    /** Index entities by normalized name while retaining entities with the same name but different types. */
    private static void addAnswerEntity(Map<String, List<AnswerEntity>> result, AnswerEntity entity) {
        List<AnswerEntity> values = result.computeIfAbsent(normalizeAnswer(entity.name()),
                ignored -> new ArrayList<>());
        if (!values.contains(entity)) values.add(entity);
    }

    /** Parse the LLM type-prediction JSON: {"types":[{"type":..,"confidence":..,"reason":..}]}. */
    private static List<PredictedType> parsePredictedTypes(String raw, int limit) {
        List<PredictedType> result = new ArrayList<>();
        try {
            JsonArray values = parseObject(raw).getAsJsonArray("types");
            if (values == null) return result;
            for (JsonElement element : values) {
                if (!element.isJsonObject()) continue;
                JsonObject value = element.getAsJsonObject();
                String type = string(value, "type");
                if (type.isBlank()) continue;
                result.add(new PredictedType(normalize(type),
                        confidence(value), string(value, "reason")));
                if (result.size() >= limit) break;
            }
        } catch (RuntimeException ignored) { }
        return List.copyOf(result);
    }

    /** Generic score parser that extracts value-to-score pairs from {"<arrayName>":[{"<valueName>":..,"score":..}]}. */
    private static List<Scored> parseScores(String raw, String arrayName, String valueName) {
        List<Scored> result = new ArrayList<>();
        try {
            JsonArray values = parseObject(raw).getAsJsonArray(arrayName);
            if (values == null) return result;
            for (JsonElement element : values) {
                if (!element.isJsonObject()) continue;
                JsonObject value = element.getAsJsonObject();
                String name = string(value, valueName);
                JsonElement score = value.get("score");
                if (!name.isBlank() && score != null) result.add(new Scored(name, score.getAsDouble()));
            }
        } catch (RuntimeException ignored) { }
        return result;
    }

    /**
     * Align model scores to candidate order, retain scored candidates, and truncate to limit by
     * descending score. If no valid score is returned, assign uniform scores to the first limit
     * candidates so one malformed model response does not stop the layer.
     */
    private static List<Scored> normalizeScores(List<Scored> scores,
                                                List<String> candidates, int limit) {
        Map<String, Double> values = new HashMap<>();
        for (Scored score : scores) values.put(normalize(score.value()), clamp(score.score()));
        List<Scored> result = new ArrayList<>();
        for (String candidate : candidates) {
            Double score = values.get(normalize(candidate));
            if (score != null) result.add(new Scored(candidate, score));
        }
        if (result.isEmpty()) {
            double uniform = 1.0 / Math.max(1, Math.min(limit, candidates.size()));
            for (String candidate : candidates.stream().limit(limit).toList()) {
                result.add(new Scored(candidate, uniform));
            }
        }
        return result.stream().sorted(Comparator.comparingDouble(Scored::score).reversed())
                .limit(limit).toList();
    }

    /** Parse a string array in {"<name>":["a","b"]} form, deduplicating and discarding empty strings. */
    private static List<String> parseStringArray(String raw, String name) {
        try {
            JsonArray array = parseObject(raw).getAsJsonArray(name);
            if (array == null) return List.of();
            List<String> result = new ArrayList<>();
            for (JsonElement value : array) {
                if (value.isJsonPrimitive() && !value.getAsString().isBlank()) {
                    result.add(value.getAsString().trim());
                }
            }
            return result.stream().distinct().toList();
        } catch (RuntimeException ignored) {
            return List.of();
        }
    }

    /**
     * Extract a JSON object from a model response, tolerating surrounding explanatory text by
     * taking the substring from the first '{' through the last '}'. Throws when no valid object
     * exists; callers handle failures according to their own recovery policy.
     */
    private static JsonObject parseObject(String raw) {
        String value = raw == null ? "" : raw.trim();
        int start = value.indexOf('{');
        int end = value.lastIndexOf('}');
        if (start < 0 || end <= start) throw new IllegalArgumentException("LLM response is not JSON");
        JsonElement parsed = JsonParser.parseString(value.substring(start, end + 1));
        if (!parsed.isJsonObject()) throw new IllegalArgumentException("LLM response is not an object");
        return parsed.getAsJsonObject();
    }

    /** Safely read a string field; return an empty string when it is absent or not primitive. */
    private static String string(JsonObject value, String field) {
        JsonElement element = value.get(field);
        return element == null || !element.isJsonPrimitive() ? "" : element.getAsString().trim();
    }

    /** Read a JSON array as a string list, skipping non-string elements and empty strings. */
    private static List<String> strings(JsonElement value) {
        if (value == null || !value.isJsonArray()) return List.of();
        List<String> result = new ArrayList<>();
        for (JsonElement element : value.getAsJsonArray()) {
            if (element != null && element.isJsonPrimitive()) {
                String text = element.getAsString().trim();
                if (!text.isBlank()) result.add(text);
            }
        }
        return List.copyOf(result);
    }

    /** Read and clamp confidence to 0..1, defaulting to 0.5 when absent. */
    private static double confidence(JsonObject value) {
        try {
            JsonElement confidence = value.get("confidence");
            return confidence == null ? 0.5 : clamp(confidence.getAsDouble());
        } catch (RuntimeException ignored) {
            return 0.5;
        }
    }

    /** Trim and truncate to limit characters to bound prompt size. */
    private static String truncate(String value, int limit) {
        if (value == null) return "";
        String normalized = value.trim();
        return normalized.length() <= limit ? normalized : normalized.substring(0, limit);
    }

    /** Normalize by trimming, lowercasing, and collapsing whitespace for key matching and deduplication. */
    private static String normalize(String value) {
        return value == null ? "" : value.trim().toLowerCase(Locale.ROOT).replaceAll("\\s+", " ");
    }

    /** Normalize an answer with {@link #normalize} and strip leading/trailing punctuation for name comparison. */
    private static String normalizeAnswer(String value) {
        return normalize(value).replaceAll("^[\\p{Punct}]+|[\\p{Punct}]+$", "");
    }

    /** Clamp a score to 0..1; NaN and infinities become 0. */
    private static double clamp(double value) {
        return Double.isFinite(value) ? Math.max(0.0, Math.min(1.0, value)) : 0.0;
    }

    /** Return a sample identifier for logs, preferring the original ID and falling back to data_set row ID. */
    private static String sampleId(DataItem item) {
        String original = item.getOriginalId();
        return original == null || original.isBlank()
                ? Long.toString(item.getDatasetId()) : original;
    }

    /** Build a concise exception description (class name and message) to keep logs readable. */
    private static String concise(Throwable error) {
        String message = error.getMessage();
        return error.getClass().getSimpleName() + (message == null ? "" : ": " + message);
    }

    /** Convert a nanosecond start timestamp into elapsed milliseconds for logging. */
    private static long elapsedMs(long startedNanos) {
        return TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - startedNanos);
    }

    /** Validate that a string is neither null nor blank. */
    private static void requireText(String value, String name) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
    }

    /**
     * Parallel helper using a fixed thread pool: submit and collect results in input order.
     *
     * <p>The thread count is min(threadCount, task count) to avoid idle workers. Runtime exceptions
     * propagate so the caller can abort the batch or mark one sample failed. Shutdown waits 30 seconds
     * before forcing termination.</p>
     */
    private static <T> List<T> parallel(List<DataItem> items, int threadCount,
                                        Function<DataItem, T> operation) {
        if (items.isEmpty()) return List.of();
        ExecutorService executor = Executors.newFixedThreadPool(Math.min(threadCount, items.size()));
        try {
            List<Future<T>> futures = new ArrayList<>(items.size());
            for (DataItem item : items) futures.add(executor.submit(() -> operation.apply(item)));
            List<T> results = new ArrayList<>(items.size());
            for (Future<T> future : futures) {
                try {
                    // Collect in submission order so results align one-to-one with input samples.
                    results.add(future.get());
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException("parallel experiment interrupted", interrupted);
                } catch (ExecutionException failed) {
                    Throwable cause = failed.getCause();
                    if (cause instanceof RuntimeException runtime) throw runtime;
                    if (cause instanceof Error error) throw error;
                    throw new IllegalStateException("parallel experiment failed", cause);
                }
            }
            return List.copyOf(results);
        } finally {
            executor.shutdown();
            try {
                if (!executor.awaitTermination(30, TimeUnit.SECONDS)) executor.shutdownNow();
            } catch (InterruptedException interrupted) {
                executor.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
    }

    /** Null-safe immutable copy, converting null to an empty list. */
    private static <T> List<T> copy(List<T> values) {
        return values == null ? List.of() : List.copyOf(values);
    }

    /**
     * Provenance-vote collector that tracks entities and relations used during search and
     * aggregates votes by source sample ID for fallback retrieval.
     *
     * <p>Each entity or relation counts at most once per sample, preventing a frequently visited
     * entity from dominating the vote counts.</p>
     */
    private static final class EvidenceCollector {
        private final Map<String, List<Long>> entities = new LinkedHashMap<>();
        private final Map<String, List<Long>> relations = new LinkedHashMap<>();

        private void recordEntity(String entityId, List<Long> sourceIds) {
            entities.putIfAbsent(entityId, copy(sourceIds));
        }

        private void record(Neo4jClient.IncidentRelationEvidence edge) {
            // When an edge is adopted, record both endpoint entities and the relation.
            recordEntity(edge.subjectEntityId(), edge.subjectSourceDatasetIds());
            recordEntity(edge.targetEntityId(), edge.targetSourceDatasetIds());
            relations.putIfAbsent(edge.relationshipId(), copy(edge.sourceDatasetIds()));
        }

        private Map<Long, Integer> votes() {
            Map<Long, Integer> counts = new HashMap<>();
            // A relation is stronger provenance than an isolated entity because search actually
            // traversed it. Relations receive 2 votes and entities 1; each counts at most once per source.
            for (List<Long> ids : entities.values()) increment(counts, ids, 1);
            for (List<Long> ids : relations.values()) increment(counts, ids, 2);
            LinkedHashMap<Long, Integer> sorted = new LinkedHashMap<>();
            counts.entrySet().stream()
                    .sorted(Map.Entry.<Long, Integer>comparingByValue().reversed()
                            .thenComparing(Map.Entry.comparingByKey()))
                    .forEach(entry -> sorted.put(entry.getKey(), entry.getValue()));
            return Collections.unmodifiableMap(sorted);
        }

        private static void increment(Map<Long, Integer> counts, List<Long> ids, int weight) {
            if (ids == null) return;
            // Deduplicate before accumulation so each source contributes once per element.
            for (Long id : new LinkedHashSet<>(ids)) {
                if (id != null && id > 0) counts.merge(id, weight, Integer::sum);
            }
        }
    }

    /**
     * Command-line entry point that creates a new experiment or loads one with --experiment-id.
     *
     * <p>Common options are --threads, --build-graphs, --resume, and --rerun-reasoning.
     * Other options correspond one-to-one with {@link Config} fields and use
     * --name=value syntax.</p>
     */
    public static void main(String[] args) {
        configureDirectConnections();
        Config defaults = Config.defaults();
        // Use defaults() for every option not supplied on the command line.
        Config config = new Config(
                intArg(args, "width", defaults.width()),
                intArg(args, "depth", defaults.depth()),
                intArg(args, "max-topic-entities", defaults.maxTopicEntities()),
                intArg(args, "max-incident-relations", defaults.maxIncidentRelations()),
                intArg(args, "max-neighbours", defaults.maxNeighbours()),
                intArg(args, "predicted-type-limit", defaults.predictedTypeLimit()),
                intArg(args, "type-vector-candidates", defaults.typeVectorCandidates()),
                doubleArg(args, "type-max-distance", defaults.typeMaxDistance()),
                intArg(args, "type-graph-hops", defaults.typeGraphHops()),
                intArg(args, "type-graph-path-limit", defaults.typeGraphPathLimit()),
                intArg(args, "answer-type-limit", defaults.answerTypeLimit()),
                intArg(args, "max-edge-context-chars", defaults.maxEdgeContextChars()),
                intArg(args, "max-fallback-context-chars", defaults.maxFallbackContextChars()),
                intArg(args, "fallback-candidate-pool", defaults.fallbackCandidatePool()),
                intArg(args, "fallback-context-limit", defaults.fallbackContextLimit()),
                intArg(args, "fallback-type-match-limit", defaults.fallbackTypeMatchLimit()),
                doubleArg(args, "fallback-type-max-distance", defaults.fallbackTypeMaxDistance()),
                doubleArg(args, "fallback-type-weight", defaults.fallbackTypeWeight()),
                intArg(args, "fallback-type-graph-edges", defaults.fallbackTypeGraphEdgeLimit()),
                booleanArg(args, "fallback-llm-rerank", defaults.fallbackLlmRerank()),
                intArg(args, "fallback-llm-candidates", defaults.fallbackLlmCandidateLimit()),
                doubleArg(args, "fallback-llm-weight", defaults.fallbackLlmWeight()),
                booleanArg(args, "full-context-fallback", defaults.fullContextFallback()),
                booleanArg(args, "ground-answer-entity", defaults.groundAnswerEntity()),
                doubleArg(args, "ground-answer-min-score", defaults.groundAnswerMinScore()));
        int threads = intArg(args, "threads", DEFAULT_THREADS);
        // buildGraphs=true builds graphs before QA; false runs QA only on existing graphs.
        boolean buildGraphs = booleanArg(args, "build-graphs", true);
        // Whether to skip a sample whose graph construction fails; false aborts the whole batch.
        boolean skipBuildFailures = booleanArg(args, "skip-build-failures", true);
        // resume continues after interruption; rerun-reasoning clears results and reruns QA.
        // Both require an existing experiment ID.
        boolean resume = booleanArg(args, "resume", false);
        boolean rerunReasoning = booleanArg(args, "rerun-reasoning", false);
        String experimentOption = option(args, "experiment-id");
        if ((resume || rerunReasoning) && experimentOption == null) {
            throw new IllegalArgumentException("resume/rerun-reasoning requires an existing experiment-id");
        }
        // Load an existing experiment when --experiment-id is given; otherwise sample N questions from each dataset.
        DataLoader loader = experimentOption == null
                ? ExperimentInitializer.create(stringArg(args, "name", "ours-" + System.currentTimeMillis()),
                longArg(args, "seed", 2026912L), intArg(args, "hotpotqa", 1000),
                intArg(args, "musique", 1000), intArg(args, "2wikimultihopqa", 1000))
                : ExperimentInitializer.loadByExperimentId(Long.parseLong(experimentOption));
        // --graph-experiment-id queries another experiment's graph while writing results here.
        String graphExperimentOption = option(args, "graph-experiment-id");
        long graphExperimentId = graphExperimentOption == null ? -1L
                : Long.parseLong(graphExperimentOption.trim());
        // When reusing an external graph, declare it graph-ready because this experiment's namespace is empty.
        boolean assumeGraphBuilt = booleanArg(args, "assume-graph-built", graphExperimentId > 0);
        // Print key configuration values so experiment settings can be recovered from logs.
        System.out.printf("Our method starting: experiment=%d samples=%d threads=%d width=%d depth=%d " +
                        "buildGraphs=%s skipBuildFailures=%s resume=%s rerunReasoning=%s " +
                        "fallbackContextLimit=%d fullContext=%s groundAnswerEntity=%s groundMinScore=%.2f " +
                        "graphExperiment=%s assumeGraphBuilt=%s search=subquestion%n",
                loader.getExperiment().getId(), loader.getDataItemList().size(), threads,
                config.width(), config.depth(), buildGraphs, skipBuildFailures, resume, rerunReasoning,
                config.fallbackContextLimit(), config.fullContextFallback(),
                config.groundAnswerEntity(), config.groundAnswerMinScore(),
                graphExperimentId > 0 ? Long.toString(graphExperimentId) : "self", assumeGraphBuilt);
        try (loader; Neo4jClient neo4j = new Neo4jClient();
             AnswerChecker checker = new AnswerChecker()) {
            // try-with-resources closes the loader, Neo4j client, and answer checker automatically.
            LlmChatClient llm = new LlmChatClient();
            if (graphExperimentId > 0) {
                runWithExternalGraph(loader, llm, neo4j, checker, config,
                        buildGraphs, skipBuildFailures, resume, rerunReasoning,
                        threads, graphExperimentId, assumeGraphBuilt);
            } else {
                run(loader, llm, neo4j, checker, config,
                        buildGraphs, skipBuildFailures, resume, rerunReasoning, threads);
            }
        }
    }

    /** Run against an external graph while resolving source context from that graph corpus. */
    private static void runWithExternalGraph(DataLoader loader, LlmChatClient llm,
                                             Neo4jClient neo4j, AnswerChecker checker,
                                             Config config,
                                             boolean buildGraphs, boolean skipBuildFailures,
                                             boolean resume, boolean rerunReasoning, int threadCount,
                                             long graphExperimentId, boolean assumeGraphBuilt) {
        try (DataLoader graphLoader = ExperimentInitializer.loadByExperimentId(graphExperimentId)) {
            Map<Long, String> contextByDatasetId = new HashMap<>();
            for (DataItem item : graphLoader.getDataItemList()) {
                contextByDatasetId.put(item.getDatasetId(), item.getContext());
            }
            LongFunction<String> resolver = contextByDatasetId::get;
            run(loader, llm, neo4j, checker, config, buildGraphs, skipBuildFailures,
                    resume, rerunReasoning, threadCount, Long.toString(graphExperimentId),
                    assumeGraphBuilt, resolver);
        }
    }

    /** Read an integer option in --name=<int> form, returning fallback when absent. */
    /**
     * Route HTTP calls directly instead of through the OS-level proxy.
     *
     * <p>A machine-wide proxy may otherwise intercept every LLM request. This
     * runner uses a direct connection so deployment-specific proxy settings do
     * not alter benchmark behavior.</p>
     */
    private static void configureDirectConnections() {
        ProxySelector.setDefault(new ProxySelector() {
            @Override
            public List<Proxy> select(URI uri) {
                return List.of(Proxy.NO_PROXY);
            }

            @Override
            public void connectFailed(URI uri, SocketAddress sa, IOException ioe) {
                // The underlying client reports connection failures.
            }
        });
    }

    private static int intArg(String[] args, String name, int fallback) {
        String value = option(args, name);
        return value == null ? fallback : Integer.parseInt(value);
    }

    /** Read a long option in --name=<long> form. */
    private static long longArg(String[] args, String name, long fallback) {
        String value = option(args, name);
        return value == null ? fallback : Long.parseLong(value);
    }

    /** Read a double option in --name=<double> form. */
    private static double doubleArg(String[] args, String name, double fallback) {
        String value = option(args, name);
        return value == null ? fallback : Double.parseDouble(value);
    }

    /** Read a Boolean switch in --name=<true|false> form. */
    private static boolean booleanArg(String[] args, String name, boolean fallback) {
        String value = option(args, name);
        return value == null ? fallback : Boolean.parseBoolean(value);
    }

    /** Read a string option in --name=<string> form; blank values count as absent. */
    private static String stringArg(String[] args, String name, String fallback) {
        String value = option(args, name);
        return value == null || value.isBlank() ? fallback : value;
    }

    /** Minimal command-line parser supporting only --name=value, returning null when absent. */
    private static String option(String[] args, String name) {
        String prefix = "--" + name + "=";
        for (String arg : args) if (arg.startsWith(prefix)) return arg.substring(prefix.length());
        return null;
    }
}
