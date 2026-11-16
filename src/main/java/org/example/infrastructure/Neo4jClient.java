package org.example.infrastructure;

import org.example.config.AppConfig;
import org.example.entity.KnowledgeEntity;
import org.example.entity.KnowledgeGraphContext;
import org.example.entity.KnowledgeRelation;
import org.example.entity.KnowledgeTriple;
import org.example.utils.EntityRelationNormalizer;
import org.example.utils.json.KnowledgeTripleParser;
import org.neo4j.driver.AuthTokens;
import org.neo4j.driver.Driver;
import org.neo4j.driver.GraphDatabase;
import org.neo4j.driver.Record;
import org.neo4j.driver.Result;
import org.neo4j.driver.Session;
import org.neo4j.driver.SessionConfig;
import org.neo4j.driver.Values;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.Semaphore;
import java.util.concurrent.locks.ReentrantLock;

/** Reusable Neo4j driver wrapper. The driver is created and closed in one place. */
public final class Neo4jClient implements AutoCloseable {
    private static final String LEGACY_DATASET_ID = "__legacy__";
    private final AppConfig config;
    private final Driver driver;
    private final EmbeddingGemmaClient embeddings;
    private final int embeddingDimension;
    /** Reuses embeddings across samples and prevents duplicate in-flight work per text. */
    private final ConcurrentMap<String, List<Float>> embeddingCache = new ConcurrentHashMap<>();
    private final Semaphore embeddingPermits;
    private static final ConcurrentMap<String, ReentrantLock> GRAPH_WRITE_LOCKS =
            new ConcurrentHashMap<>();

    public Neo4jClient() {
        this(AppConfig.load());
    }

    public Neo4jClient(AppConfig config) {
        this.config = config;
        this.driver = GraphDatabase.driver(config.neo4jUri,
                AuthTokens.basic(config.neo4jUsername, config.neo4jPassword));
        this.embeddings = new EmbeddingGemmaClient(config);
        this.embeddingDimension = EmbeddingGemmaClient.DIMENSION;
        this.embeddingPermits = new Semaphore(config.embeddingMaxConcurrency);
        ensureVectorIndexes();
    }

    public Session openSession() {
        return driver.session(SessionConfig.forDatabase(config.neo4jDatabase));
    }

    /** Parses the LLM JSON array and merges all triples into one experiment-scoped graph. */
    public void storeTripleJson(String experimentId, String tripleJson) {
        storeTriples(experimentId, LEGACY_DATASET_ID, KnowledgeTripleParser.parse(tripleJson));
    }

    /** Parses and stores triples in one experiment and dataset scope. */
    public void storeTripleJson(String experimentId, String datasetId, String tripleJson) {
        storeTriples(experimentId, datasetId, KnowledgeTripleParser.parse(tripleJson));
    }

    /**
     * Parses and stores triples while recording the source {@code data_set}
     * row on every merged entity and relationship.
     */
    public void storeTripleJson(String experimentId, String graphDatasetId,
                                long sourceDatasetId, String tripleJson) {
        storeTriples(experimentId, graphDatasetId, sourceDatasetId,
                KnowledgeTripleParser.parse(tripleJson));
    }

    public void storeTripleJson(String experimentId, long datasetId, String tripleJson) {
        storeTripleJson(experimentId, Long.toString(datasetId), tripleJson);
    }

    /**
     * Merges triples in one transaction. Entity identity is
     * experiment + normalized name + normalized type, so repeated entities
     * across samples are reused rather than recreated.
     */
    public void storeTriples(String experimentId, List<KnowledgeTriple> triples) {
        storeTriples(experimentId, LEGACY_DATASET_ID, triples);
    }

    /**
     * Merges triples in one experiment and dataset scope. Entity identity is
     * experiment + dataset + normalized name + normalized type.
     */
    public void storeTriples(String experimentId, String datasetId, List<KnowledgeTriple> triples) {
        storeTriples(experimentId, datasetId, List.of(), triples);
    }

    /**
     * Merges triples into a shared graph namespace and records the source
     * {@code data_set.id} on each participating entity and relationship.
     * The source list is an idempotent set-like Neo4j {@code List<Long>}.
     */
    public void storeTriples(String experimentId, String graphDatasetId,
                             long sourceDatasetId, List<KnowledgeTriple> triples) {
        if (sourceDatasetId <= 0) {
            throw new IllegalArgumentException("sourceDatasetId must be positive");
        }
        storeTriples(experimentId, graphDatasetId, List.of(sourceDatasetId), triples);
    }

    /**
     * Replaces one sample's contribution instead of merging into an old
     * contribution. This is used by graph rebuilds after relation/entity
     * canonicalization so stale edges from an earlier build cannot survive.
     */
    public void replaceSampleTriples(String experimentId, String graphDatasetId,
                                     long sourceDatasetId, List<KnowledgeTriple> triples) {
        if (sourceDatasetId <= 0) {
            throw new IllegalArgumentException("sourceDatasetId must be positive");
        }
        String experiment = requireId(experimentId, "experimentId");
        String dataset = requireId(graphDatasetId, "datasetId");
        ReentrantLock lock = graphWriteLock(experiment, dataset);
        lock.lock();
        try {
            removeSampleEntityGraph(experiment, dataset, sourceDatasetId);
            storeTriples(experiment, dataset, sourceDatasetId, triples);
        } finally {
            lock.unlock();
        }
    }

    private static ReentrantLock graphWriteLock(String experimentId, String datasetId) {
        return GRAPH_WRITE_LOCKS.computeIfAbsent(
                experimentId + "\u0000" + datasetId, ignored -> new ReentrantLock());
    }

    /** Removes only one sample's contribution from the shared entity graph. */
    public void removeSampleEntityGraph(String experimentId, String graphDatasetId,
                                        long sourceDatasetId) {
        String experiment = requireId(experimentId, "experimentId");
        String dataset = requireId(graphDatasetId, "datasetId");
        if (sourceDatasetId <= 0) {
            throw new IllegalArgumentException("sourceDatasetId must be positive");
        }
        ReentrantLock lock = graphWriteLock(experiment, dataset);
        lock.lock();
        try {
            Map<String, Object> params = Values.parameters("experiment_id", experiment,
                    "dataset_id", dataset, "source_id", sourceDatasetId).asMap();
            String relationMatch = "MATCH (:Entity {experiment_id: $experiment_id, dataset_id: $dataset_id})"
                    + "-[r:RELATES_TO {experiment_id: $experiment_id, dataset_id: $dataset_id}]->"
                    + "(:Entity {experiment_id: $experiment_id, dataset_id: $dataset_id})";
            try (Session session = openSession()) {
                session.executeWrite(tx -> {
                    tx.run(relationMatch
                            + " WHERE $source_id IN coalesce(r.dataset_id_list, []) "
                            + "SET r.dataset_id_list = [x IN coalesce(r.dataset_id_list, []) WHERE x <> $source_id]",
                            params);
                    tx.run(relationMatch
                            + " WHERE size(coalesce(r.dataset_id_list, [])) = 0 DELETE r",
                            params);
                    tx.run("MATCH (e:Entity {experiment_id: $experiment_id, dataset_id: $dataset_id}) "
                            + "WHERE $source_id IN coalesce(e.dataset_id_list, []) "
                            + "SET e.dataset_id_list = [x IN coalesce(e.dataset_id_list, []) WHERE x <> $source_id]",
                            params);
                    tx.run("MATCH (e:Entity {experiment_id: $experiment_id, dataset_id: $dataset_id}) "
                            + "WHERE size(coalesce(e.dataset_id_list, [])) = 0 "
                            + "AND NOT (e)-[:RELATES_TO]-() DETACH DELETE e",
                            params);
                    return null;
                });
            }
        } finally {
            lock.unlock();
        }
    }

    private void storeTriples(String experimentId, String datasetId, List<Long> sourceDatasetIds,
                              List<KnowledgeTriple> triples) {
        String experiment = requireId(experimentId, "experimentId");
        String dataset = requireId(datasetId, "datasetId");
        Objects.requireNonNull(triples, "triples");
        if (triples.isEmpty()) return;
        List<Map<String, Object>> rows = new ArrayList<>(triples.size());
        Map<String, List<Float>> entityVectors = new LinkedHashMap<>();
        Map<String, List<Float>> relationVectors = new LinkedHashMap<>();
        for (KnowledgeTriple triple : triples) {
            Objects.requireNonNull(triple, "triple");
            String subject = requireText(triple.getSubject(), "subject");
            String subjectType = requireText(triple.getSubjectType(), "subject_type");
            String relation = EntityRelationNormalizer.normalizeRelation(
                    requireText(triple.getRelation(), "relation"));
            String target = requireText(triple.getTarget(), "target");
            String targetType = requireText(triple.getTargetType(), "target_type");
            Map<String, Object> row = new HashMap<>();
            row.put("subject", subject);
            row.put("subject_type", subjectType);
            row.put("subject_id", entityId(experiment, dataset, subject, subjectType));
            row.put("relation", relation);
            row.put("target", target);
            row.put("target_type", targetType);
            row.put("target_id", entityId(experiment, dataset, target, targetType));
            row.put("relation_context", triple.getRelationContext() == null ? "" : triple.getRelationContext().trim());
            row.put("source_dataset_ids", sourceDatasetIds);
            String subjectId = (String) row.get("subject_id");
            String targetId = (String) row.get("target_id");
            entityVectors.computeIfAbsent(subjectId, ignored -> embedding(subject));
            entityVectors.computeIfAbsent(targetId, ignored -> embedding(target));
            relationVectors.computeIfAbsent(relation, ignored -> embedding(relation));
            row.put("subject_embedding", entityVectors.get(subjectId));
            row.put("target_embedding", entityVectors.get(targetId));
            row.put("relation_embedding", relationVectors.get(relation));
            rows.add(row);
        }
        ReentrantLock lock = graphWriteLock(experiment, dataset);
        lock.lock();
        try {
            try (Session session = openSession()) {
                session.executeWrite(tx -> {
                tx.run("""
                        UNWIND $triples AS t
                        MERGE (s:Entity {entity_id: t.subject_id})
                        ON CREATE SET s.name = t.subject, s.type = t.subject_type, s.experiment_id = $experiment_id,
                                      s.dataset_id = $dataset_id,
                                      s.embedding = t.subject_embedding,
                                      s.dataset_id_list = coalesce(t.source_dataset_ids, [])
                        ON MATCH SET s.name = t.subject, s.type = t.subject_type, s.experiment_id = $experiment_id,
                                     s.dataset_id = $dataset_id,
                                     s.embedding = t.subject_embedding,
                                     s.dataset_id_list = reduce(ids = coalesce(s.dataset_id_list, []),
                                         sampleId IN coalesce(t.source_dataset_ids, []) |
                                         CASE WHEN sampleId IN ids THEN ids ELSE ids + sampleId END)
                        MERGE (o:Entity {entity_id: t.target_id})
                        ON CREATE SET o.name = t.target, o.type = t.target_type, o.experiment_id = $experiment_id,
                                      o.dataset_id = $dataset_id,
                                      o.embedding = t.target_embedding,
                                      o.dataset_id_list = coalesce(t.source_dataset_ids, [])
                        ON MATCH SET o.name = t.target, o.type = t.target_type, o.experiment_id = $experiment_id,
                                     o.dataset_id = $dataset_id,
                                     o.embedding = t.target_embedding,
                                     o.dataset_id_list = reduce(ids = coalesce(o.dataset_id_list, []),
                                         sampleId IN coalesce(t.source_dataset_ids, []) |
                                         CASE WHEN sampleId IN ids THEN ids ELSE ids + sampleId END)
                        MERGE (s)-[r:RELATES_TO {experiment_id: $experiment_id, dataset_id: $dataset_id,
                                                  name: t.relation,
                                                  subject_entity_id: t.subject_id,
                                                  target_entity_id: t.target_id}]->(o)
                        ON CREATE SET r.context = t.relation_context, r.embedding = t.relation_embedding,
                                      r.dataset_id_list = coalesce(t.source_dataset_ids, [])
                        ON MATCH SET r.embedding = t.relation_embedding,
                                     r.dataset_id_list = reduce(ids = coalesce(r.dataset_id_list, []),
                                         sampleId IN coalesce(t.source_dataset_ids, []) |
                                         CASE WHEN sampleId IN ids THEN ids ELSE ids + sampleId END),
                                     r.context_count = coalesce(r.context_count, 1)
                                         + CASE WHEN t.relation_context IS NULL
                                                     OR trim(t.relation_context) = ''
                                                     OR coalesce(r.context, '') CONTAINS t.relation_context
                                                THEN 0 ELSE 1 END,
                                     r.context = CASE
                          WHEN t.relation_context IS NULL OR trim(t.relation_context) = '' THEN coalesce(r.context, '')
                          WHEN coalesce(r.context, '') = '' THEN t.relation_context
                          WHEN r.context CONTAINS t.relation_context THEN r.context
                          WHEN t.relation_context CONTAINS r.context THEN t.relation_context
                          WHEN size(t.relation_context) > size(r.context) THEN t.relation_context
                          ELSE r.context END
                        """, Values.parameters("experiment_id", experiment, "dataset_id", dataset,
                                "triples", rows)).consume();
                    return null;
                });
            }
        } finally {
            lock.unlock();
        }
    }

    public void storeTriples(String experimentId, long datasetId, List<KnowledgeTriple> triples) {
        storeTriples(experimentId, Long.toString(datasetId), triples);
    }

    /**
     * Warms the shared embedding cache before a caller enters its ontology
     * commit lock. This changes only when the same embedding request runs, not
     * the type mapping or persistence semantics.
     */
    public void preloadEmbeddings(Iterable<String> texts) {
        Objects.requireNonNull(texts, "texts");
        for (String text : texts) {
            if (text != null && !text.isBlank()) embedding(text);
        }
    }

    /** Exposes one cached embedding for in-memory entity-similarity checks. */
    public List<Float> embeddingVector(String text) {
        return embedding(text);
    }

    /** Persists extracted entity types and their vectors. */
    public void storeEntityTypes(String experimentId, String datasetId, long sourceDatasetId,
                                 List<String> types) {
        String experiment = requireId(experimentId, "experimentId");
        String dataset = requireId(datasetId, "datasetId");
        if (sourceDatasetId <= 0) throw new IllegalArgumentException("sourceDatasetId must be positive");
        Objects.requireNonNull(types, "types");

        List<Map<String, Object>> typeRows = new ArrayList<>();
        for (String type : new java.util.LinkedHashSet<>(types)) {
            if (type == null || type.isBlank()) continue;
            String normalized = normalizeType(type);
            Map<String, Object> row = new HashMap<>();
            row.put("type_id", typeId(experiment, dataset, normalized));
            row.put("name", normalized);
            row.put("source_dataset_ids", List.of(sourceDatasetId));
            row.put("embedding", embedding(normalized));
            typeRows.add(row);
        }
        if (typeRows.isEmpty()) return;

        try (Session session = openSession()) {
            session.executeWrite(tx -> {
                tx.run("""
                        UNWIND  AS type
                        MERGE (node:EntityType {type_id: type.type_id})
                        ON CREATE SET node.name = type.name, node.normalized_name = type.name,
                                      node.experiment_id = , node.dataset_id = ,
                                      node.aliases = [type.name],
                                      node.dataset_id_list = type.source_dataset_ids,
                                      node.embedding = type.embedding, node.active = true
                        ON MATCH SET node.aliases = reduce(values = coalesce(node.aliases, []),
                                         alias IN [type.name] |
                                         CASE WHEN alias IN values THEN values ELSE values + alias END),
                                     node.dataset_id_list = reduce(ids = coalesce(node.dataset_id_list, []),
                                         sampleId IN type.source_dataset_ids |
                                         CASE WHEN sampleId IN ids THEN ids ELSE ids + sampleId END),
                                     node.embedding = type.embedding,
                                     node.active = true
                        """, Values.parameters("experiment_id", experiment, "dataset_id", dataset,
                        "types", typeRows)).consume();
                return null;
            });
        }
    }

    /** Vector-searches active canonical types inside one ontology namespace. */
    public List<EntityTypeSearchResult> searchEntityTypesWithinDistance(String experimentId,
                                                                         String datasetId,
                                                                         String text, int topK,
                                                                         double maxDistance) {
        String experiment = requireId(experimentId, "experimentId");
        String dataset = requireId(datasetId, "datasetId");
        String queryText = requireText(text, "text");
        checkTopK(topK);
        checkDistance(maxDistance);
        List<Float> vector = embedding(queryText);
        try (Session session = openSession()) {
            long count = session.run("MATCH (n:EntityType) WHERE n.embedding IS NOT NULL RETURN count(n) AS count")
                    .single().get("count").asLong();
            if (count == 0) return List.of();
            int candidateK = (int) Math.min(Integer.MAX_VALUE, Math.max((long) topK, count));
            List<EntityTypeSearchResult> result = new ArrayList<>();
            for (Record record : session.run("""
                    CALL db.index.vector.queryNodes($index_name, $candidate_k, $embedding)
                    YIELD node, score
                    WHERE node.experiment_id = $experiment_id
                      AND node.dataset_id = $dataset_id
                      AND coalesce(node.active, true) = true
                      AND (1.0 - score) <= $max_distance
                    RETURN node.type_id AS type_id, node.name AS name,
                           coalesce(node.aliases, []) AS aliases, score
                    ORDER BY score DESC LIMIT $top_k
                    """, Values.parameters("index_name", config.neo4jTypeVectorIndex,
                    "candidate_k", candidateK, "embedding", vector,
                    "experiment_id", experiment, "dataset_id", dataset,
                    "max_distance", maxDistance, "top_k", topK)).list()) {
                result.add(new EntityTypeSearchResult(record.get("type_id").asString(),
                        record.get("name").asString(),
                        record.get("aliases").asList(value -> value.asString()),
                        experiment, dataset, record.get("score").asDouble()));
            }
            return List.copyOf(result);
        }
    }

    /** Returns incoming and outgoing ontology edges around recalled type nodes. */
    public List<EntitySearchResult> searchEntities(String experimentId, String text, int topK) {
        return searchEntitiesInternal(experimentId, null, text, topK, null);
    }

    /** Searches entity names inside one experiment and one dataset. */
    public List<EntitySearchResult> searchEntities(String experimentId, String datasetId,
                                                   String text, int topK) {
        return searchEntitiesInternal(experimentId, datasetId, text, topK, null);
    }

    public List<EntitySearchResult> searchEntities(String experimentId, long datasetId,
                                                   String text, int topK) {
        return searchEntities(experimentId, Long.toString(datasetId), text, topK);
    }

    /**
     * Lexical entity lookup used as a deterministic fallback for topic
     * extraction. It is deliberately dataset-scoped and does not depend on
     * the embedding service or vector-index availability.
     */
    public List<EntitySearchResult> searchEntitiesByName(String experimentId, String datasetId,
                                                         String text, int topK) {
        String experiment = requireId(experimentId, "experimentId");
        String dataset = requireId(datasetId, "datasetId");
        String queryText = requireText(text, "text");
        checkTopK(topK);
        try (Session session = openSession()) {
            List<EntitySearchResult> result = new ArrayList<>();
            for (Record record : session.run("""
                    MATCH (n:Entity {experiment_id: $experiment_id, dataset_id: $dataset_id})
                    WHERE toLower(n.name) = toLower($text)
                       OR toLower(n.name) CONTAINS toLower($text)
                       OR toLower($text) CONTAINS toLower(n.name)
                    RETURN n.entity_id AS id, n.name AS name, n.type AS type,
                           n.dataset_id AS dataset_id
                    ORDER BY size(n.name) DESC, n.entity_id
                    LIMIT $top_k
                    """, Values.parameters("experiment_id", experiment, "dataset_id", dataset,
                    "text", queryText, "top_k", topK)).list()) {
                result.add(new EntitySearchResult(record.get("id").asString(), record.get("name").asString(),
                        record.get("type").asString(""), experiment,
                        record.get("dataset_id").asString(dataset), 1.0));
            }
            return result;
        }
    }

    public List<EntitySearchResult> searchEntitiesByName(String experimentId, long datasetId,
                                                         String text, int topK) {
        return searchEntitiesByName(experimentId, Long.toString(datasetId), text, topK);
    }

    /** Alias for {@link #searchEntities(String, String, int)}. */
    public List<EntitySearchResult> vectorSearchEntities(String experimentId, String text, int topK) {
        return searchEntities(experimentId, text, topK);
    }

    public List<EntitySearchResult> vectorSearchEntities(String experimentId, String datasetId,
                                                         String text, int topK) {
        return searchEntities(experimentId, datasetId, text, topK);
    }

    /**
     * Searches entity names and returns only results whose cosine distance is
     * no greater than {@code maxDistance}. Results remain capped by topK.
     */
    public List<EntitySearchResult> searchEntitiesWithinDistance(String experimentId, String text,
                                                                  int topK, double maxDistance) {
        return searchEntitiesInternal(experimentId, null, text, topK, maxDistance);
    }

    /** Searches entities in one dataset with a cosine-distance threshold. */
    public List<EntitySearchResult> searchEntitiesWithinDistance(String experimentId, String datasetId,
                                                                  String text, int topK, double maxDistance) {
        return searchEntitiesInternal(experimentId, datasetId, text, topK, maxDistance);
    }

    public List<EntitySearchResult> searchEntitiesWithinDistance(String experimentId, long datasetId,
                                                                  String text, int topK, double maxDistance) {
        return searchEntitiesWithinDistance(experimentId, Long.toString(datasetId), text, topK, maxDistance);
    }

    private List<EntitySearchResult> searchEntitiesInternal(String experimentId, String datasetId,
                                                             String text, int topK, Double maxDistance) {
        String experiment = requireId(experimentId, "experimentId");
        String dataset = datasetId == null ? null : requireId(datasetId, "datasetId");
        String queryText = requireText(text, "text");
        checkTopK(topK);
        if (maxDistance != null) checkDistance(maxDistance);
        List<Float> vector = embedding(queryText);
        try (Session session = openSession()) {
// Candidate window: when graph.entity.candidate.limit > 0, return only a bounded set of
// nearest neighbours (the default 0 preserves the old behavior and uses the full entity count).
// After the entity index grew to millions of entries, every entity-alignment query degraded
// into a full vector scan. This directly increased graph construction time from about 3 minutes
// to 10-20 minutes per sample. Filtering is still applied to retrieval results, so semantics are unchanged.
            int candidateLimit = Integer.getInteger("graph.entity.candidate.limit", 0);
            long count;
            if (candidateLimit > 0) {
// With a candidate window, a full entity-count query is unnecessary. That count scans every
// entity node (currently millions) only to bound the candidate count; the window value can be
// used instead, avoiding a full scan for each entity.
                count = candidateLimit;
            } else {
                count = session.run("MATCH (n:Entity) WHERE n.embedding IS NOT NULL RETURN count(n) AS count")
                        .single().get("count").asLong();
                if (count == 0) return List.of();
            }
            long window = candidateLimit > 0 ? Math.max(topK, candidateLimit) : count;
            int candidateK = (int) Math.min(Integer.MAX_VALUE, Math.max((long) topK, Math.min(count, window)));
            String datasetFilter = dataset == null ? "" : " AND node.dataset_id = $dataset_id";
            String distanceFilter = maxDistance == null ? "" : " AND (1.0 - score) <= $max_distance";
            String cypher = "CALL db.index.vector.queryNodes($index_name, $candidate_k, $embedding) "
                    + "YIELD node, score WHERE node.experiment_id = $experiment_id"
                    + datasetFilter + distanceFilter
                    + " RETURN node.entity_id AS id, node.name AS name, node.type AS type, "
                    + "node.dataset_id AS dataset_id, score "
                    + "ORDER BY score DESC LIMIT $top_k";
            Map<String, Object> params = new HashMap<>(Values.parameters("index_name", config.neo4jEntityVectorIndex,
                    "candidate_k", candidateK, "embedding", vector, "experiment_id", experiment,
                    "top_k", topK).asMap());
            if (dataset != null) params.put("dataset_id", dataset);
            if (maxDistance != null) params.put("max_distance", maxDistance);
            List<EntitySearchResult> result = new ArrayList<>();
            for (Record record : session.run(cypher, params).list()) {
                result.add(new EntitySearchResult(record.get("id").asString(), record.get("name").asString(),
                        record.get("type").asString(""), experiment,
                        record.get("dataset_id").asString(dataset == null ? "" : dataset),
                        record.get("score").asDouble()));
            }
            return result;
        }
    }

    /** Alias for {@link #searchEntitiesWithinDistance(String, String, int, double)}. */
    public List<EntitySearchResult> vectorSearchEntitiesWithinDistance(String experimentId, String text,
                                                                        int topK, double maxDistance) {
        return searchEntitiesWithinDistance(experimentId, text, topK, maxDistance);
    }

    public List<EntitySearchResult> vectorSearchEntitiesWithinDistance(String experimentId, String datasetId,
                                                                        String text, int topK, double maxDistance) {
        return searchEntitiesWithinDistance(experimentId, datasetId, text, topK, maxDistance);
    }

    /** Removes only vector properties for one experiment and keeps the graph text. */
    public long deleteExperimentVectors(String experimentId) {
        return deleteExperimentVectorsInternal(experimentId, null);
    }

    /** Removes only vector properties for one experiment and dataset. */
    public long deleteExperimentVectors(String experimentId, String datasetId) {
        return deleteExperimentVectorsInternal(experimentId, datasetId);
    }

    public long deleteExperimentVectors(String experimentId, long datasetId) {
        return deleteExperimentVectors(experimentId, Long.toString(datasetId));
    }

    private long deleteExperimentVectorsInternal(String experimentId, String datasetId) {
        String experiment = requireId(experimentId, "experimentId");
        String dataset = datasetId == null ? null : requireId(datasetId, "datasetId");
        try (Session session = openSession()) {
            return session.executeWrite(tx -> {
                String entityDataset = dataset == null ? "" : " AND n.dataset_id = $dataset_id";
                String relationDataset = dataset == null ? "" : " AND r.dataset_id = $dataset_id";
                Map<String, Object> params = new HashMap<>(Values.parameters("experiment_id", experiment).asMap());
                if (dataset != null) params.put("dataset_id", dataset);
                long entities = tx.run("MATCH (n:Entity {experiment_id: $experiment_id}) "
                                + "WHERE n.embedding IS NOT NULL" + entityDataset
                                + " REMOVE n.embedding RETURN count(n) AS deleted",
                        params).single().get("deleted").asLong();
                long relations = tx.run("MATCH ()-[r:RELATES_TO {experiment_id: $experiment_id}]->() "
                                + "WHERE r.embedding IS NOT NULL" + relationDataset
                                + " REMOVE r.embedding RETURN count(r) AS deleted",
                        params).single().get("deleted").asLong();
                return entities + relations;
            });
        }
    }

    /** Alias for {@link #deleteExperimentVectors(String)}. */
    public long deleteExperimentVectorData(String experimentId) {
        return deleteExperimentVectors(experimentId);
    }

    public long deleteExperimentVectorData(String experimentId, String datasetId) {
        return deleteExperimentVectors(experimentId, datasetId);
    }

    /** Removes one entity's vector while leaving its node and relationships intact. */
    public long deleteEntityVector(String experimentId, String entityName) {
        return deleteEntityVectorInternal(experimentId, null, entityName);
    }

    /** Removes one dataset-scoped entity vector while leaving graph facts intact. */
    public long deleteEntityVector(String experimentId, String datasetId, String entityName) {
        return deleteEntityVectorInternal(experimentId, datasetId, entityName);
    }

    public long deleteEntityVector(String experimentId, long datasetId, String entityName) {
        return deleteEntityVector(experimentId, Long.toString(datasetId), entityName);
    }

    private long deleteEntityVectorInternal(String experimentId, String datasetId, String entityName) {
        String experiment = requireId(experimentId, "experimentId");
        String dataset = datasetId == null ? null : requireId(datasetId, "datasetId");
        String name = requireText(entityName, "entityName");
        try (Session session = openSession()) {
            String datasetFilter = dataset == null ? "" : " AND n.dataset_id = $dataset_id";
            Map<String, Object> params = new HashMap<>(Values.parameters("experiment_id", experiment,
                    "name", name).asMap());
            if (dataset != null) params.put("dataset_id", dataset);
            return session.executeWrite(tx -> tx.run("MATCH (n:Entity {experiment_id: $experiment_id}) "
                            + "WHERE toLower(n.name) = toLower($name) AND n.embedding IS NOT NULL"
                            + datasetFilter + " REMOVE n.embedding RETURN count(n) AS deleted", params)
                    .single().get("deleted").asLong());
        }
    }

    /** Removes vectors from all relationships with one name in an experiment. */
    public long deleteRelationVector(String experimentId, String relationName) {
        return deleteRelationVectorInternal(experimentId, null, relationName);
    }

    /** Removes dataset-scoped vectors from relationships with one name. */
    public long deleteRelationVector(String experimentId, String datasetId, String relationName) {
        return deleteRelationVectorInternal(experimentId, datasetId, relationName);
    }

    public long deleteRelationVector(String experimentId, long datasetId, String relationName) {
        return deleteRelationVector(experimentId, Long.toString(datasetId), relationName);
    }

    private long deleteRelationVectorInternal(String experimentId, String datasetId, String relationName) {
        String experiment = requireId(experimentId, "experimentId");
        String dataset = datasetId == null ? null : requireId(datasetId, "datasetId");
        String name = requireText(relationName, "relationName");
        try (Session session = openSession()) {
            String datasetFilter = dataset == null ? "" : " AND r.dataset_id = $dataset_id";
            Map<String, Object> params = new HashMap<>(Values.parameters("experiment_id", experiment,
                    "name", name).asMap());
            if (dataset != null) params.put("dataset_id", dataset);
            return session.executeWrite(tx -> tx.run("MATCH ()-[r:RELATES_TO {experiment_id: $experiment_id}]->() "
                            + "WHERE toLower(r.name) = toLower($name) AND r.embedding IS NOT NULL"
                            + datasetFilter + " REMOVE r.embedding RETURN count(r) AS deleted", params)
                    .single().get("deleted").asLong());
        }
    }

    /** Returns all entities and relationships stored for one experiment. */
    public KnowledgeGraphContext queryGraph(String experimentId, int limit) {
        return queryGraphInternal(requireId(experimentId, "experimentId"), null, null, null, limit);
    }

    /** Returns all entities and relationships stored for one dataset in an experiment. */
    public KnowledgeGraphContext queryGraph(String experimentId, String datasetId, int limit) {
        return queryGraphInternal(requireId(experimentId, "experimentId"), requireId(datasetId, "datasetId"),
                null, null, limit);
    }

    public KnowledgeGraphContext queryGraph(String experimentId, long datasetId, int limit) {
        return queryGraph(experimentId, Long.toString(datasetId), limit);
    }

    /** Returns the local graph neighborhood around an entity name in one experiment. */
    public KnowledgeGraphContext queryAroundEntity(String experimentId, String entityName, int limit) {
        return queryGraphInternal(requireId(experimentId, "experimentId"), null,
                requireText(entityName, "entityName"), null, limit);
    }

    /** Returns a dataset-scoped neighborhood around an entity name. */
    public KnowledgeGraphContext queryAroundEntityInDataset(String experimentId, String datasetId,
                                                            String entityName, int limit) {
        return queryGraphInternal(requireId(experimentId, "experimentId"), requireId(datasetId, "datasetId"),
                requireText(entityName, "entityName"), null, limit);
    }

    public KnowledgeGraphContext queryAroundEntityInDataset(String experimentId, long datasetId,
                                                            String entityName, int limit) {
        return queryAroundEntityInDataset(experimentId, Long.toString(datasetId), entityName, limit);
    }

    /**
     * Returns the directed facts incident to one exact entity id.  ToG needs
     * the direction and the neighbour id while expanding a beam; querying by
     * id also avoids ambiguities when two entities share the same display name.
     */
    public List<KnowledgeRelation> queryIncidentRelations(String experimentId, String datasetId,
                                                             String entityId, int limit) {
        String experiment = requireId(experimentId, "experimentId");
        String dataset = requireId(datasetId, "datasetId");
        String id = requireText(entityId, "entityId");
        if (limit < 1) throw new IllegalArgumentException("limit must be positive");
        try (Session session = openSession()) {
            List<KnowledgeRelation> result = new ArrayList<>();
            List<Record> records = session.run("""
                    MATCH (s:Entity {experiment_id: $experiment_id, dataset_id: $dataset_id})
                          -[r:RELATES_TO {experiment_id: $experiment_id, dataset_id: $dataset_id}]->
                          (t:Entity {experiment_id: $experiment_id, dataset_id: $dataset_id})
                    WHERE s.entity_id = $entity_id OR t.entity_id = $entity_id
                    RETURN s.entity_id AS sid, t.entity_id AS tid,
                           s.name AS sname, t.name AS tname,
                           r.name AS name, r.context AS context
                    ORDER BY r.name, s.entity_id, t.entity_id
                    LIMIT $limit
                    """, Values.parameters("experiment_id", experiment, "dataset_id", dataset,
                    "entity_id", id, "limit", limit)).list();
            for (Record record : records) {
                result.add(new KnowledgeRelation(record.get("name").asString(),
                        record.get("context").asString(""), experiment, dataset,
                        record.get("sid").asString(), record.get("tid").asString(),
                        record.get("sname").asString(), record.get("tname").asString()));
            }
            return result;
        }
    }

    /**
     * Returns incident facts together with entity types and sample provenance.
     */
    public List<IncidentRelationEvidence> queryIncidentRelationEvidence(String experimentId,
                                                                         String datasetId,
                                                                         String entityId,
                                                                         int limit) {
        String experiment = requireId(experimentId, "experimentId");
        String dataset = requireId(datasetId, "datasetId");
        String id = requireText(entityId, "entityId");
        checkTopK(limit);
        try (Session session = openSession()) {
            List<IncidentRelationEvidence> result = new ArrayList<>();
            for (Record record : session.run("""
                    MATCH (s:Entity {experiment_id: $experiment_id, dataset_id: $dataset_id})
                          -[r:RELATES_TO {experiment_id: $experiment_id, dataset_id: $dataset_id}]->
                          (t:Entity {experiment_id: $experiment_id, dataset_id: $dataset_id})
                    WHERE s.entity_id = $entity_id OR t.entity_id = $entity_id
                    RETURN elementId(r) AS relationship_id,
                           s.entity_id AS sid, s.name AS sname, s.type AS stype,
                           coalesce(s.dataset_id_list, []) AS s_dataset_ids,
                           t.entity_id AS tid, t.name AS tname, t.type AS ttype,
                           coalesce(t.dataset_id_list, []) AS t_dataset_ids,
                           r.name AS name, coalesce(r.context, '') AS context,
                           coalesce(r.dataset_id_list, []) AS r_dataset_ids
                    ORDER BY r.name, s.entity_id, t.entity_id LIMIT $limit
                    """, Values.parameters("experiment_id", experiment, "dataset_id", dataset,
                    "entity_id", id, "limit", limit)).list()) {
                result.add(new IncidentRelationEvidence(
                        record.get("relationship_id").asString(), record.get("name").asString(),
                        record.get("context").asString(""),
                        record.get("sid").asString(), record.get("sname").asString(),
                        record.get("stype").asString(""),
                        record.get("s_dataset_ids").asList(value -> value.asLong()),
                        record.get("tid").asString(), record.get("tname").asString(),
                        record.get("ttype").asString(""),
                        record.get("t_dataset_ids").asList(value -> value.asLong()),
                        record.get("r_dataset_ids").asList(value -> value.asLong()),
                        experiment, dataset));
            }
            return List.copyOf(result);
        }
    }

    /** Returns the source data_set rows that contributed one entity node. */
    public List<Long> queryEntitySourceDatasetIds(String experimentId, String datasetId,
                                                   String entityId) {
        String experiment = requireId(experimentId, "experimentId");
        String dataset = requireId(datasetId, "datasetId");
        String id = requireText(entityId, "entityId");
        try (Session session = openSession()) {
            Record record = session.run("""
                    MATCH (entity:Entity {experiment_id: $experiment_id,
                                          dataset_id: $dataset_id,
                                          entity_id: $entity_id})
                    RETURN coalesce(entity.dataset_id_list, []) AS dataset_ids
                    """, Values.parameters("experiment_id", experiment, "dataset_id", dataset,
                    "entity_id", id)).single();
            return List.copyOf(record.get("dataset_ids").asList(value -> value.asLong()));
        }
    }

    /**
     * Builds per-sample type/entity subgraphs from the shared entity graph.
     * No duplicate graph is stored: dataset_id_list induces the exact nodes
     * and facts contributed by each source sample.
     */
    public List<SampleEntityTypeGraph> querySampleEntityTypeGraphs(String experimentId,
                                                                   String datasetId,
                                                                   List<Long> sourceDatasetIds,
                                                                   int edgeLimitPerSample) {
        String experiment = requireId(experimentId, "experimentId");
        String dataset = requireId(datasetId, "datasetId");
        Objects.requireNonNull(sourceDatasetIds, "sourceDatasetIds");
        checkTopK(edgeLimitPerSample);
        List<Long> sourceIds = sourceDatasetIds.stream().filter(Objects::nonNull)
                .filter(id -> id > 0).distinct().toList();
        if (sourceIds.isEmpty()) return List.of();

        Map<Long, List<SampleEntityTypeNode>> nodesBySource = new LinkedHashMap<>();
        Map<Long, List<SampleEntityTypeEdge>> edgesBySource = new LinkedHashMap<>();
        try (Session session = openSession()) {
            for (Record record : session.run("""
                    UNWIND $source_ids AS source_id
                    MATCH (entity:Entity {experiment_id: $experiment_id, dataset_id: $dataset_id})
                    WHERE source_id IN coalesce(entity.dataset_id_list, [])
                    WITH source_id, toLower(trim(entity.type)) AS entity_type,
                         count(DISTINCT entity) AS entity_count
                    WHERE entity_type <> ''
                    RETURN source_id, entity_type, entity_count
                    ORDER BY source_id, entity_count DESC, entity_type
                    """, Values.parameters("experiment_id", experiment, "dataset_id", dataset,
                    "source_ids", sourceIds)).list()) {
                long sourceId = record.get("source_id").asLong();
                nodesBySource.computeIfAbsent(sourceId, ignored -> new ArrayList<>())
                        .add(new SampleEntityTypeNode(record.get("entity_type").asString(),
                                record.get("entity_count").asLong()));
            }
            for (Record record : session.run("""
                    UNWIND $source_ids AS source_id
                    CALL (source_id) {
                        MATCH (subject:Entity {experiment_id: $experiment_id, dataset_id: $dataset_id})
                              -[relation:RELATES_TO {experiment_id: $experiment_id,
                                                     dataset_id: $dataset_id}]->
                              (target:Entity {experiment_id: $experiment_id, dataset_id: $dataset_id})
                        WHERE source_id IN coalesce(relation.dataset_id_list, [])
                        WITH toLower(trim(subject.type)) AS source_type,
                             relation.name AS relation_name,
                             toLower(trim(target.type)) AS target_type,
                             count(DISTINCT relation) AS fact_count
                        RETURN source_type, relation_name, target_type, fact_count
                        ORDER BY fact_count DESC, source_type, relation_name, target_type
                        LIMIT $edge_limit
                    }
                    RETURN source_id, source_type, relation_name, target_type, fact_count
                    ORDER BY source_id, fact_count DESC
                    """, Values.parameters("experiment_id", experiment, "dataset_id", dataset,
                    "source_ids", sourceIds, "edge_limit", edgeLimitPerSample)).list()) {
                long sourceId = record.get("source_id").asLong();
                edgesBySource.computeIfAbsent(sourceId, ignored -> new ArrayList<>())
                        .add(new SampleEntityTypeEdge(record.get("source_type").asString(),
                                record.get("relation_name").asString(),
                                record.get("target_type").asString(),
                                record.get("fact_count").asLong()));
            }
        }
        List<SampleEntityTypeGraph> result = new ArrayList<>(sourceIds.size());
        for (Long sourceId : sourceIds) {
            result.add(new SampleEntityTypeGraph(sourceId,
                    nodesBySource.getOrDefault(sourceId, List.of()),
                    edgesBySource.getOrDefault(sourceId, List.of()), experiment, dataset));
        }
        return List.copyOf(result);
    }

    /** Vector-matches answer types against entity types present in candidate sample subgraphs. */
    public List<SampleEntityTypeMatch> matchSampleEntityTypes(String experimentId,
                                                               String datasetId,
                                                               List<Long> sourceDatasetIds,
                                                               List<String> answerTypes,
                                                               int limitPerAnswerType,
                                                               double maxDistance) {
        String experiment = requireId(experimentId, "experimentId");
        String dataset = requireId(datasetId, "datasetId");
        Objects.requireNonNull(sourceDatasetIds, "sourceDatasetIds");
        Objects.requireNonNull(answerTypes, "answerTypes");
        checkTopK(limitPerAnswerType);
        checkDistance(maxDistance);
        List<Long> sourceIds = sourceDatasetIds.stream().filter(Objects::nonNull)
                .filter(id -> id > 0).distinct().toList();
        List<String> queries = answerTypes.stream().filter(Objects::nonNull)
                .map(String::trim).filter(value -> !value.isBlank()).distinct().toList();
        if (sourceIds.isEmpty() || queries.isEmpty()) return List.of();

        List<SampleEntityTypeMatch> result = new ArrayList<>();
        try (Session session = openSession()) {
            for (String answerType : queries) {
                List<Float> vector = embedding(answerType);
                for (Record record : session.run("""
                        UNWIND $source_ids AS source_id
                        CALL (source_id) {
                            MATCH (entity:Entity {experiment_id: $experiment_id,
                                                  dataset_id: $dataset_id})
                            WHERE source_id IN coalesce(entity.dataset_id_list, [])
                            WITH toLower(trim(entity.type)) AS sample_type,
                                 count(DISTINCT entity) AS entity_count
                            MATCH (type:EntityType {experiment_id: $experiment_id,
                                                    dataset_id: $dataset_id,
                                                    name: sample_type})
                            WHERE type.embedding IS NOT NULL
                            WITH sample_type, entity_count,
                                 vector.similarity.cosine(type.embedding, $embedding) AS score
                            WHERE (1.0 - score) <= $max_distance
                            RETURN sample_type, entity_count, score
                            ORDER BY score DESC, entity_count DESC, sample_type
                            LIMIT $match_limit
                        }
                        RETURN source_id, sample_type, entity_count, score
                        ORDER BY source_id, score DESC
                        """, Values.parameters("experiment_id", experiment, "dataset_id", dataset,
                        "source_ids", sourceIds, "embedding", vector,
                        "max_distance", maxDistance, "match_limit", limitPerAnswerType)).list()) {
                    double score = record.get("score").asDouble();
                    result.add(new SampleEntityTypeMatch(record.get("source_id").asLong(),
                            answerType, record.get("sample_type").asString(),
                            record.get("entity_count").asLong(), score, 1.0 - score));
                }
            }
        }
        return List.copyOf(result);
    }

    /**
     * Retrieves real paths in the type graph around vector-recalled seed types.
     * Both hierarchy directions are included because a predicted answer type
     * can be either more general or more specific than the answer entity.
     */
    public List<EntityTypePathResult> queryEntityTypePaths(String experimentId, String datasetId,
                                                           List<String> seedTypeNames,
                                                           int maxHops, int limit) {
        String experiment = requireId(experimentId, "experimentId");
        String dataset = requireId(datasetId, "datasetId");
        Objects.requireNonNull(seedTypeNames, "seedTypeNames");
        if (maxHops < 0 || maxHops > 10) {
            throw new IllegalArgumentException("maxHops must be between 0 and 10");
        }
        checkTopK(limit);
        List<String> names = seedTypeNames.stream().filter(Objects::nonNull)
                .map(Neo4jClient::normalizeType).distinct().toList();
        if (names.isEmpty()) return List.of();
        String cypher = "MATCH (seed:EntityType {experiment_id: $experiment_id, dataset_id: $dataset_id}) "
                + "WHERE seed.name IN $seed_names "
                + "MATCH path = (seed)-[:SUBTYPE_OF|EQUIVALENT_TO|TYPE_RELATED_TO*0.."
                + maxHops + "]-(candidate:EntityType {experiment_id: $experiment_id, dataset_id: $dataset_id}) "
                + "WHERE all(node IN nodes(path) WHERE node.experiment_id = $experiment_id "
                + "AND node.dataset_id = $dataset_id) AND all(edge IN relationships(path) "
                + "WHERE edge.experiment_id = $experiment_id AND edge.dataset_id = $dataset_id) "
                + "RETURN seed.name AS seed, [node IN nodes(path) | node.name] AS type_path, "
                + "[edge IN relationships(path) | CASE type(edge) WHEN 'TYPE_RELATED_TO' THEN 'related_to' "
                + "ELSE toLower(type(edge)) END] AS relations, "
                + "[i IN range(0, size(relationships(path)) - 1) | CASE WHEN startNode(relationships(path)[i]) "
                + "= nodes(path)[i] THEN 'outgoing' ELSE 'incoming' END] AS directions, "
                + "length(path) AS hops "
                + "ORDER BY hops, candidate.name LIMIT $limit";
        try (Session session = openSession()) {
            List<EntityTypePathResult> result = new ArrayList<>();
            for (Record record : session.run(cypher, Values.parameters("experiment_id", experiment,
                    "dataset_id", dataset, "seed_names", names, "limit", limit)).list()) {
                result.add(new EntityTypePathResult(record.get("seed").asString(),
                        record.get("type_path").asList(value -> value.asString()),
                        record.get("relations").asList(value -> value.asString()),
                        record.get("directions").asList(value -> value.asString()),
                        record.get("hops").asInt(), experiment, dataset));
            }
            return List.copyOf(result);
        }
    }

    /** Checks answer-type membership through hierarchy/equivalence paths. */
    public boolean areEntityTypesCompatible(String experimentId, String datasetId,
                                             String actualType, List<String> expectedTypes,
                                             int maxHops) {
        String experiment = requireId(experimentId, "experimentId");
        String dataset = requireId(datasetId, "datasetId");
        String actual = normalizeType(actualType);
        Objects.requireNonNull(expectedTypes, "expectedTypes");
        if (maxHops < 0 || maxHops > 10) {
            throw new IllegalArgumentException("maxHops must be between 0 and 10");
        }
        List<String> expected = expectedTypes.stream().filter(Objects::nonNull)
                .map(Neo4jClient::normalizeType).distinct().toList();
        if (expected.isEmpty()) return false;
        String cypher = "MATCH (actual:EntityType {experiment_id: $experiment_id, dataset_id: $dataset_id, name: $actual}) "
                + "MATCH (expected:EntityType {experiment_id: $experiment_id, dataset_id: $dataset_id}) "
                + "WHERE expected.name IN $expected "
                + "MATCH path = (actual)-[:SUBTYPE_OF|EQUIVALENT_TO*0.." + maxHops + "]-(expected) "
                + "WHERE all(node IN nodes(path) WHERE node.experiment_id = $experiment_id "
                + "AND node.dataset_id = $dataset_id) AND all(edge IN relationships(path) "
                + "WHERE edge.experiment_id = $experiment_id AND edge.dataset_id = $dataset_id) "
                + "RETURN count(path) > 0 AS compatible";
        try (Session session = openSession()) {
            Record record = session.run(cypher, Values.parameters("experiment_id", experiment,
                    "dataset_id", dataset, "actual", actual, "expected", expected)).single();
            return record.get("compatible").asBoolean(false);
        }
    }

    /** Returns the local graph neighborhood around one exact name-and-type entity in one experiment. */
    public KnowledgeGraphContext queryAroundEntity(String experimentId, String entityName,
                                                   String entityType, int limit) {
        return queryGraphInternal(requireId(experimentId, "experimentId"), null,
                requireText(entityName, "entityName"), requireText(entityType, "entityType"), limit);
    }

    /** Returns a dataset-scoped neighborhood around an exact name-and-type entity. */
    public KnowledgeGraphContext queryAroundEntityInDataset(String experimentId, String datasetId,
                                                            String entityName, String entityType, int limit) {
        return queryGraphInternal(requireId(experimentId, "experimentId"), requireId(datasetId, "datasetId"),
                requireText(entityName, "entityName"), requireText(entityType, "entityType"), limit);
    }

    public KnowledgeGraphContext queryAroundEntityInDataset(String experimentId, long datasetId,
                                                            String entityName, String entityType, int limit) {
        return queryAroundEntityInDataset(experimentId, Long.toString(datasetId), entityName, entityType, limit);
    }

    /** Convenience method for directly injecting graph facts into an LLM prompt. */
    public String reasoningContext(String experimentId, String entityName, int limit) {
        return queryAroundEntity(experimentId, entityName, limit).toPromptText();
    }

    public String reasoningContextInDataset(String experimentId, String datasetId, String entityName, int limit) {
        return queryAroundEntityInDataset(experimentId, datasetId, entityName, limit).toPromptText();
    }

    /** Convenience method for LLM context when entity name alone is ambiguous. */
    public String reasoningContext(String experimentId, String entityName, String entityType, int limit) {
        return queryAroundEntity(experimentId, entityName, entityType, limit).toPromptText();
    }

    public String reasoningContextInDataset(String experimentId, String datasetId, String entityName,
                                            String entityType, int limit) {
        return queryAroundEntityInDataset(experimentId, datasetId, entityName, entityType, limit).toPromptText();
    }

    /** Deletes the complete graph for one experiment, including relationships. */
    public long deleteExperimentGraph(String experimentId) {
        return deleteExperimentGraphInternal(experimentId, null);
    }

    /** Deletes the complete graph for one dataset inside an experiment. */
    public long deleteExperimentGraph(String experimentId, String datasetId) {
        return deleteExperimentGraphInternal(experimentId, datasetId);
    }

    public long deleteExperimentGraph(String experimentId, long datasetId) {
        return deleteExperimentGraph(experimentId, Long.toString(datasetId));
    }

    private long deleteExperimentGraphInternal(String experimentId, String datasetId) {
        String experiment = requireId(experimentId, "experimentId");
        String dataset = datasetId == null ? null : requireId(datasetId, "datasetId");
        try (Session session = openSession()) {
            String filter = dataset == null ? "" : " AND node.dataset_id = $dataset_id";
            Map<String, Object> params = new HashMap<>(Values.parameters("experiment_id", experiment).asMap());
            if (dataset != null) params.put("dataset_id", dataset);
            return session.executeWrite(tx -> tx.run(
                    "MATCH (node {experiment_id: $experiment_id}) "
                            + "WHERE (node:Entity OR node:EntityType)" + filter
                            + " DETACH DELETE node RETURN count(node) AS deleted", params)
                    .single().get("deleted").asLong());
        }
    }

    private KnowledgeGraphContext queryGraphInternal(String experiment, String dataset, String entityName,
                                                     String entityType, int limit) {
        if (limit < 1) throw new IllegalArgumentException("limit must be positive");
        List<KnowledgeEntity> entities = new ArrayList<>();
        List<KnowledgeRelation> relations = new ArrayList<>();
        try (Session session = openSession()) {
            String datasetFilter = dataset == null ? "" : " AND e.dataset_id = $dataset_id";
            String entityQuery = entityName == null
                    ? "MATCH (e:Entity {experiment_id: $experiment_id}) WHERE true" + datasetFilter
                    + " RETURN e ORDER BY e.entity_id LIMIT $limit"
                    : entityType == null
                    ? "MATCH (e:Entity {experiment_id: $experiment_id}) WHERE toLower(e.name) = toLower($name)"
                    + datasetFilter + " RETURN e ORDER BY e.entity_id LIMIT $limit"
                    : "MATCH (e:Entity {experiment_id: $experiment_id}) "
                    + "WHERE toLower(e.name) = toLower($name) AND toLower(e.type) = toLower($type)"
                    + datasetFilter + " RETURN e ORDER BY e.entity_id LIMIT $limit";
            Map<String, Object> entityParams = new HashMap<>(entityName == null
                    ? Values.parameters("experiment_id", experiment, "limit", limit).asMap()
                    : entityType == null
                    ? Values.parameters("experiment_id", experiment, "name", entityName, "limit", limit).asMap()
                    : Values.parameters("experiment_id", experiment, "name", entityName,
                    "type", entityType, "limit", limit).asMap());
            if (dataset != null) entityParams.put("dataset_id", dataset);
            for (Record record : session.run(entityQuery, entityParams).list()) {
                var node = record.get("e").asNode();
                entities.add(new KnowledgeEntity(node.get("name").asString(), node.get("type").asString(),
                        node.get("experiment_id").asString(), node.get("dataset_id").asString(""),
                        node.get("entity_id").asString()));
            }
            String relationDatasetFilter = dataset == null ? "" : " AND r.dataset_id = $dataset_id";
            String relationQuery = entityName == null
                    ? "MATCH (s:Entity {experiment_id: $experiment_id})-[r:RELATES_TO {experiment_id: $experiment_id}]->(t:Entity {experiment_id: $experiment_id}) "
                    + "WHERE true" + relationDatasetFilter
                    + " RETURN s.entity_id AS sid, t.entity_id AS tid, s.name AS sname, t.name AS tname, "
                    + "s.type AS stype, t.type AS ttype, r.name AS name, r.context AS context, "
                    + "r.dataset_id AS dataset_id LIMIT $limit"
                    : entityType != null
                    ? "MATCH (s:Entity {experiment_id: $experiment_id})-[r:RELATES_TO {experiment_id: $experiment_id}]->(t:Entity {experiment_id: $experiment_id}) "
                    + "WHERE ((toLower(s.name) = toLower($name) AND toLower(s.type) = toLower($type)) "
                    + "OR (toLower(t.name) = toLower($name) AND toLower(t.type) = toLower($type)))"
                    + relationDatasetFilter
                    + " RETURN s.entity_id AS sid, t.entity_id AS tid, s.name AS sname, t.name AS tname, "
                    + "s.type AS stype, t.type AS ttype, r.name AS name, r.context AS context, "
                    + "r.dataset_id AS dataset_id LIMIT $limit"
                    : "MATCH (s:Entity {experiment_id: $experiment_id})-[r:RELATES_TO {experiment_id: $experiment_id}]->(t:Entity {experiment_id: $experiment_id}) "
                    + "WHERE (toLower(s.name) = toLower($name) OR toLower(t.name) = toLower($name))"
                    + relationDatasetFilter
                    + " RETURN s.entity_id AS sid, t.entity_id AS tid, s.name AS sname, t.name AS tname, "
                    + "s.type AS stype, t.type AS ttype, r.name AS name, r.context AS context, "
                    + "r.dataset_id AS dataset_id LIMIT $limit";
            Map<String, Object> relationParams = new HashMap<>(entityName == null
                    ? Values.parameters("experiment_id", experiment, "limit", limit).asMap()
                    : entityType == null
                    ? Values.parameters("experiment_id", experiment, "name", entityName, "limit", limit).asMap()
                    : Values.parameters("experiment_id", experiment, "name", entityName,
                    "type", entityType, "limit", limit).asMap());
            if (dataset != null) relationParams.put("dataset_id", dataset);
            for (Record record : session.run(relationQuery, relationParams).list()) {
                relations.add(new KnowledgeRelation(record.get("name").asString(), record.get("context").asString(""),
                        experiment, record.get("dataset_id").asString(dataset == null ? "" : dataset),
                        record.get("sid").asString(), record.get("tid").asString(),
                        record.get("sname").asString(), record.get("tname").asString()));
                String relationDataset = record.get("dataset_id").asString(dataset == null ? "" : dataset);
                addEntityIfMissing(entities, record.get("sid").asString(), record.get("sname").asString(),
                        record.get("stype").asString(), experiment, relationDataset);
                addEntityIfMissing(entities, record.get("tid").asString(), record.get("tname").asString(),
                        record.get("ttype").asString(), experiment, relationDataset);
            }
        }
        return new KnowledgeGraphContext(entities, relations);
    }

    private static void addEntityIfMissing(List<KnowledgeEntity> entities, String id, String name,
                                           String type, String experiment, String dataset) {
        for (KnowledgeEntity entity : entities) if (entity.entityId().equals(id)) return;
        entities.add(new KnowledgeEntity(name, type, experiment, dataset, id));
    }

    /** Creates native Neo4j vector indexes. Safe to call repeatedly. */
    public void ensureVectorIndexes() {
        try (Session session = openSession()) {
            session.executeWrite(tx -> {
                tx.run("CREATE CONSTRAINT kg_entity_type_id_unique IF NOT EXISTS "
                        + "FOR (n:EntityType) REQUIRE n.type_id IS UNIQUE").consume();
                tx.run("CREATE VECTOR INDEX " + quoteIdentifier(config.neo4jEntityVectorIndex)
                        + " IF NOT EXISTS FOR (n:Entity) ON (n.embedding) OPTIONS {indexConfig: {"
                        + "`vector.dimensions`: " + embeddingDimension + ", `vector.similarity_function`: 'cosine'}}").consume();
                tx.run("CREATE VECTOR INDEX " + quoteIdentifier(config.neo4jRelationVectorIndex)
                        + " IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.embedding) OPTIONS {indexConfig: {"
                        + "`vector.dimensions`: " + embeddingDimension + ", `vector.similarity_function`: 'cosine'}}").consume();
                tx.run("CREATE VECTOR INDEX " + quoteIdentifier(config.neo4jTypeVectorIndex)
                        + " IF NOT EXISTS FOR (n:EntityType) ON (n.embedding) OPTIONS {indexConfig: {"
                        + "`vector.dimensions`: " + embeddingDimension + ", `vector.similarity_function`: 'cosine'}}").consume();
                return null;
            });
        }
    }

    private List<Float> embedding(String text) {
        String key = requireText(text, "embedding text");
        return embeddingCache.computeIfAbsent(key, this::loadEmbedding);
    }

    private List<Float> loadEmbedding(String text) {
        boolean acquired = false;
        try {
            embeddingPermits.acquire();
            acquired = true;
            float[] values = embeddings.embed(text);
            if (values.length != embeddingDimension) {
                throw new IllegalStateException("Expected embedding dimension " + embeddingDimension
                        + " but got " + values.length);
            }
            List<Float> result = new ArrayList<>(values.length);
            for (float value : values) result.add(value);
            return List.copyOf(result);
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Embedding request interrupted", interrupted);
        } finally {
            if (acquired) embeddingPermits.release();
        }
    }

    private static void checkTopK(int topK) {
        if (topK < 1) throw new IllegalArgumentException("topK must be positive");
    }

    private static void checkDistance(double maxDistance) {
        if (!Double.isFinite(maxDistance) || maxDistance < 0) {
            throw new IllegalArgumentException("maxDistance must be a finite non-negative number");
        }
    }

    private static String quoteIdentifier(String identifier) {
        if (identifier == null || identifier.isBlank()) throw new IllegalArgumentException("index name must not be blank");
        return "`" + identifier.replace("`", "``") + "`";
    }

    private static String entityId(String experimentId, String datasetId, String name, String type) {
        String raw = experimentId + "\u0000" + datasetId + "\u0000"
                + EntityRelationNormalizer.normalizeName(name) + "\u0000"
                + EntityRelationNormalizer.typeBucket(type);
        return hashedId("entity-", raw);
    }

    private static String typeId(String experimentId, String datasetId, String type) {
        return hashedId("type-", experimentId + "\u0000" + datasetId + "\u0000" + normalizeType(type));
    }

    private static String hashedId(String prefix, String raw) {
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256").digest(raw.getBytes(StandardCharsets.UTF_8));
            StringBuilder result = new StringBuilder(prefix);
            for (byte value : digest) result.append(String.format("%02x", value));
            return result.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is unavailable", e);
        }
    }

    private static String normalizeType(String value) {
        return requireText(value, "type").toLowerCase(Locale.ROOT).replaceAll("\\s+", " ");
    }

    private static String requireId(String value, String name) {
        if (value == null || value.isBlank() || !value.matches("[A-Za-z0-9._-]+")) {
            throw new IllegalArgumentException(name + " must match [A-Za-z0-9._-]+: " + value);
        }
        return value;
    }

    private static String requireText(String value, String name) {
        if (value == null || value.isBlank()) throw new IllegalArgumentException(name + " must not be blank");
        return value.trim();
    }

    @Override
    public void close() {
        driver.close();
    }

    public record EntitySearchResult(String entityId, String name, String type,
                                     String experimentId, String datasetId, double score) {
        /** Cosine distance corresponding to the Neo4j vector procedure score. */
        public double distance() { return 1.0 - score; }
    }

    public record EntityTypeSearchResult(String typeId, String name, List<String> aliases,
                                         String experimentId, String datasetId, double score) {
        public double distance() { return 1.0 - score; }
    }

    public record IncidentRelationEvidence(String relationshipId, String name, String context,
                                           String subjectEntityId, String subjectName,
                                           String subjectType, List<Long> subjectSourceDatasetIds,
                                           String targetEntityId, String targetName,
                                           String targetType, List<Long> targetSourceDatasetIds,
                                           List<Long> sourceDatasetIds,
                                           String experimentId, String datasetId) {
        public IncidentRelationEvidence {
            subjectSourceDatasetIds = subjectSourceDatasetIds == null
                    ? List.of() : List.copyOf(subjectSourceDatasetIds);
            targetSourceDatasetIds = targetSourceDatasetIds == null
                    ? List.of() : List.copyOf(targetSourceDatasetIds);
            sourceDatasetIds = sourceDatasetIds == null ? List.of() : List.copyOf(sourceDatasetIds);
        }
    }

    public record SampleEntityTypeNode(String type, long entityCount) { }

    public record SampleEntityTypeEdge(String sourceType, String relation,
                                       String targetType, long factCount) { }

    public record SampleEntityTypeGraph(long sourceDatasetId,
                                        List<SampleEntityTypeNode> nodes,
                                        List<SampleEntityTypeEdge> edges,
                                        String experimentId, String datasetId) {
        public SampleEntityTypeGraph {
            nodes = nodes == null ? List.of() : List.copyOf(nodes);
            edges = edges == null ? List.of() : List.copyOf(edges);
        }
    }

    public record SampleEntityTypeMatch(long sourceDatasetId, String answerType,
                                        String sampleType, long entityCount,
                                        double score, double distance) { }

    public record EntityTypePathResult(String seedType, List<String> typePath,
                                       List<String> relations, List<String> directions, int hops,
                                       String experimentId, String datasetId) {
        public EntityTypePathResult {
            typePath = typePath == null ? List.of() : List.copyOf(typePath);
            relations = relations == null ? List.of() : List.copyOf(relations);
            directions = directions == null ? List.of() : List.copyOf(directions);
        }

        public String candidateType() {
            return typePath.isEmpty() ? seedType : typePath.get(typePath.size() - 1);
        }
    }

}
