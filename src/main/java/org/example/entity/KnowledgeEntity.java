package org.example.entity;

/** A Neo4j entity returned from an experiment/dataset-scoped graph query. */
public record KnowledgeEntity(String name, String type, String experimentId, String datasetId, String entityId) {
    public KnowledgeEntity(String name, String type, String experimentId, String entityId) {
        this(name, type, experimentId, "", entityId);
    }
}
