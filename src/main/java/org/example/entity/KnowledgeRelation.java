package org.example.entity;

/** An experiment/dataset-scoped relationship and its evidence context. */
public record KnowledgeRelation(String name, String context,
                                String experimentId, String datasetId, String subjectEntityId,
                                String targetEntityId, String subjectName,
                                String targetName) {
    public KnowledgeRelation(String name, String context, String experimentId,
                             String subjectEntityId, String targetEntityId,
                             String subjectName, String targetName) {
        this(name, context, experimentId, "", subjectEntityId, targetEntityId, subjectName, targetName);
    }
}
