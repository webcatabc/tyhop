package org.example.entity;

import java.util.List;

/** Graph facts ready to be injected into an LLM reasoning prompt. */
public record KnowledgeGraphContext(List<KnowledgeEntity> entities,
                                    List<KnowledgeRelation> relations) {
    public KnowledgeGraphContext {
        entities = List.copyOf(entities);
        relations = List.copyOf(relations);
    }

    public String toPromptText() {
        StringBuilder out = new StringBuilder("Knowledge graph facts:\n");
        for (KnowledgeRelation relation : relations) {
            out.append("- ").append(relation.subjectName()).append(" --[")
                    .append(relation.name()).append("]--> ")
                    .append(relation.targetName());
            if (relation.context() != null && !relation.context().isBlank()) {
                out.append(" (evidence: ").append(relation.context()).append(')');
            }
            out.append('\n');
        }
        return out.toString();
    }
}
