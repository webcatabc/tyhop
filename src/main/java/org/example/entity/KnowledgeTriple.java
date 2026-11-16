package org.example.entity;

import com.google.gson.annotations.SerializedName;

/** JSON shape returned by the LLM for one extracted subject-predicate-object triple. */
public class KnowledgeTriple {
    private String subject;
    @SerializedName("subject_type")
    private String subjectType;
    private String relation;
    private String target;
    @SerializedName("target_type")
    private String targetType;
    @SerializedName("relation_context")
    private String relationContext;

    public KnowledgeTriple() {
    }

    public KnowledgeTriple(String subject, String subjectType, String relation,
                           String target, String targetType, String relationContext) {
        this.subject = subject;
        this.subjectType = subjectType;
        this.relation = relation;
        this.target = target;
        this.targetType = targetType;
        this.relationContext = relationContext;
    }

    public String getSubject() { return subject; }
    public void setSubject(String subject) { this.subject = subject; }
    public String getSubjectType() { return subjectType; }
    public void setSubjectType(String subjectType) { this.subjectType = subjectType; }
    public String getRelation() { return relation; }
    public void setRelation(String relation) { this.relation = relation; }
    public String getTarget() { return target; }
    public void setTarget(String target) { this.target = target; }
    public String getTargetType() { return targetType; }
    public void setTargetType(String targetType) { this.targetType = targetType; }
    public String getRelationContext() { return relationContext; }
    public void setRelationContext(String relationContext) { this.relationContext = relationContext; }
}
