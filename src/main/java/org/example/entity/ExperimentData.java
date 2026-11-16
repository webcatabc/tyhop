package org.example.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;

/** MyBatis-Plus entity for the experiment_data table. */
@TableName("experiment_data")
public class ExperimentData {
    public static final int STATUS_NOT_STARTED = 0;
    public static final int STATUS_GRAPH_BUILT = 1;
    public static final int STATUS_QA_COMPLETED = 2;

    @TableId(value = "id", type = IdType.AUTO)
    private Long id;
    private Long experimentId;
    private Long datasetId;
    private String answer;
    private Double em;
    private Double f1;
    private Double hit;
    private Integer status;
    /** Serialized ToG reasoning paths used to produce the answer. */
    private String reasoningPath;

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public Long getExperimentId() { return experimentId; }
    public void setExperimentId(Long experimentId) { this.experimentId = experimentId; }
    public Long getDatasetId() { return datasetId; }
    public void setDatasetId(Long datasetId) { this.datasetId = datasetId; }
    public String getAnswer() { return answer; }
    public void setAnswer(String answer) { this.answer = answer; }
    public Double getEm() { return em; }
    public void setEm(Double em) { this.em = em; }
    public Double getF1() { return f1; }
    public void setF1(Double f1) { this.f1 = f1; }
    public Double getHit() { return hit; }
    public void setHit(Double hit) { this.hit = hit; }
    public Integer getStatus() { return status; }
    public void setStatus(Integer status) { this.status = status; }
    public String getReasoningPath() { return reasoningPath; }
    public void setReasoningPath(String reasoningPath) { this.reasoningPath = reasoningPath; }
}
