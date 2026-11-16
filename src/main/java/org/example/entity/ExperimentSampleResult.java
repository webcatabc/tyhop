package org.example.entity;

/** Metrics calculated for one answered experiment sample. */
public final class ExperimentSampleResult {
    private final Long experimentDataId;
    private final Long datasetId;
    private final String prediction;
    private final String goldAnswer;
    private final double em;
    private final double f1;
    /** Answer-level hit; for HotpotQA this is the same binary value as EM. */
    private final double hit;
    private final String reasoningPath;

    public ExperimentSampleResult(Long experimentDataId, Long datasetId,
                                  String prediction, String goldAnswer,
                                  double em, double f1, double hit) {
        this(experimentDataId, datasetId, prediction, goldAnswer, em, f1, hit, null);
    }

    public ExperimentSampleResult(Long experimentDataId, Long datasetId,
                                  String prediction, String goldAnswer,
                                  double em, double f1, double hit,
                                  String reasoningPath) {
        this.experimentDataId = experimentDataId;
        this.datasetId = datasetId;
        this.prediction = prediction;
        this.goldAnswer = goldAnswer;
        this.em = em;
        this.f1 = f1;
        this.hit = hit;
        this.reasoningPath = reasoningPath;
    }

    public Long getExperimentDataId() { return experimentDataId; }
    public Long getDatasetId() { return datasetId; }
    public String getPrediction() { return prediction; }
    public String getGoldAnswer() { return goldAnswer; }
    public double getEm() { return em; }
    public double getF1() { return f1; }
    public double getHit() { return hit; }
    public String getReasoningPath() { return reasoningPath; }
}
