package org.example.entity;

import java.util.Collections;
import java.util.Map;
/** Current aggregate metrics for one experiment, plus the sample just scored. */
public final class ExperimentResult {
    private final Long experimentId;
    private final int totalSamples;
    private final int evaluatedSamples;
    private final double f1;
    private final double em;
    private final double hit;
    private final ExperimentSampleResult sampleResult;
    private final Map<String, ExperimentDatasetResult> datasetResults;

    public ExperimentResult(Long experimentId, int totalSamples, int evaluatedSamples,
                            double f1, double em, double hit,
                            ExperimentSampleResult sampleResult) {
        this(experimentId, totalSamples, evaluatedSamples, f1, em, hit,
                sampleResult, Collections.emptyMap());
    }

    public ExperimentResult(Long experimentId, int totalSamples, int evaluatedSamples,
                            double f1, double em, double hit,
                            ExperimentSampleResult sampleResult,
                            Map<String, ExperimentDatasetResult> datasetResults) {
        this.experimentId = experimentId;
        this.totalSamples = totalSamples;
        this.evaluatedSamples = evaluatedSamples;
        this.f1 = f1;
        this.em = em;
        this.hit = hit;
        this.sampleResult = sampleResult;
        this.datasetResults = datasetResults == null
                ? Collections.emptyMap() : Map.copyOf(datasetResults);
    }

    public Long getExperimentId() { return experimentId; }
    public int getTotalSamples() { return totalSamples; }
    public int getEvaluatedSamples() { return evaluatedSamples; }
    public double getF1() { return f1; }
    public double getEm() { return em; }
    public double getHit() { return hit; }
    public ExperimentSampleResult getSampleResult() { return sampleResult; }
    public Map<String, ExperimentDatasetResult> getDatasetResults() { return datasetResults; }
    public ExperimentDatasetResult getDatasetResult(String dataType) {
        return datasetResults.get(dataType);
    }
}
