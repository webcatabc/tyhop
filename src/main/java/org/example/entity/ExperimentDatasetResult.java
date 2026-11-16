package org.example.entity;

/** Aggregate metrics for one dataset type inside an experiment. */
public final class ExperimentDatasetResult {
    private final String dataType;
    private final int totalSamples;
    private final int evaluatedSamples;
    private final double f1;
    private final double em;
    private final double hit;

    public ExperimentDatasetResult(String dataType, int totalSamples, int evaluatedSamples,
                                   double f1, double em, double hit) {
        this.dataType = dataType;
        this.totalSamples = totalSamples;
        this.evaluatedSamples = evaluatedSamples;
        this.f1 = f1;
        this.em = em;
        this.hit = hit;
    }

    public String getDataType() { return dataType; }
    public int getTotalSamples() { return totalSamples; }
    public int getEvaluatedSamples() { return evaluatedSamples; }
    public double getF1() { return f1; }
    public double getEm() { return em; }
    public double getHit() { return hit; }
}
