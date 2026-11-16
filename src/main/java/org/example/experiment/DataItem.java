package org.example.experiment;

import lombok.Getter;
import org.example.entity.DataSet;
import org.example.entity.ExperimentData;
import org.example.mp.KgPaperRepository;

import java.util.Objects;

/** A lazily loaded data-set sample belonging to an experiment. */
public class DataItem {
    private final KgPaperRepository repository;
    private final long datasetId;
    @Getter
    private final Long experimentDataId;
    private volatile Integer status;
    private volatile String persistedAnswer;
    private volatile String persistedReasoningPath;
    private volatile DataSet value;

    public DataItem(KgPaperRepository repository, long datasetId, Long experimentDataId) {
        this(repository, datasetId, experimentDataId, null, null, null);
    }

    public DataItem(KgPaperRepository repository, long datasetId, Long experimentDataId,
                    Integer status) {
        this(repository, datasetId, experimentDataId, status, null, null);
    }

    public DataItem(KgPaperRepository repository, long datasetId, Long experimentDataId,
                    Integer status, String persistedAnswer, String persistedReasoningPath) {
        this(repository, datasetId, experimentDataId, status,
                persistedAnswer, persistedReasoningPath, null);
    }

    public DataItem(KgPaperRepository repository, long datasetId, Long experimentDataId,
                    Integer status, String persistedAnswer, String persistedReasoningPath,
                    DataSet value) {
        this.repository = Objects.requireNonNull(repository, "repository");
        this.datasetId = datasetId;
        this.experimentDataId = experimentDataId;
        this.status = status;
        this.persistedAnswer = persistedAnswer;
        this.persistedReasoningPath = persistedReasoningPath;
        this.value = value;
    }

    public long getDatasetId() { return datasetId; }

    /** Returns the complete row, loading it from data_set on first access. */
    public DataSet getDataSet() { return dataSet(); }
    public String getContext() { return dataSet().getContext(); }
    public String getQuestion() { return dataSet().getQuestion(); }
    public String getAnswer() { return dataSet().getAnswer(); }
    public String getDataType() { return dataSet().getDataType(); }
    public String getOriginalId() { return dataSet().getOriginalId(); }
    public String getType() { return dataSet().getType(); }
    public String getDifficulty() { return dataSet().getDifficulty(); }

    public Integer getStatus() { return status; }

    /** A completed QA row also necessarily has a completed graph. */
    public boolean isGraphBuilt() {
        return status != null && status >= ExperimentData.STATUS_GRAPH_BUILT;
    }

    public boolean isQaCompleted() {
        return status != null && status >= ExperimentData.STATUS_QA_COMPLETED;
    }

    public String getPersistedAnswer() { return persistedAnswer; }
    public String getPersistedReasoningPath() { return persistedReasoningPath; }

    /** Marks this sample's graph as built and updates experiment progress. */
    public int markGraphBuilt() {
        if (experimentDataId == null) {
            throw new IllegalStateException("sample is not linked to experiment_data");
        }
        int changed = repository.markGraphBuilt(experimentDataId);
        if (status == null || status < ExperimentData.STATUS_GRAPH_BUILT) {
            status = ExperimentData.STATUS_GRAPH_BUILT;
        }
        return changed;
    }

    /** Mirrors a successful AnswerChecker update for the remainder of this run. */
    public void markQaCompleted(String answer, String reasoningPath) {
        status = ExperimentData.STATUS_QA_COMPLETED;
        persistedAnswer = answer;
        persistedReasoningPath = reasoningPath;
    }

    /** Mirrors an experiment-level QA reset while retaining the built graph state. */
    public void resetQaForRerun() {
        if (isGraphBuilt()) status = ExperimentData.STATUS_GRAPH_BUILT;
        persistedAnswer = null;
        persistedReasoningPath = null;
    }

    /**
     * Mirrors an experiment-level QA reset for runs without graph construction:
     * the sample returns to the not-started
     * state so the next run answers it again.
     */
    public void resetQaOnly() {
        status = ExperimentData.STATUS_NOT_STARTED;
        persistedAnswer = null;
        persistedReasoningPath = null;
    }

    private DataSet dataSet() {
        DataSet current = value;
        if (current == null) {
            synchronized (this) {
                current = value;
                if (current == null) {
                    current = repository.findDataSetById(datasetId);
                    if (current == null) throw new IllegalStateException("data_set row not found: " + datasetId);
                    value = current;
                }
            }
        }
        return current;
    }

    /** Drops the cached row so the next property access reads the current database value. */
    public synchronized void evict() { value = null; }

    @Override
    public String toString() {
        return "DataItem{datasetId=" + datasetId + ", experimentDataId=" + experimentDataId + '}';
    }
}
