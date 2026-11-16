package org.example.utils;

import org.example.config.AppConfig;
import org.example.entity.Experiment;
import org.example.entity.ExperimentResult;
import org.example.mp.KgPaperRepository;

import java.util.Collection;
import java.util.Objects;

/** Facade for HotpotQA scoring and transactional experiment metric updates. */
public final class AnswerChecker implements AutoCloseable {
    private final KgPaperRepository repository;

    /** Opens the configured database connection without requiring a repository argument. */
    public AnswerChecker() {
        this(new KgPaperRepository(AppConfig.load()));
    }

    public AnswerChecker(KgPaperRepository repository) {
        this.repository = Objects.requireNonNull(repository, "repository");
    }

    public Metrics.Result calculate(String prediction, String goldAnswer) {
        return Metrics.evaluate(prediction, goldAnswer);
    }

    public Metrics.Result calculate(String prediction, Collection<String> goldAnswers) {
        return Metrics.evaluate(prediction, goldAnswers);
    }

    /** Calculates the sample score and updates sample and experiment rows atomically. */
    public ExperimentResult checkAndUpdate(long experimentDataId, String predictedAnswer) {
        return repository.recordAnswer(experimentDataId, predictedAnswer);
    }

    /** Calculates metrics and persists an answer plus its reasoning path atomically. */
    public ExperimentResult checkAndUpdate(long experimentDataId, String predictedAnswer,
                                           String reasoningPath) {
        return repository.recordAnswer(experimentDataId, predictedAnswer, reasoningPath);
    }

    /** Clears prior QA outputs while retaining every constructed graph. */
    public int resetExperimentReasoning(long experimentId) {
        return repository.resetExperimentReasoning(experimentId);
    }

    /**
     * Clears prior QA outputs for methods that never build a graph, so the whole
     * experiment can be answered again from scratch.
     */
    public int resetExperimentQa(long experimentId) {
        return repository.resetExperimentQa(experimentId);
    }

    /** Reads the stored experiment row so a run can report its aggregate metrics. */
    public Experiment findExperiment(long experimentId) {
        return repository.findExperimentById(experimentId);
    }

    @Override
    public void close() {
        repository.close();
    }
}
