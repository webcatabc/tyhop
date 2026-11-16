package org.example.experiment;

import lombok.Getter;
import org.example.entity.Experiment;
import org.example.mp.KgPaperRepository;

import java.util.Collections;
import java.util.List;

/** Lazily exposes the samples selected for an experiment. */
public class DataLoader implements AutoCloseable {
    @Getter
    public final List<DataItem> dataItemList;
    private final KgPaperRepository repository;
    @Getter
    private final Experiment experiment;

    /** Empty loader retained for source compatibility with older examples. */
    public DataLoader() {
        this.repository = null;
        this.experiment = null;
        this.dataItemList = new java.util.ArrayList<>();
    }

    public DataLoader(KgPaperRepository repository, Experiment experiment, List<DataItem> items) {
        this.repository = repository;
        this.experiment = experiment;
        this.dataItemList = Collections.unmodifiableList(new java.util.ArrayList<>(items));
    }

    @Override
    public void close() {
        if (repository != null) repository.close();
    }
}
