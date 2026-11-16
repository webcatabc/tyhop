package org.example.utils;

import org.example.config.AppConfig;
import org.example.experiment.DataLoader;
import org.example.mp.KgPaperRepository;

import java.util.LinkedHashMap;
import java.util.Map;

/** Factory for creating an experiment and lazily accessible sample loader. */
public final class ExperimentInitializer {
    private ExperimentInitializer() { }

    public static DataLoader create(String name, long randSeed,
                                    int hotpotqaNum, int musiqueNum,
                                    int twoWikiMultihopqaNum) {
        Map<String, Integer> quantities = new LinkedHashMap<>();
        quantities.put("hotpotqa", hotpotqaNum);
        quantities.put("musique", musiqueNum);
        quantities.put("2wikimultihopqa", twoWikiMultihopqaNum);
        return create(AppConfig.load(), name, randSeed, quantities);
    }

    public static DataLoader create(AppConfig config, String name, long randSeed,
                                    Map<String, Integer> quantities) {
        KgPaperRepository repository = new KgPaperRepository(config);
        try {
            return repository.initializeExperiment(name, randSeed, quantities);
        } catch (RuntimeException | Error e) {
            repository.close();
            throw e;
        }
    }

    /** Loads an existing experiment by its primary key. */
    public static DataLoader loadByExperimentId(long experimentId) {
        return loadByExperimentId(AppConfig.load(), experimentId);
    }

    public static DataLoader loadByExperimentId(AppConfig config, long experimentId) {
        KgPaperRepository repository = new KgPaperRepository(config);
        try {
            return repository.loadExperimentById(experimentId);
        } catch (RuntimeException | Error e) {
            repository.close();
            throw e;
        }
    }

    /** Loads an existing experiment by its exact name. */
    public static DataLoader loadByExperimentName(String experimentName) {
        return loadByExperimentName(AppConfig.load(), experimentName);
    }

    public static DataLoader loadByExperimentName(AppConfig config, String experimentName) {
        KgPaperRepository repository = new KgPaperRepository(config);
        try {
            return repository.loadExperimentByName(experimentName);
        } catch (RuntimeException | Error e) {
            repository.close();
            throw e;
        }
    }
}
