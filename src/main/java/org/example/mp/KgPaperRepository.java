package org.example.mp;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.UpdateWrapper;
import org.apache.ibatis.session.SqlSession;
import org.example.config.AppConfig;
import org.example.entity.DataSet;
import org.example.entity.Experiment;
import org.example.entity.ExperimentDatasetResult;
import org.example.entity.ExperimentData;
import org.example.entity.ExperimentResult;
import org.example.entity.ExperimentSampleResult;
import org.example.experiment.DataLoader;
import org.example.experiment.DataItem;
import org.example.mapper.DataSetMapper;
import org.example.mapper.ExperimentDataMapper;
import org.example.mapper.ExperimentMapper;
import org.example.utils.Metrics;

import java.util.List;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Function;
import java.util.function.Supplier;

/** Small transaction-aware facade over the kg-paper MyBatis-Plus mappers. */
public final class KgPaperRepository implements AutoCloseable {
    /**
     * The paper database lives behind a remote host whose link occasionally
     * resets an established TCP connection, which surfaces as a
     * {@code Communications link failure} in the middle of a long run. Reads
     * and single-row updates are idempotent, so they are retried on a fresh
     * connection instead of aborting the experiment.
     */
    private static final int CONNECTION_RETRY_LIMIT = 4;

    private final KgPaperMybatisClient client;

    public KgPaperRepository(AppConfig config) {
        client = new KgPaperMybatisClient(config);
    }

    public DataSet findDataSetById(long id) {
        return read(mapper -> mapper.dataSetMapper().selectById(id));
    }

    public DataSet findDataSetByOriginalId(String originalId) {
        Objects.requireNonNull(originalId, "originalId");
        return read(mapper -> mapper.dataSetMapper().selectOne(
                new QueryWrapper<DataSet>().eq("original_id", originalId)));
    }

    public List<DataSet> listDataSetsByType(String dataType, int limit) {
        Objects.requireNonNull(dataType, "dataType");
        if (limit <= 0) throw new IllegalArgumentException("limit must be positive");
        return read(mapper -> mapper.dataSetMapper().selectList(
                new QueryWrapper<DataSet>().eq("data_type", dataType).last("LIMIT " + limit)));
    }

    public long countDataSetsByType(String dataType) {
        Objects.requireNonNull(dataType, "dataType");
        return read(mapper -> mapper.dataSetMapper().selectCount(
                new QueryWrapper<DataSet>().eq("data_type", dataType)));
    }

    public int insertDataSet(DataSet dataSet) {
        Objects.requireNonNull(dataSet, "dataSet");
        return write(mapper -> mapper.dataSetMapper().insert(dataSet));
    }

    public int updateDataSet(DataSet dataSet) {
        Objects.requireNonNull(dataSet, "dataSet");
        return write(mapper -> mapper.dataSetMapper().updateById(dataSet));
    }

    public int deleteDataSetById(long id) {
        return write(mapper -> mapper.dataSetMapper().deleteById(id));
    }

    public Experiment findExperimentById(long id) {
        return read(mapper -> mapper.experimentMapper().selectById(id));
    }

    /**
     * Opens a lazy loader for an existing experiment. Only experiment_data IDs
     * are read here; each DataItem loads its data_set row on first access.
     */
    public DataLoader loadExperimentById(long id) {
        if (id <= 0) throw new IllegalArgumentException("experiment id must be positive");
        return read(mapper -> {
            Experiment experiment = mapper.experimentMapper().selectById(id);
            if (experiment == null) throw new IllegalArgumentException("experiment not found: " + id);
            return createDataLoader(mapper, experiment);
        });
    }

    /** Opens a lazy loader for an existing experiment with the exact name. */
    public DataLoader loadExperimentByName(String name) {
        Objects.requireNonNull(name, "name");
        if (name.isBlank()) throw new IllegalArgumentException("experiment name must not be blank");
        return read(mapper -> {
            List<Experiment> matches = mapper.experimentMapper().selectList(
                    new QueryWrapper<Experiment>().eq("name", name).last("LIMIT 2"));
            if (matches.isEmpty()) throw new IllegalArgumentException("experiment not found: " + name);
            if (matches.size() > 1) {
                throw new IllegalStateException("multiple experiments have the name: " + name);
            }
            return createDataLoader(mapper, matches.get(0));
        });
    }

    public List<Experiment> listExperiments() {
        return read(mapper -> mapper.experimentMapper().selectList(null));
    }

    public int insertExperiment(Experiment experiment) {
        Objects.requireNonNull(experiment, "experiment");
        return write(mapper -> mapper.experimentMapper().insert(experiment));
    }

    public int updateExperimentById(Experiment experiment) {
        Objects.requireNonNull(experiment, "experiment");
        return write(mapper -> mapper.experimentMapper().updateById(experiment));
    }

    public int deleteExperimentById(long id) {
        return write(mapper -> mapper.experimentMapper().deleteById(id));
    }

    private DataLoader createDataLoader(SessionMappers mapper, Experiment experiment) {
        List<ExperimentData> samples = mapper.experimentDataMapper().selectList(
                new QueryWrapper<ExperimentData>()
                        .select("id", "dataset_id", "status", "answer", "reasoning_path")
                        .eq("experiment_id", experiment.getId())
                        .orderByAsc("id"));
        List<Long> datasetIds = samples.stream()
                .map(ExperimentData::getDatasetId)
                .filter(Objects::nonNull)
                .distinct()
                .toList();
        Map<Long, DataSet> dataSets = datasetIds.isEmpty()
                ? Map.of()
                : mapper.dataSetMapper().selectBatchIds(datasetIds).stream()
                .collect(java.util.stream.Collectors.toMap(
                        DataSet::getId, value -> value));
        List<DataItem> items = new ArrayList<>(samples.size());
        for (ExperimentData sample : samples) {
            if (sample.getDatasetId() == null) {
                throw new IllegalStateException("experiment_data has no dataset_id: " + sample.getId());
            }
            items.add(new DataItem(this, sample.getDatasetId(), sample.getId(), sample.getStatus(),
                    sample.getAnswer(), sample.getReasoningPath(),
                    dataSets.get(sample.getDatasetId())));
        }
        return new DataLoader(this, experiment, items);
    }

    /** Creates an experiment and its sample links in one transaction. */
    public DataLoader initializeExperiment(String name, long randSeed,
                                           Map<String, Integer> quantities) {
        Objects.requireNonNull(name, "name");
        Map<String, Integer> requested = normalizeQuantities(quantities);
        if (name.isBlank()) throw new IllegalArgumentException("name must not be blank");
        int total = totalQuantity(requested);

        return writeOnce(mapper -> {
            Experiment experiment = insertExperiment(mapper, name, randSeed, requested, total);
            List<DataItem> items = new ArrayList<>(total);
            for (Map.Entry<String, Integer> entry : requested.entrySet()) {
                String type = entry.getKey();
                int count = entry.getValue();
                QueryWrapper<DataSet> query = new QueryWrapper<DataSet>()
                        .select("id").eq("data_type", type)
                        .last("ORDER BY RAND(" + randSeed + ") LIMIT " + count);
                List<Object> ids = mapper.dataSetMapper().selectObjs(query);
                if (ids.size() != count) {
                    throw new IllegalArgumentException("not enough data_set rows for type " + type
                            + ": requested " + count + ", available " + ids.size());
                }
                for (Object rawId : ids) {
                    long datasetId = ((Number) rawId).longValue();
                    ExperimentData link = new ExperimentData();
                    link.setExperimentId(experiment.getId());
                    link.setDatasetId(datasetId);
                    link.setStatus(ExperimentData.STATUS_NOT_STARTED);
                    if (mapper.experimentDataMapper().insert(link) != 1) {
                        throw new IllegalStateException("failed to initialize experiment_data");
                    }
                    items.add(new DataItem(this, datasetId, link.getId(), link.getStatus(), null, null));
                }
            }
            return new DataLoader(this, experiment, items);
        });
    }

    /**
     * Creates a new experiment that reuses an explicit subset of existing samples
     * instead of selecting its own. Sources such as the graph-built part of a
     * larger experiment can therefore be reused on exactly the same questions.
     * Graph progress is copied but capped at
     * {@link ExperimentData#STATUS_GRAPH_BUILT}, so a source that already stored
     * answers does not make the new experiment look finished: the subset starts
     * with an empty QA phase while methods that reuse graphs can still run on it
     * without rebuilding anything.
     *
     * @param quantities per-type sample counts of {@code sourceSamples}, which
     *                   must add up to its size
     */
    public Experiment createSubsetExperiment(String name, long randSeed,
                                             Map<String, Integer> quantities,
                                             List<ExperimentData> sourceSamples) {
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(sourceSamples, "sourceSamples");
        Map<String, Integer> requested = normalizeQuantities(quantities);
        if (name.isBlank()) throw new IllegalArgumentException("name must not be blank");
        if (sourceSamples.isEmpty()) {
            throw new IllegalArgumentException("sourceSamples must not be empty");
        }
        int total = totalQuantity(requested);
        if (total != sourceSamples.size()) {
            throw new IllegalArgumentException("quantities add up to " + total
                    + " but " + sourceSamples.size() + " samples were supplied");
        }
        return writeOnce(mapper -> {
            Experiment experiment = insertExperiment(mapper, name, randSeed, requested, total);
            for (ExperimentData source : sourceSamples) {
                if (source.getDatasetId() == null) {
                    throw new IllegalArgumentException("source sample without dataset_id");
                }
                ExperimentData link = new ExperimentData();
                link.setExperimentId(experiment.getId());
                link.setDatasetId(source.getDatasetId());
                int status = source.getStatus() == null
                        ? ExperimentData.STATUS_NOT_STARTED : source.getStatus();
                link.setStatus(Math.min(status, ExperimentData.STATUS_GRAPH_BUILT));
                if (mapper.experimentDataMapper().insert(link) != 1) {
                    throw new IllegalStateException("failed to initialize experiment_data");
                }
            }
            return experiment;
        });
    }

    /** Validates and copies one non-negative quantity per supported data type. */
    private static Map<String, Integer> normalizeQuantities(Map<String, Integer> quantities) {
        Objects.requireNonNull(quantities, "quantities");
        if (quantities.isEmpty()) throw new IllegalArgumentException("quantities must not be empty");
        Map<String, Integer> requested = new LinkedHashMap<>();
        quantities.forEach((type, count) -> {
            if (type == null || type.isBlank()) throw new IllegalArgumentException("data type must not be blank");
            if (!type.equals("hotpotqa") && !type.equals("musique")
                    && !type.equals("2wikimultihopqa")) {
                throw new IllegalArgumentException("unsupported data type: " + type);
            }
            if (count == null || count < 0) throw new IllegalArgumentException("quantity must be non-negative");
            if (count > 0) requested.put(type, count);
        });
        if (requested.isEmpty()) throw new IllegalArgumentException("at least one quantity must be positive");
        return requested;
    }

    private static int totalQuantity(Map<String, Integer> requested) {
        long total = requested.values().stream().mapToLong(Integer::longValue).sum();
        if (total > Integer.MAX_VALUE) throw new IllegalArgumentException("total quantity is too large");
        return (int) total;
    }

    /** Inserts one empty experiment row; callers add the experiment_data links. */
    private static Experiment insertExperiment(SessionMappers mapper, String name, long randSeed,
                                               Map<String, Integer> requested, int total) {
        Experiment experiment = new Experiment();
        experiment.setName(name);
        experiment.setRandSeed(randSeed);
        experiment.setNum(total);
        experiment.setConstructedNum(0);
        experiment.setFinishNum(0);
        experiment.setHotpotqaNum(requested.getOrDefault("hotpotqa", 0));
        experiment.setMusiqueNum(requested.getOrDefault("musique", 0));
        experiment.setTwoWikiMultihopqaNum(requested.getOrDefault("2wikimultihopqa", 0));
        experiment.setF1(0.0);
        experiment.setEm(0.0);
        experiment.setHit(0.0);
        experiment.setHotpotqaF1(0.0);
        experiment.setHotpotqaEm(0.0);
        experiment.setHotpotqaHit(0.0);
        experiment.setMusiqueF1(0.0);
        experiment.setMusiqueEm(0.0);
        experiment.setMusiqueHit(0.0);
        experiment.setTwoWikiMultihopqaF1(0.0);
        experiment.setTwoWikiMultihopqaEm(0.0);
        experiment.setTwoWikiMultihopqaHit(0.0);
        if (mapper.experimentMapper().insert(experiment) != 1 || experiment.getId() == null) {
            throw new IllegalStateException("failed to create experiment");
        }
        return experiment;
    }

    public List<ExperimentData> listExperimentData(long experimentId) {
        return read(mapper -> mapper.experimentDataMapper().selectList(
                new QueryWrapper<ExperimentData>().eq("experiment_id", experimentId)));
    }

    public ExperimentData findExperimentDataById(long id) {
        return read(mapper -> mapper.experimentDataMapper().selectById(id));
    }

    public ExperimentData findExperimentData(long experimentId, long datasetId) {
        if (experimentId <= 0 || datasetId <= 0) {
            throw new IllegalArgumentException("experimentId and datasetId must be positive");
        }
        return read(mapper -> mapper.experimentDataMapper().selectOne(
                new QueryWrapper<ExperimentData>()
                        .eq("experiment_id", experimentId)
                        .eq("dataset_id", datasetId)));
    }

    public List<ExperimentData> listExperimentDataByStatus(long experimentId, int status) {
        if (status < ExperimentData.STATUS_NOT_STARTED
                || status > ExperimentData.STATUS_QA_COMPLETED) {
            throw new IllegalArgumentException("status must be 0, 1, or 2");
        }
        return read(mapper -> mapper.experimentDataMapper().selectList(
                new QueryWrapper<ExperimentData>()
                        .eq("experiment_id", experimentId)
                        .eq("status", status)));
    }

    public int insertExperimentData(ExperimentData experimentData) {
        Objects.requireNonNull(experimentData, "experimentData");
        return write(mapper -> mapper.experimentDataMapper().insert(experimentData));
    }

    public int updateExperimentData(ExperimentData experimentData) {
        Objects.requireNonNull(experimentData, "experimentData");
        return write(mapper -> mapper.experimentDataMapper().updateById(experimentData));
    }

    /**
     * Marks one sample's graph construction as complete and updates the
     * experiment-level constructed counter in the same transaction. Repeated
     * calls are idempotent, which makes rerunning a partially completed build
     * safe.
     */
    public int markGraphBuilt(long experimentDataId) {
        if (experimentDataId <= 0) {
            throw new IllegalArgumentException("experimentDataId must be positive");
        }
        return write(mapper -> {
            ExperimentData sample = mapper.experimentDataMapper().selectOne(
                    new QueryWrapper<ExperimentData>().eq("id", experimentDataId).last("FOR UPDATE"));
            if (sample == null) {
                throw new IllegalArgumentException("experiment_data row not found: " + experimentDataId);
            }
            if (sample.getStatus() != null && sample.getStatus() >= ExperimentData.STATUS_GRAPH_BUILT) {
                return 0;
            }
            sample.setStatus(ExperimentData.STATUS_GRAPH_BUILT);
            if (mapper.experimentDataMapper().updateById(sample) != 1) {
                throw new IllegalStateException("failed to mark graph built: " + experimentDataId);
            }
            Experiment experiment = mapper.experimentMapper().selectOne(
                    new QueryWrapper<Experiment>().eq("id", sample.getExperimentId()).last("FOR UPDATE"));
            if (experiment == null) {
                throw new IllegalStateException("experiment not found: " + sample.getExperimentId());
            }
            int constructed = experiment.getConstructedNum() == null ? 0 : experiment.getConstructedNum();
            experiment.setConstructedNum(constructed + 1);
            if (mapper.experimentMapper().updateById(experiment) != 1) {
                throw new IllegalStateException("failed to update constructed count: " + experiment.getId());
            }
            return 1;
        });
    }

    /**
     * Scores one sample and atomically updates both the sample and its experiment.
     * Aggregate metrics use the experiment's total sample count as denominator,
     * so unfinished samples count as zero just like missing predictions in HotpotQA.
     */
    public ExperimentResult recordAnswer(long experimentDataId, String predictedAnswer) {
        return recordAnswer(experimentDataId, predictedAnswer, null);
    }

    /** Scores one sample and persists the answer together with its reasoning path. */
    public ExperimentResult recordAnswer(long experimentDataId, String predictedAnswer,
                                         String reasoningPath) {
        if (experimentDataId <= 0) throw new IllegalArgumentException("experimentDataId must be positive");
        return write(mapper -> {
            ExperimentData sample = mapper.experimentDataMapper().selectOne(
                    new QueryWrapper<ExperimentData>().eq("id", experimentDataId).last("FOR UPDATE"));
            if (sample == null) throw new IllegalArgumentException("experiment_data row not found: " + experimentDataId);

            Experiment experiment = mapper.experimentMapper().selectOne(
                    new QueryWrapper<Experiment>().eq("id", sample.getExperimentId()).last("FOR UPDATE"));
            if (experiment == null) throw new IllegalStateException("experiment not found: " + sample.getExperimentId());
            DataSet dataSet = mapper.dataSetMapper().selectById(sample.getDatasetId());
            if (dataSet == null) throw new IllegalStateException("data_set row not found: " + sample.getDatasetId());

            String answer = predictedAnswer == null ? "" : predictedAnswer;
            Metrics.Result metrics = Metrics.evaluate(answer, dataSet.getAnswer());
            sample.setAnswer(answer);
            sample.setEm(metrics.getEm());
            sample.setF1(metrics.getF1());
            sample.setHit(metrics.getHit());
            // Legacy LLM-only/RAG callers do not provide a path. Preserve an
            // existing ToG path in that case instead of erasing it.
            if (reasoningPath != null) sample.setReasoningPath(reasoningPath);
            sample.setStatus(ExperimentData.STATUS_QA_COMPLETED);
            if (mapper.experimentDataMapper().updateById(sample) != 1) {
                throw new IllegalStateException("failed to update experiment_data: " + experimentDataId);
            }

            // Only the scored columns are needed to recompute the aggregates.
            // Selecting every column would also drag the large data_set.context
            // text through the connection on each individual answer update, which
            // dominates the runtime of a thousand-sample experiment.
            List<ExperimentData> allSamples = mapper.experimentDataMapper().selectList(
                    new QueryWrapper<ExperimentData>()
                            .select("id", "dataset_id", "status", "em", "f1", "hit")
                            .eq("experiment_id", experiment.getId()));
            Set<Long> datasetIds = new HashSet<>();
            for (ExperimentData item : allSamples) {
                if (item.getDatasetId() != null) datasetIds.add(item.getDatasetId());
            }
            Map<Long, DataSet> dataSetsById = new HashMap<>();
            if (!datasetIds.isEmpty()) {
                for (DataSet dataSetRow : mapper.dataSetMapper().selectList(
                        new QueryWrapper<DataSet>()
                                .select("id", "data_type")
                                .in("id", datasetIds))) {
                    dataSetsById.put(dataSetRow.getId(), dataSetRow);
                }
            }
            int evaluated = 0;
            double emSum = 0.0;
            double f1Sum = 0.0;
            double hitSum = 0.0;
            Map<String, DatasetAccumulator> byType = new LinkedHashMap<>();
            for (ExperimentData item : allSamples) {
                DataSet dataSetRow = dataSetsById.get(item.getDatasetId());
                String dataType = dataSetRow == null || dataSetRow.getDataType() == null
                        ? null : dataSetRow.getDataType().toLowerCase(Locale.ROOT);
                DatasetAccumulator accumulator = dataType == null ? null
                        : byType.computeIfAbsent(dataType, ignored -> new DatasetAccumulator());
                if (accumulator != null) accumulator.totalSamples++;
                if (item.getStatus() != null && item.getStatus() == ExperimentData.STATUS_QA_COMPLETED) {
                    evaluated++;
                    emSum += valueOrZero(item.getEm());
                    f1Sum += valueOrZero(item.getF1());
                    hitSum += valueOrZero(item.getHit());
                    if (accumulator != null) accumulator.add(item);
                }
            }
            int total = experiment.getNum() == null || experiment.getNum() <= 0
                    ? allSamples.size() : experiment.getNum();
            double denominator = total > 0 ? total : evaluated;
            double aggregateEm = denominator == 0.0 ? 0.0 : emSum / denominator;
            double aggregateF1 = denominator == 0.0 ? 0.0 : f1Sum / denominator;
            double aggregateHit = denominator == 0.0 ? 0.0 : hitSum / denominator;
            ExperimentDatasetResult hotpotqa = datasetResult("hotpotqa", byType.get("hotpotqa"),
                    experiment.getHotpotqaNum());
            ExperimentDatasetResult musique = datasetResult("musique", byType.get("musique"),
                    experiment.getMusiqueNum());
            ExperimentDatasetResult twoWiki = datasetResult("2wikimultihopqa",
                    byType.get("2wikimultihopqa"), experiment.getTwoWikiMultihopqaNum());
            experiment.setFinishNum(evaluated);
            experiment.setEm(aggregateEm);
            experiment.setF1(aggregateF1);
            experiment.setHit(aggregateHit);
            experiment.setHotpotqaF1(hotpotqa.getF1());
            experiment.setHotpotqaEm(hotpotqa.getEm());
            experiment.setHotpotqaHit(hotpotqa.getHit());
            experiment.setMusiqueF1(musique.getF1());
            experiment.setMusiqueEm(musique.getEm());
            experiment.setMusiqueHit(musique.getHit());
            experiment.setTwoWikiMultihopqaF1(twoWiki.getF1());
            experiment.setTwoWikiMultihopqaEm(twoWiki.getEm());
            experiment.setTwoWikiMultihopqaHit(twoWiki.getHit());
            if (mapper.experimentMapper().updateById(experiment) != 1) {
                throw new IllegalStateException("failed to update experiment: " + experiment.getId());
            }

            ExperimentSampleResult sampleResult = new ExperimentSampleResult(
                    sample.getId(), sample.getDatasetId(), answer, dataSet.getAnswer(),
                    metrics.getEm(), metrics.getF1(), metrics.getHit(), sample.getReasoningPath());
            Map<String, ExperimentDatasetResult> datasetResults = new LinkedHashMap<>();
            datasetResults.put(hotpotqa.getDataType(), hotpotqa);
            datasetResults.put(musique.getDataType(), musique);
            datasetResults.put(twoWiki.getDataType(), twoWiki);
            return new ExperimentResult(experiment.getId(), total, evaluated,
                    aggregateF1, aggregateEm, aggregateHit, sampleResult, datasetResults);
        });
    }

    /**
     * Resets completed QA rows to graph-built state for a clean reasoning-only
     * rerun. Graph progress and Neo4j data are intentionally untouched.
     */
    public int resetExperimentReasoning(long experimentId) {
        if (experimentId <= 0) throw new IllegalArgumentException("experimentId must be positive");
        return write(mapper -> {
            Experiment experiment = mapper.experimentMapper().selectOne(
                    new QueryWrapper<Experiment>().eq("id", experimentId).last("FOR UPDATE"));
            if (experiment == null) throw new IllegalArgumentException("experiment not found: " + experimentId);

            int reset = mapper.experimentDataMapper().update(null,
                    new UpdateWrapper<ExperimentData>()
                            .eq("experiment_id", experimentId)
                            .ge("status", ExperimentData.STATUS_GRAPH_BUILT)
                            .set("answer", null)
                            .set("em", null)
                            .set("f1", null)
                            .set("hit", null)
                            .set("reasoning_path", null)
                            .set("status", ExperimentData.STATUS_GRAPH_BUILT));

            clearExperimentMetrics(experiment);
            if (mapper.experimentMapper().updateById(experiment) != 1) {
                throw new IllegalStateException("failed to reset experiment metrics: " + experimentId);
            }
            return reset;
        });
    }

    /**
     * Clears every QA output for an experiment whose samples never build a graph.
     * Completed rows return to
     * {@code status = 0} so a later run answers them again, which keeps resume
     * semantics working for methods that have no graph phase.
     */
    public int resetExperimentQa(long experimentId) {
        if (experimentId <= 0) throw new IllegalArgumentException("experimentId must be positive");
        return write(mapper -> {
            Experiment experiment = mapper.experimentMapper().selectOne(
                    new QueryWrapper<Experiment>().eq("id", experimentId).last("FOR UPDATE"));
            if (experiment == null) throw new IllegalArgumentException("experiment not found: " + experimentId);

            int reset = mapper.experimentDataMapper().update(null,
                    new UpdateWrapper<ExperimentData>()
                            .eq("experiment_id", experimentId)
                            .set("answer", null)
                            .set("em", null)
                            .set("f1", null)
                            .set("hit", null)
                            .set("reasoning_path", null)
                            .set("status", ExperimentData.STATUS_NOT_STARTED));

            clearExperimentMetrics(experiment);
            if (mapper.experimentMapper().updateById(experiment) != 1) {
                throw new IllegalStateException("failed to reset experiment metrics: " + experimentId);
            }
            return reset;
        });
    }

    /** Restores every aggregate metric of one experiment to its initial value. */
    private static void clearExperimentMetrics(Experiment experiment) {
        experiment.setFinishNum(0);
        experiment.setF1(0.0);
        experiment.setEm(0.0);
        experiment.setHit(0.0);
        experiment.setHotpotqaF1(0.0);
        experiment.setHotpotqaEm(0.0);
        experiment.setHotpotqaHit(0.0);
        experiment.setMusiqueF1(0.0);
        experiment.setMusiqueEm(0.0);
        experiment.setMusiqueHit(0.0);
        experiment.setTwoWikiMultihopqaF1(0.0);
        experiment.setTwoWikiMultihopqaEm(0.0);
        experiment.setTwoWikiMultihopqaHit(0.0);
    }

    private static ExperimentDatasetResult datasetResult(String dataType,
                                                          DatasetAccumulator accumulator,
                                                          Integer configuredTotal) {
        int selected = accumulator == null ? 0 : accumulator.totalSamples;
        int total = configuredTotal == null || configuredTotal <= 0 ? selected : configuredTotal;
        int evaluated = accumulator == null ? 0 : accumulator.evaluatedSamples;
        double denominator = total > 0 ? total : evaluated;
        double emSum = accumulator == null ? 0.0 : accumulator.emSum;
        double f1Sum = accumulator == null ? 0.0 : accumulator.f1Sum;
        double hitSum = accumulator == null ? 0.0 : accumulator.hitSum;
        double em = denominator == 0.0 ? 0.0 : emSum / denominator;
        double f1 = denominator == 0.0 ? 0.0 : f1Sum / denominator;
        double hit = denominator == 0.0 ? 0.0 : hitSum / denominator;
        return new ExperimentDatasetResult(dataType, total, evaluated, f1, em, hit);
    }

    private static final class DatasetAccumulator {
        private int totalSamples;
        private int evaluatedSamples;
        private double emSum;
        private double f1Sum;
        private double hitSum;

        private void add(ExperimentData item) {
            evaluatedSamples++;
            emSum += valueOrZero(item.getEm());
            f1Sum += valueOrZero(item.getF1());
            hitSum += valueOrZero(item.getHit());
        }
    }

    private static double valueOrZero(Double value) {
        return value == null ? 0.0 : value;
    }

    public int deleteExperimentDataById(long id) {
        return write(mapper -> mapper.experimentDataMapper().deleteById(id));
    }

    private <T> T read(Function<SessionMappers, T> operation) {
        return withConnectionRetry(() -> {
            try (SqlSession session = client.openSession(true)) {
                return operation.apply(new SessionMappers(session));
            }
        });
    }

    private <T> T write(Function<SessionMappers, T> operation) {
        return withConnectionRetry(() -> writeOnce(operation));
    }

    /**
     * Runs one write transaction without retrying. Used by the bulk sample
     * initialisers, whose inserts are not idempotent: replaying them after an
     * ambiguous commit would duplicate experiment rows.
     */
    private <T> T writeOnce(Function<SessionMappers, T> operation) {
        try (SqlSession session = client.openSession(false)) {
            try {
                T result = operation.apply(new SessionMappers(session));
                session.commit();
                return result;
            } catch (RuntimeException | Error e) {
                try {
                    session.rollback();
                } catch (RuntimeException rollbackFailure) {
                    // Keep the original failure visible: it says what actually went wrong.
                    e.addSuppressed(rollbackFailure);
                }
                throw e;
            }
        }
    }

    /** Re-runs an idempotent database call when the link drops the connection. */
    private static <T> T withConnectionRetry(Supplier<T> action) {
        RuntimeException lastFailure = null;
        for (int attempt = 1; attempt <= CONNECTION_RETRY_LIMIT; attempt++) {
            try {
                return action.get();
            } catch (RuntimeException e) {
                if (!isConnectionLoss(e) || attempt == CONNECTION_RETRY_LIMIT) throw e;
                lastFailure = e;
                try {
                    Thread.sleep(1000L * attempt);
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                    throw e;
                }
            }
        }
        throw lastFailure;
    }

    /** True when the failure is a dropped/failed connection rather than bad SQL. */
    private static boolean isConnectionLoss(Throwable error) {
        for (Throwable current = error; current != null; current = current.getCause()) {
            String name = current.getClass().getName();
            if (name.contains("CommunicationsException")
                    || name.contains("ConnectionIsClosed")
                    || name.contains("CJCommunications")
                    || current instanceof java.io.IOException) {
                return true;
            }
        }
        return false;
    }

    @Override
    public void close() {
        client.close();
    }

    private final class SessionMappers {
        private final SqlSession sqlSession;

        private SessionMappers(SqlSession sqlSession) {
            this.sqlSession = sqlSession;
        }

        private DataSetMapper dataSetMapper() { return sqlSession.getMapper(DataSetMapper.class); }
        private ExperimentMapper experimentMapper() { return sqlSession.getMapper(ExperimentMapper.class); }
        private ExperimentDataMapper experimentDataMapper() { return sqlSession.getMapper(ExperimentDataMapper.class); }
    }
}
