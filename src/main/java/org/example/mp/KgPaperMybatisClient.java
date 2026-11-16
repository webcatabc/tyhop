package org.example.mp;

import org.apache.ibatis.session.SqlSession;
import org.example.config.AppConfig;
import org.example.infrastructure.MybatisPlusClient;
import org.example.mapper.DataSetMapper;
import org.example.mapper.ExperimentDataMapper;
import org.example.mapper.ExperimentMapper;

/** Convenience access point for the three kg-paper tables. */
public final class KgPaperMybatisClient implements AutoCloseable {
    private final MybatisPlusClient client;

    public KgPaperMybatisClient(AppConfig config) {
        client = new MybatisPlusClient(config,
                DataSetMapper.class, ExperimentMapper.class, ExperimentDataMapper.class);
    }

    public SqlSession openSession(boolean autoCommit) {
        return client.openSession(autoCommit);
    }

    public DataSetMapper dataSetMapper(SqlSession session) {
        return session.getMapper(DataSetMapper.class);
    }

    public ExperimentMapper experimentMapper(SqlSession session) {
        return session.getMapper(ExperimentMapper.class);
    }

    public ExperimentDataMapper experimentDataMapper(SqlSession session) {
        return session.getMapper(ExperimentDataMapper.class);
    }

    @Override
    public void close() {
        client.close();
    }
}
