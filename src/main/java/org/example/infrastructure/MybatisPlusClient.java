package org.example.infrastructure;

import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.core.MybatisSqlSessionFactoryBuilder;
import org.apache.ibatis.datasource.pooled.PooledDataSource;
import org.apache.ibatis.mapping.Environment;
import org.apache.ibatis.session.SqlSession;
import org.apache.ibatis.session.SqlSessionFactory;
import org.apache.ibatis.transaction.jdbc.JdbcTransactionFactory;
import org.example.config.AppConfig;

/** Standalone MyBatis-Plus factory; no Spring Boot starter is required. */
public final class MybatisPlusClient implements AutoCloseable {
    private final PooledDataSource dataSource;
    private final SqlSessionFactory sessionFactory;

    public MybatisPlusClient(AppConfig config, Class<?>... mapperTypes) {
        dataSource = new PooledDataSource("com.mysql.cj.jdbc.Driver", config.mysqlJdbcUrl(),
                config.mysqlUser, config.mysqlPassword);
        dataSource.setPoolMaximumActiveConnections(config.mysqlPoolMaxActive);
        dataSource.setPoolMaximumIdleConnections(config.mysqlPoolMaxIdle);
        MybatisConfiguration configuration = new MybatisConfiguration(new Environment(
                "mysql", new JdbcTransactionFactory(), dataSource));
        configuration.setMapUnderscoreToCamelCase(true);
        for (Class<?> mapperType : mapperTypes) {
            configuration.addMapper(mapperType);
        }
        sessionFactory = new MybatisSqlSessionFactoryBuilder().build(configuration);
    }

    public SqlSession openSession(boolean autoCommit) {
        return sessionFactory.openSession(autoCommit);
    }

    @Override
    public void close() {
        dataSource.forceCloseAll();
        com.mysql.cj.jdbc.AbandonedConnectionCleanupThread.checkedShutdown();
    }
}
