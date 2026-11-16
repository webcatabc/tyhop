package org.example.config;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.yaml.snakeyaml.Yaml;

import java.io.InputStream;
import java.util.Collections;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Single source of truth for application.yaml and environment configuration. */
public final class AppConfig {
    private static final Logger LOG = LoggerFactory.getLogger(AppConfig.class);
    private static final Pattern PLACEHOLDER = Pattern.compile("\\$\\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?}");

    public final String dashscopeBaseUrl;
    public final String dashscopeApiKey;
    public final String llmModel;
    public final long llmTimeoutSeconds;
    public final int llmMaxRetries;
    public final String llmSystemPrompt;
    public final boolean llmDisableThinking;
    public final int memoryMaxMessages;
    public final String neo4jUri;
    public final String neo4jUsername;
    public final String neo4jPassword;
    public final String neo4jDatabase;
    public final String neo4jEntityVectorIndex;
    public final String neo4jRelationVectorIndex;
    public final String neo4jTypeVectorIndex;
    public final String redisHost;
    public final int redisPort;
    public final String redisPassword;
    public final int redisDatabase;
    public final String redisKeyPrefix;
    public final int redisTtlSeconds;
    public final int redisSocketTimeoutMs;
    public final String embeddingBaseUrl;
    public final String embeddingIndexName;
    public final int embeddingTimeoutSeconds;
    public final int embeddingMaxConcurrency;
    public final String mysqlHost;
    public final int mysqlPort;
    public final String mysqlUser;
    public final String mysqlPassword;
    public final String mysqlDatabase;
    public final int mysqlConnectTimeoutSeconds;
    public final int mysqlPoolMaxActive;
    public final int mysqlPoolMaxIdle;
    private AppConfig(Map<String, Object> yaml) {
        dashscopeBaseUrl = value("DASHSCOPE_BASE_URL", "dashscope.base-url", "", yaml);
        dashscopeApiKey = value("DASHSCOPE_API_KEY", "dashscope.api-key", "", yaml);
        llmModel = value("DASHSCOPE_MODEL", "dashscope.model", "", yaml);
        llmTimeoutSeconds = longValue("LLM_TIMEOUT_SECONDS", "llm.timeout-seconds", 260, yaml);
        llmMaxRetries = intValue("LLM_MAX_RETRIES", "llm.max-retries", 3, yaml);
        llmSystemPrompt = value("LLM_SYSTEM_PROMPT", "llm.system-prompt", "", yaml);
        llmDisableThinking = booleanValue("LLM_DISABLE_THINKING", "llm.disable-thinking", true, yaml);
        memoryMaxMessages = intValue("MEMORY_MAX_MESSAGES", "memory.max-messages", 20, yaml);
        neo4jUri = value("NEO4J_URI", "neo4j.uri", "", yaml);
        neo4jUsername = value("NEO4J_USERNAME", "neo4j.username", "", yaml);
        neo4jPassword = value("NEO4J_PASSWORD", "neo4j.password", "", yaml);
        neo4jDatabase = value("NEO4J_DATABASE", "neo4j.database", "", yaml);
        neo4jEntityVectorIndex = value("NEO4J_ENTITY_VECTOR_INDEX", "neo4j.entity-vector-index",
                "", yaml);
        neo4jRelationVectorIndex = value("NEO4J_RELATION_VECTOR_INDEX", "neo4j.relation-vector-index",
                "", yaml);
        neo4jTypeVectorIndex = value("NEO4J_TYPE_VECTOR_INDEX", "neo4j.type-vector-index",
                "", yaml);
        redisHost = value("REDIS_HOST", "redis.host", "", yaml);
        redisPort = intValue("REDIS_PORT", "redis.port", 6379, yaml);
        redisPassword = value("REDIS_PASSWORD", "redis.password", "", yaml);
        redisDatabase = intValue("REDIS_DB", "redis.database", 0, yaml);
        redisKeyPrefix = value("REDIS_KEY_PREFIX", "redis.key-prefix", "", yaml);
        redisTtlSeconds = intValue("REDIS_TTL_SECONDS", "redis.ttl-seconds", 0, yaml);
        redisSocketTimeoutMs = intValue("REDIS_SOCKET_TIMEOUT_SECONDS", "redis.socket-timeout-seconds", 2, yaml) * 1000;
        embeddingBaseUrl = value("EMBEDDING_BASE_URL", "embedding.base-url", "", yaml);
        embeddingIndexName = value("EMBEDDING_INDEX_NAME", "embedding.index-name", "", yaml);
        embeddingTimeoutSeconds = intValue("EMBEDDING_TIMEOUT_SECONDS", "embedding.timeout-seconds", 120, yaml);
        embeddingMaxConcurrency = intValue("EMBEDDING_MAX_CONCURRENCY", "embedding.max-concurrency", 4, yaml);
        if (embeddingMaxConcurrency < 1) throw new IllegalArgumentException("EMBEDDING_MAX_CONCURRENCY must be positive");
        mysqlHost = value("MYSQL_HOST", "mysql.host", "", yaml);
        mysqlPort = intValue("MYSQL_PORT", "mysql.port", 3306, yaml);
        mysqlUser = value("MYSQL_USER", "mysql.user", "", yaml);
        mysqlPassword = value("MYSQL_PASSWORD", "mysql.password", "", yaml);
        mysqlDatabase = value("MYSQL_DATABASE", "mysql.database", "", yaml);
        mysqlConnectTimeoutSeconds = intValue("MYSQL_CONNECT_TIMEOUT_SECONDS", "mysql.connect-timeout-seconds", 5, yaml);
        mysqlPoolMaxActive = intValue("MYSQL_POOL_MAX_ACTIVE", "mysql.pool-max-active", 32, yaml);
        mysqlPoolMaxIdle = intValue("MYSQL_POOL_MAX_IDLE", "mysql.pool-max-idle", 8, yaml);
        if (mysqlPoolMaxActive < 1 || mysqlPoolMaxIdle < 1 || mysqlPoolMaxIdle > mysqlPoolMaxActive) {
            throw new IllegalArgumentException("invalid MySQL pool limits");
        }
    }

    public static AppConfig load() {
        return new AppConfig(loadYaml());
    }

    public String mysqlJdbcUrl() {
        return "jdbc:mysql://" + mysqlHost + ":" + mysqlPort + "/" + mysqlDatabase
                + "?useSSL=false&allowPublicKeyRetrieval=true&serverTimezone=UTC&connectTimeout="
                + (mysqlConnectTimeoutSeconds * 1000)
                + "&socketTimeout=60000";
    }

    private static Map<String, Object> loadYaml() {
        try (InputStream input = AppConfig.class.getClassLoader().getResourceAsStream("application.yaml")) {
            if (input == null) {
                return Collections.emptyMap();
            }
            Object parsed = new Yaml().load(input);
            return parsed instanceof Map<?, ?> map ? castMap(map) : Collections.emptyMap();
        } catch (Exception e) {
            LOG.warn("Unable to load application.yaml: {}", concise(e));
            return Collections.emptyMap();
        }
    }

    private static Map<String, Object> castMap(Map<?, ?> source) {
        Map<String, Object> result = new HashMap<>();
        source.forEach((key, value) -> result.put(String.valueOf(key), value));
        return result;
    }

    private static String value(String name, String yamlPath, String defaultValue, Map<String, Object> yaml) {
        String result = System.getProperty(name);
        if (result == null) {
            result = System.getenv(name);
        }
        if (result != null) {
            return result;
        }
        Object yamlValue = yamlValue(yaml, yamlPath);
        return yamlValue == null ? defaultValue : expand(String.valueOf(yamlValue), defaultValue);
    }

    private static int intValue(String name, String yamlPath, int defaultValue, Map<String, Object> yaml) {
        return Integer.parseInt(value(name, yamlPath, Integer.toString(defaultValue), yaml));
    }

    private static long longValue(String name, String yamlPath, long defaultValue, Map<String, Object> yaml) {
        return Long.parseLong(value(name, yamlPath, Long.toString(defaultValue), yaml));
    }

    /** Accepts true/false, 1/0, yes/no and on/off so shell overrides stay readable. */
    private static boolean booleanValue(String name, String yamlPath, boolean defaultValue, Map<String, Object> yaml) {
        String raw = value(name, yamlPath, Boolean.toString(defaultValue), yaml);
        return switch (raw.trim().toLowerCase(Locale.ROOT)) {
            case "true", "1", "yes", "on" -> true;
            case "false", "0", "no", "off" -> false;
            default -> throw new IllegalArgumentException(name + " must be a boolean but was: " + raw);
        };
    }

    private static Object yamlValue(Map<String, Object> yaml, String path) {
        Object current = yaml;
        for (String part : path.split("\\.")) {
            if (!(current instanceof Map<?, ?> map)) {
                return null;
            }
            current = map.get(part);
        }
        return current;
    }

    private static String expand(String raw, String defaultValue) {
        Matcher matcher = PLACEHOLDER.matcher(raw);
        if (!matcher.matches()) {
            return raw;
        }
        String result = System.getProperty(matcher.group(1));
        if (result == null) {
            result = System.getenv(matcher.group(1));
        }
        return result == null ? (matcher.group(2) == null ? defaultValue : matcher.group(2)) : result;
    }

    private static String concise(Exception e) {
        return e.getClass().getSimpleName() + (e.getMessage() == null ? "" : ": " + e.getMessage());
    }
}
