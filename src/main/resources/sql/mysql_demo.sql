CREATE TABLE IF NOT EXISTS kg_demo_user (
    id BIGINT NOT NULL AUTO_INCREMENT,
    username VARCHAR(64) NOT NULL,
    email VARCHAR(128) NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_kg_demo_user_username (username)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
