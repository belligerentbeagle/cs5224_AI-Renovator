-- RoomStyle PostgreSQL Schema
-- Apply with: uv run python sync_schema.py

CREATE TYPE generation_status AS ENUM ('pending', 'completed', 'failed');

-- 1. USERS
CREATE TABLE users (
    user_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email         VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    name          VARCHAR(255),
    created_at    TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. PRODUCTS  (populated by IKEA/Taobao search or URL scraper)
CREATE TABLE products (
    product_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_source     VARCHAR(100),
    external_product_id VARCHAR(255),
    name                TEXT,
    product_url         TEXT,
    image_url           TEXT,
    price               NUMERIC(10, 2),
    currency            VARCHAR(10),
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. PROJECTS
CREATE TABLE projects (
    project_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID REFERENCES users(user_id) ON DELETE CASCADE,
    title        VARCHAR(255),
    room_type    VARCHAR(100),
    style_prompt TEXT,
    budget_limit NUMERIC(10, 2),
    created_at   TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 4. PHOTOS  (both uploads and generated images live here)
CREATE TABLE photos (
    photo_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(project_id) ON DELETE CASCADE,
    photo_type VARCHAR(50),          -- 'original' | 'generated'
    s3_key     TEXT NOT NULL,
    file_name  TEXT,
    mime_type  VARCHAR(100),
    width      INTEGER,
    height     INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 5. DESIGN_GENERATIONS
CREATE TABLE design_generations (
    design_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id         UUID REFERENCES projects(project_id) ON DELETE CASCADE,
    input_photo_id     UUID REFERENCES photos(photo_id) ON DELETE SET NULL,
    generated_photo_id UUID REFERENCES photos(photo_id) ON DELETE SET NULL,
    prompt_text        TEXT,
    style_name         VARCHAR(100),
    status             generation_status DEFAULT 'pending',
    created_at         TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 6. GENERATION_PRODUCTS  (join: which products appear in which generation)
CREATE TABLE generation_products (
    design_id  UUID REFERENCES design_generations(design_id) ON DELETE CASCADE,
    product_id UUID REFERENCES products(product_id) ON DELETE CASCADE,
    x_position REAL,
    y_position REAL,
    width      INTEGER,
    height     INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (design_id, product_id, x_position, y_position)
);

-- 7. AFFILIATE_CLICKS  (cart click tracking)
CREATE TABLE affiliate_clicks (
    click_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    project_id   UUID NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    product_id   UUID NOT NULL REFERENCES products(product_id) ON DELETE CASCADE,
    redirect_url TEXT NOT NULL,
    clicked_at   TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- INDEXES (PostgreSQL does not auto-index FK columns)
CREATE INDEX idx_projects_user_id              ON projects(user_id);
CREATE INDEX idx_photos_project_id             ON photos(project_id);
CREATE INDEX idx_design_generations_project_id ON design_generations(project_id);
CREATE INDEX idx_products_external             ON products(external_source, external_product_id);
CREATE INDEX idx_affiliate_clicks_user_id      ON affiliate_clicks(user_id);
CREATE INDEX idx_affiliate_clicks_product_id   ON affiliate_clicks(product_id);
