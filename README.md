# NYC Yellow Taxi — Hourly Ride Demand Forecasting 🚕

An end-to-end machine learning project that forecasts NYC Yellow Taxi ride demand at the pickup-zone level, one hour ahead. This project was built **iteratively across four phases**, each exploring a different cloud architecture, from a local Jupyter setup to fully managed cloud services, to compare trade-offs in speed, scalability, cost, and ease of deployment.

---

## The Core Idea

> **How many taxi rides will be picked up at a given NYC zone in the next hour?**

Using historical NYC TLC trip records, we engineer time-series lag features (past 28 days of hourly ride counts per zone) and train a gradient-boosted model to forecast demand one hour ahead. The same ML problem was intentionally solved **four different ways** to understand how the choice of infrastructure changes the development experience, data throughput, and deployment complexity.

---

## Project Phases

### Phase 1 — Local + Hopsworks + Streamlit

**Stack:** Local machine (Conda + Jupyter Notebook) · Hopsworks Feature Store · Streamlit

**What we built:**
- Downloaded NYC TLC parquet files locally
- Engineered lag features (t−1 through t−672) per pickup zone in Jupyter
- Stored features and model artifacts in the **Hopsworks feature store**
- Trained a LightGBM model and registered it in the **Hopsworks model registry** with versioning
- Built a **Streamlit** frontend that fetches features from the store, loads the registered model, and displays predictions on a Folium choropleth map of NYC

**Results:**
- Feature pipeline, inference pipeline, and modeling pipeline were easy to reason about and track
- Model versioning and experiment tracking worked cleanly out of the box with Hopsworks
- Data loading was the bottleneck — everything ran on the laptop, so ingesting and processing large parquet files was noticeably slow
- Good starting point; proved the end-to-end concept works before moving to the cloud

---

### Phase 2 — AWS Full Pipeline + Power BI

**Stack:** AWS S3 · AWS Lambda · AWS EC2 · AWS Glue (PySpark) · AWS Athena · Power BI

**What we built:**
- Stored raw parquet files in **S3** and triggered ingestion via **Lambda functions**
- Used **AWS Glue** with PySpark to filter outliers and aggregate 4 years of data at scale
- Cataloged filtered data with a Glue Crawler and queried it with **AWS Athena**
- Connected **Power BI** directly to Athena via the Athena ODBC connector for dashboarding
- Deployed compute on **EC2** for running pipeline scripts

**Results:**
- Processing speed was dramatically faster — loading and filtering ~4 years of data completed in seconds using distributed PySpark on Glue
- The entire pipeline lived inside AWS, making navigation and service connections straightforward
- Power BI connected to Athena in a single step via the connection string — no custom API needed
- Cost awareness was critical: Glue sessions bill by the DPU-hour and must be explicitly terminated; idle sessions can rack up unexpected charges quickly

---

### Phase 3 — Full AWS End-to-End + Elastic Beanstalk Deployment

**Stack:** AWS S3 · AWS Lambda · AWS EC2 · AWS Glue · AWS Athena · AWS SageMaker · AWS RDS (PostgreSQL) · AWS Elastic Beanstalk · Streamlit

**What we built:**
- Repeated the full pipeline entirely within AWS — storage, filtering, transformation, model training, and serving
- Trained the model inside **AWS SageMaker** (Jupyter notebooks in the cloud), eliminating the local compute bottleneck
- Stored predictions in **AWS RDS (PostgreSQL)** for fast retrieval, alongside Athena for ad-hoc querying
- Deployed the **Streamlit dashboard on AWS Elastic Beanstalk** — the app was publicly accessible without managing a server manually
- Added **caching** (`@st.cache_data`) to the dashboard to avoid re-fetching predictions on every page load

**Results:**
- The most complete end-to-end AWS architecture — storage, cleaning, model building, and visualization all in one ecosystem
- Deployment via Elastic Beanstalk removed the need to configure Nginx or manage EC2 instances directly
- Caching made the dashboard noticeably faster for end users
- Most operationally complex phase: environment variables reset on SageMaker restarts, Glue sessions needed manual cleanup, and IAM permissions required careful management
- Demonstrated that a full MLOps pipeline is achievable entirely within AWS, but comes with meaningful infrastructure overhead

---

### Phase 4 — Snowflake

**Stack:** Snowflake (stages · tables · Snowpark · built-in dashboards)

**What we built:**
- Uploaded cleaned and transformed CSV files (gzipped) to a **Snowflake internal stage**
- Loaded ~75 million raw rows into a Snowflake table using `COPY INTO` from stage
- Ran filtering (67M rows) and transformation (44M rows → hourly aggregates) using **Snowpark** — Snowflake's Python-based distributed compute layer
- Trained the model externally (Google Colab), connected to Snowflake via the Python connector, and stored predictions as a Snowflake table
- Built **visualizations directly in Snowflake dashboards** — comparison of predicted vs. actual ride counts per zone, updated on demand

**Results:**
- The cleanest separation of concerns: storage, compute, and dashboarding all within one platform
- Snowpark processed 75M rows in under 2 minutes on an XS warehouse ($2/credit-hour)
- Smooth, scalable data retrieval — just resize the warehouse if more compute is needed
- Dashboard visualizations updated with a single click, no external BI tool or deployment required
- Best fit for teams already in the Snowflake ecosystem; the free trial ($400 credit, 30 days) was sufficient for the full project

---

## Dataset

**Source:** [NYC Taxi & Limousine Commission (TLC) Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

- **Format:** Parquet files, one per month
- **Coverage:** January 2023 – November 2024
- **Scale:** ~75M raw rows → 67M filtered → 44M aggregated (hourly per zone)

**Transformation pipeline:**

```
Raw Parquet (monthly files from NYC TLC)
        │
        ▼
Filter outliers
  - Remove invalid trip durations, distances, fare amounts
  - Remove passenger counts outside valid range
  - Drop rows with unreliable pickup datetimes (use year/month from filename instead)
        │
        ▼
Aggregate to hourly counts
  - GROUP BY pickup_location_id, floor(pickup_datetime, 'hour')
        │
        ▼
Engineer lag features
  - t-1 through t-672 (28 days × 24 hours) per pickup zone
        │
        ▼
Train → Predict → Store predictions → Visualize
```

---

## ML Model

| Detail | Value |
|---|---|
| Algorithm | LightGBM (gradient boosted trees) |
| Features | 672 lag features per pickup zone |
| Target | Ride count for the next hour |
| Evaluation metric | Mean Absolute Error (MAE) |
| Training period | January 2023 – December 2024 |
| Prediction scope | All NYC pickup zones, hourly |

---

## Repo Structure

```
nyc-taxi-demand-forecast/
│
├── notebooks/
│   ├── 01_fetch_raw_data.ipynb
│   ├── 02_filter_data.ipynb
│   ├── 03_transform_data.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training.ipynb
│   ├── 06_inference.ipynb
│   └── 07_monitoring.ipynb
│
├── src/
│   ├── feature_pipeline.py
│   ├── training_pipeline.py
│   ├── inference_pipeline.py
│   ├── monitoring.py
│   └── utils/
│       ├── data_utils.py
│       ├── model_utils.py
│       └── plot_utils.py
│
├── frontend/
│   └── app.py                      # Streamlit dashboard (Phases 1 to 3)
│
├── snowflake/
│   ├── upload_data.py              # Upload CSVs to Snowflake stage
│   ├── filter_data.py              # Snowpark filtering script
│   ├── transform_data.py           # Snowpark transformation script
│   └── train_predict.py            # Connect from Colab, push predictions
│
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml    # Scheduled feature ingestion
│       └── inference_pipeline.yml  # Hourly prediction job
│
├── docker-compose.yml              # Local dev container (Ubuntu 24.04 LTS)
├── requirements.txt                # Packages without versions (for dev)
├── requirements_with_versions.txt  # Pinned versions (for deployment)
├── .env.sample                     # Credentials template
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.11
- Conda or pip + venv
- [Hopsworks](https://app.hopsworks.ai/) account — free tier available (Phase 1)
- AWS account (Phases 2 & 3)
- [Snowflake](https://signup.snowflake.com/) account — 30-day free trial available (Phase 4)

---

## Key Lessons Learned

**Pin your dependencies.** Unpinned packages in `requirements.txt` caused silent pipeline breakage when upstream libraries released new versions. Once things work, always run `pip freeze > requirements_with_versions.txt` and commit it.

**Always stop Glue sessions manually.** AWS Glue sessions bill independently of the Jupyter notebook that launched them. Closing the notebook does not stop the session — terminate it from the Glue console.

**Watch SageMaker studio costs.** Opening a PySpark Glue notebook from inside SageMaker Studio starts a separate Glue session with 2 DPUs by default. Multiple open notebooks means multiple concurrent billing sessions.

**Set Snowflake warehouse auto-suspend to 1 minute.** The default is 10 minutes. Reducing this prevents idle charges when you forget to suspend the warehouse manually.

**Use Python's `pathlib` for file paths.** Hard-coded Windows backslashes break on Linux/EC2. `pathlib.Path` resolves the correct separator automatically across operating systems.

**Cache dashboard data.** Both `@st.cache_data` in Streamlit and Snowflake's native dashboard caching make a noticeable difference — re-fetching predictions on every user interaction is unnecessary and slow.

---

## Acknowledgments 👏

- [NYC Taxi & Limousine Commission](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) for the open trip record data
- [Hopsworks](https://www.hopsworks.ai/) for the feature store and model registry
- [Streamlit](https://streamlit.io/) for the ML app framework
- [Snowflake](https://www.snowflake.com/) for the cloud data platform
- [AWS](https://aws.amazon.com/) for the cloud infrastructure
- Professor Mohammad Zia · CDA 500 Applied Machine Learning · University at Buffalo